import numpy as np
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..model import StationCache


@dataclass
class TrackingTaskArrays:
    """
    Per-task quantities packed into arrays, so a term can evaluate every task in a few
    vectorized numpy operations.

    Attributes
    ----------
    base_stations: np.ndarray, shape (num_tasks, 3)
        Each task's point in its base frame, before scaling or offsets.
    mobod_indexes: np.ndarray, shape (num_tasks,)
        Each task's base-frame `MobilizedBodyIndex`.
    scale_groups: np.ndarray, shape (num_tasks,)
        Index of the `BodyScaleGroup` scaling each task, or -1 where none applies.
    offset_groups: np.ndarray, shape (num_tasks,)
        Index of the offset group applying to each task, or -1 where none applies.
    base_frames: list[osim.PhysicalFrame]
        The distinct base frames across all tasks, one per entry of `body_rows`.
    body_rows: np.ndarray, shape (num_tasks,)
        For each task, the index into `base_frames` of its base frame.
    reference_positions: np.ndarray, shape (num_tasks, 3)
        Each task's reference position.
    weights: np.ndarray or None, shape (num_tasks,)
        Each task's cost weight, for terms carrying a single weight per task.
        ``None`` for terms that weight position and orientation separately.
    scaled_rows, scaled_groups: np.ndarray
        The tasks a body scale applies to, and the scale group of each. Paired, so
        ``stations[scaled_rows] *= scales[scaled_groups]`` applies every scale at once.
    shifted_rows, shifted_groups: np.ndarray
        The same pairing for placement offsets.
    scaled_base_stations: np.ndarray, shape (len(scaled_rows), 3)
        ``base_stations[scaled_rows]``, used when accumulating the body-scale Jacobian.
    unique_mobod_indexes: np.ndarray
        The distinct base-frame mobod indexes across all tasks, i.e. exactly the rows a
        per-body gradient scatter can touch.
    double_weights_column: np.ndarray or None, shape (num_tasks, 1)
        ``2 * weights`` as a column, the factor the squared-error gradient needs.
    position_weights, orientation_weights: np.ndarray or None
        Each task's separate position and orientation weights, for terms that carry
        both. ``None`` for terms with a single weight per task.
    double_position_weights_column: np.ndarray or None, shape (num_tasks, 1)
        ``2 * position_weights`` as a column.
    reference_orientations: np.ndarray or None, shape (num_tasks, 4)
        Each task's reference quaternion.
    base_relative_rotations: np.ndarray or None, shape (num_tasks, 3, 3)
        Each task's rotation relative to its base frame, which is fixed. A frame's
        rotation in ground is ``R_GB @ R_BF``, so one transform read per base frame
        yields every task's orientation.
    reference_rotations: np.ndarray or None, shape (num_tasks, 3, 3)
        Each task's reference orientation as a rotation matrix.
    """
    base_stations: np.ndarray
    mobod_indexes: np.ndarray
    scale_groups: np.ndarray
    offset_groups: np.ndarray
    base_frames: list
    body_rows: np.ndarray
    reference_positions: np.ndarray
    weights: np.ndarray = None
    scaled_rows: np.ndarray = None
    scaled_groups: np.ndarray = None
    shifted_rows: np.ndarray = None
    shifted_groups: np.ndarray = None
    scaled_base_stations: np.ndarray = None
    unique_mobod_indexes: np.ndarray = None
    double_weights_column: np.ndarray = None
    position_weights: np.ndarray = None
    orientation_weights: np.ndarray = None
    double_position_weights_column: np.ndarray = None
    reference_orientations: np.ndarray = None
    base_relative_rotations: np.ndarray = None
    reference_rotations: np.ndarray = None

    def __post_init__(self):
        self.scaled_rows = np.flatnonzero(self.scale_groups >= 0)
        self.scaled_groups = self.scale_groups[self.scaled_rows]
        self.shifted_rows = np.flatnonzero(self.offset_groups >= 0)
        self.shifted_groups = self.offset_groups[self.shifted_rows]
        self.scaled_base_stations = self.base_stations[self.scaled_rows]
        self.unique_mobod_indexes = np.unique(self.mobod_indexes)
        self.double_weights_column = (None if self.weights is None
                                      else 2.0 * self.weights[:, None])


class TrackingTasks(ABC):
    """
    A base class for task-specific storage and registration.
    """
    @abstractmethod
    def initialize_tasks(self, state: osim.State, **kwargs) -> float:
        pass

    def invalidate_task_arrays(self) -> None:
        """
        Drop the cached `TrackingTaskArrays`. Called whenever a task is registered, so
        the arrays are rebuilt on next use.
        """
        self._task_arrays = None

    @property
    def task_arrays(self) -> TrackingTaskArrays:
        """
        The cached `TrackingTaskArrays` for the registered tasks, built on first use.
        """
        if getattr(self, '_task_arrays', None) is None:
            self._task_arrays = self._build_task_arrays()
        return self._task_arrays

    def _build_task_arrays(self) -> TrackingTaskArrays:
        num_tasks = self.num_tasks
        base_stations = (np.asarray(self.base_stations, dtype=float).reshape(-1, 3)
                         if num_tasks else np.zeros((0, 3)))
        mobod_indexes = np.array(
            [int(self.mobod_indexes.getElt(i)) for i in range(num_tasks)], dtype=int)
        scale_groups = np.array(
            [-1 if cache.body_scale_group_index is None
             else cache.body_scale_group_index for cache in self.station_caches],
            dtype=int)
        offset_groups = np.array(
            [-1 if g is None else g for g in self.offset_group_indexes], dtype=int)

        # One entry per distinct base frame, in first-seen order.
        frames, rows, row_of_mobod = [], [], {}
        for i, cache in enumerate(self.station_caches):
            key = cache.mobod_index
            if key not in row_of_mobod:
                row_of_mobod[key] = len(frames)
                frames.append(cache.base_frame)
            rows.append(row_of_mobod[key])

        reference_positions = (np.asarray(self.positions, dtype=float).reshape(-1, 3)
                               if num_tasks else np.zeros((0, 3)))
        single_weights = getattr(self, 'weights', None)
        weights = (np.asarray(single_weights, dtype=float)
                   if single_weights is not None else None)

        return TrackingTaskArrays(
            base_stations=base_stations, mobod_indexes=mobod_indexes,
            scale_groups=scale_groups, offset_groups=offset_groups,
            base_frames=frames, body_rows=np.array(rows, dtype=int),
            reference_positions=reference_positions, weights=weights)


class MarkerTrackingTasks(TrackingTasks):
    """
    Marker-specific task storage and registration.
    """
    def initialize_tasks(self):
        self.markers = []
        self.station_caches: list[StationCache] = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.weights = []
        self.base_frames = []
        self.base_stations = []
        self.offset_group_indexes: list[int] = []
        self._task_arrays: TrackingTaskArrays = None

    def add_marker(self, marker_path: str, position: osim.Vec3, weight: float = 1.0,
                   offset_group_index: int | None = None,
                   station_cache: StationCache | None = None):
        """
        Register a marker to track.

        Parameters
        ----------
        marker_path: str
            The OpenSim Model path to the tracking marker.
        position: osim.Vec3
            The reference position data tracked by the model marker.
        weight: float, optional
            The cost weight for the position error. Default: 1.0.
        offset_group_index: int | None, optional
            The index of the offset group whose XYZ offset applies to this marker, or
            ``None`` if this marker's placement is not offset. Default: ``None``.
        station_cache: StationCache | None, optional
            A previously resolved `StationCache` for this marker (e.g., from
            `CostCache.station_cache`), or ``None`` to resolve one here. Default:
            ``None``.
        """
        if not self.mc.model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')

        marker = osim.Marker.safeDownCast(self.mc.model.getComponent(marker_path))
        if station_cache is None:
            self.mc.model.realizePosition(self.mc.state)
            station_cache = StationCache.from_station(self.mc, marker)
        cache = station_cache
        self.station_caches.append(cache)
        self.markers.append(marker)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.weights.append(weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)
        self.invalidate_task_arrays()


class FrameTrackingTasks(TrackingTasks):
    """
    Frame-specific task storage and registration.
    """
    def initialize_tasks(self):
        self.frames = []
        self.station_caches: list[StationCache] = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.orientations = []
        self.position_weights = []
        self.orientation_weights = []
        self.base_frames = []
        self.base_stations = []
        self.offset_group_indexes: list[int] = []
        self._task_arrays: TrackingTaskArrays = None

    def add_frame(self, frame_path: str, position: osim.Vec3,
                  orientation: osim.Quaternion, position_weight: float = 1.0,
                  orientation_weight: float = 1.0,
                  offset_group_index: int | None = None,
                  station_cache: StationCache | None = None):
        """
        Register a frame to track.

        Parameters
        ----------
        frame_path: str
            The OpenSim Model path to the tracking frame.
        position: osim.Vec3
            The reference position data tracked by the model frame.
        orientation: osim.Quaternion
            The reference orientation, expressed as a quaternion, tracked by the model
            frame.
        position_weight: float, optional
            The cost weight for the position error. Default: 1.0.
        orientation_weight: float, optional
            The cost weight for the orientation error. Default: 1.0.
        offset_group_index: int | None, optional
            The index of the offset group whose XYZ offset applies to this frame, or
            ``None`` if this frame's placement is not offset. Default: ``None``.
        station_cache: StationCache | None, optional
            A previously resolved `StationCache` for this frame's origin (e.g., from
            `CostCache.frame_cache`), or ``None`` to resolve one here. Default:
            ``None``.
        """
        if not self.mc.model.hasComponent(frame_path):
            raise ValueError(f'Model does not have a component at path {frame_path}.')
        if position_weight < 0:
            raise ValueError(f'Expected position_weight to be non-negative, but got '
                             f'{position_weight}.')
        if orientation_weight < 0:
            raise ValueError(f'Expected orientation_weight to be non-negative, but got '
                             f'{orientation_weight}.')

        frame = osim.PhysicalFrame.safeDownCast(self.mc.model.getComponent(frame_path))
        cache = (StationCache.from_frame(self.mc, frame) if station_cache is None
                 else station_cache)
        self.station_caches.append(cache)
        self.frames.append(frame)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.orientations.append(np.array([orientation.get(i) for i in range(4)]))
        self.position_weights.append(position_weight)
        self.orientation_weights.append(orientation_weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)
        self.invalidate_task_arrays()

    def _build_task_arrays(self) -> TrackingTaskArrays:
        arrays = super()._build_task_arrays()
        arrays.position_weights = np.asarray(self.position_weights, dtype=float)
        arrays.orientation_weights = np.asarray(self.orientation_weights, dtype=float)
        arrays.double_position_weights_column = (
            2.0 * arrays.position_weights[:, None] if self.num_tasks
            else np.zeros((0, 1)))
        arrays.reference_orientations = (
            np.asarray(self.orientations, dtype=float).reshape(-1, 4)
            if self.num_tasks else np.zeros((0, 4)))
        base_relative_rotations = []
        for frame in self.frames:
            transform = frame.findTransformInBaseFrame()
            base_relative_rotations.append(transform.R().to_numpy())
        arrays.base_relative_rotations = (
            np.array(base_relative_rotations) if self.num_tasks
            else np.zeros((0, 3, 3)))
        arrays.reference_rotations = (
            np.array([osim.Rotation(osim.Quaternion(*[float(v) for v in q])).to_numpy()
                      for q in arrays.reference_orientations])
            if self.num_tasks else np.zeros((0, 3, 3)))
        return arrays
