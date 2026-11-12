import numpy as np
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..model import ModelCache, StationCache


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


class TrackingTerm(ABC):
    """
    A base class for tracking cost terms: a set of registered tasks (markers or frames)
    and the scalar error between the model and their reference data, along with its
    Jacobian with respect to the model's generalized coordinates and, optionally, its
    body scales, placement offsets, and joint-level parameters.

    A term evaluates each task at its cached base-frame station. `apply_state` updates
    those stations for given body scales and placement offsets; until it is called,
    the term evaluates the unscaled, unoffset stations.

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.station_caches: list[StationCache] = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.base_frames = []
        self.base_stations = []
        self.offset_group_indexes: list[int] = []
        self._task_arrays: TrackingTaskArrays = None
        self._station_array = None

    @abstractmethod
    def calc_error(self, state: osim.State) -> float:
        pass

    @abstractmethod
    def calc_jacobian(self, state: osim.State,
                      include_parameters: bool = False) -> list[np.ndarray]:
        """
        Return the error's Jacobian as ``[Jq]``, or as ``[Jq, Js, Jo, Jr, Jl]`` (the
        coordinate, body-scale, offset, ellipsoid-radii-scale, and beam-length-scale
        blocks) if `include_parameters` is ``True``.
        """

    def _add_task(self, cache: StationCache, position: osim.Vec3,
                  offset_group_index: int | None) -> None:
        self.station_caches.append(cache)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)
        self.invalidate_task_arrays()

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
        if self._task_arrays is None:
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

        return TrackingTaskArrays(
            base_stations=base_stations, mobod_indexes=mobod_indexes,
            scale_groups=scale_groups, offset_groups=offset_groups,
            base_frames=frames, body_rows=np.array(rows, dtype=int),
            reference_positions=reference_positions)

    @property
    def station_array(self) -> np.ndarray:
        cached = self._station_array
        if cached is None or len(cached) != self.num_tasks:
            cached = self._station_array = self.task_arrays.base_stations.copy()
        return cached

    def apply_state(self, body_scales: np.ndarray, offsets: np.ndarray) -> None:
        arrays = self.task_arrays
        if len(arrays.scaled_rows) or len(arrays.shifted_rows):
            stations = arrays.base_stations.copy()
            if len(arrays.scaled_rows):
                scales = np.asarray(body_scales, dtype=float).reshape(-1, 3)
                stations[arrays.scaled_rows] *= scales[arrays.scaled_groups]
            if len(arrays.shifted_rows):
                shifts = np.asarray(offsets, dtype=float).reshape(-1, 3)
                stations[arrays.shifted_rows] += shifts[arrays.shifted_groups]
        else:
            stations = arrays.base_stations
        self._station_array = stations
        self.stations = osim.SimTKArrayVec3.createFromMat(
            np.ascontiguousarray(stations, dtype=float).reshape(-1))


class MarkerTrackingTerm(TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model markers'
    positions and corresponding reference positions. Individual markers are registered
    via add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        super().__init__(mc)
        self.markers = []
        self.weights = []

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
        self.markers.append(marker)
        self.weights.append(weight)
        self._add_task(station_cache, position, offset_group_index)

    def _build_task_arrays(self) -> TrackingTaskArrays:
        arrays = super()._build_task_arrays()
        arrays.weights = np.asarray(self.weights, dtype=float)
        arrays.double_weights_column = 2.0 * arrays.weights[:, None]
        return arrays

    def calc_error(self, state) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rows = arrays.body_rows
        # p_GS = p_GB + R_GB · p_BS
        p_GB = poses[:, :, 3][rows]
        R_GB = poses[:, :, :3][rows]
        positions = (p_GB + (R_GB @ self.station_array[..., None])[..., 0])
        residuals = positions - arrays.reference_positions
        return float(np.sum(arrays.weights * np.sum(residuals * residuals, axis=1)))

    def calc_jacobian(self, state, include_parameters=False) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_q_indexes)))
        if include_parameters:
            Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
            Jo = np.zeros((1, 3 * len(self.mc.marker_offset_groups)))
            Jr = np.zeros((1, 3 * len(self.mc.ellipsoid_radii_scale_groups)))
            Jl = np.zeros((1, len(self.mc.beam_length_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo, Jr, Jl] if include_parameters else [Jq]

        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows

        # The per-marker error gradient in Ground.
        R_GB = rotations[rows]
        p_GS = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        gradient = arrays.double_weights_column * (p_GS - arrays.reference_positions)
        dp_GS = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(gradient, dtype=float).reshape(-1))

        # Calculate the Jacobian of the position error with respect to the coordinates.
        grad_u = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, dp_GS, grad_u)
        grad_q = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
        Jq[0, :] = grad_q.to_numpy()[self.mc.coordinate_q_indexes]
        if not include_parameters:
            return [Jq]

        # The sensitivity of each marker's ground position to a shift from an offset
        # variable.
        doffset = (gradient[:, None, :] @ R_GB)[:, 0, :]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GS_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GS.get(i)   # for each marker i on body k
        #
        accumulated = np.zeros((self.mc.num_mobod, 3))
        np.add.at(accumulated, arrays.mobod_indexes, gradient)
        dp_GB = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(accumulated, dtype=float).reshape(-1))

        # Calculate the position-error Jacobian with respect to body scales.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Calculate the position-error Jacobians with respect to the joint-level
        # parameters.
        Jr = self.mc.calc_position_jacobian_wrt_ellipsoid_radii_scales(state, dp_GB)
        Jl = self.mc.calc_position_jacobian_wrt_beam_length_scales(state, dp_GB)

        # Assemble the marker offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the marker offsets to the Jacobian with respect
        # to body scales.
        if len(arrays.shifted_rows):
            np.add.at(Jo.reshape(-1, 3), arrays.shifted_groups,
                      doffset[arrays.shifted_rows])
        if len(arrays.scaled_rows):
            np.add.at(Js.reshape(-1, 3), arrays.scaled_groups,
                      arrays.scaled_base_stations * doffset[arrays.scaled_rows])

        return [Jq, Js, Jo, Jr, Jl]


class FrameTrackingTerm(TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model frames'
    positions and orientations and corresponding reference data. Individual frames are
    registered via add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        super().__init__(mc)
        self.frames = []
        self.orientations = []
        self.position_weights = []
        self.orientation_weights = []

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
        self.frames.append(frame)
        self.orientations.append(np.array([orientation.get(i) for i in range(4)]))
        self.position_weights.append(position_weight)
        self.orientation_weights.append(orientation_weight)
        self._add_task(cache, position, offset_group_index)

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

    def calc_error(self, state) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        # Positions come from the scaled and offset stations applied by apply_state,
        # so the error stays consistent with any applied offsets.
        positions = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        residuals = positions - arrays.reference_positions
        position_error = arrays.position_weights * np.sum(
            residuals * residuals, axis=1)

        # trace(R_ref^T R) == 1 + 2cos(theta), so
        # (3 - trace(R_ref^T R)) / 4 == (1 - cos(theta)) / 2 == sin(theta/2)**2,
        # which is equal to the quaternion alignment loss 1 - (eps . q_ref)**2.
        frame_rotations = R_GB @ arrays.base_relative_rotations
        trace = np.einsum('nij,nij->n', arrays.reference_rotations, frame_rotations)
        orientation_error = arrays.orientation_weights * (3.0 - trace) / 4.0

        return float(np.sum(position_error + orientation_error))

    def calc_jacobian(self, state, include_parameters=False) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_q_indexes)))
        if include_parameters:
            Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
            Jo = np.zeros((1, 3 * len(self.mc.frame_offset_groups)))
            Jr = np.zeros((1, 3 * len(self.mc.ellipsoid_radii_scale_groups)))
            Jl = np.zeros((1, len(self.mc.beam_length_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo, Jr, Jl] if include_parameters else [Jq]

        # Compute the combined position and orientation error for every frame at
        # once.
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        # The frames' ground positions come from their (possibly offset) cached
        # stations, so the gradient is consistent with any applied offsets.
        positions = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        gradient = arrays.double_position_weights_column * (
            positions - arrays.reference_positions)

        # E = w_o * (1 - (eps . q_ref)**2) = w_o * (3 - trace(R_ref^T R)) / 4
        #
        # dE/dq = dE/dR * dR/dq = (-w_o / 4) * tr(R_ref^T dR/dq)
        #
        # dR/dq = [z]x R = [Jw]x R, where z is the axis of rotation of the frame, which
        #                           is equal to Jw to the angular Jacobian of the frame
        #                           in ground.
        #
        # dE/dq = (-w_o / 4) * tr(R_ref^T [Jw]x R)
        #       = (-w_o / 4) * tr([Jw]x R R_ref^T)
        #       = (-w_o / 4) * tr([Jw]x M)          -->  M = R R_ref^T
        #       = (-w_o / 4) * Jw^T [M12 - M21, M20 - M02, M01 - M10]
        frame_rotations = R_GB @ arrays.base_relative_rotations
        aligned = np.einsum('nij,nkj->nik', frame_rotations,
                            arrays.reference_rotations)
        orientation_error = (-0.25 * arrays.orientation_weights)[:, None] * np.stack(
            [aligned[:, 1, 2] - aligned[:, 2, 1],
             aligned[:, 2, 0] - aligned[:, 0, 2],
             aligned[:, 0, 1] - aligned[:, 1, 0]], axis=1)

        # A SpatialVec holds the angular half first, then the linear half.
        spatialError = osim.VectorOfSpatialVec.createFromMat(
            np.ascontiguousarray(
                np.concatenate([orientation_error, gradient], axis=1),
                dtype=float).reshape(-1))

        # Calculate the frame (position and orientation) error Jacobian.
        grad_u = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, grad_u)
        grad_q = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
        Jq[0, :] = grad_q.to_numpy()[self.mc.coordinate_q_indexes]
        if not include_parameters:
            return [Jq]

        # Sensitivity of each frame's ground position to a base-frame station shift.
        doffset = np.einsum('ni,nij->nj', gradient, R_GB)

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GF_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GF.get(i)   # for each frame i on body k
        #
        accumulated = np.zeros((self.mc.num_mobod, 3))
        np.add.at(accumulated, arrays.mobod_indexes, gradient)
        dp_GB = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(accumulated, dtype=float).reshape(-1))

        # Calculate the position-error Jacobian with respect to body scales. This does
        # not include the contributions from frame offsets, we will include that below.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Calculate the position-error Jacobians with respect to the joint-level
        # parameters.
        Jr = self.mc.calc_position_jacobian_wrt_ellipsoid_radii_scales(state, dp_GB)
        Jl = self.mc.calc_position_jacobian_wrt_beam_length_scales(state, dp_GB)

        # Assemble the frame offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the frame offsets to the Jacobian with respect
        # to body scales.
        if len(arrays.shifted_rows):
            np.add.at(Jo.reshape(-1, 3), arrays.shifted_groups,
                      doffset[arrays.shifted_rows])
        if len(arrays.scaled_rows):
            np.add.at(Js.reshape(-1, 3), arrays.scaled_groups,
                      arrays.scaled_base_stations * doffset[arrays.scaled_rows])

        return [Jq, Js, Jo, Jr, Jl]


@dataclass
class TrackingTaskSet:
    """
    The marker and frame tasks a tracking cost evaluates at one time sample of one
    trial. A tracking cost builds one per sample, and its `CostFunction` carries it as
    the callback's context.

    Attributes
    ----------
    marker_term: MarkerTrackingTerm
        The marker tasks.
    frame_term: FrameTrackingTerm
        The frame tasks.
    """
    marker_term: MarkerTrackingTerm
    frame_term: FrameTrackingTerm
