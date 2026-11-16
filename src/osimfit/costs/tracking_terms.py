import numpy as np
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from ..model import ModelCache, StationCache


@dataclass(frozen=True)
class TaskGeometry:
    """
    Task information related to model geometry or topology.

    Attributes
    ----------
    base_stations: np.ndarray, shape (num_tasks, 3)
        The task stations, before scaling or offsets, expressed in each base frame.
    mobod_indexes: np.ndarray, shape (num_tasks,)
        The base-frame `MobilizedBodyIndex` for each task.
    base_frames: list[osim.PhysicalFrame]
        The distinct base frames across all tasks, in first-seen order.
    body_rows: np.ndarray, shape (num_tasks,)
        The index into `base_frames` for each task.
    """
    base_stations: np.ndarray
    mobod_indexes: np.ndarray
    base_frames: list
    body_rows: np.ndarray


@dataclass(frozen=True)
class FrameGeometry(TaskGeometry):
    """
    Task geometry information for model frames.

    Attributes
    ----------
    base_relative_rotations: np.ndarray, shape (num_tasks, 3, 3)
        Each frame's rotation relative to its base frame. A frame's rotation in ground
        is ``R_GB @ R_BF``, so one transform read per base frame yields every frame's
        orientation.
    """
    base_relative_rotations: np.ndarray


@dataclass(frozen=True)
class ParameterMap:
    """
    Task information related to body scales and placement offsets.

    Attributes
    ----------
    scaled_rows, scaled_groups: np.ndarray
        The body scales and scale group for each task.
    scaled_base_stations: np.ndarray, shape (len(scaled_rows), 3)
        The unscaled base stations of tasks to which body scales are applied.
    shifted_rows, shifted_groups: np.ndarray
        The placement offsets and offset groups for each task.
    """
    scaled_rows: np.ndarray
    scaled_groups: np.ndarray
    scaled_base_stations: np.ndarray
    shifted_rows: np.ndarray
    shifted_groups: np.ndarray

    @classmethod
    def from_groups(cls, scale_groups: np.ndarray, offset_groups: np.ndarray,
                    base_stations: np.ndarray) -> 'ParameterMap':
        scaled_rows = np.flatnonzero(scale_groups >= 0)
        shifted_rows = np.flatnonzero(offset_groups >= 0)
        return cls(scaled_rows=scaled_rows, scaled_groups=scale_groups[scaled_rows],
                   scaled_base_stations=base_stations[scaled_rows],
                   shifted_rows=shifted_rows,
                   shifted_groups=offset_groups[shifted_rows])


@dataclass(frozen=True)
class MarkerReference:
    """
    The reference data and weights for tracked markers.

    Attributes
    ----------
    positions: np.ndarray, shape (num_tasks, 3)
        Marker reference positions.
    weights: np.ndarray, shape (num_tasks,)
        Marker tracking cost weights.
    """
    positions: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class FrameReference:
    """
    The reference data and weights for tracked frames.

    Attributes
    ----------
    positions: np.ndarray, shape (num_tasks, 3)
        Frame reference positions.
    rotations: np.ndarray, shape (num_tasks, 3, 3)
        Frame reference orientations stored as rotation matrices.
    position_weights, orientation_weights: np.ndarray, shape (num_tasks,)
        Frame position and orientation tracking cost weights.
    """
    positions: np.ndarray
    rotations: np.ndarray
    position_weights: np.ndarray
    orientation_weights: np.ndarray


class TrackingTerm(ABC):
    """
    A base class for tracking cost terms: a set of registered tasks (markers or frames)
    and the scalar error between the model and their reference data, along with its
    Jacobian with respect to the model's generalized coordinates and, optionally, its
    body scales, placement offsets, and joint-level parameters.

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.

    Attributes
    ----------
    offset_input: str
        The `CostInput` field holding this term's placement offsets.
    """
    offset_input: ClassVar[str]

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
        self._geometry = None
        self._parameter_map = None
        self._reference = None
        self._station_array = None

    @abstractmethod
    def calc_error(self, state: osim.State) -> float:
        pass

    @abstractmethod
    def calc_jacobian(self, state: osim.State,
                      inputs: tuple[str, ...]) -> dict[str, np.ndarray]:
        """
        Return the error's Jacobian as a ``(1, n)`` block for each `CostInput` field
        named in `inputs`. The block of an input the error does not depend on (e.g.,
        ``'frame_offsets'`` for marker tasks) is zero.
        """

    def _add_parameter_jacobians(self, state: osim.State,
                                 jacobian: dict[str, np.ndarray],
                                 gradient: np.ndarray, R_GB: np.ndarray) -> None:
        """
        Fill the parameter blocks of `jacobian`, given each task's error gradient with
        respect to its ground position, `gradient`, and the rotation of its base frame
        in ground, `R_GB`.
        """
        # The sensitivity of each task's ground position to a shift of its station in
        # its base frame.
        doffset = (gradient[:, None, :] @ R_GB)[:, 0, :]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GS_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GS.get(i)   # for each task i on body k
        #
        accumulated = np.zeros((self.mc.num_mobod, 3))
        np.add.at(accumulated, self.geometry.mobod_indexes, gradient)
        dp_GB = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(accumulated, dtype=float).reshape(-1))

        # Calculate the position-error Jacobian with respect to body scales, including
        # the contributions from scaling each task's station within its base frame.
        parameter_map = self.parameter_map
        if 'body_scales' in jacobian:
            Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)
            if len(parameter_map.scaled_rows):
                np.add.at(Js.reshape(-1, 3), parameter_map.scaled_groups,
                          parameter_map.scaled_base_stations
                          * doffset[parameter_map.scaled_rows])
            jacobian['body_scales'] = Js

        # Calculate the position-error Jacobians with respect to the joint-level
        # parameters.
        if 'ellipsoid_radii_scales' in jacobian:
            jacobian['ellipsoid_radii_scales'] = (
                self.mc.calc_position_jacobian_wrt_ellipsoid_radii_scales(state, dp_GB))
        if 'beam_length_scales' in jacobian:
            jacobian['beam_length_scales'] = (
                self.mc.calc_position_jacobian_wrt_beam_length_scales(state, dp_GB))

        # Assemble the offset Jacobian from the offset sensitivities.
        if self.offset_input in jacobian and len(parameter_map.shifted_rows):
            np.add.at(jacobian[self.offset_input].reshape(-1, 3),
                      parameter_map.shifted_groups,
                      doffset[parameter_map.shifted_rows])

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
        self._invalidate_cached_arrays()

    def _invalidate_cached_arrays(self) -> None:
        self._geometry = None
        self._parameter_map = None
        self._reference = None

    @property
    def geometry(self) -> TaskGeometry:
        """
        The registered tasks' `TaskGeometry`, built on first use.
        """
        if self._geometry is None:
            self._geometry = self._build_geometry()
        return self._geometry

    @property
    def parameter_map(self) -> ParameterMap:
        """
        The registered tasks' `ParameterMap`, built on first use.
        """
        if self._parameter_map is None:
            scale_groups = np.array(
                [-1 if cache.body_scale_group_index is None
                 else cache.body_scale_group_index for cache in self.station_caches],
                dtype=int)
            offset_groups = np.array(
                [-1 if g is None else g for g in self.offset_group_indexes], dtype=int)
            self._parameter_map = ParameterMap.from_groups(
                scale_groups, offset_groups, self.geometry.base_stations)
        return self._parameter_map

    @property
    def reference(self):
        """
        The registered tasks' reference data and weights, built on first use.
        """
        if self._reference is None:
            self._reference = self._build_reference()
        return self._reference

    def _geometry_fields(self) -> dict:
        num_tasks = self.num_tasks
        base_stations = (np.asarray(self.base_stations, dtype=float).reshape(-1, 3)
                         if num_tasks else np.zeros((0, 3)))
        mobod_indexes = np.array(
            [int(self.mobod_indexes.getElt(i)) for i in range(num_tasks)], dtype=int)

        # One entry per distinct base frame, in first-seen order.
        frames, rows, row_of_mobod = [], [], {}
        for cache in self.station_caches:
            key = cache.mobod_index
            if key not in row_of_mobod:
                row_of_mobod[key] = len(frames)
                frames.append(cache.base_frame)
            rows.append(row_of_mobod[key])

        return dict(base_stations=base_stations, mobod_indexes=mobod_indexes,
                    base_frames=frames, body_rows=np.array(rows, dtype=int))

    def _build_geometry(self) -> TaskGeometry:
        return TaskGeometry(**self._geometry_fields())

    def _reference_positions(self) -> np.ndarray:
        return (np.asarray(self.positions, dtype=float).reshape(-1, 3)
                if self.num_tasks else np.zeros((0, 3)))

    @abstractmethod
    def _build_reference(self):
        pass

    @property
    def station_array(self) -> np.ndarray:
        cached = self._station_array
        if cached is None or len(cached) != self.num_tasks:
            cached = self._station_array = self.geometry.base_stations.copy()
        return cached

    def apply_state(self, body_scales: np.ndarray, offsets: np.ndarray) -> None:
        base_stations = self.geometry.base_stations
        parameter_map = self.parameter_map
        if len(parameter_map.scaled_rows) or len(parameter_map.shifted_rows):
            stations = base_stations.copy()
            if len(parameter_map.scaled_rows):
                scales = np.asarray(body_scales, dtype=float).reshape(-1, 3)
                stations[parameter_map.scaled_rows] *= scales[
                    parameter_map.scaled_groups]
            if len(parameter_map.shifted_rows):
                shifts = np.asarray(offsets, dtype=float).reshape(-1, 3)
                stations[parameter_map.shifted_rows] += shifts[
                    parameter_map.shifted_groups]
        else:
            stations = base_stations
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
    offset_input = 'marker_offsets'

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

    def _build_reference(self) -> MarkerReference:
        return MarkerReference(positions=self._reference_positions(),
                               weights=np.asarray(self.weights, dtype=float))

    def calc_error(self, state) -> float:
        if self.num_tasks == 0:
            return 0.0
        geometry, reference = self.geometry, self.reference
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in geometry.base_frames])
        rows = geometry.body_rows
        # p_GS = p_GB + R_GB · p_BS
        p_GB = poses[:, :, 3][rows]
        R_GB = poses[:, :, :3][rows]
        positions = (p_GB + (R_GB @ self.station_array[..., None])[..., 0])
        residuals = positions - reference.positions
        return float(np.sum(reference.weights * np.sum(residuals * residuals, axis=1)))

    def calc_jacobian(self, state, inputs) -> dict[str, np.ndarray]:
        jacobian = {name: np.zeros((1, self.mc.num_variables(name)))
                    for name in inputs}
        if self.num_tasks == 0:
            return jacobian

        geometry, reference = self.geometry, self.reference
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in geometry.base_frames])
        rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = geometry.body_rows

        # The per-marker error gradient in Ground.
        R_GB = rotations[rows]
        p_GS = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        gradient = 2.0 * reference.weights[:, None] * (p_GS - reference.positions)

        # Calculate the Jacobian of the position error with respect to the coordinates.
        if 'coordinates' in jacobian:
            dp_GS = osim.VectorVec3.createFromMat(
                np.ascontiguousarray(gradient, dtype=float).reshape(-1))
            grad_u = osim.Vector(state.getNU(), 0.0)
            self.mc.model.multiplyByStationJacobianTranspose(
                state, self.mobod_indexes, self.stations, dp_GS, grad_u)
            grad_q = osim.Vector(state.getNQ(), 0.0)
            self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
            jacobian['coordinates'][0, :] = (
                grad_q.to_numpy()[self.mc.coordinate_q_indexes])

        if set(inputs) - {'coordinates'}:
            self._add_parameter_jacobians(state, jacobian, gradient, R_GB)
        return jacobian


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
    offset_input = 'frame_offsets'

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

    def _build_geometry(self) -> FrameGeometry:
        # Keep each Transform alive while reading its rotation: R() returns a reference
        # into the Transform, which a chained call would let Python free first.
        base_relative_rotations = []
        for frame in self.frames:
            transform = frame.findTransformInBaseFrame()
            base_relative_rotations.append(transform.R().to_numpy())
        base_relative_rotations = (np.array(base_relative_rotations) if self.num_tasks
                                   else np.zeros((0, 3, 3)))
        return FrameGeometry(**self._geometry_fields(),
                             base_relative_rotations=base_relative_rotations)

    def _build_reference(self) -> FrameReference:
        rotations = (
            np.array([osim.Rotation(osim.Quaternion(*[float(v) for v in q])).to_numpy()
                      for q in self.orientations])
            if self.num_tasks else np.zeros((0, 3, 3)))
        return FrameReference(
            positions=self._reference_positions(), rotations=rotations,
            position_weights=np.asarray(self.position_weights, dtype=float),
            orientation_weights=np.asarray(self.orientation_weights, dtype=float))

    def calc_error(self, state) -> float:
        if self.num_tasks == 0:
            return 0.0
        geometry, reference = self.geometry, self.reference
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in geometry.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = geometry.body_rows
        R_GB = base_rotations[rows]

        # Positions come from the scaled and offset stations applied by apply_state,
        # so the error stays consistent with any applied offsets.
        positions = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        residuals = positions - reference.positions
        position_error = reference.position_weights * np.sum(
            residuals * residuals, axis=1)

        # trace(R_ref^T R) == 1 + 2cos(theta), so
        # (3 - trace(R_ref^T R)) / 4 == (1 - cos(theta)) / 2 == sin(theta/2)**2,
        # which is equal to the quaternion alignment loss 1 - (eps . q_ref)**2.
        frame_rotations = R_GB @ geometry.base_relative_rotations
        trace = np.einsum('nij,nij->n', reference.rotations, frame_rotations)
        orientation_error = reference.orientation_weights * (3.0 - trace) / 4.0

        return float(np.sum(position_error + orientation_error))

    def calc_jacobian(self, state, inputs) -> dict[str, np.ndarray]:
        jacobian = {name: np.zeros((1, self.mc.num_variables(name)))
                    for name in inputs}
        if self.num_tasks == 0:
            return jacobian

        # Compute the combined position and orientation error for every frame at
        # once.
        geometry, reference = self.geometry, self.reference
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in geometry.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = geometry.body_rows
        R_GB = base_rotations[rows]

        # The frames' ground positions come from their (possibly offset) cached
        # stations, so the gradient is consistent with any applied offsets.
        positions = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        gradient = 2.0 * reference.position_weights[:, None] * (
            positions - reference.positions)

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
        frame_rotations = R_GB @ geometry.base_relative_rotations
        aligned = np.einsum('nij,nkj->nik', frame_rotations, reference.rotations)
        orientation_error = (-0.25 * reference.orientation_weights)[:, None] * np.stack(
            [aligned[:, 1, 2] - aligned[:, 2, 1],
             aligned[:, 2, 0] - aligned[:, 0, 2],
             aligned[:, 0, 1] - aligned[:, 1, 0]], axis=1)

        # Calculate the frame (position and orientation) error Jacobian with respect to
        # the coordinates. A SpatialVec holds the angular half first, then the linear
        # half.
        if 'coordinates' in jacobian:
            spatialError = osim.VectorOfSpatialVec.createFromMat(
                np.ascontiguousarray(
                    np.concatenate([orientation_error, gradient], axis=1),
                    dtype=float).reshape(-1))
            grad_u = osim.Vector(state.getNU(), 0.0)
            self.mc.model.multiplyByFrameJacobianTranspose(
                state, self.mobod_indexes, self.stations, spatialError, grad_u)
            grad_q = osim.Vector(state.getNQ(), 0.0)
            self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
            jacobian['coordinates'][0, :] = (
                grad_q.to_numpy()[self.mc.coordinate_q_indexes])

        if set(inputs) - {'coordinates'}:
            self._add_parameter_jacobians(state, jacobian, gradient, R_GB)
        return jacobian


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
