import numpy as np
import opensim as osim
from abc import ABC, abstractmethod

from ..model import ModelCache
from .tasks import FrameTasks, MarkerTasks


class TrackingTerm(ABC):
    """
    A base class for tracking cost terms that compute a scalar error and its
    Jacobian with respect to the model's generalized coordinates, body scales,
    and other optimization variables. To implement a new tracking cost term, extend this
    class and implement the abstract methods (calc_error, calc_jacobian) to compute the
    error and its Jacobian.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_error(self, state: osim.State, **kwargs) -> float:
        pass

    @abstractmethod
    def calc_jacobian(self, state: osim.State, **kwargs) -> list[np.ndarray]:
        pass


class FrameTrackingTerm(FrameTasks, TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model frames'
    positions and orientations and corresponding reference data as a function of the
    model's generalized coordinates. Individual frames are registered via add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        positions = origins[rows] + (R_GB @ arrays.base_stations[..., None])[..., 0]
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

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_q_indexes)))]

        # Compute the "spatial error" (i.e., the combined position and orientation
        # error) for every frame at once.
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        positions = origins[rows] + (R_GB @ arrays.base_stations[..., None])[..., 0]
        position_error = arrays.double_position_weights_column * (
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
                np.concatenate([orientation_error, position_error], axis=1),
                dtype=float).reshape(-1))

        # Calculate the frame (position and orientation) error Jacobian.
        Ju = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, Ju)
        Jq = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, Ju, Jq)

        return [np.expand_dims(Jq.to_numpy()[self.mc.coordinate_q_indexes], axis=0)]


class MarkerTrackingTerm(MarkerTasks, TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model markers'
    positions and corresponding reference positions as a function of the model's
    generalized coordinates. Individual markers are registered via add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rows = arrays.body_rows
        # p_GS = p_GB + R_GB · p_BS
        p_GB = poses[:, :, 3][rows]
        R_GB = poses[:, :, :3][rows]
        positions = (p_GB + (R_GB @ arrays.base_stations[..., None])[..., 0])
        residuals = positions - arrays.reference_positions
        return float(np.sum(arrays.weights * np.sum(residuals * residuals, axis=1)))

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_q_indexes)))]

        # Initialize the array used to calculate the position error Jacobian via the
        # grouped Simbody operator.
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rows = arrays.body_rows
        # p_GS = p_GB + R_GB · p_BS
        p_GB = poses[:, :, 3][rows]
        R_GB = poses[:, :, :3][rows]
        positions = (p_GB + (R_GB @ arrays.base_stations[..., None])[..., 0])
        gradient = arrays.double_weights_column * (
            positions - arrays.reference_positions)
        f_GP = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(gradient, dtype=float).reshape(-1))

        # Calculate the position error Jacobian.
        Ju = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, f_GP, Ju)
        Jq = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, Ju, Jq)

        return [np.expand_dims(Jq.to_numpy()[self.mc.coordinate_q_indexes], axis=0)]


class BilevelTerm(TrackingTerm):
    """
    An intermediate base class that provides functionality common to bilevel cost terms.

    Applying body scales and placement offsets to the cached station locations is shared
    across marker and frame terms; subclasses (via `MarkerTasks`/`FrameTasks`) supply the
    per-task `station_caches`, `stations`, and `offset_group_indexes`.
    """
    def __init__(self):
        super().__init__()
        self._station_array = None

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


class MarkerBilevelTerm(MarkerTasks, BilevelTerm):
    """
    A tracking cost term that computes the aggregate error between model markers' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual markers are registered via
    add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
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

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_q_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.marker_offset_groups)))
        Jr = np.zeros((1, 3 * len(self.mc.ellipsoid_radii_scale_groups)))
        Jl = np.zeros((1, len(self.mc.beam_length_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo, Jr, Jl]

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

        # The sensitivity of each marker's ground position to a shift from an offset
        # variable.
        doffset = (gradient[:, None, :] @ R_GB)[:, 0, :]

        # Calculate the Jacobian of the position error with respect to the coordinates.
        grad_u = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, dp_GS, grad_u)
        grad_q = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
        Jq[0, :] = grad_q.to_numpy()[self.mc.coordinate_q_indexes]

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


class FrameBilevelTerm(FrameTasks, BilevelTerm):
    """
    A tracking cost term that computes the aggregate error between model frames' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual frames are registered via
    add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
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

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_q_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.frame_offset_groups)))
        Jr = np.zeros((1, 3 * len(self.mc.ellipsoid_radii_scale_groups)))
        Jl = np.zeros((1, len(self.mc.beam_length_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo, Jr, Jl]

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

        # Sensitivity of each frame's ground position to a base-frame station shift.
        doffset = np.einsum('ni,nij->nj', gradient, R_GB)

        # Calculate the frame (position and orientation) error Jacobian.
        grad_u = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, grad_u)
        grad_q = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
        Jq[0, :] = grad_q.to_numpy()[self.mc.coordinate_q_indexes]

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
