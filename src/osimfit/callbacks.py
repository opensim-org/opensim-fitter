import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from .utilities import get_coordinate_indexes


#########
# COSTS #
#########


class TrackingCost(ABC):
    """
    A base class for tracking cost functions that compute a scalar error and its
    Jacobian with respect to the model's generalized coordinates, body scale factors,
    and other optimization variables. To implement a new tracking cost, extend this 
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


class FrameTrackingCost(TrackingCost):
    """
    A tracking cost that computes the aggregate error between model frames' positions 
    and orientations and corresponding reference data as a function of the model's 
    generalized coordinates. Individual frames are registered via add_frame().

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    """
    def __init__(self, model: osim.Model):
        self.model = model
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        self.matter = model.getMatterSubsystem()

        self.frames = []
        self.mobod_indexes = osim.SimTKArrayMobilizedBodyIndex()
        self.stations = osim.SimTKArrayVec3()
        self.positions = []
        self.orientations = []
        self.position_weights = []
        self.orientation_weights = []

    def add_frame(self, frame_path: str, position: osim.Vec3, 
                  orientation: osim.Quaternion, position_weight: float = 1.0,
                  orientation_weight: float = 1.0):
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
        position_weight: float
            (Optional) The cost weight for the position error. Default: 1.0.
        orientation_weight: float
            (Optional) The cost weight for the orientation error. Default: 1.0.
        """
        if not self.model.hasComponent(frame_path):
            raise ValueError(f'Model does not have a component at path {frame_path}.')
        if position_weight < 0:
            raise ValueError(f'Expected position_weight to be non-negative, but got '
                             f'{position_weight}.')
        if orientation_weight < 0:
            raise ValueError(f'Expected orientation_weight to be non-negative, but got '
                             f'{orientation_weight}.')

        frame = osim.PhysicalFrame.safeDownCast(self.model.getComponent(frame_path))
        self.frames.append(frame)
        self.mobod_indexes.push_back(frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(frame.findTransformInBaseFrame().p()))
        self.positions.append(position.to_numpy())
        self.orientations.append(
            np.array([orientation.get(i) for i in range(4)]))
        self.position_weights.append(position_weight)
        self.orientation_weights.append(orientation_weight)

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for i, frame in enumerate(self.frames):
            p_model = frame.getPositionInGround(state).to_numpy()
            position_error = self.position_weights[i] * np.square(
                np.linalg.norm(p_model - self.positions[i]))

            eps = self._calc_quaternion(state, frame)
            orientation_error = self.orientation_weights[i] * (
                1.0 - np.square(np.dot(eps, self.orientations[i])))

            error += position_error + orientation_error
        return error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        nb = self.mobod_indexes.size()
        if nb == 0:
            return [np.zeros((1, len(self.q_indexes)))]

        # Inialize arrays used to calculate position and orientation error Jacobians 
        # via the grouped Simbody operators.
        spatialError = osim.VectorOfSpatialVec(nb, osim.SpatialVec(0))
        for i, frame in enumerate(self.frames):
            wp = self.position_weights[i]
            wo = self.orientation_weights[i]

            # Position error
            p_model = frame.getPositionInGround(state)
            p_error = osim.Vec3(
                2.0 * wp * (p_model[0] - self.positions[i][0]),
                2.0 * wp * (p_model[1] - self.positions[i][1]),
                2.0 * wp * (p_model[2] - self.positions[i][2]))

            # Orientation error
            eps = self._calc_quaternion(state, frame)
            jac_eps = self._calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ self.orientations[i]
            scale = wo * -2.0 * np.dot(eps, self.orientations[i])
            w_error = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(w_error, p_error))

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.matter.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        J = vec.to_numpy()

        return [np.expand_dims(J[self.q_indexes], axis=0)]

    def _calc_quaternion(self, state, frame):
        rotation = frame.getRotationInGround(state)
        quaternion = rotation.convertRotationToQuaternion()
        return np.array([quaternion.get(i) for i in range(4)])

    def _calc_quaternion_jacobian(self, eps):
        # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
        e = 0.5 * eps
        return np.array([
            [-e[1], -e[2], -e[3]],
            [ e[0],  e[3], -e[2]],
            [-e[3],  e[0],  e[1]],
            [ e[2], -e[1],  e[0]],
        ])


class MarkerTrackingCost(TrackingCost):
    """
    A tracking cost that computes the aggregate error between model markers' positions 
    and corresponding reference positions as a function of the model's generalized 
    coordinates. Individual markers are registered via add_marker().

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    """
    def __init__(self, model: osim.Model):
        self.model = model
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        self.matter = model.getMatterSubsystem()

        self.markers = []
        self.mobod_indexes = osim.SimTKArrayMobilizedBodyIndex()
        self.stations = osim.SimTKArrayVec3()
        self.positions = []
        self.weights = []

    def add_marker(self, marker_path: str, position: osim.Vec3, weight: float = 1.0):
        """
        Register a marker to track.

        Parameters
        ----------
        marker_path: str
            The OpenSim Model path to the tracking marker.
        position: osim.Vec3
            The reference position data tracked by the model marker.
        weight: float
            (Optional) The cost weight for the position error. Default: 1.0.
        """
        if not self.model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')

        marker = osim.Marker.safeDownCast(self.model.getComponent(marker_path))
        frame = osim.PhysicalFrame.safeDownCast(
            marker.getParentFrame().findBaseFrame())
        state = self.model.getWorkingState()
        self.model.realizePosition(state)

        self.markers.append(marker)
        self.mobod_indexes.push_back(frame.getMobilizedBodyIndex())
        self.stations.push_back(marker.findLocationInFrame(state, frame))
        self.positions.append(position.to_numpy())
        self.weights.append(weight)

    def calc_error(self, state, **kwargs) -> float:
        error = 0.0
        for marker, position, weight in zip(
                self.markers, self.positions, self.weights):
            p_model = marker.getLocationInGround(state).to_numpy()
            error += weight * np.square(np.linalg.norm(p_model - position))
        return error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        nb = self.mobod_indexes.size()
        if nb == 0:
            return [np.zeros((1, len(self.q_indexes)))]

        # Inialize the array used to calculate the position error Jacobian via the
        # grouped Simbody operator.
        f_GP = osim.VectorVec3(nb, osim.Vec3(0))
        for i, (marker, position, weight) in enumerate(
                zip(self.markers, self.positions, self.weights)):
            p_model = marker.getLocationInGround(state)
            f_GP.set(i, osim.Vec3(
                2.0 * weight * (p_model[0] - position[0]),
                2.0 * weight * (p_model[1] - position[1]),
                2.0 * weight * (p_model[2] - position[2])))

        # Calculate the position error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.matter.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, f_GP, vec)

        return [np.expand_dims(vec.to_numpy()[self.q_indexes], axis=0)]


class MarkerBilevelCost(TrackingCost):
    """
    A tracking cost that computes the aggregate error between model markers' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scale factors. Individual markers are registered
    via add_marker().

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    scale_groups: list[list[int]]
        A list of mobilized-body-index groups. Each inner list is a group of
        bodies that share one optimized 3-vector scale factor.
    """
    def __init__(self, model: osim.Model, scale_groups: list[list[int]]):
        self.model = model
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        self.scale_groups = scale_groups
        self.matter = model.getMatterSubsystem()

        self.mobod_indexes = osim.SimTKArrayMobilizedBodyIndex()
        self.stations = osim.SimTKArrayVec3()
        self.positions = []
        self.weights = []

    def add_marker(self, marker_path: str, position: osim.Vec3, weight: float = 1.0):
        """
        Register a marker to track.

        Parameters
        ----------
        marker_path: str
            The OpenSim Model path to the tracking marker.
        position: osim.Vec3
            The reference position data tracked by the model marker.
        weight: float
            (Optional) The cost weight for the position error. Default: 1.0.
        """
        if not self.model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')

        marker = osim.Marker.safeDownCast(self.model.getComponent(marker_path))
        frame = osim.PhysicalFrame.safeDownCast(
            marker.getParentFrame().findBaseFrame())
        state = self.model.getWorkingState()

        self.mobod_indexes.push_back(frame.getMobilizedBodyIndex())
        self.stations.push_back(marker.findLocationInFrame(state, frame))
        self.positions.append(position.to_numpy())
        self.weights.append(weight)

    def calc_error(self, state, **kwargs) -> float:
        scales = kwargs['scales']
        nb = self.mobod_indexes.size() 
        if nb == 0:
            return 0.0

        # Calculate position errors across all markers.
        p_GS = osim.VectorVec3(nb, osim.Vec3(0))
        self.matter.calcScaledStationPosition(state, scales, self.mobod_indexes,
                                              self.stations, p_GS)

        error = 0.0
        for i, (position, weight) in enumerate(zip(self.positions, self.weights)):
            p = p_GS.get(i).to_numpy()
            error += weight * np.square(np.linalg.norm(p - position))
        return error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        scales = kwargs['scales']
        nb = self.mobod_indexes.size()
        if nb == 0:
            return [np.zeros((1, len(self.q_indexes))),
                    np.zeros((1, 3*len(self.scale_groups)))]

        # Calculate scaled station positions acrss all markers
        p_GS = osim.VectorVec3(nb, osim.Vec3(0))
        self.matter.calcScaledStationPosition(state, scales, self.mobod_indexes,
                                              self.stations, p_GS)

        # Position errors
        f_GS = osim.VectorVec3(nb, osim.Vec3(0))
        for i, (position, weight) in enumerate(zip(self.positions, self.weights)):
            p = p_GS.get(i)
            f_GS.set(i, osim.Vec3(
                2.0 * weight * (p[0] - position[0]),
                2.0 * weight * (p[1] - position[1]),
                2.0 * weight * (p[2] - position[2])))

        # Calculate the error Jacobian with respect to the model coordinates.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.matter.multiplyByScaledStationJacobianTranspose(
            state, scales, self.mobod_indexes, self.stations, f_GS, vec)
        Jq = np.expand_dims(vec.to_numpy()[self.q_indexes], axis=0)

        # Calculate the error Jacobian with respect to body scale factors.
        # For a shared scalar s_i applied to every body k in group G_i, the
        # chain rule gives dE/ds_i = sum_{k in G_i} dE/ds_k, so the per-group
        # Jacobian column is the sum of the Simbody per-body sensitivities.
        vecVec3 = osim.VectorVec3(self.model.getNumBodies() + 1, osim.Vec3(0))
        self.matter.multiplyByStationJacobianWrtBodyScalesTranspose(
            state, self.mobod_indexes, self.stations, f_GS, vecVec3)
        Js = np.zeros((1, 3*len(self.scale_groups)))
        for i, group in enumerate(self.scale_groups):
            col = np.zeros(3)
            for k in group:
                col += vecVec3.get(k).to_numpy()
            Js[0, 3*i:3*(i+1)] = col

        return [Jq, Js]


class Function(ca.Callback, ABC):
    """
    A base class for CasADi callback functions that evaluate the function and its
    Jacobian using OpenSim. To implement a new callback, extend this class and implement
    the abstract methods to define the number of inputs and outputs and provide the
    function evaluation and its Jacobian.

    Parameters
    ----------
    name: str
        The name of the callback function.
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    opts: dict
        A dictionary of options to pass to the CasADi callback constructor.
    """
    def __init__(self, name: str, model: osim.Model, opts: dict = {}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.getWorkingState()
        self.enable_fd = opts.get("enable_fd", False)
        self.construct(name, opts)

    def get_n_in(self): return self._get_num_inputs()
    def get_n_out(self): return self._get_num_outputs()

    def get_input_size(self, i):
        return self._get_input_size(i)

    def get_output_size(self, i):
        return self._get_output_size(i)

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.get_input_size(i), 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.get_output_size(i), 1)

    def eval(self, arg):
        return self._eval(arg)

    def has_jacobian(self): return not self.enable_fd

    def get_jacobian(self, name, inames, onames, opts):
        class JacobianFunction(ca.Callback):
            def __init__(self, callback, opts={}):
                ca.Callback.__init__(self)
                self.callback = callback
                self.construct(name, opts)

            def get_n_in(self):
                return self.callback.get_n_in() + self.callback.get_n_out()
            def get_n_out(self):
                return self.callback.get_n_in()

            def get_sparsity_in(self,i):
                if i < self.callback.get_n_in():
                    return ca.Sparsity.dense(self.callback.get_input_size(i), 1)
                elif i < self.callback.get_n_in() + self.callback.get_n_out():
                    iout = i - self.callback.get_n_in()
                    return ca.Sparsity.dense(self.callback.get_output_size(iout), 1)
                else:
                    return ca.Sparsity.dense(0, 0)

            def get_sparsity_out(self,i):
                iin = i % self.callback.get_n_in()
                iout = i // self.callback.get_n_in()
                return ca.Sparsity.dense(self.callback.get_output_size(iout),
                                         self.callback.get_input_size(iin))

            def eval(self, arg):
                return self.callback._jac_eval(arg)

        self.jacobian_callback = JacobianFunction(self)
        return self.jacobian_callback

    @abstractmethod
    def _get_num_inputs(self):
        pass

    @abstractmethod
    def _get_num_outputs(self):
        pass

    @abstractmethod
    def _get_input_size(self, i):
        pass

    @abstractmethod
    def _get_output_size(self, i):
        pass

    @abstractmethod
    def _eval(self, arg):
        pass

    @abstractmethod
    def _jac_eval(self, arg):
        pass


##################
# COST FUNCTIONS #
##################

class TrackingCostFunction(Function):
    """
    A CasADi callback that evaluates the sum of tracking costs over a set of model
    frames and markers with respect to the model's generalized coordinates.

    Parameters
    ----------
    name: str
        The name of the callback function.
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    opts: dict
        A dictionary of options to pass to the CasADi callback constructor.
    """
    def __init__(self, name: str, model: osim.Model, opts={}):
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        Function.__init__(self, name, model, opts=opts)
        self.marker_cost = MarkerTrackingCost(model)
        self.frame_cost = FrameTrackingCost(model)

    def apply_state(self, arg):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(self.state.getNQ())
        q[self.q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.model.realizePosition(self.state)

    def add_frame_tracking_cost(self, frame_path: str,
                                position: osim.Vec3,
                                orientation: osim.Quaternion,
                                position_weight: float = 1.0,
                                orientation_weight: float = 1.0):
        self.frame_cost.add_frame(frame_path, position, orientation,
                                  position_weight=position_weight,
                                  orientation_weight=orientation_weight)

    def add_marker_tracking_cost(self, marker_path: str, position: osim.Vec3,
                                 weight: float = 1.0):
        self.marker_cost.add_marker(marker_path, position, weight=weight)

    def _get_num_inputs(self):
        return 1

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return len(self.q_indexes)
        else:
            raise IndexError(f'Invalid input index {i} for TrackingCostFunction.')

    def _get_output_size(self, i):
        if i == 0:
            return 1
        else:
            raise IndexError(f'Invalid output index {i} for TrackingCostFunction.')

    def _eval(self, arg):
        self.apply_state(arg)
        error = (self.marker_cost.calc_error(self.state) +
                 self.frame_cost.calc_error(self.state))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        J = (self.marker_cost.calc_jacobian(self.state)[0] +
             self.frame_cost.calc_jacobian(self.state)[0])
        return [J]


class BilevelCostFunction(Function):
    """
    A CasADi callback that evaluates the sum of tracking costs over a set of model
    markers with respect to the model's generalized coordinates and a set of body scale
    factors.

    Parameters
    ----------
    name: str
        The name of the callback function.
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    scale_groups: list[list[int]]
        A list of mobilized-body-index groups. Each inner list is a group of
        bodies sharing one optimized 3-vector scale factor.
    opts: dict
        A dictionary of options to pass to the CasADi callback constructor.
    """
    def __init__(self, name: str, model: osim.Model,
                 scale_groups: list[list[int]], opts={}):
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        self.scale_groups = scale_groups
        Function.__init__(self, name, model, opts=opts)
        self.marker_cost = MarkerBilevelCost(model, scale_groups)

    def apply_state(self, arg):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(self.state.getNQ())
        q[self.q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.model.realizePosition(self.state)

    def pack_scales(self, arg) -> osim.VectorVec3:

        # Define a VectorVec3 of scale factors, where the scale factors for bodies not
        # being optimized are set to 1.0. This vector includes scale factors for all
        # bodies in the model, including the ground body at index 0.
        scales = osim.VectorVec3(self.model.getNumBodies()+1, osim.Vec3(1.0))

        # Broadcast each optimized 3-vector to every body in its group, so that
        # bodies sharing one scale factor receive identical per-body scales.
        for i, group in enumerate(self.scale_groups):
            s_vec = osim.Vec3(*arg[1][3*i:3*i+3].full().flatten())
            for mobod_index in group:
                scales[mobod_index] = s_vec

        return scales

    def add_marker_bilevel_cost(self, marker_path: str, position: osim.Vec3,
                                weight: float = 1.0):
        self.marker_cost.add_marker(marker_path, position, weight=weight)

    def _get_num_inputs(self):
        return 2

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return len(self.q_indexes)
        elif i == 1:
            return 3 * len(self.scale_groups)
        else:
            raise IndexError(f'Invalid input index {i} for BilevelCostFunction.')

    def _get_output_size(self, i):
        if i == 0:
            return 1
        else:
            raise IndexError(f'Invalid output index {i} for BilevelCostFunction.')

    def _eval(self, arg):
        self.apply_state(arg)
        scales = self.pack_scales(arg)
        error = self.marker_cost.calc_error(self.state, scales=scales)
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        scales = self.pack_scales(arg)
        jac = self.marker_cost.calc_jacobian(self.state, scales=scales)
        return [jac[0], jac[1]]
