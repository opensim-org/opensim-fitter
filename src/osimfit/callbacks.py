from xml.parsers.expat import model

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
    Jacobian with respect to the model's generalized coordinates. To implement a new
    tracking cost, extend this class and implement the abstract methods (calc_error,
    calc_jacobian) to compute the error and its Jacobian.
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
    A tracking cost that computes the error between a model frame's position and
    orientation and corresponding position and orientation data as a function of the
    model's generalized coordinates.

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    frame_path: str
        The path to the frame in the model to track.
    position: osim.Vec3
        The position data to track.
    orientation: osim.Quaternion
        The orientation data to track, represented as a quaternion.
    position_weight: float
        The weight to apply to the position error in the total error.
    orientation_weight: float
        The weight to apply to the orientation error in the total error.
    """
    def __init__(self, model: osim.Model, frame_path: str, position: osim.Vec3,
                 orientation: osim.Quaternion, position_weight: float = 1.0,
                 orientation_weight: float = 1.0):

        self.frame_path = frame_path
        if not model.hasComponent(frame_path):
            raise ValueError(f'Model does not have a component at path {frame_path}.')

        self.frame = osim.PhysicalOffsetFrame.safeDownCast(
                model.getComponent(frame_path))
        self.matter = model.getMatterSubsystem()
        self.mobod_index = self.frame.getMobilizedBodyIndex()
        self.station = osim.Vec3(self.frame.findTransformInBaseFrame().p())

        # Cost weights.
        if position_weight < 0:
            raise ValueError(f'Expected position_weight to be non-negative, but got '
                             f'{position_weight}.')
        if orientation_weight < 0:
            raise ValueError(f'Expected orientation_weight to be non-negative, but got '
                             f'{orientation_weight}.')
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

        # Reference data.
        self.position = position.to_numpy()
        self.orientation = np.zeros(4)
        self.orientation[0] = orientation.get(0)
        self.orientation[1] = orientation.get(1)
        self.orientation[2] = orientation.get(2)
        self.orientation[3] = orientation.get(3)

    def calc_error(self, state, **kwargs) -> float:

        # Compute the position error as the norm of the difference between model and
        # data positions.
        position = self.frame.getPositionInGround(state).to_numpy()
        position_error = np.square(np.linalg.norm(position - self.position))

        # Get a quaternion representation of the current model frame's orientation
        # with respect to ground.
        eps = self._calc_quaternion(state)

        # Compute the quaternion distance.
        # See section 2 in docs/frame_error_jacobians.pdf for details.
        orientation_error = 1 - np.square(np.dot(eps, self.orientation))

        # Compute the weight sum of the position and orientation errors.
        error = self.position_weight * position_error + \
                self.orientation_weight * orientation_error

        return error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:

        # Position error Jacobian.
        # ------------------------
        # Compute the position error, error = p_GF - p_DG.
        error = self.frame.getPositionInGround(state)
        error[0] -= self.position[0]
        error[1] -= self.position[1]
        error[2] -= self.position[2]

        # This is size NQ and not len(coordinate_indexes) because
        # multiplyByStationJacobianTranspose() returns the Jacobian for all coordinates,
        # including dependent coordinates. We will index out the dependent coordinates
        # elements of the Jacobian before returning it.
        vec = osim.Vector(state.getNQ(), 0.0)

        # Compute the Jacobian of the position error. See section 4 in
        # docs/frame_error_jacobians.pdf for details.
        self.matter.multiplyByStationJacobianTranspose(state, self.mobod_index,
                                                       self.station, error, vec)
        J_p = self.position_weight * 2.0 * vec.to_numpy()

        # Orientation error Jacobian.
        # --------------------------
        # Calculate the quaternion representation of the current model frame with
        # respect to ground.
        eps = self._calc_quaternion(state)

        # Relate the time derivative of the quaternion to the angular velocity.
        # See section 5 in docs/frame_error_jacobians.pdf for details.
        jac_eps = self._calc_quaternion_jacobian(eps)
        w = jac_eps.T.dot(self.orientation)

        # Pack the angular velocity into a SpatialVec with zero linear velocity.
        spatial_vec = osim.SpatialVec(osim.Vec3(w[0], w[1], w[2]), osim.Vec3(0))

        # This is size NQ and not len(coordinate_indexes) because
        # multiplyByStationJacobianTranspose() returns the Jacobian for all coordinates,
        # including dependent coordinates. We will index out the dependent coordinates
        # elements of the Jacobian before returning it.
        vec = osim.Vector(state.getNQ(), 0.0)

        # Compute the Jacobian of the orientation error. See section 5
        # in docs/frame_error_jacobians.pdf for details.
        self.matter.multiplyByFrameJacobianTranspose(state, self.mobod_index,
                                                     self.station, spatial_vec, vec)
        error = np.dot(eps, self.orientation)
        J_R = self.orientation_weight * -2.0*error*vec.to_numpy()

        # Return the sum of the position and orientation error Jacobians for this frame.
        return [J_p + J_R]

    def _calc_quaternion(self, state):
        """
        Get the quaternion representation of the model frame's orientation with respect
        to ground.
        """
        rotation = self.frame.getRotationInGround(state)
        quaternion = rotation.convertRotationToQuaternion()
        eps = np.array([quaternion.get(0), quaternion.get(1),
                        quaternion.get(2), quaternion.get(3)])
        return eps

    def _calc_quaternion_jacobian(self, eps):
        """
        Get the Jacobian that relates the time derivative of the quaternion to the
        angular velocity.
        """
        # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
        e = 0.5 * eps
        jac_eps = np.array([
            [-e[1], -e[2], -e[3]],
            [ e[0],  e[3], -e[2]],
            [-e[3],  e[0],  e[1]],
            [ e[2], -e[1],  e[0]],
        ])
        return jac_eps


class MarkerTrackingCost(TrackingCost):
    """
    A tracking cost that computes the error between a model marker's position and
    the position of an experimental marker as a function of the model's generalized
    coordinates.

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    marker_path: str
        The path to the marker in the model to track.
    position: osim.Vec3
        The position data to track.
    weight: float
        The weight to apply to the position error in the total error.
    """
    def __init__(self, model: osim.Model, marker_path: str, position: osim.Vec3,
                 weight: float = 1.0):

        self.marker_path = marker_path
        if not model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')

        self.marker = osim.Marker.safeDownCast(
                model.getComponent(marker_path))
        self.matter = model.getMatterSubsystem()

        frame = osim.PhysicalFrame.safeDownCast(
            self.marker.getParentFrame().findBaseFrame())
        state = model.getWorkingState()
        model.realizePosition(state)
        self.station = self.marker.findLocationInFrame(state, frame)
        self.mobod_index = frame.getMobilizedBodyIndex()

        # Cost weights.
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

        # Reference data.
        self.position = position.to_numpy()

    def calc_error(self, state, **kwargs) -> float:

        # Compute the position error as the norm of the difference between model and
        # data positions.
        position = self.marker.getLocationInGround(state).to_numpy()
        position_error = np.square(np.linalg.norm(position - self.position))

        return self.weight * position_error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:

        # Position error Jacobian.
        # ------------------------
        # Compute the position error, error = p_GF - p_DG.
        error = self.marker.getLocationInGround(state)
        error[0] -= self.position[0]
        error[1] -= self.position[1]
        error[2] -= self.position[2]

        # This is size NQ and not len(coordinate_indexes) because
        # multiplyByStationJacobianTranspose() returns the Jacobian for all coordinates,
        # including dependent coordinates. We will index out the dependent coordinates
        # elements of the Jacobian before returning it.
        vec = osim.Vector(state.getNQ(), 0.0)

        # Compute the Jacobian of the position error. See section 4 in
        # docs/frame_error_jacobians.pdf for details.
        self.matter.multiplyByStationJacobianTranspose(state, self.mobod_index,
                                                       self.station, error, vec)
        J = self.weight * 2.0 * vec.to_numpy()

        return [J]


class MarkerBilevelCost(TrackingCost):
    """
    A tracking cost that computes the error between a model marker's position and
    the position of an experimental marker as a function of the model's generalized
    coordinates and body scale factors.

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    marker_path: str
        The path to the marker in the model to track.
    position: osim.Vec3
        The position data to track.
    weight: float
        The weight to apply to the position error in the total error.
    """
    def __init__(self, model: osim.Model, marker_path: str, position: osim.Vec3,
                 weight: float = 1.0):

        self.model = model
        self.marker_path = marker_path
        if not model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')

        self.marker = osim.Marker.safeDownCast(
                model.getComponent(marker_path))
        self.matter = model.getMatterSubsystem()

        frame = osim.PhysicalFrame.safeDownCast(
            self.marker.getParentFrame().findBaseFrame())
        state = model.getWorkingState()
        self.station = self.marker.findLocationInFrame(state, frame)
        self.mobod_index = frame.getMobilizedBodyIndex()

        # Cost weights.
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

        # Reference data.
        self.position = position.to_numpy()

    def calc_error(self, state, **kwargs) -> float:

        # Compute the scaled position error as the norm of the difference between model
        # and data positions.
        scales = kwargs['scales']
        position = self.matter.calcScaledStationPosition(state,
                                                         scales,
                                                         self.mobod_index,
                                                         self.station)

        error = np.square(np.linalg.norm(position.to_numpy() - self.position))
        return self.weight * error

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:

        # Scaled position error Jacobian
        # -------------------------------
        # Compute the scaled position error.
        scales = kwargs['scales']
        error = self.matter.calcScaledStationPosition(state, scales, self.mobod_index,
                                                      self.station)
        error[0] -= self.position[0]
        error[1] -= self.position[1]
        error[2] -= self.position[2]

        # This is size NQ and not len(q_indexes) because
        # multiplyByScaledStationJacobianTranspose() returns the Jacobian for all
        # coordinates, including dependent coordinates. We will index out the dependent
        # coordinates elements of the Jacobian before returning it.
        vec = osim.Vector(state.getNQ(), 0.0)

        # Compute the Jacobian of the scaled position error with respect to the
        # coordinates.
        self.matter.multiplyByScaledStationJacobianTranspose(state, scales,
                                                             self.mobod_index,
                                                             self.station, error, vec)
        Jq = self.weight * 2.0 * vec.to_numpy()

        # Compute the Jacobian of the scaled position error with respect to the
        # body scale factors.
        vecVec3 = osim.VectorVec3(self.model.getNumBodies() + 1, osim.Vec3(0))
        self.matter.multiplyByStationJacobianWrtBodyScalesTranspose(state,
                                                                    self.mobod_index,
                                                                    self.station,
                                                                    error,
                                                                    vecVec3)

        # Flatten the VectorVec3 into a 2D array of shape (1, 3*num_scales) and apply
        # the cost weight.
        Js = np.zeros((1, 3*self.model.getNumBodies()))
        for ib in range(self.model.getNumBodies()):
            Js[0, 3*ib:3*ib+3] = vecVec3.get(ib+1).to_numpy()
        Js = self.weight * 2.0 * Js

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
    frames with respect to the model's generalized coordinates.

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
        self.tracking_costs: list[TrackingCost] = []

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
        self.tracking_costs.append(
                FrameTrackingCost(self.model,
                                  frame_path,
                                  position,
                                  orientation,
                                  position_weight=position_weight,
                                  orientation_weight=orientation_weight))

    def add_marker_tracking_cost(self, marker_path: str, position: osim.Vec3,
                                 weight: float = 1.0):
        self.tracking_costs.append(
                MarkerTrackingCost(self.model,
                                   marker_path,
                                   position,
                                   weight=weight))

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
        # Apply the optimization variables to the SimTK::State.
        self.apply_state(arg)

        # Compute the sum of the frame cost errors.
        error = 0
        for cost in self.tracking_costs:
            error += cost.calc_error(self.state)
        return [error]

    def _jac_eval(self, arg):
        # Apply the optimal coordinates to the SimTK::State.
        self.apply_state(arg)

        # Compute the Jacobian of the tracking cost by summing the Jacobians of the
        # position and orientation errors for each frame.
        # See section 6 in docs/frame_error_jacobians.pdf for details.
        J = np.zeros((self.state.getNQ()))
        for cost in self.tracking_costs:
            J += cost.calc_jacobian(self.state)[0]

        # Index out the dependent coordinates and return the Jacobian.
        return [np.expand_dims(J[self.q_indexes], axis=0)]


class BilevelCostFunction(Function):
    """
    A CasADi callback that evaluates the sum of tracking costs over a set of model
    frames with respect to the model's generalized coordinates and a set of body scale
    factors.

    Parameters
    ----------
    name: str
        The name of the callback function.
    model: osim.Model
        The OpenSim model to use for evaluating the function and its Jacobian.
    scale_indexes: list[int]
        A list of the mobilized body indexes corresponding to the body scale factors
        being optimized.
    opts: dict
        A dictionary of options to pass to the CasADi callback constructor.
    """
    def __init__(self, name: str, model: osim.Model, scale_indexes, opts={}):
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        self.scale_indexes = scale_indexes
        Function.__init__(self, name, model, opts=opts)
        self.tracking_costs: list[TrackingCost] = []

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

        # Set the scale factors from the optimization variables for the bodies being
        # optimized.
        for i, mobod_index in enumerate(self.scale_indexes):
            scales[mobod_index] = osim.Vec3(*arg[1][3*i:3*i+3].full().flatten())

        return scales

    def add_marker_bilevel_cost(self, marker_path: str, position: osim.Vec3,
                                 weight: float = 1.0):
        self.tracking_costs.append(MarkerBilevelCost(self.model,
                                                     marker_path,
                                                     position,
                                                     weight=weight))

    def _get_num_inputs(self):
        return 2

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return len(self.q_indexes)
        elif i == 1:
            return 3 * len(self.scale_indexes)
        else:
            raise IndexError(f'Invalid input index {i} for BilevelCostFunction.')

    def _get_output_size(self, i):
        if i == 0:
            return 1
        else:
            raise IndexError(f'Invalid output index {i} for BilevelCostFunction.')

    def _eval(self, arg):
        # Apply the optimization variables to the SimTK::State.
        self.apply_state(arg)
        scales = self.pack_scales(arg)

        # Compute the sum of the frame cost errors.
        error = 0
        for cost in self.tracking_costs:
            error += cost.calc_error(self.state, scales=scales)
        return [error]

    def _jac_eval(self, arg):
        # Apply the optimal coordinates to the SimTK::State.
        self.apply_state(arg)
        scales = self.pack_scales(arg)

        # Compute the Jacobian of the tracking cost by summing the Jacobians of the
        # position and orientation errors for each frame.
        # See section 6 in docs/frame_error_jacobians.pdf for details.
        Jq = np.zeros((self.state.getNQ()))
        Js = np.zeros((1, 3*len(self.scale_indexes)))
        for cost in self.tracking_costs:
            jac = cost.calc_jacobian(self.state, scales=scales)
            Jq += jac[0]
            Js_full = jac[1]  # shape (1, 3*num_bodies), indexed by mobod_index-1
            for k, mobod_index in enumerate(self.scale_indexes):
                ib = mobod_index - 1
                Js[0, 3*k:3*(k+1)] += Js_full[0, 3*ib:3*(ib+1)]

        # Index out the dependent coordinates and return the Jacobians.
        return [np.expand_dims(Jq[self.q_indexes], axis=0), Js]
