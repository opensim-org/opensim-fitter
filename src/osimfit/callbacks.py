from turtle import position

import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod

# Base Callback Classes
# ---------------------
class Callback(ca.Callback, ABC):
    def __init__(self, name, model, coordinate_indexes, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.getWorkingState()
        self.coordinate_indexes = coordinate_indexes
        self.construct(name, opts)

    def get_num_inputs(self):
        return self._get_num_inputs()
    def get_num_outputs(self):
        return self._get_num_outputs()

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(self.get_num_inputs(), 1)
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(self.get_num_outputs(), 1)

    def eval(self, arg):
        return self._eval(arg)

    def apply_state(self, arg):
        # Apply the input coordinates to the model state and realize the system to the
        # position stage.
        q = np.zeros(self.state.getNQ())
        q[self.coordinate_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.model.realizePosition(self.state)

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        class JacobianFunction(ca.Callback):
            def __init__(self, callback, opts={}):
                ca.Callback.__init__(self)
                self.callback = callback
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                if i == 0:
                    return ca.Sparsity.dense(self.callback.get_num_inputs(), 1)
                elif i == 1:
                    return ca.Sparsity.dense(self.callback.get_num_outputs(), 1)
            def get_sparsity_out(self,i):
                return ca.Sparsity.dense(self.callback.get_num_outputs(),
                                         self.callback.get_num_inputs())

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
    def _eval(self, arg):
        pass

    @abstractmethod
    def _jac_eval(self, arg):
        pass


# Data types
# ----------
class TrackingCost(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_error(self, state: osim.State) -> float:
        pass

    @abstractmethod
    def calc_jacobian(self, state: osim.State) -> np.ndarray:
        pass


class FrameTrackingCost:
    def __init__(self, model, frame_path, position, orientation, weights):
        self.frame_path = frame_path
        self.frame = osim.PhysicalOffsetFrame.safeDownCast(
                model.getComponent(frame_path))
        self.matter = model.getMatterSubsystem()
        self.mobod_index = self.frame.getMobilizedBodyIndex()
        self.station = osim.Vec3(self.frame.findTransformInBaseFrame().p())
        self.weights = weights
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.update_data(position, orientation)

    def update_data(self, position, orientation):
        self.position = position.to_numpy()
        self.orientation[0] = orientation.get(0)
        self.orientation[1] = orientation.get(1)
        self.orientation[2] = orientation.get(2)
        self.orientation[3] = orientation.get(3)

    def calc_error(self, state) -> float:

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
        error = self.weights['position']    * position_error + \
                self.weights['orientation'] * orientation_error

        return error

    def calc_jacobian(self, state) -> np.ndarray:

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
        J_p = self.weights['position'] * 2.0 * vec.to_numpy()

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
        J_R = self.weights['orientation'] * -2.0*error*vec.to_numpy()

        # Return the sum of the position and orientation error Jacobians for this frame.
        return J_p + J_R

    def _calc_quaternion(self, state):
        rotation = self.frame.getRotationInGround(state)
        quaternion = rotation.convertRotationToQuaternion()
        eps = np.array([quaternion.get(0), quaternion.get(1),
                        quaternion.get(2), quaternion.get(3)])
        return eps

    def _calc_quaternion_jacobian(self, eps):
        # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
        e = 0.5 * eps
        jac_eps = np.array([
            [-e[1], -e[2], -e[3]],
            [ e[0],  e[3], -e[2]],
            [-e[3],  e[0],  e[1]],
            [ e[2], -e[1],  e[0]],
        ])
        return jac_eps

# Tracking Cost Callbacks
# -----------------------
class TrackingCostCallback(Callback):
    def __init__(self, name, model, coordinate_indexes, frame_paths, positions,
                 orientations, weights, opts={}):
        Callback.__init__(self, name, model, coordinate_indexes, opts=opts)

        # Frames.
        self.frame_paths = frame_paths
        self.frame_costs: list[FrameTrackingCost] = []
        for iframe, frame_path in enumerate(self.frame_paths):
            frame_cost = FrameTrackingCost(model, frame_path,
                                           positions.getElt(0, iframe),
                                           orientations.getElt(0, iframe), weights)
            self.frame_costs.append(frame_cost)

    def update_data(self, positions, orientations):
        for iframe, frame_cost in enumerate(self.frame_costs):
            frame_cost.update_data(positions.getElt(0, iframe),
                                   orientations.getElt(0, iframe))

    def _get_num_inputs(self):
        return len(self.coordinate_indexes)

    def _get_num_outputs(self):
        return 1

    def _eval(self, arg):
        # Apply the optimization variables to the SimTK::State.
        self.apply_state(arg)

        # Compute the sum of the frame cost errors.
        error = 0
        for frame_cost in self.frame_costs:
            error += frame_cost.calc_error(self.state)
        return [error]

    def _jac_eval(self, arg):
        # Apply the optimal coordinates to the SimTK::State.
        self.apply_state(arg)

        # Compute the Jacobian of the tracking cost by summing the Jacobians of the
        # position and orientation errors for each frame.
        # See section 6 in docs/frame_error_jacobians.pdf for details.
        J = np.zeros((self.state.getNQ()))
        for frame_cost in self.frame_costs:
            J += frame_cost.calc_jacobian(self.state)

        # Index out the dependent coordinates and return the Jacobian.
        return [np.expand_dims(J[self.coordinate_indexes], axis=0)]
