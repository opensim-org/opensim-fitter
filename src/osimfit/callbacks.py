import opensim as osim
import casadi as ca
import numpy as np
from abc import ABC, abstractmethod

# Base Callback Classes
# ---------------------
class Callback(ca.Callback, ABC):
    def __init__(self, name, model, coordinate_indexes, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.initSystem()
        self.matter = self.model.getMatterSubsystem()
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

    @abstractmethod
    def _get_num_inputs(self):
        pass

    @abstractmethod
    def _get_num_outputs(self):
        pass

    @abstractmethod
    def _eval(self, arg):
        pass


class JacobianCallback(Callback, ABC):
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
    def _jac_eval(self, arg):
        pass


# Tracking Cost Callbacks
# -----------------------
class TrackingCostMixin:
    def _init_tracking_cost(self, model, frame_paths, positions, quaternions, weights):
        # Frames.
        self.frame_paths = frame_paths
        self.frames = []
        self.stations = []
        self.mobod_indexes = []
        for frame_path in self.frame_paths:
            frame = osim.PhysicalOffsetFrame.safeDownCast(
                model.getComponent(frame_path))
            self.frames.append(frame)
            transform = frame.findTransformInBaseFrame()
            self.stations.append(osim.Vec3(transform.p()))
            self.mobod_indexes.append(frame.getMobilizedBodyIndex())

        # Convert the position data to a numpy array.
        self.positions = np.zeros((3, positions.size()))
        for i in range(positions.size()):
            self.positions[:, i] = positions[i].to_numpy()

        # Convert the quaternion data to a numpy array.
        self.quaternions = np.zeros((4, quaternions.size()))
        for i in range(quaternions.size()):
            quaternion = quaternions.getElt(0, i)
            self.quaternions[0, i] = quaternion.get(0)
            self.quaternions[1, i] = quaternion.get(1)
            self.quaternions[2, i] = quaternion.get(2)
            self.quaternions[3, i] = quaternion.get(3)

        # Cost function weights.
        self.weights = weights

    def _get_num_inputs(self):
        return len(self.coordinate_indexes)

    def _get_num_outputs(self):
        return 1

    def _calc_quaternion(self, frame):
        rotation = frame.getRotationInGround(self.state)
        quaternion = rotation.convertRotationToQuaternion() # 40 flops
        eps = np.array([quaternion.get(0), quaternion.get(1),
                        quaternion.get(2), quaternion.get(3)])
        return eps

    def _calc_errors(self):
        position_errors = []
        orientation_errors = []
        for i, frame in enumerate(self.frames):
            # Compute the position error as the norm of the difference between model and
            # data positions.
            position = frame.getPositionInGround(self.state).to_numpy()
            position_errors.append(np.square(
                    np.linalg.norm(position - self.positions[:,i])))

            # Get a quaternion representation of the current model frame's orientation
            # with respect to ground.
            eps = self._calc_quaternion(frame)

            # Compute the quaternion distance.
            # See section 2 in docs/frame_error_jacobians.pdf for details.
            error = 1 - np.square(np.dot(eps, self.quaternions[:, i]))
            orientation_errors.append(error)

        return position_errors, orientation_errors

    def _eval(self, arg):
        # Apply the optimization variables to the SimTK::State.
        self.apply_state(arg)

        # Compute the position and orientation errors, and return the weighted sum of
        # squared errors.
        position_errors, orientation_errors = self._calc_errors()
        return [
                self.weights['position']    * np.sum(position_errors) +
                self.weights['orientation'] * np.sum(orientation_errors)
                ]


class TrackingCostCallback(TrackingCostMixin, Callback):
    def __init__(self, name, model, coordinate_indexes, frame_paths, positions,
                 quaternions, weights, opts={}):
        Callback.__init__(self, name, model, coordinate_indexes, opts)
        self._init_tracking_cost(model, frame_paths, positions, quaternions, weights)


class TrackingCostJacobianCallback(TrackingCostMixin, JacobianCallback):
    def __init__(self, name, model, coordinate_indexes, frame_paths, positions,
                 quaternions, weights, opts={}):
        JacobianCallback.__init__(self, name, model, coordinate_indexes, opts)
        self._init_tracking_cost(model, frame_paths, positions, quaternions, weights)

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

    def _calc_frame_error_jacobian(self, frame, station, mobod_index, position,
                                   quaternion):
        # Position error Jacobian.
        # ------------------------
        # Compute the position error, error = p_GF - p_DG.
        error = frame.getPositionInGround(self.state)
        error[0] -= position[0]
        error[1] -= position[1]
        error[2] -= position[2]

        # This is size NQ and not len(coordinate_indexes) because
        # multiplyByStationJacobianTranspose() returns the Jacobian for all coordinates,
        # including dependent coordinates. We will index out the dependent coordinates
        # elements of the Jacobian before returning it.
        vec = osim.Vector(self.state.getNQ(), 0.0)

        # Compute the Jacobian of the position error. See section 4 in
        # docs/frame_error_jacobians.pdf for details.
        self.matter.multiplyByStationJacobianTranspose(self.state, mobod_index,
                                                       station, error, vec)
        J_p = self.weights['position'] * 2.0 * vec.to_numpy()

        # Orientation error Jacobian.
        # --------------------------
        # Calculate the quaternion representation of the current model frame with
        # respect to ground.
        eps = self._calc_quaternion(frame)

        # Relate the time derivative of the quaternion to the angular velocity.
        # See section 5 in docs/frame_error_jacobians.pdf for details.
        jac_eps = self._calc_quaternion_jacobian(eps)
        w = jac_eps.T.dot(quaternion)

        # Pack the angular velocity into a SpatialVec with zero linear velocity.
        spatial_vec = osim.SpatialVec(osim.Vec3(w[0], w[1], w[2]), osim.Vec3(0))

        # This is size NQ and not len(coordinate_indexes) because
        # multiplyByStationJacobianTranspose() returns the Jacobian for all coordinates,
        # including dependent coordinates. We will index out the dependent coordinates
        # elements of the Jacobian before returning it.
        vec = osim.Vector(self.state.getNQ(), 0.0)

        # Compute the Jacobian of the orientation error. See section 5
        # in docs/frame_error_jacobians.pdf for details.
        self.matter.multiplyByFrameJacobianTranspose(self.state, mobod_index,
                                                     station, spatial_vec, vec)
        J_R = self.weights['orientation'] * -2.0*(np.dot(eps, quaternion))*vec.to_numpy()

        # Return the sum of the position and orientation error Jacobians for this frame.
        return J_p + J_R

    def _jac_eval(self, arg):
        # Apply the optimal coordinates to the SimTK::State.
        self.apply_state(arg)

        # Compute the Jacobian of the tracking cost by summing the Jacobians of the
        # position and orientation errors for each frame.
        # See section 6 in docs/frame_error_jacobians.pdf for details.
        J = np.zeros((self.state.getNQ()))
        for i, frame in enumerate(self.frames):
            J_i = self._calc_frame_error_jacobian(frame, self.stations[i],
                                                 self.mobod_indexes[i],
                                                 self.positions[:,i],
                                                 self.quaternions[:,i])
            J += J_i

        # Index out the dependent coordinates and return the Jacobian.
        return [np.expand_dims(J[self.coordinate_indexes], axis=0)]
