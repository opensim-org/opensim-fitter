import unittest
import numpy as np
import casadi as ca
import opensim as osim
from osimfit.callbacks import Callback

class PositionCallback(Callback):
    def __init__(self, name, model, body_name, opts={}):
        Callback.__init__(self, name, model, opts)
        self.body = model.getBodySet().get(body_name)
        self.mobod_index = self.body.getMobilizedBodyIndex()
        self.matter = self.model.getMatterSubsystem()

    def _get_num_inputs(self):
        return len(self.q_indexes)

    def _get_num_outputs(self):
        return 3

    def _eval(self, arg):
        self.apply_state(arg)
        position = self.body.getPositionInGround(self.state).to_numpy()
        return [position]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        matrix = osim.Matrix()
        self.matter.calcStationJacobian(self.state, self.mobod_index, osim.Vec3(0),
                                        matrix)
        return [matrix.to_numpy()[:, self.q_indexes]]


class PositionErrorCallback(Callback):
    def __init__(self, name, model, frame_path, reference, opts={}):
        Callback.__init__(self, name, model, opts)
        frame = osim.PhysicalOffsetFrame.safeDownCast(
            model.getComponent(frame_path))
        transform = frame.findTransformInBaseFrame()

        self.matter = self.model.getMatterSubsystem()
        self.frame = frame
        self.mobod_index = frame.getMobilizedBodyIndex()
        self.station = osim.Vec3(transform.p())
        self.reference = reference

    def _get_num_inputs(self):
        return len(self.q_indexes)

    def _get_num_outputs(self):
        return 1

    def _eval(self, arg):
        self.apply_state(arg)
        position = self.frame.getPositionInGround(self.state).to_numpy()
        error = np.square(np.linalg.norm(position - self.reference))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        error = self.frame.getPositionInGround(self.state)
        error[0] -= self.reference[0]
        error[1] -= self.reference[1]
        error[2] -= self.reference[2]

        vec = osim.Vector(self.get_num_inputs(), 0.0)
        self.matter.multiplyByStationJacobianTranspose(self.state, self.mobod_index,
                                                       self.station, error, vec)
        J = 2.0*vec.to_numpy()
        return [np.expand_dims(J[self.q_indexes], axis=0)]


class OrientationCallback(Callback):
    def __init__(self, name, model, body, opts={}):
        Callback.__init__(self, name, model, opts)
        self.body = model.getBodySet().get(body)
        self.mobod_index = self.body.getMobilizedBodyIndex()
        self.matter = self.model.getMatterSubsystem()

    def _get_num_inputs(self):
        return len(self.q_indexes)

    def _get_num_outputs(self):
        return 4

    def _calc_quaternion(self):
        rotation = self.body.getRotationInGround(self.state)
        quaternion = rotation.convertRotationToQuaternion()
        eps = np.array([quaternion.get(0), quaternion.get(1),
                        quaternion.get(2), quaternion.get(3)])
        return eps

    def _eval(self, arg):
        self.apply_state(arg)
        eps = self._calc_quaternion()
        return [eps]

    def _calc_quaternion_jacobian(self):
        e = 0.5*self._calc_quaternion()

        # SimbodyMatterSubsystem::calcFrameJacobian() returns a Jacobian that maps
        # generalized speeds to the spatial velocity of a frame: [omega; vdot]. So, this
        # Jacobian needs to account for the linear velocity component, vdot, as well.
        #
        # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
        #
        # [e_dot; vdot] = J_eps * [omega; vdot]
        # J = [ -e1 -e2 -e3   0   0   0
        #        e0  e3 -e2   0   0   0
        #       -e3  e0  e1   0   0   0
        #        e2 -e1  e0   0   0   0
        #         0   0   0   1   0   0
        #         0   0   0   0   1   0
        #         0   0   0   0   0   1 ]
        jac_eps = np.array([
            [-e[1], -e[2], -e[3],  0.0,  0.0,  0.0],
            [ e[0],  e[3], -e[2],  0.0,  0.0,  0.0],
            [-e[3],  e[0],  e[1],  0.0,  0.0,  0.0],
            [ e[2], -e[1],  e[0],  0.0,  0.0,  0.0],
            [  0.0,   0.0,   0.0,  1.0,  0.0,  0.0],
            [  0.0,   0.0,   0.0,  0.0,  1.0,  0.0],
            [  0.0,   0.0,   0.0,  0.0,  0.0,  1.0]
        ])
        return jac_eps

    def _calc_frame_jacobian(self):
        jac_frame = osim.Matrix()
        self.matter.calcFrameJacobian(self.state, self.mobod_index, osim.Vec3(0),
                                      jac_frame)
        return jac_frame.to_numpy()

    def _jac_eval(self, arg):
        self.apply_state(arg)
        jac_eps = self._calc_quaternion_jacobian()
        jac_frame = self._calc_frame_jacobian()
        jac = jac_eps.dot(jac_frame)
        return [jac[0:4, self.q_indexes]]


class OrientationErrorCallback(Callback):
    def __init__(self, name, model, frame_path, reference, opts={}):
        Callback.__init__(self, name, model, opts)
        frame = osim.PhysicalOffsetFrame.safeDownCast(
            model.getComponent(frame_path))
        transform = frame.findTransformInBaseFrame()
        self.frame = frame
        self.matter = self.model.getMatterSubsystem()
        self.mobod_index = frame.getMobilizedBodyIndex()
        self.station = osim.Vec3(transform.p())
        self.reference = reference

    def _get_num_inputs(self):
        return len(self.q_indexes)

    def _get_num_outputs(self):
        return 1

    def _calc_quaternion(self, arg):
        self.apply_state(arg)
        rotation = self.frame.getRotationInGround(self.state)
        quaternion = rotation.convertRotationToQuaternion()
        eps = np.array([quaternion.get(0), quaternion.get(1),
                        quaternion.get(2), quaternion.get(3)])
        return eps

    def _eval(self, arg):
        eps = self._calc_quaternion(arg)
        error = 1.0 - np.square(np.dot(eps, self.reference))
        return [error]

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

    def _jac_eval(self, arg):
        self.apply_state(arg)
        eps = self._calc_quaternion(arg)
        jac_eps = self._calc_quaternion_jacobian(eps)
        omega = jac_eps.T.dot(self.reference)
        spatial_vec = osim.SpatialVec(osim.Vec3(omega[0], omega[1], omega[2]),
                                      osim.Vec3(0))
        vec = osim.Vector(self.get_num_inputs(), 0.0)
        self.matter.multiplyByFrameJacobianTranspose(self.state, self.mobod_index,
                                                     self.station, spatial_vec, vec)
        J = -2.0*(np.dot(eps, self.reference))*vec.to_numpy()
        return [np.expand_dims(J[self.q_indexes], axis=0)]


# Unit tests
# ----------
MODEL_FPATH = 'unscaled_generic.osim'
FRAME_PATHS = ['/bodyset/pelvis/pelvis',
               '/bodyset/torso/torso',
               '/jointset/hip_r/femur_r_offset/r_thigh',
               '/jointset/walker_knee_r/tibia_r_offset/r_shank',
               '/jointset/ankle_r/talus_r_offset/r_foot',
               '/jointset/mtp_r/toes_r_offset/r_toes']


class TestPositionJacobians(unittest.TestCase):
    def test_position_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()

        bodyset = model.getBodySet()
        for ibody in range(bodyset.getSize()):
            body = bodyset.get(ibody)
            body_name = body.getName()

            # Callback functions.
            f_fd = PositionCallback('f_fd', model, body_name, {"enable_fd": True})
            f_jac = PositionCallback('f_jac', model, body_name)

            # Symbolic inputs.
            x = ca.SX.sym('x', len(f_fd.q_indexes))

            # Jacobian expression graphs.
            J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])

            # Test that the two Jacobians are equivalent.
            self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestPositionErrorJacobians(unittest.TestCase):
    def test_position_error_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        reference = np.array([0.1, 0.2, 0.3])

        for frame_path in FRAME_PATHS:
            f_fd = PositionErrorCallback('f_fd', model, frame_path, reference,
                                         {"enable_fd": True})
            f_jac = PositionErrorCallback('f_jac', model, frame_path, reference)

            # Symbolic inputs.
            x = ca.SX.sym('x', len(f_fd.q_indexes))

            # Jacobian expression graphs.
            J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])

            # Test that the two Jacobians are equivalent.
            self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestOrientationJacobians(unittest.TestCase):
    def test_orientation_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()

        bodyset = model.getBodySet()
        for ibody in range(bodyset.getSize()):
            body = bodyset.get(ibody)
            body_name = body.getName()

            # Callback functions.
            f_fd = OrientationCallback('f_fd', model, 'pelvis', {'enable_fd': True})
            f_jac = OrientationCallback('f_jac', model, 'pelvis')

            # Symbolic inputs.
            x = ca.SX.sym('x', len(f_fd.q_indexes))

            # Jacobian expression graphs.
            J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])

            # Test that the two Jacobians are equivalent.
            self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestOrientationErrorJacobians(unittest.TestCase):
    def test_orientation_error_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        reference = np.array([1.0, 0.0, 0.0, 0.0])

        for frame_path in FRAME_PATHS:
            f_fd = OrientationErrorCallback('f_fd', model, frame_path, reference,
                                            {"enable_fd": True})
            f_jac = OrientationErrorCallback('f_jac', model, frame_path, reference)

            # Symbolic inputs.
            x = ca.SX.sym('x', len(f_fd.q_indexes))

            # Jacobian expression graphs.
            J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])

            # Test that the two Jacobians are equivalent.
            self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))


# class TestTrackingCostErrorJacobians(unittest.TestCase):
#     def test_position_error_jacobians(self):
#         model = osim.Model(MODEL_FPATH)
#         state = model.initSystem()
#         weights = {'position': 2.0,
#                    'orientation': 0.3}
#         coordinates_map = get_coordinate_indexes(model, skip_dependent_coordinates=True)
#         coordinate_indexes = list(coordinates_map.values())

#         positions = osim.RowVectorVec3(len(FRAME_PATHS), osim.Vec3(0))
#         quaternions = osim.RowVectorQuaternion(len(FRAME_PATHS),
#                                                osim.Quaternion(1,0,0,0))
#         quaternions.setTo(osim.Quaternion(1,0,0,0))
#         f_fd = TrackingCostCallback('f_fd', model, coordinate_indexes, FRAME_PATHS,
#                                     positions, quaternions, weights,
#                                     {"enable_fd": True})
#         f_jac = TrackingCostJacobianCallback('f_jac', model, coordinate_indexes,
#                                              FRAME_PATHS, positions, quaternions,
#                                              weights)
#         # Symbolic inputs.
#         x = ca.SX.sym('x', len(coordinate_indexes))

#         # Jacobian expression graphs.
#         J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(x), x)])
#         J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(x), x)])

#         # Test that the two Jacobians are equivalent.
#         self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))