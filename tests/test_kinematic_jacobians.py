import unittest
from pathlib import Path

import numpy as np
import casadi as ca
import opensim as osim

from osimfit.callbacks import Function
from osimfit.utilities import get_coordinate_indexes


class KinematicFunction(Function):
    """
    Base for kinematics test callbacks: a single CasADi input of length
    ``len(q_indexes)``. Subclasses provide the output sizing and the eval
    / Jacobian-eval implementations.
    """
    def __init__(self, name: str, model: osim.Model, opts: dict = {}):
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        Function.__init__(self, name, model, opts=opts)

    def apply_state(self, arg):
        """
        Apply the input coordinates to the model state and realize to position.
        """
        q = np.zeros(self.state.getNQ())
        q[self.q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.model.realizePosition(self.state)

    def _get_num_inputs(self):
        return 1

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return len(self.q_indexes)
        raise IndexError(f'Invalid input index {i}.')


class PositionCallback(KinematicFunction):
    def __init__(self, name, model, body_name, opts={}):
        KinematicFunction.__init__(self, name, model, opts)
        self.body = model.getBodySet().get(body_name)
        self.mobod_index = self.body.getMobilizedBodyIndex()
        self.matter = self.model.getMatterSubsystem()

    def _get_output_size(self, i):
        if i == 0:
            return 3
        raise IndexError(f'Invalid output index {i}.')

    def _eval(self, arg):
        self.apply_state(arg)
        position = self.body.getPositionInGround(self.state).to_numpy()
        return [position]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        matrix = osim.Matrix()
        self.matter.calcStationJacobian(
            self.state, self.mobod_index, osim.Vec3(0), matrix)
        return [matrix.to_numpy()[:, self.q_indexes]]


class PositionErrorCallback(KinematicFunction):
    def __init__(self, name, model, frame_path, reference, opts={}):
        KinematicFunction.__init__(self, name, model, opts)
        frame = osim.PhysicalOffsetFrame.safeDownCast(
            model.getComponent(frame_path))
        transform = frame.findTransformInBaseFrame()

        self.matter = self.model.getMatterSubsystem()
        self.frame = frame
        self.mobod_index = frame.getMobilizedBodyIndex()
        self.station = osim.Vec3(transform.p())
        self.reference = reference

        # Grouped-form arrays for the Simbody operator, matching the pattern
        # used in src/osimfit/callbacks.py.
        self.mobod_indexes = osim.SimTKArrayMobilizedBodyIndex()
        self.mobod_indexes.push_back(self.mobod_index)
        self.stations = osim.SimTKArrayVec3()
        self.stations.push_back(self.station)

    def _get_output_size(self, i):
        if i == 0:
            return 1
        raise IndexError(f'Invalid output index {i}.')

    def _eval(self, arg):
        self.apply_state(arg)
        position = self.frame.getPositionInGround(self.state).to_numpy()
        error = np.square(np.linalg.norm(position - self.reference))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        p = self.frame.getPositionInGround(self.state)
        f_GP = osim.VectorVec3(1, osim.Vec3(0))
        f_GP.set(0, osim.Vec3(
            2.0 * (p[0] - self.reference[0]),
            2.0 * (p[1] - self.reference[1]),
            2.0 * (p[2] - self.reference[2])))

        vec = osim.Vector(self.state.getNQ(), 0.0)
        self.matter.multiplyByStationJacobianTranspose(
            self.state, self.mobod_indexes, self.stations, f_GP, vec)
        return [np.expand_dims(vec.to_numpy()[self.q_indexes], axis=0)]


class OrientationCallback(KinematicFunction):
    def __init__(self, name, model, body, opts={}):
        KinematicFunction.__init__(self, name, model, opts)
        self.body = model.getBodySet().get(body)
        self.mobod_index = self.body.getMobilizedBodyIndex()
        self.matter = self.model.getMatterSubsystem()

    def _get_output_size(self, i):
        if i == 0:
            return 4
        raise IndexError(f'Invalid output index {i}.')

    def _calc_quaternion(self):
        rotation = self.body.getRotationInGround(self.state)
        quaternion = rotation.convertRotationToQuaternion()
        return np.array([quaternion.get(0), quaternion.get(1),
                         quaternion.get(2), quaternion.get(3)])

    def _eval(self, arg):
        self.apply_state(arg)
        return [self._calc_quaternion()]

    def _calc_quaternion_jacobian(self):
        e = 0.5 * self._calc_quaternion()
        # SimbodyMatterSubsystem::calcFrameJacobian() returns a Jacobian that
        # maps generalized speeds to the spatial velocity of a frame:
        # [omega; vdot]. Account for the linear velocity component, vdot, too.
        #
        # Simbody -> SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
        #
        # [e_dot; vdot] = J_eps * [omega; vdot]
        return np.array([
            [-e[1], -e[2], -e[3],  0.0,  0.0,  0.0],
            [ e[0],  e[3], -e[2],  0.0,  0.0,  0.0],
            [-e[3],  e[0],  e[1],  0.0,  0.0,  0.0],
            [ e[2], -e[1],  e[0],  0.0,  0.0,  0.0],
            [  0.0,   0.0,   0.0,  1.0,  0.0,  0.0],
            [  0.0,   0.0,   0.0,  0.0,  1.0,  0.0],
            [  0.0,   0.0,   0.0,  0.0,  0.0,  1.0],
        ])

    def _calc_frame_jacobian(self):
        jac_frame = osim.Matrix()
        self.matter.calcFrameJacobian(
            self.state, self.mobod_index, osim.Vec3(0), jac_frame)
        return jac_frame.to_numpy()

    def _jac_eval(self, arg):
        self.apply_state(arg)
        jac_eps = self._calc_quaternion_jacobian()
        jac_frame = self._calc_frame_jacobian()
        jac = jac_eps.dot(jac_frame)
        return [jac[0:4, self.q_indexes]]


class OrientationErrorCallback(KinematicFunction):
    def __init__(self, name, model, frame_path, reference, opts={}):
        KinematicFunction.__init__(self, name, model, opts)
        frame = osim.PhysicalOffsetFrame.safeDownCast(
            model.getComponent(frame_path))
        transform = frame.findTransformInBaseFrame()
        self.frame = frame
        self.matter = self.model.getMatterSubsystem()
        self.mobod_index = frame.getMobilizedBodyIndex()
        self.station = osim.Vec3(transform.p())
        self.reference = reference

        self.mobod_indexes = osim.SimTKArrayMobilizedBodyIndex()
        self.mobod_indexes.push_back(self.mobod_index)
        self.stations = osim.SimTKArrayVec3()
        self.stations.push_back(self.station)

    def _get_output_size(self, i):
        if i == 0:
            return 1
        raise IndexError(f'Invalid output index {i}.')

    def _calc_quaternion(self, arg):
        self.apply_state(arg)
        rotation = self.frame.getRotationInGround(self.state)
        quaternion = rotation.convertRotationToQuaternion()
        return np.array([quaternion.get(0), quaternion.get(1),
                         quaternion.get(2), quaternion.get(3)])

    def _eval(self, arg):
        eps = self._calc_quaternion(arg)
        error = 1.0 - np.square(np.dot(eps, self.reference))
        return [error]

    def _calc_quaternion_jacobian(self, eps):
        # Simbody -> SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
        e = 0.5 * eps
        return np.array([
            [-e[1], -e[2], -e[3]],
            [ e[0],  e[3], -e[2]],
            [-e[3],  e[0],  e[1]],
            [ e[2], -e[1],  e[0]],
        ])

    def _jac_eval(self, arg):
        self.apply_state(arg)
        eps = self._calc_quaternion(arg)
        jac_eps = self._calc_quaternion_jacobian(eps)
        omega = jac_eps.T.dot(self.reference)
        scale = -2.0 * np.dot(eps, self.reference)
        w_error = osim.Vec3(
            scale * omega[0], scale * omega[1], scale * omega[2])
        spatial_error = osim.VectorOfSpatialVec(
            1, osim.SpatialVec(osim.Vec3(0), osim.Vec3(0)))
        spatial_error.set(0, osim.SpatialVec(w_error, osim.Vec3(0)))

        vec = osim.Vector(self.state.getNQ(), 0.0)
        self.matter.multiplyByFrameJacobianTranspose(
            self.state, self.mobod_indexes, self.stations, spatial_error, vec)
        return [np.expand_dims(vec.to_numpy()[self.q_indexes], axis=0)]


# Unit tests
# ----------
MODEL_FPATH = str(Path(__file__).parent / 'unscaled_generic.osim')
FRAME_PATHS = ['/bodyset/pelvis/pelvis',
               '/bodyset/torso/torso',
               '/jointset/hip_r/femur_r_offset/r_thigh',
               '/jointset/walker_knee_r/tibia_r_offset/r_shank',
               '/jointset/ankle_r/talus_r_offset/r_foot',
               '/jointset/mtp_r/toes_r_offset/r_toes']


class TestPositionJacobians(unittest.TestCase):
    def test_position_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        model.initSystem()

        bodyset = model.getBodySet()
        for ibody in range(bodyset.getSize()):
            body_name = bodyset.get(ibody).getName()

            f_fd = PositionCallback('f_fd', model, body_name,
                                    {'enable_fd': True})
            f_jac = PositionCallback('f_jac', model, body_name)

            x = ca.SX.sym('x', len(f_fd.q_indexes))
            J_fd = ca.Function('J_fd', [x], [ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac', [x], [ca.jacobian(f_jac(x), x)])

            self.assertTrue(np.allclose(
                J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestPositionErrorJacobians(unittest.TestCase):
    def test_position_error_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        model.initSystem()
        reference = np.array([0.1, 0.2, 0.3])

        for frame_path in FRAME_PATHS:
            f_fd = PositionErrorCallback('f_fd', model, frame_path,
                                         reference, {'enable_fd': True})
            f_jac = PositionErrorCallback('f_jac', model, frame_path,
                                          reference)

            x = ca.SX.sym('x', len(f_fd.q_indexes))
            J_fd = ca.Function('J_fd', [x], [ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac', [x], [ca.jacobian(f_jac(x), x)])

            self.assertTrue(np.allclose(
                J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestOrientationJacobians(unittest.TestCase):
    def test_orientation_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        model.initSystem()

        bodyset = model.getBodySet()
        for ibody in range(bodyset.getSize()):
            body_name = bodyset.get(ibody).getName()

            f_fd = OrientationCallback('f_fd', model, body_name,
                                       {'enable_fd': True})
            f_jac = OrientationCallback('f_jac', model, body_name)

            x = ca.SX.sym('x', len(f_fd.q_indexes))
            J_fd = ca.Function('J_fd', [x], [ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac', [x], [ca.jacobian(f_jac(x), x)])

            self.assertTrue(np.allclose(
                J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestOrientationErrorJacobians(unittest.TestCase):
    def test_orientation_error_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        model.initSystem()
        reference = np.array([1.0, 0.0, 0.0, 0.0])

        for frame_path in FRAME_PATHS:
            f_fd = OrientationErrorCallback('f_fd', model, frame_path,
                                            reference, {'enable_fd': True})
            f_jac = OrientationErrorCallback('f_jac', model, frame_path,
                                             reference)

            x = ca.SX.sym('x', len(f_fd.q_indexes))
            J_fd = ca.Function('J_fd', [x], [ca.jacobian(f_fd(x), x)])
            J_jac = ca.Function('J_jac', [x], [ca.jacobian(f_jac(x), x)])

            self.assertTrue(np.allclose(
                J_jac(2).full(), J_fd(2).full(), atol=1e-6))
