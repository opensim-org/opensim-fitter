import unittest
from abc import abstractmethod
from pathlib import Path

import numpy as np
import casadi as ca
import opensim as osim

from osimfit.callbacks import Function
from osimfit.utilities import get_coordinate_indexes


class ScaleCallback(Function):
    """
    Base for bilevel test callbacks: two CasADi inputs — model coordinates
    of length ``len(q_indexes)`` and per-body scale factors flattened to
    length ``3 * NB`` — and one scalar output. Subclasses implement
    ``_eval`` / ``_jac_eval`` for the specific cost being checked.
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

    def pack_scales(self, arg) -> osim.VectorVec3:
        """
        Build a VectorVec3 of size ``NB + 1``: index 0 is ground (held at
        1.0), indices ``1..NB`` carry the per-body scale factors in body-set
        order from ``arg[1]``.
        """
        nb = self.model.getNumBodies()
        scales = osim.VectorVec3(nb + 1, osim.Vec3(1.0))
        for ib in range(nb):
            scales[ib + 1] = osim.Vec3(
                *arg[1][3*ib:3*ib + 3].full().flatten())
        return scales

    def _get_num_inputs(self):
        return 2

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return len(self.q_indexes)
        if i == 1:
            return 3 * self.model.getNumBodies()
        raise IndexError(f'Invalid input index {i}.')

    def _get_output_size(self, i):
        if i == 0:
            return 1
        raise IndexError(f'Invalid output index {i}.')

    @abstractmethod
    def _eval(self, arg):
        pass

    @abstractmethod
    def _jac_eval(self, arg):
        pass


class ScaledPositionErrorCallback(ScaleCallback):
    def __init__(self, name, model, frame_path, reference, opts={}):
        ScaleCallback.__init__(self, name, model, opts)
        frame = osim.PhysicalOffsetFrame.safeDownCast(
            model.getComponent(frame_path))
        transform = frame.findTransformInBaseFrame()

        self.matter = self.model.getMatterSubsystem()
        self.frame = frame
        self.mobod_index = frame.getMobilizedBodyIndex()
        self.station = osim.Vec3(transform.p())
        self.reference = reference

        # Grouped-form arrays for the Simbody operators, matching the pattern
        # used in src/osimfit/callbacks.py.
        self.mobod_indexes = osim.SimTKArrayMobilizedBodyIndex()
        self.mobod_indexes.push_back(self.mobod_index)
        self.stations = osim.SimTKArrayVec3()
        self.stations.push_back(self.station)

    def _eval(self, arg):
        self.apply_state(arg)
        scales = self.pack_scales(arg)
        position = self.matter.calcScaledStationPosition(
            self.state, self.mobod_index, self.station, scales)
        return [np.square(
            np.linalg.norm(position.to_numpy() - self.reference))]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        scales = self.pack_scales(arg)
        p = self.matter.calcScaledStationPosition(
            self.state, self.mobod_index, self.station, scales)

        # Pre-double the error so subsequent Simbody calls return the
        # Jacobian of ``||p - p_ref||^2`` directly.
        f_GS = osim.VectorVec3(1, osim.Vec3(0))
        f_GS.set(0, osim.Vec3(
            2.0 * (p[0] - self.reference[0]),
            2.0 * (p[1] - self.reference[1]),
            2.0 * (p[2] - self.reference[2])))

        # Jacobian w.r.t. model coordinates.
        vec = osim.Vector(self.state.getNQ(), 0.0)
        self.matter.multiplyByScaledStationJacobianTranspose(
            self.state, scales, self.mobod_indexes, self.stations, f_GS, vec)
        Jq = np.expand_dims(vec.to_numpy()[self.q_indexes], axis=0)

        # Jacobian w.r.t. body scale factors.
        vecVec3 = osim.VectorVec3(
            self.model.getNumBodies() + 1, osim.Vec3(0))
        self.matter.multiplyByStationJacobianWrtBodyScalesTranspose(
            self.state, self.mobod_indexes, self.stations, f_GS, vecVec3)
        Js = np.zeros((1, 3 * self.model.getNumBodies()))
        for ib in range(self.model.getNumBodies()):
            Js[0, 3*ib:3*ib + 3] = vecVec3.get(ib + 1).to_numpy()

        return [Jq, Js]


# Unit tests
# ----------
MODEL_FPATH = str(Path(__file__).parent / 'unscaled_generic.osim')
FRAME_PATHS = ['/bodyset/pelvis/pelvis',
               '/bodyset/torso/torso',
               '/jointset/hip_r/femur_r_offset/r_thigh',
               '/jointset/walker_knee_r/tibia_r_offset/r_shank',
               '/jointset/ankle_r/talus_r_offset/r_foot',
               '/jointset/mtp_r/toes_r_offset/r_toes']


class TestPositionErrorJacobians(unittest.TestCase):
    def test_position_error_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        model.initSystem()
        reference = np.array([0.1, 0.2, 0.3])

        for frame_path in FRAME_PATHS:
            f_fd = ScaledPositionErrorCallback(
                'f_fd', model, frame_path, reference, {'enable_fd': True})
            f_jac = ScaledPositionErrorCallback(
                'f_jac', model, frame_path, reference)

            q = ca.SX.sym('q', len(f_fd.q_indexes))
            s = ca.SX.sym('s', 3 * model.getNumBodies())
            x = ca.vertcat(q, s)

            J_fd = ca.Function('J_fd', [x], [ca.jacobian(f_fd(q, s), x)])
            J_jac = ca.Function('J_jac', [x], [ca.jacobian(f_jac(q, s), x)])

            self.assertTrue(np.allclose(
                J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestScaledPosition(unittest.TestCase):
    def test_scaled_position(self):
        model = osim.Model(MODEL_FPATH)
        model.initSystem()

        def set_default_positions(state):
            """
            Apply a non-trivial pose so the scaled / scale-and-rebuild paths
            actually disagree if the operator is wrong.
            """
            nq = state.getNQ()
            for iq in range(nq):
                state.updQ()[iq] = 0.01 * iq

        # First scale (index 0) is always 1.0, corresponding to ground.
        nb = model.getNumBodies()
        scales = osim.VectorVec3(nb + 1, osim.Vec3(1.0))
        for ib in range(nb):
            scales[ib + 1] = osim.Vec3(
                1.0 + 0.01*ib, 1.0 - 0.01*ib, 1.0 + 0.02*ib)

        # For each frame, verify that the position computed from the unscaled
        # system via SimbodyMatterSubsystem::calcScaledStationPosition matches
        # the position computed from a freshly-scaled model.
        for frame_path in FRAME_PATHS:

            # Load fresh so prior-iteration scaling doesn't carry over.
            model = osim.Model(MODEL_FPATH)
            state = model.initSystem()
            matter = model.getMatterSubsystem()

            frame = osim.PhysicalOffsetFrame.safeDownCast(
                model.getComponent(frame_path))
            mobod_index = frame.getMobilizedBodyIndex()
            transform = frame.findTransformInBaseFrame()
            station = osim.Vec3(transform.p())
            set_default_positions(state)
            model.realizePosition(state)
            position = matter.calcScaledStationPosition(
                state, mobod_index, station, scales)

            # Scale the model. The order of the scale factors in `scales` does
            # not necessarily match body-set order, so index by each body's
            # underlying MobilizedBodyIndex.
            scaleset = osim.ScaleSet()
            for ib in range(nb):
                body = model.getBodySet().get(ib)
                mobod_idx = int(body.getMobilizedBodyIndex())
                segment = body.getName()
                scale = osim.Scale()
                scale.setSegmentName(segment)
                scale.setScaleFactors(scales.get(mobod_idx))
                scaleset.cloneAndAppend(scale)
                scaleset.get(scaleset.getSize() - 1).setName(segment)
            model.scale(state, scaleset, True)

            model.finalizeConnections()
            state = model.initSystem()
            matter = model.getMatterSubsystem()
            set_default_positions(state)
            model.realizePosition(state)
            frame = osim.PhysicalOffsetFrame.safeDownCast(
                model.getComponent(frame_path))
            position_scaled_model = frame.getPositionInGround(state)

            self.assertTrue(np.allclose(
                position.to_numpy(),
                position_scaled_model.to_numpy(), atol=1e-6))
