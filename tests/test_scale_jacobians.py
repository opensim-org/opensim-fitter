import unittest
import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from osimfit.utilities import get_coordinate_indexes


class ScaleCallback(ca.Callback, ABC):
    def __init__(self, name: str, model: osim.Model, opts: dict = {}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.getWorkingState()
        self.q_indexes = list(get_coordinate_indexes(
            model, skip_dependent_coordinates=True).values())
        self.enable_fd = opts.get("enable_fd", False)
        self.construct(name, opts)

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
        nb = self.model.getNumBodies()

        # The first scale is always 1.0 (corresponding to the ground).
        scales = osim.VectorVec3(nb+1, osim.Vec3(1.0))
        for ib in range(nb):
            scales[ib+1] = osim.Vec3(*arg[1][3*ib:3*ib+3].full().flatten())
        return scales

    def get_num_coordinates(self):
        return self._get_num_coordinates()

    def get_num_scales(self):
        return self._get_num_scales()

    def get_num_outputs(self):
        return self._get_num_outputs()

    def get_n_in(self): return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return ca.Sparsity.dense(self.get_num_coordinates(), 1)
        elif i == 1:
            return ca.Sparsity.dense(self.get_num_scales(), 1)
        else:
            return ca.Sparsity(0, 0)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.get_num_outputs(), 1)

    def eval(self, arg):
        return self._eval(arg)

    def has_jacobian(self): return not self.enable_fd

    def get_jacobian(self, name, inames, onames, opts):
        class JacobianFunction(ca.Callback):
            def __init__(self, callback, opts={}):
                ca.Callback.__init__(self)
                self.callback = callback
                self.construct(name, opts)

            def get_n_in(self): return 3
            def get_n_out(self): return 2

            def get_sparsity_in(self,i):
                if i == 0: # nominal input 0 (coordinates)
                    return ca.Sparsity.dense(self.callback.get_num_coordinates(), 1)
                if i == 1: # nominal input 1 (scales)
                    return ca.Sparsity.dense(self.callback.get_num_scales(), 1)
                if i == 2: # nominal output
                    return ca.Sparsity.dense(self.callback.get_num_outputs(), 1)

            def get_sparsity_out(self,i):
                if i == 0: # Jacobian w.r.t. input 0 (coordinates)
                    return ca.Sparsity.dense(self.callback.get_num_outputs(),
                                             self.callback.get_num_coordinates())
                if i == 1: # Jacobian w.r.t. input 1 (scales)
                    return ca.Sparsity.dense(self.callback.get_num_outputs(),
                                             self.callback.get_num_scales())

            def eval(self, arg):
                return self.callback._jac_eval(arg)

        self.jacobian_callback = JacobianFunction(self)
        return self.jacobian_callback

    @abstractmethod
    def _get_num_coordinates(self):
        pass

    @abstractmethod
    def _get_num_scales(self):
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

    def _get_num_coordinates(self):
        return len(self.q_indexes)

    def _get_num_scales(self):
        return 3*self.model.getNumBodies()

    def _get_num_outputs(self):
        return 1

    def _eval(self, arg):
        self.apply_state(arg)
        scales = self.pack_scales(arg)
        position = self.matter.calcScaledStationPosition(self.state, self.mobod_index,
                                                         self.station, scales)

        error = np.square(np.linalg.norm(position.to_numpy() - self.reference))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        scales = self.pack_scales(arg)
        error = self.matter.calcScaledStationPosition(self.state, self.mobod_index,
                                                      self.station, scales)
        error[0] -= self.reference[0]
        error[1] -= self.reference[1]
        error[2] -= self.reference[2]

        vec = osim.Vector(self.get_num_coordinates(), 0.0)
        self.matter.multiplyByScaledStationJacobianTranspose(self.state,
                                                             scales,
                                                             self.mobod_index,
                                                             self.station, error, vec)
        Jq = 2.0*vec.to_numpy()


        vecVec3 = osim.VectorVec3(self.get_num_scales(), osim.Vec3(0))
        self.matter.multiplyByScaleStationJacobianTranspose(self.state,
                                                            self.mobod_index,
                                                            self.station, error,
                                                            vecVec3)

        Js = np.zeros((1, 3*self.model.getNumBodies()))
        for ib in range(self.model.getNumBodies()):
            Js[0, 3*ib:3*ib+3] = vecVec3.get(ib+1).to_numpy()

        return [np.expand_dims(Jq[self.q_indexes], axis=0), 2.0*Js]





# Unit tests
# ----------
MODEL_FPATH = 'unscaled_generic.osim'
FRAME_PATHS = ['/bodyset/pelvis/pelvis',
               '/bodyset/torso/torso',
               '/jointset/hip_r/femur_r_offset/r_thigh',
               '/jointset/walker_knee_r/tibia_r_offset/r_shank',
               '/jointset/ankle_r/talus_r_offset/r_foot',
               '/jointset/mtp_r/toes_r_offset/r_toes']



class TestPositionErrorJacobians(unittest.TestCase):
    def test_position_error_jacobians(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        reference = np.array([0.1, 0.2, 0.3])

        nb = model.getNumBodies()
        scales = osim.VectorVec3(nb, osim.Vec3(1.0))
        for ib in range(nb):
            scales[ib] = osim.Vec3(1.0 + 0.01*ib, 1.0 - 0.01*ib, 1.0 + 0.02*ib)

        for frame_path in FRAME_PATHS:
            f_fd = ScaledPositionErrorCallback('f_fd', model, frame_path, reference,
                                               {"enable_fd": True})
            f_jac = ScaledPositionErrorCallback('f_jac', model, frame_path, reference)

            # Symbolic inputs.
            q = ca.SX.sym('q', len(f_fd.q_indexes))
            s = ca.SX.sym('s', 3*model.getNumBodies())
            x = ca.vertcat(q, s)

            # Jacobian expression graphs.
            J_fd = ca.Function('J_fd',[x],[ca.jacobian(f_fd(q, s), x)])
            J_jac = ca.Function('J_jac',[x],[ca.jacobian(f_jac(q, s), x)])

            # Test that the two Jacobians are equivalent.
            self.assertTrue(np.allclose(J_jac(2).full(), J_fd(2).full(), atol=1e-6))


class TestScaledPosition(unittest.TestCase):
    def test_scaled_position(self):
        model = osim.Model(MODEL_FPATH)
        state = model.initSystem()
        matter = model.getMatterSubsystem()

        # Helper function to set a non-trival set of default coordinates.
        def set_default_positions(state):
            nq = state.getNQ()
            for iq in range(nq):
                state.updQ()[iq] = 0.01*iq

        # Define the scale factors for each body. The first scale is always 1.0,
        # corresponding to the ground frame.
        nb = model.getNumBodies()
        scales = osim.VectorVec3(nb+1, osim.Vec3(1.0))
        scales[0] = osim.Vec3(1.0)
        for ib in range(nb):
            scales[ib+1] = osim.Vec3(1.0 + 0.01*ib, 1.0 - 0.01*ib, 1.0 + 0.02*ib)

        # For each frame, test that the position calculated from the unscaled system
        # using SimbodyMatterSubsystem::calcScaledStationPosition is equivalent to the
        # position calculated from the scaled system.
        for frame_path in FRAME_PATHS:

            # Load the model fresh to reset any scaling applied in previous iterations
            # of the loop.
            model = osim.Model(MODEL_FPATH)
            state = model.initSystem()
            matter = model.getMatterSubsystem()

            # Scaled position from unscaled system using the SimbodyMatterSubsystem
            # operator.
            frame = osim.PhysicalOffsetFrame.safeDownCast(
                model.getComponent(frame_path))
            mobod_index = frame.getMobilizedBodyIndex()
            transform = frame.findTransformInBaseFrame()
            station = osim.Vec3(transform.p())
            set_default_positions(state)
            model.realizePosition(state)
            position = matter.calcScaledStationPosition(state, mobod_index, station,
                                                        scales)

            # Scale the model. The order of the scale factors stored in the `scales`
            # vector does not necessarily match the order of the bodies in the model, so
            # we need to index the scales by each body's underlying MobilizedBodyIndex.
            scaleset = osim.ScaleSet()
            for ib in range(nb):
                body = model.getBodySet().get(ib)
                mobod_idx = int(body.getMobilizedBodyIndex())
                segment = body.getName()
                scale = osim.Scale()
                scale.setSegmentName(segment)
                scale.setScaleFactors(scales.get(mobod_idx))
                scaleset.cloneAndAppend(scale)
                scaleset.get(scaleset.getSize()-1).setName(segment)
            model.scale(state, scaleset, True)

            # Calculate the frame position from scaled system.
            model.finalizeConnections()
            state = model.initSystem()
            matter = model.getMatterSubsystem()
            set_default_positions(state)
            model.realizePosition(state)
            frame = osim.PhysicalOffsetFrame.safeDownCast(model.getComponent(frame_path))
            position_scaled_model = frame.getPositionInGround(state)

            # Test that the two positions are equivalent.
            self.assertTrue(np.allclose(position.to_numpy(),
                                        position_scaled_model.to_numpy(), atol=1e-6))
