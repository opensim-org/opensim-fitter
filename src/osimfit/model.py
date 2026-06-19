import numpy as np
import opensim as osim
from dataclasses import dataclass, field


###############
# SCALE GROUP #
###############

@dataclass
class BodyScaleGroup:
    """
    A group of mobilized bodies sharing one set of XYZ body scales. The group
    defines the list of OpenSim body paths and corresponding mobilized body indexes for
    each set of body scales.

    Attributes
    ----------
    body_paths: list[str]
        Absolute model paths to the bodies in this group.
    mobod_indexes: list[int]
        `MobilizedBodyIndex` values for the bodies in this group, paired with
        body_paths.
    inboard_joints: list[osim.Joint]
        A list of `Joint`s whose inboard frames correspond to the `MobilizedBodyIndex`
        values in `mobod_indexes`.
    outboard_joints: list[osim.Joint]
        A list of `Joint`s whose outboard frames correspond to the `MobilizedBodyIndex`
        values in `mobod_indexes`.
    """
    body_paths: list[str]
    mobod_indexes: list[int]
    inboard_joints: list[osim.Joint] = field(default_factory=list, compare=False)
    outboard_joints: list[osim.Joint] = field(default_factory=list, compare=False)


@dataclass
class TranslationScaleGroup:
    """
    A group of CustomJoints (backed by `SimTK::MobilizedBody::FunctionBased`) sharing
    one set of XYZ translation-scale factors. Translation scales multiply the three
    translation-output functions of the FunctionBased mobilizer (entries 3..5 of the
    spatial transform) before they are combined into the mobilizer translation
    `p_FM`.

    Attributes
    ----------
    joint_paths: list[str]
        Absolute model paths to the CustomJoints in this group.
    mobod_indexes: list[int]
        MobilizedBodyIndex values of the FunctionBased mobilizers backing the joints,
        paired with `joint_paths`.
    """
    joint_paths: list[str]
    mobod_indexes: list[int]
    custom_joints: list[osim.CustomJoint] = field(default_factory=list, compare=False)


#########
# MODEL #
#########

class ModelCache:
    """
    A thin wrapper around `osim.Model` that pre-computes and caches lookups
    used repeatedly by solvers and callback functions. It also provides useful methods
    for complicated calculations used by solvers (e.g., converting gradients with 
    respect to body scales).

    Parameters
    ----------
    model: str or osim.Model
        The OpenSim model to use for the optimization problem.

    Attributes
    ----------
    model: osim.Model
        The wrapped OpenSim model.
    state: osim.State
        The model's working state (snapshot at construction time).
    matter: osim.SimbodyMatterSubsystem
        The cached matter subsystem reference.
    num_mobod: int
        Total Simbody mobod count, including Ground at index 0.
    q_map: dict[str, int]
        Mapping from absolute coordinate path to its q-index in the State,
        restricted to independent coordinates (e.g., coupled coordinates are 
        excluded).
    q_indexes: list[int]
        The q-indexes of the independent coordinates, in registration order.
    parent_of: dict[int, int]
        Per-mobod parent in the multibody tree. ``parent_of[k]`` is the
        ``MobilizedBodyIndex`` of body ``k``'s parent (Ground has no entry).
    children_of: dict[int, list[int]]
        Inverse of ``parent_of``: ``children_of[k]`` is the list of mobod
        indexes whose parent is ``k``. Every mobod (including Ground at 0)
        has an entry, possibly empty.
    translation_scale_candidates: list[str]
        Auto-detected absolute paths of CustomJoints whose translation
        TransformAxes carry a non-trivial function.
    """
    def __init__(self, model: str | osim.Model):
        modelProcessor = osim.ModelProcessor(model)
        self.model = modelProcessor.process()
        self.state = self.model.initSystem()
        self.matter = self.model.getMatterSubsystem()
        self.num_mobod = self.model.getNumBodies() + 1
        self.q_map = self._get_coordinate_index_map(self.model, 
                                                    skip_dependent_coordinates=True)
        self.q_indexes = list(self.q_map.values())

        # For now, disallow models with joints where qdot != u.
        assert(self.state.getNQ() == self.state.getNU())

        # Mobilized body parents. 
        self.parent_of: dict[int, int] = {}
        for k in range(1, self.num_mobod):
            mb = self.matter.getMobilizedBody(k)
            self.parent_of[k] = int(mb.getParentMobilizedBody()
                                      .getMobilizedBodyIndex())
            
        # Mobilized body children.
        self.children_of: dict[int, list[int]] = {
            k: [] for k in range(self.num_mobod)}
        for j, kp in self.parent_of.items():
            self.children_of[kp].append(j)

        # Cache baseline (unscaled) inboard (X_PF) and outboard (X_BM) mobilizer
        # frames for every mobilized body, indexed by MobilizedBodyIndex.
        self.baseline_p_PF: dict[int, np.ndarray] = {}
        self.baseline_R_PF: dict[int, osim.Rotation] = {}
        self.baseline_p_BM: dict[int, np.ndarray] = {}
        self.baseline_R_BM: dict[int, osim.Rotation] = {}
        for k in range(1, self.num_mobod):
            mb = self.matter.getMobilizedBody(k)
            X_PF = mb.getInboardFrame(self.state)
            self.baseline_p_PF[k] = X_PF.p().to_numpy()
            self.baseline_R_PF[k] = osim.Rotation(X_PF.R())
            X_BM = mb.getOutboardFrame(self.state)
            self.baseline_p_BM[k] = X_BM.p().to_numpy()
            self.baseline_R_BM[k] = osim.Rotation(X_BM.R())

    @staticmethod
    def _get_coordinate_index_map(model: osim.Model, 
                                  skip_dependent_coordinates: bool=True) -> dict:
        """
        Get a mapping between coordinate paths and their indexes in the state vector.

        Parameters
        ----------
        model: osim.Model
            The OpenSim model from which to create the coordinate index map.
        skip_dependent_coordinates: bool, optional
            Whether to skip dependent (e.g., constrained) coordinates in the model.
        """
        state = model.getWorkingState()
        state_paths = osim.createStateVariableNamesInSystemOrder(model)
        q_map: dict[str, int] = {}
        for i, state_path in enumerate(state_paths):
            if 'value' in state_path:
                coord_path = state_path.replace('/value', '')
                coordinate = osim.Coordinate.safeDownCast(model.getComponent(coord_path))
                if skip_dependent_coordinates:
                    if not coordinate.isDependent(state):
                        q_map[coord_path] = i
                else:
                    q_map[coord_path] = i

        return q_map

    @staticmethod
    def _translation_axes_are_axis_aligned(st: osim.SpatialTransform, 
                                           tol: float = 1e-9) -> bool:
        """
        Return true if each `TransformAxis` of the provided `SpatialTransform`, when
        normalized, are the Cartesian basis vectors [1,0,0], [0,1,0], [0,0,1] in
        some order (up to sign).

        Parameters
        ----------
        st: osim.SpatialTransform
            The SpatialTransform whose translation axes (entries 3..5) are checked.
        tol: float, optional
            Absolute tolerance for the basis-vector comparison. Default 1e-9.
        """
        basis_seen = set()
        for j in range(3, 6):
            ax = osim.Vec3(0)
            st.getTransformAxis(j).getAxis(ax)
            v = np.abs([ax.get(0), ax.get(1), ax.get(2)])
            norm = np.linalg.norm(v)
            if norm == 0.0:
                return False
            v = v / norm
            k = int(np.argmax(v))
            if not np.allclose(v, np.eye(3)[k], atol=tol):
                return False
            basis_seen.add(k)
        return len(basis_seen) == 3

    @staticmethod
    def _validate_custom_joint_can_scale(cj: osim.CustomJoint) -> None:
        """
        Raise `ValueError` if `cj` cannot be part of a translation scale. A CustomJoint 
        is scalable when its translation TransformAxes are axis-aligned (see 
        `_translation_axes_are_axis_aligned`) and at least one carries a non-trivial 
        function that `SpatialTransform::scale` would promote to a `MultiplierFunction`.

        A translation axis function is non-trivial when it is present (e.g., 
        `axis.hasFunction()` is True), is not a pure prismatic identity (e.g.,
        `osim.LinearFunction(1, 0)`), and is not a zero constant (e.g.,
        `osim.Constant(0)`).

        Raises
        ------
        ValueError
            If the translation axes are not axis-aligned, or no translation axis
            carries a non-trivial function to scale.
        """
        path = cj.getAbsolutePathString()
        st = cj.getSpatialTransform()

        # The translation axes must be axis-aligned; otherwise
        # SpatialTransform::scale projects the scale Vec3 onto a non-basis axis
        # and produces a weighted average.
        if not ModelCache._translation_axes_are_axis_aligned(st):
            raise ValueError(
                f'CustomJoint {path} cannot be part of a translation scale: its '
                f'translation axes are not axis-aligned.')

        # At least one translation axis must carry a non-trivial function.
        for j in range(3, 6):
            axis = st.getTransformAxis(j)

            # 1. Is the function present?
            if not axis.hasFunction():
                continue

            # 2. Is the function not a pure prismatic identity?
            f = axis.getFunction()
            lf = osim.LinearFunction.safeDownCast(f)
            if lf is not None:
                c = lf.getCoefficients()
                if c.get(0) == 1.0 and c.get(1) == 0.0:
                    continue

            # 3. If constant, is the function non-zero?
            cf = osim.Constant.safeDownCast(f)
            if cf is not None and cf.getValue() == 0.0:
                continue

            return

        raise ValueError(
            f'CustomJoint {path} cannot be part of a translation scale: no '
            f'translation axis carries a non-trivial function to scale.')

    def create_translation_scale_group(
                self, joint_paths) -> TranslationScaleGroup:
        """
        Validate that each path in `joint_paths` refers to a CustomJoint that can
        be part of a translation scale, and return a
        :py:class:`TranslationScaleGroup` pairing the joint paths with their
        FunctionBased mobod indexes.

        Parameters
        ----------
        joint_paths: str or list[str]
            One or more absolute paths to CustomJoints sharing one Vec3
            translation scale.

        Raises
        ------
        ValueError
            If `joint_paths` is empty, any path does not resolve to a CustomJoint
            in this model, or any CustomJoint cannot be part of a translation
            scale (see :py:meth:`_validate_custom_joint_can_scale`).
        """
        if isinstance(joint_paths, str):
            joint_paths = [joint_paths]
        if not joint_paths:
            raise ValueError(
                'joint_paths must be a non-empty string or list of strings.')

        mobod_indexes: list[int] = []
        for path in joint_paths:
            joint = osim.CustomJoint.safeDownCast(self.model.getComponent(path))
            if joint is None:
                raise ValueError(f'Component at {path} is not a CustomJoint.')
            self._validate_custom_joint_can_scale(joint)
            mobod_indexes.append(joint.getChildFrame().getMobilizedBodyIndex())

        return TranslationScaleGroup(list(joint_paths), mobod_indexes, custom_joints=[])

    @staticmethod
    def get_translation_scales(model: osim.Model) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping joint paths to per-axis translation scales, each 
        currently applied to a CustomJoint as a length-3 array.

        Parameters
        ----------
        model: osim.Model
            The model to read from.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping joint paths to current [sx, sy, sz] translation scales.
        """
        scales: dict[str, np.ndarray] = {}
        jointset = model.getJointSet()
        for ijoint in range(jointset.getSize()):
            joint = jointset.get(ijoint)
            joint_path = joint.getAbsolutePathString()
            cj = osim.CustomJoint.safeDownCast(model.getComponent(joint_path))
            if cj is None:
                continue
            
            st = cj.getSpatialTransform()
            scales[joint_path] = np.ones(3)
            for i in range(3):
                axis = st.getTransformAxis(3 + i)
                if not axis.hasFunction():
                    continue
                mf = osim.MultiplierFunction.safeDownCast(axis.getFunction())
                if mf is not None:
                    scales[joint_path][i] = mf.getScale()

        return scales

    @staticmethod
    def apply_translation_scales(model: osim.Model,
                                 scales: dict) -> None:
        """
        For each `(joint_path, Vec3)` entry in `scales`, scale the
        translation TransformAxis functions of that CustomJoint by delegating
        to OpenSim's `SpatialTransform::scale`.

        Parameters
        ----------
        model: osim.Model
            The model to mutate.
        scales: dict[str, np.ndarray | osim.Vec3]
            Mapping from CustomJoint absolute path to a length-3 Vec3-like
            translation-scale value.
        """
        for joint_path, tscale in scales.items():
            cj = osim.CustomJoint.safeDownCast(model.getComponent(joint_path))
            if cj is None:
                raise ValueError(f'Component at {joint_path} is not a CustomJoint.')
            st = cj.upd_SpatialTransform()

            # Undo any scaling left on the translation functions by a prior
            # Model::scale().
            for j in range(3, 6):
                axis = st.updTransformAxis(j)
                if not axis.hasFunction():
                    continue
                mf = osim.MultiplierFunction.safeDownCast(axis.updFunction())
                if mf is not None:
                    mf.setScale(1.0)

            # Apply the desired translation scale.
            tscale_np = np.asarray(tscale, dtype=float)
            st.scale(osim.Vec3(float(tscale_np[0]), float(tscale_np[1]),
                               float(tscale_np[2])))


    def get_joint_for_mobilized_body_index(self, mobod_index: int) -> osim.Joint:
        """
        Return a `Joint` whose child body is associated with provided `MobilizedBody`
        index.

        Parameters
        ----------
        mobod_index: int
            The index to a `MobilizedBody`.

        Raises
        ------
        ValueError
            If no `Joint` is found matching provided `MobilizedBody` index.
        """
        jointset = self.model.getJointSet()
        for i in range(jointset.getSize()):
            joint = jointset.get(i)
            if mobod_index == int(joint.getChildFrame().getMobilizedBodyIndex()):
                return joint

        raise ValueError(
                f"Could not find a Joint in model '{self.model.getName()}' with "
                f"MobilizedBodyIndex {mobod_index}")

    def set_scaled_mobilizer_frame_positions(self, state: osim.State,
                                             body_scale_groups: list[BodyScaleGroup],
                                             body_scales: np.ndarray) -> None:
        """
        Set the inboard (X_PF) and outboard (X_BM) mobilizer frame positions given body
        body scales. Invalidates Stage::Instance and higher.

        For each group, the outboard frame (X_BM) of every group body's joint and
        the inboard frame (X_PF) of every joint driving a group body's child are
        scaled by the group's XYZ body scale. Each scaled frame translation is
        computed elementwise from the cached baseline (relative to the body's base
        frame), so repeated calls are absolute rather than compounding.

        Parameters
        ----------
        state: osim.State
            The State to update.
        body_scale_groups: list[BodyScaleGroup]
            Body-scale groups, each carrying the inboard/outboard Joints to scale.
        body_scales: np.ndarray, shape (3 * len(body_scale_groups),)
            Flat XYZ body-scale variables, one Vec3 per BodyScaleGroup.
        """
        for i, group in enumerate(body_scale_groups):
            s = np.asarray(body_scales[3*i : 3*i+3], dtype=float)

            # Outboard frame (X_BM) attached to each group body.
            for joint in group.outboard_joints:
                k = int(joint.getChildFrame().getMobilizedBodyIndex())
                p_BM = self.baseline_p_BM[k] * s
                X_BM = osim.Transform(self.baseline_R_BM[k], osim.Vec3(
                    float(p_BM[0]), float(p_BM[1]), float(p_BM[2])))
                self.matter.getMobilizedBody(k).setOutboardFrame(state, X_BM)

            # Inboard frame (X_PF) of every joint driving a group body's child.
            for joint in group.inboard_joints:
                c = int(joint.getChildFrame().getMobilizedBodyIndex())
                p_PF = self.baseline_p_PF[c] * s
                X_PF = osim.Transform(self.baseline_R_PF[c], osim.Vec3(
                    float(p_PF[0]), float(p_PF[1]), float(p_PF[2])))
                self.matter.getMobilizedBody(c).setInboardFrame(state, X_PF)

    def calc_position_jacobian_wrt_body_scales(self, state: osim.State, 
                dp_GB: osim.VectorVec3, body_scale_groups: list[BodyScaleGroup]) -> np.ndarray:
        """
        Return the position-error Jacobian with respect to body scales given a
        `State` object with scaled inboard and outboard applied and a vector `dp_GB` 
        representing the position-error gradient with respect to body origin positions.

        Parameters
        ----------
        state: osim.State
            The `State` from which to compute the Jacobian. Scaled inboard and outboard
            frame positions should already be applied.
        dp_GB: osim.VectorVec3
            The gradient of the position-error with respect to body origin positions.
            Length is equal to the number of mobilized bodies in the system (including 
            ground).
        body_scale_groups: list[BodyScaleGroup]
            A list of `BodyScaleGroup`, one for each body scale. The cached references
            to `Joint`s should be populated to provide to access inboard and outboard 
            frame indexes.
        """
        dp_BM = osim.VectorVec3(self.num_mobod, osim.Vec3(0))
        self.matter.multiplyByPositionJacobianWrtOutboardFramePositionsTranspose(
            state, dp_GB, dp_BM)
        dp_PF = osim.VectorVec3(self.num_mobod, osim.Vec3(0))
        self.matter.multiplyByPositionJacobianWrtInboardFramePositionsTranspose(
            state, dp_GB, dp_PF)
        
        ds_body = np.zeros((self.num_mobod, 3))
        for cx in range(1, self.num_mobod):
            px = self.parent_of[cx]
            ds_body[px] += self.baseline_p_PF[cx] * dp_PF[cx].to_numpy()
            ds_body[cx] += self.baseline_p_BM[cx] * dp_BM[cx].to_numpy()

        Js = np.zeros((1, 3 * len(body_scale_groups)))
        for i, group in enumerate(body_scale_groups):
            col = np.zeros(3)
            for k in group.mobod_indexes:
                col += ds_body[k,:]
            Js[0, 3*i:3*(i+1)] = col

        return Js
 