import copy
import numpy as np
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .bounds import Bounds


##########
# GROUPS #
##########

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

    Notes
    -----
    A group holds only model-independent descriptors, so the same group may be
    registered on more than one `ModelCache`. The `Joint`s whose mobilizer frames
    scale with the group are model-specific and are therefore cached on the
    `ModelCache`, rebuilt whenever a group is registered.
    """
    body_paths: list[str]
    mobod_indexes: list[int]


@dataclass
class OffsetGroup:
    """
    A group of markers or frames sharing one set of XYZ offsets. The offset is an
    additive translation, expressed in each component's base frame, applied to the
    component's placement (a marker's location or a frame's translation).

    Attributes
    ----------
    component_paths: list[str]
        Absolute model paths to the markers or frames in this group.
    """
    component_paths: list[str]
    mobod_indexes: list[int]


@dataclass
class MarkerOffsetGroup(OffsetGroup):
    """An `OffsetGroup` whose components are markers (offsets a marker's location)."""


@dataclass
class FrameOffsetGroup(OffsetGroup):
    """An `OffsetGroup` whose components are frames (offsets a frame's translation)."""


@dataclass
class JointParameterGroup:
    """
    A group of joints sharing one dimensionless factor on a joint-level parameter (e.g.
    a set of `EllipsoidJoint`s sharing one factor on their ellipsoid radii). The
    factor multiplies each joint's own baseline value, read from the model when the
    group is registered.

    Attributes
    ----------
    joint_paths: list[str]
        Absolute model paths to the joints in this group.

    Notes
    -----
    A group holds only model-independent descriptors, so the same group may be
    registered on more than one `ModelCache`. The `Joint`s and their baselines are
    model-specific and are therefore cached on the `ModelCache`; see
    `ModelCache.add_parameter_group`.
    """
    joint_paths: list[str]


@dataclass
class EllipsoidRadiiScaleGroup(JointParameterGroup):
    """
    A `JointParameterGroup` whose joints are `EllipsoidJoint`s sharing one Vec3 of
    factors on their baseline ellipsoid radii.
    """


@dataclass
class BeamLengthScaleGroup(JointParameterGroup):
    """
    A `JointParameterGroup` whose joints are `CantileverFreeBeamJoint`s sharing one
    factor on their baseline beam length.
    """


###############
# MODEL CACHE #
###############

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
    num_mobod: int
        Total Simbody mobod count, including Ground at index 0.
    coordinate_q_map: dict[str, int]
        Mapping from absolute coordinate path to its q-index in the State,
        restricted to independent coordinates (e.g., coupled coordinates are
        excluded).
    coordinate_u_map: dict[str, int]
        Mapping from absolute coordinate path to its u-index in the State, with the
        same keys and iteration order as `coordinate_q_map`.
    coordinate_q_indexes: list[int]
        The q-indexes of the independent coordinates, in registration order.
    coordinate_u_indexes: list[int]
        The u-indexes of the independent coordinates, in the registration order.
    body_scale_groups: list[BodyScaleGroup]
        The list of BodyScaleGroups associated with this model.
    marker_offset_groups: list[MarkerOffsetGroup]
        The list of MarkerOffsetGroups associated with this model.
    frame_offset_groups: list[FrameOffsetGroup]
        The list of FrameOffsetGroups associated with this model.
    ellipsoid_radii_scale_groups: list[EllipsoidRadiiScaleGroup]
        The list of EllipsoidRadiiScaleGroups associated with this model.
    beam_length_scale_groups: list[BeamLengthScaleGroup]
        The list of BeamLengthScaleGroups associated with this model.
    ellipsoid_radii_scale_group_joints: list[list[osim.EllipsoidJoint]]
        Per-`EllipsoidRadiiScaleGroup` `EllipsoidJoint`s, parallel to
        `ellipsoid_radii_scale_groups`.
    ellipsoid_radii_scale_group_baselines: list[list[np.ndarray]]
        Per-`EllipsoidRadiiScaleGroup` baseline XYZ radii, one length-3 array per joint,
        parallel to `ellipsoid_radii_scale_group_joints`.
    beam_length_scale_group_joints: list[list[osim.CantileverFreeBeamJoint]]
        Per-`BeamLengthScaleGroup` `CantileverFreeBeamJoint`s, parallel to
        `beam_length_scale_groups`.
    beam_length_scale_group_baselines: list[list[float]]
        Per-`BeamLengthScaleGroup` baseline beam lengths, one per joint, parallel to
        `beam_length_scale_group_joints`.
    parent_of: dict[int, int]
        Per-mobod parent in the multibody tree. ``parent_of[k]`` is the
        ``MobilizedBodyIndex`` of body ``k``'s parent (Ground has no entry).
    children_of: dict[int, list[int]]
        Inverse of ``parent_of``: ``children_of[k]`` is the list of mobod
        indexes whose parent is ``k``. Every mobod (including Ground at 0)
        has an entry, possibly empty.
    outboard_write_mobods, inboard_write_mobods: osim.SimTKArrayInt
        The mobilized bodies whose outboard (X_BM) and inboard (X_PF) mobilizer
        frames scaling the registered `BodyScaleGroup`s rewrites: a group body itself
        for the outboard frames, and each of its children for the inboard frames.
    outboard_write_rotations, inboard_write_rotations: osim.SimTKArrayRotation
        Each frame's baseline rotation.
    outboard_write_baselines, inboard_write_baselines: np.ndarray, shape (n, 3)
        Each write's baseline frame translation.
    outboard_write_group_rows, inboard_write_group_rows: np.ndarray, shape (n,)
        Each write's `BodyScaleGroup` index, for gathering its scale.
    scale_group_mobod_indexes, scale_group_rows: np.ndarray
        Paired arrays over every body in every `BodyScaleGroup`: the body's mobod
        index and the index of the group scaling it.
    """
    def __init__(self, model: str | osim.Model):
        modelProcessor = osim.ModelProcessor(model)
        self.model = modelProcessor.process()
        self.state = self.model.initSystem()
        self.num_mobod = self.model.getNumBodies() + 1
        self.coordinate_q_map, self.coordinate_u_map = (
            self._get_coordinate_index_maps(self.model,
                                            skip_dependent_coordinates=True))
        self.coordinate_q_indexes = list(self.coordinate_q_map.values())
        self.coordinate_u_indexes = list(self.coordinate_u_map.values())
        self.body_scale_groups: list[BodyScaleGroup] = []
        self.marker_offset_groups: list[MarkerOffsetGroup] = []
        self.frame_offset_groups: list[FrameOffsetGroup] = []
        self.ellipsoid_radii_scale_groups: list[EllipsoidRadiiScaleGroup] = []
        self.beam_length_scale_groups: list[BeamLengthScaleGroup] = []
        self.ellipsoid_radii_scale_group_joints: list[list[osim.EllipsoidJoint]] = []
        self.ellipsoid_radii_scale_group_baselines: list[list[np.ndarray]] = []
        self.beam_length_scale_group_joints: list[
            list[osim.CantileverFreeBeamJoint]] = []
        self.beam_length_scale_group_baselines: list[list[float]] = []

        # Joints where qdot != u are acceptable, but assert that nq == nu (i.e.,
        # disallow quaternions).
        num_q_in_use = sum({
            int(coordinate.getBodyIndex()):
                self.model.getCoordinateNumQInUse(self.state, coordinate)
            for coordinate in self.model.getCoordinateSet()}.values())
        assert(num_q_in_use == self.state.getNU())

        # Mobilized body parents.
        self.parent_of: dict[int, int] = {}
        for i in range(self.model.getNumJoints()):
            joint = self.model.getJointSet().get(i)
            cix = int(joint.getChildFrame().getMobilizedBodyIndex())
            pix = int(joint.getParentFrame().getMobilizedBodyIndex())
            self.parent_of[cix] = pix

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
        for i in range(self.model.getNumJoints()):
            # TODO: this logic breaks for joints that contain multiple mobilized bodies
            # (e.g., ScapulothoracicJoint).
            joint = self.model.getJointSet().get(i)
            mbx = int(joint.getChildFrame().getMobilizedBodyIndex())
            X_PF = joint.getInboardFrame(self.state)
            self.baseline_p_PF[mbx] = X_PF.p().to_numpy()
            self.baseline_R_PF[mbx] = osim.Rotation(X_PF.R())
            X_BM = joint.getOutboardFrame(self.state)
            self.baseline_p_BM[mbx] = X_BM.p().to_numpy()
            self.baseline_R_BM[mbx] = osim.Rotation(X_BM.R())

        # Flattened mobilizer indexes and baseline frame translations for every
        # child body.
        self.child_mobod_indexes = np.arange(1, self.num_mobod)
        self.parent_mobod_indexes = np.array(
            [self.parent_of[cx] for cx in self.child_mobod_indexes], dtype=int)
        self.baseline_p_PF_rows = np.array(
            [self.baseline_p_PF[cx] for cx in self.child_mobod_indexes],
            dtype=float).reshape(-1, 3)
        self.baseline_p_BM_rows = np.array(
            [self.baseline_p_BM[cx] for cx in self.child_mobod_indexes],
            dtype=float).reshape(-1, 3)

        self._rebuild_body_scale_group_cache()

    def add_parameter_group(self, group) -> None:
        """
        Append a parameter group to the appropriate cached list, dispatched by type.

        Parameters
        ----------
        group: BodyScaleGroup, MarkerOffsetGroup, FrameOffsetGroup,
                EllipsoidRadiiScaleGroup, or BeamLengthScaleGroup
            The parameter group to register.

        Raises
        ------
        ValueError
            If `group` is not a recognized parameter group type.
        """
        if isinstance(group, BodyScaleGroup):
            self.body_scale_groups.append(group)
            self._rebuild_body_scale_group_cache()
        elif isinstance(group, MarkerOffsetGroup):
            self.marker_offset_groups.append(group)
        elif isinstance(group, FrameOffsetGroup):
            self.frame_offset_groups.append(group)
        elif isinstance(group, EllipsoidRadiiScaleGroup):
            joints = self._resolve_joints(group.joint_paths, osim.EllipsoidJoint)
            self.ellipsoid_radii_scale_groups.append(group)
            self.ellipsoid_radii_scale_group_joints.append(joints)
            self.ellipsoid_radii_scale_group_baselines.append(
                [joint.get_radii_x_y_z().to_numpy() for joint in joints])
        elif isinstance(group, BeamLengthScaleGroup):
            joints = self._resolve_joints(group.joint_paths,
                                          osim.CantileverFreeBeamJoint)
            self.beam_length_scale_groups.append(group)
            self.beam_length_scale_group_joints.append(joints)
            self.beam_length_scale_group_baselines.append(
                [float(joint.get_beam_length()) for joint in joints])
        else:
            raise ValueError(
                f'Unsupported parameter group type {type(group).__name__}.')

    def _resolve_joints(self, joint_paths: list[str], cls: type) -> list:
        """
        Return this model's `Joint`s at `joint_paths`, downcast to `cls`.

        Parameters
        ----------
        joint_paths: list[str]
            Absolute model paths to the joints to resolve.
        cls: type
            The concrete `osim.Joint` subclass every path must downcast to.

        Raises
        ------
        ValueError
            If a component at one of `joint_paths` is not a `cls`.
        """
        joints = []
        for path in joint_paths:
            joint = cls.safeDownCast(self.model.getComponent(path))
            if joint is None:
                raise ValueError(
                    f'Component at path {path} is not a {cls.__name__}.')
            joints.append(joint)
        return joints

    @staticmethod
    def _get_coordinate_index_maps(model: osim.Model,
                                   skip_dependent_coordinates: bool=True) -> tuple:
        """
        Get mappings between coordinate paths and their q and u indexes in the state
        vector.

        Parameters
        ----------
        model: osim.Model
            The OpenSim model from which to create the coordinate index maps.
        skip_dependent_coordinates: bool, optional
            Whether to skip dependent (e.g., constrained) coordinates in the model.
        """
        state = model.getWorkingState()
        state_paths = osim.createStateVariableNamesInSystemOrder(model)
        coordinate_q_map: dict[str, int] = {}
        coordinate_u_map: dict[str, int] = {}
        u_index = 0
        for state_path in state_paths:
            if 'value' in state_path:
                coord_path = state_path.replace('/value', '')
                coordinate = osim.Coordinate.safeDownCast(model.getComponent(coord_path))
                q_index = model.getCoordinateQIndex(state, coordinate)
                if not (skip_dependent_coordinates and coordinate.isDependent(state)):
                    coordinate_q_map[coord_path] = q_index
                    coordinate_u_map[coord_path] = u_index
                u_index += 1

        return coordinate_q_map, coordinate_u_map

    def gather_coordinate_gradient(self, state: osim.State,
                                   gradient_u: osim.Vector) -> np.ndarray:
        """
        Convert a gradient with respect to the generalized speeds into a gradient with
        respect to the independent coordinates.

        The matter subsystem's `multiplyBy*JacobianTranspose` operators return a
        length-nu gradient, but solvers optimize over coordinate values (q), so the
        result has to be mapped from u into q before its entries can be read off as
        per-coordinate derivatives.

        Parameters
        ----------
        state: osim.State
            The `State` the gradient was computed at, realized through Position.
        gradient_u: osim.Vector
            The length-nu gradient to convert.

        Returns
        -------
        np.ndarray, shape (len(coordinate_q_indexes),)
            The gradient with respect to each independent coordinate, ordered to match
            `coordinate_q_map`.
        """
        gradient_q = osim.Vector(state.getNQ(), 0.0)
        self.model.multiplyByNInv(state, True, gradient_u, gradient_q)
        return gradient_q.to_numpy()[self.coordinate_q_indexes]

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

    def _rebuild_body_scale_group_cache(self) -> None:
        """
        Rebuild the model-specific state implied by the registered `BodyScaleGroup`s:
        the mobilizer frame writes that scaling each group performs, and the paired
        body/group index arrays the body-scale Jacobian accumulates over.
        """
        outboard_joints = [
            [self.get_joint_for_mobilized_body_index(int(k))
             for k in group.mobod_indexes]
            for group in self.body_scale_groups]
        inboard_joints = [
            [self.get_joint_for_mobilized_body_index(c)
             for k in group.mobod_indexes
             for c in self.children_of[int(k)]]
            for group in self.body_scale_groups]

        self.scale_group_mobod_indexes = np.array(
            [int(k) for group in self.body_scale_groups
             for k in group.mobod_indexes], dtype=int)
        self.scale_group_rows = np.array(
            [i for i, group in enumerate(self.body_scale_groups)
             for _ in group.mobod_indexes], dtype=int)

        self.outboard_write_mobods = osim.SimTKArrayInt()
        self.outboard_write_rotations = osim.SimTKArrayRotation()
        baselines, group_rows = [], []
        for igroup, group_joints in enumerate(outboard_joints):
            for joint in group_joints:
                index = int(joint.getChildFrame().getMobilizedBodyIndex())
                self.outboard_write_mobods.push_back(index)
                self.outboard_write_rotations.push_back(self.baseline_R_BM[index])
                baselines.append(self.baseline_p_BM[index])
                group_rows.append(igroup)
        self.outboard_write_baselines = np.asarray(
            baselines, dtype=float).reshape(-1, 3)
        self.outboard_write_group_rows = np.asarray(group_rows, dtype=int)

        self.inboard_write_mobods = osim.SimTKArrayInt()
        self.inboard_write_rotations = osim.SimTKArrayRotation()
        baselines, group_rows = [], []
        for igroup, group_joints in enumerate(inboard_joints):
            for joint in group_joints:
                index = int(joint.getChildFrame().getMobilizedBodyIndex())
                self.inboard_write_mobods.push_back(index)
                self.inboard_write_rotations.push_back(self.baseline_R_PF[index])
                baselines.append(self.baseline_p_PF[index])
                group_rows.append(igroup)
        self.inboard_write_baselines = np.asarray(
            baselines, dtype=float).reshape(-1, 3)
        self.inboard_write_group_rows = np.asarray(group_rows, dtype=int)

    def set_scaled_mobilizer_frame_positions(self, state: osim.State,
                                             body_scales: np.ndarray) -> None:
        """
        Set the inboard (X_PF) and outboard (X_BM) mobilizer frame positions given body
        body scales. Invalidates Stage::Instance and higher.

        Parameters
        ----------
        state: osim.State
            The State to update.
        body_scales: np.ndarray, shape (3 * len(body_scale_groups),)
            Flat XYZ body-scale variables, one Vec3 per BodyScaleGroup.
        """
        scales = np.asarray(body_scales, dtype=float).reshape(-1, 3)

        # Outboard frames (X_BM) attached to each group body, written in one call.
        if len(self.outboard_write_group_rows):
            p_BM = (self.outboard_write_baselines
                    * scales[self.outboard_write_group_rows])
            self.model.setOutboardFrames(
                state, self.outboard_write_mobods, self.outboard_write_rotations,
                osim.Vector.createFromMat(
                    np.ascontiguousarray(p_BM, dtype=float).reshape(-1)))

        # Inboard frames (X_PF) of every joint driving a group body's child.
        if len(self.inboard_write_group_rows):
            p_PF = (self.inboard_write_baselines
                    * scales[self.inboard_write_group_rows])
            self.model.setInboardFrames(
                state, self.inboard_write_mobods, self.inboard_write_rotations,
                osim.Vector.createFromMat(
                    np.ascontiguousarray(p_PF, dtype=float).reshape(-1)))

    def set_ellipsoid_radii_from_scales(
            self, state: osim.State, ellipsoid_radii_scales: np.ndarray) -> None:
        """
        Set the radii of every `EllipsoidJoint` in every registered
        `EllipsoidRadiiScaleGroup` to that joint's cached baseline radii times the
        group's factors.

        Parameters
        ----------
        state: osim.State
            The State to update.
        ellipsoid_radii_scales: np.ndarray
            Flat XYZ factors on the baseline radii, one Vec3 per
            EllipsoidRadiiScaleGroup. Length is
            ``3 * len(ellipsoid_radii_scale_groups)``.
        """
        for i, (joints, baselines) in enumerate(zip(
                self.ellipsoid_radii_scale_group_joints,
                self.ellipsoid_radii_scale_group_baselines)):
            factors = np.asarray(ellipsoid_radii_scales[3*i : 3*i+3], dtype=float)
            for joint, baseline in zip(joints, baselines):
                r = baseline * factors
                joint.setRadii(state, osim.Vec3(
                    float(r[0]), float(r[1]), float(r[2])))

    def set_beam_length_from_scales(self, state: osim.State,
                                    beam_length_scales: np.ndarray) -> None:
        """
        Set the beam length of every `CantileverFreeBeamJoint` in every registered
        `BeamLengthScaleGroup` to that joint's cached baseline length times the group's
        factor.

        Parameters
        ----------
        state: osim.State
            The State to update.
        beam_length_scales: np.ndarray, shape (len(beam_length_scale_groups),)
            Factors on the baseline beam lengths, one per BeamLengthScaleGroup.
        """
        for i, (joints, baselines) in enumerate(zip(
                self.beam_length_scale_group_joints,
                self.beam_length_scale_group_baselines)):
            factor = float(beam_length_scales[i])
            for joint, baseline in zip(joints, baselines):
                joint.setLength(state, baseline * factor)

    def calc_position_jacobian_wrt_ellipsoid_radii_scales(
            self, state: osim.State, dp_GB: osim.VectorVec3) -> np.ndarray:
        """
        Return the position-error Jacobian with respect to ellipsoid radii given a
        `State` with the current radii applied and a vector `dp_GB` representing the
        position-error gradient with respect to body origin positions.

        Parameters
        ----------
        state: osim.State
            The `State` from which to compute the Jacobian, realized through Position.
        dp_GB: osim.VectorVec3
            The gradient of the position error with respect to body origin positions.
            Length is equal to the number of mobilized bodies in the system (including
            ground).
        """
        Jr = np.zeros((1, 3 * len(self.ellipsoid_radii_scale_groups)))
        for i, (joints, baselines) in enumerate(zip(
                self.ellipsoid_radii_scale_group_joints,
                self.ellipsoid_radii_scale_group_baselines)):
            col = np.zeros(3)
            for joint, baseline in zip(joints, baselines):
                col += baseline * joint.multiplyByPositionJacobianWrtRadiiTranspose(
                    state, dp_GB).to_numpy()
            Jr[0, 3*i:3*(i+1)] = col

        return Jr

    def calc_position_jacobian_wrt_beam_length_scales(
            self, state: osim.State, dp_GB: osim.VectorVec3) -> np.ndarray:
        """
        Return the position-error Jacobian with respect to beam lengths given a `State`
        with the current beam lengths applied and a vector `dp_GB` representing the
        position-error gradient with respect to body origin positions.

        Parameters
        ----------
        state: osim.State
            The `State` from which to compute the Jacobian, realized through Position.
        dp_GB: osim.VectorVec3
            The gradient of the position error with respect to body origin positions.
            Length is equal to the number of mobilized bodies in the system (including
            ground).
        """
        Jl = np.zeros((1, len(self.beam_length_scale_groups)))
        for i, (joints, baselines) in enumerate(zip(
                self.beam_length_scale_group_joints,
                self.beam_length_scale_group_baselines)):
            Jl[0, i] = sum(
                baseline * float(joint.multiplyByPositionJacobianWrtLengthTranspose(
                    state, dp_GB))
                for joint, baseline in zip(joints, baselines))

        return Jl

    @staticmethod
    def get_custom_joint_translation_scales(model: osim.Model) -> dict[str, np.ndarray]:
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
    def apply_custom_joint_translation_scales(model: osim.Model, scales: dict) -> None:
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

    @staticmethod
    def get_ellipsoid_joint_radii(model: osim.Model) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping every `EllipsoidJoint`'s path to its current
        [rx, ry, rz] radii.

        Parameters
        ----------
        model: osim.Model
            The model to read from.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping EllipsoidJoint paths to their current radii.
        """
        radii: dict[str, np.ndarray] = {}
        jointset = model.getJointSet()
        for ijoint in range(jointset.getSize()):
            joint_path = jointset.get(ijoint).getAbsolutePathString()
            ej = osim.EllipsoidJoint.safeDownCast(model.getComponent(joint_path))
            if ej is not None:
                radii[joint_path] = ej.get_radii_x_y_z().to_numpy()

        return radii

    @staticmethod
    def apply_ellipsoid_joint_radii(model: osim.Model, radii: dict) -> None:
        """
        Write each `(joint_path, Vec3)` entry of `radii` back onto that
        `EllipsoidJoint`.

        Parameters
        ----------
        model: osim.Model
            The model to mutate.
        radii: dict[str, np.ndarray | osim.Vec3]
            Mapping from EllipsoidJoint absolute path to a length-3 Vec3-like radii
            value.
        """
        for joint_path, joint_radii in radii.items():
            ej = osim.EllipsoidJoint.safeDownCast(model.getComponent(joint_path))
            if ej is None:
                raise ValueError(
                    f'Component at {joint_path} is not an EllipsoidJoint.')
            r = np.asarray(joint_radii, dtype=float)
            ej.set_radii_x_y_z(osim.Vec3(float(r[0]), float(r[1]), float(r[2])))

    def calc_position_jacobian_wrt_body_scales(self, state: osim.State,
                                               dp_GB: osim.VectorVec3) -> np.ndarray:
        """
        Return the position-error Jacobian with respect to body scales given a
        `State` object with scaled inboard and outboard applied and a vector `dp_GB`
        representing the position-error gradient with respect to body origin
        positions.

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
        self.model.multiplyByPositionJacobianWrtOutboardFramePositionsTranspose(
            state, dp_GB, dp_BM)
        dp_PF = osim.VectorVec3(self.num_mobod, osim.Vec3(0))
        self.model.multiplyByPositionJacobianWrtInboardFramePositionsTranspose(
            state, dp_GB, dp_PF)

        # Read both gradients in one crossing each, then drop Ground's row.
        children = self.child_mobod_indexes
        gradient_PF = dp_PF.to_numpy()[children]
        gradient_BM = dp_BM.to_numpy()[children]

        ds_body = np.zeros((self.num_mobod, 3))
        np.add.at(ds_body, self.parent_mobod_indexes,
                  self.baseline_p_PF_rows * gradient_PF)
        ds_body[children] += self.baseline_p_BM_rows * gradient_BM

        Js = np.zeros((1, 3 * len(self.body_scale_groups)))
        if len(self.scale_group_mobod_indexes):
            np.add.at(Js.reshape(-1, 3), self.scale_group_rows,
                      ds_body[self.scale_group_mobod_indexes])

        return Js

    def find_body_scale_group_index(self, mobod_index: int):
        """
        Find the `BodyScaleGroup` index associated with a `MobilizedBody` in the model.

        Parameters
        ----------
        mobod_index: int
            The index to a `MobilizedBody` in the model.

        Raises
        ------
        Exception
            If multiple `BodyScaleGroup` indexes are found for the provided
            `osim.MobilizedBodyIndex`.
        """
        scale_groups = list()
        for g, group in enumerate(self.body_scale_groups):
            if mobod_index in [int(k) for k in group.mobod_indexes]:
                scale_groups.append(g)

        if len(scale_groups) > 1:
            raise Exception(f'Multiple scale groups found for body at index '
                            f'{mobod_index}')

        return scale_groups[0] if len(scale_groups) > 0 else None

    def get_tracking_marker_paths(self):
        """
        Get a list of all markers in the model whose '<fixed>' property is ``False``.
        """
        tracking_markers: list[str] = []
        for i in range(self.model.getMarkerSet().getSize()):
            marker = self.model.getMarkerSet().get(i)
            if not marker.get_fixed():
                tracking_markers.append(marker.getAbsolutePathString())

        return tracking_markers


class StationCache:
    """
    A thin wrapper around a point fixed on a body (a `osim.Station` or a
    `osim.PhysicalFrame`'s origin) that pre-computes values used repeatedly by solvers
    and callback functions.

    Construct via `from_station` or `from_frame`.

    Attributes
    ----------
    base_frame: osim.PhysicalFrame
        The base `osim.PhysicalFrame` of the frame to which the point is attached.
    mobod_index: int
        The index to the `osim.MobilizedBody` associated with the base frame.
    base_station: np.ndarray
        The location of the point in the base frame, shape (3,).
    body_scale_group_index: None | int
        The index to the `BodyScaleGroup` associated with this point's
        `osim.MobilizedBodyIndex`, if it exists.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError(
            'Construct a StationCache via StationCache.from_station(...) or '
            'StationCache.from_frame(...).')

    @classmethod
    def _create(cls, mc: ModelCache, base_frame: osim.PhysicalFrame,
                base_station: np.ndarray):
        """
        Populate a cache from an already-resolved base frame and base-frame point.
        """
        cache = cls.__new__(cls)
        cache.mc = mc
        cache.base_frame = base_frame
        cache.mobod_index = int(base_frame.getMobilizedBodyIndex())
        cache.base_station = base_station
        cache.body_scale_group_index = mc.find_body_scale_group_index(cache.mobod_index)
        return cache

    @classmethod
    def from_station(cls, mc: ModelCache, station: osim.Station):
        """
        Build a `StationCache` for an `osim.Station`. `base_station` is the station's
        location in its base frame.

        Parameters
        ----------
        mc: ModelCache
            A previously-constructed `ModelCache`.
        station: osim.Station
            The station (or a subclass, e.g. an `osim.Marker`) to wrap.

        Raises
        ------
        ValueError
            If `station` is not an `osim.Station`.
        """
        downcast = osim.Station.safeDownCast(station)
        if downcast is None:
            raise ValueError(f'Expected an osim.Station, but got {station}.')
        base_frame = osim.PhysicalFrame.safeDownCast(
            downcast.getParentFrame().findBaseFrame())
        base_station = downcast.findLocationInFrame(mc.state, base_frame).to_numpy()
        return cls._create(mc, base_frame, base_station)

    @classmethod
    def from_frame(cls, mc: ModelCache, frame: osim.PhysicalFrame):
        """
        Build a `StationCache` for a `osim.PhysicalFrame`'s origin. `base_station` is
        the frame origin's location in its base frame.

        Parameters
        ----------
        mc: ModelCache
            A previously-constructed `ModelCache`.
        frame: osim.PhysicalFrame
            The frame whose origin to wrap.

        Raises
        ------
        ValueError
            If `frame` is not an `osim.PhysicalFrame`.
        """
        downcast = osim.PhysicalFrame.safeDownCast(frame)
        if downcast is None:
            raise ValueError(f'Expected an osim.PhysicalFrame, but got {frame}.')
        base_frame = osim.PhysicalFrame.safeDownCast(downcast.findBaseFrame())
        transform = downcast.findTransformInBaseFrame()
        base_station = transform.p().to_numpy()
        return cls._create(mc, base_frame, base_station)

    def calc_scaled_base_station(self, body_scales: np.ndarray) -> np.ndarray:
        offset = self.base_station.copy()
        if self.body_scale_group_index is not None:
            g = self.body_scale_group_index
            offset = offset * np.asarray(body_scales[3*g : 3*g+3], dtype=float)
        return offset

    def calc_position(self, state: osim.State, body_scales: np.ndarray) -> osim.Vec3:
        offset = self.calc_scaled_base_station(body_scales)
        vec = osim.Vec3(float(offset[0]), float(offset[1]), float(offset[2]))
        return self.base_frame.findStationLocationInGround(state, vec)

    def calc_position_jacobian_wrt_body_scales(self, state: osim.State) -> np.ndarray:
        rotation = self.base_frame.getRotationInGround(state)
        R_GB = np.array([[rotation.get(r, c) for c in range(3)] for r in range(3)])
        jacobian = np.zeros((3, 3 * len(self.mc.body_scale_groups)))
        for axis in range(3):
            dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
            unit = [0.0, 0.0, 0.0]
            unit[axis] = 1.0
            dp_GB.set(self.mobod_index, osim.Vec3(unit[0], unit[1], unit[2]))
            row = self.mc.calc_position_jacobian_wrt_body_scales(
                state, dp_GB)[0, :].copy()

            # Add the contribution from scaling the station's base-frame location.
            if self.body_scale_group_index is not None:
                doffset = np.asarray(unit) @ R_GB
                g = self.body_scale_group_index
                row[3*g:3*g+3] += self.base_station * doffset

            jacobian[axis, :] = row

        return jacobian


##############
# PARAMETERS #
##############

class Parameter(ABC):
    """
    Base class for an optimized parameter. The parameter can be assigned to a single
    component, or a group of model components of the same type. Each parameter will
    create single block of optimization variables in a bilevel problem. Subclasses
    must supply the per-type behavior a solver needs by implementing the abstract
    methods `validate`, `to_group`, `append_guess_and_bounds`, and `apply_to_model`.

    Attributes
    ----------
    value: np.ndarray or None
        The optimized (or initial-guess) value for this parameter, or ``None`` when
        unset. Populated by solvers and carried on solution objects.
    group_type: type
        The math-layer descriptor type (e.g., `BodyScaleGroup`) for this parameter, as
        consumed by the cost callback.
    cost_input: str
        The name of the `CostInput` field this parameter's variable block feeds (e.g.,
        ``'body_scales'``). Solvers use it to order parameter blocks by
        `CostInput.INPUT_ORDER`.
    """
    value: np.ndarray = None
    group_type: type = None
    cost_input: str = None

    @abstractmethod
    def validate(self, mc: ModelCache) -> None:
        """
        Validate this parameter against the model and cache any derived data. Raise a
        ValueError if the configuration is invalid.
        """

    @abstractmethod
    def to_group(self):
        """
        Return the math-layer descriptor (e.g., `BodyScaleGroup`) for this parameter, as
        consumed by the cost callback.
        """

    @abstractmethod
    def append_guess_and_bounds(self, x0: list, lbx: list, ubx: list) -> None:
        """
        Append this parameter's initial guess and per-variable bounds, in place, to the
        solver's `x0`, `lbx`, and `ubx` arrays.
        """

    @abstractmethod
    def apply_to_model(self, model: osim.Model) -> None:
        """
        Apply this parameter's `value` to the `model`.
        """

    @property
    @abstractmethod
    def num_variables(self) -> int:
        """
        The number of optimization variables in this parameter's block.
        """

    def with_value(self, value: np.ndarray) -> "Parameter":
        """
        Return a copy of this parameter carrying `value`, leaving the original
        unchanged. Raise a ValueError if `value` does not have `num_variables` elements.
        """
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != self.num_variables:
            raise ValueError(
                f'{type(self).__name__} expected a value with {self.num_variables} '
                f'element(s), but got {value.size}.')
        new = copy.copy(self)
        new.value = value
        return new


class Vec3Parameter(Parameter):
    """
    A parameter representing a Vec3 quantity in an OpenSim model.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the component(s) sharing this parameter's Vec3 value.
    bounds: Bounds
        Bounds applied to each element of the Vec3.
    value: np.ndarray
        Initial value for the Vec3.
    """
    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError(
                'paths must be a non-empty string or list of strings.')
        self.paths = list(paths)
        self.bounds = bounds
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != self.num_variables:
            raise ValueError(
                f'{type(self).__name__} expected a value with {self.num_variables} '
                f'element(s), but got {value.size}.')
        self.value = value

    @property
    def num_variables(self) -> int:
        return 3

    def append_guess_and_bounds(self, x0: list, lbx: list, ubx: list) -> None:
        x0 += self.value.tolist()
        lbx += [self.bounds.lower_bound] * 3
        ubx += [self.bounds.upper_bound] * 3


class ScalarParameter(Parameter):
    """
    A parameter representing a scalar quantity in an OpenSim model.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the component(s) sharing this parameter's scalar
        value.
    bounds: Bounds
        Bounds applied to the scalar.
    value: float
        Initial value for the scalar.
    """
    def __init__(self, paths: str | list[str], bounds: Bounds, value: float):
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError(
                'paths must be a non-empty string or list of strings.')
        self.paths = list(paths)
        self.bounds = bounds
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != self.num_variables:
            raise ValueError(
                f'{type(self).__name__} expected a value with {self.num_variables} '
                f'element(s), but got {value.size}.')
        self.value = value

    @property
    def num_variables(self) -> int:
        return 1

    def append_guess_and_bounds(self, x0: list, lbx: list, ubx: list) -> None:
        x0 += self.value.tolist()
        lbx += [self.bounds.lower_bound]
        ubx += [self.bounds.upper_bound]


class BodyScale(Vec3Parameter):
    """
    An optimized Vec3 of body scales shared across one or more bodies. Pass a single
    body path to scale one body, or a list of body paths to share one set of body scales
    across a group of bodies (e.g., for left-right symmetric scaling).

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the body or bodies whose body scale is optimized.
    bounds: Bounds
        Bounds applied to each Vec3 scale factor.
    value: np.ndarray
        Initial [sx, sy, sz] scale.
    """
    group_type = BodyScaleGroup
    cost_input = 'body_scales'

    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def validate(self, mc: ModelCache) -> None:
        self.mobod_indexes = []
        for path in self.paths:
            body = osim.Body.safeDownCast(mc.model.getComponent(path))
            if body is None:
                raise ValueError(f'Component at path {path} is not a Body.')
            self.mobod_indexes.append(int(body.getMobilizedBodyIndex()))

    def to_group(self) -> BodyScaleGroup:
        return BodyScaleGroup(list(self.paths), list(self.mobod_indexes))

    def apply_to_model(self, model: osim.Model) -> None:
        raise NotImplementedError(
            'BodyScale.apply_to_model is not implemented.')


class MarkerOffset(Vec3Parameter):
    """
    An optimized Vec3 offset applied to one or more markers' placement, expressed in
    each marker's base frame. Pass a single marker path to offset one marker, or a list
    to share one set of offsets across a group of markers.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the marker(s) whose placement offset is optimized.
    bounds: Bounds
        Bounds applied to each Vec3 offset component.
    value: np.ndarray, optional
        Initial [ox, oy, oz] offset. Defaults to ``None`` (unset).
    """
    group_type = MarkerOffsetGroup
    cost_input = 'marker_offsets'

    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            marker = osim.Marker.safeDownCast(model.getComponent(path))
            loc = marker.get_location()
            marker.set_location(osim.Vec3(
                loc[0] + float(self.value[0]), loc[1] + float(self.value[1]),
                loc[2] + float(self.value[2])))

    def validate(self, mc: ModelCache) -> None:
        self.mobod_indexes = []
        for path in self.paths:
            marker = osim.Marker.safeDownCast(mc.model.getComponent(path))
            if marker is None:
                raise ValueError(f'Component at path {path} is not a Marker.')
            parent_frame = marker.getParentFrame()
            base_frame = osim.PhysicalFrame.safeDownCast(
                marker.getParentFrame().findBaseFrame())
            if (parent_frame.getAbsolutePathString() !=
                    base_frame.getAbsolutePathString()):
                raise ValueError(
                    f'Cannot optimize a marker offset for {path}: its parent '
                    f'frame ({parent_frame.getAbsolutePathString()}) is not its base '
                    f'frame ({base_frame.getAbsolutePathString()}). Offsets are only '
                    f'supported for markers attached directly to a body.')
            self.mobod_indexes.append(base_frame.getMobilizedBodyIndex())

    def to_group(self) -> MarkerOffsetGroup:
        return MarkerOffsetGroup(list(self.paths), list(self.mobod_indexes))


class FrameOffset(Vec3Parameter):
    """
    An optimized Vec3 offset applied to one or more `PhysicalOffsetFrame` translations,
    expressed in each frame's base frame. Pass a single frame path to offset one frame,
    or a list to share one set of offsets across a group of frames.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the frame(s) whose placement offset is optimized.
    bounds: Bounds
        Bounds applied to each Vec3 offset component.
    value: np.ndarray, optional
        Initial [ox, oy, oz] offset. Defaults to ``None`` (unset).
    """
    group_type = FrameOffsetGroup
    cost_input = 'frame_offsets'

    def __init__(self, paths: str | list[str], bounds: Bounds, value: np.ndarray):
        super().__init__(paths, bounds, value)
        self.mobod_indexes: list[int] = None

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            frame = osim.PhysicalOffsetFrame.safeDownCast(model.getComponent(path))
            t = frame.get_translation()
            frame.set_translation(osim.Vec3(
                t[0] + float(self.value[0]), t[1] + float(self.value[1]),
                t[2] + float(self.value[2])))

    def validate(self, mc: ModelCache) -> None:
        self.mobod_indexes = []
        for path in self.paths:
            frame = osim.PhysicalOffsetFrame.safeDownCast(mc.model.getComponent(path))
            if frame is None:
                raise ValueError(
                    f'Component at path {path} is not a PhysicalOffsetFrame.')
            parent_frame = frame.getParentFrame()
            base_frame = osim.PhysicalFrame.safeDownCast(
                frame.getParentFrame().findBaseFrame())
            if (parent_frame.getAbsolutePathString() !=
                    base_frame.getAbsolutePathString()):
                raise ValueError(
                    f'Cannot optimize a frame offset for {path}: its parent '
                    f'frame ({parent_frame.getAbsolutePathString()}) is not its base '
                    f'frame ({base_frame.getAbsolutePathString()}). Offsets are only '
                    f'supported for markers attached directly to a body.')
            self.mobod_indexes.append(base_frame.getMobilizedBodyIndex())

    def to_group(self) -> FrameOffsetGroup:
        return FrameOffsetGroup(list(self.paths), list(self.mobod_indexes))


class EllipsoidRadiiScale(Vec3Parameter):
    """
    An optimized Vec3 of dimensionless factors on `EllipsoidJoint` radii, shared across
    one or more joints. Pass a single joint path to optimize one joint's radii, or a
    list of joint paths to share one set of factors across a group of joints (e.g., for
    left-right symmetric shoulders).

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the `EllipsoidJoint`(s) whose radii are optimized.
    bounds: Bounds
        Bounds applied to each factor. The lower bound must be positive.
    value: np.ndarray
        Initial [fx, fy, fz] factors on the baseline radii.
    """
    group_type = EllipsoidRadiiScaleGroup
    cost_input = 'ellipsoid_radii_scales'

    def validate(self, mc: ModelCache) -> None:
        if self.bounds.lower_bound <= 0.0:
            raise ValueError(
                f'Ellipsoid radii factors must be positive, but the lower bound on '
                f'{self.paths} is {self.bounds.lower_bound}.')
        for path in self.paths:
            joint = osim.EllipsoidJoint.safeDownCast(mc.model.getComponent(path))
            if joint is None:
                raise ValueError(
                    f'Component at path {path} is not an EllipsoidJoint.')

    def to_group(self) -> EllipsoidRadiiScaleGroup:
        return EllipsoidRadiiScaleGroup(list(self.paths))

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            joint = osim.EllipsoidJoint.safeDownCast(model.getComponent(path))
            radii = joint.get_radii_x_y_z().to_numpy() * self.value
            joint.set_radii_x_y_z(osim.Vec3(
                float(radii[0]), float(radii[1]), float(radii[2])))


class BeamLengthScale(ScalarParameter):
    """
    An optimized dimensionless factor on a `CantileverFreeBeamJoint`'s beam length,
    shared across one or more joints. Pass a single joint path to optimize one joint's
    length, or a list of joint paths to share one factor across a group of joints.

    Parameters
    ----------
    paths: str or list[str]
        Absolute model path(s) to the `CantileverFreeBeamJoint`(s) whose beam length is
        optimized.
    bounds: Bounds
        Bounds applied to the factor. The lower bound must be positive.
    value: float
        Initial factor on the baseline beam length.
    """
    group_type = BeamLengthScaleGroup
    cost_input = 'beam_length_scales'

    def validate(self, mc: ModelCache) -> None:
        if self.bounds.lower_bound <= 0.0:
            raise ValueError(
                f'Beam length factors must be positive, but the lower bound on '
                f'{self.paths} is {self.bounds.lower_bound}.')
        for path in self.paths:
            joint = osim.CantileverFreeBeamJoint.safeDownCast(
                mc.model.getComponent(path))
            if joint is None:
                raise ValueError(
                    f'Component at path {path} is not a CantileverFreeBeamJoint.')

    def to_group(self) -> BeamLengthScaleGroup:
        return BeamLengthScaleGroup(list(self.paths))

    def apply_to_model(self, model: osim.Model) -> None:
        for path in self.paths:
            joint = osim.CantileverFreeBeamJoint.safeDownCast(
                model.getComponent(path))
            joint.set_beam_length(
                float(joint.get_beam_length() * self.value[0]))
