import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from .data_sources import Trial
from .model import ModelCache, StationCache
from .scaling import AnthropometricMeasurement
from .anthropometrics import build_ansur_distribution


##################
# COST INTERFACE #
##################

@dataclass
class CostInput:
    """
    Bundles the optimization variables passed to a cost evaluation. The canonical
    ordering of cost inputs is defined via the `INPUT_ORDER` attribute. Unused inputs
    default to empty symbolic arrays so they remain valid callback arguments.

    Attributes
    ----------
    INPUT_ORDER: tuple[str, ...]
        The canonical order of the optimization-variable inputs.
    coordinates: ca.MX, optional
        Coordinate values (e.g., joint angles).
    body_scales: ca.MX, optional
        Flattened per-group XYZ body-scale factors.
    marker_offsets: ca.MX, optional
        Flattened per-group XYZ marker offsets.
    frame_offsets: ca.MX, optional
        Flattened per-group XYZ frame offsets.
    ellipsoid_radii_scales: ca.MX, optional
        Flattened per-group XYZ factors on the baseline ellipsoid radii.
    beam_length_scales: ca.MX, optional
        Per-group factors on the baseline cantilever-free-beam lengths.
    """
    INPUT_ORDER: ClassVar[tuple[str, ...]] = (
        'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets',
        'ellipsoid_radii_scales', 'beam_length_scales')
    TRIPLET_INPUTS: ClassVar[tuple[str, ...]] = (
        'body_scales', 'marker_offsets', 'frame_offsets', 'ellipsoid_radii_scales')

    coordinates: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    body_scales: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    marker_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    frame_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    ellipsoid_radii_scales: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    beam_length_scales: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))

    @classmethod
    def field_index(cls, name: str) -> int:
        """
        Return the canonical position of `name` in `INPUT_ORDER`, validating that it is
        a recognized input field. Solvers use this both to reject parameters that
        declare an unknown input and to order parameter blocks by `INPUT_ORDER`.

        Parameters
        ----------
        name: str
            The `CostInput` field name to look up.

        Returns
        -------
        int
            The index of `name` in `INPUT_ORDER`.

        Raises
        ------
        ValueError
            If `name` is not a recognized `CostInput` field.
        """
        if name not in cls.INPUT_ORDER:
            raise ValueError(
                f'{name!r} is not a recognized CostInput field {cls.INPUT_ORDER}.')
        return cls.INPUT_ORDER.index(name)

    def as_triplets(self, name: str) -> ca.MX:
        """
        Return a per-group XYZ input reshaped to an ``(n, 3)`` matrix whose row ``i`` is
        the ``(x, y, z)`` triplet of group ``i``. The flat storage is laid out
        triplet-contiguous (``[s0x, s0y, s0z, s1x, ...]``), so the view is the
        column-major ``(3, n)`` reshape transposed.

        Parameters
        ----------
        name: str
            The field to view. Must be one of `TRIPLET_INPUTS` (`body_scales`,
            `marker_offsets`, `frame_offsets`, or `ellipsoid_radii_scales`).

        Returns
        -------
        ca.MX
            The field as an ``(n, 3)`` matrix.

        Raises
        ------
        ValueError
            If `name` is not a triplet input, or its length is not a multiple of 3.
        """
        if name not in self.TRIPLET_INPUTS:
            raise ValueError(
                f'{name!r} is not a triplet input; expected one of '
                f'{self.TRIPLET_INPUTS}.')
        value = getattr(self, name)
        num_entries = value.numel()
        if num_entries % 3 != 0:
            raise ValueError(
                f'{name!r} has length {num_entries}, which is not a multiple of 3.')
        return ca.reshape(value, 3, num_entries // 3).T


class CostRep(ABC):
    """
    A cost's per-solve representation that a solver evaluates.

    A `CostRep` is constructed via a cost's `create_rep` from the solver's `ModelCache`
    and lives only for the duration of one solve.
    """

    @abstractmethod
    def __call__(self, input: CostInput) -> ca.MX:
        pass


class CostBase(ABC):
    """
    A model-independent description of a cost term: the weights, targets, and reference
    data the user (or a solver) supplies, with no OpenSim state of its own. A cost is
    inert and reusable. All model-bound work happens in the `CostRep` it creates.

    Attributes
    ----------
    required_inputs: frozenset[str]
        The `CostInput` field names this cost reads and therefore requires the solver to
        provide (e.g., ``{'body_scales'}``). A solver validates that it provides every
        required input before accepting the cost; see `Solver.add_cost`.
    """
    required_inputs: frozenset[str] = frozenset()


class Cost(CostBase):
    """
    A cost whose rep is built from the `ModelCache` alone, and so the kind of cost a
    user can register on a solver via `Solver.add_cost`.
    """

    @abstractmethod
    def create_rep(self, mc: ModelCache) -> CostRep:
        """
        Build one of this cost's representations against `mc`. Solvers call this once
        for every rep the problem needs, after all parameter groups are registered, and
        hold the returned reps for the lifetime of the solve.

        Parameters
        ----------
        mc: ModelCache
            The solver's `ModelCache`, from which the rep initializes any model-derived
            quantities.
        """


class TrackingCostBase(CostBase):
    """
    A cost evaluated at a single time sample of a single trial, whose rep therefore
    needs that trial and sample index at construction.
    """

    @abstractmethod
    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> CostRep:
        """
        Build this cost's representation of one time sample of one trial. Solvers call
        this once for every sample the problem tracks, after all parameter groups are
        registered, and hold the returned reps for the lifetime of the solve.

        Parameters
        ----------
        name: str
            The name of the rep's callback function.
        mc: ModelCache
            The solver's `ModelCache`, from which the rep initializes any model-derived
            quantities.
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` that the rep tracks.
        """


class SymbolicCost(Cost):
    """
    A `Cost` whose rep is a plain CasADi expression, requiring no OpenSim evaluation.
    It is differentiated symbolically by CasADi and incurs no callback overhead.
    """

    @abstractmethod
    def evaluate(self, input: CostInput) -> ca.MX:
        """
        Return this cost's CasADi expression for `input`.
        """

    def create_rep(self, mc: ModelCache) -> 'SymbolicCostRep':
        return SymbolicCostRep(self)


class SymbolicCostRep(CostRep):
    """
    The rep of a `SymbolicCost`. It holds no model-derived state, since the cost is a
    pure expression over the optimization variables, and simply defers to
    `SymbolicCost.evaluate`.

    Parameters
    ----------
    cost: SymbolicCost
        The cost this rep represents.
    """

    def __init__(self, cost: SymbolicCost):
        self.cost = cost

    def __call__(self, input: CostInput) -> ca.MX:
        return self.cost.evaluate(input)


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
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function
        and its Jacobian and caching model information.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian (`get_jacobian`). Default is ``False``.
    """
    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        ca.Callback.__init__(self)
        self.mc = mc
        self.state = self.mc.state
        self.enable_fd = enable_fd
        self.construct(name, {'enable_fd': True} if enable_fd else {})

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


class CallbackCostRep(CostRep, Function):
    """
    A `CostRep` backed by a CasADi callback function that evaluates the cost and its
    Jacobian through OpenSim. Constructed with a fully-populated `ModelCache`, so the
    input sizes it declares to CasADi match the solver's registered parameter groups.
    """

    def __call__(self, input: CostInput) -> ca.MX:
        return ca.Function.__call__(
            self, *(getattr(input, name) for name in CostInput.INPUT_ORDER))

    def _get_num_inputs(self):
        return len(CostInput.INPUT_ORDER)

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        sizes = {
            'coordinates': len(self.mc.coordinate_q_indexes),
            'body_scales': 3 * len(self.mc.body_scale_groups),
            'marker_offsets': 3 * len(self.mc.marker_offset_groups),
            'frame_offsets': 3 * len(self.mc.frame_offset_groups),
            'ellipsoid_radii_scales': 3 * len(self.mc.ellipsoid_radii_scale_groups),
            'beam_length_scales': len(self.mc.beam_length_scale_groups),
        }
        order = CostInput.INPUT_ORDER
        if not 0 <= i < len(order):
            raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')
        return sizes[order[i]]

    def _get_output_size(self, i):
        if i == 0:
            return 1
        raise IndexError(f'Invalid output index {i} for {type(self).__name__}.')


class BodyScaleRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on body-scale factors that encourages each toward `target`:

        cost = weight * sum_i (s_i - target)^2

    Keeping the scales near ``target`` (typically 1.0, i.e., identity scaling) means the
    optimizer only deviates from the nominal scaling when doing so substantially
    improves the primary tracking cost.

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-component target value. Default is 1.0.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.body_scales - self.target)**2)


class BodyScaleIsotropyCost(SymbolicCost):
    """
    A quadratic penalty encouraging each body-scale group to scale isotropically, i.e.
    equally along X, Y, and Z:

        cost = weight * sum_g sum_axis (s_{g,axis} - mean_axis(s_g))^2

    where ``mean_axis(s_g)`` is the average of group ``g``'s three scale factors. It
    penalizes a group being stretched more along one axis than another without
    constraining its overall size.

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, weight: float):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

    def evaluate(self, input: CostInput) -> ca.MX:
        scales = input.as_triplets('body_scales')
        axis_means = ca.sum2(scales) / 3
        deviations = scales - ca.repmat(axis_means, 1, 3)
        return self.weight * ca.sumsqr(deviations)


class OffsetRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on marker and frame XYZ offsets, penalizing offsets away from
    zero:

        cost = weight * sum_i offset_i^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    """
    required_inputs = frozenset({'marker_offsets', 'frame_offsets'})

    def __init__(self, weight: float):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

    def evaluate(self, input: CostInput) -> ca.MX:
        offsets = ca.vertcat(input.marker_offsets, input.frame_offsets)
        return self.weight * ca.sum(offsets**2)


class EllipsoidRadiiScaleRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on the ellipsoid radii factors that encourages each toward
    `target`:

        cost = weight * sum_i (f_i - target)^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-factor target value. Default is 1.0.
    """
    required_inputs = frozenset({'ellipsoid_radii_scales'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.ellipsoid_radii_scales - self.target)**2)


class BeamLengthScaleRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on the beam length factors that encourages each toward
    `target`:

        cost = weight * sum_i (f_i - target)^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-factor target value. Default is 1.0.
    """
    required_inputs = frozenset({'beam_length_scales'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.beam_length_scales - self.target)**2)


class CoordinateStiffnessCost(Cost):
    """
    A quadratic penalty that acts like a spring on selected coordinates, holding each
    near a target value:

        cost = weight * sum_i k_i * (q_i - target_i)^2

    Parameters
    ----------
    stiffnesses: dict[str, float]
        Mapping from absolute coordinate path to that coordinate's non-negative
        stiffness. Only the coordinates named here are penalized.
    targets: dict[str, float], optional
        Mapping from absolute coordinate path to the value that coordinate is pulled
        toward. Any coordinate absent from this mapping is pulled toward its default
        value in the model. Defaults to ``None`` (every target taken from the model).
    weight: float, optional
        Non-negative scalar applied to the whole sum. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` or any stiffness is negative, or `stiffnesses` is empty.
        `CoordinateStiffnessCostRep` additionally validates the coordinate paths
        against the model.
    """
    required_inputs = frozenset({'coordinates'})

    def __init__(self, stiffnesses: dict[str, float],
                 targets: dict[str, float] = None, weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        if not stiffnesses:
            raise ValueError(
                'CoordinateStiffnessCost requires at least one coordinate stiffness.')
        for path, stiffness in stiffnesses.items():
            if stiffness < 0:
                raise ValueError(
                    f'Expected the stiffness for {path} to be non-negative, but got '
                    f'{stiffness}.')
        self.weight = weight
        self.stiffnesses = dict(stiffnesses)
        self.targets = dict(targets) if targets else {}

    def create_rep(self, mc: ModelCache) -> 'CoordinateStiffnessCostRep':
        return CoordinateStiffnessCostRep(self, mc)


class CoordinateStiffnessCostRep(CostRep):
    """
    The rep of a `CoordinateStiffnessCost`.

    Parameters
    ----------
    cost: CoordinateStiffnessCost
        The cost this rep represents.
    mc: ModelCache
        The solver's `ModelCache`, supplying the coordinate ordering and the default
        coordinate values.

    Raises
    ------
    ValueError
        If a coordinate path is not an independent coordinate of the model. Dependent
        (e.g. constrained) coordinates are absent from `ModelCache.coordinate_q_map` and
        so cannot be penalized directly.
    """

    def __init__(self, cost: CoordinateStiffnessCost, mc: ModelCache):
        self.cost = cost
        # Element j of the coordinates vector is the j-th entry of coordinate_q_map, so
        # a coordinate's position in that ordering is its index into the input.
        order = list(mc.coordinate_q_map)
        self.indexes: list[int] = []
        self.stiffnesses: list[float] = []
        self.targets: list[float] = []
        for path, stiffness in cost.stiffnesses.items():
            if path not in mc.coordinate_q_map:
                known = 'is a dependent coordinate' if mc.model.hasComponent(path) \
                    else 'is not a coordinate in the model'
                raise ValueError(
                    f'Cannot apply a coordinate stiffness to {path}: it {known}. '
                    f'Expected one of the model\'s independent coordinates.')
            self.indexes.append(order.index(path))
            self.stiffnesses.append(float(stiffness))
            if path in cost.targets:
                self.targets.append(float(cost.targets[path]))
            else:
                coordinate = osim.Coordinate.safeDownCast(mc.model.getComponent(path))
                self.targets.append(float(coordinate.getDefaultValue()))

    def __call__(self, input: CostInput) -> ca.MX:
        penalty = 0
        for index, stiffness, target in zip(
                self.indexes, self.stiffnesses, self.targets):
            penalty += stiffness * (input.coordinates[index] - target)**2
        return self.cost.weight * penalty


###########
# HELPERS #
###########

#########
# TASKS #
#########

@dataclass
class TaskArrays:
    """
    Per-task quantities packed into arrays, so a term can evaluate every task in a few
    vectorized numpy operations.

    Attributes
    ----------
    base_stations: np.ndarray, shape (num_tasks, 3)
        Each task's point in its base frame, before scaling or offsets.
    mobod_indexes: np.ndarray, shape (num_tasks,)
        Each task's base-frame `MobilizedBodyIndex`.
    scale_groups: np.ndarray, shape (num_tasks,)
        Index of the `BodyScaleGroup` scaling each task, or -1 where none applies.
    offset_groups: np.ndarray, shape (num_tasks,)
        Index of the offset group applying to each task, or -1 where none applies.
    base_frames: list[osim.PhysicalFrame]
        The distinct base frames across all tasks, one per entry of `body_rows`.
    body_rows: np.ndarray, shape (num_tasks,)
        For each task, the index into `base_frames` of its base frame.
    reference_positions: np.ndarray, shape (num_tasks, 3)
        Each task's reference position.
    weights: np.ndarray or None, shape (num_tasks,)
        Each task's cost weight, for terms carrying a single weight per task.
        ``None`` for terms that weight position and orientation separately.
    scaled_rows, scaled_groups: np.ndarray
        The tasks a body scale applies to, and the scale group of each. Paired, so
        ``stations[scaled_rows] *= scales[scaled_groups]`` applies every scale at once.
    shifted_rows, shifted_groups: np.ndarray
        The same pairing for placement offsets.
    scaled_base_stations: np.ndarray, shape (len(scaled_rows), 3)
        ``base_stations[scaled_rows]``, used when accumulating the body-scale Jacobian.
    unique_mobod_indexes: np.ndarray
        The distinct base-frame mobod indexes across all tasks, i.e. exactly the rows a
        per-body gradient scatter can touch.
    double_weights_column: np.ndarray or None, shape (num_tasks, 1)
        ``2 * weights`` as a column, the factor the squared-error gradient needs.
    position_weights, orientation_weights: np.ndarray or None
        Each task's separate position and orientation weights, for terms that carry
        both. ``None`` for terms with a single weight per task.
    double_position_weights_column: np.ndarray or None, shape (num_tasks, 1)
        ``2 * position_weights`` as a column.
    reference_orientations: np.ndarray or None, shape (num_tasks, 4)
        Each task's reference quaternion.
    base_relative_rotations: np.ndarray or None, shape (num_tasks, 3, 3)
        Each task's rotation relative to its base frame, which is fixed. A frame's
        rotation in ground is ``R_GB @ R_BF``, so one transform read per base frame
        yields every task's orientation.
    reference_rotations: np.ndarray or None, shape (num_tasks, 3, 3)
        Each task's reference orientation as a rotation matrix.
    """
    base_stations: np.ndarray
    mobod_indexes: np.ndarray
    scale_groups: np.ndarray
    offset_groups: np.ndarray
    base_frames: list
    body_rows: np.ndarray
    reference_positions: np.ndarray
    weights: np.ndarray = None
    scaled_rows: np.ndarray = None
    scaled_groups: np.ndarray = None
    shifted_rows: np.ndarray = None
    shifted_groups: np.ndarray = None
    scaled_base_stations: np.ndarray = None
    unique_mobod_indexes: np.ndarray = None
    double_weights_column: np.ndarray = None
    position_weights: np.ndarray = None
    orientation_weights: np.ndarray = None
    double_position_weights_column: np.ndarray = None
    reference_orientations: np.ndarray = None
    base_relative_rotations: np.ndarray = None
    reference_rotations: np.ndarray = None

    def __post_init__(self):
        self.scaled_rows = np.flatnonzero(self.scale_groups >= 0)
        self.scaled_groups = self.scale_groups[self.scaled_rows]
        self.shifted_rows = np.flatnonzero(self.offset_groups >= 0)
        self.shifted_groups = self.offset_groups[self.shifted_rows]
        self.scaled_base_stations = self.base_stations[self.scaled_rows]
        self.unique_mobod_indexes = np.unique(self.mobod_indexes)
        self.double_weights_column = (None if self.weights is None
                                      else 2.0 * self.weights[:, None])


class Tasks(ABC):
    """
    A base class for task-specific storage and registration.
    """
    @abstractmethod
    def initialize_tasks(self, state: osim.State, **kwargs) -> float:
        pass

    def invalidate_task_arrays(self) -> None:
        """
        Drop the cached `TaskArrays`. Called whenever a task is registered, so the
        arrays are rebuilt on next use.
        """
        self._task_arrays = None

    @property
    def task_arrays(self) -> TaskArrays:
        """
        The cached `TaskArrays` for the registered tasks, built on first use.
        """
        if getattr(self, '_task_arrays', None) is None:
            self._task_arrays = self._build_task_arrays()
        return self._task_arrays

    def _build_task_arrays(self) -> TaskArrays:
        num_tasks = self.num_tasks
        base_stations = (np.asarray(self.base_stations, dtype=float).reshape(-1, 3)
                         if num_tasks else np.zeros((0, 3)))
        mobod_indexes = np.array(
            [int(self.mobod_indexes.getElt(i)) for i in range(num_tasks)], dtype=int)
        scale_groups = np.array(
            [-1 if cache.body_scale_group_index is None
             else cache.body_scale_group_index for cache in self.station_caches],
            dtype=int)
        offset_groups = np.array(
            [-1 if g is None else g for g in self.offset_group_indexes], dtype=int)

        # One entry per distinct base frame, in first-seen order.
        frames, rows, row_of_mobod = [], [], {}
        for i, cache in enumerate(self.station_caches):
            key = cache.mobod_index
            if key not in row_of_mobod:
                row_of_mobod[key] = len(frames)
                frames.append(cache.base_frame)
            rows.append(row_of_mobod[key])

        reference_positions = (np.asarray(self.positions, dtype=float).reshape(-1, 3)
                               if num_tasks else np.zeros((0, 3)))
        single_weights = getattr(self, 'weights', None)
        weights = (np.asarray(single_weights, dtype=float)
                   if single_weights is not None else None)

        return TaskArrays(
            base_stations=base_stations, mobod_indexes=mobod_indexes,
            scale_groups=scale_groups, offset_groups=offset_groups,
            base_frames=frames, body_rows=np.array(rows, dtype=int),
            reference_positions=reference_positions, weights=weights)


class MarkerTasks(Tasks):
    """
    Marker-specific task storage and registration.
    """
    def initialize_tasks(self):
        self.markers = []
        self.station_caches: list[StationCache] = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.weights = []
        self.base_frames = []
        self.base_stations = []
        self.offset_group_indexes: list[int] = []
        self._task_arrays: TaskArrays = None

    def add_marker(self, marker_path: str, position: osim.Vec3, weight: float = 1.0,
                   offset_group_index: int | None = None):
        """
        Register a marker to track.

        Parameters
        ----------
        marker_path: str
            The OpenSim Model path to the tracking marker.
        position: osim.Vec3
            The reference position data tracked by the model marker.
        weight: float, optional
            The cost weight for the position error. Default: 1.0.
        offset_group_index: int | None, optional
            The index of the offset group whose XYZ offset applies to this marker, or
            ``None`` if this marker's placement is not offset. Default: ``None``.
        """
        if not self.mc.model.hasComponent(marker_path):
            raise ValueError(f'Model does not have a component at path {marker_path}.')
        if weight < 0:
            raise ValueError(f'Expected weight to be non-negative, but got {weight}.')

        self.mc.model.realizePosition(self.mc.state)
        marker = osim.Marker.safeDownCast(self.mc.model.getComponent(marker_path))
        cache = StationCache.from_station(self.mc, marker)
        self.station_caches.append(cache)
        self.markers.append(marker)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.weights.append(weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)
        self.invalidate_task_arrays()


class FrameTasks(Tasks):
    """
    Frame-specific task storage and registration.
    """
    def initialize_tasks(self):
        self.frames = []
        self.station_caches: list[StationCache] = []
        self.mobod_indexes = osim.SimTKArrayInt()
        self.stations = osim.SimTKArrayVec3()
        self.num_tasks: int = 0
        self.positions = []
        self.orientations = []
        self.position_weights = []
        self.orientation_weights = []
        self.base_frames = []
        self.base_stations = []
        self.offset_group_indexes: list[int] = []
        self._task_arrays: TaskArrays = None

    def add_frame(self, frame_path: str, position: osim.Vec3,
                  orientation: osim.Quaternion, position_weight: float = 1.0,
                  orientation_weight: float = 1.0,
                  offset_group_index: int | None = None):
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
        position_weight: float, optional
            The cost weight for the position error. Default: 1.0.
        orientation_weight: float, optional
            The cost weight for the orientation error. Default: 1.0.
        offset_group_index: int | None, optional
            The index of the offset group whose XYZ offset applies to this frame, or
            ``None`` if this frame's placement is not offset. Default: ``None``.
        """
        if not self.mc.model.hasComponent(frame_path):
            raise ValueError(f'Model does not have a component at path {frame_path}.')
        if position_weight < 0:
            raise ValueError(f'Expected position_weight to be non-negative, but got '
                             f'{position_weight}.')
        if orientation_weight < 0:
            raise ValueError(f'Expected orientation_weight to be non-negative, but got '
                             f'{orientation_weight}.')

        frame = osim.PhysicalFrame.safeDownCast(self.mc.model.getComponent(frame_path))
        cache = StationCache.from_frame(self.mc, frame)
        self.station_caches.append(cache)
        self.frames.append(frame)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(position.to_numpy())
        self.orientations.append(np.array([orientation.get(i) for i in range(4)]))
        self.position_weights.append(position_weight)
        self.orientation_weights.append(orientation_weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)
        self.invalidate_task_arrays()

    def _build_task_arrays(self) -> TaskArrays:
        arrays = super()._build_task_arrays()
        arrays.position_weights = np.asarray(self.position_weights, dtype=float)
        arrays.orientation_weights = np.asarray(self.orientation_weights, dtype=float)
        arrays.double_position_weights_column = (
            2.0 * arrays.position_weights[:, None] if self.num_tasks
            else np.zeros((0, 1)))
        arrays.reference_orientations = (
            np.asarray(self.orientations, dtype=float).reshape(-1, 4)
            if self.num_tasks else np.zeros((0, 4)))
        base_relative_rotations = []
        for frame in self.frames:
            transform = frame.findTransformInBaseFrame()
            base_relative_rotations.append(transform.R().to_numpy())
        arrays.base_relative_rotations = (
            np.array(base_relative_rotations) if self.num_tasks
            else np.zeros((0, 3, 3)))
        arrays.reference_rotations = (
            np.array([osim.Rotation(osim.Quaternion(*[float(v) for v in q])).to_numpy()
                      for q in arrays.reference_orientations])
            if self.num_tasks else np.zeros((0, 3, 3)))
        return arrays


##############
# COST TERMS #
##############

class TrackingTerm(ABC):
    """
    A base class for tracking cost terms that compute a scalar error and its
    Jacobian with respect to the model's generalized coordinates, body scales,
    and other optimization variables. To implement a new tracking cost term, extend this
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


class FrameTrackingTerm(FrameTasks, TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model frames'
    positions and orientations and corresponding reference data as a function of the
    model's generalized coordinates. Individual frames are registered via add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        positions = origins[rows] + (R_GB @ arrays.base_stations[..., None])[..., 0]
        residuals = positions - arrays.reference_positions
        position_error = arrays.position_weights * np.sum(
            residuals * residuals, axis=1)

        # trace(R_ref^T R) == 1 + 2cos(theta), so
        # (3 - trace(R_ref^T R)) / 4 == (1 - cos(theta)) / 2 == sin(theta/2)**2,
        # which is equal to the quaternion alignment loss 1 - (eps . q_ref)**2.
        frame_rotations = R_GB @ arrays.base_relative_rotations
        trace = np.einsum('nij,nij->n', arrays.reference_rotations, frame_rotations)
        orientation_error = arrays.orientation_weights * (3.0 - trace) / 4.0

        return float(np.sum(position_error + orientation_error))

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_q_indexes)))]

        # Compute the "spatial error" (i.e., the combined position and orientation
        # error) for every frame at once.
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        positions = origins[rows] + (R_GB @ arrays.base_stations[..., None])[..., 0]
        position_error = arrays.double_position_weights_column * (
            positions - arrays.reference_positions)

        # E = w_o * (1 - (eps . q_ref)**2) = w_o * (3 - trace(R_ref^T R)) / 4
        #
        # dE/dq = dE/dR * dR/dq = (-w_o / 4) * tr(R_ref^T dR/dq)
        #
        # dR/dq = [z]x R = [Jw]x R, where z is the axis of rotation of the frame, which
        #                           is equal to Jw to the angular Jacobian of the frame
        #                           in ground.
        #
        # dE/dq = (-w_o / 4) * tr(R_ref^T [Jw]x R)
        #       = (-w_o / 4) * tr([Jw]x R R_ref^T)
        #       = (-w_o / 4) * tr([Jw]x M)          -->  M = R R_ref^T
        #       = (-w_o / 4) * Jw^T [M12 - M21, M20 - M02, M01 - M10]
        frame_rotations = R_GB @ arrays.base_relative_rotations
        aligned = np.einsum('nij,nkj->nik', frame_rotations,
                            arrays.reference_rotations)
        orientation_error = (-0.25 * arrays.orientation_weights)[:, None] * np.stack(
            [aligned[:, 1, 2] - aligned[:, 2, 1],
             aligned[:, 2, 0] - aligned[:, 0, 2],
             aligned[:, 0, 1] - aligned[:, 1, 0]], axis=1)

        # A SpatialVec holds the angular half first, then the linear half.
        spatialError = osim.VectorOfSpatialVec.createFromMat(
            np.ascontiguousarray(
                np.concatenate([orientation_error, position_error], axis=1),
                dtype=float).reshape(-1))

        # Calculate the frame (position and orientation) error Jacobian.
        Ju = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, Ju)
        Jq = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, Ju, Jq)

        return [np.expand_dims(Jq.to_numpy()[self.mc.coordinate_q_indexes], axis=0)]


class MarkerTrackingTerm(MarkerTasks, TrackingTerm):
    """
    A tracking cost term that computes the aggregate error between model markers'
    positions and corresponding reference positions as a function of the model's
    generalized coordinates. Individual markers are registered via add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rows = arrays.body_rows
        # p_GS = p_GB + R_GB · p_BS
        p_GB = poses[:, :, 3][rows]
        R_GB = poses[:, :, :3][rows]
        positions = (p_GB + (R_GB @ arrays.base_stations[..., None])[..., 0])
        residuals = positions - arrays.reference_positions
        return float(np.sum(arrays.weights * np.sum(residuals * residuals, axis=1)))

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_q_indexes)))]

        # Initialize the array used to calculate the position error Jacobian via the
        # grouped Simbody operator.
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rows = arrays.body_rows
        # p_GS = p_GB + R_GB · p_BS
        p_GB = poses[:, :, 3][rows]
        R_GB = poses[:, :, :3][rows]
        positions = (p_GB + (R_GB @ arrays.base_stations[..., None])[..., 0])
        gradient = arrays.double_weights_column * (
            positions - arrays.reference_positions)
        f_GP = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(gradient, dtype=float).reshape(-1))

        # Calculate the position error Jacobian.
        Ju = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, f_GP, Ju)
        Jq = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, Ju, Jq)

        return [np.expand_dims(Jq.to_numpy()[self.mc.coordinate_q_indexes], axis=0)]


class BilevelTerm(TrackingTerm):
    """
    An intermediate base class that provides functionality common to bilevel cost terms.

    Applying body scales and placement offsets to the cached station locations is shared
    across marker and frame terms; subclasses (via `MarkerTasks`/`FrameTasks`) supply the
    per-task `station_caches`, `stations`, and `offset_group_indexes`.
    """
    def __init__(self):
        super().__init__()
        self._station_array = None

    @property
    def station_array(self) -> np.ndarray:
        cached = self._station_array
        if cached is None or len(cached) != self.num_tasks:
            cached = self._station_array = self.task_arrays.base_stations.copy()
        return cached

    def apply_state(self, body_scales: np.ndarray, offsets: np.ndarray) -> None:
        arrays = self.task_arrays
        if len(arrays.scaled_rows) or len(arrays.shifted_rows):
            stations = arrays.base_stations.copy()
            if len(arrays.scaled_rows):
                scales = np.asarray(body_scales, dtype=float).reshape(-1, 3)
                stations[arrays.scaled_rows] *= scales[arrays.scaled_groups]
            if len(arrays.shifted_rows):
                shifts = np.asarray(offsets, dtype=float).reshape(-1, 3)
                stations[arrays.shifted_rows] += shifts[arrays.shifted_groups]
        else:
            stations = arrays.base_stations
        self._station_array = stations
        self.stations = osim.SimTKArrayVec3.createFromMat(
            np.ascontiguousarray(stations, dtype=float).reshape(-1))


class MarkerBilevelTerm(MarkerTasks, BilevelTerm):
    """
    A tracking cost term that computes the aggregate error between model markers' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual markers are registered via
    add_marker().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rows = arrays.body_rows
        # p_GS = p_GB + R_GB · p_BS
        p_GB = poses[:, :, 3][rows]
        R_GB = poses[:, :, :3][rows]
        positions = (p_GB + (R_GB @ self.station_array[..., None])[..., 0])
        residuals = positions - arrays.reference_positions
        return float(np.sum(arrays.weights * np.sum(residuals * residuals, axis=1)))

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_q_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.marker_offset_groups)))
        Jr = np.zeros((1, 3 * len(self.mc.ellipsoid_radii_scale_groups)))
        Jl = np.zeros((1, len(self.mc.beam_length_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo, Jr, Jl]

        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows

        # The per-marker error gradient in Ground.
        R_GB = rotations[rows]
        p_GS = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        gradient = arrays.double_weights_column * (p_GS - arrays.reference_positions)
        dp_GS = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(gradient, dtype=float).reshape(-1))

        # The sensitivity of each marker's ground position to a shift from an offset
        # variable.
        doffset = (gradient[:, None, :] @ R_GB)[:, 0, :]

        # Calculate the Jacobian of the position error with respect to the coordinates.
        grad_u = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, dp_GS, grad_u)
        grad_q = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
        Jq[0, :] = grad_q.to_numpy()[self.mc.coordinate_q_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GS_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GS.get(i)   # for each marker i on body k
        #
        accumulated = np.zeros((self.mc.num_mobod, 3))
        np.add.at(accumulated, arrays.mobod_indexes, gradient)
        dp_GB = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(accumulated, dtype=float).reshape(-1))

        # Calculate the position-error Jacobian with respect to body scales.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Calculate the position-error Jacobians with respect to the joint-level
        # parameters.
        Jr = self.mc.calc_position_jacobian_wrt_ellipsoid_radii_scales(state, dp_GB)
        Jl = self.mc.calc_position_jacobian_wrt_beam_length_scales(state, dp_GB)

        # Assemble the marker offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the marker offsets to the Jacobian with respect
        # to body scales.
        if len(arrays.shifted_rows):
            np.add.at(Jo.reshape(-1, 3), arrays.shifted_groups,
                      doffset[arrays.shifted_rows])
        if len(arrays.scaled_rows):
            np.add.at(Js.reshape(-1, 3), arrays.scaled_groups,
                      arrays.scaled_base_stations * doffset[arrays.scaled_rows])

        return [Jq, Js, Jo, Jr, Jl]


class FrameBilevelTerm(FrameTasks, BilevelTerm):
    """
    A tracking cost term that computes the aggregate error between model frames' scaled
    positions and corresponding reference positions as a function of the model's
    generalized coordinates and body scales. Individual frames are registered via
    add_frame().

    Parameters
    ----------
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for
        evaluating the function and its Jacobian and caching model information.
    """
    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.initialize_tasks()

    def calc_error(self, state, **kwargs) -> float:
        if self.num_tasks == 0:
            return 0.0
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        # Positions come from the scaled and offset stations applied by apply_state,
        # so the error stays consistent with any applied offsets.
        positions = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        residuals = positions - arrays.reference_positions
        position_error = arrays.position_weights * np.sum(
            residuals * residuals, axis=1)

        # trace(R_ref^T R) == 1 + 2cos(theta), so
        # (3 - trace(R_ref^T R)) / 4 == (1 - cos(theta)) / 2 == sin(theta/2)**2,
        # which is equal to the quaternion alignment loss 1 - (eps . q_ref)**2.
        frame_rotations = R_GB @ arrays.base_relative_rotations
        trace = np.einsum('nij,nij->n', arrays.reference_rotations, frame_rotations)
        orientation_error = arrays.orientation_weights * (3.0 - trace) / 4.0

        return float(np.sum(position_error + orientation_error))

    def calc_jacobian(self, state, **kwargs) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_q_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.frame_offset_groups)))
        Jr = np.zeros((1, 3 * len(self.mc.ellipsoid_radii_scale_groups)))
        Jl = np.zeros((1, len(self.mc.beam_length_scale_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo, Jr, Jl]

        # Compute the combined position and orientation error for every frame at
        # once.
        arrays = self.task_arrays
        poses = np.array([frame.getTransformInGround(state).to_numpy()
                          for frame in arrays.base_frames])
        base_rotations, origins = poses[:, :, :3], poses[:, :, 3]
        rows = arrays.body_rows
        R_GB = base_rotations[rows]

        # The frames' ground positions come from their (possibly offset) cached
        # stations, so the gradient is consistent with any applied offsets.
        positions = origins[rows] + (R_GB @ self.station_array[..., None])[..., 0]
        gradient = arrays.double_position_weights_column * (
            positions - arrays.reference_positions)

        # E = w_o * (1 - (eps . q_ref)**2) = w_o * (3 - trace(R_ref^T R)) / 4
        #
        # dE/dq = dE/dR * dR/dq = (-w_o / 4) * tr(R_ref^T dR/dq)
        #
        # dR/dq = [z]x R = [Jw]x R, where z is the axis of rotation of the frame, which
        #                           is equal to Jw to the angular Jacobian of the frame
        #                           in ground.
        #
        # dE/dq = (-w_o / 4) * tr(R_ref^T [Jw]x R)
        #       = (-w_o / 4) * tr([Jw]x R R_ref^T)
        #       = (-w_o / 4) * tr([Jw]x M)          -->  M = R R_ref^T
        #       = (-w_o / 4) * Jw^T [M12 - M21, M20 - M02, M01 - M10]
        frame_rotations = R_GB @ arrays.base_relative_rotations
        aligned = np.einsum('nij,nkj->nik', frame_rotations,
                            arrays.reference_rotations)
        orientation_error = (-0.25 * arrays.orientation_weights)[:, None] * np.stack(
            [aligned[:, 1, 2] - aligned[:, 2, 1],
             aligned[:, 2, 0] - aligned[:, 0, 2],
             aligned[:, 0, 1] - aligned[:, 1, 0]], axis=1)

        # A SpatialVec holds the angular half first, then the linear half.
        spatialError = osim.VectorOfSpatialVec.createFromMat(
            np.ascontiguousarray(
                np.concatenate([orientation_error, gradient], axis=1),
                dtype=float).reshape(-1))

        # Sensitivity of each frame's ground position to a base-frame station shift.
        doffset = np.einsum('ni,nij->nj', gradient, R_GB)

        # Calculate the frame (position and orientation) error Jacobian.
        grad_u = osim.Vector(state.getNU(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, grad_u)
        grad_q = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByNInv(state, True, grad_u, grad_q)
        Jq[0, :] = grad_q.to_numpy()[self.mc.coordinate_q_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GF_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GF.get(i)   # for each frame i on body k
        #
        accumulated = np.zeros((self.mc.num_mobod, 3))
        np.add.at(accumulated, arrays.mobod_indexes, gradient)
        dp_GB = osim.VectorVec3.createFromMat(
            np.ascontiguousarray(accumulated, dtype=float).reshape(-1))

        # Calculate the position-error Jacobian with respect to body scales. This does
        # not include the contributions from frame offsets, we will include that below.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Calculate the position-error Jacobians with respect to the joint-level
        # parameters.
        Jr = self.mc.calc_position_jacobian_wrt_ellipsoid_radii_scales(state, dp_GB)
        Jl = self.mc.calc_position_jacobian_wrt_beam_length_scales(state, dp_GB)

        # Assemble the frame offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the frame offsets to the Jacobian with respect
        # to body scales.
        if len(arrays.shifted_rows):
            np.add.at(Jo.reshape(-1, 3), arrays.shifted_groups,
                      doffset[arrays.shifted_rows])
        if len(arrays.scaled_rows):
            np.add.at(Js.reshape(-1, 3), arrays.scaled_groups,
                      arrays.scaled_base_stations * doffset[arrays.scaled_rows])

        return [Jq, Js, Jo, Jr, Jl]


##################
# COST FUNCTIONS #
##################

class TrackingCost(TrackingCostBase):
    """
    The weighted, squared error between the model's markers and frames and a trial's
    reference data, as a function of the model's generalized coordinates.

    Parameters
    ----------
    position_weight: float, optional
        Weight applied to marker and frame-origin position errors. Default is 1.0.
    orientation_weight: float, optional
        Weight applied to frame orientation errors. Default is 1.0.
    """
    required_inputs = frozenset({'coordinates'})

    def __init__(self, position_weight: float = 1.0,
                 orientation_weight: float = 1.0):
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> 'TrackingCostRep':
        rep = TrackingCostRep(name, mc)

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                rep.add_frame_tracking_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight)

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                rep.add_marker_tracking_cost_term(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.position_weight)

        return rep


class TrackingCostRep(CallbackCostRep):
    """
    The rep of a `TrackingCost`: a callback that evaluates the sum of tracking cost
    terms over a set of model frames and markers with respect to the model's
    generalized coordinates.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian. Default is ``False``.
    """
    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        Function.__init__(self, name, mc, enable_fd=enable_fd)
        self.marker_term = MarkerTrackingTerm(mc)
        self.frame_term = FrameTrackingTerm(mc)

    def apply_state(self, arg):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(self.state.getNQ())
        q[self.mc.coordinate_q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_tracking_cost_term(self, marker_path: str, position: osim.Vec3,
                                      weight: float = 1.0):
        self.marker_term.add_marker(marker_path, position, weight=weight)

    def add_frame_tracking_cost_term(self, frame_path: str,
                                     position: osim.Vec3,
                                     orientation: osim.Quaternion,
                                     position_weight: float = 1.0,
                                     orientation_weight: float = 1.0):
        self.frame_term.add_frame(frame_path, position, orientation,
                                  position_weight=position_weight,
                                  orientation_weight=orientation_weight)

    def _eval(self, arg):
        self.apply_state(arg)
        error = (self.marker_term.calc_error(self.state) +
                 self.frame_term.calc_error(self.state))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        J = (self.marker_term.calc_jacobian(self.state)[0] +
             self.frame_term.calc_jacobian(self.state)[0])
        empty = np.zeros((1, 0))
        return [J] + [empty] * (len(CostInput.INPUT_ORDER) - 1)


class BilevelCost(TrackingCostBase):
    """
    The tracking cost of `TrackingCost`, as a function of the model's generalized
    coordinates, its body scales, its per-marker/frame XYZ placement offsets, and its
    joint-level geometry (e.g., ellipsoid radii).

    Parameters
    ----------
    position_weight: float, optional
        Weight applied to marker and frame-origin position errors. Default is 1.0.
    orientation_weight: float, optional
        Weight applied to frame orientation errors. Default is 1.0.
    """
    required_inputs = frozenset(
        {'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets',
         'ellipsoid_radii_scales', 'beam_length_scales'})

    def __init__(self, position_weight: float = 1.0,
                 orientation_weight: float = 1.0):
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> 'BilevelCostRep':
        rep = BilevelCostRep(name, mc)
        # Map each offset target path to the index of the offset group that applies to
        # it; paths absent from a mapping are not offset.
        marker_index_of = {path: i for i, grp in enumerate(mc.marker_offset_groups)
                           for path in grp.component_paths}
        frame_index_of = {path: i for i, grp in enumerate(mc.frame_offset_groups)
                          for path in grp.component_paths}

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                rep.add_frame_bilevel_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight,
                    offset_group_index=frame_index_of.get(frame_path))

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                rep.add_marker_bilevel_cost_term(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.position_weight,
                    offset_group_index=marker_index_of.get(marker_path))

        return rep


class BilevelCostRep(CallbackCostRep):
    """
    The rep of a `BilevelCost`: a callback that evaluates the sum of tracking cost
    terms over a set of model markers and frames with respect to the model's generalized
    coordinates, a set of body scales, a set of per-marker/frame XYZ placement offsets,
    a set of ellipsoid radii, and a set of beam lengths.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information. Contains parameter information
        (e.g., body scale groups) for relevant optimization parameters.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian. Default is ``False``.
    """

    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        Function.__init__(self, name, mc, enable_fd=enable_fd)
        self.marker_term = MarkerBilevelTerm(mc)
        self.frame_term = FrameBilevelTerm(mc)

    def apply_state(self, arg):
        """
        Apply input coordinates, body-scale variables, offset variables, and
        joint-level variables (ellipsoid radii and beam lengths) to the model State,
        then realize to Position.
        """
        body_scales = np.squeeze(arg[1].full())
        body_scales = np.atleast_1d(body_scales).astype(float)
        self.mc.set_scaled_mobilizer_frame_positions(self.state, body_scales)

        marker_offsets = np.atleast_1d(np.squeeze(arg[2].full())).astype(float)
        self.marker_term.apply_state(body_scales, marker_offsets)

        frame_offsets = np.atleast_1d(np.squeeze(arg[3].full())).astype(float)
        self.frame_term.apply_state(body_scales, frame_offsets)

        ellipsoid_radii_scales = np.atleast_1d(np.squeeze(arg[4].full())).astype(float)
        self.mc.set_ellipsoid_radii_from_scales(self.state, ellipsoid_radii_scales)

        beam_length_scales = np.atleast_1d(np.squeeze(arg[5].full())).astype(float)
        self.mc.set_beam_length_from_scales(self.state, beam_length_scales)

        q = np.zeros(self.state.getNQ())
        q[self.mc.coordinate_q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_bilevel_cost_term(self, marker_path: str, position: osim.Vec3,
                                     weight: float = 1.0,
                                     offset_group_index: int | None = None):
        self.marker_term.add_marker(marker_path, position, weight=weight,
                                    offset_group_index=offset_group_index)

    def add_frame_bilevel_cost_term(self, frame_path: str, position: osim.Vec3,
                                    orientation: osim.Quaternion,
                                    position_weight: float = 1.0,
                                    orientation_weight: float = 1.0,
                                    offset_group_index: int | None = None):
        self.frame_term.add_frame(frame_path, position, orientation, position_weight,
                                  orientation_weight,
                                  offset_group_index=offset_group_index)

    def _eval(self, arg):
        self.apply_state(arg)
        error = 0
        error += self.marker_term.calc_error(self.state)
        error += self.frame_term.calc_error(self.state)
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        Jq_m, Js_m, Jmo, Jr_m, Jl_m = self.marker_term.calc_jacobian(self.state)
        Jq_f, Js_f, Jfo, Jr_f, Jl_f = self.frame_term.calc_jacobian(self.state)
        return [Jq_m + Jq_f, Js_m + Js_f, Jmo, Jfo, Jr_m + Jr_f, Jl_m + Jl_f]


class AnthropometricRegularizationCost(Cost):
    """
    A regularization penalty on body-scale factors, ``s``, that maximizes the
    log-likelihood that a set of anthropometric measurements, ``m(s)``, fall within a
    distribution fit to the ANSUR II dataset. Since it is a multivariate normal
    distribution, we use the Mahalanobis distance to define the cost:

        cost = weight * 0.5 (m(s) - μ)^T Σ^-1 (m(s) - μ)

    which equates to minimizing the negative log-likelihood of the probability density
    function.

    Users must define the set of measurements, ``m(s)``, via the parameter
    `measurements`, a list of `AnthropometricMeasurement`. Each
    `AnthropometricMeasurement` is named after a measurement in the ANSUR II dataset
    and defines the two `Station`s on the `Model` (and optionally an axis) from which
    to compute the simulated measurement. The `sex`
    parameter can be used to specify that the distribution should be fit to either
    male or female participants only; using the default value (`None`) fits across all
    participants from the ANSUR report.

    All quantities are in meters (ANSUR II millimeters are converted on load).

    Parameters
    ----------
    measurements: list[AnthropometricMeasurement]
        The measurements (each a station pair and optional axis) to compute from the
        model. Each measurement's `name` must match a measurement from the ANSUR II
        dataset.
    sex: str, optional
        Subject sex ('male' or 'female') selecting the ANSUR II subset. Defaults to None
        (the combined male-and-female dataset).
    weight: float, optional
        Non-negative scalar applied to the penalty. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` is negative or a measurement name is not present in the ANSUR II
        dataset. `AnthropometricRegularizationCostRep` additionally validates that the
        referenced components are stations.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, measurements: list[AnthropometricMeasurement],
                 sex: str = None, weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.measurements = measurements

        # Fit the ANSUR II distribution over the requested measurements, in meters.
        measurement_names = [m.name for m in self.measurements]
        distribution = build_ansur_distribution(measurement_names, sex)
        self.mean = np.asarray(distribution.get_mean(), dtype=float).reshape(-1)
        self.precision = np.linalg.inv(
            np.asarray(distribution.get_covariance(), dtype=float))

    def create_rep(self, mc: ModelCache) -> 'AnthropometricRegularizationCostRep':
        return AnthropometricRegularizationCostRep(self, mc)


class AnthropometricRegularizationCostRep(CallbackCostRep):
    """
    The rep of an `AnthropometricRegularizationCost`. It caches the model's default
    pose and a `StationCache` pair per measurement, then evaluates the Mahalanobis
    penalty and its gradient through OpenSim.

    Parameters
    ----------
    cost: AnthropometricRegularizationCost
        The cost this rep represents.
    mc: ModelCache
        The solver's `ModelCache`, whose registered body scale groups set this
        callback's input size.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback rather than using its
        analytic Jacobian. Default is ``False``.

    Raises
    ------
    ValueError
        If a measurement references a component that is not an `osim.Station`.
    """

    def __init__(self, cost: AnthropometricRegularizationCost, mc: ModelCache,
                 enable_fd: bool = False):
        self.cost = cost
        mc.model.realizePosition(mc.state)
        self.default_q = mc.state.getQ().to_numpy().copy()
        self.station_caches = []
        for measurement in cost.measurements:
            sc1 = StationCache.from_station(
                mc, mc.model.getComponent(measurement.station1_path))
            sc2 = StationCache.from_station(
                mc, mc.model.getComponent(measurement.station2_path))
            axis = measurement.axis.value if measurement.axis is not None else None
            self.station_caches.append((sc1, sc2, axis))
        Function.__init__(self, 'anthropometric_regularization_cost', mc,
                          enable_fd=enable_fd)

    def __call__(self, input: CostInput) -> ca.MX:
        return ca.Function.__call__(self, input.body_scales)

    def _get_num_inputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return 3 * len(self.mc.body_scale_groups)
        raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')

    def _apply_body_scales(self, body_scales: np.ndarray) -> None:
        self.mc.set_scaled_mobilizer_frame_positions(self.state, body_scales)
        self.state.setQ(osim.Vector.createFromMat(self.default_q))
        self.mc.model.realizePosition(self.state)

    def _eval(self, arg):
        body_scales = np.atleast_1d(np.squeeze(arg[0].full())).astype(float)
        self._apply_body_scales(body_scales)

        measurements = np.empty(len(self.station_caches))
        for i, (sc1, sc2, axis) in enumerate(self.station_caches):
            pos1 = sc1.calc_position(self.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(self.state, body_scales).to_numpy()
            displacement = pos2 - pos1
            measurements[i] = (np.linalg.norm(displacement) if axis is None
                               else abs(displacement[axis]))

        residual = measurements - self.cost.mean
        return [float(self.cost.weight * 0.5 * residual
                      @ self.cost.precision @ residual)]

    def _jac_eval(self, arg):
        body_scales = np.atleast_1d(np.squeeze(arg[0].full())).astype(float)
        self._apply_body_scales(body_scales)

        num_scales = 3 * len(self.mc.body_scale_groups)
        m = np.empty(len(self.station_caches))
        jacobian = np.zeros((len(self.station_caches), num_scales))
        for i, (sc1, sc2, axis) in enumerate(self.station_caches):
            pos1 = sc1.calc_position(self.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(self.state, body_scales).to_numpy()
            displacement = pos2 - pos1

            jac1 = sc1.calc_position_jacobian_wrt_body_scales(self.state)
            jac2 = sc2.calc_position_jacobian_wrt_body_scales(self.state)
            displacement_jacobian = jac2 - jac1

            if axis is None:
                norm = np.linalg.norm(displacement)
                m[i] = norm
                if norm > 0.0:
                    jacobian[i, :] = (displacement / norm) @ displacement_jacobian
            else:
                value = displacement[axis]
                m[i] = abs(value)
                jacobian[i, :] = np.sign(value) * displacement_jacobian[axis, :]
        residual = m - self.cost.mean
        gradient = self.cost.weight * (self.cost.precision @ residual) @ jacobian
        return [gradient.reshape(1, num_scales)]
