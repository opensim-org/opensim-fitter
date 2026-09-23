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
    """
    INPUT_ORDER: ClassVar[tuple[str, ...]] = (
        'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets')
    TRIPLET_INPUTS: ClassVar[tuple[str, ...]] = (
        'body_scales', 'marker_offsets', 'frame_offsets')

    coordinates: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    body_scales: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    marker_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    frame_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))

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
            `marker_offsets`, or `frame_offsets`).

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


class CostBase(ABC):
    """
    A model-independent description of a cost term: the weights, targets, and reference
    data the user (or a solver) supplies, with no OpenSim state of its own. A cost is
    inert, stateless, and reusable across solves. All model-bound work is resolved by
    the `SolveCache` a solver passes to the cost when it is evaluated.

    Attributes
    ----------
    required_inputs: frozenset[str]
        The `CostInput` field names this cost reads and therefore requires the solver to
        provide (e.g., ``{'body_scales'}``). A solver validates that it provides every
        required input before accepting the cost; see `Solver.add_cost`. The set also
        determines the inputs a `CostCallback` declares to CasADi, ordered by
        `CostInput.INPUT_ORDER`.
    """
    required_inputs: frozenset[str] = frozenset()

    def input_names(self) -> tuple[str, ...]:
        """
        Return this cost's `required_inputs` in canonical `CostInput.INPUT_ORDER`.
        """
        return tuple(name for name in CostInput.INPUT_ORDER
                     if name in self.required_inputs)


class Cost(CostBase):
    """
    A cost evaluated from the model and the optimization variables alone, and so the
    kind of cost a user can register on a solver via `Solver.add_cost`.

    A cost is called with the solver's `SolveCache` and a `CostInput`, and returns the
    CasADi expression contributing to the objective.
    """

    @abstractmethod
    def __call__(self, cache: 'SolveCache', input: CostInput) -> ca.MX:
        """
        Return this cost's contribution to the objective.

        Parameters
        ----------
        cache: SolveCache
            The solver's cache, from which the cost resolves any model-bound
            quantities it needs.
        input: CostInput
            The optimization variables.
        """


class TrackingCostBase(CostBase):
    """
    A cost evaluated at a single time sample of a single trial. It is called with the
    trial and sample index in addition to the cache and inputs; the sample index is
    passed through to the callback as a constant, so one callback serves every sample
    of a trial.
    """

    @abstractmethod
    def __call__(self, cache: 'SolveCache', trial: Trial, itime: int,
                 input: CostInput, num_times: int = None) -> ca.MX:
        """
        Return this cost's contribution to the objective at one sample of one trial.

        Parameters
        ----------
        cache: SolveCache
            The solver's cache, which supplies the trial's memoized `TaskSet`.
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` to evaluate.
        input: CostInput
            The optimization variables.
        num_times: int, optional
            Passed through to `SolveCache.task_set` to bound how many samples of the
            trial's reference data are loaded. Default is ``None``, meaning all of
            them.
        """

    @property
    def is_bilevel(self) -> bool:
        """
        Whether this cost evaluates against body scales and placement offsets, and so
        requires the bilevel flavor of a trial's `TaskSet`.
        """
        return 'body_scales' in self.required_inputs


class SymbolicCost(Cost):
    """
    A `Cost` that is a plain CasADi expression, requiring no OpenSim evaluation. It is
    differentiated symbolically by CasADi and incurs no callback overhead, so it needs
    nothing from the cache.
    """

    @abstractmethod
    def evaluate(self, input: CostInput) -> ca.MX:
        """
        Return this cost's CasADi expression for `input`.
        """

    def __call__(self, cache: 'SolveCache', input: CostInput) -> ca.MX:
        return self.evaluate(input)


class CallbackCost(Cost):
    """
    A `Cost` evaluated through OpenSim via a CasADi callback. Subclasses implement
    `evaluate` and `jacobian`, which the cost's `CostCallback` invokes; the callback
    itself is memoized on the cache, so repeated calls within one solve reuse it.
    """

    def __call__(self, cache: 'SolveCache', input: CostInput) -> ca.MX:
        callback = cache.callback(
            (type(self).__name__, id(self)),
            lambda name: CostCallback(name, self, cache))
        return callback(input)

    @abstractmethod
    def evaluate(self, cache: 'SolveCache', values: dict[str, np.ndarray]) -> float:
        """
        Return this cost's scalar value.

        Parameters
        ----------
        cache: SolveCache
            The solver's cache.
        values: dict[str, np.ndarray]
            The cost's `input_names`, mapped to their numeric values.
        """

    @abstractmethod
    def jacobian(self, cache: 'SolveCache',
                 values: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Return this cost's Jacobian as one ``(1, n)`` block per entry of
        `input_names`, in that order.
        """


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
    cache: SolveCache
        The solver's cache, supplying the OpenSim model and state used for evaluating
        the function and its Jacobian.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian (`get_jacobian`). Default is ``False``.
    """
    def __init__(self, name: str, cache: 'SolveCache', enable_fd: bool = False):
        ca.Callback.__init__(self)
        self.cache = cache
        self.mc = cache.mc
        self.state = cache.state
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

    def get_jacobian_sparsity(self, iout: int, iin: int) -> ca.Sparsity:
        """
        Return the sparsity of the Jacobian block of output `iout` with respect to
        input `iin`. Defaults to dense; override to declare a block structurally zero
        so CasADi omits it from the sparsity pattern.
        """
        return ca.Sparsity.dense(self.get_output_size(iout), self.get_input_size(iin))

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
                return self.callback.get_jacobian_sparsity(iout, iin)

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


class CostCallback(Function):
    """
    The single CasADi callback that evaluates any OpenSim-backed cost. It declares one
    input per entry of the cost's `input_names`, sized from the solver's registered
    parameter groups, and delegates evaluation to the cost itself.

    For a `TrackingCostBase` it additionally declares a trailing scalar sample-index
    input. Reference data never crosses the CasADi boundary: it lives as dense arrays
    in the trial's `TaskSet` and is addressed by that index inside `eval`. The Jacobian
    block for the index is therefore structurally zero, and one callback serves every
    sample of a trial.

    Parameters
    ----------
    name: str
        The name of the callback function.
    cache: SolveCache
        The solver's cache, which supplies the model, state, and task sets.
    cost: CostBase
        The cost this callback evaluates.
    task_set: TaskSet, optional
        The trial's resolved geometry and reference data, for a `TrackingCostBase`.
        Default is ``None``, for a cost that tracks no reference data and therefore
        declares no sample-index input.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its
        analytic Jacobian. Default is ``False``.
    """

    def __init__(self, name: str, cost: CostBase, cache: 'SolveCache',
                 task_set: 'TaskSet' = None, enable_fd: bool = False):
        self.cost = cost
        self.task_set = task_set
        self.names = cost.input_names()
        Function.__init__(self, name, cache, enable_fd=enable_fd)

    @property
    def tracks_samples(self) -> bool:
        """
        Whether this callback declares a trailing sample-index input.
        """
        return self.task_set is not None

    def __call__(self, input: CostInput, itime: int = None) -> ca.MX:
        args = [getattr(input, name) for name in self.names]
        if self.tracks_samples:
            args.append(itime)
        return ca.Function.__call__(self, *args)

    def _input_sizes(self) -> dict[str, int]:
        return {
            'coordinates': len(self.mc.coordinate_indexes),
            'body_scales': 3 * len(self.mc.body_scale_groups),
            'marker_offsets': 3 * len(self.mc.marker_offset_groups),
            'frame_offsets': 3 * len(self.mc.frame_offset_groups),
        }

    def _get_num_inputs(self):
        return len(self.names) + (1 if self.tracks_samples else 0)

    def _get_num_outputs(self):
        return 1

    def _get_input_size(self, i):
        if i == len(self.names) and self.tracks_samples:
            return 1
        if not 0 <= i < len(self.names):
            raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')
        return self._input_sizes()[self.names[i]]

    def _get_output_size(self, i):
        if i == 0:
            return 1
        raise IndexError(f'Invalid output index {i} for {type(self).__name__}.')

    def get_jacobian_sparsity(self, iout, iin):
        # The cost value does not vary smoothly with the sample index; it selects which
        # reference sample to read. Declaring the block structurally zero keeps the
        # index out of the NLP's sparsity pattern entirely.
        if iin == len(self.names) and self.tracks_samples:
            return ca.Sparsity(self.get_output_size(iout), 1)
        return super().get_jacobian_sparsity(iout, iin)

    def _unpack(self, arg) -> tuple[dict[str, np.ndarray], int]:
        """
        Convert the callback's positional arguments into a mapping from input name to
        a 1-D float array, plus the sample index (``None`` when not tracking samples).
        """
        values = {name: np.atleast_1d(np.squeeze(arg[i].full())).astype(float)
                  for i, name in enumerate(self.names)}
        itime = (int(round(float(arg[len(self.names)])))
                 if self.tracks_samples else None)
        return values, itime

    def _eval(self, arg):
        values, itime = self._unpack(arg)
        if self.tracks_samples:
            return [self.cost.evaluate(self.cache, self.task_set, itime, values)]
        return [self.cost.evaluate(self.cache, values)]

    def _jac_eval(self, arg):
        values, itime = self._unpack(arg)
        if self.tracks_samples:
            blocks = self.cost.jacobian(self.cache, self.task_set, itime, values)
        else:
            blocks = self.cost.jacobian(self.cache, values)
        if len(blocks) != len(self.names):
            raise ValueError(
                f'{type(self.cost).__name__}.jacobian returned {len(blocks)} blocks, '
                f'but {type(self).__name__} declares {len(self.names)} inputs '
                f'{self.names}.')
        # The trailing block is the structurally-empty sample-index derivative.
        return blocks + ([ca.DM(1, 1)] if self.tracks_samples else [])


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


###########
# HELPERS #
###########

def _calc_quaternion(state, frame):
    rotation = frame.getRotationInGround(state)
    quaternion = rotation.convertRotationToQuaternion()
    return np.array([quaternion.get(i) for i in range(4)])

def _calc_quaternion_jacobian(eps):
    # Simbody -> /SimTKcommon/Mechanics/include/SimTKcommon/internal/Rotation.h#L712
    e = 0.5 * eps
    return np.array([
        [-e[1], -e[2], -e[3]],
        [ e[0],  e[3], -e[2]],
        [-e[3],  e[0],  e[1]],
        [ e[2], -e[1],  e[0]],
    ])

#########
# TASKS #
#########

def _as_reference_array(value, width: int) -> np.ndarray:
    """
    Normalize a task's reference data to a dense ``(num_times, width)`` array.

    Accepts a single sample (an `osim.Vec3`, an `osim.Quaternion`, or a length-`width`
    sequence), which becomes a one-row array, or an already-stacked
    ``(num_times, width)`` array.

    Parameters
    ----------
    value: osim.Vec3 | osim.Quaternion | array-like
        The reference data for one task, for one or many time samples.
    width: int
        The expected number of components per sample: 3 for a position, 4 for an
        orientation quaternion.

    Returns
    -------
    np.ndarray
        The reference data, shape ``(num_times, width)``.

    Raises
    ------
    ValueError
        If `value` cannot be interpreted as one or more `width`-component samples.
    """
    if hasattr(value, 'to_numpy'):
        value = value.to_numpy()
    elif isinstance(value, osim.Quaternion):
        value = np.array([value.get(i) for i in range(4)])
    array = np.atleast_2d(np.asarray(value, dtype=float))
    if array.shape[-1] != width:
        raise ValueError(
            f'Expected reference data with {width} components per sample, but got '
            f'an array of shape {array.shape}.')
    return array


class Tasks(ABC):
    """
    A base class for task-specific storage and registration.

    Task storage is per-trial, not per-sample: geometry (station caches, mobilized-body
    indexes, base frames) is resolved once, and each task's reference data is stored as
    a dense ``(num_times, ...)`` array that cost evaluation indexes by sample.
    """
    @abstractmethod
    def initialize_tasks(self, state: osim.State, **kwargs) -> float:
        pass

    @property
    def num_times(self) -> int:
        """
        The number of reference samples held per task, or 0 if no tasks are registered.
        """
        return 0 if not self.positions else self.positions[0].shape[0]

    def assert_sample_in_range(self, itime: int) -> None:
        """
        Verify that `itime` addresses a sample this task set holds.

        Raises
        ------
        IndexError
            If `itime` is out of range for the stored reference data.
        """
        if not 0 <= itime < self.num_times:
            raise IndexError(
                f'Sample index {itime} is out of range for {type(self).__name__} '
                f'holding {self.num_times} sample(s).')


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

    def add_marker(self, marker_path: str, positions, weight: float = 1.0,
                   offset_group_index: int | None = None):
        """
        Register a marker to track.

        Parameters
        ----------
        marker_path: str
            The OpenSim Model path to the tracking marker.
        positions: osim.Vec3 | array-like, shape (num_times, 3)
            The reference position data tracked by the model marker, either a single
            sample or one row per time sample. Every marker registered on the same task
            set must supply the same number of samples.
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

        positions = _as_reference_array(positions, 3)
        if self.num_tasks and positions.shape[0] != self.num_times:
            raise ValueError(
                f"Marker '{marker_path}' supplies {positions.shape[0]} reference "
                f'sample(s), but this task set already holds {self.num_times}.')

        self.mc.model.realizePosition(self.mc.state)
        marker = osim.Marker.safeDownCast(self.mc.model.getComponent(marker_path))
        cache = StationCache.from_station(self.mc, marker)
        self.station_caches.append(cache)
        self.markers.append(marker)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(positions)
        self.weights.append(weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)


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

    def add_frame(self, frame_path: str, positions,
                  orientations, position_weight: float = 1.0,
                  orientation_weight: float = 1.0,
                  offset_group_index: int | None = None):
        """
        Register a frame to track.

        Parameters
        ----------
        frame_path: str
            The OpenSim Model path to the tracking frame.
        positions: osim.Vec3 | array-like, shape (num_times, 3)
            The reference position data tracked by the model frame, either a single
            sample or one row per time sample.
        orientations: osim.Quaternion | array-like, shape (num_times, 4)
            The reference orientation data, expressed as quaternions, tracked by the
            model frame, either a single sample or one row per time sample.
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

        positions = _as_reference_array(positions, 3)
        orientations = _as_reference_array(orientations, 4)
        if positions.shape[0] != orientations.shape[0]:
            raise ValueError(
                f"Frame '{frame_path}' supplies {positions.shape[0]} position "
                f'sample(s) but {orientations.shape[0]} orientation sample(s).')
        if self.num_tasks and positions.shape[0] != self.num_times:
            raise ValueError(
                f"Frame '{frame_path}' supplies {positions.shape[0]} reference "
                f'sample(s), but this task set already holds {self.num_times}.')

        frame = osim.PhysicalFrame.safeDownCast(self.mc.model.getComponent(frame_path))
        cache = StationCache.from_frame(self.mc, frame)
        self.station_caches.append(cache)
        self.frames.append(frame)
        self.mobod_indexes.push_back(cache.base_frame.getMobilizedBodyIndex())
        self.stations.push_back(osim.Vec3(*[float(v) for v in cache.base_station]))
        self.num_tasks = self.mobod_indexes.size()
        self.positions.append(positions)
        self.orientations.append(orientations)
        self.position_weights.append(position_weight)
        self.orientation_weights.append(orientation_weight)
        self.base_frames.append(cache.base_frame)
        self.base_stations.append(cache.base_station)
        self.offset_group_indexes.append(offset_group_index)


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
    def calc_error(self, state: osim.State, itime: int = 0, **kwargs) -> float:
        pass

    @abstractmethod
    def calc_jacobian(self, state: osim.State, itime: int = 0,
                      **kwargs) -> list[np.ndarray]:
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

    def calc_error(self, state, itime=0, position_weight=1.0,
                   orientation_weight=1.0) -> float:
        if self.num_tasks:
            self.assert_sample_in_range(itime)
        error = 0.0
        for i, frame in enumerate(self.frames):
            reference = self.positions[i][itime]
            orientation = self.orientations[i][itime]
            p_model = frame.getPositionInGround(state).to_numpy()
            position_error = self.position_weights[i] * position_weight * np.square(
                np.linalg.norm(p_model - reference))

            eps = _calc_quaternion(state, frame)
            orientation_error = (
                self.orientation_weights[i] * orientation_weight
                * (1.0 - np.square(np.dot(eps, orientation))))

            error += position_error + orientation_error
        return error

    def calc_jacobian(self, state, itime=0, position_weight=1.0,
                      orientation_weight=1.0) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_indexes)))]
        self.assert_sample_in_range(itime)

        # Loop over all frames and compute the "spatial error" (i.e., the combined
        # position and orientation error) for each.
        spatialError = osim.VectorOfSpatialVec(self.num_tasks, osim.SpatialVec(0))
        for i, frame in enumerate(self.frames):
            wp = self.position_weights[i] * position_weight
            wo = self.orientation_weights[i] * orientation_weight
            reference = self.positions[i][itime]
            orientation = self.orientations[i][itime]

            # Position error.
            p_model = frame.getPositionInGround(state)
            p_error = osim.Vec3(
                2.0 * wp * (p_model[0] - reference[0]),
                2.0 * wp * (p_model[1] - reference[1]),
                2.0 * wp * (p_model[2] - reference[2]))

            # Orientation error.
            eps = _calc_quaternion(state, frame)
            jac_eps = _calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ orientation
            scale = wo * -2.0 * np.dot(eps, orientation)
            w_error = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(w_error, p_error))

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        J = vec.to_numpy()

        return [np.expand_dims(J[self.mc.coordinate_indexes], axis=0)]


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

    def calc_error(self, state, itime=0, weight=1.0) -> float:
        if self.num_tasks:
            self.assert_sample_in_range(itime)
        error = 0.0
        for marker, positions, task_weight in zip(
                self.markers, self.positions, self.weights):
            p_model = marker.getLocationInGround(state).to_numpy()
            error += task_weight * weight * np.square(
                np.linalg.norm(p_model - positions[itime]))
        return error

    def calc_jacobian(self, state, itime=0, weight=1.0) -> list[np.ndarray]:
        if self.num_tasks == 0:
            return [np.zeros((1, len(self.mc.coordinate_indexes)))]
        self.assert_sample_in_range(itime)

        # Inialize the array used to calculate the position error Jacobian via the
        # grouped Simbody operator.
        f_GP = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        for i, (marker, positions, task_weight) in enumerate(
                zip(self.markers, self.positions, self.weights)):
            position = positions[itime]
            w = task_weight * weight
            p_model = marker.getLocationInGround(state)
            f_GP.set(i, osim.Vec3(
                2.0 * w * (p_model[0] - position[0]),
                2.0 * w * (p_model[1] - position[1]),
                2.0 * w * (p_model[2] - position[2])))

        # Calculate the position error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, f_GP, vec)

        return [np.expand_dims(vec.to_numpy()[self.mc.coordinate_indexes], axis=0)]


class BilevelTerm(TrackingTerm):
    """
    An intermediate base class that provides functionality common to bilevel cost terms.

    Applying body scales and placement offsets to the cached station locations is shared
    across marker and frame terms; subclasses (via `MarkerTasks`/`FrameTasks`) supply the
    per-task `station_caches`, `stations`, and `offset_group_indexes`.
    """
    def __init__(self):
        super().__init__()

    def apply_scales(self, body_scales: np.ndarray) -> None:
        for itask, cache in enumerate(self.station_caches):
            s = cache.calc_scaled_base_station(body_scales)
            self.stations.updElt(itask).set(0, float(s[0]))
            self.stations.updElt(itask).set(1, float(s[1]))
            self.stations.updElt(itask).set(2, float(s[2]))

    def apply_offsets(self, offsets: np.ndarray) -> None:
        for i, g in enumerate(self.offset_group_indexes):
            if g is None:
                continue
            o = np.asarray(offsets[3*g : 3*g+3], dtype=float)
            s = self.stations.getElt(i).to_numpy() + o
            self.stations.updElt(i).set(0, float(s[0]))
            self.stations.updElt(i).set(1, float(s[1]))
            self.stations.updElt(i).set(2, float(s[2]))

    def apply_state(self, body_scales: np.ndarray, offsets: np.ndarray) -> None:
        self.apply_scales(body_scales)
        self.apply_offsets(offsets)


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

    def calc_error(self, state, itime=0, weight=1.0) -> float:
        if self.num_tasks:
            self.assert_sample_in_range(itime)
        error = 0.0
        for i, (frame, positions, task_weight) in enumerate(
                zip(self.base_frames, self.positions, self.weights)):
            p_model = frame.findStationLocationInGround(
                state, self.stations.getElt(i)).to_numpy()
            error += task_weight * weight * np.square(
                np.linalg.norm(p_model - positions[itime]))
        return error

    def calc_jacobian(self, state, itime=0, weight=1.0) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.marker_offset_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo]
        self.assert_sample_in_range(itime)

        # Calculate the per-marker error gradient in Ground. This is a force-like term
        # will be multiplied with (the transpose of) each position Jacobian below. Also,
        # precompute the sensitivity of each marker's ground position to a shift from
        # an offset variable.
        dp_GS = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        doffset = np.zeros((self.num_tasks, 3))
        for i, (frame, positions, task_weight) in enumerate(
                zip(self.base_frames, self.positions, self.weights)):
            position = positions[itime]
            w = task_weight * weight
            p_GS = frame.findStationLocationInGround(state, self.stations.getElt(i))
            dp_GS.set(i, osim.Vec3(2.0 * w * (p_GS[0] - position[0]),
                                   2.0 * w * (p_GS[1] - position[1]),
                                   2.0 * w * (p_GS[2] - position[2])))
            rotation = frame.getRotationInGround(state)
            R_GB = np.array([[rotation.get(r, c) for c in range(3)] for r in range(3)])
            doffset[i] = dp_GS.get(i).to_numpy() @ R_GB

        # Calculate the Jacobian of the position error with respect to the coordinates.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByStationJacobianTranspose(
            state, self.mobod_indexes, self.stations, dp_GS, vec)
        Jq[0, :] = vec.to_numpy()[self.mc.coordinate_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GS_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GS.get(i)   # for each marker i on body k
        #
        dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
        for i in range(self.num_tasks):
            k = int(self.mobod_indexes.getElt(i))
            cur = dp_GB.get(k).to_numpy() + dp_GS.get(i).to_numpy()
            dp_GB.set(k, osim.Vec3(float(cur[0]), float(cur[1]), float(cur[2])))

        # Calculate the position-error Jacobian with respect to body scales.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Assemble the marker offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the marker offsets to the Jacobian with respect
        # to body scales.
        for i in range(self.num_tasks):
            g_off = self.offset_group_indexes[i]
            g_scale = self.station_caches[i].body_scale_group_index
            if g_off is None and g_scale is None:
                continue
            if g_off is not None:
                Jo[0, 3*g_off:3*g_off+3] += doffset[i]
            if g_scale is not None:
                Js[0, 3*g_scale:3*g_scale+3] += self.base_stations[i] * doffset[i]

        return [Jq, Js, Jo]


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

    def calc_error(self, state, itime=0, position_weight=1.0,
                   orientation_weight=1.0) -> float:
        if self.num_tasks:
            self.assert_sample_in_range(itime)
        error = 0.0
        for i, (frame, base_frame) in enumerate(zip(self.frames, self.base_frames)):
            reference = self.positions[i][itime]
            orientation = self.orientations[i][itime]
            p_model = base_frame.findStationLocationInGround(
                state, self.stations.getElt(i)).to_numpy()
            position_error = self.position_weights[i] * position_weight * np.square(
                np.linalg.norm(p_model - reference))

            eps = _calc_quaternion(state, frame)
            orientation_error = (
                self.orientation_weights[i] * orientation_weight
                * (1.0 - np.square(np.dot(eps, orientation))))

            error += position_error + orientation_error
        return error

    def calc_jacobian(self, state, itime=0, position_weight=1.0,
                      orientation_weight=1.0) -> list[np.ndarray]:
        Jq = np.zeros((1, len(self.mc.coordinate_indexes)))
        Js = np.zeros((1, 3 * len(self.mc.body_scale_groups)))
        Jo = np.zeros((1, 3 * len(self.mc.frame_offset_groups)))
        if self.num_tasks == 0:
            return [Jq, Js, Jo]
        self.assert_sample_in_range(itime)

        # Loop over all frames and compute the "spatial error" (i.e., the combined
        # position and orientation error) for each.
        spatialError = osim.VectorOfSpatialVec(self.num_tasks, osim.SpatialVec(0))
        # Store the position-error gradient along the way. We need it for the body scale
        # and offset Jacobian calculations. Also, precompute the sensitivity of each
        # frame's ground position to a shift from an offset variable.
        dp_GF = osim.VectorVec3(self.num_tasks, osim.Vec3(0))
        doffset = np.zeros((self.num_tasks, 3))
        for i, (frame, base_frame) in enumerate(zip(self.frames, self.base_frames)):
            wp = self.position_weights[i] * position_weight
            wo = self.orientation_weights[i] * orientation_weight
            position = self.positions[i][itime]
            orientation = self.orientations[i][itime]

            # The frame's ground position is computed from its (possibly offset) cached
            # station so that the gradient is consistent with any applied offsets.
            p_GF = base_frame.findStationLocationInGround(
                state, self.stations.getElt(i))
            dp_GF.set(i, osim.Vec3(2.0 * wp * (p_GF[0] - position[0]),
                                   2.0 * wp * (p_GF[1] - position[1]),
                                   2.0 * wp * (p_GF[2] - position[2])))

            # Calculate the per-frame orientation error in Ground.
            eps = _calc_quaternion(state, frame)
            jac_eps = _calc_quaternion_jacobian(eps)
            omega = jac_eps.T @ orientation
            scale = wo * -2.0 * np.dot(eps, orientation)
            dw_GF = osim.Vec3(scale * omega[0], scale * omega[1], scale * omega[2])

            # Combine the position and orientation into a SpatialVec to pass to the
            # frame Jacobian operator below.
            spatialError.set(i, osim.SpatialVec(dw_GF, dp_GF.get(i)))

            # Precompute the position sensitivity to a base-frame station shift.
            rotation = base_frame.getRotationInGround(state)
            R_GB = np.array([[rotation.get(r, c) for c in range(3)] for r in range(3)])
            doffset[i] = dp_GF.get(i).to_numpy() @ R_GB

        # Calculate the frame (position and orientation) error Jacobian.
        vec = osim.Vector(state.getNQ(), 0.0)
        self.mc.model.multiplyByFrameJacobianTranspose(
            state, self.mobod_indexes, self.stations, spatialError, vec)
        Jq[0, :] = vec.to_numpy()[self.mc.coordinate_indexes]

        # Scatter per-station gradients for each task into a vector respresenting the
        # error gradient with respect to body origins, which we need for the Jacobian
        # operations below. Since the body scales only apply a translational shift and
        # no rotation, `dp_GF_i / dp_GB[k_i] = I`, and we can compute the vector via:
        #
        #     dp_GB.get(k) += dp_GF.get(i)   # for each frame i on body k
        #
        dp_GB = osim.VectorVec3(self.mc.num_mobod, osim.Vec3(0))
        for i in range(self.num_tasks):
            k = int(self.mobod_indexes.getElt(i))
            cur = dp_GB.get(k).to_numpy() + dp_GF.get(i).to_numpy()
            dp_GB.set(k, osim.Vec3(float(cur[0]), float(cur[1]), float(cur[2])))

        # Calculate the position-error Jacobian with respect to body scales. This does
        # not include the contributions from frame offsets, we will include that below.
        Js = self.mc.calc_position_jacobian_wrt_body_scales(state, dp_GB)

        # Assemble the frame offset Jacobian based on the offset sensitivities. Also,
        # include the contributions from the frame offsets to the Jacobian with respect
        # to body scales.
        for i in range(self.num_tasks):
            g_off = self.offset_group_indexes[i]
            g_scale = self.station_caches[i].body_scale_group_index
            if g_off is None and g_scale is None:
                continue
            if g_off is not None:
                Jo[0, 3*g_off:3*g_off+3] += doffset[i]
            if g_scale is not None:
                Js[0, 3*g_scale:3*g_scale+3] += self.base_stations[i] * doffset[i]

        return [Jq, Js, Jo]


##################
# COST FUNCTIONS #
##################

class TrackingCost(TrackingCostBase):
    """
    The weighted, squared error between the model's markers and frames and a trial's
    reference data, as a function of the model's generalized coordinates.

    The cost is stateless: the trial's geometry and reference data live in a `TaskSet`
    memoized on the solver's `SolveCache`, and the cost's weights are applied to that
    task set's per-task weights at evaluation time.

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

    def __call__(self, cache: 'SolveCache', trial: Trial, itime: int,
                 input: CostInput, num_times: int = None) -> ca.MX:
        task_set = cache.task_set(trial, self.is_bilevel, num_times)
        callback = cache.callback(
            (type(self).__name__, id(self), trial.name),
            lambda name: CostCallback(name, self, cache, task_set=task_set))
        return callback(input, itime)

    def apply_state(self, cache: 'SolveCache', values: dict[str, np.ndarray]) -> None:
        """
        Apply the input coordinates to the cache's state and realize the system to the
        position stage.
        """
        state = cache.state
        q = np.zeros(state.getNQ())
        q[cache.mc.coordinate_indexes] = values['coordinates']
        state.setQ(osim.Vector.createFromMat(q))
        cache.model.realizePosition(state)

    def evaluate(self, cache: 'SolveCache', task_set: 'TaskSet', itime: int,
                 values: dict[str, np.ndarray]) -> float:
        self.apply_state(cache, values)
        return (task_set.marker_term.calc_error(
                    cache.state, itime, weight=self.position_weight)
                + task_set.frame_term.calc_error(
                    cache.state, itime, position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight))

    def jacobian(self, cache: 'SolveCache', task_set: 'TaskSet', itime: int,
                 values: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.apply_state(cache, values)
        Jq = (task_set.marker_term.calc_jacobian(
                  cache.state, itime, weight=self.position_weight)[0]
              + task_set.frame_term.calc_jacobian(
                  cache.state, itime, position_weight=self.position_weight,
                  orientation_weight=self.orientation_weight)[0])
        return [Jq]


class BilevelCost(TrackingCostBase):
    """
    The tracking cost of `TrackingCost`, as a function of the model's generalized
    coordinates, its body scales, and its per-marker/frame XYZ placement offsets.

    Parameters
    ----------
    position_weight: float, optional
        Weight applied to marker and frame-origin position errors. Default is 1.0.
    orientation_weight: float, optional
        Weight applied to frame orientation errors. Default is 1.0.
    """
    required_inputs = frozenset(
        {'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets'})

    def __init__(self, position_weight: float = 1.0,
                 orientation_weight: float = 1.0):
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

    def __call__(self, cache: 'SolveCache', trial: Trial, itime: int,
                 input: CostInput, num_times: int = None) -> ca.MX:
        task_set = cache.task_set(trial, self.is_bilevel, num_times)
        callback = cache.callback(
            (type(self).__name__, id(self), trial.name),
            lambda name: CostCallback(name, self, cache, task_set=task_set))
        return callback(input, itime)

    def apply_state(self, cache: 'SolveCache', task_set: 'TaskSet',
                    values: dict[str, np.ndarray]) -> None:
        """
        Apply input coordinates, body-scale variables, and offset variables to the
        cache's state and the task set's cached stations, then realize to Position.
        """
        state = cache.state
        body_scales = values['body_scales']
        cache.mc.set_scaled_mobilizer_frame_positions(state, body_scales)
        task_set.marker_term.apply_state(body_scales, values['marker_offsets'])
        task_set.frame_term.apply_state(body_scales, values['frame_offsets'])

        q = np.zeros(state.getNQ())
        q[cache.mc.coordinate_indexes] = values['coordinates']
        state.setQ(osim.Vector.createFromMat(q))
        cache.model.realizePosition(state)

    def evaluate(self, cache: 'SolveCache', task_set: 'TaskSet', itime: int,
                 values: dict[str, np.ndarray]) -> float:
        self.apply_state(cache, task_set, values)
        return (task_set.marker_term.calc_error(
                    cache.state, itime, weight=self.position_weight)
                + task_set.frame_term.calc_error(
                    cache.state, itime, position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight))

    def jacobian(self, cache: 'SolveCache', task_set: 'TaskSet', itime: int,
                 values: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.apply_state(cache, task_set, values)
        Jq_m, Js_m, Jmo = task_set.marker_term.calc_jacobian(
            cache.state, itime, weight=self.position_weight)
        Jq_f, Js_f, Jfo = task_set.frame_term.calc_jacobian(
            cache.state, itime, position_weight=self.position_weight,
            orientation_weight=self.orientation_weight)
        return [Jq_m + Jq_f, Js_m + Js_f, Jmo, Jfo]


class AnthropometricRegularizationCost(CallbackCost):
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
        dataset. Evaluation additionally validates, via `SolveCache.station_cache`,
        that the referenced components are stations.
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

    def station_caches(self, cache: 'SolveCache') -> list[tuple]:
        """
        Return this cost's ``(station_cache, station_cache, axis)`` triplets, resolved
        through the solve cache so that a station referenced by several measurements
        (or by another cost) is resolved only once.
        """
        return [(cache.station_cache(m.station1_path),
                 cache.station_cache(m.station2_path),
                 m.axis.value if m.axis is not None else None)
                for m in self.measurements]

    def apply_body_scales(self, cache: 'SolveCache',
                          body_scales: np.ndarray) -> None:
        """
        Apply `body_scales` to the cache's state, restore the model's default pose, and
        realize to Position. The measurements are defined at the default pose, so the
        coordinates are reset rather than taken from the optimizer.
        """
        cache.mc.set_scaled_mobilizer_frame_positions(cache.state, body_scales)
        cache.state.setQ(osim.Vector.createFromMat(cache.default_q))
        cache.model.realizePosition(cache.state)

    def evaluate(self, cache: 'SolveCache',
                 values: dict[str, np.ndarray]) -> float:
        body_scales = values['body_scales']
        self.apply_body_scales(cache, body_scales)

        station_caches = self.station_caches(cache)
        measurements = np.empty(len(station_caches))
        for i, (sc1, sc2, axis) in enumerate(station_caches):
            pos1 = sc1.calc_position(cache.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(cache.state, body_scales).to_numpy()
            displacement = pos2 - pos1
            measurements[i] = (np.linalg.norm(displacement) if axis is None
                               else abs(displacement[axis]))

        residual = measurements - self.mean
        return float(self.weight * 0.5 * residual @ self.precision @ residual)

    def jacobian(self, cache: 'SolveCache',
                 values: dict[str, np.ndarray]) -> list[np.ndarray]:
        body_scales = values['body_scales']
        self.apply_body_scales(cache, body_scales)

        station_caches = self.station_caches(cache)
        num_scales = 3 * len(cache.mc.body_scale_groups)
        m = np.empty(len(station_caches))
        jacobian = np.zeros((len(station_caches), num_scales))
        for i, (sc1, sc2, axis) in enumerate(station_caches):
            pos1 = sc1.calc_position(cache.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(cache.state, body_scales).to_numpy()
            displacement = pos2 - pos1

            jac1 = sc1.calc_position_jacobian_wrt_body_scales(cache.state)
            jac2 = sc2.calc_position_jacobian_wrt_body_scales(cache.state)
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
        residual = m - self.mean
        gradient = self.weight * (self.precision @ residual) @ jacobian
        return [gradient.reshape(1, num_scales)]


##############
# SOLVE CACHE #
##############

@dataclass
class TaskSet:
    """
    One trial's tracking geometry and reference data, resolved once per solve.

    Both terms hold their tasks' model-bound handles (station caches, mobilized-body
    indexes, base frames) and their reference data as dense ``(num_times, ...)``
    arrays, so a cost evaluates any sample of the trial by index without rebuilding
    anything.

    Attributes
    ----------
    marker_term: MarkerTrackingTerm | MarkerBilevelTerm
        The trial's marker tasks and their evaluator.
    frame_term: FrameTrackingTerm | FrameBilevelTerm
        The trial's frame tasks and their evaluator.
    """
    marker_term: Tasks
    frame_term: Tasks

    @property
    def num_times(self) -> int:
        """
        The number of reference samples held, taken from whichever term has tasks.
        """
        return max(self.marker_term.num_times, self.frame_term.num_times)

    def offset_group_indexes(self) -> tuple[set[int], set[int]]:
        """
        Return the marker and frame offset group indexes tracked by this task set,
        excluding tasks that are not offset.
        """
        markers = {g for g in self.marker_term.offset_group_indexes if g is not None}
        frames = {g for g in self.frame_term.offset_group_indexes if g is not None}
        return markers, frames


class SolveCache:
    """
    Everything a solver resolves once per solve, memoized.

    A `SolveCache` replaces the per-cost, per-sample representation objects that costs
    previously built for themselves. Costs are stateless descriptions; every
    model-bound quantity they need is requested from the cache, keyed by the thing it
    actually varies over:

    - a station's `StationCache`, keyed by component path;
    - a trial's `TaskSet`, keyed by trial and whether the cost is bilevel;
    - a cost's `CostCallback`, keyed by the cost and, for a tracking cost, the trial.

    The cache also owns the callbacks for the lifetime of the solve, which is what
    keeps CasADi's references to them valid; solvers no longer hold them in ad-hoc
    lists.

    Parameters
    ----------
    mc: ModelCache
        The solver's `ModelCache`. All parameter groups must already be registered on
        it, since they set the input sizes the callbacks declare to CasADi.

    Attributes
    ----------
    mc: ModelCache
        The wrapped model cache.
    state: osim.State
        The state every cost evaluation mutates and realizes.
    default_q: np.ndarray
        The model's default pose, captured at construction, before any evaluation can
        perturb the state.
    """

    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.state = mc.state

        # Capture the default pose eagerly: costs that measure the model at its default
        # configuration must not observe a pose left behind by an earlier evaluation.
        mc.model.realizePosition(self.state)
        self.default_q = self.state.getQ().to_numpy().copy()

        # Body-scale group joints are needed by any cost that scales the model, and
        # caching them is idempotent.
        mc.cache_body_scale_group_joints()

        self._station_caches: dict[str, StationCache] = {}
        self._task_sets: dict[tuple, TaskSet] = {}
        self._callbacks: dict[tuple, 'CostCallback'] = {}

    @property
    def model(self) -> osim.Model:
        """
        The OpenSim model being evaluated.
        """
        return self.mc.model

    def station_cache(self, path: str) -> StationCache:
        """
        Return the memoized `StationCache` for the station at `path`.

        Parameters
        ----------
        path: str
            The model path to an `osim.Station`.

        Returns
        -------
        StationCache
            The cached station, resolved on first request.
        """
        if path not in self._station_caches:
            self._station_caches[path] = StationCache.from_station(
                self.mc, self.model.getComponent(path))
        return self._station_caches[path]

    def task_set(self, trial: Trial, bilevel: bool,
                 num_times: int = None) -> TaskSet:
        """
        Return the memoized `TaskSet` for `trial`.

        Parameters
        ----------
        trial: Trial
            The trial supplying the task paths and reference data.
        bilevel: bool
            Whether to build terms that evaluate against body scales and placement
            offsets (`MarkerBilevelTerm`/`FrameBilevelTerm`) rather than coordinates
            alone (`MarkerTrackingTerm`/`FrameTrackingTerm`).
        num_times: int, optional
            Load reference data for only the first `num_times` samples of the trial.
            Default is ``None``, meaning every sample. A solver that evaluates a single
            pose per trial passes 1 to avoid loading data it will not read.

        Returns
        -------
        TaskSet
            The trial's task set, built on first request.
        """
        key = (trial.name, bool(bilevel), num_times)
        if key not in self._task_sets:
            self._task_sets[key] = self._build_task_set(trial, bilevel, num_times)
        return self._task_sets[key]

    def task_sets(self) -> list[TaskSet]:
        """
        Return every task set built so far. Solvers use this to validate coverage
        across trials, e.g. that each registered offset group is tracked somewhere.
        """
        return list(self._task_sets.values())

    def callback(self, key: tuple, factory) -> 'CostCallback':
        """
        Return the memoized callback for `key`, building it via `factory` on first
        request and retaining it for the lifetime of this cache.

        Parameters
        ----------
        key: tuple
            The identity of the callback, e.g. ``(cost type, cost id, trial name)``.
        factory: Callable[[str], CostCallback]
            Builds the callback, given a generated CasADi function name.

        Returns
        -------
        CostCallback
            The cached callback.
        """
        if key not in self._callbacks:
            name = f'cost_callback_{len(self._callbacks)}'
            self._callbacks[key] = factory(name)
        return self._callbacks[key]

    def _build_task_set(self, trial: Trial, bilevel: bool,
                        num_times: int) -> TaskSet:
        """
        Resolve `trial`'s markers and frames against the model once, and stack their
        reference data into dense, time-indexed arrays.
        """
        marker_term = (MarkerBilevelTerm(self.mc) if bilevel
                       else MarkerTrackingTerm(self.mc))
        frame_term = (FrameBilevelTerm(self.mc) if bilevel
                      else FrameTrackingTerm(self.mc))

        # Map each offset target path to the index of the offset group that applies to
        # it; paths absent from a mapping are not offset. Only a bilevel task set reads
        # these, but they are cheap to resolve and harmless otherwise.
        marker_index_of = {path: i
                           for i, grp in enumerate(self.mc.marker_offset_groups)
                           for path in grp.component_paths}
        frame_index_of = {path: i
                          for i, grp in enumerate(self.mc.frame_offset_groups)
                          for path in grp.component_paths}

        for data in trial.frame_data:
            rows = _sample_count(data.positions, num_times)
            for iframe, frame_path in enumerate(data.labels):
                frame_term.add_frame(
                    frame_path,
                    _stack_vec3(data.positions, iframe, rows),
                    _stack_quaternion(data.orientations, iframe, rows),
                    offset_group_index=frame_index_of.get(frame_path))

        for data in trial.marker_data:
            rows = _sample_count(data.positions, num_times)
            for imarker, marker_path in enumerate(data.labels):
                marker_term.add_marker(
                    marker_path,
                    _stack_vec3(data.positions, imarker, rows),
                    offset_group_index=marker_index_of.get(marker_path))

        return TaskSet(marker_term=marker_term, frame_term=frame_term)


def _sample_count(table, num_times: int | None) -> int:
    """
    Return the number of rows of `table` to load, capped by `num_times` when given.
    """
    rows = table.getNumRows()
    return rows if num_times is None else min(rows, num_times)


def _stack_vec3(table, icolumn: int, rows: int) -> np.ndarray:
    """
    Stack the first `rows` samples of a Vec3 table column into a ``(rows, 3)`` array.
    """
    return np.array([table.getRowAtIndex(itime).getElt(0, icolumn).to_numpy()
                     for itime in range(rows)])


def _stack_quaternion(table, icolumn: int, rows: int) -> np.ndarray:
    """
    Stack the first `rows` samples of a Quaternion table column into a ``(rows, 4)``
    array.
    """
    return np.array(
        [[table.getRowAtIndex(itime).getElt(0, icolumn).get(i) for i in range(4)]
         for itime in range(rows)])
