import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ..data_sources import Trial
from ..model import ModelCache, StationCache


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


CostScope = Literal['sample', 'global']


@dataclass(frozen=True)
class TrialSample:
    """A reference-data sample, identified by trial identity and row index."""
    trial: Trial
    index: int


@dataclass(frozen=True)
class CostEvaluation:
    """Variables and optional reference sample for one scalar cost evaluation."""
    inputs: CostInput
    sample: TrialSample | None = None

    @property
    def scope(self) -> CostScope:
        return 'global' if self.sample is None else 'sample'


class CostBase(ABC):
    """
    A model-independent description of a cost term: the weights, targets, and reference
    data the user (or a solver) supplies, with no OpenSim state of its own. A cost is
    inert and reusable.

    Attributes
    ----------
    required_inputs: frozenset[str]
        The `CostInput` field names this cost reads and therefore requires the solver to
        provide (e.g., ``{'body_scales'}``). A solver validates that it provides every
        required input before accepting the cost; see `Solver.add_cost`.
    """
    required_inputs: frozenset[str] = frozenset()

    def input_names(self) -> tuple[str, ...]:
        """
        Return this cost's `required_inputs` in canonical `CostInput.INPUT_ORDER`.
        """
        return tuple(name for name in CostInput.INPUT_ORDER
                     if name in self.required_inputs)


class CostCache:
    """
    Shared evaluation data for costs in a solve.

    Costs are stateless descriptions, and this cache owns the model-bound data they need
    to evaluate. Cached information is memoized, so that it is resolved at most once per
    solve.

    Parameters
    ----------
    mc: ModelCache
        The solver's finalized `ModelCache`.

    Attributes
    ----------
    mc: ModelCache
        The solver's finalized `ModelCache`.
    state: osim.State
        The model's shared state.
    default_q: np.ndarray
        The model's default generalized coordinates.
    """

    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.state = mc.state
        mc.model.realizePosition(self.state)
        self.default_q = self.state.getQ().to_numpy().copy()
        self._station_caches: dict[str, StationCache] = {}
        self._frame_caches: dict[str, StationCache] = {}
        self._memoized: dict[tuple, tuple] = {}

    def station_cache(self, path: str) -> StationCache:
        """
        Return the memoized `StationCache` for the station at `path`.

        Parameters
        ----------
        path: str
            The model path to an `osim.Station` (e.g., an `osim.Marker`).

        Returns
        -------
        StationCache
            The cached station, resolved on first request.

        Raises
        ------
        ValueError
            If the model has no component at `path`, or it is not an `osim.Station`.
        """
        if path not in self._station_caches:
            self._assert_has_component(path)
            self.mc.model.realizePosition(self.state)
            self._station_caches[path] = StationCache.from_station(
                self.mc, self.mc.model.getComponent(path))
        return self._station_caches[path]

    def frame_cache(self, path: str) -> StationCache:
        """
        Return the memoized `StationCache` for the origin of the frame at `path`.

        Parameters
        ----------
        path: str
            The model path to an `osim.PhysicalFrame`.

        Returns
        -------
        StationCache
            The cached frame origin, resolved on first request.

        Raises
        ------
        ValueError
            If the model has no component at `path`, or it is not an
            `osim.PhysicalFrame`.
        """
        if path not in self._frame_caches:
            self._assert_has_component(path)
            self._frame_caches[path] = StationCache.from_frame(
                self.mc, self.mc.model.getComponent(path))
        return self._frame_caches[path]

    def _assert_has_component(self, path: str) -> None:
        if not self.mc.model.hasComponent(path):
            raise ValueError(f'Model does not have a component at path {path}.')

    def memoize(self, owner: CostBase, key: str, factory):
        """
        Return the value memoized for `owner` under `key`, building it via `factory`
        on first request.

        Parameters
        ----------
        owner: CostBase
            The cost the value belongs to.
        key: str
            Distinguishes multiple values memoized for the same owner.
        factory: Callable[[], Any]
            Builds the value.
        """
        memo_key = (id(owner), key)
        if memo_key not in self._memoized:
            self._memoized[memo_key] = (owner, factory())
        return self._memoized[memo_key][1]


class CostWorkspace:
    """
    The callbacks and sample-specific evaluation data for an optimization problem.

    Parameters
    ----------
    cache: CostCache
        Shared model-bound evaluation data.
    """

    def __init__(self, cache: CostCache):
        self.cache = cache
        self._callbacks: list[CostFunction] = []
        self._cost_callbacks: dict[tuple[int, TrialSample | None], CostFunction] = {}
        self._states: dict[int, osim.State] = {}

    def add_callback(self, name: str, cost: CostBase, context=None,
                     enable_fd: bool = False) -> 'CostFunction':
        """
        Create a `CostFunction` for `cost` and retain it for the lifetime of this
        workspace. Each cost is evaluated using its own copy of the model state so that
        state information (e.g., body scales) does not leak between costs.

        Parameters
        ----------
        name: str
            The name of the callback function.
        cost: CostBase
            The cost the callback evaluates.
        context: optional
            The cost's per-callback data, passed back to the cost's `calc_value` and
            `calc_jacobian`. Default is ``None``.
        enable_fd: bool, optional
            If ``True``, CasADi finite-differences the callback instead of using its
            analytic Jacobian. Default is ``False``.
        """
        if id(cost) not in self._states:
            self._states[id(cost)] = osim.State(self.cache.state)
        callback = CostFunction(name, cost, self.cache, self._states[id(cost)],
                                context, enable_fd=enable_fd)
        self._callbacks.append(callback)
        return callback

    def cost_callback(self, cost: 'CallbackCost',
                      sample: TrialSample | None = None) -> 'CostFunction':
        """
        Reuse a callback by cost, and by sample for costs requiring reference data.
        """
        if cost.requires_sample and sample is None:
            raise ValueError(f'{type(cost).__name__} requires a trial sample.')
        sample = sample if cost.requires_sample else None
        key = (id(cost), sample)
        if key not in self._cost_callbacks:
            if sample is None:
                context = self.cache.memoize(
                    cost, 'context', lambda: cost.create_context(self.cache))
            else:
                context = cost.create_context(self.cache, sample)
            self._cost_callbacks[key] = self.add_callback(
                f'{type(cost).__name__}_{len(self._callbacks)}', cost, context)
        return self._cost_callbacks[key]

    def contexts(self, cost: 'CallbackCost') -> tuple:
        """Prepared contexts for a cost, for post-assembly validation."""
        return tuple(callback.context for callback in self._callbacks
                     if callback.cost is cost)

    def validate(self, named_costs: list['CostTerm']) -> None:
        """Validate all costs after objective assembly and before optimization."""
        for term in named_costs:
            term.cost.validate(self)


class Cost(CostBase):
    """
    A cost evaluated from the model and the optimization variables that user can
    register on a solver via `Solver.add_cost`.
    """
    scope: ClassVar[CostScope] = 'sample'
    requires_sample: bool = False

    @abstractmethod
    def expression(self, workspace: CostWorkspace,
                   evaluation: CostEvaluation) -> ca.MX:
        """
        Return this cost's scalar contribution before solver reduction.

        Parameters
        ----------
        workspace: CostWorkspace
            Owns this problem's callbacks and supplies shared evaluation data via
            its `cache`.
        evaluation: CostEvaluation
            The optimization variables and optional reference sample.
        """

    def validate(self, workspace: CostWorkspace) -> None:
        """Check the fully assembled problem, raising ValueError if invalid."""
        pass


@dataclass(frozen=True)
class CostTerm:
    """A named cost whose objective placement is defined by its cost class."""
    label: str
    cost: Cost

    @property
    def scope(self) -> CostScope:
        return self.cost.scope

    def __post_init__(self):
        scope = self.cost.scope
        if scope not in ('sample', 'global'):
            raise ValueError(f'Unknown cost scope {scope!r}; expected sample or global.')
        if scope == 'global' and self.cost.requires_sample:
            raise ValueError(
                f'{type(self.cost).__name__} requires a trial sample and cannot '
                f'use global scope.')


class SymbolicCost(Cost):
    """
    A `Cost` that is a plain CasADi expression over the optimization variables,
    requiring neither OpenSim evaluation nor anything from the `CostCache`. It is
    differentiated symbolically by CasADi and incurs no callback overhead.
    """

    @abstractmethod
    def evaluate(self, input: CostInput) -> ca.MX:
        """
        Return this cost's CasADi expression for `input`.
        """

    def expression(self, workspace: CostWorkspace,
                   evaluation: CostEvaluation) -> ca.MX:
        return self.evaluate(evaluation.inputs)


class CallbackCost(Cost):
    """
    A `Cost` evaluated through OpenSim via a `CostFunction`. Subclasses implement
    `calc_value` and `calc_jacobian`, which the callback invokes, and optionally
    `create_context` to resolve model-bound data the evaluation needs.
    """

    def expression(self, workspace: CostWorkspace,
                   evaluation: CostEvaluation) -> ca.MX:
        return workspace.cost_callback(self, evaluation.sample)(evaluation.inputs)

    def create_context(self, cache: CostCache, sample: TrialSample | None = None):
        """
        Model-bound data (beyond the cache itself) needed to evaluate `calc_value` and
        `calc_jacobian` for a cost.

        Parameters
        ----------
        cache: CostCache
            The solver's cache.
        sample: TrialSample, optional
            Supplied only when `requires_sample` is True. Otherwise the context is
            built once per solve and reused across samples and workspaces.
        """
        return None

    @abstractmethod
    def calc_value(self, cache: CostCache, state: osim.State, context,
                   values: dict[str, np.ndarray]) -> float:
        """
        Return this cost's scalar value.

        Parameters
        ----------
        cache: CostCache
            The solver's cache.
        state: osim.State
            This cost's copy of the state.
        context
            Model-bound data, beyond the cache, needed to evaluate this cost.
        values: dict[str, np.ndarray]
            The cost's `input_names`, mapped to their numeric values.
        """

    @abstractmethod
    def calc_jacobian(self, cache: CostCache, state: osim.State, context,
                      values: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Return this cost's Jacobian as one ``(1, n)`` block per entry of
        `input_names`, in that order.

        Parameters
        ----------
        cache: CostCache
            The solver's cache.
        state: osim.State
            This cost's copy of the state.
        context
            Model-bound data, beyond the cache, needed to evaluate this cost.
        values: dict[str, np.ndarray]
            The cost's `input_names`, mapped to their numeric values.
        """


class CostFunction(ca.Callback):
    """
    The CasADi callback that evaluates any OpenSim-backed cost and its Jacobian. It
    declares one input per entry of the cost's `input_names`, sized from the solver's
    registered parameter groups, and delegates evaluation to the cost's `calc_value`
    and `calc_jacobian`, passing back the per-callback `context` the cost supplied.

    Parameters
    ----------
    name: str
        The name of the callback function.
    cost: CostBase
        The cost this callback evaluates.
    cache: CostCache
        The solver's cache, which supplies the model.
    state: osim.State
        The cost's copy of the state.
    context: optional
        Model-bound data, beyond the cache, needed to evaluate this cost. Default is
        ``None``.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its
        analytic Jacobian. Default is ``False``.
    """

    def __init__(self, name: str, cost: CostBase, cache: CostCache, state: osim.State,
                 context=None, enable_fd: bool = False):
        ca.Callback.__init__(self)
        self.cost = cost
        self.cache = cache
        self.mc = cache.mc
        self.state = state
        self.context = context
        self.enable_fd = enable_fd
        self.input_names = cost.input_names()
        self.construct(name, {'enable_fd': True} if enable_fd else {})

    def __call__(self, input: CostInput) -> ca.MX:
        return ca.Function.__call__(
            self, *(getattr(input, name) for name in self.input_names))

    def get_n_in(self): return len(self.input_names)
    def get_n_out(self): return 1

    def get_input_size(self, i):
        if not 0 <= i < len(self.input_names):
            raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')
        sizes = {
            'coordinates': len(self.mc.coordinate_q_indexes),
            'body_scales': 3 * len(self.mc.body_scale_groups),
            'marker_offsets': 3 * len(self.mc.marker_offset_groups),
            'frame_offsets': 3 * len(self.mc.frame_offset_groups),
            'ellipsoid_radii_scales': 3 * len(self.mc.ellipsoid_radii_scale_groups),
            'beam_length_scales': len(self.mc.beam_length_scale_groups),
        }
        return sizes[self.input_names[i]]

    def get_output_size(self, i):
        if i == 0:
            return 1
        raise IndexError(f'Invalid output index {i} for {type(self).__name__}.')

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.get_input_size(i), 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.get_output_size(i), 1)

    @staticmethod
    def _unpack_cost_inputs(input_names, arg) -> dict[str, np.ndarray]:
        """Unpack numerical CasADi arguments in the cost's input order."""
        return {name: np.atleast_1d(np.squeeze(arg[i].full())).astype(float)
                for i, name in enumerate(input_names)}

    def eval(self, arg):
        values = CostFunction._unpack_cost_inputs(self.input_names, arg)
        return [self.cost.calc_value(self.cache, self.state, self.context, values)]

    def has_jacobian(self): return not self.enable_fd

    def get_jacobian(self, name, inames, onames, opts):
        class JacobianFunction(ca.Callback):
            def __init__(self, name, cost, cache, state, context, input_names,
                         input_sizes):
                ca.Callback.__init__(self)
                self.cost = cost
                self.cache = cache
                self.state = state
                self.context = context
                self.input_names = input_names
                self.input_sizes = input_sizes
                self.construct(name, {})

            def get_n_in(self):
                # CasADi supplies the original inputs followed by the scalar cost value.
                return len(self.input_names) + 1

            def get_n_out(self):
                return len(self.input_names)

            def get_sparsity_in(self, i):
                if 0 <= i < len(self.input_sizes):
                    return ca.Sparsity.dense(self.input_sizes[i], 1)
                if i == len(self.input_sizes):
                    return ca.Sparsity.dense(1, 1)
                return ca.Sparsity.dense(0, 0)

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(1, self.input_sizes[i])

            def eval(self, arg):
                values = CostFunction._unpack_cost_inputs(self.input_names, arg)
                blocks = self.cost.calc_jacobian(self.cache, self.state, self.context,
                                                 values)
                if len(blocks) != len(self.input_names):
                    raise ValueError(
                        f'{type(self.cost).__name__}.calc_jacobian returned {len(blocks)} '
                        f'blocks, but expected {len(self.input_names)} for cost inputs '
                        f'{self.input_names}.')
                return blocks

        input_sizes = tuple(self.get_input_size(i) for i in range(self.get_n_in()))
        self.jacobian_callback = JacobianFunction(
            name, self.cost, self.cache, self.state, self.context, self.input_names,
            input_sizes)
        return self.jacobian_callback
