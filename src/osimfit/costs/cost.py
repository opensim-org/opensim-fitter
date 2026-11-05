import casadi as ca
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from ..data_sources import Trial
from ..model import ModelCache


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

    Parameters
    ----------
    position_weight: float, optional
        Overall weight on marker and frame-origin position errors. Default is 1.0.
    orientation_weight: float, optional
        Overall weight on frame orientation errors. Default is 1.0.
    marker_weights: dict[str, float], optional
        Per-marker factors on the position error, keyed by the marker label (i.e., the
        osim.Model component path) model component path. Each multiplies with the
        overall `position_weight`. Absent markers use a weight of 1.0.
    frame_position_weights: dict[str, float], optional
        Per-frame factors on the frame position error, keyed by the frame label (i.e.,
        the osim.Model component path) model component path. Each multiplies with the
        overall `position_weight`. Absent frames use a weight of 1.0.
    frame_orientation_weights: dict[str, float], optional
        Per-frame factors on the frame orientation error, keyed by the frame label
        (i.e., the osim.Model component path) model component path. Each multiplies with
        the overall `orientation_weight`. Absent frames use a weight of 1.0.
    """

    def __init__(self, position_weight: float = 1.0,
                 orientation_weight: float = 1.0,
                 marker_weights: dict[str, float] = None,
                 frame_position_weights: dict[str, float] = None,
                 frame_orientation_weights: dict[str, float] = None):
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.marker_weights = dict(marker_weights or {})
        self.frame_position_weights = dict(frame_position_weights or {})
        self.frame_orientation_weights = dict(frame_orientation_weights or {})

    def get_marker_weight(self, marker_path: str) -> float:
        return self.position_weight * self.marker_weights.get(marker_path, 1.0)

    def get_frame_weights(self, frame_path: str) -> tuple[float, float]:
        return (self.position_weight
                * self.frame_position_weights.get(frame_path, 1.0),
                self.orientation_weight
                * self.frame_orientation_weights.get(frame_path, 1.0))

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
