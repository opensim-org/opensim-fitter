import numpy as np
import opensim as osim
from abc import ABC, abstractmethod

from ..data_sources import Trial
from ..costs import Cost, CostScope, CostTerm
from ..model import ModelCache
from .solution import Solution


class Solver(ABC):
    """
    An abstract base class for CasADi-based solvers that leverage computations from
    OpenSim models. Subclasses must implement the solve() method, which should return
    a Solution object containing the solution trajectory. This base class also
    provides common functionality for building IPOPT options and managing the OpenSim
    model and state.

    Reference data is organized into `Trial` objects, one per motion, each bundling the
    data sources collected together for that motion. A trial can be registered with
    `add_trial`. Each concrete solver determines how to optimize over the registered
    trials (e.g., sequentially, simultaneously, etc.).

    Users can register additional `Cost` objects to include in the solver's objective
    via `add_cost`. Solvers may only support a subset of available costs, and will raise
    a ValueError if a cost is registered that depends on unsupported inputs (e.g., a
    time-stepping inverse kinematics solver does not support optimizing body scales).

    Parameters
    ----------
    model: str or osim.Model
        The OpenSim model to use for the optimization problem. Can be provided as a file
        path or as an already-loaded osim.Model object.
    convergence_tolerance: float, optional
        The convergence tolerance to use for the IPOPT solver. Default is 1e-4.
    """
    # Concrete subclasses override to define the set of CostInput fields supported by
    # the solver.
    SUPPORTED_INPUTS: frozenset[str] = frozenset()
    SUPPORTED_SCOPES: frozenset[CostScope] = frozenset({'sample'})
    GLOBAL_INPUTS: frozenset[str] = frozenset()

    def __init__(self, model: str | osim.Model, convergence_tolerance: float=1e-4):
        super().__init__()

        # Remove muscles and create the ModelCache.
        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpRemoveMuscles())
        self.mc = ModelCache(modelProcessor.process())
        self.state = self.mc.state

        # Convenience aliases for the cached coordinate maps.
        self.coordinate_q_map = self.mc.coordinate_q_map
        self.coordinate_q_indexes = self.mc.coordinate_q_indexes

        # Optimization settings.
        self.convergence_tolerance = convergence_tolerance

        # Additional user-registered costs with their labels and evaluation scopes.
        self.costs: list[CostTerm] = []

        # Reference data, one Trial per motion.
        self.trials: list[Trial] = []

    def add_trial(self, trial: Trial):
        """
        Register a `Trial` containing the reference data for a single motion trial.

        Parameters
        ----------
        trial: Trial
            The trial object. Must carry at least one data source, and its name must be
            unique among the trials already registered.

        Raises
        ------
        ValueError
            If `trial` is not a `Trial`, carries no reference data, or reuses the name
            of an already-registered trial.
        """
        if not isinstance(trial, Trial):
            raise ValueError(
                f'add_trial expected a Trial, but got {type(trial).__name__}.')
        if not trial.marker_data and not trial.frame_data:
            raise ValueError(
                f"Trial '{trial.name}' has no reference data; add a data source before "
                f'registering it with a solver.')
        if any(trial.name == registered.name for registered in self.trials):
            raise ValueError(
                f"A trial named '{trial.name}' is already registered with this solver; "
                f'trial names must be unique.')

        self.trials.append(trial)

    def add_cost(self, cost: Cost):
        """
        Register an additional cost, named by its type in the objective breakdown.

        Each cost class defines its scope: tracking and coordinate costs are
        evaluated at each sample, while parameter regularizers are evaluated once
        using shared inputs. This placement is independent of `required_inputs`.

        Raises
        ------
        TypeError
            If `cost` is not a `Cost`.
        ValueError
            If the scope or inputs are unsupported, the cost needs reference data in
            global scope, or a cost of the same type is already registered.
        """
        if not isinstance(cost, Cost):
            raise TypeError(
                f'{type(self).__name__}.add_cost expects a Cost, but got '
                f'{type(cost).__name__}.')
        unsupported = cost.required_inputs - self.SUPPORTED_INPUTS
        if unsupported:
            raise ValueError(
                f'{type(self).__name__} does not support {type(cost).__name__}: it '
                f'requires cost input(s) {sorted(unsupported)} that this solver does '
                f'not provide (supported: {sorted(self.SUPPORTED_INPUTS)}).')
        term = CostTerm(type(cost).__name__, cost)
        if term.scope not in self.SUPPORTED_SCOPES:
            raise ValueError(
                f'{type(self).__name__} does not support {term.scope} cost scope.')
        if term.scope == 'global':
            unsupported = cost.required_inputs - self.GLOBAL_INPUTS
            if unsupported:
                raise ValueError(
                    f'{type(cost).__name__} requires input(s) {sorted(unsupported)} '
                    f'that {type(self).__name__} does not provide in global scope.')
        if any(type(cost) is type(registered.cost) for registered in self.costs):
            raise ValueError(
                f'A {type(cost).__name__} is already registered with this solver; a '
                f'cost type may be added only once.')

        self.costs.append(term)

    def _assert_has_trials(self):
        """
        Verify that at least one `Trial` has been registered. Called at the start of
        solve(), since there is nothing to fit without reference data.
        """
        if not self.trials:
            raise ValueError(
                f'{type(self).__name__} has no reference data; register at least one '
                f'Trial via add_trial() before calling solve().')

    def _get_ipopt_options(self, print_level=0):
        """
        Get a dictionary of common IPOPT options for use with CasADi's nlpsolver.
        """
        ipopt_options = {}
        # We only support callback functions with first-order derivatives.
        ipopt_options['hessian_approximation'] = 'limited-memory'
        # Based on Simbody's settings.
        ipopt_options['tol'] = self.convergence_tolerance
        ipopt_options['dual_inf_tol'] = self.convergence_tolerance
        ipopt_options['compl_inf_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_dual_inf_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_compl_inf_tol'] = self.convergence_tolerance
        ipopt_options['print_level'] = print_level
        # Avoids crashes in CasADi for larger problems.
        ipopt_options['mumps_pivot_order'] = 0

        return ipopt_options

    def _validate_guess(self, guess: Solution):
        """
        Validate that `guess` is a `Solution` providing a usable states table for each
        registered trial. Subclasses may override to add solver-specific checks.
        """
        if not isinstance(guess, Solution):
            raise TypeError(
                f'{type(self).__name__} expected an initial guess of type Solution, '
                f'but got {type(guess).__name__}.')

        # The guess must cover exactly the registered trials. Each trial's guess is
        # looked up by name, so the order of the dict is irrelevant.
        expected = {trial.name for trial in self.trials}
        got = set(guess.states_tables)
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing or extra:
            raise ValueError(
                f'Initial guess trials do not match the solver configuration. Missing '
                f'guesses for {missing}; unexpected guesses for {extra}.')

        for name, table in guess.states_tables.items():
            self._validate_guess_states_table(name, table)

    def _validate_guess_states_table(self, name: str,
                                     table: osim.TimeSeriesTable):
        """
        Validate that a trial's guess states table is non-empty and provides a column
        for every independent coordinate.
        """
        if table is None or table.getNumRows() == 0 or table.getNumColumns() == 0:
            shape = ('None' if table is None else
                     f'{table.getNumRows()} rows, {table.getNumColumns()} columns')
            raise ValueError(
                f"Initial guess states table for trial '{name}' is empty ({shape}).")

        labels = set(table.getColumnLabels())
        missing = [coord_path + '/value' for coord_path in self.coordinate_q_map
                   if coord_path + '/value' not in labels]
        if missing:
            raise ValueError(
                f"Initial guess states table for trial '{name}' is missing required "
                f'coordinate columns: {missing}.')

    def _extract_guess_coordinates(self, table: osim.TimeSeriesTable) -> np.ndarray:
        """
        Read a guess states table into a ``(num_rows, num_coords)`` array of coordinate
        values, with columns ordered to match this solver's `coordinate_q_map`.
        """
        return np.column_stack([
            table.getDependentColumn(coord_path + '/value').to_numpy()
            for coord_path in self.coordinate_q_map])

    @staticmethod
    def _compute_average_trapezoidal_weights(times) -> np.ndarray:
        """Normalized trapezoidal weights for a trial's strictly increasing times.

        The weights sum to one, preserving cost units and the relative strength of
        global regularizers independently of trial duration or sample spacing.
        A single sample (e.g., static data) has weight one.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError('Expected a non-empty vector of finite sample times.')
        if times.size == 1:
            return np.ones(1)
        dt = np.diff(times)
        if np.any(dt <= 0):
            raise ValueError('Sample times must be strictly increasing.')
        weights = np.zeros(len(times))
        weights[:-1] += 0.5 * dt
        weights[1:] += 0.5 * dt
        return weights / (times[-1] - times[0])

    @abstractmethod
    def solve(self, guess=None) -> Solution:
        pass


class TrackingSolver(Solver):
    """
    An abstract base class for solvers that track reference data. Data within a trial
    can be position-based (e.g., marker trajectories) or orientation-based (e.g., Theia
    frames). Concrete subclasses must implement the solve() method, which should return
    one states table per registered trial.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    convergence_tolerance: float, optional
        See `Solver`.
    position_weight: float, optional
        The weight to use for position-based tracking costs. Default is 1.0.
    orientation_weight: float, optional
        The weight to use for orientation-based tracking costs. Default is 1.0.
    """
    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0):
        super().__init__(model, convergence_tolerance)

        # Cost function weights.
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.marker_weights: dict[str, float] = {}
        self.frame_position_weights: dict[str, float] = {}
        self.frame_orientation_weights: dict[str, float] = {}

    def set_marker_weight(self, marker_path: str, marker_weight: float) -> None:
        """
        Set a marker's position error weight. This weight multiplies with the overall
        position weight used by the tracking cost so that the total weight applied to
        the marker is `position_weight * marker_weight`.

        Parameters
        ----------
        marker_path: str
            The osim.Model path to the marker to weight.
        marker_weight: float
            The marker weight. Must be non-negative.

        Raises
        ------
        ValueError
            If `weight` is negative, or the model has no component at `marker_path`.
        """
        if marker_weight < 0:
            raise ValueError(
                f'Expected the marker weight for {marker_path!r} to be non-negative, '
                f'but got {marker_weight}.')
        if not self.mc.model.hasComponent(marker_path):
            raise ValueError(
                f'Cannot weight marker {marker_path!r}: model '
                f"'{self.mc.model.getName()}' has no component at that path.")
        self.marker_weights[marker_path] = marker_weight

    def set_frame_weights(self, frame_path: str, frame_position_weight: float,
                         frame_orientation_weight: float) -> None:
        """
        Set a frame's position and orientation error weights. These weights multiply
        with the overall position and orientation weights used by the tracking cost so
        that the total weights applied to the frame are
        `position_weight * frame_position_weight` and
        `orientation_weight * frame_orientation_weight`.

        Parameters
        ----------
        frame_path: str
            The osim.Model path to the frame to weight.
        frame_position_weight: float
            The frame position weight. Must be non-negative.
        frame_orientation_weight: float
            The frame orientation weight. Must be non-negative.

        Raises
        ------
        ValueError
            If either weight is negative, or the model has no component at `frame_path`.
        """
        if frame_position_weight < 0:
            raise ValueError(
                f'Expected the position weight for {frame_path!r} to be non-negative, '
                f'but got {frame_position_weight}.')
        if frame_orientation_weight < 0:
            raise ValueError(
                f'Expected the orientation weight for {frame_path!r} to be '
                f'non-negative, but got {frame_orientation_weight}.')
        if not self.mc.model.hasComponent(frame_path):
            raise ValueError(
                f'Cannot weight frame {frame_path!r}: model '
                f"'{self.mc.model.getName()}' has no component at that path.")
        self.frame_position_weights[frame_path] = frame_position_weight
        self.frame_orientation_weights[frame_path] = frame_orientation_weight

    def _validate_guess(self, guess: Solution):
        super()._validate_guess(guess)
        for trial in self.trials:
            num_rows = guess.states_tables[trial.name].getNumRows()
            if num_rows != trial.num_times:
                raise ValueError(
                    f"Initial guess states table for trial '{trial.name}' has "
                    f"{num_rows} rows but that trial's reference data has "
                    f'{trial.num_times} time samples.')
