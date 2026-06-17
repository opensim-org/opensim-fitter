import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .utilities import get_coordinate_indexes
from .data_sources import DataSource, MarkerSource, TheiaFrameSource
from .callbacks import TrackingCostFunction, BilevelCostFunction, ScaleGroup
from .scaling import Axis, Scaler, ManualScaleFactor


################
# DATA STRUCTS #
################

@dataclass
class TheiaFrameData:
    labels: list[str]
    positions: osim.TimeSeriesTableVec3
    orientations: osim.TimeSeriesTableQuaternion


@dataclass
class MarkerData:
    labels: list[str]
    positions: osim.TimeSeriesTableVec3


@dataclass
class Bounds:
    lower_bound: float
    upper_bound: float


@dataclass
class ScaleFactor:
    group: ScaleGroup
    bounds: Bounds


############
# SOLUTION #
############

@dataclass
class Solution:
    """
    Base class for solver solutions.
    """

@dataclass
class TrackingSolution(Solution):
    """
    Solution for tracking solvers. Contains the optimized model states as an OpenSim
    TimeSeriesTable and a static helper for constructing it from raw trajectory arrays.
    """
    states_table: osim.TimeSeriesTable

    @staticmethod
    def create_states_table(model, state, coordinate_indexes, times,
                            q_opt, qdot_opt=None) -> osim.TimeSeriesTable:
        """
        Build an OpenSim StatesTrajectory and export it to a TimeSeriesTable.

        Parameters
        ----------
        model: osim.Model
        state: osim.State
            An initialized state that will be mutated in place during construction.
        coordinate_indexes: list[int]
            Indexes of the independent coordinates in the full state vector.
        times: sequence of float
        q_opt: np.ndarray, shape (num_times, num_coords)
        qdot_opt: np.ndarray, shape (num_times, num_coords), optional
        """
        statesTraj = osim.StatesTrajectory()
        for i, time in enumerate(times):
            state.setTime(time)
            q = np.zeros(state.getNQ())
            q[coordinate_indexes] = q_opt[i, :]
            state.setQ(osim.Vector.createFromMat(q))
            if qdot_opt is not None:
                qdot = np.zeros(state.getNQ())
                qdot[coordinate_indexes] = qdot_opt[i, :]
                state.setU(osim.Vector.createFromMat(qdot))
            statesTraj.append(state)
        return statesTraj.exportToTable(model)


@dataclass
class SplineTrackingSolution(TrackingSolution):
    """
    TrackingSolution for spline-based solvers. Adds the optimal B-spline control
    points (nodes) for each coordinate.

    Attributes
    ----------
    spline_nodes: np.ndarray, shape (num_knots, num_coords)
    """
    spline_nodes: np.ndarray = None


@dataclass
class BilevelSolution(TrackingSolution):
    """
    Solution for bilevel solvers. Separates the optimized coordinate trajectories
    from the optimized body scale factors.

    Attributes
    ----------
    scale_factors: np.ndarray, shape (num_scale_factors, 3)
        Optimal [sx, sy, sz] scale factors, one row per scale factor group.
    scale_groups: list[ScaleGroup]
        ScaleGroup objects paired row-wise with scale_factors. Each entry
        names the bodies sharing that set of XYZ body scale factors. Single-body scale
        factors appear as a ScaleGroup with one body path and mobilized body index.
    """
    scale_factors: np.ndarray = None
    scale_groups: list[ScaleGroup] = None


@dataclass
class SplineBilevelSolution(BilevelSolution):
    """
    BilevelSolution for spline-based bilevel solvers. Adds the optimal B-spline
    control points (nodes) for each coordinate.

    Attributes
    ----------
    spline_nodes: np.ndarray, shape (num_knots, num_coords)
    """
    spline_nodes: np.ndarray = None


###########
# SOLVERS #
###########

class Solver(ABC):
    """
    An abstract base class for CasADi-based solvers that leverage computations from
    OpenSim models. Subclasses must implement the solve() method, which should return
    a Solution object containing the solution trajectory. This base class also
    provides common functionality for building IPOPT options and managing the OpenSim
    model and state.

    Parameters
    ----------
    model: str or osim.Model
        The OpenSim model to use for the optimization problem. Can be provided as a file
        path or as an already-loaded osim.Model object.
    convergence_tolerance: float, optional
        The convergence tolerance to use for the IPOPT solver. Default is 1e-4.
    """
    # Concrete subclasses set this to the exact Solution subclass they accept as
    # an initial guess (and return from solve()).
    _guess_type: type = Solution

    def __init__(self, model, convergence_tolerance=1e-4):
        super().__init__()

        # Load the model.
        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpRemoveMuscles())
        self.model =  modelProcessor.process()
        self.state = self.model.initSystem()
        # For now, disallow models with joints where qdot != u.
        assert(self.state.getNQ() == self.state.getNU())

        # Create a mapping between coordinate paths and their indexes in the state
        # vector.
        self.coordinates_map = get_coordinate_indexes(self.model,
                                                      skip_dependent_coordinates=True)
        self.coordinate_indexes = list(self.coordinates_map.values())

        # Optimization settings.
        self.convergence_tolerance = convergence_tolerance

    def get_ipopt_options(self, print_level=0):
        """
        Get a dictionary of common IPOPT options for use with CasADi's nlpsolver.
        """
        ipopt_options = {}
        ipopt_options['hessian_approximation'] = 'limited-memory'
        ipopt_options['tol'] = self.convergence_tolerance
        ipopt_options['dual_inf_tol'] = self.convergence_tolerance
        ipopt_options['compl_inf_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_dual_inf_tol'] = self.convergence_tolerance
        ipopt_options['acceptable_compl_inf_tol'] = self.convergence_tolerance
        ipopt_options['print_level'] = print_level

        return ipopt_options

    def _validate_guess(self, guess: Solution):
        """
        Validate that `guess` matches the solver's expected guess type and contains
        usable data. Subclasses may override to add solver-specific checks; in that
        case they should call `super()._validate_guess(guess)` first.
        """
        if type(guess) is not self._guess_type:
            raise TypeError(
                f'{type(self).__name__} expected an initial guess of type '
                f'{self._guess_type.__name__}, but got {type(guess).__name__}.')

        table = guess.states_table
        if table.getNumRows() == 0 or table.getNumColumns() == 0:
            raise ValueError(
                'Initial guess states_table is empty '
                f'({table.getNumRows()} rows, {table.getNumColumns()} columns).')

        labels = set(table.getColumnLabels())
        missing = [coord_path + '/value' for coord_path in self.coordinates_map
                   if coord_path + '/value' not in labels]
        if missing:
            raise ValueError(
                f'Initial guess states_table is missing required coordinate columns: '
                f'{missing}.')

    @abstractmethod
    def solve(self, guess=None) -> Solution:
        pass


class TrackingSolver(Solver):
    """
    An abstract base class for solvers that track reference data. Reference data can be
    position-based (e.g., marker trajectories) or orientation-based (e.g., Theia frames)
    and should be provided as DataSource objects via the helper methods. Concrete
    subclasses must implement the solve() method, which should return a Solution object.

    Parameters
    ----------
    model: str or osim.Model
        See :py:class:`Solver`.
    convergence_tolerance: float, optional
        See :py:class:`Solver`.
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

        # Data sources.
        self.theia_frame_data: list[TheiaFrameData] = []
        self.marker_data: list[MarkerData] = []

    def add_theia_frame_reference_data(self, theia_frame_source: TheiaFrameSource):
        """
        Add a TheiaFrameSource as reference data for this solver.
        """
        positions = theia_frame_source.get_positions_table()
        orientations = theia_frame_source.get_orientations_table()
        DataSource.assert_position_orientation_consistent(positions, orientations)
        labels = positions.getColumnLabels()

        self.theia_frame_data.append(TheiaFrameData(labels, positions, orientations))

    def add_marker_reference_data(self, marker_source: MarkerSource):
        """
        Add a MarkerSource as reference data for this solver.
        """
        positions = marker_source.get_positions_table()
        labels = positions.getColumnLabels()

        self.marker_data.append(MarkerData(labels, positions))

    def get_times_from_reference_data(self):
        """
        Extract the time vector from the reference data, asserting that all data
        sources share the same time vector.
        """
        tables = [data.positions for data in self.theia_frame_data]
        tables += [data.positions for data in self.marker_data]
        return DataSource.assert_tables_share_times(tables)

    def create_tracking_callback(self, name: str, itime: int,
                                 position_weight: float,
                                 orientation_weight: float) -> TrackingCostFunction:
        """
        Create a CasADi callback function for computing the tracking cost at a given
        time step, which can be used in the formulation of an optimization problem.
        """
        callback = TrackingCostFunction(name, self.model)

        for data in self.theia_frame_data:
            for iframe, frame_path in enumerate(data.labels):
                callback.add_frame_tracking_cost(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=position_weight,
                    orientation_weight=orientation_weight)

        for data in self.marker_data:
            for iframe, marker_path in enumerate(data.labels):
                callback.add_marker_tracking_cost(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    weight=position_weight)

        return callback
    
    def _validate_guess(self, guess: Solution):
        super()._validate_guess(guess)
        num_times = len(self.get_times_from_reference_data())
        num_rows = guess.states_table.getNumRows()
        if num_rows != num_times:
            raise ValueError(
                f'Initial guess states_table has {num_rows} rows but the reference '
                f'data has {num_times} time samples.')


##############################
# INVERSE KINEMATICS SOLVERS #
##############################


class InverseKinematicsSolver(TrackingSolver):
    """
    Solve the inverse kinematics problem to find the set of model coordinate values that
    best track provided position (e.g., marker trajectories) and/or orientation (e.g.,
    frame orientations) data.

    Parameters
    ----------
    model: str or osim.Model
        See :py:class:`Solver`.
    convergence_tolerance: float, optional
        See :py:class:`Solver`.
    position_weight: float, optional
        See :py:class:`TrackingSolver`.
    orientation_weight: float, optional
        See :py:class:`TrackingSolver`.
    """

    _guess_type = TrackingSolution

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0):
        super().__init__(model, convergence_tolerance, position_weight,
                         orientation_weight)

    def create_tracking_solver(self, itime, position_weight, orientation_weight):
        """
        A helper function to create a CasADi solver for the tracking problem at a
        given time step.
        """
        x = ca.SX.sym('x', len(self.coordinate_indexes))
        callback = self.create_tracking_callback('tracking_cost', itime,
                                                 position_weight=position_weight,
                                                 orientation_weight=orientation_weight)
        f = callback(x)
        nlp = {'x': x, 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return callback, solver

    def solve(self, guess: TrackingSolution = None) -> TrackingSolution:

        times = self.get_times_from_reference_data()
        num_times = len(times)

        if guess is not None:
            self._validate_guess(guess)

        # Per-coordinate bounds, plus an initial x0 to use for the first time step
        # when no guess is supplied. The loop below carries x0 forward from the
        # previous step's solution (or pulls from the guess if provided).
        x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinates_map:
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # When a guess is provided, pre-extract a (num_times, num_coords) array of
        # initial values from the guess states_table so each timestep can be seeded
        # from the corresponding row.
        guess_q = None
        if guess is not None:
            guess_q = np.column_stack([
                guess.states_table.getDependentColumn(
                    coord_path + '/value').to_numpy()
                for coord_path in self.coordinates_map])

        # Iterate over all of the time steps in the tracking data and solve the
        # optimization problem at each time step.
        statesTraj = osim.StatesTrajectory()
        q_traj = np.zeros((num_times, len(self.coordinate_indexes)))
        for itime, time in enumerate(times):
            print(f'Solving time {itime+1} of {num_times} (t={time:.3f} s)...')

            if guess_q is not None:
                x0 = guess_q[itime, :].tolist()

            callback, solver = self.create_tracking_solver(itime,
                    position_weight=self.position_weight,
                    orientation_weight=self.orientation_weight)
            sol = solver(x0=x0, lbx=lbx, ubx=ubx)

            q_traj[itime, :] = np.squeeze(sol['x'].full())

            # Write solution into callback.state — avoids calling initSystem() again,
            # which would invalidate the state handle held by the callback.
            # StatesTrajectory.append() copies the state by value, so reuse is safe.
            callback.state.setTime(time)
            q = np.zeros(callback.state.getNQ())
            q[self.coordinate_indexes] = q_traj[itime, :]
            callback.state.setQ(osim.Vector.createFromMat(q))
            statesTraj.append(callback.state)

            if guess_q is None:
                x0 = sol['x']

        return TrackingSolution(
            states_table=statesTraj.exportToTable(self.model),
        )


class SplineBasedSolverMixin:
    """
    A mixin class that provides common functionality for spline-based solvers, which
    represent the predicted trajectories as B-splines and optimize over the spline
    control points.

    Parameters
    ----------
    degree: int, optional
        The degree of the B-spline basis functions. Default is 3 (i.e., cubic splines).
    knot_interval: float, optional
        The interval between knots in the B-spline basis. Default is 0.05 seconds.
    """
    def __init__(self, *args, degree=3, knot_interval=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.degree = degree
        self.knot_interval = knot_interval

    def build_knots_vector(self, times, num_knots):
        """
        Create a clamped knot vector. For n control points and degree p, there are
        n+p+1 knots. The first and last p+1 knots are clamped to the first and last time,
        respectively, and the interior knots are uniformly spaced between the first
        and last time.
        """
        knots = np.concatenate([
            np.repeat(times[0], self.degree),
            np.linspace(times[0], times[-1], num_knots - self.degree + 1),
            np.repeat(times[-1], self.degree),
        ])
        return knots

    def build_spline_basis_matrix(self, times, knots):
        """
        Build the spline basis matrix B and its derivative dB. B[i,j] = N_j(t_i),
        where N_j is the j-th B-spline basis function evaluated at time t_i.
        """

        # Build basis matrix B[i,j] = N_j(t_i) numerically.
        t = ca.MX.sym("t")
        num_knots = len(knots) - self.degree - 1

        # Scalar spline function for building B matrix.
        c_temp = ca.MX.sym("c_temp", num_knots, 1)
        spline = ca.bspline(t, c_temp, [knots], [self.degree], 1)
        spline_fn = ca.Function("spline", [t, c_temp], [spline])

        # Derivative of the spline w.r.t. time.
        spline_dt = ca.jacobian(spline, t)
        spline_fn_dt = ca.Function("spline_dt", [t, c_temp], [spline_dt])

        # Build basis matrix B[i,j] = N_j(t_i) by evaluating with unit coefficient
        # vectors.
        B = np.zeros((len(times), num_knots))
        dB = np.zeros((len(times), num_knots))
        for j in range(num_knots):
            e_j = np.zeros(num_knots)
            e_j[j] = 1.0
            B[:, j] = [float(spline_fn(ti, e_j)) for ti in times]
            dB[:, j] = [float(spline_fn_dt(ti, e_j)) for ti in times]

        return ca.DM(B), ca.DM(dB)

    def extract_coordinate_initial_guess(self, states_table, B, coord_path):
        """Extract an initial guess for the spline control points for a given coordinate
          by solving a least squares problem.
        """
        q_col = states_table.getDependentColumn(coord_path + '/value').to_numpy()
        q_guess, _, _, _ = np.linalg.lstsq(np.array(B), q_col, rcond=None)
        return q_guess.tolist()


class SplineBasedInverseKinematicsSolver(SplineBasedSolverMixin, TrackingSolver):
    """
    An inverse kinematics solver that optimizes model coordinate values to minimize 
    tracking error, where the predicted trajectories are represented as B-splines and 
    the optimization variables are the spline control points.

    Parameters
    ----------
    model: str or osim.Model
        See :py:class:`Solver`.
    convergence_tolerance: float, optional
        See :py:class:`Solver`.
    position_weight: float, optional
        See :py:class:`TrackingSolver`.
    orientation_weight: float, optional
        See :py:class:`TrackingSolver`.
    degree: int, optional
        See :py:class:`SplineBasedSolverMixin`.
    knot_interval: float, optional
        See :py:class:`SplineBasedSolverMixin`.
    """

    _guess_type = SplineTrackingSolution

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0, degree=3, knot_interval=0.05):
        super().__init__(model, convergence_tolerance=convergence_tolerance,
                         position_weight=position_weight,
                         orientation_weight=orientation_weight,
                         degree=degree, knot_interval=knot_interval)

    def solve(self, guess: SplineTrackingSolution = None) -> SplineTrackingSolution:

        times = self.get_times_from_reference_data()
        num_times = len(times)

        if guess is not None:
            self._validate_guess(guess)

        # Define the knot vector.
        num_knots = int(times[-1] / self.knot_interval)
        knots = self.build_knots_vector(times, num_knots)

        # Pre-compute the spline basis matrix, which is independent of the optimization
        # variables.
        B, dB = self.build_spline_basis_matrix(times, knots)

        # Define the optimization variables, which are the spline control points for
        # each coordinate.
        coeffs = ca.MX.sym('coeffs', num_knots, len(self.coordinate_indexes))
        x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinates_map:
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0 += ([coord.getDefaultValue()] * num_knots if guess is None
                   else self.extract_coordinate_initial_guess(
                       guess.states_table, B, coord_path))
            lbx += [coord.getRangeMin()] * num_knots
            ubx += [coord.getRangeMax()] * num_knots

        # Map the control points to the full predicted trajectory via the spline basis
        # matrix.
        q = B @ coeffs

        # Compute the tracking cost at each time step via a callback.
        errors = ca.MX(num_times, 1)
        callbacks = []
        for itime in range(num_times):
            callbacks.append(self.create_tracking_callback(
                f'tracking_cost_time_{itime}', itime,
                position_weight=self.position_weight,
                orientation_weight=self.orientation_weight))
            errors[itime] = callbacks[itime](q[itime, :].T)

        # Define the overall cost as the average tracking error across all time steps.
        f = ca.sum(errors) / num_times

        nlp = {'x': ca.vec(coeffs), 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Reconstruct the optimal trajectory by evaluating the spline at the
        # input data time points.
        coeffs_opt = ca.reshape(sol['x'], num_knots, len(self.coordinate_indexes))
        q_opt = np.array(B @ coeffs_opt)    # (num_times, num_coords)
        qdot_opt = np.array(dB @ coeffs_opt)

        return SplineTrackingSolution(
            states_table=TrackingSolution.create_states_table(
                self.model, self.state, self.coordinate_indexes, times, q_opt, qdot_opt),
            spline_nodes=np.array(coeffs_opt),
        )
    
###################
# BILEVEL SOLVERS #
###################

class BilevelSolver(TrackingSolver):
    """
    An abstract base class for solvers that solve bilevel optimization problems,
    i.e., problems that optimize over both the kinematics and body scale factors to
    minimize tracking error. Concrete subclasses must implement the solve() method,
    which should return a Solution object.

    Parameters
    ----------
    model: str or osim.Model
        See :py:class:`Solver`.
    convergence_tolerance: float, optional
        See :py:class:`Solver`.
    position_weight: float, optional
        See :py:class:`TrackingSolver`.
    orientation_weight: float, optional
        See :py:class:`TrackingSolver`.
    scale_regularization_weight: float, optional
        The weight to apply to the regularization term on the scale factors in the
        bilevel optimization problem to encourage them to stay close to 1.0 if changes
        in the scale factor don't substantially improve the tracking cost. Default is
        0.0 (i.e., no regularization).
    """
    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0, scale_regularization_weight=0.0):
        super().__init__(model, convergence_tolerance, position_weight,
                         orientation_weight)
        self.scale_regularization_weight = scale_regularization_weight
        self.scale_factors: list[ScaleFactor] = []

    @property
    def scale_groups(self) -> list[ScaleGroup]:
        """
        The list of :py:class:`ScaleGroup` objects configured on this solver via
        :py:meth:`add_scale_factor`, in the order they were added. Useful for
        constructing a :py:class:`BilevelSolution` initial guess that matches the
        solver's configuration.
        """
        return [sf.group for sf in self.scale_factors]

    def add_scale_factor(self, body_paths: str | list[str],
                         lower_bound, upper_bound):
        """
        Add a set of XYZ body scale factors to be optimized over in the bilevel
        optimization problem. Pass a single body path to scale one body, or a
        list of body paths to share one set of scale factors across a group of bodies 
        (e.g., for left-right symmetric scaling).

        Parameters
        ----------
        body_paths: str or list[str]
            Absolute model path(s) to the body or bodies whose scale factor will be 
            optimized. A list shares one set of scale factors across every body in the 
            group.
        lower_bound: float
            Lower bound on each component of the XYZ body scale factors.
        upper_bound: float
            Upper bound on each component of the XYZ body scale factors.
        """
        if isinstance(body_paths, str):
            body_paths = [body_paths]
        if not body_paths:
            raise ValueError(
                'body_paths must be a non-empty string or list of strings.')
        mobod_indexes = []
        for path in body_paths:
            body = osim.Body.safeDownCast(self.model.getComponent(path))
            mobod_indexes.append(int(body.getMobilizedBodyIndex()))
        self.scale_factors.append(ScaleFactor(
            group=ScaleGroup(list(body_paths), mobod_indexes),
            bounds=Bounds(lower_bound, upper_bound)))

    def create_bilevel_callback(self, name: str, itime: int,
                                position_weight: float,
                                orientation_weight: float) -> BilevelCostFunction:
        scale_groups = [sf.group for sf in self.scale_factors]
        callback = BilevelCostFunction(name, self.model, scale_groups)

        for data in self.theia_frame_data:
            raise NotImplementedError('TheiaFrameSource is not currently supported '
                                      'with scale factor optimization.')

        for data in self.marker_data:
            for iframe, marker_path in enumerate(data.labels):
                callback.add_marker_bilevel_cost(marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    weight=position_weight)

        return callback
    
    def update_model(self, model: osim.Model, solution: BilevelSolution) -> osim.Model:
        """
        Apply the solution's optimized per-group XYZ scale factors to `model`
        in place and return it.
        """
        scaler = Scaler(model)
        axes = (Axis.XAxis, Axis.YAxis, Axis.ZAxis)
        for group, factors in zip(solution.scale_groups, solution.scale_factors):
            for body_path in group.body_paths:
                body_name = osim.Body.safeDownCast(
                    model.getComponent(body_path)).getName()
                for ax_idx, axis in enumerate(axes):
                    scaler.add_scale_factor(ManualScaleFactor(
                        body_name, axis, float(factors[ax_idx])))
        return scaler.scale()
    
    def _validate_guess(self, guess: Solution):
        super()._validate_guess(guess)
        expected = self.scale_groups
        if guess.scale_groups != expected:
            raise ValueError(
                f'Initial guess scale_groups do not match the solver configuration. '
                f'Expected {len(expected)} group(s) matching '
                f'{[g.body_paths for g in expected]}, got '
                f'{len(guess.scale_groups) if guess.scale_groups is not None else 0} '
                f'group(s) matching '
                f'{[g.body_paths for g in (guess.scale_groups or [])]}.')

        expected_shape = (len(expected), 3)
        if guess.scale_factors is None or guess.scale_factors.shape != expected_shape:
            shape = (None if guess.scale_factors is None
                     else guess.scale_factors.shape)
            raise ValueError(
                f'Initial guess scale_factors must have shape {expected_shape}, '
                f'got {shape}.')


class SplineBasedBilevelSolver(SplineBasedSolverMixin, BilevelSolver):
    """
    A solver for bilevel optimization problems that optimize over both the kinematics
    and body scale factors to minimize tracking error, where the predicted trajectories
    are represented as B-splines and the optimization variables are the spline control
    points and scale factors.

    Parameters
    ----------
    model: str or osim.Model
        See :py:class:`Solver`.
    convergence_tolerance: float, optional
        See :py:class:`Solver`.
    position_weight: float, optional
        See :py:class:`TrackingSolver`.
    orientation_weight: float, optional
        See :py:class:`TrackingSolver`.
    scale_regularization_weight: float, optional
        See :py:class:`BilevelSolver`.
    degree: int, optional
        See :py:class:`SplineBasedSolverMixin`.
    knot_interval: float, optional
        See :py:class:`SplineBasedSolverMixin`.   
    """
    _guess_type = SplineBilevelSolution

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0, scale_regularization_weight=0.0,
                 degree=3, knot_interval=0.05):
        super().__init__(model, convergence_tolerance=convergence_tolerance,
                         position_weight=position_weight,
                         orientation_weight=orientation_weight,
                         scale_regularization_weight=scale_regularization_weight,
                         degree=degree, knot_interval=knot_interval)

    def solve(self, guess: SplineBilevelSolution = None) -> SplineBilevelSolution:

        times = self.get_times_from_reference_data()
        num_times = len(times)

        if guess is not None:
            self._validate_guess(guess)

        # Define the knot vector.
        num_knots = int(times[-1] / self.knot_interval)
        knots = self.build_knots_vector(times, num_knots)

        # Pre-compute the spline basis matrix, which is independent of the optimization
        # variables.
        B, dB = self.build_spline_basis_matrix(times, knots)

        # Define the optimization variables: spline control points and scale factors.
        coeffs = ca.MX.sym('coeffs', num_knots, len(self.coordinate_indexes))
        s = ca.MX.sym('scale_factors', 3*len(self.scale_factors))
        x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinates_map:
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0 += ([coord.getDefaultValue()] * num_knots if guess is None
                   else self.extract_coordinate_initial_guess(
                       guess.states_table, B, coord_path))
            lbx += [coord.getRangeMin()] * num_knots
            ubx += [coord.getRangeMax()] * num_knots
        if guess is None:
            for sf in self.scale_factors:
                x0 += [1.0, 1.0, 1.0]
                lbx += [sf.bounds.lower_bound] * 3
                ubx += [sf.bounds.upper_bound] * 3
        else:
            x0 += guess.scale_factors.flatten().tolist()
            for sf in self.scale_factors:
                lbx += [sf.bounds.lower_bound] * 3
                ubx += [sf.bounds.upper_bound] * 3

        # Map the control points to the full predicted trajectory via the spline basis
        # matrix.
        q = B @ coeffs

        # Compute the tracking cost at each time step via a callback.
        tracking_errors = ca.MX(num_times, 1)
        callbacks = []
        for itime in range(num_times):
            callbacks.append(self.create_bilevel_callback(
                f'scaled_tracking_cost_time_{itime}', itime,
                position_weight=self.position_weight,
                orientation_weight=self.orientation_weight))
            tracking_errors[itime] = callbacks[itime](q[itime, :].T, s)

        # Define the overall cost as the average tracking error across all time steps,
        # plus an optional regularization term on the scale factors.
        f_track = ca.sum(tracking_errors) / num_times
        f_scale_reg = self.scale_regularization_weight * ca.sum((s - 1.0)**2)
        f = f_track + f_scale_reg

        nlp = {'x': ca.vertcat(ca.vec(coeffs), s), 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Reconstruct the optimal trajectory by evaluating the spline at the
        # input data time points.
        num_coeff_vars = num_knots * len(self.coordinate_indexes)
        coeffs_opt = ca.reshape(sol['x'][:num_coeff_vars], num_knots,
                                len(self.coordinate_indexes))
        q_opt = np.array(B @ coeffs_opt)    # (num_times, num_coords)
        qdot_opt = np.array(dB @ coeffs_opt)

        # Reshape scale factors from flat vector to (num_scaled_bodies, 3).
        scales_opt = np.array(sol['x'][num_coeff_vars:]).flatten()
        scale_factors_mat = scales_opt.reshape(len(self.scale_factors), 3)

        return SplineBilevelSolution(
            states_table=TrackingSolution.create_states_table(
                self.model, self.state, self.coordinate_indexes, times, q_opt, qdot_opt),
            scale_factors=scale_factors_mat,
            scale_groups=self.scale_groups,
            spline_nodes=np.array(coeffs_opt),
        )
