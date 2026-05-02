import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .utilities import get_coordinate_indexes
from .data_sources import MarkerSource, TheiaFrameSource
from .callbacks import TrackingCostCallback


@dataclass
class TheiaFrameData:
    labels: list[str]
    positions: osim.TimeSeriesTableVec3
    orientations: osim.TimeSeriesTableQuaternion


@dataclass
class MarkerData:
    labels: list[str]
    positions: osim.TimeSeriesTableVec3


class Solver(ABC):
    def __init__(self, model, convergence_tolerance=1e-4, weights={}):
        super().__init__()

        # Load the model.
        # ---------------
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
        # ----------------------
        self.convergence_tolerance = convergence_tolerance
        self.weights = weights

        # Data sources.
        # -------------
        self.theia_frame_data: list[TheiaFrameData] = []
        self.marker_data: list[MarkerData] = []

    def add_theia_frame_source(self, theia_frame_source: TheiaFrameSource):
        positions = theia_frame_source.get_positions_table()
        orientations = theia_frame_source.get_orientations_table()
        labels = positions.getColumnLabels()
        assert(labels == orientations.getColumnLabels())
        assert(positions.getNumRows() == orientations.getNumRows())

        self.theia_frame_data.append(TheiaFrameData(labels, positions, orientations))

    def add_marker_source(self, marker_source: MarkerSource):
        positions = marker_source.get_positions_table()
        labels = positions.getColumnLabels()

        self.marker_data.append(MarkerData(labels, positions))

    @abstractmethod
    def solve(self, guess=None) -> osim.TimeSeriesTable:
        pass

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
        # ipopt_options['constr_viol_tol'] = self.constraint_tolerance
        # ipopt_options['acceptable_constr_viol_tol'] = self.constraint_tolerance
        ipopt_options['print_level'] = print_level

        return ipopt_options


class InverseKinematicsSolver(Solver):

    def __init__(self, model, convergence_tolerance=1e-4, weights={}):
        super().__init__(model, convergence_tolerance, weights)

    def _build_solver(self, weights, itime):
        x = ca.SX.sym('x', len(self.coordinate_indexes))
        p = ca.SX.sym('p', len(self.coordinate_indexes))

        callback = TrackingCostCallback('tracking_cost', self.model, weights)

        for data in self.theia_frame_data:
            for iframe, frame_path in enumerate(data.labels):
                callback.add_frame_tracking_cost(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe))

        for data in self.marker_data:
            for iframe, marker_path in enumerate(data.labels):
                callback.add_marker_tracking_cost(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe))

        f = callback(x) + weights['smoothness'] * ca.sumsqr(x - p)
        nlp = {'x': x, 'p': p, 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return callback, solver

    def solve(self, guess=None) -> osim.TimeSeriesTable:

        if guess is not None:
            raise ValueError(f'InverseKinematicsSolver does not currently support '
                             f'using an initial guess, but received a guess with '
                             f"{guess.getNumRows()} rows.")

        # Verify tracking data
        # ----------------------
        times = None
        for data in self.theia_frame_data:
            if times is None:
                times = data.positions.getIndependentColumn()
            else:
                assert(np.all(times == data.positions.getIndependentColumn()))

        for data in self.marker_data:
            if times is None:
                times = data.positions.getIndependentColumn()
            else:
                assert(np.all(times == data.positions.getIndependentColumn()))

        # Inverse kinematics
        # ------------------
        # Define initial guess and bounds.
        x0 = []
        lbx = []
        ubx = []
        for coord_path, ix in self.coordinates_map.items():
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # Solve position-only optimization to create an inital guess for the full IK
        # problem.
        print('Solving initial guess optimization...')
        initial_weights = {'position': 10.0*self.weights.get('position', 0.0),
                           'orientation': 0.1*self.weights.get('orientation', 0.0),
                           'smoothness': 0.01*self.weights.get('smoothness', 0.0)}
        callback, solver = self._build_solver(initial_weights, 0)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=x0)
        x0 = sol['x']

        # Iterate over all of the time steps in the tracking data and solve the
        # optimization problem at each time step.
        statesTraj = osim.StatesTrajectory()
        for itime, time in enumerate(times):
            print(f'Solving time {itime+1} of {len(times)} (t={time:.3f} s)...')

            callback, solver = self._build_solver(self.weights, itime)
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=x0)

            # Write solution into callback.state — avoids calling initSystem() again,
            # which would invalidate the state handle held by the callback.
            # StatesTrajectory.append() copies the state by value, so reuse is safe.
            callback.state.setTime(time)
            q = np.zeros(callback.state.getNQ())
            q[self.coordinate_indexes] = np.squeeze(sol['x'].full())
            callback.state.setQ(osim.Vector.createFromMat(q))
            statesTraj.append(callback.state)

            # Use the solution for the current time step as the initial guess for the next
            # time step.
            x0 = sol['x']

        # Export the solution to a .sto file.
        statesTable = statesTraj.exportToTable(self.model)

        return statesTable


class SplineBasedInverseKinematicsSolver(Solver):

    def __init__(self, model, convergence_tolerance=1e-4, weights={}, degree=3,
                 knot_interval=0.05):
        super().__init__(model, convergence_tolerance, weights)
        self.degree = degree
        self.knot_interval = knot_interval

    def _build_knots_vector(self, times, num_knots):
        # Clamped knot vector. For n control points and degree p, there are n+p+1 knots.
        # The first and last p+1 knots are clamped to the first and last time,
        # respectively, and the interior knots are uniformly spaced between the first
        # and last time.
        knots = np.concatenate([
            np.repeat(times[0], self.degree),
            np.linspace(times[0], times[-1], num_knots - self.degree + 1),
            np.repeat(times[-1], self.degree),
        ])
        return knots

    def _build_spline_basis_matrix(self, times, knots):

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

    def _extract_coordinate_initial_guess(self, guess, B, coord_path):
        q_col = guess.getDependentColumn(coord_path + '/value').to_numpy()
        q_guess, _, _, _ = np.linalg.lstsq(np.array(B), q_col, rcond=None)
        return q_guess.tolist()

    def solve(self, guess=None) -> osim.TimeSeriesTable:

        # Verify tracking data
        # ----------------------
        times = None
        for data in self.theia_frame_data:
            if times is None:
                times = data.positions.getIndependentColumn()
            else:
                assert(np.all(times == data.positions.getIndependentColumn()))

        for data in self.marker_data:
            if times is None:
                times = data.positions.getIndependentColumn()
            else:
                assert(np.all(times == data.positions.getIndependentColumn()))

        num_times = len(times)

        # Check the initial guess.
        # ------------------------
        if guess is not None and guess.getNumRows() != num_times:
            raise ValueError(f'Expected the initial guess to have the same number of '
                             f'rows as the tracking data, but got {guess.getNumRows()} '
                             f'and {num_times} rows, respectively.')

        # Define the knot vector.
        num_knots = int(times[-1] / self.knot_interval)
        knots = self._build_knots_vector(times, num_knots)

        # Pre-compute the spline basis matrix, which is independent of the optimization
        # variables.
        B, dB = self._build_spline_basis_matrix(times, knots)

        # Define the optimization variables, which are the spline control points for
        # each coordinate.
        coeffs = ca.MX.sym('coeffs', num_knots, len(self.coordinate_indexes))
        x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinates_map:
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0 += [coord.getDefaultValue()] * num_knots if not guess else \
                  self._extract_coordinate_initial_guess(guess, B, coord_path)
            lbx += [coord.getRangeMin()] * num_knots
            ubx += [coord.getRangeMax()] * num_knots

        # Define the optimization problem.
        # --------------------------------
        # Map the control points to the full predicted trajectory via the spline basis
        # matrix.
        q = B @ coeffs

        # Compute the tracking cost at each time step via a callback.
        errors = ca.MX(num_times, 1)
        callbacks = []
        for i in range(num_times):
            callbacks.append(TrackingCostCallback(f'tracking_cost_time_{i}', self.model,
                                                  self.weights))

            for data in self.theia_frame_data:
                for iframe, frame_path in enumerate(data.labels):
                    callbacks[i].add_frame_tracking_cost(
                        frame_path,
                        data.positions.getRowAtIndex(i).getElt(0, iframe),
                        data.orientations.getRowAtIndex(i).getElt(0, iframe))

            for data in self.marker_data:
                for iframe, marker_path in enumerate(data.labels):
                    callbacks[i].add_marker_tracking_cost(
                        marker_path,
                        data.positions.getRowAtIndex(i).getElt(0, iframe))

            errors[i] = callbacks[i](q[i, :].T)

        # Define the overall cost as the average tracking error across all time steps.
        f = ca.sum(errors) / num_times

        # Define the NLP and solver.
        nlp = {'x': ca.vec(coeffs), 'f': f}
        opts = {}
        opts['ipopt'] = self.get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve!
        # ------
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Reconstruct the optimal trajectory by evaluating the spline at the
        # input data time points.
        coeffs_opt = ca.reshape(sol['x'], num_knots, len(self.coordinate_indexes))
        q_opt = np.array(B @ coeffs_opt)  # (num_times, n_coords)
        qdot_opt = np.array(dB @ coeffs_opt)

        statesTraj = osim.StatesTrajectory()
        for i, time in enumerate(times):
            self.state.setTime(time)
            q = np.zeros(self.state.getNQ())
            q[self.coordinate_indexes] = q_opt[i, :]
            self.state.setQ(osim.Vector.createFromMat(q))
            qdot = np.zeros(self.state.getNQ())
            qdot[self.coordinate_indexes] = qdot_opt[i, :]
            self.state.setU(osim.Vector.createFromMat(qdot))
            statesTraj.append(self.state)

        return statesTraj.exportToTable(self.model)
