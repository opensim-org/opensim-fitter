from os import times

import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from .utilities import get_coordinate_indexes, get_ipopt_options
from .callbacks import TrackingCostJacobianCallback


class Solver(ABC):
    def __init__(self, model, positions, orientations,
                 convergence_tolerance=1e-4, weights={}):
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

        # Tracking data.
        # --------------
        self.positions = positions
        self.orientations = orientations
        assert(self.positions.getNumRows() == self.orientations.getNumRows())

        # Optimization settings.
        # ----------------------
        self.convergence_tolerance = convergence_tolerance
        self.weights = weights

    @abstractmethod
    def solve(self, guess=None) -> osim.TimeSeriesTable:
        pass


class InverseKinematicsSolver(Solver):

    def __init__(self, model, positions, orientations, convergence_tolerance=1e-6,
                 weights={}):
        super().__init__(model, positions, orientations, convergence_tolerance, weights)

    def _build_solver(self, frame_paths, positions, quaternions, weights):
        x = ca.SX.sym('x', len(self.coordinate_indexes))
        p = ca.SX.sym('p', len(self.coordinate_indexes))
        callback = TrackingCostJacobianCallback('tracking_cost', self.model,
                                                self.coordinate_indexes,
                                                frame_paths, positions, quaternions,
                                                weights)
        tracking_cost = ca.Function('f', [x], [callback(x)])
        f = tracking_cost(x) + weights['smoothness'] * ca.sumsqr(x - p)
        nlp = {'x': x, 'p': p, 'f': f}
        opts = {}
        opts['ipopt'] = get_ipopt_options(self.convergence_tolerance)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return callback, solver

    def solve(self, guess=None) -> osim.TimeSeriesTable:

        if guess is not None:
            raise ValueError(f'InverseKinematicsSolver does not currently support '
                             f'using an initial guess, but got a guess with '
                             f"{guess.getNumRows()} rows.")

        # Load tracking data
        # ------------------
        frame_paths = self.positions.getColumnLabels()
        times = self.positions.getIndependentColumn()

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
        callback, solver = self._build_solver(
            frame_paths,
            self.positions.getRowAtIndex(0),
            self.orientations.getRowAtIndex(0),
            {'position': 10.0*self.weights['position'],
             'orientation': 0.1*self.weights['orientation'],
             'smoothness': 0.01*self.weights['smoothness']}
        )
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=x0)
        x0 = sol['x']

        # Build the callback and solver once for the main time-stepping loop.
        callback, solver = self._build_solver(
            frame_paths,
            self.positions.getRowAtIndex(0),
            self.orientations.getRowAtIndex(0),
            self.weights
        )

        # Iterate over all of the time steps in the tracking data and solve the
        # optimization problem at each time step.
        statesTraj = osim.StatesTrajectory()
        for itime, time in enumerate(times):
            print(f'Solving time {itime+1} of {len(times)} (t={time:.3f} s)...')

            callback.update_data(self.positions.getRowAtIndex(itime),
                                 self.orientations.getRowAtIndex(itime))
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


class SplineInverseKinematicsSolver(Solver):

    def __init__(self, model, positions, orientations, convergence_tolerance=1e-6,
                 weights={}, degree=3, knot_interval=0.05):
        super().__init__(model, positions, orientations, convergence_tolerance, weights)
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

        # Build basis matrix B[i,j] = N_j(t_i) by evaluating with unit coefficient
        # vectors.
        B = np.zeros((len(times), num_knots))
        for j in range(num_knots):
            e_j = np.zeros(num_knots); e_j[j] = 1.0
            B[:, j] = [float(spline_fn(ti, e_j)) for ti in times]

        return ca.DM(B)

    def _extract_coordinate_initial_guess(self, guess, B, coord_path):
        q_col = guess.getDependentColumn(coord_path + '/value').to_numpy()
        q_guess, _, _, _ = np.linalg.lstsq(np.array(B), q_col, rcond=None)
        return q_guess.tolist()

    def solve(self, guess=None) -> osim.TimeSeriesTable:

        # Preliminaries.
        # --------------
        if guess is not None and guess.getNumRows() != self.positions.getNumRows():
            raise ValueError(f'Expected the initial guess to have the same number of '
                             f'rows as the tracking data, but got {guess.getNumRows()} '
                             f'and {self.positions.getNumRows()} rows, respectively.')

        # Define the (normalized) time vector.
        times = self.positions.getIndependentColumn()
        dt = times[1] - times[0]
        num_times = len(times)

        # Define the knot vector.
        num_knots = int(times[-1] / self.knot_interval)
        knots = self._build_knots_vector(times, num_knots)

        # Pre-compute the spline basis matrix, which is independent of the optimization
        # variables.
        B = self._build_spline_basis_matrix(times, knots)

        # Define the optimization variables, which are the spline control points for
        # each coordinate.
        x = ca.MX.sym('x', num_knots, len(self.coordinate_indexes))
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
        q = B @ x

        # Compute the tracking cost at each time step via a callback.
        # Call initSystem() once and copy the state for each callback to avoid
        # repeated initSystem() calls invalidating earlier states (Simbody invalidates
        # all State objects when realizeTopology() is called again).
        frame_paths = self.positions.getColumnLabels()
        base_state = self.model.initSystem()
        errors = ca.MX(num_times, 1)
        callbacks = []
        for i in range(num_times):
            callbacks.append(TrackingCostJacobianCallback(
                    f'tracking_cost_time_{i}',
                    self.model,
                    self.coordinate_indexes,
                    frame_paths,
                    self.positions.getRowAtIndex(i),
                    self.orientations.getRowAtIndex(i),
                    self.weights))
            errors[i] = callbacks[i](q[i, :].T)

        # Define the overall cost as the sum of squared tracking errors.
        f = ca.sum(errors) / num_times

        # Define the NLP and solver.
        nlp = {'x': ca.vec(x), 'f': f}
        opts = {}
        opts['ipopt'] = get_ipopt_options(self.convergence_tolerance)
        opts['ipopt']['print_level'] = 5
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve!
        # ------
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Reconstruct the optimal trajectory by evaluating the spline at the
        # input data time points.
        x_opt = ca.reshape(sol['x'], num_knots, len(self.coordinate_indexes))
        q_opt = np.array(B @ x_opt)  # (num_times, n_coords)

        actual_times = self.positions.getIndependentColumn()
        statesTraj = osim.StatesTrajectory()
        for i, time in enumerate(actual_times):
            self.state.setTime(time)
            q = np.zeros(self.state.getNQ())
            q[self.coordinate_indexes] = q_opt[i, :]
            self.state.setQ(osim.Vector.createFromMat(q))
            statesTraj.append(self.state)

        return statesTraj.exportToTable(self.model)
