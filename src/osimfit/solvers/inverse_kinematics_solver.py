import numpy as np
import casadi as ca
import opensim as osim

from ..data_sources import Trial
from ..costs import (CostEvaluation, CostInput, CostCache, CostWorkspace,
                     TrackingCost, TrialSample)
from ..costs.cost import CostTerm
from .solution import ObjectiveTerms, Solution
from .solver import TrackingSolver


class InverseKinematicsSolver(TrackingSolver):
    """
    Solve the inverse kinematics problem to find the set of model coordinate values that
    best track provided position (e.g., marker trajectories) and/or orientation (e.g.,
    frame orientations) data. Trials are solved sequentially.

    The coordinate speeds provided to costs (e.g., `CoordinateDampingCost`) are computed
    by a backward finite difference with the previous time step's solution. No speeds
    are available at the first time step of each trial.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    convergence_tolerance: float, optional
        See `Solver`.
    position_weight: float, optional
        See `TrackingSolver`.
    orientation_weight: float, optional
        See `TrackingSolver`.
    """

    SUPPORTED_INPUTS = frozenset({'coordinates', 'speeds'})

    def solve(self, guess: Solution = None) -> Solution:
        self._assert_has_trials()

        if guess is not None:
            self._validate_guess(guess)

        # Per-coordinate bounds, plus the default initial values used to seed the first
        # time step of each trial when no guess is supplied.
        default_x0 = []
        lbx = []
        ubx = []
        for coord_path in self.coordinate_q_map:
            coord = osim.Coordinate.safeDownCast(self.mc.model.getComponent(coord_path))
            default_x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # Solve each trial sequentially, restarting the warm start from the default
        # coordinate values (or that trial's guess) at each trial's first time step.
        cache = CostCache(self.mc)
        tracking_cost = TrackingCost(
            self.position_weight, self.orientation_weight, self.marker_weights,
            self.frame_position_weights, self.frame_orientation_weights)
        named_costs = [CostTerm('tracking', tracking_cost), *self.costs]

        # Accumulate each term's value over every sample of every trial, and report the
        # mean below.
        objective_terms = ObjectiveTerms()
        states_tables: dict[str, osim.TimeSeriesTable] = {}
        for trial in self.trials:
            times = trial.times
            num_times = len(times)

            # When a guess is provided, pre-extract a (num_times, num_coords) array of
            # initial values from this trial's guess states_table so each timestep can
            # be seeded from the corresponding row.
            guess_q = None
            if guess is not None:
                guess_q = self._extract_guess_coordinates(
                    guess.states_tables[trial.name])

            # Iterate over all of the time steps in the tracking data and solve the
            # optimization problem at each time step.
            x0 = list(default_x0)
            x_prev = list(default_x0)
            statesTraj = osim.StatesTrajectory()
            q_traj = np.zeros((num_times, len(self.coordinate_q_indexes)))
            for itime, time in enumerate(times):
                print(f"Trial '{trial.name}': solving time {itime+1} of {num_times} "
                      f'(t={time:.3f} s)...')

                if guess_q is None:
                    x0 = x_prev
                else:
                    x0 = guess_q[itime, :].tolist()

                # The first time step has no previous solution from which to compute
                # speeds, and is solved to a tighter tolerance.
                dt = None
                tolerance = self.convergence_tolerance
                if itime == 0:
                    tolerance *= 0.1
                else:
                    dt = time - times[itime - 1]

                q_opt, term_values = self._solve_step(
                    cache, trial, itime, named_costs, x_prev,
                    x0, lbx, ubx, dt, tolerance)
                objective_terms.accumulate(term_values)
                q_traj[itime, :] = q_opt

                # Write solution into the cache's state.
                cache.state.setTime(time)
                q = np.zeros(cache.state.getNQ())
                q[self.coordinate_q_indexes] = q_traj[itime, :]
                cache.state.setQ(osim.Vector.createFromMat(q))
                statesTraj.append(cache.state)
                x_prev = q_opt

            states_tables[trial.name] = statesTraj.exportToTable(self.mc.model)

        return Solution(
            states_tables=states_tables,
            outputs={'cost_terms': objective_terms.summarize_mean()})

    def _solve_step(self, cache: CostCache, trial: Trial, itime: int,
                    named_costs: list[CostTerm],
                    x_prev, x0, lbx, ubx, dt,
                    convergence_tolerance) -> tuple[np.ndarray, dict[str, float]]:
        """
        Build and solve one time step of the inverse kinematics problem.
        """
        workspace = CostWorkspace(cache)
        x = ca.SX.sym('x', len(self.coordinate_q_indexes))
        speeds = ca.SX(0, 1) if dt is None else (x - x_prev) / dt
        cost_input = CostInput(coordinates=x, speeds=speeds)
        terms = ObjectiveTerms()
        evaluation = CostEvaluation(cost_input, TrialSample(trial, itime))
        terms.add_costs(named_costs, evaluation, workspace)
        workspace.validate(named_costs)
        nlp = {'x': x, 'f': terms.objective}
        opts = {}
        opts['ipopt'] = self._get_ipopt_options(
            convergence_tolerance=convergence_tolerance)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)
        term_values = terms.evaluate(x, sol['x'])
        return sol['x'].full().reshape(-1), term_values
