import numpy as np
import casadi as ca
import opensim as osim

from ..data_sources import Trial
from ..costs import (CostEvaluation, CostInput, CostCache, CostTerm, CostWorkspace,
                     TrackingCost, TrialSample)
from .solution import ObjectiveTerms, Solution
from .solver import TrackingSolver


class InverseKinematicsSolver(TrackingSolver):
    """
    Solve the inverse kinematics problem to find the set of model coordinate values that
    best track provided position (e.g., marker trajectories) and/or orientation (e.g.,
    frame orientations) data. Trials are solved sequentially.

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
    damping_weight: float, optional
        The weight to use for a quadratic damping term that penalizes large changes in
        the coordinate values between consecutive time steps. Default is 0 (no damping).
    """

    SUPPORTED_INPUTS = frozenset({'coordinates'})

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0, damping_weight=0):
        super().__init__(model, convergence_tolerance, position_weight,
                         orientation_weight)
        self.damping_weight = damping_weight

    def solve(self, guess: Solution = None) -> Solution:
        self._assert_has_trials()

        if self.damping_weight < 0:
            raise ValueError(
                f'{type(self).__name__} damping_weight must be non-negative, but got '
                f'{self.damping_weight}.')

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

                q_opt, term_values = self._solve_step(
                    cache, trial, itime, named_costs, x_prev,
                    x0, lbx, ubx)
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
                    x_prev, x0, lbx, ubx) -> tuple[np.ndarray, dict[str, float]]:
        """
        Build and solve one time step of the inverse kinematics problem.

        Parameters
        ----------
        cache: CostCache
            Evaluation data shared by every time step of this solve.
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` to track.
        named_costs: list[CostTerm]
            Tracking and user costs with their labels and evaluation scopes.
        x_prev: array-like
            The solution from the previous time step.
        x0: array-like
            The initial guess for the current time step.
        lbx, ubx: array-like
            Lower and upper coordinate bounds.
        """
        workspace = CostWorkspace(cache)
        x = ca.SX.sym('x', len(self.coordinate_q_indexes))
        cost_input = CostInput(coordinates=x)
        terms = ObjectiveTerms()
        evaluation = CostEvaluation(cost_input, TrialSample(trial, itime))
        terms.add_costs(named_costs, evaluation, workspace)
        if self.damping_weight > 0:
            terms.add('damping', self.damping_weight
                      * ca.dot(x - x_prev, x - x_prev) / x.numel())
        workspace.validate(named_costs)
        nlp = {'x': x, 'f': terms.objective}
        opts = {}
        opts['ipopt'] = self._get_ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)
        term_values = terms.evaluate(x, sol['x'])
        return sol['x'].full().reshape(-1), term_values
