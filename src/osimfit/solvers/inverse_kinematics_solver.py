import numpy as np
import casadi as ca
import opensim as osim

from ..data_sources import Trial
from ..costs import CostInput, CostRep, TrackingCost
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
        cost_reps = [(type(cost).__name__, cost.create_rep(self.mc))
                     for cost in self.costs]
        tracking_cost = TrackingCost(
            self.position_weight, self.orientation_weight, self.marker_weights,
            self.frame_position_weights, self.frame_orientation_weights)

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

                tracking_rep, solver, terms, x = self._create_tracking_solver(
                    trial, itime, tracking_cost, cost_reps, x_prev, guess_q)
                sol = solver(x0=x0, lbx=lbx, ubx=ubx)

                q_traj[itime, :] = np.squeeze(sol['x'].full())

                objective_terms.accumulate(terms.evaluate(x, sol['x']))

                # Write solution into the rep's state.
                tracking_rep.state.setTime(time)
                q = np.zeros(tracking_rep.state.getNQ())
                q[self.coordinate_q_indexes] = q_traj[itime, :]
                tracking_rep.state.setQ(osim.Vector.createFromMat(q))
                statesTraj.append(tracking_rep.state)
                x_prev = sol['x']

            states_tables[trial.name] = statesTraj.exportToTable(self.mc.model)

        return Solution(
            states_tables=states_tables,
            outputs={'cost_terms': objective_terms.summarize_mean()})

    def _create_tracking_solver(self, trial: Trial, itime: int,
                            tracking_cost: TrackingCost, cost_reps: list[CostRep],
                            x_prev: list[float], guess_q: list[float]):
        """
        A helper function to create a CasADi solver for the tracking problem at a
        given time step of a given trial.

        Parameters
        ----------
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` to track.
        tracking_cost: TrackingCost
            The tracking cost description, built once per solve.
        cost_reps: list[tuple[str, CostRep]]
            Reps of the solver's registered costs, built once per solve, each paired
            with its label in the objective breakdown.
        x_prev: list[float]
            The solution from the previous time step.
        guess_q: list[float]
            The initial guess for the current time step.

        """
        x = ca.SX.sym('x', len(self.coordinate_q_indexes))
        tracking_rep = tracking_cost.create_rep('tracking_cost', self.mc, trial, itime)
        cost_input = CostInput(coordinates=x)
        terms = ObjectiveTerms()
        terms.add('tracking', tracking_rep(cost_input))
        terms.add_costs(cost_reps, cost_input)
        if self.damping_weight > 0:
            terms.add('damping', self.damping_weight
                      * ca.dot(x - x_prev, x - x_prev) / x.numel())
        nlp = {'x': x, 'f': terms.objective}
        opts = {}
        opts['ipopt'] = self._get_ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return tracking_rep, solver, terms, x
