import numpy as np
import casadi as ca
import opensim as osim

from ..bounds import Bounds
from ..costs import BilevelCost, CostInput, CostCache, CostWorkspace
from ..model import MarkerOffset
from .solution import ObjectiveTerms, Solution
from .solver import Solver


class MarkerPlacer(Solver):
    """
    A solver for placing unfixed, e.g. "tracking", markers on the model.

    The solver minimizes the squared distance between the model's marker positions and
    the reference marker positions provided by each trial's `MarkerSource`. Markers
    whose '<fixed>' property is set to ``True`` will be used to pose the model, as in a
    typical inverse kinematics problem. Markers whose '<fixed>' property is set to
    ``False`` will have the position offsets optimized to place them as close as
    possible to the reference positions.

    The solver operates on the first time point of each registered trial. If multiple
    trials are registered, the solver operates on all time points simultaneously.

    Parameters
    ----------
    model: str or osim.Model
        See `Solver`.
    offset_bounds: Bounds, optional
        The bounds on the marker position offsets to optimize. Default is
        [-0.5, 0.5] meters in each direction.
    convergence_tolerance: float, optional
        See `Solver`.
    """
    SUPPORTED_INPUTS = frozenset({'coordinates', 'marker_offsets'})

    def __init__(self, model: osim.Model, offset_bounds: Bounds = Bounds(-0.5, 0.5),
                 convergence_tolerance=1e-4):
        super().__init__(model, convergence_tolerance)
        self.offset_bounds = offset_bounds

    def solve(self, guess: Solution = None) -> Solution:
        self._assert_has_trials()
        self._validate_trials()
        self._assert_no_costs()

        if guess is not None:
            self._validate_guess(guess)

        num_coords = len(self.coordinate_q_indexes)

        # Define the marker offset parameters. These are shared across every trial.
        marker_offsets: list[MarkerOffset] = []
        initial_offset = np.zeros(3)
        for tracking_marker in self.mc.get_tracking_marker_paths():
            marker_offset = MarkerOffset(tracking_marker, self.offset_bounds,
                                         initial_offset)
            marker_offset.validate(self.mc)
            marker_offsets.append(marker_offset)
        self.mc.marker_offset_groups = [mo.to_group() for mo in marker_offsets]

        # Define the optimization variables: one pose per trial, followed by the shared
        # offsets.
        num_markers = sum(mo.num_variables for mo in marker_offsets)
        poses = [ca.MX.sym(f'pose_{itrial}', num_coords)
                 for itrial in range(len(self.trials))]
        s = ca.MX.sym('body_scales', 0)
        mo = ca.MX.sym('marker_offsets', num_markers)
        fo = ca.MX.sym('frame_offsets', 0)

        # Define each pose's initial guess and bounds, seeding from the first row of
        # that trial's guess states table when one is supplied.
        x0 = []
        lbx = []
        ubx = []
        for trial in self.trials:
            guess_q = (None if guess is None else self._extract_guess_coordinates(
                guess.states_tables[trial.name])[0, :])
            for icoord, coord_path in enumerate(self.coordinate_q_map):
                coord = osim.Coordinate.safeDownCast(
                    self.mc.model.getComponent(coord_path))
                x0.append(coord.getDefaultValue() if guess_q is None
                          else float(guess_q[icoord]))
                lbx.append(coord.getRangeMin())
                ubx.append(coord.getRangeMax())
        for marker_offset in marker_offsets:
            marker_offset.append_guess_and_bounds(x0, lbx, ubx)

        # Accumulate the placement error at each trial's first time point. The workspace
        # holds every callback for the lifetime of the solve so CasADi's references to
        # them stay valid. _validate_trials rejects frame data, so a BilevelCost's frame
        # terms are always empty here and each callback tracks markers only.
        workspace = CostWorkspace(CostCache(self.mc))
        errors = ca.MX(len(self.trials), 1)
        placement_cost = BilevelCost(position_weight=1.0)
        for itrial, trial in enumerate(self.trials):
            errors[itrial] = placement_cost.expression(
                workspace,
                CostInput(coordinates=poses[itrial], body_scales=s,
                          marker_offsets=mo, frame_offsets=fo),
                trial, 0, f'marker_placer_cost_trial_{itrial}')

        # Average over trials so the objective's magnitude is independent of the number
        # of trials.
        terms = ObjectiveTerms()
        terms.add('tracking', ca.sum1(errors) / len(self.trials))

        # Solve.
        x = ca.vertcat(*poses, s, mo, fo)
        nlp = {'x': x, 'f': terms.objective}
        opts = {}
        opts['ipopt'] = self._get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Slice each trial's optimized pose from the flat solution vector, storing it as
        # a one-row states table stamped with that trial's first time.
        x_flat = np.array(sol['x']).flatten()
        states_tables: dict[str, osim.TimeSeriesTable] = {}
        i = 0
        for trial in self.trials:
            pose = x_flat[i : i + num_coords].reshape(1, num_coords)
            states_tables[trial.name] = Solution.create_states_table(
                self.mc.model, self.state, self.coordinate_q_indexes,
                self.mc.coordinate_u_indexes, [trial.times[0]], pose)
            i += num_coords

        # The shared offset blocks follow every trial's pose, so `i` is already at the
        # start of the first offset block.
        for marker_offset in marker_offsets:
            marker_offset.value = x_flat[
                i : i + marker_offset.num_variables].reshape(-1)
            i += marker_offset.num_variables

        return Solution(
            states_tables=states_tables,
            parameters=marker_offsets,
            outputs={'cost_terms': terms.summarize(x, sol['x'])},
        )

    def update_model(self, model: osim.Model,
                     solution: Solution) -> osim.Model:
        """
        Apply the solution's optimized marker placement offsets to `model` in place and
        return it. Each offset is baked into its marker's ``location`` property as an
        additive translation expressed in the marker's base frame.
        """
        model.initSystem()
        for parameter in solution.parameters or []:
            parameter.apply_to_model(model)
        model.finalizeConnections()
        model.initSystem()
        return model

    def _assert_no_costs(self):
        """
        Verify that no additional `Cost` is registered. This solver's objective is the
        marker placement error alone, so a registered cost has nothing to contribute to.

        Raises
        ------
        ValueError
            If any cost was registered via `add_cost`.
        """
        if self.costs:
            registered = sorted(type(cost).__name__ for cost in self.costs)
            raise ValueError(
                f'{type(self).__name__} does not accept additional costs; its '
                f'objective is the marker placement error alone, but {registered} '
                f'{"was" if len(registered) == 1 else "were"} registered.')

    def _validate_trials(self):
        """
        Verify that every registered trial carries marker data this solver can place,
        and nothing it cannot.

        Raises
        ------
        ValueError
            If a trial carries no marker data, carries frame data, or labels a marker
            that is not a marker in the model.
        """
        markerset = self.mc.model.getMarkerSet()
        model_markers = {markerset.get(i).getAbsolutePathString()
                         for i in range(markerset.getSize())}
        for trial in self.trials:
            if not trial.marker_data:
                raise ValueError(
                    f"Trial '{trial.name}' carries no marker data; MarkerPlacer places "
                    f'markers and so requires a MarkerSource in every trial.')
            if trial.frame_data:
                raise ValueError(
                    f"Trial '{trial.name}' carries frame data, which MarkerPlacer does "
                    f'not support; it places markers only.')
            for data in trial.marker_data:
                missing = sorted(set(data.labels) - model_markers)
                if missing:
                    raise ValueError(
                        f"Trial '{trial.name}' has reference data labels that are not "
                        f'markers in {self.mc.model.getName()}: {missing}.')
