import numpy as np
import casadi as ca
import opensim as osim

from ..costs import BilevelCost, BilevelCostRep, CostInput, TrackingCost
from ..model import (ModelCache, Parameter, BodyScale, MarkerOffset, FrameOffset,
                     EllipsoidRadiiScale, BeamLengthScale)
from ..scaling import Axis, Scaler, ManualBodyScale
from ..utilities import set_model_mass
from .solution import ObjectiveTerms, Solution
from .solver import TrackingSolver


class SplinedKinematicsSolver(TrackingSolver):
    """
    Solve for model kinematics by representing each coordinate trajectory as a B-spline
    and optimizing over the spline control points across the whole trial. Because the
    control points couple every time step, this solver can optionally also optimize
    global bilevel parameters (e.g., body scales, marker and frame offsets, ellipsoid
    radii, and cantilever-free-beam lengths); register them with ``add_parameter``.
    With no parameters registered, it reduces to a spline-based inverse kinematics
    problem tracking the reference data.

    All registered trials are optimized simultaneously. Each trial
    contributes its own block of spline control points, with its own knot vector sized
    from its own duration, while the bilevel parameters are shared across every trial.
    The tracking objective uses the mean over trials of each trial's time-averaged
    error. Registered `Cost` terms that depend only on the shared parameters are
    evaluated once, not per trial.

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
    degree: int, optional
        The degree of the B-spline basis functions. Default is 3 (i.e., cubic splines).
    knot_interval: float, optional
        The spacing between consecutive knots in the B-spline basis, in seconds. Default
        is 0.05.
    damping_weight: float, optional
        The weight multiplied with the coordinate derivatives to apply a quadratic
        damping penalty to the cost function. Default is 0 (no damping).
    """
    SUPPORTED_INPUTS = frozenset({'coordinates', 'body_scales', 'marker_offsets',
                                  'frame_offsets', 'ellipsoid_radii_scales',
                                  'beam_length_scales'})

    def __init__(self, model, convergence_tolerance=1e-4, position_weight=1.0,
                 orientation_weight=1.0, degree=3, knot_interval=0.05,
                 damping_weight=0.0):
        super().__init__(model, convergence_tolerance=convergence_tolerance,
                         position_weight=position_weight,
                         orientation_weight=orientation_weight)
        self._parameters_by_input: dict[str, list[Parameter]] = {}
        self.degree = degree
        self.knot_interval = knot_interval
        self.damping_weight = damping_weight

    @property
    def parameters(self) -> list[Parameter]:
        """
        All registered parameters, flattened in `CostInput.INPUT_ORDER` (the order in
        which their variable blocks are concatenated into the optimization vector).
        Within an input, registration order is preserved.
        """
        return [p for name in CostInput.INPUT_ORDER
                for p in self._parameters_by_input.get(name, [])]

    @property
    def body_scales(self) -> list[BodyScale]:
        """
        The registered `BodyScale` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(BodyScale.cost_input, []))

    @property
    def marker_offsets(self) -> list[MarkerOffset]:
        """
        The registered `MarkerOffset` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(MarkerOffset.cost_input, []))

    @property
    def frame_offsets(self) -> list[FrameOffset]:
        """
        The registered `FrameOffset` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(FrameOffset.cost_input, []))

    @property
    def ellipsoid_radii_scales(self) -> list[EllipsoidRadiiScale]:
        """
        The registered `EllipsoidRadiiScale` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(EllipsoidRadiiScale.cost_input, []))

    @property
    def beam_length_scales(self) -> list[BeamLengthScale]:
        """
        The registered `BeamLengthScale` parameters, in registration order.
        """
        return list(self._parameters_by_input.get(BeamLengthScale.cost_input, []))

    def add_parameter(self, parameter: Parameter):
        """
        Register a `Parameter` to be optimized over in the bilevel optimization problem.
        The parameter is validated against the model at registration time.

        Parameters
        ----------
        parameter: Parameter
            The parameter to optimize (e.g., a `BodyScale`).

        Raises
        ------
        ValueError
            If `parameter` is not a `Parameter`, or its `cost_input` is not a recognized
            `CostInput` field.
        """
        if not isinstance(parameter, Parameter):
            raise ValueError(
                f'add_parameter expected a Parameter, but got '
                f'{type(parameter).__name__}.')
        CostInput.field_index(parameter.cost_input)
        parameter.validate(self.mc)
        self._parameters_by_input.setdefault(parameter.cost_input, []).append(parameter)
        self.mc.add_parameter_group(parameter.to_group())

    def solve(self, guess: Solution = None) -> Solution:
        self._assert_has_trials()

        if guess is not None:
            self._validate_guess(guess)

        num_coords = len(self.coordinate_q_indexes)

        # Build a spline basis per trial.
        trial_times: list[list[float]] = []
        trial_num_control_points: list[int] = []
        trial_B: list[ca.DM] = []
        trial_dB: list[ca.DM] = []
        for trial in self.trials:
            times = trial.times
            duration = times[-1] - times[0]
            num_intervals = int(round(duration / self.knot_interval))
            if num_intervals < 1:
                raise ValueError(
                    f"Trial '{trial.name}' spans {duration:.4g} s, which is shorter "
                    f"than half the requested knot interval of "
                    f"{self.knot_interval:.4g} s, so the trial cannot be divided into "
                    f"even one knot interval. Either provide a longer trial, or reduce "
                    f"knot_interval.")
            num_control_points = num_intervals + self.degree

            knots = self._build_knots_vector(times, num_intervals)
            B, dB = self._build_spline_basis_matrix(times, knots)
            trial_times.append(times)
            trial_num_control_points.append(num_control_points)
            trial_B.append(B)
            trial_dB.append(dB)

        # Extract parameter dimensions.
        num_params = len(self.parameters)
        num_scales = sum(p.num_variables for p in self.parameters
                         if isinstance(p, BodyScale))
        num_markers = sum(p.num_variables for p in self.parameters
                          if isinstance(p, MarkerOffset))
        num_frames = sum(p.num_variables for p in self.parameters
                         if isinstance(p, FrameOffset))
        num_radii = sum(p.num_variables for p in self.parameters
                        if isinstance(p, EllipsoidRadiiScale))
        num_lengths = sum(p.num_variables for p in self.parameters
                          if isinstance(p, BeamLengthScale))

        # Apply the parameters from the initial guess to the solver's list of registered
        # parameters.
        if num_params > 0 and guess is not None and guess.parameters is not None:
            for sp, gp in zip(self.parameters, guess.parameters):
                sp.value = np.asarray(gp.value, dtype=float)

        # Define the optimization variables: one block of control points per trial,
        # followed by the parameter blocks shared across all trials.
        coeffs = [ca.MX.sym(f'coeffs_{itrial}', num_control_points, num_coords)
                  for itrial, num_control_points in
                  enumerate(trial_num_control_points)]
        s = ca.MX.sym('body_scales', num_scales)
        mo = ca.MX.sym('marker_offsets', num_markers)
        fo = ca.MX.sym('frame_offsets', num_frames)
        er = ca.MX.sym('ellipsoid_radii_scales', num_radii)
        bl = ca.MX.sym('beam_length_scales', num_lengths)
        x0 = []
        lbx = []
        ubx = []
        for itrial, trial in enumerate(self.trials):
            num_control_points = trial_num_control_points[itrial]
            guess_table = (None if guess is None else guess.states_tables[trial.name])
            for coord_path in self.coordinate_q_map:
                coord = osim.Coordinate.safeDownCast(
                    self.mc.model.getComponent(coord_path))
                x0 += ([coord.getDefaultValue()] * num_control_points
                       if guess_table is None
                       else self._extract_coordinate_initial_guess(
                           guess_table, trial_B[itrial], coord_path))
                lbx += [coord.getRangeMin()] * num_control_points
                ubx += [coord.getRangeMax()] * num_control_points

        # Append each parameter's initial guess and bounds, in type order, matching the
        # [coeffs_0, ..., coeffs_J, s, mo, fo, er, bl] layout of the optimization vector
        # below.
        for p in self.parameters:
            p.append_guess_and_bounds(x0, lbx, ubx)

        # Accumulate the tracking cost for each trial. Reps are held in a list for the
        # lifetime of the solve so CasADi's references to them stay valid. Each rep is
        # paired with its label so the objective can be reported term by term.
        named_reps = [(type(cost).__name__, cost.create_rep(self.mc))
                      for cost in self.costs]
        coordinate_cost_reps = [(name, rep) for (name, rep), cost
                                in zip(named_reps, self.costs)
                                if 'coordinates' in cost.required_inputs]
        parameter_cost_reps = [(name, rep) for (name, rep), cost
                               in zip(named_reps, self.costs)
                               if 'coordinates' not in cost.required_inputs]

        # The labels of the terms evaluated at every time sample, which therefore need
        # a per-timestep error vector within each trial.
        timestep_term_names = ['tracking'] + [name for name, _
                                              in coordinate_cost_reps]
        terms = ObjectiveTerms()
        tracking_reps = []
        cost_type = BilevelCost if num_params > 0 else TrackingCost
        tracking_cost = cost_type(
            self.position_weight, self.orientation_weight, self.marker_weights,
            self.frame_position_weights, self.frame_orientation_weights)
        for itrial, trial in enumerate(self.trials):
            times = trial_times[itrial]
            num_times = len(times)

            # Map this trial's control points to its full predicted trajectory via its
            # spline basis matrix.
            q = trial_B[itrial] @ coeffs[itrial]

            # Compute the tracking cost at each time step via a callback rep. Each
            # term gets its own per-timestep error vector; the trapezoidal time
            # average below is linear, so averaging the terms separately and summing
            # them is equivalent to averaging their sum.
            errors = {name: ca.MX(num_times, 1) for name in timestep_term_names}
            for itime in range(num_times):
                tracking_rep = tracking_cost.create_rep(
                    f'tracking_cost_trial_{itrial}_time_{itime}', self.mc, trial,
                    itime)
                tracking_reps.append(tracking_rep)
                cost_input = CostInput(coordinates=q[itime, :].T, body_scales=s,
                                       marker_offsets=mo, frame_offsets=fo,
                                       ellipsoid_radii_scales=er, beam_length_scales=bl)
                errors['tracking'][itime] = tracking_rep(cost_input)
                for name, cost_rep in coordinate_cost_reps:
                    errors[name][itime] = cost_rep(cost_input)

            for name, term_errors in errors.items():
                terms.add(name, self._compute_average_trapezoidal_error(
                    term_errors, times))
            if self.damping_weight > 0:
                qdot = trial_dB[itrial] @ coeffs[itrial]
                terms.add('damping',
                          self.damping_weight
                          * self._compute_average_trapezoidal_error(
                              ca.sum2(qdot**2), times))

        # Average across trials.
        terms.scale(1.0 / len(self.trials))

        # Every offset group must be tracked in at least one trial.
        if num_params > 0:
            self._assert_offset_groups_used(tracking_reps)

        # Add the cost terms on the parameters shared across all trials. These are
        # evaluated once, and so bypass the per-trial averaging above.
        parameter_input = CostInput(body_scales=s, marker_offsets=mo, frame_offsets=fo,
                                    ellipsoid_radii_scales=er, beam_length_scales=bl)
        terms.add_costs(parameter_cost_reps, parameter_input)

        # Solve.
        x = ca.vertcat(*[ca.vec(c) for c in coeffs], s, mo, fo, er, bl)
        nlp = {'x': x, 'f': terms.objective}
        opts = {}
        opts['ipopt'] = self._get_ipopt_options(print_level=5)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx)

        # Reconstruct each trial's optimal trajectory by evaluating its spline at that
        # trial's input data time points.
        states_tables: dict[str, osim.TimeSeriesTable] = {}
        spline_nodes: dict[str, np.ndarray] = {}
        i = 0
        for itrial, trial in enumerate(self.trials):
            num_coeff_vars = trial_num_control_points[itrial] * num_coords
            coeffs_opt = ca.reshape(sol['x'][i : i + num_coeff_vars],
                                    trial_num_control_points[itrial], num_coords)
            q_opt = np.array(trial_B[itrial] @ coeffs_opt)
            qdot_opt = np.array(trial_dB[itrial] @ coeffs_opt)
            states_tables[trial.name] = Solution.create_states_table(
                self.mc.model, self.state, self.coordinate_q_indexes,
                self.mc.coordinate_u_indexes, trial_times[itrial], q_opt, qdot_opt)
            spline_nodes[trial.name] = np.array(coeffs_opt)
            i += num_coeff_vars

        # Slice each parameter's optimized value from the flat solution vector, when any
        # parameters were registered. The parameter blocks follow every trial's control
        # points, so `i` is already at the start of the first parameter block.
        solution_parameters = None
        if num_params > 0:
            x_flat = np.array(sol['x']).flatten()
            for p in self.parameters:
                p.value = x_flat[i : i + p.num_variables].reshape(-1)
                i += p.num_variables
            solution_parameters = [p.with_value(p.value) for p in self.parameters]

        return Solution(
            states_tables=states_tables,
            parameters=solution_parameters,
            outputs={
                'spline_nodes': spline_nodes,
                'cost_terms': terms.summarize(x, sol['x']),
            },
        )

    def _assert_offset_groups_used(self, reps: list[BilevelCostRep]):
        """
        Verify that every registered offset group is tracked by at least one task in at
        least one of `reps`, ensuring that the offset is properly constrained in
        the bilevel optimization problem.

        Raises
        ------
        ValueError
            If any registered marker or frame offset group is tracked by no task.
        """
        def assert_used(used, offset_groups, label):
            for i, group in enumerate(offset_groups):
                if i not in used:
                    raise ValueError(
                        f'{label.capitalize()} offset group {group.component_paths} is '
                        f'not tracked by any registered {label} in any trial; its '
                        f'offset would be unconstrained.')

        used_markers = {g for rep in reps
                        for g in rep.marker_term.offset_group_indexes
                        if g is not None}
        used_frames = {g for rep in reps
                       for g in rep.frame_term.offset_group_indexes
                       if g is not None}
        assert_used(used_markers, self.mc.marker_offset_groups, 'marker')
        assert_used(used_frames, self.mc.frame_offset_groups, 'frame')

    def update_model(self, model: osim.Model, solution: Solution,
                    mass: float = None) -> osim.Model:
        """
        Apply the solution's optimized parameters to `model` and return it.
        """
        model.initSystem()

        # Get pre-`Model::scale()` quanities.
        translation_scales = ModelCache.get_custom_joint_translation_scales(model)
        ellipsoid_radii = ModelCache.get_ellipsoid_joint_radii(model)

        # Construct a scaler using the optimized body scales as manual scale factors.
        # This calls Model::scale() under the hood. Body scales are applied via the
        # Scaler rather than each BodyScale's apply_to_model (see
        # BodyScale.apply_to_model).
        scaler = Scaler(model)
        axes = (Axis.XAxis, Axis.YAxis, Axis.ZAxis)
        for parameter in solution.parameters:
            if not isinstance(parameter, BodyScale):
                continue
            for body_path in parameter.paths:
                body_name = osim.Body.safeDownCast(
                    model.getComponent(body_path)).getName()
                for ax_idx, axis in enumerate(axes):
                    scaler.add_body_scale(ManualBodyScale(
                        body_name, axis, float(parameter.value[ax_idx])))
        model = scaler.scale()

        # Apply pre-`Model::scale()` quanities.
        ModelCache.apply_custom_joint_translation_scales(model, translation_scales)
        ModelCache.apply_ellipsoid_joint_radii(model, ellipsoid_radii)

        # Apply the remaining optimized parameters (e.g., marker and frame offsets,
        # ellipsoid radii, beam lengths) to the scaled model.
        for parameter in solution.parameters:
            if not isinstance(parameter, BodyScale):
                parameter.apply_to_model(model)

        # Update the model mass.
        if mass is not None:
            set_model_mass(model, mass)

        # Finalize the system and return.
        model.finalizeConnections()
        model.initSystem()
        return model

    def _build_knots_vector(self, times, num_intervals):
        """
        Create a clamped knot vector spanning `times` with `num_intervals` equally-wide
        knot intervals. The first and last knots are repeated `degree` times so that the
        spline is clamped to the first and last time, giving
        ``num_intervals + 1 + 2*degree`` knots and therefore
        ``num_intervals + degree`` control points.
        """
        knots = np.concatenate([
            np.repeat(times[0], self.degree),
            np.linspace(times[0], times[-1], num_intervals + 1),
            np.repeat(times[-1], self.degree),
        ])
        return knots

    def _build_spline_basis_matrix(self, times, knots):
        """
        Build the spline basis matrix B and its derivative dB. B[i,j] = N_j(t_i),
        where N_j is the j-th B-spline basis function evaluated at time t_i.
        """

        # Build basis matrix B[i,j] = N_j(t_i) numerically.
        t = ca.MX.sym("t")
        num_control_points = len(knots) - self.degree - 1

        # Scalar spline function for building B matrix.
        c_temp = ca.MX.sym("c_temp", num_control_points, 1)
        spline = ca.bspline(t, c_temp, [knots], [self.degree], 1)
        spline_fn = ca.Function("spline", [t, c_temp], [spline])

        # Derivative of the spline w.r.t. time.
        spline_dt = ca.jacobian(spline, t)
        spline_fn_dt = ca.Function("spline_dt", [t, c_temp], [spline_dt])

        # Build basis matrix B[i,j] = N_j(t_i) by evaluating with unit coefficient
        # vectors.
        B = np.zeros((len(times), num_control_points))
        dB = np.zeros((len(times), num_control_points))
        for j in range(num_control_points):
            e_j = np.zeros(num_control_points)
            e_j[j] = 1.0
            B[:, j] = [float(spline_fn(ti, e_j)) for ti in times]
            dB[:, j] = [float(spline_fn_dt(ti, e_j)) for ti in times]

        return ca.DM(B), ca.DM(dB)

    def _extract_coordinate_initial_guess(self, states_table, B, coord_path):
        """
        Extract an initial guess for the spline control points for a given coordinate
        by solving a least squares problem.
        """
        q_col = states_table.getDependentColumn(coord_path + '/value').to_numpy()
        q_guess, _, _, _ = np.linalg.lstsq(np.array(B), q_col, rcond=None)
        return q_guess.tolist()

    def _validate_guess(self, guess: Solution):
        super()._validate_guess(guess)
        if not self.parameters:
            return

        # Check that the solver parameters and guess parameters are the same size.
        expected = self.parameters
        got = guess.parameters or []
        if len(got) != len(expected):
            raise ValueError(
                f'Initial guess has {len(got)} parameter(s) but the solver is '
                f'configured with {len(expected)}.')

        # Enforce that the guess lists its parameters grouped in CostInput.INPUT_ORDER.
        positions = [CostInput.field_index(g.cost_input) for g in got]
        if positions != sorted(positions):
            raise ValueError(
                f'Initial guess parameters must be ordered by CostInput.INPUT_ORDER '
                f'{CostInput.INPUT_ORDER}, but got {[g.cost_input for g in got]}.')

        for e, g in zip(expected, got):
            if type(g) is not type(e) or g.paths != e.paths:
                raise ValueError(
                    f'Initial guess parameters do not match the solver configuration. '
                    f'Expected {type(e).__name__} on {e.paths}, got '
                    f'{type(g).__name__} on {g.paths}.')
            if g.value is None or np.asarray(g.value).shape != (e.num_variables,):
                shape = None if g.value is None else np.asarray(g.value).shape
                raise ValueError(
                    f'Initial guess value for {type(e).__name__} on {e.paths} must '
                    f'have shape ({e.num_variables},), got {shape}.')
