"""Cost placement, reference binding, and reduction through the common interface."""

import casadi as ca
import numpy as np
import pytest

from osimfit.bounds import Bounds
from osimfit.costs import (BilevelCost, BodyScaleRegularizationCost, Cost,
                           CostCache, CostEvaluation, CostInput, CostTerm,
                           CostWorkspace, CoordinateStiffnessCost, TrackingCost,
                           TrialSample)
from osimfit.data_sources import Trial
from osimfit.model import BodyScale, FrameOffset, MarkerOffset, ModelCache
from osimfit.solvers import InverseKinematicsSolver, MarkerPlacer, SplinedKinematicsSolver
from osimfit.solvers.solution import ObjectiveTerms
from tests.test_double_pendulum import create_double_pendulum
from tests.test_data_sources import PositionsSource, PositionsAndOrientationsSource
from tests.test_marker_placer import create_placer_model, PosedMarkerSource
from tests.test_offsets import create_offset_test_model
from tests.test_solution import create_pendulum_trial


@pytest.mark.parametrize('cost_type', [TrackingCost, BilevelCost])
def test_tracking_binds_each_sample_and_trial_independently(tmp_path, cost_type):
    # Identical names must not conflate distinct reference datasets.
    first = create_pendulum_trial(tmp_path, name='trial', duration=0.1)
    second = create_pendulum_trial(tmp_path, name='trial', duration=0.1,
                                    initial_q0=0.8)
    samples = [TrialSample(first, 0), TrialSample(first, len(first.times) - 1),
               TrialSample(second, 0)]
    workspace = CostWorkspace(CostCache(ModelCache(create_double_pendulum(1, 1))))
    cost = cost_type()
    q = ca.MX.sym('q', 2)
    inputs = CostInput(coordinates=q)
    expressions = [cost.expression(workspace, CostEvaluation(inputs, sample))
                   for sample in samples]
    function = ca.Function('samples', [q], [ca.vertcat(*expressions)])
    jacobian = ca.Function('sample_jacobians', [q],
                           [ca.jacobian(ca.vertcat(*expressions), q)])

    # Preparation is reused for a sample, even with a new TrialSample wrapper.
    callback = workspace.cost_callback(cost, samples[0])
    assert workspace.cost_callback(cost, TrialSample(first, 0)) is callback
    assert len(workspace.contexts(cost)) == 3
    assert len({id(context) for context in workspace.contexts(cost)}) == 3
    other_workspace = CostWorkspace(workspace.cache)
    assert other_workspace.cost_callback(cost, samples[0]).context is not callback.context

    values = np.array([0.1, -0.2])
    numeric_inputs = CostInput(coordinates=ca.DM(values))
    unpacked = {name: np.array(getattr(numeric_inputs, name)).reshape(-1)
                for name in cost.input_names()}
    expected = [cost.calc_value(
        workspace.cache,
        cost.create_tracking_task_set(workspace.cache, sample.trial, sample.index),
        unpacked) for sample in samples]
    np.testing.assert_allclose(np.array(function(values)).reshape(-1), expected)
    assert not np.isclose(expected[0], expected[1])
    assert not np.isclose(expected[0], expected[2])

    # Evaluate after all samples have been prepared, including their derivatives.
    step = 1e-6
    finite_difference = np.column_stack([
        np.array(function(values + delta) - function(values - delta)).reshape(-1)
        / (2 * step) for delta in step * np.eye(2)])
    np.testing.assert_allclose(jacobian(values), finite_difference, atol=1e-7)


def test_tracking_requires_a_sample():
    workspace = CostWorkspace(CostCache(ModelCache(create_double_pendulum(1, 1))))
    with pytest.raises(ValueError, match='requires a trial sample'):
        TrackingCost().expression(workspace, CostEvaluation(CostInput()))


@pytest.mark.parametrize('declared_scope, needs_sample, message', [
    ('unknown', False, 'Unknown cost scope'),
    ('global', True, 'requires a trial sample'),
    ('global', False, 'global scope'),
])
def test_invalid_scope_is_rejected_at_registration(declared_scope, needs_sample, message):
    class InvalidScopeCost(Cost):
        scope = declared_scope
        requires_sample = needs_sample
        required_inputs = frozenset({'coordinates'})

        def expression(self, workspace, evaluation):
            return ca.sumsqr(evaluation.inputs.coordinates)

    solver = SplinedKinematicsSolver(create_double_pendulum(1, 1))
    with pytest.raises(ValueError, match=message):
        solver.add_cost(InvalidScopeCost())
    assert solver.costs == []


def test_ik_rejects_global_scope():
    class GlobalCoordinateCost(CoordinateStiffnessCost):
        scope = 'global'

    solver = InverseKinematicsSolver(create_double_pendulum(1, 1))
    with pytest.raises(ValueError, match='does not support global cost scope'):
        solver.add_cost(GlobalCoordinateCost({'/jointset/j0/q0': 1}))


def test_scope_is_defined_by_each_cost_class():
    solver = SplinedKinematicsSolver(create_double_pendulum(1, 1))
    costs = [TrackingCost(), CoordinateStiffnessCost({'/jointset/j0/q0': 1}),
             BodyScaleRegularizationCost(1)]
    for cost in costs:
        solver.add_cost(cost)
    assert [term.scope for term in solver.costs] == ['sample', 'sample', 'global']
    assert all(term.scope == term.cost.scope for term in solver.costs)


class ReferenceScaleCost(Cost):
    """A reference-dependent parameter cost without any coordinate dependency."""
    required_inputs = frozenset({'body_scales'})
    requires_sample = True

    def __init__(self):
        self.evaluations = []

    def expression(self, workspace, evaluation):
        self.evaluations.append(evaluation.sample)
        sample = evaluation.sample
        time = sample.trial.times[sample.index]
        return (1 + time) * ca.sumsqr(evaluation.inputs.body_scales)


class GlobalScaleCost(Cost):
    scope = 'global'
    required_inputs = frozenset({'body_scales'})

    def __init__(self):
        self.num_evaluations = 0

    def expression(self, workspace, evaluation):
        assert evaluation.sample is None
        assert evaluation.inputs.coordinates.numel() == 0
        self.num_evaluations += 1
        return 2 * ca.sumsqr(evaluation.inputs.body_scales)


def test_spline_reduces_sample_parameter_costs_and_evaluates_global_cost_once(tmp_path):
    solver = SplinedKinematicsSolver(create_double_pendulum(1, 1), knot_interval=0.1)
    for name, duration in [('short', 0.1), ('long', 0.2)]:
        solver.add_trial(create_pendulum_trial(tmp_path, name=name, duration=duration))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2), np.ones(3)))
    sample_cost, global_cost = ReferenceScaleCost(), GlobalScaleCost()
    solver.add_cost(sample_cost)
    solver.add_cost(global_cost)
    solution = solver.solve()

    scales = solution.get_parameter('/bodyset/b0', BodyScale).value
    scale_penalty = np.sum(scales**2)
    # The time average of a linear function is its midpoint value, irrespective of
    # sample count; the two trials have equal weight despite different durations.
    expected = np.mean([1 + 0.5 * (trial.times[0] + trial.times[-1])
                        for trial in solver.trials]) * scale_penalty
    terms = solution.outputs['cost_terms']['terms']
    assert terms['ReferenceScaleCost']['value'] == pytest.approx(expected)
    assert terms['GlobalScaleCost']['value'] == pytest.approx(2 * scale_penalty)
    assert global_cost.num_evaluations == 1
    assert sample_cost.evaluations == [TrialSample(trial, i)
                                      for trial in solver.trials
                                      for i in range(len(trial.times))]


def test_nonuniform_sample_reduction_preserves_global_penalty(tmp_path):
    trial = create_pendulum_trial(tmp_path, duration=0.1)
    # Values of a linear signal at nonuniform sample times, averaged analytically.
    times = np.array([2.0, 2.1, 2.6, 3.0])
    weights = SplinedKinematicsSolver._compute_average_trapezoidal_weights(times)
    assert weights @ times == pytest.approx(2.5)
    terms = ObjectiveTerms()
    registered = [
        CostTerm('sample', CoordinateStiffnessCost(
            {'/jointset/j0/q0': 1, '/jointset/j1/q1': 1},
            targets={'/jointset/j0/q0': 0, '/jointset/j1/q1': 0})),
        CostTerm('global', BodyScaleRegularizationCost(1)),
    ]
    workspace = CostWorkspace(CostCache(ModelCache(create_double_pendulum(1, 1))))
    inputs = CostInput(coordinates=ca.DM([2, 3]), body_scales=ca.DM([2, 3, 4]))
    for i, weight in enumerate(weights):
        terms.add_costs(registered, CostEvaluation(inputs, TrialSample(trial, i)),
                        workspace, weight)
    terms.add_costs(registered, CostEvaluation(inputs), workspace)
    assert float(terms.terms['sample']) == pytest.approx(13)
    assert float(terms.terms['global']) == pytest.approx(14)
    assert float(terms.objective) == pytest.approx(27)


@pytest.mark.parametrize('solver_type', [InverseKinematicsSolver, SplinedKinematicsSolver])
def test_registered_tracking_cost_uses_the_common_solver_path(tmp_path, solver_type):
    solver = solver_type(create_double_pendulum(1, 1))
    solver.add_trial(create_pendulum_trial(tmp_path, duration=0.1))
    solver.add_cost(TrackingCost(position_weight=2))
    terms = solver.solve().outputs['cost_terms']['terms']
    assert terms['TrackingCost']['value'] == pytest.approx(2 * terms['tracking']['value'])


@pytest.mark.parametrize('solver_type, declared_scope', [
    (InverseKinematicsSolver, 'sample'),
    (SplinedKinematicsSolver, 'sample'),
    (SplinedKinematicsSolver, 'global'),
])
def test_cost_validation_runs_after_assembly_before_optimization(
        tmp_path, monkeypatch, solver_type, declared_scope):
    evaluations = []
    validations = []

    class ValidatingCost(Cost):
        scope = declared_scope

        def expression(self, workspace, evaluation):
            evaluations.append(evaluation.sample)
            return ca.MX(0)

        def validate(self, workspace):
            validations.append(list(evaluations))
            raise ValueError('cost validation failed')

    def unexpected_solver(*args, **kwargs):
        pytest.fail('NLP solver constructed before cost validation succeeded')

    monkeypatch.setattr(ca, 'nlpsol', unexpected_solver)
    solver = solver_type(create_double_pendulum(1, 1))
    solver.add_cost(ValidatingCost())
    for name in ('first', 'second'):
        solver.add_trial(create_pendulum_trial(tmp_path, name=name, duration=0.1))

    with pytest.raises(ValueError, match='cost validation failed'):
        solver.solve()

    if declared_scope == 'global':
        expected = [None]
    elif solver_type is InverseKinematicsSolver:
        expected = [TrialSample(solver.trials[0], 0)]
    else:
        expected = [TrialSample(trial, i) for trial in solver.trials
                    for i in range(len(trial.times))]
    assert validations == [expected]


@pytest.mark.parametrize('parameter_type, path, source_type, label', [
    (MarkerOffset, '/markerset/m_body', PositionsSource, 'Marker'),
    (FrameOffset, '/bodyset/body/pof', PositionsAndOrientationsSource, 'Frame'),
])
def test_offset_validation_combines_trials_only_within_its_workspace(
        parameter_type, path, source_type, label):
    mc = ModelCache(create_offset_test_model())
    parameter = parameter_type(path, Bounds(-0.1, 0.1), np.zeros(3))
    parameter.validate(mc)
    mc.add_parameter_group(parameter.to_group())
    cost = BilevelCost()
    named_costs = [CostTerm('tracking', cost)]
    workspace = CostWorkspace(CostCache(mc))

    first_source = PositionsSource()
    first_source.labels_to_remove = ['b']
    first_source.label_map = {'a': '/markerset/m_pof'}
    first = Trial('first', [first_source])
    workspace.cost_callback(cost, TrialSample(first, 0))
    with pytest.raises(ValueError, match=f'{label} offset group .*not tracked'):
        workspace.validate(named_costs)

    second_source = source_type()
    second_source.labels_to_remove = ['b']
    second_source.label_map = {'a': path}
    second = Trial('second', [second_source])
    workspace.cost_callback(cost, TrialSample(second, 0))
    workspace.validate(named_costs)

    # Reusing the cost and model cache cannot carry coverage into a new problem.
    other_workspace = CostWorkspace(workspace.cache)
    other_workspace.cost_callback(cost, TrialSample(first, 0))
    with pytest.raises(ValueError, match=f'{label} offset group .*not tracked'):
        other_workspace.validate(named_costs)


@pytest.mark.parametrize('solver_type', [SplinedKinematicsSolver, MarkerPlacer])
def test_solvers_reject_untracked_offsets_through_cost_validation(monkeypatch, solver_type):
    model = create_placer_model(np.zeros(3))
    source = PosedMarkerSource(model, {}, times=(0.0, 0.1))
    source.labels_to_remove = ['/markerset/m1']
    solver = solver_type(model)
    solver.add_trial(Trial('missing_marker', [source]))
    if solver_type is SplinedKinematicsSolver:
        solver.add_parameter(MarkerOffset('/markerset/m1', Bounds(-0.1, 0.1), np.zeros(3)))

    def unexpected_solver(*args, **kwargs):
        pytest.fail('NLP solver constructed with an untracked offset')

    monkeypatch.setattr(ca, 'nlpsol', unexpected_solver)
    with pytest.raises(ValueError, match='Marker offset group .*m1.*not tracked'):
        solver.solve()
