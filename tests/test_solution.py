"""
Tests for `Solution`, including the per-term breakdown of a solver's objective
that it reports under ``outputs['cost_terms']``.
"""

import json
import os

import numpy as np
import opensim as osim
import pytest

from osimfit.bounds import Bounds
from osimfit.model import BodyScale
from osimfit.solution import Solution
from osimfit.solvers import (InverseKinematicsSolver, MarkerPlacer,
                             SplinedKinematicsSolver)
from osimfit.costs import BodyScaleRegularizationCost, CoordinateStiffnessCost
from osimfit.data_sources import MarkerSource, Trial
from tests.test_double_pendulum import (create_double_pendulum,
                                        create_synthetic_markers_file)
from tests.test_marker_placer import create_placer_model, make_trial


###########
# HELPERS #
###########

def make_states_table(num_rows=5, coordinates=('/jointset/j0/q0',
                                               '/jointset/j1/q1')):
    table = osim.TimeSeriesTable()
    labels = [f'{c}/value' for c in coordinates] + [f'{c}/speed' for c in coordinates]
    table.setColumnLabels(labels)
    for row in range(num_rows):
        table.appendRow(0.1 * row, osim.RowVector([0.01 * row] * len(labels)))
    return table


def make_solution(num_trials=1, with_parameters=True, with_spline_nodes=True):
    states = {f'trial_{i}': make_states_table() for i in range(num_trials)}
    parameters = None
    if with_parameters:
        parameters = [BodyScale('/bodyset/b0', Bounds(0.5, 2.0),
                                np.array([1.1, 1.0, 0.9]))]
    outputs = {'cost_terms': {'total': 4.0,
                              'terms': {'tracking': {'value': 3.0,
                                                     'fraction': 0.75},
                                        'reg': {'value': 1.0,
                                                'fraction': 0.25}}}}
    if with_spline_nodes:
        outputs['spline_nodes'] = {name: np.arange(6.0).reshape(3, 2)
                                   for name in states}
    return Solution(states_tables=states, parameters=parameters, outputs=outputs)


#########
# TESTS #
#########

def test_writes_one_file_per_result(tmp_path):
    solution = make_solution(num_trials=2)
    written = solution.print(str(tmp_path), prefix='run')
    names = sorted(os.path.basename(p) for p in written)
    assert names == ['run_cost_terms.json',
                     'run_parameters.json',
                     'run_trial_0_spline_nodes.sto',
                     'run_trial_0_states.sto',
                     'run_trial_1_spline_nodes.sto',
                     'run_trial_1_states.sto']
    # Every reported path exists, and nothing else was written.
    for path in written:
        assert os.path.isfile(path)
    assert sorted(os.listdir(tmp_path)) == names


def test_the_prefix_is_optional(tmp_path):
    written = make_solution().print(str(tmp_path))
    assert os.path.basename(written[0]) == 'trial_0_states.sto'


def test_the_directory_is_created(tmp_path):
    target = tmp_path / 'nested' / 'results'
    make_solution().print(str(target))
    assert target.is_dir()


def test_states_tables_round_trip(tmp_path):
    solution = make_solution()
    solution.print(str(tmp_path), prefix='run')
    reloaded = osim.TimeSeriesTable(str(tmp_path / 'run_trial_0_states.sto'))
    original = solution.states_tables['trial_0']
    assert reloaded.getNumRows() == original.getNumRows()
    assert list(reloaded.getColumnLabels()) == list(original.getColumnLabels())


def test_per_trial_arrays_are_written_with_coordinate_labels(tmp_path):
    solution = make_solution()
    solution.print(str(tmp_path), prefix='run')
    nodes = osim.TimeSeriesTable(str(tmp_path / 'run_trial_0_spline_nodes.sto'))
    # Column count matches the trial's coordinates, so the coordinate paths are
    # used rather than generic names.
    assert list(nodes.getColumnLabels()) == ['/jointset/j0/q0', '/jointset/j1/q1']
    assert nodes.getNumRows() == 3


def test_arrays_of_unexpected_width_fall_back_to_generic_labels(tmp_path):
    solution = make_solution(with_spline_nodes=False)
    solution.outputs['spline_nodes'] = {
        'trial_0': np.arange(9.0).reshape(3, 3)}   # 3 columns, 2 coordinates
    solution.print(str(tmp_path), prefix='run')
    nodes = osim.TimeSeriesTable(str(tmp_path / 'run_trial_0_spline_nodes.sto'))
    assert list(nodes.getColumnLabels()) == ['col_0', 'col_1', 'col_2']


def test_parameters_are_written_with_paths_values_and_bounds(tmp_path):
    make_solution().print(str(tmp_path), prefix='run')
    described = json.load(open(tmp_path / 'run_parameters.json'))
    assert described == [{'type': 'BodyScale',
                          'paths': ['/bodyset/b0'],
                          'value': [1.1, 1.0, 0.9],
                          'bounds': {'lower': 0.5, 'upper': 2.0}}]


def test_cost_terms_are_written_verbatim(tmp_path):
    solution = make_solution()
    solution.print(str(tmp_path), prefix='run')
    assert (json.load(open(tmp_path / 'run_cost_terms.json'))
            == solution.outputs['cost_terms'])


def test_a_solution_without_parameters_writes_no_parameter_file(tmp_path):
    written = make_solution(with_parameters=False).print(str(tmp_path))
    assert not any(p.endswith('parameters.json') for p in written)


def test_an_unsupported_output_name_raises(tmp_path):
    solution = make_solution(with_spline_nodes=False)
    solution.outputs['iterations'] = 42
    with pytest.raises(ValueError, match="'iterations'"):
        solution.print(str(tmp_path), prefix='run')


def test_an_unsupported_output_name_is_rejected_before_anything_is_written(tmp_path):
    directory = tmp_path / 'results'
    solution = make_solution()
    solution.outputs['iterations'] = 42
    with pytest.raises(ValueError, match='SUPPORTED_OUTPUTS'):
        solution.print(str(directory), prefix='run')
    assert not directory.exists()


def test_an_unserializable_output_value_raises(tmp_path):
    solution = make_solution(with_spline_nodes=False)
    solution.outputs['spline_nodes'] = osim.Model()
    with pytest.raises(TypeError, match="'spline_nodes' holds a Model"):
        solution.print(str(tmp_path), prefix='run')


def test_an_unserializable_output_value_leaves_no_truncated_file(tmp_path):
    solution = make_solution(with_spline_nodes=False)
    solution.outputs['spline_nodes'] = osim.Model()
    with pytest.raises(TypeError):
        solution.print(str(tmp_path), prefix='run')
    assert not (tmp_path / 'run_spline_nodes.json').exists()


def test_the_summary_reports_trials_parameters_and_cost_terms(tmp_path, capsys):
    make_solution().print(str(tmp_path), prefix='run')
    out = capsys.readouterr().out
    assert 'trials' in out and 'trial_0' in out and '5 rows' in out
    assert 'parameters' in out and 'BodyScale(/bodyset/b0)' in out
    assert 'cost terms' in out and 'tracking' in out and '75.0%' in out
    assert 'total' in out
    assert 'wrote 4 file(s)' in out


def test_an_empty_solution_writes_nothing_and_still_reports(tmp_path, capsys):
    written = Solution().print(str(tmp_path))
    assert written == []
    assert 'wrote 0 file(s)' in capsys.readouterr().out


#######################
# OBJECTIVE BREAKDOWN #
#######################

class BodyScalePenaltySubclass(BodyScaleRegularizationCost):
    """
    A distinct subclass of `BodyScaleRegularizationCost`, to check that labels follow
    the concrete class rather than the base one.
    """


def create_pendulum_trial(tmp_path, name='pendulum', duration=0.5,
                          initial_q0=0.0) -> Trial:
    trc_path = str(tmp_path / f'{name}.trc')
    create_synthetic_markers_file(trc_path, 1.25, 0.75, duration=duration,
                                  initial_q0=initial_q0)
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}
    return Trial(name, [MarkerSource(name, trc_path, label_map=label_map)])


PENDULUM_COORDINATES = ('/jointset/j0/q0', '/jointset/j1/q1')
PENDULUM_STIFFNESSES = {'/jointset/j0/q0': 2.0, '/jointset/j1/q1': 5.0}


def calc_stiffness_penalty_per_sample(states_table, weight) -> np.ndarray:
    q = np.column_stack([states_table.getDependentColumn(path + '/value').to_numpy()
                         for path in PENDULUM_COORDINATES])
    stiffnesses = np.array([PENDULUM_STIFFNESSES[path]
                            for path in PENDULUM_COORDINATES])
    return weight * (q ** 2) @ stiffnesses


def assert_well_formed(cost_terms, expected_names):
    assert set(cost_terms) == {'total', 'terms'}
    assert set(cost_terms['terms']) == set(expected_names)

    values = [term['value'] for term in cost_terms['terms'].values()]
    total = cost_terms['total']
    assert total > 0.0
    assert sum(values) == pytest.approx(total)
    assert sum(term['fraction'] for term in cost_terms['terms'].values()) == \
        pytest.approx(1.0)
    for term in cost_terms['terms'].values():
        assert term['value'] > 0.0
        assert term['fraction'] == pytest.approx(term['value'] / total)


##############################
# SPLINED KINEMATICS SOLVER  #
##############################

def test_splined_solver_attributes_the_objective_to_each_term(tmp_path):
    weight = 1e-2
    solver = SplinedKinematicsSolver(create_double_pendulum(1.0, 1.0),
                                     convergence_tolerance=1e-5,
                                     knot_interval=0.07,
                                     position_weight=5.0,
                                     damping_weight=1e-3)
    solver.add_trial(create_pendulum_trial(tmp_path))
    solver.add_cost(BodyScaleRegularizationCost(weight))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
    solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))

    solution = solver.solve()
    cost_terms = solution.outputs['cost_terms']
    assert_well_formed(cost_terms,
                       ['tracking', 'damping', 'BodyScaleRegularizationCost'])

    # The penalty is weight * sum_i (s_i - 1)^2 over every optimized scale, so its
    # contribution can be recomputed independently from the solution's parameters.
    scales = np.concatenate([p.value for p in solution.parameters
                             if isinstance(p, BodyScale)])
    expected = weight * float(np.sum((scales - 1.0) ** 2))
    reported = cost_terms['terms']['BodyScaleRegularizationCost']['value']
    assert reported == pytest.approx(expected, rel=1e-6)


def test_splined_solver_attributes_a_per_timestep_cost(tmp_path):
    weight = 1e-3
    solver = SplinedKinematicsSolver(create_double_pendulum(1.0, 1.0),
                                     convergence_tolerance=1e-5,
                                     knot_interval=0.07,
                                     position_weight=5.0)
    solver.add_trial(create_pendulum_trial(tmp_path))
    solver.add_cost(CoordinateStiffnessCost(PENDULUM_STIFFNESSES, weight=weight))

    solution = solver.solve()
    cost_terms = solution.outputs['cost_terms']
    assert_well_formed(cost_terms, ['tracking', 'CoordinateStiffnessCost'])

    # The trapezoidal time average the solver applies, over the one trial.
    states_table = solution.states_tables['pendulum']
    penalty = calc_stiffness_penalty_per_sample(states_table, weight)
    times = np.array(states_table.getIndependentColumn())
    dt = np.diff(times)
    weights = np.zeros(len(times))
    weights[:-1] += 0.5 * dt
    weights[1:] += 0.5 * dt
    expected = float(weights @ penalty) / (times[-1] - times[0])

    reported = cost_terms['terms']['CoordinateStiffnessCost']['value']
    assert reported == pytest.approx(expected, rel=1e-6)


def test_splined_solver_omits_damping_when_it_is_disabled(tmp_path):
    solver = SplinedKinematicsSolver(create_double_pendulum(1.0, 1.0),
                                     convergence_tolerance=1e-4,
                                     knot_interval=0.1)
    solver.add_trial(create_pendulum_trial(tmp_path))

    cost_terms = solver.solve().outputs['cost_terms']
    assert_well_formed(cost_terms, ['tracking'])


def test_splined_solver_separates_distinct_costs(tmp_path):
    weight = 1e-2
    solver = SplinedKinematicsSolver(create_double_pendulum(1.0, 1.0),
                                     convergence_tolerance=1e-4,
                                     knot_interval=0.1)
    solver.add_trial(create_pendulum_trial(tmp_path))
    solver.add_cost(BodyScaleRegularizationCost(weight))
    solver.add_cost(BodyScalePenaltySubclass(2.0 * weight))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
    solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))

    cost_terms = solver.solve().outputs['cost_terms']
    assert_well_formed(cost_terms, ['tracking',
                                    'BodyScaleRegularizationCost',
                                    'BodyScalePenaltySubclass'])

    terms = cost_terms['terms']
    assert terms['BodyScalePenaltySubclass']['value'] == pytest.approx(
        2.0 * terms['BodyScaleRegularizationCost']['value'], rel=1e-6)


def test_splined_solver_sums_cost_terms_over_trials(tmp_path):
    solver = SplinedKinematicsSolver(create_double_pendulum(1.0, 1.0),
                                     convergence_tolerance=1e-4,
                                     knot_interval=0.1)
    solver.add_trial(create_pendulum_trial(tmp_path, name='swing_a'))
    solver.add_trial(create_pendulum_trial(tmp_path, name='swing_b',
                                           initial_q0=0.3))
    solver.add_cost(BodyScaleRegularizationCost(1e-2))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))

    cost_terms = solver.solve().outputs['cost_terms']
    assert_well_formed(cost_terms, ['tracking', 'BodyScaleRegularizationCost'])


#############################
# INVERSE KINEMATICS SOLVER #
#############################

def test_inverse_kinematics_reports_cost_terms_averaged_over_samples(tmp_path):
    solver = InverseKinematicsSolver(create_double_pendulum(1.0, 1.0),
                                     convergence_tolerance=1e-5,
                                     position_weight=5.0,
                                     damping_weight=1e-3)
    solver.add_trial(create_pendulum_trial(tmp_path, duration=0.1))

    cost_terms = solver.solve().outputs['cost_terms']
    assert_well_formed(cost_terms, ['tracking', 'damping'])


def test_inverse_kinematics_attributes_a_registered_cost(tmp_path):
    weight = 1e-3
    solver = InverseKinematicsSolver(create_double_pendulum(1.0, 1.0),
                                     convergence_tolerance=1e-6,
                                     position_weight=5.0)
    solver.add_trial(create_pendulum_trial(tmp_path, duration=0.1))
    solver.add_cost(CoordinateStiffnessCost(PENDULUM_STIFFNESSES, weight=weight))

    solution = solver.solve()
    cost_terms = solution.outputs['cost_terms']
    assert_well_formed(cost_terms, ['tracking', 'CoordinateStiffnessCost'])

    penalty = calc_stiffness_penalty_per_sample(
        solution.states_tables['pendulum'], weight)
    reported = cost_terms['terms']['CoordinateStiffnessCost']['value']
    assert reported == pytest.approx(float(np.mean(penalty)), rel=1e-6)


#################
# MARKER PLACER #
#################

def test_marker_placer_reports_its_single_tracking_term():
    truth = create_placer_model(np.array([0.1, 0.05, -0.02]))
    placer = MarkerPlacer(create_placer_model(np.zeros(3)),
                          convergence_tolerance=1e-6)
    placer.add_trial(make_trial('pose_a', truth, 0.0, 0.0, first_time=0.0))
    placer.add_trial(make_trial('pose_b', truth, 0.4, -0.3, first_time=1.0))

    cost_terms = placer.solve().outputs['cost_terms']
    assert set(cost_terms['terms']) == {'tracking'}
    assert cost_terms['terms']['tracking']['value'] == pytest.approx(
        cost_terms['total'])
    assert cost_terms['terms']['tracking']['fraction'] == pytest.approx(1.0)
