"""
Tests for `ProblemRep`, the per-solve problem representation, and the `WorkerJar` that
lets one tracking callback be evaluated from several threads.

The central property these tests pin down is equivalence: a single rep per trial,
evaluated across all of a trial's samples via `ca.Function.map`, must produce exactly
what the previous design produced from one rep per (trial, time sample).
"""

import threading
import contextlib

import pytest
import numpy as np
import casadi as ca
import opensim as osim

from osimfit.bounds import Bounds
from osimfit.costs import BilevelCost, CostInput, TrackingCost
from osimfit.model import BodyScaleGroup, MarkerOffsetGroup, ModelCache
from osimfit.problem import (ProblemRep, WorkerJar, PooledTrackingRep,
                             replicate_model_cache)
from tests.test_costs import marker_trial
from tests.test_double_pendulum import (create_double_pendulum,
                                        create_synthetic_markers_file)


############
# FIXTURES #
############

@pytest.fixture(scope='module')
def trial(tmp_path_factory):
    """A short two-marker trial, synthesized once for the whole module."""
    trc = tmp_path_factory.mktemp('problem') / 'trial.trc'
    create_synthetic_markers_file(str(trc), 1.25, 0.75, duration=0.2)
    return marker_trial('t0', str(trc))


@pytest.fixture
def model():
    return create_double_pendulum(1.0, 1.0)


def scaled_cache(model):
    """A ModelCache with one body-scale group registered."""
    mc = ModelCache(model)
    mc.add_parameter_group(BodyScaleGroup(['/bodyset/b0'], [1]))
    return mc


def instrument(jar):
    """
    Wrap `jar.borrow` to record how many workers are held at once and by how many OS
    threads, so tests can assert that threading actually engages and stays capped.
    """
    record = {'threads': set(), 'live': 0, 'max_live': 0, 'lock': threading.Lock()}
    original = jar.borrow

    @contextlib.contextmanager
    def borrow():
        with original() as worker:
            with record['lock']:
                record['live'] += 1
                record['max_live'] = max(record['max_live'], record['live'])
                record['threads'].add(threading.get_ident())
            try:
                yield worker
            finally:
                with record['lock']:
                    record['live'] -= 1

    jar.borrow = borrow
    return record


########################
# MODEL CACHE REPLICAS #
########################

def test_replicate_model_cache_shares_no_model_state(model):
    """
    A replica must share no mutable OpenSim object with its source, since the whole
    point is to let two threads realize two states concurrently.
    """
    primary = scaled_cache(model)
    replica = replicate_model_cache(primary)

    assert replica.model is not primary.model
    assert replica.state is not primary.state
    assert replica.state.getNQ() == primary.state.getNQ()
    assert replica.coordinate_q_indexes == primary.coordinate_q_indexes


def test_replicate_model_cache_reregisters_parameter_groups(model):
    """
    Parameter groups are model-independent descriptors, so the replica carries the same
    groups while deriving its own per-model joints and baselines.
    """
    primary = scaled_cache(model)
    replica = replicate_model_cache(primary)

    assert replica.body_scale_groups == primary.body_scale_groups
    # The cached joints are the replica's own, not the primary's.
    assert replica.body_scale_group_outboard_joints
    primary_joints = {id(j) for group in primary.body_scale_group_outboard_joints
                      for j in group}
    replica_joints = {id(j) for group in replica.body_scale_group_outboard_joints
                      for j in group}
    assert primary_joints.isdisjoint(replica_joints)


##############
# WORKER JAR #
##############

def test_worker_jar_rejects_an_empty_pool():
    with pytest.raises(ValueError, match='at least one worker'):
        WorkerJar([])


def test_worker_jar_borrow_returns_the_worker_afterwards():
    jar = WorkerJar(['a'])
    with jar.borrow() as worker:
        assert worker == 'a'
    # Borrowable again, so it was returned to the pool.
    with jar.borrow() as worker:
        assert worker == 'a'


def test_worker_jar_returns_the_worker_even_when_the_body_raises():
    """
    A failed evaluation must not leak a worker, or the pool would drain and every
    subsequent borrow would block forever.
    """
    jar = WorkerJar(['a'])
    with pytest.raises(RuntimeError):
        with jar.borrow():
            raise RuntimeError('boom')
    with jar.borrow() as worker:
        assert worker == 'a'


def test_worker_jar_primary_is_the_first_worker():
    jar = WorkerJar(['a', 'b', 'c'])
    assert jar.primary == 'a'
    assert jar.num_workers == 3


###############
# PROBLEM REP #
###############

def test_problem_rep_rejects_a_non_positive_thread_count(model, trial):
    with pytest.raises(ValueError, match='at least 1'):
        ProblemRep(ModelCache(model), [trial], [], TrackingCost(), num_threads=0)


def test_problem_rep_builds_one_tracking_rep_per_trial(model, trial):
    """
    The whole point of the rep: one callback per trial rather than one per time sample.
    """
    rep = ProblemRep(ModelCache(model), [trial], [], TrackingCost())

    assert set(rep.tracking_reps) == {'t0'}
    assert isinstance(rep.tracking_reps['t0'], PooledTrackingRep)
    assert trial.num_times > 1


def test_problem_rep_single_thread_adds_no_model_copies(model, trial):
    """
    A single-threaded problem must reuse the solver's own ModelCache, so the default
    path pays nothing for the jar.
    """
    mc = ModelCache(model)
    rep = ProblemRep(mc, [trial], [], TrackingCost(), num_threads=1)

    jar = rep.jars['t0']
    assert jar.num_workers == 1
    assert jar.primary.mc is mc


@pytest.mark.parametrize('num_threads', [2, 4])
def test_problem_rep_gives_each_worker_its_own_model(model, trial, num_threads):
    mc = ModelCache(model)
    rep = ProblemRep(mc, [trial], [], TrackingCost(), num_threads=num_threads)

    jar = rep.jars['t0']
    assert jar.num_workers == num_threads
    assert jar.primary.mc is mc
    models = [worker.mc.model for worker in jar.workers]
    assert len({id(m) for m in models}) == num_threads


def test_build_reference_is_shaped_one_column_per_sample(model, trial):
    rep = ProblemRep(ModelCache(model), [trial], [], TrackingCost())

    reference = rep.reference['t0']
    num_markers = len(trial.marker_data[0].labels)
    assert reference.shape == (3 * num_markers, trial.num_times)
    assert rep.tracking_reps['t0'].reference_size == 3 * num_markers


def test_build_reference_columns_match_the_trial_tables(model, trial):
    """
    Column `i` must hold sample `i`'s data, in per-marker contiguous XYZ triplets.
    """
    rep = ProblemRep(ModelCache(model), [trial], [], TrackingCost())
    reference = np.array(rep.reference['t0'])

    data = trial.marker_data[0]
    for itime in (0, trial.num_times // 2, trial.num_times - 1):
        for imarker in range(len(data.labels)):
            expected = data.positions.getRowAtIndex(itime).getElt(0, imarker).to_numpy()
            got = reference[3 * imarker:3 * imarker + 3, itime]
            np.testing.assert_allclose(got, expected, rtol=0, atol=0)


###############################
# EQUIVALENCE WITH PER-SAMPLE #
###############################

def per_sample_errors(cost, mc, trial, q, **inputs):
    """
    Evaluate `cost` the previous way: one rep per time sample, each carrying that
    sample's reference data. Reps are returned so CasADi's references stay valid.
    """
    reps, errors = [], []
    for itime in range(trial.num_times):
        rep = cost.create_rep(f'per_sample_{itime}', mc, trial, itime)
        reps.append(rep)
        errors.append(float(rep(CostInput(coordinates=ca.DM(q[:, itime]), **inputs))))
    return np.array(errors), reps


@pytest.mark.parametrize('num_threads', [1, 2, 4])
def test_mapped_tracking_matches_per_sample_reps(model, trial, num_threads):
    """
    One rep per trial, mapped over samples, must reproduce one-rep-per-sample exactly.
    Reference data is the only thing that varied between the old per-sample reps, so
    supplying it as an input must be an identity-preserving change.
    """
    cost = TrackingCost(position_weight=2.0, orientation_weight=1.5)
    num_coords = len(ModelCache(model).coordinate_q_indexes)
    q = np.random.default_rng(0).uniform(-0.3, 0.3, (num_coords, trial.num_times))

    expected, _reps = per_sample_errors(cost, ModelCache(model), trial, q)

    rep = ProblemRep(ModelCache(model), [trial], [], cost, num_threads=num_threads)
    coordinates = ca.MX.sym('q', num_coords, trial.num_times)
    evaluate = ca.Function(
        'evaluate', [coordinates],
        [rep.evaluate_trial('t0', coordinates, CostInput())])
    got = np.array(evaluate(ca.DM(q))).ravel()

    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize('num_threads', [1, 4])
def test_mapped_bilevel_tracking_matches_per_sample_reps(model, trial, num_threads):
    """
    The same equivalence for the bilevel cost, whose callback also takes the shared
    parameter blocks that `evaluate_trial` broadcasts across the map.
    """
    cost = BilevelCost(position_weight=2.0, orientation_weight=1.5)
    body_scales = ca.DM([1.1, 0.95, 1.05])
    num_coords = len(scaled_cache(model).coordinate_q_indexes)
    q = np.random.default_rng(1).uniform(-0.3, 0.3, (num_coords, trial.num_times))

    expected, _reps = per_sample_errors(cost, scaled_cache(model), trial, q,
                                        body_scales=body_scales)

    rep = ProblemRep(scaled_cache(model), [trial], [], cost, num_threads=num_threads)
    coordinates = ca.MX.sym('q', num_coords, trial.num_times)
    scales = ca.MX.sym('s', 3)
    evaluate = ca.Function(
        'evaluate', [coordinates, scales],
        [rep.evaluate_trial('t0', coordinates, CostInput(body_scales=scales))])
    got = np.array(evaluate(ca.DM(q), body_scales)).ravel()

    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_mapped_tracking_gradient_matches_finite_differences(model, trial):
    """
    The analytic Jacobian must survive being mapped: CasADi differentiates the map by
    mapping the callback's own Jacobian, and the reference input must contribute
    nothing.
    """
    cost = TrackingCost()
    num_coords = len(ModelCache(model).coordinate_q_indexes)
    num_times = trial.num_times
    q = np.random.default_rng(2).uniform(-0.3, 0.3, (num_coords, num_times))

    rep = ProblemRep(ModelCache(model), [trial], [], cost, num_threads=2)
    coordinates = ca.MX.sym('q', num_coords, num_times)
    total = ca.sum2(rep.evaluate_trial('t0', coordinates, CostInput()))
    value = ca.Function('value', [coordinates], [total])
    gradient = ca.Function('gradient', [coordinates],
                           [ca.gradient(total, ca.vec(coordinates))])

    analytic = np.array(gradient(ca.DM(q))).ravel()
    # ca.vec is column-major, so perturb the column-major flattening to match.
    flat = q.ravel(order='F')
    step = 1e-6
    numeric = []
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += step
        minus[i] -= step
        numeric.append(
            (float(value(plus.reshape(num_coords, num_times, order='F'))) -
             float(value(minus.reshape(num_coords, num_times, order='F')))) /
            (2.0 * step))

    np.testing.assert_allclose(analytic, np.array(numeric), rtol=1e-5, atol=1e-7)


#############
# THREADING #
#############

def test_single_threaded_evaluation_uses_one_thread(model, trial):
    rep = ProblemRep(ModelCache(model), [trial], [], TrackingCost(), num_threads=1)
    record = instrument(rep.jars['t0'])

    coordinates = ca.DM.zeros(len(rep.mc.coordinate_q_indexes), trial.num_times)
    symbols = ca.MX.sym('q', *coordinates.shape)
    ca.Function('f', [symbols],
                [rep.evaluate_trial('t0', symbols, CostInput())])(coordinates)

    assert record['max_live'] == 1
    assert len(record['threads']) == 1


@pytest.mark.parametrize('num_threads', [2, 4])
def test_threaded_evaluation_is_capped_at_num_threads(model, trial, num_threads):
    """
    CasADi's `reduce_in` map overload spawns one OS thread per map column and silently
    ignores a thread cap, which would mean one thread per time sample. `tracking_map`
    therefore uses the capped overload, and this pins that down: concurrency must equal
    `num_threads`, not `trial.num_times`.
    """
    assert trial.num_times > num_threads

    rep = ProblemRep(ModelCache(model), [trial], [], TrackingCost(),
                     num_threads=num_threads)
    record = instrument(rep.jars['t0'])

    coordinates = ca.DM.zeros(len(rep.mc.coordinate_q_indexes), trial.num_times)
    symbols = ca.MX.sym('q', *coordinates.shape)
    ca.Function('f', [symbols],
                [rep.evaluate_trial('t0', symbols, CostInput())])(coordinates)

    assert record['max_live'] <= num_threads
    assert len(record['threads']) <= num_threads


##############
# VALIDATION #
##############

def test_assert_offset_groups_used_accepts_a_tracked_group(model, trial):
    mc = ModelCache(model)
    mc.add_parameter_group(MarkerOffsetGroup(['/markerset/m0'], [1]))
    rep = ProblemRep(mc, [trial], [], BilevelCost())

    rep.assert_offset_groups_used()


def test_assert_offset_groups_used_rejects_an_untracked_group(model, trial):
    """
    An offset group no task tracks is unconstrained in the NLP, so the rep must refuse
    it rather than let the solve wander.
    """
    mc = ModelCache(model)
    mc.add_parameter_group(MarkerOffsetGroup(['/markerset/m_untracked'], [1]))
    rep = ProblemRep(mc, [trial], [], BilevelCost())

    with pytest.raises(ValueError, match='not tracked by any registered marker'):
        rep.assert_offset_groups_used()


####################
# REGISTERED COSTS #
####################

def test_problem_rep_splits_costs_by_whether_they_read_coordinates(model, trial):
    """
    Costs reading `coordinates` are evaluated at every time sample; the rest are
    evaluated once on the parameters shared across trials.
    """
    from osimfit.costs import BodyScaleRegularizationCost, CoordinateStiffnessCost

    coordinate_cost = CoordinateStiffnessCost({'/jointset/j0/q0': 1.0})
    parameter_cost = BodyScaleRegularizationCost(weight=1.0)
    rep = ProblemRep(scaled_cache(model), [trial],
                     [coordinate_cost, parameter_cost], TrackingCost())

    assert len(rep.coordinate_cost_reps) == 1
    assert len(rep.parameter_cost_reps) == 1
