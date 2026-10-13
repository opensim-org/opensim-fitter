"""
Tests for `Cost`s: the cost classes a user registers on a solver, and the
solver-side validation of what each solver accepts.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim

from osimfit.model import ModelCache, BodyScale, StationCache, BeamLengthScale
from osimfit.bounds import Bounds
from osimfit.solvers import InverseKinematicsSolver, SplinedKinematicsSolver
from osimfit.costs import (AnthropometricRegularizationCostRep, CostInput, CostRep,
                           SymbolicCost, SymbolicCostRep, BodyScaleRegularizationCost,
                           BodyScaleIsotropyCost, OffsetRegularizationCost,
                           TrackingCost, AnthropometricRegularizationCost,
                           CoordinateStiffnessCost,
                           EllipsoidRadiiScaleRegularizationCost,
                           BeamLengthScaleRegularizationCost)
from osimfit.scaling import Axis, AnthropometricMeasurement
from osimfit.data_sources import MarkerSource, Trial
from tests.test_double_pendulum import create_double_pendulum

# Baseline mobilizer geometry of the beam test model.
ELBOW = '/jointset/elbow_r'
BEAM_LENGTH = 0.35

###########
# HELPERS #
###########

def create_beam_model():
    """
    A chain in which the beam length is strongly identifiable:

        ground --Pin-- torso --CantileverFreeBeam-- forearm

    Three markers spread across the forearm pin down the beam's endpoint pose, and one
    on the torso pins down the upstream pin rotation. No three-rotation joint upstream
    could mimic a change in beam length.
    """
    model = osim.Model()
    model.setName('beam')

    torso = osim.Body('torso', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    forearm = osim.Body('forearm', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    for body in (torso, forearm):
        model.addBody(body)

    model.addJoint(osim.PinJoint('ground_torso', model.getGround(), torso))
    model.addJoint(osim.CantileverFreeBeamJoint(
        'elbow_r', torso, osim.Vec3(0.05, -0.1, 0.02), osim.Vec3(0.1, 0.1, 0.1),
        forearm, osim.Vec3(0.0, 0.01, 0.0), osim.Vec3(-0.2, 0.1, 0.05),
        BEAM_LENGTH))

    model.addMarker(osim.Marker('torso_marker', torso, osim.Vec3(0.1, 0.05, 0.0)))
    for i, location in enumerate([osim.Vec3(0.15, 0.0, 0.0), osim.Vec3(0.0, 0.15, 0.0),
                                  osim.Vec3(0.0, 0.0, 0.15)]):
        model.addMarker(osim.Marker(f'forearm_{i}', forearm, location))

    model.finalizeConnections()
    return model


def create_prescribed_markers(model: osim.Model, trc_path: str,
                              num_times: int = 40, freq: float = 0.5) -> None:
    """
    Write marker positions for a smooth, prescribed coordinate trajectory to a TRC
    file. A prescribed trajectory is used rather than a forward simulation so the
    resulting fit is well conditioned and cheap to solve.
    """
    state = model.initSystem()
    coordinates = [model.getCoordinateSet().get(i)
                   for i in range(model.getCoordinateSet().getSize())]
    markers = [model.getMarkerSet().get(i)
               for i in range(model.getMarkerSet().getSize())]

    times = np.linspace(0.0, 1.0, num_times)
    table = osim.TimeSeriesTableVec3()
    for time in times:
        for icoord, coordinate in enumerate(coordinates):
            coordinate.setValue(
                state,
                0.3 * np.sin(2.0 * np.pi * freq * time + 0.7 * icoord), False)
        model.assemble(state)
        model.realizePosition(state)
        row = osim.RowVectorVec3(len(markers))
        for imarker, marker in enumerate(markers):
            location = marker.getLocationInGround(state)
            for i in range(3):
                row.updElt(0, imarker).set(i, location[i])
        table.appendRow(time, row)

    table.setColumnLabels([m.getAbsolutePathString() for m in markers])
    table.addTableMetaDataString('DataRate', str(num_times))
    table.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter().write(table, trc_path)


def getP_BM(model: osim.Model, joint_index: int, state: osim.State):
    """
    Return the position of the child frame from the Joint at index `joint_index`.
    """
    return model.getJointSet().get(joint_index).getOutboardFrame(state).p().to_numpy()


def marker_trial(name: str, trc_path: str):
    """
    A `Trial` over the markers in `trc_path`, mapping the labels the TRC adapter writes
    back to the component paths the solver resolves.
    """
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}
    return Trial(name, [MarkerSource('markers', trc_path, label_map=label_map)])


##############
# VALIDATION #
##############

class CoordinatePenalty(SymbolicCost):
    """A minimal cost that depends only on the coordinates."""
    required_inputs = frozenset({'coordinates'})

    def evaluate(self, input: CostInput) -> ca.MX:
        return ca.sumsqr(input.coordinates)


@pytest.fixture
def double_pendulum_model():
    m = create_double_pendulum(1.0, 1.0)
    m.initSystem()
    return m

def test_inverse_kinematics_accepts_coordinate_cost(double_pendulum_model):
    solver = InverseKinematicsSolver(double_pendulum_model)
    solver.add_cost(CoordinatePenalty())
    assert len(solver.costs) == 1


def test_inverse_kinematics_rejects_parameter_cost(double_pendulum_model):
    solver = InverseKinematicsSolver(double_pendulum_model)
    with pytest.raises(ValueError, match='body_scales'):
        solver.add_cost(BodyScaleRegularizationCost(1.0))
    assert solver.costs == []


def test_splined_accepts_parameter_costs(double_pendulum_model):
    solver = SplinedKinematicsSolver(double_pendulum_model)
    solver.add_cost(BodyScaleRegularizationCost(1.0))
    solver.add_cost(OffsetRegularizationCost(1.0))
    assert len(solver.costs) == 2


def test_splined_accepts_coordinate_cost(double_pendulum_model):
    """
    The splined solver provides the coordinates as well as the shared parameters, so a
    coordinate-dependent cost is registrable; it is evaluated at every time sample
    rather than once per solve.
    """
    solver = SplinedKinematicsSolver(double_pendulum_model)
    solver.add_cost(CoordinatePenalty())
    assert len(solver.costs) == 1


def test_registered_cost_reps_size_themselves_from_the_solvers_parameters():
    """
    A registered cost's rep is built from the solver's own ModelCache, so its declared
    CasADi input size follows the parameters registered on that solver: an
    AnthropometricRegularizationCostRep declares 3 * len(mc.body_scale_groups), which is
    only final once add_parameter() has been called for every body scale.
    """
    solver = SplinedKinematicsSolver(create_two_link_model())
    cost = AnthropometricRegularizationCost(
        [AnthropometricMeasurement('stature', '/S0', '/S1', Axis.YAxis)],
        sex='female')
    solver.add_cost(cost)
    for body_path in ('/bodyset/b0', '/bodyset/b1'):
        solver.add_parameter(BodyScale(body_path, Bounds(0.5, 2.0), np.ones(3)))

    rep = cost.create_rep(solver.mc)

    assert isinstance(rep, AnthropometricRegularizationCostRep)
    assert rep.mc is solver.mc
    num_scales = 3 * len(solver.mc.body_scale_groups)
    assert num_scales == 6
    assert rep.size1_in(0) == num_scales

    # Every solve builds a fresh rep rather than reusing the previous solve's.
    assert cost.create_rep(solver.mc) is not rep


def test_add_cost_rejects_a_tracking_cost(double_pendulum_model):
    """
    The registered-cost loop builds reps from the ModelCache alone, so a
    TrackingCostBase (which a solver is meant to construct itself, per trial and time
    sample) is not a Cost and cannot be registered.
    """
    solver = InverseKinematicsSolver(double_pendulum_model)
    with pytest.raises(TypeError, match='expects a Cost'):
        solver.add_cost(TrackingCost())
    assert solver.costs == []


def test_add_cost_rejects_a_second_cost_of_the_same_type(double_pendulum_model):
    """
    A cost's type labels its term in the objective breakdown, so registering two of a
    type would leave two terms fighting over one label.
    """
    solver = InverseKinematicsSolver(double_pendulum_model)
    solver.add_cost(CoordinateStiffnessCost({'/jointset/j0/q0': 1.0}))
    with pytest.raises(ValueError, match='already registered'):
        solver.add_cost(CoordinateStiffnessCost({'/jointset/j1/q1': 1.0}))
    assert len(solver.costs) == 1


def test_add_cost_accepts_a_subclass_of_a_registered_cost(double_pendulum_model):
    """
    The check is on the exact type, not `isinstance`: a subclass carries its own class
    name, and so its own label, and may be registered alongside its base.
    """
    class CoordinateStiffnessSubclass(CoordinateStiffnessCost):
        pass

    solver = InverseKinematicsSolver(double_pendulum_model)
    solver.add_cost(CoordinateStiffnessCost({'/jointset/j0/q0': 1.0}))
    solver.add_cost(CoordinateStiffnessSubclass({'/jointset/j0/q0': 1.0}))
    assert len(solver.costs) == 2


############################
# BODY SCALE ISOTROPY COST #
############################

# The isotropy cost is a `SymbolicCost`, so it is a pure function of the flat
# `body_scales` input and needs no model. `body_scales` is triplet-contiguous, i.e.
# [s0x, s0y, s0z, s1x, s1y, s1z, ...].
def _isotropy_cost(scales, weight=1.0):
    cost = BodyScaleIsotropyCost(weight=weight)
    return float(cost.evaluate(
        CostInput(body_scales=ca.DM(np.asarray(scales, dtype=float)))))


def _manual_isotropy_cost(scales, weight=1.0):
    """
    Independent numpy evaluation: each group's three factors are penalized against
    their own mean.
    """
    groups = np.asarray(scales, dtype=float).reshape(-1, 3)
    return weight * float(np.sum((groups - groups.mean(axis=1, keepdims=True))**2))


def test_isotropy_cost_requires_body_scales():
    assert (BodyScaleIsotropyCost(weight=1.0).required_inputs ==
            frozenset({'body_scales'}))


def test_isotropy_cost_rejects_negative_weight():
    with pytest.raises(ValueError, match='non-negative'):
        BodyScaleIsotropyCost(weight=-1.0)


def test_isotropy_cost_is_zero_for_identity_scaling():
    assert _isotropy_cost([1.0] * 6) == pytest.approx(0.0, abs=1e-12)


def test_isotropy_cost_is_zero_for_uniform_non_identity_scaling():
    # The cost constrains shape, not size: a group scaled equally along X, Y, and Z is
    # free no matter how far from 1.0 it is.
    assert _isotropy_cost([1.2, 1.2, 1.2, 0.7, 0.7, 0.7]) == pytest.approx(
        0.0, abs=1e-12)


def test_isotropy_cost_penalizes_anisotropic_group():
    # scales [1.0, 1.0, 1.3] -> mean 1.1, deviations [-0.1, -0.1, 0.2].
    assert _isotropy_cost([1.0, 1.0, 1.3]) == pytest.approx(0.06, abs=1e-12)


def test_isotropy_cost_sums_over_groups_independently():
    # Group 0 is anisotropic, group 1 is uniform, so only group 0 contributes. This also
    # pins down the triplet-contiguous layout: a group's three axes are adjacent.
    scales = [1.0, 1.0, 1.3, 0.8, 0.8, 0.8]
    assert _isotropy_cost(scales) == pytest.approx(0.06, abs=1e-12)
    assert _isotropy_cost(scales) == pytest.approx(
        _manual_isotropy_cost(scales), abs=1e-12)


def test_isotropy_cost_matches_manual_evaluation():
    rng = np.random.default_rng(0)
    for num_groups in (1, 2, 5):
        scales = 0.5 + rng.random(3 * num_groups)
        assert _isotropy_cost(scales, weight=0.25) == pytest.approx(
            _manual_isotropy_cost(scales, weight=0.25), abs=1e-12)


def test_isotropy_cost_scales_linearly_with_weight():
    scales = [1.0, 1.0, 1.3, 0.9, 1.1, 1.0]
    unit = _isotropy_cost(scales, weight=1.0)
    assert unit > 0.0
    assert _isotropy_cost(scales, weight=3.5) == pytest.approx(3.5 * unit, abs=1e-12)


def test_isotropy_cost_is_invariant_to_axis_permutation():
    # Deviations from the group mean do not depend on which axis is the odd one out.
    assert _isotropy_cost([1.3, 1.0, 1.0]) == pytest.approx(
        _isotropy_cost([1.0, 1.0, 1.3]), abs=1e-12)


def test_isotropy_cost_rejects_body_scales_not_a_multiple_of_three():
    with pytest.raises(ValueError, match='multiple of 3'):
        _isotropy_cost([1.0, 1.0, 1.0, 1.0])


def test_isotropy_cost_gradient_is_symbolic_and_matches_finite_difference():
    # Being a SymbolicCost, it must differentiate through CasADi without a callback.
    scales = np.array([1.0, 1.0, 1.3, 0.9, 1.1, 1.0])
    x = ca.MX.sym('body_scales', scales.size)
    expr = BodyScaleIsotropyCost(weight=2.0).evaluate(CostInput(body_scales=x))
    grad = ca.Function('grad', [x], [ca.gradient(expr, x)])

    analytic = np.asarray(grad(ca.DM(scales))).reshape(-1)
    eps = 1e-7
    numeric = np.zeros_like(scales)
    for k in range(scales.size):
        forward, backward = scales.copy(), scales.copy()
        forward[k] += eps
        backward[k] -= eps
        numeric[k] = (_isotropy_cost(forward, weight=2.0) -
                      _isotropy_cost(backward, weight=2.0)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


######################################
# ANTHROPOMETRIC REGULARIZATION COST #
######################################

def create_two_link_model():
    """
    A two-link pin chain with a station on each body at a non-zero body-frame offset, so
    both mobilizer-frame scaling (chain) and station-offset scaling contribute to a
    measurement spanning the two bodies.
    """
    model = osim.Model()
    model.setName('two_link')
    ground = model.getGround()

    b0 = osim.Body('b0', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b0)
    j0 = osim.PinJoint('j0', ground, osim.Vec3(0), osim.Vec3(0),
                       b0, osim.Vec3(0, -0.5, 0), osim.Vec3(0))
    model.addJoint(j0)

    b1 = osim.Body('b1', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b1)
    j1 = osim.PinJoint('j1', b0, osim.Vec3(0), osim.Vec3(0),
                       b1, osim.Vec3(0, -0.5, 0), osim.Vec3(0))
    model.addJoint(j1)

    s0 = osim.Station(b0, osim.Vec3(0.1, 0.2, 0.0))
    s0.setName('S0')
    model.addComponent(s0)
    s1 = osim.Station(b1, osim.Vec3(0.3, 0.0, 0.0))
    s1.setName('S1')
    model.addComponent(s1)

    model.finalizeConnections()
    return model


def register_body_scales(mc, body_paths):
    """Register one BodyScale group per body on the ModelCache, in order."""
    for path in body_paths:
        bs = BodyScale(path, Bounds(0.5, 2.0), np.ones(3))
        bs.validate(mc)
        mc.add_parameter_group(bs.to_group())


def station_ground_under_scale(mc, station_path, s):
    """
    Station ground position under the solver's scaling model for flat scales `s`:
    scaled mobilizer frames (set_scaled_mobilizer_frame_positions) plus the station's
    own base-frame location scaled by its body's group scale.
    """
    station = osim.Station.safeDownCast(mc.model.getComponent(station_path))
    base_frame = osim.PhysicalFrame.safeDownCast(
        station.getParentFrame().findBaseFrame())
    mobod = int(base_frame.getMobilizedBodyIndex())
    base_loc = station.findLocationInFrame(mc.state, base_frame).to_numpy()

    group = next((g for g, grp in enumerate(mc.body_scale_groups)
                  if mobod in [int(k) for k in grp.mobod_indexes]), None)
    scaled_loc = base_loc.copy()
    if group is not None:
        scaled_loc = base_loc * np.asarray(s[3*group:3*group+3])

    mc.set_scaled_mobilizer_frame_positions(mc.state, np.asarray(s, dtype=float))
    mc.model.realizePosition(mc.state)
    p = base_frame.findStationLocationInGround(
        mc.state, osim.Vec3(*[float(v) for v in scaled_loc])).to_numpy()
    return p

# The cost fits its distribution from the ANSUR II dataset, so measurement names must be
# real ANSUR labels. The station pairs are the synthetic model's stations — anatomical
# correctness is irrelevant here; we exercise the cost mechanics.
def _build_rep(label='stature', axis=Axis.YAxis, sex='female', weight=1.0):
    """
    Build an AnthropometricRegularizationCost and its rep, as a solver does in solve().
    """
    model = create_two_link_model()
    mc = ModelCache(model)
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    measurements = [AnthropometricMeasurement(label, '/S0', '/S1', axis)]
    cost = AnthropometricRegularizationCost(measurements, sex=sex, weight=weight)
    rep = cost.create_rep(mc)
    n = 3 * len(mc.body_scale_groups)
    return rep, n


def _manual_cost(rep, s):
    """
    Independent numpy evaluation of the Mahalanobis penalty: measurements are recomputed
    from explicit station ground positions under the scaling model (bypassing the rep's
    own callback), then combined with its cost's fitted mean and precision.
    """
    s = np.asarray(s, dtype=float)
    cost = rep.cost
    measurements = []
    for (sc1, sc2, axis), m in zip(rep.station_caches, cost.measurements):
        d = (station_ground_under_scale(sc2.mc, m.station2_path, s) -
             station_ground_under_scale(sc1.mc, m.station1_path, s))
        measurements.append(np.abs(d[axis]) if axis is not None else np.linalg.norm(d))
    residual = np.asarray(measurements) - cost.mean
    return cost.weight * 0.5 * residual @ cost.precision @ residual


def test_station_position_jacobian_matches_finite_difference():
    model = create_two_link_model()
    mc = ModelCache(model)
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    mc.model.realizePosition(mc.state)

    n = 3 * len(mc.body_scale_groups)
    for station_path in ('/S0', '/S1'):
        station = osim.Station.safeDownCast(mc.model.getComponent(station_path))
        station_cache = StationCache.from_station(mc, station)

        # Analytical Jacobian (computed at baseline before any state scaling).
        J = station_cache.calc_position_jacobian_wrt_body_scales(mc.state)

        # Finite-difference the same scaling model.
        p0 = station_ground_under_scale(mc, station_path, np.ones(n))
        eps = 1e-6
        J_fd = np.zeros((3, n))
        for k in range(n):
            s = np.ones(n)
            s[k] += eps
            pk = station_ground_under_scale(mc, station_path, s)
            J_fd[:, k] = (pk - p0) / eps

        np.testing.assert_allclose(J, J_fd, atol=1e-5,
                                   err_msg=f'Jacobian mismatch for {station_path}')


def test_distribution_mean_is_in_meters():
    # A female stature is ~1.6 m; without the mm->m conversion it would be ~1600.
    rep, n = _build_rep(label='stature')
    assert 1.0 < rep.cost.mean[0] < 2.5


def test_cost_matches_manual_mahalanobis():
    rep, n = _build_rep(label='stature', weight=2.0)
    for s in (np.ones(n), np.array([1.1, 1.0, 1.0, 0.9, 1.0, 1.0])):
        value = float(rep(CostInput(body_scales=ca.DM(s))))
        np.testing.assert_allclose(value, _manual_cost(rep, s), rtol=1e-9)


def test_cost_gradient_matches_finite_difference():
    rep, n = _build_rep(label='stature')
    s = ca.MX.sym('s', n)
    grad = ca.Function('grad', [s], [ca.gradient(rep(CostInput(body_scales=s)), s)])
    s0 = np.ones(n)
    g = np.array(grad(s0)).flatten()
    eps = 1e-6
    g_fd = np.zeros(n)
    for k in range(n):
        sp, sm = s0.copy(), s0.copy()
        sp[k] += eps
        sm[k] -= eps
        g_fd[k] = (float(rep(CostInput(body_scales=ca.DM(sp)))) -
                   float(rep(CostInput(body_scales=ca.DM(sm))))) / (2 * eps)
    np.testing.assert_allclose(g, g_fd, atol=1e-6)


def test_euclidean_measurement_builds_and_evaluates():
    rep, n = _build_rep(label='biacromialbreadth', axis=None)
    value = float(rep(CostInput(body_scales=ca.DM(np.ones(n)))))
    assert np.isfinite(value)


def test_cost_is_a_description_and_only_its_rep_is_callable():
    """
    A Cost carries no model state and is not evaluable; the rep it creates is.
    """
    measurements = [AnthropometricMeasurement('stature', '/S0', '/S1', Axis.YAxis)]
    cost = AnthropometricRegularizationCost(measurements, sex='female')
    assert not isinstance(cost, CostRep)
    assert not callable(cost)

    mc = ModelCache(create_two_link_model())
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    rep = cost.create_rep(mc)
    assert isinstance(rep, CostRep)
    assert rep.cost is cost
    assert rep.mc is mc


def test_creating_a_rep_twice_yields_independent_reps():
    """
    A solver builds a fresh rep on every solve, so a Cost must support create_rep more
    than once. A CasADi callback can only be constructed once per proxy, so each rep
    has to be a distinct object rather than the cost itself.
    """
    measurements = [AnthropometricMeasurement('stature', '/S0', '/S1', Axis.YAxis)]
    cost = AnthropometricRegularizationCost(measurements, sex='female', weight=2.0)

    caches, reps = [], []
    for _ in range(2):
        mc = ModelCache(create_two_link_model())
        register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
        caches.append(mc)
        reps.append(cost.create_rep(mc))

    assert reps[0] is not reps[1]
    assert [rep.mc for rep in reps] == caches
    assert all(rep.cost is cost for rep in reps)

    # The models are identical copies, so both reps evaluate to the same value, and
    # each agrees with an independent numpy evaluation through its own caches.
    s = np.array([1.1, 1.0, 1.0, 0.9, 1.0, 1.0])
    values = [float(rep(CostInput(body_scales=ca.DM(s)))) for rep in reps]
    np.testing.assert_allclose(values[0], values[1], rtol=1e-9)
    for rep, value in zip(reps, values):
        np.testing.assert_allclose(value, _manual_cost(rep, s), rtol=1e-9)


def test_symbolic_cost_rep_delegates_to_its_cost():
    cost = BodyScaleRegularizationCost(2.0, target=1.1)
    mc = ModelCache(create_two_link_model())
    rep = cost.create_rep(mc)

    assert isinstance(rep, SymbolicCostRep)
    assert rep.cost is cost

    s = ca.DM([1.2, 0.9, 1.0])
    assert float(rep(CostInput(body_scales=s))) == pytest.approx(
        float(cost.evaluate(CostInput(body_scales=s))))


def test_body_scale_groups_may_be_shared_across_model_caches():
    model = create_two_link_model()
    mc1, mc2 = ModelCache(model), ModelCache(model)
    assert mc1.model is not mc2.model

    def snapshot(mc):
        """The per-ModelCache state that scaling a body scale group writes through."""
        return ([int(mc.outboard_write_mobods.getElt(i))
                 for i in range(mc.outboard_write_mobods.size())],
                [int(mc.inboard_write_mobods.getElt(i))
                 for i in range(mc.inboard_write_mobods.size())],
                mc.outboard_write_baselines.tolist(),
                mc.inboard_write_baselines.tolist(),
                mc.outboard_write_group_rows.tolist(),
                mc.inboard_write_group_rows.tolist())

    register_body_scales(mc1, ['/bodyset/b0', '/bodyset/b1'])
    groups = list(mc1.body_scale_groups)
    before = snapshot(mc1)
    assert before[0], 'expected at least one outboard write to pin'

    # Register the very same group objects on a second ModelCache.
    for group in groups:
        mc2.add_parameter_group(group)

    # Registering on mc2 must leave mc1's cache alone, and the two caches must hold
    # their own C++ containers, one set per model copy.
    assert snapshot(mc1) == before
    assert int(mc1.outboard_write_mobods.this) != int(mc2.outboard_write_mobods.this)
    assert int(mc1.outboard_write_rotations.this) != \
        int(mc2.outboard_write_rotations.this)

    # The shared group descriptors carry no model-specific state at all.
    for group in groups:
        assert not hasattr(group, 'outboard_joints')
        assert not hasattr(group, 'inboard_joints')


def test_set_scaled_mobilizer_frames_needs_no_explicit_caching():
    mc = ModelCache(create_two_link_model())
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    mc.set_scaled_mobilizer_frame_positions(
        mc.state, np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0]))

    # Each group body's outboard frame is its baseline [0, -0.5, 0] times that
    # group's Y scale.
    np.testing.assert_allclose(getP_BM(mc.model, 0, mc.state), [0.0, -1.0, 0.0])
    np.testing.assert_allclose(getP_BM(mc.model, 1, mc.state), [0.0, -1.5, 0.0])


#############################
# COORDINATE STIFFNESS COST #
#############################

Q0 = '/jointset/j0/q0'
Q1 = '/jointset/j1/q1'


@pytest.fixture
def pendulum():
    model = create_double_pendulum(1.0, 1.0)
    model.initSystem()
    return model


def test_stiffness_rejects_negative_weight():
    with pytest.raises(ValueError, match='non-negative'):
        CoordinateStiffnessCost({Q0: 1.0}, weight=-1.0)


def test_stiffness_rejects_negative_stiffness():
    with pytest.raises(ValueError, match='non-negative'):
        CoordinateStiffnessCost({Q0: -1.0})


def test_stiffness_rejects_empty_stiffnesses():
    with pytest.raises(ValueError, match='at least one'):
        CoordinateStiffnessCost({})


def test_stiffness_rejects_unknown_coordinate(pendulum):
    cost = CoordinateStiffnessCost({'/jointset/j0/nope': 1.0})
    with pytest.raises(ValueError, match='not a coordinate in the model'):
        cost.create_rep(ModelCache(pendulum))


def test_stiffness_requires_coordinates():
    assert CoordinateStiffnessCost({Q0: 1.0}).required_inputs == frozenset(
        {'coordinates'})


def test_stiffness_penalizes_only_the_named_coordinates(pendulum):
    mc = ModelCache(pendulum)
    rep = CoordinateStiffnessCost({Q1: 2.0}).create_rep(mc)
    order = list(mc.coordinate_q_map)

    # Moving the unpenalized coordinate leaves the cost at zero; moving the penalized
    # one gives stiffness * deviation^2.
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.5
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(0.0)

    q = np.zeros(len(order))
    q[order.index(Q1)] = 0.5
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(2.0 * 0.25)


def test_stiffness_weight_scales_the_penalty(pendulum):
    mc = ModelCache(pendulum)
    order = list(mc.coordinate_q_map)
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.3

    plain = CoordinateStiffnessCost({Q0: 1.0}).create_rep(mc)
    scaled = CoordinateStiffnessCost({Q0: 1.0}, weight=5.0).create_rep(mc)
    assert float(scaled(CostInput(coordinates=ca.DM(q)))) == pytest.approx(
        5.0 * float(plain(CostInput(coordinates=ca.DM(q)))))


def test_stiffness_target_defaults_to_the_model_default_value(pendulum):
    """
    A coordinate whose default value is non-zero is pulled toward that value, not
    toward zero, so the penalty vanishes at the model's neutral posture.
    """
    coordinate = osim.Coordinate.safeDownCast(pendulum.getComponent(Q0))
    coordinate.setDefaultValue(0.4)
    pendulum.finalizeConnections()

    mc = ModelCache(pendulum)
    rep = CoordinateStiffnessCost({Q0: 3.0}).create_rep(mc)
    order = list(mc.coordinate_q_map)

    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.4
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(0.0)

    q[order.index(Q0)] = 0.5
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(
        3.0 * (0.5 - 0.4)**2)


def test_stiffness_explicit_target_overrides_the_model_default(pendulum):
    mc = ModelCache(pendulum)
    rep = CoordinateStiffnessCost({Q0: 1.0}, targets={Q0: 0.25}).create_rep(mc)
    order = list(mc.coordinate_q_map)
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.25
    assert float(rep(CostInput(coordinates=ca.DM(q)))) == pytest.approx(0.0)


def test_stiffness_gradient_matches_the_analytic_spring(pendulum):
    """
    The cost is a plain CasADi expression, so CasADi differentiates it symbolically.
    Its gradient must be the spring gradient 2 * weight * k * (q - target).
    """
    mc = ModelCache(pendulum)
    order = list(mc.coordinate_q_map)
    rep = CoordinateStiffnessCost({Q0: 2.0, Q1: 0.5}, weight=3.0).create_rep(mc)

    x = ca.SX.sym('x', len(order))
    gradient = ca.Function('g', [x],
                           [ca.jacobian(rep(CostInput(coordinates=x)), x)])
    q = np.zeros(len(order))
    q[order.index(Q0)] = 0.3
    q[order.index(Q1)] = -0.2

    expected = np.zeros(len(order))
    expected[order.index(Q0)] = 2.0 * 3.0 * 2.0 * 0.3
    expected[order.index(Q1)] = 2.0 * 3.0 * 0.5 * -0.2
    np.testing.assert_allclose(np.squeeze(gradient(q).full()), expected, atol=1e-12)


def test_inverse_kinematics_accepts_a_stiffness(pendulum):
    solver = InverseKinematicsSolver(pendulum)
    solver.add_cost(CoordinateStiffnessCost({Q0: 1.0}))
    assert len(solver.costs) == 1


def test_splined_solver_accepts_a_stiffness(pendulum):
    solver = SplinedKinematicsSolver(pendulum)
    solver.add_cost(CoordinateStiffnessCost({Q0: 1.0}))
    assert len(solver.costs) == 1


def test_stiffness_pulls_an_underdetermined_coordinate_toward_its_target(pendulum):
    """
    Drive the objective with the stiffness alone and confirm the minimizer sits at the
    target: the penalty is what determines an otherwise unconstrained coordinate.
    """
    mc = ModelCache(pendulum)
    order = list(mc.coordinate_q_map)
    rep = CoordinateStiffnessCost({Q0: 1.0, Q1: 1.0},
                                  targets={Q0: 0.2, Q1: -0.3}).create_rep(mc)

    x = ca.SX.sym('x', len(order))
    nlp = {'x': x, 'f': rep(CostInput(coordinates=x))}
    solver = ca.nlpsol('solver', 'ipopt', nlp,
                       {'ipopt': {'print_level': 0}, 'print_time': False})
    optimal = np.squeeze(solver(x0=np.zeros(len(order)))['x'].full())

    assert optimal[order.index(Q0)] == pytest.approx(0.2, abs=1e-6)
    assert optimal[order.index(Q1)] == pytest.approx(-0.3, abs=1e-6)


def test_splined_solver_applies_the_stiffness_per_time_sample(tmp_path):
    """
    A coordinate-dependent cost is a distinct code path in `SplinedKinematicsSolver`:
    it is evaluated at every time sample rather than once per solve. Solve the same
    problem with and without a stiffness pulling one coordinate away from what the
    markers alone imply, and confirm the stiffened solution moves toward the target.
    """
    coordinate = '/jointset/elbow_r/elbow_r_coord_2'
    target = 0.4

    trc_path = str(tmp_path / 'markers.trc')
    create_prescribed_markers(create_beam_model(), trc_path)

    def solve(stiffness):
        model = create_beam_model()
        model.initSystem()
        solver = SplinedKinematicsSolver(model, convergence_tolerance=1e-6,
                                         knot_interval=0.1, position_weight=5.0)
        solver.add_trial(marker_trial('beam', trc_path))
        if stiffness is not None:
            solver.add_cost(CoordinateStiffnessCost({coordinate: stiffness},
                                                    targets={coordinate: target}))
        solution = solver.solve()
        column = solution.states_tables['beam'].getDependentColumn(
            coordinate + '/value').to_numpy()
        return float(np.mean(column))

    plain = solve(None)
    stiffened = solve(1e3)

    # The stiffened solve must sit closer to the target than the unstiffened one.
    assert abs(stiffened - target) < abs(plain - target)


##################################
# MOBILIZER REGULARIZATION COSTS #
##################################

@pytest.mark.parametrize('cost_cls,input_name', [
    (EllipsoidRadiiScaleRegularizationCost, 'ellipsoid_radii_scales'),
    (BeamLengthScaleRegularizationCost, 'beam_length_scales'),
])
def test_mobilizer_regularization_requires_only_its_own_input(cost_cls, input_name):
    assert cost_cls(1.0).required_inputs == frozenset({input_name})


@pytest.mark.parametrize('cost_cls', [EllipsoidRadiiScaleRegularizationCost,
                                      BeamLengthScaleRegularizationCost])
def test_mobilizer_regularization_rejects_a_negative_weight(cost_cls):
    with pytest.raises(ValueError, match='non-negative'):
        cost_cls(weight=-1.0)


@pytest.mark.parametrize('cost_cls,input_name', [
    (EllipsoidRadiiScaleRegularizationCost, 'ellipsoid_radii_scales'),
    (BeamLengthScaleRegularizationCost, 'beam_length_scales'),
])
def test_mobilizer_regularization_penalizes_deviation(cost_cls, input_name):
    cost = cost_cls(weight=2.0)
    factors = np.array([1.1, 0.9, 1.0])

    assert float(cost.evaluate(
        CostInput(**{input_name: np.ones(3)}))) == pytest.approx(0.0)
    assert float(cost.evaluate(
        CostInput(**{input_name: factors}))) == pytest.approx(
            2.0 * (0.1**2 + 0.1**2 + 0.0))


@pytest.mark.parametrize('cost_cls,input_name', [
    (EllipsoidRadiiScaleRegularizationCost, 'ellipsoid_radii_scales'),
    (BeamLengthScaleRegularizationCost, 'beam_length_scales'),
])
def test_mobilizer_regularization_honors_a_non_unit_target(cost_cls, input_name):
    cost = cost_cls(weight=1.0, target=1.5)
    assert float(cost.evaluate(
        CostInput(**{input_name: np.array([1.5])}))) == pytest.approx(0.0)


def test_mobilizer_regularization_ignores_the_other_parameter_type():
    """
    Splitting the costs is what makes each one's weight independent: the radii cost
    must not see the beam lengths, and vice versa.
    """
    radii = np.array([1.4, 0.6, 1.0])
    lengths = np.array([1.2])

    radii_cost = EllipsoidRadiiScaleRegularizationCost(weight=1.0)
    with_lengths = radii_cost.evaluate(
        CostInput(ellipsoid_radii_scales=radii, beam_length_scales=lengths))
    without_lengths = radii_cost.evaluate(CostInput(ellipsoid_radii_scales=radii))
    assert float(with_lengths) == pytest.approx(float(without_lengths))

    length_cost = BeamLengthScaleRegularizationCost(weight=1.0)
    with_radii = length_cost.evaluate(
        CostInput(ellipsoid_radii_scales=radii, beam_length_scales=lengths))
    without_radii = length_cost.evaluate(CostInput(beam_length_scales=lengths))
    assert float(with_radii) == pytest.approx(float(without_radii))


@pytest.mark.parametrize('cost_cls', [EllipsoidRadiiScaleRegularizationCost,
                                      BeamLengthScaleRegularizationCost])
def test_splined_solver_accepts_the_mobilizer_regularization_costs(cost_cls):
    solver = SplinedKinematicsSolver(create_beam_model())
    solver.add_cost(cost_cls(weight=1e-3))
    assert len(solver.costs) == 1


@pytest.mark.parametrize('cost_cls', [EllipsoidRadiiScaleRegularizationCost,
                                      BeamLengthScaleRegularizationCost])
def test_inverse_kinematics_rejects_the_mobilizer_regularization_costs(cost_cls):
    solver = InverseKinematicsSolver(create_beam_model())
    with pytest.raises(ValueError, match='does not support'):
        solver.add_cost(cost_cls(weight=1e-3))
    assert solver.costs == []


def test_beam_length_scale_regularization_holds_a_factor_at_its_target(tmp_path):
    """
    With marker data generated from the nominal model, a regularized solve must leave
    the factor at 1.0 rather than letting it absorb numerical slack.
    """
    trc_path = str(tmp_path / 'markers.trc')
    create_prescribed_markers(create_beam_model(), trc_path)

    model = create_beam_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(
        model, convergence_tolerance=1e-6, knot_interval=0.1, position_weight=5.0)
    solver.add_trial(marker_trial('beam', trc_path))
    solver.add_parameter(BeamLengthScale(ELBOW, Bounds(0.5, 2.0), 1.0))
    solver.add_cost(BeamLengthScaleRegularizationCost(weight=1.0))

    solution = solver.solve()
    np.testing.assert_allclose(
        solution.get_parameter(ELBOW, BeamLengthScale).value, [1.0], atol=1e-3)
