"""
Unit tests for TrackingCostFunction and BilevelCostFunction in
src/osimfit/callbacks.py.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim
from pathlib import Path
from osimfit.callbacks import BilevelCostFunction, ScaleGroup, TrackingCostFunction

# Define the test model path.
MODEL_FPATH = str(Path(__file__).parent / 'subject_scale_walk.osim')


# Helper functions.

def create_sliding_mass_model():
    """
    One body sliding along ground X with two markers in the body frame:
    m0 at the origin, m1 at (0.5, 0, 0). The q axis is the world X
    translation; scaling the body X axis stretches m1's world position.
    """
    model = osim.Model()
    model.setName('sliding_mass')
    ground = model.getGround()
    body = osim.Body('body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)
    joint = osim.SliderJoint(
        'slider',
        ground, osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(0), osim.Vec3(0),
    )
    model.addJoint(joint)
    model.addMarker(osim.Marker('m0', body, osim.Vec3(0)))
    model.addMarker(osim.Marker('m1', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model


def create_n_sliding_body_model(n: int):
    """
    n independent bodies, each on its own slider joint from ground along X,
    each with one marker at body-frame (0.5, 0, 0). Mobilized body indexes
    are 1..n in body-addition order.
    """
    model = osim.Model()
    model.setName(f'{n}_sliding_mass')
    ground = model.getGround()
    for i in range(n):
        body = osim.Body(f'body_{i}', 1.0, osim.Vec3(0), osim.Inertia(1))
        model.addBody(body)
        joint = osim.SliderJoint(
            f'slider_{i}',
            ground, osim.Vec3(0), osim.Vec3(0),
            body, osim.Vec3(0), osim.Vec3(0),
        )
        model.addJoint(joint)
        model.addMarker(osim.Marker(f'm{i}', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model


# Test the TrackingCostFunction interface.

def test_tracking_cost_function_constructs_marker_and_frame_subcosts():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    assert cost.marker_cost is not None
    assert cost.frame_cost is not None


def test_tracking_cost_function_add_marker_registers_in_marker_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    assert len(cost.marker_cost.markers) == 1
    assert cost.marker_cost.mobod_indexes.size() == 1
    # frame_cost should be empty.
    assert len(cost.frame_cost.frames) == 0


def test_tracking_cost_function_add_frame_registers_in_frame_cost():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    cost.add_frame_tracking_cost(
        '/bodyset/pelvis', osim.Vec3(0), osim.Quaternion())
    assert len(cost.frame_cost.frames) == 1
    assert cost.frame_cost.mobod_indexes.size() == 1
    # frame_cost should be empty
    assert len(cost.marker_cost.markers) == 0


# Test TrackingCostFunction error calculations.

def test_empty_tracking_cost_function():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    x = ca.DM.zeros(len(cost.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_at_reference_yields_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    # At q=0, m0 sits at the world origin.
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    x = ca.DM.zeros(len(cost.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_off_reference_yields_squared_error():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    # m0 at world (0.1, 0, 0) when q=0.1; reference at the origin.
    cost.add_marker_tracking_cost(
        '/markerset/m0', osim.Vec3(0.0, 0, 0), weight=1.0)
    x = ca.DM([0.1])
    assert float(cost(x)) == pytest.approx(0.01, abs=1e-9)


# Test TrackingCostFunction error Jacobian calculations.

def test_tracking_cost_function_jacobian_sliding_mass():
    model = create_sliding_mass_model()
    model.initSystem()
    cost_jac = TrackingCostFunction('cost_jac', model)
    cost_fd = TrackingCostFunction('cost_fd', model, opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(x), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(x), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


def test_tracking_cost_function_jacobian_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost_jac = TrackingCostFunction('cost_jac', model)
    cost_fd = TrackingCostFunction('cost_fd', model, opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(x), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(x), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)



# Test the BilevelCostFunction interface.

def test_bilevel_cost_function_constructs_marker_subcost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', model,
        scale_groups=[ScaleGroup(['/bodyset/body'], [1])])
    assert cost.marker_cost is not None
    assert cost.scale_groups == [ScaleGroup(['/bodyset/body'], [1])]


def test_bilevel_cost_function_add_marker_registers_in_marker_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', model,
        scale_groups=[ScaleGroup(['/bodyset/body'], [1])])
    cost.add_marker_bilevel_cost('/markerset/m0', osim.Vec3(0))
    assert cost.marker_cost.mobod_indexes.size() == 1


# Check that BilevelCostFunction packs scale factors correctly.

def test_bilevel_pack_scales_writes_to_mobod_indexes_keeps_ground_at_one():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', model,
        scale_groups=[ScaleGroup(['/bodyset/body'], [1])])
    arg = [ca.DM.zeros(len(cost.q_indexes)), ca.DM([2.0, 3.0, 4.0])]
    scales = cost.pack_scales(arg)

    ground = scales.get(0).to_numpy()
    body = scales.get(1).to_numpy()
    assert ground[0] == pytest.approx(1.0)
    assert ground[1] == pytest.approx(1.0)
    assert ground[2] == pytest.approx(1.0)
    assert body[0] == pytest.approx(2.0)
    assert body[1] == pytest.approx(3.0)
    assert body[2] == pytest.approx(4.0)


# Test BilevelCostFunction error calculations.

def test_bilevel_cost_function_empty_eval_is_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', model,
        scale_groups=[ScaleGroup(['/bodyset/body'], [1])])
    q = ca.DM.zeros(len(cost.q_indexes))
    s = ca.DM.ones(3)
    assert float(cost(q, s)) == pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_marker_world_position():
    """
    m1 lives at body offset (0.5, 0, 0); reference at (1.0, 0, 0). Scaling
    the body X by 2.0 should moves m1 to world (1.0, 0, 0).
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', model,
        scale_groups=[ScaleGroup(['/bodyset/body'], [1])])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(1.0, 0, 0))

    q = ca.DM.zeros(len(cost.q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])

    assert float(cost(q, s_unit)) == pytest.approx(0.25, abs=1e-9)
    assert float(cost(q, s_scaled)) == pytest.approx(0.0, abs=1e-9)


# Test BilevelCostFunction error Jacobian calcluations.

def test_bilevel_cost_function_jacobians_sliding_mass():
    model = create_sliding_mass_model()
    model.initSystem()
    cost_jac = BilevelCostFunction(
        'cost_jac', model,
        scale_groups=[ScaleGroup(['/bodyset/body'], [1])])
    cost_fd = BilevelCostFunction(
        'cost_fd', model,
        scale_groups=[ScaleGroup(['/bodyset/body'], [1])],
        opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.q_indexes))
    s = ca.SX.sym('s', 3)
    x = ca.vertcat(q, s)

    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(q, s), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(q, s), x)])

    val = np.concatenate([
        np.full(len(cost_jac.q_indexes), 0.1),
        np.array([1.1, 1.0, 1.0]),
    ])
    assert np.allclose(J_jac(val).full(), J_fd(val).full(), atol=1e-6)


def test_bilevel_cost_function_jacobians_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    bodyset = model.getBodySet()
    scale_groups = []
    for i in range(bodyset.getSize()):
        body = bodyset.get(i)
        scale_groups.append(ScaleGroup(
            body_paths=[body.getAbsolutePathString()],
            mobod_indexes=[int(body.getMobilizedBodyIndex())]))

    cost_jac = BilevelCostFunction(
        'cost_jac', model, scale_groups=scale_groups)
    cost_fd = BilevelCostFunction(
        'cost_fd', model, scale_groups=scale_groups,
        opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.q_indexes))
    s = ca.SX.sym('s', 3*bodyset.getSize())
    x = ca.vertcat(q, s)

    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(q, s), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(q, s), x)])

    val = np.concatenate([
        np.full(len(cost_jac.q_indexes), 0.1),
        np.tile([1.1, 1.0, 1.0], bodyset.getSize()),
    ])
    assert np.allclose(J_jac(val).full(), J_fd(val).full(), atol=1e-6)


# Test BilevelCostFunction shared scale-factor groups.

def test_bilevel_pack_scales_broadcasts_across_shared_group():
    """
    A shared scale group must apply the same set of scale factors to every member body's
    slot in the packed VectorVec3; ground stays at (1, 1, 1).
    """
    model = create_n_sliding_body_model(2)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', model,
        scale_groups=[ScaleGroup(
            ['/bodyset/body_0', '/bodyset/body_1'], [1, 2])])
    arg = [ca.DM.zeros(len(cost.q_indexes)), ca.DM([2.0, 3.0, 4.0])]
    scales = cost.pack_scales(arg)

    np.testing.assert_allclose(scales.get(0).to_numpy(), np.ones(3))
    for k in (1, 2):
        np.testing.assert_allclose(scales.get(k).to_numpy(),
                                   np.array([2.0, 3.0, 4.0]))


def test_bilevel_pack_scales_mixed_groups_apply_independent_vectors():
    """
    With both a shared and a solo group, each group's scale factors must land on its 
    own member bodies independently.
    """
    model = create_n_sliding_body_model(3)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', model,
        scale_groups=[
            ScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
            ScaleGroup(['/bodyset/body_2'], [3]),
        ])
    arg = [ca.DM.zeros(len(cost.q_indexes)),
           ca.DM([2.0, 3.0, 4.0, 5.0, 5.0, 5.0])]
    scales = cost.pack_scales(arg)

    for k in (1, 2):
        np.testing.assert_allclose(scales.get(k).to_numpy(),
                                   np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(scales.get(3).to_numpy(),
                               np.array([5.0, 5.0, 5.0]))


def test_bilevel_cost_function_grouped_jacobian_sums_solo_and_matches_fd():
    """
    For a 2-body model with one marker per body, the shared-group Jacobian
    column for the shared scalar must (a) equal the sum of the solo Jacobian
    columns when both solo scales are set to the same value (chain rule), and
    (b) agree with the finite-difference Jacobian of the shared callback.
    """
    model = create_n_sliding_body_model(2)
    model.initSystem()

    solo_groups = [
        ScaleGroup(['/bodyset/body_0'], [1]),
        ScaleGroup(['/bodyset/body_1'], [2]),
    ]
    shared_groups = [
        ScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
    ]
    cost_solo = BilevelCostFunction(
        'cost_solo', model, scale_groups=solo_groups)
    cost_shared = BilevelCostFunction(
        'cost_shared', model, scale_groups=shared_groups)
    cost_fd = BilevelCostFunction(
        'cost_fd', model, scale_groups=shared_groups,
        opts={'enable_fd': True})

    for cost in (cost_solo, cost_shared, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.4, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    nq = len(cost_shared.q_indexes)
    q = ca.SX.sym('q', nq)

    # (b) Shared analytic ≈ FD on the shared callback.
    s_shared = ca.SX.sym('s_shared', 3)
    x_shared = ca.vertcat(q, s_shared)
    J_shared_fn = ca.Function(
        'J_shared', [x_shared],
        [ca.jacobian(cost_shared(q, s_shared), x_shared)])
    J_fd_fn = ca.Function(
        'J_fd', [x_shared],
        [ca.jacobian(cost_fd(q, s_shared), x_shared)])
    val_shared = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0]),
    ])
    J_shared = J_shared_fn(val_shared).full()
    J_fd = J_fd_fn(val_shared).full()
    assert np.allclose(J_shared, J_fd, atol=1e-6)

    # (a) Shared scale-factor column equals the sum of solo scale-factor
    # columns evaluated at the same s applied to both bodies.
    s_solo = ca.SX.sym('s_solo', 6)
    x_solo = ca.vertcat(q, s_solo)
    J_solo_fn = ca.Function(
        'J_solo', [x_solo],
        [ca.jacobian(cost_solo(q, s_solo), x_solo)])
    val_solo = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0, 1.1, 1.0, 1.0]),
    ])
    J_solo = J_solo_fn(val_solo).full()
    solo_sum_cols = J_solo[:, nq:nq+3] + J_solo[:, nq+3:nq+6]
    np.testing.assert_allclose(J_shared[:, nq:nq+3], solo_sum_cols,
                               atol=1e-9)
