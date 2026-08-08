"""
Unit tests for TrackingCostFunction and BilevelCostFunction in
src/osimfit/callbacks.py.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim
from pathlib import Path
from osimfit.callbacks import (BilevelCostFunction, BodyScaleGroup,
                               MarkerOffsetGroup, FrameOffsetGroup,
                               TrackingCostFunction)
from osimfit.model import ModelCache

# Define the test model path.
MODEL_FPATH = str(Path(__file__).parent / 'subject_scale_walk.osim')


# Helper functions.

def create_sliding_mass_model(offset_x: float = 0.0):
    """
    One body sliding along ground X with two markers in the body frame:
    m0 at the origin, m1 at (0.5, 0, 0). The q axis is the world X
    translation. With `offset_x != 0`, the joint's outboard frame on
    the body (X_BM.p) carries an X translation, so scaling the body's X
    component multiplies that offset and shifts both markers in Ground.
    """
    model = osim.Model()
    model.setName('sliding_mass')
    ground = model.getGround()
    body = osim.Body('body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)
    joint = osim.SliderJoint(
        'slider',
        ground, osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(offset_x, 0, 0), osim.Vec3(0),
    )
    model.addJoint(joint)
    model.addMarker(osim.Marker('m0', body, osim.Vec3(0)))
    model.addMarker(osim.Marker('m1', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model


def create_n_sliding_body_model(n: int, offset_x: float = 0.0):
    """
    n independent bodies, each on its own slider joint from ground along X,
    each with one marker at body-frame (0.5, 0, 0). Mobilized body indexes
    are 1..n in body-addition order. `offset_x != 0` gives every joint a
    non-trivial X_BM translation so body-scale variables produce non-zero
    sensitivities.
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
            body, osim.Vec3(offset_x, 0, 0), osim.Vec3(0),
        )
        model.addJoint(joint)
        model.addMarker(osim.Marker(f'm{i}', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model


# Test the TrackingCostFunction interface.

def test_tracking_cost_function_constructs_marker_and_frame_subcosts():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    assert cost.marker_cost is not None
    assert cost.frame_cost is not None


def test_tracking_cost_function_add_marker_registers_in_marker_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    assert len(cost.marker_cost.markers) == 1
    assert cost.marker_cost.mobod_indexes.size() == 1
    # frame_cost should be empty.
    assert len(cost.frame_cost.frames) == 0


def test_tracking_cost_function_add_frame_registers_in_frame_cost():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
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
    cost = TrackingCostFunction('cost', ModelCache(model))
    x = ca.DM.zeros(len(cost.mc.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_at_reference_yields_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    # At q=0, m0 sits at the world origin.
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    x = ca.DM.zeros(len(cost.mc.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_off_reference_yields_squared_error():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostFunction('cost', ModelCache(model))
    # m0 at world (0.1, 0, 0) when q=0.1; reference at the origin.
    cost.add_marker_tracking_cost(
        '/markerset/m0', osim.Vec3(0.0, 0, 0), weight=1.0)
    x = ca.DM([0.1])
    assert float(cost(x)) == pytest.approx(0.01, abs=1e-9)


# Test TrackingCostFunction error Jacobian calculations.

def test_tracking_cost_function_jacobian_sliding_mass():
    model = create_sliding_mass_model()
    model.initSystem()
    cost_jac = TrackingCostFunction('cost_jac', ModelCache(model))
    cost_fd = TrackingCostFunction('cost_fd', ModelCache(model),
                                   opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(x), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(x), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


def test_tracking_cost_function_jacobian_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost_jac = TrackingCostFunction('cost_jac', ModelCache(model))
    cost_fd = TrackingCostFunction('cost_fd', ModelCache(model),
                                   opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(x), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(x), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)



# Test the BilevelCostFunction interface.

def test_bilevel_cost_function_constructs_marker_subcost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    assert cost.marker_cost is not None
    assert cost.body_scale_groups == [BodyScaleGroup(['/bodyset/body'], [1])]


def test_bilevel_cost_function_add_marker_registers_in_marker_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost('/markerset/m0', osim.Vec3(0))
    assert cost.marker_cost.mobod_indexes.size() == 1


def test_bilevel_cost_function_add_frame_registers_in_frame_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion())
    assert cost.frame_cost.mobod_indexes.size() == 1
    # marker_cost should be empty.
    assert len(cost.marker_cost.markers) == 0


# A helper function for retrieving the outboard frame, `X_BM` of a mobilzed body.
def getX_BM(model, idx, state):
    return model.getJointSet().get(idx).getOutboardFrame(state).p().to_numpy()

# Check that BilevelCostFunction routes scale-group values through the per-mobod
# X_PF / X_BM overrides on the State.

def test_bilevel_apply_scales_writes_xbm_on_target_mobod():
    """
    For a body whose joint's outboard frame on the body (X_BM) has a non-trivial
    translation, applying a Vec3 body scale through the cost should multiply
    each component of X_BM.p() elementwise on the State.
    """
    model = create_sliding_mass_model(offset_x=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])

    state = cost.state
    cost.mc.set_scaled_mobilizer_frame_positions(
        state, cost.body_scale_groups, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(getX_BM(model, 0, state),
                               np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_shared_group_broadcasts_across_members():
    """
    A shared scale group must apply the same set of factors to every member body's
    X_BM override on the State.
    """
    model = create_n_sliding_body_model(2, offset_x=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(
            ['/bodyset/body_0', '/bodyset/body_1'], [1, 2])],
        marker_offset_groups=[], frame_offset_groups=[])

    state = cost.state
    cost.mc.set_scaled_mobilizer_frame_positions(
        state, cost.body_scale_groups, np.array([2.0, 3.0, 4.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getX_BM(model, k, state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_mixed_groups_apply_independent_vectors():
    """
    With both a shared and a solo group, each group's body scales must land on
    its own member bodies' X_BM overrides independently.
    """
    model = create_n_sliding_body_model(3, offset_x=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[
            BodyScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
            BodyScaleGroup(['/bodyset/body_2'], [3]),
        ],
        marker_offset_groups=[], frame_offset_groups=[])

    state = cost.state
    cost.mc.set_scaled_mobilizer_frame_positions(
        state, cost.body_scale_groups, np.array([2.0, 3.0, 4.0, 5.0, 5.0, 5.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getX_BM(model, k, state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(getX_BM(model, 2, state),
                               np.array([0.4 * 5.0, 0.0, 0.0]))


# Test BilevelCostFunction error calculations.

def test_bilevel_cost_function_empty_eval_is_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM.ones(3)
    assert float(cost(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_marker_world_position():
    """
    Scaling the body X by sx does two things: it shifts the body origin in Ground
    (segment-length scaling of the mobilizer frame) and it scales the marker's in-body
    location (geometric scaling). For a SliderJoint whose X_BM = Tx(offset_x), body B's
    origin in Ground at q=0 is -offset_x * sx, and marker m1 at body-frame (0.5, 0, 0)
    scales to (0.5 * sx, 0, 0), so its world position is (-offset_x * sx + 0.5 * sx, 0,
    0) = (0.1 * sx, 0, 0) for offset_x = 0.4.
    """
    model = create_sliding_mass_model(offset_x=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(0.5, 0, 0))

    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: m1 world = (-0.4 + 0.5) = 0.1. Error = (0.1 - 0.5)^2 = 0.16.
    assert float(cost(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.16, abs=1e-9)
    # At s_scaled X=2: m1 world = (-0.8 + 1.0) = 0.2. Error = (0.2 - 0.5)^2 = 0.09.
    assert float(cost(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.09, abs=1e-9)


def test_bilevel_cost_function_frame_at_reference_yields_zero():
    """
    A frame tracked at its own world position and orientation (the body frame
    at q=0 sits at the origin with identity orientation) yields zero error.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost('/bodyset/body', osim.Vec3(0), osim.Quaternion())
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM([1.0, 1.0, 1.0])
    assert float(cost(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_frame_world_position():
    """
    With a non-zero outboard offset, scaling the body X by 2.0 shifts the body
    frame in Ground. For a SliderJoint whose X_BM = Tx(offset_x), body B's
    origin in Ground at q=0 is -offset_x * sx. Tracking that frame against a
    reference at the world origin with position_weight w gives an error of
    w * (offset_x * sx)^2 (the orientation stays identity, so it contributes
    nothing).
    """
    model = create_sliding_mass_model(offset_x=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(), position_weight=2.0)

    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: origin = -0.4. Error = 2 * (-0.4)^2 = 0.32.
    assert float(cost(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.32, abs=1e-9)
    # At s_scaled X=2: origin = -0.8. Error = 2 * (-0.8)^2 = 1.28.
    assert float(cost(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))) == \
        pytest.approx(1.28, abs=1e-9)


# Test BilevelCostFunction error Jacobian calcluations.

def test_bilevel_cost_function_jacobians_sliding_mass():
    model = create_sliding_mass_model(offset_x=0.4)
    model.initSystem()
    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[],
        opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost(
            '/bodyset/body', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.5, orientation_weight=1.0)

    q = ca.SX.sym('q', len(cost_jac.mc.q_indexes))
    s = ca.SX.sym('s', 3)
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.q_indexes), 0.1),
        np.array([1.1, 1.0, 1.0]),
    ])
    assert np.allclose(J_jac(val).full(), J_fd(val).full(), atol=1e-6)


def test_bilevel_cost_function_jacobians_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    bodyset = model.getBodySet()
    body_scale_groups = []
    for i in range(bodyset.getSize()):
        body = bodyset.get(i)
        body_scale_groups.append(BodyScaleGroup(
            body_paths=[body.getAbsolutePathString()],
            mobod_indexes=[int(body.getMobilizedBodyIndex())]))

    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[], opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.mc.q_indexes))
    s = ca.SX.sym('s', 3*bodyset.getSize())
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.q_indexes), 0.1),
        np.tile([1.1, 1.0, 1.0], bodyset.getSize()),
    ])
    assert np.allclose(J_jac(val).full(), J_fd(val).full(), atol=1e-6)


def test_bilevel_cost_function_grouped_jacobian_sums_solo_and_matches_fd():
    """
    For a 2-body model with one marker per body, the shared-group Jacobian
    column for the shared scalar must (a) equal the sum of the solo Jacobian
    columns when both solo scales are set to the same value (chain rule), and
    (b) agree with the finite-difference Jacobian of the shared callback.
    """
    model = create_n_sliding_body_model(2, offset_x=0.4)
    model.initSystem()

    solo_groups = [
        BodyScaleGroup(['/bodyset/body_0'], [1]),
        BodyScaleGroup(['/bodyset/body_1'], [2]),
    ]
    shared_groups = [
        BodyScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
    ]
    cost_solo = BilevelCostFunction(
        'cost_solo', ModelCache(model), body_scale_groups=solo_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_shared = BilevelCostFunction(
        'cost_shared', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[], opts={'enable_fd': True})

    for cost in (cost_solo, cost_shared, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.4, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost(
            '/bodyset/body_0', osim.Vec3(0.2, 0, 0), osim.Quaternion(),
            position_weight=1.0)
        cost.add_frame_bilevel_cost(
            '/bodyset/body_1', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.2)

    nq = len(cost_shared.mc.q_indexes)
    q = ca.SX.sym('q', nq)
    # Empty offset input.
    empty = ca.DM.zeros(0, 1)

    # (b) Shared analytic ≈ FD on the shared callback.
    s_shared = ca.SX.sym('s_shared', 3)
    x_shared = ca.vertcat(q, s_shared)
    J_shared_fn = ca.Function(
        'J_shared', [x_shared],
        [ca.jacobian(cost_shared(q, s_shared, empty, empty), x_shared)])
    J_fd_fn = ca.Function(
        'J_fd', [x_shared],
        [ca.jacobian(cost_fd(q, s_shared, empty, empty), x_shared)])
    val_shared = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0]),
    ])
    J_shared = J_shared_fn(val_shared).full()
    J_fd = J_fd_fn(val_shared).full()
    assert np.allclose(J_shared, J_fd, atol=1e-6)

    # (a) Shared body-scale column equals the sum of solo body-scale
    # columns evaluated at the same s applied to both bodies.
    s_solo = ca.SX.sym('s_solo', 6)
    x_solo = ca.vertcat(q, s_solo)
    J_solo_fn = ca.Function(
        'J_solo', [x_solo],
        [ca.jacobian(cost_solo(q, s_solo, empty, empty), x_solo)])
    val_solo = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0, 1.1, 1.0, 1.0]),
    ])
    J_solo = J_solo_fn(val_solo).full()
    solo_sum_cols = J_solo[:, nq:nq+3] + J_solo[:, nq+3:nq+6]
    np.testing.assert_allclose(J_shared[:, nq:nq+3], solo_sum_cols,
                               atol=1e-9)


# Test BilevelCostFunction marker/frame offset handling.

def test_bilevel_apply_offsets_shifts_station():
    """
    apply_station_transforms sets each offset task's cached station to baseline + offset
    (absolute, not compounding) at identity body scale, leaving non-offset tasks
    untouched.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'])],
        frame_offset_groups=[])
    # m1 (offset group 0) and m0 (no offset).
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    cost.add_marker_bilevel_cost('/markerset/m0', osim.Vec3(0, 0, 0))
    mc = cost.marker_cost
    baseline_m1 = mc.base_stations[0].copy()
    baseline_m0 = mc.base_stations[1].copy()

    offset = np.array([0.1, -0.2, 0.3])
    mc.apply_station_transforms(np.ones(3), offset)
    np.testing.assert_allclose(mc.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)
    np.testing.assert_allclose(mc.stations.getElt(1).to_numpy(), baseline_m0)

    # Absolute (not compounding): applying again with the same offset is idempotent.
    mc.apply_station_transforms(np.ones(3), offset)
    np.testing.assert_allclose(mc.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)


def test_bilevel_offset_changes_marker_error():
    """
    Offsetting a marker's station shifts its Ground position (R_GB is identity for
    the slider), changing the tracking error by the expected amount.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'])],
        frame_offset_groups=[])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM.ones(3)
    # No offset: m1 world = 0.5, reference = 0.5, error = 0.
    assert float(cost(q, s, ca.DM.zeros(3), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.0, abs=1e-12)
    # Offset X by 0.2: m1 world = 0.7, error = (0.7 - 0.5)^2 = 0.04.
    assert float(cost(q, s, ca.DM([0.2, 0, 0]), ca.DM.zeros(0, 1))) == \
        pytest.approx(0.04, abs=1e-9)


def test_bilevel_offset_frame_orientation_invariant():
    """
    A translation offset shifts a frame's position but not its orientation, so an
    orientation-only frame cost (position_weight = 0) is invariant to the offset.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[],
        frame_offset_groups=[FrameOffsetGroup(['/bodyset/body'])])
    cost.add_frame_bilevel_cost(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(0.9, 0.1, 0.2, 0.3),
        position_weight=0.0, orientation_weight=1.0, offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM.ones(3)
    e0 = float(cost(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(3)))
    e1 = float(cost(q, s, ca.DM.zeros(0, 1), ca.DM([0.2, -0.1, 0.3])))
    assert e0 > 0.0
    assert e0 == pytest.approx(e1, abs=1e-12)


def test_bilevel_cost_function_offset_jacobians_full_body():
    """
    On the full-body model (which has rotational DOFs, so R_GB != identity), the
    analytic bilevel Jacobian over [q, s, o] -- including the marker/frame offset
    columns and the offset-induced coupling into the q-columns -- must match the
    finite-difference Jacobian.
    """
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    pelvis = osim.Body.safeDownCast(model.getComponent('/bodyset/pelvis'))
    body_scale_groups = [BodyScaleGroup(
        ['/bodyset/pelvis'], [int(pelvis.getMobilizedBodyIndex())])]

    marker_offset_groups = [MarkerOffsetGroup(['/markerset/R.Shoulder'])]
    frame_offset_groups = [FrameOffsetGroup(['/bodyset/pelvis'])]
    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups)
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups, opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0,
            offset_group_index=0)
        cost.add_frame_bilevel_cost(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5,
            offset_group_index=0)

    nq = len(cost_jac.mc.q_indexes)
    q = ca.SX.sym('q', nq)
    s = ca.SX.sym('s', 3)
    mo = ca.SX.sym('mo', 3)
    fo = ca.SX.sym('fo', 3)
    x = ca.vertcat(q, s, mo, fo)

    J_jac = ca.Function('J_jac', [x],
                        [ca.jacobian(cost_jac(q, s, mo, fo), x)])
    J_fd = ca.Function('J_fd', [x],
                       [ca.jacobian(cost_fd(q, s, mo, fo), x)])

    val = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0]),
        np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.0]),
    ])
    A = J_jac(val).full()
    F = J_fd(val).full()
    assert np.allclose(A, F, atol=1e-6)
    # The offset columns should be non-zero.
    assert np.any(np.abs(A[0, nq+3:nq+9]) > 1e-8)
