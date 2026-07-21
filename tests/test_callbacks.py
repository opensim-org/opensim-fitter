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
                               TrackingCostFunction, TranslationScaleGroup)
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


def create_custom_joint_translation_model():
    """
    One body attached to ground by a CustomJoint with one driven rotation
    about Y and a non-trivial sinusoidal X-translation. Has a marker at the
    body origin and is used to exercise the translation-scale optimization
    path (the X translation function value is non-zero at typical q, so its
    translation-scale Jacobian column is non-zero).
    """
    model = osim.Model()
    model.setName('custom_joint_translation')
    body = osim.Body('body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)

    st = osim.SpatialTransform()
    empty = osim.ArrayStr()
    q_names = osim.ArrayStr(); q_names.append('q0')

    ax = st.upd_rotation1()
    ax.setCoordinateNames(empty); ax.set_axis(osim.Vec3(1, 0, 0))
    ax.set_function(osim.Constant(0.0))
    ax = st.upd_rotation2()
    ax.setCoordinateNames(q_names); ax.set_axis(osim.Vec3(0, 1, 0))
    ax.set_function(osim.LinearFunction(1.0, 0.0))
    ax = st.upd_rotation3()
    ax.setCoordinateNames(empty); ax.set_axis(osim.Vec3(0, 0, 1))
    ax.set_function(osim.Constant(0.0))
    q_names_t = osim.ArrayStr(); q_names_t.append('q0')
    ax = st.upd_translation1()
    ax.setCoordinateNames(q_names_t); ax.set_axis(osim.Vec3(1, 0, 0))
    ax.set_function(osim.Sine(0.2, 1.0, 0.0))
    ax = st.upd_translation2()
    ax.setCoordinateNames(empty); ax.set_axis(osim.Vec3(0, 1, 0))
    ax.set_function(osim.Constant(0.0))
    ax = st.upd_translation3()
    ax.setCoordinateNames(empty); ax.set_axis(osim.Vec3(0, 0, 1))
    ax.set_function(osim.Constant(0.0))

    cj = osim.CustomJoint(
        'cj', model.getGround(), osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(0), osim.Vec3(0), st)
    model.addJoint(cj)
    model.addMarker(osim.Marker('m0', body, osim.Vec3(0)))
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
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])])
    assert cost.marker_cost is not None
    assert cost.body_scale_groups == [BodyScaleGroup(['/bodyset/body'], [1])]


def test_bilevel_cost_function_add_marker_registers_in_marker_cost():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])])
    cost.add_marker_bilevel_cost('/markerset/m0', osim.Vec3(0))
    assert cost.marker_cost.mobod_indexes.size() == 1


# A helper function for retrieving the outboard frame, `X_BM` of a mobilzed body.
def get_X_BM(matter, idx, state):
    return matter.getMobilizedBody(idx).getOutboardFrame(state).p().to_numpy()

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
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])])

    matter = model.getMatterSubsystem()
    state = cost.state
    cost.marker_cost.apply_scales(
        np.array([2.0, 3.0, 4.0]), np.zeros(0), state)
    np.testing.assert_allclose(get_X_BM(matter, 1, state),
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
            ['/bodyset/body_0', '/bodyset/body_1'], [1, 2])])

    matter = model.getMatterSubsystem()
    state = cost.state
    cost.marker_cost.apply_scales(
        np.array([2.0, 3.0, 4.0]), np.zeros(0), state)
    for k in (1, 2):
        np.testing.assert_allclose(get_X_BM(matter, k, state),
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
        ])

    matter = model.getMatterSubsystem()
    state = cost.state
    cost.marker_cost.apply_scales(
        np.array([2.0, 3.0, 4.0, 5.0, 5.0, 5.0]), np.zeros(0), state)
    for k in (1, 2):
        np.testing.assert_allclose(get_X_BM(matter, k, state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(get_X_BM(matter, 3, state),
                               np.array([0.4 * 5.0, 0.0, 0.0]))


# Test BilevelCostFunction error calculations.

def test_bilevel_cost_function_empty_eval_is_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])])
    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s = ca.DM.ones(3)
    assert float(cost(q, s, ca.DM.zeros(0, 1))) == pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_marker_world_position():
    """
    With a non-zero outboard offset, scaling the body X by 2.0 shifts the body
    in Ground (and the markers fixed to it). Specifically, for a SliderJoint
    whose X_BM = Tx(offset_x), body B's origin in Ground at q=0 is
    -offset_x * sx. Marker m1 at body-frame (0.5, 0, 0) is then at world
    (-offset_x * sx + 0.5, 0, 0).
    """
    model = create_sliding_mass_model(offset_x=0.4)
    model.initSystem()
    cost = BilevelCostFunction(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(0.5, 0, 0))

    q = ca.DM.zeros(len(cost.mc.q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: m1 world = (-0.4 + 0.5) = 0.1. Error = (0.1 - 0.5)^2 = 0.16.
    assert float(cost(q, s_unit, ca.DM.zeros(0, 1))) == pytest.approx(0.16, abs=1e-9)
    # At s_scaled X=2: m1 world = (-0.8 + 0.5) = -0.3. Error = (-0.3 - 0.5)^2 = 0.64.
    assert float(cost(q, s_scaled, ca.DM.zeros(0, 1))) == pytest.approx(0.64, abs=1e-9)


# Test BilevelCostFunction error Jacobian calcluations.

def test_bilevel_cost_function_jacobians_sliding_mass():
    model = create_sliding_mass_model(offset_x=0.4)
    model.initSystem()
    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])])
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.mc.q_indexes))
    s = ca.SX.sym('s', 3)
    x = ca.vertcat(q, s)

    J_jac = ca.Function('J_jac', [x],
                        [ca.jacobian(cost_jac(q, s, ca.DM.zeros(0, 1)), x)])
    J_fd = ca.Function('J_fd', [x],
                       [ca.jacobian(cost_fd(q, s, ca.DM.zeros(0, 1)), x)])

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
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups)
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        opts={'enable_fd': True})

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.mc.q_indexes))
    s = ca.SX.sym('s', 3*bodyset.getSize())
    x = ca.vertcat(q, s)

    J_jac = ca.Function('J_jac', [x],
                        [ca.jacobian(cost_jac(q, s, ca.DM.zeros(0, 1)), x)])
    J_fd = ca.Function('J_fd', [x],
                       [ca.jacobian(cost_fd(q, s, ca.DM.zeros(0, 1)), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.q_indexes), 0.1),
        np.tile([1.1, 1.0, 1.0], bodyset.getSize()),
    ])
    assert np.allclose(J_jac(val).full(), J_fd(val).full(), atol=1e-6)


def test_bilevel_cost_function_custom_joint_translation_scale_jacobian_matches_fd():
    """
    For a CustomJoint with a non-zero translation function, the third (Jt)
    block of the bilevel Jacobian must match the finite-difference Jacobian
    along the translation-scale axis at typical q.
    """
    model = create_custom_joint_translation_model()
    model.initSystem()

    body_scale_groups = [BodyScaleGroup(['/bodyset/body'], [1])]
    ts_groups = [TranslationScaleGroup(['/jointset/cj'], [1])]
    cost_jac = BilevelCostFunction(
        'cost_jac', ModelCache(model),
        body_scale_groups=body_scale_groups, translation_scale_groups=ts_groups)
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model),
        body_scale_groups=body_scale_groups, translation_scale_groups=ts_groups,
        opts={'enable_fd': True})
    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)

    nq = len(cost_jac.mc.q_indexes)
    q = ca.SX.sym('q', nq)
    s = ca.SX.sym('s', 3)
    ts = ca.SX.sym('ts', 3)
    x = ca.vertcat(q, s, ts)
    J_jac = ca.Function('J_jac', [x],
                        [ca.jacobian(cost_jac(q, s, ts), x)])
    J_fd = ca.Function('J_fd', [x],
                       [ca.jacobian(cost_fd(q, s, ts), x)])

    val = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0]),
        np.array([1.2, 1.0, 1.0]),
    ])
    A = J_jac(val).full()
    F = J_fd(val).full()
    assert np.allclose(A, F, atol=1e-6)
    # The X component of the translation-scale Jacobian column should be
    # non-zero (q is non-zero so the X translation function value is
    # non-zero, hence the local Jacobian column is non-zero).
    assert abs(A[0, nq + 3]) > 1e-8


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
        'cost_solo', ModelCache(model), body_scale_groups=solo_groups)
    cost_shared = BilevelCostFunction(
        'cost_shared', ModelCache(model), body_scale_groups=shared_groups)
    cost_fd = BilevelCostFunction(
        'cost_fd', ModelCache(model), body_scale_groups=shared_groups,
        opts={'enable_fd': True})

    for cost in (cost_solo, cost_shared, cost_fd):
        cost.add_marker_bilevel_cost(
            '/markerset/m0', osim.Vec3(0.4, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    nq = len(cost_shared.mc.q_indexes)
    q = ca.SX.sym('q', nq)

    # (b) Shared analytic ≈ FD on the shared callback.
    s_shared = ca.SX.sym('s_shared', 3)
    x_shared = ca.vertcat(q, s_shared)
    J_shared_fn = ca.Function(
        'J_shared', [x_shared],
        [ca.jacobian(cost_shared(q, s_shared, ca.DM.zeros(0, 1)), x_shared)])
    J_fd_fn = ca.Function(
        'J_fd', [x_shared],
        [ca.jacobian(cost_fd(q, s_shared, ca.DM.zeros(0, 1)), x_shared)])
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
        [ca.jacobian(cost_solo(q, s_solo, ca.DM.zeros(0, 1)), x_solo)])
    val_solo = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0, 1.1, 1.0, 1.0]),
    ])
    J_solo = J_solo_fn(val_solo).full()
    solo_sum_cols = J_solo[:, nq:nq+3] + J_solo[:, nq+3:nq+6]
    np.testing.assert_allclose(J_shared[:, nq:nq+3], solo_sum_cols,
                               atol=1e-9)
