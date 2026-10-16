"""
Tests for the tracking and bilevel cost functions and the terms they are built
from, including their Jacobians against finite differences and the per-component
weights applied to each task.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim

from pathlib import Path
from osimfit.model import (ModelCache, BodyScaleGroup, MarkerOffsetGroup,
                           FrameOffsetGroup, EllipsoidRadiiScaleGroup,
                           BeamLengthScaleGroup)
from osimfit.solvers import SplinedKinematicsSolver
from osimfit.costs import (CostInput, FrameTrackingTerm, FrameBilevelTerm,
                           MarkerTrackingTerm, BilevelCostRep, TrackingCost,
                           TrackingCostRep, BilevelCost)
from tests.test_costs import create_beam_model, getP_BM

# Define the test model path.
MODEL_FPATH = str(Path(__file__).parent / 'subject_scale_walk.osim')

def create_sliding_mass_model(child_x_offset: float = 0.0):
    """
    Create a model with ne body sliding along the X-direction in ground with two markers
    in the body frame: 'm0' at the origin, 'm1' at (0.5, 0, 0). Use `child_x_offset` to
    add offset in the X-direction for the child body's joint frame.
    """
    model = osim.Model()
    model.setName('sliding_mass')
    ground = model.getGround()
    body = osim.Body('body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)
    joint = osim.SliderJoint(
        'slider',
        ground, osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(child_x_offset, 0, 0), osim.Vec3(0),
    )
    model.addJoint(joint)
    model.addMarker(osim.Marker('m0', body, osim.Vec3(0)))
    model.addMarker(osim.Marker('m1', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model

def test_tracking_cost_function_constructs_marker_and_frame_terms():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    assert cost.marker_term is not None
    assert cost.frame_term is not None


def test_tracking_cost_function_add_marker_registers_in_marker_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    cost.add_marker_tracking_cost_term('/markerset/m0', osim.Vec3(0))
    assert len(cost.marker_term.markers) == 1
    assert cost.marker_term.mobod_indexes.size() == 1
    assert len(cost.frame_term.frames) == 0


def test_tracking_cost_function_add_frame_registers_in_frame_term():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    cost.add_frame_tracking_cost_term(
        '/bodyset/pelvis', osim.Vec3(0), osim.Quaternion())
    assert len(cost.frame_term.frames) == 1
    assert cost.frame_term.mobod_indexes.size() == 1
    assert len(cost.marker_term.markers) == 0


def test_empty_tracking_cost_function():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    x = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    assert float(cost(CostInput(coordinates=x))) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_at_reference_yields_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    # At q=0, m0 sits at the world origin.
    cost.add_marker_tracking_cost_term('/markerset/m0', osim.Vec3(0))
    x = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    assert float(cost(CostInput(coordinates=x))) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_off_reference_yields_squared_error():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = TrackingCostRep('cost', ModelCache(model))
    # m0 at world (0.1, 0, 0) when q=0.1; reference at the origin.
    cost.add_marker_tracking_cost_term(
        '/markerset/m0', osim.Vec3(0.0, 0, 0), weight=1.0)
    x = ca.DM([0.1])
    assert float(cost(CostInput(coordinates=x))) == pytest.approx(0.01, abs=1e-9)


def test_tracking_cost_function_jacobian_sliding_mass():
    model = create_sliding_mass_model()
    model.initSystem()
    cost_jac = TrackingCostRep('cost_jac', ModelCache(model))
    cost_fd = TrackingCostRep('cost_fd', ModelCache(model),
                                   enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost_term(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost_term(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.coordinate_q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(CostInput(x)), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(CostInput(x)), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


def test_tracking_cost_function_jacobian_full_body():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost_jac = TrackingCostRep('cost_jac', ModelCache(model))
    cost_fd = TrackingCostRep('cost_fd', ModelCache(model),
                                   enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_tracking_cost_term(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_tracking_cost_term(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)

    x = ca.SX.sym('x', len(cost_jac.mc.coordinate_q_indexes))
    J_jac = ca.Function('J_jac', [x], [ca.jacobian(cost_jac(CostInput(x)), x)])
    J_fd = ca.Function('J_fd', [x], [ca.jacobian(cost_fd(CostInput(x)), x)])

    assert np.allclose(J_jac(0.1).full(), J_fd(0.1).full(), atol=1e-6)


#########################
# BILEVEL COST FUNCTION #
#########################

def create_n_sliding_body_model(n: int, child_x_offset: float = 0.0):
    """
    Create a model with `n` independent bodies, each on its own slider joint along the
    X-direction in ground, each with one marker at body-frame (0.5, 0, 0). Mobilized
    body indexes are 1..n in body-addition order. Use `child_x_offset` to
    add offset in the X-direction for each child body's joint frame.
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
            body, osim.Vec3(child_x_offset, 0, 0), osim.Vec3(0),
        )
        model.addJoint(joint)
        model.addMarker(osim.Marker(f'm{i}', body, osim.Vec3(0.5, 0, 0)))
    model.finalizeConnections()
    return model


def build_bilevel_rep(name, mc, body_scale_groups=[], marker_offset_groups=[],
                       frame_offset_groups=[], ellipsoid_radii_scale_groups=[],
                       beam_length_scale_groups=[], enable_fd=False):
    """
    Register the given parameter groups on `mc` and build a BilevelCostRep, which
    reads its groups from the ModelCache. Mirrors what BilevelCost.create_rep does,
    without needing a Trial to supply reference data.

    Every group goes through `add_parameter_group`, since that is what resolves the
    model-specific state each group implies (a joint-level group's `Joint`s, a body
    scale group's mobilizer frame writes).
    """
    for group in (list(body_scale_groups) + list(marker_offset_groups)
                  + list(frame_offset_groups) + list(ellipsoid_radii_scale_groups)
                  + list(beam_length_scale_groups)):
        mc.add_parameter_group(group)
    return BilevelCostRep(name, mc, enable_fd=enable_fd)


def test_bilevel_cost_function_constructs_marker_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    assert cost.marker_term is not None
    assert cost.mc.body_scale_groups == [BodyScaleGroup(['/bodyset/body'], [1])]


def test_bilevel_cost_function_add_marker_registers_in_marker_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m0', osim.Vec3(0))
    assert cost.marker_term.mobod_indexes.size() == 1
    assert len(cost.frame_term.frames) == 0


def test_bilevel_cost_function_add_frame_registers_in_frame_term():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost_term(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion())
    assert cost.frame_term.mobod_indexes.size() == 1
    assert len(cost.marker_term.markers) == 0


def test_bilevel_apply_scales_shifts_child_frame_translation():
    """
    Applying body scales through the cost should multiply each component of the model's
    child frame translation elementswise.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(getP_BM(model, 0, cost.state),
                               np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_shared_group_broadcasts_across_members():
    """
    A scale group must apply the same set of scale factors to every member body's
    child frame translations.
    """
    model = create_n_sliding_body_model(2, child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(
            ['/bodyset/body_0', '/bodyset/body_1'], [1, 2])],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, np.array([2.0, 3.0, 4.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getP_BM(model, k, cost.state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))


def test_bilevel_apply_scales_mixed_groups_apply_independent_vectors():
    """
    Separate scale groups must apply scale factors to owned bodies independently.
    """
    model = create_n_sliding_body_model(3, child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[
            BodyScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
            BodyScaleGroup(['/bodyset/body_2'], [3]),
        ],
        marker_offset_groups=[], frame_offset_groups=[])

    cost.mc.set_scaled_mobilizer_frame_positions(
        cost.state, np.array([2.0, 3.0, 4.0, 5.0, 5.0, 5.0]))
    for k in (0, 1):
        np.testing.assert_allclose(getP_BM(model, k, cost.state),
                                   np.array([0.4 * 2.0, 0.0, 0.0]))
    np.testing.assert_allclose(getP_BM(model, 2, cost.state),
                               np.array([0.4 * 5.0, 0.0, 0.0]))


def test_bilevel_cost_function_empty_eval_is_zero():
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    q = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    s = ca.DM.ones(3)
    assert float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_marker_world_position():
    """
    Scaling a body changes both the segment length (via offset frame scaling) and the
    positions of markers on the body.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m1', osim.Vec3(0.5, 0, 0))

    q = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: m1 world = (-0.4 + 0.5) = 0.1. Error = (0.1 - 0.5)^2 = 0.16.
    assert float(cost(CostInput(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.16, abs=1e-9)
    # At s_scaled X=2: m1 world = (-0.8 + 1.0) = 0.2. Error = (0.2 - 0.5)^2 = 0.09.
    assert float(cost(CostInput(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.09, abs=1e-9)


def test_bilevel_cost_function_frame_at_reference_yields_zero():
    """
    A frame tracked at its own world position and orientation yields zero error.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost_term('/bodyset/body', osim.Vec3(0), osim.Quaternion())
    q = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    s = ca.DM([1.0, 1.0, 1.0])
    assert float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_frame_world_position():
    """
    Scaling a body with a non-zero child frame offset shifts the child frame in ground.
    The child frame offset should contribute to the squared error in the cost.
    """
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost.add_frame_bilevel_cost_term(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(), position_weight=2.0)

    q = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])
    # At s_unit: origin = -0.4. Error = 2 * (-0.4)^2 = 0.32.
    assert float(cost(CostInput(q, s_unit, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.32, abs=1e-9)
    # At s_scaled X=2: origin = -0.8. Error = 2 * (-0.8)^2 = 1.28.
    assert float(cost(CostInput(q, s_scaled, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1)))) == \
        pytest.approx(1.28, abs=1e-9)


def test_bilevel_cost_function_jacobians_sliding_mass():
    model = create_sliding_mass_model(child_x_offset=0.4)
    model.initSystem()
    cost_jac = build_bilevel_rep(
        'cost_jac', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[], frame_offset_groups=[],
        enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/m0', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost_term(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/body', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.5, orientation_weight=1.0)

    q = ca.SX.sym('q', len(cost_jac.mc.coordinate_q_indexes))
    s = ca.SX.sym('s', 3)
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.coordinate_q_indexes), 0.1),
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

    cost_jac = build_bilevel_rep(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=[], frame_offset_groups=[], enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost_term(
            '/markerset/L.ASIS', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5)

    q = ca.SX.sym('q', len(cost_jac.mc.coordinate_q_indexes))
    s = ca.SX.sym('s', 3*bodyset.getSize())
    x = ca.vertcat(q, s)

    J_jac = ca.Function(
        'J_jac', [x],
        [ca.jacobian(cost_jac(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])
    J_fd = ca.Function(
        'J_fd', [x],
        [ca.jacobian(cost_fd(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(0, 1))), x)])

    val = np.concatenate([
        np.full(len(cost_jac.mc.coordinate_q_indexes), 0.1),
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
    model = create_n_sliding_body_model(2, child_x_offset=0.4)
    model.initSystem()

    solo_groups = [
        BodyScaleGroup(['/bodyset/body_0'], [1]),
        BodyScaleGroup(['/bodyset/body_1'], [2]),
    ]
    shared_groups = [
        BodyScaleGroup(['/bodyset/body_0', '/bodyset/body_1'], [1, 2]),
    ]
    cost_solo = build_bilevel_rep(
        'cost_solo', ModelCache(model), body_scale_groups=solo_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_shared = build_bilevel_rep(
        'cost_shared', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[])
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model), body_scale_groups=shared_groups,
        marker_offset_groups=[], frame_offset_groups=[], enable_fd=True)

    for cost in (cost_solo, cost_shared, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/m0', osim.Vec3(0.4, 0, 0), weight=2.0)
        cost.add_marker_bilevel_cost_term(
            '/markerset/m1', osim.Vec3(0.7, 0, 0), weight=1.5)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/body_0', osim.Vec3(0.2, 0, 0), osim.Quaternion(),
            position_weight=1.0)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/body_1', osim.Vec3(0.5, 0, 0), osim.Quaternion(),
            position_weight=1.2)

    nq = len(cost_shared.mc.coordinate_q_indexes)
    q = ca.SX.sym('q', nq)
    offset = ca.DM.zeros(0, 1)

    # (b) Shared analytic ≈ FD on the shared callback.
    s_shared = ca.SX.sym('s_shared', 3)
    x_shared = ca.vertcat(q, s_shared)
    J_shared_fn = ca.Function(
        'J_shared', [x_shared],
        [ca.jacobian(cost_shared(CostInput(q, s_shared, offset, offset)), x_shared)])
    J_fd_fn = ca.Function(
        'J_fd', [x_shared],
        [ca.jacobian(cost_fd(CostInput(q, s_shared, offset, offset)), x_shared)])
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
        [ca.jacobian(cost_solo(CostInput(q, s_solo, offset, offset)), x_solo)])
    val_solo = np.concatenate([
        np.full(nq, 0.1),
        np.array([1.1, 1.0, 1.0, 1.1, 1.0, 1.0]),
    ])
    J_solo = J_solo_fn(val_solo).full()
    solo_sum_cols = J_solo[:, nq:nq+3] + J_solo[:, nq+3:nq+6]
    np.testing.assert_allclose(J_shared[:, nq:nq+3], solo_sum_cols,
                               atol=1e-9)


def create_ellipsoid_chain_model(radii=(0.1, 0.2, 0.3)):
    """
    Create a two-body chain: ground -> EllipsoidJoint -> 'proximal' -> PinJoint ->
    'distal', with one marker on each body. The distal marker sits downstream of the
    ellipsoid mobilizer, so its sensitivity to the radii exercises the outboard-subtree
    accumulation inside multiplyByPositionJacobianWrtRadiiTranspose().
    """
    model = osim.Model()
    model.setName('ellipsoid_chain')
    ground = model.getGround()
    proximal = osim.Body('proximal', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(proximal)
    model.addJoint(osim.EllipsoidJoint(
        'ellipsoid', ground, osim.Vec3(0), osim.Vec3(0),
        proximal, osim.Vec3(0), osim.Vec3(0), osim.Vec3(*radii)))
    distal = osim.Body('distal', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(distal)
    model.addJoint(osim.PinJoint(
        'pin', proximal, osim.Vec3(0.3, 0, 0), osim.Vec3(0),
        distal, osim.Vec3(0), osim.Vec3(0)))
    model.addMarker(osim.Marker('m_prox', proximal, osim.Vec3(0.2, 0.1, 0)))
    model.addMarker(osim.Marker('m_dist', distal, osim.Vec3(0.4, 0, 0.1)))
    model.finalizeConnections()
    return model


def create_beam_chain_model(length=1.2):
    """
    The `create_ellipsoid_chain_model` layout with the EllipsoidJoint replaced by a
    CantileverFreeBeamJoint, so the distal marker exercises subtree accumulation in
    multiplyByPositionJacobianWrtLengthTranspose().
    """
    model = osim.Model()
    model.setName('beam_chain')
    ground = model.getGround()
    proximal = osim.Body('proximal', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(proximal)
    model.addJoint(osim.CantileverFreeBeamJoint(
        'beam', ground, osim.Vec3(0), osim.Vec3(0),
        proximal, osim.Vec3(0), osim.Vec3(0), length))
    distal = osim.Body('distal', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(distal)
    model.addJoint(osim.PinJoint(
        'pin', proximal, osim.Vec3(0.3, 0, 0), osim.Vec3(0),
        distal, osim.Vec3(0), osim.Vec3(0)))
    model.addMarker(osim.Marker('m_prox', proximal, osim.Vec3(0.2, 0.1, 0)))
    model.addMarker(osim.Marker('m_dist', distal, osim.Vec3(0.4, 0, 0.1)))
    model.finalizeConnections()
    return model


def create_two_ellipsoid_model(radii=(0.1, 0.2, 0.3)):
    """
    Create two independent bodies, each on its own EllipsoidJoint in ground, with one
    marker each. Used to check that a shared EllipsoidRadiiScaleGroup broadcasts
    across its joints.
    """
    model = osim.Model()
    model.setName('two_ellipsoid')
    ground = model.getGround()
    for i in range(2):
        body = osim.Body(f'body_{i}', 1.0, osim.Vec3(0), osim.Inertia(1))
        model.addBody(body)
        model.addJoint(osim.EllipsoidJoint(
            f'ellipsoid_{i}', ground, osim.Vec3(0), osim.Vec3(0),
            body, osim.Vec3(0), osim.Vec3(0), osim.Vec3(*radii)))
        model.addMarker(osim.Marker(f'm{i}', body, osim.Vec3(0.3, 0.1, 0)))
    model.finalizeConnections()
    return model


def add_chain_cost_terms(cost):
    """
    Add the same marker and frame tracking terms used by the joint-parameter Jacobian
    tests to `cost`. The frame term pins an orientation reference as well, so the
    orientation-error path is exercised alongside the position-error path.
    """
    cost.add_marker_bilevel_cost_term(
        '/markerset/m_prox', osim.Vec3(0.25, 0.05, 0.1), weight=2.0)
    cost.add_marker_bilevel_cost_term(
        '/markerset/m_dist', osim.Vec3(0.5, -0.1, 0.2), weight=1.5)
    cost.add_frame_bilevel_cost_term(
        '/bodyset/distal', osim.Vec3(0.3, 0.1, -0.2),
        osim.Quaternion(0.9, 0.1, 0.2, 0.3),
        position_weight=2.0, orientation_weight=1.5)


def test_bilevel_cost_function_ellipsoid_radii_scale_jacobian_matches_fd():
    """
    The analytic Jacobian with respect to a single EllipsoidRadiiScaleGroup must
    match the finite-difference Jacobian of the same callback, including the
    contribution routed through the downstream (pin-jointed) body. The variables
    are factors on the joint's baseline radii, so the analytic path carries the
    baseline through the chain rule and finite differences must agree with it.
    """
    model = create_ellipsoid_chain_model()
    model.initSystem()
    groups = [EllipsoidRadiiScaleGroup(['/jointset/ellipsoid'])]

    cost_jac = build_bilevel_rep('cost_jac', ModelCache(model),
                                 ellipsoid_radii_scale_groups=groups)
    cost_fd = build_bilevel_rep('cost_fd', ModelCache(model),
                                ellipsoid_radii_scale_groups=groups, enable_fd=True)
    for cost in (cost_jac, cost_fd):
        add_chain_cost_terms(cost)

    nq = len(cost_jac.mc.coordinate_q_indexes)
    q = ca.SX.sym('q', nq)
    r = ca.SX.sym('r', 3)
    x = ca.vertcat(q, r)
    empty = ca.DM.zeros(0, 1)

    def jacobian_of(cost):
        return ca.Function('J', [x], [ca.jacobian(
            cost(CostInput(coordinates=q, body_scales=empty, marker_offsets=empty,
                           frame_offsets=empty, ellipsoid_radii_scales=r,
                           beam_length_scales=empty)), x)])

    val = np.concatenate([np.full(nq, 0.15), np.array([1.1, 0.9, 1.05])])
    np.testing.assert_allclose(jacobian_of(cost_jac)(val).full(),
                               jacobian_of(cost_fd)(val).full(), atol=1e-6)


def test_bilevel_cost_function_beam_length_scale_jacobian_matches_fd():
    """
    The analytic Jacobian with respect to a single BeamLengthScaleGroup must match the
    finite-difference Jacobian of the same callback, including the contribution routed
    through the downstream (pin-jointed) body. The variable is a factor on the joint's
    baseline length, so the analytic path carries the baseline through the chain rule
    and finite differences must agree with it.
    """
    model = create_beam_chain_model()
    model.initSystem()
    groups = [BeamLengthScaleGroup(['/jointset/beam'])]

    cost_jac = build_bilevel_rep('cost_jac', ModelCache(model),
                                 beam_length_scale_groups=groups)
    cost_fd = build_bilevel_rep('cost_fd', ModelCache(model),
                                beam_length_scale_groups=groups, enable_fd=True)
    for cost in (cost_jac, cost_fd):
        add_chain_cost_terms(cost)

    nq = len(cost_jac.mc.coordinate_q_indexes)
    q = ca.SX.sym('q', nq)
    length = ca.SX.sym('length', 1)
    x = ca.vertcat(q, length)
    empty = ca.DM.zeros(0, 1)

    def jacobian_of(cost):
        return ca.Function('J', [x], [ca.jacobian(
            cost(CostInput(coordinates=q, body_scales=empty, marker_offsets=empty,
                           frame_offsets=empty, ellipsoid_radii_scales=empty,
                           beam_length_scales=length)), x)])

    val = np.concatenate([np.full(nq, 0.15), np.array([1.15])])
    np.testing.assert_allclose(jacobian_of(cost_jac)(val).full(),
                               jacobian_of(cost_fd)(val).full(), atol=1e-6)


def test_bilevel_shared_ellipsoid_radii_scale_group_jacobian_sums_solo_and_matches_fd():
    """
    For two bodies each on their own EllipsoidJoint, the shared-group radii Jacobian
    must (a) equal the sum of the solo Jacobian columns when both solo groups hold the
    same factors, and (b) agree with the finite-difference Jacobian of the shared
    callback.
    """
    model = create_two_ellipsoid_model()
    model.initSystem()
    solo_groups = [EllipsoidRadiiScaleGroup(['/jointset/ellipsoid_0']),
                   EllipsoidRadiiScaleGroup(['/jointset/ellipsoid_1'])]
    shared_groups = [EllipsoidRadiiScaleGroup(
        ['/jointset/ellipsoid_0', '/jointset/ellipsoid_1'])]

    cost_solo = build_bilevel_rep('cost_solo', ModelCache(model),
                                  ellipsoid_radii_scale_groups=solo_groups)
    cost_shared = build_bilevel_rep('cost_shared', ModelCache(model),
                                    ellipsoid_radii_scale_groups=shared_groups)
    cost_fd = build_bilevel_rep('cost_fd', ModelCache(model),
                                ellipsoid_radii_scale_groups=shared_groups,
                                enable_fd=True)
    for cost in (cost_solo, cost_shared, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/m0', osim.Vec3(0.35, 0.05, 0.1), weight=2.0)
        cost.add_marker_bilevel_cost_term(
            '/markerset/m1', osim.Vec3(0.2, -0.1, 0.25), weight=1.5)

    nq = len(cost_shared.mc.coordinate_q_indexes)
    q = ca.SX.sym('q', nq)
    empty = ca.DM.zeros(0, 1)

    def jacobian_of(cost, r, x):
        return ca.Function('J', [x], [ca.jacobian(
            cost(CostInput(coordinates=q, body_scales=empty, marker_offsets=empty,
                           frame_offsets=empty, ellipsoid_radii_scales=r,
                           beam_length_scales=empty)), x)])

    # (b) Shared analytic vs FD on the shared callback.
    r_shared = ca.SX.sym('r_shared', 3)
    x_shared = ca.vertcat(q, r_shared)
    val_shared = np.concatenate([np.full(nq, 0.15), np.array([1.1, 0.9, 1.05])])
    J_shared = jacobian_of(cost_shared, r_shared, x_shared)(val_shared).full()
    J_fd = jacobian_of(cost_fd, r_shared, x_shared)(val_shared).full()
    np.testing.assert_allclose(J_shared, J_fd, atol=1e-6)

    # (a) Shared factor column equals the sum of the solo factor columns.
    r_solo = ca.SX.sym('r_solo', 6)
    x_solo = ca.vertcat(q, r_solo)
    val_solo = np.concatenate([np.full(nq, 0.15), np.tile([1.1, 0.9, 1.05], 2)])
    J_solo = jacobian_of(cost_solo, r_solo, x_solo)(val_solo).full()
    np.testing.assert_allclose(J_shared[:, nq:nq+3],
                               J_solo[:, nq:nq+3] + J_solo[:, nq+3:nq+6], atol=1e-9)


def test_bilevel_cost_function_all_parameter_jacobians_match_fd():
    """
    Body scales, marker offsets, frame offsets, ellipsoid radii, and beam lengths
    active at once: the assembled six-input Jacobian must match finite differences.
    Guards the ordering of the blocks returned by `BilevelCostRep._jac_eval`.
    """
    model = create_ellipsoid_chain_model()
    model.initSystem()
    body_scale_groups = [BodyScaleGroup(['/bodyset/distal'], [2])]
    marker_offset_groups = [MarkerOffsetGroup(['/markerset/m_dist'], [2])]
    frame_offset_groups = [FrameOffsetGroup(['/bodyset/distal'], [2])]
    ellipsoid_radii_scale_groups = [EllipsoidRadiiScaleGroup(['/jointset/ellipsoid'])]

    def build(name, enable_fd):
        return build_bilevel_rep(
            name, ModelCache(model), body_scale_groups=body_scale_groups,
            marker_offset_groups=marker_offset_groups,
            frame_offset_groups=frame_offset_groups,
            ellipsoid_radii_scale_groups=ellipsoid_radii_scale_groups,
            enable_fd=enable_fd)

    cost_jac = build('cost_jac', False)
    cost_fd = build('cost_fd', True)
    for cost in (cost_jac, cost_fd):
        add_chain_cost_terms(cost)

    nq = len(cost_jac.mc.coordinate_q_indexes)
    q = ca.SX.sym('q', nq)
    s = ca.SX.sym('s', 3)
    mo = ca.SX.sym('mo', 3)
    fo = ca.SX.sym('fo', 3)
    r = ca.SX.sym('r', 3)
    x = ca.vertcat(q, s, mo, fo, r)

    def jacobian_of(cost):
        return ca.Function('J', [x], [ca.jacobian(
            cost(CostInput(coordinates=q, body_scales=s, marker_offsets=mo,
                           frame_offsets=fo, ellipsoid_radii_scales=r,
                           beam_length_scales=ca.DM.zeros(0, 1))), x)])

    val = np.concatenate([
        np.full(nq, 0.15),
        np.array([1.1, 1.0, 0.95]),
        np.array([0.01, -0.02, 0.03]),
        np.array([-0.01, 0.02, 0.01]),
        np.array([1.1, 0.9, 1.05]),
    ])
    np.testing.assert_allclose(jacobian_of(cost_jac)(val).full(),
                               jacobian_of(cost_fd)(val).full(), atol=1e-6)


def test_bilevel_apply_state_shifts_station():
    """
    Use apply_state() to set each offset task's cached station to baseline + offset
    at identity body scale, leaving non-offset tasks untouched.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'], [2])],
        frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    cost.add_marker_bilevel_cost_term('/markerset/m0', osim.Vec3(0, 0, 0))
    term = cost.marker_term
    baseline_m1 = term.base_stations[0].copy()
    baseline_m0 = term.base_stations[1].copy()

    body_scale = np.ones(3)
    offset = np.array([0.1, -0.2, 0.3])
    term.apply_state(body_scale, offset)
    np.testing.assert_allclose(term.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)
    np.testing.assert_allclose(term.stations.getElt(1).to_numpy(), baseline_m0)

    # apply_state() is idempotent.
    term.apply_state(body_scale, offset)
    np.testing.assert_allclose(term.stations.getElt(0).to_numpy(),
                               baseline_m1 + offset)


def test_bilevel_offset_changes_marker_error():
    """
    Applying an offset to a marker shifts its position and ground and shoudl yield a
    change in the tracking error.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[MarkerOffsetGroup(['/markerset/m1'], [2])],
        frame_offset_groups=[])
    cost.add_marker_bilevel_cost_term('/markerset/m1', osim.Vec3(0.5, 0, 0),
                                 offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    s = ca.DM.ones(3)
    # No offset: m1 world = 0.5, reference = 0.5, error = 0.
    assert float(cost(CostInput(q, s, ca.DM.zeros(3), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.0, abs=1e-12)
    # Offset X by 0.2: m1 world = 0.7, error = (0.7 - 0.5)^2 = 0.04.
    assert float(cost(CostInput(q, s, ca.DM([0.2, 0, 0]), ca.DM.zeros(0, 1)))) == \
        pytest.approx(0.04, abs=1e-9)


def test_bilevel_offset_frame_orientation_invariant():
    """
    Applying a translation offset to frame's position should not affect its orientation,
    so an orientation-only frame cost (position_weight = 0) is invariant to the offset.
    """
    model = create_sliding_mass_model()
    model.initSystem()
    cost = build_bilevel_rep(
        'cost', ModelCache(model),
        body_scale_groups=[BodyScaleGroup(['/bodyset/body'], [1])],
        marker_offset_groups=[],
        frame_offset_groups=[FrameOffsetGroup(['/bodyset/body'], [1])])
    cost.add_frame_bilevel_cost_term(
        '/bodyset/body', osim.Vec3(0), osim.Quaternion(0.9, 0.1, 0.2, 0.3),
        position_weight=0.0, orientation_weight=1.0, offset_group_index=0)
    q = ca.DM.zeros(len(cost.mc.coordinate_q_indexes))
    s = ca.DM.ones(3)
    e0 = float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM.zeros(3))))
    e1 = float(cost(CostInput(q, s, ca.DM.zeros(0, 1), ca.DM([0.2, -0.1, 0.3]))))
    assert e0 > 0.0
    assert e0 == pytest.approx(e1, abs=1e-12)


def test_bilevel_cost_function_offset_jacobians_full_body():
    """
    On the full-body model, the analytic bilevel Jacobian over [q, s, o], including the
    marker and frame offset columns and the offset-induced coupling into the q-columns,
    must match the finite-difference Jacobian.
    """
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    pelvis = osim.Body.safeDownCast(model.getComponent('/bodyset/pelvis'))
    pelvis_mbx = int(pelvis.getMobilizedBodyIndex())
    torso = osim.Body.safeDownCast(model.getComponent('/bodyset/torso'))
    torso_mbx = int(torso.getMobilizedBodyIndex())
    body_scale_groups = [BodyScaleGroup(['/bodyset/pelvis'], [pelvis_mbx])]

    marker_offset_groups = [MarkerOffsetGroup(['/markerset/R.Shoulder'], [torso_mbx])]
    frame_offset_groups = [FrameOffsetGroup(['/bodyset/pelvis'], [pelvis_mbx])]
    cost_jac = build_bilevel_rep(
        'cost_jac', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups)
    cost_fd = build_bilevel_rep(
        'cost_fd', ModelCache(model), body_scale_groups=body_scale_groups,
        marker_offset_groups=marker_offset_groups,
        frame_offset_groups=frame_offset_groups, enable_fd=True)

    for cost in (cost_jac, cost_fd):
        cost.add_marker_bilevel_cost_term(
            '/markerset/R.Shoulder', osim.Vec3(0.3, 0, 0), weight=2.0,
            offset_group_index=0)
        cost.add_frame_bilevel_cost_term(
            '/bodyset/pelvis', osim.Vec3(0.3, 0.1, -0.2),
            osim.Quaternion(0.9, 0.1, 0.2, 0.3),
            position_weight=2.0, orientation_weight=1.5,
            offset_group_index=0)

    nq = len(cost_jac.mc.coordinate_q_indexes)
    q = ca.SX.sym('q', nq)
    s = ca.SX.sym('s', 3)
    mo = ca.SX.sym('mo', 3)
    fo = ca.SX.sym('fo', 3)
    x = ca.vertcat(q, s, mo, fo)

    J_jac = ca.Function('J_jac', [x],
                        [ca.jacobian(cost_jac(CostInput(q, s, mo, fo)), x)])
    J_fd = ca.Function('J_fd', [x],
                       [ca.jacobian(cost_fd(CostInput(q, s, mo, fo)), x)])

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


#############################
# TERM JACOBIANS VERSUS FD  #
#############################

def _chain_with_rotated_offset_frames():
    model = osim.Model()
    model.setName('offset_chain')
    previous = model.getGround()
    for i in range(3):
        body = osim.Body(f'b{i}', 1.0, osim.Vec3(0), osim.Inertia(1, 1, 1, 0, 0, 0))
        model.addBody(body)
        joint = osim.PinJoint(f'j{i}', previous, osim.Vec3(0.2 * i, 0, 0),
                              osim.Vec3(0), body, osim.Vec3(0, -0.3, 0), osim.Vec3(0))
        joint.updCoordinate().setName(f'q{i}')
        model.addJoint(joint)
        previous = body
        model.addMarker(osim.Marker(f'm{i}', body, osim.Vec3(0.05, 0.1, -0.02)))
        body.addComponent(osim.PhysicalOffsetFrame(
            f'off{i}', body,
            osim.Transform(osim.Rotation(0.4 + 0.1 * i, osim.Vec3(0.3, 0.5, 0.8)),
                           osim.Vec3(0.01, 0.02, -0.03))))
    model.finalizeConnections()
    return model


def _coordinate_jacobian_by_finite_difference(mc, term, step=1e-6):
    indexes = mc.coordinate_q_indexes
    q = np.array(mc.state.getQ().to_numpy())
    jacobian = np.zeros(len(indexes))
    for column, index in enumerate(indexes):
        errors = []
        for sign in (1.0, -1.0):
            perturbed = q.copy()
            perturbed[index] += sign * step
            mc.state.setQ(osim.Vector.createFromMat(perturbed))
            mc.model.realizePosition(mc.state)
            errors.append(term.calc_error(mc.state))
        jacobian[column] = (errors[0] - errors[1]) / (2.0 * step)
    mc.state.setQ(osim.Vector.createFromMat(q))
    mc.model.realizePosition(mc.state)
    return jacobian


def test_frame_tracking_term_jacobian_matches_finite_differences():
    mc = ModelCache(_chain_with_rotated_offset_frames())
    term = FrameTrackingTerm(mc)
    references = [
        # Reference position, and reference orientation as an unnormalized quaternion.
        ([0.13, 0.17, 0.04], [0.19, -0.20, 0.95, 0.16]),
        ([-0.20, 0.14, -0.19], [0.51, -0.38, -0.69, -0.34]),
        ([-0.08, -0.03, -0.19], [-0.79, -0.46, -0.35, -0.20]),
    ]
    for i, (position, quaternion) in enumerate(references):
        quaternion = np.array(quaternion) / np.linalg.norm(quaternion)
        term.add_frame(f'/bodyset/b{i}/off{i}', osim.Vec3(*position),
                       osim.Quaternion(*[float(v) for v in quaternion]),
                       position_weight=1.0 + 0.1 * i,
                       orientation_weight=0.5 + 0.2 * i)

    q = np.zeros(mc.state.getNQ())
    q[mc.coordinate_q_indexes] = [-0.30, 0.14, 0.12]
    mc.state.setQ(osim.Vector.createFromMat(q))
    mc.model.realizePosition(mc.state)

    analytic = term.calc_jacobian(mc.state)[0][0, :]
    numeric = _coordinate_jacobian_by_finite_difference(mc, term)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_frame_bilevel_term_coordinate_jacobian_matches_finite_differences():
    model = _chain_with_rotated_offset_frames()
    mc = ModelCache(model)
    bodies = [mc.model.getBodySet().get(i) for i in range(mc.model.getNumBodies())]
    for body in bodies:
        mc.add_parameter_group(BodyScaleGroup([body.getAbsolutePathString()],
                                              [int(body.getMobilizedBodyIndex())]))

    term = FrameBilevelTerm(mc)
    references = [
        # Reference position, and reference orientation as an unnormalized quaternion.
        ([-0.08, -0.03, 0.13], [0.21, 0.51, 0.20, -0.81]),
        ([0.02, -0.07, 0.12], [0.78, 0.49, 0.39, 0.04]),
        ([-0.12, -0.10, 0.10], [-0.59, 0.73, 0.05, -0.36]),
    ]
    for i, (position, quaternion) in enumerate(references):
        quaternion = np.array(quaternion) / np.linalg.norm(quaternion)
        term.add_frame(f'/bodyset/b{i}/off{i}', osim.Vec3(*position),
                       osim.Quaternion(*[float(v) for v in quaternion]),
                       position_weight=1.0, orientation_weight=0.7)

    # One XYZ scale triplet per body scale group, none of them uniform.
    scales = np.array([0.96, 1.02, 1.10, 1.09, 1.05, 1.01, 0.96, 0.93, 1.09])
    mc.set_scaled_mobilizer_frame_positions(mc.state, scales)
    term.apply_state(scales, np.zeros(0))
    q = np.zeros(mc.state.getNQ())
    q[mc.coordinate_q_indexes] = [0.01, -0.31, 0.10]
    mc.state.setQ(osim.Vector.createFromMat(q))
    mc.model.realizePosition(mc.state)

    analytic = term.calc_jacobian(mc.state)[0][0, :]
    numeric = _coordinate_jacobian_by_finite_difference(mc, term)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_marker_tracking_term_jacobian_matches_finite_differences():
    mc = ModelCache(_chain_with_rotated_offset_frames())
    term = MarkerTrackingTerm(mc)
    references = [
        [-0.10, -0.08, 0.13],
        [-0.16, 0.04, 0.09],
        [-0.12, -0.18, -0.09],
    ]
    for i, position in enumerate(references):
        term.add_marker(f'/markerset/m{i}', osim.Vec3(*position),
                        weight=1.0 + 0.3 * i)
    q = np.zeros(mc.state.getNQ())
    q[mc.coordinate_q_indexes] = [0.13, 0.05, -0.28]
    mc.state.setQ(osim.Vector.createFromMat(q))
    mc.model.realizePosition(mc.state)

    analytic = term.calc_jacobian(mc.state)[0][0, :]
    numeric = _coordinate_jacobian_by_finite_difference(mc, term)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_orientation_error_matches_the_quaternion_definition():
    mc = ModelCache(_chain_with_rotated_offset_frames())
    term = FrameTrackingTerm(mc)
    references = []
    for i, quaternion in enumerate([[0.61, -0.76, 0.13, -0.17],
                                    [-0.22, -0.10, -0.96, -0.11],
                                    [-0.25, 0.96, 0.07, -0.10]]):
        quaternion = np.array(quaternion) / np.linalg.norm(quaternion)
        references.append(quaternion)
        term.add_frame(f'/bodyset/b{i}/off{i}', osim.Vec3(0),
                       osim.Quaternion(*[float(v) for v in quaternion]),
                       position_weight=0.0, orientation_weight=1.0)

    q = np.zeros(mc.state.getNQ())
    q[mc.coordinate_q_indexes] = [-0.08, 0.10, 0.29]
    mc.state.setQ(osim.Vector.createFromMat(q))
    mc.model.realizePosition(mc.state)

    # position_weight is zero, so calc_error is the orientation error alone.
    quaternion_error = 0.0
    for i, reference in enumerate(references):
        frame = osim.PhysicalFrame.safeDownCast(
            mc.model.getComponent(f'/bodyset/b{i}/off{i}'))
        quaternion = frame.getRotationInGround(mc.state).convertRotationToQuaternion()
        eps = np.array([quaternion.get(k) for k in range(4)])
        quaternion_error += 1.0 - np.square(np.dot(eps, reference))

    assert term.calc_error(mc.state) == pytest.approx(
        quaternion_error, rel=1e-12, abs=1e-14)


#########################
# PER-COMPONENT WEIGHTS #
#########################

def test_per_component_weights_multiply_the_overall_weights():
    cost = TrackingCost(position_weight=2.0, orientation_weight=3.0,
                        marker_weights={'/markerset/m0': 7.0},
                        frame_position_weights={'/bodyset/b0': 5.0},
                        frame_orientation_weights={'/bodyset/b1': 9.0})
    assert cost.get_marker_weight('/markerset/m0') == 2.0 * 7.0
    assert cost.get_frame_weights('/bodyset/b0') == (2.0 * 5.0, 3.0)
    assert cost.get_frame_weights('/bodyset/b1') == (2.0, 3.0 * 9.0)
    assert cost.get_marker_weight('/markerset/other') == 2.0
    assert cost.get_frame_weights('/bodyset/other') == (2.0, 3.0)


def test_a_unit_per_component_weight_changes_nothing():
    plain = TrackingCost(position_weight=2.0, orientation_weight=3.0)
    unit = TrackingCost(position_weight=2.0, orientation_weight=3.0,
                        marker_weights={'/markerset/m0': 1.0},
                        frame_position_weights={'/bodyset/b0': 1.0},
                        frame_orientation_weights={'/bodyset/b0': 1.0})
    assert unit.get_marker_weight('/markerset/m0') == plain.get_marker_weight(
        '/markerset/m0')
    assert unit.get_frame_weights('/bodyset/b0') == plain.get_frame_weights(
        '/bodyset/b0')


def test_weight_mappings_are_copied_at_construction():
    weights = {'/markerset/m0': 7.0}
    cost = TrackingCost(marker_weights=weights)
    weights['/markerset/m0'] = 99.0
    assert cost.get_marker_weight('/markerset/m0') == 7.0


def test_bilevel_cost_shares_the_weight_behavior():
    cost = BilevelCost(position_weight=2.0, orientation_weight=3.0,
                       marker_weights={'/markerset/m0': 7.0})
    assert cost.get_marker_weight('/markerset/m0') == 2.0 * 7.0
    assert cost.get_frame_weights('/bodyset/other') == (2.0, 3.0)


def test_set_marker_weight_rejects_bad_input(tmp_path):
    model = create_beam_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model, knot_interval=0.1)
    with pytest.raises(ValueError, match='non-negative'):
        solver.set_marker_weight('/markerset/marker1', -1.0)
    with pytest.raises(ValueError, match='no component at that path'):
        solver.set_marker_weight('/markerset/not_a_marker', 1.0)


def test_set_frame_weights_scales_position_and_orientation():
    model = create_beam_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model, knot_interval=0.1,
                                     position_weight=2.0, orientation_weight=3.0)
    frame = '/bodyset/torso'
    solver.set_frame_weights(frame, frame_position_weight=5.0,
                             frame_orientation_weight=9.0)

    cost = TrackingCost(solver.position_weight, solver.orientation_weight,
                        solver.marker_weights, solver.frame_position_weights,
                        solver.frame_orientation_weights)
    assert cost.get_frame_weights(frame) == (2.0 * 5.0, 3.0 * 9.0)


def test_set_frame_weights_requires_both_weights():
    model = create_beam_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(model, knot_interval=0.1)
    with pytest.raises(TypeError):
        solver.set_frame_weights('/bodyset/torso')
    with pytest.raises(TypeError):
        solver.set_frame_weights('/bodyset/torso', frame_position_weight=1.0)
    with pytest.raises(ValueError, match='non-negative'):
        solver.set_frame_weights('/bodyset/torso', frame_position_weight=1.0,
                                 frame_orientation_weight=-1.0)
    with pytest.raises(ValueError, match='non-negative'):
        solver.set_frame_weights('/bodyset/torso', frame_position_weight=-1.0,
                                 frame_orientation_weight=1.0)
    with pytest.raises(ValueError, match='no component at that path'):
        solver.set_frame_weights('/bodyset/not_a_body', frame_position_weight=1.0,
                                 frame_orientation_weight=1.0)
