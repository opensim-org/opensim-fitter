"""
Unit tests for TrackingCostFunction and BilevelCostFunction in
src/osimfit/callbacks.py.

Wiring tests verify that the public ``add_*`` methods register entries
on the correct internal sub-cost. Behavioral tests evaluate the CasADi
callback at known reference points (empty cost, marker at reference,
marker reachable only by scaling) and check the values. Jacobian tests
compare the analytical Jacobian to the finite-difference Jacobian
produced by the same callback under ``opts={'enable_fd': True}``.

The frame-based tests use ``unscaled_generic.osim`` because the fixture
model has named PhysicalOffsetFrames but no markers. The marker- and
scale-based tests use a tiny in-memory slider rig so the Jacobians have
non-trivial dependence on every input.
"""

from pathlib import Path

import numpy as np
import casadi as ca
import opensim as osim
import pytest

from osimfit.callbacks import BilevelCostFunction, TrackingCostFunction


MODEL_FPATH = str(Path(__file__).parent / 'unscaled_generic.osim')


# -- helpers ---------------------------------------------------------------

def _build_test_rig():
    """
    One body sliding along ground X with two markers in the body frame:
    m0 at the origin, m1 at (0.5, 0, 0). The q axis is the world X
    translation; scaling the body X axis stretches m1's world position.
    """
    model = osim.Model()
    model.setName('test_rig')
    ground = model.getGround()
    body = osim.Body('rig_body', 1.0, osim.Vec3(0), osim.Inertia(1))
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


# -- TrackingCostFunction: wiring -----------------------------------------

def test_tracking_cost_function_constructs_marker_and_frame_subcosts():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    assert cost.marker_cost is not None
    assert cost.frame_cost is not None


def test_tracking_cost_function_add_marker_registers_in_marker_cost():
    model = _build_test_rig()
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    assert len(cost.marker_cost.markers) == 1
    assert cost.marker_cost.mobod_indexes.size() == 1
    # Sanity: frame_cost stays empty.
    assert len(cost.frame_cost.frames) == 0


def test_tracking_cost_function_add_frame_registers_in_frame_cost():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    cost.add_frame_tracking_cost(
        '/bodyset/pelvis/pelvis', osim.Vec3(0), osim.Quaternion())
    assert len(cost.frame_cost.frames) == 1
    assert cost.frame_cost.mobod_indexes.size() == 1
    # Sanity: marker_cost stays empty.
    assert len(cost.marker_cost.markers) == 0


# -- TrackingCostFunction: evaluation -------------------------------------

def test_tracking_cost_function_empty_eval_is_zero():
    model = osim.Model(MODEL_FPATH)
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    x = ca.DM.zeros(len(cost.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_at_reference_yields_zero():
    model = _build_test_rig()
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    # At q=0, m0 sits at the world origin.
    cost.add_marker_tracking_cost('/markerset/m0', osim.Vec3(0))
    x = ca.DM.zeros(len(cost.q_indexes))
    assert float(cost(x)) == pytest.approx(0.0, abs=1e-12)


def test_tracking_cost_function_marker_off_reference_yields_squared_error():
    model = _build_test_rig()
    model.initSystem()
    cost = TrackingCostFunction('cost', model)
    # m0 at world (0.1, 0, 0) when q=0.1; reference at the origin.
    cost.add_marker_tracking_cost(
        '/markerset/m0', osim.Vec3(0.0, 0, 0), weight=1.0)
    x = ca.DM([0.1])
    assert float(cost(x)) == pytest.approx(0.01, abs=1e-9)


# -- TrackingCostFunction: Jacobian ---------------------------------------

def test_tracking_cost_function_jacobian_matches_finite_difference():
    model = _build_test_rig()
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


# -- BilevelCostFunction: wiring ------------------------------------------

def test_bilevel_cost_function_constructs_marker_subcost():
    model = _build_test_rig()
    model.initSystem()
    cost = BilevelCostFunction('cost', model, scale_indexes=[1])
    assert cost.marker_cost is not None
    assert cost.scale_indexes == [1]


def test_bilevel_cost_function_add_marker_registers_in_marker_cost():
    model = _build_test_rig()
    model.initSystem()
    cost = BilevelCostFunction('cost', model, scale_indexes=[1])
    cost.add_marker_bilevel_cost('/markerset/m0', osim.Vec3(0))
    assert cost.marker_cost.mobod_indexes.size() == 1


# -- BilevelCostFunction: pack_scales -------------------------------------

def test_bilevel_pack_scales_writes_to_mobod_indexes_keeps_ground_at_one():
    model = _build_test_rig()
    model.initSystem()
    cost = BilevelCostFunction('cost', model, scale_indexes=[1])
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


# -- BilevelCostFunction: evaluation --------------------------------------

def test_bilevel_cost_function_empty_eval_is_zero():
    model = _build_test_rig()
    model.initSystem()
    cost = BilevelCostFunction('cost', model, scale_indexes=[1])
    q = ca.DM.zeros(len(cost.q_indexes))
    s = ca.DM.ones(3)
    assert float(cost(q, s)) == pytest.approx(0.0, abs=1e-12)


def test_bilevel_cost_function_scaling_changes_marker_world_position():
    """
    m1 lives at body offset (0.5, 0, 0); reference at (1.0, 0, 0). Scaling
    the body X by 2.0 moves m1 to world (1.0, 0, 0) — zero error.
    """
    model = _build_test_rig()
    model.initSystem()
    cost = BilevelCostFunction('cost', model, scale_indexes=[1])
    cost.add_marker_bilevel_cost('/markerset/m1', osim.Vec3(1.0, 0, 0))

    q = ca.DM.zeros(len(cost.q_indexes))
    s_unit = ca.DM([1.0, 1.0, 1.0])
    s_scaled = ca.DM([2.0, 1.0, 1.0])

    assert float(cost(q, s_unit)) == pytest.approx(0.25, abs=1e-9)
    assert float(cost(q, s_scaled)) == pytest.approx(0.0, abs=1e-9)


# -- BilevelCostFunction: Jacobian ----------------------------------------

def test_bilevel_cost_function_jacobians_match_finite_difference():
    model = _build_test_rig()
    model.initSystem()
    cost_jac = BilevelCostFunction('cost_jac', model, scale_indexes=[1])
    cost_fd = BilevelCostFunction(
        'cost_fd', model, scale_indexes=[1], opts={'enable_fd': True})

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
