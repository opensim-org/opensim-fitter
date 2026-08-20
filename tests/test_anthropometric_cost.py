"""
Tests for the anthropometric body-scale regularization cost and the analytical
station-position-wrt-body-scale Jacobian it is built from.
"""

import pytest
import numpy as np
import casadi as ca
import opensim as osim

from osimfit.model import ModelCache, BodyScale
from osimfit.bounds import Bounds
from osimfit.costs import CostInput, AnthropometricRegularizationCost
from osimfit.scaling import Axis, AnthropometricMeasurement


###########
# HELPERS #
###########

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


def cache_group_joints(mc):
    """Populate inboard/outboard joints on each group (as BilevelCostFunction does)."""
    for group in mc.body_scale_groups:
        group.outboard_joints = [
            mc.get_joint_for_mobilized_body_index(int(k))
            for k in group.mobod_indexes]
        group.inboard_joints = [
            mc.get_joint_for_mobilized_body_index(c)
            for k in group.mobod_indexes
            for c in mc.children_of[int(k)]]


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


#########
# TESTS #
#########

def test_station_position_jacobian_matches_finite_difference():
    model = create_two_link_model()
    mc = ModelCache(model)
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    cache_group_joints(mc)
    mc.model.realizePosition(mc.state)

    n = 3 * len(mc.body_scale_groups)
    for station_path in ('/S0', '/S1'):
        station = osim.Station.safeDownCast(mc.model.getComponent(station_path))
        base_frame = osim.PhysicalFrame.safeDownCast(
            station.getParentFrame().findBaseFrame())
        mobod = int(base_frame.getMobilizedBodyIndex())
        base_loc = station.findLocationInFrame(mc.state, base_frame).to_numpy()

        # Analytical Jacobian (computed at baseline before any state scaling).
        J = mc.calc_station_position_jacobian_wrt_body_scales(
            mc.state, mobod, base_frame, base_loc)

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


# The cost fits its distribution from the ANSUR II dataset, so measurement names must be
# real ANSUR labels. The station pairs are the synthetic model's stations — anatomical
# correctness is irrelevant here; we exercise the cost mechanics.
def _build_cost(label='stature', axis=Axis.YAxis, sex='female', weight=1.0):
    model = create_two_link_model()
    mc = ModelCache(model)
    register_body_scales(mc, ['/bodyset/b0', '/bodyset/b1'])
    measurements = {label: AnthropometricMeasurement('/S0', '/S1', axis)}
    cost = AnthropometricRegularizationCost(measurements, sex=sex, weight=weight)
    cost.initialize(mc)  # the solver does this in solve()
    n = 3 * len(mc.body_scale_groups)
    return cost, n


def _manual_cost(cost, s):
    """Independent numpy evaluation of the Mahalanobis penalty from the built maps."""
    measurements = []
    for (D, c), axis in zip(cost.displacement_maps, cost.axes):
        d = D @ s + c
        measurements.append(np.abs(d[axis]) if axis is not None else np.linalg.norm(d))
    residual = np.asarray(measurements) - cost.mean
    return cost.weight * 0.5 * residual @ cost.precision @ residual


def test_distribution_mean_is_in_meters():
    # A female stature is ~1.6 m; without the mm->m conversion it would be ~1600.
    cost, n = _build_cost(label='stature')
    assert 1.0 < cost.mean[0] < 2.5


def test_cost_matches_manual_mahalanobis():
    cost, n = _build_cost(label='stature', weight=2.0)
    for s in (np.ones(n), np.array([1.1, 1.0, 1.0, 0.9, 1.0, 1.0])):
        value = float(cost(CostInput(body_scales=ca.DM(s))))
        np.testing.assert_allclose(value, _manual_cost(cost, s), rtol=1e-9)


def test_cost_gradient_matches_finite_difference():
    cost, n = _build_cost(label='stature')
    s = ca.MX.sym('s', n)
    grad = ca.Function('grad', [s], [ca.gradient(cost(CostInput(body_scales=s)), s)])
    s0 = np.ones(n)
    g = np.array(grad(s0)).flatten()
    eps = 1e-6
    g_fd = np.zeros(n)
    for k in range(n):
        sp, sm = s0.copy(), s0.copy()
        sp[k] += eps
        sm[k] -= eps
        g_fd[k] = (float(cost(CostInput(body_scales=ca.DM(sp)))) -
                   float(cost(CostInput(body_scales=ca.DM(sm))))) / (2 * eps)
    np.testing.assert_allclose(g, g_fd, atol=1e-6)


def test_euclidean_measurement_builds_and_evaluates():
    cost, n = _build_cost(label='biacromialbreadth', axis=None)
    value = float(cost(CostInput(body_scales=ca.DM(np.ones(n)))))
    assert np.isfinite(value)


def test_evaluating_before_initialize_raises():
    measurements = {'stature': AnthropometricMeasurement('/S0', '/S1', Axis.YAxis)}
    cost = AnthropometricRegularizationCost(measurements, sex='female')
    with pytest.raises(RuntimeError, match='initialize'):
        cost(CostInput(body_scales=ca.DM(np.zeros(6))))
