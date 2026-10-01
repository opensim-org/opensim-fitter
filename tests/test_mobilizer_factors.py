"""
Tests for the mobilizer-geometry `Parameter`s, `EllipsoidRadii` and `BeamLength`, whose
optimization variables are dimensionless factors on each joint's own baseline geometry
rather than absolute values.

The factor semantics are what let one parameter span joints whose baselines differ:
a spine's lumbar, thoracic, and cervical beams scale in proportion instead of being
forced equal, and a left-right pair of ellipsoid joints mirrored by negated radii stays
mirrored under a single positive factor.
"""

import pytest
import numpy as np
import opensim as osim

from osimfit.model import (ModelCache, BodyScale, EllipsoidRadii, EllipsoidRadiiGroup,
                           BeamLength, BeamLengthGroup)
from osimfit.bounds import Bounds
from osimfit.solvers import SplinedKinematicsSolver, Solution

# Baseline mobilizer geometry of the test models.
RADII = (0.05, 0.03, 0.04)
BEAM_LENGTH = 0.35

ELBOW = '/jointset/elbow_r'
SHOULDER = '/jointset/shoulder_r'


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


def create_ellipsoid_model():
    """
    The `create_beam_model` layout with the beam replaced by an `EllipsoidJoint`:

        ground --Pin-- torso --Ellipsoid-- humerus
    """
    model = osim.Model()
    model.setName('ellipsoid')

    torso = osim.Body('torso', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    humerus = osim.Body('humerus', 1.0, osim.Vec3(0), osim.Inertia(0.1))
    for body in (torso, humerus):
        model.addBody(body)

    model.addJoint(osim.PinJoint('ground_torso', model.getGround(), torso))
    model.addJoint(osim.EllipsoidJoint(
        'shoulder_r', torso, osim.Vec3(0.1, 0.2, 0.3), osim.Vec3(0.2, -0.1, 0.3),
        humerus, osim.Vec3(0.01, 0.02, 0.03), osim.Vec3(0.1, 0.2, -0.1),
        osim.Vec3(*RADII)))

    model.addMarker(osim.Marker('torso_marker', torso, osim.Vec3(0.1, 0.05, 0.0)))
    for i, location in enumerate([osim.Vec3(0.15, 0.0, 0.0), osim.Vec3(0.0, 0.15, 0.0),
                                  osim.Vec3(0.0, 0.0, 0.15)]):
        model.addMarker(osim.Marker(f'humerus_{i}', humerus, location))

    model.finalizeConnections()
    return model


def create_mirrored_ellipsoid_model():
    """
    Two ellipsoid joints in ground whose radii are mirrored by negation, as a
    left-right symmetric pair is in the athlete model.
    """
    model = osim.Model()
    model.setName('mirrored_ellipsoid')
    for side, sign in (('r', 1.0), ('l', -1.0)):
        body = osim.Body(f'humerus_{side}', 1.0, osim.Vec3(0), osim.Inertia(0.1))
        model.addBody(body)
        model.addJoint(osim.EllipsoidJoint(
            f'shoulder_{side}', model.getGround(), osim.Vec3(0), osim.Vec3(0),
            body, osim.Vec3(0), osim.Vec3(0),
            osim.Vec3(*[sign * r for r in RADII])))
        model.addMarker(osim.Marker(f'm_{side}', body, osim.Vec3(0.1, 0.0, 0.0)))
    model.finalizeConnections()
    return model


def create_spine_model():
    """
    Three cantilever beams in series, each with a different baseline length, as the
    lumbar, thoracic, and cervical segments of a spine are.
    """
    model = osim.Model()
    model.setName('spine')
    lengths = {'lumbar': 0.175, 'thorax': 0.275, 'cervical': 0.075}
    parent = model.getGround()
    for name, length in lengths.items():
        body = osim.Body(name, 1.0, osim.Vec3(0), osim.Inertia(0.1))
        model.addBody(body)
        model.addJoint(osim.CantileverFreeBeamJoint(
            name, parent, osim.Vec3(0), osim.Vec3(0), body, osim.Vec3(0),
            osim.Vec3(0), length))
        model.addMarker(osim.Marker(f'm_{name}', body, osim.Vec3(0.1, 0.0, 0.0)))
        parent = body
    model.finalizeConnections()
    return model


SPINE_LENGTHS = {'/jointset/lumbar': 0.175,
                 '/jointset/thorax': 0.275,
                 '/jointset/cervical': 0.075}


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


def marker_trial(name: str, trc_path: str):
    """
    A `Trial` over the markers in `trc_path`, mapping the labels the TRC adapter writes
    back to the component paths the solver resolves.
    """
    from osimfit.data_sources import MarkerSource, Trial
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}
    return Trial(name, [MarkerSource('markers', trc_path, label_map=label_map)])


def radii_of(model: osim.Model, joint_path: str) -> np.ndarray:
    return model.getComponent(joint_path).get_radii_x_y_z().to_numpy()


def length_of(model: osim.Model, joint_path: str) -> float:
    return model.getComponent(joint_path).get_beam_length()


##############
# VALIDATION #
##############

@pytest.mark.parametrize('lower_bound', [-0.5, 0.0])
def test_factors_require_a_positive_lower_bound(lower_bound):
    """
    A non-positive factor would collapse or invert the geometry, so the bound is
    rejected up front rather than left to produce a degenerate model mid-solve.
    """
    model = create_ellipsoid_model()
    model.initSystem()
    mc = ModelCache(model)
    with pytest.raises(ValueError, match='factors must be positive'):
        EllipsoidRadii(SHOULDER, Bounds(lower_bound, 2.0), np.ones(3)).validate(mc)

    model = create_beam_model()
    model.initSystem()
    mc = ModelCache(model)
    with pytest.raises(ValueError, match='factors must be positive'):
        BeamLength(ELBOW, Bounds(lower_bound, 2.0), 1.0).validate(mc)


def test_factors_accept_a_joint_with_negative_baseline_radii():
    """
    The left joint of a mirrored pair has negative baseline radii, which an absolute
    parameter could not bound positively. A factor can, since the sign lives in the
    baseline rather than in the variable.
    """
    model = create_mirrored_ellipsoid_model()
    model.initSystem()
    mc = ModelCache(model)
    np.testing.assert_allclose(radii_of(model, '/jointset/shoulder_l'),
                               -np.array(RADII))
    EllipsoidRadii(['/jointset/shoulder_r', '/jointset/shoulder_l'],
                   Bounds(0.5, 2.0), np.ones(3)).validate(mc)


def test_rejects_the_wrong_joint_type():
    model = create_beam_model()
    model.initSystem()
    mc = ModelCache(model)
    with pytest.raises(ValueError, match='not an EllipsoidJoint'):
        EllipsoidRadii(ELBOW, Bounds(0.5, 2.0), np.ones(3)).validate(mc)
    with pytest.raises(ValueError, match='not a CantileverFreeBeamJoint'):
        BeamLength('/jointset/ground_torso', Bounds(0.5, 2.0), 1.0).validate(mc)


#####################
# BASELINE CACHING  #
#####################

def test_add_parameter_group_caches_each_joints_baseline():
    model = create_spine_model()
    model.initSystem()
    mc = ModelCache(model)
    mc.add_parameter_group(BeamLengthGroup(list(SPINE_LENGTHS)))

    assert mc.beam_length_group_baselines == [list(SPINE_LENGTHS.values())]


def test_add_parameter_group_caches_mirrored_radii_baselines():
    model = create_mirrored_ellipsoid_model()
    model.initSystem()
    mc = ModelCache(model)
    mc.add_parameter_group(EllipsoidRadiiGroup(
        ['/jointset/shoulder_r', '/jointset/shoulder_l']))

    baselines = mc.ellipsoid_radii_group_baselines[0]
    np.testing.assert_allclose(baselines[0], RADII)
    np.testing.assert_allclose(baselines[1], -np.array(RADII))


##################
# STATE SETTERS  #
##################

def test_set_beam_lengths_scales_each_joint_from_its_own_baseline():
    """
    One factor shared across beams of differing nominal length must scale each in
    proportion, not force them to a common value.
    """
    model = create_spine_model()
    model.initSystem()
    mc = ModelCache(model)
    mc.add_parameter_group(BeamLengthGroup(list(SPINE_LENGTHS)))

    mc.set_beam_lengths(mc.state, np.array([1.2]))
    for path, baseline in SPINE_LENGTHS.items():
        assert mc.model.getComponent(path).getLength(mc.state) == \
            pytest.approx(baseline * 1.2)


def test_set_beam_lengths_is_absolute_not_compounding():
    model = create_beam_model()
    model.initSystem()
    mc = ModelCache(model)
    mc.add_parameter_group(BeamLengthGroup([ELBOW]))

    for _ in range(3):
        mc.set_beam_lengths(mc.state, np.array([1.5]))
    assert mc.model.getComponent(ELBOW).getLength(mc.state) == \
        pytest.approx(BEAM_LENGTH * 1.5)


def test_set_ellipsoid_radii_preserves_a_mirrored_pair():
    """
    A single positive factor applied to a negated baseline keeps the pair mirrored,
    which is the behavior an absolute shared value could not express.
    """
    model = create_mirrored_ellipsoid_model()
    model.initSystem()
    mc = ModelCache(model)
    mc.add_parameter_group(EllipsoidRadiiGroup(
        ['/jointset/shoulder_r', '/jointset/shoulder_l']))

    mc.set_ellipsoid_radii(mc.state, np.array([1.5, 0.5, 2.0]))
    factors = np.array([1.5, 0.5, 2.0])
    np.testing.assert_allclose(
        mc.model.getComponent('/jointset/shoulder_r').getRadii(mc.state).to_numpy(),
        np.array(RADII) * factors)
    np.testing.assert_allclose(
        mc.model.getComponent('/jointset/shoulder_l').getRadii(mc.state).to_numpy(),
        -np.array(RADII) * factors)


def test_identity_factors_leave_the_state_at_baseline():
    model = create_ellipsoid_model()
    model.initSystem()
    mc = ModelCache(model)
    mc.add_parameter_group(EllipsoidRadiiGroup([SHOULDER]))

    mc.set_ellipsoid_radii(mc.state, np.ones(3))
    np.testing.assert_allclose(
        mc.model.getComponent(SHOULDER).getRadii(mc.state).to_numpy(), RADII)


##################
# MODEL APPLYING #
##################

def test_apply_to_model_is_multiplicative():
    model = create_ellipsoid_model()
    model.initSystem()
    EllipsoidRadii(SHOULDER, Bounds(0.5, 2.0),
                   np.array([1.5, 2.0, 0.5])).apply_to_model(model)
    np.testing.assert_allclose(radii_of(model, SHOULDER),
                               np.array(RADII) * np.array([1.5, 2.0, 0.5]))

    model = create_beam_model()
    model.initSystem()
    BeamLength(ELBOW, Bounds(0.5, 2.0), 2.0).apply_to_model(model)
    assert length_of(model, ELBOW) == pytest.approx(BEAM_LENGTH * 2.0)


def test_get_and_apply_ellipsoid_joint_radii_round_trip():
    model = create_ellipsoid_model()
    model.initSystem()

    saved = ModelCache.get_ellipsoid_joint_radii(model)
    assert set(saved) == {SHOULDER}

    model.getComponent(SHOULDER).set_radii_x_y_z(osim.Vec3(9.0, 9.0, 9.0))
    ModelCache.apply_ellipsoid_joint_radii(model, saved)
    np.testing.assert_allclose(radii_of(model, SHOULDER), RADII)


def test_model_scale_resizes_radii_without_the_restore():
    """
    Document the behavior `update_model` compensates for: `EllipsoidJoint::extendScale`
    multiplies the radii by the parent frame's body scale factors, and restoring the
    saved radii undoes exactly that.
    """
    model = create_ellipsoid_model()
    state = model.initSystem()
    saved = ModelCache.get_ellipsoid_joint_radii(model)

    scaleset = osim.ScaleSet()
    scale = osim.Scale()
    scale.setSegmentName('torso')
    scale.setScaleFactors(osim.Vec3(2.0, 2.0, 2.0))
    scaleset.cloneAndAppend(scale)
    scaleset.get(0).setName('torso')
    model.scale(state, scaleset, True)
    np.testing.assert_allclose(radii_of(model, SHOULDER), np.array(RADII) * 2.0)

    ModelCache.apply_ellipsoid_joint_radii(model, saved)
    np.testing.assert_allclose(radii_of(model, SHOULDER), RADII)


def test_update_model_keeps_radii_independent_of_body_scales():
    """
    This fitter treats mobilizer geometry as independent of body scaling, so an
    `EllipsoidRadii` factor must apply on top of the baseline radii rather than on top
    of the radii `Model::scale()` already resized.
    """
    solver = SplinedKinematicsSolver(create_ellipsoid_model())
    body_scale = BodyScale('/bodyset/torso', Bounds(0.5, 2.0), np.full(3, 2.0))
    radii = EllipsoidRadii(SHOULDER, Bounds(0.5, 2.0), np.full(3, 1.5))
    solver.add_parameter(body_scale)
    solver.add_parameter(radii)

    updated = solver.update_model(create_ellipsoid_model(),
                                 Solution(parameters=[body_scale, radii]))
    np.testing.assert_allclose(radii_of(updated, SHOULDER),
                               np.array(RADII) * 1.5)


def test_update_model_keeps_beam_length_independent_of_body_scales():
    """
    `CantileverFreeBeamJoint` does not override extendScale, so `Model::scale()` leaves
    'beam_length' alone and a `BeamLength` factor applies cleanly on top of the
    baseline. This is why `update_model` saves and restores the ellipsoid radii but
    needs no counterpart for the beam length. The body scaled here is the beam's own
    parent, which is the frame `EllipsoidJoint::extendScale` reads its factors from.
    """
    solver = SplinedKinematicsSolver(create_beam_model())
    body_scale = BodyScale('/bodyset/torso', Bounds(0.5, 2.0), np.full(3, 2.0))
    length = BeamLength(ELBOW, Bounds(0.5, 2.0), 1.5)
    solver.add_parameter(body_scale)
    solver.add_parameter(length)

    updated = solver.update_model(create_beam_model(),
                                  Solution(parameters=[body_scale, length]))
    assert length_of(updated, ELBOW) == pytest.approx(BEAM_LENGTH * 1.5)


def test_update_model_without_a_radii_parameter_preserves_the_radii():
    """
    A solve that scales a body but does not optimize the radii must still leave the
    radii at their baseline, since the restore is unconditional.
    """
    solver = SplinedKinematicsSolver(create_ellipsoid_model())
    body_scale = BodyScale('/bodyset/torso', Bounds(0.5, 2.0), np.full(3, 2.0))
    solver.add_parameter(body_scale)

    updated = solver.update_model(create_ellipsoid_model(),
                                  Solution(parameters=[body_scale]))
    np.testing.assert_allclose(radii_of(updated, SHOULDER), RADII)


##############
# END-TO-END #
##############

def test_solver_recovers_a_beam_length_factor(tmp_path):
    """
    Synthesize marker data from a model whose beam is 1.3x its nominal length, then
    solve against the nominal model while optimizing the beam-length factor. The
    recovered factor must match the truth and bake into the updated model.
    """
    true_factor = 1.3

    truth = create_beam_model()
    truth.getComponent(ELBOW).set_beam_length(BEAM_LENGTH * true_factor)
    truth.finalizeConnections()
    trc_path = str(tmp_path / 'markers.trc')
    create_prescribed_markers(truth, trc_path)

    model = create_beam_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(
        model, convergence_tolerance=1e-6, knot_interval=0.1, position_weight=5.0)
    solver.add_trial(marker_trial('beam', trc_path))
    solver.add_parameter(BeamLength(ELBOW, Bounds(0.5, 2.0), 1.0))

    solution = solver.solve()

    lengths = [p for p in solution.parameters if isinstance(p, BeamLength)]
    assert len(lengths) == 1
    np.testing.assert_allclose(lengths[0].value, [true_factor], atol=0.01)

    updated = solver.update_model(create_beam_model(), solution)
    assert length_of(updated, ELBOW) == \
        pytest.approx(BEAM_LENGTH * lengths[0].value[0])


def test_solver_recovers_ellipsoid_radii_factors(tmp_path):
    """
    Synthesize marker data from a model whose ellipsoid radii are scaled by known
    factors, then solve against the nominal model while optimizing those factors.
    """
    true_factors = np.array([1.4, 0.7, 1.2])

    truth = create_ellipsoid_model()
    truth.getComponent(SHOULDER).set_radii_x_y_z(
        osim.Vec3(*[float(v) for v in np.array(RADII) * true_factors]))
    truth.finalizeConnections()
    trc_path = str(tmp_path / 'markers.trc')
    create_prescribed_markers(truth, trc_path)

    model = create_ellipsoid_model()
    model.initSystem()
    solver = SplinedKinematicsSolver(
        model, convergence_tolerance=1e-6, knot_interval=0.1, position_weight=5.0)
    solver.add_trial(marker_trial('ellipsoid', trc_path))
    solver.add_parameter(EllipsoidRadii(SHOULDER, Bounds(0.2, 3.0), np.ones(3)))

    solution = solver.solve()

    radii = [p for p in solution.parameters if isinstance(p, EllipsoidRadii)]
    assert len(radii) == 1
    np.testing.assert_allclose(radii[0].value, true_factors, atol=0.02)

    updated = solver.update_model(create_ellipsoid_model(), solution)
    np.testing.assert_allclose(radii_of(updated, SHOULDER),
                               np.array(RADII) * radii[0].value)
