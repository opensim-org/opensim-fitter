"""
Tests for the model utilities in `osimfit.utilities`.
"""

import numpy as np
import opensim as osim
import pytest

from osimfit.utilities import set_model_mass


###########
# HELPERS #
###########

def create_three_body_model(masses=(3.0, 5.0, 2.0)) -> osim.Model:
    """
    A chain of sliding bodies with distinct masses, inertias and mass centers, so a
    uniform scaling is distinguishable from any per-body rescaling.
    """
    model = osim.Model()
    model.setName('three_body')
    parent = model.getGround()
    for i, mass in enumerate(masses):
        # OpenSim forces a massless body's inertia to zero.
        inertia = (osim.Inertia(0.0) if mass == 0.0
                   else osim.Inertia(1.0 + i, 2.0 + i, 3.0 + i,
                                     0.1 * i, 0.2 * i, 0.3 * i))
        body = osim.Body(f'b{i}', mass, osim.Vec3(0.1 * (i + 1), 0.02, -0.03),
                         inertia)
        model.addBody(body)
        joint = osim.SliderJoint(f'j{i}', parent, body)
        joint.updCoordinate().setName(f'q{i}')
        model.addJoint(joint)
        parent = body
    model.finalizeConnections()
    model.initSystem()
    return model


def body_masses(model):
    bodyset = model.getBodySet()
    return [bodyset.get(i).getMass() for i in range(bodyset.getSize())]


def body_inertias(model):
    bodyset = model.getBodySet()
    return np.array([[bodyset.get(i).get_inertia().get(k) for k in range(6)]
                     for i in range(bodyset.getSize())])


def body_mass_centers(model):
    bodyset = model.getBodySet()
    return np.array([[bodyset.get(i).get_mass_center().get(k) for k in range(3)]
                     for i in range(bodyset.getSize())])


#########
# TESTS #
#########

def test_total_mass_matches_the_requested_value():
    model = create_three_body_model()
    set_model_mass(model, 30.0)
    assert sum(body_masses(model)) == pytest.approx(30.0, rel=1e-12)
    assert model.getTotalMass(model.initSystem()) == pytest.approx(30.0, rel=1e-12)


def test_mass_distribution_is_preserved():
    model = create_three_body_model()
    before = np.array(body_masses(model))
    set_model_mass(model, 30.0)
    after = np.array(body_masses(model))
    # Each body keeps the same share of the total.
    np.testing.assert_allclose(after / after.sum(), before / before.sum(),
                               rtol=1e-12, atol=0)


def test_inertia_scales_with_mass_and_mass_centers_do_not():
    model = create_three_body_model()
    inertia_before = body_inertias(model)
    centers_before = body_mass_centers(model)
    total_before = sum(body_masses(model))

    set_model_mass(model, 30.0)

    factor = 30.0 / total_before
    np.testing.assert_allclose(body_inertias(model), factor * inertia_before,
                               rtol=1e-12, atol=0)
    # Mass centers are positions on the body, so holding geometry fixed leaves them
    # unchanged.
    np.testing.assert_allclose(body_mass_centers(model), centers_before,
                               rtol=0, atol=0)


def test_setting_the_current_mass_is_a_no_op():
    model = create_three_body_model()
    masses_before = body_masses(model)
    inertia_before = body_inertias(model)
    set_model_mass(model, sum(masses_before))
    np.testing.assert_allclose(body_masses(model), masses_before, rtol=1e-12, atol=0)
    np.testing.assert_allclose(body_inertias(model), inertia_before,
                               rtol=1e-12, atol=0)


def test_massless_bodies_stay_massless():
    model = create_three_body_model(masses=(4.0, 0.0, 6.0))
    set_model_mass(model, 50.0)
    masses = body_masses(model)
    assert masses[1] == 0.0
    assert sum(masses) == pytest.approx(50.0, rel=1e-12)


def test_repeated_calls_are_idempotent_in_the_target():
    model = create_three_body_model()
    set_model_mass(model, 42.0)
    first = body_masses(model)
    set_model_mass(model, 42.0)
    np.testing.assert_allclose(body_masses(model), first, rtol=1e-12, atol=0)


def test_the_updated_mass_is_visible_to_the_model_system():
    model = create_three_body_model()
    set_model_mass(model, 12.5)
    # set_model_mass reinitializes the system, so a freshly fetched state agrees.
    assert model.getTotalMass(model.initSystem()) == pytest.approx(12.5, rel=1e-12)


@pytest.mark.parametrize('mass', [0.0, -1.0])
def test_non_positive_mass_is_rejected(mass):
    model = create_three_body_model()
    with pytest.raises(ValueError, match='positive'):
        set_model_mass(model, mass)


def test_a_model_with_no_mass_is_rejected():
    # Simbody refuses to build a chain whose terminal body is massless unless it is
    # welded, so the bodies here are welded; that makes a zero-mass model that can
    # actually be initialized, which the guard then rejects.
    model = osim.Model()
    model.setName('massless')
    parent = model.getGround()
    for i in range(2):
        body = osim.Body(f'b{i}', 0.0, osim.Vec3(0), osim.Inertia(0.0))
        model.addBody(body)
        model.addJoint(osim.WeldJoint(f'j{i}', parent, body))
        parent = body
    model.finalizeConnections()
    assert model.getTotalMass(model.initSystem()) == 0.0

    with pytest.raises(ValueError, match='no mass distribution to preserve'):
        set_model_mass(model, 10.0)


def test_returns_the_same_model_object():
    model = create_three_body_model()
    assert set_model_mass(model, 20.0) is model
