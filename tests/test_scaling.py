"""
Unit tests for src/osimfit/scaling.py.

Covers the three Measurement subclasses (FrameMeasurement,
MarkerMeasurement, AnthropometricMeasurement) and the
PositionDataScaler container API (`add_scale`, `add_symmetry_pair`).
FrameMeasurement is tested against unscaled_generic.osim. The
fixture model has no markers and no stations attached at predictable
named paths, so MarkerMeasurement, AnthropometricMeasurement, and
the scaler container tests use a tiny in-memory rig with two markers
and two stations placed 0.5 m apart along the body-frame X axis.
"""

from pathlib import Path

import numpy as np
import opensim as osim
import pytest

from osimfit.data_sources import DataSource
from osimfit.scaling import (
    AnthropometricMeasurement,
    Axis,
    FrameMeasurement,
    MarkerMeasurement,
    PositionDataScaler,
    ScaleFactor,
)


MODEL_FPATH = str(Path(__file__).parent / 'unscaled_generic.osim')


# -- helpers ---------------------------------------------------------------

def _load_generic_model():
    """
    Load unscaled_generic.osim and return (model, initialized state).
    """
    model = osim.Model(MODEL_FPATH)
    state = model.initSystem()
    return model, state


def _build_test_rig():
    """
    Build a one-body model welded to ground with two markers and two
    stations placed 0.5 m apart along the body-frame X axis.
    """
    model = osim.Model()
    model.setName('test_rig')
    ground = model.getGround()

    body = osim.Body('rig_body', 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(body)

    joint = osim.WeldJoint(
        'rig_weld',
        ground, osim.Vec3(0), osim.Vec3(0),
        body, osim.Vec3(0), osim.Vec3(0),
    )
    model.addJoint(joint)

    model.addMarker(osim.Marker('m0', body, osim.Vec3(0)))
    model.addMarker(osim.Marker('m1', body, osim.Vec3(0.5, 0, 0)))

    s0 = osim.Station(body, osim.Vec3(0))
    s0.setName('s0')
    model.addComponent(s0)
    s1 = osim.Station(body, osim.Vec3(0.5, 0, 0))
    s1.setName('s1')
    model.addComponent(s1)

    model.finalizeConnections()
    return model


class _StubPositionsSource(DataSource):
    """
    Throwaway DataSource that emits a minimal positions table so
    PositionDataScaler can be constructed without real marker data.
    """
    def _create_positions_table(self):
        table = osim.TimeSeriesTableVec3()
        row = osim.RowVectorVec3(2, osim.Vec3(0))
        for t in (0.0, 0.1):
            table.appendRow(t, row)
        table.setColumnLabels(['a', 'b'])
        return table


# -- FrameMeasurement ------------------------------------------------------

def test_frame_measurement_same_frame_yields_zero():
    model, state = _load_generic_model()
    measurement = FrameMeasurement('/bodyset/pelvis', '/bodyset/pelvis')
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-12)


def test_frame_measurement_matches_direct_calc():
    model, state = _load_generic_model()
    path1 = '/bodyset/femur_r'
    path2 = '/bodyset/tibia_r'

    frame1 = osim.Frame.safeDownCast(model.getComponent(path1))
    frame2 = osim.Frame.safeDownCast(model.getComponent(path2))
    expected = float(np.linalg.norm(
        frame1.getPositionInGround(state).to_numpy()
        - frame2.getPositionInGround(state).to_numpy()))

    measurement = FrameMeasurement(path1, path2)
    actual = measurement.compute_measurement(model, state)
    assert actual == pytest.approx(expected, rel=1e-12)


# -- MarkerMeasurement -----------------------------------------------------

def test_marker_measurement_same_marker_yields_zero():
    model = _build_test_rig()
    state = model.initSystem()
    measurement = MarkerMeasurement('/markerset/m0', '/markerset/m0')
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-12)


def test_marker_measurement_returns_offset_distance():
    model = _build_test_rig()
    state = model.initSystem()
    measurement = MarkerMeasurement('/markerset/m0', '/markerset/m1')
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.5, abs=1e-12)


# -- AnthropometricMeasurement ---------------------------------------------

def test_anthropometric_no_axis_returns_mm_magnitude():
    model = _build_test_rig()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('/s0', '/s1')
    # 0.5 m * 1000 = 500 mm.
    assert measurement.compute_measurement(model, state) == pytest.approx(
        500.0, abs=1e-9)


def test_anthropometric_x_axis_returns_mm_along_x():
    model = _build_test_rig()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('/s0', '/s1', axis=Axis.XAxis)
    assert measurement.compute_measurement(model, state) == pytest.approx(
        500.0, abs=1e-9)


def test_anthropometric_y_axis_returns_zero_for_pure_x_offset():
    model = _build_test_rig()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('/s0', '/s1', axis=Axis.YAxis)
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-9)


def test_anthropometric_z_axis_returns_zero_for_pure_x_offset():
    model = _build_test_rig()
    state = model.initSystem()
    measurement = AnthropometricMeasurement('/s0', '/s1', axis=Axis.ZAxis)
    assert measurement.compute_measurement(model, state) == pytest.approx(
        0.0, abs=1e-9)


# -- PositionDataScaler container API --------------------------------------

def _make_scaler():
    """
    Build a PositionDataScaler over the rig model with a stub data source.
    """
    return PositionDataScaler(_build_test_rig(), _StubPositionsSource())


def _make_scale_factor(label='a'):
    """
    Build a throwaway ScaleFactor — its arguments aren't exercised by the
    add_scale / add_symmetry_pair container tests.
    """
    measurement = FrameMeasurement('/bodyset/rig_body', '/bodyset/rig_body')
    return ScaleFactor(label, label, measurement, Axis.XAxis)


def test_position_data_scaler_add_scale_appends_to_segment_list():
    scaler = _make_scaler()
    sf = _make_scale_factor()
    scaler.add_scale('rig_body', sf)
    assert list(scaler.segment_scale_factors.keys()) == ['rig_body']
    assert scaler.segment_scale_factors['rig_body'] == [sf]


def test_position_data_scaler_add_scale_groups_multiple_per_segment():
    scaler = _make_scaler()
    sf1 = _make_scale_factor('a')
    sf2 = _make_scale_factor('b')
    scaler.add_scale('rig_body', sf1)
    scaler.add_scale('rig_body', sf2)
    assert scaler.segment_scale_factors['rig_body'] == [sf1, sf2]


def test_position_data_scaler_add_symmetry_pair_appends_tuple():
    scaler = _make_scaler()
    scaler.add_symmetry_pair('femur_l', 'femur_r')
    scaler.add_symmetry_pair('tibia_l', 'tibia_r')
    assert scaler.symmetry_pairs == [
        ('femur_l', 'femur_r'),
        ('tibia_l', 'tibia_r'),
    ]
