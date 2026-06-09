"""
End-to-end regression test mirroring examples/example_pendulum/example_pendulum.py.

Synthesizes marker data from a double pendulum with known body lengths
(1.25 m and 0.75 m), then runs SplineBasedBilevelSolver against an unscaled
model (both lengths = 1.0 m) and asserts the recovered scale factors recover
the ground-truth lengths.
"""

import opensim as osim

from osimfit.data_sources import MarkerSource
from osimfit.solvers import SplineBasedBilevelSolver


def create_double_pendulum(length1: float, length2: float) -> osim.Model:
    """
    Build a double-pendulum model with the given body lengths.
    """
    model = osim.Model()
    model.setName("double_pendulum")
    ground = model.getGround()

    b0 = osim.Body("b0", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b0)
    j0 = osim.PinJoint(
        "j0",
        ground, osim.Vec3(0), osim.Vec3(0),
        b0, osim.Vec3(-length1, 0, 0), osim.Vec3(0),
    )
    j0.updCoordinate().setName("q0")
    model.addJoint(j0)

    b1 = osim.Body("b1", 1.0, osim.Vec3(0), osim.Inertia(1))
    model.addBody(b1)
    j1 = osim.PinJoint(
        "j1",
        b0, osim.Vec3(0), osim.Vec3(0),
        b1, osim.Vec3(-length2, 0, 0), osim.Vec3(0),
    )
    j1.updCoordinate().setName("q1")
    model.addJoint(j1)

    model.addMarker(osim.Marker("m0", b0, osim.Vec3(0)))
    model.addMarker(osim.Marker("m1", b1, osim.Vec3(0)))

    model.finalizeConnections()
    return model


def create_synthetic_markers_file(trc_path: str, length1: float,
                                  length2: float) -> None:
    """
    Forward-simulate the truth pendulum and write marker positions to a TRC.
    """
    model = create_double_pendulum(length1, length2)
    state = model.initSystem()

    manager = osim.Manager(model)
    manager.setIntegratorFixedStepSize(0.01)
    manager.initialize(state)
    manager.integrate(2.0)
    states = manager.getStatesTable()

    controls = osim.TimeSeriesTable(states.getIndependentColumn())
    output_paths = osim.StdVectorString()
    output_paths.append('/markerset/.*location')
    markers = osim.analyzeVec3(model, states, controls, output_paths)

    markers.addTableMetaDataString('DataRate', '100.0')
    markers.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter().write(markers, trc_path)


def test_pendulum_bilevel_recovers_ground_truth_lengths(tmp_path):
    """
    Bilevel solver must recover the body lengths used to synthesize the data.
    """
    true_b0_length = 1.25
    true_b1_length = 0.75

    trc_path = str(tmp_path / "markers.trc")
    create_synthetic_markers_file(trc_path, true_b0_length, true_b1_length)

    # Strip the '|location' suffix that analyzeVec3 appends to marker column
    # labels so they match the marker names ('m0', 'm1') in the unscaled model.
    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    # Solve against an unscaled model (both lengths = 1.0 m).
    unscaled_model = create_double_pendulum(1.0, 1.0)
    unscaled_model.initSystem()

    marker_source = MarkerSource(trc_path, label_map=label_map)

    solver = SplineBasedBilevelSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
        scale_regularization_weight=1e-2,
    )
    solver.add_marker_reference_data(marker_source)
    solver.add_scale_factor('/bodyset/b0', 0.5, 2.0)
    solver.add_scale_factor('/bodyset/b1', 0.5, 2.0)

    solution = solver.solve()

    # Sanity: solution table covers the simulated 2.0 s @ 100 Hz (201 samples)
    # and exposes both joint coordinates.
    assert solution.states_table.getNumRows() == 201
    state_labels = list(solution.states_table.getColumnLabels())
    assert '/jointset/j0/q0/value' in state_labels
    assert '/jointset/j1/q1/value' in state_labels

    # Regression contract: recovered X-scale factors match the ground-truth
    # lengths. Y and Z scales should stay near 1.0 since the truth model only
    # varies length along the local X axis.
    assert abs(solution.scale_factors[0, 0] - true_b0_length) < 0.02
    assert abs(solution.scale_factors[1, 0] - true_b1_length) < 0.02
    for body_idx in (0, 1):
        for axis in (1, 2):
            assert abs(solution.scale_factors[body_idx, axis] - 1.0) < 0.05


def test_pendulum_bilevel_recovers_shared_length_under_asymmetric_truth(
        tmp_path):
    """
    A single scale factor shared across both pendulum bodies must converge to
    a compromise value strictly between the two ground-truth lengths. This
    proves the grouped-scale machinery wires through end-to-end: one
    optimization variable, broadcast to two mobilized bodies, with a
    chain-rule Jacobian column.
    """
    true_b0_length = 1.25
    true_b1_length = 0.75

    trc_path = str(tmp_path / "markers.trc")
    create_synthetic_markers_file(trc_path, true_b0_length, true_b1_length)

    raw_labels = osim.TimeSeriesTableVec3(trc_path).getColumnLabels()
    label_map = {label: label.replace('|location', '') for label in raw_labels}

    unscaled_model = create_double_pendulum(1.0, 1.0)
    unscaled_model.initSystem()

    marker_source = MarkerSource(trc_path, label_map=label_map)

    solver = SplineBasedBilevelSolver(
        unscaled_model,
        convergence_tolerance=1e-5,
        knot_interval=0.05,
        position_weight=5.0,
        scale_regularization_weight=1e-2,
    )
    solver.add_marker_reference_data(marker_source)
    solver.add_scale_factor(
        ['/bodyset/b0', '/bodyset/b1'], 0.5, 2.0)

    solution = solver.solve()

    # One scale factor group → one row of 3 components.
    assert solution.scale_factors.shape == (1, 3)
    assert solution.body_paths == [['/bodyset/b0', '/bodyset/b1']]

    # The shared scale must lie strictly between the two ground-truth lengths
    # — it cannot match either body exactly. This is the load-bearing check
    # that the shared constraint was actually enforced, not silently dropped.
    shared_sx = solution.scale_factors[0, 0]
    assert true_b1_length < shared_sx < true_b0_length

    # Y and Z scales remain near 1.0 since the truth varies length along X.
    for axis in (1, 2):
        assert abs(solution.scale_factors[0, axis] - 1.0) < 0.05
