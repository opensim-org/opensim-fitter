import os
import time
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
from osimfit.data_sources import MarkerSource
from osimfit.scaling import (
    Axis, 
    PositionDataScaler, 
    MarkerMeasurement, 
    ScaleFactor,
    AnthropometricMeasurement,
    AnthropometricScaler
    )

from osimfit.solvers import InverseKinematicsSolver, SplineBasedBilevelSolver

# EXAMPLE WALK
# ------------
# This example demonstrates how to use the inverse kinematics solvers, both
# frame-by-frame and spline-based, to find the kinematics for a walking motion. The
# model and marker trajectories are taken from the sample data associated with the
# Rajagopal et al. (2016) lower-body model publication.

# Load data
# ---------
# Load the marker data and model.
markers_fpath = 'motion_capture_walk.trc'
markers_table = osim.TimeSeriesTableVec3(markers_fpath)
marker_labels = markers_table.getColumnLabels()
model = osim.Model('unscaled_generic.osim')
model.initSystem()

# Append the markerset used for scaling and inverse kinematics to the model.
markerset = model.updMarkerSet()
mset = osim.MarkerSet('markerset_walk_preScale.xml')
for i in range(mset.getSize()):
    markerset.cloneAndAppend(mset.get(i))

# Define a mapping between marker names and marker paths.
# (marker_name --> /marker/path)
marker_map = {label: f'/markerset/{label}' for label in marker_labels}

# Marker-based scaling
# --------------------

# Define scaling rules as a list of (segment, marker_1, marker_2, axis) tuples.
# Each rule specifies a segment to scale, two markers whose inter-distance defines
# the scale factor, and the axis along which to apply it.
scale_rules = [
    ('torso', 'R.PSIS', 'R.Shoulder', Axis.YAxis),
    ('torso', 'L.PSIS', 'L.Shoulder', Axis.YAxis),
    ('torso', 'R.Shoulder', 'L.Shoulder', Axis.ZAxis),

    ('pelvis', 'R.ASIS', 'L.ASIS', Axis.ZAxis),
    ('pelvis', 'R.PSIS', 'L.PSIS', Axis.ZAxis),
    ('pelvis', 'R.PSIS', 'R.ASIS', Axis.XAxis),
    ('pelvis', 'L.PSIS', 'L.ASIS', Axis.XAxis),

    ('humerus_r', 'R.Shoulder', 'R.Elbow', Axis.YAxis),
    ('humerus_l', 'L.Shoulder', 'L.Elbow', Axis.YAxis),

    ('radius_r', 'R.Elbow', 'R.Wrist', Axis.YAxis),
    ('radius_l', 'L.Elbow', 'L.Wrist', Axis.YAxis),

    ('ulna_r', 'R.Elbow', 'R.Wrist', Axis.YAxis),
    ('ulna_l', 'L.Elbow', 'L.Wrist', Axis.YAxis),

    ('hand_r', 'R.Elbow', 'R.Wrist', Axis.YAxis),
    ('hand_l', 'L.Elbow', 'L.Wrist', Axis.YAxis),

    ('femur_r', 'R.ASIS', 'R.Knee', Axis.YAxis),
    ('femur_l', 'L.ASIS', 'L.Knee', Axis.YAxis),

    ('patella_r', 'R.ASIS', 'R.Knee', Axis.YAxis),
    ('patella_l', 'L.ASIS', 'L.Knee', Axis.YAxis),

    ('tibia_r', 'R.Knee', 'R.Ankle', Axis.YAxis),
    ('tibia_l', 'L.Knee', 'L.Ankle', Axis.YAxis),

    ('calcn_r', 'R.Heel', 'R.Toe', Axis.XAxis),
    ('calcn_r', 'R.Heel', 'R.MT5', Axis.XAxis),
    ('calcn_r', 'R.Toe', 'R.MT5', Axis.ZAxis),
    ('calcn_r', 'R.Heel', 'R.Ankle', Axis.YAxis),
    ('toes_r', 'R.Heel', 'R.Toe', Axis.XAxis),
    ('toes_r', 'R.Heel', 'R.MT5', Axis.XAxis),
    ('toes_r', 'R.Toe', 'R.MT5', Axis.ZAxis),
    ('toes_r', 'R.Heel', 'R.Ankle', Axis.YAxis),

    ('calcn_l', 'L.Heel', 'L.Toe', Axis.XAxis),
    ('calcn_l', 'L.Heel', 'L.MT5', Axis.XAxis),
    ('calcn_l', 'L.Toe', 'L.MT5', Axis.ZAxis),
    ('calcn_l', 'L.Heel', 'L.Ankle', Axis.YAxis),
    ('toes_l', 'L.Heel', 'L.Toe', Axis.XAxis),
    ('toes_l', 'L.Heel', 'L.MT5', Axis.XAxis),
    ('toes_l', 'L.Toe', 'L.MT5', Axis.ZAxis),
    ('toes_l', 'L.Heel', 'L.Ankle', Axis.YAxis),
]


# Create a MarkerSource and PositionDataScaler.
marker_source = MarkerSource(markers_fpath)
position_scaler = PositionDataScaler(model, marker_source)

# Add scaling rules to the PositionDataScaler.
for segment_name, marker_1, marker_2, axis in scale_rules:
    measurement = MarkerMeasurement(marker_map[marker_1], marker_map[marker_2])
    scale_factor = ScaleFactor(marker_1, marker_2, measurement, axis)
    position_scaler.add_scale(segment_name, scale_factor)

# Add symmetry pairs. Internally, the PositionDataScaler will average the scale factors
# computed for each pair of symmetric segments to ensure left-right symmetry.
position_scaler.add_symmetry_pair('humerus_l', 'humerus_r')
position_scaler.add_symmetry_pair('radius_l', 'radius_r')
position_scaler.add_symmetry_pair('ulna_l', 'ulna_r')
position_scaler.add_symmetry_pair('hand_l', 'hand_r')
position_scaler.add_symmetry_pair('femur_l', 'femur_r')
position_scaler.add_symmetry_pair('patella_l', 'patella_r')
position_scaler.add_symmetry_pair('tibia_l', 'tibia_r')
position_scaler.add_symmetry_pair('calcn_l', 'calcn_r')
position_scaler.add_symmetry_pair('toes_l', 'toes_r')

# Scale the model.
scaled_model = position_scaler.scale()
scaled_model.printToXML('subject_marker_scaled_walk.osim')


# Anthropometry-based scaling
# ---------------------------
# Next, we will adjust the scaled model based on anthropometric measurements from the
# ANSUR II dataset.

# Define a mapping between ANSUR II measurement labels and pairs of stations (e.g.,
# body-fixed points) representing the measurement, along with the axis along which to
# apply the measurement. If no axis is specified, the measurement will be applied
# isotropically.
# ansur_label --> (station1_path, station2_path, axis)
ansur_measurements = {
    'biacromialbreadth':      ('/acromion_r', '/acromion_l', None),
    'bicristalbreadth':       ('/iliocrestale_r', '/iliocrestale_l', None),
    'bimalleolarbreadth':     ('/lateral_malleolus_r', '/medial_malleolus_r', None),
    'footbreadthhorizontal':  ('/mtp1_r', '/mtp5_r', Axis.ZAxis),
    'footlength':             ('/acropodion_r', '/pternion_r', Axis.XAxis),
    'iliocristaleheight':     ('/iliocrestale_r', '/mtp5_r', Axis.YAxis),
    'lateralmalleolusheight': ('/lateral_malleolus_r', '/mtp5_r', Axis.YAxis),
    'radialestylionlength':   ('/radiale_r', '/stylion_r', None),
    'shoulderelbowlength':    ('/acromion_r', '/olecranon_r', None),
    'stature':                ('/vertex', '/mtp5_r', Axis.YAxis),
    'suprasternaleheight':    ('/suprasternale', '/mtp5_r', Axis.YAxis),
    'tibialheight':           ('/tibiale_r', '/mtp5_r', Axis.YAxis),
    'trochanterionheight':    ('/trochanterion_r', '/mtp5_r', Axis.YAxis),
    'waistbacklength':        ('/cervicale', '/posterior_omphalion', None),
    'waistdepth':             ('/posterior_omphalion', '/anterior_omphalion', None),
}

# Create the AnthropometricScaler and add measurements based on the mapping above.
anthropometric_scaler = AnthropometricScaler(scaled_model, sex='female')

# Add measurements to the AnthropometricScaler. Each measurement is defined by a pair of
# stations and an axis along which to apply the measurement.
for ansur_label, (station1_path, station2_path, axis) in ansur_measurements.items():
    measurement = AnthropometricMeasurement(station1_path, station2_path, axis)
    anthropometric_scaler.add_measurement(ansur_label, measurement)

# Select a subset of the measurements that we will use to condition the
# multivariate normal distribution. These measurements are "trustworthy" in the
# sense that we can estimate them relatively well from the Theia frames.
anthropometric_scaler.add_conditional_measurement('iliocristaleheight')
anthropometric_scaler.add_conditional_measurement('radialestylionlength')
anthropometric_scaler.add_conditional_measurement('shoulderelbowlength')
anthropometric_scaler.add_conditional_measurement('stature')
anthropometric_scaler.add_conditional_measurement('suprasternaleheight')
anthropometric_scaler.add_conditional_measurement('tibialheight')
anthropometric_scaler.add_conditional_measurement('trochanterionheight')
anthropometric_scaler.add_conditional_measurement('waistbacklength')

# Define the scale factors that will be generated from the conditioned anthropometric
# measurements.
anthropometric_scaler.add_scale_factor('torso', 'biacromialbreadth', Axis.ZAxis)
anthropometric_scaler.add_scale_factor('pelvis', 'bicristalbreadth', Axis.ZAxis)
anthropometric_scaler.add_scale_factor('tibia_r', 'bimalleolarbreadth', Axis.YAxis)
anthropometric_scaler.add_scale_factor('tibia_r', 'bimalleolarbreadth', Axis.ZAxis)
anthropometric_scaler.add_scale_factor('tibia_l', 'bimalleolarbreadth', Axis.YAxis)
anthropometric_scaler.add_scale_factor('tibia_l', 'bimalleolarbreadth', Axis.ZAxis)
anthropometric_scaler.add_scale_factor('calcn_r', 'footlength', Axis.XAxis)
anthropometric_scaler.add_scale_factor('calcn_r', 'footbreadthhorizontal', Axis.ZAxis)
anthropometric_scaler.add_scale_factor('calcn_l', 'footlength', Axis.XAxis)
anthropometric_scaler.add_scale_factor('calcn_l', 'footbreadthhorizontal', Axis.ZAxis)

# Scale the model.
anthro_scaled_model = anthropometric_scaler.scale()
anthro_scaled_model.printToXML(os.path.join('subject_anthro_scaled_walk.osim'))


# Frame-by-frame inverse kinematics
# ---------------------------------
# Create a new MarkerSource with updated labels and tracking markers removed.
columns_to_remove = ['R.TH1', 'R.TH2', 'R.TH3', 'R.SH1', 'R.SH2', 'R.SH3', 'R.SH4',
                     'L.TH1', 'L.TH2', 'L.TH3', 'L.TH4', 'L.SH1', 'L.SH2', 'L.SH3']
marker_source = MarkerSource(markers_fpath, label_map=marker_map, 
                             labels_to_remove=columns_to_remove)

# Run the frame-by-frame IK solver.
solver = InverseKinematicsSolver(model,
                                 convergence_tolerance=1e-2,
                                 position_weight=1.0)
solver.add_marker_reference_data(marker_source)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution.states_table, 'walk_ik_solution.sto')

# Spline-based inverse kinematics
# -------------------------------
# Construct a SplineBasedBilevelSolver to solve for the model kinematics and body
# lengths that best match the marker data.
solver = SplineBasedBilevelSolver(model,
                                  convergence_tolerance=1e-3,
                                  knot_interval=0.08,
                                  position_weight=5.0,
                                  scale_regularization_weight=1e-2)
solver.add_marker_reference_data(marker_source)
# Add scale factors for the two bodies including lower and upper bounds for the
# optimization variables.
solver.add_scale_factor('/bodyset/torso', 0.5, 2.0)
solver.add_scale_factor('/bodyset/pelvis', 0.5, 2.0)
solver.add_scale_factor(['/bodyset/humerus_r', '/bodyset/humerus_l'], 0.5, 2.0)
solver.add_scale_factor(['/bodyset/radius_r', '/bodyset/radius_l',
                         '/bodyset/ulna_r', '/bodyset/ulna_l',
                         '/bodyset/hand_r', '/bodyset/hand_l'], 0.5, 2.0)
solver.add_scale_factor(['/bodyset/femur_r', '/bodyset/femur_l',
                         '/bodyset/patella_r', '/bodyset/patella_l'], 0.5, 2.0)
solver.add_scale_factor(['/bodyset/tibia_r', '/bodyset/tibia_l'], 0.5, 2.0)
solver.add_scale_factor(['/bodyset/calcn_r', '/bodyset/calcn_l',
                         '/bodyset/toes_r', '/bodyset/toes_l'], 0.5, 2.0)

# Solve!
guess = osim.TimeSeriesTable('walk_ik_solution.sto')
bilevel_solution = solver.solve(guess)
sto.write(bilevel_solution.states_table, 'walk_bilevel_solution.sto')


# # Plot joint kinematics
# # ---------------------
# coordinates  = ['/jointset/ground_pelvis/pelvis_list/value',
#                 '/jointset/hip_r/hip_adduction_r/value',
#                 '/jointset/subtalar_r/subtalar_angle_r/value']
# ylabels = ['pelvis list (deg)', 'hip adduction (deg)', 'subtalar angle (deg)']
# fig, axes = plt.subplots(3, 1, figsize=(6, 2.5 * len(coordinates)), sharex=True)

# # Plot frame-by-frame IK solution.
# solution = osim.TimeSeriesTable('walk_ik_solution.sto')
# for ax, coord in zip(axes.flatten(), coordinates):
#     t = np.array(solution.getIndependentColumn())
#     ax.plot(solution.getIndependentColumn(),
#             np.rad2deg(solution.getDependentColumn(coord).to_numpy()),
#             color='black', alpha=0.4, lw=5, label='Frame-by-frame IK')
#     ax.set_ylabel(ylabels[coordinates.index(coord)])
#     ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.75)

# # Plot the spline-based IK solutions.
# solution = osim.TimeSeriesTable('walk_spline_based_ik_solution.sto')
# for ax, coord in zip(axes.flatten(), coordinates):
#     ax.plot(solution.getIndependentColumn(),
#             np.rad2deg(solution.getDependentColumn(coord).to_numpy()),
#                 color='blue', lw=3,
#                 label=f'Spline-based IK')

# axes.flatten()[-1].set_xlabel('time (s)')
# axes.flatten()[0].legend(loc='upper right', fontsize=8)

# plt.tight_layout()
# plt.savefig('compare_joint_kinematics.png', dpi=150, bbox_inches='tight')
# # plt.show()
