import os
import time
import opensim as osim
from osimfit.data_sources import TheiaFrameSource
from osimfit.scaling import PositionDataScaler, FrameMeasurement, Axis, ScaleFactor, \
                            AnthropometricScaler, AnthropometricMeasurement
from osimfit.solvers import InverseKinematicsSolver, SplineBasedInverseKinematicsSolver

# STEP 1: POSITION-BASED SCALING
# ------------------------------

# data_label --> model_frame
frame_map = {
    'l_thigh': '/jointset/hip_l/femur_l_offset/l_thigh',
    'l_shank': '/jointset/walker_knee_l/tibia_l_offset/l_shank',
    'l_foot': '/jointset/ankle_l/talus_l_offset/l_foot',
    'l_toes': '/jointset/mtp_l/toes_l_offset/l_toes',
    'r_thigh': '/jointset/hip_r/femur_r_offset/r_thigh',
    'r_shank': '/jointset/walker_knee_r/tibia_r_offset/r_shank',
    'r_foot': '/jointset/ankle_r/talus_r_offset/r_foot',
    'r_toes': '/jointset/mtp_r/toes_r_offset/r_toes',
    'l_uarm': '/jointset/acromial_l/humerus_l_offset/l_uarm',
    'l_larm': '/jointset/elbow_l/ulna_l_offset/l_larm',
    'l_hand': '/jointset/radius_hand_l/hand_l_offset/l_hand',
    'r_uarm': '/jointset/acromial_r/humerus_r_offset/r_uarm',
    'r_larm': '/jointset/elbow_r/ulna_r_offset/r_larm',
    'r_hand': '/jointset/radius_hand_r/hand_r_offset/r_hand',
    'pelvis': '/bodyset/pelvis/pelvis',
    'torso': '/bodyset/torso/torso',
}

# segment_name --> [data_label_1, data_label_2, axis]
scale_map = {
    'pelvis': ['pelvis', 'torso', Axis.YAxis],
    'pelvis': ['l_thigh', 'r_thigh', Axis.ZAxis],
    'torso': ['torso', 'pelvis', Axis.YAxis],
    'torso': ['l_uarm', 'r_uarm', Axis.ZAxis],
    'humerus_r': ['r_uarm', 'r_larm', Axis.YAxis],
    'humerus_l': ['l_uarm', 'l_larm', Axis.YAxis],
    'radius_r': ['r_larm', 'r_hand', Axis.YAxis],
    'radius_l': ['l_larm', 'l_hand', Axis.YAxis],
    'femur_r': ['r_thigh', 'r_shank', Axis.YAxis],
    'femur_l': ['l_thigh', 'l_shank', Axis.YAxis],
    'tibia_r': ['r_shank', 'r_foot', Axis.YAxis],
    'tibia_l': ['l_shank', 'l_foot', Axis.YAxis],
    'calcn_r': ['r_foot', 'r_toes', Axis.XAxis],
    'calcn_r': ['r_foot', 'r_toes', Axis.YAxis],
    'calcn_l': ['l_foot', 'l_toes', Axis.XAxis],
    'calcn_l': ['l_foot', 'l_toes', Axis.YAxis]
}

# Scale a model from position-based (e.g., Vec3) data.
model = osim.Model('unscaled_generic.osim')
c3d_source = TheiaFrameSource('pose_0.c3d')
position_scaler = PositionDataScaler(model, c3d_source)

# Add scales.
for segment_name, (data_label_1, data_label_2, axis) in scale_map.items():
    measurement = FrameMeasurement(frame_map[data_label_1], frame_map[data_label_2])
    scale_factor = ScaleFactor(data_label_1, data_label_2, measurement, axis)
    position_scaler.add_scale(segment_name, scale_factor)

# Add symmetry pairs.
position_scaler.add_symmetry_pair('humerus_l', 'humerus_r')
position_scaler.add_symmetry_pair('radius_l', 'radius_r')
position_scaler.add_symmetry_pair('femur_l', 'femur_r')
position_scaler.add_symmetry_pair('tibia_l', 'tibia_r')
position_scaler.add_symmetry_pair('calcn_l', 'calcn_r')

# Scale the model.
scaled_model = position_scaler.scale()
scaled_model.printToXML('jump_1_scaled.osim')


# STEP 2: ANTHROPOMETRIC SCALING
# ------------------------------

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
    'waistdepth':             ('/posterior_omphalion', '/anterior_omphalion', None)
}


anthropometric_scaler = AnthropometricScaler(scaled_model, sex='female')

for ansur_label, (station1_path, station2_path, axis) in ansur_measurements.items():
    measurement = AnthropometricMeasurement(station1_path, station2_path, axis)
    anthropometric_scaler.add_measurement(ansur_label, measurement)

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

anthro_scaled_model = anthropometric_scaler.scale()
anthro_scaled_model.printToXML('jump_1_anthro_scaled.osim')


# STEP 3: INVERSE KINEMATICS
# --------------------------
columns_to_remove = ['worldbody', 'head', 'pelvis_shifted', 'l_clavicle', 'r_clavicle']

theia_frame_source = TheiaFrameSource('pose_0.c3d',
                                      labels_to_remove=columns_to_remove,
                                      label_map=frame_map)

# Run the frame-by-frame IK solver.
weights = {'position': 2.0, 'orientation': 5.0, 'smoothness': 0.5}
solver = InverseKinematicsSolver(anthro_scaled_model,
                                 convergence_tolerance=1e-4,
                                 weights=weights)
solver.add_theia_frame_source(theia_frame_source)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution, 'jump_1_ik_solution.sto')

# Run the spline IK solver, initialized with the frame-by-frame solution.
weights = {'position': 2.0, 'orientation': 5.0}
solver = SplineBasedInverseKinematicsSolver(anthro_scaled_model,
                                            convergence_tolerance=1e-4,
                                            weights=weights,
                                            knot_interval=0.06)
solver.add_theia_frame_source(theia_frame_source)
spline_ik_solution = solver.solve(osim.TimeSeriesTable('jump_1_ik_solution.sto'))
sto = osim.STOFileAdapter()
sto.write(spline_ik_solution, 'jump_1_spline_ik_solution.sto')

# STEP 4: VISUALIZATION
# ---------------------
modelProcessor = osim.ModelProcessor('jump_1_anthro_scaled.osim')
modelProcessor.append(osim.ModOpRemoveMuscles())
model = modelProcessor.process()
model.initSystem()

states = osim.TimeSeriesTable('jump_1_spline_ik_solution.sto')
states.addTableMetaDataString('inDegrees', 'no')

osim.VisualizerUtilities.showMotion(model, states)
