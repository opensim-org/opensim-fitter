
import os
import time
import opensim as osim
from osimfit.data_sources import MarkerSource, TheiaFrameSource
from osimfit.solvers import InverseKinematicsSolver, SplineInverseKinematicsSolver

columns_to_remove = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow',
                     'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee',
                     'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe',
                     'RHeel']

marker_fpath = os.path.join('postaug', 'ID5_S7_sprintNoSync_LSTM.trc')
marker_table = osim.TimeSeriesTableVec3(marker_fpath)
labels = marker_table.getColumnLabels()
labels = [label for label in labels if label not in columns_to_remove]
label_map = {label: f'/markerset/{label}' for label in labels}
marker_source = MarkerSource(marker_fpath,
                             labels_to_remove=columns_to_remove,
                             label_map=label_map)


model = osim.Model('LaiUhlrich2022_scaled.osim')
model.initSystem()


# Run the frame-by-frame IK solver.
weights = {'position': 2.0, 'orientation': 5.0, 'smoothness': 0.5}
solver = InverseKinematicsSolver(model,
                                 convergence_tolerance=1e-4,
                                 weights=weights)
solver.add_marker_source(marker_source)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution, 'sprint_ik_solution.sto')

# Run the spline IK solver, initialized with the frame-by-frame solution.
weights = {'position': 2.0, 'orientation': 5.0}
solver = SplineInverseKinematicsSolver(model,
                                       convergence_tolerance=1e-4,
                                       weights=weights,
                                       knot_interval=0.06)
solver.add_marker_source(marker_source)
spline_ik_solution = solver.solve(osim.TimeSeriesTable('sprint_ik_solution.sto'))
sto = osim.STOFileAdapter()
sto.write(spline_ik_solution, 'sprint_spline_ik_solution.sto')

# STEP 4: VISUALIZATION
# ---------------------
modelProcessor = osim.ModelProcessor(model)
modelProcessor.append(osim.ModOpRemoveMuscles())
model = modelProcessor.process()
model.initSystem()

states = osim.TimeSeriesTable('sprint_spline_ik_solution.sto')
states.addTableMetaDataString('inDegrees', 'no')

osim.VisualizerUtilities.showMotion(model, states)