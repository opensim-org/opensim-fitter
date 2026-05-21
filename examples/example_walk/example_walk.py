import os
import time
import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
from osimfit.data_sources import MarkerSource
from osimfit.solvers import InverseKinematicsSolver, SplineBasedInverseKinematicsSolver

# EXAMPLE WALK
# ------------
# This example demonstrates how to use the inverse kinematics solvers, both
# frame-by-frame and spline-based, to find the kinematics for a walking motion. The
# model and marker trajectories are taken from the sample data associated with the
# Rajagopal et al. (2016) lower-body model publication.

# Load data
# ---------
# Load the marker data and model.
marker_fpath = 'motion_capture_walk.trc'
marker_table = osim.TimeSeriesTableVec3(marker_fpath)
model = osim.Model('subject_scale_walk.osim')
model.initSystem()

# Create a MarkerSource.
label_map = {label: f'/markerset/{label}' for label in marker_table.getColumnLabels()}
marker_source = MarkerSource(marker_fpath, label_map=label_map)

# Frame-by-frame inverse kinematics
# ---------------------------------
# Run the frame-by-frame IK solver.
solver = InverseKinematicsSolver(model,
                                 convergence_tolerance=1e-2,
                                 position_weight=1.0)
solver.add_marker_source(marker_source)
ik_solution = solver.solve()
sto = osim.STOFileAdapter()
sto.write(ik_solution.states_table, 'walk_ik_solution.sto')

# Spline-based inverse kinematics
# -------------------------------
# Run the spline IK solver, initialized with the frame-by-frame solution, for varying
# knot densities.
solver = SplineBasedInverseKinematicsSolver(model,
                                            convergence_tolerance=1e-3,
                                            position_weight=1.0,
                                            knot_interval=0.075)
solver.add_marker_source(marker_source)
spline_ik_solution = solver.solve(osim.TimeSeriesTable('walk_ik_solution.sto'))
sto = osim.STOFileAdapter()
sto.write(spline_ik_solution.states_table, 'walk_spline_based_ik_solution.sto')

# Plot joint kinematics
# ---------------------
coordinates  = ['/jointset/ground_pelvis/pelvis_list/value',
                '/jointset/hip_r/hip_adduction_r/value',
                '/jointset/subtalar_r/subtalar_angle_r/value']
ylabels = ['pelvis list (deg)', 'hip adduction (deg)', 'subtalar angle (deg)']
fig, axes = plt.subplots(3, 1, figsize=(6, 2.5 * len(coordinates)), sharex=True)

# Plot frame-by-frame IK solution.
solution = osim.TimeSeriesTable('walk_ik_solution.sto')
for ax, coord in zip(axes.flatten(), coordinates):
    t = np.array(solution.getIndependentColumn())
    ax.plot(solution.getIndependentColumn(),
            np.rad2deg(solution.getDependentColumn(coord).to_numpy()),
            color='black', alpha=0.4, lw=5, label='Frame-by-frame IK')
    ax.set_ylabel(ylabels[coordinates.index(coord)])
    ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.75)

# Plot the spline-based IK solutions.
solution = osim.TimeSeriesTable('walk_spline_based_ik_solution.sto')
for ax, coord in zip(axes.flatten(), coordinates):
    ax.plot(solution.getIndependentColumn(),
            np.rad2deg(solution.getDependentColumn(coord).to_numpy()),
                color='blue', lw=3,
                label=f'Spline-based IK')

axes.flatten()[-1].set_xlabel('time (s)')
axes.flatten()[0].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('compare_joint_kinematics.png', dpi=150, bbox_inches='tight')
# plt.show()
