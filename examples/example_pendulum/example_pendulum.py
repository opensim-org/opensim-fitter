import os
import opensim as osim
import osimfit as ofit

# Example Pendulum
# ----------------

# Create a simple pendulum model.
model = osim.ModelFactory.createDoublePendulum()
state = model.initSystem()

# Run a forward simulations at a fixed framerate.
manager = osim.Manager(model)
manager.setIntegratorFixedStepSize(0.01)
manager.initialize(state)
manager.integrate(1.0)

# Extract the marker trajectories.
states = manager.getStatesTable()
controls = osim.TimeSeriesTable(states.getIndependentColumn())
output_paths = osim.StdVectorString()
output_paths.append('/markerset/.*location')
markers = osim.analyzeVec3(model, states, controls, output_paths)

# Save the marker trajectories to a .trc file.
markers.addTableMetaDataString('DataRate', '100.0')
markers.addTableMetaDataString('Units', 'm')
trc = osim.TRCFileAdapter()
trc.write(markers, 'markers.trc')

# Load the marker trajectories from the .trc file.

