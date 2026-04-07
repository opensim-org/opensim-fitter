import os
import opensim as osim

model_fpath = os.path.join('..', 'data', 'acl' , 'jump_1', 'jump_1_scaled_adjusted.osim')
model = osim.Model(model_fpath)
model.initSystem()


model.updForceSet().clearAndDestroy()
model.updDisplayHints().set_show_stations(True)

# osim.VisualizerUtilities.showModel(model)


ik_solution_fpath = os.path.join('..', 'data', 'acl' , 'jump_1',
                                 'pose_0_ik_solution.sto')
ik_solution = osim.TimeSeriesTable(ik_solution_fpath)
ik_solution.addTableMetaDataString('inDegrees', 'no')

osim.VisualizerUtilities.showMotion(model, ik_solution)