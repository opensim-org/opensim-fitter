import os
import opensim as osim
import numpy as np
from .utilities import C3D

def compute_scale_factor(state, offset_frame_1, offset_frame_2, positions,
                         frame_1, frame_2):
    # Magnitude of relative position between the two model frames.
    offset_frame_1_position = offset_frame_1.getPositionInGround(state).to_numpy()
    offset_frame_2_position = offset_frame_2.getPositionInGround(state).to_numpy()
    offset_frame_distance = np.linalg.norm(offset_frame_1_position -
                                           offset_frame_2_position)

    # Magnitude of relative position between the two Theia frames. Average over all the
    # frames in the trial.
    times = positions.getIndependentColumn()
    theia_frame_distances = np.zeros(len(times))
    for i, t in enumerate(times):
        frame_1_position = positions.getDependentColumn(frame_1)[i]
        frame_2_position = positions.getDependentColumn(frame_2)[i]
        theia_frame_distances[i] = np.linalg.norm(frame_1_position.to_numpy() -
                                                  frame_2_position.to_numpy())

    theia_frame_distance = np.mean(theia_frame_distances)

    scale_factor = theia_frame_distance / offset_frame_distance
    return scale_factor


def create_scale(segment, scale_rules, offset_frame_map, positions, model, state):
    scale = osim.Scale()
    scale.setSegmentName(segment)
    factors = osim.Vec3(1.0)
    for rule in scale_rules:
        frame_1 = rule[0]
        frame_2 = rule[1]
        index = rule[2]
        offset_frame_1 = osim.PhysicalFrame.safeDownCast(
            model.getComponent(f'{offset_frame_map[frame_1]}/{frame_1}'))
        offset_frame_2 = osim.PhysicalFrame.safeDownCast(
            model.getComponent(f'{offset_frame_map[frame_2]}/{frame_2}'))
        factors[index] = compute_scale_factor(state, offset_frame_1, offset_frame_2,
                                              positions, frame_1, frame_2)
    scale.setScaleFactors(factors)

    return scale


def scale_model(generic_model_fpath, trial_path, c3d_filename, offset_frame_map,
                scale_rules, scaled_model_name):

    # Load the model generic model.
    model = osim.Model(generic_model_fpath)
    state = model.initSystem()

    # Import the C3D file and load the frame position data.
    c3d = C3D(os.path.join(trial_path, c3d_filename))
    positions = c3d.get_positions_table()

    # Create scale factors
    # --------------------
    scaleset = osim.ScaleSet()
    for segment, rules in scale_rules.items():
        scale = create_scale(segment, rules, offset_frame_map, positions, model, state)
        scaleset.cloneAndAppend(scale)

    # Apply symmetry to scale factors.
    # --------------------------------
    for i in range(scaleset.getSize()):
        scale_r = scaleset.get(i)
        segment_name_r = scale.getSegmentName()
        if segment_name_r.endswith('_r'):
            segment_name_l = segment_name_r.replace('_r', '_l')
            for j in range(scaleset.getSize()):
                scale_l = scaleset.get(j)
                if scale_l.getSegmentName() == segment_name_l:
                    factors_r = scale_r.getScaleFactors()
                    factors_l = scale_l.getScaleFactors()
                    avg_factors = osim.Vec3(
                        0.5 * (factors_r[0] + factors_l[0]),
                        0.5 * (factors_r[1] + factors_l[1]),
                        0.5 * (factors_r[2] + factors_l[2]))
                    scale_r.setScaleFactors(avg_factors)
                    scale_l.setScaleFactors(avg_factors)

    # Scale the model
    # ---------------
    model.scale(state, scaleset, True)
    model.setName(scaled_model_name)
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(os.path.join(trial_path, f'{scaled_model_name}.osim'))
