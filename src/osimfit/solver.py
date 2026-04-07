import os
import opensim as osim
import casadi as ca
import numpy as np
from .utilities import C3D, get_coordinate_indexes, get_ipopt_options
from .callbacks import TrackingCostCallback, TrackingCostJacobianCallback


def run_optimization(model, coordinate_indexes, frame_paths, positions, quaternions,
                     weights, x0, lbx, ubx, convergence_tolerance,
                     finite_differences=False):
    # Declare optimization variables.
    x = ca.SX.sym('x', len(coordinate_indexes))

    # Construct the callback function defining the tracking cost.
    # If 'finite_differences' is True, the Jacobian will be computed using finite
    # differences. Otherwise, a callback function that provides an analytical
    # Jacobian will be used.
    if finite_differences:
        track = TrackingCostCallback('tracking_cost', model, coordinate_indexes,
                                    frame_paths, positions, quaternions,
                                    weights, {'enable_fd': True})
    else:
        track = TrackingCostJacobianCallback('tracking_cost', model, coordinate_indexes,
                                            frame_paths, positions, quaternions,
                                            weights)
    tracking_cost = ca.Function('f', [x], [track(x)])

    # The total cost function includes a smoothness term to penalize large
    # deviations from the previous solution.
    f = tracking_cost(x) + weights['smoothness'] * ca.sumsqr(x - x0)

    # Form the non-linear program (NLP).
    nlp = {'x': x, 'f': f}

    # Allocate a solver.
    opts = {}
    opts['ipopt'] = get_ipopt_options(convergence_tolerance)
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the NLP.
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)
    return sol


def run_inverse_kinematics(model_path, trial_path, c3d_filename, offset_frame_map,
                           weights, convergence_tolerance, finite_differences=False):

    # Load model
    # ----------
    modelProcessor = osim.ModelProcessor(model_path)
    modelProcessor.append(osim.ModOpRemoveMuscles())
    model = modelProcessor.process()
    state = model.initSystem()
    # For now, disallow models with joints where qdot != u.
    assert(state.getNQ() == state.getNU())

    # Load tracking data
    # ------------------
    columns_to_ignore = ['worldbody', 'head', 'pelvis_shifted', 'l_clavicle',
                         'r_clavicle']
    label_map = {k: f'{v}/{k}' for k, v in offset_frame_map.items()}
    c3d_filepath = os.path.join(trial_path, c3d_filename)
    c3d = C3D(c3d_filepath, columns_to_ignore=columns_to_ignore, label_map=label_map)
    positions_table = c3d.get_positions_table()
    quaternions_table = c3d.get_quaternions_table()
    frame_paths = positions_table.getColumnLabels()
    times = positions_table.getIndependentColumn()

    # Inverse kinematics
    # ------------------
    # Define initial guess and bounds.
    # This utility retrieves a mapping between coordinate paths and their indexes in the
    # state vector.
    coordinates_map = get_coordinate_indexes(model, skip_dependent_coordinates=True)
    coordinate_indexes = list(coordinates_map.values())
    x0 = []
    lbx = []
    ubx = []
    for coord_path, ix in coordinates_map.items():
        coord = osim.Coordinate.safeDownCast(model.getComponent(coord_path))
        x0.append(coord.getDefaultValue())
        lbx.append(coord.getRangeMin())
        ubx.append(coord.getRangeMax())

    # Declare optimization variables.
    x = ca.SX.sym('x', len(coordinate_indexes))

    # Solve position-only optimization to create an inital guess for the full IK
    # problem.
    initial_weights = {'position': 10.0,
                       'orientation': 1.0,
                       'smoothness': 0.01}
    positions = positions_table.getRowAtIndex(0)
    quaternions = quaternions_table.getRowAtIndex(0)
    print('Solving initial guess optimization...')
    sol = run_optimization(model, coordinate_indexes, frame_paths, positions,
                           quaternions, initial_weights, x0, lbx, ubx,
                           convergence_tolerance,
                           finite_differences=finite_differences)
    x0 = sol['x']

    # Iterate over all of the time steps in the tracking data and solve the
    # optimization problem at each time step.
    statesTraj = osim.StatesTrajectory()
    for itime, time in enumerate(times):
        print(f'Solving time {itime+1} of {len(times)} (t={time:.3f} s)...')

        # Construct the callback function defining the tracking cost.
        positions = positions_table.getRowAtIndex(itime)
        quaternions = quaternions_table.getRowAtIndex(itime)
        sol = run_optimization(model, coordinate_indexes, frame_paths,
                               positions, quaternions, weights, x0, lbx, ubx,
                               convergence_tolerance,
                               finite_differences=finite_differences)

        # Save solution
        state = model.initSystem()
        state.setTime(time)
        q = np.zeros(state.getNQ())
        q[coordinate_indexes] = np.squeeze(sol['x'].full())
        state.setQ(osim.Vector.createFromMat(q))
        statesTraj.append(state)

        # Use the solution for the current time step as the initial guess for the next
        # time step.
        x0 = sol['x']

    # Export the solution to a .sto file.
    statesTable = statesTraj.exportToTable(model)
    trial_name = c3d_filename.replace('.c3d', '')
    solution_path = os.path.join(trial_path, f'{trial_name}_ik_solution.sto')
    osim.STOFileAdapter.write(statesTable, solution_path)