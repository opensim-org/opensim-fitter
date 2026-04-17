import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from .utilities import get_coordinate_indexes, get_ipopt_options
from .callbacks import TrackingCostJacobianCallback


class Solver(ABC):

    @abstractmethod
    def solve(self) -> osim.TimeSeriesTable:
        pass


class InverseKinematicsSolver(Solver):

    def __init__(self, model, positions, orientations, convergence_tolerance=1e-6,
                 position_weight=1.0, orientation_weight=1.0,
                 smoothness_weight=0.01):

        # Load the model.
        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpRemoveMuscles())
        self.model =  modelProcessor.process()
        self.state = self.model.initSystem()
        # For now, disallow models with joints where qdot != u.
        assert(self.state.getNQ() == self.state.getNU())

        # Create a mapping between coordinate paths and their indexes in the state
        # vector.
        self.coordinates_map = get_coordinate_indexes(self.model,
                                                      skip_dependent_coordinates=True)
        self.coordinate_indexes = list(self.coordinates_map.values())

        # Tracking data.
        self.positions = positions
        self.orientations = orientations

        # Optimization settings.
        self.convergence_tolerance = convergence_tolerance
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.smoothness_weight = smoothness_weight


    def _build_solver(self, frame_paths, positions, quaternions, weights):
        x = ca.SX.sym('x', len(self.coordinate_indexes))
        p = ca.SX.sym('p', len(self.coordinate_indexes))
        callback = TrackingCostJacobianCallback('tracking_cost', self.model,
                                                self.coordinate_indexes,
                                                frame_paths, positions, quaternions,
                                                weights)
        tracking_cost = ca.Function('f', [x], [callback(x)])
        f = tracking_cost(x) + weights['smoothness'] * ca.sumsqr(x - p)
        nlp = {'x': x, 'p': p, 'f': f}
        opts = {}
        opts['ipopt'] = get_ipopt_options(self.convergence_tolerance)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return callback, solver

    def solve(self) -> osim.TimeSeriesTable:

        # Load tracking data
        # ------------------
        frame_paths = self.positions.getColumnLabels()
        times = self.positions.getIndependentColumn()

        # Inverse kinematics
        # ------------------
        # Define initial guess and bounds.
        # This utility retrieves a mapping between coordinate paths and their indexes in the
        # state vector.
        x0 = []
        lbx = []
        ubx = []
        for coord_path, ix in self.coordinates_map.items():
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # Solve position-only optimization to create an inital guess for the full IK
        # problem.
        print('Solving initial guess optimization...')
        callback, solver = self._build_solver(
            frame_paths,
            self.positions.getRowAtIndex(0),
            self.orientations.getRowAtIndex(0),
            {'position': 10.0*self.position_weight,
             'orientation': 0.1*self.orientation_weight,
             'smoothness': 0.01*self.smoothness_weight}
        )
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=x0)
        x0 = sol['x']

        # Build the callback and solver once for the main time-stepping loop.
        weights = {'position': self.position_weight,
                   'orientation': self.orientation_weight,
                   'smoothness': self.smoothness_weight}
        callback, solver = self._build_solver(
            frame_paths,
            self.positions.getRowAtIndex(0),
            self.orientations.getRowAtIndex(0),
            weights
        )

        # Iterate over all of the time steps in the tracking data and solve the
        # optimization problem at each time step.
        statesTraj = osim.StatesTrajectory()
        for itime, time in enumerate(times):
            print(f'Solving time {itime+1} of {len(times)} (t={time:.3f} s)...')

            callback.update_data(self.positions.getRowAtIndex(itime),
                                 self.orientations.getRowAtIndex(itime))
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=x0)

            # Write solution into callback.state — avoids calling initSystem() again,
            # which would invalidate the state handle held by the callback.
            # StatesTrajectory.append() copies the state by value, so reuse is safe.
            callback.state.setTime(time)
            q = np.zeros(callback.state.getNQ())
            q[self.coordinate_indexes] = np.squeeze(sol['x'].full())
            callback.state.setQ(osim.Vector.createFromMat(q))
            statesTraj.append(callback.state)

            # Use the solution for the current time step as the initial guess for the next
            # time step.
            x0 = sol['x']

        # Export the solution to a .sto file.
        statesTable = statesTraj.exportToTable(self.model)

        return statesTable


class BSplineInverseKinematicsSolver(Solver):

    def __init__(self, model, positions, orientations, convergence_tolerance=1e-6,
                 position_weight=1.0, orientation_weight=1.0,
                 smoothness_weight=0.01):

        # Load the model.
        modelProcessor = osim.ModelProcessor(model)
        modelProcessor.append(osim.ModOpRemoveMuscles())
        self.model =  modelProcessor.process()
        self.state = self.model.initSystem()
        # For now, disallow models with joints where qdot != u.
        assert(self.state.getNQ() == self.state.getNU())

        # Create a mapping between coordinate paths and their indexes in the state
        # vector.
        self.coordinates_map = get_coordinate_indexes(self.model,
                                                      skip_dependent_coordinates=True)
        self.coordinate_indexes = list(self.coordinates_map.values())

        # Tracking data.
        self.positions = positions
        self.orientations = orientations

        # Optimization settings.
        self.convergence_tolerance = convergence_tolerance
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.smoothness_weight = smoothness_weight


    def _build_solver(self, frame_paths, positions, quaternions, weights):
        x = ca.SX.sym('x', len(self.coordinate_indexes))
        p = ca.SX.sym('p', len(self.coordinate_indexes))
        callback = TrackingCostJacobianCallback('tracking_cost', self.model,
                                                self.coordinate_indexes,
                                                frame_paths, positions, quaternions,
                                                weights)
        tracking_cost = ca.Function('f', [x], [callback(x)])
        f = tracking_cost(x) + weights['smoothness'] * ca.sumsqr(x - p)
        nlp = {'x': x, 'p': p, 'f': f}
        opts = {}
        opts['ipopt'] = get_ipopt_options(self.convergence_tolerance)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return callback, solver

    def solve(self) -> osim.TimeSeriesTable:

        # Load tracking data
        # ------------------
        frame_paths = self.positions.getColumnLabels()
        times = self.positions.getIndependentColumn()

        # Inverse kinematics
        # ------------------
        # Define initial guess and bounds.
        # This utility retrieves a mapping between coordinate paths and their indexes in the
        # state vector.
        x0 = []
        lbx = []
        ubx = []
        for coord_path, ix in self.coordinates_map.items():
            coord = osim.Coordinate.safeDownCast(self.model.getComponent(coord_path))
            x0.append(coord.getDefaultValue())
            lbx.append(coord.getRangeMin())
            ubx.append(coord.getRangeMax())

        # Solve position-only optimization to create an inital guess for the full IK
        # problem.
        print('Solving initial guess optimization...')
        callback, solver = self._build_solver(
            frame_paths,
            self.positions.getRowAtIndex(0),
            self.orientations.getRowAtIndex(0),
            {'position': 10.0*self.position_weight,
             'orientation': 0.1*self.orientation_weight,
             'smoothness': 0.01*self.smoothness_weight}
        )
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=x0)
        x0 = sol['x']

        # Build the callback and solver once for the main time-stepping loop.
        weights = {'position': self.position_weight,
                   'orientation': self.orientation_weight,
                   'smoothness': self.smoothness_weight}
        callback, solver = self._build_solver(
            frame_paths,
            self.positions.getRowAtIndex(0),
            self.orientations.getRowAtIndex(0),
            weights
        )

        # Iterate over all of the time steps in the tracking data and solve the
        # optimization problem at each time step.
        statesTraj = osim.StatesTrajectory()
        for itime, time in enumerate(times):
            print(f'Solving time {itime+1} of {len(times)} (t={time:.3f} s)...')

            callback.update_data(self.positions.getRowAtIndex(itime),
                                 self.orientations.getRowAtIndex(itime))
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=x0)

            # Write solution into callback.state — avoids calling initSystem() again,
            # which would invalidate the state handle held by the callback.
            # StatesTrajectory.append() copies the state by value, so reuse is safe.
            callback.state.setTime(time)
            q = np.zeros(callback.state.getNQ())
            q[self.coordinate_indexes] = np.squeeze(sol['x'].full())
            callback.state.setQ(osim.Vector.createFromMat(q))
            statesTraj.append(callback.state)

            # Use the solution for the current time step as the initial guess for the next
            # time step.
            x0 = sol['x']

        # Export the solution to a .sto file.
        statesTable = statesTraj.exportToTable(self.model)

        return statesTable

