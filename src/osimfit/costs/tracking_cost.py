import numpy as np
import opensim as osim
from typing import TYPE_CHECKING

from ..data_sources import Trial
from .cost import TrackingCostBase
from .terms import FrameTrackingTerm, MarkerTrackingTerm, TaskSet

if TYPE_CHECKING:
    from .solve_cache import SolveCache


class TrackingCost(TrackingCostBase):
    """
    The weighted, squared error between the model's markers and frames and a trial's
    reference data, as a function of the model's generalized coordinates.

    Parameters
    ----------
    position_weight, orientation_weight, marker_weights, frame_position_weights,
    frame_orientation_weights
        See `TrackingCostBase`, which defines the weighting these costs apply.
    """
    required_inputs = frozenset({'coordinates'})

    def create_task_set(self, cache: 'SolveCache', trial: Trial,
                        itime: int) -> TaskSet:
        task_set = TaskSet(MarkerTrackingTerm(cache.mc), FrameTrackingTerm(cache.mc))

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                frame_position_weight, frame_orientation_weight = \
                    self.get_frame_weights(frame_path)
                task_set.frame_term.add_frame(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=frame_position_weight,
                    orientation_weight=frame_orientation_weight,
                    station_cache=cache.frame_cache(frame_path))

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                task_set.marker_term.add_marker(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.get_marker_weight(marker_path),
                    station_cache=cache.station_cache(marker_path))

        return task_set

    def _apply_state(self, cache: 'SolveCache', values: dict[str, np.ndarray]):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(cache.state.getNQ())
        q[cache.mc.coordinate_q_indexes] = values['coordinates']
        cache.state.setQ(osim.Vector.createFromMat(q))
        cache.mc.model.realizePosition(cache.state)

    def calc_value(self, cache, context, values):
        self._apply_state(cache, values)
        return (context.marker_term.calc_error(cache.state) +
                context.frame_term.calc_error(cache.state))

    def calc_jacobian(self, cache, context, values):
        self._apply_state(cache, values)
        return [context.marker_term.calc_jacobian(cache.state)[0] +
                context.frame_term.calc_jacobian(cache.state)[0]]
