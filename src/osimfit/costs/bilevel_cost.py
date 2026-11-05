import numpy as np
import opensim as osim
from typing import TYPE_CHECKING

from ..data_sources import Trial
from .cost import TrackingCostBase
from .terms import FrameBilevelTerm, MarkerBilevelTerm, TaskSet

if TYPE_CHECKING:
    from .solve_cache import SolveCache


class BilevelCost(TrackingCostBase):
    """
    The tracking cost of `TrackingCost`, as a function of the model's generalized
    coordinates, its body scales, its per-marker/frame XYZ placement offsets, and its
    joint-level geometry (e.g., ellipsoid radii).

    Parameters
    ----------
    position_weight, orientation_weight, marker_weights, frame_position_weights,
    frame_orientation_weights
        See `TrackingCostBase`, which defines the weighting these costs apply.
    """
    required_inputs = frozenset(
        {'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets',
         'ellipsoid_radii_scales', 'beam_length_scales'})

    def create_task_set(self, cache: 'SolveCache', trial: Trial,
                        itime: int) -> TaskSet:
        mc = cache.mc
        task_set = TaskSet(MarkerBilevelTerm(mc), FrameBilevelTerm(mc))
        marker_index_of = {path: i for i, grp in enumerate(mc.marker_offset_groups)
                           for path in grp.component_paths}
        frame_index_of = {path: i for i, grp in enumerate(mc.frame_offset_groups)
                          for path in grp.component_paths}

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
                    offset_group_index=frame_index_of.get(frame_path),
                    station_cache=cache.frame_cache(frame_path))

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                task_set.marker_term.add_marker(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.get_marker_weight(marker_path),
                    offset_group_index=marker_index_of.get(marker_path),
                    station_cache=cache.station_cache(marker_path))

        return task_set

    def _apply_state(self, cache: 'SolveCache', context: TaskSet,
                     values: dict[str, np.ndarray]):
        """
        Apply input coordinates, body-scale variables, offset variables, and
        joint-level variables (ellipsoid radii and beam lengths) to the model State,
        then realize to Position.
        """
        mc, state = cache.mc, cache.state
        body_scales = values['body_scales']
        mc.set_scaled_mobilizer_frame_positions(state, body_scales)
        context.marker_term.apply_state(body_scales, values['marker_offsets'])
        context.frame_term.apply_state(body_scales, values['frame_offsets'])
        mc.set_ellipsoid_radii_from_scales(state, values['ellipsoid_radii_scales'])
        mc.set_beam_length_from_scales(state, values['beam_length_scales'])

        q = np.zeros(state.getNQ())
        q[mc.coordinate_q_indexes] = values['coordinates']
        state.setQ(osim.Vector.createFromMat(q))
        mc.model.realizePosition(state)

    def calc_value(self, cache, context, values):
        self._apply_state(cache, context, values)
        error = 0
        error += context.marker_term.calc_error(cache.state)
        error += context.frame_term.calc_error(cache.state)
        return error

    def calc_jacobian(self, cache, context, values):
        self._apply_state(cache, context, values)
        Jq_m, Js_m, Jmo, Jr_m, Jl_m = context.marker_term.calc_jacobian(cache.state)
        Jq_f, Js_f, Jfo, Jr_f, Jl_f = context.frame_term.calc_jacobian(cache.state)
        return [Jq_m + Jq_f, Js_m + Js_f, Jmo, Jfo, Jr_m + Jr_f, Jl_m + Jl_f]
