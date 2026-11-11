import numpy as np
import opensim as osim

from ..data_sources import Trial
from .cost import CallbackCost, CostCache, TrialSample
from .tracking_terms import (FrameTrackingTerm, MarkerTrackingTerm,
                             FrameBilevelTerm, MarkerBilevelTerm, TrackingTaskSet)


class TrackingCost(CallbackCost):
    """
    The weighted, squared error between the model's markers and frames and a trial's
    reference data, as a function of the model's generalized coordinates. Evaluated
    through a `CostFunction` whose context is one sample's `TrackingTaskSet`.

    Parameters
    ----------
    position_weight: float, optional
        Overall weight on marker and frame-origin position errors. Default is 1.0.
    orientation_weight: float, optional
        Overall weight on frame orientation errors. Default is 1.0.
    marker_weights: dict[str, float], optional
        Per-marker factors on the position error, keyed by the marker label (i.e., the
        osim.Model component path) model component path. Each multiplies with the
        overall `position_weight`. Absent markers use a weight of 1.0.
    frame_position_weights: dict[str, float], optional
        Per-frame factors on the frame position error, keyed by the frame label (i.e.,
        the osim.Model component path) model component path. Each multiplies with the
        overall `position_weight`. Absent frames use a weight of 1.0.
    frame_orientation_weights: dict[str, float], optional
        Per-frame factors on the frame orientation error, keyed by the frame label
        (i.e., the osim.Model component path) model component path. Each multiplies with
        the overall `orientation_weight`. Absent frames use a weight of 1.0.
    """

    requires_sample = True
    required_inputs = frozenset({'coordinates'})

    def __init__(self, position_weight: float = 1.0,
                 orientation_weight: float = 1.0,
                 marker_weights: dict[str, float] = None,
                 frame_position_weights: dict[str, float] = None,
                 frame_orientation_weights: dict[str, float] = None):
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.marker_weights = dict(marker_weights or {})
        self.frame_position_weights = dict(frame_position_weights or {})
        self.frame_orientation_weights = dict(frame_orientation_weights or {})

    def get_marker_weight(self, marker_path: str) -> float:
        return self.position_weight * self.marker_weights.get(marker_path, 1.0)

    def get_frame_weights(self, frame_path: str) -> tuple[float, float]:
        return (self.position_weight
                * self.frame_position_weights.get(frame_path, 1.0),
                self.orientation_weight
                * self.frame_orientation_weights.get(frame_path, 1.0))

    def create_context(self, cache: CostCache,
                       sample: TrialSample | None = None) -> TrackingTaskSet:
        """Prepare the reference tasks for one sample in the owning workspace."""
        if sample is None:
            raise ValueError(f'{type(self).__name__} requires a trial sample.')
        return self.create_tracking_task_set(cache, sample.trial, sample.index)

    def create_tracking_task_set(self, cache: CostCache, trial: Trial,
                                 itime: int) -> TrackingTaskSet:
        """
        Register this cost's marker and frame tasks for one time sample of one trial.

        Parameters
        ----------
        cache: CostCache
            The solver's cache, which supplies the memoized `StationCache` of each
            tracked marker and frame.
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` to track.
        """
        tracking_task_set = TrackingTaskSet(MarkerTrackingTerm(cache.mc),
                                            FrameTrackingTerm(cache.mc))

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                frame_position_weight, frame_orientation_weight = \
                    self.get_frame_weights(frame_path)
                tracking_task_set.frame_term.add_frame(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=frame_position_weight,
                    orientation_weight=frame_orientation_weight,
                    station_cache=cache.frame_cache(frame_path))

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                tracking_task_set.marker_term.add_marker(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.get_marker_weight(marker_path),
                    station_cache=cache.station_cache(marker_path))

        return tracking_task_set

    def _apply_state(self, cache: CostCache, context: TrackingTaskSet,
                     values: dict[str, np.ndarray]):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(cache.state.getNQ())
        q[cache.mc.coordinate_q_indexes] = values['coordinates']
        cache.state.setQ(osim.Vector.createFromMat(q))
        cache.mc.model.realizePosition(cache.state)

    def calc_value(self, cache, context, values):
        self._apply_state(cache, context, values)
        return (context.marker_term.calc_error(cache.state) +
                context.frame_term.calc_error(cache.state))

    def calc_jacobian(self, cache, context, values):
        self._apply_state(cache, context, values)
        return [context.marker_term.calc_jacobian(cache.state)[0] +
                context.frame_term.calc_jacobian(cache.state)[0]]


class BilevelCost(TrackingCost):
    """
    The tracking cost of `TrackingCost`, as a function of the model's generalized
    coordinates, its body scales, its per-marker/frame XYZ placement offsets, and its
    joint-level geometry (e.g., ellipsoid radii).

    Parameters
    ----------
    position_weight, orientation_weight, marker_weights, frame_position_weights,
    frame_orientation_weights
        See `TrackingCost`, which defines the weighting these costs apply.
    """
    required_inputs = frozenset(
        {'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets',
         'ellipsoid_radii_scales', 'beam_length_scales'})

    def create_tracking_task_set(self, cache: CostCache, trial: Trial,
                                 itime: int) -> TrackingTaskSet:
        mc = cache.mc
        tracking_task_set = TrackingTaskSet(MarkerBilevelTerm(mc), FrameBilevelTerm(mc))
        marker_index_of = {path: i for i, grp in enumerate(mc.marker_offset_groups)
                           for path in grp.component_paths}
        frame_index_of = {path: i for i, grp in enumerate(mc.frame_offset_groups)
                          for path in grp.component_paths}

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                frame_position_weight, frame_orientation_weight = \
                    self.get_frame_weights(frame_path)
                tracking_task_set.frame_term.add_frame(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=frame_position_weight,
                    orientation_weight=frame_orientation_weight,
                    offset_group_index=frame_index_of.get(frame_path),
                    station_cache=cache.frame_cache(frame_path))

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                tracking_task_set.marker_term.add_marker(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.get_marker_weight(marker_path),
                    offset_group_index=marker_index_of.get(marker_path),
                    station_cache=cache.station_cache(marker_path))

        return tracking_task_set

    def _apply_state(self, cache: CostCache, context: TrackingTaskSet,
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

        super()._apply_state(cache, context, values)

    def calc_jacobian(self, cache, context, values):
        self._apply_state(cache, context, values)
        Jq_m, Js_m, Jmo, Jr_m, Jl_m = context.marker_term.calc_jacobian(cache.state)
        Jq_f, Js_f, Jfo, Jr_f, Jl_f = context.frame_term.calc_jacobian(cache.state)
        return [Jq_m + Jq_f, Js_m + Js_f, Jmo, Jfo, Jr_m + Jr_f, Jl_m + Jl_f]
