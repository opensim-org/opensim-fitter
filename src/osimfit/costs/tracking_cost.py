import numpy as np
import casadi as ca
import opensim as osim
from abc import abstractmethod

from ..data_sources import Trial
from .cost import CostBase, CostCache, CostInput, CostWorkspace
from .tracking_terms import FrameTrackingTerm, MarkerTrackingTerm, TrackingTaskSet


class TrackingCostBase(CostBase):
    """
    A cost evaluated at a single time sample of a single trial, through a
    `CostFunction` whose context is that sample's `TrackingTaskSet`.

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

    def expression(self, workspace: CostWorkspace, input: CostInput, trial: Trial,
                   itime: int, name: str) -> ca.MX:
        """
        Return this cost's contribution to the objective at one time sample of one
        trial, evaluated by a `CostFunction` carrying that sample's `TrackingTaskSet`.

        Parameters
        ----------
        workspace: CostWorkspace
            Owns the sample's callback and tracking task set, and supplies the shared
            cache for resolving model-bound quantities.
        input: CostInput
            The optimization variables.
        trial: Trial
            The trial supplying the reference data.
        itime: int
            Index of the time sample within `trial` to evaluate.
        name: str
            The name of the CasADi callback evaluating this sample.
        """
        tracking_task_set = self.create_tracking_task_set(workspace.cache, trial, itime)
        workspace.tracking_task_sets.append(tracking_task_set)
        return workspace.add_callback(name, self, tracking_task_set)(input)

    @abstractmethod
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

    @abstractmethod
    def calc_value(self, cache: CostCache, context: TrackingTaskSet,
                   values: dict[str, np.ndarray]) -> float:
        """
        Return this cost's scalar value at the sample `context` tracks.

        Parameters
        ----------
        cache: CostCache
            The solver's cache.
        context: TrackingTaskSet
            The tasks returned by `create_tracking_task_set`.
        values: dict[str, np.ndarray]
            The cost's `input_names`, mapped to their numeric values.
        """

    @abstractmethod
    def calc_jacobian(self, cache: CostCache, context: TrackingTaskSet,
                      values: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Return this cost's Jacobian as one ``(1, n)`` block per entry of
        `input_names`, in that order.

        Parameters
        ----------
        cache: CostCache
            The solver's cache.
        context: TrackingTaskSet
            The tasks returned by `create_tracking_task_set`.
        values: dict[str, np.ndarray]
            The cost's `input_names`, mapped to their numeric values.
        """


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

    def create_tracking_task_set(self, cache: CostCache, trial: Trial,
                                 itime: int) -> TrackingTaskSet:
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

    def _apply_state(self, cache: CostCache, values: dict[str, np.ndarray]):
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
