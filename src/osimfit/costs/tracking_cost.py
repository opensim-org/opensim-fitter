import numpy as np
import opensim as osim

from ..data_sources import Trial
from ..model import ModelCache
from .cost import CallbackCostRep, CostInput, Function, TrackingCostBase
from .tracking_terms import FrameTrackingTerm, MarkerTrackingTerm


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

    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> 'TrackingCostRep':
        rep = TrackingCostRep(name, mc)

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                frame_position_weight, frame_orientation_weight = \
                    self.get_frame_weights(frame_path)
                rep.add_frame_tracking_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=frame_position_weight,
                    orientation_weight=frame_orientation_weight)

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                rep.add_marker_tracking_cost_term(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.get_marker_weight(marker_path))

        return rep


class TrackingCostRep(CallbackCostRep):
    """
    The rep of a `TrackingCost`: a callback that evaluates the sum of tracking cost
    terms over a set of model frames and markers with respect to the model's
    generalized coordinates.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian. Default is ``False``.
    """
    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        Function.__init__(self, name, mc, enable_fd=enable_fd)
        self.marker_term = MarkerTrackingTerm(mc)
        self.frame_term = FrameTrackingTerm(mc)

    def apply_state(self, arg):
        """
        Apply the input coordinates to the model state and realize the system to the
        position stage.
        """
        q = np.zeros(self.state.getNQ())
        q[self.mc.coordinate_q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_tracking_cost_term(self, marker_path: str, position: osim.Vec3,
                                      weight: float = 1.0):
        self.marker_term.add_marker(marker_path, position, weight=weight)

    def add_frame_tracking_cost_term(self, frame_path: str,
                                     position: osim.Vec3,
                                     orientation: osim.Quaternion,
                                     position_weight: float = 1.0,
                                     orientation_weight: float = 1.0):
        self.frame_term.add_frame(frame_path, position, orientation,
                                  position_weight=position_weight,
                                  orientation_weight=orientation_weight)

    def _eval(self, arg):
        self.apply_state(arg)
        error = (self.marker_term.calc_error(self.state) +
                 self.frame_term.calc_error(self.state))
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        J = (self.marker_term.calc_jacobian(self.state)[0] +
             self.frame_term.calc_jacobian(self.state)[0])
        empty = np.zeros((1, 0))
        return [J] + [empty] * (len(CostInput.INPUT_ORDER) - 1)
