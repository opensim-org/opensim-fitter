import numpy as np
import opensim as osim

from ..data_sources import Trial
from ..model import ModelCache
from .cost import CallbackCostRep, Function, TrackingCostBase
from .tracking_terms import FrameBilevelTerm, MarkerBilevelTerm


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

    def create_rep(self, name: str, mc: ModelCache, trial: Trial,
                   itime: int) -> 'BilevelCostRep':
        rep = BilevelCostRep(name, mc)
        marker_index_of = {path: i for i, grp in enumerate(mc.marker_offset_groups)
                           for path in grp.component_paths}
        frame_index_of = {path: i for i, grp in enumerate(mc.frame_offset_groups)
                          for path in grp.component_paths}

        for data in trial.frame_data:
            for iframe, frame_path in enumerate(data.labels):
                frame_position_weight, frame_orientation_weight = \
                    self.get_frame_weights(frame_path)
                rep.add_frame_bilevel_cost_term(
                    frame_path,
                    data.positions.getRowAtIndex(itime).getElt(0, iframe),
                    data.orientations.getRowAtIndex(itime).getElt(0, iframe),
                    position_weight=frame_position_weight,
                    orientation_weight=frame_orientation_weight,
                    offset_group_index=frame_index_of.get(frame_path))

        for data in trial.marker_data:
            for imarker, marker_path in enumerate(data.labels):
                rep.add_marker_bilevel_cost_term(
                    marker_path,
                    data.positions.getRowAtIndex(itime).getElt(0, imarker),
                    weight=self.get_marker_weight(marker_path),
                    offset_group_index=marker_index_of.get(marker_path))

        return rep


class BilevelCostRep(CallbackCostRep):
    """
    The rep of a `BilevelCost`: a callback that evaluates the sum of tracking cost
    terms over a set of model markers and frames with respect to the model's generalized
    coordinates, a set of body scales, a set of per-marker/frame XYZ placement offsets,
    a set of ellipsoid radii, and a set of beam lengths.

    Parameters
    ----------
    name: str
        The name of the callback function.
    mc: ModelCache
        The `ModelCache` wrapping the OpenSim model used for evaluating the function and
        its Jacobian and caching model information. Contains parameter information
        (e.g., body scale groups) for relevant optimization parameters.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback instead of using its analytic
        Jacobian. Default is ``False``.
    """

    def __init__(self, name: str, mc: ModelCache, enable_fd: bool = False):
        Function.__init__(self, name, mc, enable_fd=enable_fd)
        self.marker_term = MarkerBilevelTerm(mc)
        self.frame_term = FrameBilevelTerm(mc)

    def apply_state(self, arg):
        """
        Apply input coordinates, body-scale variables, offset variables, and
        joint-level variables (ellipsoid radii and beam lengths) to the model State,
        then realize to Position.
        """
        body_scales = np.squeeze(arg[1].full())
        body_scales = np.atleast_1d(body_scales).astype(float)
        self.mc.set_scaled_mobilizer_frame_positions(self.state, body_scales)

        marker_offsets = np.atleast_1d(np.squeeze(arg[2].full())).astype(float)
        self.marker_term.apply_state(body_scales, marker_offsets)

        frame_offsets = np.atleast_1d(np.squeeze(arg[3].full())).astype(float)
        self.frame_term.apply_state(body_scales, frame_offsets)

        ellipsoid_radii_scales = np.atleast_1d(np.squeeze(arg[4].full())).astype(float)
        self.mc.set_ellipsoid_radii_from_scales(self.state, ellipsoid_radii_scales)

        beam_length_scales = np.atleast_1d(np.squeeze(arg[5].full())).astype(float)
        self.mc.set_beam_length_from_scales(self.state, beam_length_scales)

        q = np.zeros(self.state.getNQ())
        q[self.mc.coordinate_q_indexes] = np.squeeze(arg[0].full())
        self.state.setQ(osim.Vector.createFromMat(q))
        self.mc.model.realizePosition(self.state)

    def add_marker_bilevel_cost_term(self, marker_path: str, position: osim.Vec3,
                                     weight: float = 1.0,
                                     offset_group_index: int | None = None):
        self.marker_term.add_marker(marker_path, position, weight=weight,
                                    offset_group_index=offset_group_index)

    def add_frame_bilevel_cost_term(self, frame_path: str, position: osim.Vec3,
                                    orientation: osim.Quaternion,
                                    position_weight: float = 1.0,
                                    orientation_weight: float = 1.0,
                                    offset_group_index: int | None = None):
        self.frame_term.add_frame(frame_path, position, orientation, position_weight,
                                  orientation_weight,
                                  offset_group_index=offset_group_index)

    def _eval(self, arg):
        self.apply_state(arg)
        error = 0
        error += self.marker_term.calc_error(self.state)
        error += self.frame_term.calc_error(self.state)
        return [error]

    def _jac_eval(self, arg):
        self.apply_state(arg)
        Jq_m, Js_m, Jmo, Jr_m, Jl_m = self.marker_term.calc_jacobian(self.state)
        Jq_f, Js_f, Jfo, Jr_f, Jl_f = self.frame_term.calc_jacobian(self.state)
        return [Jq_m + Jq_f, Js_m + Js_f, Jmo, Jfo, Jr_m + Jr_f, Jl_m + Jl_f]
