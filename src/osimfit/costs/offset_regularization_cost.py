import casadi as ca

from .cost import Cost, CostEvaluation, CostWorkspace


class OffsetRegularizationCost(Cost):
    """
    A quadratic penalty on marker and frame XYZ offsets, penalizing offsets away from
    zero:

        cost = weight * sum_i offset_i^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    """
    scope = 'global'
    required_inputs = frozenset({'marker_offsets', 'frame_offsets'})

    def __init__(self, weight: float):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

    def expression(self, workspace: CostWorkspace,
                   evaluation: CostEvaluation) -> ca.MX:
        inputs = evaluation.inputs
        offsets = ca.vertcat(inputs.marker_offsets, inputs.frame_offsets)
        return self.weight * ca.sum(offsets**2)
