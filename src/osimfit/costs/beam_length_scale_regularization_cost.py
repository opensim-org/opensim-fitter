import casadi as ca

from .cost import CostInput, SymbolicCost


class BeamLengthScaleRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on the beam length factors that encourages each toward
    `target`:

        cost = weight * sum_i (f_i - target)^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-factor target value. Default is 1.0.
    """
    required_inputs = frozenset({'beam_length_scales'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.beam_length_scales - self.target)**2)
