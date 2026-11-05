import casadi as ca

from .cost import CostInput, SymbolicCost


class BodyScaleRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on body-scale factors that encourages each toward `target`:

        cost = weight * sum_i (s_i - target)^2

    Keeping the scales near ``target`` (typically 1.0, i.e., identity scaling) means the
    optimizer only deviates from the nominal scaling when doing so substantially
    improves the primary tracking cost.

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-component target value. Default is 1.0.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.body_scales - self.target)**2)
