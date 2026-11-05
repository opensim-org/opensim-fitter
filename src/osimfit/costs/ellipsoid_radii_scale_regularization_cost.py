import casadi as ca

from .cost import CostInput, SymbolicCost


class EllipsoidRadiiScaleRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on the ellipsoid radii factors that encourages each toward
    `target`:

        cost = weight * sum_i (f_i - target)^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    target: float, optional
        Per-factor target value. Default is 1.0.
    """
    scope = 'global'
    required_inputs = frozenset({'ellipsoid_radii_scales'})

    def __init__(self, weight: float, target: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.target = target

    def evaluate(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.ellipsoid_radii_scales - self.target)**2)
