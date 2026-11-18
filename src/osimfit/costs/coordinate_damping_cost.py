import casadi as ca

from ..model import ModelCache
from .cost import Cost, CostWorkspace, CostEvaluation


class CoordinateDampingCost(Cost):
    """
    A quadratic penalty that acts like a damper on selected coordinates:

        cost = weight * sum_i c_i * qdot_i^2

    Parameters
    ----------
    coefficients: dict[str, float]
        Mapping from absolute coordinate path to that coordinate's non-negative damping
        coefficient. Only the coordinates named here are penalized.
    weight: float, optional
        Non-negative scalar applied to the whole sum. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` or any coefficient is negative, `coefficients` is empty, or if
        a coordinate path does not match a coordinate in the model.
    """
    required_inputs = frozenset({'speeds'})

    def __init__(self, coefficients: dict[str, float], weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        if not coefficients:
            raise ValueError(
                'CoordinateDampingCost requires at least one coordinate damping '
                'coefficient.')
        for path, coefficient in coefficients.items():
            if coefficient < 0:
                raise ValueError(
                    f'Expected the damping coefficient for {path} to be non-negative, '
                    f'but got {coefficient}.')
        self.weight = weight
        self.coefficients = dict(coefficients)

    def expression(self, workspace: CostWorkspace,
                   evaluation: CostEvaluation) -> ca.MX:
        speeds = evaluation.inputs.speeds
        if speeds.numel() == 0:
            return 0
        cache = workspace.cache
        indexes, coefficients = cache.memoize(
            self, 'speeds', lambda: self._resolve(cache.mc))
        penalty = 0
        for index, coefficient in zip(indexes, coefficients):
            penalty += coefficient * speeds[index]**2
        return self.weight * penalty

    def _resolve(self, mc: ModelCache) -> tuple[list[int], list[float]]:
        """
        Resolve each penalized coordinate's index into the speeds input and its damping
        coefficient.
        """
        order = list(mc.coordinate_u_map)
        indexes: list[int] = []
        coefficients: list[float] = []
        for path, coefficient in self.coefficients.items():
            if path not in mc.coordinate_u_map:
                known = 'is a dependent coordinate' if mc.model.hasComponent(path) \
                    else 'is not a coordinate in the model'
                raise ValueError(
                    f'Cannot apply a damping coefficient to {path}: it {known}. '
                    f'Expected one of the model\'s independent coordinates.')
            indexes.append(order.index(path))
            coefficients.append(float(coefficient))
        return indexes, coefficients
