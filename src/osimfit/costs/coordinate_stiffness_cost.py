import casadi as ca
import opensim as osim

from typing import TYPE_CHECKING

from ..model import ModelCache
from .cost import Cost, CostInput

if TYPE_CHECKING:
    from .solve_cache import SolveCache


class CoordinateStiffnessCost(Cost):
    """
    A quadratic penalty that acts like a spring on selected coordinates, holding each
    near a target value:

        cost = weight * sum_i k_i * (q_i - target_i)^2

    Parameters
    ----------
    stiffnesses: dict[str, float]
        Mapping from absolute coordinate path to that coordinate's non-negative
        stiffness. Only the coordinates named here are penalized.
    targets: dict[str, float], optional
        Mapping from absolute coordinate path to the value that coordinate is pulled
        toward. Any coordinate absent from this mapping is pulled toward its default
        value in the model. Defaults to ``None`` (every target taken from the model).
    weight: float, optional
        Non-negative scalar applied to the whole sum. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` or any stiffness is negative, or `stiffnesses` is empty.
        The coordinate paths are additionally validated against the model when the
        cost is first evaluated in a solve.
    """
    required_inputs = frozenset({'coordinates'})

    def __init__(self, stiffnesses: dict[str, float],
                 targets: dict[str, float] = None, weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        if not stiffnesses:
            raise ValueError(
                'CoordinateStiffnessCost requires at least one coordinate stiffness.')
        for path, stiffness in stiffnesses.items():
            if stiffness < 0:
                raise ValueError(
                    f'Expected the stiffness for {path} to be non-negative, but got '
                    f'{stiffness}.')
        self.weight = weight
        self.stiffnesses = dict(stiffnesses)
        self.targets = dict(targets) if targets else {}

    def expression(self, cache: 'SolveCache', input: CostInput) -> ca.MX:
        indexes, stiffnesses, targets = cache.memoize(
            self, 'coordinates', lambda: self._resolve(cache.mc))
        penalty = 0
        for index, stiffness, target in zip(indexes, stiffnesses, targets):
            penalty += stiffness * (input.coordinates[index] - target)**2
        return self.weight * penalty

    def _resolve(self, mc: ModelCache) -> tuple[list[int], list[float], list[float]]:
        """
        Resolve each penalized coordinate's index into the coordinates input, its
        stiffness, and its target, falling back to the coordinate's default value in
        the model when no target was given.

        Raises
        ------
        ValueError
            If a coordinate path is not an independent coordinate of the model.
            Dependent (e.g. constrained) coordinates are absent from
            `ModelCache.coordinate_q_map` and so cannot be penalized directly.
        """
        # Element j of the coordinates vector is the j-th entry of coordinate_q_map, so
        # a coordinate's position in that ordering is its index into the input.
        order = list(mc.coordinate_q_map)
        indexes: list[int] = []
        stiffnesses: list[float] = []
        targets: list[float] = []
        for path, stiffness in self.stiffnesses.items():
            if path not in mc.coordinate_q_map:
                known = 'is a dependent coordinate' if mc.model.hasComponent(path) \
                    else 'is not a coordinate in the model'
                raise ValueError(
                    f'Cannot apply a coordinate stiffness to {path}: it {known}. '
                    f'Expected one of the model\'s independent coordinates.')
            indexes.append(order.index(path))
            stiffnesses.append(float(stiffness))
            if path in self.targets:
                targets.append(float(self.targets[path]))
            else:
                coordinate = osim.Coordinate.safeDownCast(mc.model.getComponent(path))
                targets.append(float(coordinate.getDefaultValue()))
        return indexes, stiffnesses, targets
