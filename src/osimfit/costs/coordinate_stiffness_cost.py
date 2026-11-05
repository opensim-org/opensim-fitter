import casadi as ca
import opensim as osim

from ..model import ModelCache
from .cost import Cost, CostInput, CostRep


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
        `CoordinateStiffnessCostRep` additionally validates the coordinate paths
        against the model.
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

    def create_rep(self, mc: ModelCache) -> 'CoordinateStiffnessCostRep':
        return CoordinateStiffnessCostRep(self, mc)


class CoordinateStiffnessCostRep(CostRep):
    """
    The rep of a `CoordinateStiffnessCost`.

    Parameters
    ----------
    cost: CoordinateStiffnessCost
        The cost this rep represents.
    mc: ModelCache
        The solver's `ModelCache`, supplying the coordinate ordering and the default
        coordinate values.

    Raises
    ------
    ValueError
        If a coordinate path is not an independent coordinate of the model. Dependent
        (e.g. constrained) coordinates are absent from `ModelCache.coordinate_q_map` and
        so cannot be penalized directly.
    """

    def __init__(self, cost: CoordinateStiffnessCost, mc: ModelCache):
        self.cost = cost
        # Element j of the coordinates vector is the j-th entry of coordinate_q_map, so
        # a coordinate's position in that ordering is its index into the input.
        order = list(mc.coordinate_q_map)
        self.indexes: list[int] = []
        self.stiffnesses: list[float] = []
        self.targets: list[float] = []
        for path, stiffness in cost.stiffnesses.items():
            if path not in mc.coordinate_q_map:
                known = 'is a dependent coordinate' if mc.model.hasComponent(path) \
                    else 'is not a coordinate in the model'
                raise ValueError(
                    f'Cannot apply a coordinate stiffness to {path}: it {known}. '
                    f'Expected one of the model\'s independent coordinates.')
            self.indexes.append(order.index(path))
            self.stiffnesses.append(float(stiffness))
            if path in cost.targets:
                self.targets.append(float(cost.targets[path]))
            else:
                coordinate = osim.Coordinate.safeDownCast(mc.model.getComponent(path))
                self.targets.append(float(coordinate.getDefaultValue()))

    def __call__(self, input: CostInput) -> ca.MX:
        penalty = 0
        for index, stiffness, target in zip(
                self.indexes, self.stiffnesses, self.targets):
            penalty += stiffness * (input.coordinates[index] - target)**2
        return self.cost.weight * penalty
