import numpy as np
import casadi as ca
import opensim as osim
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from .model import ModelCache
from .anthropometrics import build_ansur_distribution


##################
# COST INTERFACE #
##################

@dataclass
class CostInput:
    """
    Bundles the optimization variables passed to a cost evaluation. The canonical
    ordering of cost inputs is defined via the `INPUT_ORDER` attribute. Unused inputs
    default to empty symbolic arrays so they remain valid callback arguments.

    Attributes
    ----------
    INPUT_ORDER: tuple[str, ...]
        The canonical order of the optimization-variable inputs.
    coordinates: ca.MX, optional
        Coordinate values (e.g., joint angles).
    body_scales: ca.MX, optional
        Flattened per-group XYZ body-scale factors.
    marker_offsets: ca.MX, optional
        Flattened per-group XYZ marker offsets.
    frame_offsets: ca.MX, optional
        Flattened per-group XYZ frame offsets.
    """
    INPUT_ORDER: ClassVar[tuple[str, ...]] = (
        'coordinates', 'body_scales', 'marker_offsets', 'frame_offsets')

    coordinates: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    body_scales: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    marker_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))
    frame_offsets: ca.MX = field(default_factory=lambda: ca.DM.zeros(0, 1))


class Cost(ABC):
    """
    A uniform interface for the cost terms of an optimization problem. A cost is
    evaluated by calling it with a `CostInput` that bundles the canonical optimization
    variables; a cost reads only the fields it depends on.

    Attributes
    ----------
    required_inputs: frozenset[str]
        The `CostInput` field names this cost reads and therefore requires the solver to
        provide (e.g., ``{'body_scales'}``). A solver validates that it provides every
        required input before accepting the cost; see `Solver.add_cost`.
    """
    required_inputs: frozenset[str] = frozenset()

    def initialize(self, model_cache: ModelCache) -> None:
        """
        Precompute any model-derived data the cost needs, given the solver's
        `ModelCache`. The solver calls this once for every registered cost before
        building the objective, so users need not pass the model (or body-scale
        parameters) when constructing a cost. The default is a no-op; costs that depend
        on the model (e.g., `AnthropometricRegularizationCost`) override it.
        """

    @abstractmethod
    def __call__(self, input: CostInput) -> ca.MX:
        pass


class SymbolicCost(Cost):
    """
    A `Cost` defined directly as a CasADi expression, requiring no OpenSim evaluation
    (e.g., a regularization penalty on the optimization variables). Unlike
    `CallbackCost`, it is differentiated symbolically by CasADi and incurs no callback
    overhead.
    """


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

    def __call__(self, input: CostInput) -> ca.MX:
        return self.weight * ca.sum((input.body_scales - self.target)**2)


class OffsetRegularizationCost(SymbolicCost):
    """
    A quadratic penalty on marker and frame XYZ offsets, penalizing offsets away from
    zero:

        cost = weight * sum_i offset_i^2

    Parameters
    ----------
    weight: float
        Non-negative scalar applied to the sum-of-squares.
    """
    required_inputs = frozenset({'marker_offsets', 'frame_offsets'})

    def __init__(self, weight: float):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight

    def __call__(self, input: CostInput) -> ca.MX:
        offsets = ca.vertcat(input.marker_offsets, input.frame_offsets)
        return self.weight * ca.sum(offsets**2)


class AnthropometricRegularizationCost(SymbolicCost):
    """
    A regularization penalty on body-scale factors: half the squared Mahalanobis
    distance of the model's anthropometric measurements from an ANSUR II Gaussian
    distribution (the negative log-likelihood up to an additive constant). Minimizing it
    keeps the body scales in a region of anthropometrically plausible measurements.

    The distribution is fitted in the constructor from the ANSUR II dataset (chosen by
    `sex`) over the requested measurement names, in meters. Each measurement (a distance
    between two model stations) is an affine function of the body scales,
    ``m(s) = D @ s + c``, because measurements are taken at a fixed pose and body
    scaling multiplies translations (both the mobilizer frames along the kinematic
    chain and each station's own base-frame location). These affine maps are built once
    in `initialize` (called by the solver), analytically, from OpenSim station Jacobians
    (`ModelCache.calc_station_position_jacobian_wrt_body_scales`), so the cost is smooth
    and CasADi differentiates it exactly:

        cost = weight * 0.5 (m(s) - mu)^T Sigma^-1 (m(s) - mu)

    All quantities are in meters (ANSUR II millimeters are converted on load).

    Parameters
    ----------
    measurements: dict[str, AnthropometricMeasurement]
        Maps each ANSUR II measurement name to the measurement (a station pair and
        optional axis) that computes it from the model. The names (keys) select the
        distribution's columns and must be present in the ANSUR II dataset.
    sex: str, optional
        Subject sex ('male' or 'female') selecting the ANSUR II subset. Defaults to None
        (the combined male-and-female dataset).
    weight: float, optional
        Non-negative scalar applied to the penalty. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` is negative or a measurement name is not present in the ANSUR II
        dataset. `initialize` additionally validates that the referenced components are
        stations.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, measurements: dict, sex: str = None, weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.measurements = measurements
        self.labels = list(measurements.keys())

        # Fit the ANSUR II distribution over the requested measurements, in meters.
        distribution = build_ansur_distribution(self.labels, sex)
        self.mean = np.asarray(distribution.get_mean(), dtype=float).reshape(-1)
        self.precision = np.linalg.inv(
            np.asarray(distribution.get_covariance(), dtype=float))

        # Populated by initialize(), which the solver calls before evaluation.
        self.displacement_maps: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.axes: list[int | None] | None = None

    def initialize(self, model_cache: ModelCache) -> None:
        """
        Build the affine measurement maps from the solver's `ModelCache`. The body-scale
        layout follows ``model_cache.body_scale_groups``, so the maps align with the
        solver's flat ``body_scales`` vector; measurements are evaluated at the model's
        current pose.
        """
        model_cache.model.realizePosition(model_cache.state)
        num_vars = 3 * len(model_cache.body_scale_groups)
        ones = np.ones(num_vars)
        self.displacement_maps = []
        self.axes = []
        for label in self.labels:
            measurement = self.measurements[label]
            jacobian1, position1 = self._station_jacobian_and_position(
                model_cache, measurement.station1_path)
            jacobian2, position2 = self._station_jacobian_and_position(
                model_cache, measurement.station2_path)
            D = jacobian2 - jacobian1
            c = (position2 - position1) - D @ ones
            self.displacement_maps.append((D, c))
            self.axes.append(
                measurement.axis.value if measurement.axis is not None else None)

    @staticmethod
    def _station_jacobian_and_position(model_cache: ModelCache, station_path: str):
        """
        Return the (3, 3 * num_groups) Jacobian of a station's ground position with
        respect to the flat body scales, and the station's baseline ground position (m).
        """
        station = osim.Station.safeDownCast(
            model_cache.model.getComponent(station_path))
        if station is None:
            raise ValueError(f'Component at path {station_path} is not a Station.')
        base_frame = osim.PhysicalFrame.safeDownCast(
            station.getParentFrame().findBaseFrame())
        mobod_index = int(base_frame.getMobilizedBodyIndex())
        base_station = station.findLocationInFrame(
            model_cache.state, base_frame).to_numpy()
        position = station.getLocationInGround(model_cache.state).to_numpy()
        jacobian = model_cache.calc_station_position_jacobian_wrt_body_scales(
            model_cache.state, mobod_index, base_frame, base_station)
        return jacobian, position

    def __call__(self, input: CostInput) -> ca.MX:
        if self.displacement_maps is None:
            raise RuntimeError(
                'AnthropometricRegularizationCost.initialize(...) must be called '
                'before evaluation; a solver does this automatically in solve().')
        s = input.body_scales
        measurements = []
        for (D, c), axis in zip(self.displacement_maps, self.axes):
            displacement = ca.DM(D) @ s + ca.DM(c)  # meters
            if axis is None:
                measurements.append(ca.norm_2(displacement))
            else:
                measurements.append(ca.fabs(displacement[axis]))
        residual = ca.vertcat(*measurements) - ca.DM(self.mean)
        return self.weight * 0.5 * (residual.T @ ca.DM(self.precision) @ residual)
