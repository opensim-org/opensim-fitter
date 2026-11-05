import numpy as np
import casadi as ca
import opensim as osim

from ..model import ModelCache, StationCache
from ..scaling import AnthropometricMeasurement
from ..anthropometrics import build_ansur_distribution
from .cost import CallbackCostRep, Cost, CostInput, Function


class AnthropometricRegularizationCost(Cost):
    """
    A regularization penalty on body-scale factors, ``s``, that maximizes the
    log-likelihood that a set of anthropometric measurements, ``m(s)``, fall within a
    distribution fit to the ANSUR II dataset. Since it is a multivariate normal
    distribution, we use the Mahalanobis distance to define the cost:

        cost = weight * 0.5 (m(s) - μ)^T Σ^-1 (m(s) - μ)

    which equates to minimizing the negative log-likelihood of the probability density
    function.

    Users must define the set of measurements, ``m(s)``, via the parameter
    `measurements`, a list of `AnthropometricMeasurement`. Each
    `AnthropometricMeasurement` is named after a measurement in the ANSUR II dataset
    and defines the two `Station`s on the `Model` (and optionally an axis) from which
    to compute the simulated measurement. The `sex`
    parameter can be used to specify that the distribution should be fit to either
    male or female participants only; using the default value (`None`) fits across all
    participants from the ANSUR report.

    All quantities are in meters (ANSUR II millimeters are converted on load).

    Parameters
    ----------
    measurements: list[AnthropometricMeasurement]
        The measurements (each a station pair and optional axis) to compute from the
        model. Each measurement's `name` must match a measurement from the ANSUR II
        dataset.
    sex: str, optional
        Subject sex ('male' or 'female') selecting the ANSUR II subset. Defaults to None
        (the combined male-and-female dataset).
    weight: float, optional
        Non-negative scalar applied to the penalty. Default is 1.0.

    Raises
    ------
    ValueError
        If `weight` is negative or a measurement name is not present in the ANSUR II
        dataset. `AnthropometricRegularizationCostRep` additionally validates that the
        referenced components are stations.
    """
    required_inputs = frozenset({'body_scales'})

    def __init__(self, measurements: list[AnthropometricMeasurement],
                 sex: str = None, weight: float = 1.0):
        if weight < 0:
            raise ValueError(
                f'Expected weight to be non-negative, but got {weight}.')
        self.weight = weight
        self.measurements = measurements

        # Fit the ANSUR II distribution over the requested measurements, in meters.
        measurement_names = [m.name for m in self.measurements]
        distribution = build_ansur_distribution(measurement_names, sex)
        self.mean = np.asarray(distribution.get_mean(), dtype=float).reshape(-1)
        self.precision = np.linalg.inv(
            np.asarray(distribution.get_covariance(), dtype=float))

    def create_rep(self, mc: ModelCache) -> 'AnthropometricRegularizationCostRep':
        return AnthropometricRegularizationCostRep(self, mc)


class AnthropometricRegularizationCostRep(CallbackCostRep):
    """
    The rep of an `AnthropometricRegularizationCost`. It caches the model's default
    pose and a `StationCache` pair per measurement, then evaluates the Mahalanobis
    penalty and its gradient through OpenSim.

    Parameters
    ----------
    cost: AnthropometricRegularizationCost
        The cost this rep represents.
    mc: ModelCache
        The solver's `ModelCache`, whose registered body scale groups set this
        callback's input size.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the callback rather than using its
        analytic Jacobian. Default is ``False``.

    Raises
    ------
    ValueError
        If a measurement references a component that is not an `osim.Station`.
    """

    def __init__(self, cost: AnthropometricRegularizationCost, mc: ModelCache,
                 enable_fd: bool = False):
        self.cost = cost
        mc.model.realizePosition(mc.state)
        self.default_q = mc.state.getQ().to_numpy().copy()
        self.station_caches = []
        for measurement in cost.measurements:
            sc1 = StationCache.from_station(
                mc, mc.model.getComponent(measurement.station1_path))
            sc2 = StationCache.from_station(
                mc, mc.model.getComponent(measurement.station2_path))
            axis = measurement.axis.value if measurement.axis is not None else None
            self.station_caches.append((sc1, sc2, axis))
        Function.__init__(self, 'anthropometric_regularization_cost', mc,
                          enable_fd=enable_fd)

    def __call__(self, input: CostInput) -> ca.MX:
        return ca.Function.__call__(self, input.body_scales)

    def _get_num_inputs(self):
        return 1

    def _get_input_size(self, i):
        if i == 0:
            return 3 * len(self.mc.body_scale_groups)
        raise IndexError(f'Invalid input index {i} for {type(self).__name__}.')

    def _apply_body_scales(self, body_scales: np.ndarray) -> None:
        self.mc.set_scaled_mobilizer_frame_positions(self.state, body_scales)
        self.state.setQ(osim.Vector.createFromMat(self.default_q))
        self.mc.model.realizePosition(self.state)

    def _eval(self, arg):
        body_scales = np.atleast_1d(np.squeeze(arg[0].full())).astype(float)
        self._apply_body_scales(body_scales)

        measurements = np.empty(len(self.station_caches))
        for i, (sc1, sc2, axis) in enumerate(self.station_caches):
            pos1 = sc1.calc_position(self.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(self.state, body_scales).to_numpy()
            displacement = pos2 - pos1
            measurements[i] = (np.linalg.norm(displacement) if axis is None
                               else abs(displacement[axis]))

        residual = measurements - self.cost.mean
        return [float(self.cost.weight * 0.5 * residual
                      @ self.cost.precision @ residual)]

    def _jac_eval(self, arg):
        body_scales = np.atleast_1d(np.squeeze(arg[0].full())).astype(float)
        self._apply_body_scales(body_scales)

        num_scales = 3 * len(self.mc.body_scale_groups)
        m = np.empty(len(self.station_caches))
        jacobian = np.zeros((len(self.station_caches), num_scales))
        for i, (sc1, sc2, axis) in enumerate(self.station_caches):
            pos1 = sc1.calc_position(self.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(self.state, body_scales).to_numpy()
            displacement = pos2 - pos1

            jac1 = sc1.calc_position_jacobian_wrt_body_scales(self.state)
            jac2 = sc2.calc_position_jacobian_wrt_body_scales(self.state)
            displacement_jacobian = jac2 - jac1

            if axis is None:
                norm = np.linalg.norm(displacement)
                m[i] = norm
                if norm > 0.0:
                    jacobian[i, :] = (displacement / norm) @ displacement_jacobian
            else:
                value = displacement[axis]
                m[i] = abs(value)
                jacobian[i, :] = np.sign(value) * displacement_jacobian[axis, :]
        residual = m - self.cost.mean
        gradient = self.cost.weight * (self.cost.precision @ residual) @ jacobian
        return [gradient.reshape(1, num_scales)]
