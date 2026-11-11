import numpy as np
import opensim as osim

from ..scaling import AnthropometricMeasurement
from ..anthropometrics import build_ansur_distribution
from .cost import CallbackCost, CostCache


class AnthropometricRegularizationCost(CallbackCost):
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
        dataset. The referenced components are additionally validated as stations when
        the cost is first evaluated in a solve.
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

    def create_context(self, cache: CostCache) -> list[tuple]:
        """
        Resolve a `StationCache` pair and an optional axis index per measurement.

        Raises
        ------
        ValueError
            If a measurement references a component that is not an `osim.Station`.
        """
        context = []
        for measurement in self.measurements:
            sc1 = cache.station_cache(measurement.station1_path)
            sc2 = cache.station_cache(measurement.station2_path)
            axis = measurement.axis.value if measurement.axis is not None else None
            context.append((sc1, sc2, axis))
        return context

    def _apply_body_scales(self, cache: CostCache, body_scales: np.ndarray) -> None:
        cache.mc.set_scaled_mobilizer_frame_positions(cache.state, body_scales)
        cache.state.setQ(osim.Vector.createFromMat(cache.default_q))
        cache.mc.model.realizePosition(cache.state)

    def calc_value(self, cache, context, values):
        body_scales = values['body_scales']
        self._apply_body_scales(cache, body_scales)

        measurements = np.empty(len(context))
        for i, (sc1, sc2, axis) in enumerate(context):
            pos1 = sc1.calc_position(cache.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(cache.state, body_scales).to_numpy()
            displacement = pos2 - pos1
            measurements[i] = (np.linalg.norm(displacement) if axis is None
                               else abs(displacement[axis]))

        residual = measurements - self.mean
        return float(self.weight * 0.5 * residual @ self.precision @ residual)

    def calc_jacobian(self, cache, context, values):
        body_scales = values['body_scales']
        self._apply_body_scales(cache, body_scales)

        num_scales = 3 * len(cache.mc.body_scale_groups)
        m = np.empty(len(context))
        jacobian = np.zeros((len(context), num_scales))
        for i, (sc1, sc2, axis) in enumerate(context):
            pos1 = sc1.calc_position(cache.state, body_scales).to_numpy()
            pos2 = sc2.calc_position(cache.state, body_scales).to_numpy()
            displacement = pos2 - pos1

            jac1 = sc1.calc_position_jacobian_wrt_body_scales(cache.state)
            jac2 = sc2.calc_position_jacobian_wrt_body_scales(cache.state)
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
        residual = m - self.mean
        gradient = self.weight * (self.precision @ residual) @ jacobian
        return [gradient.reshape(1, num_scales)]
