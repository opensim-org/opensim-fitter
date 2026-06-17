import os
import numpy as np
import pandas as pd
import opensim as osim
from enum import Enum
import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .utilities import MultivariateNormal
from .data_sources import DataSource

class Axis(Enum):
    """
    Cartesian axis identifier used to specify the direction of a scale factor or
    measurement.
    """
    XAxis = 0
    YAxis = 1
    ZAxis = 2


class Measurement(ABC):
    """
    Abstract base class defining the interface for computing a scalar measurement from
    an OpenSim model and its current state.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_measurement(self, model: osim.Model, state: osim.State) -> float:
        pass


class ScaleFactor(ABC):
    """
    Abstract base class for a single per-body, per-axis scaling rule. Subclasses
    encapsulate how a scale value is computed (e.g., from position data, from
    anthropometric statistics).

    Parameters
    ----------
    body_name: str
        Name of the body to which this scale factor is applied.
    axis: Axis
        Axis along which the scale factor is applied.
    """
    def __init__(self, body_name: str, axis: Axis):
        super().__init__()
        self.body_name = body_name
        self.axis = axis

    @abstractmethod
    def get_scale_factor(self, model: osim.Model, state: osim.State) -> float:
        pass


class Scaler(ABC):
    """
    A base class for model scalers. Subclasses can implement a specific scaling
    strategy and return the scaled OpenSim model from `scale`, or use the default
    behavior which generates an `osim.ScaleSet` via `populate_scaleset` and uses it to
    scale the model.

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to be scaled.
    """
    def __init__(self, model: osim.Model):
        super().__init__()
        self.model = model
        self.state = model.initSystem()
        self.scale_factors: list[ScaleFactor] = []
        self.scaleset: osim.ScaleSet = None

    def add_scale_factor(self, scale_factor: ScaleFactor) -> None:
        """
        Register a `ScaleFactor`.
        """
        self.scale_factors.append(scale_factor)

    def populate_scaleset(self):
        """
        Populate the internal `osim.ScaleSet` from the set of registered `ScaleFactor`s.
        """
        factors_by_body: dict[str, dict[Axis, list[float]]] = (
            collections.defaultdict(lambda: collections.defaultdict(list)))
        for sf in self.scale_factors:
            factor = sf.get_scale_factor(self.model, self.state)
            factors_by_body[sf.body_name][sf.axis].append(factor)

        self.scaleset = osim.ScaleSet()
        bodyset = self.model.getBodySet()
        for ib in range(bodyset.getSize()):
            body_name = bodyset.get(ib).getName()
            factors = osim.Vec3(1.0)
            for axis, values in factors_by_body.get(body_name, {}).items():
                factors[axis.value] = float(np.mean(values))
            scale = osim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(factors)
            self.scaleset.cloneAndAppend(scale)
            self.scaleset.get(self.scaleset.getSize() - 1).setName(body_name)

    def scale(self) -> osim.Model:
        """
        Default scaling pipeline: build the ScaleSet from registered SFs, apply
        it to `self.model` in place, and return the scaled model. Subclasses
        override this to interpose extra steps (e.g., symmetry averaging in
        `PositionBasedScaler` or MVN conditioning in `AnthropometricScaler`).
        """
        self.populate_scaleset()
        self.model.scale(self.state, self.scaleset, True)
        self.model.finalizeConnections()
        self.model.initSystem()

        return self.model


#################
# MANUAL SCALER #
#################

class ManualScaleFactor(ScaleFactor):
    """
    A manually-prescribed scale factor for a given body and axis.

    Parameters
    ----------
    body_name: str
        See :py:class:`ScaleFactor`.
    axis: Axis
        See :py:class:`ScaleFactor`.
    scale_factor: float
        The scale factor applied to the specified body and axis.
    """
    def __init__(self, body_name: str, axis: Axis, scale_factor: float):
        super().__init__(body_name, axis)
        self.scale_factor = scale_factor

    def get_scale_factor(self, model: osim.Model, state: osim.State) -> float:
        return self.scale_factor


########################
# POSITION DATA SCALER #
########################

class FrameMeasurement(Measurement):
    """
    Computes the Euclidean distance between two named frames expressed in the model's
    ground frame.

    Parameters
    ----------
    frame1_path : str
        Component path to the first frame in the model.
    frame2_path : str
        Component path to the second frame in the model.
    """
    def __init__(self, frame1_path, frame2_path):
        super().__init__()
        self.frame1_path = frame1_path
        self.frame2_path = frame2_path

    def compute_measurement(self, model: osim.Model, state: osim.State) -> float:
        # Retrieve the model frames.
        frame1 = osim.Frame.safeDownCast(model.getComponent(self.frame1_path))
        frame2 = osim.Frame.safeDownCast(model.getComponent(self.frame2_path))

        # Magnitude of relative position between the two model frames.
        frame1_position = frame1.getPositionInGround(state).to_numpy()
        frame2_position = frame2.getPositionInGround(state).to_numpy()

        return np.linalg.norm(frame1_position - frame2_position)


class MarkerMeasurement(Measurement):
    """
    Computes the Euclidean distance between two named markers expressed in the model's
    ground frame.

    Parameters
    ----------
    marker1_path : str
        Component path to the first marker in the model.
    marker2_path : str
        Component path to the second marker in the model.
    """
    def __init__(self, marker1_path, marker2_path):
        super().__init__()
        self.marker1_path = marker1_path
        self.marker2_path = marker2_path

    def compute_measurement(self, model: osim.Model, state: osim.State) -> float:
        # Retrieve the model markers.
        marker1 = osim.Marker.safeDownCast(model.getComponent(self.marker1_path))
        marker2 = osim.Marker.safeDownCast(model.getComponent(self.marker2_path))

        # Magnitude of relative position between the two model markers.
        marker1_position = marker1.getLocationInGround(state).to_numpy()
        marker2_position = marker2.getLocationInGround(state).to_numpy()

        return np.linalg.norm(marker1_position - marker2_position)


class MeasurementScaleFactor(ScaleFactor):
    """
    Computes a scale factor as the ratio of a data-derived distance measurement to the
    corresponding model measurement, averaged over all frames in the position data.

    Parameters
    ----------
    body_name: str
        See :py:class:`ScaleFactor`.
    axis: Axis
        See :py:class:`ScaleFactor`.
    measurement: Measurement
        Model measurement corresponding to the distance between the two data positions.
    position1_data: osim.VectorVec3
        The trajectory of position data associated with the first model measurement.
    position2_data: osim.VectorVec3
        The trajectory of position data associated with the second model measurement.
    """
    def __init__(self, body_name: str, axis: Axis, measurement: Measurement,
                 position1_data: osim.VectorVec3, position2_data: osim.VectorVec3):
        super().__init__(body_name, axis)
        self.measurement = measurement
        self.position1_data = position1_data
        self.position2_data = position2_data
        assert(self.position1_data.size() == self.position2_data.size())

    def get_scale_factor(self, model: osim.Model, state: osim.State) -> float:
        model_measurement = self.measurement.compute_measurement(model, state)

        # Magnitude of relative position between the data source elements, averaged
        # over all frames in the data source.
        num_times = self.position1_data.size()
        data_frame_distances = np.zeros(num_times)
        for i in range(num_times):
            data_value1 = self.position1_data[i]
            data_value2 = self.position2_data[i]
            data_frame_distances[i] = np.linalg.norm(data_value1.to_numpy() -
                                                     data_value2.to_numpy())

        data_measurement = np.mean(data_frame_distances)
        return data_measurement / model_measurement


class PositionBasedScaler(Scaler):
    """
    Scales a model by comparing distances between position data (e.g., marker
    trajectories) to corresponding model measurements.

    Parameters
    ----------
    model: osim.Model
        OpenSim model to be scaled.
    data_source: DataSource
        A data source containing position-based data (i.e., it produces a valid
        `osim.TimeSeriesTableVec3` from `get_positions_table`).
    """
    def __init__(self, model: osim.Model, data_source: DataSource):
        super().__init__(model)
        self.symmetry_pairs: list[tuple[str, str]] = []
        self.data_source = data_source
        self.positions = data_source.get_positions_table()

    def add_measurement_scale_factor(self, body_name: str, axis: Axis,
                                     measurement: Measurement, data_label1: str,
                                     data_label2: str) -> None:
        """
        Build a `MeasurementScaleFactor` given an `Axis`, `Measurement`, and labels to
        position data columns in the `DataSource` of this `PositionBasedScaler`.
        """
        position1_data = self.positions.getDependentColumn(data_label1)
        position2_data = self.positions.getDependentColumn(data_label2)
        self.add_scale_factor(MeasurementScaleFactor(
            body_name, axis, measurement, position1_data, position2_data))

    def add_symmetry_pair(self, body1_name: str, body2_name: str) -> None:
        self.symmetry_pairs.append((body1_name, body2_name))

    def apply_scaleset_symmetry(self):
        # Apply symmetry to scale factors.
        for body1_name, body2_name in self.symmetry_pairs:
            scale1 = self.scaleset.get(body1_name)
            scale2 = self.scaleset.get(body2_name)
            factors1 = scale1.getScaleFactors()
            factors2 = scale2.getScaleFactors()
            avg_factors = osim.Vec3(
                0.5 * (factors1[0] + factors2[0]),
                0.5 * (factors1[1] + factors2[1]),
                0.5 * (factors1[2] + factors2[2]))
            scale1.setScaleFactors(avg_factors)
            scale2.setScaleFactors(avg_factors)

    def scale(self):
        self.populate_scaleset()
        self.apply_scaleset_symmetry()
        self.model.scale(self.state, self.scaleset, True)
        self.model.finalizeConnections()
        self.model.initSystem()

        return self.model


#########################
# ANTHROPOMETRIC SCALER #
#########################

class AnthropometricMeasurement(ABC):
    """
    Computes the distance between two named stations in the model, in millimeters, for
    comparison against entries in the ANSUR II anthropometric dataset.

    Parameters
    ----------
    station1_path: str
        Component path to the first station in the model.
    station2_path: str
        Component path to the second station in the model.
    axis: Axis, optional
        If provided, returns the signed distance along the specified axis rather than
        the Euclidean magnitude.
    """
    def __init__(self, station1_path: str, station2_path: str, axis: Axis = None):
        super().__init__()
        self.station1_path = station1_path
        self.station2_path = station2_path
        self.axis = axis

    def compute_measurement(self, model: osim.Model, state: osim.State) -> float:
        # Retrieve the model stations.
        station1 = osim.Station.safeDownCast(model.getComponent(self.station1_path))
        station2 = osim.Station.safeDownCast(model.getComponent(self.station2_path))

        # Magnitude of relative position between the two model frames.
        station1_position = station1.getLocationInGround(state).to_numpy()
        station2_position = station2.getLocationInGround(state).to_numpy()
        difference = station2_position - station1_position

        # If an axis is specified, return the absolute value of the difference along
        # that axis. Otherwise, return the magnitude of the difference vector. In both
        # cases, convert from meters to millimeters to match the ANSUR II dataset.
        if self.axis is not None:
            return 1000.0 * np.abs(difference[self.axis.value])
        else:
            return 1000.0 * np.linalg.norm(difference)


@dataclass
class AnthropometricContext:
    """
    Shared mutable bag holding the per-label values produced inside
    `AnthropometricScaler.scale()`. Each `AnthropometricScaleFactor` reads from this
    context at append-time to compute its ratio.
    """
    model_values: dict[str, float] = field(default_factory=dict)
    conditioned_values: dict[str, float] = field(default_factory=dict)


class AnthropometricScaleFactor(ScaleFactor):
    """
    Scale factor computed as the ratio of a conditioned anthropometric measurement
    (from a multivariate-normal model fit to the ANSUR II dataset) to the
    corresponding model-side measurement.

    Parameters
    ----------
    body_name: str
        See :py:class:`ScaleFactor`.
    axis: Axis
        See :py:class:`ScaleFactor`.
    ansur_label : str
        ANSUR II label identifying the measurement.
    measurement : AnthropometricMeasurement
        Measurement used to compute the model-side value.
    context : AnthropometricContext
        Shared context populated by the scaler before `append_scale_factor` runs.
    """
    def __init__(self, body_name: str, axis: Axis, ansur_label: str,
                 measurement: AnthropometricMeasurement,
                 context: AnthropometricContext):
        super().__init__(body_name, axis)
        self.ansur_label = ansur_label
        self.measurement = measurement
        self.context = context

    def get_scale_factor(self, model: osim.Model, state: osim.State) -> float:
        model_value = self.context.model_values[self.ansur_label]
        conditioned_value = self.context.conditioned_values[self.ansur_label]
        return conditioned_value / model_value


class AnthropometricScaler(Scaler):
    """
    Scales a model using the ANSUR II anthropometric dataset.

    The scaler fits a multivariate normal (MVN) distribution to a chosen subset of
    ANSUR II measurements, conditions that distribution on the corresponding
    model-side measurements (computed on the supplied model in its current state),
    and uses the mean of each conditioned marginal as the "target" subject value.
    Each registered scale factor then applies a per-body, per-axis scale equal to
    the ratio of the conditioned target to the model-side measurement.

    Typical workflow
    ----------------
    1. Construct the scaler with the model and, optionally, the subject's sex.
    2. Register every ANSUR II measurement that should participate in the joint
       MVN distribution via `add_measurement`. This includes both
       measurements directly tied to scale factors and "context-only"
       measurements that exist solely to improve the joint correlations.
    3. Mark a subset of the registered measurements as conditioning variables via
       `add_conditional_measurement`. These are the measurements you
       trust enough to fix at their model-side values; everything else is
       inferred from the conditioned MVN.
    4. Declare per-body, per-axis scale factors with
       `add_anthropometric_scale_factor`, referencing labels that have
       already been registered in step 2.
    5. Call `scale` to apply the result to the model.

    Step (2) must precede step (4): adding a scale factor for an unregistered
    label raises `ValueError`. Labels passed to step (3) must also have been registered
    in step (2).

    Example
    -------
    The snippet below scales the torso width and tibia geometry of a female
    subject, conditioning the MVN on stature and tibial height. `stature` and
    `tibialheight` are registered both because they condition the
    distribution and because (in the case of `tibialheight`) they drive a
    scale factor; ``biacromialbreadth`` is registered because its scale factor
    needs to read its conditioned mean.

        scaler = AnthropometricScaler(model, sex='female')

        # Register every measurement that participates in the joint MVN.
        scaler.add_measurement(
            'stature',
            AnthropometricMeasurement('/vertex', '/mtp5_r', Axis.YAxis))
        scaler.add_measurement(
            'tibialheight',
            AnthropometricMeasurement('/tibiale_r', '/mtp5_r', Axis.YAxis))
        scaler.add_measurement(
            'biacromialbreadth',
            AnthropometricMeasurement('/acromion_r', '/acromion_l'))

        # Trust the model-side values for these measurements; the others are
        # inferred from the conditioned MVN.
        scaler.add_conditional_measurement('stature')
        scaler.add_conditional_measurement('tibialheight')

        # Declare scale factors. Each ansur_label must already be registered.
        scaler.add_anthropometric_scale_factor(
            'torso', Axis.ZAxis, 'biacromialbreadth')
        scaler.add_anthropometric_scale_factor(
            'tibia_r', Axis.YAxis, 'tibialheight')

        scaled_model = scaler.scale()

    Parameters
    ----------
    model: osim.Model
        OpenSim model to be scaled.
    sex: str, optional
        Sex of the subject ('male' or 'female'). If not provided, the combined
        male-and-female dataset is used.

    Attributes
    ----------
    measurements: dict[str, AnthropometricMeasurement]
        Registered measurements keyed by ANSUR II label.
    conditional_measurements: list[str]
        Labels of measurements used to condition the MVN distribution.
    context: AnthropometricContext
        Shared bag of per-label model and conditioned values populated by
        `scale` and read by each `AnthropometricScaleFactor`.
    """
    def __init__(self, model: osim.Model, sex: str = None):
        super().__init__(model)
        self.measurements: dict[str, AnthropometricMeasurement] = {}
        self.conditional_measurements: list[str] = []
        self.context = AnthropometricContext()

        sex_tag = 'BOTH'
        if sex and sex.lower() == 'male':
            sex_tag = 'MALE'
        elif sex and sex.lower() == 'female':
            sex_tag = 'FEMALE'
        self.anthropometrics_fpath = os.path.join(os.path.dirname(__file__),
                                                  'anthropometrics',
                                                  f'ANSUR_II_{sex_tag}_Public.csv')

    def add_measurement(self, ansur_label: str,
                        measurement: AnthropometricMeasurement) -> None:
        """
        Register an anthropometric measurement. Use this for measurements that need
        to participate in the joint multivariate normal distribution but are not
        directly tied to a scale factor (e.g., conditional-only measurements, or
        measurements that just enrich the joint correlations).

        Parameters
        ----------
        ansur_label: str
            The label to an anthropometric measurement used in the ANSUR II dataset.
        measurement: AnthropometricMeasurement
            A model-based computation of anthropometric measurement from the ANSUR II
            dataset.
        """
        self.measurements[ansur_label] = measurement

    def add_conditional_measurement(self, ansur_label: str) -> None:
        """
        Register an anthropometric measurement that will be used to condition the
        multivariate normal distribution before scaling.

        Parameters
        ----------
        ansur_label: str
            The label to an anthropometric measurement used in the ANSUR II dataset.
        """
        self.conditional_measurements.append(ansur_label)

    def add_anthropometric_scale_factor(self, body_name: str, axis: Axis,
                                        ansur_label: str) -> None:
        """
        Build an `AnthropometricScaleFactor` with the scaler's shared context
        auto-filled, and register it. The associated `AnthropometricMeasurement`
        must already be registered via `add_measurement`.

        Parameters
        ----------
        body_name: str
            The name of a body in the model.
        axis: Axis
            The axis along which the scale factor is applied.
        ansur_label: str
            The label to an anthropometric measurement used in the ANSUR II dataset.
            Must match a label previously registered via `add_measurement`.

        Raises
        ------
        ValueError
            If `ansur_label` has not been registered via `add_measurement`.
        """
        if ansur_label not in self.measurements:
            raise ValueError(
                f"No anthropometric measurement registered for '{ansur_label}'. "
                f"Call add_measurement('{ansur_label}', ...) before adding a "
                f"scale factor for it.")
        self.add_scale_factor(AnthropometricScaleFactor(
            body_name, axis, ansur_label, self.measurements[ansur_label],
            self.context))

    def scale(self):
        # Load anthropometric measurements from the ANSUR II dataset.
        df = pd.read_csv(self.anthropometrics_fpath)
        ansur_labels = list(self.measurements.keys())
        for label in ansur_labels:
            if label not in df.columns:
                raise ValueError(f"The anthropometric measurement '{label}' was "
                                 f"provided, but it is not present in the ANSUR II "
                                 f"dataset.")

        # Construct a multivariate normal distribution over the anthropometric
        # measurements.
        df = df[self.measurements.keys()]
        mvn = MultivariateNormal.from_data(df.columns.tolist(), df.values)

        # Compute the values of the model measurements corresponding to the provided
        # anthropometric measurements.
        self.context.model_values.clear()
        for ansur_label, measurement in self.measurements.items():
            self.context.model_values[ansur_label] = measurement.compute_measurement(
                self.model, self.state)

        # Condition the multivariate normal distribution on the selected measurements.
        condition_values = {label: self.context.model_values[label]
                            for label in self.conditional_measurements}
        mvn_conditioned = mvn.condition(condition_values)

        # Extract the means of the conditioned distribution for use in scale factors.
        self.context.conditioned_values.clear()
        for var, mean in zip(mvn_conditioned.get_variables(),
                             mvn_conditioned.get_mean()):
            self.context.conditioned_values[var] = mean

        # Each ASF reads from self.context to overwrite its axis on self.scaleset.
        self.populate_scaleset()
        self.model.scale(self.state, self.scaleset, True)
        self.model.finalizeConnections()
        self.model.initSystem()

        return self.model
