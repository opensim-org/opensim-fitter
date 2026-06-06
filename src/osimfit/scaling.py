import os
import numpy as np
import pandas as pd
import opensim as osim
from enum import Enum
from abc import ABC, abstractmethod
from .data_sources import DataSource
from .utilities import MultivariateNormal

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


class Scaler(ABC):
    """
    Abstract base class for model scalers. Subclasses implement a specific scaling
    strategy and return the scaled OpenSim model from `scale`.

    Parameters
    ----------
    model: osim.Model
        The OpenSim model to be scaled.
    """
    def __init__(self, model: osim.Model):
        super().__init__()
        self.model = model
        self.state = model.initSystem()

    @abstractmethod
    def scale(self) -> osim.Model:
        pass

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


class ScaleFactor:
    """
    Computes a scale factor as the ratio of a data-derived distance measurement to the
    corresponding model measurement, averaged over all frames in the position data.

    Parameters
    ----------
    data_label1 : str
        Column label for the first position in the data table.
    data_label2 : str
        Column label for the second position in the data table.
    measurement : Measurement
        Model measurement corresponding to the distance between the two data positions.
    axis : Axis
        Axis along which the scale factor is applied to the model segment.
    """
    def __init__(self, data_label1: str, data_label2: str, measurement: Measurement,
                 axis: Axis):
        self.data_label1 = data_label1
        self.data_label2 = data_label2
        self.measurement = measurement
        self.axis = axis

    def compute_scale_factor(self, model: osim.Model, state: osim.State,
                             positions: osim.TimeSeriesTableVec3) -> float:
        model_measurement = self.measurement.compute_measurement(model, state)

        # Magnitude of relative position between the data source elements. Average over
        # all the frames in the data source.
        data_column1 = positions.getDependentColumn(self.data_label1)
        data_column2 = positions.getDependentColumn(self.data_label2)

        data_frame_distances = np.zeros(data_column1.size())
        for i in range(data_column1.size()):
            data_value1 = data_column1[i]
            data_value2 = data_column2[i]
            data_frame_distances[i] = np.linalg.norm(data_value1.to_numpy() -
                                                     data_value2.to_numpy())

        data_measurement = np.mean(data_frame_distances)
        return data_measurement / model_measurement


class PositionDataScaler(Scaler):
    """
    Scales a model by comparing distances between position data (e.g., marker
    trajectories) to corresponding model measurements.

    Parameters
    ----------
    model : osim.Model
        OpenSim model to be scaled.
    data_source : DataSource
        Data source providing the position table used to compute data-side measurements.
    """
    def __init__(self, model: osim.Model, data_source: DataSource):
        super().__init__(model)
        self.positions = data_source.get_positions_table()
        self.symmetry_pairs: list[tuple[str, str]] = []
        self.segment_scale_factors: dict[str, list[ScaleFactor]] = {}

    def add_scale(self, segment, scale_factor: ScaleFactor):
        if segment not in self.segment_scale_factors:
            self.segment_scale_factors[segment] = []
        self.segment_scale_factors[segment].append(scale_factor)

    def add_symmetry_pair(self, segment1, segment2):
        self.symmetry_pairs.append((segment1, segment2))

    def scale(self):

        # Create scale factors
        # --------------------
        scaleset = osim.ScaleSet()
        for segment, scale_factors in self.segment_scale_factors.items():
            scale = self._create_scale(segment, scale_factors)
            scaleset.cloneAndAppend(scale)
            scaleset.get(scaleset.getSize()-1).setName(segment)

        # Apply symmetry to scale factors.
        # --------------------------------
        for segment1, segment2 in self.symmetry_pairs:
            scale1 = scaleset.get(segment1)
            scale2 = scaleset.get(segment2)
            factors1 = scale1.getScaleFactors()
            factors2 = scale2.getScaleFactors()
            avg_factors = osim.Vec3(
                0.5 * (factors1[0] + factors2[0]),
                0.5 * (factors1[1] + factors2[1]),
                0.5 * (factors1[2] + factors2[2]))
            scale1.setScaleFactors(avg_factors)
            scale2.setScaleFactors(avg_factors)

        # Scale the model
        # ---------------
        self.model.scale(self.state, scaleset, True)
        self.model.finalizeConnections()
        self.model.initSystem()

        return self.model

    def _create_scale(self, segment: str,
                      scale_factors: list[ScaleFactor]) -> osim.Scale:

        scale = osim.Scale()
        scale.setSegmentName(segment)

        axis_factors = {Axis.XAxis: [], Axis.YAxis: [], Axis.ZAxis: []}
        for scale_factor in scale_factors:
            axis_factors[scale_factor.axis].append(scale_factor.compute_scale_factor(
                self.model, self.state, self.positions))

        factors = osim.Vec3(1.0)
        for axis, axis_factors in axis_factors.items():
            if len(axis_factors) > 0:
                factors[axis.value] = np.mean(axis_factors)

        scale.setScaleFactors(factors)

        return scale


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


class AnthropometricScaler(Scaler):
    """
    Scales a model using the ANSUR II anthropometric dataset. Model measurements are
    compared against a multivariate normal distribution fit to the dataset, conditioned
    on a subset of measurements, and the resulting mean values are used to compute
    per-axis scale factors for each body segment.

    Parameters
    ----------
    model: osim.Model
        OpenSim model to be scaled.
    sex: str, optional
        Sex of the subject ('male' or 'female'). If not provided, the combined
        male-and-female dataset is used.
    """
    def __init__(self, model: osim.Model, sex: str = None):
        super().__init__(model)
        self.measurements: dict[str, AnthropometricMeasurement] = {}
        self.conditional_measurements: list[str] = []
        self.scale_factors: list[tuple[str, str, Axis]] = []

        sex_tag = 'BOTH'
        if sex and sex.lower() == 'male':
            sex_tag = 'MALE'
        elif sex and sex.lower() == 'female':
            sex_tag = 'FEMALE'
        self.anthropometrics_fpath = os.path.join(os.path.dirname(__file__),
                                                  'anthropometrics',
                                                  f'ANSUR_II_{sex_tag}_Public.csv')

    def add_measurement(self, ansur_label: str, measurement: AnthropometricMeasurement):
        self.measurements[ansur_label] = measurement

    def add_conditional_measurement(self, ansur_label: str):
        self.conditional_measurements.append(ansur_label)

    def add_scale_factor(self, segment: str, ansur_label: str, axis: Axis):
        self.scale_factors.append((segment, ansur_label, axis))

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
        model_values = dict()
        for ansur_label, measurement in self.measurements.items():
            model_value = measurement.compute_measurement(self.model, self.state)
            model_values[ansur_label] = model_value

        # Condition the multivariate normal distribution on the selected measurements.
        condition_values = dict()
        for ansur_label in self.conditional_measurements:
            condition_values[ansur_label] = model_values[ansur_label]
        mvn_conditioned = mvn.condition(condition_values)

        # Create a new dictionary to hold the mean values from the conditioned
        # distribution, which will be used to compute the scale factors.
        variables_conditioned = mvn_conditioned.get_variables()
        mean_conditioned = mvn_conditioned.get_mean()
        values_conditioned = dict()
        for var, mean in zip(variables_conditioned, mean_conditioned):
            values_conditioned[var] = mean

        # Create scale factors based on the anthropometric measurements.
        scale_dict = {}
        for segment, ansur_label, axis in self.scale_factors:
            measurement_value = model_values[ansur_label]
            conditioned_value = values_conditioned[ansur_label]
            scale_factor = conditioned_value / measurement_value
            if segment not in scale_dict:
                scale_dict[segment] = np.ones(3)
            scale_dict[segment][axis.value] = scale_factor

        # Combine the scale fators into a ScaleSet and apply to the model.
        scaleset = osim.ScaleSet()
        for segment, factors in scale_dict.items():
            scale = osim.Scale()
            scale.setSegmentName(segment)
            scale.setScaleFactors(osim.Vec3(factors[0], factors[1], factors[2]))
            scaleset.cloneAndAppend(scale)
            scaleset.get(scaleset.getSize()-1).setName(segment)

        # Scale the model
        self.model.scale(self.state, scaleset, True)
        self.model.finalizeConnections()
        self.model.initSystem()

        return self.model
