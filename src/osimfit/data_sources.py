import ezc3d
import numpy as np
import opensim as osim
from abc import ABC, abstractmethod

class DataSource(ABC):
    def __init__(self, labels_to_remove=None, label_map=None, trim_to_range=None):
        super().__init__()
        self.labels_to_remove = labels_to_remove
        self.label_map = label_map

        if trim_to_range is not None:
            assert(isinstance(trim_to_range, tuple) and len(trim_to_range) == 2,
                   "trim_to_range should be a tuple of (start_time, end_time)")
        self.trim_to_range = trim_to_range

    def get_positions_table(self) -> osim.TimeSeriesTableVec3:
        table = self._create_positions_table()
        if self.labels_to_remove:
            self.remove_columns(table, self.labels_to_remove)
        if self.label_map:
            self.update_column_labels(table, self.label_map)
        if self.trim_to_range:
            self.trim_table_to_range(table, self.trim_to_range)
        return table

    def get_orientations_table(self) -> osim.TimeSeriesTableQuaternion:
        table = self._create_orientations_table()
        if self.labels_to_remove:
            self.remove_columns(table, self.labels_to_remove)
        if self.label_map:
            self.update_column_labels(table, self.label_map)
        if self.trim_to_range:
            self.trim_table_to_range(table, self.trim_to_range)
        return table

    @abstractmethod
    def _create_positions_table(self) -> osim.TimeSeriesTableVec3:
        pass

    @abstractmethod
    def _create_orientations_table(self) -> osim.TimeSeriesTableQuaternion:
        pass

    @staticmethod
    def remove_columns(table, columns_to_remove):
        for col in columns_to_remove:
            table.removeColumn(col)
        return table

    @staticmethod
    def update_column_labels(table, label_map):
        if not label_map:
            return table

        labels = list(table.getColumnLabels())
        for ilabel in range(len(labels)):
            labels[ilabel] = label_map.get(labels[ilabel], labels[ilabel])
        table.setColumnLabels(labels)
        return table

    @staticmethod
    def trim_table_to_range(table, time_range):
        table.trim(time_range[0], time_range[1])
        return table


class MarkerSource(DataSource):
    def __init__(self, trc_filepath, labels_to_remove=None, label_map=None,
                 trim_to_range=None):
        super().__init__(labels_to_remove=labels_to_remove, label_map=label_map,
                         trim_to_range=trim_to_range)
        self.trc_filepath = trc_filepath

    def _create_positions_table(self) -> osim.TimeSeriesTableVec3:
        table = osim.TimeSeriesTableVec3(self.trc_filepath)
        return table

    def _create_orientations_table(self) -> osim.TimeSeriesTableQuaternion:
        raise NotImplementedError("Orientation data is not available in MarkerSource.")


class TheiaFrameSource(DataSource):
    """
    Data source for Theia markerless motion capture data. Theia outputs 4x4 homogeneous
    transformation matrices for various frames from its internal representation of the
    human skeletal model. This class extracts the position and orientation of each
    frame in Theia's output and converts them to OpenSim's coordinate system.

     Parameters
    ----------
    c3d_filepath : str
        Path to the C3D file containing Theia's output. The C3D file should contain
        4x4 homogeneous transformation matrices for each frame, stored in the
        'rotations' field. Each frame's transformation matrix should be labeled with a
        unique name in the C3D file.
    """
    def __init__(self, c3d_filepath, labels_to_remove=None, label_map=None,
                 trim_to_range=None):
        super().__init__(labels_to_remove=labels_to_remove, label_map=label_map,
                         trim_to_range=trim_to_range)

        self.filepath = c3d_filepath
        self.c3d = ezc3d.c3d(c3d_filepath)

        # This is a Y-Z space-fixed rotation needed to convert data collected from Theia
        # to OpenSim's ground reference frame convention (X forward, Y up, Z right).
        osim_rotation = osim.Rotation()
        osim_rotation.setRotationFromTwoAnglesTwoAxes(1, # space-fixed
                -0.5*np.pi, osim.CoordinateAxis(1), # Y rotation
                -0.5*np.pi, osim.CoordinateAxis(2)) # Z rotation
        self.osim_rotation = osim_rotation

        # This is an additional body-fixed rotation that effectively swaps the axes of
        # the rotations collected from Theia to match OpenSim's ground reference frame
        # convention (X forward, Y up, Z right), which is the convention used by the
        # matching Frame elements in the generic model.
        frame_rotation = osim.Rotation()
        frame_rotation.setRotationToBodyFixedXY(osim.Vec2(0.5*np.pi))
        self.frame_rotation = frame_rotation

        # Extract the column labels from the C3D file and remove the '_4X4' suffix.
        raw_labels = self.c3d.parameters['ROTATION']['LABELS']['value']
        self.labels = [label.replace('_4X4', '') for label in raw_labels]

        self.data = self.c3d.data['rotations']
        self.rate = self.c3d.parameters['ROTATION']['RATE']['value'][0]
        self.num_frames = self.data.shape[3]
        self.times = np.array([i/self.rate for i in range(self.num_frames)])


    def _create_positions_table(self) -> osim.TimeSeriesTableVec3:
        table = osim.TimeSeriesTableVec3()
        for iframe in range(self.num_frames):
            row = osim.RowVectorVec3(len(self.labels), osim.Vec3(0))
            for ilabel, label in enumerate(self.labels):
                position = self.data[:, 3, ilabel, iframe] / 1000.0  # mm to m
                row[ilabel] = osim.Vec3(position[0], position[1], position[2])
                row[ilabel] = self.osim_rotation.multiply(row[ilabel])

            table.appendRow(self.times[iframe], row)

        table.setColumnLabels(self.labels)
        table.addTableMetaDataString("Units", "m")
        table.addTableMetaDataString("DataRate", str(self.rate))

        return table


    def _create_orientations_table(self) -> osim.TimeSeriesTableQuaternion:
        table = osim.TimeSeriesTableQuaternion()
        for iframe in range(self.num_frames):
            row = osim.RowVectorQuaternion(len(self.labels), osim.Quaternion())
            for ilabel, label in enumerate(self.labels):
                rot = self.data[:3, :3, ilabel, iframe]
                data_rotation = osim.Rotation()
                data_rotation.set(0,0, rot[0,0])
                data_rotation.set(1,0, rot[1,0])
                data_rotation.set(2,0, rot[2,0])
                data_rotation.set(0,1, rot[0,1])
                data_rotation.set(1,1, rot[1,1])
                data_rotation.set(2,1, rot[2,1])
                data_rotation.set(0,2, rot[0,2])
                data_rotation.set(1,2, rot[1,2])
                data_rotation.set(2,2, rot[2,2])
                rotation = self.osim_rotation.multiply(data_rotation)
                rotation = rotation.multiply(self.frame_rotation)

                # Store as a quaternion.
                new_quat = rotation.convertRotationToQuaternion()
                upd_quat = row.updElt(0, ilabel)
                upd_quat.set(0, new_quat.get(0))
                upd_quat.set(1, new_quat.get(1))
                upd_quat.set(2, new_quat.get(2))
                upd_quat.set(3, new_quat.get(3))

            table.appendRow(self.times[iframe], row)

        table.setColumnLabels(self.labels)
        table.addTableMetaDataString("DataRate", str(self.rate))

        return table

