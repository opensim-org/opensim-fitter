"""
Unit tests for src/osimfit/data_sources.py.

Covers the override-detection properties (`provides_positions`,
`provides_orientations`), the corresponding `get_*_table` guards, the
static table helpers, the consistency-check helpers, and `__init__`
validation of `trim_to_range`. No real model file is needed — tests use
hand-built TimeSeriesTables and throwaway DataSource subclasses.
"""

import pytest
import opensim as osim
from osimfit.data_sources import DataSource, MarkerSource


# -- helpers ---------------------------------------------------------------

def _make_vec3_table(times, labels):
    """
    Build a TimeSeriesTableVec3 with rows of zero-valued Vec3 entries.
    """
    table = osim.TimeSeriesTableVec3()
    for t in times:
        row = osim.RowVectorVec3(len(labels), osim.Vec3(0))
        table.appendRow(t, row)
    table.setColumnLabels(labels)
    return table


def _make_quat_table(times, labels):
    """
    Build a TimeSeriesTableQuaternion with identity-quaternion rows.
    """
    table = osim.TimeSeriesTableQuaternion()
    for t in times:
        row = osim.RowVectorQuaternion(len(labels), osim.Quaternion())
        table.appendRow(t, row)
    table.setColumnLabels(labels)
    return table


class _PositionsOnly(DataSource):
    """
    Throwaway subclass that provides positions only.
    """
    def _create_positions_table(self):
        return _make_vec3_table([0.0, 0.1], ['a', 'b'])


class _OrientationsOnly(DataSource):
    """
    Throwaway subclass that provides orientations only.
    """
    def _create_orientations_table(self):
        return _make_quat_table([0.0, 0.1], ['a', 'b'])


class _Both(DataSource):
    """
    Throwaway subclass that provides both positions and orientations.
    """
    def _create_positions_table(self):
        return _make_vec3_table([0.0, 0.1], ['a', 'b'])

    def _create_orientations_table(self):
        return _make_quat_table([0.0, 0.1], ['a', 'b'])


class _Neither(DataSource):
    """
    Throwaway subclass that provides neither positions nor orientations.
    """
    pass


# -- provides_* properties -------------------------------------------------

def test_positions_only_provides_positions_not_orientations():
    src = _PositionsOnly()
    assert src.provides_positions is True
    assert src.provides_orientations is False


def test_orientations_only_provides_orientations_not_positions():
    src = _OrientationsOnly()
    assert src.provides_positions is False
    assert src.provides_orientations is True


def test_both_subclass_provides_both():
    src = _Both()
    assert src.provides_positions is True
    assert src.provides_orientations is True


def test_neither_subclass_provides_neither():
    src = _Neither()
    assert src.provides_positions is False
    assert src.provides_orientations is False


def test_marker_source_provides_positions_only():
    # MarkerSource.__init__ only stores the path — does not open the file —
    # so a nonexistent path is safe for property-level checks.
    src = MarkerSource('nonexistent.trc')
    assert src.provides_positions is True
    assert src.provides_orientations is False


# -- get_*_table raises when hook isn't overridden -------------------------

def test_get_positions_raises_with_subclass_name_when_not_provided():
    with pytest.raises(NotImplementedError, match='_Neither'):
        _Neither().get_positions_table()


def test_get_orientations_raises_with_subclass_name_when_not_provided():
    with pytest.raises(NotImplementedError, match='_Neither'):
        _Neither().get_orientations_table()


def test_get_orientations_raises_on_positions_only_subclass():
    with pytest.raises(NotImplementedError, match='_PositionsOnly'):
        _PositionsOnly().get_orientations_table()


def test_get_positions_raises_on_orientations_only_subclass():
    with pytest.raises(NotImplementedError, match='_OrientationsOnly'):
        _OrientationsOnly().get_positions_table()


# -- static table helpers --------------------------------------------------

def test_remove_columns_drops_listed_columns():
    table = _make_vec3_table([0.0, 0.1], ['a', 'b', 'c'])
    DataSource.remove_columns(table, ['b'])
    assert list(table.getColumnLabels()) == ['a', 'c']


def test_update_column_labels_renames_via_mapping():
    table = _make_vec3_table([0.0, 0.1], ['a', 'b', 'c'])
    DataSource.update_column_labels(table, {'a': 'x', 'c': 'z'})
    assert list(table.getColumnLabels()) == ['x', 'b', 'z']


def test_update_column_labels_no_op_for_empty_map():
    table = _make_vec3_table([0.0, 0.1], ['a', 'b'])
    DataSource.update_column_labels(table, {})
    assert list(table.getColumnLabels()) == ['a', 'b']


def test_trim_table_to_range_keeps_inclusive_window():
    table = _make_vec3_table([0.0, 0.5, 1.0, 1.5, 2.0], ['a'])
    DataSource.trim_table_to_range(table, (0.5, 1.5))
    assert list(table.getIndependentColumn()) == [0.5, 1.0, 1.5]


# -- assert_position_orientation_consistent --------------------------------

def test_assert_position_orientation_consistent_happy_path():
    positions = _make_vec3_table([0.0, 0.1], ['a', 'b'])
    orientations = _make_quat_table([0.0, 0.1], ['a', 'b'])
    # Returns None; absence of exception is the contract.
    DataSource.assert_position_orientation_consistent(positions, orientations)


def test_assert_position_orientation_consistent_raises_on_label_mismatch():
    positions = _make_vec3_table([0.0, 0.1], ['a', 'b'])
    orientations = _make_quat_table([0.0, 0.1], ['a', 'x'])
    with pytest.raises(ValueError, match='mismatched column'):
        DataSource.assert_position_orientation_consistent(
            positions, orientations)


def test_assert_position_orientation_consistent_raises_on_row_count_mismatch():
    positions = _make_vec3_table([0.0, 0.1, 0.2], ['a', 'b'])
    orientations = _make_quat_table([0.0, 0.1], ['a', 'b'])
    with pytest.raises(ValueError, match='mismatched row'):
        DataSource.assert_position_orientation_consistent(
            positions, orientations)


def test_assert_position_orientation_consistent_raises_on_time_mismatch():
    positions = _make_vec3_table([0.0, 0.1], ['a', 'b'])
    orientations = _make_quat_table([0.0, 0.2], ['a', 'b'])
    with pytest.raises(ValueError, match='mismatched time'):
        DataSource.assert_position_orientation_consistent(
            positions, orientations)


# -- assert_tables_share_times ---------------------------------------------

def test_assert_tables_share_times_returns_shared_time_vector():
    a = _make_vec3_table([0.0, 0.1, 0.2], ['x'])
    b = _make_vec3_table([0.0, 0.1, 0.2], ['y'])
    assert DataSource.assert_tables_share_times([a, b]) == [0.0, 0.1, 0.2]


def test_assert_tables_share_times_raises_on_mismatch():
    a = _make_vec3_table([0.0, 0.1], ['x'])
    b = _make_vec3_table([0.0, 0.2], ['y'])
    with pytest.raises(ValueError, match='differs from'):
        DataSource.assert_tables_share_times([a, b])


# -- assert_sources_share_times (positions + orientation fallback) ---------

def test_assert_sources_share_times_falls_back_to_orientations():
    # _PositionsOnly and _OrientationsOnly both expose times [0.0, 0.1] —
    # the orientation-only source must contribute via its orientations table.
    sources = [_PositionsOnly(), _OrientationsOnly()]
    assert DataSource.assert_sources_share_times(sources) == [0.0, 0.1]


def test_assert_sources_share_times_raises_when_source_has_neither():
    sources = [_PositionsOnly(), _Neither()]
    with pytest.raises(ValueError, match='_Neither'):
        DataSource.assert_sources_share_times(sources)


# -- trim_to_range validation in __init__ ----------------------------------

def test_init_raises_for_non_tuple_trim_to_range():
    with pytest.raises(ValueError, match='tuple'):
        _Neither(trim_to_range=[0.0, 1.0])


def test_init_raises_for_wrong_length_trim_to_range():
    with pytest.raises(ValueError, match='tuple'):
        _Neither(trim_to_range=(0.0, 0.5, 1.0))
