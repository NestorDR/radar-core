# tests/test_strategy_crud.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- App modules ---
from radar_core.infrastructure.crud.strategy_crud import StrategyCrud


def test_strategy_crud_get_by_acronym_existing():
    """
    GIVEN an existing strategy acronym 'SMA'
    WHEN get_by_acronym is called with mocked database connection
    THEN it returns the corresponding Strategy instance.
    """
    crud_ = StrategyCrud()
    mock_row_ = (1, "Simple Moving Average", "SMA", "Default", "SMA Unit")

    with patch("radar_core.infrastructure.crud.strategy_crud.read_connection_scope") as mock_conn_func_:
        mock_cur_ = MagicMock()
        mock_cur_.fetchone.return_value = mock_row_
        mock_conn_func_.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur_

        strategy_ = crud_.get_by_acronym("SMA")
        assert strategy_ is not None
        assert strategy_.id == 1
        assert strategy_.acronym == "SMA"
        assert strategy_.name == "Simple Moving Average"


def test_strategy_crud_get_by_acronym_nonexistent():
    """
    GIVEN a non-existent strategy acronym 'UNKNOWN'
    WHEN get_by_acronym is called with mocked database connection
    THEN it returns None.
    """
    crud_ = StrategyCrud()

    with patch("radar_core.infrastructure.crud.strategy_crud.read_connection_scope") as mock_conn_func_:
        mock_cur_ = MagicMock()
        mock_cur_.fetchone.return_value = None
        mock_conn_func_.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur_

        strategy_ = crud_.get_by_acronym("UNKNOWN_ACRONYM")
        assert strategy_ is None
