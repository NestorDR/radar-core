# tests/infrastructure/crud/test_strategy_crud.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- App modules ---
from radar_core.helpers.constants import SMA
from radar_core.infrastructure.crud import StrategyCrud
from tests.conftest import STRATEGY_NAME_SMA


def test_strategy_crud_get_by_acronym_existing(
    mock_connection_scope: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """
    GIVEN an existing strategy acronym
    WHEN get_by_acronym is called
    THEN it returns the mapped strategy.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = (
        1,
        STRATEGY_NAME_SMA,
        SMA,
        'Default',
        'SMA Unit'
    )

    with patch(
        'radar_core.infrastructure.crud.strategy_crud.read_connection_scope',
        return_value=scope_
    ) as read_scope_:
        strategy_ = StrategyCrud.get_by_acronym(SMA)

    assert strategy_ is not None
    assert strategy_.id == 1
    assert strategy_.acronym == SMA
    assert strategy_.name == STRATEGY_NAME_SMA
    assert strategy_.pool == 'Default'
    assert strategy_.unit_label == 'SMA Unit'
    read_scope_.assert_called_once_with(None)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_strategy_crud_get_by_acronym_nonexistent(
    mock_connection_scope: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """
    GIVEN a non-existent strategy acronym
    WHEN get_by_acronym is called
    THEN it returns None.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = None

    with patch(
        'radar_core.infrastructure.crud.strategy_crud.read_connection_scope',
        return_value=scope_
    ) as read_scope_:
        strategy_ = StrategyCrud.get_by_acronym('UNKNOWN_ACRONYM')

    assert strategy_ is None
    read_scope_.assert_called_once_with(None)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_strategy_crud_get_by_acronym_reuses_supplied_connection(
    mock_connection_scope: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """
    GIVEN a supplied read connection
    WHEN get_by_acronym is called
    THEN the lookup reuses that connection.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = (
        1,
        STRATEGY_NAME_SMA,
        SMA,
        'Default',
        'SMA Unit'
    )

    with patch(
        'radar_core.infrastructure.crud.strategy_crud.read_connection_scope',
        return_value=scope_
    ) as read_scope_:
        strategy_ = StrategyCrud.get_by_acronym(
            SMA,
            conn=connection_
        )

    assert strategy_ is not None
    assert strategy_.id == 1
    read_scope_.assert_called_once_with(connection_)
    scope_.__exit__.assert_called_once_with(None, None, None)