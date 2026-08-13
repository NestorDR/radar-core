# tests/infrastructure/crud/test_ratio_crud.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
import datetime
from unittest.mock import patch

# --- Third Party Libraries ---
# pytest: testing framework
import pytest

# --- App modules ---
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud.ratio_crud import RatioCrud
# models: result of Object-Relational Mapping
from radar_core.models import Ratios


def _create_sample_ratio(symbol: str = 'BTC-USD',
                         strategy_id: int = 1,
                         inputs: str = '{"period": 10}',
                         timeframe: int = 2,
                         is_long_position: bool = True,
                         net_profit: float = 0.15,
                         expected_value: float = 0.05,
                         last_output_date: datetime.date | None = datetime.date(2025, 12, 31)) -> Ratios:
    """
    Helper function to build a sample Ratios instance for testing.
    """
    return Ratios(
        symbol=symbol,
        strategy_id=strategy_id,
        timeframe=timeframe,
        inputs=inputs,
        is_long_position=is_long_position,
        is_in_process=False,
        from_date=datetime.date(2025, 1, 1),
        to_date=datetime.date(2025, 12, 31),
        initial_price=100,
        final_price=150,
        net_change=0.5,
        signals=5,
        winnings=60,
        losses=10,
        net_profit=net_profit,
        expected_value=expected_value,
        win_probability=0.8,
        loss_probability=0.2,
        average_win=15,
        average_loss=5,
        min_percentage_change_to_win=0.01,
        max_percentage_change_to_win=0.10,
        total_sessions=250,
        winning_sessions=200,
        losing_sessions=50,
        percentage_exposure=0.8,
        first_input_date=datetime.date(2025, 1, 10),
        last_input_date=datetime.date(2025, 11, 1),
        last_output_date=last_output_date,
    )


def test_remove_unlisted_symbols_empty_deletes_all_rows(mock_connection_scope):
    """
    GIVEN an empty symbol list
    WHEN remove_unlisted_symbols is called
    THEN it executes the delete-all statement with no parameters.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.rowcount = 7

    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope',
            return_value=scope_
    ):
        result_ = RatioCrud.remove_unlisted_symbols([])

    assert result_ == 7
    cursor_.execute.assert_called_once()
    assert cursor_.execute.call_args.args[1] == ()
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_remove_unlisted_symbols_reuses_supplied_connection(mock_connection_scope):
    """
    GIVEN a supplied database connection
    WHEN remove_unlisted_symbols is called
    THEN the operation reuses that connection and passes the keep-list.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    cursor_.rowcount = 2

    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope',
            return_value=scope_
    ) as connection_scope_:
        result_ = RatioCrud.remove_unlisted_symbols(
            ['SPY', 'NDQ'],
            conn=connection_
        )

    assert result_ == 2
    connection_scope_.assert_called_once_with(connection_)
    assert cursor_.execute.call_args.args[1] == (['SPY', 'NDQ'],)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_flag_in_process_returns_rowcount_and_reuses_connection(mock_connection_scope):
    """
    GIVEN a supplied database connection
    WHEN flag_in_process is called
    THEN it executes the flag operation and returns the affected-row count.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    cursor_.rowcount = 5

    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope',
            return_value=scope_
    ) as connection_scope_:
        result_ = RatioCrud.flag_in_process(
            'SPY',
            3,
            2,
            conn=connection_
        )

    assert result_ == 5
    connection_scope_.assert_called_once_with(connection_)
    assert cursor_.execute.call_args.args[1] == ('SPY', 3, 2)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_delete_flagged_in_process_returns_rowcount_and_reuses_connection(mock_connection_scope):
    """
    GIVEN a supplied database connection
    WHEN delete_flagged_in_process is called
    THEN it executes the cleanup operation and returns the affected-row count.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    cursor_.rowcount = 4

    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope',
            return_value=scope_
    ) as connection_scope_:
        result_ = RatioCrud.delete_flagged_in_process(
            'SPY',
            3,
            2,
            conn=connection_
        )

    assert result_ == 4
    connection_scope_.assert_called_once_with(connection_)
    assert cursor_.execute.call_args.args[1] == ('SPY', 3, 2)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_upsert_many_empty_returns_zero():
    """
    GIVEN an empty list of ratios
    WHEN upsert_many is called
    THEN it returns 0 immediately without executing any database query.
    """
    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope'
    ) as connection_scope_:
        result_ = RatioCrud.upsert_many([])

    assert result_ == 0
    connection_scope_.assert_not_called()


def test_upsert_many_propagates_connection_scope_error():
    """
    GIVEN a batch of ratio records and a connection-scope failure
    WHEN upsert_many is executed
    THEN the original connection error is propagated.
    """
    sample_ratio_ = _create_sample_ratio()
    failure_ = RuntimeError('psycopg3 database error')

    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope'
    ) as scope_:
        scope_.return_value.__enter__.side_effect = failure_

        with pytest.raises(RuntimeError, match='psycopg3 database error'):
            RatioCrud.upsert_many([sample_ratio_])

    scope_.return_value.__exit__.assert_not_called()


def test_upsert_many_reuses_supplied_connection(mock_connection_scope):
    """
    GIVEN a non-empty ratio list and a supplied database connection
    WHEN upsert_many is called
    THEN executemany uses the supplied connection and its row count is returned.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    cursor_.rowcount = 1
    sample_ratio_ = _create_sample_ratio()

    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope',
            return_value=scope_
    ) as connection_scope_:
        result_ = RatioCrud.upsert_many(
            [sample_ratio_],
            conn=connection_
        )

    assert result_ == 1
    connection_scope_.assert_called_once_with(connection_)
    cursor_.executemany.assert_called_once()
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_upsert_many_creates_operation_scoped_connection_when_none_is_supplied(mock_connection_scope):
    """
    GIVEN a non-empty ratio list without a supplied connection
    WHEN upsert_many is called
    THEN the operation creates an operation-scoped connection.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.rowcount = 1

    with patch(
            'radar_core.infrastructure.crud.ratio_crud.connection_scope',
            return_value=scope_
    ) as connection_scope_:
        result_ = RatioCrud.upsert_many([_create_sample_ratio()])

    assert result_ == 1
    connection_scope_.assert_called_once_with(None)
    cursor_.executemany.assert_called_once()
    scope_.__exit__.assert_called_once_with(None, None, None)