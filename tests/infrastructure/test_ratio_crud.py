# tests/infrastructure/test_ratio_crud.py

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


def test_upsert_many_empty_returns_zero():
    """
    GIVEN an empty list of ratios
    WHEN upsert_many is called
    THEN it returns 0 immediately without executing any database query.
    """
    crud_ = RatioCrud()
    res_ = crud_.upsert_many([])
    assert res_ == 0



def test_upsert_many_rollback_on_error():
    """
    GIVEN a batch of ratio records and a database execution failure
    WHEN upsert_many is executed
    THEN it catches the exception and raises the error via connection_scope context manager.
    """
    crud_ = RatioCrud()
    sample_ratio_ = _create_sample_ratio()

    exception_message_ = 'psycopg3 database error'

    with patch('radar_core.infrastructure.crud.ratio_crud.connection_scope') as mock_conn_scope_:
        mock_conn_scope_.return_value.__enter__.side_effect = Exception(exception_message_)

        with pytest.raises(Exception, match=exception_message_):
            crud_.upsert_many([sample_ratio_])
