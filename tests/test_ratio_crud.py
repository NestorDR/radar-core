# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
import datetime
# unittest.mock: provides tools for creating mock objects for use in testing.
from unittest.mock import MagicMock

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


def test_deduplicate_batch_empty():
    """
    GIVEN an empty list of ratio records
    WHEN _deduplicate_batch is called
    THEN it returns an empty list.
    """
    result_ = RatioCrud._deduplicate_batch([])
    assert result_ == []


def test_deduplicate_batch_distinct():
    """
    GIVEN a list of ratio records with distinct conflict keys
    WHEN _deduplicate_batch is called
    THEN all distinct records are preserved in the result.
    """
    records_ = [
        _create_sample_ratio(inputs='{"period": 10}', net_profit=0.15),
        _create_sample_ratio(inputs='{"period": 20}', net_profit=0.20),
    ]
    result_ = RatioCrud._deduplicate_batch(records_)
    assert len(result_) == 2


def test_deduplicate_batch_duplicate_keys():
    """
    GIVEN a list of ratio records containing duplicate conflict keys
    WHEN _deduplicate_batch is called
    THEN it keeps only the record with higher net profit and expected value.
    """
    records_ = [
        _create_sample_ratio(inputs='{"period": 10}', net_profit=0.10, expected_value=0.02),
        _create_sample_ratio(inputs='{"period": 10}', net_profit=0.25, expected_value=0.09),
    ]
    result_ = RatioCrud._deduplicate_batch(records_)
    assert len(result_) == 1
    assert result_[0]['net_profit'] == 0.25


def test_deduplicate_batch_with_ratios_objects_mixed_nulls():
    """
    GIVEN Ratios ORM objects with mixed null values (e.g. open and closed trades)
    WHEN _deduplicate_batch is called
    THEN it outputs dictionaries with homogeneous keys preserving None for nullable columns.
    """
    closed_trade_ = _create_sample_ratio(inputs='{"period": 10}', last_output_date=datetime.date(2025, 12, 31))
    open_trade_ = _create_sample_ratio(inputs='{"period": 20}', last_output_date=None)

    result_ = RatioCrud._deduplicate_batch([closed_trade_, open_trade_])

    assert len(result_) == 2
    # Ensure both output dictionaries contain exact same keys
    assert set(result_[0].keys()) == set(result_[1].keys())
    assert result_[0]['last_output_date'] == datetime.date(2025, 12, 31)
    assert result_[1]['last_output_date'] is None
    assert 'id' not in result_[0]


def test_upsert_many_empty_returns_zero():
    """
    GIVEN an empty list of ratios
    WHEN upsert_many is called
    THEN it returns 0 immediately without executing any database query.
    """
    crud_ = MagicMock(spec=RatioCrud)
    crud_.upsert_many = RatioCrud.upsert_many.__get__(crud_)
    res_ = crud_.upsert_many([])
    assert res_ == 0


def test_upsert_many_rollback_on_error():
    """
    GIVEN a batch of ratio records and a database execution failure
    WHEN upsert_many is executed
    THEN it catches the exception and rolls back the session transaction.
    """
    mock_session_ = MagicMock()
    exception_message_ = 'Database connection error'
    mock_session_.execute.side_effect = Exception(exception_message_)

    crud_ = MagicMock(spec=RatioCrud)
    crud_.session = mock_session_
    crud_._deduplicate_batch = RatioCrud._deduplicate_batch
    crud_.upsert_many = RatioCrud.upsert_many.__get__(crud_)

    sample_ratio_ = _create_sample_ratio()

    with pytest.raises(Exception, match=exception_message_):
        crud_.upsert_many([sample_ratio_])

    mock_session_.rollback.assert_called_once()
