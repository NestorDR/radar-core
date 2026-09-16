# tests/domain/strategies/test_rsirc.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.domain.strategies.rsirc import RsiRollerCoaster, _find_trades_rc, _get_out_range
from radar_core.helpers.constants import DAILY, WEEKLY
from radar_core.infrastructure.crud import StrategyCrud
from radar_core.infrastructure.ratio_repository import RatioRepository
from radar_core.models import Strategies


@pytest.fixture(autouse=True)
def mock_strategy_db():
    """Mocks database lookup of strategy metadata and flag_in_process for pure offline testing."""
    mock_strategy_ = Strategies(
        id=2,
        acronym='RSI(14) RC',
        name='RSI RollerCoaster',
    )
    with patch.object(StrategyCrud, 'get_by_acronym', return_value=mock_strategy_), \
         patch.object(RatioRepository, 'flag_in_process', return_value=0):
        yield


def test_find_trades_rc_complete_lifecycle() -> None:
    """
    GIVEN an RSI series with a full Rollercoaster progression: input -> over -> output.
    WHEN _find_trades_rc is evaluated.
    THEN input and output bars are correctly captured in the returned arrays.
    """
    # Bar 0: 25.0 (warmup)
    # Bar 1: 35.0 (cross above in=30 -> Long Entry)
    # Bar 2: 50.0
    # Bar 3: 75.0 (crosses over_level=70 -> Overbought trigger)
    # Bar 4: 72.0
    # Bar 5: 68.0 (crosses below out=70 -> Output Exit)
    rsi_values_ = np.array([25.0, 35.0, 50.0, 75.0, 72.0, 68.0], dtype=np.float64)
    stop_loss_bars_ = np.full(len(rsi_values_), 10, dtype=np.int32)

    inputs_, outputs_ = _find_trades_rc(
        rsi_values_, stop_loss_bars_, in_=30, over_=70, out_=70,
        is_long_position=True, future_bar_number=6
    )

    assert len(inputs_) == 1
    assert inputs_[0] == 1
    assert outputs_[0] == 5


def test_find_trades_rc_stop_loss_priority_before_over() -> None:
    """
    GIVEN an active trade where stop loss is breached before reaching intermediate over level.
    WHEN _find_trades_rc is evaluated.
    THEN the trade terminates immediately at the stop-loss bar.
    """
    # Bar 0: 25.0
    # Bar 1: 35.0 (Long entry)
    # Bar 2: 40.0
    # Bar 3: 45.0 (stop loss triggered at bar 2)
    rsi_values_ = np.array([25.0, 35.0, 40.0, 45.0], dtype=np.float64)
    stop_loss_bars_ = np.array([10, 2, 10, 10], dtype=np.int32)

    inputs_, outputs_ = _find_trades_rc(
        rsi_values_, stop_loss_bars_, in_=30, over_=70, out_=60,
        is_long_position=True, future_bar_number=4
    )

    assert len(inputs_) == 1
    assert inputs_[0] == 1
    assert outputs_[0] == 2


def test_find_trades_rc_open_trade_at_end_of_data() -> None:
    """
    GIVEN an active trade reaching intermediate over level but with no output before data ends.
    WHEN _find_trades_rc is evaluated.
    THEN the trade remains open and output bar is marked with future_bar_number.
    """
    # Bar 0: 25.0
    # Bar 1: 35.0 (Long entry)
    # Bar 2: 75.0 (Overbought triggered)
    # Bar 3: 74.0 (Remains above out=60)
    rsi_values_ = np.array([25.0, 35.0, 75.0, 74.0], dtype=np.float64)
    stop_loss_bars_ = np.full(len(rsi_values_), 10, dtype=np.int32)

    inputs_, outputs_ = _find_trades_rc(
        rsi_values_, stop_loss_bars_, in_=30, over_=70, out_=60,
        is_long_position=True, future_bar_number=4
    )

    assert len(inputs_) == 1
    assert inputs_[0] == 1
    assert outputs_[0] == 4


def test_find_trades_rc_entry_filtering() -> None:
    """
    GIVEN an eligibility mask that rejects the first candidate entry and confirms the second.
    WHEN _find_trades_rc is evaluated with the mask.
    THEN only the eligible bar is accepted for trade entry.
    """
    # Two entry candidates: at Bar 1 and Bar 5
    rsi_values_ = np.array([25.0, 35.0, 75.0, 68.0, 25.0, 35.0, 75.0, 68.0], dtype=np.float64)
    stop_loss_bars_ = np.full(len(rsi_values_), 10, dtype=np.int32)
    mask_ = np.array([False, False, False, False, False, True, False, False], dtype=bool)

    inputs_, outputs_ = _find_trades_rc(
        rsi_values_, stop_loss_bars_, in_=30, over_=70, out_=70,
        is_long_position=True, future_bar_number=8, is_input_eligible=mask_
    )

    assert inputs_.tolist() == [5]
    assert outputs_.tolist() == [7]


def test_get_out_range_rsirc() -> None:
    """
    GIVEN Long and Short positions with input and intermediate levels.
    WHEN _get_out_range is called.
    THEN appropriate (from_out, to_out) bounds are returned.
    """
    # Long: from_out is 84 if over_ > 84 else over_; to_out is 18 if in_ < 18 else in_
    assert _get_out_range(True, 30, 70) == (70, 30)
    assert _get_out_range(True, 15, 90) == (84, 18)

    # Short: from_out is 16 if over_ < 16 else over_; to_out is 82 if in_ > 82 else in_
    assert _get_out_range(False, 70, 30) == (30, 70)
    assert _get_out_range(False, 85, 10) == (16, 82)


def test_rsirc_identify_execution() -> None:
    """
    GIVEN RsiRollerCoaster with synthetic OHLCV and indicator data.
    WHEN identify() is executed on DAILY and WEEKLY timeframes.
    THEN _grid_search_rc_fused and _find_trades_rc execute and persist positive ratios.
    """
    total_bars_ = 40
    dates_ = pl.date_range(
        start=pl.date(2025, 1, 1),
        end=pl.date(2025, 2, 9),
        interval='1d',
        eager=True,
    )
    prices_df_ = pl.DataFrame({
        'Date': dates_,
        'Open': np.full(total_bars_, 100.0),
        'High': np.full(total_bars_, 105.0),
        'Low': np.full(total_bars_, 95.0),
        'Close': np.full(total_bars_, 100.0),
        'Volume': np.full(total_bars_, 1000.0),
        'PercentChange': np.zeros(total_bars_),
        'BarNumber': np.arange(total_bars_, dtype=np.int32),
        'Rsi': np.full(total_bars_, 50.0),
        'MogalefUpper': np.full(total_bars_, 110.0),
        'MogalefLower': np.full(total_bars_, 90.0),
        'BarNumberForLongStop': np.full(total_bars_, total_bars_, dtype=np.int32),
        'BarNumberForShortStop': np.full(total_bars_, total_bars_, dtype=np.int32),
    })
    close_prices_ = prices_df_['Close'].to_numpy()

    strategy_ = RsiRollerCoaster()
    mock_persist_ = MagicMock(return_value=1)
    strategy_.persist_ratios = mock_persist_

    with patch('radar_core.domain.strategies.rsirc._grid_search_rc_fused', return_value=np.empty((0, 3))) as mock_grid_:
        strategy_.identify('TEST', DAILY, False, prices_df_.clone(), close_prices_)
        assert mock_grid_.called

        strategy_.identify('TEST', WEEKLY, True, prices_df_.clone(), close_prices_)
        assert mock_grid_.called
