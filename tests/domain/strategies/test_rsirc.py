# tests/domain/strategies/test_rsirc.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
from radar_core.domain.strategies.rsirc import RsiRollerCoaster, _find_trades_rc, _get_out_range
from radar_core.helpers.constants import DAILY
from radar_core.models import Ratios




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




def test_rsirc_identify_filters_by_win_probability_threshold(sample_ohlcv_with_indicators_df: pl.DataFrame) -> None:
    """
    GIVEN RsiRollerCoaster strategy with candidates having win_probability below, at, and above threshold.
    WHEN identify() is executed with win_probability_threshold=0.5.
    THEN setups with win_probability < 0.5 are filtered out, while setups with win_probability >= 0.5 are persisted.
    """
    strategy_ = RsiRollerCoaster()
    prices_df_ = sample_ohlcv_with_indicators_df
    close_prices_ = prices_df_['Close'].to_numpy()

    mock_candidates_ = np.array([[20, 60, 70], [25, 60, 70], [30, 60, 70]], dtype=np.int32)
    mock_trades_ = (np.array([1], dtype=np.int32), np.array([2], dtype=np.int32))
    ratios_list_ = [
        Ratios(net_profit=10.0, expected_percentage=1.0, win_probability=0.49),
        Ratios(net_profit=10.0, expected_percentage=1.0, win_probability=0.50),
        Ratios(net_profit=10.0, expected_percentage=1.0, win_probability=0.51),
    ]

    mock_persist_ = MagicMock(return_value=1)
    strategy_.persist_ratios = mock_persist_

    with patch('radar_core.domain.strategies.rsirc._grid_search_rc_fused', return_value=mock_candidates_), \
         patch('radar_core.domain.strategies.rsirc._find_trades_rc', return_value=mock_trades_), \
         patch.object(strategy_, 'perfile_performance', side_effect=ratios_list_):
        strategy_.identify('TEST', DAILY, True, prices_df_, close_prices_, 0.5)

    assert mock_persist_.called
    persisted_ratios_ = mock_persist_.call_args[0][0]
    assert [r_.win_probability for r_ in persisted_ratios_] == [0.50, 0.51]

