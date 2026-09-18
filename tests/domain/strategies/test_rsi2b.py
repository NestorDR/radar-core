# tests/domain/strategies/test_rsi2b.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.domain.strategies.rsi2b import RsiTwoBands, _find_trades_2b, _get_out_range, _grid_search_2b_fused
from radar_core.helpers.constants import DAILY, INTRADAY, WEEKLY
from radar_core.models import Ratios




def test_find_trades_2b_dwell_flicker_rejection() -> None:
    """
    GIVEN an RSI series containing a 1-bar flicker vs a sustained 2-bar dwell at entry.
    WHEN _find_trades_2b is evaluated with dwell_bars=1 and dwell_bars=2.
    THEN dwell_bars=1 accepts the flicker, while dwell_bars=2 rejects the flicker and requires prior bar dwell.
    """
    # Bar 0: 60.0 (warmup)
    # Bar 1: 35.0 (flicker entry setup: prior=60.0, previous=35.0, current=45.0) -> in=40
    # Bar 2: 45.0
    # Bar 3: 75.0 (exit setup: previous=45.0, current=75.0) -> out=70
    # Bar 4: 30.0 (sustained dwell entry setup: prior=30.0, previous=30.0, current=45.0)
    # Bar 5: 30.0
    # Bar 6: 45.0
    # Bar 7: 75.0
    rsi_values_ = np.array([60.0, 35.0, 45.0, 75.0, 30.0, 30.0, 45.0, 75.0], dtype=np.float64)
    stop_loss_bars_ = np.full(len(rsi_values_), 8, dtype=np.int32)

    # 1. Baseline dwell_bars = 1 accepts Bar 2 (previous was 35.0 <= 40, current 45.0 > 40)
    inputs_1_, outputs_1_ = _find_trades_2b(rsi_values_, stop_loss_bars_, in_=40, out_=70, is_long_position=True,
                                            future_bar_number=8, dwell_bars=1)
    assert 2 in inputs_1_.tolist(), 'dwell_bars=1 should accept 1-bar entry cross'

    # 2. Dwell persistence dwell_bars = 2 rejects Bar 2 because prior (Bar 0) was 60.0 > 40
    # But accepts Bar 6 because Bar 4 (30.0 <= 40) and Bar 5 (30.0 <= 40) both dwelt below in_
    inputs_2_, outputs_2_ = _find_trades_2b(rsi_values_, stop_loss_bars_, in_=40, out_=70, is_long_position=True,
                                            future_bar_number=8, dwell_bars=2)
    assert 2 not in inputs_2_.tolist(), 'dwell_bars=2 must reject 1-bar flicker'
    assert 6 in inputs_2_.tolist(), 'dwell_bars=2 must accept 2-bar dwell entry'


def test_find_trades_2b_boundary_safety() -> None:
    """
    GIVEN a short RSI series starting immediately at bar 0 or 1.
    WHEN _find_trades_2b evaluates with dwell_bars=2.
    THEN start_bar_ = max(1, dwell_bars) prevents accessing negative indices without error.
    """
    rsi_values_ = np.array([30.0, 45.0], dtype=np.float64)
    stop_loss_bars_ = np.full(len(rsi_values_), 2, dtype=np.int32)

    inputs_, outputs_ = _find_trades_2b(rsi_values_, stop_loss_bars_, in_=40, out_=70, is_long_position=True,
                                        future_bar_number=2, dwell_bars=2)
    assert len(inputs_) == 0
    assert len(outputs_) == 0


def test_rsi2b_adaptive_dwell_timeframe_resolution(sample_ohlcv_with_indicators_df: pl.DataFrame) -> None:
    """
    GIVEN an RsiTwoBands strategy instance.
    WHEN identify() is executed on DAILY vs WEEKLY timeframes.
    THEN it adaptively resolves dwell_bars=2 on DAILY (and INTRADAY) and dwell_bars=1 on WEEKLY.
    """
    strategy_ = RsiTwoBands()
    prices_df_ = sample_ohlcv_with_indicators_df
    close_prices_ = prices_df_['Close'].to_numpy()

    with patch('radar_core.domain.strategies.rsi2b._grid_search_2b_fused', return_value=np.empty((0, 2))) as mock_grid_:
        strategy_.persist_ratios = MagicMock()

        # DAILY evaluation -> dwell_bars=2 (9th positional argument to _grid_search_2b_fused, index 9)
        strategy_.identify('TEST', DAILY, False, prices_df_.clone(), close_prices_, 0.5)
        assert mock_grid_.call_args[0][8] == 0.5, 'RsiTwoBands must pass win_probability_threshold'
        assert mock_grid_.call_args[0][9] == 2, 'RsiTwoBands on DAILY must resolve dwell_bars=2'

        # INTRADAY evaluation -> dwell_bars=2 (timeframe <= DAILY)
        strategy_.identify('TEST', INTRADAY, False, prices_df_.clone(), close_prices_, 0.5)
        assert mock_grid_.call_args[0][8] == 0.5, 'RsiTwoBands must pass win_probability_threshold'
        assert mock_grid_.call_args[0][9] == 2, 'RsiTwoBands on INTRADAY must resolve dwell_bars=2'

        # WEEKLY evaluation -> dwell_bars=1
        strategy_.identify('TEST', WEEKLY, False, prices_df_.clone(), close_prices_, 0.5)
        assert mock_grid_.call_args[0][8] == 0.5, 'RsiTwoBands must pass win_probability_threshold'
        assert mock_grid_.call_args[0][9] == 1, 'RsiTwoBands on WEEKLY must resolve dwell_bars=1'


@pytest.mark.parametrize('invalid_dwell', [0, 3], ids=['zero', 'three'])
def test_find_trades_2b_dwell_bars_validation(invalid_dwell: int) -> None:
    """
    GIVEN unsupported dwell_bars parameter values (< 1 or > 2).
    WHEN _find_trades_2b or _grid_search_2b_fused is invoked.
    THEN ValueError is raised by the JIT kernels.
    """
    rsi_values_ = np.array([50.0, 60.0, 70.0], dtype=np.float64)
    stop_loss_bars_ = np.full(len(rsi_values_), 3, dtype=np.int32)
    close_prices_ = np.array([100.0, 101.0, 102.0], dtype=np.float64)

    with pytest.raises(ValueError, match=r'dwell_bars'):
        _find_trades_2b(
            rsi_values_, stop_loss_bars_, in_=40, out_=70, is_long_position=True,
            future_bar_number=3, dwell_bars=invalid_dwell
        )

    with pytest.raises(ValueError, match=r'dwell_bars'):
        _grid_search_2b_fused(
            rsi_values_, stop_loss_bars_, close_prices_, 20, 60, 5, True, 3, 0.5, dwell_bars=invalid_dwell
        )


@pytest.mark.parametrize(
    ('is_long', 'in_val', 'expected_range'),
    [
        (True, 30, (84, 30)),
        (True, 85, (85, 85)),
        (False, 70, (16, 70)),
        (False, 10, (10, 10)),
    ],
    ids=['long_standard', 'long_boundary', 'short_standard', 'short_boundary'],
)
def test_get_out_range_rsi2b(is_long: bool, in_val: int, expected_range: tuple[int, int]) -> None:
    """
    GIVEN Long and Short positions with different input levels.
    WHEN _get_out_range is called.
    THEN appropriate (from_out, to_out) bounds are returned.
    """
    assert _get_out_range(is_long, in_val) == expected_range


def test_rsi2b_identify_filters_by_win_probability_threshold(sample_ohlcv_with_indicators_df: pl.DataFrame) -> None:
    """
    GIVEN RsiTwoBands strategy and candidate setups with win_probability below, at, and above threshold.
    WHEN identify() is executed with win_probability_threshold=0.5.
    THEN candidate setups with win_probability < 0.5 are filtered out, while setups with win_probability >= 0.5 are persisted.
    """
    strategy_ = RsiTwoBands()
    prices_df_ = sample_ohlcv_with_indicators_df
    close_prices_ = prices_df_['Close'].to_numpy()

    mock_candidates_ = np.array([[20, 70], [25, 70], [30, 70]], dtype=np.int32)
    mock_trades_ = (np.array([1], dtype=np.int32), np.array([2], dtype=np.int32))
    ratios_list_ = [
        Ratios(net_profit=10.0, expected_percentage=1.0, win_probability=0.49),
        Ratios(net_profit=10.0, expected_percentage=1.0, win_probability=0.50),
        Ratios(net_profit=10.0, expected_percentage=1.0, win_probability=0.51),
    ]

    mock_persist_ = MagicMock(return_value=1)
    strategy_.persist_ratios = mock_persist_

    with patch('radar_core.domain.strategies.rsi2b._grid_search_2b_fused', return_value=mock_candidates_), \
         patch('radar_core.domain.strategies.rsi2b._find_trades_2b', return_value=mock_trades_), \
         patch.object(strategy_, 'perfile_performance', side_effect=ratios_list_):
        strategy_.identify('TEST', DAILY, True, prices_df_, close_prices_, 0.5)

    assert mock_persist_.called
    persisted_ratios_ = mock_persist_.call_args[0][0]
    assert [r_.win_probability for r_ in persisted_ratios_] == [0.50, 0.51]


