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
from radar_core.infrastructure.crud import StrategyCrud
from radar_core.infrastructure.ratio_repository import RatioRepository
from radar_core.models import Ratios, Strategies


@pytest.fixture(autouse=True)
def mock_strategy_db():
    """Mocks database lookup of strategy metadata and flag_in_process for pure offline testing."""
    mock_strategy_ = Strategies(
        id=1,
        acronym='RSI(14) 2B',
        name='RSI Two Bands',
    )
    with patch.object(StrategyCrud, 'get_by_acronym', return_value=mock_strategy_), \
         patch.object(RatioRepository, 'flag_in_process', return_value=0):
        yield


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


def test_rsi2b_adaptive_dwell_timeframe_resolution() -> None:
    """
    GIVEN an RsiTwoBands strategy instance.
    WHEN identify() is executed on DAILY vs WEEKLY timeframes.
    THEN it adaptively resolves dwell_bars=2 on DAILY (and INTRADAY) and dwell_bars=1 on WEEKLY.
    """
    strategy_ = RsiTwoBands()

    # Synthetic DataFrame
    total_bars_ = 30
    dates_ = pl.date_range(
        start=pl.date(2025, 1, 1),
        end=pl.date(2025, 1, 30),
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


def test_find_trades_2b_dwell_bars_validation() -> None:
    """
    GIVEN unsupported dwell_bars parameter values (< 1 or > 2).
    WHEN _find_trades_2b or _grid_search_2b_fused is invoked.
    THEN ValueError is raised by the JIT kernels.
    """
    rsi_values_ = np.array([50.0, 60.0, 70.0], dtype=np.float64)
    stop_loss_bars_ = np.full(len(rsi_values_), 3, dtype=np.int32)
    close_prices_ = np.array([100.0, 101.0, 102.0], dtype=np.float64)

    # _find_trades_2b validation
    with pytest.raises(ValueError, match='dwell_bars must be 1 or 2'):
        _find_trades_2b(rsi_values_, stop_loss_bars_, in_=40, out_=70, is_long_position=True,
                        future_bar_number=3, dwell_bars=0)

    with pytest.raises(ValueError, match='dwell_bars must be 1 or 2'):
        _find_trades_2b(rsi_values_, stop_loss_bars_, in_=40, out_=70, is_long_position=True,
                        future_bar_number=3, dwell_bars=3)

    # _grid_search_2b_fused validation
    with pytest.raises(ValueError, match='dwell_bars must be 1 or 2'):
        _grid_search_2b_fused(rsi_values_, stop_loss_bars_, close_prices_, 20, 60, 5, True, 3, 0.5, dwell_bars=0)

    with pytest.raises(ValueError, match='dwell_bars must be 1 or 2'):
        _grid_search_2b_fused(rsi_values_, stop_loss_bars_, close_prices_, 20, 60, 5, True, 3, 0.5, dwell_bars=3)


def test_get_out_range_rsi2b() -> None:
    """
    GIVEN Long and Short positions with different input levels.
    WHEN _get_out_range is called.
    THEN appropriate (from_out, to_out) bounds are returned.
    """
    # Long: from_out is 84 if in_ < 84 else in_, to_out is in_
    assert _get_out_range(True, 30) == (84, 30)
    assert _get_out_range(True, 85) == (85, 85)

    # Short: from_out is 16 if in_ > 16 else in_, to_out is in_
    assert _get_out_range(False, 70) == (16, 70)
    assert _get_out_range(False, 10) == (10, 10)


def test_rsi2b_identify_filters_by_win_probability_threshold() -> None:
    """
    GIVEN RsiTwoBands strategy and candidate setups with win_probability below, at, and above threshold.
    WHEN identify() is executed with win_probability_threshold=0.5.
    THEN candidate setups with win_probability < 0.5 are filtered out, while setups with win_probability >= 0.5 are persisted.
    """
    strategy_ = RsiTwoBands()

    total_bars_ = 30
    dates_ = pl.date_range(
        start=pl.date(2025, 1, 1),
        end=pl.date(2025, 1, 30),
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
    assert len(persisted_ratios_) == 2
    assert all(r_.win_probability >= 0.5 for r_ in persisted_ratios_)
    assert [r_.win_probability for r_ in persisted_ratios_] == [0.50, 0.51]


