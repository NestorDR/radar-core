# tests/domain/strategies/test_base_strategy.py

# --- Python modules ---
from collections.abc import Callable
from datetime import date
import inspect
from unittest.mock import patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.domain.strategies import MovingAverage, RsiRollerCoaster, RsiTwoBands, StrategyABC
from radar_core.domain.strategies.base_strategy import AnalysisContext, RsiStrategyABC, _find_stop_loss_bars
from radar_core.helpers.constants import DAILY, SMA, WEEKLY
from radar_core.infrastructure.crud import StrategyCrud
from radar_core.models import Strategies
from radar_core.settings import get_settings


def _find_stop_loss_bars_legacy(
    close_prices: np.ndarray,
    long_stop_loss: np.ndarray,
    short_stop_loss: np.ndarray,
    future_bar_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy interpreted Python implementation for baseline parity comparison."""
    total_bars_ = len(close_prices)
    bar_for_long_stop_ = future_bar_number * np.ones(total_bars_, dtype=np.int32)
    bar_for_short_stop_ = future_bar_number * np.ones(total_bars_, dtype=np.int32)

    for i in range(total_bars_ - 1):
        long_condition_ = np.asarray(close_prices[i + 1:] < long_stop_loss[i]).nonzero()[0]
        if long_condition_.size > 0:
            bar_for_long_stop_[i] = i + 1 + long_condition_[0]

        short_condition_ = np.asarray(close_prices[i + 1:] > short_stop_loss[i]).nonzero()[0]
        if short_condition_.size > 0:
            bar_for_short_stop_[i] = i + 1 + short_condition_[0]

    return bar_for_long_stop_, bar_for_short_stop_


def test_set_mogalef_stop_loss_standard_bounds(
    ohlcv_factory: Callable[..., pl.DataFrame],
) -> None:
    """
    GIVEN a Polars DataFrame with OHLC columns.
    WHEN set_mogalef_stop_loss is executed.
    THEN LongStopLoss and ShortStopLoss match MogalefLower and MogalefUpper across all valid rows.
    """
    df_ = ohlcv_factory(30)
    result_df_ = RsiStrategyABC.set_mogalef_stop_loss(df_, period_reg=3, period_dev=7, multiplier=2.0)

    assert 'LongStopLoss' in result_df_.columns
    assert 'ShortStopLoss' in result_df_.columns

    valid_rows_ = result_df_.filter(pl.col('LongStopLoss').is_not_null() & pl.col('ShortStopLoss').is_not_null())
    assert (valid_rows_['LongStopLoss'] == valid_rows_['MogalefLower']).all()
    assert (valid_rows_['ShortStopLoss'] == valid_rows_['MogalefUpper']).all()


def test_set_mogalef_stop_loss_idempotency(
    ohlcv_factory: Callable[..., pl.DataFrame],
) -> None:
    """
    GIVEN a DataFrame that already contains LongStopLoss and ShortStopLoss columns.
    WHEN set_mogalef_stop_loss is called.
    THEN the DataFrame is returned unchanged.
    """
    df_ = ohlcv_factory(10).with_columns([
        pl.lit(95.0).alias('LongStopLoss'),
        pl.lit(105.0).alias('ShortStopLoss')
    ])
    result_df_ = RsiStrategyABC.set_mogalef_stop_loss(df_)

    assert (result_df_['LongStopLoss'] == 95.0).all()
    assert (result_df_['ShortStopLoss'] == 105.0).all()


def test_set_stop_loss_backwards_compatibility(
    ohlcv_factory: Callable[..., pl.DataFrame],
) -> None:
    """
    GIVEN a DataFrame with High, Low, and Close columns.
    WHEN the baseline set_stop_loss is executed.
    THEN LongStopLoss and ShortStopLoss columns are produced using ATR and rolling window.
    """
    df_ = ohlcv_factory(30)
    result_df_ = RsiStrategyABC.set_stop_loss(df_, bars_for_stop_loss=10)

    assert 'LongStopLoss' in result_df_.columns
    assert 'ShortStopLoss' in result_df_.columns
    assert 'Atr' in result_df_.columns


def test_identify_where_to_stop_loss_daily_and_weekly(
    ohlcv_factory: Callable[..., pl.DataFrame],
) -> None:
    """
    GIVEN a price DataFrame and close prices array.
    WHEN identify_where_to_stop_loss is called for DAILY and WEEKLY timeframes.
    THEN stop loss columns and trigger bar numbers are generated.
    """
    df_ = ohlcv_factory(35)
    close_prices_ = df_['Close'].to_numpy()

    # Test Daily
    result_daily_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, df_, close_prices_)
    assert {'LongStopLoss', 'ShortStopLoss', 'BarNumberForLongStop', 'BarNumberForShortStop'}.issubset(
        result_daily_.columns
    )
    assert result_daily_['BarNumberForLongStop'].dtype == pl.Int32
    assert result_daily_['BarNumberForShortStop'].dtype == pl.Int32

    # Test Weekly
    result_weekly_ = RsiStrategyABC.identify_where_to_stop_loss(WEEKLY, df_, close_prices_)
    assert {'LongStopLoss', 'ShortStopLoss', 'BarNumberForLongStop', 'BarNumberForShortStop'}.issubset(
        result_weekly_.columns
    )


def test_stop_loss_jit_parity_with_synthetic_data() -> None:
    """
    GIVEN synthetic price series and stop loss levels.
    WHEN both legacy nonzero slicing and _find_stop_loss_bars JIT are evaluated.
    THEN the returned long and short stop-loss bar indices match exactly.
    """
    close_prices_ = np.array([100.0, 102.0, 99.0, 95.0, 92.0, 105.0, 110.0, 108.0], dtype=np.float64)
    long_stops_ = np.array([98.0, 100.0, 96.0, 94.0, 90.0, 100.0, 105.0, 104.0], dtype=np.float64)
    short_stops_ = np.array([103.0, 104.0, 102.0, 100.0, 98.0, 108.0, 112.0, 110.0], dtype=np.float64)
    future_bar_ = 999

    long_legacy_, short_legacy_ = _find_stop_loss_bars_legacy(
        close_prices_, long_stops_, short_stops_, future_bar_
    )
    long_jit_, short_jit_ = _find_stop_loss_bars(
        close_prices_, long_stops_, short_stops_, future_bar_
    )

    np.testing.assert_array_equal(long_jit_, long_legacy_)
    np.testing.assert_array_equal(short_jit_, short_legacy_)


def test_stop_loss_jit_handles_no_breaches() -> None:
    """
    GIVEN prices that never breach the stop-loss levels.
    WHEN _find_stop_loss_bars is evaluated.
    THEN all elements remain set to future_bar_number.
    """
    close_prices_ = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    long_stops_ = np.array([90.0, 90.0, 90.0], dtype=np.float64)
    short_stops_ = np.array([110.0, 110.0, 110.0], dtype=np.float64)
    future_bar_ = 100

    long_jit_, short_jit_ = _find_stop_loss_bars(
        close_prices_, long_stops_, short_stops_, future_bar_
    )

    assert np.all(long_jit_ == future_bar_)
    assert np.all(short_jit_ == future_bar_)


def test_identify_where_to_stop_loss_parity_on_real_market_data(frozen_spy_prices: pl.DataFrame) -> None:
    """
    GIVEN real SPY daily market prices.
    WHEN RsiStrategyABC.identify_where_to_stop_loss is executed.
    THEN BarNumberForLongStop and BarNumberForShortStop columns are created and match legacy values.
    """
    df_ = frozen_spy_prices.clone()
    close_prices_ = df_['Close'].to_numpy()

    result_df_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, df_, close_prices_)

    assert 'BarNumberForLongStop' in result_df_.columns
    assert 'BarNumberForShortStop' in result_df_.columns
    assert result_df_.height == frozen_spy_prices.height

    # Check that calling again returns the cached DataFrame immediately
    cached_df_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, result_df_, close_prices_)
    assert cached_df_ is result_df_

    # Verify parity against legacy logic on the exact same Mogalef stop loss columns
    long_stops_ = result_df_['LongStopLoss'].to_numpy()
    short_stops_ = result_df_['ShortStopLoss'].to_numpy()
    future_bar_ = RsiStrategyABC.future_bar_number(result_df_)

    long_legacy_, short_legacy_ = _find_stop_loss_bars_legacy(
        close_prices_, long_stops_, short_stops_, future_bar_
    )

    np.testing.assert_array_equal(result_df_['BarNumberForLongStop'].to_numpy(), long_legacy_)
    np.testing.assert_array_equal(result_df_['BarNumberForShortStop'].to_numpy(), short_legacy_)


def test_rsi_strategy_abc_get_current_indicators() -> None:
    """
    GIVEN a Polars DataFrame with Rsi, MogalefUpper, and MogalefLower columns.
    WHEN RsiStrategyABC.get_current_indicators is called.
    THEN it returns a dictionary with 'rsi', 'up', and 'low' rounded to 1 decimal place.
    """
    df_ = pl.DataFrame({
        'Rsi': [45.123, 50.456, 55.449],
        'MogalefUpper': [120.555, 125.649, 128.531],
        'MogalefLower': [110.123, 115.449, 118.219],
    })
    indicators_ = RsiStrategyABC.get_current_indicators(df_)

    assert indicators_ == {'rsi': 55.4, 'up': 128.5, 'low': 118.2}


def test_rsi_strategy_abc_get_current_indicators_with_null_values() -> None:
    """
    GIVEN a Polars DataFrame where Rsi, MogalefUpper, or MogalefLower contains null values at the latest bar.
    WHEN RsiStrategyABC.get_current_indicators is called.
    THEN it gracefully returns None for null fields without raising a TypeError.
    """
    df_nulls_ = pl.DataFrame({
        'Rsi': [45.1, 50.4, None],
        'MogalefUpper': [120.5, 125.6, None],
        'MogalefLower': [110.1, 115.4, 118.2],
    })
    indicators_ = RsiStrategyABC.get_current_indicators(df_nulls_)

    assert indicators_ == {'rsi': None, 'up': None, 'low': 118.2}


def test_perfile_performance_with_current_indicators() -> None:
    """
    GIVEN an AnalysisContext, price data, and a current_indicators dictionary.
    WHEN StrategyABC.perfile_performance is executed.
    THEN the returned Ratios object has current_indicators populated with the serialized JSON string.
    """
    mock_strategy_ = Strategies(id=1, acronym=SMA, name='Moving Average')
    with patch.object(StrategyCrud, 'get_by_acronym', return_value=mock_strategy_):
        strategy_ = MovingAverage(SMA, 'Close', 'Sma')

    analysis_context_ = AnalysisContext(
        symbol='TEST',
        timeframe=DAILY,
        from_date=date(2025, 1, 1),
        to_date=date(2025, 1, 30),
        initial_price=100.0,
        final_price=110.0,
        last_bar_number=29,
        future_bar_number=30,
    )

    close_prices_ = np.array([100.0, 105.0, 110.0, 115.0], dtype=np.float64)
    percent_changes_ = np.array([0.0, 0.05, 0.0476, 0.0455], dtype=np.float64)
    prices_df_ = pl.DataFrame({
        'Date': [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)],
        'Close': close_prices_,
        'PercentChange': percent_changes_,
        'BarNumber': [0, 1, 2, 3],
    })

    inputs_ = {'period': 20}
    current_indicators_ = {'sma': 112.5}
    input_bars_ = np.array([0], dtype=np.int32)
    output_bars_ = np.array([3], dtype=np.int32)

    ratios_ = strategy_.perfile_performance(
        analysis_context_,
        inputs_,
        input_bars_,
        output_bars_,
        close_prices_,
        prices_df_,
        current_indicators_,
    )

    assert ratios_ is not None
    assert ratios_.current_indicators == '{"sma": 112.5}'


def test_perfile_performance_without_current_indicators_fallback() -> None:
    """
    GIVEN an AnalysisContext and trade arrays without current_indicators specified.
    WHEN StrategyABC.perfile_performance is executed.
    THEN the returned Ratios object has current_indicators set to None.
    """
    mock_strategy_ = Strategies(id=1, acronym=SMA, name='Moving Average')
    with patch.object(StrategyCrud, 'get_by_acronym', return_value=mock_strategy_):
        strategy_ = MovingAverage(SMA, 'Close', 'Sma')

    analysis_context_ = AnalysisContext(
        symbol='TEST',
        timeframe=DAILY,
        from_date=date(2025, 1, 1),
        to_date=date(2025, 1, 30),
        initial_price=100.0,
        final_price=110.0,
        last_bar_number=29,
        future_bar_number=30,
    )

    close_prices_ = np.array([100.0, 105.0, 110.0, 115.0], dtype=np.float64)
    percent_changes_ = np.array([0.0, 0.05, 0.0476, 0.0455], dtype=np.float64)
    prices_df_ = pl.DataFrame({
        'Date': [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 4)],
        'Close': close_prices_,
        'PercentChange': percent_changes_,
        'BarNumber': [0, 1, 2, 3],
    })

    inputs_ = {'period': 20}
    input_bars_ = np.array([0], dtype=np.int32)
    output_bars_ = np.array([3], dtype=np.int32)

    ratios_ = strategy_.perfile_performance(
        analysis_context_,
        inputs_,
        input_bars_,
        output_bars_,
        close_prices_,
        prices_df_,
    )

    assert ratios_ is not None
    assert ratios_.current_indicators is None


def test_strategy_identify_signatures_conform_to_lsp() -> None:
    """
    GIVEN StrategyABC and its concrete subclasses (MovingAverage, RsiTwoBands, RsiRollerCoaster).
    WHEN their identify method signatures are inspected.
    THEN all signatures match the base StrategyABC contract with identical parameter names and defaults.
    """
    base_sig_ = inspect.signature(StrategyABC.identify)
    base_params_ = list(base_sig_.parameters.keys())
    expected_params_ = [
        'self',
        'symbol',
        'timeframe',
        'only_long_positions',
        'prices_df',
        'close_prices',
        'win_probability_threshold',
        'is_input_eligible',
        'verbosity_level',
    ]
    assert base_params_ == expected_params_

    for strategy_cls_ in (MovingAverage, RsiTwoBands, RsiRollerCoaster):
        cls_sig_ = inspect.signature(strategy_cls_.identify)
        cls_params_ = list(cls_sig_.parameters.keys())
        assert cls_params_ == expected_params_, f'{strategy_cls_.__name__}.identify signature violates LSP'
        assert cls_sig_.parameters['is_input_eligible'].default is None
        assert cls_sig_.parameters['verbosity_level'].default == base_sig_.parameters['verbosity_level'].default


def test_identify_where_to_stop_loss_daily_clamping() -> None:
    """
    GIVEN a price DataFrame with wide Mogalef stop loss corridors (> 12% from Close)
    WHEN identify_where_to_stop_loss is evaluated on the DAILY timeframe with default 12% cap
    THEN LongStopLoss is clamped to Close * 0.88 and ShortStopLoss is clamped to Close * 1.12.
    """
    df_ = pl.DataFrame({
        'Date': [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
        'Open': [100.0, 100.0, 100.0],
        'High': [105.0, 105.0, 105.0],
        'Low': [95.0, 95.0, 95.0],
        'Close': [100.0, 100.0, 100.0],
        'BarNumber': [0, 1, 2],
        'LongStopLoss': [70.0, 75.0, 80.0],
        'ShortStopLoss': [130.0, 125.0, 120.0],
    })
    close_prices_ = df_['Close'].to_numpy()

    result_df_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, df_, close_prices_)

    assert np.allclose(result_df_['LongStopLoss'].to_numpy(), [88.0, 88.0, 88.0])
    assert np.allclose(result_df_['ShortStopLoss'].to_numpy(), [112.0, 112.0, 112.0])


def test_identify_where_to_stop_loss_weekly_clamping() -> None:
    """
    GIVEN a price DataFrame with wide Mogalef stop loss corridors (> 18% from Close)
    WHEN identify_where_to_stop_loss is evaluated on the WEEKLY timeframe with default 18% cap
    THEN LongStopLoss is clamped to Close * 0.82 and ShortStopLoss is clamped to Close * 1.18.
    """
    df_ = pl.DataFrame({
        'Date': [date(2025, 1, 1), date(2025, 1, 8), date(2025, 1, 15)],
        'Open': [100.0, 100.0, 100.0],
        'High': [105.0, 105.0, 105.0],
        'Low': [95.0, 95.0, 95.0],
        'Close': [100.0, 100.0, 100.0],
        'BarNumber': [0, 1, 2],
        'LongStopLoss': [70.0, 75.0, 80.0],
        'ShortStopLoss': [130.0, 125.0, 120.0],
    })
    close_prices_ = df_['Close'].to_numpy()

    result_df_ = RsiStrategyABC.identify_where_to_stop_loss(WEEKLY, df_, close_prices_)

    assert np.allclose(result_df_['LongStopLoss'].to_numpy(), [82.0, 82.0, 82.0])
    assert np.allclose(result_df_['ShortStopLoss'].to_numpy(), [118.0, 118.0, 118.0])


def test_identify_where_to_stop_loss_preserves_tighter_corridors() -> None:
    """
    GIVEN a price DataFrame where Mogalef stop loss corridors are narrower than the clamping cap (e.g. 5% span)
    WHEN identify_where_to_stop_loss is evaluated
    THEN LongStopLoss and ShortStopLoss retain the tighter Mogalef levels without artificial widening.
    """
    df_ = pl.DataFrame({
        'Date': [date(2025, 1, 1), date(2025, 1, 2)],
        'Open': [100.0, 200.0],
        'High': [105.0, 205.0],
        'Low': [95.0, 195.0],
        'Close': [100.0, 200.0],
        'BarNumber': [0, 1],
        'LongStopLoss': [95.0, 192.0],
        'ShortStopLoss': [105.0, 208.0],
    })
    close_prices_ = df_['Close'].to_numpy()

    # Daily cap is 12%, but Mogalef is tighter (5% / 4%)
    result_daily_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, df_, close_prices_)
    assert np.allclose(result_daily_['LongStopLoss'].to_numpy(), [95.0, 192.0])
    assert np.allclose(result_daily_['ShortStopLoss'].to_numpy(), [105.0, 208.0])

    # Weekly cap is 18%, but Mogalef is tighter (5% / 4%)
    result_weekly_ = RsiStrategyABC.identify_where_to_stop_loss(WEEKLY, df_, close_prices_)
    assert np.allclose(result_weekly_['LongStopLoss'].to_numpy(), [95.0, 192.0])
    assert np.allclose(result_weekly_['ShortStopLoss'].to_numpy(), [105.0, 208.0])


def test_identify_where_to_stop_loss_disabled_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN settings where stop_loss_cap_daily and stop_loss_cap_weekly are disabled (0.0)
    WHEN identify_where_to_stop_loss is evaluated
    THEN LongStopLoss and ShortStopLoss match unconstrained Mogalef levels without modification.
    """
    monkeypatch.setattr(get_settings(), 'stop_loss_cap_daily', 0.0)
    monkeypatch.setattr(get_settings(), 'stop_loss_cap_weekly', 0.0)

    df_ = pl.DataFrame({
        'Date': [date(2025, 1, 1), date(2025, 1, 2)],
        'Open': [100.0, 100.0],
        'High': [105.0, 105.0],
        'Low': [95.0, 95.0],
        'Close': [100.0, 100.0],
        'BarNumber': [0, 1],
        'LongStopLoss': [60.0, 65.0],
        'ShortStopLoss': [140.0, 135.0],
    })
    close_prices_ = df_['Close'].to_numpy()

    result_df_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, df_, close_prices_)
    assert np.allclose(result_df_['LongStopLoss'].to_numpy(), [60.0, 65.0])
    assert np.allclose(result_df_['ShortStopLoss'].to_numpy(), [140.0, 135.0])


def test_identify_where_to_stop_loss_full_calculation_respects_clamping(
    ohlcv_factory: Callable[..., pl.DataFrame],
) -> None:
    """
    GIVEN a standard OHLC DataFrame requiring full Mogalef indicator calculation
    WHEN identify_where_to_stop_loss is evaluated on DAILY and WEEKLY timeframes
    THEN LongStopLoss is strictly >= Close * (1 - cap) and ShortStopLoss is strictly <= Close * (1 + cap).
    """
    df_ = ohlcv_factory(50)
    close_prices_ = df_['Close'].to_numpy()

    # Daily evaluation (cap = 0.12)
    result_daily_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, df_, close_prices_)
    min_allowed_long_daily_ = result_daily_['Close'] * (1.0 - 0.12)
    max_allowed_short_daily_ = result_daily_['Close'] * (1.0 + 0.12)
    assert (result_daily_['LongStopLoss'] >= min_allowed_long_daily_ - 1e-9).all()
    assert (result_daily_['ShortStopLoss'] <= max_allowed_short_daily_ + 1e-9).all()

    # Weekly evaluation (cap = 0.18)
    result_weekly_ = RsiStrategyABC.identify_where_to_stop_loss(WEEKLY, df_, close_prices_)
    min_allowed_long_weekly_ = result_weekly_['Close'] * (1.0 - 0.18)
    max_allowed_short_weekly_ = result_weekly_['Close'] * (1.0 + 0.18)
    assert (result_weekly_['LongStopLoss'] >= min_allowed_long_weekly_ - 1e-9).all()
    assert (result_weekly_['ShortStopLoss'] <= max_allowed_short_weekly_ + 1e-9).all()



