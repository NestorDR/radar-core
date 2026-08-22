# tests/domain/strategies/test_base_strategy.py

# --- Python modules ---
from datetime import date, timedelta
from unittest.mock import patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.domain.strategies.base_strategy import RsiStrategyABC, _find_stop_loss_bars
from radar_core.helpers.constants import DAILY, WEEKLY
from radar_core.infrastructure import PriceProvider
from radar_core.infrastructure.security_repository import SecurityRepository


def _sample_price_df(rows: int = 30) -> pl.DataFrame:
    """Helper to generate sample OHLC and BarNumber data for stop loss testing."""
    np.random.seed(42)
    base_price_ = 100.0 + np.cumsum(np.random.randn(rows) * 1.5)
    high_prices_ = base_price_ + np.random.uniform(0.5, 2.0, size=rows)
    low_prices_ = base_price_ - np.random.uniform(0.5, 2.0, size=rows)
    open_prices_ = base_price_ + np.random.uniform(-0.5, 0.5, size=rows)
    close_prices_ = base_price_ + np.random.uniform(-0.5, 0.5, size=rows)

    return pl.DataFrame({
        'Date': [date(2025, 1, 1) + timedelta(days=i_) for i_ in range(rows)],
        'Open': open_prices_,
        'High': high_prices_,
        'Low': low_prices_,
        'Close': close_prices_,
        'BarNumber': np.arange(rows, dtype=np.int32),
    })


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


def test_set_mogalef_stop_loss_standard_bounds() -> None:
    """
    GIVEN a Polars DataFrame with OHLC columns.
    WHEN set_mogalef_stop_loss is executed.
    THEN LongStopLoss and ShortStopLoss match MogalefLower and MogalefUpper across all valid rows.
    """
    df_ = _sample_price_df(30)
    result_df_ = RsiStrategyABC.set_mogalef_stop_loss(df_, period_reg=3, period_dev=7, multiplier=2.0)

    assert 'LongStopLoss' in result_df_.columns
    assert 'ShortStopLoss' in result_df_.columns

    valid_rows_ = result_df_.filter(pl.col('LongStopLoss').is_not_null() & pl.col('ShortStopLoss').is_not_null())
    assert (valid_rows_['LongStopLoss'] == valid_rows_['MogalefLower']).all()
    assert (valid_rows_['ShortStopLoss'] == valid_rows_['MogalefUpper']).all()


def test_set_mogalef_stop_loss_idempotency() -> None:
    """
    GIVEN a DataFrame that already contains LongStopLoss and ShortStopLoss columns.
    WHEN set_mogalef_stop_loss is called.
    THEN the DataFrame is returned unchanged.
    """
    df_ = _sample_price_df(10).with_columns([
        pl.lit(95.0).alias('LongStopLoss'),
        pl.lit(105.0).alias('ShortStopLoss')
    ])
    result_df_ = RsiStrategyABC.set_mogalef_stop_loss(df_)

    assert (result_df_['LongStopLoss'] == 95.0).all()
    assert (result_df_['ShortStopLoss'] == 105.0).all()


def test_set_stop_loss_backwards_compatibility() -> None:
    """
    GIVEN a DataFrame with High, Low, and Close columns.
    WHEN the baseline set_stop_loss is executed.
    THEN LongStopLoss and ShortStopLoss columns are produced using ATR and rolling window.
    """
    df_ = _sample_price_df(30)
    result_df_ = RsiStrategyABC.set_stop_loss(df_, bars_for_stop_loss=10)

    assert 'LongStopLoss' in result_df_.columns
    assert 'ShortStopLoss' in result_df_.columns
    assert 'Atr' in result_df_.columns


def test_identify_where_to_stop_loss_daily_and_weekly() -> None:
    """
    GIVEN a price DataFrame and close prices array.
    WHEN identify_where_to_stop_loss is called for DAILY and WEEKLY timeframes.
    THEN stop loss columns and trigger bar numbers are generated.
    """
    df_ = _sample_price_df(35)
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


@pytest.fixture(scope='module')
def real_spy_prices() -> pl.DataFrame:
    """Fixture downloading real SPY daily data from Yahoo Finance."""
    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value={'SPY': 'SPY'}):
        prices_data_ = PriceProvider(long_term=False).get_prices(['SPY'])

    prices_df_ = prices_data_['SPY']
    return prices_df_.with_columns(
        pl.arange(0, pl.len(), eager=False).cast(pl.Int32).alias('BarNumber')
    )


def test_identify_where_to_stop_loss_parity_on_real_market_data(real_spy_prices: pl.DataFrame) -> None:
    """
    GIVEN real SPY daily market prices.
    WHEN RsiStrategyABC.identify_where_to_stop_loss is executed.
    THEN BarNumberForLongStop and BarNumberForShortStop columns are created and match legacy values.
    """
    df_ = real_spy_prices.clone()
    close_prices_ = df_['Close'].to_numpy()

    result_df_ = RsiStrategyABC.identify_where_to_stop_loss(DAILY, df_, close_prices_)

    assert 'BarNumberForLongStop' in result_df_.columns
    assert 'BarNumberForShortStop' in result_df_.columns
    assert result_df_.height == real_spy_prices.height

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
