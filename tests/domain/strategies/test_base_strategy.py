# tests/domain/strategies/test_base_strategy.py

# --- Python modules ---
from datetime import date, timedelta

# --- Third Party Libraries ---
import numpy as np
import polars as pl

# --- App modules ---
from radar_core.domain.strategies.base_strategy import RsiStrategyABC
from radar_core.helpers.constants import DAILY, WEEKLY


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
