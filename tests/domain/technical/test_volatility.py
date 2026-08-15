# tests/domain/technical/test_volatility.py

# --- Python modules ---
from datetime import date, timedelta

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest
import talib

# --- App modules ---
from radar_core.domain.technical.volatility import ATR, MogalefBands


def _sample_ohlc_df(rows: int = 30) -> pl.DataFrame:
    """Helper to generate sample OHLC data for testing."""
    np.random.seed(42)
    base_price_ = 100.0 + np.cumsum(np.random.randn(rows) * 1.5)
    high_prices_ = base_price_ + np.random.uniform(0.5, 2.0, size=rows)
    low_prices_ = base_price_ - np.random.uniform(0.5, 2.0, size=rows)
    open_prices_ = base_price_ + np.random.uniform(-0.5, 0.5, size=rows)
    close_prices_ = base_price_ + np.random.uniform(-0.5, 0.5, size=rows)

    return pl.DataFrame(
        {
            'Date': [date(2025, 1, 1) + timedelta(days=i_) for i_ in range(rows)],
            'Open': open_prices_,
            'High': high_prices_,
            'Low': low_prices_,
            'Close': close_prices_,
        }
    )


def test_mogalef_bands_standard_calculation() -> None:
    """
    GIVEN a Polars DataFrame with valid OHLC columns.
    WHEN MogalefBands is executed with default parameters.
    THEN MogalefCentral, MogalefUpper, and MogalefLower columns are added with correct relationship.
    """
    df_ = _sample_ohlc_df(30)
    result_df_ = MogalefBands(df_)

    assert 'MogalefCentral' in result_df_.columns
    assert 'MogalefUpper' in result_df_.columns
    assert 'MogalefLower' in result_df_.columns
    assert result_df_.height == 30

    # Calculate expected typical price and indicators directly
    typical_price_ = (df_['Open'].to_numpy() + df_['High'].to_numpy() + df_['Low'].to_numpy() + 2.0 * df_[
        'Close'].to_numpy()) / 5.0
    expected_central_ = talib.LINEARREG(typical_price_, 3)
    expected_std_ = talib.STDDEV(typical_price_, 7, nbdev=1.0)
    expected_upper_ = expected_central_ + 2.0 * expected_std_
    expected_lower_ = expected_central_ - 2.0 * expected_std_

    # Verify values at last index
    last_idx_ = 29
    assert np.isclose(result_df_['MogalefCentral'][last_idx_], expected_central_[last_idx_])
    assert np.isclose(result_df_['MogalefUpper'][last_idx_], expected_upper_[last_idx_])
    assert np.isclose(result_df_['MogalefLower'][last_idx_], expected_lower_[last_idx_])

    # Upper band must be >= Central >= Lower band where not null
    valid_rows_ = result_df_.filter(pl.col('MogalefUpper').is_not_null())
    assert (valid_rows_['MogalefUpper'] >= valid_rows_['MogalefCentral']).all()
    assert (valid_rows_['MogalefCentral'] >= valid_rows_['MogalefLower']).all()


def test_mogalef_bands_custom_parameters() -> None:
    """
    GIVEN a Polars DataFrame and custom period/multiplier settings.
    WHEN MogalefBands is executed.
    THEN the calculated values reflect the custom configuration.
    """
    df_ = _sample_ohlc_df(40)
    result_df_ = MogalefBands(df_, period_reg=5, period_dev=10, multiplier=3.0)

    typical_price_ = (df_['Open'].to_numpy() + df_['High'].to_numpy() + df_['Low'].to_numpy() + 2.0 * df_[
        'Close'].to_numpy()) / 5.0
    expected_central_ = talib.LINEARREG(typical_price_, 5)
    expected_std_ = talib.STDDEV(typical_price_, 10, nbdev=1.0)
    expected_upper_ = expected_central_ + 3.0 * expected_std_

    last_idx_ = 39
    assert np.isclose(result_df_['MogalefCentral'][last_idx_], expected_central_[last_idx_])
    assert np.isclose(result_df_['MogalefUpper'][last_idx_], expected_upper_[last_idx_])


def test_mogalef_bands_missing_columns() -> None:
    """
    GIVEN a DataFrame missing one of the required OHLC columns.
    WHEN MogalefBands is called.
    THEN a ValueError is raised specifying the missing column(s).
    """
    df_missing_ = pl.DataFrame({'High': [10.0, 11.0], 'Low': [9.0, 9.5], 'Close': [9.8, 10.5]})
    with pytest.raises(ValueError, match='Missing required columns.*Open'):
        MogalefBands(df_missing_)


def test_mogalef_bands_invalid_parameters() -> None:
    """
    GIVEN invalid parameters (period < 1 or multiplier < 0).
    WHEN MogalefBands is called.
    THEN a ValueError is raised.
    """
    df_ = _sample_ohlc_df(10)
    with pytest.raises(ValueError, match='Lookback periods must be greater than or equal to 1'):
        MogalefBands(df_, period_reg=0)

    with pytest.raises(ValueError, match='Lookback periods must be greater than or equal to 1'):
        MogalefBands(df_, period_dev=-1)

    with pytest.raises(ValueError, match='Multiplier must be non-negative'):
        MogalefBands(df_, multiplier=-0.5)


def test_mogalef_bands_empty_and_short_dataframe() -> None:
    """
    GIVEN an empty or very short DataFrame.
    WHEN MogalefBands is called.
    THEN it returns null/empty columns gracefully without crashing.
    """
    # Empty DataFrame
    empty_df_ = pl.DataFrame(schema={'Open': pl.Float64, 'High': pl.Float64, 'Low': pl.Float64, 'Close': pl.Float64})
    result_empty_ = MogalefBands(empty_df_)
    assert result_empty_.height == 0
    assert {'MogalefCentral', 'MogalefUpper', 'MogalefLower'}.issubset(result_empty_.columns)

    # Short DataFrame (fewer bars than lookback)
    short_df_ = pl.DataFrame({'Open': [10.0, 10.5], 'High': [11.0, 11.5], 'Low': [9.0, 9.5], 'Close': [10.2, 10.8]})
    result_short_ = MogalefBands(short_df_, period_reg=3, period_dev=7)
    assert result_short_.height == 2
    assert result_short_['MogalefCentral'].is_null().all()
    assert result_short_['MogalefUpper'].is_null().all()
    assert result_short_['MogalefLower'].is_null().all()


def test_atr_standard_calculation() -> None:
    """
    GIVEN a DataFrame with High, Low, and Close columns.
    WHEN ATR is called.
    THEN the Atr column is added matching talib.ATR output.
    """
    df_ = _sample_ohlc_df(30)
    result_df_ = ATR(df_, period=14)

    assert 'Atr' in result_df_.columns
    expected_atr_ = talib.ATR(df_['High'].to_numpy(), df_['Low'].to_numpy(), df_['Close'].to_numpy(), 14)
    assert np.isclose(result_df_['Atr'][29], expected_atr_[29])


def test_atr_missing_columns() -> None:
    """
    GIVEN a DataFrame missing required columns for ATR.
    WHEN ATR is called.
    THEN a ValueError is raised.
    """
    df_missing_ = pl.DataFrame({'Open': [10.0], 'Close': [10.5]})
    with pytest.raises(ValueError, match='Missing required columns'):
        ATR(df_missing_)
