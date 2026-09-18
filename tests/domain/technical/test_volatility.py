# tests/domain/technical/test_volatility.py

# --- Python modules ---
from collections.abc import Callable
from typing import Any

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest
import talib

# --- App modules ---
from radar_core.domain.technical.volatility import ATR, MogalefBands


def test_mogalef_bands_default_output_contract_and_warmup() -> None:
    """
    GIVEN a constant weighted typical-price series with valid OHLC columns.
    WHEN MogalefBands is executed with default parameters.
    THEN the two output bands contain null warm-up values followed by the constant corridor.
    """
    prices_df_ = pl.DataFrame({
        'Open': [10.0] * 10,
        'High': [12.0] * 10,
        'Low': [8.0] * 10,
        'Close': [10.0] * 10,
    })
    result_df_ = MogalefBands(prices_df_)

    assert 'MogalefUpper' in result_df_.columns
    assert 'MogalefLower' in result_df_.columns
    assert result_df_.height == 10
    assert 'MogalefCentral' not in result_df_.columns

    assert result_df_['MogalefUpper'][:8].is_null().all()
    assert result_df_['MogalefLower'][:8].is_null().all()
    np.testing.assert_allclose(result_df_['MogalefUpper'][8:].to_numpy(), [10.0, 10.0])
    np.testing.assert_allclose(result_df_['MogalefLower'][8:].to_numpy(), [10.0, 10.0])


def test_mogalef_bands_stepped_levels_hold_and_reset() -> None:
    """
    GIVEN a price series whose regression enters, remains within, and exits a corridor.
    WHEN MogalefBands is executed with short lookback periods.
    THEN levels hold inside the corridor and reset after a breakout for both log and linear scales.
    """
    values_ = np.array([10.0, 10.0, 14.0, 15.0, 20.0])
    prices_df_ = pl.DataFrame({
        'Open': values_,
        'High': values_,
        'Low': values_,
        'Close': values_,
    })

    # 1. Linear scale (log_scale=False)
    result_lin_df_ = MogalefBands(prices_df_, period_reg=2, period_dev=2, multiplier=1.0, log_scale=False)
    expected_upper_lin_ = np.array([np.nan, np.nan, 16.0, 16.0, 22.5])
    expected_lower_lin_ = np.array([np.nan, np.nan, 12.0, 12.0, 17.5])
    np.testing.assert_allclose(result_lin_df_['MogalefUpper'].to_numpy(), expected_upper_lin_, equal_nan=True)
    np.testing.assert_allclose(result_lin_df_['MogalefLower'].to_numpy(), expected_lower_lin_, equal_nan=True)

    # 2. Logarithmic scale (default log_scale=True)
    result_log_df_ = MogalefBands(prices_df_, period_reg=2, period_dev=2, multiplier=1.0)
    expected_upper_log_ = np.array([np.nan, np.nan, 16.565023, 16.565023, 23.094011])
    expected_lower_log_ = np.array([np.nan, np.nan, 11.832160, 11.832160, 17.320508])
    np.testing.assert_allclose(result_log_df_['MogalefUpper'].to_numpy(), expected_upper_log_, rtol=1e-4, equal_nan=True)
    np.testing.assert_allclose(result_log_df_['MogalefLower'].to_numpy(), expected_lower_log_, rtol=1e-4, equal_nan=True)


def test_mogalef_bands_custom_parameters() -> None:
    """
    GIVEN a Polars DataFrame and custom period/multiplier settings.
    WHEN MogalefBands is executed.
    THEN the calculated values reflect the custom configuration.
    """
    values_ = np.array([10.0, 10.0, 14.0, 15.0, 20.0])
    prices_df_ = pl.DataFrame({
        'Open': values_,
        'High': values_,
        'Low': values_,
        'Close': values_,
    })

    result_df_ = MogalefBands(prices_df_, period_reg=2, period_dev=2, multiplier=2.0, log_scale=False)

    expected_upper_ = np.array([np.nan, np.nan, 18.0, 18.0, 25.0])
    expected_lower_ = np.array([np.nan, np.nan, 10.0, 10.0, 15.0])
    np.testing.assert_allclose(result_df_['MogalefUpper'].to_numpy(), expected_upper_, equal_nan=True)
    np.testing.assert_allclose(result_df_['MogalefLower'].to_numpy(), expected_lower_, equal_nan=True)


def test_mogalef_bands_high_volatility_decay_prevents_negative_lower() -> None:
    """
    GIVEN a price series experiencing a steep collapse from 1000 down to 10.
    WHEN MogalefBands is executed under linear vs logarithmic scale.
    THEN linear mode produces a negative lower band and stays frozen,
    WHILE logarithmic mode guarantees strictly positive lower band and active corridor stepping.
    """
    values_ = np.array([1000.0, 900.0, 800.0, 600.0, 400.0, 300.0, 200.0, 150.0, 100.0, 80.0, 50.0, 30.0, 20.0, 15.0, 10.0])
    prices_df_ = pl.DataFrame({
        'Open': values_,
        'High': values_,
        'Low': values_,
        'Close': values_,
    })

    # Linear calculation produces negative lower band that stays permanently frozen
    linear_df_ = MogalefBands(prices_df_, period_reg=3, period_dev=5, multiplier=1.5, log_scale=False)
    linear_lower_ = linear_df_['MogalefLower'].to_numpy()
    assert np.nanmin(linear_lower_) < 0.0
    # Linear lower band remains completely flat/frozen once negative
    valid_linear_ = linear_df_['MogalefLower'].drop_nulls().to_numpy()
    assert (valid_linear_ == valid_linear_[0]).all()

    # Logarithmic calculation guarantees strictly positive lower band and active corridor stepping
    log_df_ = MogalefBands(prices_df_, period_reg=3, period_dev=5, multiplier=1.5, log_scale=True)
    valid_log_lower_ = log_df_['MogalefLower'].drop_nulls().to_numpy()
    assert (valid_log_lower_ > 0.0).all()
    # Corridor steps down dynamically as prices decline
    assert valid_log_lower_[-1] < valid_log_lower_[0]


def test_mogalef_bands_missing_columns() -> None:
    """
    GIVEN a DataFrame missing one of the required OHLC columns.
    WHEN MogalefBands is called.
    THEN a ValueError is raised specifying the missing column(s).
    """
    df_missing_ = pl.DataFrame({'High': [10.0, 11.0], 'Low': [9.0, 9.5], 'Close': [9.8, 10.5]})
    with pytest.raises(ValueError, match=r'Missing required columns'):
        MogalefBands(df_missing_)


@pytest.mark.parametrize(
    ('kwargs', 'match_pattern'),
    [
        ({'period_reg': 0}, r'Lookback periods'),
        ({'period_dev': -1}, r'Lookback periods'),
        ({'multiplier': -0.5}, r'Multiplier'),
    ],
    ids=['zero_period_reg', 'negative_period_dev', 'negative_multiplier'],
)
def test_mogalef_bands_invalid_parameters(
    ohlcv_factory: Callable[..., pl.DataFrame],
    kwargs: dict[str, Any],
    match_pattern: str,
) -> None:
    """
    GIVEN invalid parameters (period < 1 or multiplier < 0).
    WHEN MogalefBands is called.
    THEN a ValueError is raised with a descriptive error message.
    """
    df_ = ohlcv_factory(10)
    with pytest.raises(ValueError, match=match_pattern):
        MogalefBands(df_, **kwargs)


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
    assert {'MogalefUpper', 'MogalefLower'}.issubset(result_empty_.columns)

    # Short DataFrame (fewer bars than lookback)
    short_df_ = pl.DataFrame({'Open': [10.0, 10.5], 'High': [11.0, 11.5], 'Low': [9.0, 9.5], 'Close': [10.2, 10.8]})
    result_short_ = MogalefBands(short_df_, period_reg=3, period_dev=7)
    assert result_short_.height == 2
    assert result_short_['MogalefUpper'].is_null().all()
    assert result_short_['MogalefLower'].is_null().all()


def test_atr_standard_calculation(ohlcv_factory: Callable[..., pl.DataFrame]) -> None:
    """
    GIVEN a DataFrame with High, Low, and Close columns.
    WHEN ATR is called.
    THEN the Atr column is added matching talib.ATR output.
    """
    df_ = ohlcv_factory(30)
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
    with pytest.raises(ValueError, match=r'Missing required columns'):
        ATR(df_missing_)
