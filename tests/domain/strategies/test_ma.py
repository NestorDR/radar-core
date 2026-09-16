# tests/domain/strategies/test_ma.py

# --- Python modules ---
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.domain.strategies import MovingAverage
from radar_core.domain.strategies.ma import _find_trades_sma
from radar_core.domain.technical import RSI
from radar_core.helpers.constants import DAILY, RSI_SMA, SMA, WEEKLY
from radar_core.helpers.datetime_helper import to_weekly_timeframe
from radar_core.infrastructure import PriceProvider
from radar_core.infrastructure.crud import StrategyCrud
from radar_core.infrastructure.ratio_repository import RatioRepository
from radar_core.infrastructure.security_repository import SecurityRepository
from radar_core.models import Strategies


@pytest.fixture(autouse=True)
def mock_strategy_db():
    """Mocks database lookup of strategy metadata and flag_in_process for pure offline testing."""
    mock_strategy_ = Strategies(
        id=1,
        acronym='SMA',
        name='Simple Moving Average',
    )
    with patch.object(StrategyCrud, 'get_by_acronym', return_value=mock_strategy_), \
         patch.object(RatioRepository, 'flag_in_process', return_value=0):
        yield


def test_find_trades_sma_buffer_preallocation_multiple_trades() -> None:
    """
    GIVEN price data containing two independent completed long trades across SMA(3).
    WHEN _find_trades_sma evaluates period 3.
    THEN both input and output bars are correctly populated into pre-allocated NumPy buffers.
    """
    values_ = np.array(
        [10.0, 10.0, 10.0, 12.0, 14.0, 8.0, 6.0, 12.0, 15.0, 7.0],
        dtype=np.float64,
    )

    input_bar_numbers_, output_bar_numbers_ = _find_trades_sma(
        values_,
        3,
        True,
        len(values_),
    )

    assert isinstance(input_bar_numbers_, np.ndarray)
    assert isinstance(output_bar_numbers_, np.ndarray)
    assert input_bar_numbers_.tolist() == [3, 7]
    assert output_bar_numbers_.tolist() == [5, 9]


def test_find_trades_sma_handles_leading_nans() -> None:
    """
    GIVEN an indicator series with leading NaNs (e.g. RSI) and a period of 3.
    WHEN _find_trades_sma evaluates crossovers.
    THEN the kernel skips leading NaNs, calculates running sum over valid window, and generates signals.
    """
    values_ = np.array(
        [np.nan, np.nan, 10.0, 10.0, 10.0, 15.0, 5.0],
        dtype=np.float64,
    )

    input_bar_numbers_, output_bar_numbers_ = _find_trades_sma(
        values_,
        3,
        True,
        len(values_),
    )

    assert input_bar_numbers_.tolist() == [5]
    assert output_bar_numbers_.tolist() == [6]


@pytest.fixture(scope='module')
def real_spy_prices() -> pl.DataFrame:
    """
    Fixture that downloads real historical price data for SPY from Yahoo Finance
    and prepares it with BarNumber and shared RSI indicator.

    :return: Polars DataFrame containing historical prices for SPY.
    """
    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value={'SPY': 'SPY'}):
        prices_data_ = PriceProvider(long_term=False).get_prices(['SPY'], datetime.now(timezone.utc))

    assert 'SPY' in prices_data_, 'Failed to retrieve real SPY prices from Yahoo Finance.'
    prices_df_ = prices_data_['SPY']

    # Add 0-indexed BarNumber column
    prices_df_ = prices_df_.with_columns(
        pl.arange(0, pl.len(), eager=False).cast(pl.Int32).alias('BarNumber')
    )
    return RSI(prices_df_)


def test_moving_average_identify_sma_daily_execution(real_spy_prices: pl.DataFrame) -> None:
    """
    GIVEN real SPY daily price data.
    WHEN MovingAverage.identify is executed for SMA with pre-allocated buffers.
    THEN positive ratios are persisted and best_long strategy is correctly tracked.
    """
    df_ = real_spy_prices.clone()
    close_prices_ = df_['Close'].to_numpy()

    strategy_ = MovingAverage(SMA, 'Close', 'Sma', min_period=8, max_period=50)
    mock_persist_ = MagicMock(return_value=1)
    strategy_.persist_ratios = mock_persist_

    strategy_.identify('SPY', DAILY, False, df_, close_prices_, 0.0)

    assert mock_persist_.called
    positive_ratios_ = mock_persist_.call_args[0][0]
    analysis_context_ = mock_persist_.call_args[0][1]

    assert len(positive_ratios_) > 0
    assert all(r_.win_probability > 0.0 for r_ in positive_ratios_)
    assert analysis_context_.best_long.inputs != ''


def test_moving_average_identify_rsi_sma_weekly_execution(real_spy_prices: pl.DataFrame) -> None:
    """
    GIVEN real SPY weekly price data with RSI indicator.
    WHEN MovingAverage.identify is executed for RSI(14) SMA with pre-allocated buffers.
    THEN positive ratios are evaluated and persisted without error.
    """
    weekly_df_ = to_weekly_timeframe(real_spy_prices.clone())
    weekly_df_ = weekly_df_.with_columns(
        pl.arange(0, pl.len(), eager=False).cast(pl.Int32).alias('BarNumber')
    )
    weekly_df_ = RSI(weekly_df_)

    close_prices_ = weekly_df_['Close'].to_numpy()

    strategy_ = MovingAverage(RSI_SMA, 'Rsi', 'RsiSma', min_period=8, max_period=50)
    mock_persist_ = MagicMock(return_value=1)
    strategy_.persist_ratios = mock_persist_

    strategy_.identify('SPY', WEEKLY, False, weekly_df_, close_prices_, 0.0)

    assert mock_persist_.called
    positive_ratios_ = mock_persist_.call_args[0][0]
    analysis_context_ = mock_persist_.call_args[0][1]

    assert len(positive_ratios_) > 0
    assert all(r_.win_probability > 0.0 for r_ in positive_ratios_)
    assert analysis_context_.best_long.inputs != ''


def test_moving_average_identify_filters_by_win_probability_threshold(real_spy_prices: pl.DataFrame) -> None:
    """
    GIVEN real SPY daily price data.
    WHEN MovingAverage.identify is executed with win_probability_threshold=0.5.
    THEN setups with win_probability < 0.5 are filtered out.
    """
    df_ = real_spy_prices.clone()
    close_prices_ = df_['Close'].to_numpy()

    strategy_ = MovingAverage(SMA, 'Close', 'Sma', min_period=8, max_period=50)
    mock_persist_ = MagicMock(return_value=1)
    strategy_.persist_ratios = mock_persist_

    strategy_.identify('SPY', DAILY, False, df_, close_prices_, 0.5)

    assert mock_persist_.called
    positive_ratios_ = mock_persist_.call_args[0][0]
    assert all(r_.win_probability >= 0.5 for r_ in positive_ratios_)


