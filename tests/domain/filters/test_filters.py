# tests/domain/filters/test_filters.py

# --- Python modules ---
import datetime

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.domain.filters import (
    AtrVolatilityFilter,
    FilterABC,
    PriceActionFilter,
    SmaTrendFilter,
    get_filter,
    get_filter_masks,
)


class _DummyPassthroughFilter(FilterABC):
    """Test-local implementation of FilterABC to verify abstract contract."""

    @property
    def name(self) -> str:
        return 'dummy_passthrough'

    def get_masks(self, prices_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        total_bars_ = prices_df.height
        mask_ = np.ones(total_bars_, dtype=np.bool_)
        return mask_, mask_


def _build_dummy_ohlc_df(rows: int) -> pl.DataFrame:
    """Helper to generate a synthetic OHLC DataFrame for filter tests."""
    dates_ = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i_) for i_ in range(rows)]
    # Trending prices: 100.0, 100.5, 101.0, ...
    base_prices_ = np.linspace(100.0, 200.0, rows)
    return pl.DataFrame({
        'Date': dates_,
        'Open': base_prices_ - 0.5,
        'High': base_prices_ + 1.0,
        'Low': base_prices_ - 1.0,
        'Close': base_prices_ + 0.5,
        'Volume': np.full(rows, 1000000, dtype=np.int64),
    })


def test_filter_abc_contract_with_dummy_filter() -> None:
    """
    GIVEN a test-local FilterABC subclass.
    WHEN generating masks for a DataFrame with 10 price bars.
    THEN both Long and Short masks have length 10 and all elements are True.
    """
    df_ = _build_dummy_ohlc_df(10)
    filter_ = _DummyPassthroughFilter()

    assert filter_.name == 'dummy_passthrough'
    long_mask_, short_mask_ = filter_.get_masks(df_)

    assert len(long_mask_) == 10
    assert len(short_mask_) == 10
    assert np.all(long_mask_)
    assert np.all(short_mask_)



def test_price_action_filter_directional_logic() -> None:
    """
    GIVEN candles with known close locations and directions.
    WHEN PriceActionFilter evaluates the bars with default thresholds (0.40 Long, 0.25 Short).
    THEN bullish bars with location >= 0.40 are Long eligible,
         bearish bars with location <= 0.25 are Short eligible,
         and non-confirming or zero-range bars are ineligible for both.
    """
    df_ = pl.DataFrame({
        'Date': [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), datetime.date(2020, 1, 3),
                 datetime.date(2020, 1, 4)],
        # Bar 0: Bullish: Open 10, Low 10, High 20, Close 18 (location = 8/10 = 0.80 >= 0.40, Close > Open) -> Long True, Short False
        # Bar 1: Bearish: Open 20, Low 10, High 20, Close 12 (location = 2/10 = 0.20 <= 0.25, Close < Open) -> Long False, Short True
        # Bar 2: Bearish but location > 0.25: Open 15, Low 10, High 20, Close 13 (location = 3/10 = 0.30 > 0.25, Close < Open) -> Long False, Short False
        # Bar 3: Zero-range: Open 15, Low 15, High 15, Close 15 (range == 0) -> Long False, Short False
        'Open': [10.0, 20.0, 15.0, 15.0],
        'High': [20.0, 20.0, 20.0, 15.0],
        'Low': [10.0, 10.0, 10.0, 15.0],
        'Close': [18.0, 12.0, 13.0, 15.0],
        'Volume': [1000, 1000, 1000, 1000],
    })
    filter_ = PriceActionFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    assert long_mask_[0] and not short_mask_[0]
    assert not long_mask_[1] and short_mask_[1]
    assert not long_mask_[2] and not short_mask_[2]
    assert not long_mask_[3] and not short_mask_[3]


def test_atr_volatility_filter_insufficient_history() -> None:
    """
    GIVEN a DataFrame with fewer than 252 bars.
    WHEN AtrVolatilityFilter generates eligibility masks.
    THEN all elements in the masks evaluate to False due to insufficient quantile lookback history.
    """
    df_ = _build_dummy_ohlc_df(100)
    filter_ = AtrVolatilityFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    assert len(long_mask_) == 100
    assert not np.any(long_mask_)
    assert not np.any(short_mask_)


def test_atr_volatility_filter_percentile_boundary() -> None:
    """
    GIVEN a stationary series with 300 bars of fluctuating ATR and an extreme shock on the final bar.
    WHEN AtrVolatilityFilter evaluates the series.
    THEN bars in the middle volatility distribution are eligible,
         and extreme outliers beyond the 90th percentile are rejected.
    """
    rng_ = np.random.default_rng(42)
    rows_ = 300
    dates_ = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i_) for i_ in range(rows_)]
    # Stationary price around 100 with random typical daily ranges [1.0, 3.0]
    ranges_ = rng_.uniform(1.0, 3.0, rows_)
    # On the final bar, introduce an extreme volatility shock (range = 25.0)
    ranges_[-1] = 25.0

    closes_ = 100.0 + rng_.uniform(-0.5, 0.5, rows_)
    highs_ = closes_ + ranges_ / 2.0
    lows_ = closes_ - ranges_ / 2.0
    opens_ = closes_ + rng_.uniform(-0.2, 0.2, rows_)

    df_ = pl.DataFrame({
        'Date': dates_,
        'Open': opens_,
        'High': highs_,
        'Low': lows_,
        'Close': closes_,
        'Volume': np.full(rows_, 1000000, dtype=np.int64),
    })
    filter_ = AtrVolatilityFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    # Prior 252 bars are required for the quantile window, so bar 0..251 are ineligible
    assert not long_mask_[0]
    assert not long_mask_[251]
    # In the stationary regime with typical ranges, verify that eligible bars exist
    assert np.any(long_mask_)
    # The extreme shock bar on the final bar (range 25.0 vs historical 1.0-3.0) must be rejected
    assert not long_mask_[-1]
    assert not short_mask_[-1]


def test_sma_trend_filter_insufficient_history() -> None:
    """
    GIVEN a DataFrame with fewer than 220 bars.
    WHEN SmaTrendFilter generates eligibility masks.
    THEN all elements in the masks evaluate to False.
    """
    df_ = _build_dummy_ohlc_df(150)
    filter_ = SmaTrendFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    assert len(long_mask_) == 150
    assert not np.any(long_mask_)
    assert not np.any(short_mask_)


def test_sma_trend_filter_uptrend_and_downtrend() -> None:
    """
    GIVEN an upward-trending price series with 250 bars.
    WHEN SmaTrendFilter evaluates the series.
    THEN after bar 220, Long setups are eligible while Short setups are ineligible.
    """
    rows_ = 250
    df_ = _build_dummy_ohlc_df(rows_)
    filter_ = SmaTrendFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    assert not long_mask_[0]
    assert not long_mask_[210]
    # In a continuous uptrend, Close > SMA and SMA slope > 0
    assert long_mask_[230]
    assert not short_mask_[230]


def test_filter_registry_and_helper_functions() -> None:
    """
    GIVEN registered filter names ('atr_volatility', 'price_action', 'sma_trend') and baseline.
    WHEN get_filter and get_filter_masks are invoked.
    THEN instances of the corresponding FilterABC subclasses are returned,
         baseline/None queries return (None, None), and unknown filter names raise KeyError.
    """
    df_ = _build_dummy_ohlc_df(10)

    for name_ in ['atr_volatility', 'price_action', 'sma_trend']:
        filter_ = get_filter(name_)
        assert filter_ is not None
        long_mask_, short_mask_ = get_filter_masks(name_, df_)
        assert long_mask_ is not None
        assert short_mask_ is not None
        assert len(long_mask_) == 10
        assert len(short_mask_) == 10

    # Baseline queries return (None, None) with zero allocation
    for baseline_name_ in [None, '', 'none', 'baseline']:
        long_mask_, short_mask_ = get_filter_masks(baseline_name_, df_)
        assert long_mask_ is None
        assert short_mask_ is None

    with pytest.raises(KeyError):
        get_filter('unregistered_filter_name')

    with pytest.raises(KeyError):
        get_filter('baseline')

