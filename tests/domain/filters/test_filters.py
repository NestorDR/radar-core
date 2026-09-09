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
    GIVEN candles with known prior closes, directional spans, and close locations.
    WHEN PriceActionFilter evaluates the bars with default thresholds.
    THEN bullish bars with upward retention >= DEFAULT_LONG_THRESHOLD are Long eligible,
         bearish bars with downward retention >= DEFAULT_SHORT_THRESHOLD are Short eligible,
         and unconfirmed or zero-span bars are ineligible for both.
    """
    df_ = pl.DataFrame({
        'Date': [
            datetime.date(2020, 1, 1),
            datetime.date(2020, 1, 2),
            datetime.date(2020, 1, 3),
            datetime.date(2020, 1, 4),
            datetime.date(2020, 1, 5),
        ],
        # Bar 0: Benchmark setup session: Open 10, Low 9, High 15, Close 10 (Prior close is null -> ineligible)
        # Bar 1: Bullish confirmation: Prior close 10. Open 10, Low 10, High 20, Close 18
        #        Span = 20 - 10 = 10, Retention = (18 - 10) / 10 = 0.80 >= DEFAULT_LONG_THRESHOLD, Close > Open -> Long True, Short False
        # Bar 2: Bearish confirmation: Prior close 18. Open 18, High 18, Low 8, Close 10
        #        Span = 18 - 8 = 10, Retention = (18 - 10) / 10 = 0.80 >= DEFAULT_SHORT_THRESHOLD, Close < Open -> Long False, Short True
        # Bar 3: Bearish but retention < DEFAULT_SHORT_THRESHOLD: Prior close 10. Open 10, High 12, Low 2, Close 5
        #        Span = 10 - 2 = 8, Retention = (10 - 5) / 8 = 0.625 < DEFAULT_SHORT_THRESHOLD, Close < Open -> Long False, Short False
        # Bar 4: Flat zero-span day: Prior close 5. Open 5, High 5, Low 5, Close 5 (Close == Open) -> Long False, Short False
        'Open': [10.0, 10.0, 18.0, 10.0, 5.0],
        'High': [15.0, 20.0, 18.0, 12.0, 5.0],
        'Low': [9.0, 10.0, 8.0, 2.0, 5.0],
        'Close': [10.0, 18.0, 10.0, 5.0, 5.0],
        'Volume': [1000, 1000, 1000, 1000, 1000],
    })
    filter_ = PriceActionFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    assert not long_mask_[0] and not short_mask_[0]
    assert long_mask_[1] and not short_mask_[1]
    assert not long_mask_[2] and short_mask_[2]
    assert not long_mask_[3] and not short_mask_[3]
    assert not long_mask_[4] and not short_mask_[4]


def test_price_action_filter_directional_gap_and_submerged_handling() -> None:
    """
    GIVEN candles where an opening gap down occurs and produces an intraday bounce.
    WHEN PriceActionFilter evaluates the series.
    THEN directional retention rejects the submerged session (High <= Close_prev) as Long ineligible,
         while a subsequent breakout session penetrating above prior close is confirmed.
    """
    df_ = pl.DataFrame({
        'Date': [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)],
        # Bar 0: Benchmark session: High 105, Low 95, Close 100
        # Bar 1: Submerged gap down with green candle: Open 80, High 85, Low 75, Close 82 (Close > Open)
        #        Long span = 85 - 100 = -15 <= 0 -> Long False!
        #        Short span = 100 - 75 = 25, Retention = (100 - 82) / 25 = 0.72 < DEFAULT_SHORT_THRESHOLD (and Close > Open) -> Short False!
        # Bar 2: Bullish recovery penetrating prior close: Prior close 82. Open 83, Low 82, High 92, Close 90
        #        Long span = 92 - 82 = 10, Retention = (90 - 82) / 10 = 0.80 >= DEFAULT_LONG_THRESHOLD -> Long True!
        'Open': [98.0, 80.0, 83.0],
        'High': [105.0, 85.0, 92.0],
        'Low': [95.0, 75.0, 82.0],
        'Close': [100.0, 82.0, 90.0],
        'Volume': [1000, 1000, 1000],
    })
    filter_ = PriceActionFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    # Bar 0 is unanchored -> False
    assert not long_mask_[0] and not short_mask_[0]
    # Bar 1 is submerged below prior close -> Long False
    assert not long_mask_[1] and not short_mask_[1]
    # Bar 2 breaks above prior close and retains 80% of upside span -> Long True
    assert long_mask_[2] and not short_mask_[2]


def test_price_action_filter_initial_bar_fallback() -> None:
    """
    GIVEN a single price bar where prior close is null.
    WHEN PriceActionFilter evaluates the bar.
    THEN unanchored directional spans safely evaluate to False without raising null or index errors.
    """
    df_ = pl.DataFrame({
        'Date': [datetime.date(2020, 1, 1)],
        # Bar 0: Bullish bar: Open 10, High 20, Low 10, Close 18 (prior close is null)
        'Open': [10.0],
        'High': [20.0],
        'Low': [10.0],
        'Close': [18.0],
        'Volume': [1000],
    })
    filter_ = PriceActionFilter()

    long_mask_, short_mask_ = filter_.get_masks(df_)

    assert len(long_mask_) == 1
    assert len(short_mask_) == 1
    assert long_mask_[0] is False or long_mask_[0] == False  # noqa: E712
    assert short_mask_[0] is False or short_mask_[0] == False  # noqa: E712


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

