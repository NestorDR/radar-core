# src/radar_core/domain/filters/sma_trend.py

# --- Python modules ---
from typing import Final

# --- Third Party Libraries ---
import numpy as np
import polars as pl

# --- App modules ---
from radar_core.domain.filters.base_filter import FilterABC

DEFAULT_SMA_PERIOD: Final[int] = 200
DEFAULT_SLOPE_PERIOD: Final[int] = 20


class SmaTrendFilter(FilterABC):
    """
    SMA trend and slope filter.
    Evaluates whether the closing price agrees with the SMA(200) position and whether
    the SMA's 20-bar slope confirms continuation.
    Long setups require Close > SMA(200) and SMA(200)[t] > SMA(200)[t-20].
    Short setups require Close < SMA(200) and SMA(200)[t] < SMA(200)[t-20].
    """

    def __init__(
            self,
            sma_period: int = DEFAULT_SMA_PERIOD,
            slope_period: int = DEFAULT_SLOPE_PERIOD,
    ) -> None:
        """
        :param sma_period: Lookback window for simple moving average (default 200).
        :param slope_period: Lag period to calculate slope delta (default 20).
        """
        self.sma_period = sma_period
        self.slope_period = slope_period

    @property
    def name(self) -> str:
        """
        Return the filter identifier.

        :return: String identifier 'sma_trend'.
        """
        return 'sma_trend'

    def get_masks(self, prices_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute directional trend and slope eligibility masks for Long and Short setups.

        :param prices_df: A Polars DataFrame containing the Close column.
        
        :return: Tuple of (long_mask, short_mask) as 1D boolean NumPy arrays.
        """
        total_bars_ = prices_df.height
        if total_bars_ == 0:
            empty_mask_ = np.empty(0, dtype=np.bool_)
            return empty_mask_, empty_mask_

        sma_expr_ = pl.col('Close').rolling_mean(
            window_size=self.sma_period,
            min_samples=self.sma_period,
        ).alias('sma')

        df_ = prices_df.with_columns(sma_expr_)

        slope_expr_ = (pl.col('sma') - pl.col('sma').shift(self.slope_period)).alias('sma_slope')
        df_ = df_.with_columns(slope_expr_)

        long_condition_expr_ = (
                (pl.col('Close') > pl.col('sma'))
                & (pl.col('sma_slope') > 0.0)
                & pl.col('sma').is_not_null()
                & pl.col('sma_slope').is_not_null()
        ).fill_null(False)

        short_condition_expr_ = (
                (pl.col('Close') < pl.col('sma'))
                & (pl.col('sma_slope') < 0.0)
                & pl.col('sma').is_not_null()
                & pl.col('sma_slope').is_not_null()
        ).fill_null(False)

        long_mask_series_ = df_.select(long_condition_expr_.alias('long_eligible')).to_series()
        short_mask_series_ = df_.select(short_condition_expr_.alias('short_eligible')).to_series()

        long_mask_ = long_mask_series_.to_numpy().astype(np.bool_)
        short_mask_ = short_mask_series_.to_numpy().astype(np.bool_)

        return long_mask_, short_mask_
