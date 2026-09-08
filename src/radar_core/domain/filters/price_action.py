# src/radar_core/domain/filters/price_action.py

# --- Python modules ---
from typing import Final

# --- Third Party Libraries ---
import numpy as np
import polars as pl

# --- App modules ---
from radar_core.domain.filters.base import FilterABC

DEFAULT_LONG_THRESHOLD: Final[float] = 0.60
DEFAULT_SHORT_THRESHOLD: Final[float] = 0.25


class PriceActionFilter(FilterABC):
    """
    Price-action candle confirmation filter based on True Range.
    Evaluates candle direction and close location relative to the True Range
    (incorporating prior close and session gaps) on the input bar.
    Long setups require a bullish candle (Close > Open) with close location >= 0.60.
    Short setups require a bearish candle (Close < Open) with close location <= 0.25.
    """

    def __init__(
            self,
            long_threshold: float = DEFAULT_LONG_THRESHOLD,
            short_threshold: float = DEFAULT_SHORT_THRESHOLD,
    ) -> None:
        """
        :param long_threshold: Minimum close location for Long setups (default 0.60).
        :param short_threshold: Maximum close location for Short setups (default 0.25).
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    @property
    def name(self) -> str:
        """
        Return the filter identifier.

        :return: String identifier 'price_action'.
        """
        return 'price_action'

    def get_masks(self, prices_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the directional price-action eligibility masks for Long and Short setups.

        :param prices_df: A Polars DataFrame containing Open, High, Low, and Close columns.

        :return: Tuple of (long_mask, short_mask) as 1D boolean NumPy arrays.
        """
        total_bars_ = prices_df.height
        if total_bars_ == 0:
            empty_mask_ = np.empty(0, dtype=np.bool_)
            return empty_mask_, empty_mask_

        prior_close_expr_ = pl.col('Close').shift(1)
        true_high_expr_ = pl.max_horizontal(pl.col('High'), prior_close_expr_.fill_null(pl.col('High')))
        true_low_expr_ = pl.min_horizontal(pl.col('Low'), prior_close_expr_.fill_null(pl.col('Low')))
        true_range_expr_ = (true_high_expr_ - true_low_expr_).alias('true_range')
        close_location_expr_ = (
                (pl.col('Close') - true_low_expr_) / pl.col('true_range')
        ).alias('close_location')

        df_ = (
            prices_df
            .with_columns(true_high_expr_.alias('true_high'), true_low_expr_.alias('true_low'))
            .with_columns(true_range_expr_)
            .with_columns(close_location_expr_)
        )

        long_condition_expr_ = (
                (pl.col('true_range') > 0.0)
                & (pl.col('Close') > pl.col('Open'))
                & (pl.col('close_location') >= self.long_threshold)
                & pl.col('close_location').is_not_null()
        ).fill_null(False)

        short_condition_expr_ = (
                (pl.col('true_range') > 0.0)
                & (pl.col('Close') < pl.col('Open'))
                & (pl.col('close_location') <= self.short_threshold)
                & pl.col('close_location').is_not_null()
        ).fill_null(False)

        long_mask_series_ = df_.select(long_condition_expr_.alias('long_eligible')).to_series()
        short_mask_series_ = df_.select(short_condition_expr_.alias('short_eligible')).to_series()

        long_mask_ = long_mask_series_.to_numpy().astype(np.bool_)
        short_mask_ = short_mask_series_.to_numpy().astype(np.bool_)

        return long_mask_, short_mask_
