# src/radar_core/domain/filters/price_action.py

# --- Python modules ---
from typing import Final

# --- Third Party Libraries ---
import numpy as np
import polars as pl

# --- App modules ---
from radar_core.domain.filters.base_filter import FilterABC

DEFAULT_LONG_THRESHOLD: Final[float] = 0.10
DEFAULT_SHORT_THRESHOLD: Final[float] = 0.95
DEFAULT_BEAR_LONG_THRESHOLD: Final[float] = 0.85


class PriceActionFilter(FilterABC):
    """
    Price-action candle confirmation filter based on Directional Retention Ratios.
    Evaluates candle direction and directional progress relative to previous close
    on the input bar.

    Asymmetric Market Model:
    - Regular assets (is_bear=False): Long setups remain unfiltered (None, zero-allocation baseline),
      while Short setups require a bearish candle (Close < Open) retaining >= short_threshold (0.95)
      of the downward drop.
    - Inverse ETFs / bear assets (is_bear=True): Long setups require a bullish candle (Close > Open)
      retaining >= bear_long_threshold (0.85) of the upward span, while Short setups remain unfiltered (None).
    """

    def __init__(
            self,
            long_threshold: float = DEFAULT_LONG_THRESHOLD,
            short_threshold: float = DEFAULT_SHORT_THRESHOLD,
            bear_long_threshold: float = DEFAULT_BEAR_LONG_THRESHOLD,
    ) -> None:
        """
        :param long_threshold: Minimum upward span retention for Long setups (defaults to DEFAULT_LONG_THRESHOLD).
        :param short_threshold: Minimum downward drop retention for Short setups (defaults to DEFAULT_SHORT_THRESHOLD).
        :param bear_long_threshold: Minimum upward span retention for bear asset Long setups (defaults to DEFAULT_BEAR_LONG_THRESHOLD).
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.bear_long_threshold = bear_long_threshold

    @property
    def name(self) -> str:
        """
        Return the filter identifier.

        :return: String identifier 'price_action'.
        """
        return 'price_action'

    def get_masks(
        self,
        prices_df: pl.DataFrame,
        is_bear: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Compute the directional price-action eligibility masks for Long and Short setups.

        :param prices_df: A Polars DataFrame containing Open, High, Low, and Close columns.
        :param is_bear: Whether the security is an inverse ETF (bear asset).

        :return: Tuple of (long_mask, short_mask) as 1D boolean NumPy arrays or None for unfiltered directions.
        """
        total_bars_ = prices_df.height
        if total_bars_ == 0:
            empty_mask_ = np.empty(0, dtype=np.bool_)
            return (empty_mask_, None) if is_bear else (None, empty_mask_)

        previous_close_expr_ = pl.col('Close').shift(1)

        if not is_bear:
            # Regular asset: Long trades remain unfiltered (None), only Short trades require candle confirmation
            short_span_expr_ = (previous_close_expr_ - pl.col('Low')).alias('Short_span')
            short_retention_expr_ = (
                (previous_close_expr_ - pl.col('Close')) / pl.col('Short_span')
            ).alias('short_retention')

            df_ = (
                prices_df
                .with_columns(previous_close_expr_.alias('Previous_close'))
                .with_columns(short_span_expr_)
                .with_columns(short_retention_expr_)
            )

            short_condition_expr_ = (
                (pl.col('Short_span') > 0.0)
                & (pl.col('Close') < pl.col('Open'))
                & (pl.col('short_retention') >= self.short_threshold)
                & pl.col('short_retention').is_not_null()
            ).fill_null(False)

            short_mask_ = df_.select(short_condition_expr_.alias('Short_eligible')).to_series().to_numpy().astype(np.bool_)
            return None, short_mask_

        # Inverse ETF / bear asset: Long trades require bullish candle confirmation, Short trades remain unfiltered (None)
        long_span_expr_ = (pl.col('High') - previous_close_expr_).alias('Long_span')
        long_retention_expr_ = (
            (pl.col('Close') - previous_close_expr_) / pl.col('Long_span')
        ).alias('Long_retention')

        df_ = (
            prices_df
            .with_columns(previous_close_expr_.alias('Previous_close'))
            .with_columns(long_span_expr_)
            .with_columns(long_retention_expr_)
        )

        long_condition_expr_ = (
            (pl.col('Long_span') > 0.0)
            & (pl.col('Close') > pl.col('Open'))
            & (pl.col('Long_retention') >= self.bear_long_threshold)
            & pl.col('Long_retention').is_not_null()
        ).fill_null(False)

        long_mask_ = df_.select(long_condition_expr_.alias('Long_eligible')).to_series().to_numpy().astype(np.bool_)
        return long_mask_, None

