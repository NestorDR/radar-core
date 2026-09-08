# src/radar_core/domain/filters/atr_volatility.py

# --- Python modules ---
from typing import Final

# --- Third Party Libraries ---
import numpy as np
import polars as pl

# --- App modules ---
from radar_core.domain.filters.base import FilterABC
from radar_core.domain.technical import ATR

DEFAULT_ATR_PERIOD: Final[int] = 14
DEFAULT_LOOKBACK_BARS: Final[int] = 252
DEFAULT_LOWER_QUANTILE: Final[float] = 0.10
DEFAULT_UPPER_QUANTILE: Final[float] = 0.90


class AtrVolatilityFilter(FilterABC):
    """
    ATR-normalized volatility regime filter.
    Accepts entry signals only when the normalized ATR falls between the 10th and 90th percentiles
    of the previous 252 completed bars.
    """

    def __init__(
            self,
            atr_period: int = DEFAULT_ATR_PERIOD,
            lookback_bars: int = DEFAULT_LOOKBACK_BARS,
            lower_quantile: float = DEFAULT_LOWER_QUANTILE,
            upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    ) -> None:
        """
        :param atr_period: Lookback period for ATR calculation.
        :param lookback_bars: Window of preceding bars used for rolling quantile distribution.
        :param lower_quantile: Lower bound percentile (e.g. 0.10 for 10th percentile).
        :param upper_quantile: Upper bound percentile (e.g. 0.90 for 90th percentile).
        """
        self.atr_period = atr_period
        self.lookback_bars = lookback_bars
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    @property
    def name(self) -> str:
        """
        Return the filter identifier.

        :return: String identifier 'atr_volatility'.
        """
        return 'atr_volatility'

    def get_masks(self, prices_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the volatility regime eligibility mask for Long and Short setups.

        :param prices_df: A Polars DataFrame containing High, Low, and Close columns.
        
        :return: Tuple of (long_mask, short_mask) as 1D boolean NumPy arrays.
        """
        total_bars_ = prices_df.height
        if total_bars_ == 0:
            empty_mask_ = np.empty(0, dtype=np.bool_)
            return empty_mask_, empty_mask_

        # Use existing 'Atr' column if present, otherwise compute ATR
        df_ = prices_df if 'Atr' in prices_df.columns else ATR(prices_df, self.atr_period)

        # Compute normalized ATR (Atr / Close)
        natr_expr_ = (pl.col('Atr') / pl.col('Close')).alias('nATR')
        df_ = df_.with_columns(natr_expr_)

        # Shift by 1 bar to strictly evaluate prior completed history, excluding the current input bar
        prev_natr_expr_ = pl.col('nATR').shift(1)

        p10_expr_ = prev_natr_expr_.rolling_quantile(
            quantile=self.lower_quantile,
            window_size=self.lookback_bars,
            min_samples=self.lookback_bars,
        ).alias('p10')

        p90_expr_ = prev_natr_expr_.rolling_quantile(
            quantile=self.upper_quantile,
            window_size=self.lookback_bars,
            min_samples=self.lookback_bars,
        ).alias('p90')

        evaluated_df_ = df_.with_columns([p10_expr_, p90_expr_])

        # A bar is eligible when normalized ATR is between p10 and p90 and both bounds are valid
        condition_expr_ = (
                (pl.col('nATR') >= pl.col('p10'))
                & (pl.col('nATR') <= pl.col('p90'))
                & pl.col('nATR').is_not_null()
                & pl.col('p10').is_not_null()
                & pl.col('p90').is_not_null()
        ).fill_null(False)

        mask_series_ = evaluated_df_.select(condition_expr_.alias('is_eligible')).to_series()
        mask_ = mask_series_.to_numpy().astype(np.bool_)

        # ATR volatility regime is direction-neutral; Long and Short share identical eligibility
        return mask_, mask_
