# src/radar_core/domain/filters/registry.py

# --- Python modules ---
from typing import Final

# --- Third Party Libraries ---
import numpy as np
import polars as pl

# --- App modules ---
from radar_core.domain.filters.atr_volatility import AtrVolatilityFilter
from radar_core.domain.filters.base import FilterABC
from radar_core.domain.filters.price_action import PriceActionFilter
from radar_core.domain.filters.sma_trend import SmaTrendFilter


FILTER_REGISTRY: Final[dict[str, type[FilterABC]]] = {
    'atr_volatility': AtrVolatilityFilter,
    'price_action': PriceActionFilter,
    'sma_trend': SmaTrendFilter,
}


def get_filter(filter_name: str) -> FilterABC:
    """
    Instantiate a registered filter by its name.

    :param filter_name: Registered identifier ('atr_volatility', 'price_action', 'sma_trend').

    :return: An instance of the corresponding FilterABC subclass.
    :raises KeyError: If the filter name is not registered.
    """
    normalized_name_ = filter_name.strip().lower()
    if normalized_name_ not in FILTER_REGISTRY:
        raise KeyError(f"Unknown filter '{filter_name}'. Available: {list(FILTER_REGISTRY.keys())}")
    return FILTER_REGISTRY[normalized_name_]()


def get_filter_masks(
    filter_name: str | None,
    prices_df: pl.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Directly obtain Long and Short eligibility masks for a named filter or baseline.

    :param filter_name: Registered filter name, or None/'none'/'baseline' for unfiltered baseline.
    :param prices_df: A Polars DataFrame containing OHLCV price series.

    :return: Tuple of (long_mask, short_mask) as 1D boolean NumPy arrays, or (None, None) for baseline.
    """
    if filter_name is None:
        return None, None

    normalized_name_ = filter_name.strip().lower()
    if normalized_name_ in ('', 'none', 'baseline'):
        return None, None

    filter_instance_ = get_filter(normalized_name_)
    return filter_instance_.get_masks(prices_df)
