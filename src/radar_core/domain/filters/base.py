# src/radar_core/domain/filters/base.py

# --- Python modules ---
from abc import ABC, abstractmethod

# --- Third Party Libraries ---
import numpy as np
import polars as pl


class FilterABC(ABC):
    """
    Abstract base class for technical and statistical entry-bar filters.
    Generates precomputed boolean eligibility masks for Long and Short trade setups.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique identifier for the filter.

        :return: String identifier for the filter.
        """
        pass

    @abstractmethod
    def get_masks(self, prices_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate Long and Short eligibility masks for all price bars.

        :param prices_df: A Polars DataFrame containing OHLCV price series.

        :return: Tuple of (long_mask, short_mask) as 1D boolean NumPy arrays.
        """
        pass
