# src/radar_core/domain/technical/momentum.py

# --- Third Party Libraries ---
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl
# TA-Lib: Python wrapper for TA-LIB based on Cython, for TA indicator calculations
#  Visit: https://github.com/ta-lib/ta-lib-python/   https://ta-lib.org/functions/
import talib


# noqa: N802 – instructs the linter (Ruff/Bandit) to ignore function name should be lowercase
def RSI(prices_df: pl.DataFrame,  # noqa: N802
        period: int = 14) -> pl.DataFrame:
    """
    Calculates the Relative Strength Index (RSI - Momentum Indicators) and add it as a new column to a Polars DataFrame.

    :param prices_df: Historical prices. It must include at least a column named "Close" representing closing prices.
    :param period: An integer representing the time period over which the indicator will be calculated.
    :return: A dataframe with the column 'Rsi' added to the input dataframe.
    """

    # Validate required column exists
    if "Close" not in prices_df.columns:
        raise ValueError('Missing required column: "Close"')

    # Calculate technical indicator to analyze
    # - talib.RSI()   : calculates technical indicator RSI
    # - to_list()     : converts the resulting Pandas Series into a list
    #                   (practical and efficient way to ensure seamless integration)
    # - pl.Series()   : converts the list back to a Polars Series
    # - with_columns(): adds the RSI Series as a new column in the Polars DataFrame.
    return prices_df.with_columns(pl.Series('Rsi',
                                            talib.RSI(prices_df['Close'].to_numpy(),
                                                      period)
                                            ).fill_nan(value=None))
