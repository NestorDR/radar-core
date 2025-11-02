# src/radar_core/domain/technical/volatility.py

# --- Third Party Libraries ---
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl
# TA-Lib: Python wrapper for TA-LIB based on Cython, for TA indicator calculations
#  Visit: https://github.com/ta-lib/ta-lib-python/   https://ta-lib.org/functions/
import talib


# noqa: N802 – instructs the linter (Ruff/Bandit) to ignore function name should be lowercase
def ATR(prices_df: pl.DataFrame,  # noqa: N802
        period: int = 14) -> pl.DataFrame:
    """
    Calculates the Average True Range (ATR - Momentum Indicators) and add it as a new column to a Polars DataFrame.

    :param prices_df: Historical prices. It must include at least the columns named "High", "Low" and "Close"
    :param period: An integer representing the time period over which the indicator will be calculated.
    :return: A dataframe with the column 'ATR' added to the input dataframe.
    """
    # Validate required columns exist
    required_cols = ["High", "Low", "Close"]
    if not all(col in prices_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in prices_df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Calculate technical indicator to analyze
    # - talib.ATR()   : calculates technical indicator ATR
    # - to_list()     : converts the resulting Pandas Series into a list
    #                   (practical and efficient way to ensure seamless integration)
    # - pl.Series()   : converts the list back to a Polars Series
    # - with_columns(): adds the ATR Series as a new column in the Polars DataFrame.
    return prices_df.with_columns(pl.Series('Atr',
                                            talib.ATR(prices_df['High'].to_numpy(),
                                                      prices_df['Low'].to_numpy(),
                                                      prices_df['Close'].to_numpy(),
                                                      period)
                                            ).fill_nan(value=None))
