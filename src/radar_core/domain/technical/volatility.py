# src/radar_core/domain/technical/volatility.py

# --- Third Party Libraries ---
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
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
    required_cols_ = ['High', 'Low', 'Close']
    if not all(col_ in prices_df.columns for col_ in required_cols_):
        missing_ = [col_ for col_ in required_cols_ if col_ not in prices_df.columns]
        raise ValueError(f'Missing required columns: {missing_}')

    # Calculate technical indicator to analyze
    # - talib.ATR()   : calculates technical indicator ATR
    # - pl.Series()   : converts the array to a Polars Series
    # - with_columns(): adds the ATR Series as a new column in the Polars DataFrame.
    return prices_df.with_columns(pl.Series('Atr',
                                            talib.ATR(prices_df['High'].to_numpy(),
                                                      prices_df['Low'].to_numpy(),
                                                      prices_df['Close'].to_numpy(),
                                                      period)
                                            ).fill_nan(value=None)
                                  )


# noqa: N802 – instructs the linter (Ruff/Bandit) to ignore function name should be lowercase
def MogalefBands(prices_df: pl.DataFrame,  # noqa: N802
                 period_reg: int = 3,
                 period_dev: int = 7,
                 multiplier: float = 2.0) -> pl.DataFrame:
    """
    Calculates Mogalef Bands (Central, Upper, and Lower volatility corridor bands)
     and adds them as new columns to a Polars DataFrame.

    Typical Weighted Price (CP): (Open + High + Low + 2 * Close) / 5
    Central Line: Linear regression of CP over period_reg
    Upper Band: Central + (multiplier * Standard Deviation of CP over period_dev)
    Lower Band: Central - (multiplier * Standard Deviation of CP over period_dev)

    :param prices_df: Historical prices. It must include at least ['Open', 'High', 'Low', 'Close'].
    :param period_reg: Lookback period for linear regression calculation of the central line.
    :param period_dev: Lookback period for standard deviation calculation of the bands.
    :param multiplier: Volatility multiplier applied to standard deviation for upper and lower bands.

    :return: A dataframe with columns 'MogalefCentral', 'MogalefUpper', and 'MogalefLower' added.
    """
    # Validate required columns exist
    required_cols_ = ['Open', 'High', 'Low', 'Close']
    if not all(col_ in prices_df.columns for col_ in required_cols_):
        missing_ = [col_ for col_ in required_cols_ if col_ not in prices_df.columns]
        raise ValueError(f'Missing required columns: {missing_}')

    # Validate parameters
    if period_reg < 1 or period_dev < 1:
        raise ValueError('Lookback periods must be greater than or equal to 1.')
    if multiplier < 0:
        raise ValueError('Multiplier must be non-negative.')

    # Short series guard: if DataFrame height is 0, add empty float columns
    if prices_df.height == 0:
        return prices_df.with_columns(
            [
                pl.Series('MogalefCentral', [], dtype=pl.Float64),
                pl.Series('MogalefUpper', [], dtype=pl.Float64),
                pl.Series('MogalefLower', [], dtype=pl.Float64),
            ]
        )

    # Calculate Weighted Typical Price: CP = (Open + High + Low + 2 * Close) / 5
    open_prices_ = prices_df['Open'].to_numpy().astype(np.float64)
    high_prices_ = prices_df['High'].to_numpy().astype(np.float64)
    low_prices_ = prices_df['Low'].to_numpy().astype(np.float64)
    close_prices_ = prices_df['Close'].to_numpy().astype(np.float64)

    typical_price_ = (open_prices_ + high_prices_ + low_prices_ + 2.0 * close_prices_) / 5.0

    # Calculate Linear Regression for Central Equilibrium Line
    central_line_ = talib.LINEARREG(typical_price_, period_reg)

    # Calculate Standard Deviation for Volatility Envelope
    std_dev_ = talib.STDDEV(typical_price_, period_dev, nbdev=1.0)

    # Compute Upper and Lower Bands
    upper_band_ = central_line_ + (multiplier * std_dev_)
    lower_band_ = central_line_ - (multiplier * std_dev_)

    # Add columns with null-filled NaNs
    return prices_df.with_columns(
        [
            pl.Series('MogalefCentral', central_line_).fill_nan(value=None),
            pl.Series('MogalefUpper', upper_band_).fill_nan(value=None),
            pl.Series('MogalefLower', lower_band_).fill_nan(value=None),
        ]
    )
