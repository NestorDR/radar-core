# src/radar_core/domain/technical/volatility.py

# --- Third Party Libraries ---
# numba: JIT compiler for numerical Python functions
from numba import njit
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl
# TA-Lib: Python wrapper for TA-LIB based on Cython, for TA indicator calculations
#  Visit: https://github.com/ta-lib/ta-lib-python/ and https://ta-lib.org/functions/
import talib


# noqa: N802 – instructs the linter (Ruff/Bandit) to ignore function name should be lowercase
def ATR(prices_df: pl.DataFrame,  # noqa: N802
        period: int = 14) -> pl.DataFrame:
    """
    Calculates the Average True Range (ATR - Momentum Indicators) and add it as a new column to a Polars DataFrame.

    :param prices_df: Historical prices. It must include at least the columns named "High", "Low" and "Close"
    :param period: An integer representing the time period over which the indicator will be calculated.

    :return: A dataframe with the column `ATR` added to the input dataframe.

    :raises ValueError: If an OHLC column is missing.
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


# In HPC (High Performance Computing), stateful numerical logic is isolated from
# dataframe orchestration so it can execute without Python interpreter overhead.
@njit(cache=True)
def _compute_stepped_mogalef_bands(
        linear_regression: np.ndarray,
        standard_deviation: np.ndarray,
        multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates stepped Mogalef upper and lower bands from indicator arrays.

    :param linear_regression: Rolling linear-regression values.
    :param standard_deviation: Rolling standard-deviation values for the regression series.
    :param multiplier: Non-negative standard-deviation multiplier.

    :return: Tuple containing the upper and lower stepped band arrays.
    """
    rows_ = len(linear_regression)
    upper_band_ = np.full(rows_, np.nan, dtype=np.float64)
    lower_band_ = np.full(rows_, np.nan, dtype=np.float64)

    initialized_ = False
    current_upper_level_ = np.nan
    current_lower_level_ = np.nan

    for row_ in range(rows_):
        linear_regression_value_ = linear_regression[row_]
        standard_deviation_value_ = standard_deviation[row_]

        if np.isnan(linear_regression_value_) or np.isnan(standard_deviation_value_):
            continue

        # Initialize on the first valid bar or reset after a corridor breakout.
        if (
                not initialized_
                or linear_regression_value_ > current_upper_level_
                or linear_regression_value_ < current_lower_level_
        ):
            current_upper_level_ = linear_regression_value_ + multiplier * standard_deviation_value_
            current_lower_level_ = linear_regression_value_ - multiplier * standard_deviation_value_
            initialized_ = True

        upper_band_[row_] = current_upper_level_
        lower_band_[row_] = current_lower_level_

    return upper_band_, lower_band_


def MogalefBands(  # noqa: N802
        prices_df: pl.DataFrame,
        period_reg: int = 3,
        period_dev: int = 7,
        multiplier: float = 2.0,
) -> pl.DataFrame:
    """
    Calculates standard Mogalef Bands (stepped volatility corridor bands)
     and adds them as new columns to a Polars DataFrame.

    Typical Weighted Price (CP): (Open + High + Low + 2 * Close) / 5
    Linear Regression: TA-Lib linear regression of CP over period_reg
    Standard Deviation: TA-Lib standard deviation of the linear regression line over period_dev
    Stepped Bands: Bands hold horizontal levels until the linear regression line breaks
     outside the corridor [Lower Band, Upper Band].

    :param prices_df: Historical prices. It must include at least ['Open', 'High', 'Low', 'Close'].
    :param period_reg: Lookback period for the linear regression line. Must be >= 1.
    :param period_dev: Lookback period for standard deviation of the regression line. Must be >= 1.
    :param multiplier: Non-negative standard-deviation multiplier for upper and lower bands.

    :return: The input DataFrame with 'MogalefUpper' and 'MogalefLower' columns added.

    :raises ValueError: If an OHLC column is missing, either lookback period is < 1, or multiplier is negative.
    """
    # Validate required columns exist
    required_cols_ = ["Open", "High", "Low", "Close"]
    if not all(col_ in prices_df.columns for col_ in required_cols_):
        missing_ = [
            col_ for col_ in required_cols_ if col_ not in prices_df.columns
        ]
        raise ValueError(f"Missing required columns: {missing_}")

    # Validate parameters
    if period_reg < 1 or period_dev < 1:
        raise ValueError("Lookback periods must be greater than or equal to 1.")
    if multiplier < 0:
        raise ValueError("Multiplier must be non-negative.")

    # Short series guard: if DataFrame height is 0, return empty columns
    if prices_df.height == 0:
        return prices_df.with_columns(
            [
                pl.Series("MogalefUpper", [], dtype=pl.Float64),
                pl.Series("MogalefLower", [], dtype=pl.Float64),
            ]
        )

    # Calculate Weighted Typical Price (Eric Lefort's classic formula): (Open + High + Low + 2 * Close) / 5
    open_prices_ = prices_df["Open"].to_numpy().astype(np.float64)
    high_prices_ = prices_df["High"].to_numpy().astype(np.float64)
    low_prices_ = prices_df["Low"].to_numpy().astype(np.float64)
    close_prices_ = prices_df["Close"].to_numpy().astype(np.float64)

    typical_price_ = (open_prices_ + high_prices_ + low_prices_ + 2.0 * close_prices_) / 5.0

    # Calculate Linear Regression for Central Equilibrium Line
    linear_regression_ = talib.LINEARREG(typical_price_, period_reg)

    # Standard Deviation of the Linear Regression Line
    std_deviation_ = talib.STDDEV(linear_regression_, period_dev, nbdev=1.0)

    # Calculate stepped bands in the array-only JIT kernel.
    upper_band_, lower_band_ = _compute_stepped_mogalef_bands(linear_regression_, std_deviation_, multiplier)

    # 5. Add columns with null-filled NaNs
    return prices_df.with_columns(
        [
            pl.Series("MogalefUpper", upper_band_).fill_nan(value=None),
            pl.Series("MogalefLower", lower_band_).fill_nan(value=None),
        ]
    )
