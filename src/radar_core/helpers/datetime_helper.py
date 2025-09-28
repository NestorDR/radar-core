# -*- coding: utf-8 -*-

# --- Python modules ---
# datetime: provides classes for simple and complex date and time manipulation.
from datetime import date, timedelta

# --- Third Party Libraries ---
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl

# --- App modules ---
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import DAILY, INTRADAY


def propose_start_dt(timeframe: int = DAILY,
                     long_term: bool = False,
                     use_tuning_days: bool = True,
                     randomize_from: bool = False) -> date:
    """
    Sets a start date for the acquisition (reading, capture, etc.) of historical prices.

    If the parameter long_term_ = is True, the start date to return will be:
        - Jan'01-2020, for intraday frame
        - Jan'01-1973, for daily frame (to include the minimum in the October 1974 prices)
    Otherwise, it will adopt today's date minus:
        - 1 year for intraday frame
        - 12 years for daily frame
    In the latter case with the possibility of tuning for AT and randomizing the resulting date.

    :param timeframe: Specifies daily or intraday frame.
    :param long_term: Specifies whether taking an old date.
    :param use_tuning_days: If the parameter long_term_ = False, flag to add tuning days for Technical Analysis
     indicators, which will backdate the initial date, and then perhaps discarded by a speculation strategy. In intraday
     frame adds 21 running days. For daily frame, it adds 280 running days. Both cases are equivalent to 200 price bars.
    :param randomize_from: Flag to request a small additional random time window

    :return: Default date from which historical prices will be acquired.
    """

    # Ensure a standard time frame
    if timeframe not in (DAILY, INTRADAY):
        raise ValueError(f"Invalid time frame: {timeframe}")

    if long_term:
        return date(2020, 1, 1) if timeframe == INTRADAY else date(1973, 1, 1)

    days_tuning_range_ = 0
    if timeframe == INTRADAY:
        # Window of days for prices acquisition: 1 year
        days_window_ = 365
        # Add 21 days (= 200 price bars) for fine-tuning of technical indicators
        if use_tuning_days:
            days_tuning_range_ = 21
    else:
        # Window of days for prices acquisition: 12 years
        # days_window_ = 365.25 * 12

        # Elapsed days since 1 March 2000, which was the peak of the bubble ".com"
        delta_ = date.today() - date(2000, 3, 1)
        days_window_ = delta_.days + 1

        if use_tuning_days:
            # Add 280 days (= 200 price bars) for fine-tuning of technical indicators
            days_tuning_range_ = 280

    if randomize_from:
        # Randomize start date
        import random
        random.seed()

        # noqa: S311 – instructs the linter (Ruff/Bandit) to ignore insecure use of the random module
        if timeframe == INTRADAY:
            # Add to the date to be returned a window between 1 and 20 days prior, at random
            days_random_prefix_ = random.randint(1, 20)  # noqa: S311
        else:
            # Add to the date to be returned a window between 1 and 6 months prior, at random
            days_random_prefix_ = random.randint(30, 182)  # noqa: S311
    else:
        # Do not add days at the beginning
        days_random_prefix_ = 0

    # Return start date
    return date.today() - timedelta(days=days_window_ + days_tuning_range_ + days_random_prefix_)


def to_weekly_timeframe(daily_df: pl.DataFrame,
                        column_name: str = 'Date') -> pl.DataFrame:
    """
    Group daily prices into weekly timeframes, using Polars.

    :param daily_df: DataFrame with [OHLCV%] prices on daily time frame.
    :param column_name: Name of the column containing the DateTime value.

    :return: Polars.DataFrame with [OHLCV%] prices on weekly time frame.
    """
    # Convert to a lazy frame for optimization
    lazy_df = daily_df.lazy()

    # Use group_by_dynamic for resampling weekly prices
    lazy_weekly_df = lazy_df.group_by_dynamic(
        index_column=column_name,
        every="1w",
        closed="left",  # Assume week starts on a Monday
    ).agg([
        pl.col('Open').first().alias("Open"),
        pl.col('High').max().alias("High"),
        pl.col('Low').min().alias("Low"),
        pl.col('Close').last().alias("Close"),
        pl.col('Volume').sum().alias("Volume"),
    ])

    # Calculate percentage change if 'PercentChange' exists
    if 'PercentChange' in daily_df.columns:
        lazy_weekly_df = lazy_weekly_df.with_columns(
            (pl.col("Close").pct_change() * 100).alias("PercentChange")
        )

    # Collect the result to a DataFrame and return
    return lazy_weekly_df.collect()


def monday_n_weeks_ago(weeks_ago: int) -> date:
    """
    Determines the date of the Monday that occurred weeks_ago_ weeks prior to the current date.

    :param weeks_ago: Number of weeks prior to the current date.

    :return: Date of the Monday that occurred weeks prior to the current date.
    """
    weeks_ago_date = date.today() - timedelta(weeks=weeks_ago)
    return weeks_ago_date - timedelta(days=weeks_ago_date.weekday())


def monday_n_workdays_ago(workdays_ago):
    """
    Determines the date of the Monday that occurred workdays_ago_ workdays prior to the current date.

    :param workdays_ago: Number of workdays prior to the current date.

    :return: Date of the Monday that occurred workdays prior to the current date.
    """
    # Calculate the number of full weeks (5 workdays per week), plus 3 preventive weeks
    weeks_ago_ = workdays_ago // 5 + 3
    return monday_n_weeks_ago(weeks_ago_)
