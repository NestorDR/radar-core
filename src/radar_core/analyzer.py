# src/radar_core/analyzer.py

# --- Python modules ---
# datetime: provides classes for simple and complex date and time manipulation.
from datetime import datetime
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
# os: allows access to functionalities dependent on the Operating System
import os
# sys: provides access to some variables used or maintained by the interpreter and to functions that interact strongly
#      with the interpreter.
import sys
# time: provides various time-related functions
import time

# --- Third Party Libraries ---
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl
# sqlalchemy: SQL and ORM toolkit for accessing relational databases - Import the specific exception
from sqlalchemy.exc import OperationalError

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies import MovingAverage, RsiRollerCoaster, RsiTwoBands, RsiStrategyABC
from radar_core.domain.strategies.constants import SMA, RSI_SMA
# technical: provides calculations of TA indicators
from radar_core.domain.technical import RSI
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import DAILY, WEEKLY, TIMEFRAMES, REQUIRED_PRICE_COLS
from radar_core.helpers.datetime_helper import to_weekly_timeframe
from radar_core.helpers.log_helper import DEFAULT_VERBOSITY_LEVEL, get_verbosity_level, setup_logger, end_logger, \
    verbose
# infrastructure: allows access to the own database and/or integration with external prices providers
from radar_core.infrastructure import price_provider
from radar_core.infrastructure.crud import RatioCrud
# Settings: has the configuration for the radar_core
from radar_core.settings import Settings


def clean(symbols: list[str],
          verbosity_level: int = DEBUG):
    """
    Deletes deprecated symbols from the database.
    """
    deleted_rows_ = 0

    with RatioCrud() as ratio_crud_:
        deleted_ratios = ratio_crud_.delete_symbols_not_in(symbols)
        deleted_rows_ += deleted_ratios

    if verbosity_level <= INFO:
        print(f'Cleaned {deleted_rows_} rows from the database for deprecated symbols.')


def valid_prices(timeframe: int,
                 symbol: str,
                 prices_df: pl.DataFrame | None,
                 verbosity_level: int = DEBUG) -> bool:
    """
    Determines whether the provided DataFrame contains valid price information.

    :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
    :param symbol: Security symbol to analyze strategies.
    :param prices_df: Dataframe with prices to validate.
    :param verbosity_level: Minimum importance level of messages reporting the progress of the process

    :return: True if the DataFrame meets all validation criteria; False otherwise.
    """

    result_ = (isinstance(prices_df, pl.DataFrame)
               and prices_df.height > 0
               and REQUIRED_PRICE_COLS.issubset(set(prices_df.columns)))

    if not result_:
        message = f"[{symbol}]: its {TIMEFRAMES[timeframe]} prices dataframe is not valid."
        verbose(message, WARNING, verbosity_level)
        logger_.warning(message)

    return result_


def analyze(timeframe: int,
            symbol: str,
            only_long_positions: bool,
            prices_df: pl.DataFrame,
            verbosity_level: int = DEBUG):
    """
    Analyze the prices dataframe for the specified timeframe.

    :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
    :param symbol: Security symbol to analyze strategies.
    :param only_long_positions: True if only long positions are evaluated, otherwise False.
    :param prices_df: Dataframe with prices to process.
    :param verbosity_level: Minimum importance level of messages reporting the progress of the process
    """

    # Show pricing frame information with prices to process
    if verbosity_level <= INFO:
        print(f'Starting the {TIMEFRAMES[timeframe]} time frame analysis for {symbol}...')
        if verbosity_level == DEBUG:
            print(prices_df.head(1))
        print(prices_df.tail(1))

    # Add a row counter as a column, required to the analysis (is zero-based bar number)
    prices_df = prices_df.with_columns(pl.arange(0, pl.len()).cast(pl.Int32).alias('BarNumber'))

    # Get profitable strategies for daily time frame (using global variables)
    close_prices_ = None
    p_ma_.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_, verbosity_level)
    # Calculate RSI only once for the following strategies
    prices_df = RSI(prices_df)
    p_rsi_ma_.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_, verbosity_level)
    # Extract Close prices to an array to speed up prices access
    close_prices_ = prices_df['Close'].to_numpy()
    # Calculate the stop loss prices only once for the following strategies
    bars_for_stop_loss_ = 10 if timeframe <= DAILY else 3
    prices_df = RsiStrategyABC.identify_where_to_stop_loss(prices_df, close_prices_, bars_for_stop_loss_)
    p_rsi_2b_.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_, verbosity_level)
    p_rsi_rc_.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_, verbosity_level)
    # Release memory
    del close_prices_


# Use of __name__ & __main__
# When the Python interpreter reads a code file, it completely executes the code in it.
# For example, in a file my_module.py, when executed as the main program, the __name__ attribute will be '__main__',
#  however, if it is called by importing it from another module: import my_module, the __name__ attribute will be
#  'my_module'
if __name__ == '__main__':
    # Set information about the start of the process
    init_dt_ = datetime.now()  # Identify the date and time when the process is started
    script_name_ = os.path.basename(__file__)
    verbosity_level_ = DEFAULT_VERBOSITY_LEVEL

    # Initialize logger to ensure that the exception handler can work even if logger setup fails.
    # It will be configured inside the try block.
    logger_ = None

    try:
        # Initialize application settings
        settings_ = Settings()

        # Initialize logging settings
        message_ = f'{script_name_.capitalize()} started at {init_dt_.strftime("%Y-%m-%d %H:%M:%S")}.'
        verbosity_level_ = get_verbosity_level()
        logger_ = setup_logger(verbosity_level_, str(script_name_))
        verbose(message_, INFO, verbosity_level_)
        logger_.info(message_)

        # Get configured symbols to analyze
        symbols_ = settings_.get_symbols()
        shortable_symbols_ = settings_.get_shortables()

        # Clean deprecated symbols in the database
        if symbols_:
            clean(settings_.get_undeletable(), verbosity_level_)

        # For a specific test
        # symbols_ = ['BTC-USD']

        if symbols_:
            # Instantiate strategies to analyze
            p_ma_ = MovingAverage(SMA, 'Close', 'Sma', verbosity_level=verbosity_level_)
            p_rsi_ma_ = MovingAverage(RSI_SMA, 'Rsi', 'RsiSma', verbosity_level=verbosity_level_)
            p_rsi_rc_ = RsiRollerCoaster(verbosity_level=verbosity_level_)
            p_rsi_2b_ = RsiTwoBands(verbosity_level=verbosity_level_)

            # Iterate over symbols
            for symbol_ in symbols_:
                symbol_started_at_ = time.monotonic()
                symbol_ = symbol_.upper()
                only_long_positions_ = symbol_ not in shortable_symbols_

                try:
                    # Get daily historical prices in a Pandas dataFrame.
                    prices_df_ = price_provider.get_daily_prices(symbol_, long_term=False,
                                                                 verbosity_level=verbosity_level_)

                    if valid_prices(DAILY, symbol_, prices_df_):
                        analyze(DAILY, symbol_, only_long_positions_, prices_df_, verbosity_level_)

                        # Prepare weekly prices dataframe
                        prices_df_ = to_weekly_timeframe(prices_df_)
                        if valid_prices(WEEKLY, symbol_, prices_df_):
                            print()
                            analyze(WEEKLY, symbol_, only_long_positions_, prices_df_, verbosity_level_)

                    symbol_elapsed_ = time.monotonic() - symbol_started_at_
                    message_ = f"[{symbol_}]: Analysis completed in {(symbol_elapsed_ / 60):.1f} min\n"
                    verbose(message_, INFO, verbosity_level_)

                except Exception as e:
                    message_ = f"[{symbol_}]: Error while analyzing prices due to error: {e}."
                    verbose(message_, ERROR, verbosity_level_)
                    logger_.exception(message_, exc_info=e)
        else:
            message_ = 'No available securities to analyze in the settings file'
            verbose(message_, WARNING, verbosity_level_)
            logger_.warning(message_)

        message_ = (init_dt_.strftime(f'{script_name_.capitalize()} - Started at %Y-%m-%d %H:%M:%S ...')
                    + datetime.now().strftime(' Ended at %Y-%m-%d %H:%M:%S')
                    + f' - Elapsed time {(datetime.now() - init_dt_).total_seconds() / 60:.1f} min')
        verbose(message_, INFO, verbosity_level_)
        logger_.info(message_)

        # Logger finalization
        end_logger(logger_)

        # Terminate normally
        sys.exit(0)

    except OperationalError as e:
        # Log the critical error using your application's logger
        message_ = f"CRITICAL ({script_name_} __main__): Database connection error. App terminating. Error: {e}"
        verbose(message_, CRITICAL, DEFAULT_VERBOSITY_LEVEL)
        if logger_:
            logger_.exception(message_, exc_info=e)

        sys.exit(1)  # Exit with a non-zero code to indicate failure

    except Exception as e:
        # Catch any other unexpected exceptions
        message_ = f"CRITICAL ({script_name_} __main__): An unexpected error occurred. App terminating. Error: {e}"
        verbose(message_, CRITICAL, DEFAULT_VERBOSITY_LEVEL)
        if logger_:
            logger_.exception(message_, exc_info=e)

        # Include traceback for unexpected errors
        import traceback

        traceback.print_exc()
        sys.exit(2)  # Different error code for unexpected errors
