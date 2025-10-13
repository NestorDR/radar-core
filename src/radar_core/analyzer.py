# src/radar_core/analyzer.py

# --- Python modules ---
# datetime: provides classes for simple and complex date and time manipulation.
from datetime import datetime
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, getLogger
# os: allows access to functionalities dependent on the Operating System
import os
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
from radar_core.helpers.log_helper import get_verbosity_level, verbose
# infrastructure: allows access to the own database and/or integration with external prices providers
from radar_core.infrastructure import price_provider
from radar_core.infrastructure.crud import RatioCrud
# settings: has the configuration for the radar_core
from radar_core.settings import settings

logger_ = getLogger(__name__)


def clean(symbols: list[str],
          verbosity_level: int = DEBUG):
    """
    Deletes obsolete ratios from the database. Deletes ratios from symbols that are not in the list provided.

    :param symbols: List of symbols whose ratios will be maintained in the database.
    :param verbosity_level: Minimum importance level of messages reporting the progress of the process
    """
    with RatioCrud() as ratio_crud_:
        deleted_ratios = ratio_crud_.delete_symbols_not_in(symbols)

    if verbosity_level <= INFO:
        print(f'Cleaned {deleted_ratios} rows from the database for deprecated symbols.')


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
        message_ = f"[{symbol}]: its {TIMEFRAMES[timeframe]} prices dataframe is not valid."
        verbose(message_, WARNING, verbosity_level)
        logger_.warning(message_)

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


def analyzer(symbols: list[str] | None = None) -> int:
    """
    Analyzes financial symbols using various technical strategies. The analysis includes retrieving daily and weekly
     historical price data, validating the data, and applying multiple trading strategies.
    The function supports verbosity for logging and handles errors effectively to provide robust execution.

    :param symbols: A list of symbols (e.g., stock tickers) to analyze. Defaults to None,
     in which case the function retrieves symbols from the application settings.

    :return: An integer status code:
     - 0 if the process executes successfully without any errors.
     - 1 if there is a critical database connection error.
     - 2 if an unexpected error occurs.
    """

    # Set information about the start of the process
    init_dt_ = datetime.now()  # Identify the date and time when the process is started
    verbosity_level_ = get_verbosity_level()

    try:
        # Initialize logging settings
        message_ = f'Analysis started at {init_dt_.strftime("%Y-%m-%d %H:%M:%S")}.'
        verbose(message_, INFO, verbosity_level_)
        logger_.info(message_)

        # Get configured symbols to analyze
        if symbols is None:
            symbols = settings.get_symbols()
        shortable_symbols_ = settings.get_shortables()

        # Clean deprecated symbols in the database
        if symbols:
            clean(settings.get_undeletable(), verbosity_level_)

        if symbols:
            # Instantiate strategies to analyze
            # TODO 2025-10-12 NestorDR: Replace global variables with?
            global p_ma_, p_rsi_ma_, p_rsi_rc_, p_rsi_2b_
            p_ma_ = MovingAverage(SMA, 'Close', 'Sma', verbosity_level=verbosity_level_)
            p_rsi_ma_ = MovingAverage(RSI_SMA, 'Rsi', 'RsiSma', verbosity_level=verbosity_level_)
            p_rsi_rc_ = RsiRollerCoaster(verbosity_level=verbosity_level_)
            p_rsi_2b_ = RsiTwoBands(verbosity_level=verbosity_level_)

            # Iterate over symbols
            for symbol_ in symbols:
                symbol_started_at_ = time.monotonic()
                symbol_ = symbol_.upper()
                only_long_positions_ = symbol_ not in shortable_symbols_

                try:
                    # Get daily historical prices in a Pandas dataFrame.
                    prices_df_ = price_provider.get_daily_prices(symbol_, long_term=False,
                                                                 verbosity_level=verbosity_level_)

                    if valid_prices(DAILY, symbol_, prices_df_, verbosity_level_):
                        analyze(DAILY, symbol_, only_long_positions_, prices_df_, verbosity_level_)

                        # Prepare weekly prices dataframe
                        prices_df_ = to_weekly_timeframe(prices_df_)
                        if valid_prices(WEEKLY, symbol_, prices_df_, verbosity_level_):
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

        # Terminate normally
        return 0

    except OperationalError as e:
        # Log the critical error using your application's logger
        message_ = "Database connection error. CRITICAL app terminating."
        verbose(f"{message_} Error: {e}", CRITICAL, verbosity_level_)
        logger_.exception(message_, exc_info=e)

        return 1  # Exit code to indicate failure

    except Exception as e:
        # Catch any other unexpected exceptions
        message_ = "An unexpected error occurred. CRITICAL app terminating."
        verbose(f"{message_} Error: {e}", CRITICAL, verbosity_level_)
        logger_.exception(message_, exc_info=e)

        # Include traceback for unexpected errors
        import traceback

        traceback.print_exc()
        return 2  # Different error code for unexpected errors


# Use of __name__ & __main__
# When the Python interpreter reads a code file, it completely executes the code in it.
# For example, in a file my_module.py, when executed as the main program, the __name__ attribute will be '__main__',
#  however, if it is called by importing it from another module: import my_module, the __name__ attribute will be
#  'my_module'
if __name__ == "__main__":
    script_name_ = os.path.basename(__file__)

    # Logger initialisation
    import logging.config
    from radar_core.helpers.log_helper import get_logging_config, begin_logging, end_logging

    logging.config.dictConfig(get_logging_config(filename=str(script_name_)))
    logger_ = logging.getLogger()
    begin_logging(logger_, script_name_, INFO)

    # Set symbol for a specific test
    symbols_ = ['BTC-USD']

    #  Analyze strategies over historical prices
    exit_code = analyzer(symbols_)

    # Logger finalization
    end_logging(logger_)

    # Return exit code
    raise SystemExit(exit_code)
