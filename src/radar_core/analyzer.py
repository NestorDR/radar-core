# src/radar_core/analyzer.py

# --- Python modules ---
# concurrent.futures: provides a high-level interface for asynchronously executing callables.
import concurrent.futures
# contextlib: provides utilities for working with context managers, including stream redirection.
from contextlib import redirect_stderr, redirect_stdout
# datetime: provides classes for manipulating dates and times.
from datetime import datetime
# io: implements the core facilities for file-like objects and I/O streams.
import io
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, config, getLogger
# os: allows access to functionalities dependent on the Operating System
import os
# time: provides various time-related functions
import time

# --- Third Party Libraries ---
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl
# sqlalchemy: SQL and ORM toolkit for accessing relational databases - Import the specific exception
from sqlalchemy.exc import OperationalError

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies import MovingAverage, RsiRollerCoaster, RsiTwoBands, RsiStrategyABC
from radar_core.domain.types import Strategies
# technical: provides calculations of TA indicators
from radar_core.domain.technical import RSI
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import DAILY, WEEKLY, TIMEFRAMES, REQUIRED_PRICE_COLS, RSI_SMA, SMA
from radar_core.helpers.datetime_helper import to_weekly_timeframe
from radar_core.helpers.log_helper import verbose
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.price_provider import PriceProvider
from radar_core.infrastructure.crud import RatioCrud
# settings: has the configuration for the radar_core
from radar_core.settings import Settings

logger_ = getLogger(__name__)


def clean(symbols: list[str],
          verbosity_level: int = DEBUG) -> None:
    """
    Deletes obsolete ratios from the database. Deletes ratios from symbols that are not in the list provided.

    :param symbols: List of symbols whose ratios will be maintained in the database.
    :param verbosity_level: Minimum importance level of messages reporting the progress of the process
    """
    with RatioCrud() as ratio_crud_:
        deleted_ratios = ratio_crud_.delete_symbols_not_in(symbols)

    verbose(f'Cleaned {deleted_ratios} rows from the database for deprecated symbols.', INFO, verbosity_level)


def init_worker(log_config: dict) -> None:
    """
    Initializes the logging configuration for the worker process, required for Windows & macOS (spawn).
    This ensures that logs from child processes are correctly handled and written to the log file.


    :param log_config: Dictionary with logging configuration from the main process.
    """
    if log_config:
        config.dictConfig(log_config)


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
        message_ = f"[{symbol}]: Its {TIMEFRAMES[timeframe]} prices dataframe is not valid."
        verbose(message_, WARNING, verbosity_level)
        logger_.warning(message_)

    return result_


def analyze(timeframe: int,
            symbol: str,
            only_long_positions: bool,
            prices_df: pl.DataFrame,
            strategies: Strategies,
            verbosity_level: int = DEBUG) -> None:
    """
    Analyze the prices dataframe for the specified timeframe.

    :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
    :param symbol: Security symbol to analyze strategies.
    :param only_long_positions: True if only long positions are evaluated, otherwise False.
    :param prices_df: Dataframe with prices to process.
    :param strategies: Pre-instantiated strategies container.
    :param verbosity_level: Minimum importance level of messages reporting the progress of the process
    """

    # Show pricing frame information with prices to process
    if verbosity_level <= INFO:
        print(
            f'\n[{symbol}]: {TIMEFRAMES[timeframe]} time frame analysis started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        if verbosity_level == DEBUG:
            print(prices_df.head(1))
        print(prices_df.tail(1))

    # Add a row counter as a column, required to the analysis (is a zero-based bar number)
    prices_df = prices_df.with_columns(pl.arange(0, pl.len(), eager=False).cast(pl.Int32).alias('BarNumber'))

    # Extract vectors only once per timeframe to maximize performance
    close_prices_ = prices_df['Close'].to_numpy()
    percent_changes_ = prices_df['PercentChange'].to_numpy()

    # Profitable SMAs identification
    if strategies.sma:
        strategies.sma.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_, percent_changes_, verbosity_level)

    # Profitable RSI-based identification
    if strategies.rsi_sma or strategies.rsi_rc or strategies.rsi_2b:
        # Calculate RSI only once for the RSI-based strategies
        prices_df = RSI(prices_df)

        if strategies.rsi_sma:
            strategies.rsi_sma.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_,
                                        percent_changes_, verbosity_level)

        # Calculate the stop loss prices only once for the following strategies
        if strategies.rsi_2b or strategies.rsi_rc:
            # Identify and calculate where to stop losses for both long and short positions.
            prices_df = RsiStrategyABC.identify_where_to_stop_loss(timeframe, prices_df, close_prices_)

            if strategies.rsi_2b:
                strategies.rsi_2b.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_,
                                           percent_changes_, verbosity_level)
            if strategies.rsi_rc:
                strategies.rsi_rc.identify(symbol, timeframe, only_long_positions, prices_df, close_prices_,
                                           percent_changes_, verbosity_level)

    # Release memory
    del close_prices_, percent_changes_


def process_symbol(symbol: str,
                   prices_df: pl.DataFrame,
                   strategies: Strategies,
                   shortable_symbols: list[str],
                   verbosity_level: int) -> str:
    """
    Worker function to analyze a single symbol.
    Captures activity to prevent interleaved logs.

    :param symbol: The symbol to analyze.
    :param prices_df: The price data for the symbol.
    :param strategies: The container with strategy instances.
    :param shortable_symbols: A list of symbols that can be shorted.
    :param verbosity_level: The logging verbosity level.

    :return: A string containing the captured activity logs.
    """
    symbol_started_at_ = time.monotonic()
    symbol_ = symbol.upper()

    # Log inside the child process (traceable)
    message_ = f'[{symbol}]: Launching parallel worker process at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}...'
    verbose(message_, INFO, verbosity_level)
    logger_.info(message_)

    only_long_positions_ = symbol_ not in shortable_symbols

    # Create an in-memory buffer to capture text output
    log_capture_buffer_ = io.StringIO()

    # Redirect both stdout (print/verbose) and stderr (some logs) to the buffer
    with redirect_stdout(log_capture_buffer_), redirect_stderr(log_capture_buffer_):
        try:
            # Strategy Analysis
            if valid_prices(DAILY, symbol_, prices_df, verbosity_level):
                analyze(DAILY, symbol_, only_long_positions_, prices_df, strategies, verbosity_level)

                # Prepare weekly prices dataframe
                prices_df_ = to_weekly_timeframe(prices_df)
                if valid_prices(WEEKLY, symbol_, prices_df_, verbosity_level):
                    analyze(WEEKLY, symbol_, only_long_positions_, prices_df_, strategies, verbosity_level)

            symbol_elapsed_ = time.monotonic() - symbol_started_at_
            message_ = f"[{symbol_}]: Analysis completed in {(symbol_elapsed_ / 60):.1f} min"
            verbose(message_ + "\n", INFO, verbosity_level)
            logger_.info(message_)

        except Exception as e:
            message_ = f"[{symbol_}]: Error while analyzing prices due to error: {e}."
            verbose(message_, ERROR, verbosity_level)
            logger_.exception(message_, exc_info=e)

        # Get the captured stdout value for returning, close the buffer (although GC handles it)
        captured_output_ = log_capture_buffer_.getvalue()
        log_capture_buffer_.close()

        return captured_output_


def analyzer(settings: Settings,
             symbols: list[str] | None = None) -> int:
    """
    Orchestrates the parallel analysis of financial symbols using various technical strategies.
    The analysis includes retrieving daily and weekly historical price data, validating the data,
     and applying multiple trading strategies.
    The function supports verbosity for logging and handles errors effectively to provide robust execution.

    Architecture:
        - Pandas is the ‘adapter’ needed to interact with yfinance.
        - Polars is the ‘efficient in-memory database’.
        - NumPy/Numba is the ‘high-speed calculator’.

    :param settings: Application settings object.
    :param symbols: A list of symbols (e.g., stock tickers) to analyze. Defaults to None,
     in which case the function retrieves symbols from the application settings.

    :return: An integer status code:
     - 0 if the process executes successfully without any errors.
     - 1 if there is a critical database connection error.
     - 2 if an unexpected error occurs.
    """

    # Set information about the start of the process
    init_dt_ = datetime.now()  # Identify the date and time when the process is started
    verbosity_level_ = settings.verbosity_level

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
            # Map configuration keys directly to strategy factory functions.
            # Key: Attribute name in Strategies class (and key in settings.yml)
            # Value: Factory lambda to create the instance
            strategy_map_ = {
                'sma': lambda: MovingAverage(SMA, 'Close', 'Sma', verbosity_level=verbosity_level_),
                'rsi_sma': lambda: MovingAverage(RSI_SMA, 'Rsi', 'RsiSma', verbosity_level=verbosity_level_),
                'rsi_rc': lambda: RsiRollerCoaster(verbosity_level=verbosity_level_),
                'rsi_2b': lambda: RsiTwoBands(verbosity_level=verbosity_level_),
            }

            # Build kwargs dynamically based on enabled strategies in settings.yml
            # get_evaluable_strategies() returns a list of strings matching the keys in strategy_map_
            # The strategy key in the map is the attribute name
            active_strategies_: dict = {strategy_key_: factory_() for strategy_key_, factory_ in strategy_map_.items() if
                                        strategy_key_ in settings.get_evaluable_strategies()}
            # Instantiate strategies container only with active strategies
            strategies_ = Strategies(**active_strategies_)

            # If no strategy is active, skip processing
            if not any(vars(strategies_).values()):
                message_ = "No active strategies configured to run."
                verbose(message_, WARNING, verbosity_level_)
                logger_.warning(message_)
                return 0

            # Download prices data for all symbols
            prices_data_ = PriceProvider(long_term=False).get_prices(symbols)

            # Determine the number of workers using the new property. os.cpu_count() will automatically use the available cores.
            num_workers_ = settings.max_workers
            if num_workers_ <= 0:
                num_workers_ = (os.cpu_count() or 2)

            # Use a ProcessPoolExecutor to analyze symbols in parallel
            message_ = f"Starting parallel analysis for {len(prices_data_)} symbols using {num_workers_} workers..."
            verbose(message_, INFO, verbosity_level_)
            logger_.info(message_)

            # Use initializer to configure logging once per worker process
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers_,
                    initializer=init_worker,
                    initargs=(settings.log_config,)
            ) as executor:
                # Use a set to store futures
                futures_ = []

                # Create a future for each symbol analysis task using destructive iteration to free memory in the main
                # process immediately. The items are popped from the dictionary one by one.
                # Once passed to executor.submit, the main process no longer needs the DataFrame reference.
                # ---
                # Update symbols to those whose prices have actually been downloaded
                # and keep the original order (FIFO: First-In, First-Out) by emptying the dictionary to free resources
                symbols = list(prices_data_.keys())
                for symbol_ in symbols:
                    # .pop(symbol_) returns the DataFrame and removes the entry from the dict immediately
                    prices_df_ = prices_data_.pop(symbol_)

                    # Submit the task to the Executor Pool
                    future_ = executor.submit(process_symbol,
                                              symbol_, prices_df_, strategies_, shortable_symbols_, verbosity_level_)
                    futures_.append(future_)

                    # Explicitly delete the local reference to the DataFrame to encourage GC
                    del prices_df_

                # Loop over every future to run its process. Wait for all futures to complete and process results
                for future_ in concurrent.futures.as_completed(futures_):
                    try:
                        captured_logs_ = future_.result()  # Get the captured logs string

                        if captured_logs_:
                            # Print the captured atomic block of logs to the console
                            # flush=True ensures immediate output in containerized environments (Docker)
                            print(captured_logs_, end='', flush=True)

                    except Exception as e:
                        # This will catch errors from within the process_symbol function
                        message_ = f"A task generated an exception: {e}"
                        verbose(message_, ERROR, verbosity_level_)
                        logger_.exception(message_, exc_info=e)
        else:
            message_ = 'No available securities to analyze in the settings file'
            verbose(message_, WARNING, verbosity_level_)
            logger_.warning(message_)

        message_ = (init_dt_.strftime('Analysis executed from %Y-%m-%d %H:%M:%S ')
                    + datetime.now().strftime('to %Y-%m-%d %H:%M:%S')
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
    # --- Python modules ---
    import logging.config
    # --- App modules ---
    from radar_core.helpers.log_helper import begin_logging, end_logging

    # Initialize app settings
    settings_ = Settings()
    # Logger initialisation
    logging.config.dictConfig(settings_.log_config)
    logger_ = getLogger(__name__)
    script_name_ = os.path.basename(__file__)
    begin_logging(logger_, script_name_, INFO)

    # Set symbols for a specific test
    symbols_ = ['SPY']

    #  Analyze strategies over historical prices
    exit_code = analyzer(settings_, symbols_)

    # Logger finalization
    end_logging(logger_)

    # Return exit code
    raise SystemExit(exit_code)
