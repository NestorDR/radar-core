# src/radar_core/infrastructure/price_provider.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
from datetime import date, timedelta
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, INFO, getLogger

# --- Third Party Libraries ---
# pandas: required by `yfinance`, provides powerful data structures and data analysis tools.
import pandas as pd
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl
# yfinance: offers a threaded way to download market prices from Yahoo!Ⓡ Finance.
import yfinance as yf

# --- App modules ---
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import DAILY, ORDERED_PRICE_COLS
from radar_core.helpers.datetime_helper import propose_start_dt
from radar_core.helpers.log_helper import verbose
# infrastructure: provides access to persisted data.
from radar_core.infrastructure.security_repository import SecurityRepository

logger_ = getLogger(__name__)


class PriceProvider:
    """
    Provides security price data from Yahoo Finance.
    """

    def __init__(self,
                 long_term: bool = False,
                 verbosity_level: int = DEBUG):
        """
        Initializes the PriceProvider for a specific time period.

        :param long_term: Specifies whether taking an old date.
        :param verbosity_level: Minimum importance level of messages reporting the process progress.
        """
        self.start_date = propose_start_dt(DAILY, long_term=long_term)
        self.end_date = date.today() + timedelta(days=1)
        self.verbosity_level = verbosity_level

    def _process_dataframe(self, symbol: str, prices_df: pd.DataFrame) -> pl.DataFrame:
        """
        Internal helper to convert a Pandas DataFrame into a clean Polars DataFrame.

        :param symbol: The security symbol the dataframe belongs to.
        :param prices_df: The raw pandas DataFrame to process.
        :return: A processed Polars DataFrame.
        """
        # Reset Date index as a column Date and convert the Pandas DataFrame to a Polars DataFrame
        prices_pl_df_ = pl.from_pandas(prices_df.reset_index())

        # Remove rows without "Close" price
        prices_pl_df_ = prices_pl_df_.filter(pl.col('Close').is_not_nan())

        # Round prices to 4 decimal places with Polars DataFrame
        prices_pl_df_ = prices_pl_df_.with_columns([
            pl.col('Date').cast(pl.Date).alias('Date'),
            pl.col('Open').round(4).alias('Open'),
            pl.col('High').round(4).alias('High'),
            pl.col('Low').round(4).alias('Low'),
            pl.col('Close').round(4).alias('Close')
        ])

        # To check if a DataFrame is empty, use the shape attribute and check if the row count is zero
        if not prices_pl_df_.height == 0:
            # Report the last Close price
            last_close_ = prices_pl_df_.select('Close').tail(1).to_numpy()[0][0]
            message_ = f'{symbol} - Last Close: ${last_close_:.2f}'
            verbose(message_, DEBUG, self.verbosity_level)
            logger_.info(message_)

        # Calculate percentage change
        prices_pl_df_ = prices_pl_df_.with_columns((pl.col("Close").pct_change() * 100).alias("PercentChange"))

        return prices_pl_df_[ORDERED_PRICE_COLS]

    def get_prices(self,
                   symbols: list[str],
                   max_workers: int = 10) -> dict[str, pl.DataFrame]:
        """
        Downloads historical prices for a list of symbols concurrently using yfinance's built-in capabilities.

        :param symbols: A list of security symbols to download (e.g., ['SPY', 'NDQ']).
        :param max_workers: The maximum number of threads yfinance should use for the concurrent downloads.

        :return: A dictionary mapping each symbol to its Polars DataFrame. Symbols with errors will be omitted.
        """
        if not symbols:
            logger_.warning("List of symbols empty.")
            return {}

        # Step 1: Translate internal symbols to provider tickers (currently only Yahoo Finance is supported)
        symbol_to_ticker_map_ = SecurityRepository(self.verbosity_level).map_symbol_to_ticker(symbols)
        tickers_ = list(symbol_to_ticker_map_.values())

        # Step 2: Download data using the translated tickers
        results_: dict[str, pl.DataFrame] = {}
        message_ = f"Starting download for {len(tickers_)} tickers from Yahoo Finance..."
        verbose(message_, INFO, self.verbosity_level)
        logger_.info(message_)

        try:
            # Visit: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
            #        https://github.com/pydata/pandas-datareader/issues/170
            #        https://pypi.org/project/yfinance/

            # Download prices from prices source with this parameter list:
            # tickers, start = None, end = None, actions = False, threads = True,
            # ignore_tz = None, group_by = 'column', auto_adjust = None, back_adjust = False,
            # repair = False, keepna = False, progress = True, period = None, interval = "1d",
            # prepost = False, proxy = _SENTINEL_, rounding = False, timeout = 10, session = None,
            # multi_level_index = True
            multi_symbol_df_ = yf.download(tickers_, self.start_date, self.end_date,
                                           auto_adjust=True, progress=bool(self.verbosity_level == DEBUG),
                                           threads=max_workers, group_by='ticker')

            if multi_symbol_df_.empty:
                logger_.warning("Download returned an empty DataFrame for all tickers.")
                return {}

            # Step 3: Process results, mapping tickers back to original symbols
            for symbol, ticker in symbol_to_ticker_map_.items():
                # For single ticker downloads, yfinance might not use multi-level columns unless group_by is used.
                # The current code handles both multi-level and single-level column structures.
                if ticker not in multi_symbol_df_.columns:
                    logger_.warning(f"No data downloaded for symbol: {symbol} (ticker: {ticker})")
                    continue

                symbol_df_ = multi_symbol_df_[ticker].dropna(how='all')

                if not symbol_df_.empty:
                    results_[symbol] = self._process_dataframe(symbol, symbol_df_)

        except Exception as e_:
            logger_.error(f'An exception occurred during download: {e_}', exc_info=True)

        logger_.log(self.verbosity_level,
                    f"Successfully processed data for {len(results_)} out of {len(symbols)} symbols.")
        return results_


# Use of __name__ & __main__
if __name__ == '__main__':
    # --- Python modules ---
    import os
    from datetime import datetime
    import logging.config
    # --- App modules ---
    from radar_core.settings import Settings
    from radar_core.helpers.log_helper import begin_logging, end_logging

    # Initialize app settings
    settings = Settings()
    # Logger initialisation
    script_name_ = os.path.basename(__file__)
    logging.config.dictConfig(settings.log_config)
    logger_ = getLogger(__name__)
    begin_logging(logger_, script_name_, INFO)

    price_provider_ = PriceProvider()

    # --- Test Case 1: Download a single symbol that requires translation ---
    print("--- Testing single download ---")
    test_symbol_ = 'NDQ'
    prices_data_ = price_provider_.get_prices([test_symbol_])
    if test_symbol_ in prices_data_:
        data_ = prices_data_[test_symbol_]
        print(f"{test_symbol_} - Shape: {data_.shape}")
        print(data_.head(2))
        print(data_.tail(2))

    # --- Test Case 2: Download multiple symbols ---
    print("\n--- Testing multiple symbols download ---")
    test_symbols_ = settings.get_symbols()
    init_dt_ = datetime.now()  # Identify the date and time when the process is started
    prices_data_ = price_provider_.get_prices(test_symbols_)
    end_dt_ = datetime.now()

    print("\nConcurrent download complete. Results:")
    for test_symbol_, data_ in prices_data_.items():
        print(f"{test_symbol_} - Shape: {data_.shape}")
        print(data_.tail(2))

    message_ = (init_dt_.strftime('Concurrent download executed from %Y-%m-%d %H:%M:%S ')
                + end_dt_.strftime('to %Y-%m-%d %H:%M:%S')
                + f' - Elapsed time {(end_dt_ - init_dt_).total_seconds() / 60:.1f} min')
    verbose(message_, INFO, settings.verbosity_level)
    logger_.info(message_)

    end_logging(logger_)
    raise SystemExit(0)
