# src/radar_core/infrastructure/price_provider.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
from datetime import datetime, timedelta, timezone
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, ERROR, INFO, WARNING, getLogger
# typing: provides runtime support for type hints
from typing import Final
# uuid: generates Universally Unique Identifiers.
import uuid

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
from radar_core.infrastructure.price_cache import PriceCache, PriceCacheMetadata
from radar_core.infrastructure.security_repository import SecurityRepository
# settings: has the configuration for the radar_core
from radar_core.settings import get_settings

logger_ = getLogger(__name__)

# Centralized percentage change expression for DRY computation across pipelines
_PERCENT_CHANGE_EXPR: Final[pl.Expr] = (pl.col('Close').pct_change() * 100).alias('PercentChange')


class PriceProvider:
    """
    Provides security price data from Yahoo Finance.
    """

    def __init__(
            self,
            long_term: bool = False,
            verbosity_level: int = DEBUG
    ):
        """
        Initializes the PriceProvider for a specific time period.

        :param long_term: Specifies whether taking an old date.
        :param verbosity_level: Minimum importance level of messages reporting the process progress.
        """
        # Load application environment and price cache configuration
        settings_ = get_settings()
        self.app_environment = settings_.app_environment
        self.max_workers = settings_.max_workers
        self._cache_settings = settings_.price_cache_kwargs
        self._price_cache = PriceCache(self._cache_settings['dir'])
        self.start_date = propose_start_dt(DAILY, long_term=long_term)
        self.end_date = datetime.now(self._cache_settings['timezone']).date() + timedelta(days=1)
        self.verbosity_level = verbosity_level

    def _process_dataframe(
            self,
            symbol: str,
            prices_df: pd.DataFrame,
            verbosity_level: int = DEBUG
    ) -> pl.DataFrame:
        """
        Internal helper to convert a Pandas DataFrame into a clean Polars DataFrame.

        :param symbol: The security symbol the dataframe belongs to.
        :param prices_df: The raw pandas DataFrame to process.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: A processed Polars DataFrame.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Extract naive datetime64[ns] date array from index or column directly without intermediate DataFrame
        date_array_ = (
            prices_df['Date'].to_numpy(dtype='datetime64[ns]', copy=False)
            if 'Date' in prices_df.columns
            else prices_df.index.to_numpy(dtype='datetime64[ns]', copy=False)
        )

        # Extract raw NumPy arrays with to_numpy(copy=False) directly from the Series/Index without intermediate DataFrame
        raw_data_ = {
            'Date': date_array_,
            'Open': prices_df['Open'].to_numpy(dtype=float, copy=False),
            'High': prices_df['High'].to_numpy(dtype=float, copy=False),
            'Low': prices_df['Low'].to_numpy(dtype=float, copy=False),
            'Close': prices_df['Close'].to_numpy(dtype=float, copy=False),
            'Volume': prices_df['Volume'].to_numpy(copy=False),
        }

        # Convert to Polars directly, filter invalid rows, cast types, and compute percentage change
        # Note: rows without 'Close' prices are removed, and OHLC prices are rounded to 4 decimals
        prices_pl_df_ = (
            pl.DataFrame(raw_data_)
            .filter(pl.col('Close').is_not_nan() & pl.col('Close').is_not_null())
            .with_columns([
                pl.col('Date').cast(pl.Date),
                pl.col('Open').round(4),
                pl.col('High').round(4),
                pl.col('Low').round(4),
                pl.col('Close').round(4),
                pl.col('Volume').cast(pl.Int64, strict=False),
            ])
            .with_columns(_PERCENT_CHANGE_EXPR)
        )

        # To check if a DataFrame is empty, use the height attribute and check if the row count is zero
        if prices_pl_df_.height > 0:
            # Report the last Close price
            last_close_ = prices_pl_df_['Close'][-1]
            message_ = f'{symbol} - Last Close: ${last_close_:.2f}'
            verbose(message_, DEBUG, verbosity_level)
            logger_.info(message_)

        return prices_pl_df_.select(ORDERED_PRICE_COLS)

    def _save_cache(
            self,
            symbol_to_ticker_map: dict[str, str],
            results: dict[str, pl.DataFrame],
            session_date: str,
            verbosity_level: int = DEBUG
    ) -> None:
        """
        Saves the processed price data for all symbols to the price cache.

        :param symbol_to_ticker_map: Mapping of internal symbols to provider tickers.
        :param results: Dictionary mapping symbol to its processed Polars DataFrame.
        :param session_date: Local session date string (YYYY-MM-DD).
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Skip saving if cache is disabled or cache writes are disabled via RADAR_PRICE_CACHE_WRITE
        if not self._cache_settings['enabled'] or not self._cache_settings['write']:
            return

        try:
            # Exclude derived PercentChange from persistence and identify rows with their symbol
            combined_df_ = pl.concat([
                symbol_df_.drop('PercentChange').with_columns(pl.lit(symbol_).alias('Symbol'))
                for symbol_, symbol_df_ in results.items()
            ])
            # Record metadata with the provided local session date
            metadata_ = PriceCacheMetadata(
                symbol_to_ticker=symbol_to_ticker_map,
                start_date=str(self.start_date),
                session_date=session_date,
                generation_id=uuid.uuid4().hex,
                is_complete=True,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            # Atomically persist data and metadata to disk
            self._price_cache.save(combined_df_, metadata_)
            del combined_df_
            message_ = f'Price cache saved for {len(results)} symbols.'
            verbose(message_, INFO, verbosity_level)
            logger_.info(message_)

        except Exception as e_:
            message_ = f'Failed to save price cache: {e_}'
            verbose(message_, ERROR, verbosity_level)
            logger_.exception(message_)

    def _is_cache_eligible(
            self,
            symbol_to_ticker_map: dict[str, str],
            now: datetime,
            verbosity_level: int = DEBUG
    ) -> bool:
        """
        Checks if the local price cache is eligible for current-day refresh.

        :param symbol_to_ticker_map: Expected symbol-to-ticker mapping.
        :param now: Current datetime in market timezone for evaluation.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: True if eligible, False otherwise.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        if not self._cache_settings['enabled'] or self._cache_settings['ignore']:
            return False

        metadata_ = self._price_cache.read_metadata()
        if metadata_ is None:
            return False

        is_compatible_, reason_ = metadata_.is_compatible(
            symbol_to_ticker=symbol_to_ticker_map,
            start_date=str(self.start_date),
            session_date=str(now.date()),
        )
        if not is_compatible_:
            message_ = f'Price cache is not compatible: {reason_}.'
            verbose(message_, INFO, verbosity_level)
            logger_.info(message_)
            return False

        # Active market window: both environments refresh current-day bar
        is_trading_window_ = (now.weekday() < 5) and (
                self._cache_settings['window_start'] <= now.time() <= self._cache_settings['window_end']
        )
        if is_trading_window_:
            return True

        # Outside market window: dev mode avoids frequent downloads using dev_max_age_minutes TTL
        try:
            # Outside market window: allow reuse on the same session date if cache was generated after market close
            updated_at_market_tz_ = datetime.fromisoformat(metadata_.updated_at_utc).astimezone(self._cache_settings['timezone'])
            is_post_close_settled_ = (
                (now.weekday() < 5)
                and (now.time() > self._cache_settings['window_end'])
                and (metadata_.session_date == str(now.date()))
                and (updated_at_market_tz_.time() >= self._cache_settings['window_end'])
            )
            if is_post_close_settled_:
                return True

            # In development mode, also allow reuse within dev_max_age_minutes TTL (e.g. pre-market or weekends)
            if self.app_environment == 'dev':
                return 0 <= metadata_.age_in_minutes(now) <= self._cache_settings['dev_max_age_minutes']

            return False
        except (ValueError, TypeError) as exc_:
            logger_.warning(f'Failed to parse price cache timestamp: {exc_}. Bypassing cache.')
            return False


    def _refresh_cache(
            self,
            symbol_to_ticker_map: dict[str, str],
            tickers: list[str],
            now: datetime,
            verbosity_level: int = DEBUG
    ) -> dict[str, pl.DataFrame] | None:
        """
        Updates cached historical data by downloading only the current session's OHLCV bar from Yahoo Finance,
        replacing today's row, and updating percentage changes in memory without modifying the on-disk cache.

        :param symbol_to_ticker_map: Mapping of symbols to provider tickers.
        :param tickers: List of provider tickers.
        :param now: Current datetime in market timezone.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: A dictionary mapping symbol to updated Polars DataFrame, or None if refresh failed.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        cached_df_ = self._price_cache.read_data()
        if cached_df_ is None:
            message_ = 'Cannot refresh price cache: cached data unavailable.'
            verbose(message_, WARNING, verbosity_level)
            logger_.warning(message_)
            return None
        current_date_ = now.date()

        message_ = f'Refreshing price cache for {len(tickers)} tickers for session date {current_date_}...'
        verbose(message_, INFO, verbosity_level)
        logger_.info(message_)

        try:
            # Visit: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
            #        https://github.com/pydata/pandas-datareader/issues/170
            #        https://pypi.org/project/yfinance/

            # Download prices from price source with this parameter list:
            # tickers, start = None, end = None, actions = False, threads = True,
            # ignore_tz = None, group_by = 'column', auto_adjust = None, back_adjust = False,
            # repair = False, keepna = False, progress = True, period = None, interval = '1d',
            # prepost = False, proxy = _SENTINEL_, rounding = False, timeout = 10, session = None,
            # multi_level_index = True
            today_df_ = yf.download(tickers, current_date_, self.end_date,
                                    auto_adjust=True, progress=bool(verbosity_level == DEBUG),
                                    threads=self.max_workers, group_by='ticker')

            if today_df_.empty:
                message_ = 'Current-day download returned an empty DataFrame.'
                verbose(message_, WARNING, verbosity_level)
                logger_.warning(message_)
                return None

            # Partition cached data by Symbol for O(1) lookups and release original cache frame
            # Result is a dictionary mapping { 'symbol': DataFrame(Date, OHLCV), ... }
            empty_history_template_ = cached_df_.clear().drop('Symbol')
            history_by_symbol_ = {
                symbol_: partition_df_.drop('Symbol')
                for (symbol_,), partition_df_ in cached_df_.filter(pl.col('Date') != current_date_)
                .partition_by('Symbol', as_dict=True)
                .items()
            }
            del cached_df_

            results_: dict[str, pl.DataFrame] = {}
            for symbol_, ticker_ in symbol_to_ticker_map.items():
                if ticker_ not in today_df_.columns:
                    message_ = f'Ticker {ticker_} missing from current-day download.'
                    verbose(message_, WARNING, verbosity_level)
                    logger_.warning(message_)
                    return None

                ticker_df_ = today_df_[ticker_].dropna(how='all')
                if ticker_df_.empty:
                    message_ = f'No current-day data returned for {symbol_} (ticker: {ticker_}).'
                    verbose(message_, WARNING, verbosity_level)
                    logger_.warning(message_)
                    return None

                # Process today's row and combine with historical rows
                today_row_ = self._process_dataframe(symbol_, ticker_df_).drop('PercentChange')
                history_ = history_by_symbol_.get(symbol_, empty_history_template_)

                # Recompute PercentChange across the merged series
                results_[symbol_] = (
                    pl.concat([history_, today_row_], how='vertical_relaxed')
                    .sort('Date')
                    .with_columns(_PERCENT_CHANGE_EXPR)
                    .select(ORDERED_PRICE_COLS)
                )

            message_ = f'Price cache successfully refreshed in memory for {len(results_)} symbols.'
            verbose(message_, INFO, verbosity_level)
            logger_.info(message_)
            return results_

        except Exception as e_:
            message_ = f'Current-day refresh failed: {e_}'
            verbose(message_, ERROR, verbosity_level)
            logger_.exception(message_)

        return None

    def get_prices(
            self,
            symbols: list[str],
            now: datetime,
            verbosity_level: int = DEBUG
    ) -> dict[str, pl.DataFrame]:
        """
        Downloads historical prices for a list of symbols concurrently using yfinance built-in capabilities.

        :param symbols: A list of security symbols to download (e.g., ['SPY', 'NDQ']).
        :param now: Current evaluation datetime in or convertible to the configured market timezone.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: A dictionary mapping each symbol to its Polars DataFrame. Symbols with errors will be omitted.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        if not symbols:
            logger_.warning('List of symbols empty.')
            return {}

        tz_ = self._cache_settings['timezone']
        now_ = now.astimezone(tz_) if now.tzinfo is not None else now.replace(tzinfo=tz_)

        # Step 1: Translate internal symbols to provider tickers (currently only Yahoo Finance is supported)
        symbol_to_ticker_map_ = SecurityRepository(verbosity_level).map_symbol_to_ticker(symbols)
        tickers_ = list(symbol_to_ticker_map_.values())

        if not tickers_:
            logger_.warning('No valid tickers found to download.')
            return {}

        # Step 2: In either environment, refresh price cache if eligible
        if self._is_cache_eligible(symbol_to_ticker_map_, now_):
            message_ = 'Price cache is eligible. Attempting current-day refresh...'
            verbose(message_, INFO, verbosity_level)
            logger_.info(message_)
            refreshed_ = self._refresh_cache(symbol_to_ticker_map_, tickers_, now_, verbosity_level)
            if refreshed_ is not None:
                return refreshed_
            logger_.warning('Current-day refresh failed; falling back to full download.')

        # Step 3: Download data using the translated tickers
        results_: dict[str, pl.DataFrame] = {}
        message_ = f'Starting download for {len(tickers_)} tickers from Yahoo Finance...'
        verbose(message_, INFO, verbosity_level)
        logger_.info(message_)

        try:
            # Visit: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
            #        https://github.com/pydata/pandas-datareader/issues/170
            #        https://pypi.org/project/yfinance/

            # Download prices from price source with this parameter list:
            # tickers, start = None, end = None, actions = False, threads = True,
            # ignore_tz = None, group_by = 'column', auto_adjust = None, back_adjust = False,
            # repair = False, keepna = False, progress = True, period = None, interval = '1d',
            # prepost = False, proxy = _SENTINEL_, rounding = False, timeout = 10, session = None,
            # multi_level_index = True
            multi_symbol_df_ = yf.download(tickers_, self.start_date, self.end_date,
                                           auto_adjust=True, progress=bool(verbosity_level == DEBUG),
                                           threads=self.max_workers, group_by='ticker')

            if multi_symbol_df_.empty:
                logger_.warning('Download returned an empty DataFrame for all tickers.')
                return {}

            message_ = 'Download from Yahoo Finance completed.'
            verbose(message_, INFO, verbosity_level)
            logger_.info(message_)

            # Step 4: Process results, mapping tickers back to original symbols and converting Pandas to Polars
            for symbol_, ticker_ in symbol_to_ticker_map_.items():
                # For single ticker downloads, yfinance might not use multi-level columns unless group_by is used.
                # The current code handles both multi-level and single-level column structures.
                if ticker_ not in multi_symbol_df_.columns:
                    logger_.warning(f'No data downloaded for symbol: {symbol_} (ticker: {ticker_}).')
                    continue

                symbol_df_ = multi_symbol_df_[ticker_].dropna(how='all')

                if not symbol_df_.empty:
                    # Convert Pandas DataFrame into a Polars DataFrame.
                    results_[symbol_] = self._process_dataframe(symbol_, symbol_df_)

        except Exception as e_:
            logger_.exception(f'An exception occurred during download: {e_}', exc_info=True)

        message_ = f'Successfully downloaded and converted data for {len(results_)} of {len(symbols)} symbols.'
        verbose(message_, INFO, verbosity_level)
        logger_.info(message_)

        # Step 5: Save to price cache if all requested symbols were downloaded successfully
        if len(results_) == len(symbols):
            session_date_ = str(now_.date())
            self._save_cache(symbol_to_ticker_map_, results_, session_date_)

        return results_


# Use of __name__ & __main__
if __name__ == '__main__':
    # --- Python modules ---
    import logging.config
    import os
    from datetime import datetime

    # --- App modules ---
    from radar_core.helpers.log_helper import begin_logging, end_logging, rotate_log_at_startup
    from radar_core.settings import get_settings

    # Initialize app settings
    settings = get_settings()
    # Logger initialization
    logging.config.dictConfig(settings.log_config)
    rotate_log_at_startup()
    # Get root logger and log start messages
    logger_ = getLogger(__name__)
    script_name_ = os.path.basename(__file__)
    begin_logging(logger_, script_name_, INFO)

    price_provider_ = PriceProvider()

    # --- Test Case 1: Download a single symbol that requires translation ---
    print('--- Testing single download ---')
    test_symbol_ = 'QQQ'
    init_dt_ = datetime.now()  # Identify the date and time when the process is started
    prices_data_ = price_provider_.get_prices([test_symbol_], init_dt_)
    if test_symbol_ in prices_data_:
        data_ = prices_data_[test_symbol_]
        print(f'{test_symbol_} - Shape: {data_.shape}')
        print(data_.head(2))
        print(data_.tail(2))

    # --- Test Case 2: Download multiple symbols ---
    print('\n--- Testing multiple symbols download ---')
    test_symbols_ = settings.symbols
    prices_data_ = price_provider_.get_prices(test_symbols_, init_dt_)
    end_dt_ = datetime.now()

    print('\nConcurrent download complete. Results:')
    for test_symbol_, data_ in prices_data_.items():
        print(f'{test_symbol_} - Shape: {data_.shape}')
        print(data_.tail(2))

    message = (init_dt_.strftime('Concurrent download executed from %Y-%m-%d %H:%M:%S ')
               + end_dt_.strftime('to %Y-%m-%d %H:%M:%S')
               + f' - Elapsed time {(end_dt_ - init_dt_).total_seconds() / 60:.1f} min')
    verbose(message, INFO, settings.verbosity_level)
    logger_.info(message)

    end_logging(logger_)
    raise SystemExit(0)
