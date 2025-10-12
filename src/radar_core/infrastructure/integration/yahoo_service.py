# src/radar_core/infrastructure/integration/yahoo_service.py

# --- Python modules ---
# datetime: provides classes for simple and complex date and time manipulation.
from datetime import date, timedelta
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import CRITICAL, DEBUG, ERROR, INFO, getLogger
# os: allows access to functionalities dependent on the Operating System
import os
# threading: provides a higher-level interface for concurrent execution of different processes
import threading
# time: provides various time-related functions
import time

# --- Third Party Libraries ---
# pandas: required by `yfinance`, provides powerful data structures and data analysis tools.
import pandas as pd
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl
# yfinance: offers a threaded way to download market prices from Yahoo!Ⓡ Finance
# https://pypi.org/project/yfinance/
# https://snyk.io/advisor/python/yfinance/functions/yfinance.pdr_override
# https://aroussi.com/post/python=yahoo=finance
# https://github.com/ranaroussi/yfinance
import yfinance as yf

# --- App modules ---
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import DEFAULT_ATTEMPT_LIMIT, DEFAULT_SLEEP, NOT_FOUND
from radar_core.helpers.log_helper import verbose

logger_ = getLogger(__name__)


class YahooClient(object):
    """
    Infrastructure to download prices from Yahoo!Ⓡ Finance
    """

    def __init__(self,
                 verbosity_level: int = DEBUG,
                 stop_event: threading.Event = None):
        """
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process,
         for all methods of the class.
         Message levels to be reported: 0-discard messages, 1-report important messages, 2-report details.
        :param stop_event: Event to stop thread execution within multithreaded environment
        """
        self.stop_event = stop_event
        self.verbosity_level = verbosity_level

    def get_company_info(self,
                         ticker: str,
                         verbosity_level: int = DEBUG) -> tuple[str, str]:
        """
        Get company info from Yahoo!

        :param ticker: Security symbol to download prices.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: A tuple with [company name, business summary].
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)
        message_ = f"Requesting information from Yahoo! Finance about {ticker}..."
        verbose(message_, DEBUG, verbosity_level)
        logger_.debug(message_)

        company_name_ = business_summary_ = NOT_FOUND
        try:
            ticker_info_ = yf.Ticker(ticker).info

            company_name_ = ticker_info_.get('longName', NOT_FOUND)
            business_summary_ = ticker_info_.get('longBusinessSummary', NOT_FOUND)

            return company_name_, business_summary_

        except Exception as e:
            # Log error
            message_ = f'Error downloading information about {ticker}.'
            verbose(message_, ERROR, verbosity_level)
            logger_.exception(message_, exc_info=e)

        return company_name_, business_summary_

    def get_daily_data(self,
                       ticker: str,
                       start: date,
                       end: date,
                       verbosity_level: int = DEBUG) -> pl.DataFrame | None:
        """
        Gets and returns daily historical prices in a Polars DataFrame.

        :param ticker: Security ticker to download prices.
        :param start: Start date of the time-window for downloading historical prices.
        :param end: End date of the time-window for downloading historical prices.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: Polars.DataFrame formatted as [Date, Open, High, Low, Close, Volume] index datetime.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        message_ = (f"Downloading from Yahoo! Finance daily prices for {ticker} from {start.strftime('%Y-%m-%d')}"
                    f" to {end.strftime('%Y-%m-%d')}...")
        verbose(message_, INFO, verbosity_level)
        logger_.info(message_)

        # Initialize results
        prices_df_ = None

        # Repeat reading/downloading prices from the price provider until successful or reaching the limit
        attempts_ = 0
        while attempts_ < DEFAULT_ATTEMPT_LIMIT:
            try:
                # Visit: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
                #        https://github.com/pydata/pandas-datareader/issues/170
                #        https://pypi.org/project/yfinance/

                # Download prices from prices source with this parameter list:
                #  tickers, start=None, end=None, actions=False, threads=True, ignore_tz=None,
                #  group_by='column', auto_adjust=False, back_adjust=False, repair=False, keepna=False,
                #  progress=True, period="max", interval="1d", prepost=False,
                #  proxy=None, rounding=False, timeout=10, session=None
                prices_df_ = yf.download(ticker, start, end, auto_adjust=True,
                                         progress=bool(verbosity_level == DEBUG), multi_level_index=False)

                if (isinstance(prices_df_, pd.DataFrame)
                        and not prices_df_.empty
                        and {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(set(prices_df_.columns))):
                    # Successful
                    break

                attempts_ += 1

            except Exception as e:
                attempts_ += 1
                if attempts_ == DEFAULT_ATTEMPT_LIMIT:
                    # Log error
                    message_ = f'Error downloading historical prices for {ticker}.'
                    verbose(message_, CRITICAL, verbosity_level)
                    logger_.exception(message_, exc_info=e)

                    # Return df_prices = None
                    return None
                # Wait a few seconds before retrying
                if self.stop_event is None:
                    time.sleep(DEFAULT_SLEEP)
                else:
                    self.stop_event.wait(DEFAULT_SLEEP)

        # Reset Date index as a column to prepare for conversion to Polars
        prices_df_ = prices_df_.reset_index()

        # Convert the Pandas DataFrame to a Polars DataFrame
        prices_df_ = pl.from_pandas(prices_df_)

        # Remove rows without "Close" price
        prices_df_ = prices_df_.filter(pl.col("Close").is_not_nan())

        # Round prices to 4 decimal places with Polars DataFrame
        prices_df_ = prices_df_.with_columns([
            pl.col('Date').cast(pl.Date).alias("Date"),
            pl.col('Open').round(4).alias('Open'),
            pl.col('High').round(4).alias('High'),
            pl.col('Low').round(4).alias('Low'),
            pl.col('Close').round(4).alias('Close')
        ])

        # To check if a DataFrame is empty, use the shape attribute and check if the row count is zero
        if not prices_df_.height == 0:
            # Report the last Close price
            last_close_ = prices_df_.select('Close').tail(1).to_numpy()[0][0]
            message_ = f'Last close: ${last_close_:.2f}'
            verbose(message_, DEBUG, verbosity_level)
            logger_.info(message_)

        # Return historical prices
        return prices_df_[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


# Use of __name__ & __main__
# When the Python interpreter reads a code file, it completely executes the code in it.
# For example, in a file my_module.py, when executed as the main program, the __name__ attribute will be '__main__',
#  however, if it is called by importing it from another module: import my_module, the __name__ attribute will be
#  'my_module'
if __name__ == '__main__':
    script_name_ = os.path.basename(__file__)

    # Logger initialisation
    import logging.config
    from radar_core.helpers.log_helper import get_logging_config, begin_logging, end_logging
    logging.config.dictConfig(get_logging_config(filename=str(script_name_)))
    logger_ = logging.getLogger()
    begin_logging(logger_, script_name_, INFO)

    # Set ticker
    ticker_ = '^GSPC'

    # Set the time window for downloading historical prices
    to_dt_ = date.today()
    from_dt_ = to_dt_ - timedelta(days=365)

    # Download prices
    yahoo_client_ = YahooClient()
    company_info_ = yahoo_client_.get_company_info(ticker_)
    data = yahoo_client_.get_daily_data(ticker_, from_dt_, to_dt_)
    if type(data) is pl.DataFrame:
        print(f"{ticker_} - {company_info_[0]}")
        print(f"{company_info_[1]}")
        print(data.head(5))
        print(data.tail(5))

    end_logging(logger_)

    # Terminate normally
    raise SystemExit(0)
