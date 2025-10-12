# src/radar_core/infrastructure/integration/integration_service.py

# --- Python modules ---
# datetime: provides classes for simple and complex date and time manipulation.
from datetime import date, timedelta
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, INFO, WARNING, getLogger
# os: allows access to functionalities dependent on the Operating System
import os
# threading: provides a higher-level interface for concurrent execution of different processes
import threading
# typing: provides runtime support for type hints
from typing import List

# --- Third Party Libraries ---
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl

# --- App modules ---
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import DAILY, NOT_FOUND, ORDERED_PRICE_COLS
from radar_core.helpers.datetime_helper import propose_start_dt
from radar_core.helpers.log_helper import verbose
# infrastructure: allows access to the own database and/or integration with external prices providers
from radar_core.infrastructure.crud import DailyDataCrud, SecurityCrud
# yahoo_service: downloads prices from Yahoo!Ⓡ Finance
from radar_core.infrastructure.integration.yahoo_service import YahooClient
# models: result of Object-Relational Mapping
from radar_core.models import Securities

logger_ = getLogger(__name__)

# --- Integration constants.py ---
YAHOO_ID = 1


# noinspection PyPackageRequirements
class IntegrationDataAccess(object):
    """
    Infrastructure to download prices from external providers
    """

    def __init__(self,
                 verbosity_level: int = DEBUG,
                 stop_event: threading.Event = None):
        """
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process for all
         methods of the class.
         Message levels to be reported: 0-discard messages, 1-report important messages, 2-report details.
        :param stop_event: Event to stop thread execution within multithreaded environment
        """
        self.verbosity_level = verbosity_level
        self.stop_event = stop_event
        self.provider_id = YAHOO_ID
        self.yahoo_client = YahooClient(self.verbosity_level, self.stop_event)

    def add_security(self,
                     symbol: str,
                     verbosity_level: int = DEBUG) -> Securities | None:
        """
        Download company info to add new security to the database.

        :param symbol: Security symbol to download prices.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: Created instance of Securities if the addition was successful, otherwise None
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        company_info_: tuple[str, str] = self.yahoo_client.get_company_info(symbol)

        if company_info_[0] != NOT_FOUND:
            # Instantiate prices access class
            security_crud_ = SecurityCrud()

            # Add new security
            security_ = Securities(symbol=symbol, description=company_info_[0])
            security_crud_.add_security(security_)

            # Release memory
            del security_crud_

            message_ = f"Added new security: {symbol} to the database."
            verbose(message_, INFO, verbosity_level)
            logger_.info(message_)

            return security_

        message_ = f"Security {symbol} not found in external provider."
        verbose(message_, WARNING, verbosity_level)
        logger_.warning(message_)

        return None

    def bulk_update_daily_data(self, symbols: List[str],
                               from_dt: date = None,
                               verbosity_level: int = DEBUG) -> None:
        """
        Update daily prices stored in the database from external providers for certain securities identified
         by their symbols/tickers.

        :param symbols: Security symbol list to update
        :param from_dt: Start date for downloading historical prices.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: None
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        try:
            security_crud_ = SecurityCrud()
            daily_data_crud_ = DailyDataCrud()

            for symbol_ in symbols:
                message_ = f"Starting with {symbol_}..."
                verbose(message_, INFO, verbosity_level)
                logger_.info(message_)

                # Identify security to update based on symbol
                security_ = security_crud_.get_by_symbol(symbol_, self.provider_id)
                if security_ is None:
                    message_ = f"Security {symbol_} not found in database."
                    verbose(message_, WARNING, verbosity_level)
                    logger_.warning(message_)

                    continue
                if not security_.store_locally:
                    message_ = f"Security {symbol_} not storable in database."
                    verbose(message_, INFO, verbosity_level)
                    logger_.info(message_)
                    continue

                prices_df_ = self.get_daily_data(security=security_, from_dt=from_dt)

                if type(prices_df_) is pl.DataFrame:
                    daily_data_crud_.upsert(security_.id, prices_df_)
        finally:
            del daily_data_crud_
            del security_crud_

    def check_update(self,
                     security: Securities,
                     alternative_start_dt: date,
                     verbosity_level: int = DEBUG) -> None:
        """
        Check if daily prices are updated, if not, update them

        :param security: Security to check and update prices.
        :param alternative_start_dt: Alternative start date for historical price checking and downloading.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class

        :return: None
        """
        daily_data_crud_ = DailyDataCrud()

        # Get prices for the most recent date in the DB
        latest_prices_ = daily_data_crud_.get_latest_prices_by_security(security.id)

        # Get prices from the external provider starting from last date minus seven days to ensure the best calculations
        from_dt_ = alternative_start_dt if latest_prices_ is None else latest_prices_.date - timedelta(days=7)
        prices_df_ = self.get_daily_data(security=security, from_dt=from_dt_, verbosity_level=verbosity_level)

        if latest_prices_ is not None and type(prices_df_) is pl.DataFrame:
            # Identify the newly downloaded prices for the last date saved in the database
            downloaded_prices_ = prices_df_.filter(pl.col('Date') == latest_prices_.date)

            if downloaded_prices_.height > 0:
                # Get the newly downloaded Open price
                downloaded_open_ = downloaded_prices_.select('Open').to_numpy()[0][0]
                # Get the Open price of the last available saved session
                saved_open_ = float(latest_prices_.open)  # Convert the Decimal type to float
                if abs(downloaded_open_ - saved_open_) > 0.0001:
                    # There is a different opening price at the external provider, perhaps for a split,
                    # a reversal split, a dividend payment, or something else.
                    # Download prices from the beginning of time to update the database completely.
                    prices_df_ = self.get_daily_data(security=security, from_dt=alternative_start_dt)

        # Update database
        if type(prices_df_) is pl.DataFrame:
            daily_data_crud_.upsert(security.id, prices_df_)

        del daily_data_crud_

    def get_daily_data(self, symbol: str = "",
                       security: Securities = None,
                       from_dt: date = None,
                       verbosity_level: int = DEBUG) -> pl.DataFrame | None:
        """
        Returns daily historical prices in a Polars DataFrame.
        January 2024: get daily prices from Yahoo Finance.

        :param symbol: Security symbol to download prices.
        :param security: Security to download prices.
        :param from_dt: Start date for downloading historical prices.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: Polars.DataFrame formatted as [Open, High, Low, Close, Volume, PercentChange] index datetime.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Identify and format ticker
        if symbol != "":
            ticker_ = symbol

        elif security:
            # If the synonym list has the symbol's synonym (ticker) in Yahoo, get it
            synonym_ = next((synonym for synonym in security.synonyms if synonym.provider_id == YAHOO_ID), None)
            if not synonym_:
                # Search a synonym in the database
                security_crud_ = SecurityCrud()
                synonym_ = security_crud_.get_synonym(security.id, self.provider_id)
                if synonym_ is not None:
                    security.synonyms.append(synonym_)
                del security_crud_

            ticker_ = security.symbol if not synonym_ else synonym_.ticker
        else:
            message_ = "Symbol or security not specified."
            verbose(message_, WARNING, verbosity_level)
            logger_.warning(message_)

            return None
        ticker_ = ticker_.upper()

        # Set the time window for downloading historical prices
        end_ = date.today() + timedelta(days=1)
        start_ = from_dt

        # Get daily historical prices from Yahoo! Finance
        prices_df_ = self.yahoo_client.get_daily_data(ticker_, start_, end_, verbosity_level)

        # Return historical prices
        if isinstance(prices_df_, pl.DataFrame):
            # Calculate percentage change
            # prices_df_['PercentChange'] = (prices_df_['Close'] / prices_df_['Close'].shift(1) - 1) * 100
            prices_df_ = prices_df_.with_columns((pl.col("Close").pct_change() * 100).alias("PercentChange"))

            if verbosity_level == DEBUG:
                print(prices_df_.tail(5))

            return prices_df_[ORDERED_PRICE_COLS]

        # Return no prices
        message_ = f"No prices found for {ticker_}."
        verbose(message_, WARNING, verbosity_level)
        logger_.warning(message_)

        return None

    @staticmethod
    def empty_df() -> pl.DataFrame:
        """
        :return: Empty polars.DataFrame with columns [Date, Open, High, Low, Close, Volume, PercentChange]
        """

        # Create an empty Polars DataFrame with specified columns
        return pl.DataFrame(schema={
            'Date': pl.Date,
            'Open': pl.Float64,
            'High': pl.Float64,
            'Low': pl.Float64,
            'Close': pl.Float64,
            'Volume': pl.Int64,
            'PercentChange': pl.Float64
        })


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

    # Set symbol and get its identifier
    security_symbol_ = "SOXX"
    security_instance_ = SecurityCrud().get_by_symbol(security_symbol_, YAHOO_ID)

    # Get a standard start date
    start_dt_ = propose_start_dt(DAILY)

    if security_instance_:
        # Download prices with security instance
        data_ = IntegrationDataAccess(INFO).get_daily_data(security=security_instance_, from_dt=start_dt_)
    else:
        # Download prices with ticker
        data_ = IntegrationDataAccess(INFO).get_daily_data(security_symbol_, from_dt=start_dt_)

    if type(data_) is pl.DataFrame:
        print(security_symbol_)
        print(data_.head(5))
        print(data_.tail(5))

    end_logging(logger_)

    # Terminate normally
    raise SystemExit(0)
