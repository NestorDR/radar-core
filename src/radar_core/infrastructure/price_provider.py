# src/radar_core/infrastructure/price_provider.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, INFO, WARNING, getLogger
# os: allows access to functionalities dependent on the Operating System
import os

# --- Third Party Libraries ---
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl

# --- App modules ---
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import DAILY
from radar_core.helpers.datetime_helper import propose_start_dt
from radar_core.helpers.log_helper import verbose
# infrastructure: allows access to the own database and/or integration with external prices providers
from radar_core.infrastructure.integration import IntegrationDataAccess
from radar_core.infrastructure.crud import DailyDataCrud, SecurityCrud

logger_ = getLogger(__name__)


def get_daily_prices(symbol: str = '',
                     long_term: bool = False,
                     verbosity_level: int = DEBUG) -> pl.DataFrame | None:
    """
    Returns daily historical prices in a Polars.DataFrame.

    :param symbol: Security symbol to download prices.
    :param long_term: Specifies whether taking an old date.
    :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
     it will be taken into account only if it is greater than the level of detail specified for the entire class.

    :return: Polars.DataFrame formatted as [DateTime, Open, High, Low, Close, Volume, PercentChange] with index integer.
    """
    # Get security for the symbol using a context manager
    with SecurityCrud() as security_crud:
        security_ = security_crud.get_by_symbol(symbol)

    integration_data_access_ = IntegrationDataAccess(verbosity_level)
    if security_ is None:
        # Download company info from an Internet financial provider, to add new security to the database
        security_ = integration_data_access_.add_security(symbol, verbosity_level)

    if security_:
        # Get a standard start date
        from_dt_ = propose_start_dt(DAILY, long_term=long_term)

        if security_.store_locally:
            # Ensure the local database is updated completely
            integration_data_access_.check_update(security_, from_dt_, verbosity_level)

            # Get daily prices from the local database
            prices_df_ = DailyDataCrud().get_prices_by_security(security_.id, from_dt=from_dt_)
        else:
            # Get daily prices from external provider
            prices_df_ = integration_data_access_.get_daily_data(security=security_, from_dt=from_dt_,
                                                                 verbosity_level=verbosity_level)

    else:
        prices_df_ = None
        message_ = f"Security {symbol} not found in database or external provider."
        verbose(message_, WARNING, verbosity_level)
        logger_.warning(message_)

    # Release memory
    del integration_data_access_

    return prices_df_


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

    # Set symbol
    symbol_ = "NDQ"

    # Get daily historical prices
    data_ = get_daily_prices(symbol_, verbosity_level=INFO)

    if type(data_) is pl.DataFrame:
        print(symbol_)
        print(data_.head(5))
        print(data_.tail(5))

    # Logger finalization
    end_logging(logger_)

    # Terminate normally
    raise SystemExit(0)
