# src/radar_core/infrastructure/security_repository.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, ERROR, INFO, WARNING, getLogger

# --- Third Party Libraries ---
# yfinance: offers a threaded way to download market prices from Yahoo!Ⓡ Finance
# https://pypi.org/project/yfinance/
# https://snyk.io/advisor/python/yfinance/functions/yfinance.pdr_override
# https://aroussi.com/post/python=yahoo=finance
# https://github.com/ranaroussi/yfinance
import yfinance as yf

# --- App modules ---
# radar_core.helpers.log_helper: provides logging configuration and utilities.
from radar_core.helpers.log_helper import verbose
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import SecurityCrud
# models: result of Object-Relational Mapping
from radar_core.models import Securities

logger_ = getLogger(__name__)

# --- Integration constants.py ---
YAHOO_ID = 1


class SecurityRepository:
    """
    Manages persistence and retrieval of Security entities from the DB and external providers.
    """

    def __init__(self,
                 verbosity_level: int = DEBUG):
        """
        Initializes the SecurityRepository.

        :param verbosity_level: Minimum importance level of messages reporting the process progress.
        """
        self.__security_crud = SecurityCrud()
        self.verbosity_level = verbosity_level

    def __del__(self):
        """
        Destructor fallback to ensure resource release if close() was not called explicitly.
        """
        del self.__security_crud

    def _get_or_create_security(self, symbol: str) -> Securities | None:
        """
        Checks if the security symbol exists in the DB. If not, it fetches info from Yahoo Finance and creates the record.

        :param symbol: The security symbol to check/create (e.g., 'SPY').

        :return: The Security entity if it is found, otherwise None.
        """
        try:
            # Check if the security already exists
            security_ = self.__security_crud.get_by_symbol(symbol)
            if security_:
                return security_

            # Does not exist, fetch info and create it
            try:
                message_ = f'Security {symbol} not found in DB. Downloading info from Yahoo Finance...'
                logger_.info(message_)
                verbose(message_, DEBUG, self.verbosity_level)

                ticker_info_ = yf.Ticker(symbol).info
                company_name_ = ticker_info_.get('longName', 'Not found')
                # business_summary_ = ticker_info_.get('longBusinessSummary', 'Not found')

            except Exception as e:
                # Log error
                message_ = f'Error downloading information about {symbol} from Yahoo Finance.'
                verbose(message_, ERROR, self.verbosity_level)
                logger_.exception(message_, exc_info=e)
                company_name_ = 'Not found'

            if company_name_ == 'Not found':
                message_ = f"Security {symbol} not found in Yahoo Finance."
                verbose(message_, WARNING, self.verbosity_level)
                logger_.warning(message_)
                return None

            # Add new security
            self.__security_crud.add_security(
                Securities(
                    symbol=symbol,
                    description=company_name_,
                    is_bear=any(keyword in company_name_.lower() for keyword in ['bear', 'inverse', 'short'])))

            message_ = f"Added new security: {symbol} to the DB."
            verbose(message_, INFO, self.verbosity_level)
            logger_.info(message_)

            return security_

        except Exception as e_:
            message_ = f'Failed to ensure existence of security {symbol}: {e_}'
            verbose(message_, WARNING, self.verbosity_level)
            logger_.warning(message_)
            return None

    def _get_ticker(self, symbol: str, provider_id: int = YAHOO_ID) -> str:
        """
        Fetches the appropriate ticker (synonym) for the given symbol and provider.

        Parameters:
        :param symbol: Security symbol for which the ticker is to be retrieved.
        :param provider_id: The ID of the provider to fetch the synonym for. Defaults to YAHOO_ID.

        :returns: The corresponding ticker or the provided symbol if no synonym is found.
        """
        security_ = self._get_or_create_security(symbol)

        if security_:
            # If the synonym list has the symbol's synonym (ticker) in Yahoo, get it
            synonym_ = next((synonym for synonym in security_.synonyms if synonym.provider_id == provider_id), None)

            if not synonym_:
                # Search a synonym in the DB
                synonym_ = self.__security_crud.get_synonym(security_.id, provider_id)
                if synonym_:
                    security_.synonyms.append(synonym_)

            return synonym_.ticker if synonym_ else security_.symbol

        return symbol

    def map_symbol_to_ticker(self, symbols: list[str], provider_id: int = YAHOO_ID) -> dict[str, str]:
        """
        Translates a list of internal symbols to their corresponding provider tickers.

        :param symbols: List of security symbols to translate.
        :param provider_id: The ID of the provider to fetch the synonym for. Defaults to YAHOO_ID.

        :return: Dictionary mapping each valid original symbol to its provider ticker.
        """
        symbol_to_ticker_map_: dict[str, str] = {}
        message_ = f"Translating {len(symbols)} symbols to provider tickers..."
        verbose(message_, INFO, self.verbosity_level)
        logger_.info(message_)

        for symbol in symbols:
            if not symbol:
                message_ = "Empty symbol string provided in the list."
                verbose(message_, WARNING, self.verbosity_level)
                logger_.warning(message_)
                continue

            ticker_ = self._get_ticker(symbol, provider_id)
            if ticker_:
                symbol_to_ticker_map_[symbol] = ticker_

        return symbol_to_ticker_map_
