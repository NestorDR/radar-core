# src/radar_core/infrastructure/crud/security_crud.py

# --- Third Party Libraries ---
# psycopg: PostgreSQL database adapter for Python
from psycopg.sql import Identifier, SQL

# --- App modules ---
from radar_core.database import get_psycopg_connection
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import Securities, Synonyms

# Module-level table identifier constants
_SECURITIES_TABLE = Identifier(Securities.__tablename__)
_SYNONYMS_TABLE = Identifier(Synonyms.__tablename__)

_GET_SECURITY_BY_SYMBOL_SQL = SQL(
    "SELECT id, symbol, description, is_bear, store_locally FROM ") + _SECURITIES_TABLE + SQL(
    " WHERE symbol = %s")

_GET_SYNONYM_SQL = SQL("SELECT id, provider_id, security_id, ticker FROM ") + _SYNONYMS_TABLE + SQL(
    " WHERE security_id = %s AND provider_id = %s")

_GET_TICKERS_BY_SYMBOLS_SQL = SQL(
    "SELECT security.symbol, COALESCE(synonym.ticker, security.symbol) AS ticker FROM ") + _SECURITIES_TABLE + SQL(
    " AS security LEFT JOIN ") + _SYNONYMS_TABLE + SQL(
    " AS synonym ON synonym.security_id = security.id AND synonym.provider_id = %s WHERE security.symbol = ANY (%s)")

_ADD_SECURITY_SQL = SQL("INSERT INTO ") + _SECURITIES_TABLE + SQL(
    " (symbol, description, is_bear, store_locally) VALUES (%s, %s, %s, %s) RETURNING id")


class SecurityCrud(BaseCrud):
    def __init__(self):
        super().__init__(Securities)

    def get_by_symbol(self,
                      symbol: str,
                      provider_id: int | None = None) -> Securities | None:
        """
        Get security based on its symbol, optionally with a synonym ticker.

        :param symbol: Security symbol.
        :param provider_id: If it is present, then the synonym for that external provider will be appended.

        :return: Security instance or None.
        """
        with get_psycopg_connection() as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_GET_SECURITY_BY_SYMBOL_SQL, (symbol,))
                row_ = cur_.fetchone()
                if not row_:
                    return None

                security_ = Securities(
                    id=row_[0],
                    symbol=row_[1],
                    description=row_[2],
                    is_bear=row_[3],
                    store_locally=row_[4]
                )

                if provider_id:
                    synonym_ = self.get_synonym(security_.id, provider_id)
                    if synonym_:
                        security_.synonyms.append(synonym_)

                return security_

    @staticmethod
    def get_synonym(security_id: int, provider_id: int) -> Synonyms | None:
        """
        Get the ticker synonym for given security in a given financial prices provider

        :param security_id: Security id
        :param provider_id: Provider id

        :return: Synonyms instance or None
        """
        with get_psycopg_connection() as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_GET_SYNONYM_SQL, (security_id, provider_id))
                row_ = cur_.fetchone()
                if row_:
                    return Synonyms(id=row_[0], provider_id=row_[1], security_id=row_[2], ticker=row_[3])
                return None

    @staticmethod
    def get_tickers_by_symbols(symbols: list[str], provider_id: int) -> dict[str, str]:
        """
        Retrieves provider tickers for multiple security symbols in one query.
        Securities without a synonym for the requested provider use their
        internal symbol as the fallback ticker.

        :param symbols: Security symbols to retrieve.
        :param provider_id: Provider identifier.

        :return: Mapping from each existing symbol to its provider ticker.
        """
        if not symbols:
            return {}

        with get_psycopg_connection() as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_GET_TICKERS_BY_SYMBOLS_SQL, (provider_id, symbols))
                rows_ = cur_.fetchall()

        # Map symbols to tickers
        result_map_ = {
            symbol_: ticker_
            for symbol_, ticker_ in rows_
            if symbol_ and ticker_
        }

        # Sort the result map to preserve the order of input symbols
        return {
            symbol_: result_map_[symbol_]
            for symbol_ in symbols
            if symbol_ in result_map_
        }

    @staticmethod
    def add_security(security: Securities) -> None:
        """
        Persist a new Security instance to the database using psycopg3.

        :param security: Securities model instance to persist.
        """
        is_bear_ = security.is_bear if security.is_bear is not None else False
        store_locally_ = security.store_locally if security.store_locally is not None else False

        with get_psycopg_connection() as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_ADD_SECURITY_SQL, (security.symbol, security.description, is_bear_, store_locally_))
                row_ = cur_.fetchone()
                if row_:
                    security.id = row_[0]
