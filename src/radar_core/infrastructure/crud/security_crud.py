# src/radar_core/infrastructure/crud/security_crud.py

# --- Python modules ---
# typing: provides runtime support for type hints
from typing import Final

# --- Third Party Libraries ---
# psycopg: PostgreSQL database adapter
from psycopg import Connection
from psycopg.sql import Composed, Identifier, SQL

# --- App modules ---
# database: provides access to database connections
from radar_core.database import connection_scope, read_connection_scope
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import SECURITIES_COLUMNS, SECURITIES_PAYLOAD_COLUMNS, Securities, Synonyms

# Module-level table identifier constants
_SECURITIES_TABLE = Identifier(Securities.__tablename__)
_SYNONYMS_TABLE = Identifier(Synonyms.__tablename__)

# SQL statements
_GET_SECURITY_BY_SYMBOL_SQL: Final[Composed] = (
    SQL("SELECT ") +
    SQL(', ').join(map(Identifier, SECURITIES_COLUMNS)) +
    SQL(" FROM ") + _SECURITIES_TABLE + SQL(" WHERE symbol = %s")
)

_GET_SHORTABLE_SYMBOLS_SQL: Final[Composed] = SQL(
    "SELECT symbol FROM "
) + _SECURITIES_TABLE + SQL(" WHERE is_shortable = TRUE AND symbol = ANY (%s)")

_GET_SYNONYM_SQL: Final[Composed] = SQL("SELECT id, provider_id, security_id, ticker FROM ") + _SYNONYMS_TABLE + SQL(
    " WHERE security_id = %s AND provider_id = %s")

_GET_TICKERS_BY_SYMBOLS_SQL: Final[Composed] = SQL(
    "SELECT security.symbol, COALESCE(synonym.ticker, security.symbol) AS ticker FROM ") + _SECURITIES_TABLE + SQL(
    " AS security LEFT JOIN ") + _SYNONYMS_TABLE + SQL(
    " AS synonym ON synonym.security_id = security.id AND synonym.provider_id = %s WHERE security.symbol = ANY (%s)")

_ADD_SECURITY_SQL: Final[Composed] = (
    SQL("INSERT INTO ") + _SECURITIES_TABLE + SQL(" (") +
    SQL(', ').join(map(Identifier, SECURITIES_PAYLOAD_COLUMNS)) +
    SQL(") VALUES (") +
    SQL(', ').join([SQL('%s')] * len(SECURITIES_PAYLOAD_COLUMNS)) +
    SQL(") RETURNING id")
)


class SecurityCrud(BaseCrud):
    def __init__(self):
        super().__init__(Securities)

    def get_by_symbol(self,
                      symbol: str,
                      provider_id: int | None = None,
                      conn: Connection | None = None) -> Securities | None:

        """
        Get security based on its symbol, optionally with a synonym ticker.

        :param symbol: Security symbol.
        :param provider_id: If it is present, then the synonym for that external provider will be appended.
        :param conn: Optional active autocommit read connection to reuse.

        :return: Security instance or None.
        """
        with read_connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_GET_SECURITY_BY_SYMBOL_SQL, (symbol,))
                security_row_ = cur_.fetchone()
                if not security_row_:
                    return None

                security_ = Securities(**dict(zip(SECURITIES_COLUMNS, security_row_, strict=True)))

                if provider_id:
                    synonym_ = self.get_synonym(security_.id, provider_id, conn=conn_)
                    if synonym_:
                        security_.synonyms.append(synonym_)

                return security_

    @staticmethod
    def get_synonym(security_id: int,
                    provider_id: int,
                    conn: Connection | None = None) -> Synonyms | None:
        """
        Get the ticker synonym for a given security and financial prices provider.

        :param security_id: Security id.
        :param provider_id: Provider id.
        :param conn: Optional active autocommit read connection to reuse.

        :return: A Synonyms instance or None if not found.
        """
        with read_connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_GET_SYNONYM_SQL, (security_id, provider_id))
                row_ = cur_.fetchone()
                if row_:
                    return Synonyms(
                        id=row_[0],
                        provider_id=row_[1],
                        security_id=row_[2],
                        ticker=row_[3]
                    )
                return None

    @staticmethod
    def get_tickers_by_symbols(symbols: list[str],
                               provider_id: int,
                               conn: Connection | None = None) -> dict[str, str]:
        """
        Retrieves provider tickers for multiple security symbols in one query.
        Securities without a synonym for the requested provider use their
        internal symbol as the fallback ticker.

        :param symbols: Security symbols to retrieve.
        :param provider_id: Provider identifier.
        :param conn: Optional active autocommit read connection to reuse.

        :return: Mapping from each existing symbol to its provider ticker.
        """
        if not symbols:
            return {}

        with read_connection_scope(conn) as conn_:
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
    def get_shortable_symbols(symbols: list[str],
                              conn: Connection | None = None) -> set[str]:
        """
        Retrieves symbols that are eligible for short trading from the provided list.

        :param symbols: Security symbols to check.
        :param conn: Optional active autocommit read connection to reuse.

        :return: Set of symbols where is_shortable is True.
        """
        if not symbols:
            return set()

        with read_connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_GET_SHORTABLE_SYMBOLS_SQL, (symbols,))
                rows_ = cur_.fetchall()

        return {row_[0] for row_ in rows_ if row_ and row_[0]}

    @staticmethod
    def add_security(security: Securities,
                     conn: Connection | None = None) -> None:
        """
        Persist a new Security instance to the database using psycopg3.

        :param security: A Securities model instance to persist.
        :param conn: Optional active Connection for transaction reuse.
        """
        with connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(
                    _ADD_SECURITY_SQL,
                    tuple(getattr(security, col_) for col_ in SECURITIES_PAYLOAD_COLUMNS)
                )
                row_ = cur_.fetchone()
                if row_:
                    security.id = row_[0]
