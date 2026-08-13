# src/radar_core/infrastructure/crud/ratio_crud.py

# --- Python modules ---
# operator: exports a set of efficient functions corresponding to intrinsic operators of Python
#  (e.g., attrgetter for fast attribute access)
from operator import attrgetter

# --- Third Party Libraries ---
# psycopg: PostgreSQL database adapter
from psycopg import Connection
from psycopg.sql import Identifier, SQL

# --- App modules ---
# database: provides access to database connections
from radar_core.database import connection_scope
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import RATIOS_CONFLICT_COLUMNS, RATIOS_PAYLOAD_COLUMNS, Ratios

# Module-level table identifier constant
_RATIOS_TABLE = Identifier(Ratios.__tablename__)

# Precompute metadata and attribute getters once at import-time
_cols_sql = SQL(', ').join(Identifier(col_) for col_ in RATIOS_PAYLOAD_COLUMNS)
_values_sql = SQL(', ').join(SQL('%({})s').format(SQL(col_)) for col_ in RATIOS_PAYLOAD_COLUMNS)
_on_conflict_sql = SQL(', ').join(Identifier(col_) for col_ in RATIOS_CONFLICT_COLUMNS)
_updatable_col_names = tuple(
    col_
    for col_ in RATIOS_PAYLOAD_COLUMNS
    # Explicitly reset flag field is_in_process to False on update
    if col_ not in set(RATIOS_CONFLICT_COLUMNS) | {'id', 'is_in_process'}
)
_update_sql = SQL(', ').join(
    SQL('{col} = EXCLUDED.{col}')
    .format(col=Identifier(col_)) for col_ in _updatable_col_names
)

# Fast C-level compiled attribute getters to avoid string lookup overhead in loops
_payload_attr_getters = tuple(
    (key_, attrgetter(key_))
    for key_ in RATIOS_PAYLOAD_COLUMNS
)

# SQL statements
_FLAG_IN_PROCESS_SQL = SQL("UPDATE ") + _RATIOS_TABLE + SQL(
    " SET is_in_process = TRUE WHERE symbol = %s AND strategy_id = %s AND timeframe = %s")

# Explicitly reset flag field is_in_process to False on update
_UPSERT_RATIOS_SQL = SQL("INSERT INTO ") + _RATIOS_TABLE + SQL(
    " ({cols}) VALUES ({values}) ON CONFLICT ({on_conflict}) DO UPDATE SET {update}, is_in_process = FALSE"
).format(cols=_cols_sql, values=_values_sql, on_conflict=_on_conflict_sql, update=_update_sql)

_DELETE_ALL_RATIOS_SQL = SQL("DELETE FROM ") + _RATIOS_TABLE
_DELETE_UNLISTED_SYMBOLS_SQL = _DELETE_ALL_RATIOS_SQL + SQL(" WHERE symbol != ALL(%s)")
_DELETE_FLAGGED_IN_PROCESS_SQL = _DELETE_ALL_RATIOS_SQL + SQL(
    " WHERE symbol = %s AND strategy_id = %s  AND timeframe = %s AND is_in_process = TRUE")


class RatioCrud(BaseCrud):
    def __init__(self):
        super().__init__(Ratios)

    @staticmethod
    def remove_unlisted_symbols(symbols: list[str],
                                conn: Connection | None = None) -> int:
        """
        Delete rows where the symbol is not in the provided list.

        :param symbols: List of symbols to keep in the database.
        :param conn: Optional active Connection for transaction reuse.

        :return: The number of deleted rows.
        """
        if not symbols:
            query_ = _DELETE_ALL_RATIOS_SQL
            params_ = ()
        else:
            query_ = _DELETE_UNLISTED_SYMBOLS_SQL
            params_ = (symbols,)

        with connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(query_, params_)
                return cur_.rowcount

    @staticmethod
    def delete_flagged_in_process(symbol: str,
                                  strategy_id: int,
                                  timeframe: int,
                                  conn: Connection | None = None) -> int:
        """
        Delete rows in which its column `is_in_process` is flagged as True.

        :param symbol: Security symbol flagged as in process.
        :param strategy_id: Identifier of the trading strategy flagged as in process.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param conn: Optional active Connection for transaction reuse.

        :return: The number of deleted rows.
        """
        with connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(
                    _DELETE_FLAGGED_IN_PROCESS_SQL,
                    (symbol, strategy_id, timeframe),
                )
                return cur_.rowcount

    @staticmethod
    def flag_in_process(symbol: str,
                        strategy_id: int,
                        timeframe: int,
                        conn: Connection | None = None) -> int:
        """
        Update the flag field `is_in_process` to True for a specific symbol, trading strategy, and timeframe.

        :param symbol: Security symbol to flag.
        :param strategy_id: Identifier of the trading strategy to flag.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param conn: Optional active Connection for transaction reuse.

        :return: The number of rows updated.
        """
        with connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(
                    _FLAG_IN_PROCESS_SQL,
                    (symbol, strategy_id, timeframe),
                )
                return cur_.rowcount

    @staticmethod
    def upsert_many(ratios_list: list[Ratios],
                    conn: Connection | None = None) -> int:
        """
        Perform a batch PostgreSQL upsert (INSERT ... ON CONFLICT DO UPDATE) for strategy ratios using psycopg3.

        :param ratios_list: List of Ratios objects to insert or update.
        :param conn: Optional active Connection for transaction reuse.

        :return: The number of rows affected by the batch operation.
        """
        if not ratios_list:
            return 0

        payload_data_ = [
            {
                key_: getter_(ratio_)
                for key_, getter_ in _payload_attr_getters
            }
            for ratio_ in ratios_list
        ]

        with connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.executemany(_UPSERT_RATIOS_SQL, payload_data_)
                return cur_.rowcount
