# src/radar_core/infrastructure/crud/strategy_crud.py

# --- Python modules ---
# typing: provides runtime support for type hints
from typing import Final

# --- Third Party Libraries ---
# psycopg: PostgreSQL database adapter
from psycopg import Connection
from psycopg.sql import Composed, Identifier, SQL

# --- App modules ---
# database: provides access to database connections
from radar_core.database import read_connection_scope
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import Strategies

# Module-level table identifier constant
_STRATEGIES_TABLE = Identifier(Strategies.__tablename__)

# SQL statements
_GET_STRATEGY_BY_ACRONYM_SQL: Final[Composed] = SQL(
    "SELECT id, name, acronym, pool, unit_label FROM ") + _STRATEGIES_TABLE + SQL(" WHERE acronym = %s")


class StrategyCrud(BaseCrud):
    def __init__(self):
        super().__init__(Strategies)

    @staticmethod
    def get_by_acronym(acronym: str,
                       conn: Connection | None = None) -> Strategies | None:
        """
        Retrieves a Strategies model instance by acronym.

        :param acronym: Strategy acronym.
        :param conn: Optional active autocommit read connection to reuse.

        :return: A Strategies model instance if found, or None if not found.
        """
        with read_connection_scope(conn) as conn_:
            with conn_.cursor() as cur_:
                cur_.execute(_GET_STRATEGY_BY_ACRONYM_SQL, (acronym,))
                row_ = cur_.fetchone()
                if row_:
                    return Strategies(
                        id=row_[0],
                        name=row_[1],
                        acronym=row_[2],
                        pool=row_[3],
                        unit_label=row_[4]
                    )
                return None
