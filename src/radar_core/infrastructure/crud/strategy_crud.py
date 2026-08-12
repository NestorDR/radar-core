# src/radar_core/infrastructure/crud/strategy_crud.py

# --- Third Party Libraries ---
# psycopg: PostgreSQL database adapter for Python
from psycopg.sql import Identifier, SQL

# --- App modules ---
from radar_core.database import get_psycopg_connection
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import Strategies

# Module-level table identifier constant
_STRATEGIES_TABLE = Identifier(Strategies.__tablename__)

_GET_STRATEGY_BY_ACRONYM_SQL = SQL("SELECT id, name, acronym, pool, unit_label FROM ") + _STRATEGIES_TABLE + SQL(
    " WHERE acronym = %s")


def _get_strategy_by_acronym_cached(acronym: str) -> Strategies | None:
    """
    Module-level cached helper function to prevent B019 method lru_cache memory leak issues.
    """
    with get_psycopg_connection() as conn_:
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


class StrategyCrud(BaseCrud):
    def __init__(self):
        super().__init__(Strategies)

    @staticmethod
    def get_by_acronym(acronym: str) -> Strategies | None:
        """
        Get identifier and unit_label of a strategy based on its acronym.

        :param acronym: Strategy acronym.

        :return: A Strategies instance if found, or None if not found.
        """
        return _get_strategy_by_acronym_cached(acronym)
