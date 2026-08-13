# src/radar_core/infrastructure/ratio_repository.py

# --- App modules ---
# database: provides access to database connections
from radar_core.database import get_psycopg_connection
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import RatioCrud
# models: result of Object-Relational Mapping
from radar_core.models import Ratios


class RatioRepository:
    """
    Coordinates ratio persistence operations and their transaction boundaries.
    """

    def __init__(self):
        self.__ratio_crud = RatioCrud()

    def remove_unlisted_symbols(self,
                                symbols: list[str]) -> int:
        """
        Deletes ratios for symbols not included in the supplied list.

        :param symbols: Symbols whose ratios must be retained.

        :return: The number of deleted rows.
        """
        return self.__ratio_crud.remove_unlisted_symbols(symbols)
    
    def flag_in_process(self,
                        symbol: str,
                        strategy_id: int,
                        timeframe: int) -> int:
        """
        Flags existing ratios before strategy calculation.

        The operation uses an independent short-lived transaction. A successful
        commit makes the failure indicator visible before calculation starts.

        :param symbol: Security symbol to flag.
        :param strategy_id: Strategy identifier.
        :param timeframe: Timeframe indicator.

        :return: The number of rows updated.
        """
        with get_psycopg_connection() as conn_:
            return self.__ratio_crud.flag_in_process(
                symbol,
                strategy_id,
                timeframe,
                conn=conn_
            )

    def persist_and_cleanup(self,
                            positive_ratios: list[Ratios],
                            symbol: str,
                            strategy_id: int,
                            timeframe: int) -> int:
        """
        Atomically persists positive ratios and removes remaining flagged rows.

        :param positive_ratios: Positive ratios to insert or update.
        :param symbol: Security symbol whose stale flagged rows must be removed.
        :param strategy_id: Strategy identifier.
        :param timeframe: Timeframe indicator.

        :return: The number of rows affected by the upsert.
        """
        upserted_rows_ = 0

        with get_psycopg_connection() as conn_:
            if positive_ratios:
                upserted_rows_ = self.__ratio_crud.upsert_many(
                    positive_ratios,
                    conn=conn_
                )

            self.__ratio_crud.delete_flagged_in_process(
                symbol,
                strategy_id,
                timeframe,
                conn=conn_
            )
            return upserted_rows_
