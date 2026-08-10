# src/radar_core/infrastructure/crud/ratio_crud.py

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases
from sqlalchemy import and_, ColumnElement, not_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.future import select
from sqlalchemy.inspection import inspect  # Use mapper inspection to remain refactor-friendly

# --- App modules ---
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import RATIOS_UNIQUE_CONSTRAINT, Ratios

# Precompute the list of attributes to copy once at import-time (refactor-safe and fast at runtime, minimizes overhead)
_mapper = inspect(Ratios)
_excluded_attrs = (
    # keys to exclude from the copy operation
    Ratios.symbol,
    Ratios.strategy_id,
    Ratios.inputs,
    Ratios.timeframe,
    Ratios.is_long_position,
    # flag
    Ratios.is_in_process,
)
# Operator |: for union of sets: primary key and excluded attributes
_exclude_keys = {col.key for col in _mapper.primary_key} | {attr.key for attr in _excluded_attrs}
# Updatable attributes after re-evaluating a strategy
_COPY_ATTRS = tuple(k for k in _mapper.columns.keys() if k not in _exclude_keys)


class RatioCrud(BaseCrud):
    def __init__(self):
        super().__init__(Ratios)

    @staticmethod
    def _base_clause_to_flag(symbol: str,
                             strategy_id: int,
                             timeframe: int) -> ColumnElement[bool]:
        """
        Build the base where clause for flag conditions.

        :param symbol: Security symbol.
        :param strategy_id: Identifier of the trading strategy.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).

        :return: Base `and_` clause with common conditions.
        """
        return and_(Ratios.symbol == symbol,
                    Ratios.strategy_id == strategy_id,
                    Ratios.timeframe == timeframe)

    def delete_symbols_not_in(self,
                              symbols: list[str]) -> int:
        """
        Delete rows where the symbol is not in the provided list.

        :param symbols: List of symbols to keep in the database.
        """

        # Create where clause for symbols not in the provided list
        where_clause_ = not_(Ratios.symbol.in_(symbols))

        # Delete rows that don't have symbols in the list
        return super()._delete_for([where_clause_])

    def delete_flagged_in_process(self,
                                  symbol: str,
                                  strategy_id: int,
                                  timeframe: int) -> None:
        """
        Delete rows in which its column `is_in_process` is flagged as True.

        :param symbol: Security symbol flagged as in process.
        :param strategy_id: Identifier of the trading strategy flagged as in process.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        """
        where_clause_ = and_(self._base_clause_to_flag(symbol, strategy_id, timeframe),
                             Ratios.is_in_process)

        # Flagged rows deletion
        super()._delete_for(where_clause_)

    def flag_in_process(self,
                        symbol: str,
                        strategy_id: int,
                        timeframe: int) -> None:
        """
        Update the flag field `is_in_process` to True for a specific symbol, trading strategy, and timeframe.

        :param symbol: Security symbol to flag.
        :param strategy_id: Identifier of the trading strategy to flag.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        """
        where_clause_ = self._base_clause_to_flag(symbol, strategy_id, timeframe)

        super()._flag_in_process(where_clause_)

    @staticmethod
    def _deduplicate_batch(ratios_list: list[Ratios]) -> list[dict]:
        """
        Deduplicate ratio records in-memory by conflict key (symbol, strategy_id, inputs, timeframe, is_long_position).
        Keep the best performing ratio (highest net_profit and expected_value) on duplicate keys.

        :param ratios_list: List of Ratios objects.

        :return: List of deduplicated dictionaries with uniform column keys for PostgreSQL insert.
        """
        deduped_map_ = {}
        for item_ in ratios_list:
            key_ = (
                item_.symbol,
                item_.strategy_id,
                item_.inputs,
                item_.timeframe,
                item_.is_long_position,
            )
            if key_ not in deduped_map_:
                deduped_map_[key_] = item_
            else:
                existing_ = deduped_map_[key_]
                new_score_ = (item_.net_profit, item_.expected_value)
                existing_score_ = (existing_.net_profit, existing_.expected_value)
                if new_score_ > existing_score_:
                    deduped_map_[key_] = item_

        # Convert deduplicated Ratios objects to homogeneous dictionaries for pg_insert
        return [
            {
                col_.key: getattr(ratio_, col_.key, None)
                for col_ in _mapper.column_attrs
                if col_.key != 'id'
            }
            for ratio_ in deduped_map_.values()
        ]

    def upsert_many(self,
                    ratios_list: list[Ratios]) -> int:
        """
        Perform a batch PostgreSQL upsert (INSERT ... ON CONFLICT DO UPDATE) for strategy ratios.

        :param ratios_list: List of Ratios objects to insert or update.

        :return: The number of rows affected by the batch operation.
        """
        if not ratios_list:
            return 0

        self.session.expire_on_commit = False

        deduped_data_ = self._deduplicate_batch(ratios_list)
        if not deduped_data_:
            return 0

        # Build PostgreSQL insert statement
        insert_stmt_ = pg_insert(Ratios).values(deduped_data_)

        # Identify columns to update (all payload columns except primary key 'id' and conflict key columns)
        conflict_keys_ = {'symbol', 'strategy_id', 'inputs', 'timeframe', 'is_long_position', 'id'}
        update_cols_ = {
            col_.name: insert_stmt_.excluded[col_.name]
            for col_ in Ratios.__table__.columns
            if col_.name not in conflict_keys_
        }
        # Explicitly reset flag field is_in_process to False on update
        update_cols_['is_in_process'] = False

        upsert_stmt_ = insert_stmt_.on_conflict_do_update(
            constraint=RATIOS_UNIQUE_CONSTRAINT,
            set_=update_cols_
        )

        try:
            result_ = self.session.execute(upsert_stmt_)
            self.session.commit()
            return result_.rowcount
        except Exception:
            self.session.rollback()
            raise

    def upsert(self,
               ratios: Ratios) -> None:
        """
        Update or insert the ratios for a trading strategy.

        :param ratios: An object Ratios with the calculated ratios to update existent or insert new.
        """
        self.session.expire_on_commit = False

        # Get stored ratios for the strategy with input settings
        statement_ = (select(Ratios)
                      # Generate the clause 'where' according to the parameterized strategy with specified 'inputs' values
                      .where(and_(Ratios.symbol == ratios.symbol,
                                  Ratios.strategy_id == ratios.strategy_id,
                                  Ratios.inputs == ratios.inputs,
                                  Ratios.timeframe == ratios.timeframe,
                                  Ratios.is_long_position == ratios.is_long_position))
                      .with_for_update())  # Lock the row if exists to prevent race conditions

        row_ = self.session.execute(statement_).first()

        if row_:
            saved_record_: Ratios = row_[0]
            saved_record_.is_in_process = False

            # Copy attributes using a precomputed list, refactor-safe list (no per-call reflection)
            for name in _COPY_ATTRS:
                setattr(saved_record_, name, getattr(ratios, name))
        else:
            self.session.add(ratios)

        self.session.commit()

