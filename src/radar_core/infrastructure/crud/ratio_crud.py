# src/radar_core/infrastructure/crud/ratio_crud.py

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases
from sqlalchemy import and_, ColumnElement, not_
from sqlalchemy.future import select
from sqlalchemy.inspection import inspect  # Use mapper inspection to remain refactor-friendly

# --- App modules ---
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import Ratios

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

            """
            # Set prices
            saved_record_.from_date = ratios.from_date
            saved_record_.to_date = ratios.to_date
            saved_record_.initial_price = ratios.initial_price
            saved_record_.final_price = ratios.final_price
            saved_record_.net_change = ratios.net_change

            saved_record_.signals = ratios.signals
            saved_record_.winnings = ratios.winnings
            saved_record_.losses = ratios.losses
            # saved_record_.profit_factor = ratios.profit_factor
            saved_record_.net_profit = ratios.net_profit
            saved_record_.expected_value = ratios.expected_value
            saved_record_.win_probability = ratios.win_probability
            saved_record_.loss_probability = ratios.loss_probability
            saved_record_.average_win = ratios.average_win
            saved_record_.average_loss = ratios.average_loss

            saved_record_.min_percentage_change_to_win = ratios.min_percentage_change_to_win
            saved_record_.max_percentage_change_to_win = ratios.max_percentage_change_to_win

            saved_record_.total_sessions = ratios.total_sessions
            saved_record_.winning_sessions = ratios.winning_sessions
            saved_record_.losing_sessions = ratios.losing_sessions
            saved_record_.percentage_exposure = ratios.percentage_exposure

            saved_record_.first_input_date = ratios.first_input_date
            saved_record_.last_input_date = ratios.last_input_date
            saved_record_.last_output_date = ratios.last_output_date

            saved_record_.last_input_price = ratios.last_input_price
            saved_record_.last_output_price = ratios.last_output_price
            saved_record_.last_stop_loss = ratios.last_stop_loss

            self.session.merge(saved_record_)
            """
        else:
            self.session.add(ratios)

        self.session.commit()
