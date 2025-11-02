# src/radar_core/infrastructure/crud/strategy_crud.py

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases
from sqlalchemy.future import select

# --- App modules ---
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import Strategies


class StrategyCrud(BaseCrud):
    def __init__(self):
        super().__init__(Strategies)

    def get_by_acronym(self, acronym: str) -> Strategies | None:
        """
        Get identifier and unit_label of a strategy based on its acronym.

        :param acronym: Strategy acronym.

        :return: A tuple with the identifier and unit_label (int, str) if a strategy is found, or None if not found.
        """
        statement_ = select(Strategies).where(Strategies.acronym == acronym)

        row_ = self.session.execute(statement_).first()

        if row_:
            return row_[0]
        return None