# src/radar_core/infrastructure/crud/security_crud.py

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases
from sqlalchemy.future import select

# --- App modules ---
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import BaseCrud
# models: result of Object-Relational Mapping
from radar_core.models import Securities, Synonyms


class SecurityCrud(BaseCrud):
    def __init__(self):
        super().__init__(Securities)

    def get_by_symbol(self,
                      symbol: str,
                      provider_id: int = None) -> Securities | None:
        """
        Get security based on its symbol, optionally with a synonym ticker.

        :param symbol: Security symbol.
        :param provider_id: If it is present, then the synonym for that external provider will be appended.

        :return: Security instance or None.
        """
        # noinspection PyTypeChecker
        statement_ = select(Securities).where(Securities.symbol == symbol)
        row_ = self.session.execute(statement_).first()

        if row_:
            security_ = row_[0]

            if provider_id:
                # Append synonym ticker for the external provider specified
                synonym_ = self.get_synonym(security_.id, provider_id)
                if synonym_:
                    security_.synonyms.append(synonym_)

            return security_

        return None

    def get_synonym(self,
                    security_id: int,
                    provider_id: int) -> Synonyms | None:
        """
        Get the ticker synonym for given security in a given financial prices provider
        :param security_id: Security id
        :param provider_id: Provider id
        :return: Synonyms instance or None
        """
        # noinspection PyTypeChecker
        statement_ = select(Synonyms).where(
            (Synonyms.provider_id == provider_id) &
            (Synonyms.security_id == security_id))
        row_ = self.session.execute(statement_).first()

        if row_:
            return row_[0]
        return None

    def add_security(self,
                     security: Securities) -> None:
        self.session.expire_on_commit = False
        self.session.add(security)
        self.session.commit()
