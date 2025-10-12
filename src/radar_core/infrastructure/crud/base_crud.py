# src/radar_core/infrastructure/crud/base_crud.py

# --- Python modules ---
# json: library for encoding and decoding prices in JSON format.
import json

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases
from sqlalchemy import delete, update
from sqlalchemy.exc import OperationalError

# --- App modules ---
# database: provides the database engine
from radar_core.database import session_factory


class BaseCrud(object):
    def __init__(self,
                 base_model):
        self.base_model = base_model
        try:
            self.session = session_factory()
        except OperationalError:
            print(f"Error: Failed to initialize database session for {self.__class__.__name__}."
                  f" CRUD operations will be unavailable.")
            raise

    def __del__(self) -> None:
        # Robustly check if 'session' attribute exists and is not None before trying to close
        if hasattr(self, 'session') and self.session is not None:
            try:
                self.session.close()
            except Exception as e:
                # In case session.close() itself throws an error for some reason
                print(f"ERROR (BaseCrud.__del__): Exception while closing session for {self.__class__.__name__}: {e}")

    @staticmethod
    def serialize_inputs(inputs: dict) -> str:
        """
        Accessing and serializing dictionary 'inputs' with parameters of a strategy.
        :param inputs: Input prices that parameterizes a strategy.
        :return: Serialized inputs parameters.
        """
        return json.dumps(inputs)

    def get_all(self):
        return self.session.query(self.base_model).all()

    def get_by_id(self,
                  entity_id: int):
        return self.session.query(self.base_model).get(entity_id)

    def _delete_for(self,
                    where_clause: list) -> int:
        """
        Delete rows in the table for a specific condition.

        :return: The number of rows deleted
        """
        # Flagged rows deletion
        statement_ = (delete(self.base_model)
                      .where(*where_clause))

        result = self.session.execute(statement_)
        self.session.commit()

        # Return the count of affected rows
        return result.rowcount

    def _flag_in_process(self,
                         where_clause: list) -> int:
        """
        Update `is_in_process` flag field to True for a specific condition.

        :return: The number of rows deleted
        """
        self.session.expire_on_commit = False

        statement_ = (update(self.base_model)
                      .where(*where_clause)
                      .values(is_in_process=True))

        result = self.session.execute(statement_)
        self.session.commit()

        # Return the count of affected rows
        return result.rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'session') and self.session is not None:
            try:
                self.session.close()
            except Exception as e:
                print(f"ERROR: Exception while closing session: {e}")
