# src/radar_core/models/base_model.py

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases.
from sqlalchemy import Identity, Integer
from sqlalchemy.orm import Mapped, mapped_column

# --- App modules ---
# database: provides the database engine
from radar_core.database import Base


# Continue from your base class definition
class BaseModel(Base):
    __abstract__ = True
    id: Mapped[int] = mapped_column(Integer,
                                    Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647,
                                             cycle=False, cache=1), primary_key=True)
    # created_by: Mapped[Optional[str]] = mapped_column(String(100))
    # created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('now()'))
    # updated_by: Mapped[Optional[str]] = mapped_column(String(100))
    # updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime)
