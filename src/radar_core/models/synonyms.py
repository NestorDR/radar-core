# src/radar_core/models/synonyms.py

# --- Python modules ---
# dataclasses: provides support for defining data-oriented classes.
from dataclasses import dataclass

# --- App modules ---
# base_model: provides a base class for all models.
from radar_core.models.base_model import BaseModel


# kw_only=True: indicates that all fields in the dataclass must be passed as arguments
@dataclass(kw_only=True)
class Synonyms(BaseModel):
    __tablename__ = 'synonyms'

    provider_id: int = 0
    security_id: int = 0
    ticker: str = ''
