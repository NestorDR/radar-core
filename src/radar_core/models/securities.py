# src/radar_core/models/securities.py

# --- Python modules ---
# dataclasses: provides support for defining data-oriented classes.
from dataclasses import dataclass, field
# typing: provides runtime support for type hints
from typing import TYPE_CHECKING

# --- App modules ---
# base_model: provides a base class for all models.
from radar_core.models.base_model import BaseModel

if TYPE_CHECKING:
    from radar_core.models.synonyms import Synonyms


# kw_only=True: indicates that all fields in the dataclass must be passed as arguments
@dataclass(kw_only=True)
class Securities(BaseModel):
    __tablename__ = 'securities'

    symbol: str = ''
    description: str = ''
    is_bear: bool = False
    store_locally: bool = False
    synonyms: list['Synonyms'] = field(default_factory=list)
