# src/radar_core/models/securities.py

# --- Python modules ---
# dataclasses: provides support for defining data-oriented classes.
from dataclasses import dataclass, field, fields
# typing: provides runtime support for type hints
from typing import Final, TYPE_CHECKING, get_origin

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
    is_shortable: bool = False
    is_crypto: bool = False
    is_near_continuous: bool = False
    store_locally: bool = False
    synonyms: list['Synonyms'] = field(default_factory=list)


# Exclude relational collections (e.g., synonyms) from table columns.
# get_origin() is used because parameterized generic aliases like list['Synonyms']
# are GenericAlias instances and do not evaluate to `is list`.
SECURITIES_COLUMNS: Final[tuple[str, ...]] = tuple(
    field_.name
    for field_ in fields(Securities)
    if get_origin(field_.type) is not list
)

SECURITIES_PAYLOAD_COLUMNS: Final[tuple[str, ...]] = tuple(
    col_
    for col_ in SECURITIES_COLUMNS
    if col_ != 'id'
)

