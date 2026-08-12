# src/radar_core/models/base_model.py

# --- Python modules ---
# dataclasses: provides support for defining data-oriented classes.
from dataclasses import dataclass, field


# kw_only=True: indicates that all fields in the dataclass must be passed as arguments
# BaseModel(id=123) → Correct
# BaseModel() → Correct; id is None
# BaseModel(123) → TypeError
@dataclass(kw_only=True)
class BaseModel:
    id: int | None = field(default=None)
