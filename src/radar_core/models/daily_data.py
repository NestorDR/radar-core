# src/radar_core/models/daily_data.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
import datetime
# dataclasses: provides support for defining data-oriented classes.
from dataclasses import dataclass
# Decimal: provides fast, correctly rounded decimal floating-point arithmetic with advantages over the built-in float.
from decimal import Decimal

# --- App modules ---
# base_model: provides a base class for all models.
from radar_core.models.base_model import BaseModel


# kw_only=True: indicates that all fields in the dataclass must be passed as arguments
@dataclass(kw_only=True)
class DailyData(BaseModel):
    __tablename__ = 'daily_data'

    security_id: int = 0
    date: datetime.date | None = None
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal | None = None
    volume: int = 0
    percent_change: Decimal | None = None
