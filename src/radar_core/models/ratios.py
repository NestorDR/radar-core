# src/radar_core/models/ratios.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
import datetime
# dataclasses: provides decorator and functions for data-oriented classes.
from dataclasses import dataclass, field, fields
# Decimal: provides fast, correctly rounded decimal floating-point arithmetic with advantages over the built-in float.
from decimal import Decimal

# --- App modules ---
# base_model: provides a base class for all models.
from radar_core.models.base_model import BaseModel


# kw_only=True: indicates that all fields in the dataclass must be passed as arguments
@dataclass(kw_only=True)
class Ratios(BaseModel):
    __tablename__ = 'ratios'

    symbol: str = field(default='', metadata={'conflict': True})
    strategy_id: int = field(default=0, metadata={'conflict': True})
    timeframe: int = field(default=0, metadata={'conflict': True})
    inputs: str = field(default='', metadata={'conflict': True})
    is_long_position: bool = field(default=True, metadata={'conflict': True})
    is_in_process: bool = False
    from_date: datetime.date | None = None
    to_date: datetime.date | None = None
    initial_price: Decimal | float = 0.0
    final_price: Decimal | float = 0.0
    current_indicators: str | None = None
    net_change: float = 0.0
    signals: int = 0
    winnings: float = 0.0
    losses: float = 0.0
    net_profit: float = 0.0
    expected_value: float = 0.0
    win_probability: float = 0.0
    loss_probability: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    min_percentage_change_to_win: Decimal | float = 0.0
    max_percentage_change_to_win: Decimal | float = 0.0
    total_sessions: int = 0
    winning_sessions: int = 0
    losing_sessions: int = 0
    percentage_exposure: float = 0.0
    first_input_date: datetime.date | None = None
    last_input_date: datetime.date | None = None
    last_output_date: datetime.date | None = None
    last_input_price: Decimal | float | None = None
    last_output_price: Decimal | float | None = None
    last_stop_loss: Decimal | float | None = None


RATIOS_CONFLICT_COLUMNS = tuple(
    field_.name
    for field_ in fields(Ratios)
    if field_.metadata.get('conflict', False)
)

RATIOS_PAYLOAD_COLUMNS = tuple(
    field_.name
    for field_ in fields(Ratios)
    if field_.name != 'id'
)

RATIOS_UNIQUE_CONSTRAINT = 'ratios_symbol_strategy_inputs_timeframe_islong_unique'