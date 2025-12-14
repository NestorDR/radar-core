# src/radar_core/domain/types.py
"""
Defines shared data structures and domain types for the application.
"""

# --- Python modules ---
# dataclasses: provides decorator and functions for auto-generating special methods in classes that primarily store data,
# such as __init__, __repr__, and __eq__, simplifying class definitions and reducing boilerplate code.
from dataclasses import dataclass

# --- App modules ---
# Import the specific strategy classes needed for the type hints
from radar_core.domain.strategies import MovingAverage, RsiRollerCoaster, RsiTwoBands


# frozen=True converts the class to an immutable dataclass (its fields cannot be modified after instantiation)
@dataclass(frozen=True)
class Strategies:
    """Small DI container for strategy instances supported by the analyzer."""
    sma: MovingAverage | None = None
    rsi_sma: MovingAverage | None = None
    rsi_rc: RsiRollerCoaster | None = None
    rsi_2b: RsiTwoBands | None = None
