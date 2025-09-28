# -*- coding: utf-8 -*-
"""
Models auto-generated through sqlacodegen (https://github.com/agronholm/sqlacodegen/tree/master)
"""

# --- Python modules ---
# datetime: provides classes for simple and complex date and time manipulation.
import datetime
# decimal: provides support for fast correctly rounded decimal floating point arithmetic
#          it offers several advantages over the float datatype.
import decimal
# typing: provides runtime support for type hints
from typing import TYPE_CHECKING

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases.
from sqlalchemy import Boolean, Date, Float, ForeignKeyConstraint, Integer, Numeric, PrimaryKeyConstraint, \
    SmallInteger, String, text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

# --- App modules ---
# models: result of Object-Relational Mapping
from radar_core.models.base_model import BaseModel
# Only import for type checking to avoid runtime circular imports
if TYPE_CHECKING:
    from radar_core.models.strategies import Strategies

class Ratios(BaseModel):
    __tablename__ = 'ratios'
    __table_args__ = (
        ForeignKeyConstraint(['strategy_id'], ['strategies.id'], name='ratios_strategies_fkey'),
        PrimaryKeyConstraint('id', name='ratios_pkey'),
        UniqueConstraint('symbol', 'strategy_id', 'inputs', 'timeframe', 'is_long_position',
                         name='ratios_symbol_strategy_inputs_timeframe_islong_unique'),
        {'comment': 'ratios to evaluate the performance of speculation/investment '
                    'strategies'}
    )

    symbol: Mapped[str] = mapped_column(String(10), comment='Acronym identifier of financial instrument')
    strategy_id: Mapped[int] = mapped_column(Integer)
    timeframe: Mapped[int] = mapped_column(SmallInteger,
                                           comment='Time frames: 1.Intraday, 2.Daily, 3.Weekly, 4.Monthly')
    inputs: Mapped[str] = mapped_column(String(50),
                                        comment='Defines the independent variables of the strategy, for example for a moving average it is the length in sessions of the period for calculating the average')
    is_long_position: Mapped[bool] = mapped_column(Boolean,
                                                   comment='Defines whether the ratio relates to a long or short market position')
    is_in_process: Mapped[bool] = mapped_column(Boolean, server_default=text('false'),
                                                comment='Flag indicating whether the record can be deleted during any running process')
    from_date: Mapped[datetime.date] = mapped_column(Date,
                                                     comment='Indicates the date from which the strategy was tested to identify results and ratios')
    to_date: Mapped[datetime.date] = mapped_column(Date,
                                                   comment='Indicates the date up to which the strategy was tested to identify results and ratios.')
    initial_price: Mapped[decimal.Decimal] = mapped_column(Numeric(12, 2),
                                                           comment='Initial price of the period in which the strategy was tested')
    final_price: Mapped[decimal.Decimal] = mapped_column(Numeric(9, 2),
                                                         comment='Final price of the period in which the strategy was tested')
    net_change: Mapped[float] = mapped_column(Float,
                                              comment='Percentage change of the final price over the initial price')
    signals: Mapped[int] = mapped_column(SmallInteger, comment='Number of trade signals identified by the strategy')
    winnings: Mapped[float] = mapped_column(Float, comment='Total gain on positive/winning operations')
    losses: Mapped[float] = mapped_column(Float, comment='Total loss on negative/losing operations.')
    net_profit: Mapped[float] = mapped_column(Float,
                                              comment='Percentage of net profit got following the input and output signals. Formula: (winnings_ - losses_) / initial_price')
    expected_value: Mapped[float] = mapped_column(Float,
                                                  comment='Mathematical expectation of the strategy. Formula: (win_probability * average_win) + (loss_probability * average_loss).')
    win_probability: Mapped[float] = mapped_column(Float, comment='Percentage of positive/winning operations')
    loss_probability: Mapped[float] = mapped_column(Float, comment='Percentage of negative/losing operations')
    average_win: Mapped[float] = mapped_column(Float, comment='Average profit of the positive/winning operations')
    average_loss: Mapped[float] = mapped_column(Float, comment='Average loss of the negative/losing operations')
    min_percentage_change_to_win: Mapped[decimal.Decimal] = mapped_column(Numeric(6, 2),
                                                                          comment='Minimum percentage change of input sessions for the positive/winning operations')
    max_percentage_change_to_win: Mapped[decimal.Decimal] = mapped_column(Numeric(6, 2),
                                                                          comment='Maximum percentage change of input sessions for the positive/winning operations')
    total_sessions: Mapped[int] = mapped_column(SmallInteger,
                                                comment='Total number of sessions to which the strategy has been evaluated')
    winning_sessions: Mapped[int] = mapped_column(SmallInteger,
                                                  comment='Number of sessions spent/elapsed during positive/winning operations')
    losing_sessions: Mapped[int] = mapped_column(SmallInteger,
                                                 comment='Number of sessions spent/elapsed during negative/losing operations')
    percentage_exposure: Mapped[float] = mapped_column(Float,
                                                       comment='Percentage time during which the strategy was active. Formula: (winning_sessions + losing_sessions) / total_sessions')
    first_input_date: Mapped[datetime.date] = mapped_column(Date, comment='Date of the first input signal')
    last_input_date: Mapped[datetime.date] = mapped_column(Date, comment='Date of the last input signal')
    last_output_date: Mapped[datetime.date | None] = mapped_column(Date, comment='Date of the last output signal')
    last_input_price: Mapped[decimal.Decimal] = mapped_column(Numeric(12, 2),
                                                              comment='Input price for the last position opened by the tested strategy')
    last_output_price: Mapped[decimal.Decimal] = mapped_column(Numeric(12, 2),
                                                               comment='Output price for the last position opened by the tested strategy')
    last_stop_loss: Mapped[decimal.Decimal] = mapped_column(Numeric(12, 2),
                                                            comment='Stop loss for the last position opened by the tested strategy')

    strategy: Mapped['Strategies'] = relationship('Strategies', back_populates='ratios', lazy='noload')
