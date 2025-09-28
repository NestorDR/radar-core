# -*- coding: utf-8 -*-
"""
Models auto-generated through sqlacodegen (https://github.com/agronholm/sqlacodegen/tree/master)
"""

# --- Python modules ---
# datetime: provides classes for simple and complex date and time manipulation.
import datetime
# decimal: provides support for fast correctly rounded decimal floating point arithmetic
#          it offers several advantages over the float datatype.
from decimal import Decimal
# typing: provides runtime support for type hints
from typing import TYPE_CHECKING

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases
from sqlalchemy import BigInteger, Date, ForeignKeyConstraint, Integer, Numeric, \
    PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

# --- App modules ---
# models: result of Object-Relational Mapping
from radar_core.models.base_model import BaseModel

if TYPE_CHECKING:
    from radar_core.models.securities import Securities


class DailyData(BaseModel):
    __tablename__ = 'daily_data'
    __table_args__ = (
        ForeignKeyConstraint(['security_id'], ['securities.id'], name='dailyData_securities_fkey'),
        PrimaryKeyConstraint('id', name='dailydata_pkey'),
        UniqueConstraint('security_id', 'date', name='dailydata_securityid_date_unique'),
        {'comment': 'Daily prices (OHLC) and indicators for the securities'}
    )

    security_id: Mapped[int] = mapped_column(Integer)
    date: Mapped[datetime.date] = mapped_column(Date)
    open: Mapped[Decimal | None] = mapped_column(Numeric(13, 4))
    high: Mapped[Decimal | None] = mapped_column(Numeric(13, 4))
    low: Mapped[Decimal | None] = mapped_column(Numeric(13, 4))
    close: Mapped[Decimal | None] = mapped_column(Numeric(13, 4))
    volume: Mapped[int] = mapped_column(BigInteger)
    percent_change: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))

    security: Mapped['Securities'] = relationship('Securities', back_populates='daily_data', lazy='noload')
