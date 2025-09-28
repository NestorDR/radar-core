# -*- coding: utf-8 -*-
"""
Models auto-generated through sqlacodegen (https://github.com/agronholm/sqlacodegen/tree/master)
"""

# --- Python modules ---
# typing: provides runtime support for type hints
from typing import TYPE_CHECKING

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases.
from sqlalchemy import PrimaryKeyConstraint, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

# --- App modules ---
# models: result of Object-Relational Mapping
from radar_core.models.base_model import BaseModel
# Only import for type checking to avoid runtime circular imports
if TYPE_CHECKING:
    from radar_core.models.ratios import Ratios

class Strategies(BaseModel):
    __tablename__ = 'strategies'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='strategies_pkey'),
        UniqueConstraint('acronym', name='strategies_acronym_unique'),
        {'comment': 'Speculation/investment strategies on financial instruments'}
    )

    name: Mapped[str] = mapped_column(String(50))
    acronym: Mapped[str] = mapped_column(String(25))
    unit_label: Mapped[str] = mapped_column(String(5), server_default=text("''::character varying"))
    pool: Mapped[str] = mapped_column(String(10), server_default=text("''::character varying"))

    ratios: Mapped[list['Ratios']] = relationship('Ratios', back_populates='strategy', lazy='noload')
