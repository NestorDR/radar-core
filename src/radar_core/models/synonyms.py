# -*- coding: utf-8 -*-
"""
Models auto-generated through sqlacodegen (https://github.com/agronholm/sqlacodegen/tree/master)
"""

# --- Python modules ---
# typing: provides runtime support for type hints
from typing import TYPE_CHECKING

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases.
from sqlalchemy import ForeignKeyConstraint, Index, Integer, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

# --- App modules ---
# models: result of Object-Relational Mapping
from radar_core.models.base_model import BaseModel
if TYPE_CHECKING:
    from radar_core.models.securities import Securities

class Synonyms(BaseModel):
    __tablename__ = 'synonyms'
    __table_args__ = (
        ForeignKeyConstraint(['security_id'], ['securities.id'], name='synonyms_securities_fkey'),
        PrimaryKeyConstraint('id', name='synonyms_pkey'),
        Index('synonyms_providerId_securityId_idx', 'provider_id', 'security_id', unique=True),
        {'comment': 'Synonyms of security symbols in different quote providers.\n'
                    'The providerId column is managed without a master table, and its '
                    'values are:\n'
                    '1 -> Yahoo'}
    )

    provider_id: Mapped[int] = mapped_column(Integer)
    security_id: Mapped[int] = mapped_column(Integer)
    ticker: Mapped[str] = mapped_column(String(10))

    security: Mapped['Securities'] = relationship('Securities', back_populates='synonyms', lazy='noload')
