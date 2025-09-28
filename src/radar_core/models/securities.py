# -*- coding: utf-8 -*-
"""
Models auto-generated through sqlacodegen (https://github.com/agronholm/sqlacodegen/tree/master)
"""

# --- Python modules ---
# typing: provides runtime support for type hints
from typing import TYPE_CHECKING

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases.
from sqlalchemy import Boolean, PrimaryKeyConstraint, String, text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

# --- App modules ---
# models: result of Object-Relational Mapping
from radar_core.models.base_model import BaseModel
# Only import for type checking to avoid runtime circular imports
if TYPE_CHECKING:
    from radar_core.models.daily_data import DailyData as DailyData
    from radar_core.models.synonyms import Synonyms as Synonyms


class Securities(BaseModel):
    __tablename__ = 'securities'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='securities_pkey'),
        UniqueConstraint('symbol', name='securities_symbol_unique'),
        {'comment': 'Marketable financial instruments'}
    )

    symbol: Mapped[str] = mapped_column(String(10), comment='Acronym identifier of financial instrument')
    description: Mapped[str] = mapped_column(String(100))
    store_locally: Mapped[bool] = mapped_column(Boolean, server_default=text('false'),
                                                comment='Flag indicating whether prices obtained from the cloud should be saved in the database')

    # Relationships declaration, visit
    #  https://docs.sqlalchemy.org/en/20/orm/relationship_api.html#sqlalchemy.orm.relationship
    #  https://docs.sqlalchemy.org/en/20/orm/large_collections.html#write-only-relationship
    # Parameters
    #  argument      : the class that is to be related
    #  back_populates: name of a relationship() on the related class that will be synchronized with this one
    #  lazy='noload' : no loading should occur at any time, the related collection will remain empty,
    #                  the noload strategy is not recommended for general use.
    daily_data: Mapped[list['DailyData']] = relationship('DailyData', back_populates='security', lazy='noload')
    synonyms: Mapped[list['Synonyms']] = relationship('Synonyms', back_populates='security', lazy='noload')
