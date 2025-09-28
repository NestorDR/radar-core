# -*- coding: utf-8 -*-

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import getLogger
# os: allows access to functionalities dependent on the Operating System
from os import getenv
# urllib: collects several modules for working with URLs
from urllib import parse

# --- Third Party Libraries ---
# sqlalchemy: SQL and ORM toolkit for accessing relational databases.
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.exc import OperationalError

logger_ = getLogger(__name__)


class Base(DeclarativeBase):
    pass


# Lazy initialization global variables will be set in the first call to session_factory().
_engine = None
_SessionFactory: sessionmaker[Session] | None = None


# Visit https://auth0.com/blog/sqlalchemy-orm-tutorial-for-python-developers/#SQLAlchemy-Introduction
# Use session_factory() to get a new Session
def session_factory() -> Session:
    """
    Provides a SQLAlchemy Session.

    The engine and session factory are initialized on the first call.

    SQLAlchemy ORM uses Sessions to implement the Unit of Work design pattern. As explained by Martin Fowler,
     a Unit of Work is used to maintain a list of objects affected by a business transaction and to coordinate the
     writing out of changes.
    This means that all modifications tracked by Sessions (Units of Works) will be applied to the underlying database
     together, or none of them will. In other words, Sessions are used to guarantee the database consistency.
    """
    global _engine, _SessionFactory

    # The first call creates the engine and the session factory.
    if _engine is None:
        connection_str_ = _get_connection_str()

        # Add connect_args with connect_timeout for PostgreSQL
        _engine = create_engine(connection_str_, connect_args={"connect_timeout": 10})

        # Sessionmaker factory generates new Session objects when is called, creating them given configuration arguments, visit
        #  https://docs.sqlalchemy.org/en/20/orm/session_api.html#sqlalchemy.orm.sessionmaker
        # Parameters
        #  bind: an individual session to a connection
        #  expire_on_commit: default value is True. When True, all instances will completely time out after each commit(),
        #                    so all access to saved objects after a committed transaction will require a database reread.
        _SessionFactory = sessionmaker(bind=_engine)

    try:
        # Attempt to create tables (it's an idempotent operation, safe to call multiple times)
        # and return a new session.
        Base.metadata.create_all(_engine)
        return _SessionFactory()
    except OperationalError as e:
        logger_.exception('Database connection failed. Could not create session.', exc_info=e)
        raise  # Re-raise the signal failure exception to the call code


def _get_connection_str():
    """
    Builds the database connection string from environment variables.
    This is called only when needed, ensuring .env has already been loaded.
    """

    # Configure PostgreSQL Database Connection String
    # Visit https://www.postgresql.org/docs/16/libpq-envars.html
    db_host = parse.quote(getenv('POSTGRES_HOST', 'host-here'))
    db_port = parse.quote(getenv('POSTGRES_PORT', '5432'))
    db_name = parse.quote(getenv('POSTGRES_DB', 'dbname-here'))
    db_username = parse.quote(getenv('POSTGRES_USER', 'user-here'))
    db_password = parse.quote(getenv('POSTGRES_PASSWORD', 'pwd-here'))
    db_ssl_mode = parse.quote(getenv('POSTGRES_SSL_MODE', 'prefer'))
    db_options = getenv('POSTGRES_OPTIONS', None)

    # Visit https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#module-sqlalchemy.dialects.postgresql.psycopg
    connection_str_ = (
        f'postgresql+psycopg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
        f'?sslmode={db_ssl_mode}'
    )
    if db_options:
        connection_str_ += f"&options={db_options}"

    return connection_str_
