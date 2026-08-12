# src/radar_core/database.py

# --- Python modules ---
# contextlib: provides utilities for common tasks involving context managers
from contextlib import contextmanager, suppress
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import getLogger
# os: provides operating system interfaces and functionality
from os import getenv
# typing: provides runtime support for type hints
from typing import Generator
# urllib: collects several modules for working with URLs
from urllib import parse

# --- Third Party Libraries ---
# psycopg: PostgreSQL database adapter for Python
import psycopg
from psycopg import Connection

logger_ = getLogger(__name__)


def _get_psycopg_conn_kwargs() -> dict:
    """
    Builds the connection parameters dictionary for psycopg3 from environment variables.

    :return: A dictionary of connection parameters for psycopg.connect.
    """
    kwargs_ = {
        'host': getenv('POSTGRES_HOST', 'localhost'),
        'port': int(getenv('POSTGRES_PORT', '5432')),
        'dbname': getenv('POSTGRES_DB', 'radar'),
        'user': getenv('POSTGRES_USER', 'postgres'),
        'password': getenv('POSTGRES_PASSWORD', ''),
        'sslmode': getenv('POSTGRES_SSL_MODE', 'prefer'),
        'connect_timeout': 10
    }
    options_ = getenv('POSTGRES_OPTIONS', None)
    if options_:
        kwargs_['options'] = parse.unquote(options_)

    return kwargs_


@contextmanager
def get_psycopg_connection() -> Generator[Connection, None, None]:
    """
    Provides an operation-scoped psycopg3 Connection context manager.
    - Connects using settings derived from environment variables.
    - Sets autocommit=False for explicit transaction management.
    - Automatically commits on normal block completion.
    - Automatically rolls back on unhandled exceptions.
    - Deterministically closes connection on exit.

    :return: Yields an active psycopg.Connection object.
    """
    kwargs_ = _get_psycopg_conn_kwargs()
    conn_ = psycopg.connect(**kwargs_, autocommit=False)
    try:
        yield conn_
        conn_.commit()
    except Exception as e_:
        with suppress(Exception):
            conn_.rollback()
        logger_.exception('psycopg3 transaction failed. Rolled back.', exc_info=e_)
        raise
    finally:
        conn_.close()
