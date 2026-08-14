# tests/test_psycopg_connection.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from radar_core.database import _get_psycopg_conn_kwargs, get_psycopg_connection
from radar_core.settings import Settings


def test_get_psycopg_conn_kwargs_formatting():
    """
    GIVEN environment variables set for database connection
    WHEN _get_psycopg_conn_kwargs is called
    THEN it formats a valid connection dictionary for psycopg3.
    """
    kwargs_ = _get_psycopg_conn_kwargs()
    assert 'host' in kwargs_
    assert 'port' in kwargs_
    assert 'dbname' in kwargs_
    assert kwargs_['connect_timeout'] == 10



def test_get_psycopg_connection_commit_on_success():
    """
    GIVEN a successful context block
    WHEN get_psycopg_connection is executed
    THEN it yields the connection, calls commit on exit, and closes the connection.
    """
    mock_conn_ = MagicMock()

    with patch('psycopg.connect', return_value=mock_conn_) as mock_connect_:
        with get_psycopg_connection() as conn_:
            assert conn_ == mock_conn_
            mock_conn_.execute('SELECT 1')

        mock_connect_.assert_called_once()
        mock_conn_.commit.assert_called_once()
        mock_conn_.rollback.assert_not_called()
        mock_conn_.close.assert_called_once()


def test_get_psycopg_connection_rollback_on_error():
    """
    GIVEN an exception raised inside the context block
    WHEN get_psycopg_connection is executed
    THEN it rolls back the transaction, closes the connection, and re-raises the exception.
    """
    mock_conn_ = MagicMock()
    error_message_ = 'Simulated SQL execution error'

    with patch('psycopg.connect', return_value=mock_conn_):
        with pytest.raises(RuntimeError, match=error_message_):
            with get_psycopg_connection():
                raise RuntimeError(error_message_)

        mock_conn_.commit.assert_not_called()
        mock_conn_.rollback.assert_called_once()
        mock_conn_.close.assert_called_once()


@pytest.mark.integration
def test_get_psycopg_connection_live_query():
    """
    GIVEN an active PostgreSQL server container
    WHEN get_psycopg_connection is used to execute a live query
    THEN it executes successfully and returns expected query result.
    """
    Settings()
    with get_psycopg_connection() as conn_:
        with conn_.cursor() as cursor_:
            cursor_.execute('SELECT 1 AS num')
            row_ = cursor_.fetchone()
            assert row_ is not None
            assert row_[0] == 1
