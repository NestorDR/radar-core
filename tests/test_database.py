# tests/test_database.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from radar_core.database import (
    _get_psycopg_conn_kwargs,
    connection_scope,
    get_psycopg_connection,
    get_psycopg_read_connection,
    read_connection_scope,
)
from radar_core.settings import Settings


def test_get_psycopg_conn_kwargs_default(monkeypatch):
    """
    GIVEN default environment variables without POSTGRES_OPTIONS
    WHEN _get_psycopg_conn_kwargs is called
    THEN it formats a valid connection dictionary for psycopg3 without options key.
    """
    monkeypatch.delenv('POSTGRES_OPTIONS', raising=False)
    kwargs_ = _get_psycopg_conn_kwargs()
    assert 'host' in kwargs_
    assert 'port' in kwargs_
    assert 'dbname' in kwargs_
    assert 'user' in kwargs_
    assert 'password' in kwargs_
    assert 'sslmode' in kwargs_
    assert kwargs_['connect_timeout'] == 10
    assert 'options' not in kwargs_


def test_get_psycopg_conn_kwargs_with_custom_options(monkeypatch):
    """
    GIVEN environment variables including encoded POSTGRES_OPTIONS
    WHEN _get_psycopg_conn_kwargs is called
    THEN it unquotes the options string and sets custom connection parameters.
    """
    monkeypatch.setenv('POSTGRES_HOST', 'custom-db-host')
    monkeypatch.setenv('POSTGRES_PORT', '5433')
    monkeypatch.setenv('POSTGRES_DB', 'custom_db')
    monkeypatch.setenv('POSTGRES_USER', 'custom_user')
    monkeypatch.setenv('POSTGRES_PASSWORD', 'custom_secret')
    monkeypatch.setenv('POSTGRES_SSL_MODE', 'require')
    monkeypatch.setenv('POSTGRES_OPTIONS', '-c%20statement_timeout%3D5000')

    kwargs_ = _get_psycopg_conn_kwargs()

    assert kwargs_['host'] == 'custom-db-host'
    assert kwargs_['port'] == 5433
    assert kwargs_['dbname'] == 'custom_db'
    assert kwargs_['user'] == 'custom_user'
    assert kwargs_['password'] == 'custom_secret'  # noqa: S105
    assert kwargs_['sslmode'] == 'require'
    assert kwargs_['options'] == '-c statement_timeout=5000'


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
        _, kwargs_ = mock_connect_.call_args
        assert kwargs_['autocommit'] is False
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


def test_get_psycopg_connection_suppresses_rollback_failure_on_error():
    """
    GIVEN an exception raised inside the context block and rollback itself fails
    WHEN get_psycopg_connection is executed
    THEN it suppresses the rollback error, closes the connection, and re-raises the primary exception.
    """
    mock_conn_ = MagicMock()
    mock_conn_.rollback.side_effect = RuntimeError('Rollback failed')
    error_message_ = 'Original query failure'

    with patch('psycopg.connect', return_value=mock_conn_):
        with pytest.raises(RuntimeError, match=error_message_):
            with get_psycopg_connection():
                raise RuntimeError(error_message_)

        mock_conn_.commit.assert_not_called()
        mock_conn_.rollback.assert_called_once()
        mock_conn_.close.assert_called_once()


def test_connection_scope_reuses_supplied_connection():
    """
    GIVEN an active caller-supplied connection
    WHEN connection_scope is executed with the connection
    THEN it yields the connection without committing, rolling back, or closing it.
    """
    supplied_conn_ = MagicMock(name='supplied_connection')

    with patch('radar_core.database.get_psycopg_connection') as mock_get_conn_:
        with connection_scope(supplied_conn_) as conn_:
            assert conn_ == supplied_conn_

        mock_get_conn_.assert_not_called()
        supplied_conn_.commit.assert_not_called()
        supplied_conn_.rollback.assert_not_called()
        supplied_conn_.close.assert_not_called()


def test_connection_scope_creates_new_connection_when_none(mock_connection_scope):
    """
    GIVEN a None connection argument
    WHEN connection_scope is executed
    THEN it opens and yields a new write connection from get_psycopg_connection.
    """
    expected_conn_, _, scope_ = mock_connection_scope

    with patch('radar_core.database.get_psycopg_connection', return_value=scope_) as mock_get_conn_:
        with connection_scope(None) as conn_:
            assert conn_ == expected_conn_

        mock_get_conn_.assert_called_once_with()
        scope_.__enter__.assert_called_once()
        scope_.__exit__.assert_called_once()


def test_get_psycopg_read_connection_yields_and_closes_on_success():
    """
    GIVEN a successful context block
    WHEN get_psycopg_read_connection is executed
    THEN it connects with autocommit=True, yields the connection, and closes it on exit.
    """
    mock_conn_ = MagicMock()

    with patch('psycopg.connect', return_value=mock_conn_) as mock_connect_:
        with get_psycopg_read_connection() as conn_:
            assert conn_ == mock_conn_
            mock_conn_.execute('SELECT 1')

        mock_connect_.assert_called_once()
        _, kwargs_ = mock_connect_.call_args
        assert kwargs_['autocommit'] is True
        mock_conn_.commit.assert_not_called()
        mock_conn_.rollback.assert_not_called()
        mock_conn_.close.assert_called_once()


def test_get_psycopg_read_connection_closes_on_error():
    """
    GIVEN an exception raised inside the read context block
    WHEN get_psycopg_read_connection is executed
    THEN it closes the connection and re-raises the exception.
    """
    mock_conn_ = MagicMock()
    error_message_ = 'Simulated read query error'

    with patch('psycopg.connect', return_value=mock_conn_):
        with pytest.raises(RuntimeError, match=error_message_):
            with get_psycopg_read_connection():
                raise RuntimeError(error_message_)

        mock_conn_.commit.assert_not_called()
        mock_conn_.rollback.assert_not_called()
        mock_conn_.close.assert_called_once()


def test_read_connection_scope_reuses_supplied_connection():
    """
    GIVEN an active caller-supplied read connection
    WHEN read_connection_scope is executed with the connection
    THEN it yields the connection without closing it.
    """
    supplied_conn_ = MagicMock(name='supplied_read_connection')

    with patch('radar_core.database.get_psycopg_read_connection') as mock_get_read_conn_:
        with read_connection_scope(supplied_conn_) as conn_:
            assert conn_ == supplied_conn_

        mock_get_read_conn_.assert_not_called()
        supplied_conn_.commit.assert_not_called()
        supplied_conn_.rollback.assert_not_called()
        supplied_conn_.close.assert_not_called()


def test_read_connection_scope_creates_new_connection_when_none(mock_connection_scope):
    """
    GIVEN a None connection argument
    WHEN read_connection_scope is executed
    THEN it opens and yields a new read connection from get_psycopg_read_connection.
    """
    expected_conn_, _, scope_ = mock_connection_scope

    with patch('radar_core.database.get_psycopg_read_connection', return_value=scope_) as mock_get_read_conn_:
        with read_connection_scope(None) as conn_:
            assert conn_ == expected_conn_

        mock_get_read_conn_.assert_called_once_with()
        scope_.__enter__.assert_called_once()
        scope_.__exit__.assert_called_once()


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


@pytest.mark.integration
def test_get_psycopg_read_connection_live_query():
    """
    GIVEN an active PostgreSQL server container
    WHEN get_psycopg_read_connection is used to execute a live query
    THEN it executes successfully and returns expected query result in autocommit mode.
    """
    Settings()
    with get_psycopg_read_connection() as conn_:
        with conn_.cursor() as cursor_:
            cursor_.execute('SELECT 1 AS num')
            row_ = cursor_.fetchone()
            assert row_ is not None
            assert row_[0] == 1
