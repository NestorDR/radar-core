# tests/conftest.py

# --- Python modules ---
from unittest.mock import MagicMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from radar_core.settings import Settings


@pytest.fixture(scope="session", autouse=True)
def initialize_environment():
    """
    Session fixture to initialize application settings and load environment variables
    from .env before running any pytest test suite.
    """
    Settings()

@pytest.fixture
def mock_connection_scope() -> tuple[MagicMock, MagicMock, MagicMock]:
    """
    Creates an isolated mocked database connection scope.

    :return: A tuple containing the database connection, cursor, and scope.
    """
    connection_ = MagicMock(name='connection')
    cursor_ = MagicMock(name='cursor')

    connection_.cursor.return_value.__enter__.return_value = cursor_

    scope_ = MagicMock(name='connection_scope')
    scope_.__enter__.return_value = connection_
    scope_.__exit__.return_value = False

    return connection_, cursor_, scope_