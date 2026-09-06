# tests/conftest.py

# --- Python modules ---
import os
from pathlib import Path
import shutil
from unittest.mock import MagicMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from radar_core.settings import Settings, get_settings


@pytest.fixture(scope='session', autouse=True)
def initialize_environment():
    """
    Session fixture to initialize application settings and load environment variables
    from .env before running any pytest test suite.
    Directs the default price cache directory to tests/cache rather than the repository root.
    """
    test_cache_dir_ = Path(__file__).parent / 'cache'
    os.environ['RADAR_PRICE_CACHE_DIR'] = str(test_cache_dir_)
    Settings._reset()
    get_settings()
    yield
    if test_cache_dir_.exists():
        shutil.rmtree(test_cache_dir_, ignore_errors=True)


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