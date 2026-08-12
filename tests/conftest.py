# tests/conftest.py

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
