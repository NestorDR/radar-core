# tests/test_settings.py

# --- Python modules ---

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from radar_core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_settings_state():
    """
    Ensures that Settings singleton state is cleanly reset before and after each test.
    """
    Settings._reset()
    yield
    Settings._reset()


def test_singleton_identity():
    """
    GIVEN the Settings class and get_settings accessor
    WHEN both are called repeatedly in the current process
    THEN they return the exact same singleton instance object reference.
    """
    s1_ = get_settings()
    s2_ = Settings()
    s3_ = get_settings()

    assert s1_ is s2_
    assert s2_ is s3_
    assert s1_.verbosity_level is not None
    assert s1_.log_config is not None
    assert s1_.db_conn_kwargs is not None


def test_singleton_attributes_preserved_on_repeated_calls():
    """
    GIVEN an already initialized Settings singleton
    WHEN Settings() or get_settings() is called again
    THEN instance attributes such as verbosity_level, max_workers, and clean_unlisted remain intact.
    """
    s1_ = get_settings()
    level_ = s1_.verbosity_level
    workers_ = s1_.max_workers

    s2_ = Settings()
    assert s2_.verbosity_level == level_
    assert s2_.max_workers == workers_
    assert s2_.clean_unlisted is False or s2_.clean_unlisted is True


def test_initialization_retry_on_failure(monkeypatch):
    """
    GIVEN a failure during the first initialization attempt
    WHEN Settings() raises an exception
    THEN no partial instance is published and a later call can successfully retry.
    """
    monkeypatch.setenv('RADAR_SETTING_FILE', 'non_existent_settings_file_12345.yml')

    with pytest.raises(FileNotFoundError):
        get_settings()

    assert Settings._instance is None

    # Fix setting file path back to default and retry
    monkeypatch.delenv('RADAR_SETTING_FILE', raising=False)
    s_ = get_settings()
    assert s_ is not None
    assert Settings._instance is s_


def test_environment_variable_parsing(monkeypatch):
    """
    GIVEN custom RADAR_* environment variables
    WHEN get_settings() is initialized
    THEN it correctly parses verbosity level, max_workers, and boolean flags.
    """
    monkeypatch.setenv('RADAR_ENV', 'test')
    monkeypatch.setenv('RADAR_LOG_LEVEL', '20')
    monkeypatch.setenv('RADAR_MAX_WORKERS', '8')
    monkeypatch.setenv('RADAR_CLEAN_UNLISTED', 'true')
    monkeypatch.setenv('RADAR_ENABLE_FILE_LOGGING', 'true')

    s_ = get_settings()

    assert s_.verbosity_level == 20
    assert s_.max_workers == 8
    assert s_.clean_unlisted is True


def test_database_connection_kwargs_builder(monkeypatch):
    """
    GIVEN custom POSTGRES_* environment variables including encoded options
    WHEN get_settings() is initialized
    THEN db_conn_kwargs correctly builds connection parameters with decoded options.
    """
    monkeypatch.setenv('RADAR_ENV', 'test')
    monkeypatch.setenv('POSTGRES_HOST', 'db.internal')
    monkeypatch.setenv('POSTGRES_PORT', '5434')
    monkeypatch.setenv('POSTGRES_DB', 'radar_test')
    monkeypatch.setenv('POSTGRES_USER', 'test_user')
    monkeypatch.setenv('POSTGRES_PASSWORD', 'test_pass')
    monkeypatch.setenv('POSTGRES_SSL_MODE', 'require')
    monkeypatch.setenv('POSTGRES_OPTIONS', '-c%20statement_timeout%3D3000')

    s_ = get_settings()
    kwargs_ = s_.db_conn_kwargs

    assert kwargs_['host'] == 'db.internal'
    assert kwargs_['port'] == 5434
    assert kwargs_['dbname'] == 'radar_test'
    assert kwargs_['user'] == 'test_user'
    assert kwargs_['password'] == 'test_pass'  # noqa: S105
    assert kwargs_['sslmode'] == 'require'
    assert kwargs_['options'] == '-c statement_timeout=3000'
    assert kwargs_['connect_timeout'] == 10


def test_yaml_accessors():
    """
    GIVEN standard settings.yml file
    WHEN get_settings() is initialized
    THEN symbols, shortables, evaluable_strategies, and undeletable lists are accessible.
    """
    s_ = get_settings()

    assert isinstance(s_.get_symbols(), list)
    assert isinstance(s_.get_shortables(), list)
    assert isinstance(s_.get_evaluable_strategies(), list)
    assert isinstance(s_.get_undeletable(), list)

    assert set(s_.get_shortables()).issubset(set(s_.get_symbols()))
    assert all(sym_ in s_.get_undeletable() for sym_ in s_.get_symbols())


def test_reset_clears_singleton():
    """
    GIVEN an initialized Settings singleton
    WHEN Settings._reset() is invoked
    THEN Settings._instance is cleared and subsequent access creates a new instance.
    """
    _ = get_settings()
    Settings._reset()
    assert Settings._instance is None

    settings_ = get_settings()
    assert settings_ is not None
    assert Settings._instance is settings_
