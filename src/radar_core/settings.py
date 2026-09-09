# src/radar_core/settings.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
from datetime import time
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import ERROR, DEBUG, INFO, WARNING, getLogger
# os: provides operating system interfaces and functionality
import os
# pathlib: provides an interface to work with file paths in a more readable and easier way than the older 'os.path'.
from pathlib import Path
# sys: provides access to some variables used or maintained by the interpreter and to functions that interact strongly
#      with the interpreter.
import sys
# urllib.parse: provides URL parsing and unquoting facilities
from urllib import parse
# zoneinfo: provides concrete time zone implementations representing the system's time zones.
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# --- Third Party Libraries ---
# dotenvy-py: loads environment variables from .env files (first occurrence wins)
import dotenvy_py
# pyyaml: is a YAML parser and emitter
import yaml

# --- App modules ---
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.log_helper import verbose, DEFAULT_CONSOLE_LOG_LEVEL

logger_ = getLogger(__name__)


class Settings:
    """Application settings manager (Lazy Singleton)."""

    _instance: 'Settings | None' = None

    def __new__(cls,
                log_filename: str | None = None) -> 'Settings':
        """
        Guarantees that every instantiation returns the same initialized singleton instance.

        :param log_filename: Optional log file name passed on first initialization.
        :return: The process-local singleton Settings instance.
        """
        if cls._instance is None:
            instance_ = super().__new__(cls)
            instance_._initialize(log_filename)
            cls._instance = instance_
        return cls._instance

    def __init__(self,
                 log_filename: str | None = None) -> None:
        """
        Initialization is a no-op because singleton construction is performed in __new__.

        :param log_filename: Optional name for the log file.
        """
        pass

    def _initialize(self,
                    log_filename: str | None = None) -> None:
        """
        Initializes the configuration snapshot, loading environment variables,
        PostgreSQL connection parameters, and YAML configuration.

        :param log_filename: Name for the log file.
        """
        self._module_folder = Path(__file__).resolve().parent

        # Load environment variables
        self.verbosity_level = self._initialize_environment()
        self.app_environment = (os.getenv('RADAR_ENV') or 'dev').strip().lower()
        self.clean_unlisted = self._parse_bool('RADAR_CLEAN_UNLISTED', False)
        self.log_config = self._get_log_config(log_filename)
        self.max_workers = self._get_max_workers()

        # Database connection parameters
        self.db_conn_kwargs = self._get_db_conn_kwargs()

        # Price cache parameters
        self.price_cache_kwargs = self._get_price_cache_kwargs()

        # Load YAML settings file
        config_ = self._read_yaml_file() or {}
        self.symbols: list[str] = config_.get('symbols', [])
        raw_shortables_ = config_.get('shortables', [])
        symbols_set_ = set(self.symbols)
        self.shortables: list[str] = [s for s in raw_shortables_ if s in symbols_set_]
        self.undeletable_symbols: list[str] = config_.get('done', []) or []
        self.undeletable_symbols += self.symbols
        self.evaluable_strategies: list[str] = config_.get('evaluable_strategies', [])
        self.rsi_input_filter: str | None = config_.get('rsi_input_filter', 'none')

    @classmethod
    def _reset(cls) -> None:
        """
        Resets the singleton instance for isolated test fixtures.
        """
        cls._instance = None

    # region Environment Variables

    def _initialize_environment(self) -> int:
        """
        If the RADAR_ENV environment variable is 'dev', loads the .env file and configures the log/verbosity level.
        Otherwise, it sets the verbosity level based on the variable from the real environment or default.

        :return: The resolved logging verbosity level integer.
        """
        message_verbosity_level_ = DEBUG
        if (os.getenv('RADAR_ENV') or 'dev') == 'dev':
            # Attempts to find an .env file in the current directory or any of its parent dirs up to 2 levels.
            env_path_ = dotenvy_py.find_upwards(str(self._module_folder / '.env'), 2)
            if env_path_:
                # Load the environment variables from the file
                dotenvy_py.from_filename(env_path_)
                message_ = f'Found and loaded environment vars file {env_path_}'
            else:
                # Carry on without loading and sets the verbosity level based on the real environment or default.
                message_ = f'No environment variables file found ({env_path_}), Continuing without it.'
                message_verbosity_level_ = WARNING
        else:
            message_ = f'RADAR_ENV is in mode `{os.getenv('RADAR_ENV')}`.'

        # Set logging/verbosity level based on the env vars from the real environment or `.env` file read
        verbosity_level_ = self._get_log_level()
        # Display result
        verbose(message_, message_verbosity_level_, verbosity_level_)

        return verbosity_level_

    @staticmethod
    def _get_log_level() -> int:
        """
        Determines and returns the appropriate logging level based on the environment variable
        'RADAR_LOG_LEVEL', or defaults to the predefined INFO level if the variable is unset or invalid.

        :return: The obtained or default logging level.
        """
        env_log_level_ = os.getenv('RADAR_LOG_LEVEL')
        try:
            log_level_ = int(env_log_level_) if env_log_level_ else INFO
            # Calculate THE level by flooring to the nearest 10. If outside range 10-59, return default
            return (log_level_ // 10) * 10 if 10 <= log_level_ <= 59 else INFO

        except ValueError:
            return INFO

    def _get_log_config(self, log_filename: str | None = None) -> dict:
        """
        Generates a declarative log configuration dictionary,
        which will allow or not file logging based on the RADAR_ENABLE_FILE_LOGGING value.

        :param log_filename: Name to the log file.

        :return: A dictionary with the logging configuration.
        """
        enable_file_logging_ = self._parse_bool('RADAR_ENABLE_FILE_LOGGING', False)
        handlers_ = ['console']

        logger_config_ = {
            'level': self.verbosity_level,
            'handlers': [],
            'propagate': True,
        }
        loggers_ = ['radar-core', 'numba', 'numpy', 'peewee', 'polars', 'psycopg', 'yfinance']

        config_: dict = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(asctime)s; %(name)-45s; %(levelname)-8s; line %(lineno)3d; %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'level': DEFAULT_CONSOLE_LOG_LEVEL if enable_file_logging_ else self.verbosity_level,
                }
            },
            'root': {
                'level': DEBUG,
                'handlers': handlers_,
            },
            'loggers': {x_: logger_config_.copy() for x_ in loggers_}
        }

        config_['loggers']['numba']['level'] = 'WARNING'
        config_['loggers']['numpy']['level'] = 'WARNING'
        config_['loggers']['peewee']['level'] = 'WARNING'
        config_['loggers']['polars']['level'] = 'WARNING'
        config_['loggers']['psycopg']['level'] = 'WARNING'
        config_['loggers']['yfinance']['level'] = 'WARNING'

        if enable_file_logging_:
            log_folder_name_ = os.getenv('RADAR_LOG_FOLDER') or 'logs'
            log_folder_name_ = os.path.expandvars(log_folder_name_)

            # Build the log folder path (with `pathlib.Path` the result depends on whether `log_folder_name_` is relative or absolute.)
            # - relative paths are resolved from the module folder,
            # - while absolute paths are preserved.
            logs_folder_ = self._module_folder / Path(log_folder_name_).expanduser()
            logs_folder_.mkdir(parents=True, exist_ok=True)

            if not log_filename:
                # Get the main file of the running stack
                main_module_ = sys.modules['__main__']
                main_file_ = getattr(main_module_, '__file__', None)
                # stem: final component of the path without extension
                log_filename = Path(str(main_file_)).stem if main_file_ else 'app'

            log_file_path_ = logs_folder_ / f'{log_filename}.log'

            config_['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': str(log_file_path_),
                'maxBytes': 1024 * 1024,
                'backupCount': 12,
                'level': self.verbosity_level,
            }
            handlers_.append('file')

        return config_

    def _get_max_workers(self) -> int:
        """
        Retrieves the maximum number of workers based on the RADAR_MAX_WORKERS env var.
        If the value is a positive integer, it is returned; otherwise, the default value is 0.
        Handles invalid values gracefully by logging a warning message.

        :return: The maximum number of workers based on the environment variable,
            or 0 if the value is not a positive integer or invalid.
        """
        env_max_workers_ = os.getenv('RADAR_MAX_WORKERS') or '1'

        try:
            # Return the value only if it's a positive integer, otherwise 0.
            return max(int(env_max_workers_), 1)

        except ValueError:
            message_ = f'Invalid value for RADAR_MAX_WORKERS: `{env_max_workers_}`. Must be an integer. Defaulting to all available cores.'
            verbose(message_, WARNING, self.verbosity_level)
            logger_.warning(message_)
            return 1

    @staticmethod
    def _get_db_conn_kwargs() -> dict:
        """
        Builds the connection parameters dictionary for psycopg3 from environment variables.

        :return: A dictionary of connection parameters for psycopg.connect.
        """
        kwargs_: dict = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'dbname': os.getenv('POSTGRES_DB', 'radar'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
            'sslmode': os.getenv('POSTGRES_SSL_MODE', 'prefer'),
            'connect_timeout': 10
        }
        options_ = os.getenv('POSTGRES_OPTIONS', None)
        if options_:
            kwargs_['options'] = parse.unquote(options_)

        return kwargs_

    @staticmethod
    def _get_price_cache_kwargs() -> dict:
        """
        Builds the configuration dictionary for the price cache from environment variables.

        :return: A dictionary containing the parsed price cache configuration.
        """
        return {
            'dir': Settings._parse_dir('RADAR_PRICE_CACHE_DIR', '/home/default/app/cache'),
            'enabled': Settings._parse_bool('RADAR_PRICE_CACHE_ENABLED', True),
            'ignore': Settings._parse_bool('RADAR_PRICE_CACHE_IGNORE', False),
            'write': Settings._parse_bool('RADAR_PRICE_CACHE_WRITE', True),
            'timezone': Settings._parse_timezone('RADAR_PRICE_CACHE_TIMEZONE', 'America/New_York'),
            'trading_start': Settings._parse_time('RADAR_PRICE_CACHE_TRADING_START', '09:30'),
            'dev_max_age_minutes': Settings._parse_int('RADAR_PRICE_CACHE_DEV_MAX_AGE_MINUTES', 10),
        }

    # endregion Environment Variables

    # region Parsers

    @staticmethod
    def _parse_bool(env_var: str,
                    default: bool = False) -> bool:
        """
        Parse a boolean environment variable.

        :param env_var: Name of the environment variable.
        :param default: Default value if not set.

        :return: Boolean value.
        """
        return os.getenv(env_var, str(default)).strip().lower() in ('true', '1', 't')

    @staticmethod
    def _parse_int(env_var: str,
                   default: int = 0) -> int:
        """
        Parses an integer environment variable.

        :param env_var: Name of the environment variable.
        :param default: Default integer value if not set.

        :return: Parsed integer.
        :raises ValueError: If the value cannot be parsed as an integer.
        """
        value_ = os.getenv(env_var, str(default)).strip()
        try:
            return int(value_)
        except ValueError as e_:
            raise ValueError(f'Invalid integer value for {env_var}: "{value_}".') from e_

    @staticmethod
    def _parse_time(env_var: str,
                    default: str) -> time:
        """
        Parses a time environment variable in HH:MM format.

        :param env_var: Name of the environment variable.
        :param default: Default time string (HH:MM) if not set.

        :return: Parsed time object.
        :raises ValueError: If the supplied value is not in a valid HH:MM format.
        """
        value_ = os.getenv(env_var, default).strip()
        try:
            return time.fromisoformat(value_)
        except ValueError as e_:
            raise ValueError(f'Invalid time format for {env_var}: "{value_}". Expected HH:MM.') from e_

    @staticmethod
    def _parse_timezone(env_var: str,
                        default: str = 'America/New_York') -> ZoneInfo:
        """
        Parses and validates an IANA timezone from an environment variable.

        :param env_var: Name of the environment variable.
        :param default: Default IANA timezone name if not set.

        :return: Validated ZoneInfo instance.
        :raises ValueError: If the timezone cannot be found or is empty.
        """
        value_ = os.getenv(env_var, default).strip()
        try:
            return ZoneInfo(value_)
        except (ZoneInfoNotFoundError, ValueError) as e_:
            raise ValueError(f'Invalid timezone for {env_var}: "{value_}".') from e_

    @staticmethod
    def _parse_dir(env_var: str,
                   default: str) -> Path:
        """
        Parses and validates a directory path from an environment variable.

        :param env_var: Name of the environment variable.
        :param default: Default path to use if the environment variable is not set.

        :return: Path instance for the directory.
        :raises ValueError: If the directory setting is empty.
        """
        value_ = os.getenv(env_var, default).strip()
        if not value_:
            raise ValueError(f'{env_var} cannot be empty.')

        return Path(value_)

    # endregion Parsers

    # region YAML Settings File

    def _read_yaml_file(self) -> dict | None:
        """
        Reads and parses a YAML file, converting it into a Python object. Handles errors gracefully.

        :return: A dictionary representation of the parsed YAML file. If there is an error during parsing, None is returned.
        :raises FileNotFoundError: If the YAML settings file does not exist.
        """
        # Get the settings file path from the environment variable or use a default
        file_name_ = os.getenv('RADAR_SETTING_FILE', 'settings.yml')

        # Build the settings file path (with `pathlib.Path` the result depends on whether `file_name_` is relative or absolute.)
        # - relative paths are resolved from the module folder,
        # - while absolute paths are preserved.
        file_path_ = self._module_folder / file_name_
        message_ = f'Reading YAML file {file_path_}...'
        verbose(message_, INFO, self.verbosity_level)
        logger_.info(message_)

        try:
            with open(file_path_, 'r') as file_:
                return yaml.safe_load(file_)

        except yaml.YAMLError as e_:
            # Log error
            message_ = f'Error reading YAML file {file_path_}.'
            verbose(message_, ERROR, self.verbosity_level)
            logger_.exception(message_, exc_info=e_)
            return None

        except FileNotFoundError as e_:
            # Log error
            message_ = f'Settings file not found at {file_path_}. Please check the SETTING_FILE environment variable or ensure settings.yml exists.'
            verbose(message_, ERROR, self.verbosity_level)
            logger_.exception(message_, exc_info=e_)
            raise FileNotFoundError(message_) from e_

    # endregion YAML Settings File


def get_settings(log_filename: str | None = None) -> Settings:
    """
    Returns the singleton Settings instance for the current process.

    :param log_filename: Optional log file name passed on first initialization.
    :return: The shared Settings singleton object.
    """
    return Settings(log_filename=log_filename)
