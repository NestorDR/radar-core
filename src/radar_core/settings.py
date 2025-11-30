# src/radar_core/settings.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import ERROR, DEBUG, INFO, WARNING, getLogger
# os: allows access to functionalities dependent on the Operating System
import os
# pathlib: provides an interface to work with file paths in a more readable and easier way than the older 'os.path'.
from pathlib import Path
# sys: provides access to some variables used or maintained by the interpreter and to functions that interact strongly
#      with the interpreter.
import sys

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
    """Application settings manager"""
    _config = None  # Class-level flag to ensure .env & YAML config are loaded only once

    def __init__(self):
        """
        Initializes the settings object.
        - Ensures environment variables from .env are loaded.
        - Reads the main YAML configuration file.
        This method ensures configuration is loaded only once.
        """
        # Preserve the current working path
        if Settings._config is not None:
            return

        # Load environment variables
        self.verbosity_level = INFO
        self.load_env()
        self.log_config = self._get_log_config()
        self.max_workers = self._get_max_workers()

        # Get settings file path from environment variable, or use a default
        module_folder_ = Path(__file__).resolve().parent
        file_path = os.getenv('RADAR_SETTING_FILE', module_folder_ / 'settings.yml')

        # Load YAML settings file
        Settings._config = self._read_yaml_file(file_path)

    def load_env(self) -> None:
        """
        Finds and loads the .env file into the process's environment variables.
        This method is idempotent and will only run once per application lifecycle.
        """
        # Find an .env file in the current or parent directories
        env_path_ = dotenvy_py.find_upwards('.env', 2)
        self.verbosity_level = self._get_log_level()
        if env_path_:
            dotenvy_py.from_filename(env_path_)
            message_ = f"Found and loaded environment file {env_path_}"
            message_verbosity_level_ = DEBUG
        else:
            message_ = "No environment file found. Continuing without loading environment variables."
            message_verbosity_level_ = WARNING

        verbose(message_, message_verbosity_level_, self.verbosity_level)

    @staticmethod
    def _get_log_level() -> int:
        """Returns the log level from RADAR_LOG_LEVEL env var, ensuring a valid numeric value, defaulting to INFO."""
        default_log_level_ = INFO
        env_log_level_ = os.getenv('RADAR_LOG_LEVEL') or str(default_log_level_)

        try:
            log_level_ = int(env_log_level_)

            # Calculate level by flooring to the nearest 10. If outside range 10-59, return default
            return (log_level_ // 10) * 10 if 10 <= log_level_ <= 59 else default_log_level_

        except ValueError:
            return default_log_level_

    def _get_log_config(self) -> dict:
        """
        Generates a declarative log configuration dictionary,
         which will allow or not file logging based on the RADAR_ENABLE_FILE_LOGGING value.

        :return: A dictionary with the logging configuration.
        """
        main_folder_ = Path(__file__).resolve().parent  # radar_core folder
        enable_file_logging_ = os.getenv('RADAR_ENABLE_FILE_LOGGING', 'true').lower() in ('true', '1', 't')
        handlers_ = ["console"]
        config_ = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s; %(name)-45s; %(levelname)-8s; line %(lineno)3d; %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": DEFAULT_CONSOLE_LOG_LEVEL if enable_file_logging_ else self.verbosity_level,
                }
            },
            "root": {
                "level": DEBUG,
                "handlers": handlers_,
            },
        }

        if enable_file_logging_:
            logs_folder_ = main_folder_ / "logs"
            os.makedirs(logs_folder_, exist_ok=True)

            main_file = getattr(sys.modules["__main__"], "__file__", None)  # Get main file of the running stack
            log_filename = Path(main_file).name.removesuffix('.py') if main_file else "app"

            log_file_path_ = logs_folder_ / f'{log_filename}.log'

            config_["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": str(log_file_path_),
                "maxBytes": 524288,
                "backupCount": 3,
                "level": self.verbosity_level,
            }
            handlers_.append("file")

        return config_

    def _get_max_workers(self) -> int:
        """
        Returns number of parallel workers from RADAR_MAX_WORKERS env var.
        Defaults to 0 (auto-detect all cores) if not set, invalid, or non-positive.
        """
        env_max_workers_ = os.getenv('RADAR_MAX_WORKERS') or '0'

        try:
            max_workers_ = int(env_max_workers_)

            # Return the value only if it's a positive integer, otherwise 0.
            return max_workers_ if max_workers_ > 0 else 0

        except ValueError:
            message_ = f"Invalid value for RADAR_MAX_WORKERS: '{env_max_workers_}'. Must be an integer. Defaulting to all available cores."
            verbose(message_, WARNING, self.verbosity_level)
            logger_.warning(message_)
            return 0

    def _read_yaml_file(self, file_path):
        message_ = f'Reading YAML file {file_path}...'
        verbose(message_, INFO, self.verbosity_level)
        logger_.info(message_)

        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)

        except yaml.YAMLError as e:
            # Log error
            message_ = f'Error reading YAML file {file_path}.'
            verbose(message_, ERROR, self.verbosity_level)
            logger_.exception(message_, exc_info=e)
            return None

        except FileNotFoundError as e:
            # Log error
            message_ = f'Settings file not found at {file_path}. Please check the SETTING_FILE environment variable or ensure settings.yml exists.'
            verbose(message_, ERROR, self.verbosity_level)
            logger_.exception(message_, exc_info=e)
            raise FileNotFoundError(message_) from e

    def get_symbols(self) -> list[str]:
        return self._config.get('symbols', []) if self._config else []

    def get_undeletable(self) -> list[str]:
        done_list_ = [] if self._config is None else self._config.get('done', []) or []
        return done_list_ + self.get_symbols()

    def get_shortables(self) -> list[str]:
        # Get the symbol list and convert it to a set for more efficient search
        symbols_set_ = set(self.get_symbols())

        # Get the 'raw' shortables list
        raw_shortables_ = self._config.get('shortables', [])

        # Filter shortables: only those that are also in the symbol set, and return the result as a list
        # Use list comprehension: [expresión for item in iterable if condición]
        return [shortable_ for shortable_ in raw_shortables_ if shortable_ in symbols_set_]

    def get_evaluable_strategies(self) -> list[int]:
        return self._config.get('evaluable_strategies', []) if self._config else []

# Importable singleton
# settings = Settings()
