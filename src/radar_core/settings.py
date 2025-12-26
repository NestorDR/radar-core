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
    _config: dict | None = None  # Class-level flag to ensure .env & YAML config are loaded only once

    def __init__(self,
                 log_filename: str | None = None):
        """
        Initializes the settings object, ensuring configuration is loaded only once.
        - Ensures environment variables from .env are loaded.
        - Reads the main YAML configuration file.

        :param log_filename: Name to the log file.
        """
        # Preserve the current working path
        if Settings._config is not None:
            return

        self.verbosity_level = INFO  # Set the default verbosity level
        self.module_folder = Path(__file__).resolve().parent

        # Load environment variables
        self.load_env()
        self.log_config = self._get_log_config(log_filename)
        self.max_workers = self._get_max_workers()

        # Load YAML settings file
        Settings._config = self._read_yaml_file()

    def load_env(self) -> None:
        """
        Finds and loads the .env file into the process's environment variables.
        This method is idempotent and will only run once per application lifecycle.
        """
        # Find an .env file in the current or parent directories
        env_path_ = dotenvy_py.find_upwards(str(self.module_folder / '.env'), 2)
        self.verbosity_level = self._get_log_level()
        if env_path_:
            dotenvy_py.from_filename(env_path_)
            message_ = f"Found and loaded environment vars file {env_path_}"
            message_verbosity_level_ = DEBUG
        else:
            message_ = f"No environment vars file found ({env_path_}), Continuing without it."
            message_verbosity_level_ = WARNING

        verbose(message_, message_verbosity_level_, self.verbosity_level)

    @staticmethod
    def _get_log_level() -> int:
        """
        Determines and returns the appropriate logging level based on the environment variable
        'RADAR_LOG_LEVEL', or defaults to the predefined INFO level if the variable is unset or invalid.

        :return: The obtained or default logging level.
        """
        default_log_level_ = INFO
        env_log_level_ = os.getenv('RADAR_LOG_LEVEL') or str(default_log_level_)

        try:
            log_level_ = int(env_log_level_)

            # Calculate level by flooring to the nearest 10. If outside range 10-59, return default
            return (log_level_ // 10) * 10 if 10 <= log_level_ <= 59 else default_log_level_

        except ValueError:
            return default_log_level_

    def _get_log_config(self,
                        log_filename: str | None = None) -> dict:
        """
        Generates a declarative log configuration dictionary,
         which will allow or not file logging based on the RADAR_ENABLE_FILE_LOGGING value.

        :param log_filename: Name to the log file.

        :return: A dictionary with the logging configuration.
        """
        main_folder_ = Path(__file__).resolve().parent  # radar_core folder
        enable_file_logging_ = os.getenv('RADAR_ENABLE_FILE_LOGGING', 'true').lower() in ('true', '1', 't')
        handlers_ = ["console"]
        config_: dict = {
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

            if not log_filename:
                main_file = getattr(sys.modules["__main__"], "__file__", None)  # Get the main file of the running stack
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
        Retrieves the maximum number of workers based on the RADAR_MAX_WORKERS env var.
        If the value is a positive integer, it is returned; otherwise, the default value is 0.
        Handles invalid values gracefully by logging a warning message.

        :return: The maximum number of workers based on the environment variable,
            or 0 if the value is not a positive integer or invalid.
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

    def _read_yaml_file(self) -> dict | None:
        """
        Reads and parses a YAML file, converting it into a Python object. Handles errors gracefully.

        :return: A dictionary representation of the parsed YAML file. If there is an error during parsing, None is returned.
        """
        # Get the settings file path from the environment variable or use a default
        file_path_ = self.module_folder / os.getenv('RADAR_SETTING_FILE', 'settings.yml')
        message_ = f'Reading YAML file {file_path_}...'
        verbose(message_, INFO, self.verbosity_level)
        logger_.info(message_)

        try:
            with open(file_path_, 'r') as file:
                return yaml.safe_load(file)

        except yaml.YAMLError as e:
            # Log error
            message_ = f'Error reading YAML file {file_path_}.'
            verbose(message_, ERROR, self.verbosity_level)
            logger_.exception(message_, exc_info=e)
            return None

        except FileNotFoundError as e:
            # Log error
            message_ = f'Settings file not found at {file_path_}. Please check the SETTING_FILE environment variable or ensure settings.yml exists.'
            verbose(message_, ERROR, self.verbosity_level)
            logger_.exception(message_, exc_info=e)
            raise FileNotFoundError(message_) from e

    def get_symbols(self) -> list[str]:
        """Returns the list of symbols to analyze."""
        return self._config.get('symbols', []) if self._config else []

    def get_undeletable(self) -> list[str]:
        """Returns the list of symbols that cannot be deleted from the database."""
        done_list_ = [] if self._config is None else self._config.get('done', []) or []
        return done_list_ + self.get_symbols()

    def get_shortables(self) -> list[str]:
        """Returns the list of symbols that can be shorted."""
        # Get the symbol list and convert it to a set for more efficient search
        symbols_set_ = set(self.get_symbols())

        # Get the 'raw' shortables list
        raw_shortables_ = self._config.get('shortables', [])

        # Filter shortables: only those that are also in the symbol set, and return the result as a list
        # Use list comprehension: [expression for item in iterable if condición]
        return [shortable_ for shortable_ in raw_shortables_ if shortable_ in symbols_set_]

    def get_evaluable_strategies(self) -> list[str]:
        """Returns the list of strategy Acronyms that can be evaluated."""
        return self._config.get('evaluable_strategies', []) if self._config else []
