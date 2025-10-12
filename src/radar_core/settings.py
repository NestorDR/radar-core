# src/radar_core/settings.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import ERROR, DEBUG, INFO, WARNING, getLogger
# os: allows access to functionalities dependent on the Operating System
import os
# pathlib: provides an interface to work with file paths in a more readable and easier way than the older 'os.path'.
from pathlib import Path

# --- Third Party Libraries ---
# dotenvy-py: loads environment variables from .env files (first occurrence wins)
import dotenvy_py
# pyyaml: is a YAML parser and emitter
import yaml

# --- App modules ---
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.log_helper import get_verbosity_level, verbose

logger_ = getLogger(__name__)


class Settings:
    """Application settings manager"""
    _env_loaded = False  # Class-level flag to ensure .env is loaded only once

    def __init__(self):
        """
        Initializes the settings object.
        - Ensures environment variables from .env are loaded.
        - Reads the main YAML configuration file.
        """
        self.verbosity_level = DEBUG

        # Preserve the current working path
        original_folder_ = os.getcwd()

        # Set folder from which `dotenvy_py.find_upwards` will start searching for the .env file.
        module_folder_ = Path(__file__).resolve().parent
        os.chdir(module_folder_)

        # Load environment variables
        self.load_env()
        # Get settings file path from environment variable, or use a default
        file_path = os.getenv('SETTING_FILE', module_folder_ / 'settings.yml')
        # Load YAML settings file
        self._config = self._read_yaml_file(file_path)

        # Return to the original working path
        os.chdir(original_folder_)

    @classmethod
    def load_env(cls):
        """
        Finds and loads the .env file into the process's environment variables.
        This method is idempotent and will only run once per application lifecycle.
        """
        if cls._env_loaded:
            return

        # Find an .env file in the current or parent directories
        env_path_ = dotenvy_py.find_upwards('.env', 2)
        if env_path_:
            dotenvy_py.from_filename(env_path_)
            message_ = f"Found and loaded environment file {env_path_}"
            message_verbosity_level_ = DEBUG
        else:
            message_ = "No environment file found. Continuing without loading environment variables."
            message_verbosity_level_ = WARNING

        verbosity_level_ = get_verbosity_level()
        verbose(message_, message_verbosity_level_, verbosity_level_)

        cls._env_loaded = True

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
settings = Settings()
