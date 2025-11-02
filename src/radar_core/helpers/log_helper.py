# src/radar_core/helpers/log_helper.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
from datetime import datetime
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, INFO, WARNING, Logger
# os: allows access to functionalities dependent on the Operating System
import os
# pathlib: provides an interface to work with file paths in a more readable and easier way than the older 'os.path'.
from pathlib import Path

LOG_FILENAME = "app.log"
LOG_FOLDER = "logs"
DEFAULT_VERBOSITY_LEVEL = INFO
DEFAULT_CONSOLE_LOG_LEVEL = WARNING  # Console handler logs only warning, error and critical levels


def get_logging_config(log_level: int = INFO, filename: str = LOG_FILENAME) -> dict:
    """
    Generates a declarative logging configuration dictionary.
    :param log_level: The logging level for the file handler.
    :param filename: The name for the log file.
    :return: A dictionary with the logging configuration.
    """
    main_folder_ = Path(__file__).resolve().parent.parent  # radar_core folder
    enable_file_logging_ = os.getenv('ENABLE_FILE_LOGGING', 'true').lower() in ('true', '1', 't')
    handlers_ = ["console"]
    config_ = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)-45s - %(levelname)-8s - line %(lineno)3d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": WARNING if enable_file_logging_ else log_level,
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
        log_file_path_ = logs_folder_ / f'{filename.removesuffix('.py')}.log'

        config_["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": str(log_file_path_),
            "maxBytes": 524288,
            "backupCount": 3,
            "level": log_level,
        }
        handlers_.append("file")

    return config_


def begin_logging(logger: Logger, script_name: str, verbosity_level: int = INFO) -> None:
    """
    Logs the startup process for a given script using the provided logger.

    :param logger: To be used.
    :param script_name: The name of the script being executed.
    :param verbosity_level: Importance level of messages reporting the progress of the process for this method
    """
    startup_message_ = f'{script_name.capitalize()} started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.'
    verbose(startup_message_, INFO, verbosity_level)
    logger.info('=' * 80)
    logger.info(startup_message_)
    logger.info('-' * 80)


def end_logging(logger: Logger) -> None:
    """
    Finish logging and close handlers during a graceful shutdown.
    :param logger: To be closed.
    """
    # Finish logging
    logger.info('Finished')
    logger.info('=' * 80)

    # Remove handlers
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

    # Release memory
    del logger


def verbose(message: str,
            message_verbosity_level: int,
            task_verbosity_level: int,
            end: str = '\n') -> None:
    """
    Display the message on console if the level of verbosity allows so

    :param message: Message to be displayed on console.
    :param message_verbosity_level: Level of the message to be displayed.
    :param task_verbosity_level: Minimum level of importance allowed for messages to be displayed by the
     caller task/process.
    :param end: String appended after the last value, default a newline.
    """
    if task_verbosity_level <= message_verbosity_level <= DEFAULT_CONSOLE_LOG_LEVEL:
        print(message, end=end)


def get_verbosity_level() -> int:
    """Reads LOG_LEVEL from the environment and returns a valid numeric logging level."""
    log_level_env_ = os.getenv('LOG_LEVEL')
    default_level_ = DEFAULT_VERBOSITY_LEVEL

    # Return default if env var is not set or not a digit
    if not log_level_env_ or not log_level_env_.isdigit():
        return default_level_

    level = int(log_level_env_)

    # If outside range 10-59, return default
    if not 10 <= level <= 59:
        return default_level_

    # Calculate level by flooring to the nearest 10
    return (level // 10) * 10
