# -*- coding: utf-8 -*-

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
import logging
from logging.handlers import RotatingFileHandler
# os: allows access to functionalities dependent on the Operating System
import os

LOG_FILENAME = "radar_core.log"
LOG_FOLDER = "logs"
DEFAULT_VERBOSITY_LEVEL = 20  # 20 == INFO
DEFAULT_CONSOLE_LOG_LEVEL = logging.WARNING  # Console handler logs only warning, error and critical levels


def setup_logger(log_level: int = logging.INFO,
                 filename: str = LOG_FILENAME,
                 start_logging: bool = True) -> logging.Logger:
    """
    Set up consistent logging for the entire project.
    :param log_level: Set the logging level of the file handler,
      visit https://docs.python.org/3.13/library/logging.html#levels.
    :param filename: Specifies the file to use as the stream for logging.
    :param start_logging: Specifies whether to start logging immediately.
    :return: A logger for the entire project.
    """
    if not logging.DEBUG <= log_level <= logging.CRITICAL:
        raise ValueError(
            f'Parameter verbosity_level_ is out of range, should be between {logging.DEBUG} and {logging.CRITICAL}.')

    # Check environment variable to decide if file logging should be enabled.
    enable_file_logging = os.getenv('ENABLE_FILE_LOGGING', 'true').lower() in ('true', '1', 't')

    # Get root logger and clears existing handlers on the root logger before adding new ones to ensure idempotency.
    root_logger_ = logging.getLogger()
    if root_logger_.hasHandlers():
        root_logger_.handlers.clear()

    # Set the root logger's level that acts as a filter before handlers (set all levels >= DEBUG at the logger level)
    # Set it to the most verbose level you want ANY handler to process.
    # Handlers will have their own (and more restrictive) levels.
    root_logger_.setLevel(logging.DEBUG)

    # Create formatter
    log_format_ = '%(asctime)s - %(name)-45s - %(levelname)-8s - line %(lineno)3d - %(message)s'
    date_format_ = '%Y-%m-%d %H:%M:%S'
    text_formatter_ = logging.Formatter(log_format_, datefmt=date_format_)

    if enable_file_logging:
        # Get logger folder
        helper_folder_ = os.path.dirname(os.path.abspath(__file__))
        app_folder_ = os.path.dirname(helper_folder_)
        logs_folder_ = os.path.join(app_folder_, LOG_FOLDER)
        os.makedirs(logs_folder_, exist_ok=True)  # Create directories' path recursively if it didn't exist. Catch error

        base, ext = os.path.splitext(filename)
        # Ensure that the file name contains the extension “.log”
        if ext.lower() != ".log":
            filename = base + ".log"

        # Set the logger file
        path_filename_ = os.path.join(logs_folder_, filename)

        # Create a file handler which logs even debug messages
        file_handler_ = RotatingFileHandler(path_filename_, maxBytes=524288, backupCount=3)
        file_handler_.setLevel(log_level)  # File handler uses the specified level
        file_handler_.setFormatter(text_formatter_)
        root_logger_.addHandler(file_handler_)

    # Create a console handler with a higher log level
    console_handler_ = logging.StreamHandler()
    console_handler_.setLevel(DEFAULT_CONSOLE_LOG_LEVEL if enable_file_logging else logging.getLevelName(log_level))
    console_handler_.setFormatter(text_formatter_)
    root_logger_.addHandler(console_handler_)

    if start_logging:
        root_logger_.info('=' * 80)
        root_logger_.info('Started')
        root_logger_.info('-' * 80)

    return root_logger_


def end_logger(logger: logging.Logger) -> None:
    """
    Close handlers during a graceful shutdown
    :param logger: to close
    """
    logger.info('Finished')

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
