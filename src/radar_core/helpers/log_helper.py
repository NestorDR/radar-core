# src/radar_core/helpers/log_helper.py

# --- Python modules ---
# datetime: provides classes for manipulating dates and times.
from datetime import datetime
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import INFO, WARNING, Logger


DEFAULT_CONSOLE_LOG_LEVEL = WARNING  # Console handler logs only warning, error and critical levels


def begin_logging(logger: Logger,
                  script_name: str,
                  verbosity_level: int = INFO) -> None:
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
    Display the message on the console if the level of verbosity allows so.
    Uses flush=True to ensure immediate output in containerized environments.

    :param message: Message to be displayed on the console.
    :param message_verbosity_level: Level of the message to be displayed.
    :param task_verbosity_level: A minimum level of importance is allowed, it reduces the verbosity and
     should be set at the module/class level.
    :param end: String appended after the last value, default a newline.
    """
    # task_verbosity_level ....: reduces the verbosity and should be set at the module/class level.
    # DEFAULT_CONSOLE_LOG_LEVEL: it avoids displaying a message which will be logged anyway in the console by logger.
    if task_verbosity_level <= message_verbosity_level <= DEFAULT_CONSOLE_LOG_LEVEL and message != '':
        print(message, end=end, flush=True)  # flush=True is critical for Docker logs to appear in real-time
