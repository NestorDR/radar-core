# src/radar_core/__main__.py
"""Minimal CLI entrypoint for the package: python -m radar_core"""

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import INFO, getLogger
import logging.config
# signal: handles process signals and interrupts (e.g., Ctrl+C)
import signal

# --- App modules ---
# settings: has the configuration for the radar_core
from radar_core.settings import get_settings
# analyzer: defines the application's main logic.
from radar_core.analyzer import analyzer
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.log_helper import begin_logging, end_logging, rotate_log_at_startup


# noinspection unused-parameter
def handle_sigterm(signum: int,
                   frame) -> None:
    """
    Handles OS-level SIGTERM signals sent by container runtimes or system managers.

    Intercepts automated process termination requests ('docker stop', Kubernetes pod eviction, or systemd shutdown) and
     converts them into a KeyboardInterrupt exception.
    This unifies graceful shutdown logic across both interactive user cancellations (Ctrl+C / SIGINT) and
     automated OS signals.

    :param signum: The integer signal number received from the operating system.
    :param frame: The current execution stack frame at the time of signal arrival.

    :raises KeyboardInterrupt: Always raised to trigger the main application cleanup path.
    """
    raise KeyboardInterrupt('Received SIGTERM from OS/Docker')


# Use of __name__ & __main__
# When the Python interpreter reads a code file, it completely executes the code in it.
# For example, in a file my_module.py, when executed as the main program, the __name__ attribute will be equal to '__main__'.
# However, if it is called by importing it from another module: import my_module, the __name__ attribute will be 'my_module'.
if __name__ == '__main__':
    log_filename_ = 'main.analyzer'
    # Initialize app settings
    settings_ = get_settings(log_filename_)
    # Logger initialization
    logging.config.dictConfig(settings_.log_config)
    rotate_log_at_startup()
    # Get root logger and log start messages
    logger_ = getLogger(__name__)
    begin_logging(logger_, log_filename_, INFO)

    # Register SIGTERM to raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        # Run the application
        exit_code = analyzer()
    except KeyboardInterrupt:
        logger_.warning('Execution interrupted by user or Docker (SIGINT/SIGTERM).')
        exit_code = 130  # Standard Unix exit code for SIGINT/SIGTERM termination
    finally:
        # Finish logging, remove logger handlers and release memory
        end_logging(logger_)

    # Return exit code
    raise SystemExit(exit_code)
