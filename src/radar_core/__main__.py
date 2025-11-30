# src/radar_core/__main__.py
"""Minimal CLI entrypoint for the package: python -m radar_core"""

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import INFO, getLogger
import logging.config

# --- App modules ---
# settings: has the configuration for the radar_core
from radar_core.settings import Settings
# analyzer: defines the application's main logic.
from radar_core.analyzer import analyzer
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.log_helper import begin_logging, end_logging

# Use of __name__ & __main__
# When the Python interpreter reads a code file, it completely executes the code in it.
# For example, in a file my_module.py, when executed as the main program, the __name__ attribute will be '__main__',
# however, if it is called by importing it from another module: import my_module, the __name__ attribute will be
# 'my_module'
if __name__ == "__main__":
    # Initialize app settings
    settings = Settings()
    # Get root logger and log start messages
    logging.config.dictConfig(settings.log_config)
    logger_ = getLogger(__name__)
    begin_logging(logger_, "main.analyzer", INFO)

    # Run the application
    exit_code = analyzer(settings)

    # Finish logging, remove logger handlers and release memory
    end_logging(logger_)

    # Return exit code
    raise SystemExit(exit_code)
