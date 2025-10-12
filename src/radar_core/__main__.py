# src/radar_core/__main__.py
"""Minimal CLI entrypoint for the package: python -m radar_core"""

# --- Import and apply logging settings BEFORE importing other app modules ---
import logging.config
from radar_core.helpers.log_helper import get_logging_config, begin_logging, end_logging

log_name_ = "main.analyzer"
logging.config.dictConfig(get_logging_config(filename=log_name_))

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import INFO, getLogger

# --- App modules ---
# analyzer: defines the application's main logic.
from radar_core.analyzer import analyzer

# Use of __name__ & __main__
# When the Python interpreter reads a code file, it completely executes the code in it.
# For example, in a file my_module.py, when executed as the main program, the __name__ attribute will be '__main__',
# however, if it is called by importing it from another module: import my_module, the __name__ attribute will be
# 'my_module'
if __name__ == "__main__":
    # Get root logger and log start messages
    logger_ = getLogger()
    begin_logging(logger_, log_name_, INFO)

    # Run the application
    exit_code = analyzer()

    # Finish logging, remove logger handlers and release memory
    end_logging(logger_)

    # Return exit code
    raise SystemExit(exit_code)
