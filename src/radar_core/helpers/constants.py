# -*- coding: utf-8 -*-

DEFAULT_ATTEMPT_LIMIT = 3
DEFAULT_SLEEP = 3  # seconds
NOT_FOUND = 'Not found'
UNKNOWN = 'Unknown'

# Time frames
INTRADAY = 1
DAILY = 2
WEEKLY = 3
MONTHLY = 4
TIMEFRAMES = [UNKNOWN, 'Intraday', 'Daily', 'Weekly', 'Monthly']
TIMEUNITS = [UNKNOWN, 'Minutes', 'Days', 'Weeks', 'Months']

# Required columns for a valid prices dataframe
ORDERED_PRICE_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'PercentChange']
REQUIRED_PRICE_COLS = set(ORDERED_PRICE_COLS)
