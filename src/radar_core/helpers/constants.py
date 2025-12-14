# src/radar_core/helpers/constants.py

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

# Positioning constants
SHORT = -1
NO_POSITION = 0
LONG = 1

# Profitability calculation constants
# Average commission per trade in 2017 = 0.07% of average trade amount $8,700 = $6.09
COMMISSION_PERCENT = 0.0007

# Minimum acceptable or tolerable for trading evaluation
WIN_PROBABILITY_THRESHOLD = 0.40

# Step length in RSI levels
STEP_LENGTH_RSI_LEVELS = 1

# Strategy acronym
RSI_2B = 'RSI(14) 2B'
RSI_RC = 'RSI(14) RC'
RSI_SMA = 'RSI(14) SMA'
SMA = 'SMA'