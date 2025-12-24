# src/radar_core/domain/strategies/sma.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG

# --- Third Party Libraries ---
# numba: JIT compiler that compiles a subset of Python and NumPy code into optimized machine code
from numba import njit
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.base_strategy import StrategyABC
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import LONG, SHORT, TIMEFRAMES


# In HPC (High Performance Computing), it is the best practice to decouple compute-intensive logic (the kernel)
# from orchestration logic (the class). `_find_trades_2b` acts as a pure function: it accepts Numpy arrays and integers,
# and returns lists, without accessing or modifying the class state. Keeping it at the module level reinforces this separation.
@njit(cache=True)
def _find_trades_sma(values: np.ndarray,
                     period: int,
                     is_long_position: bool,
                     future_bar_number: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast JIT-compiled kernel to identify trades based on Moving Average crossovers.
    It calculates the SMA on-the-fly to avoid memory allocation for intermediate arrays.

    :param values: Array of values (e.g., Close prices) to compute MA and check crosses.
    :param period: The moving average period.
    :param is_long_position: True for Long strategy (Buy if Value > SMA), False for Short.
    :param future_bar_number: The number of a price bar that will be available in the future.

    :return: Tuple of numpy arrays with the input and output bar numbers for each trade.
    """
    total_bars_ = len(values)
    input_bar_numbers_ = []
    output_bar_numbers_ = []

    # It needs at least 'period' elements to calculate the first valid SMA.
    # The first valid SMA corresponds to index 'period - 1'.
    # Start iterating from 'period' to check the cross between (i-1) and (i).
    if total_bars_ <= period:
        return np.array(input_bar_numbers_, dtype=np.int32), np.array(output_bar_numbers_, dtype=np.int32)

    # Efficient SMA calculation using a running `sum`
    # Initialize the `sum` for the first window [0: period]
    # Identify the first bar number with a valid value (non-NaN)
    first_nan_bar_ = 0
    while first_nan_bar_ < total_bars_ and np.isnan(values[first_nan_bar_]):
        first_nan_bar_ += 1

    # Check if there are enough valid elements after leading NaNs to get the first window
    # It needs at least 'period' elements to calculate the first valid SMA.
    if (total_bars_ - first_nan_bar_) <= period:
        return np.array(input_bar_numbers_, dtype=np.int32), np.array(output_bar_numbers_, dtype=np.int32)

    # State variables
    in_position_ = False

    # Initialize the running sum from the first valid index
    current_sum_ = 0.0
    for i in range(first_nan_bar_, first_nan_bar_ + period):
        current_sum_ += values[i]

    # Calculate the first valid SMA at the end of the first valid window
    # Example: if first_nan_bar_ is 14, due to a RSI(14), and period is 20, the first SMA is at index 33
    first_valid_sma_bar_ = first_nan_bar_ + period - 1
    previous_sma_ = current_sum_ / period
    previous_value_ = values[first_valid_sma_bar_]

    # Iterate starting from the first bar AFTER the initial window
    for i in range(first_valid_sma_bar_ + 1, total_bars_):
        current_value_ = values[i]

        # Update Running Sum: add new value, remove old value leaving the window
        # Value leaving is at index (i - period)
        current_sum_ = current_sum_ + current_value_ - values[i - period]
        current_sma_ = current_sum_ / period

        # Check Crossovers
        # Long.: Open if (previous value <= previous SMA) and (current value > current SMA)
        #       Close if (previous value >  previous SMA) and (current value < current SMA)
        # Short: Open if (previous value >= previous SMA) and (current value < current SMA)
        #       Close if (previous value <  previous SMA) and (current value > current SMA)
        is_above_ = current_value_ > current_sma_
        was_above_ = previous_value_ > previous_sma_

        # Cross Over: Value crosses SMA from below to above
        cross_over_ = is_above_ and not was_above_
        # Cross Under: Value crosses SMA from above to below
        cross_under_ = not is_above_ and was_above_

        if not in_position_:
            # Look for input
            input_signal_ = cross_over_ if is_long_position else cross_under_
            if input_signal_:
                input_bar_numbers_.append(i)
                in_position_ = True
        else:
            # Look for output
            output_signal_ = cross_under_ if is_long_position else cross_over_
            if output_signal_:
                output_bar_numbers_.append(i)
                in_position_ = False

        # Update previous state for next iteration
        previous_sma_ = current_sma_
        previous_value_ = current_value_

    # Handle Open Position at the end of data (Mark-to-Market)
    if in_position_:
        output_bar_numbers_.append(future_bar_number)

    return np.array(input_bar_numbers_, dtype=np.int32), np.array(output_bar_numbers_, dtype=np.int32)


class MovingAverage(StrategyABC):
    """
    Class to identify and evaluate the Profitable Moving Average strategy
    """

    def __init__(self,
                 strategy_acronym: str,
                 value_column_name: str,
                 ma_column_name: str,
                 min_period: int = 8,
                 max_period: int = 233,
                 verbosity_level: int = DEBUG):
        """
        :param strategy_acronym: Strategy acronym to be analyzed.
        :param value_column_name: Column name on whose values the moving average will be calculated.
        :param ma_column_name: Column name where the moving average will be stored.
        :param min_period: Minimum number of time periods for the moving average calculation.
        :param max_period: Maximum number of time periods for the moving average calculation.
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process for all
         methods of the class.
        """
        super().__init__(strategy_acronym, verbosity_level)
        self.value_column_name = value_column_name
        self.ma_column_name = ma_column_name
        self.min_period = min_period
        self.max_period = max_period

    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions: bool,
                 prices_df: pl.DataFrame,
                 close_prices: np.ndarray,
                 verbosity_level: int = DEBUG) -> dict:
        """
        Iterate from the minimum to the maximum number of periods to calculate the MA and evaluate its profitability
         on positions:
         - long: open when the value rises above MA and closed when the value falls below MA
         - short: open when the value falls below MA and closed when the value rises above MA.
        Save the profitable setups (identified Moving Average) in the Database.
        Returns a dictionary with the strategies with the best ratios.

        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe at least with required columns
         [DateTime, {self.value_column_name}, PercentChange, BarNumber].
        :param close_prices: Close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: Dictionary of strategies with the best ratios based on its profitability.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Check if the column exists
        if self.value_column_name not in prices_df.columns:
            raise KeyError(f'The column [{self.value_column_name}] on whose values the moving average will be'
                           f' calculated does not exist in the Prices DataFrame.')

        # Logs initialization and prepares the necessary variables for the process
        init_dt_, analysis_context_, original_column_names_, verbosity_level = \
            self.initialize_identification(symbol, timeframe, prices_df, verbosity_level)

        # Pre-calculate arrays for Numba/Vectorized operations
        # Extract values to calculate SMA (usually Close or RSI)
        values_ = prices_df[self.value_column_name].to_numpy()
        # Extract percent change for stats
        percent_changes_ = prices_df['PercentChange'].to_numpy()

        # Identify
        future_bar_number_ = analysis_context_.future_bar_number

        # Position types to iterate
        position_types_ = [LONG] + ([] if only_long_positions else [SHORT])

        for position_type_ in position_types_:
            # Initialize bad strategy to be evaluated and to get better MAs
            best_ratios_ = self.initialize_bad_strategy()
            is_long_position_ = position_type_ == LONG
            analysis_context_.is_long_position = is_long_position_

            # Iterate from the min to the max number of periods to calculate the MA and evaluate its profitability
            for period_ in range(self.min_period, self.max_period + 1):
                if verbosity_level == DEBUG:
                    print('', end='\r')
                    print(
                        f'Evaluating profitability {TIMEFRAMES[timeframe]} of {self.strategy_acronym}({period_}) for {symbol}...',
                        end='')

                if len(values_) <= period_:
                    # The minimum number of periods to calculate the average is not reached
                    continue

                # Calculate SMA signals using Numba (no intermediate Polars objects)
                input_bar_numbers_, output_bar_numbers_ = _find_trades_sma(
                    values_, period_, is_long_position_, future_bar_number_)

                if len(input_bar_numbers_) == 0:
                    # There are no valid signals, skip further processing
                    continue

                # Set strategy Inputs. Period that parameterizes the analyzed strategy.
                inputs_ = {'period': period_}

                # Evaluate trades identified, calculate trading performance ratios and aggregates
                ratios_ = self.perfile_performance_fast(analysis_context_, inputs_,
                                                        input_bar_numbers_, output_bar_numbers_,
                                                        close_prices, percent_changes_, prices_df)
                if not ratios_:
                    continue

                if ratios_.net_profit > 0.0 and ratios_.expected_value > 0.0:
                    # Save only to positive ratios
                    self.ratio_crud.upsert(ratios_)

                # Check if MA just analyzed is a better indicator for positionings than the previous calculated ones.
                best_ratios_ = self.track_best_strategy(ratios_, best_ratios_)

            if verbosity_level == DEBUG:
                print('', end='\r')

            # Gather the best strategies
            if analysis_context_.is_long_position:
                # Set dictionary for better Long strategies
                analysis_context_.best_long = self.validate_best_strategy(best_ratios_)
            else:
                # Set dictionary for better Short strategies
                analysis_context_.best_short = self.validate_best_strategy(best_ratios_)

        # Reset to the original columns
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization and return results
        return self.finalize_identification(init_dt_, analysis_context_, verbosity_level)
