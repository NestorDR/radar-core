# src/radar_core/domain/strategies/rsirc.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG
# typing: provides runtime support for type hints.

# --- Third Party Libraries ---
# numba: JIT compiler that compiles a subset of Python and NumPy code into optimized machine code using the industry-standard LLVM compiler library
from numba import njit
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.base_strategy import RsiStrategyABC
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import RSI_RC, LONG, SHORT, STEP_LENGTH_RSI_LEVELS, TIMEFRAMES

# Column constants for work matrices: inputs and outputs
BAR_NUMBER = 0
PRICE = 1
PERCENT_CHANGE = 2


# In HPC (High Performance Computing), it is the best practice to decouple compute-intensive logic (the kernel)
# from orchestration logic (the class). `_find_trades_rc` acts as a pure function: it accepts Numpy arrays and integers,
# and returns lists, without accessing or modifying the class state. Keeping it at the module level reinforces this separation.
@njit(cache=True)
def _find_trades_rc(rsi_values: np.ndarray,
                    stop_loss_bar_numbers: np.ndarray,
                    in_: int,
                    over_: int,
                    out_: int,
                    is_long_position: bool,
                    future_bar_number: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast JIT-compiled kernel to identify trades based on RSI Rollercoaster logic.
    Logic: Input Signal -> (Check Stop Loss) -> Over[bought|sold] Signal -> Output Signal.

    :param rsi_values: Array of RSI values.
    :param stop_loss_bar_numbers: Array of stop-loss bar numbers.
    :param in_: Input level for the strategy.
    :param over_: Level of overbought/oversold for the strategy.
    :param out_: Output level for the strategy.
    :param is_long_position: Flag of the position type under analysis: long (True) or short (False).
    :param future_bar_number: The number of a price bar that will be available in the future.

    :return: Tuple of numpy arrays with the input and output bar numbers for each trade.
    """
    total_bars_ = len(rsi_values)
    input_bar_numbers_ = []
    output_bar_numbers_ = []
    last_bar_number_processed_ = -1

    # Loop through the time series
    for bar_number_ in range(1, total_bars_):
        if bar_number_ <= last_bar_number_processed_:
            continue

        # 1. Check input signal: RSI cross over (Long) or cross under (Short) the input level (in_)
        previous_rsi_ = rsi_values[bar_number_ - 1]
        rsi_ = rsi_values[bar_number_]
        # Long.: RSI > in_ AND Previous <= in_
        # Short: RSI < in_ AND Previous >= in_
        if not (rsi_ > in_ >= previous_rsi_ if is_long_position else rsi_ < in_ <= previous_rsi_):
            continue

        # This assignment is purely semantic, indicating that once the input condition was met,
        # the bar becomes a market input bar. Active market position, start trading.
        input_bar_number_ = bar_number_

        # Retrieve the pre-calculated stop-loss bar number for this input signal
        stop_loss_bar_number_ = stop_loss_bar_numbers[input_bar_number_]

        # 2. Find over[bought|sold] signal, look for the first signal strictly after the input bar number
        # But if stop_loss happens before output, close there.
        over_bar_number_ = -1
        for active_position_bar_number_ in range(input_bar_number_ + 1, total_bars_):
            # Check stop-loss priority
            if 0 < stop_loss_bar_number_ < active_position_bar_number_:
                # Stop-loss break happens before finding over[bought|sold], close losing position
                break

            # Check over[bought|sold] signal: RSI cross over (Long) or cross under (Short) the over level (over_)
            previous_rsi_ = rsi_values[active_position_bar_number_ - 1]
            rsi_ = rsi_values[active_position_bar_number_]
            # Long.: RSI <= over_ AND Previous > over_
            # Short: RSI >= over_ AND Previous < over_
            if rsi_ >= over_ > previous_rsi_ if is_long_position else rsi_ <= over_ < previous_rsi_:
                over_bar_number_ = active_position_bar_number_
                break

        # 3. Determine Outcome (stop-loss vs. output)
        # Case A: stop-loss triggered before over[bought|sold] was found or reached
        # Note: If over_bar_number_ is -1 (not found), it would be effective in the future
        if 0 < stop_loss_bar_number_ < (over_bar_number_ if over_bar_number_ != -1 else future_bar_number):
            # Add losing trade
            input_bar_numbers_.append(input_bar_number_)
            output_bar_numbers_.append(stop_loss_bar_number_)
            last_bar_number_processed_ = stop_loss_bar_number_
            continue

        # Case B: over[bought|sold] signal not found (and no stop-loss triggered)
        # This implies the trade is still open at the end of the analysis period (Buy & Hold scenario).
        # Return future_bar_number to indicate "Open Position at end of data".
        if over_bar_number_ == -1:
            # Add trade still open
            input_bar_numbers_.append(input_bar_number_)
            output_bar_numbers_.append(future_bar_number)
            # Strategy lifecycle consumes the rest of the timeline as it remains open
            last_bar_number_processed_ = total_bars_
            continue

        # 4. Find output signal, look for output strictly after the over[bought|sold] bar number
        output_bar_number_ = -1
        for active_position_bar_number_ in range(over_bar_number_ + 1, total_bars_):
            # Check output signal: RSI cross under (Long) or cross over (Short) the output level (out_)
            previous_rsi_ = rsi_values[active_position_bar_number_ - 1]
            rsi_ = rsi_values[active_position_bar_number_]
            # Long.: RSI <= out_ AND Previous > out_
            # Short: RSI >= out_ AND Previous < out_
            if rsi_ <= out_ < previous_rsi_ if is_long_position else rsi_ >= out_ > previous_rsi_:
                output_bar_number_ = active_position_bar_number_
                break

        # Add Trade
        input_bar_numbers_.append(input_bar_number_)
        if output_bar_number_ != -1:
            # Case C: Output signal found, closed position
            # Add trade with the completed life cycle for the strategy input, over[bought|sold] and output reached
            output_bar_numbers_.append(output_bar_number_)
            last_bar_number_processed_ = output_bar_number_
            continue

        # Case D: No Output found after over[bought|sold] → "Open Position at end of data"
        # Add trade still open: Mark-to-market using future_bar_number
        output_bar_numbers_.append(future_bar_number)
        last_bar_number_processed_ = total_bars_

    # Convert lists to arrays facilitating further processing
    return np.array(input_bar_numbers_, dtype=np.int32), np.array(output_bar_numbers_, dtype=np.int32)


class RsiRollerCoaster(RsiStrategyABC):
    """
    Class to identify Profitable Roller Coaster strategy on the RSI (PRSIRC)
    Visit https://www.tecnicasdetrading.com/2011/09/tecnica-de-trading-rsi-rollercoaster.html
    """

    def __init__(self,
                 verbosity_level: int = DEBUG):
        """
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process for all
         methods of the class.
        """
        super().__init__(RSI_RC, verbosity_level)

    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions,
                 prices_df: pl.DataFrame,
                 close_prices: np.ndarray,
                 verbosity_level: int = DEBUG) -> dict:
        """
        Identifies the best combinations of levels input, overbought/oversold, and output for the RSI Rollercoaster
        strategy, both for Long and Short positions, and evaluate its profitability on positions:
         - long: open when RSI rises above the input level and closed when RSI falls below the output level
         - short: open when RSI falls below the output level and closed when RSI rises above the input level.
        Save the profitable setups (identified levels and associated ratios) in the Database.
        Returns a dictionary with the strategies with the best ratios.

        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe with required columns [Date, Close, Volume, PercentChange], indexed by numbers.
        :param close_prices: Close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: Dictionary of strategies with the best ratios based on its profitability.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Logs initialization and prepares the necessary variables for the process
        init_dt_, analysis_context_, original_column_names_, verbosity_level = \
            self.initialize_identification(symbol, timeframe, prices_df, verbosity_level)

        # Identify and calculate where to stop losses for both long and short positions.
        prices_df = self.identify_where_to_stop_loss(timeframe, prices_df, close_prices)

        # Pre-calculate arrays for Numba. Convert Polars columns to Numpy-arrays once to avoid overhead due to loops.
        rsi_values_ = prices_df['Rsi'].to_numpy()
        percent_changes_ = prices_df['PercentChange'].to_numpy()

        # Pre-calculate min/max RSI to skip impossible conditions in loops; nanmin/nanmax ignore initial NaN values (first 14 periods)
        min_rsi_ = np.nanmin(rsi_values_)
        max_rsi_ = np.nanmax(rsi_values_)

        # Stop loss arrays (Bar Numbers / Indices)
        long_stops_ = prices_df['BarNumberForLongStop'].to_numpy().astype(np.int32)
        short_stops_ = prices_df['BarNumberForShortStop'].to_numpy().astype(np.int32)

        future_bar_number_ = analysis_context_.future_bar_number

        # Contexts to iterate:
        #  Position type: LONG.  Levels: '1st input', 'last input', '1st overbought', 'last overbought' & 'step to increase'
        #  Position type: SHORT. Levels: '1st input', 'last input', '1st oversold', 'last oversold' & 'step to decrease'
        # contexts_ = [ (LONG, 20, 41, 50, 81, 1), (SHORT, 75, 64, 50, 19, -1) ]
        contexts_ = [(LONG, 16, 61, 40, 81, STEP_LENGTH_RSI_LEVELS)] + \
                    ([] if only_long_positions else [(SHORT, 84, 58, 60, 19, -STEP_LENGTH_RSI_LEVELS)])

        for position_type_, from_in_, to_in_, from_over_, to_over_, step_ in contexts_:
            # Initialize bad strategy to be evaluated and to get better RSI-RCs
            best_ratios_ = self.initialize_bad_strategy()
            is_long_position_ = position_type_ == LONG
            analysis_context_.is_long_position = is_long_position_

            stop_loss_bar_numbers_ = long_stops_ if is_long_position_ else short_stops_

            # Iterate over the input level of the RSI
            for in_ in range(from_in_, to_in_, step_):
                if verbosity_level == DEBUG:
                    print('', end='\r')
                    print(
                        f'Evaluating profitability {TIMEFRAMES[timeframe]} of RSI({self.period}) Rollercoaster input level {in_} for {symbol}...',
                        end='')

                # Skip if RSI never reaches the input level necessary for a cross
                # Long needs rsi > in (strict), so skip if max <= in
                if (is_long_position_ and max_rsi_ <= in_) or (not is_long_position_ and min_rsi_ >= in_):
                    continue

                # Initialize the best strategy for this input level
                best_ratios_for_in_ = self.initialize_bad_strategy()

                # Iterate over the overbought/oversold level of the RSI
                for over_ in range(from_over_, to_over_, step_):
                    # Skip if RSI never reaches the over level necessary for trigger
                    # Long needs rsi >= over, so skip if max < over
                    if (is_long_position_ and max_rsi_ < over_) or (not is_long_position_ and min_rsi_ > over_):
                        continue

                    # Set a range of output levels to be analyzed based on pre-set input and over[bought|sold] levels
                    from_out_, to_out_ = self.__get_out_range(is_long_position_, in_, over_)

                    # Iterate over the output level of the RSI
                    for out_ in range(from_out_, to_out_, -step_):
                        # Evaluate the life cycle for the RSI Rollercoaster strategy
                        # (input, over[bought|sold] and output) with the current combination
                        input_bar_numbers_, output_bar_numbers_ = _find_trades_rc(rsi_values_, stop_loss_bar_numbers_,
                                                                                  in_, over_, out_,
                                                                                  is_long_position_, future_bar_number_)
                        # If no trades identified, skip
                        if len(input_bar_numbers_) == 0:
                            continue

                        # Set strategy Inputs. Period and levels that parameterize the analyzed strategy
                        inputs_ = {'period': self.period, 'in': in_, 'over': over_, 'out': out_}

                        # Evaluate trades identified, calculate trading performance ratios and aggregates
                        ratios_ = self.perfile_performance_fast(analysis_context_, inputs_,
                                                            input_bar_numbers_, output_bar_numbers_,
                                                            close_prices, percent_changes_, prices_df)
                        if not ratios_:
                            continue

                        # Check if RSI RC just analyzed for this input level, is a better indicator for positionings
                        #  than the previous calculated ones.
                        best_ratios_for_in_ = self.track_best_strategy(ratios_, best_ratios_for_in_)

                if best_ratios_for_in_.net_profit > 0.0 and best_ratios_for_in_.expected_value > 0.0:
                    # Save only positive ratios
                    self.ratio_crud.upsert(best_ratios_for_in_)

                # Check if the best RSI RC for this input level is a better indicator for positions
                # than the previously calculated input levels.
                best_ratios_ = self.track_best_strategy(best_ratios_for_in_, best_ratios_)

            if verbosity_level == DEBUG:
                print('', end='\r')

            # Gather the best strategies
            if analysis_context_.is_long_position:
                # Set dictionary for better Long strategies
                analysis_context_.best_long = self.validate_best_strategy(best_ratios_)
            else:
                # Set dictionary for better Short strategies
                analysis_context_.best_short = self.validate_best_strategy(best_ratios_)

        # Release memory
        del contexts_

        # Reset to the original columns, relevant to allow re-use of the same dataframe for other strategies
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization and return results.
        return self.finalize_identification(init_dt_, analysis_context_, verbosity_level)

    @staticmethod
    def __get_out_range(is_long_position: bool,
                        in_: int,
                        over_: int) -> tuple[int, int]:
        """
        Identify the range of levels for iteration over the output level of the RSI based on input
         and overbought/oversold levels.

        :param is_long_position: Flag of the position type under analysis: long (True) or short (False).
        :param in_: Input level.
        :param over_: Overbought/oversold level.

        :return: Range of output levels to iterate on RSI Rollercoaster.
        """

        if is_long_position:
            # It will be used in a loop ─► for range(from_out_, to_out_, -step):
            from_out_ = 84 if over_ > 84 else over_
            to_out_ = (18 if in_ < 18 else in_)

        else:
            # It will be used in a loop ─► range(from_out_, to_out_, step):
            from_out_ = 16 if over_ < 16 else over_
            to_out_ = (82 if in_ > 82 else in_)

        return from_out_, to_out_
