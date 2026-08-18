# src/radar_core/domain/strategies/rsirc.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG

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
from radar_core.helpers.constants import COMMISSION_PERCENT, RSI_RC, LONG, SHORT, STEP_LENGTH_RSI_LEVELS, TIMEFRAMES

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

        # 2. Find over[bought|sold] signal, look for the first signal including the input bar number,
        #  because in the same first session RSI can cross over (Long) or cross under (Short) the over level (over_)
        # But if stop_loss happens before output, close there.
        over_bar_number_ = -1
        for active_position_bar_number_ in range(input_bar_number_, total_bars_):
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


@njit(cache=True)
def _grid_search_rc_fused(rsi_values: np.ndarray,
                          stop_loss_bar_numbers: np.ndarray,
                          close_prices: np.ndarray,
                          from_in: int,
                          to_in: int,
                          from_over: int,
                          to_over: int,
                          step: int,
                          is_long_position: bool,
                          future_bar_number: int) -> np.ndarray:
    """
    Fast JIT-compiled fused grid search kernel for RSI Rollercoaster.
    Scans the parameter grid (in_, over_, out_), simulates trades, evaluates PnL (Profit and Loss) in CPU registers,
    and returns the best winning parameter triplet (in, over, out) for each evaluated input level.

    :param rsi_values: Array of RSI values.
    :param stop_loss_bar_numbers: Array of stop-loss bar numbers.
    :param close_prices: Array of Close prices.
    :param from_in: Starting input level.
    :param to_in: Ending input level (exclusive).
    :param from_over: Starting over level.
    :param to_over: Ending over level (exclusive).
    :param step: Step size for input and over levels.
    :param is_long_position: Flag of the position type under analysis: long (True) or short (False).
    :param future_bar_number: The number of a price bar that will be available in the future.

    :return: 2D numpy array of shape (K, 3) with [in, over, out] parameters for surviving winning setups.
    """
    total_bars_ = len(rsi_values)
    direction_ = 1.0 if is_long_position else -1.0

    min_rsi_ = np.nanmin(rsi_values)
    max_rsi_ = np.nanmax(rsi_values)

    max_candidates_ = abs((to_in - from_in) // step) + 2
    candidates_ = np.empty((max_candidates_, 3), dtype=np.int32)
    cand_count_ = 0

    for in_ in range(from_in, to_in, step):
        # Skip if RSI never reaches the input level necessary for a cross
        if (is_long_position and max_rsi_ <= in_) or (not is_long_position and min_rsi_ >= in_):
            continue

        best_net_profit_ = -np.inf
        best_expected_value_ = -np.inf
        best_over_ = -1
        best_out_ = -1
        found_valid_for_in_ = False

        for over_ in range(from_over, to_over, step):
            # Skip if RSI never reaches the over level necessary for trigger
            if (is_long_position and max_rsi_ < over_) or (not is_long_position and min_rsi_ > over_):
                continue

            if is_long_position:
                from_out_ = 84 if over_ > 84 else over_
                to_out_ = 18 if in_ < 18 else in_
                out_step_ = -step
            else:
                from_out_ = 16 if over_ < 16 else over_
                to_out_ = 82 if in_ > 82 else in_
                out_step_ = -step

            for out_ in range(from_out_, to_out_, out_step_):
                last_bar_number_processed_ = -1
                winnings_ = 0.0
                losses_ = 0.0
                winn_trades_ = 0
                loss_trades_ = 0
                signals_ = 0
                first_input_price_ = 0.0

                for bar_number_ in range(1, total_bars_):
                    if bar_number_ <= last_bar_number_processed_:
                        continue

                    # 1. Check input signal: RSI cross over (Long) or cross under (Short) the input level (in_)
                    previous_rsi_ = rsi_values[bar_number_ - 1]
                    rsi_ = rsi_values[bar_number_]
                    if not (rsi_ > in_ >= previous_rsi_ if is_long_position else rsi_ < in_ <= previous_rsi_):
                        continue

                    input_bar_number_ = bar_number_
                    stop_loss_bar_number_ = stop_loss_bar_numbers[input_bar_number_]

                    # 2. Find over[bought|sold] signal
                    over_bar_number_ = -1
                    for active_position_bar_number_ in range(input_bar_number_, total_bars_):
                        if 0 < stop_loss_bar_number_ < active_position_bar_number_:
                            break

                        previous_rsi_ = rsi_values[active_position_bar_number_ - 1]
                        rsi_ = rsi_values[active_position_bar_number_]
                        if rsi_ >= over_ > previous_rsi_ if is_long_position else rsi_ <= over_ < previous_rsi_:
                            over_bar_number_ = active_position_bar_number_
                            break

                    # 3. Determine Outcome (stop-loss vs. output)
                    over_target_ = over_bar_number_ if over_bar_number_ != -1 else future_bar_number
                    if 0 < stop_loss_bar_number_ < over_target_:
                        output_bar_number_ = stop_loss_bar_number_
                        last_bar_number_processed_ = stop_loss_bar_number_
                    elif over_bar_number_ == -1:
                        output_bar_number_ = future_bar_number
                        last_bar_number_processed_ = total_bars_
                    else:
                        # 4. Find output signal strictly after over bar
                        output_bar_number_ = -1
                        for active_position_bar_number_ in range(over_bar_number_ + 1, total_bars_):
                            previous_rsi_ = rsi_values[active_position_bar_number_ - 1]
                            rsi_ = rsi_values[active_position_bar_number_]
                            if rsi_ <= out_ < previous_rsi_ if is_long_position else rsi_ >= out_ > previous_rsi_:
                                output_bar_number_ = active_position_bar_number_
                                break

                        if output_bar_number_ != -1:
                            last_bar_number_processed_ = output_bar_number_
                        else:
                            output_bar_number_ = future_bar_number
                            last_bar_number_processed_ = total_bars_

                    # Scalar Trade PnL
                    input_price_ = close_prices[input_bar_number_]
                    output_bar_ = output_bar_number_ if output_bar_number_ < total_bars_ else total_bars_ - 1
                    output_price_ = close_prices[output_bar_]

                    pnl_ = (output_price_ - input_price_) * direction_ - COMMISSION_PERCENT * (
                                input_price_ + output_price_)
                    if pnl_ > 0.0:
                        winnings_ += pnl_
                        winn_trades_ += 1
                    else:
                        losses_ += pnl_
                        loss_trades_ += 1

                    if signals_ == 0:
                        first_input_price_ = max(input_price_, 0.00001)
                    signals_ += 1

                # Calculate performance metrics for current (in_, over_, out_)
                if signals_ > 0:
                    net_profit_ = (winnings_ + losses_) / first_input_price_
                    win_prob_ = winn_trades_ / signals_
                    loss_prob_ = loss_trades_ / signals_
                    avg_win_ = winnings_ / winn_trades_ if winn_trades_ > 0 else 0.0
                    avg_loss_ = losses_ / loss_trades_ if loss_trades_ > 0 else 0.0
                    expected_value_ = win_prob_ * avg_win_ + loss_prob_ * avg_loss_

                    if net_profit_ > 0.0 and expected_value_ > 0.0:
                        new_is_better_ = (net_profit_ > best_net_profit_
                                          or (net_profit_ == best_net_profit_
                                              and expected_value_ > best_expected_value_))
                        if new_is_better_:
                            best_net_profit_ = net_profit_
                            best_expected_value_ = expected_value_
                            best_over_ = over_
                            best_out_ = out_
                            found_valid_for_in_ = True

        if found_valid_for_in_:
            candidates_[cand_count_, 0] = in_
            candidates_[cand_count_, 1] = best_over_
            candidates_[cand_count_, 2] = best_out_
            cand_count_ += 1

    return candidates_[:cand_count_]


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

    def identify_old(self,
                     symbol: str,
                     timeframe: int,
                     only_long_positions,
                     prices_df: pl.DataFrame,
                     close_prices: np.ndarray,
                     percent_changes: np.ndarray,
                     verbosity_level: int = DEBUG) -> None:
        """
        [DEPRECATED] Legacy baseline method to identify combinations of levels for RSI Rollercoaster.
        Use identify() instead for the accelerated JIT fused screening implementation.

        Identifies the best combinations of levels input, overbought/oversold, and output for the RSI Rollercoaster
        strategy, both for Long and Short positions, and evaluate its profitability on positions:
         - long: open when RSI rises above the input level and closed when RSI falls below the output level
         - short: open when RSI falls below the output level and closed when RSI rises above the input level.
        Save the profitable setups (identified levels and associated ratios) in the Database.

        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe with required columns [Date, Close, Volume, PercentChange], indexed by numbers.
        :param close_prices: Close prices for the given symbol and timeframe.
        :param percent_changes: Percent change of the close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Logs initialization and prepares the necessary variables for the process
        init_dt_, analysis_context_, original_column_names_, verbosity_level = \
            self.initialize_identification(symbol, timeframe, prices_df, verbosity_level)

        # Identify and calculate where to stop losses for both long and short positions.
        prices_df = self.identify_where_to_stop_loss(timeframe, prices_df, close_prices)

        # Pre-calculate arrays for Numba. Convert Polars columns to Numpy-arrays once to avoid overhead due to loops.
        rsi_values_ = prices_df['Rsi'].to_numpy()

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

        # Collect positive ratios across position types / levels for batch upsert
        positive_ratios_ = []

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
                        ratios_ = self.perfile_performance(analysis_context_, inputs_,
                                                           input_bar_numbers_, output_bar_numbers_,
                                                           close_prices, percent_changes, prices_df)
                        if not ratios_:
                            continue

                        # Check if RSI RC just analyzed for this input level, is a better indicator for positionings
                        #  than the previous calculated ones.
                        best_ratios_for_in_ = self.track_best_strategy(ratios_, best_ratios_for_in_)

                if best_ratios_for_in_.net_profit > 0.0 and best_ratios_for_in_.expected_value > 0.0:
                    # Save only positive ratios
                    positive_ratios_.append(best_ratios_for_in_)

                # Check if the best RSI RC for this input level is a better indicator for positions
                # than the previously calculated input levels.
                best_ratios_ = self.track_best_strategy(best_ratios_for_in_, best_ratios_)

            if verbosity_level == DEBUG:
                print('', end='\r')

            # Gather the best strategies
            if analysis_context_.is_long_position:
                analysis_context_.best_long = best_ratios_  # Best Long strategies
            else:
                analysis_context_.best_short = best_ratios_  # Best Short strategies

        # Perform atomic batch upsert for all positive ratios identified and remove remaining flagged rows atomically.
        self.persist_ratios(positive_ratios_, analysis_context_)

        # Release memory
        del contexts_

        # Reset to the original columns, relevant to allow re-use of the same dataframe for other strategies
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization
        self.finalize_identification(init_dt_, analysis_context_, verbosity_level)

    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions: bool,
                 prices_df: pl.DataFrame,
                 close_prices: np.ndarray,
                 percent_changes: np.ndarray,
                 verbosity_level: int = DEBUG) -> None:
        """
        Identifies the best combinations of levels input, overbought/oversold, and output for the RSI Rollercoaster
        strategy using the fused Numba JIT grid search kernel, both for Long and Short positions, and evaluates
        its profitability on positions:
         - long: open when RSI rises above the input level and closed when RSI falls below the output level
         - short: open when RSI falls below the output level and closed when RSI rises above the input level.
        Save the profitable setups (identified levels and associated ratios) in the Database.

        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe with required columns [Date, Close, Volume, PercentChange], indexed by numbers.
        :param close_prices: Close prices for the given symbol and timeframe.
        :param percent_changes: Percent change of the close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Logs initialization and prepares the necessary variables for the process
        init_dt_, analysis_context_, original_column_names_, verbosity_level = \
            self.initialize_identification(symbol, timeframe, prices_df, verbosity_level)

        # Identify and calculate where to stop losses for both long and short positions.
        prices_df = self.identify_where_to_stop_loss(timeframe, prices_df, close_prices)

        # Pre-calculate arrays for Numba. Convert Polars columns to Numpy-arrays once to avoid overhead due to loops.
        rsi_values_ = prices_df['Rsi'].to_numpy()

        # Stop loss arrays (Bar Numbers / Indices)
        long_stops_ = prices_df['BarNumberForLongStop'].to_numpy().astype(np.int32)
        short_stops_ = prices_df['BarNumberForShortStop'].to_numpy().astype(np.int32)

        future_bar_number_ = analysis_context_.future_bar_number

        # Contexts to iterate:
        contexts_ = [(LONG, 16, 61, 40, 81, STEP_LENGTH_RSI_LEVELS)] + \
                    ([] if only_long_positions else [(SHORT, 84, 58, 60, 19, -STEP_LENGTH_RSI_LEVELS)])

        # Collect positive ratios across position types / levels for batch upsert
        positive_ratios_ = []

        for position_type_, from_in_, to_in_, from_over_, to_over_, step_ in contexts_:
            # Initialize bad strategy to be evaluated and to get better RSI-RCs
            best_ratios_ = self.initialize_bad_strategy()
            is_long_position_ = position_type_ == LONG
            analysis_context_.is_long_position = is_long_position_

            stop_loss_bar_numbers_ = long_stops_ if is_long_position_ else short_stops_

            if verbosity_level == DEBUG:
                print('', end='\r')
                print(
                    f'Evaluating profitability {TIMEFRAMES[timeframe]} of RSI({self.period}) '
                    f'Rollercoaster (fused) for {symbol}...',
                    end=''
                )

            # Fast in-register JIT grid screening across all parameter combinations
            winning_candidates_ = _grid_search_rc_fused(
                rsi_values_, stop_loss_bar_numbers_, close_prices,
                from_in_, to_in_, from_over_, to_over_, step_,
                is_long_position_, future_bar_number_
            )

            # Materialize full Ratios objects only for surviving winning parameter combinations
            for i_ in range(len(winning_candidates_)):
                in_ = int(winning_candidates_[i_, 0])
                over_ = int(winning_candidates_[i_, 1])
                out_ = int(winning_candidates_[i_, 2])

                input_bar_numbers_, output_bar_numbers_ = _find_trades_rc(
                    rsi_values_, stop_loss_bar_numbers_, in_, over_, out_,
                    is_long_position_, future_bar_number_
                )
                if len(input_bar_numbers_) == 0:
                    continue

                inputs_ = {'period': self.period, 'in': in_, 'over': over_, 'out': out_}

                ratios_ = self.perfile_performance(
                    analysis_context_, inputs_, input_bar_numbers_, output_bar_numbers_,
                    close_prices, percent_changes, prices_df
                )
                if not ratios_:
                    continue

                if ratios_.net_profit > 0.0 and ratios_.expected_value > 0.0:
                    positive_ratios_.append(ratios_)

                best_ratios_ = self.track_best_strategy(ratios_, best_ratios_)

            if verbosity_level == DEBUG:
                print('', end='\r')

            # Gather the best strategies
            if analysis_context_.is_long_position:
                analysis_context_.best_long = best_ratios_
            else:
                analysis_context_.best_short = best_ratios_

        # Perform atomic batch upsert for all positive ratios identified and remove remaining flagged rows atomically.
        self.persist_ratios(positive_ratios_, analysis_context_)

        # Release memory
        del contexts_

        # Reset to the original columns, relevant to allow re-use of the same dataframe for other strategies
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization
        self.finalize_identification(init_dt_, analysis_context_, verbosity_level)

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
