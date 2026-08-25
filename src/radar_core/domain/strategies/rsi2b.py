# src/radar_core/domain/strategies/rsi2b.py

# --- Python modules ---
# json: provides functions for working with JSON data.
import json
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG
# typing: provides runtime support for type hints
from typing import Final

# --- Third Party Libraries ---
# numba: JIT compiler that compiles a subset of Python and NumPy code into optimized machine code
from numba import njit
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.base_strategy import RsiStrategyABC
from radar_core.domain.strategies._kernel_helpers import (
    _calculate_trade_pnl,
    _crosses_input,
    _crosses_output,
    _finalize_screening_metrics,
    _is_better_candidate,
    _is_profitable_candidate,
    _mark_to_market_bar,
)
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import COMMISSION_PERCENT, RSI_2B, LONG, SHORT, STEP_LENGTH_RSI_LEVELS, TIMEFRAMES

# Column constants for work matrices
INPUT: Final[int] = 0
OUTPUT: Final[int] = 1


# In HPC (High Performance Computing), it is the best practice to decouple compute-intensive logic (the kernel)
# from orchestration logic (the class). `_find_trades_2b` acts as a pure function: it accepts Numpy arrays and integers
# and returns NumPy arrays, without accessing or modifying the class state. Keeping it at the module level reinforces this separation.
@njit(cache=True)
def _find_trades_2b(
    rsi_values: np.ndarray,
    stop_loss_bar_numbers: np.ndarray,
    in_: int,
    out_: int,
    is_long_position: bool,
    future_bar_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast JIT-compiled kernel to identify trades based on RSI Two Bands logic.
    Logic: Input Signal -> (Check Stop Loss) -> Output Signal.

    :param rsi_values: Array of RSI values.
    :param stop_loss_bar_numbers: Array of stop-loss bar numbers.
    :param in_: Input level for the strategy.
    :param out_: Output level for the strategy.
    :param is_long_position: Flag of the position type under analysis: long (True) or short (False).
    :param future_bar_number: The number of a price bar that will be available in the future.

    :return: Tuple of numpy arrays with the input and output bar numbers for each trade.
    """
    total_bars_ = len(rsi_values)

    # Pre-allocated buffers
    # A trade lifecycle always advances to a later bar, so completed trades cannot overlap.
    # Therefore, no more than half of the available bars can become trade-entry bars.
    # The extra slot also safely accommodates the final open position.
    maximum_trades_ = total_bars_ // 2 + 1
    input_bar_numbers_ = np.empty(maximum_trades_, dtype=np.int32)
    output_bar_numbers_ = np.empty(maximum_trades_, dtype=np.int32)
    trade_count_ = 0
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
        if not _crosses_input(previous_rsi_, rsi_, in_, in_, is_long_position):
            continue

        # This assignment is purely semantic, indicating that once the input condition was met,
        # the bar becomes a market input bar. Active market position, start trading.
        input_bar_number_ = bar_number_

        # Retrieve the pre-calculated stop-loss bar number for this input signal
        stop_loss_bar_number_ = stop_loss_bar_numbers[input_bar_number_]

        # 2. Find output signal, look for output strictly after the input bar number
        # But if stop_loss happens before output, close there.
        output_bar_number_ = -1
        for active_position_bar_number_ in range(input_bar_number_ + 1, total_bars_):
            # Check stop-loss priority
            if 0 < stop_loss_bar_number_ < active_position_bar_number_:
                # Stop-loss break happens before finding output signal, close losing position
                break

            # Check output signal: RSI cross under (Long) or cross over (Short) the output level (out_)
            previous_rsi_ = rsi_values[active_position_bar_number_ - 1]
            rsi_ = rsi_values[active_position_bar_number_]
            # Long.: RSI <= out_ AND Previous > out_
            # Short: RSI >= out_ AND Previous < out_
            if _crosses_output(previous_rsi_, rsi_, out_, out_, is_long_position):
                output_bar_number_ = active_position_bar_number_
                break

        # 3. Determine Outcome (stop-loss vs. output)
        # Case A: stop-loss triggered before output was found or reached
        # Note: If output_bar_number_ is -1 (not found), it would be effective in the future
        if 0 < stop_loss_bar_number_ < (output_bar_number_ if output_bar_number_ != -1 else future_bar_number):
            # Add losing trade
            input_bar_numbers_[trade_count_] = input_bar_number_
            output_bar_numbers_[trade_count_] = stop_loss_bar_number_
            trade_count_ += 1
            last_bar_number_processed_ = stop_loss_bar_number_
            continue

        # Add Trade
        input_bar_numbers_[trade_count_] = input_bar_number_
        if output_bar_number_ != -1:
            # Case B: Output signal found, closed position
            # Add trade with the completed lifecycle for the strategy input and output reached
            output_bar_numbers_[trade_count_] = output_bar_number_
            trade_count_ += 1
            last_bar_number_processed_ = output_bar_number_
            continue

        # Case C: No Output and No Stop Loss found → 'Open Position at end of data'
        # Add trade still open: Mark-to-market using future_bar_number
        output_bar_numbers_[trade_count_] = future_bar_number
        trade_count_ += 1
        last_bar_number_processed_ = total_bars_

    return input_bar_numbers_[:trade_count_], output_bar_numbers_[:trade_count_]


@njit(cache=True)
def _grid_search_2b_fused(
    rsi_values: np.ndarray,
    stop_loss_bar_numbers: np.ndarray,
    close_prices: np.ndarray,
    from_in: int,
    to_in: int,
    step: int,
    is_long_position: bool,
    future_bar_number: int,
) -> np.ndarray:
    """
    Fast JIT-compiled fused grid search kernel for RSI Two Bands.
    Scans the parameter grid (in_, out_), simulates trades, evaluates PnL (Profit and Loss) in CPU registers,
    and returns winning parameter combinations (in_, out_) that achieve positive net profit and expected value.

    :param rsi_values: Array of RSI values.
    :param stop_loss_bar_numbers: Array of precalculated stop-loss bar numbers.
    :param close_prices: Array of closing prices for PnL calculation.
    :param from_in: Starting input level for RSI.
    :param to_in: Ending input level for RSI.
    :param step: Step size for input level iteration.
    :param is_long_position: Flag of the position type under analysis: long (True) or short (False).
    :param future_bar_number: The number of a price bar that will be available in the future.

    :return: 2D NumPy array of shape (K, 2) containing [in_, out_] for winning candidates.
    """
    total_bars_ = len(rsi_values)
    direction_ = 1.0 if is_long_position else -1.0
    out_step_ = -step

    min_rsi_ = np.nanmin(rsi_values)
    max_rsi_ = np.nanmax(rsi_values)

    # Pre-allocated buffers with the maximum/optimal potential number of winning candidates
    max_candidates_ = abs((to_in - from_in) // step) + 2
    candidates_ = np.empty((max_candidates_, 2), dtype=np.int32)
    candidate_count_ = 0

    # Iterate over the input level of the RSI
    for in_ in range(from_in, to_in, step):
        # Skip if RSI never reaches the input level necessary for a cross (Input)
        # Long needs RSI > in (strict), so skip if max <= in
        # Short needs RSI < in (strict), so skip if min >= in
        if (is_long_position and max_rsi_ <= in_) or (not is_long_position and min_rsi_ >= in_):
            continue

        # Initialize worst ratios
        best_net_profit_ = -np.inf
        best_expected_value_ = -np.inf
        best_out_ = -1

        # Set a range of output levels to be analyzed based on pre-set input and overbought/oversold levels
        from_out_, to_out_ = _get_out_range(is_long_position, in_)

        # Iterate over the output level of the RSI
        for out_ in range(from_out_, to_out_, out_step_):
            # Skip if RSI never reaches the output level necessary for a cross (Output)
            # Long: needs RSI > out_ to cross under
            # Short: needs RSI < out_ to cross over
            if (is_long_position and max_rsi_ <= out_) or (not is_long_position and min_rsi_ >= out_):
                continue

            # Evaluate the life cycle for the RSI Two Bands strategy
            # (input and output) with the current combination
            last_bar_number_processed_ = -1
            winnings_ = 0.0
            losses_ = 0.0
            winn_trades_ = 0
            loss_trades_ = 0
            signals_ = 0
            first_input_price_ = 1.0

            for bar_number_ in range(1, total_bars_):
                if bar_number_ <= last_bar_number_processed_:
                    continue

                # 1. Check input signal: RSI cross over (Long) or cross under (Short) the input level (in_)
                previous_rsi_ = rsi_values[bar_number_ - 1]
                rsi_ = rsi_values[bar_number_]
                # Long.: RSI > in_ AND Previous <= in_
                # Short: RSI < in_ AND Previous >= in_
                if not _crosses_input(previous_rsi_, rsi_, in_, in_, is_long_position):
                    continue

                # This assignment is purely semantic, indicating that once the input condition was met,
                # the bar becomes a market input bar. Active market position, start trading.
                input_bar_number_ = bar_number_

                # Retrieve the pre-calculated stop-loss bar number for this input signal
                stop_loss_bar_number_ = stop_loss_bar_numbers[input_bar_number_]

                # 2. Find output signal, look for output strictly after the input bar number
                # But if stop_loss happens before output, close there.
                output_bar_number_ = -1
                for active_position_bar_number_ in range(input_bar_number_ + 1, total_bars_):
                    # Check stop-loss priority
                    if 0 < stop_loss_bar_number_ < active_position_bar_number_:
                        # Stop-loss break happens before finding output signal, close losing position
                        break

                    # Check output signal: RSI cross under (Long) or cross over (Short) the output level (out_)
                    previous_rsi_out_ = rsi_values[active_position_bar_number_ - 1]
                    rsi_out_ = rsi_values[active_position_bar_number_]
                    # Long.: RSI <= out_ AND Previous > out_
                    # Short: RSI >= out_ AND Previous < out_
                    if _crosses_output(previous_rsi_out_, rsi_out_, out_, out_, is_long_position):
                        output_bar_number_ = active_position_bar_number_
                        break

                # 3. Determine Outcome (stop-loss vs. output)
                # Case A: stop-loss triggered before output was found or reached
                # Note: If output_bar_number_ is -1 (not found), it would be effective in the future
                if 0 < stop_loss_bar_number_ < (output_bar_number_ if output_bar_number_ != -1 else future_bar_number):
                    # Identify a losing trade
                    output_bar_number_ = stop_loss_bar_number_
                    last_bar_number_processed_ = stop_loss_bar_number_

                # Case B: Output signal found, closed position
                # Identify trade with the completed lifecycle for the strategy input and output reached
                elif output_bar_number_ != -1:
                    last_bar_number_processed_ = output_bar_number_

                # Case C: No Output and No Stop Loss found → 'Open Position at end of data'
                # Add trade still open: Mark-to-market using future_bar_number
                else:
                    # Identify a trade remains open
                    output_bar_number_ = future_bar_number
                    last_bar_number_processed_ = total_bars_

                # Scalar Trade PnL (Profit and Loss)
                input_price_ = close_prices[input_bar_number_]
                output_bar_ = _mark_to_market_bar(output_bar_number_, total_bars_)
                output_price_ = close_prices[output_bar_]

                pnl_ = _calculate_trade_pnl(input_price_, output_price_, direction_, COMMISSION_PERCENT)
                if pnl_ > 0.0:
                    winnings_ += pnl_
                    winn_trades_ += 1
                else:
                    losses_ += pnl_
                    loss_trades_ += 1

                if signals_ == 0:
                    first_input_price_ = max(input_price_, 0.00001)
                signals_ += 1

            # Calculate performance metrics for current (in_, out_)
            if signals_ > 0:
                (
                    net_profit_,
                    win_probability_,
                    loss_probability_,
                    average_win_,
                    average_loss_,
                    expected_value_,
                ) = _finalize_screening_metrics(
                    signals_, first_input_price_, winnings_, winn_trades_, losses_, loss_trades_
                )

                if _is_profitable_candidate(net_profit_, expected_value_):
                    new_is_better_ = _is_better_candidate(
                        net_profit_, expected_value_, best_net_profit_, best_expected_value_
                    )
                    if new_is_better_:
                        best_net_profit_ = net_profit_
                        best_expected_value_ = expected_value_
                        best_out_ = out_

        if best_out_ != -1:
            # Valid life cycle found:
            candidates_[candidate_count_, INPUT] = in_
            candidates_[candidate_count_, OUTPUT] = best_out_
            candidate_count_ += 1

    return candidates_[:candidate_count_]


@njit(cache=True)
def _get_out_range(
    is_long_position: bool,
    in_: int,
) -> tuple[int, int]:
    """
    Identify the range of levels for iteration over the RSI output band based on the input level.

    :param is_long_position: Flag of the position type under analysis: long (True) or short (False).
    :param in_: Input level.

    :return: Range of output levels to iterate on RSI Two Bands strategy.
    """
    if is_long_position:
        # It will be used in a loop ─► for range(from_out_, to_out_, -step):
        from_out_ = 84 if in_ < 84 else in_
    else:
        # It will be used in a loop ─► range(from_out_, to_out_, step):
        from_out_ = 16 if in_ > 16 else in_

    to_out_ = in_

    return from_out_, to_out_


class RsiTwoBands(RsiStrategyABC):
    """
    Class to identify Profitable Two Bands (input/output) strategy on the RSI
    Visit https://www.tecnicasdetrading.com/2011/09/tecnica-de-trading-rsi-rollercoaster.html
    """

    def __init__(self, verbosity_level: int = DEBUG):
        """
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process for all
         methods of the class.
         Message levels to be reported: 0-discard messages, 1-report important messages, 2-report details.
        """
        super().__init__(RSI_2B, verbosity_level)

    def identify_old(
        self,
        symbol: str,
        timeframe: int,
        only_long_positions,
        prices_df: pl.DataFrame,
        close_prices: np.ndarray,
        percent_changes: np.ndarray,
        verbosity_level: int = DEBUG,
    ) -> None:
        """
        [DEPRECATED] Legacy baseline method to identify combinations of levels for RSI Two Bands.
        Use identify() instead for an accelerated JIT fused screening implementation.

        Identifies the best combinations of input-output bands for the RSI strategy,
        both for Long and Short positions, and evaluate its profitability on positions:
         - long: open when RSI rises above the lower band and closed when RSI falls below the upper band
         - short: open when RSI falls below the upper band and closed when RSI rises above the lower band.
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
        init_dt_, analysis_context_, original_column_names_, verbosity_level = self.initialize_identification(
            symbol, timeframe, prices_df, verbosity_level
        )

        # Identify and calculate where to stop losses for both long and short positions.
        prices_df = self.identify_where_to_stop_loss(timeframe, prices_df, close_prices)

        # Pre-calculate arrays for Numba. Convert Polars columns to Numpy-arrays once to avoid overhead due to loops.
        rsi_values_ = prices_df['Rsi'].to_numpy()

        # Pre-calculate min/max RSI to skip impossible conditions in loops; nanmin/nanmax ignore initial NaN values (first 14 periods)
        min_rsi_ = np.nanmin(rsi_values_)
        max_rsi_ = np.nanmax(rsi_values_)

        long_stops_ = prices_df['BarNumberForLongStop'].to_numpy().astype(np.int32)
        short_stops_ = prices_df['BarNumberForShortStop'].to_numpy().astype(np.int32)

        future_bar_number_ = analysis_context_.future_bar_number

        # Contexts to iterate:
        #  Position type: LONG.  Levels: '1st input', 'last input', & 'step to increase'
        #  Position type: SHORT. Levels: '1st input', 'last input', & 'step to decrease'
        contexts_ = [(LONG, 16, 61, STEP_LENGTH_RSI_LEVELS)] + ([] if only_long_positions else [(SHORT, 84, 39, -STEP_LENGTH_RSI_LEVELS)])

        # Collect positive ratios across position types / levels for batch upsert
        positive_ratios_ = []

        for position_type_, from_in_, to_in_, step_ in contexts_:
            # Initialize bad strategy to be evaluated and to get better RSI-2Bs
            best_ratios_ = self.initialize_bad_strategy()
            is_long_position_ = position_type_ == LONG
            analysis_context_.is_long_position = is_long_position_

            stop_loss_bar_numbers_ = long_stops_ if is_long_position_ else short_stops_

            # Iterate over the input level of the RSI
            for in_ in range(from_in_, to_in_, step_):
                if verbosity_level == DEBUG:
                    print('', end='\r')
                    print(
                        f'Evaluating profitability {TIMEFRAMES[timeframe]} of RSI({self.period}) input band {in_} for {symbol}...', end=''
                    )

                # Skip if RSI never reaches the input level necessary for a cross (Entry)
                # Long needs rsi > in (strict), so skip if max <= in
                if (is_long_position_ and max_rsi_ <= in_) or (not is_long_position_ and min_rsi_ >= in_):
                    continue

                # Initialize the best strategy for this input level
                best_ratios_for_in_ = self.initialize_bad_strategy()
                # Initialize the strategy for the same level as input and output (strategy of 1 level)
                ratios_for_1_level_ = self.initialize_bad_strategy()

                # Set a range of output levels to be analyzed based on pre-set input and overbought/oversold levels
                from_out_, to_out_ = _get_out_range(is_long_position_, in_)

                # Iterate over the output level of the RSI
                for out_ in range(from_out_, to_out_, -step_):
                    # Skip if RSI never reaches the output level necessary for a cross (Exit)
                    # Long: needs RSI > out_ to cross under. Short: needs RSI < out_ to cross over.
                    if (is_long_position_ and max_rsi_ <= out_) or (not is_long_position_ and min_rsi_ >= out_):
                        continue

                    # Evaluate the life cycle for the RSI Two Bands strategy
                    # (input and output) with the current combination
                    input_bar_numbers_, output_bar_numbers_ = _find_trades_2b(
                        rsi_values_, stop_loss_bar_numbers_, in_, out_, is_long_position_, future_bar_number_
                    )

                    # If no trades identified, skip
                    if len(input_bar_numbers_) == 0:
                        continue

                    # Set strategy Inputs. Period and levels that parameterize the analyzed strategy
                    inputs_ = {'period': self.period, 'in': in_, 'out': out_}

                    # Evaluate trades identified, calculate trading performance ratios and aggregates
                    ratios_ = self.perfile_performance(
                        analysis_context_, inputs_, input_bar_numbers_, output_bar_numbers_, close_prices, percent_changes, prices_df
                    )
                    if not ratios_:
                        continue

                    # Check if RSI 2B just analyzed for this input level, is a better indicator for positioning
                    #  than the previous calculated ones.
                    best_ratios_for_in_ = self.track_best_strategy(ratios_, best_ratios_for_in_)

                    if in_ == out_:
                        ratios_for_1_level_ = ratios_

                if best_ratios_for_in_.inputs != '':
                    strategy_inputs = json.loads(str(best_ratios_for_in_.inputs))
                    best_is_1_level_strategy_ = strategy_inputs['in'] == strategy_inputs['out']
                else:
                    best_is_1_level_strategy_ = False

                if best_ratios_for_in_.net_profit > 0.0 and best_ratios_for_in_.expected_value > 0.0 and not best_is_1_level_strategy_:
                    # Save only positive ratios
                    positive_ratios_.append(best_ratios_for_in_)

                if ratios_for_1_level_.net_profit > 0.0 and ratios_for_1_level_.expected_value > 0.0:
                    # Save only positive ratios for a particular strategy of only 1 level (input-output) analysis
                    positive_ratios_.append(ratios_for_1_level_)

                # Check if the best RSI 2B for this input level is a better indicator for positions
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
        prices_df = prices_df.select(original_column_names_)  # noqa: F841 - unused variable

        # Finalize the process to identify profitable strategies and logs finalization
        self.finalize_identification(init_dt_, analysis_context_, verbosity_level)

    def identify(
        self,
        symbol: str,
        timeframe: int,
        only_long_positions: bool,
        prices_df: pl.DataFrame,
        close_prices: np.ndarray,
        percent_changes: np.ndarray,
        verbosity_level: int = DEBUG,
    ) -> None:
        """
        Identifies the best combinations of input-output bands for the RSI strategy using the fused Numba JIT grid
         search kernel, both for Long and Short positions, and evaluates its profitability on positions:
         - long: open when RSI rises above the lower band and closed when RSI falls below the upper band
         - short: open when RSI falls below the upper band and closed when RSI rises above the lower band.
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
        init_dt_, analysis_context_, original_column_names_, verbosity_level = self.initialize_identification(
            symbol, timeframe, prices_df, verbosity_level
        )

        # Identify and calculate where to stop losses for both long and short positions.
        prices_df = self.identify_where_to_stop_loss(timeframe, prices_df, close_prices)

        # Pre-calculate arrays for Numba. Convert Polars columns to Numpy-arrays once to avoid overhead due to loops.
        rsi_values_ = prices_df['Rsi'].to_numpy()

        long_stops_ = prices_df['BarNumberForLongStop'].to_numpy().astype(np.int32)
        short_stops_ = prices_df['BarNumberForShortStop'].to_numpy().astype(np.int32)

        future_bar_number_ = analysis_context_.future_bar_number

        # Contexts to iterate:
        #  Position type: LONG.  Levels: '1st input', 'last input', & 'step to increase'
        #  Position type: SHORT. Levels: '1st input', 'last input', & 'step to decrease'
        contexts_ = [(LONG, 16, 61, STEP_LENGTH_RSI_LEVELS)] + ([] if only_long_positions else [(SHORT, 84, 39, -STEP_LENGTH_RSI_LEVELS)])

        # Collect positive ratios across position types / levels for batch upsert
        positive_ratios_ = []

        for position_type_, from_in_, to_in_, step_ in contexts_:
            # Initialize bad strategy to be evaluated and to get better RSI-2Bs
            best_ratios_ = self.initialize_bad_strategy()
            is_long_position_ = position_type_ == LONG
            analysis_context_.is_long_position = is_long_position_

            stop_loss_bar_numbers_ = long_stops_ if is_long_position_ else short_stops_

            if verbosity_level == DEBUG:
                print('', end='\r')
                print(f'Evaluating profitability {TIMEFRAMES[timeframe]} of RSI({self.period}) Two Bands (fused) for {symbol}...', end='')

            best_candidates_ = _grid_search_2b_fused(
                rsi_values_,
                stop_loss_bar_numbers_,
                close_prices,
                from_in_,
                to_in_,
                step_,
                is_long_position_,
                future_bar_number_,
            )

            # Materialize complete Ratios objects only for surviving best candidates/combinations
            for i_ in range(len(best_candidates_)):
                # Extract RSI level values from the best candidates
                in_ = int(best_candidates_[i_, INPUT])
                out_ = int(best_candidates_[i_, OUTPUT])

                # Reconstruct the trade lifecycle using the existing JIT kernel.
                # (input and output) with the current candidate/combination.
                input_bar_numbers_, output_bar_numbers_ = _find_trades_2b(
                    rsi_values_, stop_loss_bar_numbers_, in_, out_, is_long_position_, future_bar_number_
                )
                # If no trades identified, skip
                if len(input_bar_numbers_) == 0:
                    continue

                # Set strategy Inputs. Period and levels that parameterize the analyzed strategy
                inputs_ = {'period': self.period, 'in': in_, 'out': out_}

                # Evaluate trades identified, calculate trading performance ratios and aggregates
                ratios_ = self.perfile_performance(
                    analysis_context_, inputs_, input_bar_numbers_, output_bar_numbers_, close_prices, percent_changes, prices_df
                )
                if not ratios_:
                    continue

                # Save positive ratios
                if ratios_.net_profit > 0.0 and ratios_.expected_value > 0.0:
                    positive_ratios_.append(ratios_)

                # Check if the best RSI 2B for this input level is a better indicator for positions
                # than the previously calculated input levels.
                best_ratios_ = self.track_best_strategy(ratios_, best_ratios_)

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
        prices_df = prices_df.select(original_column_names_)  # noqa: F841 - unused variable

        # Finalize the process to identify profitable strategies and logs finalization
        self.finalize_identification(init_dt_, analysis_context_, verbosity_level)
