# src/radar_core/domain/strategies/ma.py

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
from radar_core.helpers.constants import COMMISSION_PERCENT, LONG, SHORT, TIMEFRAMES


# In HPC (High Performance Computing), it is the best practice to decouple compute-intensive logic (the kernel)
# from orchestration logic (the class). `_find_trades_sma` acts as a pure function: it accepts Numpy arrays and integers,
# and returns NumPy arrays, without accessing or modifying the class state. Keeping it at the module level reinforces this separation.
@njit(cache=True)
def _find_trades_sma(
    values: np.ndarray,
    period: int,
    is_long_position: bool,
    future_bar_number: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast JIT-compiled kernel to identify trades based on Moving Average crossovers.
    It calculates the SMA on-the-fly to avoid memory allocation for intermediate arrays.

    :param values: Array of values (e.g., Close prices or RSI) to compute MA and check crosses.
    :param period: The moving average period.
    :param is_long_position: True for Long strategy (Buy if Value > SMA), False for Short.
    :param future_bar_number: The number of a price bar that will be available in the future.

    :return: Tuple of numpy arrays with the input and output bar numbers for each trade.
    """
    total_bars_ = len(values)
    maximum_trades_ = total_bars_ // 2 + 1
    input_bar_numbers_ = np.empty(maximum_trades_, dtype=np.int32)
    output_bar_numbers_ = np.empty(maximum_trades_, dtype=np.int32)
    trade_count_ = 0

    # It needs at least 'period' elements to calculate the first valid SMA.
    # The first valid SMA corresponds to index 'period - 1'.
    # Start iterating from 'period' to check the cross between (i-1) and (i).
    if total_bars_ <= period:
        return input_bar_numbers_[:0], output_bar_numbers_[:0]

    # Efficient SMA calculation using a running `sum`
    # Initialize the `sum` for the first window [0: period]
    # Identify the first bar number with a valid value (non-NaN)
    first_nan_bar_ = 0
    while first_nan_bar_ < total_bars_ and np.isnan(values[first_nan_bar_]):
        first_nan_bar_ += 1

    # Check if there are enough valid elements after leading NaNs to get the first window
    # It needs at least 'period' elements to calculate the first valid SMA.
    if (total_bars_ - first_nan_bar_) <= period:
        return input_bar_numbers_[:0], output_bar_numbers_[:0]

    # State variables
    in_position_ = False

    # Initialize the running sum from the first valid index
    current_sum_ = 0.0
    for i_ in range(first_nan_bar_, first_nan_bar_ + period):
        current_sum_ += values[i_]

    # Calculate the first valid SMA at the end of the first valid window
    # Example: if first_nan_bar_ is 14, due to a RSI(14), and period is 20, the first SMA is at index 33
    first_valid_sma_bar_ = first_nan_bar_ + period - 1
    previous_sma_ = current_sum_ / period
    previous_value_ = values[first_valid_sma_bar_]

    # Iterate starting from the first bar AFTER the initial window
    for i_ in range(first_valid_sma_bar_ + 1, total_bars_):
        current_value_ = values[i_]

        # Update Running Sum: add new value, remove old value leaving the window
        # Value leaving is at index (i - period)
        current_sum_ = current_sum_ + current_value_ - values[i_ - period]
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
                input_bar_numbers_[trade_count_] = i_
                in_position_ = True
        else:
            # Look for output
            output_signal_ = cross_under_ if is_long_position else cross_over_
            if output_signal_:
                output_bar_numbers_[trade_count_] = i_
                trade_count_ += 1
                in_position_ = False

        # Update previous state for next iteration
        previous_sma_ = current_sma_
        previous_value_ = current_value_

    # Handle Open Position at the end of data (Mark-to-Market)
    if in_position_:
        output_bar_numbers_[trade_count_] = future_bar_number
        trade_count_ += 1

    return input_bar_numbers_[:trade_count_], output_bar_numbers_[:trade_count_]


@njit(cache=True)
def _grid_search_ma_fused(
    values: np.ndarray,
    close_prices: np.ndarray,
    min_period: int,
    max_period: int,
    is_long_position: bool,
    future_bar_number: int,
) -> tuple[np.ndarray, int]:
    """
    Fast JIT-compiled fused grid search kernel for Moving Average crossover strategies.
    Evaluates all periods from min_period to max_period, simulates trades on-the-fly,
    computes scalar PnL directly in registers, and returns the array of winning periods
    where net_profit > 0.0 and expected_value > 0.0, along with the single best overall period.

    :param values: Array of values (e.g. Close prices or RSI) to compute MA and check crossovers.
    :param close_prices: Array of underlying asset close prices used for trade PnL valuation.
    :param min_period: Minimum period for the moving average calculation.
    :param max_period: Maximum period for the moving average calculation.
    :param is_long_position: True for Long positions, False for Short positions.
    :param future_bar_number: Future bar number marker for open trades.

    :return: Tuple containing 1D array of winning period integers and the overall best period integer.
    """
    total_bars_ = len(values)
    direction_ = 1.0 if is_long_position else -1.0
    last_bar_number_ = total_bars_ - 1

    total_periods_ = max_period - min_period + 1
    candidates_ = np.empty(total_periods_, dtype=np.int32)
    candidate_count_ = 0

    best_net_profit_ = -np.inf
    best_expected_value_ = -np.inf
    best_period_ = -1

    first_nan_bar_ = 0
    while first_nan_bar_ < total_bars_ and np.isnan(values[first_nan_bar_]):
        first_nan_bar_ += 1

    valid_bars_ = total_bars_ - first_nan_bar_

    for period_ in range(min_period, max_period + 1):
        if valid_bars_ <= period_:
            continue

        in_position_ = False
        current_sum_ = 0.0
        for i_ in range(first_nan_bar_, first_nan_bar_ + period_):
            current_sum_ += values[i_]

        first_valid_sma_bar_ = first_nan_bar_ + period_ - 1
        previous_sma_ = current_sum_ / period_
        previous_value_ = values[first_valid_sma_bar_]

        winnings_ = 0.0
        losses_ = 0.0
        winn_trades_ = 0
        loss_trades_ = 0
        signals_ = 0
        first_input_price_ = 0.0
        current_input_bar_ = -1

        for i_ in range(first_valid_sma_bar_ + 1, total_bars_):
            current_value_ = values[i_]
            current_sum_ = current_sum_ + current_value_ - values[i_ - period_]
            current_sma_ = current_sum_ / period_

            is_above_ = current_value_ > current_sma_
            was_above_ = previous_value_ > previous_sma_

            cross_over_ = is_above_ and not was_above_
            cross_under_ = not is_above_ and was_above_

            if not in_position_:
                input_signal_ = cross_over_ if is_long_position else cross_under_
                if input_signal_:
                    current_input_bar_ = i_
                    in_position_ = True
            else:
                output_signal_ = cross_under_ if is_long_position else cross_over_
                if output_signal_:
                    input_price_ = close_prices[current_input_bar_]
                    output_price_ = close_prices[i_]

                    pnl_ = (output_price_ - input_price_) * direction_ - COMMISSION_PERCENT * (input_price_ + output_price_)
                    if pnl_ > 0.0:
                        winnings_ += pnl_
                        winn_trades_ += 1
                    else:
                        losses_ += pnl_
                        loss_trades_ += 1

                    if signals_ == 0:
                        first_input_price_ = max(input_price_, 0.00001)
                    signals_ += 1
                    in_position_ = False

            previous_sma_ = current_sma_
            previous_value_ = current_value_

        # Handle open position at end of data (Mark-to-Market)
        if in_position_:
            input_price_ = close_prices[current_input_bar_]
            output_price_ = close_prices[last_bar_number_]

            pnl_ = (output_price_ - input_price_) * direction_ - COMMISSION_PERCENT * (input_price_ + output_price_)
            if pnl_ > 0.0:
                winnings_ += pnl_
                winn_trades_ += 1
            else:
                losses_ += pnl_
                loss_trades_ += 1

            if signals_ == 0:
                first_input_price_ = max(input_price_, 0.00001)
            signals_ += 1

        if signals_ > 0:
            net_profit_ = (winnings_ + losses_) / first_input_price_
            win_probability_ = winn_trades_ / signals_
            loss_probability_ = loss_trades_ / signals_
            average_win_ = winnings_ / winn_trades_ if winn_trades_ > 0 else 0.0
            average_loss_ = losses_ / loss_trades_ if loss_trades_ > 0 else 0.0
            expected_value_ = win_probability_ * average_win_ + loss_probability_ * average_loss_

            new_is_better_ = net_profit_ > best_net_profit_ or (
                net_profit_ == best_net_profit_ and expected_value_ > best_expected_value_
            )
            if new_is_better_:
                best_net_profit_ = net_profit_
                best_expected_value_ = expected_value_
                best_period_ = period_

            if net_profit_ > 0.0 and expected_value_ > 0.0:
                candidates_[candidate_count_] = period_
                candidate_count_ += 1

    return candidates_[:candidate_count_], best_period_


class MovingAverage(StrategyABC):
    """
    Class to identify and evaluate the Profitable Moving Average strategy.
    """

    def __init__(
        self,
        strategy_acronym: str,
        value_column_name: str,
        ma_column_name: str,
        min_period: int = 8,
        max_period: int = 233,
        verbosity_level: int = DEBUG,
    ) -> None:
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

    def identify_old(
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
        [DEPRECATED] Legacy baseline method to identify and evaluate the Profitable Moving Average strategy.
        Iterate from the minimum to the maximum number of periods to calculate the MA and evaluate its profitability
         on positions:
         - long: open when the value rises above MA and closed when the value falls below MA
         - short: open when the value falls below MA and closed when the value rises above MA.
        Save the profitable setups (identified Moving Average) in the Database.

        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe at least with required columns
         [DateTime, {self.value_column_name}, PercentChange, BarNumber].
        :param close_prices: Close prices for the given symbol and timeframe.
        :param percent_changes: Percent change of the close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Check if the column exists
        if self.value_column_name not in prices_df.columns:
            raise KeyError(
                f'The column [{self.value_column_name}] on whose values the moving average will be'
                f' calculated does not exist in the Prices DataFrame.'
            )

        # Logs initialization and prepares the necessary variables for the process
        init_dt_, analysis_context_, original_column_names_, verbosity_level = (
            self.initialize_identification(symbol, timeframe, prices_df, verbosity_level)
        )

        # Pre-calculate arrays for Numba/Vectorized operations
        # Extract values to calculate SMA (usually Close or RSI)
        values_ = prices_df[self.value_column_name].to_numpy()

        # Identify
        future_bar_number_ = analysis_context_.future_bar_number

        # Position types to iterate
        position_types_ = [LONG] + ([] if only_long_positions else [SHORT])

        # Collect positive ratios across position types / periods for batch upsert
        positive_ratios_ = []

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
                        end=''
                    )

                if len(values_) <= period_:
                    # The minimum number of periods to calculate the average is not reached
                    continue

                # Calculate SMA signals using Numba (no intermediate Polars objects)
                input_bar_numbers_, output_bar_numbers_ = _find_trades_sma(
                    values_, period_, is_long_position_, future_bar_number_
                )

                if len(input_bar_numbers_) == 0:
                    # There are no valid signals, skip further processing
                    continue

                # Set strategy Inputs. Period that parameterizes the analyzed strategy.
                inputs_ = {'period': period_}

                # Evaluate trades identified, calculate trading performance ratios and aggregates
                ratios_ = self.perfile_performance(
                    analysis_context_,
                    inputs_,
                    input_bar_numbers_,
                    output_bar_numbers_,
                    close_prices,
                    percent_changes,
                    prices_df,
                )
                if not ratios_:
                    continue

                if ratios_.net_profit > 0.0 and ratios_.expected_value > 0.0:
                    # Save only positive ratios
                    positive_ratios_.append(ratios_)

                # Check if MA just analyzed is a better indicator for positionings than the previous calculated ones.
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

        # Reset to the original columns
        prices_df = prices_df.select(original_column_names_)

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
        Identifies the best moving average periods using the fused Numba JIT grid search kernel,
        both for Long and Short positions, and evaluates its profitability on positions:
         - long: open when the value rises above MA and closed when the value falls below MA
         - short: open when the value falls below MA and closed when the value rises above MA.
        Save the profitable setups (identified Moving Average and associated ratios) in the Database.

        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe at least with required columns
         [DateTime, {self.value_column_name}, PercentChange, BarNumber].
        :param close_prices: Close prices for the given symbol and timeframe.
        :param percent_changes: Percent change of the close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Check if the column exists
        if self.value_column_name not in prices_df.columns:
            raise KeyError(
                f'The column [{self.value_column_name}] on whose values the moving average will be'
                f' calculated does not exist in the Prices DataFrame.'
            )

        # Logs initialization and prepares the necessary variables for the process
        init_dt_, analysis_context_, original_column_names_, verbosity_level = (
            self.initialize_identification(symbol, timeframe, prices_df, verbosity_level)
        )

        # Pre-calculate arrays for Numba/Vectorized operations
        # Extract values to calculate SMA (usually Close or RSI)
        values_ = prices_df[self.value_column_name].to_numpy()

        future_bar_number_ = analysis_context_.future_bar_number

        # Position types to iterate
        position_types_ = [LONG] + ([] if only_long_positions else [SHORT])

        # Collect positive ratios across position types / periods for batch upsert
        positive_ratios_ = []

        for position_type_ in position_types_:
            # Initialize bad strategy to be evaluated and to get better MAs
            best_ratios_ = self.initialize_bad_strategy()
            is_long_position_ = position_type_ == LONG
            analysis_context_.is_long_position = is_long_position_

            if verbosity_level == DEBUG:
                print('', end='\r')
                print(
                    f'Evaluating profitability {TIMEFRAMES[timeframe]} of {self.strategy_acronym} '
                    f'(fused) for {symbol}...',
                    end=''
                )

            winning_periods_, best_overall_period_ = _grid_search_ma_fused(
                values_,
                close_prices,
                self.min_period,
                self.max_period,
                is_long_position_,
                future_bar_number_,
            )

            # Determine surviving periods to materialize into Ratios
            if len(winning_periods_) > 0:
                periods_to_materialize_ = winning_periods_
            elif best_overall_period_ != -1:
                periods_to_materialize_ = np.array([best_overall_period_], dtype=np.int32)
            else:
                periods_to_materialize_ = np.empty(0, dtype=np.int32)

            # Materialize complete Ratios objects only for surviving winning periods
            for period_ in periods_to_materialize_:
                input_bar_numbers_, output_bar_numbers_ = _find_trades_sma(
                    values_,
                    int(period_),
                    is_long_position_,
                    future_bar_number_,
                )

                if len(input_bar_numbers_) == 0:
                    continue

                inputs_ = {'period': int(period_)}

                ratios_ = self.perfile_performance(
                    analysis_context_,
                    inputs_,
                    input_bar_numbers_,
                    output_bar_numbers_,
                    close_prices,
                    percent_changes,
                    prices_df,
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
                analysis_context_.best_long = best_ratios_  # Best Long strategies
            else:
                analysis_context_.best_short = best_ratios_  # Best Short strategies

        # Perform atomic batch upsert for all positive ratios identified and remove remaining flagged rows atomically.
        self.persist_ratios(positive_ratios_, analysis_context_)

        # Reset to the original columns
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization
        self.finalize_identification(init_dt_, analysis_context_, verbosity_level)
