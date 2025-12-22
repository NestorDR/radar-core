# src/radar_core/domain/strategies/rsi2b.py

# --- Python modules ---
# json: library for encoding and decoding prices in JSON format.
import json
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG
# typing: provides runtime support for type hints.

# --- Third Party Libraries ---
# numba: JIT compiler that compiles a subset of Python and NumPy code into optimized machine code
from numba import njit
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.base_strategy import AnalysisContext, RsiStrategyABC
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import COMMISSION_PERCENT, RSI_2B, LONG, SHORT, STEP_LENGTH_RSI_LEVELS, TIMEFRAMES
# models: result of Object-Relational Mapping
from radar_core.models import Ratios

# Column constants for work matrices: inputs and outputs
BAR_NUMBER = 0
PRICE = 1
PERCENT_CHANGE = 2


# In HPC (High Performance Computing), it is a best practice to decouple compute-intensive logic (the "kernel")
# from orchestration logic (the class). `_find_trades_2b` acts as a pure function: it accepts Numpy arrays and integers,
# and returns lists, without accessing or modifying the class state. Keeping it at the module level reinforces this separation.
@njit(cache=True)
def _find_trades_2b(rsi_values: np.ndarray,
                    stop_loss_bar_numbers: np.ndarray,
                    in_: int,
                    out_: int,
                    is_long_position: bool,
                    future_bar_number: int) -> tuple[np.ndarray, np.ndarray]:
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
            # Long: RSI <= out_ AND Prev > out_
            # Short: RSI >= out_ AND Prev < out_
            if rsi_ <= out_ < previous_rsi_ if is_long_position else rsi_ >= out_ > previous_rsi_:
                output_bar_number_ = active_position_bar_number_
                break

        # 3. Determine Outcome (stop-loss vs. output)
        # Case A: stop-loss triggered before output was found or reached
        # Note: If output_bar_number_ is -1 (not found), it would be effectively in the future
        if 0 < stop_loss_bar_number_ < (output_bar_number_ if output_bar_number_ != -1 else future_bar_number):
            # Add losing trade
            input_bar_numbers_.append(input_bar_number_)
            output_bar_numbers_.append(stop_loss_bar_number_)
            last_bar_number_processed_ = stop_loss_bar_number_
            continue

        # Add Trade
        input_bar_numbers_.append(input_bar_number_)
        if output_bar_number_ != -1:
            # Case B: Output signal found, closed position
            # Add trade with the completed life cycle for the strategy input and output reached
            output_bar_numbers_.append(output_bar_number_)
            last_bar_number_processed_ = output_bar_number_
            continue

        # Case D: No Output and No Stop Loss found → "Open Position at end of data"
        # Add trade still open: Mark-to-market using future_bar_number
        output_bar_numbers_.append(future_bar_number)
        last_bar_number_processed_ = total_bars_

    # Convert lists to arrays facilitating further processing
    return np.array(input_bar_numbers_, dtype=np.int32), np.array(output_bar_numbers_, dtype=np.int32)


class RsiTwoBands(RsiStrategyABC):
    """
    Class to identify Profitable Two Bands (input/output) strategy on the RSI
    Visit https://www.tecnicasdetrading.com/2011/09/tecnica-de-trading-rsi-rollercoaster.html
    """

    def __init__(self,
                 verbosity_level: int = DEBUG):
        """
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process for all
         methods of the class.
         Message levels to be reported: 0-discard messages, 1-report important messages, 2-report details.
        """
        super().__init__(RSI_2B, verbosity_level)

    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions,
                 prices_df: pl.DataFrame,
                 close_prices: np.ndarray | None,  # type: ignore
                 verbosity_level: int = DEBUG) -> dict:
        """
        Identifies the best combinations of bands input and output for the RSI strategy,
        both for Long and Short positions, and evaluate its profitability on positions:
         - long: open when RSI rises above the lower band and closed when RSI falls below the upper band
         - short: open when RSI falls below the upper band and closed when RSI rises above the lower band.
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

        if close_prices is None:
            # Extract Close prices to an array to speed up prices access
            close_prices = prices_df['Close'].to_numpy()

        prices_df = self.identify_where_to_stop_loss(timeframe, prices_df, close_prices)

        # Pre-calculate arrays for Numba. Convert Polars columns to Numpy arrays once to avoid overhead due to loops.
        rsi_values_ = prices_df['Rsi'].to_numpy()
        pct_change_values_ = prices_df['PercentChange'].to_numpy()

        # Pre-calculate min/max RSI to skip impossible conditions in loops; nanmin/nanmax ignore initial NaN values (first 14 periods)
        min_rsi_ = np.nanmin(rsi_values_)
        max_rsi_ = np.nanmax(rsi_values_)

        long_stops_ = prices_df['BarNumberForLongStop'].to_numpy().astype(np.int32)
        short_stops_ = prices_df['BarNumberForShortStop'].to_numpy().astype(np.int32)

        future_bar_number_ = analysis_context_.future_bar_number

        # Contexts to iterate:
        #  Position type: LONG.  Levels: '1st input', 'last input', & 'step to increase'
        #  Position type: SHORT. Levels: '1st input', 'last input', & 'step to decrease'
        contexts_ = [(LONG, 16, 61, STEP_LENGTH_RSI_LEVELS)] + \
                    ([] if only_long_positions else [(SHORT, 84, 39, -STEP_LENGTH_RSI_LEVELS)])

        for position_type_, from_in_, to_in_, step_ in contexts_:
            # Initialize bad strategy to be evaluated and to get better RSI-2Bs
            best_ratios_ = self.initialize_bad_strategy()
            analysis_context_.is_long_position = position_type_ == LONG

            is_long_position_ = analysis_context_.is_long_position
            stop_loss_bar_numbers_ = long_stops_ if is_long_position_ else short_stops_

            # Iterate over the input level of the RSI
            for in_ in range(from_in_, to_in_, step_):
                if verbosity_level == DEBUG:
                    print('', end='\r')
                    print(f'Evaluating profitability {TIMEFRAMES[timeframe]} of RSI({self.period})'
                          f' input band {in_} for {symbol}...', end='')

                # Skip if RSI never reaches the input level necessary for a cross (Entry)
                # Long needs rsi > in (strict), so skip if max <= in
                if (is_long_position_ and max_rsi_ <= in_) or (not is_long_position_ and min_rsi_ >= in_):
                    continue

                # Initialize the best strategy for this input level
                best_ratios_for_in_ = self.initialize_bad_strategy()
                # Initialize the strategy for a same level as input and output (strategy of 1 level)
                ratios_for_1_level_ = self.initialize_bad_strategy()

                # Set a range of output levels to be analyzed based on pre-set input and overbought/oversold levels
                from_out_, to_out_ = self.__get_out_range(is_long_position_, in_)

                # Iterate over the output level of the RSI
                for out_ in range(from_out_, to_out_, -step_):
                    # Skip if RSI never reaches the output level necessary for a cross (Exit)
                    # Long: needs RSI > out_ to cross under. Short: needs RSI < out_ to cross over.
                    if (is_long_position_ and max_rsi_ <= out_) or (not is_long_position_ and min_rsi_ >= out_):
                        continue

                    # Evaluate the life cycle for the RSI Two Bands strategy
                    # (input and output) with the current combination
                    input_bar_numbers_, output_bar_numbers_ = _find_trades_2b(rsi_values_, stop_loss_bar_numbers_,
                                                                              in_, out_,
                                                                              is_long_position_, future_bar_number_)

                    # If no trades identified, skip
                    if len(input_bar_numbers_) == 0:
                        continue

                    # Evaluate trades identified, calculate trading performance ratios and aggregates
                    ratios_ = self.__analyze_fast(in_, out_, input_bar_numbers_, output_bar_numbers_,
                                                  close_prices, pct_change_values_, position_type_,
                                                  analysis_context_, prices_df)
                    if not ratios_:
                        continue

                    # Check if RSI 2B just analyzed for this input level, is a better indicator for positionings
                    #  than the previous calculated ones.
                    best_ratios_for_in_ = self.track_best_strategy(ratios_, best_ratios_for_in_)

                    if in_ == out_:
                        ratios_for_1_level_ = ratios_

                if best_ratios_for_in_.inputs != '':
                    strategy_inputs = json.loads(str(best_ratios_for_in_.inputs))
                    best_is_1_level_strategy_ = strategy_inputs['in'] == strategy_inputs['out']
                else:
                    best_is_1_level_strategy_ = False

                if (best_ratios_for_in_.net_profit > 0 and best_ratios_for_in_.expected_value > 0
                        and not best_is_1_level_strategy_):
                    # Save only positive ratios
                    self.ratio_crud.upsert(best_ratios_for_in_)

                if ratios_for_1_level_.net_profit > 0 and ratios_for_1_level_.expected_value > 0:
                    # Save only positive ratios for a particular strategy of only 1 level (input-output) analysis
                    self.ratio_crud.upsert(ratios_for_1_level_)

                # Check if the best RSI 2B for this input level is a better indicator for positions
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

    def __analyze_fast(self,
                       in_: int,
                       out_: int,
                       input_bar_numbers: np.ndarray,
                       output_bar_numbers: np.ndarray,
                       close_prices: np.ndarray,
                       pct_change_values: np.ndarray,
                       position_type: int,
                       analysis_context: AnalysisContext,
                       prices_df: pl.DataFrame) -> Ratios | None:
        """
         Calculates the results and ratios of applying the RSI Two Bands strategy, with the combination of RSI levels
         and flag of the position type (Long or Short) received as parameters.

        :param in_: Input level for the strategy.
        :param out_: Output level for the strategy.
        :param input_bar_numbers: Array of input signal bar numbers.
        :param output_bar_numbers: Array of output signal bar numbers.
        :param close_prices: Array of 'Close' prices extracted from prices_df.
        :param position_type: Position type under analysis: long (1) or short (-1).
        :param analysis_context: Analysis context for the strategy.
        :param prices_df: The dataFrame with prices, indexed by bar numbers and containing the required column Date.

        :return: If the winnings exceed the losses return a Ratios object with the ratios and aggregates calculated for
          trade performance, otherwise returns None.
        """
        signals_ = len(input_bar_numbers)

        # Retrieve values using vectorization (fancy indexing)
        input_prices_ = close_prices[input_bar_numbers]
        input_pct_change_ = pct_change_values[input_bar_numbers]

        # Handling output prices for Open Positions (Mark-to-Market - valuation of assets at current market prices):
        # If OutputBarNumber is future_bar_number, we must use the last available price,
        # but we preserve future_bar_number in the DataFrame for semantics.
        last_bar_number_ = len(close_prices) - 1

        # Create a temporary index array clamped to the last valid index
        # np.minimum ensures that any index >= len(close_prices) (like future_bar_number) becomes last_bar_number_
        safe_output_bar_numbers_ = np.minimum(output_bar_numbers, last_bar_number_)
        output_prices_ = close_prices[safe_output_bar_numbers_]

        trades_df_ = pl.DataFrame(
            {
                "InputBarNumber": input_bar_numbers,
                "InputPrice": input_prices_,
                "InputPercentChange": input_pct_change_,
                "OutputBarNumber": output_bar_numbers,
                "OutputPrice": output_prices_
            },
            schema=["InputBarNumber", "InputPrice", "InputPercentChange", "OutputBarNumber", "OutputPrice"],
            orient="col"
        ).with_columns([
            ((pl.col('OutputPrice') - pl.col('InputPrice')) * position_type
             - COMMISSION_PERCENT * (pl.col('InputPrice') + pl.col('OutputPrice')))
            .alias('Result').cast(pl.Float64),
            (pl.col('OutputBarNumber') - pl.col('InputBarNumber')).alias('Sessions').cast(pl.Int32),
            pl.col("InputBarNumber").cast(pl.Int32),
            pl.col("InputPrice").cast(pl.Float64),
            pl.col("InputPercentChange").cast(pl.Float64),
            pl.col("OutputBarNumber").cast(pl.Int32),
            pl.col("OutputPrice").cast(pl.Float64),
        ])

        # Period and levels that parameterize the analyzed strategy
        inputs_ = {'period': self.period, 'in': in_, 'out': out_}

        # Calculate trading performance ratios and aggregates
        ratios_ = self.perfile_performance(analysis_context, inputs_, signals_, trades_df_, prices_df)

        # Release memory
        del trades_df_

        return ratios_

    @staticmethod
    def __get_out_range(is_long_position: bool,
                        in_: int) -> tuple[int, int]:
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
