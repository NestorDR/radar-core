# src/radar_core/domain/strategies/rsirc.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG
# typing: provides runtime support for type hints.
from typing import Any

# --- Third Party Libraries ---
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
from numpy.typing import NDArray
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.constants import COMMISSION_PERCENT, RSI_RC, LONG, SHORT, STEP_LENGTH_RSI_LEVELS
from radar_core.domain.strategies.base_strategy import AnalysisContext, RsiStrategyABC
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import TIMEFRAMES
# models: result of Object-Relational Mapping
from radar_core.models import Ratios

# Column constants for work matrices: inputs and outputs
BAR_NUMBER = 0
PRICE = 1
PERCENT_CHANGE = 2


class RsiRollerCoaster(RsiStrategyABC):
    """
    Class to identify Profitable Roller Coaster strategy on the RSI (PRSIRC)
    Visit> https://www.tecnicasdetrading.com/2011/09/tecnica-de-trading-rsi-rollercoaster.html
    """

    def __init__(self,
                 verbosity_level: int = DEBUG):
        """
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process for all
         methods of the class.
         Message levels to be reported: 0-discard messages, 1-report important messages, 2-report details.
        """
        super().__init__(RSI_RC, verbosity_level)

    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions,
                 prices_df: pl.DataFrame,
                 close_prices: NDArray[Any] | None,  # type: ignore
                 verbosity_level: int = DEBUG) -> dict:
        """
        Identifies the best combinations of levels input, overbought/oversold and output for the RSI Rollercoaster
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

        if close_prices is None:
            # Extract Close prices to an array to speed up prices access
            close_prices = prices_df['Close'].to_numpy()

        prices_df = self.identify_where_to_stop_loss(timeframe, prices_df, close_prices)

        # Contexts to iterate:
        #  Position type: LONG.  Levels: '1st input', 'last input', '1st overbought', 'last overbought' & 'step to increase'
        #  Position type: SHORT. Levels: '1st input', 'last input', '1st oversold', 'last oversold' & 'step to decrease'
        # contexts_ = [ (LONG, 20, 41, 50, 81, 1), (SHORT, 75, 64, 50, 19, -1) ]
        contexts_ = [(LONG, 16, 61, 40, 81, STEP_LENGTH_RSI_LEVELS)] + \
                    ([] if only_long_positions else [(SHORT, 84, 58, 60, 19, -STEP_LENGTH_RSI_LEVELS)])

        for position_type_, from_in_, to_in_, from_over_, to_over_, step_ in contexts_:
            # Initialize bad strategy to be evaluated and to get better RSI-RCs
            best_ratios_ = self.initialize_bad_strategy()
            analysis_context_.is_long_position = position_type_ == LONG

            # Iterate over the input level of the RSI
            for in_ in range(from_in_, to_in_, step_):
                if verbosity_level == DEBUG:
                    print('', end='\r')
                    print(f'Evaluating profitability {TIMEFRAMES[timeframe]} of RSI({self.period}) Rollercoaster input level {in_} for {symbol}...', end='')

                # Initialize the best strategy for this input level
                best_ratios_for_in_ = self.initialize_bad_strategy()

                # Define the condition based on the position type
                # [Long|Short] input signals: RSI [raises over|falls below] input level
                in_condition_ = (pl.col('Rsi') > in_) if analysis_context_.is_long_position else (pl.col('Rsi') < in_)
                # Generate input signals
                prices_df = prices_df.with_columns([in_condition_.cast(pl.Int8).diff().alias('In')])

                # Generate start of life cycle with positionings or inputs
                inputs_ = prices_df.filter(pl.col('In') == 1).select(
                    ['BarNumber', 'PercentChange',
                     'BarNumberForLongStop' if analysis_context_.is_long_position else 'BarNumberForShortStop']
                ).to_numpy()

                # If no input signals, skip the combination of levels: input, over[bought|sold], output
                if len(inputs_) == 0:
                    continue

                # Iterate over the overbought/oversold level of the RSI
                for over_ in range(from_over_, to_over_, step_):
                    # Define condition based on the position type
                    # Over[bought|sold] signals: RSI [raises over|falls below] rollercoaster over[bought|sold] level
                    over_condition_ = (pl.col('Rsi') >= over_) if analysis_context_.is_long_position else (pl.col('Rsi') <= over_)
                    # Generate over signals
                    prices_df = prices_df.with_columns([over_condition_.cast(pl.Int8).diff().alias('Over')])

                    # Generate the life cycle part for overbought/oversold
                    overs_ = prices_df.filter(pl.col('Over') == 1)['BarNumber'].to_numpy()

                    # If no over signals, skip the combination of levels: input, over[bought|sold], output.
                    # It would be Buy-&-Hold.
                    if len(overs_) == 0:
                        continue

                    # Set a range of output levels to be analyzed based on pre-set input and over[bought|sold] levels
                    from_out_, to_out_ = self.__get_out_range(analysis_context_.is_long_position, in_, over_)

                    # Iterate over the output level of the RSI
                    for out_ in range(from_out_, to_out_, -step_):
                        # Define the condition based on the position type
                        # [Long|Short] output signals: RSI [falls below|raises over] rollercoaster output Level
                        out_condition_ = (pl.col('Rsi') <= out_) if analysis_context_.is_long_position else (pl.col('Rsi') >= out_)
                        # Generate output signals
                        prices_df = prices_df.with_columns([out_condition_.cast(pl.Int8).diff().alias('Out')])

                        # Generate the life cycle part for outputs
                        outputs_ = prices_df.filter(pl.col('Out') == 1)['BarNumber'].to_numpy()

                        # if in_ == 38 and over_ == 79 and out_ == 50:
                        #     print('in_ = {in_}, over_ = {over_}, out_ = {out_}')
                        #     print('inputs_ = {inputs_}')
                        #     print('overs_ = {overs_}')
                        #     print('outputs_ = {outputs_}')

                        # If no output signals, skip the combination of levels: input, over[bought|sold], output.
                        # It would be Buy-&-Hold.
                        if len(outputs_) == 0:
                            continue

                        ratios_ = self.__analyze(in_, over_, out_, inputs_, overs_, outputs_, close_prices,
                                                 position_type_, analysis_context_, prices_df)

                        if not ratios_:
                            continue

                        # Check if RSI RC just analyzed for this input level, is a better indicator for positionings
                        #  than the previous calculated ones.
                        best_ratios_for_in_ = self.track_best_strategy(ratios_, best_ratios_for_in_)

                if best_ratios_for_in_.net_profit > 0 and best_ratios_for_in_.expected_value > 0:
                    # Save only positive ratios
                    self.ratio_crud.upsert(best_ratios_for_in_)

                # Check if the best RSI RC for this input level is a better indicator for positions
                # than the previously calculated input levels.
                best_ratios_ = self.track_best_strategy(best_ratios_for_in_, best_ratios_)

            # Release memory
            del inputs_, overs_
            if 'outputs_' in locals():
                del outputs_

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

        # Reset to the original columns
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization and return results.
        return self.finalize_identification(init_dt_, analysis_context_, verbosity_level)

    def __analyze(self,
                  in_: int,
                  over_: int,
                  out_: int,
                  inputs: np.ndarray,
                  overs: np.ndarray,
                  outputs: np.ndarray,
                  close_prices: np.ndarray,
                  position_type: int,
                  analysis_context: AnalysisContext,
                  prices_df: pl.DataFrame) -> Ratios | None:
        """
        Calculates the results and ratios of applying the RSI Rollercoaster system, with the combination of RSI levels
         and flag of the position type (Long or Short) received as parameters.

        :param in_: Input level for the strategy.
        :param over_: Level of overbought/oversold for the strategy.
        :param out_: Output level for the strategy.
        :param inputs: Array of input signals extracted from prices_df.
        :param overs: Array of overbought/oversold signals extracted from prices_df.
        :param outputs: Array of output signals extracted from prices_df.
        :param close_prices: Array of 'Close' prices extracted from prices_df.
        :param position_type: Position type under analysis: long (1) or short (-1).
        :param analysis_context: Analysis context for the strategy.
        :param prices_df: The dataFrame with prices, indexed by bar numbers and containing the required column Date.

        :return: If the winnings exceed the losses return a Ratios object with the ratios and aggregates calculated for
          trade performance, otherwise returns None.
        """
        overs_length_ = len(overs)
        outputs_length_ = len(outputs)

        # Define a list to collect trades during the iteration
        trades_ = []
        last_bar_number_processed_ = 0
        for input_bar_number_, input_percentage_change_, stop_loss_bar_number_ in inputs:
            if input_bar_number_ <= last_bar_number_processed_:
                # Input prior to the closing of the position analyzed previously
                continue
            input_bar_number_ = int(input_bar_number_)
            stop_loss_bar_number_ = int(stop_loss_bar_number_)

            # Update the date of the last input signal.
            analysis_context.last_input_date = prices_df[input_bar_number_, 'Date']

            # Find first over.BarNumber >= input.BarNumber
            idx_ = np.searchsorted(overs, input_bar_number_, side="right")
            over_bar_number_ = overs[idx_] if idx_ < overs_length_ else analysis_context.future_bar_number

            if 0 < stop_loss_bar_number_ < over_bar_number_:
                # Close losing position due to stop-loss break
                trades_.append((
                    input_bar_number_,                  # InputBarNumber
                    close_prices[input_bar_number_],    # InputPrice
                    input_percentage_change_,           # InputPercentChange
                    stop_loss_bar_number_,              # OutputBarNumber
                    close_prices[stop_loss_bar_number_]  # OutputPrice
                ))
                # Refresh
                last_bar_number_processed_ = stop_loss_bar_number_
                continue

            # Find the first output.BarNumber >= over_bar_number_
            idx_ = np.searchsorted(outputs, over_bar_number_, side="right")
            output_bar_number_ = outputs[idx_] if idx_ < outputs_length_ else analysis_context.future_bar_number

            # Close position due to completed life cycle for the strategy input, overbought/oversold and output
            trades_.append((
                input_bar_number_,                  # InputBarNumber
                close_prices[input_bar_number_],    # InputPrice
                input_percentage_change_,           # InputPercentChange
                output_bar_number_,                 # OutputBarNumber
                close_prices[                       # OutputPrice
                    output_bar_number_ if output_bar_number_ < analysis_context.future_bar_number
                    else analysis_context.last_bar_number]
            ))
            # Refresh
            last_bar_number_processed_ = output_bar_number_

        # If there are no valid signals, skip further processing
        signals_ = len(trades_)
        if signals_ == 0:
            return None

        trades_df_ = pl.DataFrame(
            # Step 1: Create trades_df_ after the loop, explicitly defining the column names
            trades_,
            schema=["InputBarNumber", "InputPrice", "InputPercentChange", "OutputBarNumber", "OutputPrice"],
            orient="row"
        ).with_columns([
            # Step 2: Calculate the final Result subtracting prices and commissions, ...
            ((pl.col('OutputPrice') - pl.col('InputPrice')) * position_type
              - COMMISSION_PERCENT * (pl.col('InputPrice') + pl.col('OutputPrice')))
            .alias('Result').cast(pl.Float64),
            # ... and the number of Sessions positioned
            (pl.col('OutputBarNumber') - pl.col('InputBarNumber')).alias('Sessions').cast(pl.Int32),
            # Explicit cast
            pl.col("InputBarNumber").cast(pl.Int32),
            pl.col("InputPrice").cast(pl.Float64),
            pl.col("InputPercentChange").cast(pl.Float64),
            pl.col("OutputBarNumber").cast(pl.Int32),
            pl.col("OutputPrice").cast(pl.Float64),
        ])

        # Release memory
        del trades_

        # Input prices that parameterize the analyzed strategy
        inputs_ = {'period': self.period, 'in': in_, 'over': over_, 'out': out_}

        # Calculate trading performance ratios and aggregates
        ratios_ = self.perfile_performance(analysis_context, inputs_, signals_, trades_df_, prices_df)

        # Release memory
        del trades_df_

        return ratios_

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
