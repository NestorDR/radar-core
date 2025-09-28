# -*- coding: utf-8 -*-

# --- Python modules ---
# json: library for encoding and decoding prices in JSON format.
import json
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, ERROR, getLogger
# typing: provides runtime support for type hints.
from typing import Any

# --- Third Party Libraries ---
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.constants import COMMISSION_PERCENT, NO_POSITION, LONG, SHORT
from radar_core.domain.strategies.base_strategy import StrategyABC
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import TIMEFRAMES
from radar_core.helpers.log_helper import verbose

logger_ = getLogger(__name__)


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
         Message levels to be reported: 0-discard messages, 1-report important messages, 2-report details.
        """
        super().__init__(strategy_acronym, verbosity_level)
        self.value_column_name = value_column_name
        self.ma_column_name = ma_column_name
        self.min_period = min_period
        self.max_period = max_period

    def evaluate(self,
                 symbol: str,
                 timeframe: int,
                 is_long_position: bool,
                 profitable_setting: dict[str, Any],
                 prices_df: pl.DataFrame,
                 verbosity_level: int = DEBUG) -> tuple[int, str, str, object]:
        """
        Evaluates if the value under analysis crosses the MA, and if it crosses, adds notice to the list received.

        :param symbol: Security symbol to evaluate.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param is_long_position: True if long trading positions, otherwise False.
        :param profitable_setting: Input setting for a profitable trading strategy.
        :param prices_df: Historical prices.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: A tuple containing:
         - the position to take as zero (long), 1 (short), or -1 (no position),
         - parameters of the setup,
         - and a string comment explaining the suggested position to take.
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # if '"period": 85' in str(profitable_setting['inputs']):
        #     print('85')
        inputs_, net_profit_, win_probability_, min_percentage_change_to_win_, max_percentage_change_to_win, \
            last_output_date_ = self.scatter_ratios(profitable_setting)
        # Parse the inputs_ JSON and extract the relevant parameters
        period_ = json.loads(inputs_)['period']
        parameters_ = f'{period_}'

        if verbosity_level == DEBUG:
            print(f"{symbol}: evaluating {self.strategy_acronym}({parameters_})...")

        # Calculate moving average to evaluate
        prices_df = prices_df.with_columns(
            pl.col(self.value_column_name).rolling_mean(window_size=period_).alias(self.ma_column_name))

        # Extract the value (e.g. 'Close') and its moving average (e.g. 'Sma') for the last and second-last rows
        # Get the last row
        row_index_ = prices_df.height - 1
        value_ = prices_df[self.value_column_name][row_index_]
        ma_ = prices_df[self.ma_column_name][row_index_]
        percent_change_ = prices_df['PercentChange'][row_index_]
        date = prices_df['Date'][row_index_]
        # Get the second-last row
        row_index_ -= 1
        previous_value_ = prices_df[self.value_column_name][row_index_]
        previous_ma_ = prices_df[self.ma_column_name][row_index_]

        # Drop columns that are no longer needed
        prices_df.drop([self.ma_column_name])

        position_ = NO_POSITION
        comment_ = ''
        if is_long_position:
            # Check if the evaluated value rises above long MA
            if value_ > ma_ and previous_value_ <= previous_ma_:
                if min_percentage_change_to_win_ <= percent_change_ <= max_percentage_change_to_win:
                    position_ = LONG
                    comment_ = f'{self.unit_label:1}{value_:.2f} > {self.unit_label}{ma_:.2f} ({percent_change_:.2f}%)'

            # Check if the evaluated value falls below long MA
            elif value_ < ma_ and previous_value_ >= previous_ma_:
                position_ = SHORT
                comment_ = f'{self.unit_label:1}{value_:.2f} < {self.unit_label}{ma_:.2f} ({percent_change_:.2f}%)'

        # Check if the evaluated value falls below short MA
        elif value_ < ma_ and previous_value_ >= previous_ma_:
            if min_percentage_change_to_win_ <= percent_change_ <= max_percentage_change_to_win:
                position_ = SHORT
                comment_ = f'{self.unit_label:1}{value_:.2f} < {self.unit_label}{ma_:.2f} ({percent_change_:.2f}%)'

        # Check if the evaluated value rises above short MA
        elif value_ > ma_ and previous_value_ <= previous_ma_:
            position_ = LONG
            comment_ = f'{self.unit_label:1}{value_:.2f} > {self.unit_label}{ma_:.2f} ({percent_change_:.2f}%)'

        if position_ != NO_POSITION:
            super().save_notice(symbol, self.strategy_id, inputs_, timeframe, is_long_position, date, position_ == LONG,
                                ma_, value_)

        return position_, parameters_, comment_, last_output_date_

    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions,
                 prices_df: pl.DataFrame,
                 close_prices: None = None,
                 verbosity_level: int = DEBUG) -> dict:
        """
        Iterate from the minimum to the maximum number of periods to calculate the MA and evaluate its profitability
         on positions:
         - long: open when the value rises above MA and closed when the value falls below MA
         - short: open when the value falls below MA and closed when the value rises above MA.
        Save the profitable setups (identified Moving Average) in the Database.

        :param symbol: Security symbol to analyze the PMA.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe at least with required columns
         [DateTime, {self.value_column_name}, PercentChange, BarNumber].
        :param close_prices: Only for signature compatibility, not used in this method.
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

        # Position types to iterate
        position_types_ = [LONG] + ([] if only_long_positions else [SHORT])

        for position_type_ in position_types_:
            # Initialize bad strategy to be evaluated and to get better MAs
            best_ratios_ = self.initialize_bad_strategy()
            analysis_context_.is_long_position = position_type_ == LONG

            # Iterate from the min to the max number of periods to calculate the MA and evaluate its profitability
            for period_ in range(self.min_period, self.max_period + 1):
                if verbosity_level == DEBUG:
                    print('', end='\r')
                    print(
                        f'Evaluating profitability {TIMEFRAMES[timeframe]} of {self.strategy_acronym}({period_}) for {symbol}...',
                        end='')

                if prices_df.height - self.min_period < period_:
                    # The minimum number of periods to calculate the average is not reached
                    continue

                try:
                    # Calculate MA for the number of periods in process for the loop, to analyze
                    prices_df = prices_df.with_columns(
                        pl.col(self.value_column_name).rolling_mean(window_size=period_).alias(self.ma_column_name))

                except Exception as e:
                    # Log error
                    message_ = (f'Error calculating the {self.strategy_acronym}({period_})'
                                f' over {self.value_column_name} for {symbol}.')
                    verbose(message_, ERROR, verbosity_level)
                    logger_.exception(e, exc_info=e)
                    continue

                # Calculate results and ratios of applying the price crossover system on the MA of "period_" periods
                # Identify signals or position changes when the value crosses the MA of "period_" periods
                prices_df = prices_df.with_columns(
                    ((pl.col(self.value_column_name) > pl.col(self.ma_column_name))
                     if analysis_context_.is_long_position
                     else (pl.col(self.value_column_name) < pl.col(self.ma_column_name))
                     ).cast(pl.Int8).diff().alias('Position')
                )

                # Generate a dataframe of position starts
                inputs_df_ = prices_df.filter(pl.col('Position') == 1).select(['BarNumber', 'Close', 'PercentChange'])
                # Generate a dataframe of position outputs
                outputs_df_ = prices_df.filter(pl.col('Position') == -1).select(['BarNumber', 'Close'])

                # Perform early skip checks
                if inputs_df_.height < 2 or outputs_df_.height < 2:
                    # Skipping the MA, not enough signals
                    continue

                # If there is an output prior to an input: remove the first output row
                if inputs_df_[0, 'BarNumber'] > outputs_df_[0, 'BarNumber']:
                    outputs_df_ = outputs_df_.slice(1)  # Remove with slicing

                # If there are more inputs than outputs...
                if inputs_df_.height > outputs_df_.height:
                    # ...append an output row, fake because the position is open, with the last [Close] price
                    outputs_df_ = pl.concat([outputs_df_,
                                             pl.DataFrame({'BarNumber': [analysis_context_.future_bar_number],
                                                           'Close': [float(analysis_context_.final_price)]})
                                            # Match the type explicitly
                                            .with_columns(pl.col('BarNumber').cast(pl.Int32))])

                # At this point, the lengths of inputs_df and outputs_df should match
                if inputs_df_.height != outputs_df_.height:
                    # Log error
                    message_ = f'Error due to differences between number of inputs and outputs {symbol}.'
                    verbose(message_, ERROR, verbosity_level)
                    logger_.error(message_)
                    continue

                # If there are no valid signals, skip further processing
                signals_ = inputs_df_.height
                if signals_ == 0:
                    continue

                position_factor_ = 1 if analysis_context_.is_long_position else -1

                # Method lazy() starts a lazy query from this point, returns a LazyFrame object in which operations
                # are not executed until they are triggered by the collect() call
                trades_df_ = (pl.concat(
                    # Step 1: Create trades_df_ by combining aligned input/output rows into a single DataFrame
                    [
                        inputs_df_.lazy(),
                        outputs_df_.lazy()
                        .rename({'BarNumber': 'OutputBarNumber', 'Close': 'OutputPrice'})
                        .select(['OutputBarNumber', 'OutputPrice'])  # Select the necessary columns
                    ],
                    how="horizontal"  # Horizontally stack the rows, keeping row alignment
                ).rename({
                    'BarNumber': 'InputBarNumber',
                    'Close': 'InputPrice',
                    'PercentChange': 'InputPercentChange'
                }).with_columns([
                    # Step 2: Calculate the final Result subtracting prices and commissions, ...
                    ((pl.col('OutputPrice') - pl.col('InputPrice')) * position_factor_
                     - COMMISSION_PERCENT * (pl.col('InputPrice') + pl.col('OutputPrice')))
                    .alias('Result').cast(pl.Float64),
                    # ... and the number of Sessions positioned
                    (pl.col('OutputBarNumber') - pl.col('InputBarNumber')).alias('Sessions').cast(pl.Int32)
                ])).collect()

                # Input prices that parameterize the analyzed strategy
                inputs_ = {'period': period_}
                # Calculate trading performance ratios and aggregates
                ratios_ = self.perfile_performance(analysis_context_, inputs_, signals_, trades_df_, prices_df)

                if not ratios_:
                    continue

                # Save only to positive ratios
                if ratios_.net_profit > 0 and ratios_.expected_value > 0:
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

        # Release memory
        if 'inputs_df_' in locals():
            del inputs_df_, outputs_df_
            if 'trades_df_' in locals():
                del trades_df_

        # Reset to the original columns
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization and return results
        return self.finalize_identification(init_dt_, analysis_context_, verbosity_level)
