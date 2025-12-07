# src/radar_core/domain/strategies/ma.py

# --- Python modules ---
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import DEBUG, ERROR, getLogger

# --- Third Party Libraries ---
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.constants import COMMISSION_PERCENT, LONG, SHORT
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
                 close_prices: None = None,
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
            position_factor_ = pl.lit(1 if analysis_context_.is_long_position else -1)  # lit = literal

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
                    # Calculate MA for the number of periods in process for the loop and Position, to analyze,
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
                # IF is_long_position THEN identify 'close' > 'ma' ELSE identify 'close' < 'ma'
                prices_df = prices_df.with_columns(
                    (position_factor_ * (pl.col(self.value_column_name) - pl.col(self.ma_column_name)) > 0)
                    .cast(pl.Int8).diff().alias("Position")
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
                                             pl.DataFrame({
                                                 'BarNumber': [analysis_context_.future_bar_number],
                                                 'Close': [float(analysis_context_.final_price)],
                                             }, schema={'BarNumber': pl.Int32, 'Close': pl.Float64})
                                             ])

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

                if ratios_.net_profit > 0 and ratios_.expected_value > 0:
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

        # Release memory
        if 'inputs_df_' in locals():
            del inputs_df_, outputs_df_
            if 'trades_df_' in locals():
                del trades_df_

        # Reset to the original columns
        prices_df = prices_df.select(original_column_names_)

        # Finalize the process to identify profitable strategies and logs finalization and return results
        return self.finalize_identification(init_dt_, analysis_context_, verbosity_level)
