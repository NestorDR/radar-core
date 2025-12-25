# src/radar_core/domain/strategies/base_strategy.py

# --- Python modules ---
# abc: abstract base classes (abc) allow implementing interfaces effectively
from abc import ABC, abstractmethod
# decimal: provides support for fast correctly rounded decimal floating point arithmetic
#  it offers several advantages over the float datatype.
from decimal import Decimal
# datetime: provides classes for manipulating dates and times.
from datetime import date, datetime
# json: library for encoding and decoding prices in JSON format.
import json
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import CRITICAL, DEBUG, INFO, getLogger

# --- Third Party Libraries ---
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
# technical: provides calculations of TA indicators
from radar_core.domain.technical import ATR
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import COMMISSION_PERCENT, DAILY, TIMEFRAMES
from radar_core.helpers.log_helper import verbose
# infrastructure: allows access to the own DB and/or integration with external prices providers
from radar_core.infrastructure.crud import RatioCrud, StrategyCrud
# models: result of Object-Relational Mapping
from radar_core.models import Ratios

# Constant for price format
PRICE_PRECISION = Decimal('1.00')

logger_ = getLogger(__name__)


class AnalysisContext:
    """
    This class encapsulates the prices and parameters necessary for performing financial analysis on specified security
     over a given timeframe.
     It includes details as the security symbol being analyzed, the timeframe, date range, initial and final prices,
     percentage change, and the number of the last price bar under analysis.
     Also, it provides placeholders for the best long and short strategies using the `Ratios` class.
    """

    def __init__(self,
                 symbol: str,
                 timeframe: int,
                 from_date: date,
                 to_date: date,
                 initial_price: float,
                 final_price: float,
                 last_bar_number: int,
                 future_bar_number: int):
        """
        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param from_date: The start date of the analysis period.
        :param to_date: The end date of the analysis period.
        :param initial_price: The price of the security at the start of the analysis period.
        :param final_price: The price of the security at the end of the analysis period.
        :param last_bar_number: The number of the last prices bar.
        :param future_bar_number: The number of a price bar that will be available in the future.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.is_long_position = True
        self.from_date = from_date
        self.to_date = to_date
        self.initial_price = Decimal(initial_price).quantize(PRICE_PRECISION)
        self.final_price = Decimal(final_price).quantize(PRICE_PRECISION)
        self.percent_change = (final_price - initial_price) / initial_price
        self.last_bar_number = last_bar_number
        self.future_bar_number = future_bar_number
        # Initialize empty objects Ratios for the best long and short strategies
        self.best_long = Ratios()
        self.best_short = Ratios()


class StrategyABC(ABC):
    """
    Provides a base class to identify profitable trading strategies.
    This class is an abstract base class (ABC) meant to be implemented for specific strategies.
    """

    def __init__(self,
                 strategy_acronym: str,
                 verbosity_level: int):
        """
        :param strategy_acronym: Strategy acronym to be analyzed.
        :param verbosity_level: Minimum importance level of messages reporting the progress of the process for all
        """
        self.strategy_acronym = strategy_acronym
        strategy = StrategyCrud().get_by_acronym(self.strategy_acronym)
        if strategy is None:
            # Report error
            message_ = f'Strategy {self.strategy_acronym} not found in database.'
            verbose(message_, CRITICAL, verbosity_level)
            logger_.critical(message_)
            raise LookupError(message_)

        self.strategy_id = strategy.id
        self.unit_label = strategy.unit_label
        self.pool = strategy.pool
        self.verbosity_level = verbosity_level
        self.ratio_crud: RatioCrud | None = None

    # region Support to identification

    @abstractmethod
    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions: bool,
                 prices_df: pl.DataFrame,
                 close_prices: np.ndarray,
                 percent_changes: np.ndarray,
                 verbosity_level: int = DEBUG) -> dict:
        """
        Iterates a number of periods or levels to calculate a tech indicator and evaluate its profitability.
        Returns a dictionary with the strategies with the best ratios.

        :param only_long_positions:
        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe with required columns.
        :param close_prices: Close prices for the given symbol and timeframe.
        :param percent_changes: Percent change of the close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages.

        :return: Dictionary of strategies with the best ratios based on its profitability.
        """
        pass

    def initialize_identification(self,
                                  symbol: str,
                                  timeframe: int,
                                  prices_df: pl.DataFrame,
                                  verbosity_level: int) -> tuple[datetime, AnalysisContext, list, int]:
        """
        Logs initialization and prepares the necessary variables for the process that will identify profitable strategies.

        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param prices_df: Dataframe with the required column [DateTime, ...]
        :param verbosity_level: An integer specifying the level of verbosity for logging.

        :return: A tuple with (init_dt_, analysis_context_, original_column_names_, verbosity_level)
        """
        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Set the date and time when the process starts
        init_dt_ = datetime.now()

        # Logging initial message
        message_ = init_dt_.strftime(f'{self.strategy_acronym:11} on {symbol}: start %Y-%m-%d %H:%M:%S')
        verbose(message_, INFO, verbosity_level, '' if verbosity_level >= INFO else '\n')
        logger_.info(message_)

        # Initialize the analysis context
        analysis_context_ = AnalysisContext(
            symbol,
            timeframe,
            # Identify the initial and final dates of the period to analyze.
            prices_df.select(pl.col('Date').first()).to_series().item(),
            prices_df.select(pl.col('Date').last()).to_series().item(),
            # Identify the initial and final prices of the period to analyze
            prices_df.select(pl.col('Close').first()).to_series().item(),
            prices_df.select(pl.col('Close').last()).to_series().item(),
            # Identify the number of sessions
            prices_df.select(pl.col('BarNumber').last()).to_series().item(),
            self.future_bar_number(prices_df))

        # Instantiate prices access
        self.ratio_crud = RatioCrud()
        # Flag as `is_in_process` the ratios for a specific symbol, strategy_id, and timeframe
        self.ratio_crud.flag_in_process(symbol, self.strategy_id, timeframe)

        # Capture the original column names to ensure only these are returned
        original_column_names_ = prices_df.columns

        return init_dt_, analysis_context_, original_column_names_, verbosity_level

    @staticmethod
    def future_bar_number(prices_df: pl.DataFrame) -> int:
        """
        Generates a future bar number based on the height of the DataFrame.

        :param prices_df: A DataFrame with price prices for a financial instrument.

        :return: A bar number larger than those contained in the dataframe.
        """
        return prices_df.height

    def finalize_identification(self,
                                init_dt: datetime,
                                analysis_context: AnalysisContext,
                                verbosity_level: int = DEBUG) -> dict:
        """
        Finalize the process that identified profitable strategies and logs finalization.

        :param analysis_context: Analysis context with the identified profitable strategies.
        :param init_dt: The date and time when the process started.
        :param verbosity_level: An integer specifying the level of verbosity for logging.

        :return: Dictionary with a summary of the best long and short strategies.
        """

        # Delete Ratios where is_in_process is True.
        self.ratio_crud.delete_flagged_in_process(analysis_context.symbol, self.strategy_id, analysis_context.timeframe)

        # Close SQL session and release memory
        self.ratio_crud.session.close()
        self.ratio_crud = None

        # Validate on best strategies Long and Short if they remain as initial "bad seeds"
        # Check for -infinite in net_profit to determine if a valid strategy was ever found.
        if analysis_context.best_long.net_profit == -float('inf'):
            logger_.debug(f"[{analysis_context.symbol}]: No profitable Long {self.strategy_acronym} "
                          f"found in {TIMEFRAMES[analysis_context.timeframe]} timeframe.")

        if analysis_context.best_short.net_profit == -float('inf'):
            logger_.debug(f"[{analysis_context.symbol}]: No profitable Short {self.strategy_acronym} "
                          f"found in {TIMEFRAMES[analysis_context.timeframe]} timeframe.")

        # Set dictionary for better indicator strategies (Long and Short)
        result_ratios_ = dict(symbol=analysis_context.symbol,
                              updatedAt=datetime.now().strftime('%Y-%m-%d %H:%M'),
                              # Convert Ratios objects to dictionaries
                              bestLong=self.serialize_ratios(analysis_context.best_long),
                              bestShort=self.serialize_ratios(analysis_context.best_short))

        # Final logging and output
        if verbosity_level == DEBUG:
            print(json.dumps(result_ratios_, indent=4))

        message_ = (init_dt.strftime(f'{self.strategy_acronym:11} on {analysis_context.symbol}:'
                                     f' start %Y-%m-%d %H:%M:%S ...')
                    + datetime.now().strftime(' end %Y-%m-%d %H:%M:%S')
                    + f'  {(datetime.now() - init_dt).total_seconds() / 60:6.1f} min')
        if verbosity_level == INFO:
            print('', end='\r')
        verbose(message_, INFO, verbosity_level)
        logger_.info(message_)

        return result_ratios_

    # endregion Support to identification

    # region Manipulation of strategies

    @staticmethod
    def initialize_bad_strategy() -> Ratios:
        """
        Set negative infinities as (bad) reference seed values, to be evaluated and to get better strategies setups.
        :return: An object Ratios to support the best strategy.
        """
        return Ratios(inputs='',
                      net_profit=-float('inf'),
                      expected_value=-float('inf'),
                      winnings=-float('inf'),
                      losses=0)

    @staticmethod
    def track_best_strategy(strategy_to_compare: Ratios,
                            best_ratios: Ratios) -> Ratios:
        """
        Check if the ratios of the strategy to compare describe a better indicator for positioning than those
         calculated previously, and thus keeps track of the best strategy based on net profit and expected value.

        :param strategy_to_compare: Strategy to compare with the best strategy.
        :param best_ratios: Best strategy based on net profit and expected value.

        :return: A tuple with the best strategies after comparison.
        """

        new_is_better_ = (strategy_to_compare.net_profit > best_ratios.net_profit
                          or (strategy_to_compare.net_profit == best_ratios.net_profit
                              and strategy_to_compare.expected_value > best_ratios.expected_value))
        return strategy_to_compare if new_is_better_ else best_ratios

    # endregion Manipulation of strategies

    # region Ratios

    def perfile_performance_fast(self,
                                 analysis_context: AnalysisContext,
                                 inputs: dict,
                                 input_bar_numbers: np.ndarray,
                                 output_bar_numbers: np.ndarray,
                                 close_prices: np.ndarray,
                                 percent_changes: np.ndarray,
                                 prices_df: pl.DataFrame) -> Ratios | None:
        """
        Calculates and organizes aggregates and ratios to profile the strategy’s trade performance,
         using vectorized NumPy operations.

        Visit
            - https://estrategiastrading.com/ratios-para-evaluar-sistemas-de-trading/
            - https://estrategiastrading.com/calcular-la-esperanza-matematica-del-sistema-de-trading/
            - https://estrategiastrading.com/profit-factor/

        :param analysis_context: Analysis context for the strategy.
        :param inputs: Input prices that parameterize a strategy.
        :param input_bar_numbers: Array of bar numbers where trades were opened.
        :param output_bar_numbers: Array of bar numbers where trades were closed.
        :param close_prices: Array of Close prices extracted from the prices dataframe.
        :param percent_changes: Array of PercentChange values extracted from the prices dataframe.
        :param prices_df: The dataframe with prices, indexed by bar numbers and containing the required column Date.

        :return: A Ratios object (with ratios and aggregates) if winnings exceed losses, otherwise None.
          The Ratios object contains the following attributes:
            - symbol: Security symbol to update or insert the ratios for.
            - strategy_id: Identifier of the trading strategy.
            - inputs: Input prices that parameterizes a strategy.
            - timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
            - is_long_position: True if the ratios are for long trading positions,
            - from_date: The initial date from which the strategy was evaluated.
            - to_date: The final date to which the strategy was evaluated.
            - initial_price: Price of the security when the 1º input signal was identified.
            - final_price: Price of the security when the last output signal was identified.
            - net_change: Percentage change of the final price over the initial price
            - signals: Number of trade signals identified by the strategy.
            - winnings: Sum of profits (money) generated by the strategy.
            - losses: Sum of losses (money) supported through the strategy.
            - net_profit: Percentage of net profit got following the input and output signals
                          Formula: (winnings_ - losses_) / initial_price_.
            - expected_value: Mathematical expectation of the strategy
                              Formula: (win_probability_ * average_win_) + (loss_probability_ * average_loss_).
            - win_probability: Percentage of positive/winning operations.
            - loss_probability: Percentage of negative/losing operations.
            - average_win: Average profit of the positive/winning operations.
            - average_loss: Average loss of the negative/losing operations.
            - min_percentage_change_to_win: Minimum % change of input sessions for the positive/winning operations.
            - max_percentage_change_to_win: Maximum % change of input sessions for the positive/winning operations.
            - total_sessions: Total number of sessions to which the strategy has been evaluated
            - winn_sessions: Number of sessions elapsed during positive/winning trades.
            - loss_sessions: Number of sessions elapsed during negative/losing trades.
            - percentage_exposure: Percentage time during which the strategy was active.
                                   Formula: (winning_sessions + losing_sessions) / total_sessions
            - first_input_date: Date of the first input signal.
            - last_input_date: Date of the last input signal.
            - last_output_date: Date of the last output signal.
        """
        signals_ = len(input_bar_numbers)
        if signals_ == 0:
            return None

        # Prepare price arrays (vectorized slicing / fancy indexing)
        # Extract input prices and percentages directly using the indices
        input_prices_ = close_prices[input_bar_numbers]
        input_pct_changes_ = percent_changes[input_bar_numbers]

        # Handle output prices for open positions (Mark-to-Market - valuation of assets at current market prices)
        # If OutputBarNumber is future_bar_number (trade open), must be used the last available price.
        last_bar_number_ = len(close_prices) - 1

        # Create a safe index array subject to the last valid index of close_prices.
        # np.minimum ensures that any index >= len(close_prices) (like future_bar_number) becomes last_bar_number_
        output_prices_ = close_prices[np.minimum(output_bar_numbers, last_bar_number_)]

        # Identify the position type under analysis: long (1) or short (-1).
        position_type_ = 1 if analysis_context.is_long_position else -1

        # Vectorized Calculation of Results
        # Formula: (Output - Input) * Direction - Commission * (Input + Output)
        results_ = ((output_prices_ - input_prices_) * position_type_
                    - COMMISSION_PERCENT * (input_prices_ + output_prices_))

        # Calculate session durations using vectorized element-wise subtraction,
        #  leveraging SIMD (Single Instruction, Multiple Data)
        sessions_ = output_bar_numbers - input_bar_numbers

        # Boolean masking for grouping (Winnings versus Losses)
        # Identify winning trades (Net Result > 0)
        winn_mask_ = results_ > 0.0
        # Identify losing trades (Net Result <= 0)
        loss_mask_ = ~winn_mask_

        # Winnings aggregations using NumPy
        winnings_ = np.sum(results_[winn_mask_]) if np.any(winn_mask_) else 0.0
        winn_trades_ = int(np.count_nonzero(winn_mask_))
        winning_sessions_ = np.sum(sessions_[winn_mask_]) if np.any(winn_mask_) else 0

        # Min/Max Percentage Change (only on winning trades)
        if winn_trades_ > 0:
            winn_pcts_ = input_pct_changes_[winn_mask_]
            min_percentage_change_to_win_ = np.min(winn_pcts_)
            max_percentage_change_to_win_ = np.max(winn_pcts_)
        else:
            min_percentage_change_to_win_ = 0.0
            max_percentage_change_to_win_ = 0.0

        # Losses aggregations using NumPy (sum of negative results)
        losses_ = np.sum(results_[loss_mask_]) if np.any(loss_mask_) else 0.0
        loss_trades_ = int(np.count_nonzero(loss_mask_))
        losing_sessions_ = np.sum(sessions_[loss_mask_]) if np.any(loss_mask_) else 0

        # Calculate ratios for the strategy
        first_input_price_ = max(float(input_prices_[0]), 0.00001)

        net_profit_, expected_value_, win_probability_, loss_probability_, average_win_, average_loss_ = \
            self.compute_key_ratios(signals_, first_input_price_, winnings_, winn_trades_, losses_, loss_trades_)

        total_sessions_ = analysis_context.last_bar_number + 1

        # Extracts the relevant dates
        # Accessing Polars by index (.item()) is efficient for single scalars
        first_input_date_ = prices_df[int(input_bar_numbers[0]), 'Date']

        # Extract last trade info
        # Identify the input price and date for the last trade/position
        last_input_price_ = Decimal(input_prices_[-1]).quantize(PRICE_PRECISION)
        last_input_date_ = prices_df[int(input_bar_numbers[-1]), 'Date']

        # Identify the output price and date for the last trade/position
        last_output_bar_number_ = int(output_bar_numbers[-1])
        if last_output_bar_number_ >= analysis_context.future_bar_number:
            # The position is still opened
            last_output_price_ = None
            last_output_date_ = None
        else:
            # The position is already closed
            last_output_price_ = Decimal(output_prices_[-1]).quantize(PRICE_PRECISION)
            last_output_date_ = prices_df[last_output_bar_number_, 'Date']

        # Identify stop loss for the last trade/position
        if {'LongStopLoss', 'ShortStopLoss'}.issubset(prices_df.columns):
            stop_loss_column_ = 'LongStopLoss' if analysis_context.is_long_position else 'ShortStopLoss'
            # Accessing scalar value
            last_stop_loss_val_ = prices_df[int(input_bar_numbers[-1]), stop_loss_column_]
            last_stop_loss_ = Decimal(last_stop_loss_val_).quantize(PRICE_PRECISION)
        else:
            last_stop_loss_ = None

        # Return initialized support to ratios with available prices at this point
        return Ratios(
            # Set the unique restriction key
            symbol=analysis_context.symbol,
            strategy_id=self.strategy_id,
            timeframe=analysis_context.timeframe,
            inputs=RatioCrud.serialize_inputs(inputs),
            is_long_position=analysis_context.is_long_position,
            is_in_process=False,

            # Set prices
            from_date=analysis_context.from_date,
            to_date=analysis_context.to_date,
            initial_price=analysis_context.initial_price.quantize(PRICE_PRECISION),
            final_price=analysis_context.final_price.quantize(PRICE_PRECISION),
            net_change=analysis_context.percent_change if analysis_context.is_long_position else -analysis_context.percent_change,

            # Results
            signals=signals_,
            winnings=winnings_,
            losses=losses_,
            # saved_record_.profit_factor = winnings_ / losses_

            # Ratios
            net_profit=net_profit_,
            expected_value=expected_value_,
            win_probability=win_probability_,
            loss_probability=loss_probability_,
            average_win=average_win_,
            average_loss=average_loss_,

            # Filters
            min_percentage_change_to_win=Decimal(min_percentage_change_to_win_).quantize(PRICE_PRECISION),
            max_percentage_change_to_win=Decimal(max_percentage_change_to_win_).quantize(PRICE_PRECISION),

            # Sessions
            total_sessions=total_sessions_,
            winning_sessions=winning_sessions_,
            losing_sessions=losing_sessions_,
            percentage_exposure=(winning_sessions_ + losing_sessions_) / total_sessions_,

            # Relevant dates
            first_input_date=first_input_date_,
            last_input_date=last_input_date_,
            last_output_date=last_output_date_,

            # Last trade info
            last_input_price=last_input_price_,
            last_output_price=last_output_price_,
            last_stop_loss=last_stop_loss_,
        )

    @staticmethod
    def compute_key_ratios(signals_: int,
                           first_input_price_: float,
                           winnings_: float,
                           winn_trades_: int,
                           losses_: float,
                           loss_trades_: int) -> tuple[float, float, float, float, float, float]:
        """
        Calculates key ratios of a strategy to evaluate its trade performance.

        :param signals_: Number of trade signals identified by the strategy.
        :param first_input_price_: Price of the security when the 1º input signal was identified.
        :param winnings_: Sum of profits (money) generated by the strategy.
        :param winn_trades_: Number of positive/winning trades.
        :param losses_: Sum of losses (money) supported through the strategy.
        :param loss_trades_: Number of negative/losing trades.

        :return: A tuple containing the following calculated metrics:
                 - net_profit_: Percentage of net profit got following the input and output signals
                                Formula: (winnings_ - losses_) / initial_price_.
                 - expected_value_: Mathematical expectation of the strategy
                                    Formula: (win_probability_ * average_win_) + (loss_probability_ * average_loss_).
                 - win_probability_: Percentage of positive/winning trades.
                 - loss_probability_: Percentage of negative/losing trades.
                 - average_win_: Average profit of the positive/winning trades.
                 - average_loss_: Average loss of the negative/losing trades.
        """
        # Calculate the net profit; it is the profitability of the strategy, which can also be expressed as a percentage
        #  of initial capital.
        net_profit_ = (winnings_ + losses_) / first_input_price_

        # Calculate mathematical expectation of the strategy
        win_probability_ = 0.0 if signals_ <= 0 else winn_trades_ / signals_
        average_win_ = 0.0 if winn_trades_ <= 0 else winnings_ / winn_trades_
        loss_probability_ = 0.0 if signals_ <= 0 else loss_trades_ / signals_
        average_loss_ = 0.0 if loss_trades_ <= 0 else losses_ / loss_trades_
        expected_value_ = win_probability_ * average_win_ + loss_probability_ * average_loss_
        return net_profit_, expected_value_, win_probability_, loss_probability_, average_win_, average_loss_

    def serialize_ratios(self,
                         ratios: Ratios) -> dict:
        """
        Convert a Ratios object into a JSON-serializable dictionary while handling non-serializable fields and removing
         unnecessary properties.
        
        :param ratios: The Ratios object to serialize.
        
        :return: A serialized dictionary with all attributes JSON-ready.
        """

        serializable_dict_ = {}

        # Use vars() to get the object's attributes as a dictionary and then convert it to a JSON-serializable format
        for key, value in vars(ratios).items():
            # Ignore some properties
            if key in ('_sa_instance_state', 'id', 'symbol', 'is_in_process'):
                continue
            # Handle datetime objects, convert to string
            if isinstance(value, (date, datetime)):
                serializable_dict_[key] = value.isoformat()
            # Handle Decimal, float and NumPy floating types, convert to float and round to 2 decimals
            elif isinstance(value, (Decimal, float, np.floating)):
                serializable_dict_[key] = round(float(value), 2)
            # Handle Python int and NumPy integer types
            elif isinstance(value, (int, np.integer)):
                # Handle the 'strategy_id' field replace it with its acronym
                if key == 'strategy_id' and isinstance(value, int):
                    serializable_dict_['strategy'] = self.strategy_acronym
                # Handle the 'timeframe' field: replace it with its name
                elif key == 'timeframe':
                    serializable_dict_['timeframe'] = TIMEFRAMES[int(value)]
                # Convert NumPy integers to native Python int
                else:
                    serializable_dict_[key] = int(value)
            # Handle the 'inputs' field if it's a serialized dictionary as string
            elif key == 'inputs' and isinstance(value, str):
                try:
                    # Attempt to parse the string as JSON
                    serializable_dict_[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If parsing fails, keep the original string as-is
                    serializable_dict_[key] = value
            # Handle other serializable types (or leave unchanged)
            else:
                serializable_dict_[key] = value  # Keep as-is if already serializable

        return serializable_dict_

    # endregion Ratios


class RsiStrategyABC(StrategyABC, ABC):
    """
    Base class for strategies using the Relative Strength Index (RSI).
    It encapsulates common RSI-specific methods.
    """

    def __init__(self,
                 strategy_acronym: str,
                 verbosity_level: int = DEBUG,
                 period: int = 14):
        super().__init__(strategy_acronym, verbosity_level)
        self.period = period  # Common RSI period, used by RSI strategies

    # region Stop Loss

    @staticmethod
    def identify_where_to_stop_loss(timeframe: int,
                                    prices_df: pl.DataFrame,
                                    close_prices: np.ndarray) -> pl.DataFrame:
        """
        Identifies and calculates where to stop losses for both long and short positions.
        The method calculates the stop-loss trigger levels and associates bars where these triggers occur.

        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param prices_df: A DataFrame containing at least the price columns.
        :param close_prices: A Series containing the close prices.

        :return: An updated version of the input DataFrame with new stop-loss information:
         - LongStopLoss: Calculated stop loss levels for long positions.
         - ShortStopLoss: Calculated stop loss levels for short positions.
         - BarNumberForLongStop: Bar number where the stop loss for long positions is triggered.
         - BarNumberForShortStop: Bar number where the stop loss for short positions is triggered.
        """

        if ({'LongStopLoss', 'ShortStopLoss', 'BarNumberForLongStop', 'BarNumberForShortStop'}
                .issubset(prices_df.columns)):
            # Stop loss is already calculated
            return prices_df

        # Identify when (at which bar) loss stops are triggered
        bars_for_stop_loss_ = 10 if timeframe <= DAILY else 3

        # Set the stop loss
        prices_df = RsiStrategyABC.set_stop_loss(prices_df, bars_for_stop_loss_)

        # Extract relevant prices as NumPy arrays for efficient slicing and speeding up prices access
        bar_numbers_ = prices_df['BarNumber'].to_numpy()
        long_stop_loss_ = prices_df['LongStopLoss'].to_numpy()
        short_stop_loss_ = prices_df['ShortStopLoss'].to_numpy()

        # Generate a price bar number beyond the last session under analysis (in the future).
        future_bar_number_ = RsiStrategyABC.future_bar_number(prices_df)

        # Initialize all stop loss in the distant future
        bar_for_long_stop_ = future_bar_number_ * np.ones(prices_df.height, dtype=np.int32)
        bar_for_short_stop_ = future_bar_number_ * np.ones(prices_df.height, dtype=np.int32)

        for i in range(prices_df.height - 1):
            long_condition_ = np.asarray(close_prices[i + 1:] < long_stop_loss_[i]).nonzero()[0]
            if long_condition_.size > 0:
                bar_for_long_stop_[i] = bar_numbers_[i + 1 + long_condition_[0]]

            short_condition_ = np.asarray(close_prices[i + 1:] > short_stop_loss_[i]).nonzero()[0]
            if short_condition_.size > 0:
                bar_for_short_stop_[i] = bar_numbers_[i + 1 + short_condition_[0]]

        # Add the new columns to the Polars DataFrame
        prices_df = prices_df.with_columns([
            pl.Series('BarNumberForLongStop', bar_for_long_stop_),
            pl.Series('BarNumberForShortStop', bar_for_short_stop_)
        ])

        # Release memory
        del bar_numbers_, bar_for_long_stop_, bar_for_short_stop_

        return prices_df

    @staticmethod
    def set_stop_loss(prices_df: pl.DataFrame,
                      bars_for_stop_loss: int) -> pl.DataFrame:
        """
        Calculates and sets stop loss values for a dataframe containing price prices.
        This method uses the Average True Range (ATR) and rolling window calculations to determine stop loss levels
         for both long and short trading positions.
        Specifically, it computes the rolling minimum of the low prices and the rolling maximum of the high prices over
         a fixed window size, allowing the identification of recent significant price levels.
        The resulting stop loss values are adjusted using the ATR to provide more dynamic thresholds.

        :param prices_df: A DataFrame containing at least the price columns ['High', 'Low', 'Close'].
        :param bars_for_stop_loss: The number of bars (window size) to identify recent lowest/highest prices and
         consider it for stop loss calculation.

        :return: A modified version of the input DataFrame with two additional columns:
         - LongStopLoss: Calculated stop loss levels for long positions.
         - ShortStopLoss: Calculated stop loss levels for short positions.
         But if 'LongStopLoss' and 'ShortStopLoss' already exist, the DataFrame is returned unchanged.
        """
        if {'LongStopLoss', 'ShortStopLoss'}.issubset(prices_df.columns):
            return prices_df

        if 'Atr' not in prices_df.columns:
            prices_df = ATR(prices_df)  # Add a column with the ATR

        # Add columns of stop loss for long and short positions
        return prices_df.with_columns([
            # Stop loss for long positions: maximum of (Latest low price, Close - 2 ATR)
            pl.max_horizontal([
                # Get the recent lowest price of the last X bars using a rolling window
                (pl.col('Low').rolling_min(window_size=bars_for_stop_loss)),
                (pl.col('Close') - pl.col('Atr').clip(lower_bound=0))
            ]).alias('LongStopLoss'),
            # Stop loss for short positions: minimum of (Latest high price, Close + 2 ATR)
            pl.min_horizontal([
                # Get the recent highest price of the last X bars using a rolling window
                (pl.col('High').rolling_max(window_size=bars_for_stop_loss)),
                (pl.col('Close') + pl.col('Atr'))
            ]).alias('ShortStopLoss'),
        ])

    # endregion Stop Loss
