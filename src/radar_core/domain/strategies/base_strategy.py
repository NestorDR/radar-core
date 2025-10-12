# src/radar_core/domain/strategies/base_strategy.py

# --- Python modules ---
# abc: abstract base classes (abc) allow implementing interfaces effectively
from abc import ABC, abstractmethod
# decimal: provides support for fast correctly rounded decimal floating point arithmetic
#  it offers several advantages over the float datatype.
from decimal import Decimal, InvalidOperation
# datetime: provides classes for simple and complex date and time manipulation.
from datetime import date, datetime
# json: library for encoding and decoding prices in JSON format.
import json
# logging: defines functions and classes which implement a flexible event logging system for applications and libraries.
from logging import CRITICAL, DEBUG, INFO, getLogger
# typing: provides runtime support for type hints.
from typing import Any, Optional

# --- Third Party Libraries ---
# numpy: provides greater support for vectors and matrices, with high-level mathematical functions to operate on them
import numpy as np
from numpy.typing import NDArray
# polars: is a fast, memory-efficient DataFrame library designed for manipulation and analysis,
#  optimized for performance and parallelism
import polars as pl

# --- App modules ---
# strategies: provides identification and evaluation of speculation/investment strategies on financial instruments
from radar_core.domain.strategies.constants import NO_POSITION, LONG, SHORT
# technical: provides calculations of TA indicators
from radar_core.domain.technical import ATR
# helpers: constants and functions that provide miscellaneous functionality
from radar_core.helpers.constants import TIMEFRAMES
from radar_core.helpers.log_helper import verbose
# infrastructure: allows access to the own database and/or integration with external prices providers
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
        :param future_bar_number: A number of price bars will be available in the future.
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
    Provides a base class to define and evaluate profitable trading strategies.
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
            self.strategy_id, self.unit_label, self.pool = None, None, None
        else:
            self.strategy_id = strategy.id
            self.unit_label = strategy.unit_label
            self.pool = strategy.pool
        self.verbosity_level = verbosity_level
        self.ratio_crud: Optional[RatioCrud] = None

    # region Support to evaluation

    @abstractmethod
    def evaluate(self,
                 symbol: str,
                 timeframe: int,
                 is_long_position: bool,
                 profitable_setting: dict[str, Any],
                 prices_df: pl.DataFrame,
                 verbosity_level: int = DEBUG) -> tuple[int, str, str, date]:
        """
        Evaluates if the tech indicator actives a strategy setting, and if it is yes, notice to the list received.
        
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
         - and a string comment explaining the decision,
         - last output date for a position taken with the strategy under analysis.
        """
        pass

    @staticmethod
    def scatter_ratios(ratios: dict[str, Any]) -> tuple[str, float, float, Decimal, Decimal, date | None]:
        """
        Extracts specific financial ratio values from a dictionary, converts them, and returns them as a tuple.

        :param ratios: A dictionary containing financial ratios with the following keys:
         inputs, net_profit, win_probability, min_percentage_change_to_win, and max_percentage_change_to_win.

        :return: A tuple[str, float, float, decimal.Decimal, decimal.Decimal]
         A tuple containing the extracted and converted financial ratios in the following order:
         inputs, expected_value, win_probability, min_percentage_change_to_win, max_percentage_change_to_win,
         last_output_date_

        :raises: KeyError: If any of the expected keys are missing in the input Series.
                 TypeError: If any of the values cannot be converted to decimal.Decimal.
        """
        try:
            inputs_ = str(ratios['inputs'])
            net_profit_ = round(float(ratios['net_profit']), 2)
            win_probability_ = round(float(ratios['win_probability']), 2)
            min_percentage_change_to_win_ = Decimal(ratios['min_percentage_change_to_win'])
            max_percentage_change_to_win_ = Decimal(ratios['max_percentage_change_to_win'])
            last_output_date_ = ratios['last_output_date']
            if last_output_date_ == date.today():
                last_output_date_ = None

        except KeyError as e:
            raise KeyError(f"Key {e} is missing from the input Series.") from e
        except InvalidOperation as e:
            raise TypeError(f"Value for {e} cannot be converted to decimal.Decimal.") from e

        return (inputs_,
                net_profit_,
                win_probability_,
                min_percentage_change_to_win_,
                max_percentage_change_to_win_,
                last_output_date_)

    def save_notice(self,
                    symbol: str,
                    strategy_id: int,
                    inputs: str,
                    timeframe: int,
                    is_long_position: bool,
                    dt: datetime,
                    open_position: bool,
                    threshold: float,
                    reached_value: float):
        return
        # Deprecating...
        # self.notice_crud.upsert(symbol, strategy_id, inputs, timeframe, is_long_position, dt, open_position,
        #                         Decimal(threshold), Decimal(reached_value))

    # endregion Support to evaluation

    # region Support to identification

    @abstractmethod
    def identify(self,
                 symbol: str,
                 timeframe: int,
                 only_long_positions: bool,
                 prices_df: pl.DataFrame,
                 close_prices: NDArray[Any] | None,  # type: ignore
                 verbosity_level: int = DEBUG) -> dict:
        """
        Iterates a number of periods or levels to calculate a tech indicator and evaluate its profitability.

        :param only_long_positions:
        :param symbol: Security symbol to analyze.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param only_long_positions: True if only long positions are evaluated, otherwise False.
        :param prices_df: Dataframe with required columns.
        :param close_prices: Close prices for the given symbol and timeframe.
        :param verbosity_level: Importance level of messages.

        :return: Dictionary with profitability ratios.
        """
        pass

    def initialize_identification(self,
                                  symbol: str,
                                  timeframe: int,
                                  prices_df: pl.DataFrame,
                                  verbosity_level: int
                                  ) -> tuple[datetime, AnalysisContext, list, int]:
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

        # # Identify the initial and final dates of the period to analyze.
        # prices_df[0, 'Date'],
        # prices_df[-1, 'Date'],
        # # Identify the initial and final prices of the period to analyze
        # prices_df[0, 'Close'],
        # prices_df[-1, 'Close'],
        # # Identify the number of sessions
        # prices_df[-1, 'BarNumber'],
        # self.future_bar_number(prices_df))

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

        # Set dictionary for better indicator strategies (Long and Short)
        result_ratios_ = dict(symbol=analysis_context.symbol,
                              updatedAt=datetime.now().strftime('%Y-%m-%d %H:%M'),
                              # Convert Ratios objects to dictionaries
                              bestLong=self.serialize_ratios(analysis_context.best_long),
                              bestShort=self.serialize_ratios(analysis_context.best_short))

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
    def validate_best_strategy(best_ratios: Ratios) -> Ratios | None:
        """
        Validate the best strategy.
        :param best_ratios: Best strategy based on net profit and expected value.
        :return: A 'Ratios' object with the best strategy validated, or None if the strategy is invalid.
        """
        if best_ratios.net_profit == -float('inf') or best_ratios.expected_value == -float('inf'):
            best_ratios = None

        return best_ratios

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

    def perfile_performance(self,
                            analysis_context: AnalysisContext,
                            inputs: dict,
                            signals: int,
                            trades_df: pl.DataFrame,
                            prices_df: pl.DataFrame) -> Ratios | None:
        """
        Calculates and organizes aggregates and ratios to profile the strategy’s trade performance.

        Visite
            - https://estrategiastrading.com/ratios-para-evaluar-sistemas-de-trading/
            - https://estrategiastrading.com/calcular-la-esperanza-matematica-del-sistema-de-trading/
            - https://estrategiastrading.com/profit-factor/

        :param analysis_context: Analysis context for the strategy.
        :param inputs: Input prices that parameterize a strategy.
        :param signals: Number of trade signals identified by the strategy.
        :param trades_df: The dataframe with info about trades including prices, session details, and results.
        :param prices_df: The dataFrame with prices, indexed by bar numbers and containing the required column Date.

        :return: If the gains exceed the losses returns a Ratios object with the ratios and aggregates calculated for
          trade performance, otherwise returns None.
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
        # Identify prices and session numbers
        first_input_price_ = max(float(trades_df[0, 'InputPrice']), 0.00001)

        # Filter the winning trades and calculate aggregates
        aggregates_ = (
            trades_df.lazy()
            .filter(pl.col('Result') > 0)
            .select(
                pl.sum('Result').alias('TotalWinnings'),  # Total winnings
                pl.count('Result').alias('WinningTradesCount'),  # Count of winning trades
                pl.sum('Sessions').alias('TotalSessions'),  # Total sessions for winning trades
                pl.min('InputPercentChange').alias('MinPercentChange'),  # Minimum percent change
                pl.max('InputPercentChange').alias('MaxPercentChange')  # Maximum percent change
            ).collect())

        # Access the calculated values for winnings
        winnings_ = aggregates_['TotalWinnings'][0]
        winn_trades_ = aggregates_['WinningTradesCount'][0]
        winning_sessions_ = aggregates_['TotalSessions'][0]
        min_percentage_change_to_win_ = aggregates_['MinPercentChange'][0] or 0  # Handle None or null
        max_percentage_change_to_win_ = aggregates_['MaxPercentChange'][0] or 0  # Handle None or null

        # Filter the losing trades and calculate aggregates
        aggregates_ = (
            trades_df.lazy()
            .filter(pl.col('Result') <= 0)
            .select(
                pl.sum('Result').alias('TotalLosses'),  # Total losses
                pl.count('Result').alias('LosingTradesCount'),  # Count of losing trades
                pl.sum('Sessions').alias('TotalSessions')
            ).collect())

        # Access the calculated values for losses
        losses_ = aggregates_['TotalLosses'][0]
        loss_trades_ = aggregates_['LosingTradesCount'][0]
        losing_sessions_ = aggregates_['TotalSessions'][0]

        if winnings_ <= losses_:
            return None

        # Calculate ratios for the strategy
        net_profit_, expected_value_, win_probability_, loss_probability_, average_win_, average_loss_ = \
            self.compute_key_ratios(signals, first_input_price_, winnings_, winn_trades_, losses_, loss_trades_)
        total_sessions_ = analysis_context.last_bar_number + 1

        # Extracts the relevant dates
        first_input_date_, last_input_date_, last_output_date_ = self.get_relevant_dates(trades_df, prices_df)

        # Identify input price and stop loss for the last trade/position
        last_input_price_ = (Decimal(prices_df[int(trades_df[-1, 'InputBarNumber']), 'Close'])
                             .quantize(PRICE_PRECISION))
        last_output_price_ = None if last_output_date_ is None else \
            Decimal(prices_df[int(trades_df[-1, 'OutputBarNumber']), 'Close']).quantize(PRICE_PRECISION)

        if {'LongStopLoss', 'ShortStopLoss'}.issubset(prices_df.columns):
            stop_loss_column_ = 'LongStopLoss' if analysis_context.is_long_position else 'ShortStopLoss'
            last_stop_loss_ = (Decimal(prices_df[int(trades_df[-1, 'InputBarNumber']), stop_loss_column_])
                               .quantize(PRICE_PRECISION))
        else:
            last_stop_loss_ = None

        # Initialize support to ratios with available prices at this point
        return Ratios(
            # Set the relevant part at this point, of the unique restriction key
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

            signals=signals,
            winnings=winnings_,
            losses=losses_,
            # saved_record_.profit_factor = winnings / losses
            net_profit=net_profit_,
            expected_value=expected_value_,
            win_probability=win_probability_,
            loss_probability=loss_probability_,
            average_win=average_win_,
            average_loss=average_loss_,

            min_percentage_change_to_win=Decimal(min_percentage_change_to_win_).quantize(PRICE_PRECISION),
            max_percentage_change_to_win=Decimal(max_percentage_change_to_win_).quantize(PRICE_PRECISION),

            total_sessions=total_sessions_,
            winning_sessions=winning_sessions_,
            losing_sessions=losing_sessions_,
            percentage_exposure=(winning_sessions_ + losing_sessions_) / total_sessions_,

            first_input_date=first_input_date_,
            last_input_date=last_input_date_,
            last_output_date=last_output_date_,

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

    @staticmethod
    def get_relevant_dates(trades_df: pl.DataFrame,
                           prices_df: pl.DataFrame) -> tuple[date, date, date | None]:
        """
        Extracts the relevant dates (of the first and last trades) based on bar numbers from the provided
         trades and prices DataFrames.

        :param prices_df: The dataFrame with prices, indexed by bar numbers and containing the required column Date.
        :param trades_df: The dataframe with info about trades, including session details as initial and final bars.

        :return: A tuple of three dates representing the start date of the first trade, and the start and close dates
         of the last trade, in the format "YYYY-MM-DD".
        """

        # Extracts the relevant dates:
        # - start date of the first trade
        first_input_date_ = prices_df[trades_df[0, 'InputBarNumber'], 'Date']
        # - start date of the last trade
        last_input_date_ = prices_df[int(trades_df[-1, 'InputBarNumber']), 'Date']
        # - close date of the last trade.
        last_output_bar_number_ = int(trades_df[-1, 'OutputBarNumber'])
        last_output_date_ = None if last_output_bar_number_ >= StrategyABC.future_bar_number(prices_df) \
            else prices_df[last_output_bar_number_, 'Date']

        return first_input_date_, last_input_date_, last_output_date_

    def serialize_ratios(self,
                         ratios_: Ratios) -> dict:
        """
        Convert a Ratios object into a JSON-serializable dictionary while handling non-serializable fields and removing
         unnecessary properties.
        :param ratios_: The Ratios object to serialize.
        :return: A serialized dictionary with all attributes JSON-ready.
        """

        serializable_dict_ = {}

        # Use vars() to get the object's attributes as a dictionary and then convert it to a JSON-serializable format
        for key, value in vars(ratios_).items():
            # Ignore some properties
            if key in ('_sa_instance_state', 'id', 'symbol', 'is_in_process'):
                continue
            # Handle datetime objects, convert to string
            if isinstance(value, (date, datetime)):
                serializable_dict_[key] = value.isoformat()
            # Handle Decimal objects, convert to float values and round to 2 decimals
            elif isinstance(value, Decimal):
                serializable_dict_[key] = round(float(value), 2)
            # Handle float values and round to 2 decimals
            elif isinstance(value, float):
                serializable_dict_[key] = round(value, 2)
            # Handle the 'strategy_id' field replace it with its acronym
            elif key == 'strategy_id' and isinstance(value, int):
                serializable_dict_['strategy'] = self.strategy_acronym
            # Handle the 'strategy_id' field replace it with its acronym
            elif key == 'timeframe' and isinstance(value, int):
                serializable_dict_['timeframe'] = TIMEFRAMES[value]
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

    # region Support to evaluation

    def evaluate_rsi_break(self, symbol: str,
                           inputs: str,
                           timeframe: int,
                           is_long_position: bool,
                           min_percentage_change_to_win: Decimal,
                           max_percentage_change_to_win: Decimal,
                           last_output_date: date,
                           prices_df: pl.DataFrame,
                           verbosity_level: int = DEBUG) -> tuple[int, str, str]:
        """
        Evaluates whether a position should be taken or closed based on the Relative Strength Index (RSI) for the given
         financial instrument.
        The method compares the current and previous RSI values against the defined input/output levels and assesses the
         percent change for validity.
        Depending on the configuration, it determines whether to take a long or short position or none.

        :param symbol: Security symbol to evaluate.
        :param inputs: Input prices that parameterize the applied strategy.
        :param timeframe: Timeframe indicator (1.Intraday, 2.Daily, 3.Weekly, 4.Monthly).
        :param is_long_position: True if long trading positions, otherwise False.
        :param max_percentage_change_to_win: The maximum allowable percentage price change for taking or continuing the position
          within the strategy's constraints.
        :param min_percentage_change_to_win: The minimum allowable percentage price change for taking or continuing the position
            within the strategy's constraints.
        :param last_output_date: The date of the last output for the strategy.
        :param prices_df: Historical prices.
        :param verbosity_level: Importance level of messages reporting the progress of the process for this method,
         it will be taken into account only if it is greater than the level of detail specified for the entire class.

        :return: A tuple containing:
         - the suggested position to take as zero (long), 1 (short), or -1 (no position),
         - parameters of the setup,
         - and a string comment explaining the suggested position to take,
        """

        verbosity_level = min(verbosity_level, self.verbosity_level)

        # Parse the inputs_ JSON and extract the relevant parameters
        parsed_inputs_ = json.loads(inputs)
        if 'over' in parsed_inputs_:
            in_ = parsed_inputs_['in']
            over_ = parsed_inputs_['over']
            out_ = parsed_inputs_['out']
            parameters_ = f'{in_}, {over_}, {out_}'
        else:
            in_ = parsed_inputs_['in']
            out_ = parsed_inputs_['out']
            parameters_ = f'{in_}, {out_}'

        if verbosity_level == DEBUG:
            print(f'{symbol}: evaluating {self.strategy_acronym}({parameters_})...')

        # Extract the 'Rsi' values for the last and second-last rows
        # Get the last row
        row_index_ = prices_df.height - 1
        rsi_ = prices_df['Rsi'][row_index_]
        percent_change_ = prices_df['PercentChange'][row_index_]
        date_ = prices_df['Date'][row_index_]
        # Get the second-last row
        row_index_ -= 1
        previous_rsi_ = prices_df['Rsi'][row_index_]
        position_ = NO_POSITION
        comment_ = ''

        if is_long_position:
            threshold_ = in_
            # Check if RSI rises above the long RSI input level
            if rsi_ > in_ >= previous_rsi_ and last_output_date is not None:
                if min_percentage_change_to_win <= percent_change_ <= max_percentage_change_to_win:
                    position_ = LONG
                    comment_ = f' {rsi_:.2f} > {in_:.2f} ({percent_change_:.2f}%)'

            # Check if RSI falls below the long RSI output level
            elif previous_rsi_ > out_ >= rsi_ and last_output_date is None:
                position_ = SHORT
                comment_ = f' {rsi_:.2f} < {out_:.2f}'

        else:
            threshold_ = out_

            # Check if RSI falls below the short RSI input level
            if rsi_ < in_ <= previous_rsi_:
                if min_percentage_change_to_win <= percent_change_ <= max_percentage_change_to_win:
                    position_ = SHORT
                    comment_ = f' {rsi_:.2f} < {in_:.2f} ({percent_change_:.2f}%)'

            # Check if RSI rises above the short RSI output level
            elif previous_rsi_ < out_ <= rsi_:
                position_ = LONG
                comment_ = f' {rsi_:.2f} > {out_:.2f}'

        if position_ != NO_POSITION:
            self.save_notice(symbol, self.strategy_id, inputs, timeframe, is_long_position, date_, position_ == LONG,
                             threshold_, rsi_)

        return position_, parameters_, comment_

    # endregion Support to evaluation

    # region Stop Loss

    @staticmethod
    def identify_where_to_stop_loss(prices_df: pl.DataFrame,
                                    close_prices: pl.Series,
                                    bars_for_stop_loss: int) -> pl.DataFrame:
        """
        Identifies and calculates where to stop losses for both long and short positions.
        The method calculates the stop-loss trigger levels and associates bars where these triggers occur.

        :param prices_df: A DataFrame containing at least the price columns
         ['LongStopLoss', 'ShortStopLoss', 'BarNumberForLongStop', 'BarNumberForShortStop'].
        :param close_prices: A Series containing the close prices.
        :param bars_for_stop_loss: The number of bars (window size) to identify recent lowest/highest prices and
         consider it for stop loss calculation.

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

        # Set stop loss prices
        prices_df = RsiStrategyABC.set_stop_loss(prices_df, bars_for_stop_loss)

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

        # Release resources
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
