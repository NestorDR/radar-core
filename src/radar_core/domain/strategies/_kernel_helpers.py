# src/radar_core/domain/strategies/_kernel_helpers.py
"""Shared pure Numba helpers for strategy execution kernels."""

# --- Third Party Libraries ---
# numba: JIT compiler that compiles a subset of Python and NumPy code into optimized machine code
from numba import njit

"""
inline='always' tells Numba to substitute the helper’s compiled body directly into each compiled caller 
 instead of emitting a normal function call.
For example when _crosses_input() is used inside _find_trades_* or _grid_search_*_fused, 
 Numba attempts to compile the comparison logic directly into the surrounding loop.
 
Benefits:
- removes function-call overhead;
- exposes comparisons to further compiler optimization;
- allows constant propagation and better register allocation;
- is useful for tiny helpers called repeatedly inside hot loops.

It does not inline Python source text. It is a compile-time Numba decision. 

cache=True has a different purpose: it stores compiled specializations on disk so later processes can reuse them.
 It does not imply inlining.
 
There are trade-offs:
- larger generated machine code;
- potentially longer compilation;
- possible register pressure or instruction-cache impact;
- little benefit for helpers called only once per candidate.

Therefore, inline='always' is appropriate for tiny predicates
"""

@njit(cache=True, inline='always')
def _crosses_input(
    previous_value: float,
    current_value: float,
    previous_threshold: float,
    current_threshold: float,
    is_long_position: bool,
) -> bool:
    """
    Determine whether a value crosses into a position.

    :param previous_value: Value on the preceding bar.
    :param current_value: Value on the current bar.
    :param previous_threshold: Threshold on the preceding bar.
    :param current_threshold: Threshold on the current bar.
    :param is_long_position: Whether the position is long.
    :return: True when the long or short entry crossing occurs.
    """
    if is_long_position:
        return previous_value <= previous_threshold and current_value > current_threshold

    return previous_value >= previous_threshold and current_value < current_threshold


@njit(cache=True, inline='always')
def _crosses_output(
    previous_value: float,
    current_value: float,
    previous_threshold: float,
    current_threshold: float,
    is_long_position: bool,
) -> bool:
    """
    Determine whether a value crosses out of a position.

    :param previous_value: Value on the preceding bar.
    :param current_value: Value on the current bar.
    :param previous_threshold: Threshold on the preceding bar.
    :param current_threshold: Threshold on the current bar.
    :param is_long_position: Whether the position is long.

    :return: True when the long or short exit crossing occurs.
    """
    if is_long_position:
        return previous_value > previous_threshold and current_value <= current_threshold

    return previous_value < previous_threshold and current_value >= current_threshold


@njit(cache=True, inline='always')
def _crosses_over_level(
    previous_value: float,
    current_value: float,
    level: float,
    is_long_position: bool,
) -> bool:
    """
    Determine whether RSI reaches a RollerCoaster intermediate level.

    :param previous_value: RSI value on the preceding bar.
    :param current_value: RSI value on the current bar.
    :param level: Overbought or oversold threshold.
    :param is_long_position: Whether the position is long.

    :return: True when the intermediate RSI crossing occurs.
    """
    if is_long_position:
        return current_value >= level > previous_value

    return current_value <= level < previous_value


@njit(cache=True, inline='always')
def _calculate_trade_pnl(
    input_price: float,
    output_price: float,
    direction: float,
    commission_percent: float,
) -> float:
    """
    Calculate the commission-adjusted profit or loss for one trade.

    :param input_price: Price at which the position was opened.
    :param output_price: Price at which the position was closed or valued.
    :param direction: Position direction, positive for long and negative for short.
    :param commission_percent: Commission multiplier applied to both prices.

    :return: Commission-adjusted trade profit or loss.
    """
    return (output_price - input_price) * direction - commission_percent * (input_price + output_price)


@njit(cache=True, inline='always')
def _is_profitable_candidate(net_profit: float, expected_value: float) -> bool:
    """
    Check the profitability criteria used by fused screening kernels.

    :param net_profit: Candidate net-profit ratio.
    :param expected_value: Candidate expected value.

    :return: True when both profitability criteria are strictly positive.
    """
    return net_profit > 0.0 and expected_value > 0.0


@njit(cache=True, inline='always')
def _is_better_candidate(
    net_profit: float,
    expected_value: float,
    best_net_profit: float,
    best_expected_value: float,
) -> bool:
    """
    Compare a candidate using the project's ranking order.

    :param net_profit: Candidate net-profit ratio.
    :param expected_value: Candidate expected value.
    :param best_net_profit: Best net-profit ratio found so far.
    :param best_expected_value: Best expected value found so far.

    :return: True when the candidate outranks the current best candidate.
    """
    return net_profit > best_net_profit or (
        net_profit == best_net_profit and expected_value > best_expected_value
    )


@njit(cache=True, inline='always')
def _finalize_screening_metrics(
    signals: int,
    first_input_price: float,
    winnings: float,
    winning_trades: int,
    losses: float,
    losing_trades: int,
) -> tuple[float, float, float, float, float, float]:
    """
    Finalize scalar performance metrics for a screened candidate.

    :param signals: Number of trades identified for the candidate.
    :param first_input_price: Price used to normalize net profit.
    :param winnings: Sum of positive trade results.
    :param winning_trades: Number of positive trade results.
    :param losses: Sum of non-positive trade results.
    :param losing_trades: Number of non-positive trade results.

    :return: Net profit, win probability, loss probability, average win, average loss, and expected value.
    """
    if signals <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    net_profit_ = (winnings + losses) / first_input_price
    win_probability_ = winning_trades / signals
    loss_probability_ = losing_trades / signals
    average_win_ = winnings / winning_trades if winning_trades > 0 else 0.0
    average_loss_ = losses / losing_trades if losing_trades > 0 else 0.0
    expected_value_ = win_probability_ * average_win_ + loss_probability_ * average_loss_

    return net_profit_, win_probability_, loss_probability_, average_win_, average_loss_, expected_value_


@njit(cache=True, inline='always')
def _mark_to_market_bar(output_bar: int, total_bars: int) -> int:
    """
    Convert a future marker into the final valid price-bar index.

    :param output_bar: Trade output bar, possibly the future marker.
    :param total_bars: Number of bars in the price series.

    :return: Valid index to be used to get the mark-to-market price.
    """
    return output_bar if output_bar < total_bars else total_bars - 1
