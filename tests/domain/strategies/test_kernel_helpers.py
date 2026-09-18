# tests/domain/strategies/test_kernel_helpers.py

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from radar_core.domain.strategies._kernel_helpers import (
    _calculate_trade_pnl,
    _crosses_input,
    _crosses_input_persistent,
    _crosses_output,
    _crosses_over_level,
    _finalize_screening_metrics,
    _is_better_candidate,
    _is_profitable_candidate,
    _mark_to_market_bar,
)


@pytest.mark.parametrize(
    ('prev_val', 'curr_val', 'prev_th', 'curr_th', 'is_long', 'expected'),
    [
        (60.0, 60.0, 60.0, 60.0, True, False),
        (60.0, 60.1, 60.0, 60.0, True, True),
        (60.0, 59.9, 60.0, 60.0, False, True),
        (60.0, 60.0, 60.0, 60.0, False, False),
    ],
    ids=['long_equal', 'long_cross_above', 'short_cross_below', 'short_equal'],
)
def test_crosses_input_preserves_long_short_boundaries(
    prev_val: float, curr_val: float, prev_th: float, curr_th: float, is_long: bool, expected: bool
) -> None:
    """
    GIVEN values at and around an entry threshold.
    WHEN the shared entry predicate is evaluated for long and short positions.
    THEN strict current and inclusive previous boundaries are preserved.
    """
    assert _crosses_input(prev_val, curr_val, prev_th, curr_th, is_long) is expected


def test_crosses_input_persistent_preserves_boundaries() -> None:
    """
    GIVEN values at t-2, t-1, and t around an entry threshold.
    WHEN the persistent entry predicate is evaluated for long and short positions.
    THEN 1-bar flickers are rejected and 2-bar dwell periods are required.
    """
    # Long positions: prior <= in_ AND previous <= in_ AND current > in_
    assert _crosses_input_persistent(60.0, 60.0, 60.1, 60.0, 60.0, True) is True
    assert _crosses_input_persistent(55.0, 58.0, 62.0, 60.0, 60.0, True) is True
    # 1-bar flicker rejected (prior was above threshold)
    assert _crosses_input_persistent(60.1, 59.9, 60.1, 60.0, 60.0, True) is False
    # Equality boundaries
    assert _crosses_input_persistent(60.0, 60.0, 60.0, 60.0, 60.0, True) is False

    # Short positions: prior >= in_ AND previous >= in_ AND current < in_
    assert _crosses_input_persistent(60.0, 60.0, 59.9, 60.0, 60.0, False) is True
    assert _crosses_input_persistent(65.0, 62.0, 58.0, 60.0, 60.0, False) is True
    # 1-bar flicker rejected (prior was below threshold)
    assert _crosses_input_persistent(59.9, 60.1, 59.9, 60.0, 60.0, False) is False
    # Equality boundaries
    assert _crosses_input_persistent(60.0, 60.0, 60.0, 60.0, 60.0, False) is False


@pytest.mark.parametrize(
    ('prev_val', 'curr_val', 'prev_th', 'curr_th', 'is_long', 'expected'),
    [
        (12.0, 10.0, 11.0, 11.0, True, True),
        (10.0, 12.0, 11.0, 11.0, False, True),
        (11.0, 10.0, 11.0, 11.0, True, False),
    ],
    ids=['long_cross_below', 'short_cross_above', 'long_start_at_threshold'],
)
def test_crosses_output_supports_dynamic_thresholds(
    prev_val: float, curr_val: float, prev_th: float, curr_th: float, is_long: bool, expected: bool
) -> None:
    """
    GIVEN values compared with different previous and current thresholds.
    WHEN the shared exit predicate is evaluated.
    THEN moving-average-style threshold crossings use the expected boundaries.
    """
    assert _crosses_output(prev_val, curr_val, prev_th, curr_th, is_long) is expected


@pytest.mark.parametrize(
    ('prev_rsi', 'curr_rsi', 'level', 'is_long', 'expected'),
    [
        (79.0, 80.0, 80.0, True, True),
        (80.0, 80.0, 80.0, True, False),
        (81.0, 80.0, 80.0, False, True),
    ],
    ids=['long_cross_up', 'long_equal_previous', 'short_cross_down'],
)
def test_crosses_over_level_preserves_rollercoaster_boundaries(
    prev_rsi: float, curr_rsi: float, level: float, is_long: bool, expected: bool
) -> None:
    """
    GIVEN RSI values around an intermediate RollerCoaster level.
    WHEN the shared over-level predicate is evaluated.
    THEN inclusive current and strict previous boundaries are preserved.
    """
    assert _crosses_over_level(prev_rsi, curr_rsi, level, is_long) is expected


def test_scalar_screening_helpers_match_strategy_formulas() -> None:
    """
    GIVEN scalar trade aggregates and candidate metrics.
    WHEN shared screening helpers are evaluated.
    THEN PnL, ratios, profitability, ranking, and mark-to-market results match the expected percentage formulas.
    """
    pnl_ = _calculate_trade_pnl(100.0, 110.0, 1.0, 0.01)
    assert pnl_ == 7.9

    (
        net_profit_,
        win_probability_,
        loss_probability_,
        average_win_percentage_,
        average_loss_percentage_,
        expected_percentage_,
    ) = _finalize_screening_metrics(2, 100.0, 10.0, 1, -2.0, 1, 0.10, -0.02)

    assert net_profit_ == 0.08
    assert win_probability_ == 0.5
    assert loss_probability_ == 0.5
    assert average_win_percentage_ == 0.10
    assert average_loss_percentage_ == -0.02
    assert _is_profitable_candidate(net_profit_, expected_percentage_, win_probability_, 0.5) is True
    assert _is_profitable_candidate(net_profit_, expected_percentage_, 0.49, 0.5) is False
    assert _is_profitable_candidate(net_profit_, expected_percentage_, win_probability_, 0.51) is False
    assert _is_better_candidate(0.08, 0.04, 0.07, 0.05) is True
    assert _is_better_candidate(0.08, 0.04, 0.08, 0.04) is False
    assert _mark_to_market_bar(2, 3) == 2
    assert _mark_to_market_bar(3, 3) == 2
