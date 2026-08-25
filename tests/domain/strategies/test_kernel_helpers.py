# tests/domain/strategies/test_kernel_helpers.py

from radar_core.domain.strategies._kernel_helpers import (
    _calculate_trade_pnl,
    _crosses_input,
    _crosses_output,
    _crosses_over_level,
    _finalize_screening_metrics,
    _is_better_candidate,
    _is_profitable_candidate,
    _mark_to_market_bar,
)


def test_crosses_input_preserves_long_short_boundaries() -> None:
    """
    GIVEN values at and around an entry threshold.
    WHEN the shared entry predicate is evaluated for long and short positions.
    THEN strict current and inclusive previous boundaries are preserved.
    """
    assert _crosses_input(60.0, 60.0, 60.0, 60.0, True) is False
    assert _crosses_input(60.0, 60.1, 60.0, 60.0, True) is True
    assert _crosses_input(60.0, 59.9, 60.0, 60.0, False) is True
    assert _crosses_input(60.0, 60.0, 60.0, 60.0, False) is False


def test_crosses_output_supports_dynamic_thresholds() -> None:
    """
    GIVEN values compared with different previous and current thresholds.
    WHEN the shared exit predicate is evaluated.
    THEN moving-average-style threshold crossings use the expected boundaries.
    """
    assert _crosses_output(12.0, 10.0, 11.0, 11.0, True) is True
    assert _crosses_output(10.0, 12.0, 11.0, 11.0, False) is True
    assert _crosses_output(11.0, 10.0, 11.0, 11.0, True) is False


def test_crosses_over_level_preserves_rollercoaster_boundaries() -> None:
    """
    GIVEN RSI values around an intermediate RollerCoaster level.
    WHEN the shared over-level predicate is evaluated.
    THEN inclusive current and strict previous boundaries are preserved.
    """
    assert _crosses_over_level(79.0, 80.0, 80.0, True) is True
    assert _crosses_over_level(80.0, 80.0, 80.0, True) is False
    assert _crosses_over_level(81.0, 80.0, 80.0, False) is True


def test_scalar_screening_helpers_match_strategy_formulas() -> None:
    """
    GIVEN scalar trade aggregates and candidate metrics.
    WHEN shared screening helpers are evaluated.
    THEN PnL, ratios, profitability, ranking, and mark-to-market results match the existing formulas.
    """
    pnl_ = _calculate_trade_pnl(100.0, 110.0, 1.0, 0.01)
    assert pnl_ == 7.9

    (
        net_profit_,
        win_probability_,
        loss_probability_,
        average_win_,
        average_loss_,
        expected_value_,
    ) = _finalize_screening_metrics(2, 100.0, 10.0, 1, -2.0, 1)

    assert net_profit_ == 0.08
    assert win_probability_ == 0.5
    assert loss_probability_ == 0.5
    assert average_win_ == 10.0
    assert average_loss_ == -2.0
    assert expected_value_ == 4.0
    assert _is_profitable_candidate(net_profit_, expected_value_) is True
    assert _is_better_candidate(0.08, 4.0, 0.07, 5.0) is True
    assert _is_better_candidate(0.08, 4.0, 0.08, 4.0) is False
    assert _mark_to_market_bar(2, 3) == 2
    assert _mark_to_market_bar(3, 3) == 2
