# tests/test_analyzer.py

# --- Python modules ---
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.analyzer import analyze, analyzer, process_symbol
from radar_core.domain.strategies import EvaluableStrategies, RsiStrategyABC
from radar_core.helpers.constants import DAILY
from tests.conftest import QQQ, SOXS, SPY, SQQQ


def test_analyze_injects_price_action_masks_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ohlcv_factory: Callable[..., pl.DataFrame]
) -> None:
    """
    GIVEN an analyzer execution where rsi_input_filter is 'price_action'
    WHEN analyze is invoked for RSI strategies (rsi_2b and rsi_rc)
    THEN it precomputes and forwards the (long_mask, short_mask) tuple to identify.
    """
    custom_yaml_ = tmp_path / 'settings.yml'
    custom_yaml_.write_text("symbols:\n  - SPY\nevaluable_strategies:\n  - rsi_2b\n  - rsi_rc\nrsi_input_filter: 'price_action'\n")
    monkeypatch.setenv('RADAR_SETTING_FILE', str(custom_yaml_))

    mock_rsi_2b_ = MagicMock()
    mock_rsi_rc_ = MagicMock()
    strategies_ = EvaluableStrategies(rsi_2b=mock_rsi_2b_, rsi_rc=mock_rsi_rc_)
    prices_df_ = ohlcv_factory(60)

    with (
        patch('radar_core.analyzer.RSI', side_effect=lambda df: df.with_columns(pl.lit(50.0).alias('Rsi'))),
        patch.object(RsiStrategyABC, 'identify_where_to_stop_loss', side_effect=lambda tf, df, cp: df),
    ):
        analyze(
            timeframe=DAILY,
            symbol=SPY,
            only_long_positions=False,
            prices_df=prices_df_,
            strategies=strategies_,
            is_bear=False,
        )

    assert mock_rsi_2b_.identify.called
    args_2b_ = mock_rsi_2b_.identify.call_args.args
    assert args_2b_[5] == 0.5
    masks_2b_ = args_2b_[6]
    assert isinstance(masks_2b_, tuple)
    assert len(masks_2b_) == 2
    assert masks_2b_[0] is None
    assert isinstance(masks_2b_[1], np.ndarray)
    assert len(masks_2b_[1]) == 60

    assert mock_rsi_rc_.identify.called
    args_rc_ = mock_rsi_rc_.identify.call_args.args
    assert args_rc_[5] == 0.5
    masks_rc_ = args_rc_[6]
    assert masks_rc_ is masks_2b_

    # Verify inverse ETF / bear asset (is_bear=True): Long is ndarray, Short is None
    analyze(
        timeframe=DAILY,
        symbol=SQQQ,
        only_long_positions=False,
        prices_df=prices_df_,
        strategies=strategies_,
        is_bear=True,
    )
    args_2b_bear_ = mock_rsi_2b_.identify.call_args.args
    assert args_2b_bear_[5] == 0.5
    masks_2b_bear_ = args_2b_bear_[6]
    assert isinstance(masks_2b_bear_, tuple)
    assert len(masks_2b_bear_) == 2
    assert isinstance(masks_2b_bear_[0], np.ndarray)
    assert len(masks_2b_bear_[0]) == 60
    assert masks_2b_bear_[1] is None


@pytest.mark.parametrize(
    'filter_config_line',
    ['', "rsi_input_filter: 'none'\n"],
    ids=['omitted', 'none_string'],
)
def test_analyze_passes_none_tuple_for_unfiltered_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ohlcv_factory: Callable[..., pl.DataFrame], filter_config_line: str
) -> None:
    """
    GIVEN an analyzer execution where rsi_input_filter is omitted or explicitly 'none'
    WHEN analyze is invoked for RSI strategies
    THEN it forwards is_input_eligible=(None, None) for unfiltered baseline execution.
    """
    custom_yaml_ = tmp_path / 'settings.yml'
    custom_yaml_.write_text(f'symbols:\n  - SPY\nevaluable_strategies:\n  - rsi_2b\n  - rsi_rc\n{filter_config_line}')
    monkeypatch.setenv('RADAR_SETTING_FILE', str(custom_yaml_))

    mock_rsi_2b_ = MagicMock()
    mock_rsi_rc_ = MagicMock()
    strategies_ = EvaluableStrategies(rsi_2b=mock_rsi_2b_, rsi_rc=mock_rsi_rc_)
    prices_df_ = ohlcv_factory(60)

    with (
        patch('radar_core.analyzer.RSI', side_effect=lambda df: df.with_columns(pl.lit(50.0).alias('Rsi'))),
        patch.object(RsiStrategyABC, 'identify_where_to_stop_loss', side_effect=lambda tf, df, cp: df),
    ):
        analyze(
            timeframe=DAILY,
            symbol=SPY,
            only_long_positions=False,
            prices_df=prices_df_,
            strategies=strategies_,
            is_bear=False,
        )

    for mock_strat_ in (mock_rsi_2b_, mock_rsi_rc_):
        assert mock_strat_.identify.called
        assert mock_strat_.identify.call_args.args[5] == 0.5
        assert mock_strat_.identify.call_args.args[6] == (None, None)


@pytest.mark.parametrize(
    ('shortable_symbols', 'expected_only_long'),
    [
        ({SPY, QQQ}, False),
        ({QQQ}, True),
    ],
    ids=['shortable', 'not_shortable'],
)
def test_process_symbol_short_eligibility(
    ohlcv_factory: Callable[..., pl.DataFrame], shortable_symbols: set[str], expected_only_long: bool
) -> None:
    """
    GIVEN a symbol and a shortable_symbols set
    WHEN process_symbol is called to evaluate strategies
    THEN only_long_positions matches whether the symbol is in the shortable set.
    """
    mock_strategy_ = MagicMock()
    strategies_ = EvaluableStrategies(sma=mock_strategy_)
    prices_df_ = ohlcv_factory(60)

    with patch('radar_core.analyzer.analyze') as mock_analyze_:
        process_symbol(
            symbol=SPY,
            prices_df=prices_df_,
            strategies=strategies_,
            shortable_symbols=shortable_symbols,
            bear_symbols=set(),
            verbosity_level=20,
        )

    assert mock_analyze_.called
    daily_call_args_ = mock_analyze_.call_args_list[0].args
    assert daily_call_args_[1] == SPY
    assert daily_call_args_[2] is expected_only_long


@pytest.mark.parametrize(
    ('symbol', 'expected_is_bear'),
    [
        (SQQQ, True),
        (SPY, False),
    ],
    ids=['bear_asset', 'standard_asset'],
)
def test_process_symbol_propagates_is_bear_flag(
    ohlcv_factory: Callable[..., pl.DataFrame], symbol: str, expected_is_bear: bool
) -> None:
    """
    GIVEN a symbol classified as bear or standard asset
    WHEN process_symbol is called to evaluate strategies
    THEN analyze is invoked with the corresponding is_bear flag.
    """
    mock_strategy_ = MagicMock()
    strategies_ = EvaluableStrategies(sma=mock_strategy_)
    prices_df_ = ohlcv_factory(60)
    bear_set_ = {SQQQ, SOXS}

    with patch('radar_core.analyzer.analyze') as mock_analyze_:
        process_symbol(
            symbol=symbol,
            prices_df=prices_df_,
            strategies=strategies_,
            shortable_symbols=set(),
            verbosity_level=20,
            bear_symbols=bear_set_,
        )

    assert mock_analyze_.called
    daily_call_ = mock_analyze_.call_args_list[0]
    is_bear_arg_ = daily_call_.kwargs.get('is_bear', daily_call_.args[5] if len(daily_call_.args) > 5 else None)
    assert is_bear_arg_ is expected_is_bear


def test_analyzer_resolves_shortable_and_bear_symbols_via_security_repository() -> None:
    """
    GIVEN an analyzer invocation with symbols
    WHEN analyzer is called
    THEN it resolves shortable and bear symbols via SecurityRepository.
    """
    with (
        patch('radar_core.analyzer.clean'),
        patch('radar_core.analyzer.MovingAverage'),
        patch('radar_core.analyzer.RsiRollerCoaster'),
        patch('radar_core.analyzer.RsiTwoBands'),
        patch('radar_core.analyzer.SecurityRepository') as mock_repo_cls_,
        patch('radar_core.analyzer.PriceProvider') as mock_provider_cls_,
    ):
        mock_repo_ = MagicMock()
        mock_repo_.get_shortable_symbols.return_value = {SPY}
        mock_repo_.get_bear_symbols.return_value = set()
        mock_repo_cls_.return_value = mock_repo_

        mock_provider_ = MagicMock()
        mock_provider_.get_prices.return_value = {}
        mock_provider_cls_.return_value = mock_provider_

        exit_code_ = analyzer(symbols=[SPY])

    assert exit_code_ == 0
    mock_repo_cls_.assert_called_once()
    mock_repo_.get_shortable_symbols.assert_called_once_with([SPY])
    mock_repo_.get_bear_symbols.assert_called_once_with([SPY])



