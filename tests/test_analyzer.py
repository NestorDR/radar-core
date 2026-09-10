# tests/test_analyzer.py

# --- Python modules ---
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.analyzer import analyze
from radar_core.domain.strategies import EvaluableStrategies, RsiStrategyABC
from radar_core.helpers.constants import DAILY
from radar_core.settings import Settings


@pytest.fixture(autouse=True)
def clean_settings_state():
    """
    Ensures that Settings singleton state is cleanly reset before and after each test.
    """
    Settings._reset()
    yield
    Settings._reset()


def _make_sample_prices_df(bar_count: int = 50) -> pl.DataFrame:
    """
    Creates a synthetic Polars DataFrame with required OHLCV columns.

    :param bar_count: Total number of price bars.
    :return: Polars DataFrame with Open, High, Low, Close, Volume, PercentChange, Date.
    """
    dates_ = [datetime(2025, 1, 1) + timedelta(days=i_) for i_ in range(bar_count)]
    close_ = np.linspace(100.0, 150.0, bar_count)
    open_ = close_ - 0.5
    high_ = close_ + 1.0
    low_ = close_ - 1.0
    volume_ = np.full(bar_count, 1000.0)
    pct_change_ = np.zeros(bar_count)
    pct_change_[1:] = (close_[1:] - close_[:-1]) / close_[:-1]

    return pl.DataFrame({
        'Date': dates_,
        'Open': open_,
        'High': high_,
        'Low': low_,
        'Close': close_,
        'Volume': volume_,
        'PercentChange': pct_change_,
    })


def test_analyze_injects_price_action_masks_when_configured(monkeypatch, tmp_path):
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
    prices_df_ = _make_sample_prices_df(bar_count=60)

    with (
        patch('radar_core.analyzer.RSI', side_effect=lambda df: df.with_columns(pl.lit(50.0).alias('Rsi'))),
        patch.object(RsiStrategyABC, 'identify_where_to_stop_loss', side_effect=lambda tf, df, cp: df),
    ):
        analyze(
            timeframe=DAILY,
            symbol='SPY',
            only_long_positions=False,
            prices_df=prices_df_,
            strategies=strategies_,
        )

    assert mock_rsi_2b_.identify.called
    args_2b_ = mock_rsi_2b_.identify.call_args.args
    masks_2b_ = args_2b_[5]
    assert isinstance(masks_2b_, tuple)
    assert len(masks_2b_) == 2
    assert isinstance(masks_2b_[0], np.ndarray)
    assert isinstance(masks_2b_[1], np.ndarray)
    assert len(masks_2b_[0]) == 60
    assert len(masks_2b_[1]) == 60

    assert mock_rsi_rc_.identify.called
    args_rc_ = mock_rsi_rc_.identify.call_args.args
    masks_rc_ = args_rc_[5]
    assert masks_rc_ is masks_2b_


def test_analyze_passes_none_when_filter_is_omitted(monkeypatch, tmp_path):
    """
    GIVEN an analyzer execution where rsi_input_filter is omitted from YAML
    WHEN analyze is invoked for RSI strategies
    THEN it forwards is_input_eligible=None for unfiltered baseline execution.
    """
    custom_yaml_ = tmp_path / 'settings.yml'
    custom_yaml_.write_text('symbols:\n  - SPY\nevaluable_strategies:\n  - rsi_2b\n  - rsi_rc\n')
    monkeypatch.setenv('RADAR_SETTING_FILE', str(custom_yaml_))

    mock_rsi_2b_ = MagicMock()
    mock_rsi_rc_ = MagicMock()
    strategies_ = EvaluableStrategies(rsi_2b=mock_rsi_2b_, rsi_rc=mock_rsi_rc_)
    prices_df_ = _make_sample_prices_df(bar_count=60)

    with (
        patch('radar_core.analyzer.RSI', side_effect=lambda df: df.with_columns(pl.lit(50.0).alias('Rsi'))),
        patch.object(RsiStrategyABC, 'identify_where_to_stop_loss', side_effect=lambda tf, df, cp: df),
    ):
        analyze(
            timeframe=DAILY,
            symbol='SPY',
            only_long_positions=False,
            prices_df=prices_df_,
            strategies=strategies_,
        )

    assert mock_rsi_2b_.identify.called
    assert mock_rsi_2b_.identify.call_args.args[5] is None

    assert mock_rsi_rc_.identify.called
    assert mock_rsi_rc_.identify.call_args.args[5] is None


def test_analyze_passes_none_when_filter_is_none_string(monkeypatch, tmp_path):
    """
    GIVEN an analyzer execution where rsi_input_filter is explicitly 'none'
    WHEN analyze is invoked for RSI strategies
    THEN it forwards is_input_eligible=None for unfiltered baseline execution.
    """
    custom_yaml_ = tmp_path / 'settings.yml'
    custom_yaml_.write_text("symbols:\n  - SPY\nevaluable_strategies:\n  - rsi_2b\n  - rsi_rc\nrsi_input_filter: 'none'\n")
    monkeypatch.setenv('RADAR_SETTING_FILE', str(custom_yaml_))

    mock_rsi_2b_ = MagicMock()
    mock_rsi_rc_ = MagicMock()
    strategies_ = EvaluableStrategies(rsi_2b=mock_rsi_2b_, rsi_rc=mock_rsi_rc_)
    prices_df_ = _make_sample_prices_df(bar_count=60)

    with (
        patch('radar_core.analyzer.RSI', side_effect=lambda df: df.with_columns(pl.lit(50.0).alias('Rsi'))),
        patch.object(RsiStrategyABC, 'identify_where_to_stop_loss', side_effect=lambda tf, df, cp: df),
    ):
        analyze(
            timeframe=DAILY,
            symbol='SPY',
            only_long_positions=False,
            prices_df=prices_df_,
            strategies=strategies_,
        )

    assert mock_rsi_2b_.identify.called
    assert mock_rsi_2b_.identify.call_args.args[5] is None

    assert mock_rsi_rc_.identify.called
    assert mock_rsi_rc_.identify.call_args.args[5] is None

