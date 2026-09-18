# tests/conftest.py

# --- Python modules ---
from collections.abc import Callable, Generator
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
from typing import Any, Final
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import numpy as np
import polars as pl
import pytest

# --- App modules ---
from radar_core.domain.technical import RSI
from radar_core.helpers.constants import RSI_2B, RSI_RC, RSI_SMA, SMA
from radar_core.infrastructure.price_provider import PriceProvider
from radar_core.infrastructure.security_repository import SecurityRepository
from radar_core.models import Ratios, Strategies
from radar_core.settings import Settings, get_settings

# --- Strategy Display Name Constants ---
STRATEGY_NAME_SMA: Final[str] = 'Simple Moving Average'
STRATEGY_NAME_RSI_2B: Final[str] = 'RSI Two Bands'
STRATEGY_NAME_RSI_RC: Final[str] = 'RSI RollerCoaster'
STRATEGY_NAME_RSI_SMA: Final[str] = 'RSI Simple Moving Average'

# --- Common Test Security Symbols ---
SPY: Final[str] = 'SPY'
QQQ: Final[str] = 'QQQ'
BTC_USD: Final[str] = 'BTC-USD'
AAPL: Final[str] = 'AAPL'
NDQ: Final[str] = 'NDQ'
NVDA: Final[str] = 'NVDA'
SOXS: Final[str] = 'SOXS'
SPX: Final[str] = 'SPX'
SQQQ: Final[str] = 'SQQQ'


@pytest.fixture(scope='session', autouse=True)
def initialize_environment() -> Generator[None, None, None]:
    """
    Session fixture to initialize application settings and load environment variables
    from .env before running any pytest test suite.
    Directs the default price cache directory to tests/cache rather than the repository root.
    """
    test_cache_dir_ = Path(__file__).parent / 'cache'
    os.environ['RADAR_PRICE_CACHE_DIR'] = str(test_cache_dir_)
    Settings._reset()
    get_settings()
    yield
    if test_cache_dir_.exists():
        shutil.rmtree(test_cache_dir_, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_settings_state() -> Generator[None, None, None]:
    """
    Function-scoped autouse fixture that cleanly resets the Settings singleton state
    before and after each test function.
    """
    Settings._reset()
    yield
    Settings._reset()


@pytest.fixture
def mock_connection_scope() -> tuple[MagicMock, MagicMock, MagicMock]:
    """
    Creates an isolated mocked database connection scope.

    :return: A tuple containing the database connection, cursor, and scope.
    """
    connection_ = MagicMock(name='connection')
    cursor_ = MagicMock(name='cursor')

    connection_.cursor.return_value.__enter__.return_value = cursor_

    scope_ = MagicMock(name='connection_scope')
    scope_.__enter__.return_value = connection_
    scope_.__exit__.return_value = False

    return connection_, cursor_, scope_


@pytest.fixture
def mock_crud_scope(
    mock_connection_scope: tuple[MagicMock, MagicMock, MagicMock]
) -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
    """
    Creates a pre-patched connection scope for CRUD operations across models.

    :param mock_connection_scope: The underlying mocked database connection, cursor, and scope.
    :return: Tuple of (connection, cursor, scope) with CRUD scopes pre-patched.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    with (
        patch('radar_core.infrastructure.crud.ratio_crud.connection_scope', return_value=scope_),
        patch('radar_core.infrastructure.crud.security_crud.connection_scope', return_value=scope_),
        patch('radar_core.infrastructure.crud.security_crud.read_connection_scope', return_value=scope_),
        patch('radar_core.infrastructure.crud.strategy_crud.read_connection_scope', return_value=scope_),
    ):
        yield connection_, cursor_, scope_


@pytest.fixture(autouse=True)
def mock_strategy_db() -> Generator[None, None, None]:
    """
    Autouse fixture that mocks database strategy lookup and ratio persistence in StrategyABC
    to ensure domain strategy tests run purely offline without live database connections.
    """
    def _mock_get_by_acronym(acronym: str, conn: Any = None) -> Strategies:
        if acronym == RSI_2B:
            return Strategies(id=1, acronym=RSI_2B, name=STRATEGY_NAME_RSI_2B)
        if acronym == RSI_RC:
            return Strategies(id=2, acronym=RSI_RC, name=STRATEGY_NAME_RSI_RC)
        if acronym == SMA:
            return Strategies(id=3, acronym=SMA, name=STRATEGY_NAME_SMA)
        if acronym == RSI_SMA:
            return Strategies(id=4, acronym=RSI_SMA, name=STRATEGY_NAME_RSI_SMA)
        return Strategies(id=99, acronym=acronym, name=acronym)

    mock_crud_instance_ = MagicMock()
    mock_crud_instance_.get_by_acronym.side_effect = _mock_get_by_acronym

    mock_repo_instance_ = MagicMock()
    mock_repo_instance_.flag_in_process.return_value = 0
    mock_repo_instance_.persist_and_cleanup.return_value = 0

    with (
        patch('radar_core.domain.strategies.base_strategy.StrategyCrud', return_value=mock_crud_instance_),
        patch('radar_core.domain.strategies.base_strategy.RatioRepository', return_value=mock_repo_instance_),
    ):
        yield


@pytest.fixture(scope='session')
def live_spy_prices() -> pl.DataFrame:
    """
    Session-scoped fixture that downloads real historical price data for SPY from Yahoo Finance
    and prepares it with BarNumber and shared RSI indicator.
    Used by unversioned parity tests.

    :return: Polars DataFrame containing historical prices for SPY.
    """
    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value={SPY: SPY}):
        prices_data_ = PriceProvider(long_term=False).get_prices([SPY], datetime.now(timezone.utc))

    assert SPY in prices_data_, f'Failed to retrieve real {SPY} prices from Yahoo Finance.'
    prices_df_ = prices_data_[SPY]

    prices_df_ = prices_df_.with_columns(
        pl.arange(0, pl.len(), eager=False).cast(pl.Int32).alias('BarNumber')
    )
    return RSI(prices_df_)


@pytest.fixture
def real_spy_prices(live_spy_prices: pl.DataFrame) -> pl.DataFrame:
    """
    Function-scoped fixture providing a deep copy clone of the live SPY dataset.

    :param live_spy_prices: The session-scoped live SPY prices DataFrame.
    :return: An isolated cloned Polars DataFrame.
    """
    return live_spy_prices.clone()


@pytest.fixture(scope='session')
def _cached_frozen_spy_prices() -> pl.DataFrame:
    """
    Internal session fixture that reads the frozen SPY daily sample parquet file.
    If the fixture file does not exist, generates a synthetic oscillating price dataset.

    :return: Polars DataFrame containing frozen sample prices.
    """
    parquet_path_ = Path(__file__).parent / 'fixtures' / 'spy_daily_sample.parquet'
    if parquet_path_.is_file():
        df_ = pl.read_parquet(parquet_path_)
    else:
        # Graceful fallback: synthesize 350 bars with oscillations so moving average crosses occur
        dates_ = [date(2024, 1, 1) + timedelta(days=i_) for i_ in range(350)]
        t_ = np.linspace(0, 8 * np.pi, 350)
        close_ = 400.0 + 30.0 * np.sin(t_) + 0.1 * np.arange(350)
        open_ = close_ - 0.5
        high_ = close_ + 2.0
        low_ = close_ - 2.0
        volume_ = np.full(350, 1000000.0)
        pct_change_ = np.zeros(350)
        pct_change_[1:] = (close_[1:] - close_[:-1]) / close_[:-1] * 100.0
        bar_number_ = np.arange(350, dtype=np.int32)
        df_ = pl.DataFrame({
            'Date': dates_,
            'Open': open_,
            'High': high_,
            'Low': low_,
            'Close': close_,
            'Volume': volume_,
            'PercentChange': pct_change_,
            'BarNumber': bar_number_,
        })
    if 'Rsi' not in df_.columns:
        df_ = RSI(df_)
    return df_


@pytest.fixture
def frozen_spy_prices(_cached_frozen_spy_prices: pl.DataFrame) -> pl.DataFrame:
    """
    Function-scoped fixture providing an isolated deep copy clone of the frozen SPY dataset.

    :param _cached_frozen_spy_prices: The session-scoped frozen SPY DataFrame.
    :return: An isolated cloned Polars DataFrame.
    """
    return _cached_frozen_spy_prices.clone()


def make_sample_ohlcv(
    rows: int = 30,
    with_indicators: bool = False,
    start_date: date = date(2025, 1, 1),
    base_price: float = 100.0,
) -> pl.DataFrame:
    """
    Factory function creating a synthetic OHLCV Polars DataFrame for unit testing.
    Always returns a fresh, independent DataFrame.

    :param rows: Number of price bars to generate.
    :param with_indicators: Whether to append mock indicator columns (Rsi, Mogalef, StopLoss bars).
    :param start_date: Starting date for the time series.
    :param base_price: Base price level for the series.
    :return: An isolated Polars DataFrame with standard price and indicator columns.
    """
    dates_ = [start_date + timedelta(days=i_) for i_ in range(rows)]
    close_ = np.linspace(base_price, base_price + rows * 0.5, rows)
    open_ = close_ - 0.5
    high_ = close_ + 1.0
    low_ = close_ - 1.0
    volume_ = np.full(rows, 1000.0)

    pct_change_ = np.zeros(rows)
    if rows > 1:
        pct_change_[1:] = (close_[1:] - close_[:-1]) / close_[:-1] * 100

    bar_number_ = np.arange(rows, dtype=np.int32)

    data_ = {
        'Date': dates_,
        'Open': open_,
        'High': high_,
        'Low': low_,
        'Close': close_,
        'Volume': volume_,
        'PercentChange': pct_change_,
        'BarNumber': bar_number_,
    }

    if with_indicators:
        data_['Rsi'] = np.full(rows, 50.0)
        data_['MogalefUpper'] = np.full(rows, 110.0)
        data_['MogalefLower'] = np.full(rows, 90.0)
        data_['BarNumberForLongStop'] = np.full(rows, rows, dtype=np.int32)
        data_['BarNumberForShortStop'] = np.full(rows, rows, dtype=np.int32)

    return pl.DataFrame(data_)


@pytest.fixture
def ohlcv_factory() -> Callable[..., pl.DataFrame]:
    """
    Fixture returning the make_sample_ohlcv factory function.

    :return: Callable make_sample_ohlcv function.
    """
    return make_sample_ohlcv


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """
    Function-scoped fixture providing a synthetic 30-bar OHLCV Polars DataFrame.

    :return: Isolated cloned Polars DataFrame.
    """
    return make_sample_ohlcv(rows=30)


@pytest.fixture
def sample_ohlcv_with_indicators_df() -> pl.DataFrame:
    """
    Function-scoped fixture providing a synthetic 30-bar OHLCV Polars DataFrame
    with mock RSI, Mogalef bands, and stop loss columns.

    :return: Isolated cloned Polars DataFrame.
    """
    return make_sample_ohlcv(rows=30, with_indicators=True)


def create_sample_ratio(
    symbol: str = BTC_USD,
    strategy_id: int = 1,
    inputs: str = '{"period": 10}',
    timeframe: int = 2,
    is_long_position: bool = True,
    net_profit: float = 0.15,
    expected_percentage: float = 0.05,
    last_output_date: date | None = date(2025, 12, 31),
) -> Ratios:
    """
    Helper function to build a sample Ratios instance for testing with sensible defaults.

    :param symbol: Security symbol.
    :param strategy_id: Strategy database identifier.
    :param inputs: JSON serialized input parameters.
    :param timeframe: Evaluation timeframe enum.
    :param is_long_position: Long vs Short trade direction flag.
    :param net_profit: Net profit ratio value.
    :param expected_percentage: Expectation ratio value.
    :param last_output_date: Date of the last output trade bar.
    :return: Populated Ratios dataclass instance.
    """
    return Ratios(
        symbol=symbol,
        strategy_id=strategy_id,
        timeframe=timeframe,
        inputs=inputs,
        is_long_position=is_long_position,
        is_in_process=False,
        from_date=date(2025, 1, 1),
        to_date=date(2025, 12, 31),
        initial_price=100.0,
        final_price=150.0,
        net_change=0.5,
        signals=5,
        winnings=60,
        losses=10,
        net_profit=net_profit,
        expected_percentage=expected_percentage,
        win_probability=0.8,
        loss_probability=0.2,
        average_win_percentage=15.0,
        average_loss_percentage=5.0,
        total_sessions=250,
        winning_sessions=200,
        losing_sessions=50,
        percentage_exposure=0.8,
        first_input_date=date(2025, 1, 10),
        last_input_date=date(2025, 11, 1),
        last_output_date=last_output_date,
    )


@pytest.fixture
def sample_ratio() -> Ratios:
    """
    Function-scoped fixture providing a default populated Ratios dataclass instance.

    :return: Populated Ratios dataclass instance.
    """
    return create_sample_ratio()