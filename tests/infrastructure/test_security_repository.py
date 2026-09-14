# tests/infrastructure/test_security_repository.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---

# --- App modules ---
from radar_core.infrastructure.security_repository import SecurityRepository
from radar_core.models import Securities


def test_get_or_create_security_returns_new_security():
    """
    GIVEN a security symbol 'NVDA' not present in the DB
    WHEN _get_or_create_security is called and Yahoo Finance returns info
    THEN it creates and returns the new Securities instance.
    """
    repo_ = SecurityRepository()

    with patch.object(repo_._SecurityRepository__security_crud, 'get_by_symbol', return_value=None), \
            patch.object(repo_._SecurityRepository__security_crud, 'add_security') as mock_add_, \
            patch('yfinance.Ticker') as mock_ticker_cls_:
        mock_ticker_inst_ = MagicMock()
        mock_ticker_inst_.info = {'longName': 'NVIDIA Corporation'}
        mock_ticker_cls_.return_value = mock_ticker_inst_

        security_ = repo_._get_or_create_security('NVDA')

        assert security_ is not None
        assert security_.symbol == 'NVDA'
        assert security_.description == 'NVIDIA Corporation'
        assert security_.is_bear is False
        assert security_.is_shortable is False
        assert security_.is_crypto is False
        assert security_.is_near_continuous is False
        mock_add_.assert_called_once()


def test_get_or_create_security_derives_crypto_and_continuous():
    """
    GIVEN a cryptocurrency symbol missing from DB
    WHEN _get_or_create_security is called
    THEN it derives is_crypto and is_near_continuous as True.
    """
    repo_ = SecurityRepository()

    with patch.object(repo_._SecurityRepository__security_crud, 'get_by_symbol', return_value=None), \
            patch.object(repo_._SecurityRepository__security_crud, 'add_security') as mock_add_, \
            patch('yfinance.Ticker') as mock_ticker_cls_:
        mock_ticker_inst_ = MagicMock()
        mock_ticker_inst_.info = {'longName': 'Bitcoin USD', 'quoteType': 'CRYPTOCURRENCY'}
        mock_ticker_cls_.return_value = mock_ticker_inst_

        security_ = repo_._get_or_create_security('BTC-USD')

        assert security_ is not None
        assert security_.symbol == 'BTC-USD'
        assert security_.is_crypto is True
        assert security_.is_near_continuous is True
        assert security_.is_shortable is False
        mock_add_.assert_called_once()


def test_security_repository_get_shortable_symbols():
    """
    GIVEN a list of symbols
    WHEN get_shortable_symbols is called on SecurityRepository
    THEN it delegates to SecurityCrud.get_shortable_symbols.
    """
    repo_ = SecurityRepository()

    with patch.object(repo_._SecurityRepository__security_crud, 'get_shortable_symbols', return_value={'SPY', 'QQQ'}) as mock_get_:
        result_ = repo_.get_shortable_symbols(['SPY', 'QQQ', 'AAPL'])

    assert result_ == {'SPY', 'QQQ'}
    mock_get_.assert_called_once_with(['SPY', 'QQQ', 'AAPL'])


def test_security_repository_get_bear_symbols():
    """
    GIVEN a list of symbols
    WHEN get_bear_symbols is called on SecurityRepository
    THEN it delegates to SecurityCrud.get_bear_symbols.
    """
    repo_ = SecurityRepository()

    with patch.object(repo_._SecurityRepository__security_crud, 'get_bear_symbols', return_value={'SQQQ', 'SOXS'}) as mock_get_:
        result_ = repo_.get_bear_symbols(['SQQQ', 'SOXS', 'AAPL'])

    assert result_ == {'SQQQ', 'SOXS'}
    mock_get_.assert_called_once_with(['SQQQ', 'SOXS', 'AAPL'])


def test_map_symbol_to_ticker_auto_creates_missing_symbols():
    """
    GIVEN symbols ['SPY', 'NEW_SYM'] where 'NEW_SYM' is missing from DB
    WHEN map_symbol_to_ticker is called
    THEN missing symbols are created and mapped successfully.
    """
    repo_ = SecurityRepository()
    mock_db_map_ = {'SPY': 'SPY'}
    new_sec_ = Securities(id=2, symbol='NEW_SYM', description='New Symbol Inc')

    with patch.object(repo_._SecurityRepository__security_crud, 'get_tickers_by_symbols', return_value=mock_db_map_), \
            patch.object(repo_, '_get_or_create_security', return_value=new_sec_) as mock_create_, \
            patch.object(repo_, '_get_ticker', return_value='NEW_SYM'):
        result_ = repo_.map_symbol_to_ticker(['SPY', 'NEW_SYM'])

        assert result_ == {'SPY': 'SPY', 'NEW_SYM': 'NEW_SYM'}
        mock_create_.assert_called_once_with('NEW_SYM')


def test_map_symbol_to_ticker_omits_symbols_not_in_yahoo():
    """
    GIVEN a symbol 'INVALID_SYM' not in DB and not found on Yahoo Finance
    WHEN map_symbol_to_ticker is called
    THEN the invalid symbol is omitted from the returned mapping.
    """
    repo_ = SecurityRepository()

    with patch.object(repo_._SecurityRepository__security_crud, 'get_tickers_by_symbols', return_value={}), \
            patch.object(repo_, '_get_or_create_security', return_value=None):
        result_ = repo_.map_symbol_to_ticker(['INVALID_SYM'])

        assert result_ == {}
