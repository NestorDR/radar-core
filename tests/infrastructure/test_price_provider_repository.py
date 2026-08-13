# tests/infrastructure/test_price_provider_repository.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- App modules ---
from radar_core.infrastructure.price_provider import PriceProvider
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
        mock_add_.assert_called_once()


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


def test_price_provider_empty_tickers_guard():
    """
    GIVEN a list of symbols that maps to 0 tickers
    WHEN PriceProvider.get_prices is called
    THEN it returns an empty dictionary cleanly without calling yfinance download.
    """
    provider_ = PriceProvider()

    with patch.object(SecurityRepository, 'map_symbol_to_ticker', return_value={}), \
         patch('yfinance.download') as mock_download_:

        result_ = provider_.get_prices(['INVALID_SYMBOL'])

        assert result_ == {}
        mock_download_.assert_not_called()
