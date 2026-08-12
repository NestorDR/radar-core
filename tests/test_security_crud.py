# tests/test_security_crud.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- App modules ---
from radar_core.infrastructure.crud.security_crud import SecurityCrud
from radar_core.models import Securities


def test_security_crud_get_by_symbol_existing():
    """
    GIVEN an existing security symbol 'SPX'
    WHEN get_by_symbol is called with mocked database connection
    THEN it maps and returns the corresponding Securities record.
    """
    crud_ = SecurityCrud()
    mock_row_ = (1, 'SPX', 'S&P 500 Index', False, True)

    with patch('radar_core.infrastructure.crud.security_crud.get_psycopg_connection') as mock_conn_func_:
        mock_cur_ = MagicMock()
        mock_cur_.fetchone.return_value = mock_row_
        mock_conn_func_.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur_

        security_ = crud_.get_by_symbol('SPX')
        assert security_ is not None
        assert security_.id == 1
        assert security_.symbol == 'SPX'
        assert security_.description == 'S&P 500 Index'


def test_security_crud_get_by_symbol_nonexistent():
    """
    GIVEN a non-existent symbol 'UNKNOWN'
    WHEN get_by_symbol is called with mocked database connection
    THEN it returns None.
    """
    crud_ = SecurityCrud()

    with patch('radar_core.infrastructure.crud.security_crud.get_psycopg_connection') as mock_conn_func_:
        mock_cur_ = MagicMock()
        mock_cur_.fetchone.return_value = None
        mock_conn_func_.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur_

        security_ = crud_.get_by_symbol('UNKNOWN_SYMBOL')
        assert security_ is None


def test_security_crud_add_security():
    """
    GIVEN a new Securities instance
    WHEN add_security is called with mocked database connection returning an assigned id
    THEN the instance id attribute is populated correctly.
    """
    crud_ = SecurityCrud()
    security_to_add_ = Securities(symbol='TEST-STOCK', description='Test Stock Description')

    with patch('radar_core.infrastructure.crud.security_crud.get_psycopg_connection') as mock_conn_func_:
        mock_cur_ = MagicMock()
        mock_cur_.fetchone.return_value = (42,)
        mock_conn_func_.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur_

        crud_.add_security(security_to_add_)
        assert security_to_add_.id == 42


def test_security_crud_get_tickers_by_symbols_preserves_order():
    """
    GIVEN a list of symbols in a specific order
    WHEN get_tickers_by_symbols is called
    THEN the returned dictionary keys match the input list order.
    """
    crud_ = SecurityCrud()
    symbols_ = ['MSFT', 'AAPL', 'GOOGL']
    mock_rows_ = [('GOOGL', 'GOOGL'), ('AAPL', 'AAPL'), ('MSFT', 'MSFT')]

    with patch('radar_core.infrastructure.crud.security_crud.get_psycopg_connection') as mock_conn_func_:
        mock_cur_ = MagicMock()
        mock_cur_.fetchall.return_value = mock_rows_
        mock_conn_func_.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur_

        result_ = crud_.get_tickers_by_symbols(symbols_, provider_id=1)
        assert list(result_.keys()) == symbols_
