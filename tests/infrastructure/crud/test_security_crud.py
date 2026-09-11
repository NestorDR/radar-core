# tests/infrastructure/crud/test_security_crud.py

# --- Python modules ---
from unittest.mock import call, patch

# --- App modules ---
from radar_core.infrastructure.crud import SecurityCrud
from radar_core.models import Securities


def test_security_crud_get_by_symbol_existing(mock_connection_scope):
    """
    GIVEN an existing security symbol
    WHEN get_by_symbol is called
    THEN it returns the mapped security.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = (
        1,
        'SPX',
        'S&P 500 Index',
        False,
        False,
        False,
        False,
        True
    )

    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope',
            return_value=scope_
    ) as read_scope_:
        security_ = SecurityCrud().get_by_symbol('SPX')

    assert security_ is not None
    assert security_.id == 1
    assert security_.symbol == 'SPX'
    assert security_.description == 'S&P 500 Index'
    assert security_.is_bear is False
    assert security_.is_shortable is False
    assert security_.is_crypto is False
    assert security_.is_near_continuous is False
    assert security_.store_locally is True
    read_scope_.assert_called_once_with(None)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_security_crud_get_by_symbol_nonexistent(mock_connection_scope):
    """
    GIVEN a non-existent security symbol
    WHEN get_by_symbol is called
    THEN it returns None without performing a synonym lookup.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = None

    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope',
            return_value=scope_
    ):
        security_ = SecurityCrud().get_by_symbol('UNKNOWN_SYMBOL')

    assert security_ is None
    cursor_.execute.assert_called_once()
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_security_crud_get_by_symbol_reuses_connection_for_synonym(mock_connection_scope):
    """
    GIVEN an existing security and provider identifier
    WHEN get_by_symbol is called
    THEN the security and synonym queries use one read connection.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.side_effect = [
        (1, 'SPX', 'S&P 500 Index', False, False, False, False, True),
        (10, 1, 1, '^GSPC'),
    ]

    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope',
            return_value=scope_
    ) as read_scope_:
        security_ = SecurityCrud().get_by_symbol(
            'SPX',
            provider_id=1
        )

    assert security_ is not None
    assert len(security_.synonyms) == 1
    assert security_.synonyms[0].id == 10
    assert security_.synonyms[0].provider_id == 1
    assert security_.synonyms[0].security_id == 1
    assert security_.synonyms[0].ticker == '^GSPC'
    read_scope_.assert_has_calls([
        call(None),
        call(connection_)
    ])
    assert read_scope_.call_count == 2
    assert cursor_.execute.call_count == 2
    assert scope_.__exit__.call_count == 2

    # Both lookups reuse the same connection and cursor.
    assert connection_.cursor.call_count == 2
    assert connection_.cursor.return_value.__enter__.return_value is cursor_


def test_security_crud_get_synonym_existing(mock_connection_scope):
    """
    GIVEN an existing security synonym
    WHEN get_synonym is called
    THEN it returns the mapped synonym.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = (10, 1, 1, '^GSPC')

    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope',
            return_value=scope_
    ) as read_scope_:
        synonym_ = SecurityCrud.get_synonym(1, 1)

    assert synonym_ is not None
    assert synonym_.id == 10
    assert synonym_.provider_id == 1
    assert synonym_.security_id == 1
    assert synonym_.ticker == '^GSPC'
    read_scope_.assert_called_once_with(None)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_security_crud_get_synonym_nonexistent(mock_connection_scope):
    """
    GIVEN no synonym for a security and provider
    WHEN get_synonym is called
    THEN it returns None.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = None

    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope',
            return_value=scope_
    ):
        synonym_ = SecurityCrud.get_synonym(1, 1)

    assert synonym_ is None
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_security_crud_get_tickers_by_symbols_preserves_order(
        mock_connection_scope
):
    """
    GIVEN symbols returned by the database in a different order
    WHEN get_tickers_by_symbols is called
    THEN the result follows the input symbol order.
    """
    _, cursor_, scope_ = mock_connection_scope
    symbols_ = ['MSFT', 'AAPL', 'GOOGL']
    cursor_.fetchall.return_value = [
        ('GOOGL', 'GOOGL'),
        ('AAPL', 'AAPL'),
        ('MSFT', 'MSFT')
    ]

    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope',
            return_value=scope_
    ) as read_scope_:
        result_ = SecurityCrud.get_tickers_by_symbols(
            symbols_,
            provider_id=1
        )

    assert result_ == {
        'MSFT': 'MSFT',
        'AAPL': 'AAPL',
        'GOOGL': 'GOOGL'
    }
    read_scope_.assert_called_once_with(None)
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_security_crud_get_tickers_by_symbols_empty_does_not_open_connection():
    """
    GIVEN an empty symbol list
    WHEN get_tickers_by_symbols is called
    THEN it returns an empty mapping without opening a read connection.
    """
    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope'
    ) as read_scope_:
        result_ = SecurityCrud.get_tickers_by_symbols([], provider_id=1)

    assert result_ == {}
    read_scope_.assert_not_called()


def test_security_crud_add_security(mock_connection_scope):
    """
    GIVEN a new security
    WHEN add_security is called
    THEN it assigns the database-generated identifier to the model.
    """
    connection_, cursor_, scope_ = mock_connection_scope
    cursor_.fetchone.return_value = (42,)
    security_ = Securities(
        symbol='TEST-STOCK',
        description='Test Stock Description'
    )

    with patch(
            'radar_core.infrastructure.crud.security_crud.connection_scope',
            return_value=scope_
    ) as write_scope_:
        SecurityCrud.add_security(security_)

    assert security_.id == 42
    write_scope_.assert_called_once_with(None)
    assert cursor_.execute.call_args.args[1] == (
        'TEST-STOCK',
        'Test Stock Description',
        False,
        False,
        False,
        False,
        False
    )
    scope_.__exit__.assert_called_once_with(None, None, None)


def test_security_crud_get_shortable_symbols_empty():
    """
    GIVEN an empty list of symbols
    WHEN get_shortable_symbols is called
    THEN it returns an empty set without opening a read connection.
    """
    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope'
    ) as read_scope_:
        result_ = SecurityCrud.get_shortable_symbols([])

    assert result_ == set()
    read_scope_.assert_not_called()


def test_security_crud_get_shortable_symbols_filters_correctly(mock_connection_scope):
    """
    GIVEN a list of symbols
    WHEN get_shortable_symbols is called
    THEN it returns a set of symbols where is_shortable is True.
    """
    _, cursor_, scope_ = mock_connection_scope
    cursor_.fetchall.return_value = [('SPY',), ('QQQ',)]

    with patch(
            'radar_core.infrastructure.crud.security_crud.read_connection_scope',
            return_value=scope_
    ) as read_scope_:
        result_ = SecurityCrud.get_shortable_symbols(['SPY', 'UNKNOWN', 'QQQ'])

    assert result_ == {'SPY', 'QQQ'}
    read_scope_.assert_called_once_with(None)
    cursor_.execute.assert_called_once()
    scope_.__exit__.assert_called_once_with(None, None, None)
