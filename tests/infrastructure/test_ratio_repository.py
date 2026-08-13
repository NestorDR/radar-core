# tests/infrastructure/test_ratio_repository.py

# --- Python modules ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from radar_core.infrastructure.ratio_repository import RatioRepository


def test_remove_unlisted_symbols_delegates_to_crud():
    """
    GIVEN a ratio repository
    WHEN remove_unlisted_symbols is called
    THEN the request is delegated and its row count is returned.
    """
    crud_ = MagicMock()
    crud_.remove_unlisted_symbols.return_value = 6

    with patch(
            'radar_core.infrastructure.ratio_repository.RatioCrud',
            return_value=crud_
    ):
        repository_ = RatioRepository()

        result_ = repository_.remove_unlisted_symbols(['SPY', 'NDQ'])

    assert result_ == 6
    crud_.remove_unlisted_symbols.assert_called_once_with(['SPY', 'NDQ'])


def test_remove_unlisted_symbols_passes_empty_symbols_to_crud():
    """
    GIVEN an empty symbol list
    WHEN repository cleanup is requested
    THEN the empty list is passed unchanged to the CRUD layer.
    """
    crud_ = MagicMock()
    crud_.remove_unlisted_symbols.return_value = 0

    with patch(
            'radar_core.infrastructure.ratio_repository.RatioCrud',
            return_value=crud_
    ):
        repository_ = RatioRepository()

        result_ = repository_.remove_unlisted_symbols([])

    assert result_ == 0
    crud_.remove_unlisted_symbols.assert_called_once_with([])


def test_flag_in_process_uses_one_write_connection(mock_connection_scope):
    """
    GIVEN a ratio repository and an operation-scoped connection
    WHEN flag_in_process is called
    THEN the CRUD operation receives the scoped connection and returns its row count.
    """
    repository_connection_, _, connection_scope_ = mock_connection_scope
    crud_ = MagicMock()
    crud_.flag_in_process.return_value = 3

    with (
        patch(
            'radar_core.infrastructure.ratio_repository.RatioCrud',
            return_value=crud_
        ),
        patch(
            'radar_core.infrastructure.ratio_repository.get_psycopg_connection',
            return_value=connection_scope_
        ) as connection_factory_,
    ):
        repository_ = RatioRepository()

        result_ = repository_.flag_in_process('SPY', 1, 2)

    assert result_ == 3
    connection_factory_.assert_called_once_with()
    crud_.flag_in_process.assert_called_once_with(
        'SPY',
        1,
        2,
        conn=repository_connection_
    )
    connection_scope_.__exit__.assert_called_once_with(None, None, None)


def test_persist_and_cleanup_deletes_flags_when_no_positive_ratios_exist(mock_connection_scope):
    """
    GIVEN no positive ratios
    WHEN persist_and_cleanup is called
    THEN upsert is skipped, cleanup is executed, and zero is returned.
    """
    repository_connection_, _, connection_scope_ = mock_connection_scope
    crud_ = MagicMock()

    with (
        patch(
            'radar_core.infrastructure.ratio_repository.RatioCrud',
            return_value=crud_
        ),
        patch(
            'radar_core.infrastructure.ratio_repository.get_psycopg_connection',
            return_value=connection_scope_
        ) as connection_factory_,
    ):
        repository_ = RatioRepository()

        result_ = repository_.persist_and_cleanup([], 'SPY', 1, 2)

    assert result_ == 0
    connection_factory_.assert_called_once_with()
    crud_.upsert_many.assert_not_called()
    crud_.delete_flagged_in_process.assert_called_once_with(
        'SPY',
        1,
        2,
        conn=repository_connection_
    )
    connection_scope_.__exit__.assert_called_once_with(None, None, None)


def test_persist_and_cleanup_does_not_cleanup_when_upsert_fails(
        mock_connection_scope
):
    """
    GIVEN an upsert failure
    WHEN persist_and_cleanup is called
    THEN the failure is propagated and cleanup is not executed.
    """
    repository_connection_, _, connection_scope_ = mock_connection_scope
    crud_ = MagicMock()
    positive_ratios_ = [MagicMock()]
    failure_ = RuntimeError('upsert failed')
    crud_.upsert_many.side_effect = failure_

    with (
        patch(
            'radar_core.infrastructure.ratio_repository.RatioCrud',
            return_value=crud_
        ),
        patch(
            'radar_core.infrastructure.ratio_repository.get_psycopg_connection',
            return_value=connection_scope_
        ),
    ):
        repository_ = RatioRepository()

        with pytest.raises(RuntimeError, match='upsert failed'):
            repository_.persist_and_cleanup(
                positive_ratios_,
                'SPY',
                1,
                2
            )

    crud_.delete_flagged_in_process.assert_not_called()
    connection_scope_.__exit__.assert_called_once()
    assert connection_scope_.__exit__.call_args.args[0] is RuntimeError


def test_persist_and_cleanup_propagates_cleanup_failure(
        mock_connection_scope
):
    """
    GIVEN a successful upsert and a cleanup failure
    WHEN persist_and_cleanup is called
    THEN the cleanup failure is propagated.
    """
    repository_connection_, _, connection_scope_ = mock_connection_scope
    crud_ = MagicMock()
    positive_ratios_ = [MagicMock()]
    cleanup_failure_ = RuntimeError('cleanup failed')

    crud_.upsert_many.return_value = 3
    crud_.delete_flagged_in_process.side_effect = cleanup_failure_

    with (
        patch(
            'radar_core.infrastructure.ratio_repository.RatioCrud',
            return_value=crud_
        ),
        patch(
            'radar_core.infrastructure.ratio_repository.get_psycopg_connection',
            return_value=connection_scope_
        ),
    ):
        repository_ = RatioRepository()

        with pytest.raises(RuntimeError, match='cleanup failed'):
            repository_.persist_and_cleanup(
                positive_ratios_,
                'SPY',
                1,
                2
            )

    crud_.upsert_many.assert_called_once_with(
        positive_ratios_,
        conn=repository_connection_
    )
    crud_.delete_flagged_in_process.assert_called_once_with(
        'SPY',
        1,
        2,
        conn=repository_connection_
    )
    connection_scope_.__exit__.assert_called_once()
    assert connection_scope_.__exit__.call_args.args[0] is RuntimeError
