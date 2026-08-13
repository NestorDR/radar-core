# tests/infrastructure/test_ratio_repository.py

from unittest.mock import MagicMock, patch

import pytest

from radar_core.infrastructure.ratio_repository import RatioRepository


def _mock_repository_dependencies():
    repository_connection_ = MagicMock()
    connection_scope_ = MagicMock()
    connection_scope_.__enter__.return_value = repository_connection_

    crud_ = MagicMock()

    return repository_connection_, connection_scope_, crud_


def test_flag_in_process_uses_one_write_connection():
    """
    GIVEN a ratio repository and an existing database connection scope
    WHEN flag_in_process is called
    THEN the CRUD operation receives the scoped connection and returns its row count.
    """
    repository_connection_, connection_scope_, crud_ = _mock_repository_dependencies()
    crud_.flag_in_process.return_value = 3

    with (
        patch('radar_core.infrastructure.ratio_repository.RatioCrud', return_value=crud_),
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


def test_persist_and_cleanup_uses_one_connection_for_upsert_and_delete():
    """
    GIVEN positive ratios and a ratio repository
    WHEN persist_and_cleanup is called
    THEN upsert and cleanup use the same connection and the upsert count is returned.
    """
    repository_connection_, connection_scope_, crud_ = _mock_repository_dependencies()
    crud_.upsert_many.return_value = 4

    positive_ratios_ = [MagicMock()]

    with (
        patch('radar_core.infrastructure.ratio_repository.RatioCrud', return_value=crud_),
        patch(
            'radar_core.infrastructure.ratio_repository.get_psycopg_connection',
            return_value=connection_scope_
        ),
    ):
        repository_ = RatioRepository()

        result_ = repository_.persist_and_cleanup(
            positive_ratios_,
            'SPY',
            1,
            2
        )

    assert result_ == 4
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


def test_persist_and_cleanup_deletes_flags_when_no_positive_ratios_exist():
    """
    GIVEN no positive ratios
    WHEN persist_and_cleanup is called
    THEN upsert is skipped, cleanup is executed, and zero is returned.
    """
    repository_connection_, connection_scope_, crud_ = _mock_repository_dependencies()

    with (
        patch('radar_core.infrastructure.ratio_repository.RatioCrud', return_value=crud_),
        patch(
            'radar_core.infrastructure.ratio_repository.get_psycopg_connection',
            return_value=connection_scope_
        ),
    ):
        repository_ = RatioRepository()

        result_ = repository_.persist_and_cleanup([], 'SPY', 1, 2)

    assert result_ == 0
    crud_.upsert_many.assert_not_called()
    crud_.delete_flagged_in_process.assert_called_once_with(
        'SPY',
        1,
        2,
        conn=repository_connection_
    )


def test_persist_and_cleanup_does_not_cleanup_when_upsert_fails():
    """
    GIVEN an upsert failure
    WHEN persist_and_cleanup is called
    THEN the failure is propagated and cleanup is not executed.
    """
    repository_connection_, connection_scope_, crud_ = _mock_repository_dependencies()
    failure_ = RuntimeError('upsert failed')
    crud_.upsert_many.side_effect = failure_

    with (
        patch('radar_core.infrastructure.ratio_repository.RatioCrud', return_value=crud_),
        patch(
            'radar_core.infrastructure.ratio_repository.get_psycopg_connection',
            return_value=connection_scope_
        ),
    ):
        repository_ = RatioRepository()

        with pytest.raises(RuntimeError, match='upsert failed'):
            repository_.persist_and_cleanup([MagicMock()], 'SPY', 1, 2)

    crud_.delete_flagged_in_process.assert_not_called()