# tests/infrastructure/test_price_cache.py

# --- Python modules ---
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

# --- Third Party Libraries ---
import polars as pl
import pytest

# --- App modules ---
from radar_core.infrastructure.price_cache import (
    CACHE_DATA_FILENAME,
    CACHE_METADATA_FILENAME,
    PriceCache,
    PriceCacheMetadata,
)


def test_price_cache_constants():
    """
    GIVEN price cache store constants
    WHEN inspected
    THEN they match expected data and metadata filenames.
    """
    assert CACHE_DATA_FILENAME == 'price_cache_data.parquet'
    assert CACHE_METADATA_FILENAME == 'price_cache_metadata.json'


def test_metadata_requires_all_attributes():
    """
    GIVEN PriceCacheMetadata class
    WHEN instantiated with all metadata attributes
    THEN every metadata attribute is stored explicitly.
    """
    metadata_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )

    assert metadata_.is_complete is True
    assert metadata_.symbol_to_ticker == {'SPY': 'SPY'}
    assert metadata_.start_date == '2020-01-01'
    assert metadata_.session_date == '2026-09-05'
    assert metadata_.generation_id == 'generation-1'
    assert metadata_.updated_at_utc == '2026-09-05T23:00:00+00:00'

    with pytest.raises(TypeError):
        PriceCacheMetadata(
            symbol_to_ticker={'SPY': 'SPY'},
            start_date='2020-01-01',
            session_date='2026-09-05',
        )


def test_metadata_json_roundtrip():
    """
    GIVEN a PriceCacheMetadata instance
    WHEN serialized to JSON and deserialized back
    THEN all attributes match the original instance.
    """
    original_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY', 'QQQ': 'QQQ'},
        start_date='2022-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    json_str_ = original_.to_json()
    assert isinstance(json_str_, str)

    parsed_dict_ = json.loads(json_str_)
    assert parsed_dict_['symbol_to_ticker'] == {'SPY': 'SPY', 'QQQ': 'QQQ'}
    assert parsed_dict_['start_date'] == '2022-01-01'

    restored_ = PriceCacheMetadata.from_json(json_str_)
    assert restored_.generation_id == original_.generation_id
    assert restored_.symbol_to_ticker == original_.symbol_to_ticker
    assert restored_.start_date == original_.start_date
    assert restored_.session_date == original_.session_date
    assert restored_.updated_at_utc == original_.updated_at_utc
    assert restored_.is_complete == original_.is_complete


def test_metadata_from_json_ignores_unknown_fields():
    """
    GIVEN a JSON string with extra unknown fields
    WHEN deserialized via from_json
    THEN it ignores the unknown fields without raising errors.
    """
    data_ = {
        'symbol_to_ticker': {'AAPL': 'AAPL'},
        'start_date': '2021-01-01',
        'session_date': '2026-09-05',
        'generation_id': 'generation-1',
        'is_complete': True,
        'updated_at_utc': '2026-09-05T23:00:00+00:00',
        'unknown_field_x': 999,
    }
    json_str_ = json.dumps(data_)
    restored_ = PriceCacheMetadata.from_json(json_str_)
    assert restored_.symbol_to_ticker == {'AAPL': 'AAPL'}
    assert not hasattr(restored_, 'unknown_field_x')


def test_is_compatible_success():
    """
    GIVEN a complete metadata snapshot
    WHEN is_compatible is called with matching mapping, start_date, and session_date
    THEN it returns (True, 'Compatible').
    """
    metadata_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    is_compat_, reason_ = metadata_.is_compatible(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
    )
    assert is_compat_ is True
    assert reason_ == 'Compatible'


def test_is_compatible_rejects_incomplete_cache():
    """
    GIVEN a metadata entry marked as is_complete=False
    WHEN is_compatible is evaluated
    THEN it returns False with reason 'Cache is incomplete'.
    """
    metadata_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=False,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    is_compat_, reason_ = metadata_.is_compatible(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
    )
    assert is_compat_ is False
    assert reason_ == 'Cache is incomplete'


def test_is_compatible_rejects_start_date_mismatch():
    """
    GIVEN a metadata entry with start_date '2020-01-01'
    WHEN requested with start_date '2021-01-01'
    THEN it returns False with start date mismatch reason.
    """
    metadata_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    is_compat_, reason_ = metadata_.is_compatible(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2021-01-01',
        session_date='2026-09-05',
    )
    assert is_compat_ is False
    assert 'Start date mismatch' in reason_


def test_is_compatible_rejects_mapping_mismatch():
    """
    GIVEN a metadata entry with symbol_to_ticker {'SPY': 'SPY'}
    WHEN requested with different symbol mapping {'SPY': 'SPY', 'QQQ': 'QQQ'}
    THEN it returns False with mapping mismatch reason.
    """
    metadata_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    is_compat_, reason_ = metadata_.is_compatible(
        symbol_to_ticker={'SPY': 'SPY', 'QQQ': 'QQQ'},
        start_date='2020-01-01',
        session_date='2026-09-05',
    )
    assert is_compat_ is False
    assert 'Symbol or ticker mapping mismatch' in reason_


def test_is_compatible_accepts_subset_mapping():
    """
    GIVEN a metadata entry with multiple cached symbols {'SPY': 'SPY', 'QQQ': 'QQQ'}
    WHEN requested with a subset of cached symbols {'SPY': 'SPY'}
    THEN it returns True with reason 'Compatible'.
    """
    metadata_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY', 'QQQ': 'QQQ'},
        start_date='2020-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    is_compat_, reason_ = metadata_.is_compatible(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
    )
    assert is_compat_ is True
    assert reason_ == 'Compatible'


def test_is_compatible_rejects_session_date_mismatch():
    """
    GIVEN a metadata entry from session_date '2026-09-04'
    WHEN requested with current session_date '2026-09-05'
    THEN it returns False with session date mismatch reason.
    """
    metadata_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-04',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    is_compat_, reason_ = metadata_.is_compatible(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-09-05',
    )
    assert is_compat_ is False
    assert 'Session date mismatch' in reason_


def test_timezone_resolution_new_york():
    """
    GIVEN the IANA timezone identifier 'America/New_York'
    WHEN resolved via zoneinfo.ZoneInfo
    THEN it successfully resolves without raising an exception.
    """
    tz_ = ZoneInfo('America/New_York')
    assert tz_.key == 'America/New_York'


def test_price_cache_paths(tmp_path):
    """
    GIVEN a PriceCache initialized with a cache directory
    WHEN data_path and metadata_path are accessed
    THEN they point to expected filenames within that directory.
    """
    cache_ = PriceCache(tmp_path)
    assert cache_.data_path == tmp_path / CACHE_DATA_FILENAME
    assert cache_.metadata_path == tmp_path / CACHE_METADATA_FILENAME


def test_price_cache_save_roundtrip(tmp_path):
    """
    GIVEN a Polars DataFrame and PriceCacheMetadata
    WHEN save is called
    THEN the data and metadata can be read back accurately via read_data and read_metadata.
    """
    cache_ = PriceCache(tmp_path)
    df_ = pl.DataFrame({
        'Date': ['2026-09-01', '2026-09-02'],
        'Open': [100.0, 101.0],
        'Close': [101.0, 102.0],
        'Symbol': ['SPY', 'SPY'],
    })
    meta_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2026-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )

    cache_.save(df_, meta_)
    loaded_df_ = cache_.read_data()
    loaded_meta_ = cache_.read_metadata()

    assert loaded_df_ is not None
    assert loaded_meta_ is not None
    assert loaded_df_.equals(df_)
    assert loaded_meta_.generation_id == meta_.generation_id
    assert loaded_meta_.symbol_to_ticker == meta_.symbol_to_ticker


def test_price_cache_read_metadata_missing_or_corrupt(tmp_path):
    """
    GIVEN a PriceCache instance
    WHEN metadata is missing, corrupted, or incomplete
    THEN read_metadata returns None.
    """
    cache_ = PriceCache(tmp_path)
    assert cache_.read_metadata() is None

    cache_.metadata_path.write_text('{ corrupted json ', encoding='utf-8')
    assert cache_.read_metadata() is None

    incomplete_meta_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2026-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=False,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    cache_.metadata_path.write_text(incomplete_meta_.to_json(), encoding='utf-8')
    assert cache_.read_metadata() is None


def test_price_cache_read_data_corrupt_file(tmp_path):
    """
    GIVEN a cache directory with a corrupted parquet data file
    WHEN read_data is called
    THEN it returns None cleanly.
    """
    cache_ = PriceCache(tmp_path)
    cache_.data_path.write_bytes(b'not a parquet file')
    assert cache_.read_data() is None


def test_price_cache_save_atomic_failure_safety(tmp_path):
    """
    GIVEN an existing valid cache
    WHEN a subsequent save operation fails
    THEN the prior cache remains readable and temporary staging files are cleaned up.
    """
    cache_ = PriceCache(tmp_path)
    df_ = pl.DataFrame({'A': [1, 2]})
    meta_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2026-01-01',
        session_date='2026-09-05',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )
    cache_.save(df_, meta_)

    failing_meta_ = PriceCacheMetadata(
        symbol_to_ticker={'QQQ': 'QQQ'},
        start_date='2026-01-01',
        session_date='2026-09-05',
        generation_id='generation-2',
        is_complete=True,
        updated_at_utc='2026-09-05T23:00:00+00:00',
    )

    with patch.object(pl.DataFrame, 'write_parquet', side_effect=IOError('Disk full')), \
         pytest.raises(IOError, match='Disk full'):
        cache_.save(df_, failing_meta_)

    loaded_df_ = cache_.read_data()
    loaded_meta_ = cache_.read_metadata()
    assert loaded_df_ is not None
    assert loaded_meta_ is not None
    assert loaded_meta_.symbol_to_ticker == {'SPY': 'SPY'}
    assert list(tmp_path.glob('.tmp_*')) == []


def test_price_cache_metadata_age_in_minutes():
    """
    GIVEN a metadata entry with updated_at_utc
    WHEN age_in_minutes is called with a timezone-aware datetime
    THEN it returns the elapsed time in minutes accurately.
    """
    now_utc_ = datetime(2026, 1, 9, 12, 0, tzinfo=timezone.utc)
    meta_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc=(now_utc_ - timedelta(minutes=7, seconds=30)).isoformat(),
    )
    age_ = meta_.age_in_minutes(now_utc_)
    assert abs(age_ - 7.5) < 1e-4


def test_price_cache_read_data_success(tmp_path):
    """
    GIVEN a saved Parquet data file
    WHEN read_data is called
    THEN it reads and returns the DataFrame without reading metadata.
    """
    cache_ = PriceCache(tmp_path)
    df_ = pl.DataFrame({'A': [1, 2, 3]})
    meta_ = PriceCacheMetadata(
        symbol_to_ticker={'SPY': 'SPY'},
        start_date='2020-01-01',
        session_date='2026-01-09',
        generation_id='generation-1',
        is_complete=True,
        updated_at_utc='2026-01-09T12:00:00+00:00',
    )
    cache_.save(df_, meta_)
    read_df_ = cache_.read_data()
    assert read_df_ is not None
    assert read_df_.shape == (3, 1)


def test_price_cache_read_data_missing_file(tmp_path):
    """
    GIVEN a cache directory without data file
    WHEN read_data is called
    THEN it returns None cleanly.
    """
    cache_ = PriceCache(tmp_path)
    assert cache_.read_data() is None


