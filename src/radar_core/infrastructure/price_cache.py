# src/radar_core/infrastructure/price_cache.py

# --- Python modules ---
# json: provides functions for working with JSON data.
import json
# dataclasses: provides support for defining data-oriented classes.
from dataclasses import asdict, dataclass, fields
# datetime: provides classes for manipulating dates and times.
from datetime import datetime, timezone
# logging: provides flexible event logging.
from logging import getLogger
# pathlib: provides an interface to work with file paths in a more readable and easier way than the older 'os.path'.
from pathlib import Path
# typing: provides runtime support for type hints
from typing import Final

# --- Third Party Libraries ---
# polars: high-performance DataFrame library for in-memory analytics.
import polars as pl

logger_ = getLogger(__name__)

CACHE_DATA_FILENAME: Final[str] = 'price_cache_data.parquet'
CACHE_METADATA_FILENAME: Final[str] = 'price_cache_metadata.json'


@dataclass(kw_only=True)
class PriceCacheMetadata:
    """Metadata associated with cached price data."""

    symbol_to_ticker: dict[str, str]
    start_date: str
    session_date: str
    generation_id: str
    is_complete: bool
    updated_at_utc: str

    def to_json(self) -> str:
        """
        Serializes metadata to a formatted JSON string.

        :return: JSON formatted string.
        """
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> 'PriceCacheMetadata':
        """
        Deserializes metadata from a JSON string.

        :param json_str: JSON string containing metadata.

        :return: PriceCacheMetadata instance.
        """
        data_ = json.loads(json_str)
        field_names_ = {f_.name for f_ in fields(cls)}
        return cls(**{k_: v_ for k_, v_ in data_.items() if k_ in field_names_})

    def is_compatible(
        self,
        symbol_to_ticker: dict[str, str],
        start_date: str,
        session_date: str,
    ) -> tuple[bool, str]:
        """
        Evaluates whether this cached data is compatible with request criteria.

        :param symbol_to_ticker: Expected symbol-to-ticker mapping.
        :param start_date: Expected historical start date (YYYY-MM-DD).
        :param session_date: Expected local session date (YYYY-MM-DD).

        :return: Tuple of (is_compatible, reason).
        """
        if not self.is_complete:
            return False, 'Cache is incomplete'
        if self.start_date != start_date:
            return False, f'Start date mismatch ({self.start_date} != {start_date})'
        if self.symbol_to_ticker != symbol_to_ticker:
            return False, 'Symbol or ticker mapping mismatch'
        if self.session_date != session_date:
            return False, f'Session date mismatch ({self.session_date} != {session_date})'

        return True, 'Compatible'

    def age_in_minutes(self, now: datetime) -> float:
        """
        Calculates cache age in minutes relative to the given timezone-aware datetime.

        :param now: Current timezone-aware datetime.

        :return: Cache age in minutes.
        """
        updated_dt_ = datetime.fromisoformat(self.updated_at_utc)
        if updated_dt_.tzinfo is None:
            updated_dt_ = updated_dt_.replace(tzinfo=timezone.utc)
        now_utc_ = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return (now_utc_ - updated_dt_).total_seconds() / 60.0


class PriceCache:
    """
    Manages local filesystem persistence and retrieval of OHLCV Parquet data
    and accompanying metadata.
    """

    def __init__(self, cache_dir: Path | str):
        """
        Initializes the cache store for the given directory path.

        :param cache_dir: Filesystem path to the cache directory.
        """
        self.cache_dir = Path(cache_dir)
        self.data_path = self.cache_dir / CACHE_DATA_FILENAME
        self.metadata_path = self.cache_dir / CACHE_METADATA_FILENAME

    def read_metadata(self) -> PriceCacheMetadata | None:
        """
        Reads and parses cache metadata from the disk.

        :return: PriceCacheMetadata if valid and complete, else None.
        """
        if not self.metadata_path.is_file():
            return None

        try:
            json_str_ = self.metadata_path.read_text(encoding='utf-8')
            metadata_ = PriceCacheMetadata.from_json(json_str_)
            if not metadata_.is_complete:
                logger_.warning('Cache metadata at %s is incomplete; treating as cache miss', self.metadata_path)
                return None
            return metadata_
            
        except Exception as e_:
            logger_.exception('Failed to read or parse cache metadata at %s: %s', self.metadata_path, e_)
            return None

    def read_data(self) -> pl.DataFrame | None:
        """
        Reads and returns the cached Parquet DataFrame from disk.

        :return: Polars DataFrame if file exists and reads cleanly, else None.
        """
        if not self.data_path.is_file():
            logger_.warning('Cached Parquet data file missing at %s', self.data_path)
            return None

        try:
            return pl.read_parquet(self.data_path)
        except Exception as e_:
            logger_.exception('Failed reading cached Parquet data at %s: %s', self.data_path, e_)
            return None

    def load(self) -> tuple[pl.DataFrame, PriceCacheMetadata] | None:
        """
        Loads the cached data (DataFrame and metadata) from the disk.

        :return: Tuple of (DataFrame, PriceCacheMetadata) if valid, else None.
        """
        metadata_ = self.read_metadata()
        if metadata_ is None:
            return None

        df_ = self.read_data()
        if df_ is None:
            return None

        return df_, metadata_

    def save(self, df: pl.DataFrame, metadata: PriceCacheMetadata) -> None:
        """
        Atomically saves a dataframe and its metadata to disk via staging files.

        :param df: Polars dataframe containing OHLCV price data.
        :param metadata: Associated metadata describing this cache entry.

        :raises OSError: If the cache directory cannot be created or written to.
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e_:
            logger_.exception('Failed to create cache directory %s: %s', self.cache_dir, e_)
            raise

        temp_data_path_ = self.cache_dir / f'.tmp_{metadata.generation_id}_{CACHE_DATA_FILENAME}'
        temp_meta_path_ = self.cache_dir / f'.tmp_{metadata.generation_id}_{CACHE_METADATA_FILENAME}'

        try:
            df.write_parquet(temp_data_path_)
            temp_meta_path_.write_text(metadata.to_json(), encoding='utf-8')

            # Atomic swap: data first, then metadata acts as the commit gatekeeper
            temp_data_path_.replace(self.data_path)
            temp_meta_path_.replace(self.metadata_path)
            
        except Exception as e_:
            logger_.exception('Failed saving cache generation %s: %s', metadata.generation_id, e_)
            temp_data_path_.unlink(missing_ok=True)
            temp_meta_path_.unlink(missing_ok=True)
            raise
