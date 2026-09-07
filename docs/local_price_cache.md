# Local Price Cache Architecture and Operational Guide

`PriceProvider` (`src/radar_core/infrastructure/price_provider.py`) and `PriceCache` (`src/radar_core/infrastructure/price_cache.py`) persist normalized daily Yahoo Finance OHLCV market data locally in Parquet format with companion JSON metadata, avoiding redundant full downloads across repeated executions during market hours and development sessions.

## Motivation & Design Rationale

1. **Network Overhead & Latency Reduction**:
   - Downloading multi-decade historical OHLCV series across numerous assets on every analyzer run consumes substantial network bandwidth, adds multi-second latency, and risks upstream rate-limiting or IP throttling from Yahoo Finance.
   - Local caching eliminates up to 99% of downloaded data volume during repeat runs.

2. **Corporate Actions & Price Adjustment Timing**:
   - Crucial historical price adjustments—such as stock splits, reverse splits, dividend distributions, and spinoffs—are settled and applied by exchanges and providers *prior* to the market opening bell (overnight or pre-market).
   - Once the trading session opens (`09:30`), historical bars are static and immutable; only the current session's bar fluctuates.
   - Consequently, local cache reuse is safest and most effective once the session starts, requiring only the current day's single bar to be fetched and merged in memory.
   - Once trading closes (`16:00`), a complete post-market snapshot captures finalized daily settlements, remaining valid for the remainder of the evening until the next morning's corporate action adjustment window.

## 1. File Layout & Storage Contracts

Cache files are managed by `PriceCache` in the directory specified by `RADAR_PRICE_CACHE_DIR` (`/home/default/app/cache` by default):

- **`price_cache_data.parquet`**: Clean, normalized daily OHLCV bars (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Symbol`). Derived metrics such as `PercentChange` are excluded from disk persistence and computed on demand in Polars when loaded into memory.
- **`price_cache_metadata.json`**: Operational metadata verifying cache validity and compatibility:
  - `symbol_to_ticker`: Exact mapping of internal security symbols to Yahoo Finance tickers.
  - `start_date`: Configured historical start date (ISO format `YYYY-MM-DD`).
  - `session_date`: Market session date string (ISO format `YYYY-MM-DD`).
  - `generation_id`: Unique generation identifier (UUID4 hex).
  - `is_complete`: Boolean flag indicating whether all requested symbols were successfully downloaded and persisted.
  - `updated_at_utc`: UTC timestamp of the last successful save or refresh.

### Compatibility & Age Evaluation

- **Compatibility**: Verified via `PriceCacheMetadata.is_compatible(symbol_to_ticker, start_date, session_date)`:
  - Validates that `is_complete` is True, `symbol_to_ticker` exactly matches the requested map, `start_date` matches the provider's configured start date, and `session_date` matches the current session date.
  - Returns a tuple `(is_compatible: bool, reason: str)`. When incompatible, the human-readable `reason` is logged directly by `PriceProvider`.
- **Age Calculation**: Computed via `PriceCacheMetadata.age_in_minutes(now)`:
  - Encapsulates ISO timestamp parsing and timezone-aware delta calculations relative to the current market time.

### Direct Access APIs

`PriceCache` exposes decoupled methods for granular disk access:
- `read_metadata() -> PriceCacheMetadata | None`: Reads and deserializes JSON metadata.
- `read_data() -> pl.DataFrame | None`: Reads Parquet data directly with zero-copy memory mapping (`memory_map=True`) without re-reading metadata.
- `save(df, metadata) -> None`: Atomically persists data (compressed with `zstd` level 3) and metadata via temporary staging files.

### Atomic Temporary-File Replacement

To prevent data corruption from interrupted processes, power loss, or disk write failures:
1. Parquet data and JSON metadata are written to temporary staging files: `.tmp_{generation_id}_price_cache_data.parquet` and `.tmp_{generation_id}_price_cache_metadata.json`.
2. Once fully written, the temporary files atomically replace (`.replace()`) the target files.
3. If an exception occurs during write or replace, temporary files are removed immediately in the exception handler and the existing cache remains untouched and valid.

## 2. Environment Variables & Configuration

All cache settings are loaded via `Settings` (`src/radar_core/settings.py`) in accordance with the project architecture.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `RADAR_PRICE_CACHE_DIR` | Path | `/home/default/app/cache` | Container-side or local directory where cache files are stored. |
| `RADAR_PRICE_CACHE_ENABLED` | bool | `true` | Master toggle to enable or disable price cache use. |
| `RADAR_PRICE_CACHE_IGNORE` | bool | `false` | When `true`, forces `PriceProvider` to bypass cache reads and perform a full download. |
| `RADAR_PRICE_CACHE_WRITE` | bool | `true` | When `false`, downloads complete normally but are not written to the price cache. |
| `RADAR_PRICE_CACHE_TIMEZONE` | IANA tz | `America/New_York` | Timezone for evaluating the market session and trading window. |
| `RADAR_PRICE_CACHE_WINDOW_START` | time | `09:30` | Start time of the production market window (HH:MM). |
| `RADAR_PRICE_CACHE_WINDOW_END` | time | `16:00` | End time of the production market window (HH:MM). |
| `RADAR_PRICE_CACHE_DEV_MAX_AGE_MINUTES` | int | `10` | Maximum age in minutes for automatic reuse when `RADAR_ENV=dev`. |

## 3. Operational Policies & Unified Retrieval

`PriceProvider.get_prices()` follows a unified execution path for both development and production environments, accepting a mandatory `now: datetime` injected by the caller:

```
get_prices(symbols, now)
  │
  ├── Convert now to market timezone (now_market_ = now.astimezone(timezone))
  │
  ├── _is_cache_eligible(symbol_to_ticker_map, now_market_) ?
  │     ├── YES ──> _refresh_cache(...)
  │     │             ├── Success ──> Return in-memory refreshed prices (provisional prices not saved to disk)
  │     │             └── Failure ──> Fall back to full download
  │     │
  │     └── NO ───> Proceed to full download
  │
  └── Full Download from Yahoo Finance
        └── Save to price cache if all requested symbols succeed
```

### Eligibility Gatekeeper (`_is_cache_eligible`)

The `_is_cache_eligible(symbol_to_ticker_map, now)` method is the sole gatekeeper for cache reuse:
1. **Disabled / Ignored Check**: If cache is globally disabled or ignored, returns `False`.
2. **Metadata Compatibility**: Evaluates `metadata.is_compatible(symbol_to_ticker=symbol_to_ticker_map, start_date=str(self.start_date), session_date=str(now.date()))` once. If incompatible, logs the reason and returns `False`.
3. **Active Market Window**:
   - Evaluated as: `(now.weekday() < 5) and (window_start <= now.time() <= window_end)`.
   - When inside active market hours on a weekday, **both DEV and PROD** return `True` to refresh today's price bar in memory.
4. **Outside Market Window**:
   - **Post-Market Close Reuse (both DEV and PROD)**: Outside the market window on a weekday, allows cache reuse if the current time is after `window_end` (16:00), the session date matches today, and the cache was generated after market close today (`updated_at >= 16:00`).
   - **Development TTL (`RADAR_ENV=dev`)**: When not post-market settled (e.g. weekends or pre-market hours), evaluates `0 <= metadata.age_in_minutes(now) <= dev_max_age_minutes` to avoid hitting Yahoo Finance repeatedly during rapid local development sessions.
   - **Production (`RADAR_ENV!=dev`)**: Returns `False` when not post-market close settled, forcing a complete historical download to reconcile final end-of-day market settlements.

### Current-Day Refresh (`_refresh_cache`)

When cache is eligible, `_refresh_cache(symbol_to_ticker_map, tickers, now)`:
1. Loads cached historical data directly via `PriceCache.read_data()` (using memory mapping) without redundant metadata re-reads.
2. Downloads the current session's bar from Yahoo Finance:
   `yf.download(tickers, current_date, self.end_date, threads=self.max_workers, ...)`
3. Filters out the existing current-day bar and partitions historical data by `'Symbol'` in a single pass (`partition_by('Symbol', as_dict=True)`) for $O(1)$ dictionary lookups, immediately releasing the original cache DataFrame (`del cached_df_`).
4. Extracts today's bar for each ticker, vertically combines historical rows with today's row, and recalculates `PercentChange` across the merged series.
5. Retains the refreshed series in-memory for strategy analysis without calling `_save_cache()`. This ensures provisional intraday data is discarded at the end of the run and cannot pollute subsequent executions before official end-of-day market settlement.

### Fallback Policy

If any step of the cache pipeline fails (metadata incompatibility, missing parquet file, corrupted data, empty current-day download, or network errors during refresh):
- `PriceProvider` treats the event as a clean cache miss.
- A warning is logged explaining the cause.
- Execution seamlessly falls back to a complete historical download from Yahoo Finance without crashing.

## 4. Docker Container Persistence

In `docker/docker-compose.core.yml`, the core container mounts the cache volume:

```yaml
    volumes:
      - ${RADAR_HOST_CACHE_DIR:-../cache}:${RADAR_PRICE_CACHE_DIR:-/home/default/app/cache}
```

This ensures price cache data persists on the host machine across container restarts, image updates, and rebuilds.