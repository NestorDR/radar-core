# Radar Core — Financial Strategy Analyzer

Radar Core is a Python application that evaluates four complementary, pattern-based strategies ([SMA](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/moving-average-trading-strategies#price_crossovers), SMA applied to the RSI, [RSI Two Bands](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi#overbought_and_oversold_rsi_levels-1), and [RSI Rollercoaster](https://www.tecnicasdetrading.com/2011/09/tecnica-de-trading-rsi-rollercoaster.html)) on daily and weekly data to identify historical opportunities in price trends and momentum.

The strategies progress from simple price or RSI moving-average crossovers to increasingly selective RSI movements between defined levels, with Rollercoaster requiring an intermediate extreme. Positions normally close on strategy-defined reversal or output crossings; RSI Two Bands and the initial phase of RSI Rollercoaster can also close on Mogalef-based price stop-losses. If neither condition occurs before the end of the analysis period, the position is valued at the final available bar. Only configurations with strictly positive net [profit](https://estrategiastrading.com/profit-factor/) and [expected](https://estrategiastrading.com/calcular-la-esperanza-matematica-del-sistema-de-trading/) value, after strategy-specific candidate screening, are persisted. These results remain subject to false signals, changing market conditions, and historical overfitting.

For details, see the [Strategy Implementation Overview](docs/strategy-implementation-overview.md).

The analyzer downloads financial asset prices from Yahoo Finance, converts the external Pandas data to **Polars** DataFrames, and dispatches per-symbol worker processes. Each worker derives the weekly data and executes high-speed strategy evaluation using **NumPy arrays** and **Numba JIT-compiled kernels**.

The project follows High Performance Practices, using concurrent symbol processing and CPU-optimized JIT kernels. Daily and weekly analyses for each symbol are evaluated sequentially within its worker. Its external runtime infrastructure is supported by the [Radar Infra](https://github.com/NestorDR/radar-infra) project.

The fully operational results can be visited for public use: 
- [All Ratios](https://radar.ndromero.com/public/dashboard/6e547cac-cbc3-4354-97c3-6745d8540d83?gain_prob=0.51&profit_vs_change=-0.20&security=&signals=2&strategy=&time_frame=#theme=night)
- [Ratios for Stocks](https://radar.ndromero.com/public/dashboard/147ee420-badb-451c-a2d5-c30e78688ed0?profit_vs_change=&security=&strategy=&tab=6-day---open-----%7C#theme=night) 
- [Ratios for Crypto](https://radar.ndromero.com/public/dashboard/0af531b3-df15-4aa4-a665-69704e95451e?profit_vs_change=&security=&strategy=&tab=10-day-open-----%7C#theme=night)

## Features
- **Hybrid Data Architecture**:
    - **Polars**: High-performance DataFrame management for data ingestion and storage.
    - **NumPy & Numba**: Strategy logic is decoupled into JIT-compiled kernels for near-native execution speed.
- **Concurrent Analysis**: Multi-symbol processing using Python's `ProcessPoolExecutor`.
- **Yahoo Finance Integration**: Automated download of historical daily prices and local weekly aggregation.
- **Technical Analysis & Strategies**: Built-in support for Moving Averages (SMA), RSI-based variants (RSI SMA, Two Bands, Rollercoaster), and Mogalef Bands used as stop-loss levels for RSI band strategies.
- **Modular Input Filtering**: Trade entry confirmation filters (True Range price action candle confirmation with overnight gap handling, ATR volatility regime, SMA 200 trend) with timeframe-level precomputed boolean masks and zero-allocation baseline bypass (see [RSI Strategies Input Filtering Guide](docs/rsi-strategies-input-filtering.md)).
- **Local Price Caching**: Persistent Parquet OHLCV caching to eliminate redundant external downloads (see [Price Cache Guide](docs/local_price_cache.md)).
- **Performance Metrics**: Detailed profiling including net profit, success rate, mathematical expectation, trade averages, and exposure.
- **Database Synchronization**: Transactional management of trading ratios and optional cleanup of unlisted symbols via `psycopg3`.
- **Configurable settings**: Symbols, verbosity, concurrency, enabled strategies, and input confirmation filters.
- **Database-Driven Shortability**: Dynamic resolution of long-only vs. short positions based on database security records.
    
## Prerequisites
- Python 3.13+
- Recommended OS: Windows, Linux, or macOS
- Required libraries (managed via pyproject.toml):
  - polars
  - yfinance
  - numba, numpy, PyYAML, dotenvy-py, psycopg, psycopg-binary, setuptools, tzdata
  - TA-Lib (see notes below)

TA-Lib on Windows: install the prebuilt wheel noted in pyproject.toml (example shown in Installation). On non‑Windows platforms, TA-Lib can be installed from PyPI (see environment markers in pyproject.toml).

Note: The project is developed on a Windows 11 host using Python 3.13, PyCharm 2026.1+, PostgreSQL 17.x, and Docker Desktop v4.88+.

## Installation
The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

1. Create a virtual environment and install dependencies:
   - `uv venv`
   - `uv sync`
   - For development tools (`ruff`, `ty`, and `pre-commit`), run: `uv sync --group dev --active`
2. **Windows + TA-Lib** (if needed):
   - `uv pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.4/ta_lib-0.6.4-cp313-cp313-win_amd64.whl --no-cache-dir`

## Quick Start
You can run the analyzer directly from the repository without installing the package system‑wide.

- Run as a module (using the CLI entry point):
  - `python -m radar_core`

- Or run as an independent script (useful for specific tests, since it supports the `if __name__ == '__main__':` block):
  - `python src/radar_core/analyzer.py`

The direct `analyzer.py` script uses its hard-coded smoke-test symbol list; use the module command for settings-driven execution.

By default, the analyzer will:
- Initialize settings and database connections.
- Download daily prices and prepare in-memory Polars DataFrames.
- Convert price columns to NumPy arrays and dispatch JIT-compiled kernels for strategy identification.
- Use the configured number of worker processes for symbol-level parallel execution (the `Settings` default is one); daily and weekly analysis run sequentially within each worker.
- Evaluate strategies for daily and weekly timeframes.
- Print atomic, buffered logs per symbol above DEBUG verbosity; DEBUG output is streamed live.

## Architecture
The system follows a three-tier performance model:
1. **Adapter Layer**: Pandas/yfinance for external data compatibility.
2. **Storage Layer**: **Polars** for lightning-fast in-memory data manipulation and grouping.
3. **Execution Layer**: **NumPy + Numba** for the heavy mathematical lifting (JIT-compiled backtesting kernels).

```mermaid
flowchart TD
    CLI["CLI & Settings"] --> Analyzer["Analyzer Orchestrator"]

    subgraph Tier1 ["1. Adaptation & Ingestion Layer"]
        PriceProvider["PriceProvider<br>(Yahoo Finance API / Parquet Cache)"]
    end

    subgraph Tier2 ["2. In-Memory Storage Layer (Polars)"]
        PolarsData["Polars DataFrames<br>(Daily / Weekly & Technical Indicators)"]
    end

    subgraph Tier3 ["3. Execution & Calculation Layer (NumPy + Numba)"]
        Workers["Worker Pool (ProcessPoolExecutor)"] --> JITKernels["Numba JIT-Compiled Strategy Kernels<br>(SMA, RSI, Stop Loss)"]
    end

    subgraph DBBoundary ["Persistence Boundary (psycopg3)"]
        DB[("PostgreSQL Database")]
    end

    Analyzer --> PriceProvider
    PriceProvider --> PolarsData
    Analyzer --> Workers
    PolarsData --> Workers
    JITKernels -->|Transactional Upsert| DB
    PriceProvider -.->|Symbol Auto-Registration| DB
```

For each symbol, `analyzer.py` downloads daily prices, derives weekly prices with Polars, and evaluates only the strategies enabled in `settings.yml`. Before requesting historical prices from Yahoo Finance, `PriceProvider` checks a local Parquet cache (`PriceCache`) to eliminate redundant downloads when eligible (see [docs/local_price_cache.md](docs/local_price_cache.md)). When RSI strategies are enabled, shared RSI and, when required, Mogalef stop-loss indicators are calculated once per timeframe, including JIT-accelerated stop-loss bar scanning. If `rsi_input_filter` is configured in `settings.yml` (e.g. `'price_action'`), Long and Short trade eligibility masks are precomputed once per timeframe using True Range candle confirmation (or ATR regime / SMA trend) and injected into `RsiTwoBands` and `RsiRollerCoaster`, gating trade entries to confirmed setups with zero runtime penalty or array allocations during baseline runs (see [docs/rsi-strategies-input-filtering.md](docs/rsi-strategies-input-filtering.md)). `PriceProvider` translates internal symbols to Yahoo Finance tickers auto-registering missing symbols from Yahoo Finance into database and guards against empty ticker downloads before converting the Pandas response to Polars. Strategy execution kernels leverage shared inlined Numba helpers for crossover detection, trade math, and candidate screening. Shortable symbols are queried from database to determine whether each symbol evaluates long-only or both long and short strategy positions. Strategy execution results (`Ratios`) are managed transactionally in the database, atomically persisting positive ratios.

Mogalef bands are used directly as `LongStopLoss` and `ShortStopLoss` for the RSI Two Bands and Rollercoaster strategies. Those strategies retain `identify_old` for baseline comparison while `identify` runs the fused implementation. Serialized current-indicator metadata (`Ratios.current_indicators`) preserves dashboard keys across all strategies, including `sma` (and `rsi` for RSI SMA) for Moving Average variants and `rsi`, `up`, and `low` for RSI band strategies.

## Minimal Example
Below is a minimal snippet that shows how you might pull prices and run a simple analysis, similar to what the analyzer does internally. It requires the project dependencies, database connection settings, and an initialized Radar database with the strategy records.

```python
from datetime import datetime, timezone
import polars as pl
from radar_core.infrastructure.price_provider import PriceProvider
from radar_core.domain.strategies import MovingAverage
from radar_core.helpers.constants import DAILY, SMA

# Define a list of symbols to analyze
symbols_ = ["BTC-USD"]

# Download prices data for all symbols to be analyzed
now_ = datetime.now(timezone.utc)
prices_data_ = PriceProvider(long_term=False).get_prices(symbols_, now_)

# Configure analyzer
ma = MovingAverage(SMA, value_column_name="Close", ma_column_name="Sma")
only_long_positions_ = False

# Iterate over symbols
for symbol_, prices_df_ in prices_data_.items():
    # The analyzer orchestrates identify() and logging; here we just demonstrate the objects.
    prices_df_ = prices_df_.with_columns(pl.arange(0, pl.len(), eager=False).cast(pl.Int32).alias("BarNumber"))
    close_prices_ = prices_df_["Close"].to_numpy()
    ma.identify(symbol_, DAILY, only_long_positions_, prices_df_, close_prices_)

    # See src/radar_core/analyzer.py for a full run.
```

## Example Output
A typical console output (truncated) may look like:

```console
Reading YAML file settings.yml...
Analyzer.py started at 2025-12-22 09:58:52.
Cleaned 0 rows from the database for deprecated symbols.
Starting parallel analysis for 1 symbols using X workers...

[BTC-USD]: Analysis started at 2025-12-22 09:58:55...
[BTC-USD]: Daily time frame analysis started at 2025-12-22 09:58:55
shape: (1, 7)
┌─────────────────────┬───────┬───────┬───────┬──────┬──────────┐
│ Date                ┆ Open  ┆ High  ┆ Low   ┆ Close┆ Volume   │
├─────────────────────┼───────┼───────┼───────┼──────┼──────────┤
│ 2020-01-01 00:00:00 ┆ …     ┆ …     ┆ …     ┆ …    ┆ …        │
│ …                   ┆ …     ┆ …     ┆ …     ┆ …    ┆ …        │
└─────────────────────┴───────┴───────┴───────┴──────┴──────────┘
SMA         on BTC-USD: start 2025-12-22 09:58:55 ... end 2025-12-22 09:58:58  0.0 min
[BTC-USD]: Analysis completed in 0.0 min
...
Analysis executed from 2025-12-22 09:58:52 to 2025-12-22 09:58:59 - Elapsed time 0.1 min
```

Note: Actual output will vary based on a symbol list, dates, and verbosity. Output blocks per symbol are buffered atomically above DEBUG verbosity; DEBUG output is streamed live.

## Configuration
Project settings are managed by the `Settings` class, implemented as a process-local lazy singleton accessed via `get_settings()` (or `Settings()`). You can configure the application via the `src/radar_core/settings.yml` file for financial strategies and using **Environment Variables** for infrastructure-oriented settings (logging, concurrency, database connection, etc.). The singleton reads and parses both sources once during initial construction, providing a centralized snapshot (`Settings.db_conn_kwargs`, log configuration, strategy filters, and symbol lists) across all modules. The `evaluable_strategies` list accepts `sma`, `rsi_sma`, `rsi_rc`, and `rsi_2b`. The optional `rsi_input_filter` setting configures trade entry confirmation filtering for RSI band strategies (`'price_action'`, `'atr_volatility'`, `'sma_trend'`, or `'none'` / omitted for unfiltered baseline execution). Shortable eligibility is resolved from the database.

### Key Environment Variables

| Variable                    | Description                                                                                  | Default                       |
|:----------------------------|:---------------------------------------------------------------------------------------------|:------------------------------|
| `RADAR_ENV`                 | `dev` loads `.env` from the `radar_core` package directory or up to two parent directories; other values use process environment | `dev`                         |
| `RADAR_CLEAN_UNLISTED`      | Delete stored ratios for symbols not listed in `settings.yml`                                | `false`                       |
| `RADAR_LOG_LEVEL`           | Logging verbosity (10=DEBUG, 20=INFO, etc.)                                                  | `20` (INFO)                   |
| `RADAR_ENABLE_FILE_LOGGING` | Write logs to a rotating file                                                                | `false`                       |
| `RADAR_LOG_FOLDER`          | File-log folder, relative to the `radar_core` package when not absolute                      | `logs`                        |
| `RADAR_MAX_WORKERS`         | Number of parallel processes; non-positive values are clamped to one by `Settings`          | `1`                           |
| `RADAR_SETTING_FILE`        | Custom settings YAML path, relative to `src/radar_core` when not absolute                    | `settings.yml`                |
| `RADAR_PRICE_CACHE_DIR`     | Local or container directory where OHLCV Parquet and metadata files are stored               | `/home/default/app/cache`     |
| `RADAR_PRICE_CACHE_ENABLED` | Master toggle to enable or disable price cache use                                           | `true`                        |
| `RADAR_PRICE_CACHE_IGNORE`  | When `true`, forces `PriceProvider` to bypass cache reads and perform a full download        | `false`                       |
| `RADAR_PRICE_CACHE_WRITE`   | When `false`, downloads complete normally but are not written to disk                        | `true`                        |
| `RADAR_PRICE_CACHE_TIMEZONE`| IANA timezone for evaluating market session dates and trading start time    | `America/New_York`            |
| `RADAR_PRICE_CACHE_TRADING_START` | Market trading start time (HH:MM) when corporate adjustments settle       | `09:30`                       |
| `RADAR_PRICE_CACHE_DEV_MAX_AGE_MINUTES` | Maximum cache age in minutes for automatic reuse outside trading when `RADAR_ENV=dev` | `10`           |
| `POSTGRES_*`                | PostgreSQL host, port, database, user, and password settings                                  |                               |
| `POSTGRES_SSL_MODE`         | PostgreSQL connection SSL mode                                                               | `prefer`                      |
| `POSTGRES_OPTIONS`          | Optional PostgreSQL connection options passed to the connection                          | unset                         |

## Docker
Containerization is available for the application environment. The multi-stage image builds the TA-Lib C library inside the container, eliminating host setup requirements. Running is envisaged through `docker compose`.

### Build and Run
```bash
docker build -t radar-core:dev-0.5.0 -f docker/Dockerfile .
```

### Docker Compose
Two Compose targets are provided in `docker/`, configured via environment files in `envs/`:
- **Core Analyzer**: `docker compose -f docker/docker-compose.core.yml up -d --build` (helper: `auto\dc.cmd core`).
- **Metabase Dashboard**: `docker compose -f docker/docker-compose.mb.yml up -d` (helper: `auto\dc.cmd mb`).

Comprehensive multi-environment deployments (dev, e2e, prod) and database infrastructure are managed in [Radar Infra](https://github.com/NestorDR/radar-infra).

## Automation Scripts
The `auto/` directory contains Windows Command scripts to simplify common tasks:

- **`auto\dc.cmd <target>`**: Helper for Docker Compose.
  - Usage: `auto\dc.cmd core`.
  - It handles environment file injection and project naming.

- **`auto\update.cmd`**: Updates the development environment.
  - Updates `uv`, upgrades `uv.lock`, syncs dependencies, and re-installs TA-Lib from the prebuilt wheel.

- **`auto\lint.cmd`**: Automatic lint checks and corrections.
  - Runs Ruff lint checks and autofixes; Ruff formatting is currently disabled in the script.

- **`auto\test.cmd`**: Testing is implemented using `pytest`. Unit tests are located under the `tests/` directory.
  - The test suite includes fast, in-memory unit tests using `pytest` and `unittest.mock` covering symbol translation and auto-creation, price provider download guards, model instantiation, error handling, and CRUD methods without external database dependencies.

- **`auto\cleanup.cmd`**: Cache and temporary file cleanup.
  - Clears Python bytecode caches (`__pycache__`) and cleans Ruff cache using `uvx ruff clean`.

## Project Status
In active development and continuous improvement. External runtime infrastructure and end-to-end environments are managed in the [Radar Infra](https://github.com/NestorDR/radar-infra) project. This repository contains the application, database schema files, Docker services, and CI/CD workflow.

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for the full license text.

Under this license, if you modify this software or run it as a network service (SaaS), you must make your modified source code publicly available under the same terms.
