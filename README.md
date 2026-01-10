# Radar Core — Financial Strategy Analyzer

Radar Core is a Python application that downloads financial asset prices from Yahoo Finance, manages them using **Polars** as an efficient in-memory database, and executes high-speed strategy evaluation using **vectorized NumPy operations** and **Numba JIT compilation**.

The project follows High Performance Practices, utilizing concurrent processing and hardware-accelerated math to analyze multiple symbols and timeframes simultaneously.

## Features
## Features
- **Hybrid Data Architecture**:
    - **Polars**: High-performance DataFrame management for data ingestion and storage.
    - **NumPy & Numba**: Strategy logic is decoupled into JIT-compiled kernels for near-native execution speed.
- **Concurrent Analysis**: Multi-symbol processing using Python's `ProcessPoolExecutor`.
- **Yahoo Finance Integration**: Automated download of historical daily and weekly prices.
- **Technical Analysis & Strategies**: Built-in support for Moving Averages (SMA) and complex RSI-based variants (Two Bands, Rollercoaster).
- **Performance Metrics**: Detailed profiling including net profit, success rate, mathematical expectation, and risk-adjusted ratios.
- **Database Synchronization**: Automated management of trading ratios and symbol cleanup via SQLAlchemy.
- Configurable settings (symbols, shortable assets, verbosity, concurrency)
    

## Prerequisites
- Python 3.13+
- Recommended OS: Windows, Linux, or macOS
- Required libraries (managed via pyproject.toml):
  - polars
  - yfinance
  - SQLAlchemy
  - numpy, PyYAML, dotenvy-py
  - TA‑Lib (see notes below)

TA‑Lib on Windows: install the prebuilt wheel noted in pyproject.toml (example shown in Installation). On non‑Windows platforms, TA‑Lib can be installed from PyPI (see environment markers in pyproject.toml).

Note: The project was developed on Windows 11, Python 3.13, Pycharm 2025, and Docker Desktop 4.50 

## Installation
You can install with either uv (recommended for this project) or pip.

Option A — using uv:
1. Install uv if you don’t have it yet: https://docs.astral.sh/uv/
2. Create a virtual environment and install dependencies:
   - uv venv
   - uv sync
3. Windows + TA‑Lib only (if needed):
   - uv pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.8/ta_lib-0.6.8-cp313-cp313-win_amd64.whl --no-cache-dir

Option B — using pip:
1. Create and activate a virtual environment
2. Install dependencies from pyproject (via pip):
   - pip install -e .
3. Windows + TA‑Lib only (if needed):
   - pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.8/ta_lib-0.6.8-cp313-cp313-win_amd64.whl --no-cache-dir

## Quick Start
You can run the analyzer directly from the repository without installing the package system‑wide.

- Run as a script:
  - python src/radar_core/analyzer.py

- Or run as a module (after installing the project into your environment):
  - python -m radar_core.analyzer

By default, the analyzer will:
- Initialize settings and database connections.
- Download prices and synchronize the in-memory Polars store.
- **Vectorize price data** and dispatch high-speed kernels for strategy identification.
- **Auto-detect CPU cores** for parallel worker execution.
- Evaluate strategies for daily and weekly timeframes.
- Print atomic, buffered logs per symbol.

## Architecture
The system follows a three-tier performance model:
1. **Adapter Layer**: Pandas/yfinance for external data compatibility.
2. **Storage Layer**: **Polars** for lightning-fast in-memory data manipulation and grouping.
3. **Execution Layer**: **NumPy + Numba** for the heavy mathematical lifting (vectorized backtesting).

## Minimal Example
Below is a minimal snippet that shows how you might pull prices and run a simple analysis, similar to what the analyzer does internally.

```python
import polars as pl
from radar_core.infrastructure.price_provider import PriceProvider
from radar_core.domain.strategies import MovingAverage
from radar_core.helpers.constants import DAILY, SMA

# Define a list of symbols to analyze
symbols_ = ['BTC-USD']

# Download prices data for all symbols to be analyzed
prices_data_ = PriceProvider(long_term=False).get_prices(symbols_)

# Configure analyzer
ma = MovingAverage(SMA, value_column_name="Close", ma_column_name="Sma")
only_long_positions_ = False

# Iterate over symbols
for symbol_, prices_df_ in prices_data_.items():
    # The analyzer orchestrates identify() and logging; here we just demonstrate the objects.
    ma.identify(symbol_, DAILY, only_long_positions_, prices_df_, None)

    # See src/radar_core/analyzer.py for a full run.
```

## Example Output
A typical console output (truncated) may look like:

```
Reading YAML file settings.yml...
Analyzer.py started at 2025-12-22 09:58:52.
Cleaned 0 rows from the database for deprecated symbols.
Starting parallel analysis for 1 symbols using X workers...

[BTC-USD]: Launching parallel worker process at 2025-12-22 09:58:55...
[BTC-USD]: Daily time frame analysis started at 2025-12-22 09:58:55
shape: (1, 7)
[BTC-USD]: Daily time frame analysis started...
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

Note: Actual output will vary based on a symbol list, dates, and verbosity. Output blocks per symbol are printed atomically to prevent interleaving.

## Configuration
Project settings are managed by the `Settings` class. You can configure the application via the `src/radar_core/settings.yml` file or by using **Environment Variables** (which take precedence).

### Key Environment Variables

| Variable                    | Description                                                         | Default |
|:----------------------------|:--------------------------------------------------------------------| :--- |
| `RADAR_LOG_LEVEL`           | Logging verbosity (10=DEBUG, 20=INFO, etc.)                         | `20` (INFO) |
| `RADAR_ENABLE_FILE_LOGGING` | Write logs to `src/radar_core/logs/`                                | `true` |
| `RADAR_MAX_WORKERS`         | Number of parallel processes. Set to `0` to use all available CPUs. | `0` (Auto) |
| `RADAR_SETTING_FILE`        | Custom path to the settings YAML file                               | `src/radar_core/settings.yml` |
| `POSTGRES_*`                | Database settings                                                   | |

## Docker
Containerization is available for a fully reproducible environment. The image is multi-stage and builds the TA-Lib C library inside the container, so you don’t need any TA-Lib setup on your host.

Prerequisites:
- Docker Engine 24+ (included in all modern Docker Desktop versions)
- Docker Compose v2 (optional, recommended for local DB + app)

Build the image:
```
docker build -t radar-core:dev-0.5.0 .
```

Run the analyzer directly with Docker (connecting to an existing PostgreSQL):
- Example (Windows PowerShell):
```textmate
docker run --rm \
    -e POSTGRES_HOST=host.docker.internal \
    -e POSTGRES_PORT=5432 \
    -e POSTGRES_DB=radar \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=your_password \
    -e RADAR_ENABLE_FILE_LOGGING=false \
    -e RADAR_LOG_LEVEL=20 \
    -e RADAR_MAX_WORKERS=4 \
    radar-core:dev-0.5.0
```

Using Docker Compose (spins up Postgres + the app):
- Ensure you have an env file with DB credentials at envs/.env.e2e (can be created from envs/.env.template). Start both services:
```textmate
docker compose -f docker-compose.e2e.yml up -d --build
```

Notes:
- The Compose file builds the image and waits for the database to become healthy before starting the analyzer.
- To override configuration without rebuilding, you can bind-mount a custom settings.yml:
  - `docker run --rm -v %cd%\src\radar_core\settings.yml:/home/default/app/settings.yml:ro radar-core:dev-0.5.0`
  - On Linux/macOS, adjust the host path accordingly.
- If you connect the containerized app to a host PostgreSQL, `POSTGRES_HOST=host.docker.internal` is convenient on Docker Desktop. On native Linux, you may need an extra_hosts entry mapping host.docker.internal to the host gateway.

## Project Status
In active development and continuous improvement. Expect updates, refactoring, and performance tuning.

## License
This project is licensed under the MIT License. See the LICENSE file if available; otherwise, you may consider the standard MIT terms applicable by default.