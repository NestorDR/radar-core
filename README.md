# Radar Core — Financial Strategy Analyzer

Radar Core is a Python application that downloads financial asset prices from Yahoo Finance, processes them with Polars DataFrames, and evaluates speculative trading strategies using performance metrics such as net profit and percentage of success. The project is under active development and continuous optimization.

## Features
- Yahoo Finance integration via yfinance for historical daily prices
- High‑performance data processing using Polars
- Built‑in technical analysis and strategies (e.g., Moving Average and RSI‑based variants)
- Performance metrics and logs (e.g., net profit, success rate)
- Configurable settings (symbols, shortable assets, verbosity)

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

Note: The project was developed on Windows 11, Python 3.13, Pycharm 2025 and Docker Desktop 4.47 

## Installation
You can install with either uv (recommended for this project) or pip.

Option A — using uv:
1. Install uv if you don’t have it yet: https://docs.astral.sh/uv/
2. Create a virtual environment and install dependencies:
   - uv venv
   - uv sync
3. Windows + TA‑Lib only (if needed):
   - uv pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.7/ta_lib-0.6.7-cp313-cp313-win_amd64.whl --no-cache-dir

Option B — using pip:
1. Create and activate a virtual environment
2. Install dependencies from pyproject (via pip):
   - pip install -e .
3. Windows + TA‑Lib only (if needed):
   - pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.7/ta_lib-0.6.7-cp313-cp313-win_amd64.whl --no-cache-dir

## Quick Start
You can run the analyzer directly from the repository without installing the package system‑wide.

- Run as a script:
  - python src/radar_core/analyzer.py

- Or run as a module (after installing the project into your environment):
  - python -m radar_core.analyzer

By default, the analyzer will:
- Initialize settings (symbols, logging)
- Download daily prices from Yahoo Finance (e.g., BTC-USD is used in the current example config)
- Process data using Polars
- Evaluate strategies for daily and weekly timeframes
- Print progress and summary logs

## Minimal Example
Below is a minimal snippet that shows how you might pull prices and run a simple analysis, similar to what the analyzer does internally.

```python
import polars as pl
from radar_core.infrastructure.price_provider import PriceProvider
from radar_core.domain.strategies import MovingAverage
from radar_core.domain.strategies.constants import SMA
from radar_core.helpers.constants import DAILY

# Define a list of symbols to analyze
symbols_ = ['BTC-USD', 'NDQ', 'QQQ']

# Download prices data for all symbols to be analyzed
prices_data_ = PriceProvider(long_term=False).get_prices(symbols_)

# Configure analyzer
ma = MovingAverage(SMA, value_column_name="Close", ma_column_name="Sma")
only_long_positions_=False

# Iterate over symbols
for symbol_, prices_df_ in prices_data_.items():
    # The analyzer orchestrates identify() and logging; here we just demonstrate the objects.
    ma.identify(symbol_, DAILY, only_long_positions_, prices_df_, None)
    
    # See src/radar_core/analyzer.py for a full run.
```

## Example Output
A typical console output (truncated) may look like:

```
Analyzer.py started at 2025-09-28 10:35:00.
Starting the Daily time frame analysis for BTC-USD...
┌─────────────────────┬───────┬───────┬───────┬──────┬──────────┐
│ Date                ┆ Open  ┆ High  ┆ Low   ┆ Close┆ Volume   │
├─────────────────────┼───────┼───────┼───────┼──────┼──────────┤
│ 2020-01-01 00:00:00 ┆ …     ┆ …     ┆ …     ┆ …    ┆ …        │
│ …                   ┆ …     ┆ …     ┆ …     ┆ …    ┆ …        │
└─────────────────────┴───────┴───────┴───────┴──────┴──────────┘
[BTC-USD]: Analysis completed in 4.0 min
Analyzer.py - Started at 2025-09-28 10:35:00 ... Ended at 2025-09-28 10:39:00 - Elapsed time 0.4 min
```

Note: Actual output will vary based on a symbol list, dates, and verbosity.

## Configuration
Project settings are managed by the Settings class and YAML files located under src/radar_core/ (e.g., settings.yml and environment‑specific overrides). You can:
- Configure the list of symbols to analyze
- Mark shortable assets
- Adjust verbosity/logging

See src/radar_core/settings.py and the provided YAML files for details.

## Docker
Containerization is available for a fully reproducible environment. The image is multi-stage and builds the TA-Lib C library inside the container, so you don’t need any TA-Lib setup on your host.

Prerequisites:
- Docker 24+ (or Docker Desktop on Windows/macOS)
- Docker Compose v2 (optional, recommended for local DB + app)

Build the image:
```
docker build -t radar-core:dev-0.4.0 .
```

Run the analyzer directly with Docker (connecting to an existing PostgreSQL):
- Example (Windows PowerShell):
```shell
  docker run --rm \
    -e POSTGRES_HOST=host.docker.internal \
    -e POSTGRES_PORT=5432 \
    -e POSTGRES_DB=radar \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=your_password \
    -e ENABLE_FILE_LOGGING=false \
    -e LOG_LEVEL=20 \
    radar-core:dev-0.4.0
```

Using Docker Compose (spins up Postgres + the app):
- Ensure you have an env file with DB credentials at src/radar_core/.env.production (can be created from src/radar_core/.env.template). Start both services:
```shell
  docker compose -f docker-compose.dev.yml up -d --build
```

Notes:
- The Compose file builds the image and waits for the database to become healthy before starting the analyzer.
- To override configuration without rebuilding, you can bind-mount a custom settings.yml:
  - docker run --rm -v %cd%\src\radar_core\settings.yml:/home/default/app/settings.yml:ro radar-core:dev-0.4.0
  - On Linux/macOS, adjust the host path accordingly.
- If you connect the containerized app to a host PostgreSQL, POSTGRES_HOST=host.docker.internal is convenient on Docker Desktop. On native Linux, you may need an extra_hosts entry mapping host.docker.internal to the host gateway.

## Project Status
In active development and continuous improvement. Expect updates, refactoring, and performance tuning.

## License
This project is licensed under the MIT License. See the LICENSE file if available; otherwise, you may consider the standard MIT terms applicable by default.
