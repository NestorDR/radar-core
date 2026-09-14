# RSI Strategies Input Filtering Architecture & Codebase Status

## 1. Executive Summary

This document summarizes the current architecture, implementation, and operational status of **RSI Strategies Input Filtering** across the Radar Core codebase.

The input filtering subsystem introduces decoupled, vectorized trade entry filtering for the RSI-based strategies—**RSI Two Bands** (`RsiTwoBands`) and **RSI Rollercoaster** (`RsiRollerCoaster`)—without modifying exit criteria, stop-loss mechanics, or position lifecycles.

```text
       ┌────────────────────────────────────────────────────────┐
       │                   Timeframe Ingestion                  │
       │             (Daily or Resampled Weekly OHLCV)          │
       └───────────────────────────┬────────────────────────────┘
                                   │
                    Precompute Shared Indicators
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
     RSI Series              Mogalef Bands             Domain Filter
     (TA-Lib RSI)        (Stop-Loss Channels)      (Precomputed Masks)
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
               Strategy Orchestration (analyzer.py)
                    identify(..., is_input_eligible)
                                   │
                                   ▼
               Numba JIT Decoupled Compute Kernels
          _grid_search_*_fused(...) & _find_trades_*(...)
                                   │
                                   ▼
                   Filtered Trade Entry Decisions
```

### Key Architectural Tenets
1. **Precomputed Vectorized Masks**: Evaluated once per timeframe in Polars/NumPy rather than repeatedly re-evaluating conditions inside high-dimensional parameter loops.
2. **Decoupled Numba JIT Kernels**: Pure module-level functions decorated with `@njit(cache=True)` accept 1D boolean NumPy arrays (`eligible_mask: np.ndarray | None`) and perform zero-overhead register-level checks.
3. **Zero-Allocation Baseline**: When filtering is set to `'none'`, `'baseline'`, `''`, or `None`, the registry returns `(None, None)` directly, bypassing memory allocation and preserving raw baseline performance.
4. **Independent Entry Boundary**: Filters act strictly on the input signal bar $t$; once a trade entry is accepted, the existing exit rules, stop-loss scanning, and trade finalization remain unchanged.

---

## 2. Domain Filter Hierarchy (`src/radar_core/domain/filters/`)

All trade entry filters reside in `src/radar_core/domain/filters/` and subclass `FilterABC`.

### 2.1 Base Abstraction (`base_filter.py`)
The abstract base class defines the filter contract:

```python
class FilterABC(ABC):
    @abstractmethod
    def get_masks(
        self,
        prices_df: pl.DataFrame,
        is_bear: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Calculates 1D boolean NumPy masks for Long and Short trade entries.
        
        :param prices_df: Polars DataFrame containing OHLCV price series.
        :param is_bear: Whether the security is an inverse ETF (bear asset).

        :return: Tuple of (long_mask, short_mask) as 1D boolean NumPy arrays (or None for unfiltered directions).
        """
```

### 2.2 Filter Registry & Factory (`registry.py`)
`registry.py` provides centralized discovery and mask computation:
- **`FILTER_REGISTRY`**: Maps registered string identifiers to filter classes:
  - `'price_action'`: `PriceActionFilter`
  - `'atr_volatility'`: `AtrVolatilityFilter`
  - `'sma_trend'`: `SmaTrendFilter`
- **`get_filter(filter_name: str, **kwargs)`**: Instantiates the requested filter class or raises `KeyError` with available filters.
- **`get_filter_masks(filter_name: str | None, prices_df: pl.DataFrame, is_bear: bool)`**: Factory method that handles baseline queries (`None`, `''`, `'none'`, `'baseline'`) by returning `(None, None)` without array allocation, forwarding `is_bear` to the filter instance.

### 2.3 Price Action Filter with Directional Retention Ratio (`price_action.py`)
Evaluates candle confirmation against **Directional Retention Ratios** relative to the prior close anchor ($Close_{t-1}$), calibrated with **Market Asymmetry**:

In technical analysis, relative to the prior close anchor $Close_{t-1}$:

- $High_t - Close_{t-1}$ is the Upside True Range ($UTR_t$) (maximum upward excursion/buying reach).
- $Close_{t-1} - Low_t$ is the Downside True Range ($DTR_t$) (maximum downward excursion/selling reach).
Whenever yesterday's close falls within today's range ($Low_t \le Close_{t-1} \le High_t$): $$(High_t - Close_{t-1}) + (Close_{t-1} - Low_t) = High_t - Low_t = \text{True Range}_t$$

The proposed expressions are the two directional halves of the True Range.
```
       High_t ────────────────┐
                              │  Upside Span:
                              │  High_t - Close_{t-1}
  Close_{t-1} ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄─┼───────────────────────
                              │  Downside Span:
                              │  Close_{t-1} - Low_t
        Low_t ────────────────┘
```

- **Directional Spans**:
  $$\text{Span}_{\text{long}} = High_t - Close_{t-1}$$
  $$\text{Span}_{\text{short}} = Close_{t-1} - Low_t$$

- **Market Asymmetry & Regime Dynamics**:
  The empirical finding that **price action filtering improves short setups but degrades long setups** aligns directly with the structural mechanics of equity and financial markets:

  ```text
                            ┌────────────────────────────────────────────────────────┐
                            │                Market Regime Dynamics                  │
                            └───────────────────────────┬────────────────────────────┘
                                                        │
                           ┌────────────────────────────┴────────────────────────────┐
                           ▼                                                         ▼
              ┌─────────────────────────┐                               ┌─────────────────────────┐
              │   Bull / Long Regimes   │                               │   Bear / Short Regimes  │
              ├─────────────────────────┤                               ├─────────────────────────┤
              │ • Structural upward bias│                               │ • Violent cascades      │
              │ • "Escalator up"        │                               │ • "Elevator down"       │
              │ • Grinding bounces      │                               │ • Bear market squeezes  │
              │ • Low intra-bar urgency │                               │ • High intra-bar panic  │
              └────────────┬────────────┘                               └────────────┬────────────┘
                           │                                                         │
                           ▼                                                         ▼
              ┌─────────────────────────┐                               ┌─────────────────────────┐
              │ Candle Filtering Risk:  │                               │ Candle Filtering Edge:  │
              │ Rejection of valid dips │                               │ Rejection of weak       │
              │ & late entries          │                               │ bounces & bear traps    │
              └─────────────────────────┘                               └─────────────────────────┘
  ```

  1. **Standard Assets (`is_bear = False`)**:
     - **Long Positions on Standard Equities**:
       - In secular or cyclical uptrends, oversold RSI bounces often start quietly with indecisive candles, lower shadows, or small inside bars before trending higher ("escalator up").
       - Requiring candle confirmation ($Close > Open$ and directional retention) filters out high-expectancy mean-reversion entries or forces the entry after initial price expansion has already occurred.
       - Earlier historical calibration with `DEFAULT_LONG_THRESHOLD = 0.10` confirmed this phenomenon: even a lenient 10% threshold rejected valid entries without adding statistical edge.
       - **Implementation**: Regular long entries remain **unfiltered** (`long_mask = None`), bypassing array allocations and executing as pure zero-allocation baseline.
     - **Short Positions on Standard Equities**:
       - Downtrends and market sell-offs are dominated by volatility expansion and rapid cascades ("elevator down").
       - Short signals occurring on candles with significant lower shadows (where intraday buyers stepped in off the lows) frequently get caught in sharp bear-market squeeze rallies.
       - Requiring `DEFAULT_SHORT_THRESHOLD = 0.95` (closing in the bottom 5% of the downward span, at/near session lows):
         $$\frac{Close_{t-1} - Close_t}{Close_{t-1} - Low_t} \ge 0.95$$
         along with bearish candle close ($Close_t < Open_t$) and positive downward span ($\text{Span}_{\text{short}} > 0$) ensures short trades only trigger when selling pressure completely overwhelms the session, eliminating premature fading into strong bull moves.

  2. **Inverse ETFs / Bear Assets (`is_bear = True`, e.g., SPXS, SOXS, SQQQ, DUST, TZA, LABD)**:
     - **Database Evidence**: A query to the PostgreSQL production database confirms the market mechanics:
       - In `securities`: all `is_bear = True` securities have `is_shortable = False`.
       - In `ratios`: every single ratio row for inverse ETFs has `is_long_position = True`.
     - **Economic Trade Mapping**:
       When trading an inverse ETF, positions are always taken **LONG**. However, the underlying economic trade is a **SHORT** market bet:

       ```text
       Economic Event: Market / Sector Crash (e.g., XBI drops sharply)
       ─────────────────────────────────────────────────────────────────────────────
       Asset Category        Position Type     Price Motion       Intra-bar Profile Required
       ─────────────────────────────────────────────────────────────────────────────
       Standard Asset (XBI)  SHORT             Price drops        Close near Lows (Ret >= 0.95)
       Inverse ETF    (LABD) LONG              Price surges       Close near Highs (Ret >= 0.85)
       ```

     - **Intraday Expansion & Decay Trap Mitigation**:
       - Inverse ETFs surge during market panic. However, if an inverse ETF exhibits a large upper shadow (indicating the underlying market attempted an intraday bounce), entering long on that bar carries severe decay and bull-trap risks.
       - Long trades on inverse ETFs require bullish candle close ($Close_t > Open_t$), positive upward span ($\text{Span}_{\text{long}} > 0$), and upward retention $\ge \text{DEFAULT\_BEAR\_LONG\_THRESHOLD}$ (0.85):
         $$\frac{Close_t - Close_{t-1}}{High_t - Close_{t-1}} \ge 0.85$$
         The 0.85 threshold (calibrated within the 0.85–0.95 window) ensures entries trigger only when panic expansion holds firmly near highs, while safeguarding viable trade sample volume (`signals >= 5`).
     - **Short Trades on Inverse ETFs**:
       - Economically equivalent to market longs; remain **unfiltered** (`short_mask = None`) with zero-allocation baseline execution.

- **Boundary Conditions**:
  - On bar 0 (where $Close_{t-1}$ is null), directional spans cannot be anchored and evaluate safely to `False`.
  - Where directional span $\le 0.0$ (e.g. submerged sessions where $High_t \le Close_{t-1}$ or gap-ups where $Low_t \ge Close_{t-1}$), evaluates to `False`.


### 2.4 ATR Volatility Regime Filter (`atr_volatility.py`)
Filters entries based on whether prevailing volatility falls within historically normal regimes:
- **Normalized ATR**:
  $$nATR_t = \frac{ATR(14)_t}{Close_t}$$
- **Rolling Percentiles**: Computes 10th and 90th rolling percentiles of $nATR_t$ over the preceding 252 bars.
- **Condition**: Both Long and Short entries are accepted if and only if:
  $$p_{10} \le nATR_t \le p_{90}$$
- Bars with fewer than 252 prior sessions evaluate to `False`.

### 2.5 SMA Trend & Slope Filter (`sma_trend.py`)
Filters entries to trade strictly with the macro trend:
- **Macro Trend**: Computes 200-period Simple Moving Average ($SMA_{200}$).
- **Momentum Slope**: Computes the 20-bar slope:
  $$\text{Slope}_t = SMA_{200}[t] - SMA_{200}[t-20]$$
- **Long Condition**: $Close_t > SMA_{200}[t]$ and $\text{Slope}_t > 0$.
- **Short Condition**: $Close_t < SMA_{200}[t]$ and $\text{Slope}_t < 0$.
- Bars with fewer than 220 prior sessions evaluate to `False`.

---

## 3. High-Performance Execution Integration

### 3.1 Analyzer Orchestration (`src/radar_core/analyzer.py`)
`analyzer.py` governs indicator calculation and dispatch:
1. Resolves `bear_symbols_ = security_repo_.get_bear_symbols(symbols)` once before symbol dispatch.
2. Ingests configuration from `get_settings().rsi_input_filter`.
3. In `process_symbol`, resolves `is_bear_ = symbol_ in bear_symbols_` and forwards to `analyze(..., is_bear=is_bear_)`.
4. Computes TA-Lib RSI once per timeframe across all symbols.
5. Computes Mogalef Bands once per timeframe when RSI band strategies are active.
6. Invokes `get_filter_masks(filter_name, prices_df, is_bear=is_bear)` once per timeframe.
7. Injects `is_input_eligible` directly into `rsi_2b.identify` and `rsi_rc.identify`.

```python
# analyzer.py timeframe loop snippet
filter_name_ = get_settings().rsi_input_filter
is_input_eligible_ = get_filter_masks(filter_name_, prices_df, is_bear=is_bear)
...
if strategies.rsi_2b:
    strategies.rsi_2b.identify(
        symbol,
        timeframe,
        only_long_positions,
        prices_df,
        close_prices_,
        is_input_eligible_,
        verbosity_level,
    )
```

### 3.2 Strategy Entry Point Contract (`identify`)
Strategy `identify(...)` methods across `StrategyABC`, `RsiTwoBands`, `RsiRollerCoaster`, and `MovingAverage` type the entry eligibility parameter as:

```python
is_input_eligible: tuple[np.ndarray | None, np.ndarray | None] | None = None
```

Inside `RsiTwoBands.identify()` and `RsiRollerCoaster.identify()`, the eligibility mask for each position direction is selected via direct 2-way evaluation:

```python
eligible_mask_ = None if is_input_eligible is None else (is_input_eligible[0] if is_long_position_ else is_input_eligible[1])
```

When baseline is configured or omitted, `get_filter_masks(...)` returns `(None, None)`; `eligible_mask_` evaluates directly to `None` without requiring conditional checks in the analyzer.

The selected 1D array (`eligible_mask: np.ndarray | None`) is then dispatched to the decoupled Numba JIT screening and trade extraction kernels.

### 3.3 Numba JIT Screening Kernels (`rsi2b.py`, `rsirc.py`)
The screening kernels evaluate parameter grids entirely within JIT scalar registers:
- `_grid_search_2b_fused`: Evaluates combinations of $(in, out)$.
- `_grid_search_rc_fused`: Evaluates combinations of $(in, over, out)$.

Both kernels accept `eligible_mask: np.ndarray | None`. Inside the loop scanning price bars:

```python
# Inlined scalar eligibility check in Numba kernel
if eligible_mask is not None and not eligible_mask[i_]:
    continue
```

When `eligible_mask is None`, Numba compiles out the branch, ensuring identical performance to unfiltered execution.

### 3.4 Zero-Allocation Baseline Architecture & Asymmetric Execution Matrix

In High-Performance Computing (HPC) and Python/Numba systems, **"zero-allocation baseline"** refers to executing unfiltered code paths without allocating heap memory or introducing array lookups into the CPU execution loop.

#### The Naive Alternative vs. The Zero-Allocation Implementation
If an unfiltered direction (e.g. Long positions on standard equities) were implemented naively, the system would allocate a dummy boolean array of `True`:

```python
# Naive approach: allocates memory just to denote "unfiltered / eligible"
long_mask = np.ones(total_bars, dtype=np.bool_)
```

This naive pattern introduces two severe performance penalties:
1. **Heap Allocation Churn**: For every symbol and timeframe, the system allocates memory buffers, executes array assignments, and manages garbage collection overhead.
2. **CPU Cache & Memory Indexing Overhead**: Inside Numba JIT screening kernels ([`_grid_search_2b_fused`](file:///c:/Development/Repos/local-radar/radar-core/src/radar_core/domain/strategies/rsi2b.py) and [`_grid_search_rc_fused`](file:///c:/Development/Repos/local-radar/radar-core/src/radar_core/domain/strategies/rsirc.py)), high-dimensional parameter grids evaluate between **1,800 and 55,000 combinations**. If an array is passed, the CPU must fetch `mask[bar_index]` from memory or L1 cache on every potential crossover across tens of thousands of parameter passes.

`radar-core` resolves this by using **`None` as a first-class bypass sentinel**:
- When a filter is set to `'none'`, `'baseline'`, `''`, or `None`, [`get_filter_masks`](file:///c:/Development/Repos/local-radar/radar-core/src/radar_core/domain/filters/registry.py) returns `(None, None)` directly: 0 bytes allocated, 0 computations performed.
- Under asymmetric filtering, the unfiltered direction returns `None` (e.g., `(None, short_mask)` for normal assets, or `(long_mask, None)` for bear assets).
- Inside the Numba JIT inner loop:
  ```python
  # Inlined scalar eligibility check in Numba kernel
  if eligible_mask is not None and not eligible_mask[i_]:
      continue
  ```
- When `eligible_mask` is `None`:
  - **Zero Heap Allocation**: Not a single byte of array memory is allocated.
  - **CPU Branch Elimination**: Numba evaluates `eligible_mask is not None` in scalar registers; the branch is eliminated or skipped with zero memory reads.
  - **Exact Baseline Speed**: Strategy execution matches raw unfiltered throughput.

#### Asymmetric Filtering Execution Matrix

| Asset Type | Position | Economic Bet | Market Motion | Eligibility Mask | Execution Mechanics |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Standard Asset** | **Long** | Long Market | Market Rallies | **`None`** | **Zero-allocation baseline (0 bytes, zero CPU memory read overhead)** |
| **Standard Asset** | **Short** | Short Market | Market Falls | `short_mask` | Evaluates candle retention $\ge 0.95$ ($Close_t < Open_t$) |
| **Inverse ETF** | **Long** | Short Market | Market Falls | `long_mask` | Evaluates upward retention $\ge 0.85$ ($Close_t > Open_t$) |
| **Inverse ETF** | **Short** | Long Market | Market Rallies | **`None`** | **Zero-allocation baseline (0 bytes, zero CPU memory read overhead)** |

Whenever an asset direction is unfiltered, `None` is passed directly into the Numba grid search and trade extraction kernels, preserving maximum compute throughput.

### 3.5 Multi-Trade Extraction Kernels
When extracting actual trade sequences for winning setups (`_find_trades_2b`, `_find_trades_rc`), the entry signal bar is checked:

```python
if eligible_mask is not None and not eligible_mask[i_]:
    continue
```

Buffers are pre-allocated using `maximum_trades_ = total_bars_ // 2 + 1` with `np.empty(maximum_trades_, dtype=np.int32)` and sliced views `[:trade_count_]` are returned to eliminate dynamic memory allocations.

---

## 4. Configuration & Database Persistence

### 4.1 Configuration Management (`settings.yml` & `settings.py`)
The active filter is configured in `src/radar_core/settings.yml` under `rsi_input_filter`:

```yaml
# Available options: 'price_action', 'atr_volatility', 'sma_trend', 'none'
rsi_input_filter: 'price_action'
```

- Process-local lazy singleton `Settings` in `src/radar_core/settings.py` ingests this configuration via `get_settings()`.
- Explicit configuration is mandatory ('explicit is better than implicit'); no fallback assumptions are made if omitted.

### 4.2 Database Persistence (`positive_ratios`)
Strategy outputs are persisted transactionally via `RatioRepository`:
- For each evaluated input level $in$, strategies track the best setup (`best_ratios_for_in_`).
- If `net_profit > 0` and `expected_percentage > 0`, the record is added to `positive_ratios_` and batch-upserted to the PostgreSQL `ratios` table.
- The `public.ratios_dashboard` database view computes BI fields, including:
  - `"Gain Prob"`: Strategy win rate ($WinTrades / Signals$).
  - `"Profit vs Change"`: Alpha over asset buy-and-hold:
    $$\text{Profit vs Change} = \frac{\text{Net Profit} - \text{Net Change}}{|\text{Net Change}|}$$

---

## 5. Testing & Verification Status

### 5.1 Test Suite (`tests/domain/filters/test_filters.py`)
Comprehensive unit tests validate the filtering subsystem:
- **Registry Lookup**: Verifies mapping for `'price_action'`, `'atr_volatility'`, and `'sma_trend'`.
- **Zero-Allocation Baseline**: Asserts that `get_filter_masks` with `None`, `''`, `'none'`, or `'baseline'` returns `(None, None)`.
- **Unknown Filter Handling**: Confirms `KeyError` is raised with the registry key list on invalid names.
- **Price Action Directional Retention**:
  - Validates asymmetric directional retention calculations for standard vs. inverse ETF instruments.
  - Confirms standard assets (`is_bear=False`): Long entries evaluate to `None` (unfiltered, zero-allocation baseline), Short entries require $Close < Open$, positive downward span, and $\ge 0.95$ downward retention.
  - Confirms inverse ETFs (`is_bear=True`): Long entries require $Close > Open$, positive upward span, and $\ge 0.85$ upward retention, while Short entries evaluate to `None` (unfiltered).
  - Rejects submerged sessions where price fails to penetrate or hold territory relative to prior close.
  - Validates unanchored bar 0 safely evaluates to `False`.
  - Validates flat-line zero directional spans evaluate to `False`.
- **ATR Volatility Regime**: Validates percentile bounds and history length requirements.
- **SMA Trend & Slope**: Validates uptrend, downtrend, and history threshold guards.

### 5.2 Integration Tests (`tests/test_analyzer.py`)
- Verifies analyzer orchestration, `is_bear` resolution from `SecurityRepository`, and injection of `is_input_eligible` into `identify()`.
- Validates backward compatibility when filters are disabled.

### 5.3 Standalone Evaluation & Database Auditing (`evaluation/`)
The standalone evaluation framework evaluates strategy performance and compares tournament filter arms against baseline:
- **Multi-Arm Benchmarking**: Benchmarks candidate filter regimes against unfiltered baseline (`'baseline'`, `'price_action'`, `'atr_volatility'`, `'sma_trend'`).
- **Asymmetric Filter Integration**: Resolves `is_bear` for each symbol via `SecurityRepository.get_bear_symbols`, forwarding `is_bear` to `get_filter_masks`.
- **Database Auditing Mode (`--from-db`)**: Queries PostgreSQL via `DISTINCT ON (r.symbol, s.acronym, r.timeframe, r.is_long_position)` to fetch champion positive ratios stored in the database, allowing cross-verification between in-memory simulation runs and production database records.
- **Reporting**: Outputs console breakdown and persistent Markdown tournament summaries.

### 5.4 Quality & Linting Verification
- Fully compliant with Python 3.13, Polars, and Numba JIT standards.
- 100% clean check under `auto/lint.cmd` (Ruff).
- All 164 unit tests pass cleanly under `auto/test.cmd`.

