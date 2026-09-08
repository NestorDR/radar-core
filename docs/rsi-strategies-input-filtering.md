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

### 2.1 Base Abstraction (`base.py`)
The abstract base class defines the filter contract:

```python
class FilterABC(ABC):
    @abstractmethod
    def get_masks(self, prices_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates 1D boolean NumPy masks for Long and Short trade entries.
        :param prices_df: Polars DataFrame containing OHLCV price series.
        :return: Tuple of (long_mask, short_mask) of length equal to len(prices_df).
        """
```

### 2.2 Filter Registry & Factory (`registry.py`)
`registry.py` provides centralized discovery and mask computation:
- **`FILTER_REGISTRY`**: Maps registered string identifiers to filter classes:
  - `'price_action'`: `PriceActionFilter`
  - `'atr_volatility'`: `AtrVolatilityFilter`
  - `'sma_trend'`: `SmaTrendFilter`
- **`get_filter(filter_name: str, **kwargs)`**: Instantiates the requested filter class or raises `KeyError` with available filters.
- **`get_filter_masks(filter_name: str | None, prices_df: pl.DataFrame)`**: Factory method that handles baseline queries (`None`, `''`, `'none'`, `'baseline'`) by returning `(None, None)` without array allocation.

### 2.3 Price Action Filter with True Range (`price_action.py`)
Upgraded to evaluate candle confirmation against **True Range (TR)** rather than nominal High-Low range to account for overnight gaps:

$$\text{TR}_t = \max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)$$

$$\text{TrueLow}_t = \min(Low_t, Close_{t-1})$$

$$\text{TrueHigh}_t = \max(High_t, Close_{t-1})$$

- **Long Entry Eligibility**:
  - Requires bullish candle close: $Close_t > Open_t$
  - Requires close in the upper portion of the True Range:
    $$\frac{Close_t - \text{TrueLow}_t}{\text{TR}_t} \ge \text{long\_threshold} \quad (\text{default: } 0.60)$$
- **Short Entry Eligibility**:
  - Requires bearish candle close: $Close_t < Open_t$
  - Requires close in the lower portion of the True Range:
    $$\frac{Close_t - \text{TrueLow}_t}{\text{TR}_t} \le \text{short\_threshold} \quad (\text{default: } 0.25)$$
- **Boundary Conditions**:
  - On bar 0 (where $Close_{t-1}$ is null), falls back to the nominal bar range: $\text{TR}_0 = High_0 - Low_0$.
  - Where $\text{TR}_t == 0.0$, evaluates to `False`.

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
1. Ingests configuration from `get_settings().rsi_input_filter`.
2. Computes TA-Lib RSI once per timeframe across all symbols.
3. Computes Mogalef Bands once per timeframe when RSI band strategies are active.
4. Invokes `get_filter_masks(filter_name, prices_df)` once per timeframe.
5. Injects `is_input_eligible=(long_mask, short_mask)` into `rsi_2b.identify` and `rsi_rc.identify`.

```python
# analyzer.py timeframe loop snippet
is_input_eligible_ = get_filter_masks(rsi_input_filter_, prices_df)
...
if strategies.rsi_2b:
    strategies.rsi_2b.identify(
        symbol,
        timeframe,
        only_long_positions,
        prices_df,
        close_prices_,
        percent_changes_,
        is_input_eligible_,
        verbosity_level,
    )
```

### 3.2 Numba JIT Screening Kernels (`rsi2b.py`, `rsirc.py`)
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

### 3.3 Multi-Trade Extraction Kernels
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
- If `net_profit > 0` and `expected_value > 0`, the record is added to `positive_ratios_` and batch-upserted to the PostgreSQL `ratios` table.
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
- **Price Action True Range**:
  - Validates gap detection and True Range calculation.
  - Confirms Long eligibility requires $Close > Open$ and upper range close.
  - Confirms Short eligibility requires $Close < Open$ and lower range close.
  - Validates bar 0 fallback behavior when prior close is absent.
  - Validates flat-line zero range ($TR = 0$) evaluates to `False`.
- **ATR Volatility Regime**: Validates percentile bounds and history length requirements.
- **SMA Trend & Slope**: Validates uptrend, downtrend, and history threshold guards.

### 5.2 Integration Tests (`tests/test_analyzer.py`)
- Verifies analyzer orchestration and injection of `is_input_eligible` into `identify()`.
- Validates backward compatibility when filters are disabled.

### 5.3 Quality & Linting Verification
- Fully compliant with Python 3.13, Polars, and Numba JIT standards.
- 100% clean check under `auto/lint.cmd` (Ruff).
- All 145 unit tests pass cleanly under `auto/test.cmd`.

