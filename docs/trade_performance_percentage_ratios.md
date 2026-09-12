# Trade Performance Percentage Ratios Architecture & Specification

## 1. Executive Summary

This document describes the design, mathematical formulation, and system architecture for the transition from nominal currency points (`expected_value`, `average_win`, `average_loss`) to asset-normalized percentage returns (`expected_percentage`, `average_win_percentage`, `average_loss_percentage`) across the Radar Core platform.

Previously, expectation metrics were evaluated in nominal price points:
$$\text{expected\_value} = P_{\text{win}} \cdot \overline{\text{Win}}_{\$} + P_{\text{loss}} \cdot \overline{\text{Loss}}_{\$}$$

For equities and crypto assets with widely varying price scales (e.g. BTC at $60,000 vs. NVDA at $120 vs. penny stocks at $5), nominal metrics distorted cross-asset comparison and Metabase reporting.

The percentage return subsystem replaces nominal expectation with mathematically rigorous trade return percentages while simultaneously maintaining exact nominal tracking (`winnings`, `losses`) for currency accounting in a single pass.

```text
       ┌────────────────────────────────────────────────────────┐
       │                   Trade Lifecycle                      │
       │           (Entry at Bar t_in, Exit at Bar t_out)       │
       └───────────────────────────┬────────────────────────────┘
                                   │
                     Vectorized Trade PnL Evaluation
                                   │
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
     Nominal Trade PnL                            Return Percentage
  pnl = (out - in)*dir - comm*(in + out)       r_i = pnl / in_price
           │                                               │
           ▼                                               ▼
   Accounting Aggregates                         Performance Ratios
 winnings = sum(pnl[pnl > 0])              winnings_percentage = sum(r_i[r_i > 0])
 losses   = sum(pnl[pnl <= 0])               losses_percentage = sum(r_i[r_i <= 0])
           │                                               │
           └───────────────────────┬───────────────────────┘
                                   ▼
                       Strategy Key Ratios
          net_profit          = (winnings + losses) / first_input_price
          avg_win_percentage  = winnings_percentage / winning_trades
          avg_loss_percentage = losses_percentage / losing_trades
          expected_percentage = P_win * avg_win_percentage + P_loss * avg_loss_percentage
```

---

## 2. Mathematical Formulation

### 2.1 Micro Per-Trade Return
For any individual trade $i$ with entry price $P_{\text{entry}}$, exit price $P_{\text{exit}}$, position direction $d \in \{+1, -1\}$, and proportional commission rate $c = \text{COMMISSION\_PERCENT} = 0.001$ (0.1% per leg):

$$\text{PnL}_i = (P_{\text{exit}} - P_{\text{entry}}) \cdot d - c \cdot (P_{\text{entry}} + P_{\text{exit}})$$

The normalized return percentage $r_i$ is computed relative to entry capital:

$$r_i = \frac{\text{PnL}_i}{P_{\text{entry}}}$$

### 2.2 Segmented Averages
Let $\mathcal{W} = \{i : \text{PnL}_i > 0\}$ be the set of winning trades and $\mathcal{L} = \{i : \text{PnL}_i \le 0\}$ be the set of losing trades.

The average win percentage and average loss percentage are:

$$\overline{\text{Win}}_{\%} = \begin{cases} \frac{1}{|\mathcal{W}|} \sum_{i \in \mathcal{W}} r_i & \text{if } |\mathcal{W}| > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

$$\overline{\text{Loss}}_{\%} = \begin{cases} \frac{1}{|\mathcal{L}|} \sum_{i \in \mathcal{L}} r_i & \text{if } |\mathcal{L}| > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

### 2.3 Mathematical Expectation (Expected Percentage)
Given total signal count $N = |\mathcal{W}| + |\mathcal{L}|$, win probability $P_{\text{win}} = \frac{|\mathcal{W}|}{N}$, and loss probability $P_{\text{loss}} = \frac{|\mathcal{L}|}{N}$:

$$\text{expected\_percentage} = P_{\text{win}} \cdot \overline{\text{Win}}_{\%} + P_{\text{loss}} \cdot \overline{\text{Loss}}_{\%}$$

When $N = 0$, all ratios evaluate to $0.0$.

### 2.4 Payoff Ratio
In database views and reporting, the payoff ratio (profit/loss ratio) measures the magnitude of winning trades against losing trades:

$$\text{Payoff Ratio} = \frac{\overline{\text{Win}}_{\%}}{|\overline{\text{Loss}}_{\%}|}$$

---

## 3. High Performance Computing & Numba JIT Integration

### 3.1 Scalar Helpers (`src/radar_core/domain/strategies/_kernel_helpers.py`)
All screening math is centralized in inlined, pure scalar Numba functions:
- `_finalize_screening_metrics`: Accepts dual-tracked aggregates `(signals, first_input_price, winnings, winning_trades, losses, losing_trades, winnings_percentage, losses_percentage)` and computes `(net_profit_, win_probability_, loss_probability_, average_win_percentage_, average_loss_percentage_, expected_percentage_)`.
- `_is_profitable_candidate`: Ensures both `net_profit > 0.0` and `expected_percentage > 0.0`.
- `_is_better_candidate`: Enforces hierarchical candidate ordering:
  1. Primary: `net_profit > best_net_profit`
  2. Secondary: `net_profit == best_net_profit and expected_percentage > best_expected_percentage`

### 3.2 Fused Grid Search Kernels (`rsi2b.py`, `rsirc.py`)
In `_grid_search_2b_fused` and `_grid_search_rc_fused`:
- The trade return percentage `return_percentage_ = pnl_ / input_price_` is computed **only upon trade exit** (10–50 times per candidate lifecycle), introducing zero measurable CPU overhead into the 55,000 parameter inner loop.
- Scalar registers accumulate both nominal (`winnings_`, `losses_`) and percentage (`winnings_percentage_`, `losses_percentage_`) sums.
- Parity between baseline and fused execution remains bit-identical with high speedups (e.g. 17.5x on weekly SPY data).

---

## 4. Database & Persistence Layer

### 4.1 Schema Migration (`database/migrations/2026-09-12_replace_expected_value_with_expected_percentage.sql`)
The PostgreSQL table `public.ratios` was updated via `ALTER TABLE ... RENAME COLUMN`:
```sql
ALTER TABLE public.ratios RENAME COLUMN expected_value TO expected_percentage;
ALTER TABLE public.ratios RENAME COLUMN average_win TO average_win_percentage;
ALTER TABLE public.ratios RENAME COLUMN average_loss TO average_loss_percentage;
```

### 4.2 Metabase View (`database/02_radar_views.sql`)
The canonical view `public.ratios_dashboard` exposes 4-decimal precision for percentage expectation metrics and adds `"Payoff Ratio"`:
```sql
ROUND(ratios_cte.expected_percentage::numeric, 4)     AS "Expected Value",
ROUND(ratios_cte.average_win_percentage::numeric, 4)  AS "Average Gain",
ROUND(ratios_cte.average_loss_percentage::numeric, 4) AS "Average Loss",
ROUND(ABS(ratios_cte.average_win_percentage / NULLIF(ratios_cte.average_loss_percentage, 0))::numeric, 2)
                                                      AS "Payoff Ratio",
```
Existing column aliases `"Expected Value"`, `"Average Gain"`, and `"Average Loss"` are preserved so Metabase cards and dashboards continue working without query rewrites, requiring only a field suffix configuration change from `($)` to `(%)`.

### 4.3 Data Model (`src/radar_core/models/ratios.py`)
The `Ratios` dataclass defines:
```python
expected_percentage: float = 0.0
win_probability: float = 0.0
loss_probability: float = 0.0
average_win_percentage: float = 0.0
average_loss_percentage: float = 0.0
```
Because `RATIOS_PAYLOAD_COLUMNS` is dynamically derived from `fields(Ratios)`, `RatioCrud` automatically generates parameterized SQL targeting the renamed database columns without manual schema synchronization.

---

## 5. Evaluation Framework (`evaluation/`)

The multi-arm filtering evaluation framework evaluates candidate strategies using the percentage-based hierarchy:
1. `ArmResult`: Stores `expected_percentage` alongside `net_profit` and `win_probability`.
2. `CellResult`: Stores `delta_expected_percentage` between champion and baseline arms.
3. `is_better_arm`: Evaluates superiority under `net_profit` (primary) $\to$ `expected_percentage` (secondary) $\to$ `win_probability` (tertiary).
4. `report_generator`: Outputs formatted markdown tables with `Exp Return` columns and `Δ Expected Return` differential callouts.

---

## 6. Verification and Regression Testing

The implementation is verified via the automated test suite:
- `tests/domain/strategies/test_kernel_helpers.py`: Unit tests for screening math and tie-breakers.
- `tests/domain/strategies/test_base_strategy.py`: Vectorized trade evaluation and key ratios computation.
- `tests/domain/strategies/test_rsi2b_fused_parity.py`: Bit-identical candidate extraction and metric parity between baseline and fused RSI-2B.
- `tests/domain/strategies/test_rsirc_fused_parity.py`: Bit-identical candidate extraction and metric parity between baseline and fused RSI-RC.
- `tests/evaluation/test_evaluator.py`: Multi-arm tournament ranking and reporting verification.
- `tests/infrastructure/crud/test_ratio_crud.py`: Database model and repository CRUD integration.
