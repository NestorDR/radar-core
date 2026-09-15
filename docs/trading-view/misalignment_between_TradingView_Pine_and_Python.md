## Misalignments between TradingView Pine and Python

### 1. Separate upper/lower multipliers vs. one shared multiplier

The Pine script exposes two independent inputs:

```
upperMultiplier = input.float(2.0, ...)
lowerMultiplier = input.float(2.0, ...)
```


It calculates:

```
activeUpperBand := linearRegression + upperMultiplier * regressionStdev
activeLowerBand := linearRegression - lowerMultiplier * regressionStdev
```


Python accepts only one `multiplier` and applies it to both bands:

```python
current_upper_level_ = linear_regression_value_ + multiplier * standard_deviation_value_
current_lower_level_ = linear_regression_value_ - multiplier * standard_deviation_value_
```


Therefore, the implementations produce different results whenever Pine is configured with unequal upper and lower multipliers. With the default values of `2.0` and `2.0`, there is no difference.

### 2. Multiplier validation differs

Pine requires both multipliers to be at least `0.1`:

```
minval = 0.1
```


Python allows `multiplier = 0.0` because it only rejects negative values.

This means Python accepts configurations that cannot be selected in the Pine UI. The Python implementation also uses a single shared multiplier, so it cannot fully represent Pine’s input contract.

### 3. Pine maintains and plots a central equilibrium line

Pine maintains:

```
activeMidLine := linearRegression
```


only when the corridor is initialized or reset, then plots that stepped central line.

Python calculates only:

- `MogalefUpper`
- `MogalefLower`

It does not return the equivalent stepped middle line. This is intentional according to the project contract, but it means the Python output is not a complete visual equivalent of the Pine indicator.

### 4. Different behavior when either input series contains an intermittent `NaN`

Python explicitly skips a row if either the regression or standard deviation is `NaN`:

```python
if np.isnan(linear_regression_value_) or np.isnan(standard_deviation_value_):
    continue
```


The output for that row remains `NaN`, later converted to Polars `null`.

Pine only checks whether the active upper band is `na` or whether the regression has broken outside the corridor. If the corridor has already been initialized and a later `linearRegression` or `regressionStdev` becomes `na`, Pine’s behavior is not identical to Python’s explicit skip-and-output-null behavior:

- existing Pine levels may continue to plot because the `var` values persist;
- a breakout condition involving a valid regression and `na` active levels can potentially assign `na` to the active levels;
- Python preserves the previous levels internally but emits `null` for the invalid row.

For the normal TA-Lib warm-up sequence, where `NaN` values occur only before the first valid result, this difference generally does not affect the output. It matters for intermittent missing or invalid data.

### 5. Initialization condition is structurally different

Pine initializes based on the upper band being `na`:

```
if na(activeUpperBand) or (...)
```


Python uses an explicit boolean:

```python
initialized_ = False
```


For normal valid inputs, these are equivalent. Python is more explicit and safely requires both regression and standard deviation to be valid before initialization. Pine may attempt initialization while one of the calculated inputs is still `na`, although the initial warm-up assignments remain `na` until valid values are available.

### 6. Output handling differs during warm-up and invalid rows

Python initializes its output arrays with `NaN` and converts those values to Polars `null`:

```python
pl.Series("MogalefUpper", upper_band_).fill_nan(value=None)
```


TradingView represents unavailable values as Pine `na` and plots gaps or unavailable values according to Pine’s plotting behavior.

This is conceptually equivalent for the initial warm-up period, but not necessarily for later intermittent invalid rows because Python explicitly emits nulls on those rows while Pine can continue plotting previously active levels.

## Logic that is aligned

The core stepped-corridor recurrence is otherwise aligned:

- Both use the weighted price:

```plain text
(Open + High + Low + 2 * Close) / 5
```


- Both calculate linear regression with a lookback of `3` by default.
- Both calculate standard deviation over `7` periods by default.
- Both support scale-invariant logarithmic calculation (`log_scale=True` in Python, `useLogScale=true` in Pine Script) by default, computing $\ln(CP)$ and exponentiating back via $\exp()$, which prevents negative lower bands and trapped corridors on decaying assets (e.g. `DUST`, `SOXS`).
- Both initialize the corridor on the first valid calculation.
- Both hold the band levels horizontally until a breakout.
- Both reset when the regression is strictly above the upper band or strictly below the lower band.
- Both calculate the new levels around the current regression value.
- Both use a population-style standard deviation configuration in their respective indicator APIs under the normal defaults.

## Main practical conclusion

With default settings and normal OHLC data, the upper and lower stepped bands are equivalent.

The material differences are:

1. Pine supports independent upper and lower multipliers; Python accepts a single multiplier applied to both bands.
2. Pine requires multipliers of at least `0.1`; Python allows zero.
3. Pine plots a stepped middle line; Python intentionally omits it (preserves the strict two-column contract `MogalefUpper` and `MogalefLower`).
4. Intermittent `NaN` values after initialization are handled differently.
5. Both implementations now support scale-invariant logarithmic calculation (`log_scale`), ensuring `MogalefLower > 0` and preventing negative stop loss levels on high-volatility assets.