# How Radar Core Strategies Try to Find Opportunities

## 1. Introduction

Radar Core evaluates four different ways of interpreting price and momentum movements:

- Simple Moving Average (SMA);
- RSI with an RSI moving average;
- RSI Two Bands; and
- RSI Rollercoaster.

These strategies are pattern-based methods. They do not predict the future with certainty. Each one defines a situation that has historically preceded a favorable movement, then measures how often that situation produced a profitable result.

The strategies can be used to examine long positions and, when an asset is configured as shortable, short positions.

```text
Historical market data
          ↓
Identify a repeatable pattern
          ↓
Open when the pattern appears
          ↓
Close when the pattern ends or risk protection is reached
```

The most important difference between the strategies is the kind of market movement they are waiting to see.

## 2. The Four Ideas at a Glance

| Strategy | Main question |
|---|---|
| SMA | Has price changed direction relative to its average? |
| RSI SMA | Has short-term momentum changed relative to its recent momentum? |
| RSI Two Bands | Has RSI recovered from a weak or strong area, and can it travel to the opposite band? |
| RSI Rollercoaster | Has RSI recovered, reached an extreme, and then completed the expected reversal? |

The strategies become progressively more selective:

```text
SMA       → one price-versus-average movement
RSI SMA   → one momentum-versus-average movement
RSI 2B    → RSI input movement followed by an RSI output movement
RSI RC    → RSI input movement, intermediate extreme, then output movement
```

## 3. Simple Moving Average Strategy

### 3.1 The basic idea

A moving average represents the average price over a selected number of recent periods. It smooths out individual daily or weekly movements and gives a reference for judging whether price is moving above or below its recent trend.

The SMA strategy watches for price to cross that reference:

```text
Price below its average → Price crosses above its average
                                        ↓
                              Possible long opportunity
```

The idea is that a move from below the average to above it may indicate that upward momentum is beginning. The opposite movement may indicate that downward momentum is beginning.

### 3.2 Long positions

A long opportunity is identified when price moves from at or below its moving average to above it.

In plain terms:

1. Price has not been stronger than its recent average.
2. Price crosses upward through that average.
3. The strategy opens a long position.
4. The position closes if price later crosses back below the average.

The strategy therefore attempts to participate in an upward movement while it remains above the selected average.

### 3.3 Short positions

A short opportunity is identified when price moves from at or above its moving average to below it.

In plain terms:

1. Price has not been weaker than its recent average.
2. Price crosses downward through that average.
3. The strategy opens a short position.
4. The position closes if price later crosses back above the average.

The strategy therefore attempts to participate in a downward movement while price remains below the selected average.

### 3.4 What this strategy is trying to capture

The SMA strategy is a simple trend-following approach. It does not try to buy at the lowest price or sell at the highest price. It waits for evidence that price has moved to the other side of its recent average.

The choice of period determines the type of movement being observed:

- a short period reacts quickly but may produce many false changes;
- a long period reacts slowly but may ignore short-lived movements.

Radar Core evaluates multiple periods and measures which ones have produced the most favorable historical results for each asset and timeframe.

### 3.5 Strengths and limitations

The main strengths are simplicity and transparency. A reader can easily understand why an input or output occurred.

The main limitation is delay. A crossing confirms that a movement has already started, so the position may open after part of the movement has occurred. In a sideways market, price may cross the average repeatedly without developing a sustained trend.

## 4. RSI with a Moving Average

### 4.1 The basic idea

RSI, the Relative Strength Index, describes the recent balance between upward and downward price movements. It is expressed on a scale from 0 to 100.

RSI does not directly measure the price level. It measures the strength and direction of recent price changes:

- a rising RSI indicates that recent upward movements are becoming stronger;
- a falling RSI indicates that recent downward movements are becoming stronger.

This strategy compares RSI with its own moving average instead of comparing price with a moving average.

```text
RSI below its average → RSI crosses above its average
                                      ↓
                           Possible long opportunity
```

The strategy is asking whether momentum has changed relative to its recent behavior.

### 4.2 Long positions

A long opportunity is identified when RSI crosses upward through its moving average.

The interpretation is:

1. Momentum has recently been weaker than its own average.
2. Momentum begins to strengthen.
3. RSI crosses above its average.
4. The strategy opens a long position.
5. The position closes when RSI later crosses back below its average.

This can identify a change in momentum before a large price trend is obvious.

### 4.3 Short positions

A short opportunity is identified when RSI crosses downward through its moving average.

The interpretation is:

1. Momentum has recently been stronger than its own average.
2. Momentum begins to weaken.
3. RSI crosses below its average.
4. The strategy opens a short position.
5. The position closes when RSI later crosses back above its average.

### 4.4 What this strategy is trying to capture

The RSI SMA strategy attempts to identify changes in momentum rather than changes in the absolute price trend. Two securities can have very different prices but produce comparable RSI values.

The RSI period is fixed at 14 in the current implementation. The period of the moving average over RSI is evaluated over a range of values. A shorter RSI average reacts faster, while a longer one requires a more persistent change in momentum.

### 4.5 Strengths and limitations

The strategy can identify momentum changes while the price is still near a recent range. It may therefore react differently from a price/SMA strategy.

Its main limitation is that RSI can change direction frequently in a sideways market. A crossing may represent only a short-lived change rather than the beginning of a sustained movement.

## 5. RSI Two Bands Strategy

### 5.1 The basic idea

The Two Bands strategy uses two RSI levels instead of comparing RSI with an average:

- an input level, where the position begins; and
- an output level, where the position ends.

The strategy looks for a recovery from one area of the RSI scale and expects that recovery to travel far enough to reach another area.

```text
RSI crosses input → Position opens → RSI crosses output → Position closes
```

The input and output levels are selected as a pair. Different pairs represent different expectations about how far RSI should move during a successful trade.

### 5.2 Long positions

For a long position, the input level is lower and the output level is higher.

The idea is:

1. RSI has been relatively weak.
2. RSI recovers and crosses upward through the lower input level.
3. The strategy opens a long position.
4. RSI continues upward toward the higher output level.
5. When RSI later falls back through the output level, the strategy closes the position.

This does not mean that RSI must reach the maximum possible value. It means that the selected output level represents the point at which the expected upward phase is considered complete.

Example:

```text
RSI moves below 30 → crosses above 30 → long position
                                   ↓
                     rises toward 70, then falls below 70
                                   ↓
                              Position closes
```

### 5.3 Short positions

For a short position, the input level is higher and the output level is lower.

The idea is:

1. RSI has been relatively strong.
2. RSI weakens and crosses downward through the higher input level.
3. The strategy opens a short position.
4. RSI continues downward toward the lower output level.
5. When RSI later rises back through the output level, the strategy closes the position.

Example:

```text
RSI moves above 70 → crosses below 70 → short position
                                    ↓
                      falls toward 30, then rises above 30
                                    ↓
                               Position closes
```

### 5.4 What this strategy is trying to capture

The Two Bands strategy tries to capture a complete RSI swing between two selected levels. The input crossing indicates that the initial recovery or decline has begun. The output crossing indicates that the expected swing has ended.

Compared with the RSI SMA strategy, Two Bands uses absolute RSI zones rather than RSI's position relative to its own average. It can therefore distinguish a recovery from a low RSI area from a general momentum change.

### 5.5 Stop-loss protection

The strategy also has a price-based stop-loss. For a long position, the stop-loss is based on a lower price band. For a short position, it is based on an upper price band.

These bands are calculated using Mogalef Bands, which create a moving price corridor. If the price breaches the relevant stop-loss before the RSI output completes the expected lifecycle, the position is closed as a loss.

The stop-loss and RSI output are therefore two alternative ways for a Two Bands position to end:

```text
Accepted input
      │
      ├─► RSI output reached → normal close
      │
      └─► Price stop-loss reached first → loss close
```

### 5.6 Strengths and limitations

The two-level structure is more specific than a single RSI moving-average crossing. It expresses an expected journey for RSI rather than only a change in direction.

Its main limitation is that the selected levels strongly affect the number and type of trades. A shallow pair may produce many signals with small movements. A wide pair may produce fewer signals and require a much larger RSI swing.

## 6. RSI Rollercoaster Strategy

### 6.1 The basic idea

The Rollercoaster strategy is a staged version of the RSI swing idea. It uses three RSI levels:

- input level;
- overbought or oversold intermediate level; and
- output level.

The intermediate level acts as a confirmation that the initial RSI recovery or decline developed far enough to enter the next stage of the strategy.

```text
Input crossing → Intermediate extreme → Output crossing
       ↓
   Open trade
```

The strategy does not consider the initial input crossing alone sufficient to complete the expected opportunity. RSI must first reach the intermediate extreme.

### 6.2 Long positions

For a long position, the strategy expects RSI to move from relative weakness to overbought territory and then turn downward.

The idea is:

1. RSI has been weak.
2. RSI crosses upward through the input level.
3. The strategy opens a long position.
4. RSI continues upward and reaches the overbought level.
5. The upward phase is considered confirmed.
6. RSI later falls through the output level.
7. The strategy closes the long position.

In plain language, the strategy attempts to participate in a complete upward RSI journey, from recovery through strong momentum, and exits when that strong momentum has meaningfully weakened.

```text
Weak RSI → recovery above input → reaches overbought
                                             ↓
                                  falls below output
                                             ↓
                                         Exit long
```

### 6.3 Short positions

For a short position, the strategy expects RSI to move from relative strength to oversold territory and then turn upward.

The idea is:

1. RSI has been strong.
2. RSI crosses downward through the input level.
3. The strategy opens a short position.
4. RSI continues downward and reaches the oversold level.
5. The downward phase is considered confirmed.
6. RSI later rises through the output level.
7. The strategy closes the short position.

```text
Strong RSI → decline below input → reaches oversold
                                            ↓
                                  rises above output
                                            ↓
                                       Exit short
```

### 6.4 What this strategy is trying to capture

The Rollercoaster strategy tries to avoid treating every input crossing as a complete opportunity. It waits for evidence that the movement became substantial enough to reach the intermediate extreme.

For a long position, the expected sequence is:

```text
Weakness → recovery → strong upward momentum → loss of that momentum
```

For a short position, the expected sequence is:

```text
Strength → decline → strong downward momentum → loss of that momentum
```

The output level is used after the intermediate extreme to identify when the final phase of the movement has ended.

### 6.5 Stop-loss protection and lifecycle assumption

The Rollercoaster uses the same Mogalef-based price stop-loss concept as Two Bands during the initial phase of the lifecycle. If price reaches the applicable stop-loss before RSI reaches the intermediate overbought or oversold level, the position is closed as a loss.

The strategy's intended assumption is that once RSI reaches the intermediate level before that stop-loss condition, the expected lifecycle has been validated and proceeds toward the output crossing. This is a deliberate part of the strategy concept.

If RSI never reaches the intermediate level and no stop-loss closes the position, the position remains open for end-of-period valuation. Likewise, if the intermediate level is reached but the final output crossing has not occurred by the end of the data, the position remains open for valuation.

### 6.6 Strengths and limitations

The intermediate level makes Rollercoaster more selective than Two Bands. It can avoid treating a small RSI recovery as a completed upward or downward cycle.

The trade-off is that many initial inputs may never reach the intermediate level. The strategy may therefore produce fewer trades and may enter a position that remains open for a long time while waiting for the expected stages to occur.

## 7. How the Strategies Differ

### 7.1 The reference used for the signal

```text
SMA       watches price against an average price
RSI SMA   watches RSI against its average RSI
RSI 2B    watches RSI move between two absolute levels
RSI RC    watches RSI complete a three-stage movement
```

### 7.2 The type of market behavior each one seeks

| Strategy | Intended behavior |
|---|---|
| SMA | A change from below to above, or above to below, a price trend reference |
| RSI SMA | A change in short-term momentum |
| RSI Two Bands | A directional RSI swing between an input and output zone |
| RSI Rollercoaster | A larger, confirmed RSI cycle with an intermediate extreme |

### 7.3 Selectivity and timing

The SMA and RSI SMA strategies can produce an input as soon as a crossing occurs. Two Bands requires the input and output levels to describe a meaningful RSI swing. Rollercoaster adds another required stage and is consequently more selective.

Greater selectivity does not automatically mean better performance. It may improve the quality of accepted trades, but it can also reduce the number of opportunities and make results less statistically reliable.

## 8. How Radar Core Compares the Strategies

Radar Core applies each strategy to historical daily and weekly price data. It evaluates multiple periods or RSI-level combinations, then measures the resulting trade lifecycles.

The comparison includes:

- how many opportunities were found;
- how many positions were profitable;
- the average size of wins and losses;
- the total result after commission;
- the expected value of a trade; and
- how much time positions remained active.

The system keeps strategy configurations with positive net profit and positive expected value. The configuration with the highest net profit is treated as the best candidate for a given direction, with expected value breaking ties.

These results describe historical behavior. A strategy configuration that performed well in the past is not guaranteed to perform well in the future.

## 9. Common Risks and Trade-offs

### 9.1 False signals

Markets do not always develop sustained trends or complete expected RSI movements. A crossing can occur and then reverse immediately.

### 9.2 Delayed signals

Every strategy waits for some evidence before opening. That evidence can improve signal quality, but it means the position may not open at the beginning of the movement.

### 9.3 Fewer signals

Adding conditions, especially an intermediate Rollercoaster level or a future input filter, reduces the number of accepted positions. Fewer observations make performance statistics less reliable.

### 9.4 Market changes

A configuration that worked during a trending period may not work during a range-bound period. The reverse can also occur.

### 9.5 Historical selection risk

Radar Core evaluates many periods and level combinations. Selecting the best historical result can overstate how well that configuration will perform on new data. Walk-forward and out-of-sample evaluation are needed to assess whether the pattern is stable.

## 10. Brief Technical Reference

The user-facing ideas above are implemented using three layers:

1. Price data and technical indicators are prepared in Polars DataFrames.
2. High-volume signal and lifecycle calculations run on NumPy arrays through Numba-compiled functions.
3. Python orchestration profiles the trades, creates `Ratios` results, and persists positive configurations.

The main implementation components are:

```text
MovingAverage     →  SMA and RSI SMA
RsiTwoBands       →  RSI Two Bands
RsiRollerCoaster  →  RSI Rollercoaster
RsiStrategyABC    →  shared RSI stop-loss preparation
```

RSI(14) is calculated once per timeframe and shared by the RSI strategies. Mogalef Bands are also calculated once per timeframe when either RSI band strategy is enabled. The detailed filtering and lifecycle rules are then applied separately to each strategy's own signal process.
