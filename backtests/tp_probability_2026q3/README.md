# Calibrated TP Probability: July-September 2026

## Result

The score now has a probability-preserving research definition: **+p for LONG,
-p for SHORT**, where p estimates the specified action hitting TP before SL or
15:15 square-off. A sigmoid is nonnegative; direction supplies the sign. No
`2*p-1` rescaling or implicit 50% floor is used. Threshold selection is separate.

The policy selected before final evaluation was **TP 0.20%, SL 3 ATR,
threshold 0.60**. On September 21-28 it made seven SHORT trades across four days,
six exited at TP, and net simulated P&L was Rs +406.94. It made no LONG trades.
Its mean forecast was **61.17%**, not the observed **85.71%**. Seven trades are
insufficient to establish a dependable future hit rate or calibration.

The largest selected-trade sample, TP 0.15% / SL 3 ATR, forecast 55.06% and
achieved 56.60% over 53 trades, but lost Rs 5,191.66. Mean probability agreement
does not imply profitable execution. Neither result is promoted to live trading.

Most importantly, the 0.20% TP models did not beat a frozen per-side base-rate
forecast on final-test hypothetical-action Brier loss. The experiment therefore
does **not** establish that the desired accurate conditional probabilities have
been achieved. Forecasts have not been changed after observing the test.

Research sources and actionable takeaways are in
[the strategy notes](../../docs/strategies/README.md). The exact event and scoring
semantics are in [the probability contract](../../docs/strategies/TP_PROBABILITY.md).

## Data and Chronology

The user authorized a broader date range for this study. The audited tape has
63 RELIANCE sessions from July 1 through September 28, with all 360 pre-cutoff
minutes present each day. July-August comes from the existing local cache;
August 31 onward comes from the previously frozen September source. No broker
or credential access was needed. A separate combined snapshot is frozen under
the local research data directory and pinned by [data_manifest.json](data_manifest.json).

| Purpose | Dates | Sessions | What can use these outcomes |
| --- | --- | --- | --- |
| Warm-up | Jul 1-7 | 5 | Feature initialization only |
| Initial fit | Jul 8-Aug 21 | 33 | Logistic coefficients |
| Model selection | Aug 24-31 | 6 | Choose feature family and L2; then refit through Aug31 |
| Calibration | Sep 1-11 | 9 | Fit separate LONG and SHORT probability mappings |
| Policy selection | Sep 15-18 | 4 | Choose identity versus calibrated output, then threshold |
| Final evaluation | Sep 21-28 | 6 | Metrics only; no model, calibration or threshold changes |

The final evaluation dates are Sep21,22,23,24,25,28. September 14 is not treated
as a missing session. Some dates appeared in earlier aggregate backtests, and
Sep1-4 informed earlier feature research. This is a newly separated chronological
probability experiment, **not a pristine prospective test**. The formerly reserved
September validation/holdout periods are now evaluated under the expanded scope.

## Fitting and Calibration

Six event definitions are compared: TP 0.15%, 0.20% or 0.30%, each with a fixed
2-ATR or 3-ATR stop. Every event receives its own labels, fitted coefficients
and calibration; a model trained for one TP is not silently reused as a
probability estimator for another.

Each event compares the 11-input current-feature model and 17-input context
model, both with a side indicator, at L2=0.01 and L2=0.1. That is four initial
models per event, 24 initial model fits in total, plus six selected refits.
All six selections chose the current-feature family; the added prior-session
context did not win the independent August model-selection comparison.

Inputs are fixed normalized transforms; no full-sample normalization. Only
feasible, executable hypothetical actions are fitted, every fifth decision,
with equal session weights and no class rebalancing. Core price indicators
already carry history across sessions. Explicit prior-session features use the
complete 09:15-15:15 frame, not an official 15:30 daily bar.

A monotone two-parameter mapping `sigmoid(a*logit(p)+b)` is fitted separately
for LONG and SHORT on September 1-11. The slope is nonnegative and weakly
regularized toward identity. September 15-18 chooses between this mapping and
the original forecasts by equal-day log loss. Calibration is retained only
when that selection comparison improves; four events retained identity and
two chose the fitted mapping.

For each event, 21 entry thresholds from 0 through 1 in steps of 0.05 are
compared on completed September 15-18 trades. Highest TP rate wins, with the
declared tie breaks. Sparse selections are allowed and zero-trade rates remain
undefined. The selected event, model, mapping and threshold are written to
[selection_lock.json](selection_lock.json) **before** test labels are generated.
Final evaluation reloads that saved lock. No final-test result selects a policy.

## All Long and Short Results

These are alternative policies over the same six test sessions, not a combined
portfolio. Do not sum their trades or P&L as if they were independent systems.
Forecast means are probabilities at entry for the actual selected trades.

| TP | SL | Threshold | Side | Trades | Mean forecast | TP hits / rate | Net P&L, Rs | Mean initial SL distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.15% | 2 ATR | 0.50 | LONG | 0 | n/a | n/a | 0 | n/a |
| 0.15% | 2 ATR | 0.50 | SHORT | 1 | 50.15% | 1/1, 100% | +70.82 | 0.136% |
| 0.15% | 3 ATR | 0.50 | LONG | 8 | 55.00% | 5/8, 62.50% | -1,168.19 | 0.360% |
| 0.15% | 3 ATR | 0.50 | SHORT | 45 | 55.07% | 25/45, 55.56% | -4,023.47 | 0.213% |
| 0.20% | 2 ATR | 0.45 | LONG | 2 | 45.82% | 1/2, 50.00% | -198.01 | 0.307% |
| 0.20% | 2 ATR | 0.45 | SHORT | 13 | 46.37% | 8/13, 61.54% | -418.08 | 0.217% |
| 0.20% | 3 ATR | 0.60 | LONG | 0 | n/a | n/a | 0 | n/a |
| 0.20% | 3 ATR | 0.60 | SHORT | 7 | 61.17% | 6/7, 85.71% | +406.94 | 0.331% |
| 0.30% | 2 ATR | 0 | LONG | 10 | 39.94% | 2/10, 20.00% | -2,423.55 | 0.256% |
| 0.30% | 2 ATR | 0 | SHORT | 0 | n/a | n/a | 0 | n/a |
| 0.30% | 3 ATR | 0 | LONG | 1 | 44.43% | 0/1, 0.00% | -414.97 | 0.298% |
| 0.30% | 3 ATR | 0 | SHORT | 7 | 46.30% | 5/7, 71.43% | -51.89 | 0.449% |

The machine-readable [policies.csv](policies.csv) retains unrounded values.
[trades.csv](trades.csv) includes entry, TP and SL prices, times, direction,
probability, exit reason, fees and P&L for each actual trade. ATR stops have
different percentage distances for different entries; 3 ATR is not 3%.

### Selected 0.20% TP Case

On the four policy-selection days, threshold 0.60 / SL 3 ATR produced 7/9 TP
exits and forecast 61.54% on average. This choice was frozen before testing.
Test trades were 2/2 TP on Sep21, 2/3 on Sep22, none on Sep23/24, 1/1 on Sep25
and 1/1 on Sep28. All were SHORT.

Successful test trades averaged Rs 120.81 net; the one loss was Rs 317.94.
Median time to TP was about 12.48 minutes. Thus the result is not evidence of
near-instant target hits. The descriptive Wilson interval for 6/7 is roughly
48.7%-97.4%, even before allowing for session dependence and model selection.
There is no corresponding LONG execution evidence at this threshold.

## Does the Probability Match?

For all feasible hypothetical actions on the final test, not only selected
trades:

| TP / SL | Actions | Mean forecast | Observed TP rate | Brier | Frozen base-rate Brier |
| --- | --- | --- | --- | --- | --- |
| 0.15% / 2 ATR | 1,772 | 34.08% | 32.34% | 0.21035 | 0.21733 |
| 0.15% / 3 ATR | 1,772 | 46.08% | 43.62% | 0.23394 | 0.23774 |
| 0.20% / 2 ATR | 720 | 36.53% | 35.00% | 0.21850 | 0.21395 |
| 0.20% / 3 ATR | 720 | 45.49% | 44.72% | 0.23467 | 0.22583 |
| 0.30% / 2 ATR | 136 | 39.62% | 43.38% | 0.24868 | 0.24908 |
| 0.30% / 3 ATR | 136 | 43.82% | 50.00% | 0.23347 | 0.24919 |

Brier is mean squared probability error; lower is better. The control is a
constant per-side TP rate estimated from calibration days, not from the test.
Different TP values have different feasibility populations, so their Brier
values should not be used alone to pick an easier event and call it a better
strategy. Hypothetical actions overlap and are not independent trials.

**Combined averages can hide directional miscalibration.** For TP 0.20% / SL
3 ATR, LONG actions forecast 39.38% but hit TP only 26.67% of the time. SHORT
actions forecast 51.59% but hit 62.78%. These errors almost cancel in the
combined 45.49% versus 44.72% average. Reliability-bin errors remain material:
overall expected calibration error is 8.41 percentage points, with different
errors by side. The selected 7-trade result does not repair that finding.

The full [report](report.json) includes raw and mapped probability losses,
side-specific reliability bins, actual-trade calibration, constant controls,
daily results, payoff sizes and holding times. The local per-action prediction
export, [predictions.csv](../../data/research/tp_probability_2026q3/predictions.csv),
records raw, mapped and used probabilities with the corresponding TP outcome;
its hash is stored in the report.

## Reproduction and Limits

```powershell
.venv/Scripts/python.exe -m scripts.research_tp_probability
```

The optional research dependency is required. Existing report outputs require
explicit `--overwrite`; the frozen data snapshot and earlier experiments are
not overwritten. The source hashes and numerical toolchain are recorded.

Execution remains approximately Rs 100,000 per position, INR 0.10 tick,
1-second stated latency, 2bp half-spread, 1bp slippage, Groww charges and 15:15
square-off. These are scenarios, not verified second-level live fills. Labels
reuse `ReplayPortfolio`; the single-instrument scheduler has exact trade-parity
tests against the ordinary engine. Entry-bar touches are not credited,
ambiguous exits are stops, and cutoff exits count as TP failures even if profitable.

These saved models are **research probability policies**, not the old live
StrategyConfig signed-utility profiles. Default scanner behavior is unchanged.
No orders were placed. The promising short case deserves further forward
evaluation, but the requested reliable conditional probability has not yet been
demonstrated. Future research should improve regime/direction discrimination and
validate calibration on new sessions, not tune these final-test probabilities
to reproduce their already-observed outcomes.

## Verification

On 2026-09-30: **958 tests passed**, repository lint passed, 108 files passed
the formatting check, and touched-file editor diagnostics were clear. Focused
tests cover probability-preserving signs, calibration on known outcomes,
later-label exclusion, disjoint phase boundaries, LONG/SHORT reporting, and
the inherited execution-parity tests. The final full-study run reproduced the
same choices and metrics while generating the per-action export and exact
trade TP/SL prices. No live orders, broker calls or commits were made.