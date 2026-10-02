# TP Probability: July Fit, August Validation, September Evaluation

Experiment dated 2026-10-01, NSE:RELIANCE. **Modest ranking evidence, but no
validated high-precision or profitable policy.** All six alternatives lost money
after modeled costs. These models remain research-only; no live defaults changed.

The policy selected in August for the requested 0.2% TP made 36 September trades,
all SHORT: 18 TP hits (50.0%) against a mean forecast of 58.10%, net -Rs 4,823.48.
The overall August winner had only one selection-period trade. Its September
result was 2/6 TP hits, not a confirmation of the selection-period 1/1.

## Chronology and Data

| Purpose | Dates in 2026 | Sessions |
| --- | --- | ---: |
| Feature warm-up only | July 1 | 1 |
| Base-coefficient fitting | July 2-31 | 22 |
| Feature family and regularization selection | August 3-7 | 5 |
| Separate LONG/SHORT calibration fitting | August 10-21 | 10 |
| Calibration choice, entry threshold and policy selection | August 24-31 | 6 |
| Final evaluation for this run | September 1-28 | 19 |

August does not enter base-coefficient fitting: the
[protocol](protocol.json) sets `refit_model_selection=false`. It does fit the
calibration mappings, which is explicitly separate from the July model. All
choices are serialized by the August 31 boundary in
[selection_lock.json](selection_lock.json), then reloaded before test labels
are computed. No September outcome was used to change this run's choices.

The 63-session tape reuses the hash-verified immutable source from the
[previous study](../tp_probability_2026q3/README.md), rather than a refreshed
broker cache. Every session has all 360 minutes before the 15:15 cutoff and
prices on the Rs 0.10 tick grid. A separate snapshot and
[data_manifest.json](data_manifest.json) preserve provenance. September 14 is
not a session; September 29 remains excluded for its incomplete vendor tail.

**September was already examined in earlier research. This is a chronological
evaluation, not an untouched prospective test.** A month contains many decision
points, but adjacent features and future price paths overlap. The 0.2% model's
July fit contains 1,238 sampled side-actions at 619 timestamps over 22 sessions,
not 1,238 independent market regimes. The 0.3% context model has only 304 sampled
side-actions over 21 sessions after feasibility and five-minute sampling.

## Stops in Percent

The execution rule still uses a finite ATR stop. We have changed its reporting,
not replaced it with a fixed-percent stop. Exact initial distance is:

$$SL_{percent}=100\left|\frac{stop\_price}{entry\_price}-1\right|.$$

For the August-selected 0.2% TP / 2 ATR policy, September's SL was:

- Median **0.259%**, mean **0.275%**, minimum **0.144%**, maximum **0.459%**.
- At an illustrative Rs 1,300 entry, 0.259% is about Rs 3.36 per share,
  versus Rs 2.60 for a 0.2% target, before tick rounding and costs.
- A stop-price distance is not a guaranteed maximum loss: gaps, exit slippage
  and fees can make the realized loss larger.

The per-direction summary includes median/min/max SL percentages; the report
also includes means and actual TP-distance percentages. Exact entry, stop and
target prices remain available for every trade in [trades.csv](trades.csv).

## September Results

These are alternative policies, not additive portfolio returns. Thresholds and
calibration choices below were fixed in August. A blank direction has no trades,
not a 0% or 100% success rate. Each trade uses the existing roughly Rs 100,000
whole-share clip, one-second modeled latency, 2 bp half-spread, 1 bp slippage,
fees, conservative same-candle ambiguity and no entry-candle exit.

| TP | SL rule | Threshold | LONG TP/trades | SHORT TP/trades | Combined TP rate | Mean forecast | Median SL % | Net Rs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15% | 2 ATR | 0.50 | none | 33/71 | 46.48% | 52.97% | 0.215 | -10,436.91 |
| 0.15% | 3 ATR | 0.55 | 12/24 | 69/121 | 55.86% | 58.58% | 0.284 | -20,547.97 |
| **0.20%** | **2 ATR** | **0.55** | **none** | **18/36** | **50.00%** | **58.10%** | **0.259** | **-4,823.48** |
| 0.20% | 3 ATR | 0.60 | none | 22/38 | 57.89% | 63.60% | 0.360 | -5,024.09 |
| **0.30%** | **2 ATR** | **0.55** | **2/6** | **none** | **33.33%** | **59.21%** | **0.285** | **-1,122.08** |
| 0.30% | 3 ATR | 0.45 | 8/21 | 13/28 | 42.86% | 56.45% | 0.415 | -10,667.33 |

Bold rows are the preselected fixed-0.2%-TP and overall policies. Their August
selection outcomes were 4/5 and 1/1 respectively. The 0.2% / 3 ATR row has the
highest September TP rate in this table, but choosing it now would use the test
for selection. The original choices are preserved. The rate-only selection
criterion's one-trade winner exposes fragility; sparse trading is acceptable,
but sparse evidence cannot establish high precision.

Full LONG/SHORT forecasts, SL ranges and P&L are in [policies.csv](policies.csv).
Higher hit rates do not ensure profit: stop losses can exceed target gains and
every completed trade pays costs. The earlier six-session positive example
does not establish that the monthly procedure generalizes.

## Does a Higher Score Mean a Better Chance?

For the selected 0.2% / 2 ATR model, there is a modest positive association.
These are all feasible hypothetical SHORT actions, including overlapping ones,
not just the 36 non-overlapping executed trades:

| Probability band | Actions | Mean forecast | Actual TP rate | TP within 5 min, all actions | Median time among TP hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20-30% | 111 | 26.59% | 26.13% | 0.00% | 17.98 min |
| 30-40% | 435 | 35.85% | 34.94% | 0.46% | 21.98 min |
| 40-50% | 763 | 45.17% | 45.35% | 5.24% | 18.98 min |
| 50-60% | 326 | 53.96% | 49.08% | 13.50% | 8.98 min |
| 60-70% | 22 | 61.24% | 59.09% | 27.27% | 5.98 min |

SHORT ROC-AUC is **0.5768**, LONG **0.5774** (0.5 is chance ordering; 1 is
perfect ordering). Combined Brier loss is **0.23652**, versus **0.24078** for the
frozen per-side base-rate control. These descriptive improvements are small;
no session-clustered significance claim is made.

Calibration is not uniformly good. Across all feasible LONG actions the mean
forecast is **22.57%** but actual TP rate is **33.74%**. SHORT is 43.42% versus
42.25% overall, but the actually selected shorts forecast 58.10% and hit 50.00%.
Thus global calibration, side calibration, ranking and selected-trade precision
are distinct. The 0.3% models have AUC below 0.5 and worse Brier loss than their
controls; those larger-target models do not show comparable substance here.

## The Fitted Functions

The current model is **regularized logistic regression on bounded nonlinear
feature families**, not a neural network. It estimates an executable event,
not a signed return or a time-discounted utility. For direction $d$ (+1 LONG,
-1 SHORT), let $x_d$ contain the direction-adjusted inputs below:

$$z_d=b+\sum_j w_j x_{d,j},\qquad p_d=\sigma(a_d z_d+c_d),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.$$

The signed output is +p_LONG if LONG is larger, -p_SHORT if SHORT is larger;
ties abstain. The 0.55 entry threshold is separate. LONG and SHORT probabilities
are not complements. The original utility model's reachability multiplier is
not applied to this logistic model.

Coefficients below are rounded for reading. The full-precision vector, feature
order and calibrators in [selection_lock.json](selection_lock.json) are canonical.
Both displayed policies use L2 penalty 0.01.

| Input | Selected 0.20% / 2 ATR | Selected overall 0.30% / 2 ATR |
| --- | ---: | ---: |
| Intercept | -1.528884 | -1.911358 |
| Trend alignment | -0.132640 | +0.529734 |
| Momentum | -0.180501 | -0.481600 |
| Directional pressure | +0.225715 | +0.054162 |
| Range position | -0.124339 | +0.022826 |
| Mean reversion | +0.218268 | +0.187697 |
| Trend pullback | +0.130132 | +0.837543 |
| Session pressure | +0.078556 | +0.646502 |
| Volume confirmation | -0.151408 | -0.259631 |
| Volatility relative to TP | +0.554809 | +0.643909 |
| Time remaining | +0.759789 | +0.798442 |
| ADX | +0.246783 | +0.124819 |
| Prior cutoff-close distance | absent | -0.587798 |
| Prior range position | absent | -0.126146 |
| Opening from prior cutoff close | absent | +0.184174 |
| Prior pre-cutoff return | absent | -0.113016 |
| Candle body / ATR | absent | -0.014025 |
| Candle close location | absent | -0.067269 |
| Side (+1/-1) | -0.119259 | +0.179733 |

For the 0.2% model, August calibration gives:

$$p_{LONG}=\sigma(0.548968744z_{LONG}-0.807640775),\qquad
p_{SHORT}=\sigma(1.258156602z_{SHORT}+0.437838511).$$

The overall 0.3% model retains identity calibration: $a_d=1,c_d=0$.

**Feature meanings and philosophy:** the eight core families summarize EMA
alignment/slopes, returns/MACD, DI pressure, range location, RSI/Bollinger/VWAP
stretch, trend pullbacks, session/opening-range pressure and relative volume.
They are side-adjusted, bounded and normalized to avoid incomparable raw units.
The unsigned extras are `tanh(ATR_fraction / TP_fraction)`,
`max(0, 1 - minutes_since_open / 360)` and `min(1, ADX / 100)`.
The context model also uses the previous complete 09:15-15:15 session, not an
official 15:30 daily bar. Exact transforms are in
[scripts/research_tp_policy.py](../../scripts/research_tp_policy.py) and
[src/ai_trader/scanner/opportunity.py](../../src/ai_trader/scanner/opportunity.py).

Raw coefficient size alone is not importance. Using the same equal-day weighted
July sample as fitting, we measured $E[|w_j(x_j-E[x_j])|]$, a centered logit
contribution scale. Leading inputs for the 0.2% policy are:

| Input | Mean absolute centered logit effect |
| --- | ---: |
| Time remaining | 0.1990 |
| Side | 0.1193 |
| Directional pressure | 0.0672 |
| Trend alignment | 0.0622 |
| Volatility relative to TP | 0.0414 |
| Momentum | 0.0410 |

The model principally favors time to reach the barrier, short-side baseline
conditions and some directional pressure, with mild reversion/pullback effects
rather than simply buying strong momentum. This is an interpretation of the
fitted function, not proof of an economic cause. Correlated features substitute
for each other, and per-side calibration rescales their effective logit effects.
The 0.3% context model emphasizes prior-close distance, trend pullback, alignment
and session pressure, but its September failure does not validate that story.

## Probability and Speed

A high probability of hitting TP by 15:15 does **not** imply a fast TP. It can
reflect a longer remaining opportunity window. In the selected 0.2% policy:

- TP within 5 minutes: **6/36 = 16.67%** of all trades.
- TP within 15 minutes: **14/36 = 38.89%**.
- TP within 30 minutes: **17/36 = 47.22%**.
- Eventual TP by cutoff: **18/36 = 50.00%**.
- Median holding time among the 18 successes: **7.48 minutes**.

These are observed diagnostics, not fitted short-horizon probability forecasts.
Higher short probability bands tended to be faster at the top of this sample,
but the timing relationship is not monotonic across all bands and is not
guaranteed by the current objective. One-minute bars and modeled latency also
do not support measured second-level timing claims.

The next explicit target should be
$p_{d,h}=P(TP\text{ first and within }h\text{ minutes}\mid X_t,d)$ for
$h=5,15,30$ and the remaining session. A discrete-time competing-event model
can distinguish TP, SL and still-open paths, with consistent cumulative TP
probabilities. An SL is a competing failure, not independent censoring. Do not
multiply probability by a speed weight and keep calling the result probability.

## Recommendation and Reproduction

Keep this logistic model and frozen base rates as controls. Benchmark a shallow,
regularized boosted-tree model on the same executable labels and chronological
folds before considering a neural replacement. Add explicit horizon outcomes
for speed. The [research notes](../../docs/strategies/README.md#scorer-model-choice)
explain the supporting papers and why order-book neural results do not establish
an optimal RELIANCE candle scorer. No new model was fitted after this test.

```powershell
& .\.venv\Scripts\python.exe -m scripts.research_tp_probability --root backtests/tp_probability_month_split
```

Use `--overwrite` only to regenerate this experiment's own outputs; the frozen
tape is still hash-checked and not overwritten. The script's default root is
the older study, so keep the explicit `--root` above. Full evidence is in
[report.json](report.json), including all alternatives, reliability and speed
bins, per-side outcomes, frozen controls, daily diagnostics and code hashes.
[predictions.csv](../../data/research/tp_probability_month_split/predictions.csv)
records raw, mapped and used probabilities for every feasible September action.

Verification: 961 tests passed, including monthly training isolation, ranking
ties, percentage conversion, timing diagnostics, training contribution scale
and existing cached/real replay equivalence. Full lint and the 109-file format
check passed; editor diagnostics are clear and local evidence links resolve.
The final rerun reproduced choices and results with current source hashes.
No credentials were inspected, no broker calls or orders were made, and no
model was promoted to live use.