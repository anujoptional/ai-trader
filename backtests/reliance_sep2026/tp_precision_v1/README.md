# TP Precision, Thresholds and Exit Research

## Conclusion

The objective is now **actual take-profit exits / completed trades**, not total
speed utility, analytical first-barrier accuracy, or the percentage of profitable
trades. Fewer trades are allowed: no 30-signal minimum is imposed. Sparse
results are retained with their counts, not called statistically reliable.

For the existing signed model, higher score thresholds do **not** reliably give
higher TP-hit rates. At TP 0.20% and a fixed 2-ATR stop, September 4 has 6/8 TP
exits at thresholds 0.020 and 0.025, but only 1/3 at 0.050 and 0/1 at 0.060.
Thresholds above the observed score range simply produce no trades.

Selection on September 2-3 chooses threshold 0.035. Allowing SL tuning while
keeping TP at 0.20% chooses 3 ATR: 5/7 TP hits in tuning, then 3/5 on September 4.
Allowing TP to change chooses TP 0.15%, SL 3 ATR, threshold 0.050: 6/8 in tuning
and 3/4 on September 4. These are very small samples, and both check-day policies
lose money after costs. They are research results, not a validated trading edge.

Two new classifiers trained directly for TP success were also tested, including
six additional candle/prior-session features. Neither estimated enough confidence
to trade under the predeclared rule. Adding features did not establish an
improvement on this check day. Production features and defaults are unchanged.

## What Was Tested

- Data: RELIANCE September 1-4, with August 31 warm-up. Nothing later was scored.
- Current model: the existing eight-component, L2=0.1 signed utility model.
  Each session uses coefficients fitted only on earlier sessions.
- Selection: September 2-3 completed simulated trades. September 4 is a separate
  chronological check, but **was already examined in earlier work**, not a fresh
  untouched test.
- Thresholds: 113 grid points covering 0 through 1 inclusive, with finer steps
  below 0.10. Full output includes all no-trade rows.
- TP: 0.15%, 0.20%, 0.30%. SL: fixed 1, 2, 3 or 4 one-minute ATR at entry.
  This is a bounded grid, not a claim of global optimality.
- Existing-model grid: 1,356 combinations. New-model comparison: another 226
  threshold rows at TP 0.20%, SL 2 ATR. All trials are reported, not just winners.
- Selection ranks TP rate first. Ties prefer more represented sessions, then
  more trades, then smaller SL, larger TP, lower threshold. No minimum trading
  frequency or positive-P&L constraint is used to hide sparse or losing choices.

The [protocol](protocol.json) was written before this experiment's outcomes were
computed. It explicitly replaces the older speed-utility selection objective.
Repeatedly researching these same four days still creates selection bias; a new
protocol does not turn previously examined data into independent evidence.

## Threshold Curve

TP is fixed at 0.20% and SL at 2 ATR throughout this table. Counts mean actual
non-overlapping completed trades, with portfolio suppression and feasibility
applied. A profitable STOP or SESSION_END exit is **not** a TP hit.

| Absolute threshold | Sep2-3 TP/trades | Sep2-3 TP rate | Sep4 TP/trades | Sep4 TP rate | Sep4 net P&L, Rs |
| --- | --- | --- | --- | --- | --- |
| 0 | 12/36 | 33.3% | 8/15 | 53.3% | -1,021.41 |
| 0.0025 | 12/36 | 33.3% | 8/15 | 53.3% | -1,051.84 |
| 0.0050 | 12/36 | 33.3% | 6/11 | 54.5% | -909.65 |
| 0.0075 | 11/30 | 36.7% | 6/11 | 54.5% | -902.05 |
| 0.010 | 10/30 | 33.3% | 6/11 | 54.5% | -917.27 |
| 0.015 | 8/24 | 33.3% | 6/9 | 66.7% | -424.34 |
| 0.020 | 6/19 | 31.6% | 6/8 | 75.0% | -151.32 |
| 0.025 | 5/15 | 33.3% | 6/8 | 75.0% | -151.32 |
| 0.030 | 4/11 | 36.4% | 4/7 | 57.1% | -775.45 |
| 0.035 | 4/8 | 50.0% | 3/5 | 60.0% | -528.62 |
| 0.040 | 3/7 | 42.9% | 3/5 | 60.0% | -528.62 |
| 0.045 | 1/3 | 33.3% | 2/4 | 50.0% | -628.21 |
| 0.050 | 1/3 | 33.3% | 1/3 | 33.3% | -742.83 |
| 0.055 | 1/3 | 33.3% | 1/2 | 50.0% | -295.34 |
| 0.060 | 1/3 | 33.3% | 0/1 | 0.0% | -409.89 |
| 0.065 | 0/1 | 0.0% | 0/1 | 0.0% | -417.52 |
| 0.070, 0.075 | 0/1 | 0.0% | 0/0 | undefined | 0 |
| Every tested point 0.080-1.000 | 0/0 | undefined | 0/0 | undefined | 0 |

The complete [threshold curve](threshold_curve.csv) retains every grid point.
The September 4 score range is approximately -0.00968 to +0.06674. Across the
two tuning days the largest magnitude is 0.07889. A threshold of 0.8 is valid
but selects nothing; rescaling observed scores to fill [0,1] would not create
information or improve ordering.

Raising the threshold makes the set of *eligible raw signals* a subset of the
lower-threshold set. Their quality need not improve if magnitude ranks the wrong
property. The current score estimates signed speed-discounted utility, not TP
success under a 2-ATR stop. Large scores can correspond to already-extended moves
that reverse. With one position per instrument, actual trade sets also depend
on which earlier entries occupy the position, so counts are not merely nested
filters of one fixed trade list.

The attractive 75% September 4 rows were **not** selected using September 4.
Their tuning hit rates were only 31.6%-33.3%. Picking them after seeing the check
day would be another fit to that day's outcome, not a forward improvement.

## TP and SL Tuning

These choices were selected using September 2-3 only:

| Allowed changes | Threshold | TP | SL | Tuning TP/trades | Sep4 TP/trades | Sep4 net P&L |
| --- | --- | --- | --- | --- | --- | --- |
| Threshold only | 0.035 | 0.20% | 2 ATR | 4/8 (50.0%) | 3/5 (60.0%) | Rs -528.62 |
| Threshold and SL, TP fixed | 0.035 | 0.20% | 3 ATR | 5/7 (71.4%) | 3/5 (60.0%) | Rs -870.62 |
| Threshold, TP and SL | 0.050 | 0.15% | 3 ATR | 6/8 (75.0%) | 3/4 (75.0%) | Rs -369.81 |

Full results: [barrier_sweep.csv](barrier_sweep.csv). The check-day mean initial
stop distance is about 0.305%, 0.461% and 0.441% respectively. Median holding
time for successful TP trades is about 1.98 minutes in all three choices; the
minimum reflects the engine's exclusion of entry-candle exits. Losses and full
holding-time distributions remain important and are in [report.json](report.json).

The descriptive Wilson interval for 3/5 is approximately 23.1%-88.2%; for 3/4 it
is 30.1%-95.4%. These already wide intervals assume independent Bernoulli trials.
Trades from a single session are dependent, and the parameters were selected
from many alternatives, so these intervals are **not** guaranteed future bounds.
Low frequency is allowed; low evidence must still be disclosed.

A wider stop and smaller TP can raise hit rate mechanically. For intuition,
a driftless continuous-price model with unlimited time hits an upper barrier
TP before a lower barrier SL with probability `SL / (TP + SL)`. TP=0.2% and
SL=0.6% therefore gives 75% hits even with no predictive advantage; the larger
losses balance the gains before fees. This is a stylized explanation, not a fit
to these candles or a claim that the finite intraday distribution follows it.

The user-requested TP percentage was the selection objective. Net expectancy,
profit factor, drawdown and payoff sizes are reported to expose these trade-offs,
not substituted for that objective. No stop is unlimited, and no position is
carried past 15:15.

## New Features and Strategies

The existing 47-feature catalog is not a constraint on research. This study
tests a different label aligned with the requested objective: for each side,
would the *executable* TP be reached before SL or square-off?

Two small regularized binary logistic models are compared, without a new
hyperparameter search:

1. **TP current:** the eight directional feature families, adjusted for the
   hypothetical side, plus volatility relative to target, time remaining and ADX.
2. **TP context:** those 11 inputs plus six additions: distance to the prior
   session's cutoff close, location in its range, current opening distance from
   that close, its open-to-cutoff return, current candle body/ATR and current
   candle close location. An intercept is fitted in both models.

The prior session frame is the complete 09:15-15:15 period, not the official
15:30 daily close. It becomes available only in a later session. Incomplete
prior frames and unknown session-open context are rejected rather than filled
with invented values. Prefix tests change later prices and verify that earlier
features do not change.

Each model fits binary cross-entropy, L2=0.1, equal session weights, every fifth
minute, both directions, with no class reweighting. Fits use Sep1 for Sep2,
Sep1-2 for Sep3 and Sep1-3 for Sep4. Every label uses production execution
semantics, including fees for reporting, tick rounding, pessimistic ambiguous
bars, entry-bar exclusion and the cutoff. No check-day labels enter fitting.

For each decision, the larger of the two estimated TP probabilities supplies
direction. Confidence is `direction * max(0, 2*p(TP)-1)`, with ties abstaining.
This retains a centered signed output but has **different semantics** from the
old time-discounted utility. Its numerical threshold is not interchangeable
with the old threshold. Probabilities are estimates, not established calibration.

### New-Model Result

Neither model produced eligible trades on the tuning days or the check day
under the predeclared confidence and execution rules. On September 4, all
estimated probabilities were below 40%, so both returned neutral scores. There
is no selected trading policy for either; their TP-hit rate is undefined, not
zero or 100%.

On September 4's 716 hypothetical side/decision outcomes:

| Model | Brier probability error, lower is better | Executed trades |
| --- | --- | --- |
| TP current | 0.19906 | 0 |
| TP context | 0.20631 | 0 |

The overall hypothetical TP base rate is about 28.2%. These 716 labels are two
sides at overlapping minutes, not 716 independent executable trades. The added
context did not improve this check's probability error. This does not establish
that prior-session information is useless, or that a more flexible model cannot
work; it means this small, regularized experiment has not demonstrated it.

Full confidence curves: [tp_model_sweep.csv](tp_model_sweep.csv). Model weights,
prior-day training dates, score/probability ranges and reliability bins are in
[report.json](report.json). Neither model is wired into the live scanner.

## Research Read

These papers informed the experiment. The first three were read in full via
their accessible HTML; for Kaminski and Lo, the published abstract was reviewed.
None supplies a ready-made winning strategy for this instrument.

| Paper | Applicable lesson | Limitation here |
| --- | --- | --- |
| Geifman and El-Yaniv, 2017, [Selective Classification for Deep Neural Networks](https://arxiv.org/html/1705.08500v2) | Measure error against coverage and select a rejection threshold using separate labeled data. Fewer accepted predictions can be desirable. | Their guaranteed-risk argument assumes i.i.d. samples. Correlated minute paths and market regime changes do not satisfy that premise. We do not claim its guarantees. |
| Guo, Pleiss, Sun and Weinberger, ICML 2017, [On Calibration of Modern Neural Networks](https://arxiv.org/html/1706.04599v2) | A sigmoid/softmax output is not automatically a calibrated probability. Check reliability and probability error; use separate calibration data. | Temperature scaling changes confidence, not the underlying directional ranking. Four correlated days are insufficient for a convincing independent calibration exercise. No cosmetic score amplification was applied. |
| Cont, Kukanov and Stoikov, 2014, [The Price Impact of Order Book Events](https://arxiv.org/html/1011.6402v3) | Best-bid/ask order-flow imbalance and depth provide information absent from ordinary OHLCV. Their study finds stronger relationships than trade volume alone. | Their main regressions explain contemporaneous price changes in US equities; they do not establish future TP prediction for RELIANCE. Quote/depth event data is required and cannot be reconstructed from minute candle volume. |
| Kaminski and Lo, 2014, [When do stop-loss rules stop losses?](https://ideas.repec.org/a/eee/finmar/v18y2014icp234-254.html) | Evaluate the effect of stops on both return and risk rather than assume a universal optimal stop. | The published abstract describes daily index futures and longer sampling frequencies, not one-minute single-stock TP probabilities. It does not justify a particular ATR multiple here. |

Priority data/features for a later, separately validated experiment are lagged
order-flow imbalance, bid/ask depth and spread; prior-session volatility and
range regimes; same-time-of-day relative volume over multiple earlier sessions;
and aligned market/sector relative strength. Quote features need historical
quotes; seasonality features need enough prior sessions. No missing history was
fabricated, no external market data was downloaded, and no large model was fitted
to manufacture confidence from four days.

## Execution and Reproduction

Approximately Rs 100,000 per entry, Groww charges, 1-second stated latency,
2bp half-spread and 1bp slippage, fixed ATR stops, and 15:15 square-off are used.
TP is a resting limit at the tick-rounded price; stops and square-off pay
market-exit friction. One-minute data cannot verify second-level fills or the
order of intrabar touches. Ambiguous exits are stops; no entry-bar exit is allowed.

The fast sweep does not implement alternative fill or P&L arithmetic. It obtains
each potential action's trade from the existing `ReplayPortfolio`, then schedules
those trades chronologically with one position per instrument and the same
feasibility and cooldown rules. Fixture comparisons and nine real selected-policy
comparisons matched `ReplayEngine` trades exactly. This scheduler is explicitly
restricted to one instrument. It is research tooling, not a new live engine.

```powershell
.venv/Scripts/python.exe -m scripts.research_tp_policy
```

Existing outputs require `--overwrite`. The command verifies the frozen parent
data, plan and baseline configuration and stores its own source hashes. Earlier
study reports and profiles are not overwritten.

Shared, opt-in configurations fitted only through September 3:

- [Threshold only](threshold_only_strategy.json), TP 0.20%, SL 2 ATR, threshold 0.035.
- [TP fixed, SL tuned](fixed_tp_0.2_percent_strategy.json), TP 0.20%, SL 3 ATR, threshold 0.035.
- [TP and SL tuned](tp_and_sl_strategy.json), TP 0.15%, SL 3 ATR, threshold 0.050.

For alternative TP levels, the old model's weights remain fixed but its target
parameter changes both the score's reachability gate and the execution target,
as required by the shared configuration. These are sensitivity profiles, not
newly calibrated forecasts for those targets. A changed TP therefore need not
retain exactly the same raw score values.

All three profiles are unvalidated. Default strategies are unchanged. No live
orders or broker calls were made. A meaningful next test needs a frozen choice
and new forward development days; repeatedly optimizing Sep1-4 cannot establish
high future TP precision. The previously reserved validation/holdout dates remain
unused for this model.

## Verification

On 2026-09-30: **953 tests passed**, repository lint passed, 104 files passed
the formatting check, and touched-file editor diagnostics were clear. Tests
cover TP-versus-net-win counting, entry-bar exclusion, ambiguous stops, cutoff,
future-label isolation, prior-session feature causality, sparse-policy selection,
and exact cached/engine trade equality. The final artifact run reproduced all
results and recorded the final code hashes. Changes remain uncommitted.