# Signed Intraday Opportunity Score: September 1-4

## Conclusion

The requested signed, time-aware score is implemented as an **opt-in research
model**. It is not a validated trading strategy and has not replaced the
original scanner defaults. Four sessions do not justify claiming that a fitted
function reliably identifies near-instant 0.2% opportunities.

The selected compact model reduced forward-day mean squared error by only
**0.40% versus predicting zero**. Its rank correlation was negative on two of
the three forward test days, and the short side had negative realized
directional utility. Amplifying these small predictions to look decisive would
not improve their predictive value.

No outcomes after September 4 were evaluated for this study. Parent data,
baseline strategy and original manifest remain unchanged. This experiment has
its own [protocol](protocol.json), [fitted profile](strategy.json) and
[report](report.json), including source hashes for the revised scoring code.

The [threshold follow-up](#threshold-follow-up) below tests the user's later
trade/abstain rule. It preserves the original model study and files above.

## Meaning and Labels

An ordinary sigmoid has range 0 to 1. The signed form is:

$$S_t = 2\sigma(z_t)-1 = \tanh(z_t/2).$$

Positive means LONG, negative means SHORT. The scanner ranks absolute magnitude.
The endpoints describe ideal strength, not guaranteed order outcomes.

From a completed signal candle's close, consider the first of two tick-rounded
barriers: +0.2% LONG or -0.2% SHORT. Let D be that first direction, tau the
minutes to the touching candle's end, and R the minutes left before the existing
15:15 square-off. The training label is:

$$Y_t = D\exp(-(\tau-1)/5)\,\mathbf{1}_{\{\tau\le R\}}.$$

- First target in the next one-minute candle: label +1 or -1.
- Two minutes: magnitude 0.819; five minutes: 0.449; ten minutes: 0.165.
- Neither target before square-off: zero. No overnight outcomes are credited.
- Both targets first touched in one candle: excluded as ambiguous.
- Missing future minutes: excluded, not classified as a failed opportunity.

Competing barriers avoid rewarding a long that falls 0.2% first and only rallies
later, or its short-side mirror. This is a research definition of path quality,
not a new stop-loss policy. It is stricter than crediting any eventual target.

Five-minute decay is a stated interpretation of "almost no time", fixed before
inspecting outcomes, not a discovered optimum. One-minute OHLC cannot establish
second-level target times or execution. The close is an analytical reference,
not a claim that an order can fill there after the decision. Labels are gross;
costs, latency, spread, slippage and actual fills are separate questions.

The model estimates E[Y | currently available features]. Zero can therefore
mean no fast opportunity, uncertainty, or balanced opposing possibilities.
A score of 0.8 is **not** an 80% win probability. A later failed opportunity
can still receive a nonzero estimate beforehand; inference cannot know its future.

## Four Sessions

August 31 supplies warm-up only. Each day has 359 decision times, 09:16 through
15:14: 1,436 observations total. Paths end at 15:15. Adjacent decisions overlap;
these are not counts of independent trades.

| Session | Open to 15:15 | Range/open | LONG first | SHORT first | Neither | Either first target within 5m |
| --- | --- | --- | --- | --- | --- | --- |
| Sep 1 | +1.322% | 2.466% | 172 | 60 | 127 | 59/359 (16.4%) |
| Sep 2 | +1.139% | 2.171% | 222 | 118 | 19 | 58/359 (16.2%) |
| Sep 3 | -0.509% | 0.889% | 160 | 185 | 14 | 29/359 (8.1%) |
| Sep 4 | +1.714% | 2.043% | 231 | 86 | 42 | 42/359 (11.7%) |

**September 1:** fast opportunities concentrated near the open: 65.9% of
09:16-09:59 decisions reached either target within five minutes. After 13:00,
none did. Every 14:00-15:14 decision failed to reach either target by cutoff.
A bullish day, or price still above its opening range, does not justify a high
late-session score.

**September 2:** first-target direction alternated despite the rising day.
During 10:00-10:59 there were 32 LONG-first and 28 SHORT-first observations;
40% reached a target within five minutes. During 14:00-14:59 there were 14
LONG-first, 40 SHORT-first and six neither. Daily direction hides these changes.

**September 3:** narrower and slower, with a slight SHORT-first majority. Only
8.1% reached either target within five minutes. Conditional on an eventual hit,
median LONG target time was 38 minutes and SHORT was 30 minutes, regardless of
which side hit first. These are weak opportunities under the speed objective
even when eventual direction is correct.

**September 4:** the largest open-to-cutoff rise did not imply uniformly fast
long entries. Early moves supplied most fast targets; later extensions reversed
or required long waits. Exact forward-prediction failures below illustrate this.

## All 47 Features

All 47 derived features were audited for availability and per-day rank
association with the target. The full audit is in [report.json](report.json).
These are descriptive associations, not 47 independently established effects.

| Feature | Sep 1 rho | Sep 2 rho | Sep 3 rho | Sep 4 rho |
| --- | --- | --- | --- | --- |
| EMA9 five-bar slope | +0.115 | +0.282 | -0.026 | -0.148 |
| Five-bar return | +0.111 | +0.174 | -0.084 | -0.127 |
| RSI14 | +0.207 | +0.190 | -0.049 | -0.118 |
| Price versus VWAP sigma | +0.317 | -0.075 | -0.299 | -0.301 |
| Distance from opening-range high | -0.213 | -0.539 | -0.746 | -0.626 |

Momentum relationships change sign. Large opening-range extension associates
with weaker subsequent signed utility on all four days, but session age,
exhaustion and diminishing time remaining can confound this relationship. It
does not establish that reversing breakout direction will work on new dates.

The old five-rule score, converted to strongest LONG minus strongest SHORT,
had mean absolute values of 0.81-0.91 and negative rank correlation on all four
days. Those heuristic strengths were never calibrated to the new target-time
label. Their errors demonstrate incompatible semantics and saturation, not a
fair comparison of realized trading profits.

The new function consumes the shared feature snapshot and compresses correlated
inputs into eight dimensionless families. It deliberately does not fit 47 free
coefficients to four days. Raw price levels, cumulative OBV and session volume
are not independent fitted predictors; normalized distances and relative volume
avoid much of their arbitrary scale. The choice of families is itself a model
assumption, not a proof that omitted features are useless.

## Implemented Function

Runtime code is in [opportunity.py](../../../src/ai_trader/scanner/opportunity.py).
Let a = ATR14/close, T = 0.002 and R = 360 - elapsed session minutes. The fixed
reachability gate is:

$$g_t=(1-\exp[-5(a/T)^2])(1-\exp[-R/5]).$$

This is a structural heuristic, **not** a calibrated target-hit probability.
It suppresses very low volatility and opportunities near cutoff. The final
model is:

$$S_t=2\sigma\left(g_t\sum_{j=1}^{8}w_j x_{j,t}\right)-1.$$

| Component | Construction | Fitted weight |
| --- | --- | --- |
| Trend alignment | EMA9-21 and EMA21-50 gaps in ATR units; EMA9/21 slopes normalized by ATR fraction | +0.016199 |
| Momentum | 1/5/15-bar returns scaled by ATR fraction and square-root horizon; MACD histogram and change in ATR units | +0.022786 |
| Directional pressure | (+DI - -DI) / (+DI + -DI) | +0.021279 |
| Range position | Location in rolling high/low, weighted by ADX regime | +0.011655 |
| Mean reversion | Negative RSI, Bollinger %B and optional VWAP-sigma stretch, weighted toward weak trends | -0.007135 |
| Trend pullback | Negative distance from EMA9 in ATR units, weighted toward strong trends | -0.023417 |
| Session pressure | Session-open distance, session-range position and opening-range location, weighted by ADX | +0.022768 |
| Volume confirmation | Mean trend/momentum/DI pressure times relative-volume saturation | +0.028355 |

The source defines every transform exactly; [strategy.json](strategy.json)
retains full-precision weights. Scalar stretches are bounded with tanh, the
trend regime is clipped ADX/50, and available signals within a family are
averaged. These transformations were fixed before the fit. Negative fitted
reversion/pullback weights do not change their definitions; they reflect this
small sample's fitted relationship, which may be unstable.

Missing core inputs fail closed. Outside the configured session, at cutoff or
without volatility, the output is neutral. Unavailable volume contributes no
directional confirmation. All inference uses deterministic Decimal arithmetic;
fitting is offline and never runs inside the scanner.

## Fitting and Forward Tests

Three predeclared fitted specifications were compared: three trend components
with L2=0.1, eight components with L2=0.1, and eight with L2=1.0. Neutral zero and
the legacy diagnostic are controls. Fits use soft-label logistic loss on
(Y+1)/2, no intercept, bounded weights, every fifth minute and equal session
weights. There is no random row split or full-sample feature standardization.
SciPy 1.16.1 is an optional research dependency, not a runtime scorer dependency.

Folds are Sep1 -> Sep2, Sep1-2 -> Sep3, and Sep1-3 -> Sep4. All 1,077 forward
predictions preceded fitting on their test day's labels. The folds informed
model selection, so they are **not an untouched final test**.

| Model | Forward MSE | Signed rank correlation |
| --- | --- | --- |
| Neutral zero | 0.0704364 | undefined |
| Three trend components, L2=0.1 | 0.0701767 | +0.0801 |
| Eight components, L2=0.1 | 0.0701540 | +0.0609 |
| Eight components, L2=1.0 | 0.0703716 | +0.0632 |

For the selected eight-component model, per-day MSE improvement over zero is
+0.0014438, -0.0008326 and +0.0002360 on Sep2/3/4. Exhaustive resampling of the
three day blocks gives a descriptive 95% range of -0.0008326 to +0.0010412.
Three sessions are too few for a dependable population confidence interval.

The chosen direction reached its target first in 51.25% of test observations;
it did so within five minutes in 7.06%. In the top absolute-score quintile,
14.88% reached the chosen target first within five minutes. This is an observed
rate, not a calibrated model probability. LONG predictions had positive mean
realized directional utility (+0.0539); SHORT predictions had negative utility
(-0.0131). Predictive value on both sides is not established.

Sampling every 30th minute leaves 36 observations and 52.8% first-direction
accuracy. This is a dependence sensitivity check, not independent evidence:
full-session outcome paths can still overlap.

The final four-day refit outputs approximately **-0.0256 to +0.0542** on those
same days. Forward-fold predictions range **-0.0424 to +0.0789**. Leaving them
small is intentional. Rescaling observed extrema to +/-1 would misrepresent
uncertainty. A sigmoid creates a range, not predictive information.

## Exact-Time Failures

These are forward-fold predictions, not all-four-days fitted results:

| Decision IST | Reference close | Score | Actual first target |
| --- | --- | --- | --- |
| Sep 2 10:31 | 1316.10 | +0.06347 | LONG in 1 minute |
| Sep 2 10:32 | 1321.00 | +0.07889 | SHORT in 2 minutes |
| Sep 4 09:54 | 1331.60 | +0.06674 | SHORT in 6 minutes |
| Sep 4 09:58 | 1331.40 | +0.06461 | SHORT in 2 minutes |

The strongest predictions include exhausted moves. Later research should test
whether extension/exhaustion features distinguish continuation from reversal
on new development dates. It must not erase these losing rows or invent a
threshold after inspecting them.

## Reproduce and Use

From the repository root, with the optional `research` extra installed:

```powershell
.venv/Scripts/python.exe -m scripts.research_signed_score
```

The command refuses existing outputs. `--overwrite` regenerates this experiment's
reports and fitted profile; it never changes the frozen parent. It verifies
parent data/config hashes, uses only Aug31 plus Sep1-4, and records current
script/model hashes. Raw features/labels and forward predictions are in
the local research outputs under `data/research/signed_score_v1/`.

The schema-v2 [strategy.json](strategy.json) works with BOTH replay and
`check_scanner` through `--strategy-config`. It is explicitly experimental, not
the default or an approved trading fallback. Portfolio suppression and the
existing feasibility screen remain separate from raw score evaluation.

The original [manifest](../manifest.json) still pins the old decision code.
Its full source verification correctly reports drift after this scorer change;
do not rewrite that historical evidence. The child report records new code
hashes while checking that parent data and configuration stayed unchanged.

Next validation requires later development days without first fitting their
outcomes. September 21-23 validation and September 24-28 holdout remain
unexamined for this model. No live orders or broker calls were needed.

## Verification

On 2026-09-30: **930 tests passed**, repository lint passed, and the formatter
reported 102 files already formatted. Tests cover first-touch ordering, missing
minutes, cutoff, tick rounding, sigmoid symmetry, a synthetic known signal,
future-prefix invariance, schema-v1 identity, schema-v2 round trips, portfolio
suppression, and exact signed-score parity through the real replay engine.
Editor diagnostics were clear. Changes remain uncommitted.

## Threshold Follow-Up

### Definitions in Plain Language

**Open-to-cutoff return** is the underlying stock's price change from the 09:15
opening price to the close of the candle ending at the 15:15 square-off. It is
not model performance. For September 1, `(1302.60 / 1285.60 - 1) * 100 = 1.322%`.
It describes buying at the day's open and marking at cutoff, before any costs.

**What the formula does:** compress overlapping indicators into eight normalized
summaries, weight them, shrink the result when volatility or time remaining is
insufficient, and map it into [-1, 1]. The fitted relationships largely favour
trend/momentum continuation, including some already-extended moves. Magnitude
helps locate fast movement, but the tested model often gets its direction wrong.
A score of 0.03 is neither a 3% success probability nor an expected 3% return.

**Signal accuracy** here means the chosen direction's 0.2% target is reached
before the opposite 0.2% target and before 15:15. Neither target counts as a miss.
It does not mean a profitable executable trade. **Fast precision** is the share
of selected signals that reach that chosen target first within five minutes.
**Coverage** is the fraction of decision minutes selected. **Fast recall** is
the fraction of all fast opportunities caught in the correct direction.

### Selection Without Reading the Check Day

The [threshold protocol](threshold_protocol.json) fixes 16 absolute thresholds
for the three existing model specifications: **48 combinations**, not an
unbounded search. The action is LONG when `score > 0` and `abs(score) >= threshold`,
SHORT when `score < 0` and the same magnitude condition holds, otherwise abstain.
Zero always abstains, including when the threshold is zero.

Threshold/model selection uses September 2-3 predictions generated by models
trained only on preceding days. At least 30 selected signals in total and five
on each tuning day are required. The primary objective is equal-day average
speed-discounted directional utility per decision; abstention is worth zero.
This balances correct fast opportunities against wrong calls and lost coverage.
It does NOT optimize trading profits or guarantee a positive net expectancy.

A secondary accuracy-only choice was added after inspecting the tuning grid,
before checking that choice on September 4. It uses the same minimum support;
ties favour more signals, then a lower threshold. This is transparently an
additional diagnostic, not a replacement objective chosen to hide a bad result.

Both choices are frozen for the September 4 check. Their coefficients were
fitted only through September 3. The earlier four-day-fitted profile was NOT
used to claim forward performance on one of its own training days. No outcomes
after September 4 were used. September 4 was already examined in the preceding
study, so this remains a chronological replay check, **not a fresh untouched
test**. There are too few independent days for a dependable accuracy interval.

### Results

Both criteria choose the eight-component model with L2=0.1, but different cutoffs:

| Tuning metric, Sep2-3 | Speed objective: threshold 0 | Accuracy objective: threshold 0.03 |
| --- | --- | --- |
| Selected signals | 718/718 | 43/718 |
| Coverage | 100% | 5.99% |
| LONG / SHORT signals | 378 / 340 | 32 / 11 |
| Correct first target | 366/718 (50.97%) | 27/43 (62.79%) |
| Opposite target first / neither | 319 / 33 | 13 / 3 |
| Chosen target first within 5m | 52/718 (7.24%) | 8/43 (18.60%) |
| Median correct-target time | 16.5 minutes | 12 minutes |
| Speed utility per decision | 0.02942 | 0.00266 |

The larger cutoff improves tuning accuracy and selected-signal quality, but
misses most opportunities. It does not improve the primary total-utility
objective. Its 43 tuning signals comprise 30 on Sep2 and 13 on Sep3; even at
this threshold, Sep3's signed speed utility is negative.

| Frozen check, Sep4 | Threshold 0 | Threshold 0.03 |
| --- | --- | --- |
| Selected signals | 359/359 | 18/359 |
| Abstentions | 0 | 341 |
| LONG / SHORT signals | 253 / 106 | 18 / 0 |
| Correct first target | 186/359 (51.81%) | 9/18 (50.00%) |
| Opposite target first / neither | 131 / 42 | 9 / 0 |
| Chosen target first within 5m | 24/359 (6.69%) | 9/18 (50.00%) |
| Fast-opportunity recall | 57.14% | 21.43% |
| Median correct-target time | 23 minutes | 1 minute |
| Eventual chosen-target hit, ignoring opposite-first | 56.27% | 77.78% |

At 0.03, the nine right calls are fast, all within three minutes. Nine others
point the wrong way. The apparently impressive 77.78% eventual-hit rate counts
five additional signals that suffered the opposite move first; reporting that
alone would disguise path risk. Adjacent signals cluster around a few price
moves and are not independent trials. An always-LONG signal control is correct
64.35% of the time on this rising check day; this is context, not a proposed
always-long strategy, and shows why beating 50% alone does not establish edge.

### Actual Simulated Trades

The existing replay engine applies position suppression, feasibility, fills and
exits, so it cannot open a new independent position at every selected minute.
These diagnostics use approximately Rs 100,000 per entry, the unchanged fixed
2-ATR stop, 0.2% gross target, Groww charges and 15:15 square-off. Fill assumptions
remain the prior 1-second latency, 2bp half-spread and 1bp slippage scenario.
No exits, costs or fills were retuned to improve these results.

| Sep4 replay metric | Threshold 0 | Threshold 0.03 |
| --- | --- | --- |
| Completed trades | 15 | 7 |
| Net winners / losers | 8 / 7 | 4 / 3 |
| Net win rate | 53.33% | 57.14% |
| Gross P&L before fees, after modelled fill friction | Rs +223.10 | Rs -194.90 |
| Fees | Rs 1,244.51 | Rs 580.55 |
| Net P&L | **Rs -1,021.41** | **Rs -775.45** |
| Net expectancy per trade | Rs -68.09 | Rs -110.78 |
| Profit factor | 0.490 | 0.388 |
| Realized maximum drawdown | Rs 1,268.78 | Rs 1,022.70 |
| Mean holding time | 16.38 minutes | 12.41 minutes |
| Median holding time | 3.98 minutes | 1.98 minutes |
| Target / stop / square-off exits | 8 / 6 / 1 | 4 / 3 / 0 |

All executed trades in this check were LONG; short-side execution performance
is untested here, despite the raw model emitting short signals. No ambiguous
exits, unfilled entries or open positions remained. Drawdown measures realized
P&L only, not intratrade mark-to-market loss. One-minute candles cannot verify
one-second fills; the engine ignores target/stop touches in the entry candle.

At threshold 0.03, four winners average about Rs 123 net and three losers about
Rs 422 each. Thus a 57% win rate still loses money. A high-volatility signal can
also imply a wider 2-ATR stop while the profit target remains 0.2%. The lower
total loss at 0.03 reflects fewer trades; its per-trade expectancy is worse.

**Read of the outcome:** the threshold can identify a smaller group with fast
movement, but it has not established reliable direction or profitable execution.
Neither tested choice is ready for live trading. "Best" here means best among
this bounded grid under a stated tuning objective, not a globally optimal formula.

### Artifacts and Reproduction

- [Full threshold report](threshold_report.json): all metrics, every grid choice,
  per-day and per-side results, exact trades and provenance hashes.
- [Threshold sweep](threshold_sweep.csv): the complete 48-combination tuning table.
- [Speed-objective profile](threshold_strategy.json): threshold 0, Sep1-3 fit.
- [Accuracy-only profile](accuracy_strategy.json): threshold 0.03, the same fit.

```powershell
.venv/Scripts/python.exe -m scripts.research_signed_score --threshold-study
```

Existing outputs require explicit `--overwrite`. The original model report,
profile and frozen parent files are preserved. `score_threshold` is shared by
both scanner paths; nonzero thresholds serialize as schema 3. Schemas 1 and 2
load with their original behavior and fingerprints. A threshold does not alter
the raw model score, position sizing, or deterministic risk controls.

Future work should freeze a candidate before using later development dates,
then validate both directional quality and net expectancy. The reserved
validation and holdout sessions remain unused for this model.

Threshold follow-up verification on 2026-09-30: **946 tests passed**, full lint
passed, 102 files passed the formatting check, and editor diagnostics were
clear. The final artifact rerun reproduced the same choices and results after
formatting. No broker calls, live orders, commits or changes to earlier evidence.