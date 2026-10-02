# Symmetric TP/SL and Time-to-Target Probabilities

Research completed 2026-10-01, NSE:RELIANCE. The scorer now estimates
$P(TP\text{ before SL and within }h\text{ hours of entry}\mid X,d)$ with equal
initial TP/SL distances. **The probability contract is implemented, but a useful
high-confidence trading policy is not established.** No live scorer was changed.

## Selected Result

The choice made using August, before this run's September labels:

| Parameter | Selected value |
| --- | --- |
| Model | L2 multinomial logistic regression, `C=0.1` |
| Features | 11 bounded intraday inputs plus direction |
| Nominal TP = SL | **0.40% of executed entry** |
| Horizon | **0.1 hours = 6 minutes** |
| Calibration | Separate LONG/SHORT joint softmax recalibration |
| Selection criterion | Probability-loss improvement over a frozen side-specific base-rate control |

Across 11,362 hypothetical September side-actions, the mean forecast was
**0.619%**, versus **73/11,362 = 0.642%** actual TP-before-SL hits within six
minutes. ROC-AUC was 0.933; Brier loss 0.006090 versus control 0.006397, a
**4.80% relative improvement**. Log loss improved 28.07%. A paired whole-session
bootstrap gives a descriptive 95% Brier-skill range of +1.09% to +8.75%.
This interval uses 19 resampled sessions, not independent minute trials, and
does not make previously researched September an untouched prospective test.

**This is a rare-event forecast, not a 93% chance of profit.** The maximum
six-minute forecast was only 20.89%. The highest forecast over all 16 tested
event definitions was 62.90%; no tested September action was forecast at 70%.
Low-probability calibration can be accurate without producing an attractive
entry. Choosing the best probability estimator is not the same as choosing a
profitable trading policy.

The separately selected threshold was zero. The predefined threshold grid
jumped from zero to 35%; all positive grid thresholds for this selected event
abstained. The zero-threshold diagnostic made 826 non-overlapping trades,
11 TP hits (1.33%) versus a 0.974% mean forecast, and lost **Rs 121,688.20** after
modeled costs. This is not a recommendation to enter low-probability trades.
The optimal entry threshold below 35% was not tested, and is not established.

## Event and Execution

With executed entry price $P$, requested fraction $f$ and tick $t$:

$$D=t\left\lceil\frac{Pf}{t}\right\rceil,\qquad
TP=P+dD,\qquad SL=P-dD,\qquad d\in\{+1,-1\}.$$

Both barriers have exactly the same distance, including after tick rounding.
Entry means the buy fill for LONG and the sell fill for SHORT, never the future
short-cover buy price. The clock starts at that entry. For the selected policy,
actual TP and SL distances both had median **0.40425%**, range 0.40000-0.40793%.
Tick rounding explains the small difference from nominal 0.40%. Fees and adverse
stop fills mean equal price distances do not imply equal net gain/loss sizes.

Labels and trades use the existing `ReplayPortfolio`, deterministic whole-share
Rs 100,000 clip, Rs 0.10 tick, one-second modeled latency, 2 bp half-spread,
1 bp slippage and the fee schedule. There is no exit on the entry candle; STOP
wins when both barriers touch in one candle, and stops that gap pay the worse
open. Only fully completed candles by the deadline can establish a barrier hit.
A remaining position closes with an explicit `horizon` reason at the deadline,
using the last completed close and adverse market-fill assumptions.

Minute bars do not measure one-second fills or order-book queue execution. A
touch inside the entry candle or final partial minute is unobservable under
these assumptions. These are probabilities for the conservative replay event,
not yet independently verified probabilities of live fills.

Every event uses the same entry population, **09:16-14:14 IST**, with a full
one-hour outcome window before 15:15. Shorter-horizon afternoon entries are
outside this benchmark. The previous three-ATR reachability prefilter is
explicitly off in this offline study so the model can learn reachability.
Finite stops, sizing, target-after-fees validation, cutoff and one-position
scheduling remain deterministic. Production defaults remain unchanged. This
different population makes raw hit rates incomparable with the older filtered
studies.

## Data and Selection

The [protocol](protocol.json) was saved before fitting. The tape is copied from
the hash-verified immutable [previous snapshot](../tp_probability_2026q3/data_manifest.json),
not refreshed from a broker. All 63 sessions have every pre-cutoff minute and
tick-aligned prices. See [data_manifest.json](data_manifest.json).

| Purpose | 2026 dates | Sessions |
| --- | --- | ---: |
| Indicator and prior-week warm-up | July 1-7 | 5 |
| Coefficients and feature standardization | July 8-31 | 18 |
| Model and feature-family selection | August 3-7 | 5 |
| Per-side calibration and frozen base rates | August 10-21 | 10 |
| Calibration choice, TP/horizon and threshold selection | August 24-31 | 6 |
| Evaluation | September 1-28 | 19 |

Base fitting uses every fifth decision and equal day weights: 2,160 side-actions
at 1,080 timestamps over 18 sessions. LONG and SHORT share future price paths;
these are not 2,160 independent market regimes. There is no random validation
split and no August refit of base coefficients. All 72 fits converged.

Four TP=SL fractions (0.15%, 0.20%, 0.30%, 0.40%) and four horizons
(0.1, 0.25, 0.5, 1 hour) were compared. For each target, six model configurations
were tested on three feature sets: volatility/time/direction control, intraday,
and intraday plus prior-week context. Models were logistic `C=0.1/1`, boosted
trees of depth 2/3, and one-layer tanh neural nets with 8/16 units and L2
`alpha=10/100`. Tree settings are 80 iterations, learning rate 0.05, minimum
60 samples per leaf and L2=5. Numerical thread pools and random seeds are fixed.

Model selection minimizes equal-day TP Brier loss averaged across the four
horizons. Calibration is fitted later on separate dates. Event selection
maximizes the smaller of Brier and log-loss skill against the frozen control,
with shorter-horizon and smaller-target tie breaks. The base control averages
per-session, per-side event frequencies with half-count smoothing for each of
the nine classes. Absolute low Brier loss alone is not used to select rare
events, but even relative probability skill need not identify a useful trade.

Threshold selection uses a whole-session bootstrap precision statistic rather
than preferring an unsupported 1/1 hit rate. Empty resamples get zero only for
this conservative selection statistic; an actually empty policy's hit rate
remains undefined. No minimum trade frequency is imposed. Six validation days
still provide limited evidence for threshold choice.

[selection_lock.json](selection_lock.json) and model artifacts were written,
then reloaded before September labels were built. **September was already
examined in earlier research. This remains retrospective evaluation, not a
pristine prospective test.** No parameters were changed after these results.

## Function Used

Each model predicts nine mutually exclusive outcomes: TP in 0-6, 6-15, 15-30
or 30-60 minutes; SL in those same intervals; or neither by one hour. Let $r_d$
be the direction-adjusted feature vector standardized with July-only mean and
scale. For the selected logistic model:

$$z_{d,k}=b_k+W_k r_d,\qquad q_d=\operatorname{softmax}(z_d),$$
$$\widetilde q_d=\operatorname{softmax}
\left(a_d\log(\max(q_d,10^{-12}))+c_d\right),\qquad
p_{d,h_j}=\sum_{k=0}^{j}\widetilde q_{d,k}.$$

For the selected six-minute horizon, only TP class 0 enters the sum. Calibration
slopes are 1.023919611 for LONG and 1.017549251 for SHORT, with separate nine-class
bias vectors. The full-precision 9-by-12 coefficient matrix, intercepts, feature
order, means, scales and calibration vectors are in
[selection_lock.json](selection_lock.json), model `0.004`. The exact estimator
is [model_tp_0.004.pkl](../../data/research/tp_horizon_symmetric_v1/model_tp_0.004.pkl),
with its hash and dependency versions recorded in the lock. Only deserialize
trusted, locally generated model artifacts; a hash is not permission to load an
untrusted pickle.

The signed score is +p_LONG or -p_SHORT for the larger side probability; ties
abstain. LONG/SHORT probabilities are not complements. The magnitude is never
centered with `2*p-1`, amplified or multiplied by a speed reward. For fixed
inputs and target, cumulative TP probability cannot decrease as h grows.
Speed preference means asking for a short explicit horizon, not changing the
meaning of probability.

**Philosophy:** learn direction and the distribution of time needed to reach a
barrier, including the competing stop and timeout. This is more precise than
inferring speed from an eventual-hit score. Separate model fitting, calibration
and action selection prevent confusing a forecast with permission to trade.

Before September, permutation checks on August 3-7 identified these leading
inputs for the selected model's joint TP curve:

| Feature | Increase in validation Brier when shuffled |
| --- | ---: |
| Volatility relative to target | 0.001777 |
| Session/opening-range pressure | 0.000970 |
| Direction indicator | 0.000400 |
| EMA alignment/slopes | 0.000342 |
| ADX | 0.000260 |

This measures model reliance, not causal importance. Shuffling correlated inputs
can create unrealistic combinations. Directional pressure and volume confirmation
had slightly negative permutation effects in this slice. The model's strong
rare-event ranking largely reflects when a large fast move is plausible;
that is not equivalent to finding a tradable directional advantage.

## Did Weekly Context or Neural Networks Help?

Weekly inputs use only the previous five complete 09:15-15:15 sessions: return,
log-close slope, close-return volatility, daily range, current location versus
the prior-week range and distance from the prior cutoff close. The current day's
future high, low and close are never used. These are price-regime inputs, not
economic macroeconomic data or official 15:30 daily bars.

Best August integrated Brier within each feature family, lower is better:

| TP=SL | Time/volatility control | Intraday | Intraday + week | Selected model |
| --- | ---: | ---: | ---: | --- |
| 0.15% | 0.171764 | 0.170275 | **0.168060** | Logistic C=0.1 + week |
| 0.20% | 0.144722 | 0.142485 | **0.139128** | Logistic C=0.1 + week |
| 0.30% | 0.099346 | 0.097467 | **0.097285** | Depth-2 boosted trees + week |
| 0.40% | 0.058400 | **0.056892** | 0.059017 | Logistic C=0.1, no week |

Weekly context helped some development comparisons, especially at 0.15-0.20%,
but not all. No neural candidate won. This is evidence about two small neural
configurations and 18 fitting sessions, not a claim that neural networks cannot
work. The [research notes](../../docs/strategies/README.md#scorer-model-choice)
explain the tabular benchmarks, order-book neural studies and the limited
weekly-reversal evidence behind these choices.

## All Locked September Policies

The table reports actual non-overlapping trades at thresholds selected in
August. These are **alternative policies**, not additive portfolio results.
An expiry is a TP failure even if it earns a small positive return.

| TP=SL | h hours | Threshold | TP/trades | Mean forecast | Actual TP % | Net Rs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15% | 0.10 | 0 | 85/970 | 7.30% | 8.76% | -146,475.01 |
| 0.15% | 0.25 | 0 | 137/597 | 19.92% | 22.95% | -85,797.10 |
| 0.15% | 0.50 | 0 | 156/480 | 33.00% | 32.50% | -66,346.19 |
| 0.15% | 1.00 | 0.40 | 119/317 | 43.34% | 37.54% | -45,477.57 |
| 0.20% | 0.10 | 0 | 53/892 | 4.73% | 5.94% | -129,918.43 |
| 0.20% | 0.25 | 0 | 77/480 | 13.28% | 16.04% | -67,073.79 |
| 0.20% | 0.50 | 0.35 | 26/56 | 39.50% | 46.43% | -5,709.34 |
| 0.20% | 1.00 | 0.40 | 47/110 | 43.25% | 42.73% | -13,640.49 |
| 0.30% | 0.10 | 0 | 23/842 | 2.12% | 2.73% | -126,876.63 |
| 0.30% | 0.25 | 0 | 35/405 | 6.68% | 8.64% | -62,698.11 |
| 0.30% | 0.50 | 0 | 47/252 | 13.56% | 18.65% | -32,044.46 |
| 0.30% | 1.00 | 0 | 60/180 | 24.60% | 33.33% | -20,873.14 |
| **0.40%** | **0.10** | **0** | **11/826** | **0.974%** | **1.33%** | **-121,688.20** |
| 0.40% | 0.25 | 0 | 19/382 | 3.13% | 4.97% | -57,066.41 |
| 0.40% | 0.50 | 0 | 23/223 | 7.76% | 10.31% | -32,348.15 |
| 0.40% | 1.00 | 0 | 27/148 | 16.03% | 18.24% | -25,075.68 |

All 16 combined policies lost money. The 0.20% / 0.5-hour row has the highest
observed selected-trade TP rate here, but it was not the overall probability-
quality choice. It used logistic plus weekly features and produced 24/53 LONG
and 2/3 SHORT TP hits. Three shorts are not evidence of a reliable 66.7% model.
Do not replace the locked winner using this hindsight comparison.

For the selected 0.40% / six-minute event, per-side hypothetical calibration was:

| Side | Forecast | Actual | TP/actions | AUC |
| --- | ---: | ---: | ---: | ---: |
| LONG | 0.788% | 0.933% | 53/5,681 | 0.909 |
| SHORT | 0.450% | 0.352% | 20/5,681 | 0.963 |

The combined 10-20% probability band had 85 overlapping actions, mean forecast
13.47%, and 15 hits (17.65%). Above 20% there was just one action and no hit.
Mean calibration and a high AUC cannot establish reliable high-score precision
from that tail. See [policies.csv](policies.csv) for every LONG/SHORT result and
[report.json](report.json) for reliability bins, daily outcomes and controls.

## Next Steps

1. Keep this as a probability benchmark, not an execution recommendation. Require
   separately validated, cost-aware entry selection and an abstention outcome.
   With timeouts, TP probability alone does not determine expected P&L; timeout
   returns and stop costs matter as well. Do not force weak forecasts upward.
2. Gather more independent months and instruments, and use repeated forward
   validation with session-level uncertainty. Freeze a choice before genuinely
   new data. The current 18 fitting sessions are a pilot, not proof of broad
   regime coverage; adjacent minutes and two sides share information.
3. Add and ablate market/sector-relative movement and same-time relative-volume
   surprises. Retain weekly context where development evidence supports it.
   The present features detect movement potential better than directional edge.
4. Validate quotes, entry/stop friction and intrabar outcomes in forward shadow
   data. Six-minute labels are particularly sensitive to entry-bar exclusion and
   bid/ask effects; minute OHLCV cannot validate second-level execution.
5. Improve evidence and labels before expanding model size. Keep logistic and
   shallow trees as controls; retest neural sequences only with suitable richer
   history. No finite study can guarantee a perfect scanner or future profits.

## Reproduction and Verification

```powershell
& .\.venv\Scripts\python.exe -m scripts.research_horizon_probability --root backtests/tp_horizon_symmetric_v1
```

The optional research extra now includes pinned scikit-learn 1.7.2. The lock
records Python, library versions, all coefficients/parameters and source hashes.
Use `--overwrite` only to reproduce this experiment's outputs; the frozen tape
is never overwritten. All 72 candidate outcomes are in
[model_trials.csv](model_trials.csv). Per-action forecasts are in
[predictions.csv](../../data/research/tp_horizon_symmetric_v1/predictions.csv),
and exact executed prices, percentages and expiries in [trades.csv](trades.csv).

988 offline tests, full lint and the 111-file formatting check passed. Editor
diagnostics are clear and local evidence links resolve. Artifact checks verified
all 7,160 alternative-policy
trades have exact equal price distances and exit by their respective deadlines;
all 45,448 action forecasts are bounded and monotone over the four horizons.
Saved model hashes were checked before deserialization and evaluation. No
credentials were inspected, broker calls made, orders placed or commits created.