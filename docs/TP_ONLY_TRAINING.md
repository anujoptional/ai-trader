# TP-Only Model Research

Updated October 10, 2026 with the scanner-window model study below. The
October 5 study remains frozen and unchanged; its data/model reproduction and
live-input checks were last executed October 7.

## Scanner Model Study: October 10, 2026

**A fresh CatBoost model for the 09:40-13:30 IST scanner window now exists and
is preferred over the October 5 model by a predeclared rule.** It is research
only: not wired into a runtime scanner, journal or order path. Driver:
[research_tp_scanner.py](../scripts/research_tp_scanner.py); artifacts:
[tp_scanner_kite_20261010](../data/research/tp_scanner_kite_20261010/report.json).

- **Inputs:** the same 13 features plus side. Each feature carries an input no
  other feature uses, so none is derivable from the rest. On 1,079,085 training
  LONG rows the largest linear R2 from the other twelve is 0.43 (`vwap_z`); the
  strongest pair is `vwap_z`/`day_return`, r = 0.53. All 13 are kept.
- **Output:** raw probability that +/-0.20% is touched within the next 60 minutes.
- **Calls:** 09:40-13:30 IST inclusive, five-minute grid, 47 decisions/session.
  The frozen dataset already holds these rows; no rebuild. TP-by-15:00 and 15:00
  returns are the verified October 6 arrays, joined by exact symbol/time/side keys.

The protocol was written before fitting; the selection lock before evaluation.

| Phase | Dates | Window Rows |
|---|---|---:|
| Train | January 2024-December 31, 2025 | 2,158,170 |
| Validation: early stopping and selection | January 1-March 31, 2026 | 281,530 |
| Evaluation, opened once | April 1-September 28, 2026 | 575,844 |

Four candidates: depth 4 or 6, uniform or 365-day half-life recency weights,
LR 0.03, leaf L2 20, other settings as October 5. Validation log loss: `depth4`
0.645002 (**selected, 511 trees**), `depth6` 0.645054, `depth4_recent` 0.645224,
`depth6_recent` 0.645279. Recency weighting did not help.

Evaluation covers 575,844 calls, 123 sessions and 50 stocks. Overall touch rate
is 57.17% within 60 minutes and 73.24% by 15:00.

| Model | Log Loss | Brier | AUC | 10-Band ECE |
|---|---:|---:|---:|---:|
| Fresh `depth4` | 0.666407 | 0.236922 | 0.6035 | 0.63 pp |
| October 5 raw | 0.667096 | 0.237247 | 0.6017 | 0.82 pp |
| October 5 calibrated, locked selection | 0.667653 | 0.237493 | 0.6024 | 2.15 pp |

The fresh model's log-loss reduction is 0.000690 versus raw, paired five-session
block 95% interval [0.000141, 0.001345], and 0.001247 versus calibrated,
[0.000262, 0.002337]. Both intervals exceed zero, so the predeclared rule
prefers it. The improvement is real but small.

| Band | Fresh Calls | Mean P | TP Within 60m | TP By 15:00 | No-TP 15:00 Mean Return | Oct 5 Raw Calls | Raw TP 60m | Raw TP By 15:00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-20% | 0 | - | - | - | - | 0 | - | - |
| 20-30% | 226 | 28.95% | 22.12% | 44.69% | -0.180% | 228 | 34.21% | 55.70% |
| 30-40% | 17,762 | 37.28% | 36.67% | 55.79% | -0.287% | 20,070 | 37.29% | 56.60% |
| 40-50% | 112,411 | 45.91% | 46.34% | 64.76% | -0.408% | 110,959 | 46.65% | 65.21% |
| 50-60% | 214,459 | 55.22% | 55.36% | 72.42% | -0.554% | 205,081 | 55.23% | 72.27% |
| 60-70% | 171,528 | 64.40% | 63.51% | 78.09% | -0.780% | 178,196 | 63.22% | 77.88% |
| 70-80% | 55,516 | 73.60% | 71.56% | 83.23% | -1.087% | 57,334 | 71.10% | 82.68% |
| 80-90% | 3,941 | 81.88% | 80.23% | 88.30% | -1.493% | 3,976 | 80.51% | 88.15% |
| 90-100% | 1 | 90.02% | 100% | 100% | - | 0 | - | - |

Bands are lower-inclusive. TP by 15:00 includes the 14:59-15:00 candle; the
15:00 return is side * (that candle's close / decision close - 1), gross.

| P >= 0.75 | Fresh | Oct 5 Raw | Oct 5 Calibrated |
|---|---:|---:|---:|
| Calls / per session | 19,997 / 162.6 | 20,332 / 165.3 | 22,016 / 179.0 |
| Distinct candle-times / both sides qualify | 12,243 / 7,754 | 12,564 / 7,768 | 14,152 / 7,864 |
| TP within 60 minutes | 75.70% | 75.59% | 75.10% |
| TP by 15:00, session-block 95% interval | 85.22% [84.34, 85.97] | 85.12% [84.14, 86.19] | 84.69% [83.58, 85.92] |
| No TP by 15:00, sold at 15:00 | 2,955 | 3,025 | 3,371 |
| Their gross return: mean / median | -1.288% / -1.122% | -1.304% / -1.134% | -1.306% / -1.120% |
| Their gross return: p10 / worst | -2.525% / -5.393% | -2.546% / -5.393% | -2.584% / -5.393% |
| Their mean after estimated charges | -1.371% | -1.387% | -1.389% |
| All calls, TP-or-15:00 gross mean | -0.0199% | -0.0238% | -0.0306% |
| All calls, after estimated charges | -0.103% | -0.106% | -0.113% |
| Break-even TP by 15:00, gross / after charges | 86.6% / 92.1% | 86.7% / 92.2% | 86.7% / 92.2% |

Charges are the repository's Zerodha intraday estimate at INR 100,000,
0.0827% per round trip, excluding spread and slippage. TP calls are credited
exactly +0.20%; a touch is not a fill. The fresh model's gross-mean interval is
[-0.0380%, -0.0019%]. Calls overlap across times and sides; these are not trades.

**Interpretation.** The probabilities are well calibrated for their 60-minute
event and the fresh model ranks slightly better, but P >= 0.75 is not an edge
under TP-or-15:00 with no stop: misses lose about 6.4 times the +0.20% target.
In 63% of high-score candle-times both directions qualify, so a high score
mainly predicts movement, not direction. Miss losses grow with the score band.
By side (fresh, P >= 0.75, retrospective): LONG gross +0.021% per call, SHORT
-0.057%; both are negative after charges. Further tuning of this target is
unlikely to change the economics; the next lever is the scanner policy or a
payoff-aware target, judged on new forward data.

April-September 2026 was examined by earlier research, so this comparison is
retrospective, not a pristine holdout. The October 5 model, calibration and
reports are unchanged. Independent checks: saved band and P >= 0.75 figures
recomputed from the prediction arrays, and the selected weights reload to
identical forecasts.

| Artifact | SHA256 |
|---|---|
| `protocol.json` | `94f1d0b1246febb77cf600ee6b12e4bac8b5bffe2251d24f3f7432e7bdea02be` |
| `final/best.cbm` | `fd005606634d4a02c4d11288beb6bbe7a0df4657e8f60b37ca9f096ae1e349dd` |
| `final/selection_lock.json` | `fce5968ca6271a34b1cc3bd2bedcc7f569847ac88194305dffb8a8cdb25c9c0d` |
| `evaluation_predictions.npz` | `987cecbe3e7c138e62dd381f4ef09629f69b6ebf3821264cb8ba834df59d892d` |
| `report.json` | `debd75c08c5dd94feec281791f7139ef5fa736ad59b69cc55286a7aa0fd36d63` |

Reproduce only into a new directory; it refuses an existing output:

```powershell
& .\.venv\Scripts\python.exe -m scripts.research_tp_scanner `
  --data data/research/tp_only_kite_20261005_final `
  --current data/research/tp_catboost_kite_20261005/final `
  --outcomes data/research/tp_frequency_returns_kite_20261006 `
  --output data/research/tp_scanner_kite_NEXT
```

To score features offline, load `final/best.cbm` with `CatBoostClassifier`,
check its SHA256 against `final/selection_lock.json`, pass the 14 ordered
columns and apply the sigmoid to `RawFormulaVal`. No calibration map is used.

## October 5 Study: Locked Predecessor

**Completed: eight fresh Kite-only CatBoost candidates, one diagnostic seed
repeat, chronological calibration and retrospective evaluation.** The validation
winner is `depth4_l2`: depth 4, learning rate 0.03, leaf L2 20, **462 trees**.
It is a frozen research candidate, not a deployed scanner or trading system.
Orders remain unavailable. The October 7 market-hours follow-up now verifies
five in-window target records and independent TP-input arithmetic. The built
historical/live paths are accepted to the tested scope for scanner research;
exact source equivalence and future automatic recovery are separate limitations.
The model stays frozen, not promoted; see Readiness And Next Work below.

The winning model has useful TP-touch ranking signal, but **calibration did not
improve the later evaluation**. Preserve the protocol-selected calibrated output
and its predeclared raw comparator. Do not retrospectively rename raw as the
selected winner or tune another mapping against these evaluation outcomes.

This section and the October 10 study above are the current Kite reports. The
collapsed history at the end concerns
deleted Groww datasets/models and is not current setup, retention guidance or
Kite performance. No old model weights were reused for these fits.

## Dataset And Target

The [Kite manifest](../data/research/tp_only_kite_20261005_final/manifest.json)
pins 200 shards, source data, feature code, ordered columns and chronological
boundaries. The underlying immutable cache contains 66 instruments: 50 equity
targets and 16 context instruments, January 2024 through September 30, 2026.
Research stops on September 28. The model uses NIFTY, GOLDBEES and INDIAVIX
context; the other context instruments remain reusable research inputs.

| Phase | Dates | Side-Action Rows |
|---|---|---:|
| Train | January 2024-June 30, 2025, after warm-up | 1,905,708 |
| Validation | July 1-December 31, 2025 | 665,616 |
| Calibration | January 1-March 31, 2026 | 335,440 |
| Retrospective evaluation | April 1-September 28, 2026 | 686,112 |
| Total | | 3,592,876 |

One decision yields paired LONG and SHORT actions. The reference price is the
last completed one-minute close. The label is whether any of the **next 60
complete minutes**, including the first, reaches reference times 1.002 for LONG
or 0.998 for SHORT. Adverse movement never ends the label. Both directions can
succeed. Decisions are every five minutes, 09:40-14:15 IST; a full future window
must be present and ends by 15:15. No synthetic candles or missing-data negatives.

The model receives 13 numerical features plus `side`, in this exact order:

```text
log_return_5m, log_return_previous_10m, day_return,
gap_from_vendor_daily_close, realized_volatility_20m,
prior_5_daily_close_return, prior_5_cutoff_volatility,
vwap_z, same_minute_rvol_5, minutes_from_open,
benchmark_log_return_15m, gold_log_return_15m, vix_level, side
```

Side is -1 for SHORT and +1 for LONG. Stock identity, timestamps, labels,
excursions, TP and horizon are not input columns. Returns are fractions, not
percentages. Exact formulas live in
[tp_only.py](../src/ai_trader/features/tp_only.py). CatBoost receives these raw
finite features; it does not use the neural models' clipping/standardization.

The [independent data validation](../data/kite_migration_20261005/validation.json)
recomputed **every label, first-touch time, adverse excursion and terminal
return** from integer-scaled Kite prices; all 3,592,876 matched. Daily inputs
come from hash-verified Kite daily files. A daily close outside the continuous
minute range is not automatically invalid: an official auction can cause that.
Invalid OHLC and other price inconsistencies are quarantined upstream.

The calendar includes captured daily dates and quarantines. Missing prior dates
remain unavailable, not replaced with older complete dates. Five-session close
return needs six consecutive daily closes; prior volatility requires five
complete 09:15-15:15 cutoff windows; same-minute volume requires the previous
five sessions. Missing inputs exclude decisions. This is complete-case research,
not a claim that the live feature stream always has full coverage.

**Execution is a separate contract:** TP or 15:00 IST, no price stop-loss and no
forced 60-minute exit. A 14:15 research prediction includes touches through
15:15, even though a position would already be closed at 15:00. The separate
October 6 follow-up below now measures 15:00 price returns and their conditional
distribution when TP was not hit. It does not simulate fills, costs or portfolio
profit. The old Groww no-stop analysis and legacy 2-ATR replay are not results
for this Kite model.

## Bounded Training

The fresh baseline and seven variants were trained locally on CPU with
`catboost==1.2.10`, eight threads per fit and a 1,200-second fit limit. Total
reported fitting time for **all nine fits was 544.853 seconds** (9 minutes
4.9 seconds), excluding data loading, auditing and post-fit reporting. Every fit
stopped by validation early stopping, not the time limit. No Azure training job,
GPU resource, model registration or endpoint was created for this study.

Common settings: Logloss, CPU Plain boosting, SymmetricTree, 254 borders,
Bernoulli bootstrap, subsample 0.8, random strength 1, natural class frequencies,
`use_best_model=True`, seed 20261003. Default maximum 2,000 trees and patience
100; the two 0.015 learning-rate variants use 4,000 and 200 respectively.
Neural-only fields in the shared configuration are unused by CatBoost.

The [predeclared protocol](../data/research/tp_catboost_kite_20261005/protocol.json)
tests shallower/deeper trees, learning rate and leaf regularization, consistent
with [CatBoost's tuning guidance](https://catboost.ai/en/docs/concepts/parameter-tuning).
This is a compact comparison suited to the finite numerical feature table,
not an exhaustive search or proof of globally optimal hyperparameters. Earlier
neural experiments motivated the model family, but their Groww scores are not
directly comparable with this replacement dataset.

### Validation Ranking

All eight candidates were evaluated on the same 665,616 validation actions.
Lower log loss and Brier are better. Log loss penalizes confident mistakes;
Brier is the mean squared probability error. Selection uses log loss, with
shallower/fewer-tree/name tie breaks, before any calibration or evaluation read.

| Rank | Candidate | Depth | LR | L2 | Trees | Log Loss | Brier | Fit Seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | depth4_l2 | 4 | 0.030 | 20 | 462 | 0.676245893 | 0.241717203 | 57.60 |
| 2 | depth4_slow | 4 | 0.015 | 5 | 1,011 | 0.676262878 | 0.241726600 | 118.70 |
| 3 | depth4 | 4 | 0.030 | 5 | 507 | 0.676363943 | 0.241775014 | 61.33 |
| 4 | strong_l2 | 6 | 0.030 | 20 | 261 | 0.676454170 | 0.241816657 | 44.27 |
| 5 | slow_lr | 6 | 0.015 | 5 | 532 | 0.676468899 | 0.241824878 | 88.15 |
| 6 | original, fresh baseline | 6 | 0.030 | 5 | 273 | 0.676481970 | 0.241829611 | 48.72 |
| 7 | fast_lr | 6 | 0.060 | 5 | 116 | 0.676605721 | 0.241890971 | 26.92 |
| 8 | depth8 | 8 | 0.030 | 5 | 198 | 0.676694847 | 0.241931747 | 43.78 |

The winner improves baseline log loss by only 0.000236076; its advantage over
`depth4_slow` is 0.000016985. These are small validation differences, not a
demonstrated large or statistically established hyperparameter advantage.

A diagnostic repeat of the winner with seed 20261004 retained 462 trees,
log loss **0.676250033**, Brier **0.241719323**, in 55.38 fit seconds. It was
not selected, averaged into the winner or evaluated as another test candidate.
One repeat checks sensitivity to this seed change; it does not establish
cross-regime robustness.

### Calibration Protocol

The exact winning weights were locked, not refitted. Identity, one global
sigmoid and separate LONG/SHORT sigmoids were fitted on January-February and
compared on March. Eligible mappings must not worsen overall Brier or either
side's log loss versus identity. Choose the lowest eligible March log loss,
then refit only the chosen mapping on all 335,440 January-March actions.

| Mapping | March Log Loss | March Brier |
|---|---:|---:|
| Identity / raw | 0.621697502 | 0.215440051 |
| Global sigmoid | 0.620943616 | 0.215072775 |
| Side sigmoid, selected | 0.620900080 | 0.215048280 |

For tree margin `z`, raw means `sigmoid(z)`, not the unbounded CatBoost
`RawFormulaVal` API result. Final calibration is `sigmoid(slope * z + intercept)`:
LONG slope 0.9323125205/intercept 0.0556196208; SHORT slope
0.9163422379/intercept 0.1320754999. It changes probabilities and cross-side
ranking, not the trees or within-side ordering. The calibration/selection lock
was saved before loading April-September evaluation.

## Retrospective Evaluation

**686,112 actions across 123 represented sessions**, including 389,127 TP
touches (56.7148%). These are paired and overlapping forecasts, not 686,112
independent trades. Stock/time metadata is used for audits, not model input.
The calendar period has also been examined in earlier research, so it is not
an untouched prospective holdout despite chronological separation in this run.

| Forecast | Log Loss | Brier | ROC AUC | Mean Forecast | 10-Bin ECE |
|---|---:|---:|---:|---:|---:|
| Raw winner, predeclared comparator | 0.668632934 | 0.237986933 | 0.600981 | 57.3498% | 0.9014 pp |
| Selected side-calibrated winner | 0.669463747 | 0.238367858 | 0.601297 | 59.0343% | 2.3215 pp |
| Constant 50% | 0.693147181 | 0.250000000 | 0.500000 | 50.0000% | 6.7148 pp |
| Frozen training-side priors | 0.686706480 | 0.246745590 | 0.516664 | 60.5888% | 3.8740 pp |
| Frozen calibration-side priors | 0.689982835 | 0.248287414 | 0.516664 | 62.2478% | 5.5330 pp |

Both model outputs beat the predeclared base-rate controls on these loss
metrics. **Calibration worsened log loss by 0.000830813 and Brier by 0.000380925.**
The selected-minus-raw log-loss difference has a descriptive 95% interval
**[0.000229138, 0.001433176]**, from 1,000 paired circular bootstrap resamples of
five-session blocks with all assets kept together, seed 20261004. This accounts
for some local dependence, not nonstationarity, all selection bias or future risk.

ROC AUC near 0.60 indicates modest ranking discrimination; it is not 60% trade
profitability. ECE is weighted absolute forecast/hit-rate difference in ten
disjoint probability bins. It is bin-dependent and can hide opposing errors.

### Direction And Month

| Direction | Actions | TP Touch Rate | Raw Mean P | Calibrated Mean P | Raw Log Loss | Calibrated Log Loss | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG | 343,056 | 55.0785% | 56.0380% | 56.9844% | 0.668086673 | 0.668666877 | 0.237786467 | 0.238072645 |
| SHORT | 343,056 | 58.3511% | 58.6616% | 61.0841% | 0.669179194 | 0.670260617 | 0.238187399 | 0.238663072 |

Within-side AUC is unchanged by monotone calibration: LONG 0.613781, SHORT
0.583254. Calibration worsens both sides' overall loss and four of six monthly
losses; May and June improve. The substantial base-rate shift is visible below.

| Month | Actions | TP Touch Rate | Raw Log Loss | Calibrated Log Loss |
|---|---:|---:|---:|---:|
| April | 109,312 | 62.0316% | 0.654181788 | 0.655726562 |
| May | 106,400 | 61.3327% | 0.655612175 | 0.655582298 |
| June | 117,600 | 59.1216% | 0.665003201 | 0.664590778 |
| July | 128,800 | 56.0070% | 0.672355565 | 0.673214515 |
| August | 117,600 | 49.7874% | 0.681252979 | 0.683327512 |
| September through 28th | 106,400 | 52.4878% | 0.682057339 | 0.682980746 |

### Fixed Threshold Diagnostics

Thresholds were fixed before evaluation. Counts include overlapping opportunities,
potentially both directions and repeated decisions in the same stock. There is
no position suppression, capital limit, entry fill or trading threshold selection.

| Output | Minimum P | Actions | TP Hits | Touch Rate | Mean P | Sessions |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.60 | 275,655 | 179,636 | 65.1670% | 66.8188% | 123 |
| Raw | 0.70 | 68,385 | 48,886 | 71.4864% | 74.0958% | 123 |
| Raw | 0.80 | 4,480 | 3,600 | 80.3571% | 81.9257% | 113 |
| Raw | 0.90 | 0 | 0 | N/A | N/A | 0 |
| Calibrated | 0.60 | 322,667 | 207,232 | 64.2247% | 66.8154% | 123 |
| Calibrated | 0.70 | 79,286 | 56,050 | 70.6934% | 73.9071% | 123 |
| Calibrated | 0.80 | 4,245 | 3,417 | 80.4947% | 81.7807% | 114 |
| Calibrated | 0.90 | 0 | 0 | N/A | N/A | 0 |

At raw P >= 0.80, LONG is 1,723/2,121 (81.2353%) and SHORT 1,877/2,359
(79.5676%). At calibrated P >= 0.80, LONG is 1,318/1,598 (82.4781%) and SHORT
2,099/2,647 (79.2973%). These different subsets do **not** establish that
calibration is better or that an 80%-winning executable strategy exists.

### Raw Probability Versus Remaining-Session Outcomes

The October 5 follow-up audits **the same 686,112 raw forecasts**, without
retraining, remapping or choosing another model. Every probability reproduced
from the frozen weights; exact scaled-integer candle calculations matched all
60-minute labels. All 15:00 and session-close outcome windows were complete,
so the three columns below use identical actions and denominators.

The 3,600/4,480 result is indeed **within the literal next 60 minutes**. LONG
uses reference close times 1.002; SHORT uses times 0.998, without a stop-loss.
The original threshold is P >= 0.80, and that group's mean forecast is 81.93%.

| Raw Probability Band | Actions | Mean Forecast | TP Within 60m | TP By 15:00 | TP By Session Close |
|---|---:|---:|---:|---:|---:|
| 20%-30% | 258 | 28.75% | 33.33% | 52.71% | 55.04% |
| 30%-40% | 24,677 | 37.31% | 37.57% | 54.06% | 59.13% |
| 40%-50% | 137,187 | 45.79% | 46.40% | 61.69% | 67.07% |
| 50%-60% | 248,335 | 55.25% | 54.96% | 69.00% | 73.63% |
| 60%-70% | 207,270 | 64.42% | 63.08% | 75.48% | 78.84% |
| 70%-80% | 63,905 | 73.55% | 70.86% | 81.16% | 82.91% |
| 80%-90% | 4,480 | 81.93% | 80.36% | 87.08% | 87.97% |

Bands are lower-inclusive and upper-exclusive; no forecasts fall outside these
populated bands. The 20%-30% band is a small sample. The 70%-80% band is
overconfident by 2.68 percentage points; good average calibration is not perfect
calibration in every band, month or direction.

"Session close" means the recorded continuous-equity session: **15:30 before
August 3, 2026 and 15:15 thereafter**, using the migration's CAS policy and
excluding the closing auction. This does not change the user's **15:00 exit**,
and the historical CAS eligibility limitation remains.

| Measure, Same Actions | 60 Minutes | By 15:00 | By Continuous Close |
|---|---:|---:|---:|
| Actual TP touches | 389,127 | 481,684 | 509,933 |
| Actual TP-touch rate | 56.7148% | 70.2049% | 74.3221% |
| Mean unchanged raw probability | 57.3498% | 57.3498% | 57.3498% |
| Weighted 10-bin calibration error | 0.9014 pp | 12.8551 pp | 16.9723 pp |
| Row-weighted correlation of bin mean P with bin hit rate | 0.999379 | 0.997711 | 0.990457 |
| Individual P versus binary outcome correlation | 0.176385 | 0.153678 | 0.136118 |
| ROC AUC | 0.600981 | 0.595561 | 0.588257 |

**Interpretation:** the raw percentages track one-hour frequencies closely in
aggregate, with modest discrimination between individual hits and misses. The
near-one bin correlation is not 99.9% individual accuracy. For longer windows,
higher-probability bands still hit more often, but the numerical probabilities
systematically understate the changed event, and individual ranking is weaker.
A 60-minute probability must not be presented as a calibrated remaining-session
probability; that would need a separately specified and validated target study.

For the identical P >= 0.80 group: **3,600/4,480 within 60 minutes,
3,901/4,480 by 15:00 and 3,941/4,480 by continuous close**. These are price-touch
outcomes, not independent trades, realized win rates or net profits.

Remaining time to 15:00 is not always longer than an hour. There are 36,756
decisions after 14:00. Across the full cohort, 2,753 one-hour successes first
hit only after 15:00, while 95,310 other actions first hit after the hour but
before 15:00. In the 14:00-14:15 decision group, touch rate is 56.34% within the
literal hour but 50.72% by 15:00. Keep the two event definitions separate.

The [horizon audit](../data/research/tp_probability_horizons_kite_20261005/report.json)
contains full direction, month, decision-hour and threshold diagnostics. Its
protocol, executed analysis source and all outcome arrays are retained beside
it. Report SHA256:
`206dd5409c9e811228e5c1d0e9ec51cfda72ec3a064c5a99cdc164683bf39b1b`.
Research sample coverage is separately recorded in the dataset manifest:
1,897,197 candidate decisions, 99,850 missing-feature exclusions, 909
incomplete-future exclusions and 1,796,438 included decision pairs. Thus 5.31%
of candidates are excluded, including warm-up/prior-history unavailability;
this is not the raw candle-loss percentage. See the
[quality accounting](BROKER_DATA_QUALITY.md).

To repeat this offline audit, use a new output directory:

```powershell
& .\.venv\Scripts\python.exe -m scripts.analyze_tp_no_stop `
  --data data/research/tp_only_kite_20261005_final `
  --study data/research/tp_catboost_kite_20261005 `
  --cache data/candles `
  --validation data/kite_migration_20261005/validation.json `
  --output data/research/tp_probability_horizons_kite_NEXT
```

### What The Evaluation Terms Mean

A **horizon** is the allowed time for the price to touch TP: the next 60 minutes,
until the fixed 15:00 user cutoff, or until the recorded continuous-session
close. These are three different labels for the same decisions, not three
different train/test periods. The audited decisions run from **April 1, 2026
09:40 IST through September 28, 2026 14:15 IST**, across 123 sessions and 50
stocks. For example, a 10:00 decision has a one-hour deadline of 11:00, while
its other deadlines are 15:00 and that date's continuous close.

**Unseen by this model's fitting:** tree training ends June 30, 2025; July-
December 2025 chooses the candidate/early stop; January-March 2026 fits/selects
calibration. April-September 2026 was not used for fitting, early stopping,
hyperparameter choice or calibration in this run. It is an out-of-time test on
the same stock universe, not a test of unseen stocks. The project had examined
these calendar dates in earlier research, so it is not a pristine never-touched
holdout or prospective live/shadow evidence. Repeating audits does not add new
independent test observations.

**Calibration** means that, across many forecasts near 80%, approximately 80%
of the specified events happen. The one-hour band table shows good aggregate
calibration, with remaining band/month/side errors. It does not tell us which
individual calls will fail or the losses when they do.

**ROC AUC** measures ranking: pick one actual hit and one actual miss at random;
how often did the model give the hit the higher score, with ties worth half?
0.5 means no useful ranking, 1.0 is perfect. The observed AUCs are 0.600981 for
60 minutes, 0.595561 for 15:00 and 0.588257 for continuous close. Thus the first
score wins that comparison about 60.1% of the time. This is modest discrimination,
not 60.1% trade accuracy and not the 80%-band hit rate. A model can have good
group-average calibration and only modest individual discrimination.

### Frequency And Miss Returns: October 6

The new [frequency and return audit](../data/research/tp_frequency_returns_kite_20261006/report.json)
reuses the unchanged frozen forecasts and checks every 15:00 hit again. It
records **686,112 directional forecasts from 343,056 distinct symbol/candle
decision times**, with one LONG and one SHORT forecast at each. The cadence is
every five minutes from 09:40 to 14:15, not every one-minute candle.

An independent CSV count found **2,276,246 retained one-minute candles** for the
50 target stocks over those evaluation dates. Only **15.0711%** were scored on
this research schedule; **1,933,190 / 84.9289%** are unscored, not low-probability
forecasts. There is no evidence here for frequency from scoring every minute.

To give each scored candle exactly one bucket, this descriptive table uses
`max(LONG probability, SHORT probability)`. This does not select a trading side,
resolve ties or simulate a strategy. The return table below still uses all
directional forecasts in their original bands, not this maximum-side grouping.

| Higher-Side Probability Band | Scored Candles | % Of Scored Candles | % Of All Retained Minutes | Mean Candle-Times / Session |
|---|---:|---:|---:|---:|
| Below 30% | 0 | 0.0000% | 0.0000% | 0.00 |
| 30%-40% | 5,549 | 1.6175% | 0.2438% | 45.11 |
| 40%-50% | 56,088 | 16.3495% | 2.4641% | 456.00 |
| 50%-60% | 125,752 | 36.6564% | 5.5245% | 1,022.37 |
| 60%-70% | 114,971 | 33.5138% | 5.0509% | 934.72 |
| 70%-80% | 37,666 | 10.9795% | 1.6547% | 306.23 |
| 80%-90% | 3,030 | 0.8832% | 0.1331% | 24.63 |
| 90%-100% | 0 | 0.0000% | 0.0000% | 0.00 |
| Unscored minutes | N/A | N/A | 84.9289% | N/A |

For the original directional forecasts, the shares below sum to 100%. The daily
averages divide by all 123 sessions, including days without qualifying scores.
**"No TP" means never touched before or at 15:00**, not merely failed to hit
within the first hour. The cutoff price is the close of the 14:59-15:00 bar,
never the next bar. LONG return is cutoff/reference minus one; SHORT reverses
the sign. Returns are gross percentages of entry notional, excluding fees,
spread, slippage, leverage and portfolio overlap.

| Raw Forecast Band | Forecasts | % Of All Forecasts | Mean Forecasts / Session | No TP By 15:00 | Mean Return For Those Misses |
|---|---:|---:|---:|---:|---:|
| 20%-30% | 258 | 0.0376% | 2.10 | 122 | -0.1615% |
| 30%-40% | 24,677 | 3.5966% | 200.63 | 11,336 | -0.2654% |
| 40%-50% | 137,187 | 19.9948% | 1,115.34 | 52,554 | -0.3519% |
| 50%-60% | 248,335 | 36.1945% | 2,018.98 | 76,978 | -0.4646% |
| 60%-70% | 207,270 | 30.2094% | 1,685.12 | 50,817 | -0.6640% |
| 70%-80% | 63,905 | 9.3141% | 519.55 | 12,042 | -0.9870% |
| 80%-90% | 4,480 | 0.6530% | 36.42 | 579 | -1.3930% |
| All forecasts | 686,112 | 100.0000% | 5,578.15 | 204,428 | -0.5074% |

Overall non-hit mean is -0.507360%, median -0.361862%, and worst -7.481425%.
LONG has 106,834 non-hits averaging -0.490304%; SHORT has 97,594 averaging
-0.526031%. Some non-hits still close slightly positive below the +0.20% target;
they remain in the conditional mean, not discarded as non-losses.

At P >= 0.80, the 579 non-hits average **-1.392984%**, median -1.180858%, with a
worst -4.841853%. Of those, 570 are negative and nine positive but below TP.
The 4,480 opportunities occur across 113 sessions, with ten zero-opportunity
sessions. Daily forecast count has mean 36.42, median 20 and maximum 364. They
represent 3,030 distinct scored candle-times; **1,450 of those qualify in both
directions**. Repeated signals in one stock and signals during an open position
would need suppression by an explicit trading policy. These are opportunity
counts, not an estimate of actual executed trades or a daily quota.

### Why Good Calibration Is Not Yet A Profitable Strategy

The high-probability bucket illustrates the missing economic dimension. If
each of its 3,901 TP hits earns exactly +0.20% and each of its 579 misses uses
the measured 15:00 return, the equal-notional per-opportunity mean is:

```text
(3,901 * 0.20% + 579 * -1.392983565%) / 4,480 = -0.005878903%
```

That is slightly negative **before costs**, despite an 87.0759% TP-by-15:00
hit rate. With that miss severity, the gross break-even hit rate would be
87.4449%, before adding transaction costs. Across all forecasts, the same
idealized gross average is -0.010758951%. Neither is portfolio P&L: fills,
capital, simultaneous holdings and duplicate opportunities are not simulated.
This does not test a newly chosen one-direction-per-candle strategy. It does
show why touch calibration and a high hit rate alone cannot establish a
profitable no-stop strategy. No stop-loss was introduced and no threshold was
selected by looking at these new returns.

All **686,112 cutoff returns** were independently recomputed directly from CSV
closing prices with Decimal arithmetic and matched the saved arrays exactly.
There are no unavailable returns in this cohort. Tests check cutoff inclusion,
SHORT signs, missing-window exclusion, paired-candle counts and conditional
averages. The audit retains its protocol, executed source and per-action
returns; the earlier horizon audit/model/weights/calibration remain unchanged.
Report SHA256:
`e7eeae02013c03f1fd1c6c09cee248080c9eab507dedc71e9a52cee177a4e112`.

Reproduce only into a new output directory:

```powershell
& .\.venv\Scripts\python.exe -m scripts.analyze_tp_no_stop `
  --data data/research/tp_only_kite_20261005_final `
  --horizon-audit data/research/tp_probability_horizons_kite_20261005 `
  --cache data/candles `
  --output data/research/tp_frequency_returns_kite_NEXT
```

### Model Interpretation

CatBoost `PredictionValuesChange` importance: current 20-minute volatility
49.59%, prior-five-cutoff volatility 13.34%, side 9.64%, minutes from open 6.82%,
day return 5.50%, VIX 4.78%. Other features collectively contribute 10.33%.
This is model sensitivity, not causal evidence or an ablation. The dominant
volatility features are consistent with predicting small target touches in
either direction, not necessarily profitable directional trades.

## Frozen Artifacts And Verification

| Artifact | Purpose |
|---|---|
| [dataset manifest](../data/research/tp_only_kite_20261005_final/manifest.json) | Active Kite shards and source fingerprints |
| [saved best.cbm](../data/research/tp_catboost_kite_20261005/final/best.cbm) | Exact validation-selected depth-four, 462-tree weights |
| [preprocessing.json](../data/research/tp_catboost_kite_20261005/final/preprocessing.json) | Ordered model-input/preprocessing metadata; keep with the weights |
| [fresh baseline report](../data/research/tp_catboost_kite_20261005_baseline/report.json) | Depth-six reference fitted on Kite, not old weights |
| [study protocol](../data/research/tp_catboost_kite_20261005/protocol.json) | Bounded search design and executed source hashes |
| [model selection](../data/research/tp_catboost_kite_20261005/model_selection.json) | Validation ranking and pre-calibration winner lock |
| [model lock](../data/research/tp_catboost_kite_20261005/final/model_lock.json) | Exact 462-tree weights, preprocessing and schema |
| [calibration](../data/research/tp_catboost_kite_20261005/final/calibration.json) | Temporal mapping fit/check and selected coefficients |
| [selection lock](../data/research/tp_catboost_kite_20261005/final/selection_lock.json) | Mapping, controls and thresholds locked before evaluation |
| [final report](../data/research/tp_catboost_kite_20261005/final/report.json) | Frozen evaluation, reliability and block uncertainty |
| [independent verification v2](../data/research/tp_catboost_kite_20261005/verification_v2.json) | Exact counts, sklearn metrics, AUC, ECE and full forecast reload |
| [October 7 reverification](../data/research/tp_catboost_kite_20261005/verification_20261007.json) | Fresh successful reload of all validation candidates, seed repeat and 686,112 evaluation forecasts; no refit |

The study also retains all seven trial directories, the seed repeat, executed
trainer/study source copies and final prediction arrays. **All 686,112 saved
raw and calibrated forecasts reproduce exactly** after model reload. Independent
scikit-learn losses/Brier and validation rankings agree. Audit v2 corrects an
exact-decimal-boundary ECE implementation issue; the original audit is retained,
and no model, selection, probability or frozen report was rewritten.

Key SHA256 fingerprints:

```text
dataset manifest  5ccd2091a19f94e896f9570fbfb58bcf04998a18b523a217d501a3c8c7651c81
study protocol    1df30b18e0388071cc9f84dd0a2dd768e7f33eac66ee4e4a8bd47c85523e0203
final best.cbm    b8d86c6338096ea8ce3f1e9ce0cb0297f8be2dbcb84a056534c0b33752724aad
calibration       2090378d6407f2a0db93d7b3b825b69b8f547cdd076eb8d357b02c355bf64f0b
selection lock    6bc6909ec82be45b5406c33eee198e8db433b8bdb4962441139f47dbfabf1e8e
```

These artifacts are Git-ignored local files, not a remote model registry or
backup. Retain them separately. Never overwrite this study, relabel old weights
as Kite, or change a pinned feature/data source under the same manifest.

Offline verification, using a **new** output filename:

```powershell
& .\.venv\Scripts\python.exe -m scripts.report_tp_catboost `
  --data data/research/tp_only_kite_20261005_final `
  --study data/research/tp_catboost_kite_20261005 `
  --baseline data/research/tp_catboost_kite_20261005_baseline `
  --output data/research/tp_catboost_kite_20261005/verification_NEXT.json
```

For offline inference, `scripts.research_tp_catboost.predict_frozen` accepts
the final artifact directory and a finite NumPy matrix with the 14 ordered
columns. It checks the lock and returns the selected calibrated probabilities.
`load_frozen_model` plus the trainer's `predict_logits` and `scipy.special.expit`
provides the predeclared raw comparator. Neither path places orders or creates
a runtime scanner. Do not use the 63-field generic feature snapshot as this
14-column input without an explicit, tested mapping.

## Azure Retirement

The user authorized deletion of obsolete remote training artifacts while
preserving reusable infrastructure. Cleanup completed October 5 at
**17:37:15 UTC / 23:07:15 IST**. The
[receipt](../data/azure_cleanup_20261005/receipt.json) records **753 blobs /
384,257,861 bytes deleted**, all eight old jobs archived, zero remaining blobs
in the two inventoried containers and zero compute nodes. Blob soft delete was
disabled. Archive retains job metadata; it is not a claim of metadata erasure.

The [reviewed plan](../data/azure_cleanup_20261005/plan.json) matched only the
three job-recorded Groww dataset uploads and the eight jobs' outputs, logs and
setup artifacts. Live ETags, job dependencies and idle compute were checked
before writes; deletion used per-object ETag conditions and a verified receipt.
One batch-authentication failure required a tested in-memory token cache and
receipt-checked retry; all 753 targets were subsequently verified absent.

Preserved: `rg-training-model`, workspace `mlw-tp-research`, storage
`sttpresearchc34920261003`, `kv-tpresearch-c349`, `law-tp-research`,
`appi-tp-research`, datastores, reusable code/environment registrations, and
`cpu-tp-training` (Standard_DS4_v2, min 0/max 3). Azure Files shares were outside
the deletion scope and were not audited for content. No new Kite data/model was
uploaded. Old jobs cannot be replayed from their deleted inputs; a future job
needs a new hash-verified Kite bundle. Source training helpers remain reusable.

The earlier USD100 experiment envelope was not a hard billing limit. This
local study created no Azure training-job spend, but retained storage and
monitoring can still incur charges. No resource-group deletion, role expansion,
tenant change or live model deployment was performed.

## Readiness And Next Work

The latest workspace regression gate after the market-hours runner and guards
is **1,005 tests passed in 109.30 seconds** on October 7. The
focused tests cover no-later-phase selection, exact reloaded predictions,
corruption refusal, ECE boundaries, explicit TP deadlines, exact-price batch
outcomes, conditional cutoff returns, paired-candle frequency denominators,
quarantine-slot accounting, and cleanup scope/ETags/partial retry.
The October 6 capture was outside the trained decision window. The October 7
capture now produced five eligible target records at 12:45, or ten directional
forecasts, with complete inputs and exact saved-tick replay. Ten later records
were correctly withheld after the controlled reconnect. Independent batch
arithmetic matched all 195 TP input values across the 15 records at 60-digit
reference precision; the model's weights/calibration were not changed. All five
issued records' raw/calibrated probabilities matched subsequent-history outputs.
This is technical in-window evidence, not prospective TP accuracy or profitability.
Exact historical candle parity still failed: 74/120 OHLCV rows, 91/120 OHLC rows
and 58/120 generic-feature rows match. Read
[the live-validation findings](BROKER_DATA_QUALITY.md).

**Next: scanner-layer research using this frozen baseline.** Superseded for the
09:40-13:30 window by the October 10 study at the top of this report. The user scoped
technical validation to the built components; missing future services and exact
quote/history equivalence are not failed research prerequisites. The construction
investigation found no observed candle arithmetic bug and retained the current
algorithm. Use [saved captures](LIVE_API_SAMPLES.md#offline-development-first)
to measure source sensitivity when relevant to a proposed selection policy.

Predeclare the comparison, eligibility/side/cadence/threshold choices, duplicate
handling and outcome definitions. Keep rules, raw forecasts and the selected
calibrated output separately identifiable; do not infer a deployed P >= 0.80 or
max-side policy from diagnostic tables. Evaluate frequency, false signals,
coverage, TP-or-15:00 loss tails and explicitly assumed costs, not only next-hour
hit rates. The known negative high-score gross-return result remains important.
The studied dates are already examined, so new work on them is exploratory,
not a pristine confirmatory holdout. See the
[scanner research handoff](ARCHITECTURE.md#scanner-research-handoff).

Model weights, calibration and this evaluation stay frozen. No hindsight
reselection, automatic retraining, capital deployment or order enablement is
implied. Broader forward shadow and automatic recovery can be scoped separately
when needed; no new live session is required just to resume research.

Current membership, current tick sizes, vendor-adjusted history, observed
calendar coverage, CAS eligibility, complete-case selection and overlapping
horizons limit interpretation. A TP touch is not a fill. The measured no-TP
cutoff-return distribution is a gross price diagnostic, not realized portfolio
losses or net return. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full current/target boundary and
[BROKER_DATA_QUALITY.md](BROKER_DATA_QUALITY.md) for source validation limits.

<details>
<summary>Retired Groww protocols and results, not current Kite instructions</summary>

## Historical Protocol And Results

Everything below records pre-migration experiments. Their datasets, model
weights and local output paths were deleted; their eight remote jobs are now
archived and old blob payloads deleted. Preserve the rationale as history,
not instructions to restore, rerun or promote those results.

## Contract

The current research target is the probability that price touches a directional
0.20% target within the next 60 minutes. The reference is the last completed
one-minute candle close. All 60 subsequent complete minutes count, including the
first; adverse moves do not end the event. Stop loss is deliberately absent.
LONG and SHORT may both succeed. This is not executable-fill or profit probability.

There are 13 numerical inputs plus direction, with one logit and a calibrated
sigmoid probability. Stock identity, date, TP and horizon are not model inputs.
Stock and timestamp remain audit metadata. Decisions are every five minutes from
09:40 through 14:15 IST, with a full future window ending by 15:15.

The exact ordered columns and formulas are in the dataset manifest and
[`../src/ai_trader/features/tp_only.py`](../src/ai_trader/features/tp_only.py).
This bounded state uses the existing Decimal context and session VWAP primitive.
The 63-field FeatureSnapshot contract and live scanner defaults are unchanged.

## Prior History

The earlier 45.45% missingness was feature unavailability on 20 of 44 sampled
July/August sessions, not 45% missing minute data. In four audited stocks, August
sessions lack 15:16-15:19 and sometimes later minutes. The old whole-session
feature contract requires every one of 375 minutes and remains unchanged.

The new contract separates data requirements:

- Gap uses today's actual 09:15 open and the preceding vendor daily close.
- Five-session return needs six ordered daily closes and no missing intervening
  market session. It is close-to-close, not the old five-day open-to-close field.
- Prior volatility is the square root of the mean of five complete
  09:15-15:15 session realized variances. Each includes open-to-first-close plus
  consecutive close-to-close log returns. It is not full-day volatility.
- Same-minute volume needs that minute on the preceding five market sessions.
- Missing features exclude the decision. Missing future minutes are never
  converted into a negative label. No older session replaces a missing one.

Groww's sampled daily OHLC rows omit open. The independent `get_daily_closes`
broker method validates dates and positive finite closes without inventing open;
the strict OHLC reader is unchanged and the minute cache rejects daily candles.
Four stocks across June-August returned all 65 dates each, with all closes inside
their observed session range. Daily closes often differ from final minute trades;
August values frequently equal the final available minute. These checks do not
establish official exchange-close semantics or corporate-action adjustment.

The full snapshot invalidated 21 vendor closes outside their observed ranges.
Missing or invalid endpoints remain unavailable. Calendar dates come from
observed NIFTY sessions, not an independently verified exchange holiday calendar.

## Frozen Dataset

Local version: `data/research/tp_only_20261003`.
Daily-close snapshot: `data/research/tp_only_daily_20261003`.
Both are ignored by Git and must be retained separately.

| Phase | Dates | Side-Action Rows |
|---|---|---:|
| Training | January 2024-June 2025, after warm-up | 1,733,876 |
| Validation | July-December 2025 | 564,340 |
| Calibration | January-March 2026 | 335,060 |
| Retrospective evaluation | April-September 28, 2026 | 644,418 |

There are 199 compressed shards, about 90 MB. From 1,848,946 candidate decisions,
208,646 lack required features and 1,453 further decisions lack full future
coverage; 1,638,847 decisions produce 3,277,694 paired side-actions.
Training and validation contain 347 and 106 represented session dates.

Manifest SHA256:
`78325450cabbf6a14b6a861e452f5286ec125c4c7189e7bddbadf2bbce84a001`.

The manifest pins source data, feature code, schema, dates and every shard hash.
The builder refuses an existing output directory. Model inputs use only the
allowlisted feature array, never audit keys or excursion/label arrays.

## Training

[`../scripts/train_tp_only.py`](../scripts/train_tp_only.py) uses the authors'
`rtdl-revisiting-models==0.0.2` FTTransformer with PyTorch 2.9.1:
three blocks, width 192, eight attention heads, 897,217 parameters, attention
dropout 0.2, feed-forward dropout 0.1, batch size 1024, AdamW learning rate 1e-4
and weight decay 1e-5 with the reference parameter groups.

Training-only 0.5/99.5-percentile clipping and standardization are frozen before
validation. Direction stays -1/+1. Binary cross-entropy uses logits and the
natural label frequencies, without class balancing. Checkpoints are selected
by validation log loss, with patience eight and at most 40 epochs.
Monotone logit calibration fits the later calibration period. Weights,
preprocessing and calibration are hashed in a selection lock before the trainer
opens evaluation shards. Both raw and calibrated results, per-side reliability
bins and frozen training-side base-rate controls are reported.

`--omit-prior-history` provides a same-row feature ablation for a separately
authorized run. The first FT model is a candidate, not a demonstrated winner over
residual MLPs or boosted trees. No claim that the history change improves model
performance can be made before chronological comparison. Test-period inspection
must not be used to retune the current run.

A local eight-thread benchmark of ten batches after two warm-up batches measured
0.799 seconds per 1024-row training batch: about 22.5 minutes per training epoch
and 15 hours for 40 epochs, excluding validation, initialization and Azure
hardware differences. This is a rough sizing benchmark, not a completion SLA.

## Azure Run

The user authorized an initial USD100 total experiment envelope. On October 4,
the user additionally authorized the six fixed comparison trials below, with at
most three concurrent jobs including the reference. No unbounded search, GPU
quota increase or live deployment is authorized.

- Resource group: `rg-training-model`, East US.
- Workspace: `mlw-tp-research`, identity-based system datastores.
- Compute: `cpu-tp-training`, Standard_DS4_v2, eight vCPUs and 28 GiB RAM,
  minimum zero and maximum three nodes, 180-second idle scale-down, no public SSH
  or node IP, system-assigned identity. The maximum was increased from one by
  a scoped compute PATCH for the comparison batch; the other settings are unchanged.
- Storage: `sttpresearchc34920261003`, Standard LRS, anonymous blob and shared-key
  access disabled. Public authenticated endpoint enabled for local upload;
  this is not a private-endpoint-only storage account.
- Dependencies: `kv-tpresearch-c349`, `law-tp-research`, `appi-tp-research`.
  Log Analytics has a 0.1 GB/day ingestion cap. Managed network allows outbound
  internet for public training dependencies. No broker secrets are uploaded.
- Submitting user and compute identity have Blob Data Contributor on this
  storage account only. No subscription-wide role was added.
- Reference job: `tp-only-ft-20261003-v2`, submitted October 3 and verified Completed
  on October 4. It stopped early after 11 epochs; best validation log loss was
  0.67913583 at epoch 3. Its code asset and 36-hour fit / 40-hour job limits were
  unchanged. Its evaluation report was not used to select subsequent models.
- Original job `tp-only-ft-20261003-v1` failed before the first training epoch.
  Azure's input downloader materialized `M&M.npz` as `M%26M.npz`, causing a
  filename mismatch with the manifest. Its node scaled to zero after failure.

[`../scripts/azure_tp_job.py`](../scripts/azure_tp_job.py) stages only one trainer,
the manifest and the verified shards: 201 files, 89,687,262 bytes for v2. It never
uploads the workspace root, credentials, broker code or internal Azure reports.
The remote job downloads the dataset, trains in an isolated Python 3.12 CPU
environment, and mounts persistent model output for checkpoints and reports.
After successful submission it does not depend on the local PC or chat.

The v2 bundle is `data/azure/tp_only_bundle_20261003_v2`. Staging now encodes shard
basenames as UTF-8 hex, for example `M&M.npz` becomes `shard-4d264d.npz`. These
names are collision-free and unchanged by URL encoding. Only staged manifest
paths change; the original manifest, all 199 shard bytes, labels, feature values
and trainer remain unchanged. A regression test reproduces the old downloader
failure and loads all four phases through the URL-encoding simulation.
The real 1,733,876 training and 564,340 validation rows also loaded successfully
from the corrected bundle; evaluation labels were not inspected during repair.

Transport manifest SHA256:
`0fdeee8287764e5258824a151234ec0abc26a0a1fcde118f9e5957f97ce43d9f`.
Its `source_manifest_sha256` links to the unchanged frozen dataset above.
Azure read-back confirmed the same trainer asset, environment, command and limits.

The fit is capped at 36 hours; the job has a 40-hour timeout including setup and
evaluation. At the checked public Linux DS4_v2 rate of USD0.585/hour, 40 node
hours is USD23.40 compute-only, not an invoice forecast. Storage, private network
dependencies and logs also cost money, including after nodes scale to zero.
USD100 is an approved envelope, not an Azure hard billing stop or a tag-enforced
limit. The six-trial reservation below is additional to this reference allowance.
Jobs beyond these six require remaining-budget review; persistent resources need
cleanup or cost review after the experiment.

The ML CLI extension installer failed; the verified fallback uses
`azure-ai-ml==1.30.0`, `azure-identity==1.25.1` and AzureCliCredential. Azure MCP
startup also failed; ARM deployment tools and the SDK were used without changing
Azure login defaults or inspecting tokens.

```powershell
& .\.venv\Scripts\python.exe -m scripts.azure_tp_job status --name tp-only-ft-20261003-v2
```

## Parallel Comparison: October 4

Six command jobs have been submitted under experiment `tp-compare-20261004`.
Each job name is that prefix followed by its trial suffix below. Read-back on
October 4 at 15:44 IST confirmed all six and the reference Completed; all stopped
early, with no timeouts. Node enumeration at 15:46 IST confirmed zero allocated
nodes. The cluster allowed at most three single-node jobs concurrently, including
the reference. Supporting resources can still incur charges at zero nodes.

| Trial | Model / Architecture | Parameters | Learning Rate | Weight Decay / L2 | Dropout / Sampling |
|---|---|---:|---:|---|---|
| Existing reference v2 | FT-Transformer, 3 blocks, width 192, 8 heads | 897,217 | 1e-4 | AdamW 1e-5 | attention 0.2, FFN 0.1 |
| ft-low-lr | FT-Transformer, 3 blocks, width 192, 8 heads | 897,217 | 3e-5 | AdamW 1e-5 | attention 0.2, FFN 0.1 |
| ft-high-lr | FT-Transformer, 3 blocks, width 192, 8 heads | 897,217 | 3e-4 | AdamW 1e-5 | attention 0.2, FFN 0.1 |
| ft-regularized | FT-Transformer, 3 blocks, width 192, 8 heads | 897,217 | 1e-4 | AdamW 1e-3 | attention 0.3, FFN 0.2 |
| resnet | ResNet, 4 blocks, width 256, hidden width 512 | 1,058,305 | 1e-3 | AdamW 1e-4 | hidden 0.2, residual 0 |
| tabm | TabM, 3 blocks, width 512, 32 shared ensemble members | 679,392 | 2e-3 | AdamW 3e-4 | hidden 0.1 |
| catboost | Symmetric trees, depth 6, maximum 2,000 trees | Not comparable to NN weights | 0.03 | leaf L2 5 | Bernoulli subsample 0.8 |

All trials use seed 20261003 and eight CPU threads. Neural candidates use batch
size 1024, at most 40 epochs, validation patience eight, gradient clipping at 1,
and no class balancing or learning-rate schedule. FT residual dropout is zero.
FT uses the authors' parameter groups; ResNet and TabM use AdamW over their
parameters. A singleton trailing training batch is merged into the preceding
batch so ResNet's batch normalization remains valid without dropping rows.

TabM uses the reference `arch_type="tabm"`, without numerical embeddings in this
initial comparison. Each member's BCE is optimized; inference averages member
probabilities, not logits. The 32 members share weights and occupy one Azure node,
not 32 jobs. CatBoost uses CPU Plain boosting, 254 numerical borders, random
strength 1, validation patience 100 trees, and retains the best validation model.
Neural-only fields in a CatBoost configuration are unused, not tree settings.

### Research Rationale

- [RTDL's FT-Transformer and ResNet study](https://arxiv.org/abs/2106.11959) and
  [reference implementation](https://github.com/yandex-research/rtdl-revisiting-models)
  provide the architectures and published search ranges. The FT trials bracket
  the reference learning rate; the regularized trial tests increased decay and
  dropout. ResNet supplies a different inductive bias at comparable parameter count.
- [TabM, ICLR 2025](https://arxiv.org/abs/2410.24210) and its
  [official usage and tuning guidance](https://github.com/yandex-research/tabm)
  motivate the efficient ensemble control. The authors recommend AdamW learning
  rate 0.002, decay 0.0003 and 32 members as starting settings. Their numerical
  embedding variants can be stronger; this raw-feature TabM trial does not cover
  that search space. The abstract and official guidance were reviewed, not a
  complete reproduction of every paper experiment.
- [CatBoost's tuning guidance](https://catboost.ai/en/docs/concepts/parameter-tuning)
  supports depth six, a conservative learning rate and validation early stopping.
  The tree control is important because [tabular benchmarking](https://arxiv.org/abs/2305.02997)
  does not support assuming neural networks universally win.

These are six research-informed candidates, not known optimal hyperparameters.
None of these papers establishes NSE one-hour target-touch prediction quality.
Fixed seeds and one configuration per non-FT family do not measure seed variance
or exhaust each family's tuning space. Time limits also allow different numbers
of epochs across architectures; elapsed time and actual progress must accompany
the final comparison.

### Selection And Provenance

The trial bundle contains only the 99 training/validation shards: 1,733,876 and
564,340 rows respectively. Calibration and evaluation files are physically absent.
Inputs, labels and chronological boundaries match the reference frozen dataset.
Neural clipping/scaling is fitted on training only; CatBoost receives the same
raw feature rows without clipping or standardization. This preprocessing difference
is intentional and is recorded in each model's artifact.

Primary selection is minimum chronological validation log loss. Report Brier loss,
training-side base-rate controls, and per-side / per-month behavior alongside it.
Each trial writes reloadable best weights, progress, preprocessing, validation
predictions and a report with hashes. Trials stop before calibration or evaluation;
the chosen configuration must be locked before any later calibration/evaluation
workflow. Do not use the reference's evaluation results to choose this batch's
hyperparameters. Multiple comparisons can still overfit validation.

Frozen bundle: `data/azure/tp_compare_bundle_20261004`. Its `trial_plan.json` stores
all six full configurations and their hashes; `submission.json` records accepted
job IDs and Studio links. Retain this ignored local evidence separately from Git.
Only the seven allowlisted code/config files (31,741 bytes) and training/validation
dataset (62,728,412 bytes) were uploaded, not the workspace or broker credentials.

- Source manifest SHA256 remains
  `78325450cabbf6a14b6a861e452f5286ec125c4c7189e7bddbadf2bbce84a001`.
- Trial transport manifest SHA256:
  `bcd0b62e4ede7e978e9db8cafcb8f610b95e92c6795a0fc07acb3b9ab7e43ce4`.
- Trial trainer SHA256:
  `57009d18858a5baf849df41ef3212440decf811d8f8c5bba3ff43c7151071a9e`.
- Trial plan SHA256:
  `e8e8b1df482b0707180b753aea26c96f45f903fcfff7967126d47a798058293b`.

The new code asset is `e985b09a-9ab9-42df-bb66-a3b7ba598746`, version 1. The reference
retains `d2bb88ad-a2a4-4234-8388-92a1a6793676`, version 1. Trial dependencies pin
PyTorch 2.9.1 CPU, RTDL 0.0.2, TabM 0.0.3, numerical-embeddings 0.0.12,
CatBoost 1.2.10, NumPy 2.5.3 and SciPy 1.16.1 in an isolated Python 3.12 environment.

### Limits And Verification

Each new fit is capped at 10 hours, leaving headroom within the 12-hour Azure job
limit for setup, validation and artifact output. Six full job allowances reserve
72 node-hours, or USD42.12 at USD0.585/hour. Reference plus batch allowances total
USD65.52 compute-only. The USD34.48 remainder of the approved envelope must cover
the prior failed attempt, startup/idle time and supporting services; actual billing
lags and no hard USD100 spend stop is installed. At three active nodes, compute
accrues at USD1.755/hour. No new GPU, workspace, role grant or service was created.

The batch launcher verifies hashes and allowlists, refuses mismatched existing job
names, and resumes partial submission without creating duplicate trials. Azure
read-back confirmed all six one-node / 43,200-second limits and the preserved
cluster identity, network flags, VM size and 180-second idle scale-down.

Full offline suite: 1,064 tests passed; Ruff lint and all 121 formatting checks
passed. Tests include real fitting/checkpoint reload for all four model families,
the JSON-config CLI, phase isolation, TabM loss semantics and bounded/idempotent
submission. Full-size local synthetic batches measured approximately 0.073 seconds
for ResNet and 1.335 seconds for TabM, excluding validation and cloud differences.
These are sizing observations, not market-data training results or cloud SLAs.

```powershell
& .\.venv\Scripts\python.exe -m scripts.azure_tp_job status --name tp-compare-20261004-tabm
```

## Completed CatBoost Follow-Up

The user authorized further CatBoost research, bounded variants and calibration
on October 4. The original depth-six CatBoost was the seven-model comparison's
validation winner. Its model, report and preprocessing were downloaded from the
completed Azure job and hash-verified. Reloading it on the original validation
rows reproduced log loss 0.6757031545 before any new fitting.

[`../scripts/research_tp_catboost.py`](../scripts/research_tp_catboost.py) then
ran five deliberately small refinements and one seed repeat on the local CPU.
No Azure training job or new infrastructure was started. Each new fit had an
eight-thread, 1,200-second limit; all stopped early. Their combined fitting time
was about 5 minutes 50 seconds, excluding data loading and later analysis.

### Variant Results

All configurations retained the same data, raw features, Bernoulli subsample 0.8,
254 numerical borders, Plain boosting and symmetric trees. Learning rates,
depths and L2 values were fixed before fitting from the official CatBoost tuning
guidance already cited above, not selected from later evaluation outcomes.
The tree limit was 2,000 with patience 100, except the slower-learning-rate
candidate used 4,000 with patience 200. Seed was 20261003.

| Candidate | Depth | Learning Rate | Leaf L2 | Retained Trees | Validation Log Loss |
|---|---:|---:|---:|---:|---:|
| Selected: depth4 | 4 | 0.03 | 5 | 607 | 0.675096 |
| slow_lr | 6 | 0.015 | 5 | 681 | 0.675460 |
| strong_l2 | 6 | 0.03 | 20 | 346 | 0.675568 |
| Original Azure model | 6 | 0.03 | 5 | 299 | 0.675703 |
| fast_lr | 6 | 0.06 | 5 | 163 | 0.675714 |
| depth8 | 8 | 0.03 | 5 | 210 | 0.675890 |

Selection used the same 564,340 validation rows from July-December 2025. The
absolute improvement over the original is only 0.000608, not a large gain.
A separate depth-four fit with seed 20261004 retained 573 trees and reached
0.675275. This repeat was diagnostic only; it did not replace the selected model
or form an ensemble. One repeat is not a comprehensive seed-variance study.

### Calibration Selection

The 607-tree weights and calibration protocol were locked before loading any
calibration rows. The base model was not refitted. Following the
[probability-calibration guidance](https://scikit-learn.org/stable/modules/calibration.html),
calibration used dates disjoint from fitting and validation, with no shuffled
cross-validation. Candidates were identity, global monotone logit sigmoid, and
separate monotone logit sigmoids for LONG and SHORT. Sigmoid slopes were bounded
to [0, 10] with the existing identity-centered penalty 0.001. Isotonic was not
included: its additional flexibility was not justified for this small set of
dependent market sessions despite the large row count.

January-February supplied 228,660 rows across 41 sessions for fitting mappings.
March supplied 106,400 rows across 19 sessions for choosing the mapping. The
predeclared selector minimized March log loss, provided overall Brier and each
direction's log loss were no worse than identity; exact ties preferred simplicity.

| Mapping | March Log Loss | March Brier |
|---|---:|---:|
| Identity / raw | 0.623321 | 0.216213 |
| Global sigmoid | 0.622208 | 0.215663 |
| Selected: per-direction sigmoid | 0.622102 | 0.215612 |

The selected mapping family was refitted on all 335,060 January-March rows, then
hashed and locked before evaluation. Final LONG slope/intercept were
0.90883839 / 0.08193463; SHORT values were 0.87879613 / 0.17958065. These are
stored parameters, not runtime fitting. Raw probabilities remained a predeclared
comparator and were not chosen instead after examining evaluation.

### Later Evaluation

Only the selected base model was evaluated on April-September 28, 2026:
644,418 paired side-action rows across 116 represented sessions. This period is
retrospective, not pristine prospective evidence, because parts of 2026 had been
examined in older studies. No configuration or calibration changes followed these
results. Lower log loss and Brier are better; neither is an accuracy percentage.

| Output / Control | Log Loss | Brier | Mean Forecast | Actual Touch Rate |
|---|---:|---:|---:|---:|
| Locked calibrated output | 0.670667 | 0.238961 | 58.99% | 56.23% |
| Raw comparator, same weights | 0.669367 | 0.238355 | 56.57% | 56.23% |
| Frozen training-side priors | 0.687903 | 0.247348 | 60.00% | 56.23% |
| Frozen calibration-side priors | 0.692027 | 0.249296 | 62.07% | 56.23% |
| Constant 50% | 0.693147 | 0.250000 | 50.00% | 56.23% |

The calibrated model beats the frozen controls but calibration itself did not
transfer reliably. It worsened loss relative to raw on both directions and in
five of six months; June was the exception. Ten-bin calibration error increased
from 0.86 to 2.77 percentage points. Raw LONG forecast 55.45% versus 54.66%
observed, and raw SHORT 57.68% versus 57.80%; after calibration these forecasts
rose to 56.94% and 61.04%. This is consistent with changing event rates, not
evidence that probability calibration always helps.

A paired circular moving-block bootstrap used five consecutive trading dates,
all stocks and both sides together, 1,000 repeats and seed 20261004. The
calibrated-minus-raw log-loss increase was 0.001299, with descriptive 95% interval
[0.000489, 0.002246]. This accounts for some temporal dependence but does not
establish stationarity or eliminate retrospective/model-selection bias.

Predeclared thresholds are diagnostics, not optimized trading rules:

| Output | Probability At Least | Rows | Mean Forecast | Actual Touch Rate |
|---|---:|---:|---:|---:|
| Raw comparator | 0.60 | 241,145 | 66.69% | 65.28% |
| Raw comparator | 0.70 | 57,497 | 74.15% | 72.00% |
| Raw comparator | 0.80 | 3,914 | 81.88% | 80.56% |
| Locked calibrated output | 0.60 | 302,747 | 66.66% | 63.91% |
| Locked calibrated output | 0.70 | 70,351 | 73.86% | 70.99% |
| Locked calibrated output | 0.80 | 3,624 | 81.67% | 80.10% |

Neither output produced probabilities at least 0.90. Actions overlap in time and
share market exposure; they are not independent executed trades. TP touch ignores
SL, fills, fees, position overlap and the cost of unsuccessful exits. No profitable
or dependable live policy is established, and there is no automatic promotion.
Preserve both frozen outputs for a separately specified prospective shadow check;
do not redesign calibration using these evaluation outcomes and call it unseen.

### Artifacts And Verification

The completed, Git-ignored study is
[`../data/research/tp_catboost_followup_20261004`](../data/research/tp_catboost_followup_20261004).
It retains the pre-fit protocol, source snapshot, all variant reports/weights,
validation predictions, selection, seed diagnostic, March check predictions,
final model/calibration locks and full later-period predictions. Original Azure
artifacts and all older frozen research remain unchanged.

The research-only `predict_frozen(output, features)` function verifies artifact
hashes and returns the locked calibrated probability for each finite 14-column
row in the recorded order. It does not load credentials, retrain, size positions
or send orders. The existing scanner is not wired to this model.

- Final weights SHA256:
  `002d1eec3be38b2e335c772a4918001f339eb318ff9249cc40d941f95dc1ffd2`.
- Final selection-lock SHA256:
  `3fae3b58b570364d255dc3a817f709d9fe6d48c3a4edb25abbdb4b18c4321d60`.
- Protocol SHA256:
  `d493fe3cb084a76ea9bf25e7e2298fc902424dcaf32acf4bfe0442fbfc4e9e5c`.

All 644,418 exported probabilities exactly matched reloaded inference. Independent
scikit-learn log-loss/Brier calculations matched the report, and source, data,
weights, calibration and report hashes passed. The full offline suite passed
1,068 tests; Ruff lint, all 122 formatting checks and editor diagnostics passed.
Tests cover chronology, no base-model refit, identity fallback, one-direction
regression rejection, frozen-output refusal, tamper detection and paired blocks.
These checks establish implementation consistency, not live model quality.

## No-Stop Accuracy: October 5

The user clarified that calls exit only at TP or 15:00 IST, with no price stop
and no forced 60-minute exit. The earlier 2-ATR execution scenario below does
not represent this contract and must not be cited as its performance. The model
and original data remain frozen; this is a separate outcome analysis, not new
training, threshold selection or a change to the legacy scanner defaults.

[`../scripts/analyze_tp_no_stop.py`](../scripts/analyze_tp_no_stop.py) analyzes
the same 32,451 raw-ranked calls and, separately, the same 32,451 calibrated-ranked
calls from the prior comparison. These are the frozen, at-most-five calls per
five-minute cycle, with the earlier eligibility screen and rule-derived candidate
count. No probability threshold was added or optimized. They are not every
possible model forecast or independent portfolio trades.

The sample is April 1-September 28, 2026, 50 stocks and 116 sessions. The reference
price is the signal candle's close. LONG TP is reference times 1.002; SHORT TP is
reference times 0.998. A future minute high/low touching that price counts,
including the first subsequent minute. An adverse move never terminates a call.
"Remaining day" means through the 14:59-15:00 candle, not until the exchange close.
Every selected call has complete outcome data; none was dropped or imputed.

### TP Accuracy

| Forecast Ranking | Direction | Calls | TP Within Literal 60 Minutes | TP By 15:00 |
|---|---|---:|---:|---:|
| Raw | LONG | 14,239 | 9,775 / 68.65% | 11,174 / 78.47% |
| Raw | SHORT | 18,212 | 12,544 / 68.88% | 14,093 / 77.38% |
| Raw | Combined | 32,451 | 22,319 / 68.78% | 25,267 / 77.86% |
| Calibrated | LONG | 10,599 | 7,436 / 70.16% | 8,437 / 79.60% |
| Calibrated | SHORT | 21,852 | 14,915 / 68.25% | 16,854 / 77.13% |
| Calibrated | Combined | 32,451 | 22,351 / 68.88% | 25,291 / 77.94% |

There are 1,733 calls after 14:00 in each set, so some literal 60-minute windows
extend past square-off. Raw has 87 hourly successes whose first touch is after
15:00; calibrated has 91. Counting only touches by both 60 minutes and 15:00 gives
22,232 / 68.51% for raw, and 22,260 / 68.60% for calibrated. These are the hourly
rates consistent with the time exit. Among the 30,718 calls with a full hour
before 15:00, the rates are 68.79% and 68.93% respectively.

Holding beyond 60 minutes, but only until 15:00, adds 3,035 valid TP hits for raw
and 3,031 for calibrated compared with the cutoff-capped hourly counts. These are
price-touch precision measurements under the user's definition of accuracy,
not classification accuracy including true negatives or profitable-trade rates.

### Calls That Never Hit TP

Only calls with no TP touch by 15:00 enter this return table. The direction-aware
gross return is `side * (price_at_1500 / signal_close - 1)`, with LONG side +1
and SHORT side -1. The square-off price is the close of the candle ending at
exactly 15:00. Fees, spread, slippage and leverage are not included.

| Forecast Ranking | Direction | Never-Hit Calls | Mean Return | Median Return | Worst Return |
|---|---|---:|---:|---:|---:|
| Raw | LONG | 3,065 | -0.7782% | -0.5979% | -6.3294% |
| Raw | SHORT | 4,119 | -0.7585% | -0.5427% | -5.9495% |
| Raw | Combined | 7,184 | -0.7669% | -0.5685% | -6.3294% |
| Calibrated | LONG | 2,162 | -0.8195% | -0.6370% | -5.8409% |
| Calibrated | SHORT | 4,998 | -0.7468% | -0.5385% | -5.9495% |
| Calibrated | Combined | 7,160 | -0.7687% | -0.5662% | -5.9495% |

Missing TP does not always imply a loss. Of the 7,184 raw failures, 6,767 were
negative at 15:00, 27 flat and 390 positive but below TP. The raw-failure 10th and
90th return percentiles were -1.7152% and -0.0717%; the best failure returned
+0.1828%. The report also retains the equal-call gross TP-or-time-exit average
(-0.01405% raw, -0.01374% calibrated), not a portfolio or net-cost return.

Artifacts are in
[`../data/research/tp_no_stop_1500_20261005`](../data/research/tp_no_stop_1500_20261005):
the frozen protocol, report with monthly/directional breakdowns, and a 64,902-row
call export covering both alternative rankings. Each row includes TP outcomes,
first-touch times, reference and 15:00 prices, and directional return. Treat the
two alternatives separately, not as an additive book. The data directory is
Git-ignored and requires separate retention.

- Protocol SHA256:
  `e102d922fd1ca18d5e590e52b18c41499bdcc0cdab12bcd5d616254e72c4b89f`.
- Call export SHA256:
  `216a82070793223da4f661fcf84faa0338d6edd84a9a69991d89eaea05f20f81`.
- Report SHA256:
  `a38b2d2b6bec94b9fcfa6fc916623d70cc0666869e2523114b6de72d5336634b`.

All literal hourly outcomes were recomputed from candles and matched the frozen
labels. Independent checks reconciled all exported TP/time boundaries, return
signs, aggregates and hashes. Tests cover adverse moves followed by TP, first-bar
touches, exact cutoff inclusion, post-cutoff exclusion, short-return signs,
missing-minute rejection and conditional failure denominators. The full suite
passed 1,080 tests. No model fits, broker calls, live orders or historical artifact
rewrites were performed.

## CatBoost Versus Current Rules

The October 4 follow-up compared the frozen 607-tree CatBoost with the current
five-rule scanner, without any new fitting or changes to production defaults.
The driver is [`../scripts/compare_tp_scanners.py`](../scripts/compare_tp_scanners.py).
This is a retrospective diagnostic on an already-examined period, not a new
untouched test or a basis for automatic live promotion.

### Raw Versus Calibrated

Here, **raw probability** means the CatBoost estimate after the sigmoid converts
the tree ensemble's margin to a number between zero and one. CatBoost's API name
`RawFormulaVal` instead refers to the unbounded margin before that sigmoid; the
two uses of "raw" must not be confused.

**Calibrated probability** applies the saved, separately fitted LONG/SHORT slope
and intercept to that margin before the sigmoid. It adjusts confidence, not the
input features or the trees. For example, a raw 60% forecast becomes about 61.07%
for LONG or 63.09% for SHORT under the saved mapping. This is not inherently
better: the mapping helped March but became overconfident in later months.
Neither probability predicts net profit or accounts for a stop loss.

### Comparison Contract

- Same April 1-September 28, 2026 dates, 50 stocks and 116 represented sessions.
- Same 644,418 paired side-action rows, 09:40-14:15 IST five-minute decisions,
  fixed 0.20% TP-touch label and complete 60-minute future windows.
- Rebuilt the real `FeatureEngine` and `StrategyConfig().scanner()` causally
  from hash-verified candles, with five earlier sessions for warm-up.
- Shared ATR readiness and the current 3-ATR cost/feasibility screen retained
  431,382 eligible side-actions; rules fired on 158,624 of them.
- Matched-ranking comparison uses the number of rule candidates at each time,
  capped at five, for every policy: 32,451 selected actions across 6,496 cycles.
- Four fixed alternatives: current rules, raw CatBoost, calibrated CatBoost,
  and current rule triggers reranked by raw CatBoost. No threshold sweep.

The rule scanner is restricted to the model's eligible cohort and cadence here.
This is not its unrestricted every-minute live configuration. Complete-case
eligibility and previous examination of the period limit generalization.
Rule scores are heuristic rankings, not probabilities, so probability log loss
was not computed by treating those scores as calibrated forecasts.

| Ranking Policy | Matched Actions | TP Touches | TP Touch Rate |
|---|---:|---:|---:|
| Current rules | 32,451 | 19,521 | 60.16% |
| CatBoost raw | 32,451 | 22,319 | 68.78% |
| CatBoost calibrated | 32,451 | 22,351 | 68.88% |
| Rule triggers reranked by raw CatBoost | 32,451 | 21,435 | 66.05% |

Raw CatBoost improves matched TP-touch rate by 8.62 percentage points; a paired
circular five-session block bootstrap gives a descriptive 95% range of
[6.90, 10.34] points. Its touch-rate advantage appears in both directions and
all six months. The calibrated-vs-raw difference is much smaller; calibration
changes the mix of LONG and SHORT selections and is not established as superior.

### Shared Execution Check

Historical scenario only: the user clarified on October 5 that the requested
exit contract has no 2-ATR stop and uses 15:00, not 15:15. The figures below are
not a measurement of that contract; use the no-stop analysis above for it.

Selection quality is separate from executable profit. The second comparison uses
the same deterministic `ReplayPortfolio` sizing, fees, fixed 2-ATR stop, adverse
fills and stop-first resolution of ambiguous candles. The initial clip is the
current INR100,000 whole-share notional; at most five candidates are considered
per cycle and three positions are held, with no overlapping position in one stock.
The entry candle cannot trigger an exit. All policies close remaining positions
at decision plus 60 minutes or 15:15 IST, whichever applies. This added 60-minute
cap and the common cadence are stated benchmark constraints, not default changes.

Actual trade counts differ because positions exit at different times. The
standard scenario uses one-second latency, two-basis-point half-spread and
one-basis-point slippage. The adverse scenario uses ten seconds, four basis
points and three basis points respectively. These are assumptions, not measured
fills. The inherited uniform INR0.05 tick grid is not an exchange-verified
per-symbol/date schedule; no investment return percentage is inferred from it.

| Policy | Standard Trades | TP Exit Rate | Net INR / Trade | Total Net INR | Adverse Net INR / Trade |
|---|---:|---:|---:|---:|---:|
| Current rules | 6,817 | 40.85% | -139.52 | -951,138.15 | -209.90 |
| CatBoost raw | 7,608 | 52.06% | -131.57 | -1,001,019.11 | -200.56 |
| CatBoost calibrated | 7,609 | 53.13% | -127.34 | -968,948.65 | -196.97 |
| Rule triggers reranked by raw CatBoost | 7,596 | 49.46% | -137.46 | -1,044,121.11 | -205.00 |

Every alternative lost money in every month under both scenarios. CatBoost's
better hit rate does not make it profitable: raw CatBoost's average net winner
was about INR121 versus an average net loser of INR407. Trades can touch TP
eventually but stop out first, or miss the executable target after entry friction.
Total losses also depend on turnover, so a lower loss per trade need not mean a
lower aggregate loss. These alternative backtests are not additive strategies.
The fixed-clip simulation does not establish cash/margin sufficiency or implement
the unbuilt production loss limits.

### Decision And Next Work

CatBoost is the stronger TP-touch ranker among these fixed alternatives, and
the rule triggers are not a prerequisite for that advantage. Neither is a
production-grade trading policy under the tested execution contract. Keep the
rules as a control and the frozen raw/calibrated model as research candidates;
the default scanner remains unchanged.

No further variants were trained in this comparison. The evidence points to
objective/execution mismatch, not insufficient compute. A useful next model
study would separately specify a cost-aware abstention or TP-before-stop/net-payoff
target, keep deterministic controls outside the model, and use chronological
walk-forward development plus a genuinely future shadow evaluation. That would
be a new contract, not a silent relabeling of the current no-SL model. CatBoost
remains practical locally; Azure is unnecessary until measured runtime warrants it.
Corporate-action/tick validation, live feature parity, spread/fill observations,
decision journaling and fail-closed checks are still production-readiness gates.
Do not launch larger sweeps on this already-examined evaluation sample and call
the winner independently validated.

Artifacts are retained in
[`../data/research/scanner_comparison_20261004`](../data/research/scanner_comparison_20261004):
the pre-run protocol, aligned opportunity arrays, all scenario trades and report.
These files are Git-ignored and need separate retention.

- Protocol SHA256:
  `ae35ce9103bb9e48e66b5572abcb3b950af44a58bca92006435d7a4a9bc22884`.
- Report SHA256:
  `5c2a44e2ea884022c9e3f231b5d62b04f3f2c9a7bee0b7ce955364846cd911ae`.
- Trade export SHA256:
  `310c455f172bee2bd9474db68d3c6af64676afea26501f44feadc45045436c10`.

All 60,503 alternative-scenario trades reconciled with their reports and satisfied
the shared position, latency and deadline checks. Tests verify exact isolated
trade equivalence with `ReplayEngine` for both directions and fill scenarios,
matched candidate counts, future-candle independence and end-to-end exports.
The full suite passed 1,075 tests. Existing model, calibration, source candles
and runtime scanner were not modified; no training jobs or live orders were run.

## Limitations

Current Nifty membership is not point-in-time membership. Symbol history,
corporate actions and daily source semantics are not independently verified.
Strict complete-case filtering can select a cleaner subset than live conditions;
coverage must accompany metrics. Five-minute decisions have overlapping horizons
and assets share market regimes, so millions of rows are not millions of
independent observations. July-September 2026 has already been researched;
evaluation is retrospective. Session-block uncertainty and prospective shadow
data are needed before claiming dependable probabilities. No live order path,
production scorer or risk-control override is introduced.

</details>