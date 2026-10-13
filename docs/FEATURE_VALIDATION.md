# Feature Accuracy Validation

## Current Conclusion

Documentation consolidated October 9, 2026; latest executed checks are dated
below. Independent arithmetic, availability and replay checks support the built
feature paths as a validated baseline for **scanner-layer research**. Unbuilt
portfolio/execution services are outside this scope, not missing feature tests.
This does not assert that all features were independently checked at every
historical minute or that live and historical source candles are identical.

The generic 63-field rule path and the 13 numerical TP inputs plus side remain
separate contracts. Preserve their readiness/missingness and ordered model
schema in the next research protocol. Correct calculations establish what a
signal measures, not that selecting it produces a useful or profitable scanner.

## Kite Migration Update

The October 5 provider migration adds fail-closed intraday continuity: an absent
minute resets rolling readiness and invalidates session-wide VWAP, volume and
range values for that day. The independent batch reference follows that stated
contract separately. Live/replay tests cover both a continuous native-volume
handoff and a late join with one deliberately discarded minute.

The active cache is hash-verified and refuses ordinary reads across recorded
gaps or quarantines. The TP-specific builder explicitly handles incomplete
ranges and a calendar retaining quarantined dates. All 3,592,876 new Kite TP
labels and outcomes passed independent recomputation; see
[the validation report](../data/kite_migration_20261005/validation.json).
This does not claim all 63 generic features were compared at every one of the
16.8 million usable minutes, or that an exchange daily benchmark certifies them.

The [fresh Kite arithmetic audit](../data/kite_migration_20261005/feature_arithmetic.json)
covers September 1-4, 2026 for RELIANCE, HDFCBANK, INFY, SBIN, NIFTY and GOLDBEES:
8,760 candles and 443,634 numeric comparisons, all passed, plus availability
checks for all 63 derived fields. Six five-/twenty-session fields lack numeric
coverage in the four-day sample; synthetic multi-session tests cover them.
The audit is not a live/history parity test.

The generic engine retains its conservative 375-minute previous-session
readiness; shortened CAS days are not declared complete under that contract.
The TP research contract separately specifies 360-minute cutoff volatility.
The prior audits below used Groww and are historical. Their local datasets and
outputs are retired; do not reuse the old paths as current data.

The new [Kite CatBoost study](TP_ONLY_TRAINING.md) uses the separate 13 numerical
TP features plus side, not all 63 generic fields. Its exact forecast reload and
independent loss checks passed on 686,112 evaluation actions; those checks do
not replace the arithmetic or source-quality audits above. No feature formulas
or data inputs were changed to improve an evaluation score.

## Market-Hours Check: October 6

The new [live capture](../data/kite_live_validation_20261006_retry1/report.json)
produced 64 candles and 64 generic feature rows across eight instruments.
Saved-tick replay was exact. The [independent review](../data/kite_live_validation_20261006_review/report.json)
rebuilt every live OHLC/volume from the tick tape and passed the existing
60-digit arithmetic reference: **268,685 numerical comparisons plus 74,854
unavailable-value agreements over 5,453 warm-up-plus-live bars**. These totals
include warm-up; they are not 268,685 comparisons on only 64 live candles.

All 64 live snapshots were core-ready, not necessarily ready for every
long-history field. Against subsequently fetched Kite history, only 31 of 56
comparable generic feature rows were identical because some underlying candles
differed. This separates arithmetic/replay correctness from feed parity.

All ten model diagnostic records occurred after the trained 14:15 cutoff. The
runner now rejects those records for inference and also vetoes inference after
an unreconciled reconnect. Corrected offline replay retains the same features
but emits no model predictions for this capture. Full regression: **992 tests
passed in 92.69 seconds** at that point. This capture did not establish in-window
inference; the newer result below does so for a bounded sample. No model or
runtime feature formula changed.

## Latest Check: October 7

The [new live capture](../data/kite_live_validation_20261007/report.json) produced
128 complete candles and generic feature rows, with exact saved-tick replay.
The [current independent review](../data/kite_live_validation_20261007_review_v3/report.json)
passed **229,133 numeric comparisons and 66,526 unavailable-value agreements**
across **4,693 warm-up-plus-live bars**. Only 58/120 comparable generic feature
rows match subsequent Kite history; correct arithmetic is not source parity.
Long-history coverage remains bounded as described above.

The current review independently recomputes all 13 TP feature formulas directly from
saved candles and prior daily closes, without `TpFeatureState` or incremental
VWAP helpers. **195/195 values across 15 records** passed at 60-digit reference
precision, with maximum absolute error `1.438327742582764731178543642E-24` under
the declared `1E-18 * max(1, abs(reference))` bound. Missing-minute, prior-day,
volume, stale-context, flat-price, extreme-future and corruption controls are
covered by the live-path tests. No production feature formula was modified.
Review v3 additionally checks that a gap after the lagged-return interval still
invalidates that input until its rolling continuity is restored. Both retained
captures passed with credential loading, browser login and sockets blocked.

Five target records at 12:45 produced ten LONG/SHORT forecasts; ten records at
12:50/12:55 were withheld after reconnect. Their input arithmetic can be checked
without treating them as eligible forecasts. The five issued records matched
historical-input raw/calibrated probabilities exactly, but that does not erase
the measured candle differences. Full suite: **1,005 tests passed in 109.30
seconds**, with Ruff lint/format passed. Automatic source reconciliation,
portfolio execution and prospective model outcomes are not validated here.

## Candle Inputs And Research Use

The later [construction investigation](BROKER_DATA_QUALITY.md#candle-construction-investigation-october-7)
independently reproduced all 192 candles across both saved captures and found
no observed aggregation arithmetic error. Historical extrema missing from the
received prices and adjacent-minute volume offsets explain why correct input
processing need not yield exact source parity. Alternative timestamp shifts
and volume-based filtering made agreement worse; production formulas and candle
construction were not changed. The 74 focused tests passed in 6.16 seconds.

Only 58/120 complete generic-feature rows matched the subsequent-history path
on October 7, but a row fails strict equality if any field differs. This is not
the percentage of correctly calculated indicators: independent arithmetic passed
on the actual inputs. All five issued TP-model records matched historical-input
raw/calibrated probabilities. Neither fact proves source differences will never
alter another model/rule decision. Measure that sensitivity in scanner research
where relevant; do not require universal exact parity before proceeding.

October 9 documentation work adds no new full-suite or live evidence. The
[architecture](ARCHITECTURE.md#research-acceptance-and-scope) records the accepted
scope and [handover.txt](handover.txt) records the scanner-first next session.

## Historical Evidence And Formula Reference

The remaining dated numerical results used retired Groww inputs. Formula
explanations remain useful; old dataset paths and run instructions are not
current Kite artifacts. The current verification scope is stated above.

The feature engine's job is to turn candles into 63 numbers for scanner research.
Tests prove those numbers are *stable* and that unavailable
inputs stay unavailable; they do not prove the numbers are *right*, because a
test written from the same understanding as the code agrees with the code even
when both are wrong.

This document records an independent check of that second question, and the
conventions a reader comparing against TradingView or any charting package will
trip over. It exists so the comparison does not have to be re-derived, and so a
difference against a chart can be classified as expected or investigated
without a live market.

## All-Feature Audit: October 3, 2026

The retained stdlib-only batch reference now covers **all 63 derived fields**,
including availability, rather than only the 17 scanner inputs. It computes
at 60-digit precision independently of the production indicator helpers.

On June 1-August 31, 2026, RELIANCE, HDFCBANK, INFY, SBIN, NIFTY and GOLDBEES
supplied **145,758 regular candles**. All **8,379,404 numeric comparisons** and
unavailable-value comparisons passed, using the larger of `1e-16` absolute
and `1e-20` relative tolerance. Every feature had numeric coverage. The audit
also records per-symbol and per-field counts; it does not claim that all
16 million cached bars or exchange-ground-truth values were checked.

The audit found and fixed a real corner case: the old VWAP dispersion formula
`E[price**2] - E[price]**2` could report a tiny positive variance for one weighted
price. RELIANCE's August 21 opening candle yielded a deviation near `3.16e-11`
instead of zero, which could create a huge, meaningless VWAP z-score. Production
now uses weighted Welford central moments; the VWAP price formula is unchanged.
The independent reference also required centering its two-pass variance to
avoid its own `2e-57` first-price residue. Exact-zero and large-price/small-spread
regressions cover both problems. No tolerance was relaxed to make either pass.

The tests additionally cover a complete 21-session tape with populated
twenty-session features, cold starts, late starts, UTC/IST equivalence, missing
minutes, unknown/zero volume, flat prices, and rejection of corrupted output.
Full offline suite after the repair: **1,024 passed**. This fixes arithmetic;
it does not establish a profitable signal or validate broker fill assumptions.
Historical frozen research reports and their hashes were not rewritten.

Reproduce a fresh offline report using a new output path:

```bash
python -m scripts.validate_scanner_features --audit-all \
  --symbols RELIANCE HDFCBANK INFY SBIN NIFTY GOLDBEES \
  --start 2026-06-01 --end 2026-08-31 \
  --output data/research/feature_audit_20261003/arithmetic_final.json
```

`--audit-all` refuses external-fetch options. The existing Yahoo comparison
still defaults to the original 17-input surface for compatibility.

## Historical 47-Feature Check

The original 47 derived features were recomputed by a separate float64 implementation
written from textbook definitions rather than from the engine's source, and
compared element-wise against the engine's Decimal output over 362 real
one-minute RELIANCE candles from the 2026-09-21 NSE session (09:15 to 15:29
IST). Only rows where both sides produced a value were compared, so a feature
withheld for want of history was never scored against a reference that had
invented one.

**All 47 agree.** Typical agreement is 1e-13 relative or better, which is
float64 noise rather than a difference in method. The three features that
first appeared to disagree were each traced to a fault in the reference, not
in the engine:

| Apparent mismatch | Cause | Resolution |
| --- | --- | --- |
| `bollinger_percent_b_20` | Reference lost precision to float64 cancellation. At the worst row the absolute deviation is 2.1e-10 against a %B of 0.01 — the relative figure is large only because the denominator is small. | Engine is the more accurate of the two. Not a defect. |
| `volume_ratio_20` | Reference used TradingView's convention, which includes the current bar in its own baseline. | Matches to 1.3e-15 once the lagged baseline is used. Deliberate divergence — see below. |
| `session_range_pct` | Reference divided the session range by the session open. | Engine divides by the session low, as its docstring states. Matches to 2.1e-14. |

Worst relative deviations, once the reference was corrected, by family:

```
ema9/21/50            5.5e-16 .. 2.6e-15      sma20                 1.8e-16
rsi14                 1.4e-13                 bollinger_upper/lower 1.8e-13
macd/signal/histogram 3.0e-10 .. 5.1e-11      bollinger_bandwidth   5.5e-10
true_range / atr14    1.4e-12 / 1.6e-13       vwap / deviation      7.3e-16 / 2.1e-10
plus_di / minus_di    2.8e-13 / 3.8e-13       rolling_high/low_20   exactly 0
adx14                 2.4e-13                 obv                   exactly 0
return_1/5/15         ~3.7e-12                session aggregates    exactly 0
```

## Added Features: October 3, 2026

Sixteen additions share the production/replay `FeatureEngine` path. Their
formulas and readiness are covered by retained tests in
[`../tests/test_features.py`](../tests/test_features.py); they were **not** part
of the earlier 47-feature external comparison above.

All returns below are fractions, not percentage points. Let `C` be the current
completed close and `O/H/L/K` a prior session's open/high/low/final close.

| Fields | Definition | Required history |
| --- | --- | --- |
| `intraday_return_30m`, `intraday_return_60m` | `C / close_h_minutes_ago - 1` | 31/61 consecutive same-session one-minute bars |
| `realized_volatility_20m`, `realized_volatility_60m` | `sqrt(sum(log(C_i/C_(i-1))**2))` over 20/60 returns | 21/61 consecutive same-session bars; unannualized, not standard deviation |
| `previous_session_return` | `K/O - 1` | Immediately previous observed session, all 375 regular minutes |
| `previous_session_range_pct` | `(H-L)/O` | Same complete prior session; unlike the existing current-session range, denominator is open |
| `gap_from_previous_close` | `current_session_open/K - 1` | Complete prior session plus today's actual 09:15 bar |
| `distance_from_previous_close/high/low` | `C/K - 1`, `C/H - 1`, `C/L - 1` | Complete prior session |
| `prior_5_session_return`, `prior_20_session_return` | Last prior close / first prior open - 1 | All of the previous 5/20 observed sessions complete |
| `prior_5_session_range_pct`, `prior_20_session_range_pct` | `(maximum high - minimum low) / first prior open` | Same complete prior 5/20 sessions |
| `position_in_prior_5_session_range` | `(C - minimum low) / (maximum high - minimum low)` | Prior five complete sessions; may fall outside 0..1 |
| `volume_ratio_same_minute_5` | Current minute volume / mean volume at that minute in the previous five observed sessions | All five exact-minute volumes known and baseline positive |

The minute-window features reset on gaps or date changes; the older
`return_1/5/15` retain their documented candle-count/overnight semantics. A prior
session is promoted only when a later session starts. Incomplete sessions are
retained as unavailable entries rather than skipped in favor of older days.
Completely unobserved dates cannot be distinguished from holidays without a
calendar; these are explicitly **observed-session**, not calendar-week, windows.

The same-minute volume baseline excludes today and remains useful even if an
unrelated minute was missing on a prior day. A missing required minute or
unknown volume gives `None`; observed zero volume remains zero, but a zero
baseline is unavailable. Index volumes are not synthesized. Zero price-range
denominators likewise give `None`, while flat-price realized volatility is zero.

Validation covers independent log-return formulas, hand-computed session/range
and volume examples, twenty-session window eviction, cold starts, incomplete
prior sessions, a missing current open, per-instrument isolation, bounded
memory, and warm-up/live equivalence. `core_ready` is unchanged; consumers must
check each new field's readiness. Corporate-action adjustment and cross-provider
agreement are not established by these tests.

## Comparing against TradingView

The conventions already match the common charting default in every case where
one exists, so a chart comparison is meaningful rather than apples-to-oranges:

- **RSI, ATR, DI and ADX use Wilder smoothing**, not a simple moving average.
  This is what TradingView's built-in `ta.rsi`, `ta.atr` and `ta.adx` use.
- **EMAs are seeded with the SMA of the first `period` closes.** Charting
  packages that seed from the first close instead will differ for the first few
  dozen bars and converge after that; the difference is a seeding choice, not
  an error in either.
- **Bollinger Bands use population standard deviation** (ddof=0) at 2σ around
  a 20-period SMA, which is how the indicator is defined and what charting
  packages compute.
- **VWAP uses the typical price** `(high + low + close) / 3`, anchored to the
  session, matching TradingView's session VWAP.
- **MACD is 12/26/9** on closes.

Three differences are expected and are **not** defects:

1. **`volume_ratio_20` excludes the current bar from its own baseline.** It
   compares this candle's volume against the mean of the *previous* twenty,
   where TradingView's usual `volume / sma(volume, 20)` includes the current
   bar in the denominator. On a volume spike the two diverge sharply — the
   lagged form reports a larger ratio, because the spike has not been allowed
   to dilute its own baseline. The reasoning is in `_volume_ratio`'s docstring
   in `features/engine.py`. Neither is time-of-day adjusted, so neither is a
   true RVOL.

2. **Session aggregates are withheld unless the session was observed from its
   opening range.** An engine first attached at 11:00 has not seen the day's
   high, so it reports `None` for the whole session rather than the highest
   price it happens to have seen. A chart always shows a value. VWAP is the
   documented exception and is computed from whatever the engine has seen.

3. **Volume-derived features go unavailable rather than approximate.** A single
   unknown volume in a window withholds the whole value, and one unknown volume
   disables strict session VWAP for the rest of the day. A chart backfills from
   the exchange and will show a number where this engine shows `None`. That
   asymmetry is the point — see ARCHITECTURE section 8.

### A row to spot-check by hand

Final candle of the 2026-09-21 session, RELIANCE NSE, one-minute, timestamps in
UTC (add 5:30 for IST):

```
start 2026-09-21T09:58:00Z   end 2026-09-21T09:59:00Z
OHLC  1247.4 / 1247.4 / 1247.4 / 1247.4      volume 531468

ema9    1246.873350   rsi14   55.348082   sma20   1247.375000
ema21   1247.001863   macd    -0.028439   bb_up   1248.787622
ema50   1246.494934   atr14    0.576205   bb_low  1245.962378
adx14     21.667573   vwap  1240.955698   obv     1120524
session_high 1248.9   session_low 1232.5   session_volume 9885947
```

`volume_ratio_20` on this candle is 11.99 — the closing-minute spike, and a
good row on which to observe divergence 1 above, since TradingView's
convention will report a visibly smaller figure.

## Reproducing

```bash
python -m ai_trader.cli.check_features --export-csv features.csv
```

That writes one row per candle at full precision, which is the input the
comparison consumed. The reference implementation itself was a throwaway
script outside the repository and was not kept.

## Retained Reference Tests

The reference used numpy and pandas. Both are present in `.venv`, but only as
transitive dependencies of `growwapi` — `constraints.txt` records that AGENTS.md
rules them out as things this codebase may use, and nothing under `src/` imports
either. Adding a test that imports them would quietly convert a broker SDK's
dependency into one of ours.

The original 47-feature reference was not retained. The September 30 stdlib
reference initially covered 17 scanner inputs; the October 3 extension now
independently covers every one of the 63 fields in
[`../scripts/validate_scanner_features.py`](../scripts/validate_scanner_features.py).
[`../tests/test_source_independence.py`](../tests/test_source_independence.py)
exercises both the full reference and the original compatibility path.

TA-Lib and pandas-ta were considered as third-party references and are not
installable here: `files.pythonhosted.org` is unreachable from this machine
(`SSLV3_ALERT_HANDSHAKE_FAILURE`) even though the index itself resolves.

## What this does and does not establish

It establishes that the arithmetic is right: every feature matches an
independently written implementation of its textbook definition over a real
session, and the Decimal path is at least as accurate as float64 everywhere,
better where cancellation bites.

It does not establish that the feature *set* is the right one to trade on, that
the parameters (9/21/50, 14, 12/26/9, 20) suit intraday NSE equities, or that
the engine behaves correctly on a session it has not seen — a halt, a corporate
action, an illiquid symbol with minute gaps. Those are questions for replay and
forward validation, not numerical precision alone.

## Timestamp comparison, 2026-09-30

An online comparison of Yahoo and Groww one-minute RELIANCE candles over
September 23-24 is recorded in
[`../backtests/reliance_sep2026/feature_parity.json`](../backtests/reliance_sep2026/feature_parity.json).
All 23,891 numeric comparisons against the independent formulas passed. Exact
cross-feed feature equality did not: the providers supplied different OHLCV at
many of the 722 common timestamps. In particular, close matched exactly in 562
cases and volume in 66; a one-minute offset did not explain the differences.
This establishes formula consistency on both feeds, not exchange-ground-truth
accuracy or identity with TradingView. The research README contains exact-time
examples and the offline reproduction command.

## Dataset and Information Review

[`../scripts/analyze_feature_dataset.py`](../scripts/analyze_feature_dataset.py)
is a bounded offline diagnostic, not a trainer or a production dataset builder.
It uses four equity targets (RELIANCE, HDFCBANK, INFY, SBIN), five context series
(NIFTY, BANKNIFTY, NIFTYIT, INDIAVIX, GOLDBEES), June warm-up and July/August
observations. Exact-time context joins refuse future snapshots and do not carry
stale values forward. RELIANCE's missing energy-sector input is explicitly null.

The local report and six real sample rows are under
`data/research/feature_audit_20261003/dataset_diagnostic/`. The sample contains
the first six eligible 10:30 rows in chronological/symbol order, not selected
winners. One row is a decision timestamp, stock, side and barrier specification.
The 24 numeric feature candidates cover momentum, market-relative movement,
trend strength, range/VWAP location, volatility, same-minute volume, candle
shape, prior-session context, clock, sector movement, India VIX and gold.
An additional missingness flag accompanies each candidate. Side and TP fraction
are inputs; date/symbol are audit keys in this prototype, and all `label_*`
columns are future training targets, never model inputs. The example has 50
potential numeric inputs after training-only imputation/scaling: 24 values,
24 masks, side and target fraction. This is a candidate schema, not a proven
optimal feature set or a model that has been trained.

Labels use TP=SL=0.20% of the next candle's open, skip the entry candle, resolve
same-bar double touches to SL, require the entire 60-minute future window and
finish by 15:15. They are **unrounded diagnostic price barriers**, without
spread, fees or actual fill/tick validation. Recorded event time is the candle
end, not an observed intrabar touch. There are nine mutually exclusive classes:
TP in 0-6/6-15/15-30/30-60 minutes, SL in the same bins, or neither by 60 minutes.
The six-row export also includes binary TP-by-horizon labels for inspection.
For example, INFY LONG at July 1 10:30 reached diagnostic TP at 35 minutes:
`label_tp_30m=0`, `label_tp_60m=1`.

A suitable neural output is a calibrated nine-class probability vector per
direction and target fraction. Summing the first TP bins returns
`P(TP before SL within h | X, direction, target)`, and preserves nondecreasing
TP probability for longer horizons. Train/evaluate both directions; do not
present `2*p-1`, AUC, or an arbitrary confidence score as a probability.
TP/SL distances and horizon selection remain development-set choices, with
chronological all-symbol validation and separate calibration. The prototype
does not establish historical tick schedules, corporate-action treatment or
production fill labels; those are required before production training.

### What the Data Shows

There were 10,560 decision times, paired into **21,120 side-actions over 44
sessions**. All selected diagnostic horizons were complete. July supplies the
ATR-fraction quartile boundaries; the same boundaries are used in August.
These are overlapping, correlated observations, not independent trades.

| Diagnostic TP within 30 minutes | July | August |
| --- | ---: | ---: |
| All actions | 34.62% (11,040 actions) | 29.16% (10,080 actions) |
| Highest July-defined volatility quartile | 46.45% (2,760) | 44.85% (1,126) |
| Lowest July-defined volatility quartile | 23.26% (2,760) | 20.10% (4,438) |
| First session hour | 44.61% (2,208) | 37.55% (2,016) |
| Later observations | 32.12% (8,832) | 27.06% (8,064) |
| Same-minute volume at least 1.5 times baseline | 37.73% (2,404) | 33.56% (2,166) |
| Lower relative volume | 33.75% (8,636) | 27.95% (7,914) |

The main evidence is time-to-barrier reachability, not dependable directional
edge. Market-relative momentum-aligned directions hit 33.64%/28.99% versus
35.60%/29.33% for opposed directions. This is weak and not a recommendation to
reverse momentum. Volatility, time of day, symbol and volume are confounded;
no independent causal effect or statistically significant trading edge is
claimed. Overall probability changed across months. A neural network may learn
these conditional frequencies and interactions, but accurate future
calibration must still beat frozen base-rate and time/volatility controls on
later periods, using Brier/log loss, per-side reliability and day-block
uncertainty. No NN was fitted, no accuracy forecast is justified, and August
here is a descriptive later period rather than pristine forward evidence.

### A Smarter Feature Collection

- Raw price levels are heavily redundant even within individual stocks:
  July EMA21 versus SMA20 correlation exceeded 0.9995 in all four stocks;
  EMA9 versus EMA21 exceeded 0.9982. Use normalized EMA spread/slope rather
  than feeding all EMA levels, SMA and close as independent evidence.
- Some duplication is algebraic: MACD histogram is MACD minus signal; band
  position/width plus mean reconstruct the bands; ATR fraction is ATR/close.
  Retain purposeful representatives (position plus scale), not every rendering.
- The 24-column candidate set still has a correlated pair: VWAP z-score and
  current session-range position correlated 0.909-0.942 across these stocks.
  Compare versions dropping either on development folds; high correlation alone
  does not prove the residual contains no predictive information. Avoid using
  evaluation outcomes for correlation filtering or feature selection.
- Prior-session and prior-5/20-session price inputs were unavailable on 45.45%
  of the sampled rows under the strict 375-minute completeness contract.
  This is a real coverage limitation, not a calculation failure. Retain masks
  and training-only imputation; do not pretend missing observations are zero.
  A verified point-in-time daily OHLC source or a separately named covered
  pre-cutoff session summary could improve availability without inventing data.
- Sixty-minute intraday features were unavailable for the first hour (20% of
  these decision rows), as intended. The older 1/5/15 returns count observed
  candles and can bridge gaps/overnight; use explicit strict-minute returns or
  a gap flag in a future production schema where that distinction matters.
- Context features add information beyond rearranging the same OHLCV: index
  and sector-relative movement, volatility regime, same-time volume and earlier
  session state. Broader market breadth, earnings/news timing and genuine
  bid/ask spread/depth or signed order flow are not all present. OHLCV cannot
  reconstruct true order-book imbalance. No feature collection currently covers
  every determinant of executable TP-first outcomes.
- A small tabular network with missingness and regularization is a sensible
  first test; short return/volume sequences are a later challenger, not proof
  that more features or deeper models will improve probabilities. Keep model
  selection and cost-aware trade selection separate.

Reproduce the diagnostic into a new directory:

```bash
python -m scripts.analyze_feature_dataset \
  --output data/research/feature_audit_20261003/dataset_diagnostic_NEW
```
