# Kite Migration And Historical Data Quality

## F&O Cache Versus NSE Official Prices: October 11

Scope: the promoted F&O cache `data/candles_fno_20261011` (217 equities plus
indices, 2020-01-01 to 2026-09-30) against 1,676 NSE `sec_bhavdata_full`
bhavcopies, NSE's symbol-change list and 16,066 NSE corporate actions
(`data/research/data_quality_fno_20261011`). Independent audit:
`scripts/validate_candle_quality.py`; repair: `scripts/build_price_basis.py`.

Raw cache: 689 files hash-verified, 0 schema issues, 0 minutes outside session,
0.19% minutes missing, 96.2% of instrument-sessions complete. Volume equals NSE
exactly on every unadjusted date. Kite's adjusted history, however, is not fit
for percentage-based research as delivered:

- Dividend back-adjustment is sometimes additive (`kite = nse - D`) and sometimes
  multiplicative in yearly steps (PETRONET 0.9218 -> 0.9552 -> 0.9749 -> 1.0).
  Older intraday percentage moves are inflated; in 2020 the 90th/99th percentile
  inflation was 3.7%/25% (IOC 36%, VEDL 25%).
- Adjusted prices are re-rounded at every adjustment step, sometimes to Rs 1
  (GODFRYPHLP) or Rs 0.10 (HDFCAMC). 12,330 stock-days have a vendor-induced
  price step above 10 bp (MCX, GODFRYPHLP, NMDC, BSE, SUZLON, ADANIPOWER...).
- Some volume histories are not split-adjusted (BPCL, CANBK, NMDC, HDFCAMC, IOC,
  EICHERMOT).
- Several demerger adjustments are wrong or missing: Kite ratio vs NSE ex-date
  open/prior close: ITC 2025-01-06 0.865 vs 0.946, SIEMENS 0.762 vs 0.596,
  TMPV 0.689 vs 0.610, VEDL 0.523 vs 0.358; HINDUNILVR 2025-12-05 unadjusted.
- Kite's NIFTY daily series misses five sessions (2020-03-13, 2020-03-23,
  2022-03-07, 2024-03-02, 2025-09-18); the session calendar is the union of all
  instruments (1,677 sessions).

NSE reference repairs: the 2022-08-08 bhavcopy arrived as XLSX and was converted
(original kept); NSE serves the 2021-11-03 file under the 2021-11-04 Muhurat name,
so that session has no anchor; RUCHI -> PATANJALI (2022-07-13) is missing from
NSE's symbol-change list and is supplied in code.

Corrected dataset `data/candles_fno_basis_20261011` (provider kite, derived):
each day's Kite O/H/L/C is fitted to NSE's as `kite = a * nse + c` (tolerance
2.5 vendor quanta + 1 paisa + 2 bp), minutes are mapped back to NSE's raw price
basis, then multiplied by later structural ratios: 75 split/bonus ratios are
Kite's own regime change snapped to the exact ratio, 17 rights/demergers and 6
Kite-unadjusted actions use NSE's ex-date open over prior close (flagged).
Dividends are not adjusted. Results: 342,996 of 343,234 equity sessions kept
(dropped: 197 Muhurat-2021 sessions, 38 other days without an NSE EQ/BE row,
3 misfits); 228,463 sessions needed no change (Kite already equals NSE). The
independent re-audit finds daily
close and volume exactly equal to NSE on all 252,226 unscaled dates, zero
unexplained factor changes and the latest factor 1 everywhere. Minute extremes
exceed NSE's daily range by a median 1.2 bp (p99 20 bp, 3,985 days), the limit
of Kite's rounding.

Use policy: train and test only on the corrected dataset, excluding
`exclusions.json` flags (coarse vendor rounding, imprecise fit, structural
breaks; 12,644 stock-days, mostly 2020-2023) and the audit's illiquid sessions;
loaders already skip incomplete sessions. Known limits: survivorship (today's
F&O list), illiquid early years for recent entrants, minute timing not
tick-aligned, and reconstruction precision bounded by Kite's rounding.

## Current Research Assessment

Documentation consolidated October 9, 2026 after the October 7 checks and
candle-construction investigation. **The built historical and real-time flows
behave as expected under the tested conditions and are suitable for continued
scanner research.** The historical cache is good quality for the explicit
gap-aware research contract; the live feed was healthy in the recorded windows.
Independent arithmetic and exact replay support the correctness of processing
those inputs, not perfect market coverage or a profitable trading policy.

The user scoped validation to built components. Unbuilt risk/position/journal/
execution services are future work, not failed checks or blockers to scanner
research. Exact candle equivalence, full-session/all-universe coverage and
automatic recovery were not established and remain visible limitations. The
next work is scanner quality and policy research, not another general validation
campaign. No new market run, model fitting or data refresh occurred October 9.

Saved reports keep their original exact-parity and whole-product statuses,
including `not_live_trade_ready`. The later scoped acceptance does not rewrite
those criteria as passes or authorize orders. See
[the canonical acceptance scope](ARCHITECTURE.md#research-acceptance-and-scope).

## Latest Technical Validation: October 7

**Historical research reproducibility and bounded live inference passed.**
This is the latest executed validation; whole-product live trading is outside
the accepted built-layer scope described above.
The migration and October 6 sections below retain their dated observations,
not the current in-window prediction count. No orders, new research fitting,
model promotion, cloud operations or changes to the immutable cache occurred.
The [consolidated readiness receipt](../data/kite_validation_20261007/technical_readiness.json)
pins the reports and distinguishes passed checks from remaining requirements.

### Implemented-Layer Results

| Layer / Check | October 7 Result | Limit |
|---|---|---|
| Historical Kite cache | All 66 instruments reverified; all 3,592,876 labels/outcomes independently matched; all 108 retained NSE daily comparisons passed | Existing adjusted-price/calendar/reference-data caveats remain |
| Frozen CatBoost | All eight validation candidates, seed repeat, locks and 686,112 raw/calibrated evaluation forecasts reverified | No refit, new holdout or profitability claim |
| Read-only live broker/market | 13,819 moving ticks across eight instruments; native equity/ETF volume; 128 complete candles/features | One bounded session slice, not all market regimes |
| Generic feature arithmetic | 229,133 numeric plus 66,526 unavailable comparisons over 4,693 warm-up/live bars passed | Core readiness is not full long-history readiness |
| TP model-input arithmetic | All 195 values across 15 records independently checked at 60-digit precision | Formula agreement does not establish vendor-source parity |
| In-window inference | Five target records at 12:45, ten LONG/SHORT forecasts, complete inputs and exact replay | One decision time; no TP outcomes or positions evaluated |
| Reconnect guard | One controlled reconnect; ten later records withheld | Veto passed; automatic history/feature reconciliation is not built |
| Rules/feasibility | Saved-tape diagnostic: 16 cycles, 80 target snapshots, 41 repeatable candidates, network blocked | Legacy rule policy, empty portfolio; not TP entry selection or trade frequency |
| Full code gates | 1,005 tests passed in 109.30 seconds; Ruff lint and 122-file format check passed | Does not certify absent execution/portfolio services |

The actual tick window was **12:39:57-12:56:17 IST**, with history/tick cutover
at 12:40. The 900-second supervised capture recorded 12,808 ticks after 1,011
startup ticks. All eight names moved: RELIANCE, HDFCBANK, ICICIBANK, INFY, SBIN,
NIFTY, GOLDBEES and INDIAVIX. There were no timestamp/volume regressions, late
ticks, duplicate candles or pending unaligned batches. The initial partial
minute was discarded and the trailing partial minute was not flushed. Thirty-two
index candles retain unknown volume, not manufactured zero volume.

The reconnect probe now waits for an eligible prediction and at least three
collection windows. The earlier fixed three-window timing could interrupt
before the next five-minute model decision. This change was tested before the
capture. The probe disconnected at 12:45:17 and reopened at 12:45:19; the 12:45
predictions had already been recorded. All ten 12:50/12:55 records were vetoed
for unreconciled continuity, despite complete numerical inputs. This validates
fail-closed behavior, not outage repair or readiness to resume automatically.
Normal operation keeps one healthy WebSocket open across 60-second collection
windows; it does not periodically unsubscribe/reconnect. The run had 15
collection windows and two opens only because of this deliberate test. There
were zero unexpected disconnects and zero silent windows. Retryable errors or
silent collection can require reconnect; doing so does not recover missing data.

### Source Parity Remains Open

All **120 settled comparison timestamps** were present, but only **74 OHLCV**,
**91 OHLC** and **58 generic-feature rows** matched subsequent Kite history.
The last eight live candles were excluded by the settling rule.

These are two candles for the **same stock, date and minute**: our OHLCV formed
from received WebSocket updates, versus the historical API response fetched
after that minute. "Historical" here does not mean an older training day.
Strict equality fails on any one differing field; 74/120 is not a bad-tick or
missing-candle percentage. The other 46 rows are present, not discarded as corrupt.

| Comparison Scope, October 7 | Exact / Compared |
|---|---:|
| Stock/ETF full OHLCV | 74/90 |
| Stock/ETF OHLC only | 76/90 |
| Stock/ETF close | 88/90 |
| Stock/ETF volume | 86/90 |
| Index volume | 0/30, expected unknown live versus zero historical |

The largest stock-price field difference in this sample was ICICIBANK open
1,350.60 versus 1,350.30, INR 0.30 or about 0.0222%. NIFTY's largest difference,
1.15 points, is about 0.00508%. The magnitude supports close agreement rather
than a poor-feed conclusion, but it can matter near a 0.20% target or a scanner
threshold. Evaluate the effect rather than declaring every small difference
harmless or insisting every field be bit-for-bit equal.

For the concrete ICICIBANK minute **12:46-12:47 IST**:

| Field | Our Quote-Derived Candle | Subsequent Kite Historical Candle |
|---|---:|---:|
| Open | 1,350.60 | 1,350.30 |
| High | 1,350.60 | 1,350.60 |
| Low | 1,350.00 | 1,350.00 |
| Close | 1,350.50 | 1,350.30 |
| Volume, shares | 12,734 | 12,878 |

The full instrument-level comparison follows. None of these differences was
silently rounded away or replaced with a more favorable source.

| Instrument | Exact OHLCV / 15 | Exact OHLC / 15 | Largest Recorded Differences |
|---|---:|---:|---|
| RELIANCE | 13 | 14 | Close INR 0.20; volume 11 |
| HDFCBANK | 10 | 10 | High INR 0.15; low INR 0.10; open INR 0.05 |
| ICICIBANK | 10 | 11 | Open INR 0.30; close INR 0.20; volume 144 |
| INFY | 11 | 11 | Open/low INR 0.15 |
| SBIN | 15 | 15 | None |
| GOLDBEES | 15 | 15 | None |
| NIFTY | 0 | 0 | Close 1.15; high 0.55; low 0.80; open 0.70 points |
| INDIAVIX | 0 | 15 | Live volume unknown versus historical zero |

Expected index-volume representation differences do not explain the real OHLC
and equity-volume differences. No post-hoc tolerance or source substitution was
adopted. All five issued model records matched corresponding historical-input
raw/calibrated probabilities exactly; that small result does not cancel the
broader discrepancy. Ten later records were withheld, so their absent outputs
must not be counted as numerical prediction mismatches.

Review v2 adds a batch TP reference independent of `TpFeatureState` and its
incremental VWAP. It checks returns, volatility, prior-session inputs, VWAP
dispersion and aligned context directly from saved candles. Maximum absolute
error was `1.438327742582764731178543642E-24`, within the declared
`1E-18 * max(1, abs(reference))` tolerance. Synthetic missing-minute, prior-day,
volume, stale-context, flat-price, extreme-future and corrupted-input controls
passed. The first October 7 review is retained; v2 corrects its static missing-
in-window gate and distinguishes withheld outputs from numeric disagreement.
Current v3 also checks intervening continuity for lagged returns. Both retained
captures were reviewed with credential loading, private authentication, browser
launch and socket connections blocked; none of those operations was called.
The October 6 compatibility review retained 64 candles, zero eligible predictions
and 130 independently checked TP inputs, without altering the original evidence.

### Evidence And Remaining Work

Retain the [capture](../data/kite_live_validation_20261007/report.json),
[current review v3](../data/kite_live_validation_20261007_review_v3/report.json),
[historical-data recheck](../data/kite_validation_20261007/validation.json),
[model recheck](../data/research/tp_catboost_kite_20261005/verification_20261007.json)
and [rule-scanner check](../data/kite_validation_20261007/rule_scanner_check.json).
The live/review directories contain their distinct executed source snapshots;
the independent-review additions were made after the live capture. Original
October 6 evidence and the frozen model, calibration and training inputs remain
unchanged. Fresh reports use new paths; the historical recheck even reproduced
the original validation report's hash exactly.

```text
October 7 capture SHA256 3f10f2e101951258720937bf3fc1064f4afc1b52e0b94ffb2a602f5d9e89472d
October 7 tape SHA256    0a84da9f19bc26cba5b26cf307b2bdcfc080ecdaaf8411bdb85004559f094e85
October 7 review v3      ab2b6f06701acf84e77c390fbb8b92c8469a76dff7d3a0bde8cbd4cdbe5ad4ef
```

Next: use the built baseline for scanner research, with explicit source and
missingness assumptions and separate gross/after-cost conclusions. The later
construction analysis below narrows the explanation for source differences.
Optional richer timestamp capture and future automatic state repair are targeted
follow-ups, not prerequisites for resuming research. Any future post-gap inference
resumption still requires reconciliation, and prospective/shadow/execution work
needs its own scope. No live order adapter exists. The normal TP-or-15:00 contract
still excludes a price stop-loss and a forced 60-minute exit.

## Candle-Construction Investigation: October 7

Scope: the **built** tick-normalization, candle and volume path. Unbuilt trading
components are not failed checks in this investigation. The
[construction audit](../data/kite_validation_20261007/candle_construction_analysis.json)
records two-capture results and diagnostic alternatives. No production code,
model, original evidence or candle-source policy was changed; no broker call
was needed. The 74 focused candle/volume/live-path tests passed in 6.16 seconds.

**No aggregation arithmetic error was found in the observed data.** Independent
Decimal grouping reproduced all 192 saved candles: 128 from October 7 and 64
from October 6. It checked exchange-minute flooring, timestamp/delivery-order
open and close, min/max over all received prices and differences between the
last cumulative readings of consecutive minutes. No observed timestamp or
volume regression, late-tick discard, duplicated candle or imputation accounts
for the measured mismatches. This verifies those operations on the saved tape;
it does not prove that exchange timestamp is the ideal trade-time definition.

Three observations distinguish the possible causes:

1. **Unobserved extrema:** all 22 differing high/low values on October 7 and all
  nine on October 6 were absent from the received prices in their minute.
  For example, INFY's October 7 12:49 historical low is INR 994.05, while the
  lowest received price is INR 994.20. INR 994.05 never appears anywhere in
  that symbol's captured tape. A different min/max implementation cannot
  recover a price that was not recorded. The historical service is a comparison
  source, not independently verified exchange truth for that individual trade.
2. **Boundary attribution:** October 7 stock-volume differences are exactly
  -144/+144 shares for adjacent ICICIBANK minutes and -11/+11 for RELIANCE.
  October 6 has -402/+402 for ICICIBANK and -11/+11 for INFY. Each pair preserves
  combined volume. ICICIBANK's 12:46 candle uses 5,710,018 minus 5,697,284 =
  12,734 shares; history has 12,878. The next packet, stamped 12:47:00, has
  cumulative 5,710,162, supplying exactly the 144-share difference, and price
  1,350.30, matching the preceding historical close. This is evidence of
  timing/attribution differences, not a subtraction error or proof of which
  exchange events the historical aggregator used.
3. **Not solely reconnect loss:** before the deliberately interrupted minute,
  October 7 already has eight OHLC-different candles out of 40, and October 6
  has five out of 24. The controlled interruption cannot explain all differences.

Whole-cohort alternatives were tested diagnostically, counting regressions as
well as improvements. The following includes the five stocks and GOLDBEES,
excluding the expected unknown-versus-zero index-volume difference:

| Construction Variant | Exact OHLCV, October 7 / 90 | October 6 / 42 |
|---|---:|---:|
| Current exchange-minute boundaries | 74 | 35 |
| Put timestamp-at-boundary updates into the previous minute | 12 | 4 |
| Shift all timestamps one second earlier | 12 | 4 |
| Shift all timestamps one second later | 11 | 2 |
| Use prices only when cumulative volume increases | 28 | 20 |

The two earlier-boundary variants coincide because recorded exchange times have
whole-second resolution. They are not independent confirmations. Uniform shifts
or dropping unchanged-volume quotes make agreement materially worse; no such
change was selected. Post-hoc alignment of a few discrepant bars would not be a
valid repair. In particular, unchanged cumulative volume is not sufficient to
discard a price update when price and volume snapshots need not advance together.

**One normalization question remains identifiable but untested:** Kite full-mode
[quote packets](https://kite.trade/docs/connect/v3/websocket/#quote-packet-structure)
have separate last-traded and exchange timestamps. The current adapter stores
`exchange_timestamp` in `MarketTick.timestamp` and does not retain last-trade
time. A quote received in a new minute can describe an earlier last trade;
the saved normalized tape cannot prove which discrepancies arise that way.
The existing `collected_at` is queue-consumption time, not network receive time.
It would be unjustified either to blame the vendor for every difference or to
switch clocks and promise historical equality from this evidence alone.

The next discriminating diagnostic, if requested, is a bounded capture of both
timestamps, last-traded quantity, cumulative volume and callback receive time,
with current aggregation kept as a control. Even last-trade time cannot recover
every trade inside a cumulative-volume jump. **Retain the existing algorithm
for now:** it faithfully aggregates the received observations, while exact
historical trade-candle reconstruction requires more information than this tape
contains. This is a narrow input-semantics limitation, not evidence of a generally
bad live feed or an arithmetic failure in the built candle layer.

## Current Full-Universe Migration

The October 5, 2026 migration replaces active Groww access and data with Kite.
The separate RELIANCE comparison below is retained as dated evidence, not the
current system's provider or preservation policy. The user explicitly authorized
retirement of old local Groww data and derived artifacts after validation.

| Measure | Validated Replacement |
|---|---:|
| Requested dates | January 1, 2024-September 30, 2026 |
| Instruments | 50 equity targets plus 16 context instruments |
| Usable minute candles | 16,787,986 |
| Usable daily candles | 45,052 |
| Missing minute slots on non-quarantined observed dates | 494 |
| Quarantined instrument-days | 26 across 15 instruments |
| Additional entire days absent from captured calendar | 0 |
| TP side-action rows, through September 28 | 3,592,876 |
| Independently recomputed labels and outcomes | 3,592,876, all matched |
| Recent NSE daily comparisons | 108 across 54 stocks/ETFs, all passed |

### Completeness Recheck

The October 5 follow-up reverified every active cache file against the promoted
manifest and reconciled minute slots using the existing date/session rules.
The **26 quarantines are instrument-days, not 26 individual candles**.

| Scope | Expected Minute Slots | Missing | Quarantined Slots | Total Excluded | Excluded Percentage |
|---|---:|---:|---:|---:|---:|
| All 66 instruments | 16,795,530 | 494 | 7,050 | 7,544 | 0.044917% |
| 50 equity targets | 12,716,250 | 379 | 2,625 | 3,004 | 0.023623% |
| Targets plus the three model context instruments | 13,481,115 | 401 | 3,105 | 3,506 | 0.026007% |

For the full universe, missing slots are 0.002941% and quarantined slots
0.041975%; **99.955083% of expected slots are usable**. Nothing is double-counted:
missing slots are counted only outside quarantined dates. Known special-session
durations are used rather than multiplying all 26 dates by 375. No additional
absent dates, outside-session rows, unknown-volume rows or critical consistency
issues remain in the usable data under the recorded audit rules.

**Assessment: suitable for the current gap-aware percentage-touch research.**
That conclusion combines completeness, OHLC/schema checks, full label
recomputation, immutable source hashes and the independent daily sample. It is
not a claim that 99.955% of prices are independently known to be correct, or
that the tape is certified for cash execution/live orders. The denominator is
the 683-date captured union calendar; independently missing exchange dates,
historical membership/ticks, adjustments and CAS eligibility remain limitations.

Low candle loss is not identical to low research-sample loss. The frozen
research manifest records 1,897,197 candidate decisions, 99,850 excluded for
unavailable features and 909 for incomplete future windows. The 1,796,438
included decisions yield 3,592,876 paired actions: **94.6891% included / 5.3109%
excluded**. Warm-up and strict prior-session requirements also cause exclusions;
do not attribute all of them to bad candles. Missing windows are never imputed
or counted as TP failures. The new evaluation audit found all 686,112 selected
evaluation actions complete for 60-minute, 15:00 and continuous-close outcomes.

Evidence: [horizon and quality audit](../data/research/tp_probability_horizons_kite_20261005/report.json)
and [frozen dataset manifest](../data/research/tp_only_kite_20261005_final/manifest.json).

Artifacts: [active cache manifest](../data/candles/dataset.json),
[validation](../data/kite_migration_20261005/validation.json),
[staging report](../data/kite_migration_20261005/report.json),
[deletion receipt](../data/kite_migration_20261005/retirement.json), and
[research manifest](../data/research/tp_only_kite_20261005_final/manifest.json).
The raw Kite checkpoints and rejection metadata remain separate from usable data.

Deletion completed and was independently checked on disk: **1,188 files,
1,626,923,933 bytes**, with no reviewed retired target remaining. Active cache,
research-shard and pinned-source hashes still match after deletion. Only the
final Kite research version remains; its superseded intermediate rebuild had
identical shards and was removed separately. Historical aggregate comparisons,
redacted documentation, standalone public references and source scripts remain
clearly separated from usable datasets. The subsequent authorized remote cleanup
also completed: **753 blobs / 384,257,861 bytes deleted, eight old jobs archived**,
zero remaining blobs in the two inventoried containers and zero compute nodes.
Reusable infrastructure remains; see the
[cloud receipt](../data/azure_cleanup_20261005/receipt.json). Azure Files shares
were preserved and not content-audited; job metadata remains archived.

A fresh [63-feature arithmetic audit](../data/kite_migration_20261005/feature_arithmetic.json)
passed **443,634 numeric comparisons**, plus unavailable-value checks, over
**8,760 Kite candles** for RELIANCE, HDFCBANK, INFY, SBIN, NIFTY and GOLDBEES on
September 1-4, 2026. Six five-/twenty-session fields had no numeric coverage in
this short sample; their longer-history behavior is covered by synthetic tests.
The migration's post-deletion offline suite passed **979 tests in 93.00 seconds**.
Four tests that previously loaded retired study artifacts now use explicit
synthetic protocols, preserving chronological and schema checks. Whole-repo
lint, formatting (119 files), dependency compatibility and whitespace checks
passed; touched-file editor diagnostics are clear. The final summary is in
[verification.json](../data/kite_migration_20261005/verification.json).

**Kite is materially better in these checks, not flawless.** Five June 25, 2024
opening bars have open outside their high/low: BHARTIARTL, BSE, HDFCBANK,
ICICIBANK and SBILIFE. TMPV's October 14, 2025 minute prices conflict with its
daily range around a demerger. Other index/price-range inconsistencies also
triggered whole-day quarantine. No malformed row was silently fixed or accepted.
Even when all values are absent, missing minutes can invalidate rolling
features and labels. Absence is safe only when consumers detect and reject it.

The independent daily sample is September 1 and September 30, 2026. All 108
OHLC comparisons differ by at most INR 0.10; all daily volumes match exactly.
The original NSE CSV hashes and URLs are stored with validation. Daily totals
cannot certify minute timing, intraminute paths or executable fill quality.

Four worker clients share one history limiter at 2.5 starts/second. Downloads
resume only matching, hash-verified checkpoints. Promotion verifies every
minute/daily file and copies into a new directory before activation. Normal
cache reads verify fingerprints, refuse in-place refresh, and reject known
gaps/quarantines. The TP consumer preserves the full captured calendar and
explicitly excludes missing feature/future windows. Whole-day omissions do not
compress the prior-five-session lookback.

The research target, feature identities, chronological splits and no-stop
semantics were not tuned. The new phase counts are 1,905,708 training, 665,616
validation, 335,440 calibration and 686,112 retrospective evaluation actions.
Migration itself did not fit models. The subsequent
[Kite CatBoost study](TP_ONLY_TRAINING.md) completed eight fresh candidates and
one seed repeat locally, with no new Azure training job. A 462-tree depth-four
model won validation; calibration worsened later loss. All evaluation forecasts
reproduce exactly. Old model performance does not carry over to replacement data.

Important remaining limits: current rather than point-in-time membership;
Kite-adjusted prices and volumes rather than original execution prices;
current instrument ticks; CAS classification by target role rather than verified
historical eligibility; and a provider-observed, not independent holiday calendar.
The generic feature engine conservatively requires 375 minutes for complete
previous-session features. Its shortened-session readiness is not redefined
by this migration. Special sessions remain in raw history but do not create
normal-session TP decisions when required features are unavailable.

The user-assisted runtime retry on October 5 at 22:18 IST **passed authentication,
profile, LTP, detailed quote and WebSocket streaming**. It received one RELIANCE
snapshot with native cumulative volume and one NIFTY snapshot, with timestamps
and prices normalized through the production adapter. The
[runtime receipt](../data/kite_migration_20261005/live_check.json) records the
bounded check; earlier failed attempts remain in dated diagnostic files.

The first authenticated stream attempts closed with code 1006 before connecting.
Credential-free probes isolated a TLS compatibility issue: the default SDK path
dropped TCP, explicit hostname-aware TLS reached certificate validation, and
using existing system CAs then reached the expected HTTP 403 for synthetic
credentials. Production now uses hostname/peer-verified platform trust and the
same SDK protocol. No verification was disabled and no certificates installed.
The authenticated production retry then succeeded. SDK automatic retries are
disabled; the existing supervisor owns reconnects.

After this fix, **983 tests passed in 134.54 seconds**, including real SDK/TLS
construction with only the final TCP dial mocked, peer-verification assertions,
empty-trust refusal and untrusted-endpoint refusal. Lint and formatting passed.
No historical dataset, pinned research source, model or order behavior changed.
After-hours snapshots do not establish moving market-hours prices, candle
parity or general automated-trading reliability. The subsequently authorized
October 6 market-hours check below supplies actual moving-feed evidence, but
does not establish exact vendor candle parity or unattended execution readiness.
The system remains research only; no model is wired into live trading decisions
and no order-placement path exists.

## Market-Hours Validation: October 6

**Result: bounded data-path validation completed with gaps, not end-to-end
trading approval.** Private authentication, profile, LTP, detailed equity quotes
and full-mode streaming passed. The first attempt ended at browser authentication
without capturing market data; the user-coordinated retry succeeded. Tokens
stayed in process memory; neither credentials nor callback URLs were inspected.

The successful capture received **8,566 ticks**, from **14:22:34 through
14:31:54 IST**, for RELIANCE, HDFCBANK, ICICIBANK, INFY, SBIN, NIFTY, GOLDBEES
and INDIAVIX. All eight prices moved. All five stocks and GOLDBEES carried
native cumulative volume; indices did not, as expected from their packet type.
There were no timestamp/volume regressions, late ticks or duplicate candles.

The runner used prior history plus a buffered current-session overlap. It
discarded the initial partial minute and did not flush the trailing partial
minute. The eight-minute supervised window produced **64 complete candles and
64 generic-feature rows**, with no pending unaligned feature batches. One
intentional client disconnect at 14:26:54 was followed by a successful new
connection at 14:26:56; no other connection failures or silent sessions occurred.
This verifies socket recovery, not reconstruction of data missed while offline.

Saved-tick replay exactly reproduced all 64 candles and feature rows. An
independent review also rebuilt their OHLC and cumulative-volume differences
directly from the recorded ticks. The independent 60-digit feature reference
passed **268,685 numerical comparisons and 74,854 unavailable-value agreements**
across **5,453 warm-up-plus-live bars**. These totals include warm-up, not just
64 live bars. Generic `core_ready` was true for all live feature rows; that does
not mean every long-history feature was populated.

### Subsequent History Comparison

History was fetched again after capture. A one-minute settling exclusion left
**56 comparable candles** (seven per instrument, 14:23-14:29 starts); the last
eight live candles were not included in this comparison. All 56 timestamps
were present, **35 matched OHLCV exactly**, and **45 matched OHLC exactly**.

| Instrument | Exact OHLCV / Compared | Exact OHLC / Compared | Largest Recorded Differences |
|---|---:|---:|---|
| RELIANCE | 7/7 | 7/7 | None |
| HDFCBANK | 7/7 | 7/7 | None |
| ICICIBANK | 5/7 | 6/7 | Low INR 0.40; volume 402 shares |
| INFY | 2/7 | 4/7 | Open INR 0.30; low INR 1.05; volume 11 shares |
| SBIN | 7/7 | 7/7 | None |
| GOLDBEES | 7/7 | 7/7 | None |
| NIFTY | 0/7 | 0/7 | Close 1.30 points; open 0.90; high 0.20; low 0.55 |
| INDIAVIX | 0/7 | 7/7 | Only volume: live unknown versus historical zero |

NIFTY volume also differs as unknown versus zero on every compared minute.
The [official packet specification](https://kite.trade/docs/connect/v3/websocket/#index-packet-structure)
has no index-volume field. This representation difference is expected and must
not be fixed by fabricating traded index volume. The OHLC and tradable-volume
differences are genuine measured discrepancies. They occur both before and
after the controlled reconnect, so the reconnect alone does not explain them.
ICICIBANK's volume differences are -402/+402 in adjacent minutes; INFY's are
-11/+11, consistent with boundary attribution rather than missing session totals.
These observations do not prove the vendor's exact aggregation algorithm.

Only **31/56 comparable generic feature rows** were identical to the subsequent
history path. The correct calculation on sampled ticks is not proof that those
ticks reproduce vendor minute history. No tolerance was selected after the fact
to turn the exact-parity failure into a pass.

### Model And Recovery Limits

The original runner made ten complete-input diagnostic records at 14:25/14:30
(five stocks, both directions). Their raw/calibrated outputs reproduced exactly
from saved ticks and matched the corresponding historical-input outputs in
this small sample, despite seven records having at least one differing input.
**All ten are outside the trained 09:40-14:15 decision window.** They demonstrate
inference plumbing only, not in-distribution prediction or TP-outcome accuracy.
No TP labels, paper positions, costs, fills or live orders were evaluated.

Review found two gaps in the new validation runner: no last-decision-time gate,
and no inference veto after a reconnect without history reconciliation. Both
are now guarded and tested. The corrected offline replay withholds all ten
late records and retains exactly the same feature calculations. Discontinuity
events are now recorded and replayed, with inference disabled for the remainder
of an unreconciled run. This safeguard is in the diagnostic runner; it does not
implement a production recovery/orchestration layer or rewrite the captured run.

At the end of the October 6 run, the following were its remaining data-path and
future-operation checks. This is a **dated checklist, not the current next-session
plan**: October 7 subsequently supplied in-window inference, and the construction
investigation supported the scoped research acceptance at the top of this guide.
Use the [saved-tick runbook](LIVE_API_SAMPLES.md#offline-development-first) for
any separately scoped follow-up:

1. Explicitly choose and validate the completed-candle source for model inputs.
   Vendor minute candles offer training-source consistency but require measured
   publication delay/revision behavior; sampled-tick candles require a measured
   live/history mismatch policy. Do not silently substitute one for the other.
2. Reconcile history and feature state after a feed interruption before resuming
   inference. A connected socket and a present candle are insufficient evidence.
3. When requested, run a new authenticated capture inside 09:40-14:15 with the
  corrected guards, chosen source policy and completed recovery behavior.
4. Separately validate prospective shadow behavior and any future portfolio,
   risk, journal or execution components. They are not certified by this run.

Evidence: [original capture](../data/kite_live_validation_20261006_retry1/report.json),
[candle differences](../data/kite_live_validation_20261006_retry1/candle_comparison.json),
and [independent review](../data/kite_live_validation_20261006_review/report.json).
Each directory retains its executed source snapshot. All 40 captured evidence
files and protected dataset/model fingerprints verified unchanged. No active
training-cache file, frozen model, runtime broker or feature formula was changed.
The capture process and private callback listener have ended.

```text
capture report SHA256 0ea0665ed735196da3709a0923659b71ff8750f49250d61bb744d0013f0718eb
review report SHA256  bf34116602a92388448f5d3fe26e3b77ea2cc0bdbd99e00b03469c98c63960bc
```

October 6 full regression after the new runner and safeguards: **992 tests passed
in 92.69 seconds**. That is a dated code gate, not a fresh test or live run for
this documentation update. Use new output paths for any repeat. Start with the
OFFLINE review; it requires retained artifacts but no broker login or market wait:

```powershell
& .\.venv\Scripts\python.exe -m scripts.validate_kite_live `
  --review-capture data/kite_live_validation_20261006_retry1 `
  --output data/kite_live_review_NEW
```

Keep `--review-capture`; omitting it selects LIVE authentication/capture. The
separate real-service acceptance command, only when requested and ready, is:

```powershell
& .\.venv\Scripts\python.exe -m scripts.validate_kite_live `
  --output data/kite_live_validation_NEW --duration-seconds 480
```

Active dataset SHA256:
`8220c2dd6c82fc832c2057f8a8fe584a9d2af7a4c5295754d662a289b1f191d5`.
Research manifest SHA256:
`5ccd2091a19f94e896f9570fbfb58bcf04998a18b523a217d501a3c8c7651c81`.

## Earlier RELIANCE Comparison

Everything below records the pre-migration study. Statements about retaining
Groww snapshots or leaving runtime defaults unchanged applied at that time,
not after this migration. Old raw Groww captures and row-level differences are
retired; aggregate reports and independent NSE/Kite evidence are historical.

Verified October 5, 2026. Scope: NSE RELIANCE EQ, January 1, 2024 through
September 30, 2026, one-minute history plus an independent daily-data benchmark.

## Recommendation

**Use Kite as the primary historical-data source for subsequent research.**
Retain Groww as a separate diagnostic comparison, not an automatic gap filler.
This recommendation is supported by this RELIANCE study, not a claim that every
Kite instrument or candle is correct. Validate several other liquid stocks and
live/history parity before a full runtime migration.

Kite had substantially fewer missing minutes, no unknown minute volumes, and no
minute prices outside the independently sampled NSE daily ranges. Groww had four
severe price errors that were reproduced by fresh API requests. Kite still has
eight missing minutes in this capture; none were invented or filled.

No runtime broker default, source Groww cache, frozen research dataset, model,
strategy exit or live-order capability was changed. The user's research exit
remains TP or 15:00 IST, with no stop-loss.

## Download And Coverage

All 33 monthly Kite requests completed, plus a separate daily reference with
683 daily candles. The minute requests returned 254,317 candles. Authentication
used manual browser login with an in-memory token; no credential file was read
by the assistant, and no access token was saved.

The authoritative quality audit accounts for split DR sessions, Muhurat sessions
and the August 2026 closing-auction change. On the union of 683 observed dates,
the verified session windows imply 254,325 minute slots:

| Measure | Kite | Groww |
|---|---:|---:|
| Continuous-session candles | 254,317 | 253,467 |
| Expected slots without a candle | 8 | 858 |
| Observed-session coverage | 99.996854% | 99.662636% |
| Missing slots before 15:00 | 5 | 768 |
| Complete observed sessions | 681 | 676 |
| Entire observed sessions absent | 0 | 2 |
| Unknown volume in continuous candles | 0 | 837 |
| Sampled candles outside NSE daily range | 0 | 4 |

Groww's file has 262,893 rows in total, but 9,426 are outside continuous-session
windows. More rows does not imply better coverage. Auction and post-close
records are retained in the snapshots, not automatically labeled erroneous.

Kite omissions:
- April 23, 2024: 10:52, 10:53, 10:55, 10:56 and 10:57 IST.
- December 15, 2025: 15:15, 15:16 and 15:17 IST.

Groww omissions by date: July 16, 2024 (2); December 18, 2024 (1);
January 10, 2025 (8); September 26, 2025 (37); November 6, 2025 (375);
September 29, 2026 (75); September 30, 2026 (360).

The date universe uses both minute feeds, Kite daily dates and 49 NSE sample
dates. It is **not** a fully independent exchange-calendar enumeration; a date
missing from every source could remain undetected. Thus the download covers the
requested range but is not represented as a perfectly complete market tape.

## Corporate Actions

RELIANCE's 1:1 bonus became effective October 28, 2024. The official
[NSE circular FAOP64621](https://nsearchives.nseindia.com/content/circulars/FAOP64621.pdf)
states an adjustment factor of 2. [Zerodha documents](https://support.zerodha.com/category/trading-and-markets/charts-and-orders/charts/articles/kite-charts-not-matching-as-per-the-records-in-nse-or-bse)
that its historical prices are adjusted for corporate actions, while exchange
archives show prices as traded on the original date.

This capture confirms that Kite's pre-bonus prices are roughly half Groww's,
with corresponding doubled volume. That is a convention difference, not a 50%
data error. For comparison only, the audit multiplies pre-bonus Kite OHLC by 2
and divides volume by 2 to restore the original trading-price basis. Source
files remain byte-for-byte unchanged. Small rounding differences remain.

Do not mix adjusted and unadjusted data in rolling features. Conversely, do not
treat adjusted historical prices as original cash execution prices when
calculating share counts, tick sizes, turnover, fees or position sizes. Preserve
provider, retrieval time and adjustment convention in each dataset version.

## Minute Agreement

After that adjustment and special-session handling, 253,459 minute timestamps
are shared. These are **agreement rates**, not either provider's accuracy score:

| Field | Exact Agreement, Whole Period | Exact Agreement, April 2025 Onward |
|---|---:|---:|
| Open | 46.6178% | 49.8573% |
| High | 80.2560% | 96.0597% |
| Low | 78.3330% | 96.3429% |
| Close | 78.6474% | 99.2782% |
| Volume, known pairs only | 77.4568% | 94.0739% |

Older periods differ more; a September-only comparison was not representative
of the entire history. Opens differ frequently even when closes agree. The
September previous-close/open pattern remains an observation, not a proven
vendor algorithm. High/low differences can change 0.20% TP-touch labels; open
differences can change candle features and simulated entries. No predictive
model was retrained or reselected to resolve these discrepancies.

## Independent NSE Checks

The audit selected 49 dates: the first observed daily date in each month,
provider-only dates, large price discrepancies and election/bonus/CAS boundaries.
This is a targeted diagnostic sample, not a random sample or an estimate of
population-wide error rates.

All 49 official NSE `sec_bhavdata_full` daily CSVs were retrieved, and each
RELIANCE EQ row's identity, date and numeric schema were checked. Files and
fingerprints are retained locally. After reversing the bonus adjustment:

- Kite daily open, high, low and close were each within **INR 0.05** of NSE on
  every sampled date. Exact counts were 45, 46, 43 and 40 of 49 respectively.
- Kite daily traded volume matched NSE **exactly on all 49 dates**.
- Kite minute OHLC stayed inside the official daily low/high, allowing INR 0.10
  for adjustment rounding. Four Groww candles violated this bound.

Daily NSE close is not the last traded price or last minute's close. Daily
volume can include auction trades absent from continuous-minute history.
Accordingly, neither difference is mislabeled as an error in this audit.
Public daily records cannot establish exact minute-level open/high/low truth;
that would require an independent exchange-derived intraday/tick archive.

### Reproduced Groww Errors

On **May 12, 2025, 10:32-10:35 IST**, Groww reports several prices around
INR 142,000, including a high of **INR 142,450**. Kite reports prices around
INR 1,424. The [official NSE daily archive](https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_12052025.csv)
gives a day low of **INR 1,400** and high of **INR 1,439**. The Groww values
therefore cannot be genuine trades for that instrument/date.

Fresh read-only Groww requests reproduced all four bad candles. Fresh probes
also returned no candles for November 6, 2025 or September 30, 2026, and still
stopped at 13:59 on September 29, 2026. NSE daily records confirm trading on
those dates. These findings are not explained only by an old local cache.

No automatic division-by-100 repair was made: some fields in a bad row are
plausible and others are corrupted. Existing Groww-based features, labels and
results need a data-quality revalidation before reliance on their conclusions;
the frozen studies are preserved, not silently replaced or retrospectively tuned.

## Session Evidence

The audit's instrument/date-specific windows are separate from runtime clocks:

- Ordinary sessions before CAS: 09:15-15:30 IST.
- March 2 and May 18, 2024: 09:15-10:00 and 11:30-12:30, per
  [MSD60677](https://nsearchives.nseindia.com/content/circulars/MSD60677.pdf) and
  [MSD61893](https://nsearchives.nseindia.com/content/circulars/MSD61893.pdf).
- November 1, 2024: 18:00-19:00 Muhurat session, per
  [CMTR64628](https://nsearchives.nseindia.com/content/circulars/CMTR64628.pdf).
- October 21, 2025: 13:45-14:45 Muhurat session, per
  [CMTR70319](https://nsearchives.nseindia.com/content/circulars/CMTR70319.pdf).
- RELIANCE continuous trading from August 3, 2026: 09:15-15:15, with CAS
  separate. See [NSE CAS rules](https://www.nseindia.com/static/products-services/closing-auction-session)
  and [historical CAS records](https://www.nseindia.com/static/reports/closing-auction-session-historical-data).

The generic monthly comparison uses uniform per-month windows. Its raw totals
therefore exclude the 2024 evening session and include some special-session
non-trading times. Use the **quality report** for the corrected coverage numbers.

## Artifacts And Commands

Root: `data/broker_comparisons/reliance_202401_202609`, gitignored and local.

- [Full Kite minute CSV](../data/broker_comparisons/reliance_202401_202609/combined/kite.csv)
- [Full Groww comparison CSV](../data/broker_comparisons/reliance_202401_202609/combined/groww.csv)
- [Kite daily CSV](../data/broker_comparisons/reliance_202401_202609/kite_daily.csv)
- [Quality report](../data/broker_comparisons/reliance_202401_202609/quality/report.json)
- [49-date NSE comparison](../data/broker_comparisons/reliance_202401_202609/quality/nse_comparison.csv)

Monthly snapshots have source/data hashes and can resume only against the same
range plan and Groww source fingerprint. Historical calls are paced below
Kite's documented three-per-second limit. A new range download uses:

```powershell
& .\.venv\Scripts\python.exe -m ai_trader.cli.compare_broker_history `
  --browser-login --start-month 2024-01 --end-month 2026-09 `
  --symbol RELIANCE --session-close 15:30 `
  --session-change-month 2026-08 --changed-session-close 15:15 `
  --output data/broker_comparisons/reliance_202401_202609_NEW
```

The original NSE capture attempt through Python timed out; the same public files
were successfully downloaded with Windows `Invoke-WebRequest`, without disabling
certificate validation. The original capture log is preserved, including those
failures. The final quality report validates the successful files and records
their hashes; it does not mistake the initial transport failure for missing data.

Reproduce the final audit entirely offline, using a new destination:

```powershell
& .\.venv\Scripts\python.exe -m scripts.audit_broker_history `
  --source data/broker_comparisons/reliance_202401_202609 `
  --evidence data/broker_comparisons/reliance_202401_202609/evidence `
  --output data/broker_comparisons/reliance_202401_202609/quality_RECHECK
```

Next migration gates are explicit gap/invalid-price rejection, consistent
adjustment metadata, date-aware sessions and multi-stock live/history validation.
This study does not certify live latency, fills, execution quality or profitability.