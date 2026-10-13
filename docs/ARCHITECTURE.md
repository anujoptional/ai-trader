# AI Trader: Current And Target Architecture

Canonical system contract. Section 0 describes the system as implemented on
October 13, 2026 (live paper trading). Sections 1-8 are the October 9
consolidation of the research-phase contract and still apply where section 0
does not supersede them. Read this before architectural or cross-module changes.
No live orders, order methods or execution adapter exist; the paper trader
simulates fills. The old design is preserved in a collapsed archive below; its
Groww providers, stop-based exits and dated status are not current contracts.

## 0. Implemented System (October 13, 2026)

A daily paper-trading session, `ai_trader.cli.run_paper`, started by a Windows
scheduled task at 08:30 IST on weekdays (the Kite login stays manual). Long-only
delivery (CNC), Rs 2 lakh per trade, +1% target and -1% stop with no time limit,
at most 25 positions and Rs 50 lakh capital per AI. Operation, setup, results
and next steps: [handover.txt](handover.txt) section "CURRENT STATE"; full
detail: [AI_LAYER.md](AI_LAYER.md).

```text
Kite history (start of day)            Kite WebSocket ticks (09:14-15:31)
  daily rows on NSE's price basis                  |
  (paper/basis.py: NSE bhavcopies)                 |
        |                                          |
per-stock minute and daily history      LiveBars: minute bars, 652 instruments
        |                                 (decide only on complete, fresh bars)
        |                                          |
        +--> scanner: CatBoost P(+1% before the close), 500 stocks, every 5 min
                  | >= 0.60 (shortlist re-scored exactly on Kite history bars)
        order book (Kite depth), futures OI, NSE filings, features
                  |
        AI request v4 (facts only, web search) -> main AI + shadow AIs
                  | BUY / NO_BUY per stock, journaled
        deterministic paper broker per AI: size, limits, fill at the asks
                  |
        exit check every minute on completed bars (+1% / -1%)
                  |
        SQLite portfolio and journal, reports, logs, raw input capture
```

| Layer | Modules | Notes |
|---|---|---|
| Session owner | `cli/run_paper.py`, `paper/session.py` | decision, exit and stream threads; lock; NSE holidays |
| Live bars | `paper/live.py`, `broker/kite_stream.py`, `market/stream.py` | ticks to minute bars; decisions only on complete, fresh stream bars (gaps repaired from Kite history, else the cycle is skipped; no full-fetch fallback) |
| Price basis | `paper/basis.py` | daily history on NSE's official prices, structural actions only (the research basis); a stock without a provable basis is left out that day |
| Scanner | `scanner/upside.py`, `data/models/scanner_up1_by_close_v1` | features identical to research; trained on F&O stocks |
| Components | `orderbook/`, `oi/`, `news/` | built as of the decision time |
| AI | `ai/decision.py`, `ai/openai_client.py`, `ai/anthropic_client.py`, `ai/packet.py`, `ai/journal.py` | strict JSON; failures become NO_BUY; daily budget per AI |
| Paper execution | `paper/broker.py`, `paper/store.py`, `paper/charges.py`, `paper/report.py` | deterministic sizing and exits; the LLM never sizes |
| Validation | `scripts/validate_paper_inputs.py`, `scripts/replay_paper_session.py`, `scripts/compare_ai_decisions.py` | NSE files, offline replay, AI comparison |

What changed from sections 1-8: the exit contract is +1% / -1% with no time
limit (delivery, may carry overnight) instead of "0.20% TP or 15:00"; the AI
decision layer, journal and paper portfolio are built; the scanner is the
CatBoost model above, not the rules scanner. Kite remains the sole broker and
market-data source; NSE's official files (daily bhavcopies, corporate actions,
filings) are exchange reference data for the price basis and the filings. Still
absent: live orders, a risk engine beyond the paper limits, and automated Kite
login (not allowed).

## 1. Goal And Binding Decisions

Build an AI-assisted intraday research and, only after separately authorized
validation, execution system for liquid Indian equities. Both LONG and SHORT
are in scope. Holding periods are minutes to hours, not overnight or
high-frequency. Zerodha Kite is the **sole** broker and historical-data source.

- Current direction: carry the validated built layers into scanner research.
  October 7 checks covered historical data/model reproduction, read-only live
  ingestion, features, in-window inference, reconnect veto and saved-candle rule
  scanning. Unbuilt services are outside that validation scope, not failed checks
  or prerequisites for starting scanner research. The October 9 task consolidates
  documentation only; it does not start a new experiment, capture or deployment.
- **October 11 user direction (supersedes the LONG/SHORT scope for now):** the
  scanner is a recall-first filter that removes clutter and passes every
  high-potential candidate to the AI layer; it is not the trade decider. Focus is
  LONG-only CNC (delivery product) for safety, medium frequency (ballpark ~100
  trades/day at ~0.2% each) and Rs 2 lakh per trade. A same-session CNC buy and
  sell is charged at intraday rates (~0.059% round trip at Rs 2 lakh); holding
  overnight pays delivery charges (~0.23%), so same-day exits are the default and
  an overnight carry is an exception that still needs an explicit policy.
- User's intended normal exit: **0.20% directional TP or 15:00 IST, no price
  stop-loss and no forced 60-minute exit**. No LLM or inherited replay default
  may silently add another exit. Entry/allocation rules still need a frozen
  forward-study contract before model integration.
- The research label is a literal next-60-minute TP touch. It is not a fill,
  net-profit probability, or the TP-or-15:00 execution outcome. Late research
  decisions can count a TP after the intended square-off time.
- Runtime rule diagnostics consume one-minute candle closes. The new model was
  researched on a five-minute decision grid, 09:40-14:15. Every-minute model
  use is a different deployment distribution and has not been validated.
- AI may eventually select an entry/no-entry and direction from an approved
  action space. Sizing, exposure, eligibility, validity and exits belong to
  deterministic code. A learned forecast never overrides a veto.

Binding safety rules remain in [AGENTS.md](../AGENTS.md). Do not read credential
files, log secrets, place orders, add a broker fallback or promote an evaluated
model without an explicit new task. Keep infrastructure simple: typed Python
3.12, small modules, tests and structured records; SQLite is the intended
journal store, not a reason to add other databases or orchestration frameworks.

### Research Acceptance And Scope

**The implemented historical and real-time pipelines behave as expected under
the tested conditions. Their calculations and outputs are reproducible, and the
input data is suitable for continued scanner research.** This is the agreed
built-system conclusion after the candle-construction investigation, not a new
test run or a change to any frozen result.

| What Was Validated | Evidence Scope | What It Does Not Establish |
|---|---|---|
| Historical cache and TP outcomes | 66 instruments, all 3,592,876 labels/outcomes, 108 retained NSE daily comparisons | Independent truth for every minute or original cash-execution prices |
| Frozen CatBoost | All candidates/repeat verified; 686,112 raw/calibrated evaluation forecasts reproduced | A newly trained/promoted model, new holdout or profitable scanner |
| Live market/feature/model path | Two bounded eight-instrument captures; October 7 had five eligible target records at one decision time | Full-day/all-universe coverage or every possible outage |
| Candle and feature arithmetic | Independent saved-tape reconstruction and generic/TP references; missingness and boundary tests | Exact equality between quote-derived and vendor historical candles |
| Rule scanner and feasibility | 16 offline cycles over 80 saved live target snapshots, 41 repeatable candidates | A prospective TP entry policy, actual trades or portfolio performance |
| Existing safeguards/replay | Full regression suite plus live reconnect and inference veto | Automatic history repair or a production position/execution service |

Keep three decisions separate: **built-flow correctness is validated to this
scope; scanner usefulness is the next research question; future unattended or
live execution remains a separate project stage.** Candle-source differences,
observed-calendar coverage and adjusted prices remain explicit limitations.
They do not require another full validation campaign before studying scanner
quality. Investigate their material effect on a proposed rule or model policy
when relevant; do not erase differences by choosing a favorable tolerance.

The dated reports retain their original `bounded_validation_with_gaps`,
`review_complete_not_end_to_end_approved` and `not_live_trade_ready` statuses.
Those flags include exact source parity or whole-product readiness criteria;
they are not being rewritten as all-green results. This scoped acceptance is
compatible with those still-unmet broader criteria. No orders are authorized.

### Operating Modes And Scope

| Mode | Meaning | Current State |
|---|---|---|
| Offline research | Immutable historical inputs, model fitting/evaluation and diagnostic price outcomes | Implemented; versioned Kite study is frozen, not scheduled to retrain |
| Offline tick replay | Reprocess retained market evidence and synthetic faults without credentials or network | Implemented and verified with authentication/browser/socket calls blocked |
| Read-only live diagnostics | Bounded quote/tick/history capture and calculation checks against current market data | Built path validated to the recorded scope; source differences and recovery limits documented |
| Forward shadow | Journal prospective decisions and outcomes without broker orders | Running since October 12 as the paper session: main AI plus shadow AIs, all journaled |
| Paper portfolio | Simulated accepted entries, capital/exposure, positions and exits | Implemented October 12 (`paper/`): one portfolio per AI, +1% / -1% exits, reports |
| Live execution | Submit and reconcile real broker orders | Absent and disabled; requires separate explicit authorization and validation |

An authenticated data connection is not a trading mode switch. A successful
command, a passing test suite or a high target-touch rate is not authorization
to progress to the next mode. The current scope is NSE cash-equity research;
indices and ETFs provide context. Derivatives, overnight holdings, leverage
assumptions and broker order capabilities are not approved extensions.

The economic hypothesis is many small intraday captures that remain positive
after losses and transaction costs. The current 0.20% target is **gross**, not
0.20% net income. The measured high-score miss severity has not established
that hypothesis. Position size, entry thresholds, daily loss limits and capital
allocation for a future policy remain to be specified; the older fixed-clip
and stop-based replay settings are experiments, not implied live defaults.

## 2. Pipeline And Build Status

October 9 state. Section 0 supersedes it for the scanner, AI, journal and
paper layers, which are now built.

```text
Kite read-only history                 Kite read-only full-mode ticks
         |                                         |
versioned capture + validation          StreamSupervisor + native volume
         |                                         |
immutable CandleStore                  CandleBuilder -> completed candles
         |                                         |
         +---------------> MarketState <-----------+
                                |
                  +-------------+----------------+
                  |                              |
          generic FeatureEngine           TP-specific feature builder
            63 derived fields             13 numeric features + side
                  |                              |
        deterministic rules scanner       frozen CatBoost study/inference
                  |                              |
       legacy offline replay/control      offline probabilities + audits
                  |                              |
                  +-------- future integration --+
                                |
                constrained AI decision + journal          NOT BUILT
                                |
                  deterministic portfolio risk             NOT BUILT
                                |
                position manager / paper execution         NOT BUILT
                                |
                 separately authorized live executor       DEFERRED
```

The two feature paths share time/data conventions, not interchangeable arrays.
The trained model is a frozen research consumer with bounded read-only live
diagnostics. It is **not** wired into the runtime rules scanner, and no full
shadow-trading loop is running.

| Layer | Implemented Today | Boundary / Remaining Work |
|---|---|---|
| Broker | `KiteBroker`, safe profile/LTP/quotes, daily/minute history, ticks | Read-only abstraction; no order methods |
| Market state | Candle construction, native volume, bounded live capture and exact saved-tick replay | Observed aggregation correct; vendor equivalence not assumed; automatic post-outage state repair unbuilt |
| History | Versioned download, hashes, quarantine, immutable promoted cache | Not independent exchange truth or fully point-in-time reference data |
| Features | Generic 63-field engine and independent reference; separate TP feature builder | Generic shortened-session prior readiness remains conservative |
| Rules / costs | Deterministic hypotheses, feasibility, sizing arithmetic, Zerodha fee estimates | A research control, not an approved profitable strategy or full risk engine |
| Replay | Shared deterministic event/accounting and friction diagnostics | Existing stop-based policy is not the user's no-stop contract |
| Learned research | Eight Kite CatBoost candidates, locked winner/calibration, exact offline inference and audit | No runtime promotion, forward study or profitable execution claim |
| AI orchestration | Intended constrained decision layer | No LLM endpoint, prompt loop, session owner or model-to-scanner adapter built |
| Risk / positions | Useful replay primitives and portfolio state types | No production risk engine, position manager or reconciled paper trader |
| Journal / execution | JSON/NPZ research evidence and diagnostic outputs | Persistent end-to-end decision journal and live execution deferred |

### State And Feedback

The target pipeline is not a one-way call chain. A future session owner must
supply the same as-of portfolio state to AI selection and deterministic risk,
and receive updated state after every accepted/rejected instruction, simulated
fill or reconciled broker event. A candidate is not an order or an open position.
Existing diagnostic CLIs and offline forecast arrays do not implement this loop.

```text
completed market data -> features -> candidates / forecasts -> decision intent
                     |              |
                   portfolio state -> risk veto
                     ^              |
                     |       accepted instruction
                     |              |
                  position lifecycle <- paper/execution
                     |
               journal + realized outcomes + reconciliation
```

Journaled decisions, data coverage, model IDs and stage timings are the feedback
needed to compare rules, raw CatBoost, calibrated CatBoost and any future LLM
selection on the same opportunities. These comparisons are not permission for
automatic model updates. A replacement requires a separately versioned research
protocol, held-out evidence and explicit promotion review.

## 3. Data And Time Contracts

### Broker And Stream

Broker-specific code stays behind
[kite.py](../src/ai_trader/broker/kite.py) and the read-only boundary. Authentication
uses privately loaded `KITE_API_KEY`/`KITE_API_SECRET`, state-checked browser
login on `127.0.0.1:8765/kite/callback`, and an in-memory session token. The user
completes login/2FA; no TOTP automation or credential inspection is allowed.

Full-mode ticks carry exchange timestamps and native cumulative volume.
`StreamVolume` observes those values; missing volume remains unknown, not
REST-imputed or forward-filled. `StreamSupervisor` owns reconnects and bounded
collection. **One healthy WebSocket is reused throughout the run:** 60-second
collection windows do not close it or resubscribe. Only retryable failures or a
silent collection cause replacement, and final shutdown closes the transport.
The recorded disconnect was a deliberate diagnostic probe, not scheduled normal
operation; October 7 had zero unexpected disconnects or silent collections.
SDK retries are disabled to avoid two independent retry owners.
[kite_stream.py](../src/ai_trader/broker/kite_stream.py) reuses the pinned SDK
protocol with explicit hostname/peer TLS verification against existing system
CA roots, and rejects an unexpected host, scheme, port or empty trust store.
No certificate verification is disabled and no new CA was installed.

October 5 after-hours access was followed by two market-hours captures. October 6
retains 8,566 ticks/64 candles and the after-cutoff rejection case. October 7
retains 13,819 ticks/128 candles, five eligible records at 12:45 and ten later
records vetoed after the controlled disconnect. Saved-tape replay and independent
arithmetic passed. The [current review](../data/kite_live_validation_20261007_review_v3/report.json)
and each capture's executed source preserve what ran; later review changes do
not retrospectively modify an observation.

### Candle Meaning And Source Comparison

An input tick is a normalized price update, not a complete minute or guaranteed
record of every exchange trade. A built `Candle` contains the instrument,
start/end timestamps, open, high, low, close and minute volume. Open/close use
the first/last observed prices in event-time order, with arrival order resolving
equal timestamps; high/low are extrema of the observed prices. Minute volume
is the difference between the last native cumulative reading in this minute
and the previous minute, when that baseline is valid. Unknown is not zero.
Indicators and forecasts are downstream derived records, not candle fields.

The comparison is **our WebSocket-derived minute against Kite's historical API
for the same stock, date and exact completed minute**, fetched after capture.
It is not a comparison with a different day's training candle. October 7 had all
120 timestamps present: 74/120 exact OHLCV, 91/120 exact OHLC and 58/120 exact
generic-feature rows. Strict equality fails for any one differing field; it is
not a bad-tick percentage. Thirty index rows have live unknown/history-zero
volume. Excluding indices, 74/90 stock/ETF OHLCV rows, 88/90 closes and 86/90
volumes matched exactly. The largest stock OHLC difference was INR 0.30 on
ICICIBANK around INR 1,350, about 0.0222%; small, but not automatically irrelevant
to a 0.20% target or a threshold-sensitive rule.

The [construction investigation](../data/kite_validation_20261007/candle_construction_analysis.json)
independently reproduced all 192 candles from both tapes. Differing historical
extrema were absent from the received minute's prices; stock-volume differences
cancelled across adjacent minutes. Uniform timestamp shifts and filtering prices
by volume increase made agreement worse. No observed aggregation arithmetic
bug was found, and no production algorithm was changed to force parity.

The adapter retains exchange time, not separate last-trade time. The latter
could help attribute quote updates across a minute boundary, but the saved tape
cannot test that hypothesis. `collected_at` is queue-consumption time, not socket
arrival or a latency measurement. Keep the present construction as the research
baseline; a targeted dual-timestamp capture is optional follow-up when needed,
not a prerequisite for resuming scanner research. Healthy observed input and
correct arithmetic do not imply perfect historical equivalence or every-trade
coverage. Detailed facts and limitations belong in
[BROKER_DATA_QUALITY.md](BROKER_DATA_QUALITY.md).

### Immutable History

The [active manifest](../data/candles/dataset.json) records **66 instruments,
16,787,986 usable minutes and 45,052 daily rows**, January 2024-September 30,
2026. Four isolated history clients share one 2.5-request/second start budget.
Captures are staged, checkpointed and hash-verified before promotion. Invalid
or inconsistent instrument-days are quarantined in full, not patched by guess.

There are 494 missing minute slots on usable observed dates and 26 quarantined
instrument-days across 15 instruments. No additional full-day absence was found
on the captured union calendar; this is not proof against an independent
exchange calendar. Raw rejected evidence is retained outside the active cache.
The completeness recheck converts those quarantines to 7,050 minute slots:
7,544/16,795,530 expected slots excluded, or 0.044917%. The 53 instruments used
as model inputs lose 0.026007%. Feature/lookback and future-window requirements
separately exclude 5.3109% of candidate research decisions, including warm-up.

Ordinary `CandleStore` reads reject recorded gaps/quarantines. Only explicitly
gap-aware research consumers use `allow_incomplete=True`, then exclude affected
windows. The calendar preserves missing/quarantined prior dates rather than
substituting an older complete session. Promoted files cannot be refreshed or
appended in place. A changed source or schema requires a new dataset version.

### Features And Labels

A same-session gap resets generic rolling readiness and withholds session-wide
values. Overnight transitions retain their separately tested return semantics.
The generic engine's prior-session contract still requires 375 minutes; a
shortened CAS session is not declared complete under that contract. The TP
builder explicitly uses 360-minute 09:15-15:15 cutoff volatility and prior
daily-close history. Do not silently unify these two different definitions.

The [TP manifest](../data/research/tp_only_kite_20261005_final/manifest.json)
contains 3,592,876 paired side-actions in 200 shards. Training ends June 2025,
validation December 2025, calibration March 2026, and retrospective evaluation
September 28, 2026. Inputs are 13 numerical values plus side, with no identity,
timestamp, future label or excursion feature. All required future minutes must
exist; unknown outcome is not a failure label. Every label/first touch/adverse
excursion/terminal return was independently recomputed successfully.

### Why Execution History Needs More Work

These are three distinct issues, not three new model bugs:

1. **Corporate actions:** after a 1:1 bonus, a vendor may halve earlier prices
   and double volumes. Percentage research can remain meaningful, but historical
   shares, turnover and fees cannot be reconstructed by treating adjusted prices
   as the original cash trades. No adjusted/native provider splicing is allowed.
2. **Sessions:** CAS-eligible stocks can end continuous trading at 15:15 while
   other instruments use 15:30; holidays and split/Muhurat sessions differ too.
   Expecting the wrong close creates false gaps and wrong feature windows.
   Migration checks handle known date regimes; runtime does not yet provide a
   fully instrument/date-aware exchange calendar or historical CAS membership.
3. **Point-in-time reference data:** today's tick grid and index constituents
   are not necessarily those of 2024. Using them can bias fills and universe
   selection. Current instrument ticks and membership are documented limitations.

The 108 recent NSE daily checks passed, but daily references cannot certify
intraminute timing, bid/ask liquidity or fills. These gaps do not invalidate the
completed label/forecast reproducibility checks; they limit claims about cash
execution and generalization. Full details: [BROKER_DATA_QUALITY.md](BROKER_DATA_QUALITY.md).

## 4. Current Model Contract

**October 10 update:** a fresh CatBoost for 09:40-13:30 IST calls (same 13
features plus side, raw output, 511 trees, trained through December 2025 and
selected on January-March 2026) beats the model below on April-September window
calls by a predeclared paired-loss rule. It is the preferred research model but
not runtime-integrated. P >= 0.75 calls still lose after estimated charges under
TP-or-15:00; see [TP_ONLY_TRAINING.md](TP_ONLY_TRAINING.md).

The new Kite-only study trained eight CPU CatBoost candidates plus one seed
repeat, totaling 544.853 reported fit seconds. Validation selected depth 4,
LR 0.03, leaf L2 20, **462 trees**. The repeat was diagnostic only. Exact weights
were locked before January-March calibration, and the chosen side-sigmoid map
was locked before April-September evaluation. There was no refit on later data.

Across 686,112 evaluation actions / 123 sessions, raw log loss is **0.668632934**
and selected calibrated log loss **0.669463747**; Brier is respectively
0.237986933 / 0.238367858. Calibration worsened later probability accuracy.
Both beat the frozen base-rate controls, but none is an executable profit
estimate. All saved probabilities reproduce exactly on reload.

The separate raw-probability horizon audit reuses all 686,112 actions without
changing training or selection. Overall hit rates are 56.7148% in 60 minutes,
70.2049% by 15:00 and 74.3221% by recorded continuous close; no outcome windows
are missing. Raw one-hour calibration error is 0.9014 percentage points across
ten bins, versus 12.8551 and 16.9723 points for the two changed horizons. Ranking
persists, but the percentages are not calibrated probabilities for those other
events. No returns, fills or portfolio profit are inferred from these counts.

The October 6 frequency/return follow-up separately measures the unchanged
calls' gross 15:00 returns. The 204,428 no-TP calls average -0.5074%; the 579
no-TP calls in the raw P >= 0.80 subset average -1.3930%. That subset's idealized
TP-or-15:00 average is -0.00588% before costs per opportunity, not portfolio P&L.
It contains 3,030 distinct scored candle-times (24.6/session across 50 stocks),
not 4,480 independent tradable entries. Both directions qualify at 1,450 of
those times. Trading frequency requires a separately fixed side/position/
duplicate policy; no such policy or order capability was added by this audit.

The October 6 capture had no in-window decisions. The October 7 run now recorded
five eligible targets at 12:45, producing ten directional forecasts, followed
by ten correctly withheld records after an unreconciled disconnect. The probe
waits for an eligible prediction before injecting the fault. Exact replay and
an independent 60-digit check of all 195 TP input values passed; all five issued
records also matched the corresponding historical-input probabilities. These
are bounded diagnostic checks, not a production session owner, automatic history
recovery, prospective TP-outcome evaluation or model promotion.

[TP_ONLY_TRAINING.md](TP_ONLY_TRAINING.md) is the full report, including settings,
calibration, thresholds, uncertainty, hashes and inference/verification usage.
The final model lives under `data/research/tp_catboost_kite_20261005/final`.
`predict_frozen` returns the locked calibrated output from a finite 14-column
matrix. Keep raw as the predeclared comparison, not a hindsight-selected
replacement. This calendar period is retrospectively examined, not a pristine
future holdout. Further tuning needs a new approved study and honest holdouts.

### Meanings That Must Stay Separate

| Term | Meaning In This Project |
|---|---|
| Minute candle | One instrument's OHLCV over one minute, timestamped at its start |
| Scored candle-time | A feature-ready decision point on the five-minute research grid, not every retained minute |
| Side-action / forecast | One LONG or SHORT prediction; each scored candle-time produces two |
| TP touch | A high/low reaches the directional reference-price target inside a specified window; not an executable fill |
| Raw probability | Sigmoid of the frozen CatBoost margin; not the unbounded margin itself |
| Calibrated probability | The frozen, chronologically fitted side-specific mapping; not proven better than raw on later data |
| Candidate / intent | A proposed action for selection/risk, not an accepted position |
| Trade / position | A lifecycle requiring accepted entry, sizing, fills and exits; not established by a forecast array |
| Gross return | Direction-adjusted price return on entry notional before fees, spread and slippage; not leveraged return on capital |
| Parity | Equality under a stated comparison: same-tick replay and subsequent vendor-candle equality are different checks |

## 5. Intended Decision And Safety Layers

These are target responsibilities, **not implemented production capabilities**:

- **Session orchestration:** own warm-up, universe resolution, completed-candle
  cycles, reconnect/data recovery and ordered shutdown. Do not turn a diagnostic
  CLI into an unreviewed trading process.
- **AI selection:** structured approved candidates and portfolio context in;
  constrained entry/no-entry/direction/priority intents out. A future LLM could
  provide batched synthesis, while CatBoost supplies a forecast. The provider,
  budget, input mapping and actual selection policy are still open. Pin model
  versions and enforce a bounded per-cycle timeout; unavailable AI means no new
  entry. No AI-selected sizing or invented stop/target/exit rule.
- **Deterministic risk:** independently validate eligibility, data freshness,
  intent expiry, cash/margin, whole-share sizing, gross/net/sector exposure,
  concurrent positions, cooldown and session limits. Repeat expiry validation
  immediately before any future submission. Existing sizing math is not a
  complete portfolio-risk implementation.
- **Position management:** deterministic TP/15:00 lifecycle, fill awareness,
  reconciliation and restart recovery must work without the AI. A no-stop
  contract does not eliminate loss risk or make outage handling optional.
  Emergency/portfolio-limit behavior requires an explicit design; do not
  silently reinterpret the user contract as the old price-stop strategy.
- **Journal:** record feature/source versions, opportunities, predictions,
  decisions, rejection reasons, model/calibration IDs, timestamps and stage
  latency, then future risk/position/fill outcomes. SQLite is sufficient. This
  measurement layer must accompany forward shadow work, not follow deployment.
- **Execution:** remains absent. Any future implementation requires separate
  authorization and validation of idempotency, partial fills, reconciliation,
  timeouts and broker truth. Research success cannot enable it automatically.

Fail closed on missing/stale data, schema/hash mismatch, model failure, timeout
or failed reconciliation. Do not substitute Groww, carry forward unknown volume,
invent features or choose a model based on evaluation results to keep a cycle
running. Intended exits must not depend on a future LLM call.

### Interface And Ownership Contract

The following describes target boundary information, not newly implemented
classes or finalized wire schemas. Reuse existing broker-neutral candle,
feature, candidate and portfolio types where their contracts fit; validate any
extension explicitly. Exact schemas must be versioned before integration.

| Boundary | Required Information And Invariant | Owner |
|---|---|---|
| Market observation | Instrument identity, source, exchange time, observed/consumed time and known/unknown volume; distinguish complete candles from partial observations | Broker adapter and market layer |
| Feature state | Candle as-of time, source/feature version and availability; no future or stale context and no invented missing values | Feature engine |
| Model forecast | Ordered 14-column inputs, side, target/horizon, exact weights/calibration ID, raw/calibrated outputs and validity window | Frozen model adapter, still diagnostic only |
| Candidate set | Stable identity, side, reference/as-of time, eligibility reasons and score meaning; a rule score is not a calibrated TP probability | Scanner / future opportunity builder |
| Portfolio snapshot | As-of holdings, pending exposure, cash/margin and realized/unrealized outcomes; reconcile before using it for new risk decisions | Future position/session owner |
| AI intent | Only an allowed candidate/action, decision time, expiry, model/policy version and auditable reason; no executable quantity or new exit rule | Future constrained decision layer |
| Risk result | Allow/deny with reason, deterministic quantity and exposure/cost checks; expired or invalid intent cannot proceed | Future deterministic risk engine |
| Position / execution event | Accepted instruction identity, acknowledgement/fill/rejection/cancellation/reconciliation and cumulative state | Future paper or separately authorized live executor |
| Audit event | Stable correlation IDs, prior/resulting state references, source/model/policy versions, times and failure reasons; no secrets | Future persistent journal |

One minute's context must refer to the same completed point across target and
context instruments; later-arriving data must not enter an earlier decision.
All timestamps are timezone-aware, with exchange-session logic in IST. Prices
and money use deterministic arithmetic; the model adapter performs its explicit
finite numerical conversion in the locked input order. Raw and calibrated
values must remain distinguishable all the way into any comparison journal.

A future LLM should receive structured context and allowed choices, not raw
ticks, credentials or direct broker access. One bounded batched decision over
the candidate set permits comparison across names and existing exposure. Reject
malformed, unknown, expired or out-of-policy output. Pin and journal the model
version and prompt/schema version, enforce per-cycle and per-session budgets,
and make timeout or exhaustion a no-new-entry condition. Provider/model choice,
the actual prompt, numeric budgets and whether an LLM adds value remain open;
no such API integration or unconditional LLM requirement has been implemented.

Proposed AI inputs (October 11; implemented the same day except F&O breadth,
sector moves, global cues, bulk/block deals, macro calendar and the ban list).
Price/volume filters measured movement, not direction, so direction must come
from information the scanner does not use:

| Group | Inputs |
|---|---|
| Candidate | Scanner rank/score, 20-minute volatility vs own normal, relative volume, range vs ATR, day/gap return, VWAP distance, position in day range, strength vs NIFTY and sector, turnover |
| Market | NIFTY/BANKNIFTY and sector moves, India VIX, F&O breadth, pre-open global cues |
| Events | Company news (last 24 hours), results calendar, corporate actions, bulk/block deals, rating changes, macro events (RBI, CPI, Fed) |
| Derivatives (Kite) | Stock futures OI change and basis, F&O ban list |
| Tradability | Distance to circuit limits, spread/depth from quotes |
| Portfolio | Open positions, cash, sector exposure, today's trades and P&L |
orchestrator exists today:

1. **Preflight:** select research/shadow mode explicitly; verify schema/source/
  model fingerprints, current instrument metadata, configured session policy
  and read-only boundary. Restore journal/portfolio state when those components
  exist. A stale or unreconciled prior state cannot authorize new entries.
2. **Warm-up:** establish required prior-session and current-session context.
  Buffer the live/history overlap, avoid double counting, discard partial
  candles and retain unknown inputs. Advance to decision-ready only after
  alignment, freshness and data-quality checks pass.
3. **Decision cycle:** close the agreed candle, update features, apply the
  approved cadence/time window, create eligible opportunities, then record
  forecast/AI intent and deterministic risk result. An empty set or veto is a
  journaled no-trade decision, not a pipeline failure to work around.
4. **Position supervision:** after an accepted paper or future live entry,
  maintain its deterministic lifecycle and exposure independently of model
  availability. TP and the 15:00 normal time exit remain the user contract;
  no stop-loss or 60-minute forced exit is implied by the research label.
5. **Degrade and recover:** stop new entries when continuity, source validity
  or reconciliation is uncertain. Reconnect, recover confirmed missing data,
  rebuild affected feature state and reconcile positions before considering
  resumption. Never infer data continuity solely from a healthy socket.
6. **End the session:** stop new entries according to the separately frozen
  policy, enforce the deterministic cutoff for any simulated/future positions,
  verify final state, persist outcomes/errors and close transports. Do not
  flush a trailing partial market candle as a completed one.

An earlier new-entry cutoff than 15:00 may be required by a future policy, but
its value must be stated rather than borrowed from legacy replay. The current
research model's latest supported decision remains 14:15. A normal exit
contract does not define crash, broker outage or emergency behavior by itself;
those procedures need an explicit reviewed design before a position can exist.

### Failure And Recovery Policy

| Condition | Required Safe Behavior | Current Evidence / Gap |
|---|---|---|
| Missing, partial, stale or misaligned market input | Preserve unavailability and withhold dependent decisions; never synthesize a clean window | Gap/readiness tests exist; broader live acceptance remains bounded |
| Socket reconnect or suspected lost ticks | Veto inference until source history and feature state are reconciled | Supervisor reconnect and diagnostic veto tested; automatic reconciliation not built |
| Live/history discrepancy | Retain both sources and identify differences; apply only a predeclared source/parity policy | Measured on October 6; no acceptable tolerance or substitution policy approved |
| Model hash/schema/input failure | Refuse prediction; do not silently load another model or pad missing fields | Frozen-artifact/inference checks exist |
| Late decision / expired intent | Reject before selection/risk and again before any future submission | Diagnostic 14:15 cutoff tested; future intent/execution layer absent |
| AI timeout or invalid response | No new entries; deterministic position supervision continues if implemented | Target requirement; no LLM decision loop exists |
| Risk rejection / unknown cash, margin or exposure | Refuse the proposed entry with a reason; AI cannot override | Target requirement; complete production risk engine absent |
| Unknown order/position state after restart | Reconcile against broker truth before new actions; do not assume a retry is harmless | Future execution requirement; current adapter exposes no orders |
| Journal unavailable or audit state inconsistent | Fail closed for new decisions rather than run without attributable records | Target requirement; persistent end-to-end journal not built |

For a future live executor, submission timeout does not prove that an order was
rejected. Reconciliation and application-level intent identity must prevent
duplicate action after retry, restart or partial fill. The design cannot assume
a broker provides exactly-once submission. No such behavior is exercised by the
current quote/tick/history validation, and no order endpoint is to be added in
this phase.

## 6. Infrastructure And Retention

Runtime and the new model study are local Python. Azure remains reusable
training infrastructure only: `rg-training-model`, workspace `mlw-tp-research`,
storage `sttpresearchc34920261003`, key vault, monitoring, identity datastores,
code/environment registrations and DS4_v2 cluster `cpu-tp-training` (min 0/max 3).
Cleanup verified zero allocated compute nodes. Persistent services can still bill.

Authorized retirement removed 1,188 old local files (1,626,923,933 bytes), then
753 remote obsolete blobs (384,257,861 bytes). All eight old Azure jobs were
archived; their metadata remains, but their input/output payloads are deleted.
Azure Files shares were preserved and not content-audited. No new Kite dataset
or model was uploaded, no endpoint created, and no infrastructure/RBAC expansion
performed. Future cloud training needs a fresh verified Kite bundle, not an old
job rerun. See [cloud receipt](../data/azure_cleanup_20261005/receipt.json).

Keep active Kite data, raw rejection/NSE evidence, current model/trial/repeat
artifacts, locks, executed source snapshots and audit/cleanup receipts. Data and
models are Git-ignored local artifacts; that is not a backup. Retired aggregates
and source tools are historical evidence, never a reason to resurrect old data.

## 7. Validation And Next Work

Latest executed code gate, October 10: **1,011 tests passed in 81.17 seconds**
after the scanner model study; Ruff lint and 124-file formatting passed. October 7: 1,005 tests.
Tests cover deterministic market/replay behavior,
source/phase boundaries, frozen inference and guarded deletion. The separate
data audit verified all labels; the generic feature sample verified 443,634
numerical comparisons over 8,760 bars, not all features on all 16.8 million bars.

The latest October 7 capture has 13,819 ticks and 128 complete candles/features.
The independent generic reference passed 229,133 numerical and 66,526 unavailable
comparisons across 4,693 warm-up/live bars; the independent TP reference passed
195 numerical comparisons. The existing rule/feasibility path processed 80 target
snapshots in 16 offline cycles with 41 repeatable candidates, not positions.
All historical labels and frozen evaluation predictions were reverified. Exact
vendor parity remains open: 74/120 OHLCV, 91/120 OHLC and 58/120 feature rows match.
The [readiness receipt](../data/kite_validation_20261007/technical_readiness.json)
links all evidence and retains its whole-product `not_live_trade_ready` status.
The later construction investigation found no observed aggregation arithmetic
error, reproduced all 192 candles from both captures and passed 74 focused tests.
Together these establish the built-system baseline in section 1. They do not
convert the original exact-parity or full-product gates into passes. October 9
is a documentation consolidation date, not a fresh full-suite or market run.
**Offline development comes first:** use the retained tick tape, warm-up/daily
history, expected candles/features and synthetic fault tests to implement and
regress changes without waiting for an exchange session. The executable replay
workflow and full schemas are in
[LIVE_API_SAMPLES.md](LIVE_API_SAMPLES.md#offline-development-first); it has been
verified with credentials, browser authentication and sockets blocked.

### Scanner Research Handoff

**Scanner-layer research is the next work, not another general data-validation
pass.** Start from the retained baseline and existing implementations:
[scanner](../src/ai_trader/scanner),
[strategy configuration](../src/ai_trader/strategy/config.py), the distinct
generic and TP feature paths, and the frozen model/report. Code may contain
legacy stop and clip defaults; reading or comparing those rules does not adopt
them as the user's no-stop policy.

1. State the research question and freeze a versioned protocol before selecting
  thresholds or evaluating a new policy: eligibility, side choice, cadence,
  reference/entry convention, ranking, abstention and duplicate handling.
2. Keep rules, raw CatBoost and selected calibrated CatBoost identifiable as
  separate controls on the same eligible opportunities. Do not retrofit the
  63 generic fields into the 14-column model schema or silently score every
  minute when the model was studied on a five-minute grid.
3. Measure coverage/missingness, candidate frequency with explicit denominators,
  false signals, directional/time-of-day behavior and uncertainty. Distinguish
  next-hour touches from TP-or-15:00 outcomes and quantify non-hit loss tails.
4. Report gross and cost-adjusted diagnostics separately, naming assumptions
  about fills, spread, slippage, overlap, sizing and capital. No price stop-loss
  or forced 60-minute exit may be introduced as an inherited default.
5. Keep model weights, calibration, manifests and earlier reports unchanged.
  The studied calendar period has already been inspected: analysis on it is
  retrospective/exploratory, not a newly untouched holdout. Any later confirmatory
  study needs separately reserved or prospective evidence and an explicit policy.

Use saved live versus historical inputs to measure the sensitivity of proposed
scanner behavior where practical. A rule that changes materially at these small
differences needs source-specific evidence; a blanket bit-for-bit candle match
is not a prerequisite for all research. Do not change timestamps, fill missing
data, retrain or switch sources merely to improve an agreement statistic.

Automatic reconciliation, broader session coverage, vendor publication timing
and a dual-timestamp diagnostic remain targeted follow-ups when their behavior
is needed. Before future live inference resumes after a gap, recovery must be
implemented and validated; until then preserve the existing veto. Unbuilt
portfolio/execution layers remain out of the present research-validation scope.

### Acceptance Gates And Open Decisions

The accepted built baseline and remaining expansion criteria are deliberately
separate. The rows below are not a list of prerequisites for beginning scanner
research, and describing them does not authorize implementation or trading.

| Scope | Required Evidence | Current State |
|---|---|---|
| Built historical/live baseline | Reproducibility, independent arithmetic, tested safeguards and bounded real inputs | Accepted to the documented scope; sufficient to proceed with scanner research |
| Scanner policy research | Predeclared comparison, eligibility/side/cadence/threshold policy, coverage and economic diagnostics | Next work; no 80% threshold or max-side policy has been promoted |
| Source-sensitive policy deployment | Explicit input provenance and materiality/revision checks for the chosen scanner | Differences measured; keep existing construction, do not assume exact interchangeability |
| Automatic post-gap resumption | Gap repair, feature rebuild, duplicate handling and controlled-resumption tests | Veto tested; complete history/state reconciliation not built |
| Broader live or forward shadow evaluation | Frozen policy, broader coverage, real-service timing, attributable decisions/outcomes | Requires separately scoped evidence, not supplied by five records at one decision time |
| Portfolio/paper/live execution | Risk, position, accounting, order/retry/reconciliation and emergency tests | Future work outside built-layer validation; orders remain absent |

Further decisions still requiring evidence or user agreement: historical
calendar/CAS eligibility, point-in-time membership/ticks and price basis for cash
simulation; practical publication delay for vendor candles; instrument/session
coverage and promotion criteria; allowable latency and stale-data thresholds;
LLM provider/role and budget if used; deterministic clip size/cash/margin rules;
concurrent and correlated exposure limits; journal retention and approved backup
location; and emergency handling under the no-price-stop contract. Documentation
is complete about these uncertainties by naming them, not by inventing defaults.

The latest completed evidence and artifact locations belong in
[handover.txt](handover.txt). Detailed score/return results belong in
[TP_ONLY_TRAINING.md](TP_ONLY_TRAINING.md), data validation in
[BROKER_DATA_QUALITY.md](BROKER_DATA_QUALITY.md), and offline live-tick operation
in [LIVE_API_SAMPLES.md](LIVE_API_SAMPLES.md). The historical material below is
not needed to infer any current safety, operating-mode or promotion decision.

## 8. Document Map

Read [AGENTS.md](../AGENTS.md), this current architecture, then
[handover.txt](handover.txt) for the fresh-session state and next actions.

| Document | Current Responsibility |
|---|---|
| [README.md](../README.md) | Setup, commands and concise current status |
| [handover.txt](handover.txt) | Authorization, retained paths, environment traps and exact next task |
| [TP_ONLY_TRAINING.md](TP_ONLY_TRAINING.md) | Current Kite model protocol, results, inference and artifact provenance |
| [BROKER_DATA_QUALITY.md](BROKER_DATA_QUALITY.md) | Migration, quarantine, exchange checks and data limitations |
| [FEATURE_VALIDATION.md](FEATURE_VALIDATION.md) | Feature formulas, independent arithmetic and coverage limitations |
| [LIVE_API_SAMPLES.md](LIVE_API_SAMPLES.md) | Current Kite tick/artifact schemas, offline replay runbook and development coverage; retired Groww captures are separate |
| [strategies/README.md](strategies/README.md) | Scanner-research starting point; archived literature and retired stop-based studies |
| [strategies/TP_PROBABILITY.md](strategies/TP_PROBABILITY.md) | Active TP-only event/output contract; historical signed TP/SL contract is archived |
| [backtests/README.md](../backtests/README.md) | Retired execution experiments, not a current Kite profit report |

Update the owning current section when behavior or state changes. Do not append
another contradictory "latest" block and leave active guidance ambiguous.
Keep four categories explicit: implemented behavior, intended requirements,
unresolved decisions and historical evidence. AGENTS.md owns binding safety;
the current architecture owns intent; dated manifests/reports and their executed
source snapshots establish observed results. A later fix must not retrospectively
change an older run's status. Record the date and evidence path when updating a
measurement, and qualify old test/resource counts rather than implying continuous
verification. The document map and current handover should remain sufficient to
resume work without reading the archive.

<details>
<summary>Archived original design and pre-Kite implementation history</summary>

The following material is retained for technical rationale and chronology.
References to Groww, two selectable broker fees, live stops, old studies and
build ordering describe earlier decisions. They do not override sections 1-8
above or the user's research-only, no-stop TP/15:00 contract. Historical paths
may have been intentionally deleted; do not use these sections as a runbook.

## 1. Goal and operating frequency

Build a robust AI-assisted intraday trading research and execution system for
highly liquid Indian markets.

Runtime broker: Zerodha Kite only, behind the read-only broker abstraction.
Historical access and authenticated profile/LTP/quote/stream access are verified.
The stream check received after-hours snapshots; market-hours parity is pending.

The system must remain testable, deterministic where possible,
broker-independent above the adapter layer, and safe by construction.

### 1.1 Operating frequency

This is a **medium-frequency** system. Holding periods are minutes to hours,
decisions are made on one-minute bar closes, and positions are squared off
before the session ends.

That choice sets everything below it, so it is stated first:

| Property | Value | Consequence |
|---|---|---|
| Decision primitive | one-minute candle | entries are minute-quantized |
| Decision cadence | on bar close | ≤ 375 decision cycles per session |
| Tolerable decision latency | seconds | an LLM fits inside the loop |
| Holding period | minutes to hours | overnight risk is not carried |

The LLM is viable here **only** because of this frequency band. A round trip is
typically 2–8 seconds, with a tail into the tens of seconds on retry. That fits
inside a one-minute bar with room to spare, and would not fit at all at
seconds-scale. If the target frequency ever changes, the LLM leaves the hot
path — it does not get optimized.

### 1.2 Where the AI sits

The AI is the **decision-maker within a constrained action space**.

It holds the trade / no-trade call, the direction, and the selection among
candidates. Deterministic code holds the action space it chooses from, and the
size of whatever it chooses.

Both halves of that sentence are load-bearing. "Advisory" undersells it — the
AI genuinely decides whether a trade happens. "The brain" oversells its
autonomy — it cannot invent a candidate, cannot size a position, cannot
override a veto, and **must never communicate directly with the broker
execution API**.

### 1.3 The questions the system exists to answer

Two questions, easy to conflate and important not to:

1. **Do the scanner's hypotheses have edge at all?** Answered by historical
   replay, deterministically, offline. **Asked once and answered no**: at the
   default thresholds, over 61 sessions of 15 cached NSE symbols, the rule set
   loses at every stop multiple swept and its gross hit rate sits near 47% with
   friction removed. That is a result about those thresholds over that window,
   not a verdict on the hypotheses in general — but it is the honest current
   answer, and `backtests/README.md` holds the run.
2. **Does the AI's selection improve on simply taking every candidate?**
   Answered only by forward shadow trading. Replay cannot answer it — see
   section 7.1.

The deterministic baseline is therefore a **control**, not a gate the AI must
clear before being built. But if the AI cannot beat it in shadow mode, that is
a finding, and it should be acted on rather than explained away. A control that
currently loses money is still a control; what it is not is a set of entries
worth handing to a ranker untouched, which is why section 11 puts threshold
tuning ahead of the AI layer in practice even though nothing in the
architecture forces that order.

### 1.4 The trading objective, and why it is arithmetic rather than a constant

The intended behaviour is **many small round trips per session on liquid names,
each exited as soon as it is a little ahead of what the round trip cost**. Both
directions are in scope: a short sells then buys where a long buys then sells,
and since either way there is exactly one buy and exactly one sell, the two pay
the same charges. Holding periods stay inside the band in section 1.1.

The tempting way to write that down is a target percentage — "exit at +0.2%".
Section 7.2 forbids it, and the cost model in `costs/` shows why concretely
rather than as a principle. At Groww's rates a round trip costs **0.2715% of a
₹20,000 clip and 0.0827% of a ₹1,00,000 one**, because brokerage is capped per
leg and the cap stops binding as size grows. So a 0.2% gross capture is a *net
loss* on the smaller clip and a *net gain* on the larger one. No single
percentage describes both, and one written into the code would be silently
wrong at every size except the one it was chosen for — and, since two broker
schedules are modelled, wrong at a different size for each of them.

What is stable is the shape:

```
required gross move  =  round-trip cost at this size  +  the margin asked for
```

Both terms are inputs. The cost term comes from a published fee schedule. The
margin term is a strategy parameter, and it is now **stated rather than
assumed**, alongside the size it is measured against:

| Input | Value | Constant |
|---|---|---|
| Clip size | ₹1,00,000, the smallest whole number of shares worth at least that | `FIXED_CLIP_NOTIONAL` |
| Gross target | **0.2% above the buy price** — a gross move, not a net one | `STATED_GROSS_TARGET` |

Those two fix the third. At ₹1,00,000 the round trip costs ₹82.68, or 0.0827%,
so a 0.2% gross exit **keeps 0.1173% — about ₹117.32 a trade, after costs**.
That figure, repeated across many round trips in a session, is the entire
economic thesis of the system. It is also the same at either broker: both cap
brokerage at ₹20 a leg, and a lakh is past both caps. Section 4.4 carries the
arithmetic, the tick-alignment that makes the exit price reachable, and the two
readings of "0.2%" that it disambiguates.

Stating a number is not the same as measuring it. The *hypothesis* under test is
that a margin in that region can be captured often enough, on enough names, to
pay for the session — and section 4.5's replay is what turns that into evidence,
including evidence that the number should move. What the code refuses to do is
*invent* one. A stated parameter lives in one place, can be changed there, and
can be re-measured; a constant chosen for convenience is indistinguishable in
the source from a finding. Section 7.2 is that principle, and `costs/` is where
the two stated values live.


## 2. Target architecture

### 2.1 Forward path

```
                    EXTERNAL MARKET
                           │
                    Groww / Future Kite
                           │
              ┌────────────┴────────────┐
              │                         │
       Historical Data             Live Market Feed
              │                         │
              │                    MarketTick
              │                         │
              └────────────┬────────────┘
                           │
                      Market Layer
                           │
                 CandleBuilder / Volume
                           │
                     Completed Candle
                           │
                       MarketState
                           │
                       FeatureEngine
                           │
                    FeatureSnapshot
                           │
    Cost Model ────────────┤
      costs/               │
   Groww | Zerodha (Kite)  │
                Deterministic Scanner
                           │
                  Candidate / No Trade
                           │
               ┌───────────┴───────────┐
               │                       │
       Historical Replay           Live / Shadow
       & Strategy Research             │
               │                       │
               └───────────┬───────────┘
                           │
                    AI Decision Layer
                           │
                 BUY / SELL / HOLD /
                    CLOSE intent
                           │
                 Deterministic Risk
                           │
             sizing / exposure / limits /
             daily loss / liquidity /
             cooldown / kill switch
                           │
                    Position Manager
                           │
                    Execution Engine
                           │
                 Broker Abstraction
                           │
                  Groww / Future Kite
                           │
                        MARKET
```

Three structural details carry most of this diagram's weight.

**The fork after the Scanner.** Historical replay and live/shadow trading
consume the *same* candidate stream from the *same* deterministic code. Replay
is not a separate offline tool bolted on later; it is a first-class consumer
sitting at the same level as live trading.

**The AI is sandwiched.** See section 3.

**The cost model enters sideways, not in the flow.** `costs/` is not a stage —
nothing passes through it. It is a pure fee schedule the scanner consults to ask
whether a move worth capturing is even available at the configured position
size (section 4.4), and the same schedule replay will score net expectancy with
and the risk engine will size against. Drawing it in the spine would suggest it
transforms the data; it does not, and keeping it stdlib-only and dependency-free
is what lets all three layers share one answer. Two schedules are modelled —
Groww and Zerodha/Kite — and a schedule is an *argument* to the arithmetic
rather than an import inside it, which is what rule 10 of `AGENTS.md` asks for
and what lets replay hold a strategy fixed while varying who executes it.

### 2.2 Feedback path

The diagram above is the **data** path. It is not the whole system. A trading
system is a control loop, and three layers above the position manager need to
read state that the position manager owns.

This is not stylistic. It follows from commitments made elsewhere in this
document:

- The AI output schema includes **CLOSE**. Emitting CLOSE requires knowing that
  a position exists.
- Risk enforces **max open positions, max exposure, max trades per symbol,
  cooldown**. All four require live book state.
- The scanner should not emit a candidate for a name already at its position
  limit. Such a candidate can only ever be rejected downstream, so producing it
  burns a slot in the candidate budget and part of an LLM call for nothing.

To keep this implementable rather than a tangle of arrows, the feedback is a
**named artifact**, on equal footing with `Candle`, `FeatureSnapshot` and
`Candidate`:

```
   PortfolioState        owned and published by the Position Manager
        │
        ├──► Scanner     names at limit, per-symbol trade counts, cooldown
        │                state, stale-feed flags  → suppress candidates
        │
        ├──► AI Layer    open positions with entry and unrealized P&L, recent
        │                decisions on these names, session P&L so far
        │
        └──► Risk        gross and net exposure, open count, realized and
                         unrealized daily P&L, kill-switch state

   Execution ──fills──► Position Manager ──rebuilds──► PortfolioState
```

`PortfolioState` must be an **immutable snapshot taken once per decision cycle**
and passed down, not a live mutable object read at different instants by
different layers. Otherwise the scanner, the AI and risk can each see a
different book within a single cycle, and the resulting bugs are
timing-dependent and effectively unreproducible.

In live trading `PortfolioState` is built from real fills, in shadow mode from
simulated fills, in replay from simulated fills over historical candles. Same
object, same consumers, three sources — which is exactly what makes the three
modes comparable.

### 2.3 Layer status — current versus final

Every layer in section 2.1's pipeline appears below whether or not it exists.
The **Responsibility** column points at the section that defines what the layer
must do — written for the unbuilt ones too, so there is something to build
against. The **Maturity** column uses the ladder in section 11: *implemented* is
the weakest claim on it and *production-ready* the strongest, and the distance
between them is the point of having the ladder at all.

| Layer | Module | Responsibility | Maturity | Qualifier |
|---|---|---|---|---|
| Broker adapter | `broker/` | [4.1](#41-broker-adapters--broker--built-read-only) | Kite history/profile/quotes/stream verified read-only | After-hours snapshots only; market-hours parity pending; no orders |
| Market layer | `market/` | [4.2](#42-market-layer--market--built) | validated live | supervisor's failure path proven live, its recovery path not yet |
| Feature engine | `features/` | [4.3](#43-feature-engine--features--built) | validated live / offline | All 63 features independently checked on six instruments; retained formula/readiness tests |
| Transaction costs + sizing | `costs/` | [4.4](#44-deterministic-scanner--scanner--built) | tested offline | two schedules modelled; neither reconciled against a real contract note; spread not modelled |
| Deterministic scanner | `scanner/` | [4.4](#44-deterministic-scanner--scanner--built) | tested offline | thresholds are still convention; replay has now measured them, and the answer was negative |
| Strategy configuration | `strategy/` | [4.4](#44-deterministic-scanner--scanner--built) | tested offline | one object both replay and live build their scanner from |
| Candle cache | `history/` | [4.5](#45-replay--research-engine--replay--built) | tested offline | replay's tape; `--offline` re-runs read only this |
| Diagnostic CLIs | `cli/` | [4.11](#411-diagnostic-clis--cli--built) | validated live | eight checks, one per layer up to the scanner, plus `backtest` |
| Replay / research | `replay/` | [4.5](#45-replay--research-engine--replay--built) | tested offline | never run against a live session, because it is not meant to be |
| AI decision layer | — | [4.6](#46-ai-decision-layer--not-built) | not built | replay's first result is in; see section 1.3 |
| Deterministic risk | — | [4.7](#47-deterministic-risk-engine--not-built) | not built | AGENTS.md rules 7 and 8 live here |
| Position manager | — | [4.8](#48-position-manager--not-built) | not built | owns `PortfolioState` construction |
| Execution engine | — | [4.9](#49-execution-engine--deferred) | deferred | by AGENTS.md, not by sequencing |
| Journal / observability | — | [4.10](#410-journal--observability--not-built) | not built | AGENTS.md rule 9 |

Read down the Maturity column and the current state of the project is the
answer: everything from the broker through the scanner to the replay engine
exists, and nothing downstream of replay does. The AI layer is the next thing to
build, and it is the first layer whose precondition is a *result* rather than
another layer — section 11 says what that result was and why it does not block
the build. Section 11 also says the same status in build order and explains why
that order is not negotiable.

**What the replay engine has already reported.** Over 61 sessions of 15 cached
NSE symbols, the default rule set and the modelled fill produce a loss at every
stop multiple swept, and removing friction entirely moves the gross hit rate to
roughly 47% — a coin flip. That is a measurement about the *thresholds in
section 4.4*, which have always been labelled conventions, not about the
pipeline that computed them; section 7.2 is the standing warning against reading
it as more. The entries need tuning before the AI layer has anything worth
ranking, and `backtests/README.md` records the sweep.


## 3. Safety boundary

The most important architectural rule:

```
Market Data
    ↓
Deterministic Scanner
    ↓
GPT
    ↓
Deterministic Risk
    ↓
Execution
```

GPT is deliberately placed **between two deterministic layers**.

The scanner decides what it is allowed to see. Risk decides what its output is
allowed to become. It cannot create arbitrary broker actions and it cannot
override risk. It can rank, reject, and explain — nothing else.

This is AGENTS.md rules 7 and 8 expressed structurally rather than by
convention, and it is the one boundary that must never be collapsed for
convenience. The feedback edges in section 2.2 add *inputs* to the AI; they add
no outputs, and must never be used to give it one.


## 4. Layer responsibilities

### 4.1 Broker adapters — `broker/` — BUILT (read-only)

Broker-specific authentication, historical data, quotes, streaming data, and
eventually order execution. **Nothing above this layer should depend directly
on Groww-specific data structures.**

`ReadOnlyBroker` is a `Protocol`, not a base class, so a Kite implementation
never imports Groww code. Normalized types are frozen slots dataclasses:

```
Instrument  LastTradedPrice  MarketTick  MarketQuote  OHLCVCandle
BrokerProfile  CandleInterval(StrEnum)  ReadOnlyBroker(Protocol)
```

"Read-only" is a property of the abstraction itself — there is no
order-placing method to call. That is the structural half of AGENTS.md rules 1
and 2; the other half is that no such method gets added until live trading is
explicitly enabled.

`KiteHistoricalBroker` implements only historical capabilities: exact cash-equity
instrument resolution, normalized `OHLCVCandle` history and daily closes. It does
not implement the full `ReadOnlyBroker` protocol or replace Groww in runtime
consumers. The optional official SDK stays behind this adapter. Authentication
uses a loopback-only, state-checked browser callback and an in-memory token;
no order or portfolio endpoint is called. RELIANCE minute history and browser
authentication were verified live on October 5, 2026, followed by 33 months of
minute history and 683 daily candles. Historical calls are paced below the
vendor's three-per-second limit. The separate daily-close convenience method
has offline coverage; daily OHLCV was retrieved through the historical method.

`cli/compare_broker_history.py` writes separate, source-hashed provider snapshots,
coverage and OHLCV diagnostics without refreshing the Groww cache. Its offline
mode verifies saved snapshot hashes and writes a new comparison without reading
credentials or calling a broker. The continuous-session close is an explicit
CLI argument, not inferred from missing bars: September RELIANCE uses 15:15 under
NSE closing-auction rules, while non-CAS securities retain 15:30. The old global
runtime session defaults and frozen research are unchanged; a full migration
needs instrument- and date-aware sessions. The user's research exit stays 15:00.

The September comparison found complete Kite continuous-session coverage, 435
minutes absent from the older Groww cache, and material open-price differences
despite 99.2140% exact close agreement. Neither feed is proven exchange ground
truth. See the README's [results and onboarding commands](../README.md#kite-historical-data)
before mixing provider datasets or changing model/runtime inputs.

The full January 2024-September 2026 audit now recommends Kite for subsequent
historical research: eight missing minutes versus 858 in Groww, and four severe
Groww price errors reproduced by fresh API calls and contradicted by official
NSE daily bounds. Kite daily OHLCV was independently checked on 49 dates after
reversing RELIANCE's 2024 bonus adjustment. This is an instrument-scoped quality
recommendation, not a full runtime migration or proof of every intraday price.
`scripts/audit_broker_history.py` retains separate adjustment/session-aware
analysis without rewriting source tapes. See [BROKER_DATA_QUALITY.md](BROKER_DATA_QUALITY.md).
Monthly download checkpoints preserve the original Groww fingerprint and provider
snapshots; special-session coverage is refined in the audit, not the runtime clock.

Broker quirks are absorbed here rather than leaked upward:

- Historical candle volumes are **per-candle**; the live LTP feed reports
  **cumulative** day volume. `market/volume.py` reconciles them.
- The historical endpoint is **end-inclusive**.
- Sessions have genuine gaps — a minute with no prints yields no candle. Never
  assume contiguity anywhere above this layer.

When the order-placing counterpart is eventually added, it must support
**resting stop orders** (SL / SL-M) held at the exchange, for the reason given
in section 4.8. Confirm Groww's support for those order types before committing
to that exit design.

### 4.2 Market layer — `market/` — BUILT

Convert raw broker data into canonical broker-independent objects:

```
MarketTick → Candle → MarketState
```

Handle timestamps, volume, late ticks, duplicates, gaps and session boundaries
deterministically.

**`CandleBuilder`** (`market/candles.py`) aggregates a live tick stream into
independent one-minute candles. It **discards the first minute it observes for
each instrument**: a stream is joined at an arbitrary moment, so that minute is
a fragment whose open is whichever tick happened to arrive first and whose
high/low span only the watched portion — downstream it would be
indistinguishable from a real candle, which makes it worse than no candle at
all. It is also the only minute that can never carry volume, since cumulative
differencing needs a prior reading. One rule fixes both defects.

The discarded minute is still recorded as finalized. Omitting that would let a
late tick reopen it as a second, smaller fragment, costing a real candle.

This applies **per instrument**, and to the live path only. Historical backfill
bypasses the builder entirely.

**`MarketState`** (`market/state.py`) holds normalized rolling history per
instrument, fed from both paths:

```python
record_tick(tick) -> Candle | None      # live path, through CandleBuilder
record_candle(candle) -> bool
backfill(instrument, candles) -> int    # historical path, direct
flush() -> tuple[Candle, ...]
snapshot(instrument) -> InstrumentState | None
snapshots() -> tuple[InstrumentState, ...]
instruments() -> tuple[Instrument, ...]
latest_candle / late_tick_count / duplicate_candle_count    # properties
```

An optional `on_candle` callback fires **outside** the internal lock, on the
broker's feed thread. That callback is the intended integration point for the
feature engine — the two layers are deliberately not coupled directly.

The position manager also subscribes below this layer, to the raw tick stream.
See section 4.8.

### 4.3 Feature engine — `features/` — BUILT

Convert completed candles into the 63 quantitative features a scanner rule may
read: returns, trend, momentum, volatility, trend strength, bands, VWAP, volume
and the session frame.

Feature calculation is **deterministic and incremental**. Historical warm-up,
historical replay and live trading must use the same calculation path.

```python
update(candle) -> FeatureSnapshot | None
warm_up(candles) -> int
snapshot(instrument) -> FeatureSnapshot | None
snapshots() -> tuple[FeatureSnapshot, ...]
is_ready(instrument) -> bool
instruments() -> tuple[Instrument, ...]
duplicate_candle_count / out_of_order_candle_count          # properties
```

Computed: 63 features. [`FEATURE_VALIDATION.md`](FEATURE_VALIDATION.md) separates
the historical 47-feature comparison from the current retained all-feature audit:

| Family | Features |
|---|---|
| Returns | `return_1`, `return_5`, `return_15` |
| Intraday clock windows | `intraday_return_30m`, `intraday_return_60m`, `realized_volatility_20m`, `realized_volatility_60m` |
| Previous session | `previous_session_return`, `previous_session_range_pct`, `gap_from_previous_close`, `distance_from_previous_close`, `distance_from_previous_high`, `distance_from_previous_low` |
| Multi-session context | `prior_5_session_return`, `prior_20_session_return`, `prior_5_session_range_pct`, `prior_20_session_range_pct`, `position_in_prior_5_session_range` |
| Trend | `ema9`, `ema21`, `ema50`, `ema9_slope_5`, `ema21_slope_5` |
| Momentum | `rsi14`, `macd`, `macd_signal`, `macd_histogram`, `macd_histogram_change` |
| Volatility | `true_range`, `atr14`, `atr_pct`, `candle_range_pct` |
| Trend strength | `plus_di14`, `minus_di14`, `adx14` |
| Rolling window | `rolling_high_20`, `rolling_low_20`, `distance_from_high_20`, `distance_from_low_20` |
| Bands | `sma20`, `bollinger_upper_20`, `bollinger_lower_20`, `bollinger_bandwidth_20`, `bollinger_percent_b_20` |
| VWAP | `vwap`, `price_vs_vwap`, `vwap_deviation`, `price_vs_vwap_sigma` |
| Volume | `volume_ratio_20`, `obv`, `volume_ratio_same_minute_5` |
| Session frame | `session_open`, `session_high`, `session_low`, `session_range_pct`, `position_in_session_range`, `distance_from_session_open`, `minutes_since_session_open`, `session_volume` |
| Opening range | `opening_range_high`, `opening_range_low`, `distance_from_opening_range_high`, `distance_from_opening_range_low` |

Every rule in section 4.4 reads from this list and nothing else — which is why
the list is reproduced here in full rather than summarized. A rule referencing a
feature that is not in it does not exist, and one referencing a VWAP, volume or
session-frame feature has inherited a readiness obligation the next property
describes.

Four properties matter more than the indicator list:

**Warm-up and live are the same code path.** `warm_up` is literally a loop over
`update`. An engine fed 60 candles in one batch and an engine fed 30 then 30
produce equal snapshots — asserted by test. This is what makes replay a valid
proxy for live behaviour below the scanner; without it, a backtest and a live
run could diverge for purely structural reasons.

**Readiness is mechanically checkable.** Every flag on `FeatureReadiness`
equals `getattr(snapshot, name) is not None`. `core_ready` names one feature per
family — `return_15`, `ema50`, `rsi14`, `macd_histogram_change`, `atr14`,
`adx14`, `rolling_high_20`, `bollinger_upper_20` — rather than all 63.
This existing scanner contract is unchanged. Longer clock-window and prior-session
features require their own readiness checks and do not delay core readiness.

It excludes **two whole classes**, for two different reasons. Volume-dependent
features (VWAP and its derivatives, `volume_ratio_20`, `obv`) are excluded
because they can be unavailable for an entire session while trend, momentum and
volatility stay perfectly valid. Session-frame features are excluded for a
sharper reason: an engine started mid-session withholds its session aggregates
for the rest of that day by design, so including them would wedge `core_ready`
false until the next open no matter how much price history accumulated.

That exclusion has a consequence worth stating as a contract rather than a
footnote, because it was observed live on 2026-09-21: across the
historical-to-live seam, `core_ready` stayed `true` while `vwap`,
`price_vs_vwap` and `volume_ratio_20` all went `None` and stayed `None`. This
is correct behaviour — price and momentum really were ready — but it means
**`core_ready` is not an authorization to read every field.** Any consumer
touching a volume-derived feature must check that field's own flag. A scanner
rule that gates on `core_ready` and then dereferences `vwap` will silently
evaluate against nothing for the entire live session.

**Ordering is guarded before any state mutates.** A duplicate or out-of-order
candle increments a counter and returns `None`, touching nothing. A retry or a
replayed message cannot corrupt an EMA.

**Memory per instrument is constant** — 15 closes, two slope deques of 6, 20
highs, 20 lows, 20 volumes, a 20-wide dispersion window for the bands, and some
scalars. The additions retain 60 intraday closes, 60 squared log returns, up to
375 current-minute volumes and 20 fixed session records with 375 volume slots
each. EMA seed buffers are discarded once seeded. ADX, DMI and OBV remain
Wilder-smoothed or cumulative.

The earlier throughput estimate predates the clock-window and session-history
additions. Re-measure the complete feature set before claiming a specific
latency at the intended scale of roughly 200 names; bounded storage alone does
not prove a throughput budget.

### 4.4 Deterministic scanner — `scanner/` — BUILT

Filter the market and generate a small number of candidate setups.

Inputs are `FeatureSnapshot` and `PortfolioState` — **not** raw broker data and
**not** raw ticks. It must respect readiness flags rather than treating `None`
as zero.

It should encode measurable trading hypotheses such as: trend, momentum,
breakout, mean reversion, VWAP relationship, volatility, volume confirmation,
liquidity, and time-of-day constraints.

**What exists.** `scanner/` holds five rules — `trend_continuation`,
`range_breakout`, `band_mean_reversion`, `vwap_reversion`,
`opening_range_breakout` — behind a `Rule` protocol, plus `Scanner`, the
`PortfolioState` contract below, an optional cost-derived `FeasibilityPolicy`,
and `MarketContext` as an empty reserved seam. Two of the rules are deliberately
contradictory: trend continuation and mean reversion are gated on opposite ADX
regimes, and which of them earns its place is a question for section 4.5 rather
than one to settle by picking the more convincing-sounding hypothesis now.

**Optional signed target-time model.** `StrategyConfig.target_score` can select
`scanner/opportunity.py` instead of the five-rule merge. Eight bounded feature
families produce `2 * sigmoid(raw) - 1`: positive LONG, negative SHORT, ranked by
absolute magnitude. Offline-fitted coefficients are immutable at inference;
core readiness, portfolio suppression and feasibility still apply. The model
shares the strategy's gross target and intraday cutoff. It estimates a signed,
speed-discounted first-target outcome, not a win probability or permission to
trade. Missing core inputs and decisions at/after cutoff yield no candidate.

The optional model's `score_threshold` requires `abs(score) >= threshold` before
emitting a candidate, with direction from the sign. Zero always abstains. The
threshold is an action filter and does not change the raw score, sizing or risk
rules. The September 1-4 threshold study found higher tuning accuracy at 0.03,
but neither that choice nor the speed-objective choice established profitability.

The later [TP-precision study](../backtests/reliance_sep2026/tp_precision_v1/README.md)
selects by actual TP exits rather than utility or net wins, permits sparse
selections, and varies thresholds and finite TP/SL levels. Its executable labels
reuse `ReplayPortfolio` and selected trades match `ReplayEngine` exactly. The
study also tests TP-specific classifiers with prior-session context, but those
features and classifiers remain in research tooling, not the live scanner.

The [broader probability study](../backtests/tp_probability_2026q3/README.md)
uses a different explicit contract: direction times estimated TP probability,
with separate calibration for LONG and SHORT and a separate entry threshold.
It freezes fit/calibration/selection choices before later-session evaluation.
These research policies are not the production signed-utility profile and are
not automatically promoted. [Strategy notes](strategies/README.md) record the
evidence and the [probability contract](strategies/TP_PROBABILITY.md).

The [month-split follow-up](../backtests/tp_probability_month_split/README.md)
keeps base-coefficient fitting entirely in July and all calibration/selection
in August, then evaluates September 1-28. It reports initial stop distances in
percent, per-side ranking/calibration, TP timing and training-only feature
contribution scales. End-of-session TP probability is not a speed forecast;
explicit horizon probabilities were not part of that experiment. None of the six tested
policies was profitable, and no research model was promoted. September is
previously examined data, not a fresh prospective test.

The [symmetric horizon study](../backtests/tp_horizon_symmetric_v1/README.md)
now fits a joint distribution over TP time bins, SL time bins and timeout.
Its cumulative TP probabilities directly estimate TP before SL within the
specified hours from executed entry, with equal tick-rounded initial TP/SL
distances. Research-only sizing and exit-policy adapters reuse `ReplayPortfolio`;
an explicit `ExitReason.HORIZON` distinguishes expiry from session square-off.
The ordinary engine and strategy defaults are unchanged. Previous five-session
context is causal and compared against intraday-only and time/volatility inputs.
Logistic, tree and neural candidates are fitted offline and saved with hashes;
none is wired into the production scanner. The selected result predicts a rare
event reasonably on this retrospective sample, not a high-confidence trade;
all 16 combined policy alternatives lose after modeled costs.

**Latest TP-only neural research.** The separate
[training protocol](TP_ONLY_TRAINING.md) now fixes a 0.20% directional price-touch
target within 60 minutes of the last completed close, with no SL competing event.
`features/tp_only.py` owns 13 compact causal inputs; direction is the 14th tensor
column and stock identity is metadata only. Vendor daily closes have their own
broker reader, while prior volatility requires complete 09:15-15:15 windows.
The old 375-minute FeatureSnapshot history contract is unchanged. A hash-pinned,
50-stock dataset and checkpointed FT-Transformer, ResNet, TabM and CatBoost
research trainers are implemented. The six comparison jobs completed using only
training/validation uploads; CatBoost won validation. A bounded local refinement
then selected a 607-tree depth-four model. Its weights were locked before separate
chronological calibration selection, followed by a locked later-period evaluation.
Calibration improved the March selection score but worsened April-September loss
versus the raw comparator, so neither output is promoted to live trading.
Research-only, hash-verified frozen inference and full probability diagnostics
are retained. Azure resources and limits are documented in that protocol; the
compute scaled to zero after the original batch. This does not implement the
production AI decision layer; the default scanner remains unchanged.

A subsequent fixed, retrospective comparison reused the real rule scanner and
frozen CatBoost forecasts on common eligible five-minute opportunities. It found
better TP-touch ranking for CatBoost. Its separate 2-ATR/15:15 execution simulation
does not represent the user's October 5 clarification: **only TP or 15:00 exits,
no price stop, and 60 minutes as an accuracy window rather than an exit deadline**.
A separate frozen-call analysis measures that clarified contract: 68.78% literal
hourly TP touches and 77.86% by 15:00 for raw-ranked calls, with directional returns
at 15:00 for non-hits. Late calls' hourly windows past square-off are reported
separately. The training protocol holds the counts and limitations. No model is
promoted, and the runtime scanner, legacy strategy defaults and old studies remain
unchanged.

The [September 1-4 study](../backtests/reliance_sep2026/signed_score_v1/README.md)
audits all 47 features and provides an opt-in fitted profile. Forward-day error
improved only 0.40% over predicting zero, with negative rank correlation on two
of three test days. It remains unvalidated; default rules and their scores are
unchanged. The rule descriptions below concern that original baseline.

**The rules, stated.** Each one is a hypothesis written down so replay can
refute it. All five are symmetric — every one can fire `SHORT` as readily as
`LONG`, which section 1.4 requires. Every threshold named below is a
conventional level, not a measurement; section 7.2 applies to all of them.

| Rule | Needs | Prefers | Fires on |
|---|---|---|---|
| `trend_continuation` | `ema9`, `ema21`, `ema50`, `adx14`, `macd_histogram`, `rsi14`, `atr14` | — | a stacked EMA ladder with MACD pushing the same way, while ADX says it is trending |
| `range_breakout` | `rolling_high_20`, `rolling_low_20`, `atr14` | `volume_ratio_20` | a close within a fraction of an ATR of the 20-candle extreme |
| `band_mean_reversion` | `bollinger_percent_b_20`, `rsi14`, `adx14` | — | price outside the band with RSI agreeing, while ADX says it is *not* trending |
| `vwap_reversion` | `vwap`, `price_vs_vwap_sigma` | — | price stretched two or more sigma from session VWAP |
| `opening_range_breakout` | `opening_range_high`, `opening_range_low`, `atr14` | — | a close beyond the range the first fifteen minutes established |

- **`trend_continuation`** declines unless `adx14 ≥ 25`. It then reads the EMA
  ladder: `ema9 > ema21 > ema50` with a positive MACD histogram is `LONG`,
  the mirror image is `SHORT`, and anything else declines. An already-exhausted
  RSI **vetoes** the signal — at or above 80 for a long, at or below 20 for a
  short — because joining a trend at the point it has gone parabolic is this
  rule's characteristic way of losing money. The score is the mean of two
  ramps: ADX mapped over 25→50, and `|macd_histogram| / atr14` mapped over
  0→0.5. The histogram is divided by ATR so the same score means the same thing
  on a ₹200 name and a ₹3,000 one; a raw histogram is denominated in price.

- **`range_breakout`** tests *closeness* to the window edge rather than a break
  of it, for the reason in the first finding below. It takes a tolerance of
  `0.1 × atr14` and fires `LONG` when the close is within it of the high and
  nearer the high than the low, `SHORT` when it is within it of the low. The
  score is `1 − ramp(gap, 0, tolerance)` — tighter to the edge scores higher.
  When `volume_ratio_20` is available it is ramped over 1.5→3 and averaged in;
  when it is not, the tightness stands alone rather than being averaged against
  a zero.

- **`band_mean_reversion`** declines unless `adx14 < 20`. That ceiling is the
  whole point of the rule: price outside the band during a strong trend is the
  trend working, not an excess to fade. It fires `LONG` on `%B ≤ 0` with
  `rsi14 ≤ 30` and `SHORT` on `%B ≥ 1` with `rsi14 ≥ 70`. The score is the mean
  of how far outside the band price sits (ramped over 0→0.5 of band width) and
  how far past the RSI threshold it is (ramped over a 15-point span).

- **`vwap_reversion`** fires `LONG` at `−2σ` or beyond and `SHORT` at `+2σ` or
  beyond, scoring the magnitude ramped over 2→4. Both inputs are
  volume-derived, which makes this the rule the third finding below is about.

- **`opening_range_breakout`** fires `LONG` above `opening_range_high` and
  `SHORT` below `opening_range_low`, scoring the extension divided by `atr14`
  and ramped over 0→1. It contains **no clock check**, and does not need one:
  the feature is withheld until the opening range has closed and frozen once it
  has, so availability *is* the time-of-day gate — and it is the accurate one,
  since it also withholds itself on a session the engine joined late, where an
  "opening range" computed from an 11:00 start would be fiction.

**The feasibility screen — can this name pay for its own round trip?** Every
rule above answers "is this name set up to move?". None answers "is the move
big enough to be worth the fees?", and at a one-minute horizon the second
question disqualifies more names than the first (section 1.4).
`FeasibilityPolicy` asks it, using the `costs/` schedule:

```
quantity       = floor(target_notional ÷ price)
required_gross = round_trip_fraction(price × quantity) + net_margin_fraction
```

The caller states all three strategy parameters explicitly — `target_notional`
(the clip one leg aims at), `net_margin_fraction` (what it wants left over) and
`max_atr_multiple` — for the same reason section 7.2 demands the universe be
stated: a result is meaningless unless the conditions that produced it were
recorded beside it. The screen then rejects a name when

```
required_gross_fraction > max_atr_multiple × atr_pct
```

reading "the required move is implausible if it is more than this many
one-minute ATRs away". Note the unit: `atr_pct` is a fourteen-period average
true range over **one-minute** candles expressed as a fraction of close, so a
multiple of three is a small intrabar move and a multiple of thirty is most of a
session. It is multiplied, never divided — a zero ATR is a real reading on a
stock that has not moved, and dividing by it would raise on exactly the names
the screen exists to reject.

**The hurdle is priced per name, not once per cycle.** A fixed clip buys a
whole number of shares, so the notional that fills is the clip rounded *up* to
the next whole share — never below it, and on a dear name substantially above.
Every cost here is a fraction *of what filled*, and brokerage is capped per leg,
so a larger fill pays a smaller fraction: a share quoted at ₹1,40,000 turns over
forty per cent more than a one-lakh clip and clears a materially lower hurdle
than the clip figure says. Screening it against the full-clip figure would
therefore be too strict rather than too loose — rejecting names the buying model
can in fact afford. So the screen sizes at the snapshot's close, through the
same `SizingPolicy` the decision will use rather than a second implementation
free to disagree with it. The policy's own `required_gross_fraction` remains the
full-clip figure, which is a ceiling no real quote exceeds and the one number
that describes the screen itself.

Four design commitments hold this in place:

- **It is a filter, never a term in the score.** Folding headroom into the
  ranking would mean deciding how many points a spare basis point of ATR is
  worth against a point of trend strength, and there is nothing behind such a
  number. Whether headroom *should* influence rank is a real question — one for
  section 4.5, which can measure it.
- **It fails closed, and distinguishes the ways of failing.** A name whose ATR
  was *measured and found too small* counts as `unreachable`. A name whose ATR
  *could not be read at all* counts as `not_ready`, alongside the cold-engine
  case. Merging them would make an engine that never warmed up read as a market
  too quiet to trade. There is no third case about price: the clip is a floor
  on turnover, so one share always clears it and no name is ever refused for
  being too expensive to buy. What the screen refuses is a name that cannot
  plausibly move far enough, which is a judgement about the market rather than
  about the configured size.
- **`max_atr_multiple` has no default.** A caller must state the assumption it
  is making; section 7.2 is why. The time gate (`min_minutes_remaining`) is off
  unless configured, on the same reasoning that leaves the session window
  unbounded.
- **The whole screen is off by default.** `ScannerConfig.feasibility` is `None`
  until a caller supplies a policy. That is not an opinion that costs do not
  matter. Two of the three inputs now have stated values — `FIXED_CLIP_NOTIONAL`
  at ₹1,00,000 and `STATED_GROSS_TARGET` at 0.2% above the buy price — but
  `max_atr_multiple` has none, and giving it one would reintroduce by the side
  door exactly the invented threshold section 7.2 forbids. So the screen stays
  off until a caller states it.

Every check is attached to the candidate it let through, passing or failing, so
the hurdle that was in force is recorded with the result rather than inferred
later. **The screen still cannot see the spread**, which at this horizon is
frequently the largest cost of all and which no layer yet produces; when the
reserved `MarketContext` microstructure fields arrive they belong in this
function, for the same reason and in the same place.

**The cost model has two layers, and the split is deliberate.** `costs/model.py`
is pure fraction arithmetic — what a round trip costs as a fraction of notional,
and the gross move that clears it. That is what the scanner screens with, since
screening compares against `atr_pct` and never needs a price. `costs/sizing.py`
is the decision-time layer: handed a quote, it turns the clip into whole shares
and the required fraction into an exit price the exchange will accept. Keeping
them apart is what lets the scanner stay free of position sizing, which under
AGENTS.md rule 8 belongs to deterministic code rather than to anything upstream
of it. The package is stdlib-only and imports nothing else in this system —
the scanner screens with it, replay will score with it and the risk engine will
size with it, so anything it imported would become a dependency of all three.
A test asserts that rather than trusting it.

**The buying model is one fixed clip, and exits are the whole position.**
`FIXED_CLIP_NOTIONAL` is ₹1,00,000 of turnover per entry; a sell is everything
held. Nothing in `sizing.py` can express a partial exit — a single `quantity`
serves both legs — so scaling out would arrive as a visible change to the shape
of `TradeCostEstimate` rather than quietly as a new code path. The clip is a
*stated* strategy parameter, not a measured one, which is why it is a named
constant a caller passes in rather than a default hidden inside `SizingPolicy`.

**Quantity is integral, so the clip is a floor and never the notional.** A
one-lakh clip of a stock at ₹2,450 is forty-one shares and ₹1,00,450 — forty
would be ₹98,000, which is not a lakh. Every cost here is a fraction *of
notional*, so using the target where the filled value belongs prices a trade
that was never placed — an error under two percent, which is far too small to
look wrong and far too large to ignore against a target measured in tenths of a
percent. A name quoted above the clip buys one share and overshoots; NSE lists
several, and `excess_notional` reports the overshoot rather than hiding it, so a
position limit has a number to refuse when one exists.

**Size is stated before strategy because the cost curve bends.** Brokerage caps
at ₹20 *per leg*, so below the cap the charge is a flat fraction of turnover
and above it the fraction falls away. Both schedules cap at the same ₹20 but
charge different rates to reach it — Groww 0.1%, Zerodha 0.03% — so the bend
sits at ₹20,000 a leg for one and ₹66,666.67 for the other, and past the higher
of those the two are the same number:

| Clip | Groww | Zerodha (Kite) |
|---|---|---|
| ₹10,000 | ₹27.15 (0.2715%) | ₹10.63 (0.1063%) |
| ₹20,000 | ₹54.30 (0.2715%) | ₹21.26 (0.1063%) |
| ₹50,000 | ₹64.94 (0.1299%) | ₹53.14 (0.1063%) |
| ₹1,00,000 | ₹82.68 (0.0827%) | ₹82.68 (0.0827%) |
| ₹5,00,000 | ₹224.61 (0.0449%) | ₹224.61 (0.0449%) |

At Groww's rates the same strategy is a loser at ₹20,000 and a winner at
₹1,00,000, with the break-even for a 0.2% gross capture at ₹28,690. A hit rate
quoted without the clip it was measured at — and now, with two schedules
modelled, without the schedule it was priced under — is not a result.

**"A small amount above costs" read two ways, and the reading is now stated.**
Asking to keep 0.2% *after* charges and asking the price to *move* 0.2% are
different requests, and at the one-lakh clip they differ by a third:

| Stated | Read as net kept → gross needed | Read as gross move → net kept |
|---|---|---|
| 0.1% | 0.1827% | 0.0173% |
| 0.2% | 0.2827% | 0.1173% |

That table is the same under either schedule, because both caps bind well below
a lakh. The strategy takes the second column: `STATED_GROSS_TARGET` is 0.2%
**above the buy price**, so what a trade keeps is 0.2% minus the round trip —
0.1173%, or ₹117.32 on a lakh. The table stays because the distinction is worth
the space: read the other way a 0.1% target would leave ₹17.32 on a lakh,
four-fifths of it taken by charges, and a figure that did not say which reading
it was would be worse than none. `SizingPolicy.from_gross_target` performs the
conversion in one place and refuses a target its own costs consume — at Groww's
rates 0.2% at a ₹20,000 clip is a loss, and the constructor raises rather than
clamping to zero. At Zerodha's it is not, which is why the schedule is an
argument to the conversion rather than an import inside it.

**Converting the target once, at the clip, is safe in the only direction that
matters.** The conversion needs a notional, and the clip is the smallest fill
the buying model permits, so it pays the largest cost fraction. Every real quote
rounds up to a whole share, turns over more and pays less, which means the
hurdle a name actually faces lands at or *below* the figure the strategy was
stated in — never above it. That is what makes stating the target gross
legitimate rather than merely convenient, and it is asserted across a spread of
quotes rather than argued for in a comment.

**Exit prices round to a tick, and always away from the entry.** The arithmetic
gives an exact price the exchange will not accept, so it has to move to a tick
boundary, and the direction is a correctness question rather than a preference:
rounding a long's exit *down* to the nearer tick leaves it under the price that
pays for the round trip, so the trade fills, looks like a win, and returns less
than the margin that justified taking it. Long exits round up, short exits round
down. Both are computed and neither is selected — importing the scanner's
`Direction` would invert the dependency and break the stdlib-only contract, and
cost is direction-symmetric anyway, so there is nothing to choose between until
a rule has picked a side.

**The tick is a floor on how fine a target can be.** `tick_fraction` is
`tick_size / entry_price`, and it moves inversely with price: at ₹2,450 a
five-paisa tick is 0.002% and rounding is a rounding error, but at ₹100 it is
0.05% — half of a 0.1% target. Past that point the exit can only land on the
tick grid, so what the trade ends up asking for depends on where the entry sits
in that grid rather than on the target. At the 0.1% target an entry at ₹100.00
is exactly on a tick and the exit returns the ₹17.32 it asked for, while an
entry one tick higher at ₹100.05 must reach ₹100.20 and returns ₹67.30 —
roughly four times the target, off a five-paisa difference in entry. The
overshoot is not a windfall: the trade is now asking the market for a move four
times larger and will fill correspondingly less often. **Cheap stocks are a
coarser instrument for this strategy than expensive ones**, and this is the
number that says so. The authoritative tick is a per-instrument attribute;
`Instrument` carries only an exchange and a trading symbol, so `NSE_EQUITY_TICK`
is a legacy fallback, not an instrument lookup. The examples above illustrate
the arithmetic on a hypothetical five-paisa grid, not current tick eligibility.
NSE CMTR67133 introduced monthly price-band tick sizes from April 15, 2025;
the RELIANCE September 2026 research profile explicitly uses INR 0.10. A study
must state the applicable tick rather than treat the fallback as universal.

**What these numbers are worth.** Both schedules are transcribed from published
tables and **neither has been reconciled against a real contract note**, and
the spread is not modelled at all. On the section 11 ladder the cost model
therefore sits a rung below the scanner: the exit price it produces is a
*floor* — the price below which a trade certainly does not pay — rather than a
prediction of what will be realised.

**Order of operations in one cycle.** The sequence is load-bearing and each step
short-circuits the rest:

1. **Duplicate guard** — two snapshots for one instrument in a cycle raises.
2. **Suppression** from `PortfolioState`, then the session window. Reported
   first because a kill-switched session that reported `outside_window` for
   every name would bury the fact that entries are blocked at all.
3. **Feasibility screen**, if configured. Before any rule runs: a name that
   cannot pay for its own round trip is not worth scoring, and scoring it anyway
   would let it take part of the candidate budget from a name that could.
4. **Rules**, each skipped unless every feature it requires is available.
5. **Rank**, on a total key — `(−score, symbol, exchange, direction)` — so ties
   cannot be broken by the order instruments happened to arrive in.
6. **Truncate** to the budget, recording how many were discarded.

Three findings from building it are worth carrying forward, because each is a
trap that fails silently rather than loudly:

- **The rolling window includes the candle being measured** (section 10), so
  `close > rolling_high_20` is unsatisfiable and a breakout rule written that
  way emits nothing for an entire session without erroring once. `BreakoutRule`
  tests ATR-scaled *closeness* to the window edge instead, and a test drives
  that through the real `FeatureEngine` rather than a fixture that could assert
  the convenient thing.
- **Volume splits two ways.** A rule that *needs* volume (`vwap_reversion`) is
  skipped when it is unavailable. A rule that merely *prefers* it
  (`range_breakout`) scores without it rather than averaging against a zero —
  otherwise every breakout in the first twenty minutes of a session, when
  `volume_ratio_20` does not yet exist, is systematically marked down.
- **Every feature is read through its own readiness flag**, never through
  `is not None`, which makes the section 4.3 trap structural rather than
  remembered. The defence is tested against a snapshot whose flags and values
  disagree; withholding a feature outright cannot catch the mistake, because
  that removes the value too and both spellings then decline for the same
  reason.

**A scanner result is a research hypothesis, not permission to trade.** This is
the sentence that resolves what would otherwise be a chicken-and-egg problem:
the scanner does not need the single correct strategy decided in advance. It
needs candidate hypotheses that the replay engine can then measure. Strategy
selection is an empirical *output* of section 4.5, not a prerequisite for this
layer. `Candidate` carries no quantity, stop or target, and a test asserts the
absence rather than trusting the convention.

**Candidate budget.** The scanner emits at most **N candidates per decision
cycle**, ranked by score, with N explicit and tunable. Start at 3–5; the
implemented default is 5, and like every other number in this layer it is a
placeholder rather than a measurement (section 7). Section 4.5's engine now
exists and could settle it — `--max-candidates` is a backtest flag — but no
sweep has yet varied it, so "awaiting measurement" has become "awaiting
someone's attention", which is a weaker excuse and worth naming as such.
`ScanResult.truncated` reports how many candidates the budget discarded, which
is the signal for tuning it: a cycle that truncates is one where the budget, not
the market, chose what the AI saw.

That single number is the most consequential tuning knob in the system: it sets
LLM cost per session, rate-limit exposure, and how much attention the AI can
give each candidate. An unbounded scanner makes the AI layer simultaneously
expensive and shallow. A hard cap on LLM calls per session backs it up as a
cost circuit breaker (section 4.6).

**Scores are normalized per rule, and comparability across rules is an open
question.** Each rule maps its own evidence onto 0..1, where 1 means "as
convincing as this rule can be". The scanner ranks across rules as though a 0.8
from one meant the same as a 0.8 from another, because it must rank somehow —
that is an assumption replay has to confirm, not a property being claimed. Where
two rules agree on a name and direction the higher score wins and both are named
in `Candidate.rules`; no agreement bonus is invented, since a confirmation
multiplier here would be a tuning constant with nothing behind it.

**Suppression from `PortfolioState`.** Do not emit candidates for names already
at their position limit, inside a cooldown window, or flagged stale (section 6).
A suppressed name is never evaluated and consumes no part of the budget, which
is why `ScanResult` counts suppression separately from `not_ready` (the engine
has not warmed up) and from `considered` (evaluated, nothing fired). Those three
are indistinguishable from the outside otherwise, and they call for opposite
responses.

**Reserved: `MarketContext`.** `FeatureSnapshot` and `PortfolioState` are what
the scanner consumes *today*, and they are enough to build and replay the first
hypotheses. They are not sufficient for short-horizon intraday trading
indefinitely, and the gap is already visible inside this document: section 4.7
contemplates spread and liquidity limits in risk, but no layer currently
produces a spread. Nothing in `src/` computes bid, ask, depth or imbalance.

So a third input is reserved rather than designed now — a per-cycle
`MarketContext` carrying, at minimum:

- **Microstructure per symbol** — bid, ask, spread, top-of-book depth, order
  imbalance. A candidate whose theoretical edge is smaller than its spread is
  not a candidate.
- **Index / regime state** — NIFTY trend and volatility, so a long breakout can
  be suppressed when the index is breaking down.
- **Staleness per symbol** — last tick age, which section 6 already requires as
  a suppression input and which has to come from somewhere.

Two rules attach to it. Build it when a rule actually needs it, not
speculatively — but shape the scanner's signature so that adding it is not a
rewrite. And treat every field as optional and explicitly unavailable when
missing, exactly as `FeatureSnapshot` treats an unready feature: a rule that
needs a spread must be suppressed when the spread is unknown, never run against
a guessed one.

### 4.5 Replay / research engine — `replay/` — BUILT

Runs the exact deterministic pipeline over historical candles:

```
Candle → FeatureEngine → Scanner → Candidate → simulated outcome
```

Four modules, read in the order a trade meets them. `replay/models.py` holds the
fill assumptions and the trade record, `replay/portfolio.py` the book,
`replay/engine.py` the loop and the no-lookahead argument, `replay/report.py` the
output. Two supporting packages exist for it and are used by nothing else today:
`history/store.py` caches the bars so a published sweep can be re-run against the
same tape, and `cli/backtest.py` is the front end. Section 4.11 lists its
options.

**The strategy is configured once, above the fork.** `strategy/config.py` holds
`StrategyConfig` — clip size, cost model, rules, cost screen, exits, book limits,
square-off — and both replay and, when it exists, the live path build their
scanner by calling `StrategyConfig.scanner()`. Not by building equal configs,
which is a property somebody has to keep true, but by calling one constructor,
which is a property that cannot come apart. That is the mechanical form of the
claim in section 7.1 that the fork between replay and live happens *after* the
scanner. `cli/backtest.py` deliberately states no strategy default of its own;
every strategy flag parses to `None` and is forwarded only when passed.

Built-in rule thresholds and component weights are now held in an immutable
`ScoringConfig` on that strategy. Both `backtest` and the live diagnostic load
the same complete `--strategy-config` JSON, reject mixed overrides, and report
its SHA256. A saved configuration has every field and a schema version so new
defaults cannot silently alter an old experiment. Feature definitions remain
shared code; research manifests pin their source hashes separately.

Unsigned baseline files retain schema version 1 and their original fingerprints.
An explicit signed target-time model uses version 2 and stores its transform
version, exact coefficients, target, decay and cutoff. Both CLI paths load the
same profile, and replay/live parity tests cover negative as well as positive
candidate scores. Do not compare the two score conventions as probabilities.

Nonzero `score_threshold` values use schema version 3. Version 1 and 2 profiles
retain their fingerprints and behavior, including zero threshold for version 2.
Both CLI paths and the real replay cycle tests cover the same inclusive cutoff.

`ReplayEngine.run(..., warmup_candles=...)` updates only the feature engine on
the pre-period input. It rejects overlap with the measured window and does not
count warm-up as scans, trades or sessions. `backtest --warmup-start` fetches that
input through the same candle store. `prepare_scanner_research` verifies declared
session coverage and freezes data, protocol, settings and code hashes before
outcomes are examined. Its `--verify` mode never authenticates or fetches.

Equal configuration is necessary but not sufficient for equal scanner output:
the normalized history, warm-up, universe and portfolio/context must also match.
The exact timestamp cross-provider comparison in
`backtests/reliance_sep2026/feature_parity.json` found real candle differences,
despite agreement with independent higher-precision indicator formulas. This
does not license a blanket live-versus-history equality claim.

Measures: forward returns, MFE / MAE, hit rate, expectancy, drawdown, turnover,
transaction costs, slippage sensitivity, time-of-day performance, regime
sensitivity.

**Score net, not gross.** The objective in section 1.4 is many small round trips
that clear their own costs, so a gross hit rate is not a result — the question
each candidate has to answer is whether it reached the stated 0.2% gross target
before it reached its stop, and what the round trip kept after `costs/` was
charged at the size `costs/sizing.py` would actually have bought. A replay that
reports gross returns is measuring a strategy this system is not running.

The two headline numbers to come out of it are therefore **net expectancy per
round trip** and **round trips per session**. Their product is the thesis; each
one alone can be made to look good while the other is fatal.

Its correctness below the scanner rests on a property that is established and
pinned by tests rather than assumed: the same session replayed produces the same
candles, snapshots and candidates as the live path did, and no layer in the
decision path can read the wall clock, so a scan at historical time *t* is the
scan that would have been live at *t*. Section 11 states the guarantee and its
one deliberate asymmetry — withheld volume makes a live scan see *less* than a
replay, so replay flatters `VwapReversionRule` specifically. Replay is not an
approximation of live behaviour at the feature level; it is the same computation
over the same objects. Re-run `tests/test_source_independence.py` if anything
under `features/`, `scanner/` or `costs/` is touched, because that is the test
that keeps this paragraph true.

**Decision latency is modelled explicitly.** This is where a naive replay
will lie.

Replay naturally fills at the signal candle's close. Live, the LLM sits in the
middle, so the order reaches the market several seconds later at a different
price. For momentum setups that bias runs in the *favourable* direction, which
is the worst kind: it makes a strategy look profitable in backtest and bleed in
production.

So `FillModel` carries a **`latency_seconds`** field and the book fills at the
price N seconds after the signal bar, not at its close. N should be the latency
**measured** in shadow mode and recorded in the journal (section 4.10), not a
guess. Until shadow data exists, sweep N across a range and report the
sensitivity — a strategy whose edge disappears between 2 and 10 seconds of
latency has no edge. `FillModel` refuses to default `latency_seconds`,
`half_spread_fraction` or `slippage_fraction`, and `cli/backtest.py` refuses to
invent them: a caller passes `--frictionless` or all three, because a silent zero
is the most flattering thing a backtest can assume and the one a reader is least
likely to notice.

Fill assumptions — spread, queue position, slippage — are modelled here too,
and are a larger source of replay error than anything in the feature layer.

**What it has measured so far.** The first sweep says the default thresholds in
section 4.4 do not pay: a loss at every stop multiple tried, and a gross hit rate
near 47% once friction is removed. Section 7.2 is the standing caution on how to
read that — it is a result about those conventions, and the universe and window
are themselves strategy parameters. `backtests/README.md` records the run.

### 4.6 AI decision layer — BUILT FOR RESEARCH/SHADOW (October 11)

Implemented in `ai/` (see [AI_LAYER.md](AI_LAYER.md) for inputs, commands and
costs): `openai_client.py` (Responses API over stdlib HTTP, strict JSON schema,
bounded retries, usage and cost), `decision.py` (one batched call per cycle
returning LONG / SHORT / NO_TRADE, confidence and a short reason per candidate;
fail-closed to NO_TRADE; deterministic `RuleDecider` baseline), `packet.py` and
`cycle.py` (the per-candidate scanner, order-book, OI and filings blocks, shared
by replay and live), `journal.py` (SQLite: exact input, model, prompt version,
tokens, cost, latency, verdicts, outcomes; the daily budget is enforced from it).
Inputs come from `scanner/movement.py` (live features parity-tested against the
research assembly), `orderbook/`, `oi/` and `news/`. `cli/run_decisions.py` is
the shadow loop. The verdict is a research record: it carries no size, stop,
target or order, and the BUY/SELL/HOLD/CLOSE intent schema below remains the
target once risk, position and execution layers exist. On all 688 post-cutoff
replay cycles (June-September 2026) no decider, OpenAI or deterministic, has a
statistically reliable edge after charges yet; see AI_LAYER.md.

GPT receives structured candidate context and `PortfolioState` — never raw tick
streams. Its purpose is higher-level synthesis: candidate quality, conflicting
signals, market context, regime context, the trade / no-trade judgement, and
whether a new position makes sense given what is already held.

Output must use a strict structured schema:

```
BUY | SELL | HOLD | CLOSE
confidence
reason / setup
entry constraints
proposed stop / target
valid-until
```

"Proposed" is load-bearing — a proposed stop or target is an input to the risk
engine, not a decision. **The AI must never determine actual position size or
bypass risk controls.**

**One batched call per decision cycle, not one call per candidate.** The call
carries all N candidates plus `PortfolioState` and returns a decision for each.

The cost argument is the smaller one. The real reason is that weighing
candidates against each other and against existing exposure *is* the judgement
being asked for, and N independent calls structurally cannot do it — each would
decide in ignorance of the others and of the book. The trade-off is that one
timeout costs the whole cycle rather than one candidate, which is acceptable
because the timeout policy is already "no new entries this cycle" (section 6).

**`valid-until` is enforced at two points**: when risk evaluates the intent, and
again immediately before execution submits. An intent that expired between
those points is dropped, not submitted late. Without both checks the field is
decorative.

**Latency and cost budget.** A per-cycle timeout, a bounded retry policy, and a
hard cap on calls per session. Exceeding the session cap disables new entries
for the remainder of the day; it does not silently degrade.

**The model ID is pinned and journaled with every decision.** Endpoints drift.
Validating against one model version and then being silently upgraded voids the
validation. Pinning makes a model change a deliberate act, and journaling makes
it visible afterwards as a regime break in the system's own results.

Every request and response is journaled (AGENTS.md rule 9), both for audit and
because that record is the raw material for answering whether the layer adds
value.

Still to be decided: model choice, and the concrete token and latency budget.

### 4.7 Deterministic risk engine — NOT BUILT

**Absolute veto layer.** Max risk per trade, max daily loss, max open
positions, max exposure, max trades per symbol, spread and liquidity limits,
cooldown, stale-data checks, disconnect handling, kill switch.

**Position sizing must be deterministic** (AGENTS.md rule 8). No LLM output
bypasses this layer.

**Portfolio-level exposure, not only per-trade.** Per-trade and per-symbol
limits do not prevent holding five correlated NSE financials as one
undiversified bet carrying five times the intended risk. Risk therefore also
caps gross exposure, net directional exposure, and exposure per correlation
bucket — sector, or a measured correlation grouping. This is the difference
between "no single trade can hurt much" and "no single *event* can hurt much",
and only the second is actually risk management.

Reads `PortfolioState` for live exposure, open count, and daily P&L.

The control categories are settled; their numeric thresholds are not, and should
be set from replay and shadow evidence rather than intuition.

### 4.8 Position manager — NOT BUILT

Own the lifecycle of an accepted trade: entry state, stop loss, take profit,
time exit, partial fills, position reconciliation, forced close. Publishes
`PortfolioState` (section 2.2).

**It must continue functioning even when the AI layer is unavailable.** An open
position with a live stop cannot depend on a network call to a language model.
This is a hard availability boundary: the position manager holds its own
complete exit logic rather than asking anything upstream what to do.

**Exits cannot be candle-driven.** A manager that sees only completed candles
can exit up to 60 seconds after a stop is breached mid-minute. At this
frequency, with intraday stops, that is material. Two mechanisms, with a clear
primary:

1. **Broker-native resting stop orders (primary).** The stop sits at the
   exchange from the moment the entry fills. It triggers at exchange latency,
   needs no round trip, and — most importantly — survives this process dying.
   A stop that exists only in local memory is not a stop.
2. **Tick-level supervision (secondary).** The position manager subscribes to
   the tick stream directly, below the candle layer, to manage discretionary and
   time-based exits and to detect divergence between local state and broker
   state.

The resting order is the safety net; the tick supervisor is the active manager.
Neither alone suffices — a resting order cannot express a time-based exit, and a
tick supervisor cannot survive a crash.

**Forced flat before the close.** Two hard times: a **no-new-entries** cutoff,
and a **force-flat** deadline after which any remaining position is closed at
market. Both are position-manager rules, not strategy parameters, and neither is
overridable by the AI.

Indian brokers auto-square-off intraday positions before the close. A system
that does not beat the broker to it takes whatever fill the broker's bulk unwind
produces. Confirm Groww's exact square-off policy and set the force-flat
deadline comfortably ahead of it.

### 4.9 Execution engine — DEFERRED

Translate approved deterministic trade instructions into broker orders. Order
type, limit price logic, slippage protection, partial fills, retries,
idempotency, broker reconciliation.

An order is a state machine — pending, acknowledged, partially filled, filled,
rejected, cancelled — and with several concurrent positions this is the most
bug-prone part of the system. Two non-negotiables: every order carries an
idempotency key so a retry cannot double-fill, and on startup or reconnect the
engine reconciles against **broker truth** before any new decision is made.
Local state is never authoritative about what is actually in the market.

**Live execution remains disabled until research, replay and shadow validation
are complete.** AGENTS.md: *"Do not implement live order placement during the
initial development phase."* The broker abstraction currently exposes no order
method, which is the point.

### 4.10 Journal / observability — NOT BUILT

Every important event is persisted: market snapshot, features, scanner
decision, AI request/response, **model ID**, risk decision, orders, fills,
position state, PnL, errors, and **latency at each stage**.

SQLite is sufficient initially.

**This data is required to determine whether each layer actually adds value.**
The journal is not an operational nicety added at the end — it is the
measurement instrument for section 1.3, and every layer above the feature engine
should be built with its journal records defined alongside it.

It also closes a loop: measured end-to-end latency from shadow mode is the input
to the replay engine's latency parameter (section 4.5). Without the journal,
that number is a guess.

### 4.11 Diagnostic CLIs — `cli/` — BUILT

| Command | Checks |
|---|---|
| `check_kite` | read-only Kite authentication and safe profile |
| `check_historical_data` | historical candle retrieval |
| `check_market_data` | quotes and LTP |
| `check_stream` | live tick streaming |
| `check_market_state` | backfill → `MarketState` |
| `check_features` | backfill → `MarketState` → `FeatureEngine` |
| `check_scanner` | backfill or live → `FeatureEngine` → `Scanner` |
| `check_costs` | round-trip cost, fixed-clip sizing, exit prices |

The seven broker-facing checks share exit codes: **0** success, **1**
broker/session failure, **2** configuration error. Each backfills the most
recent completed NSE session, so they work outside market hours and are the
fastest end-to-end smoke test.

One command in `cli/` is not a check: **`backtest`** is the front end to the
replay engine of section 4.5, and it is the only entry point here that produces
a result rather than a diagnosis. Its options fall into five groups — universe
(`--symbols` or `--symbols-file`, `--exchange`), window (`--start`, `--end`,
`--cache`, `--offline`), fill model (`--frictionless`, or all three of
`--latency-seconds`, `--half-spread`, `--slippage`; plus `--ambiguous-as`),
strategy (`--clip`, `--gross-target`, `--max-candidates`,
`--max-open-positions`, `--stop-atr`, `--trailing`, `--no-screen`,
`--max-atr-multiple`, `--cooldown-minutes`, `--square-off-minutes`) and output
(`--label`, `--report`, `--history`, `--no-write`, `--json`). It shares the
checks' exit codes: **0** success, **1** the run could not be completed, **2**
the configuration or the environment was wrong.

Two of those groups are deliberately default-free, for opposite reasons. **The
fill model has no default at all** — a caller passes `--frictionless` or all
three friction figures, because a silent zero is the most flattering thing a
backtest can assume and the one a reader is least likely to notice. **The
strategy group states no default either**: every strategy flag parses to `None`
and is forwarded only when actually passed, so `StrategyConfig` stays the one
place any of it is written down and a sweep cannot quietly diverge from what
live would run. `--offline` never loads credentials, because re-running a
published sweep is the mode that has to be reproducible: it reads only the
cached tape and fails loudly on a gap rather than filling it from the vendor.

`check_costs` is the exception and deliberately so. It touches no broker and
backfills nothing — the cost model is arithmetic over a published schedule — so
it has no configuration to get wrong and no exit code 2, and it prints the same
figures at midnight on a Sunday as at 09:20 on a Tuesday. It exits **1** only
on an unusable price argument. The numbers that decide whether a strategy is
viable at a given clip should not be reachable only while the market is open.

`check_features --export-csv PATH` exists so feature values can be compared
against a trusted external implementation before a scanner is built on them.

**`check_scanner` is the only check that runs the full stack**, and one of three
with a live mode — `check_stream` streams by definition, `check_features --live`
has driven a supervised window since 2026-09-22, and this is the third. By
default it backfills the most recent completed
session and scans every minute of it, which answers "what would this scanner
have said, minute by minute, over a real day" without a market. `--live`
appends to that rather than replacing it: it backfills first — which is what
leaves the indicators warm — then drives a `StreamSupervisor` window for
`--live-seconds`, scanning each candle as ticks assemble it. That is the path
that produced the 2026-09-24 evidence in section 11, and the warm start is why
those runs reported `not_ready: 0` from their first live candle. Both modes
take `--export-csv`, and both emit the same per-cycle counts: `considered`,
`not_ready`, `unreachable`, `suppressed`, `candidates`, broken down by rule and
by direction. The screen is tunable from the command line — `--clip`,
`--gross-target`, `--max-atr-multiple`, `--max-candidates`, `--broker` — so the
cost hurdle can be re-priced against either schedule without editing code.

Two cautions on reading its output. First, it constructs
`PortfolioState.empty(as_of)` on every cycle, so nothing is ever suppressed by
cooldown or position limit — **the candidate rate it reports is an upper
bound**, not what a running system would emit. Second, a candidate is not a
signal that the hypothesis is sound; section 7 applies to every threshold that
produced it.


## 5. Session lifecycle

An intraday system has a day-shaped state machine. It spans layers, so it is
specified here rather than inside any one of them.

```
  pre-open      warm up features from the previous session
  09:15 IST     session open — features already warm, trading enabled
  ...           active: scan → AI → risk → position → execute
  T_cutoff      no new entries
  T_flat        force flat any remaining position
  15:30 IST     close; reconcile, finalize journal, compute session P&L
```

### 5.1 Warm-up: the 10:05 problem

`core_ready` requires `ema50`, which needs 50 candles. From a cold start at
09:15 that means no complete feature set until **10:05 IST** — and the opening
window is the most active part of the Indian session.

**Resolution: pre-warm from the previous session before the open.**

The mechanism already exists and already behaves correctly.
`FeatureEngine.warm_up()` accepts arbitrary candles, and because session VWAP
resets on the IST date change and `volume_ratio_20` is same-session scoped,
pre-loading the prior session warms EMA / RSI / MACD / ATR while correctly
resetting the session-scoped features. No code change is required — what was
missing was the decision.

**The decision: carry trend and volatility state across the overnight gap.**

An EMA50 is a 50-minute trend estimate. Discarding it at 09:15 does not make it
more accurate, it makes it unavailable. And the carried state decays quickly —
with α = 2/51, pre-gap data holds roughly 13% of the weight at 10:05 and under
2% by 11:00. It matters most exactly when nothing else is available, and stops
mattering as the session establishes itself.

The overnight gap is now explicit in `gap_from_previous_close`, requiring a
complete previous regular session and today's actual opening minute. Prior
5/20-session context needs correspondingly longer warm-up. None changes the
carried EMA/ATR contract or the default scanner.

Session-scoped features (VWAP, `volume_ratio_20`) must **not** carry over, and
already do not.

### 5.2 Trading window

`T_cutoff` and `T_flat` are configuration, enforced by the position manager
(section 4.8), not by any strategy. `T_flat` sits comfortably ahead of the
broker's auto-square-off.

Time-of-day is also a legitimate scanner constraint (section 4.4), with one
caveat: `volume_ratio_20` is unavailable for the first 20 minutes of each
session (section 10), which overlaps the open. A scanner leaning on relative
volume at the open needs the seasonality-adjusted RVOL described there first.


## 6. Degradation policy

What the system does when something fails. Specified centrally, because
scattering these decisions across layers is how a system ends up with
inconsistent failure behaviour.

| Failure | Response |
|---|---|
| AI timeout or malformed output | No new entries this cycle. Open positions unaffected and fully managed. |
| AI rate-limited, or session call cap reached | As above; back off. Never queue stale candidates for later submission. |
| Broker feed disconnect | Halt new entries immediately. Broker-side resting stops remain the active protection. Reconcile against broker truth on reconnect. |
| Stale data — no tick for a subscribed symbol beyond a threshold | Treat that symbol as untradeable and suppress its candidates until the feed recovers. |
| Daily loss limit reached | Kill switch: no new entries for the session. Open positions may only be exited. |
| Process restart | Reconcile positions against broker truth **before** any new decision. Local state is never authoritative. |

**An unavailable AI does not fall back to trading deterministically — unless an
explicitly approved fallback strategy exists.**

The default is no new entries, because the tempting alternative is wrong. The
scanner's candidates were never validated as a standalone strategy — they are
hypotheses (section 4.4), and the AI's selection is part of the system being
evaluated. Trading them unfiltered because the AI timed out means running an
unvalidated strategy at the exact moment the system is already degraded.

The prohibition is on trading something unvalidated, not on deterministic
trading as such. A deterministic fallback may be enabled only when that
standalone rule set has independently passed the section 7 bar on its own
evidence, and has been explicitly configured as an approved fallback with its
own risk limits. Absent that configuration, the answer is no new entries. "The
AI is down" is never itself the justification.

Existing positions are a different matter and are managed normally throughout.
That is precisely why section 4.8 requires the position manager to be
independent of AI availability.


## 7. Research principle

**Every strategy is a hypothesis.**

Before live trading it must pass, where applicable: historical replay,
out-of-sample testing, walk-forward testing, transaction-cost modelling,
slippage stress tests, parameter sensitivity, regime analysis, shadow trading,
and small-capital validation.

Prefer simple rules until additional complexity demonstrates measurable
out-of-sample value.

This principle is why the architecture looks the way it does. Deterministic,
reproducible, incremental computation is not an aesthetic preference — it is
the precondition for any of those tests meaning anything. A pipeline whose
numbers shift depending on how data arrived cannot be walk-forward tested,
because the test and the deployment would be measuring different systems.

### 7.1 What replay can and cannot validate

| Claim | Validated by |
|---|---|
| Feature correctness | unit tests against hand-computed values |
| Scanner hypotheses have edge | historical replay |
| Sensitivity to latency and cost | replay, swept across parameters |
| **AI selection adds value** | **primarily forward shadow trading** |
| Execution and reconciliation | shadow, then small-capital live |

**Historical replay is not authoritative evidence of AI value.** Replaying past
candidates through an LLM is possible and may be useful for research — for
prompt iteration, for sanity-checking output schemas, for catching a decision
rule that is obviously broken. It is simply not the evidence the go-live
decision rests on. Three independent reasons:

1. LLM output is non-deterministic. Temperature 0 narrows the distribution; it
   does not collapse it. A single replay pass is one sample, not a measurement.
2. Model endpoints drift. A validation run is only valid for the pinned model ID
   it ran against (section 4.6).
3. Replaying months of candidates through an LLM API is slow and expensive
   enough to distort how often it actually gets done, which is its own failure
   mode.

A fourth reason is specific to replay rather than to LLMs: a model trained on
text through some cutoff has plausibly seen commentary about the very sessions
being replayed. That contaminates a historical evaluation in a way no amount of
careful data handling fixes, and it does not affect forward testing at all.

The primary test is therefore **forward shadow trading against a pinned model ID
with every input and output journaled** (section 4.6), measured against the
control of taking every candidate the scanner emitted. So replay establishes
whether the candidates are worth anything, and shadow trading establishes
whether the AI picks the right ones. Different questions, different instruments
— and the build order (scanner, replay, shadow and journal, *then* AI) exists to
answer them in that order.

### 7.2 Three ways a result can be false

Each of these produces a number that looks like evidence and is not. They are
listed here because every one of them is cheap to prevent at design time and
expensive to detect afterwards.

**The universe is a strategy parameter.** Which symbols get scanned is a choice
made before any rule runs, and it is silently load-bearing. Validating on a
handful of large liquid names says nothing about the mid-caps a live scanner
would surface; picking today's universe from names already known to have moved
is lookahead wearing a different hat. State the selection rule explicitly —
liquidity floor, price band, exclusions — apply the same rule in replay and
live, and record it alongside the result. A backtest whose universe cannot be
reconstructed is not reproducible, whatever its numbers.

**Gross hit rate is not edge.** The system must clear brokerage, exchange fees,
STT, stamp duty, GST, spread and slippage before anything is left, and at
intraday horizons those costs are a large fraction of the move being captured.
Any target move is therefore a *hypothesis to be tested net of costs*, never a
constant to design around — which is why no such number is hard-coded anywhere
in this architecture. What `costs/` hard-codes instead is the *fee schedule*,
which is published fact rather than opinion, and the arithmetic over it:
`required gross = round-trip cost at this size + the margin asked for`. Both
inputs are the caller's to state and the second is the hypothesis itself. The
distinction is not pedantry — at Groww's rates the same round trip costs
0.2715% of a ₹20,000 clip and 0.0827% of a ₹1,00,000 one, because brokerage is
capped per leg, so a target written in as a constant would be wrong at every
size except the one it was picked for, and wrong at a different size under each
of the two schedules modelled (sections 1.4 and 4.4). Evaluate on **net
expectancy per trade after realistic costs**; a strategy can win most of its
trades and still lose money, and at this horizon that outcome is common enough
to be the default suspicion.

**Invented thresholds are not evidence.** Every numeric cut-off a scanner or
risk rule applies — an RSI level, a volume multiple, a stop distance, a
candidate budget — must come from measurement or be marked openly as a
placeholder awaiting it. Section 4.4's "start at 3–5" is a starting point, not a
finding. Writing a plausible-sounding number into a document does not make it
true, and a threshold that entered as a guess and was later cited as settled is
among the harder errors to unwind.


## 8. Cross-cutting concerns

**Exact arithmetic.** Every price and indicator is a `Decimal` under
`FEATURE_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)`, installed
per-computation via `localcontext`. Binary floats accumulate error differently
depending on operation order, which would make a replay and a live run disagree
for reasons unrelated to the market. The context inherits the default traps, so
`DivisionByZero`, `InvalidOperation` and `Overflow` **raise** rather than
producing NaN; consequently every division is explicitly guarded against a zero
or `None` denominator before it executes. `localcontext` installs a copy, so
sharing the context across threads is safe. The cost has been measured and is
negligible at one-minute resolution across a realistic universe; the current
figure lives in [`handover.txt`](handover.txt), since it is a property of a
machine rather than of the design.

**No hidden lookahead.** At decision time `T`, only information that would
genuinely have been available by `T` may be used. This binds replay, the
scanner, and any future research tooling equally. Concretely: a candle may be
acted on only after its close, a feature may not incorporate a candle later than
the one it is stamped with, a fill may not be priced better than what the
decision-time book would have offered, and a universe may not be chosen using
knowledge of how the session turned out (section 7.2). Lookahead does not
announce itself — it shows up as a strategy that is excellent in replay and
unremarkable in shadow, which is exactly the comparison the build order exists
to make. Any replay result that cannot be reproduced under this rule is void.

**Reproducible by hand.** Any number a strategy might act on should be
derivable on paper from the candles that produced it. This is why indicators
are hand-written rather than pulled from TA-Lib or pandas-ta: a scanner firing
on an opaque number is a scanner nobody can debug at 09:20 with money at stake.

**Unavailable beats approximately right.** A feature that cannot be computed
honestly reports `None`, never a plausible substitute. Consumers check
readiness flags rather than treating `None` as zero. This costs coverage and
buys the ability to trust a number when it does appear.

**Time.** **Everything is IST.** `src/ai_trader/clock.py` owns the single
`INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")` the whole system reads, and
timestamps are *normalized* to that zone at every boundary rather than merely
made timezone-aware. The distinction matters because an aware UTC instant and an
aware IST instant compare and hash identically, so the choice is invisible to
arithmetic — but it is not invisible to `.date()`, to `.hour`, or to a cached
CSV, each of which answers a *local* question. Carrying UTC internally would
mean every session boundary, every daily cache key and every VWAP session key
converted first, and the one that forgot would be wrong only between 00:00 and
05:30 IST, which is outside market hours and therefore nearly untestable. NSE's
regular session is 09:15–15:30: 375 one-minute slots, plus a closing-auction
print at 15:30. Real sessions run short of 375 because of genuine gaps.

The clock is a top-level leaf that imports nothing from `ai_trader`, which is
the property that makes "one zone" enforceable rather than aspirational —
before it existed, `broker/groww.py` had grown its own `ZoneInfo("Asia/Kolkata")`
as a second answer to the same question. The two agreed, which is exactly why
nothing caught it.

**Concurrency.** Broker SDKs deliver ticks on their own feed threads.
`CandleBuilder`, `MarketState` and `FeatureEngine` each hold their own `Lock`,
and callbacks fire *outside* the lock protecting the state they report on. No
layer may assume its caller serializes it.

**Errors.** Invalid input raises (`InvalidTickError`, `ConfigurationError`);
stale or repeated input is counted and dropped. The distinction is deliberate:
a malformed tick is a bug, a duplicate is a fact of network life. Runtime
degradation is a third category, handled per section 6.

**Configuration and secrets.** `load_groww_settings()` in
`src/ai_trader/config.py` returns a frozen `GrowwSettings` whose credential
fields are `SecretStr`, so `repr()` masks them and an accidental log line or
traceback cannot leak a token — AGENTS.md rule 5 enforced by the type rather
than by reviewer discipline. The loader names only the *missing* variables in
its error, never a value.

| Variable | Used by |
|---|---|
| `GROWW_TOTP_TOKEN`, `GROWW_TOTP_SECRET` | broker authentication |
| `OPENAI_API_KEY` | OpenAI decider in `ai/` (see AI_LAYER.md) |
| `GITHUB_PAT`, `GITHUB_USERNAME` | `scripts/sync.sh` only |

`.env` is gitignored and must never be committed, printed, or inspected —
including by agents (AGENTS.md rule 11). `.env.example` is the authoritative
list and is a template with empty values, safe to read. `scripts/sync.sh`
passes the GitHub PAT through `GIT_ASKPASS` — never into `.git/config`, a
remote URL, or a credential helper — and refuses to run against a non-GitHub
remote.

**Dependencies.** `growwapi`, `pydantic`, `pyotp`, `python-dotenv`. Dev:
`pytest`, `ruff`. No pandas, NumPy, TA-Lib or pandas-ta. No Docker, Kubernetes,
Redis, MCP, or non-SQLite databases without a demonstrated requirement.


## 9. The two data paths

They differ, and every layer above must tolerate both.

| | Historical | Live |
|---|---|---|
| Entry | `MarketState.backfill` | `MarketState.record_tick` |
| Through `CandleBuilder`? | **no** | yes |
| Volume | per-candle, as given | polled and stamped — see below |
| First minute | present | **discarded** (section 4.2) |
| Ordering | ascending, as returned | arbitrary; guarded |
| Gaps | real, from the exchange | real, plus thin-tick minutes |

**Groww's live LTP feed carries no volume.** This was measured against the live
market on 2026-09-21: across 114 ticks, not one carried a usable volume. The raw
payload zeroes every field except `tsInMillis` and `ltp`. All seven live candles
built in that session reported `volume=None`, and VWAP and `volume_ratio_20`
were unavailable from the first live candle onward.

The code already handles this correctly, and that must be preserved. The stream
parses `volume` as optional (`broker/groww_stream.py`); because Groww transports
it as a protobuf double where an unset field is indistinguishable from a genuine
zero, a zero is reported as unknown rather than accepted as a differencing
baseline. The live run vindicated that guard: Groww does send `volume: 0.0`, and
without the guard every live candle would have silently anchored on a fabricated
zero baseline. The field stays optional the whole way down — `cumulative_volume`
on `MarketTick` is `int | None`, `CandleBuilder` skips the volume path on `None`,
and `CumulativeVolumeTracker` reports no volume rather than guessing.

The consequence was architectural, not incidental: **every volume-derived
feature was historical-only** — VWAP, `volume_ratio_20`, and anything a scanner
would build on them, for the entire live session rather than just the seam. The
one live volume Groww does serve is the running session total on the REST quote
endpoint, confirmed monotonic across eight polls in the same run (5,951,700 →
6,051,078), and a `VolumePoller` now differences that at each minute boundary to
supply live volume; section 10 records what it does and does not close. What
survives unchanged is the rule underneath: never fabricate a volume, and never
substitute zero for an unavailable one.

At the handoff — backfill up to the present, then attach a live stream — the
straddled minute is lost. That is one minute, and it is the correct trade
against emitting a fragment whose open, high and low are all wrong. Both
`MarketState` and `FeatureEngine` already tolerate gaps, so nothing downstream
needs to special-case it.

**This handoff has now run against a live market and held.** Joining mid-minute
at 12:23:59 IST, the partial minute was discarded as designed and the first live
candle opened at 12:25:00; 188 backfilled candles plus 7 live ones left 195
retained, with `late_tick_count`, `duplicate_candle_count`, and the engine's
duplicate and out-of-order counters all zero, and the state's last price equal to
the last candle's close.

One live-only hazard the handoff must respect: when the backfill runs against
**today's in-progress session**, the final candle Groww returns is partial and
keeps mutating — the 12:22 candle read volume 1799, then 2181 seconds later. A
live pre-warm must request candles up to the current minute and discard the last
one. The existing CLIs backfill completed prior sessions, so none of them are
exposed to this, but any live warm-up path is.

**Where the two paths must nevertheless agree.** Everything above is a
difference in plumbing, and none of it may become a difference in output: a
minute assembled from ticks and the same minute delivered whole by the broker
must produce the same `Candle`, the same `FeatureSnapshot` and the same
`ScanResult`. Without that, every threshold replay establishes belongs to a
market the live system never sees, and section 4.5 measures nothing.
`tests/test_source_independence.py` holds the guarantee against the whole chain
— ticks in at one end, candidates out at the other — rather than against any
single function, because it is a property of the chain.

*Source independence.* One deterministic session is expressed twice, as ticks
through `CandleBuilder` and as `OHLCVCandle`s through `MarketState.backfill`,
and compared field for field and then snapshot for snapshot and candidate for
candidate. The values are regenerated on each side rather than copied across, so
a builder that mis-assembled a minute cannot supply its own answer to both.

*No lookahead.* The snapshot at step *k* is identical whether the engine was fed
*k* candles or the whole session, checked at every step rather than sampled —
a lookahead that only appeared once the 20-period window filled would sit in the
middle of the session, where sampling the ends would miss it. This is what
licenses the claim that at any historical time *t* the scanner presents what it
would have presented if *t* were now. It rests on there being one code path:
`warm_up` is a convenience over `update`, not a second implementation. It also
rests on `minutes_since_session_open` being arithmetic against 09:15 on the
candle's own timestamp rather than a count of candles seen, so an engine that
attaches mid-session reports the same elapsed minutes as one that ran from the
open.

Neither property can be established by a fixture alone. A wall-clock read inside
the decision path would be reproducible in a test — the test also runs now — and
wrong in replay, where "now" is years after the candle. So a third check walks
the source of `features/`, `scanner/` and `costs/` and fails on any clock read.
`market/` is exempt on purpose: the volume poller stamps readings from the clock
to measure their staleness and the stream transport uses a monotonic deadline,
but neither reaches a candle's values and no historical candle passes through
either.

**The one asymmetry that is real** is volume, and it is an asymmetry of
knowledge rather than of behaviour. A failed poll leaves a live minute's volume
genuinely unknown, a state the historical endpoint never reports because it
always serves a figure. The answer is to withhold every affected feature rather
than estimate it: the readiness flags fall in step with the values, no price or
momentum feature moves, and a candidate's `evidence` can only cite what its
snapshot actually held. A live scan therefore sees *less* than a replay of the
same minutes, never something different about the same question. The consequence
for section 4.5 is that replay exercises the VWAP rule more often than a live
session with an imperfect poller does, so that rule's measured contribution is
an upper bound on its live one.


## 10. Known limitations

Deliberate, recorded, non-blocking. The two with architectural consequences:

**`volume_ratio_20` is unavailable for the first 20 minutes of every session**
(09:15–09:35 IST) because the baseline is same-session scoped. That is prime
scanning time, and unlike the `ema50` cold start (section 5.1) it is *not* fixed
by pre-warming, since the feature is deliberately session-scoped. The real fix
is a time-of-day seasonality-adjusted RVOL. `volume_ratio_same_minute_5` now
compares 09:20 volume against the previous five observed sessions' 09:20 bars.
It requires those five exact-minute observations. The default scanner still
uses the old rolling ratio; adopting the new feature is a separate research choice.

**Session VWAP is strict.** One candle with `volume=None` disables VWAP for the
remainder of that session rather than reporting a subtly wrong number. That
strictness used to be crippling in live operation: because Groww's live feed
carries no volume at all, the *first* live candle disabled VWAP and
`volume_ratio_20` for the rest of the day, and a live run of the whole chain on
2026-09-21 confirmed it at the feature layer — four live candles, every one with
`vwap`, `price_vs_vwap` and `volume_ratio_20` `None`, while every price and
momentum feature crossed the seam unbroken.

That gap is now closed. A `VolumePoller` reads the REST quote's running session
total on a background thread and stamps it onto each bare tick, so the candle
builder differences consecutive totals into a per-minute volume exactly as it
does for the historical path. A second live run the same day — 165 ticks, 100%
stamped, five live candles — carried `vwap`, `price_vs_vwap` and
`volume_ratio_20` populated across the seam with `core_ready` true throughout.
The remaining constraint is the one-minute resolution of the poll and its
loss-tolerance: a poll failure degrades that minute's volume to `None`, which
still disables VWAP for the rest of the session. Volume is therefore an
enrichment a scanner may lean on but must not assume, and every volume-derived
feature keeps its own readiness flag for exactly that reason.

The full list, including minor items, is section 10 of
[`handover.txt`](handover.txt).


## 11. Current build boundary

Implemented:

```
Broker abstraction
Historical market data
Live market feed
MarketTick normalization
CandleBuilder
Volume handling
MarketState
FeatureEngine
Deterministic Scanner          + PortfolioState contract (section 2.2)
Transaction cost model         + feasibility screen (section 4.4)
Fixed-clip sizing              + tick-aligned exit prices (section 4.4)
Strategy configuration         + one object both sides build from (section 4.5)
Candle cache                   + the tape a published sweep re-runs against
Historical Replay / Strategy Evaluation
```

Next, in order:

```
Shadow Trading + Journal
AI Decision Layer
Deterministic Risk
Position Management
Execution
```

**No layer should be skipped merely to reach live trading faster.**

Two notes on that ordering.

`PortfolioState` was defined with the scanner rather than after it — as
`PortfolioState.empty()`, since the position manager does not exist yet. A
scanner built as a pure function of `FeatureSnapshot` alone is architecturally
unable to suppress a name already at its position limit, and retrofitting that
parameter later is more disruptive than accepting it from the start. The stub
means the real signature is the one being tested and replayed today.

One qualification on "implemented", which matters enough to be made general.
**"Implemented" is not one state.** These are distinct, and calling a component
"done" without saying which one is how a system acquires unearned confidence:

| Level | Means |
|---|---|
| Implemented | the code exists |
| Tested offline | unit tests and replay over recorded data pass |
| Smoke-tested against the broker | a bounded diagnostic run reached the real API |
| Validated under live market conditions | ran sustained, during market hours, and behaved |
| Production-ready | plus supervision, recovery and reconciliation |

Against that ladder, live tick normalization, `CandleBuilder`, the first-minute
discard and the historical→live seam are now **validated under live market
conditions** — a sustained run on 2026-09-21 built seven contiguous live candles
onto 188 backfilled ones with every ordering and duplication counter at zero
(section 9). What that run also established is that the live tick payload
carries no volume of its own. That is a vendor data limitation rather than a
code defect, and it is no longer an unfilled gap: the `VolumePoller` supplies
cumulative volume from the REST quote endpoint, and a later live run the same
day carried `vwap`, `price_vs_vwap` and `volume_ratio_20` across the seam
populated (section 10). The volume-derived features are therefore validated
under live market conditions too — at the poller's one-minute resolution, which
is the remaining constraint on them rather than their absence.

What remains short of validated is everything around the stream rather than in
it. A bounded diagnostic subscription is not a resilient stream supervisor.
`StreamSupervisor` now exists (`market/stream.py`) and enforces most of what
section 6 specifies — it reuses a productive stream, rebuilds one that has gone
silent, discards on a retryable failure, backs off exponentially and gives up
after a bounded run of consecutive failures. Since 2026-09-22 it drives the
`--live` window of `check_features`, which is the one shipped path that streams
for longer than a probe, and the reconnect counters reach that command's printed
summary so a run that limped is not read as one that went cleanly. It still
reconciles nothing that was missed during a gap.

`check_stream` and `check_market_state` stay deliberately unsupervised. Both are
bounded probes measured in seconds, where a single backoff would outlast the
check it was protecting; what they need is to hand the transport back when they
exit early, which they do through the stream's context manager.

The supervisor's failure path *is* live-validated, and through the shipped
command rather than only a harness. Driven against the real broker during the
2026-09-22 outage, a standalone harness made three real connect attempts, backed
off 1s then 2s, stopped at its configured limit and let no exception escape
(section 9); `check_features --live` then reproduced the same behaviour inside
its own window — two connect attempts with a real backoff between them, zero
sessions opened, no exception escaping, accurate counters in the summary, and
the `VolumePoller` polling on through the outage untouched. **The success side
closed on 2026-09-24**, when the vendor outage had lifted and a fifteen-minute
`check_scanner --live` window on `RELIANCE` delivered 143 real ticks through the
supervisor into eleven live candles. Three of the four things that path was
waiting on are now observed rather than assumed: ticks reached the scanner
through the supervisor, the silent-session tripwire fired on seven of fourteen
one-minute windows that opened cleanly and delivered nothing, and the transport
was rebuilt six times after sessions that had been productive first. The fourth
— the consecutive-failure counter resetting on success — is consistent with that
run but not isolated by it: only two failures occurred against a limit of ten,
so the run never approached the bound the reset protects. It stays a unit-tested
claim.

Note also that
`check_market_state` bounds its tick window at fifteen seconds. That window is
shorter than a minute, so it can never contain a whole one — it either sits
inside a single minute or straddles one boundary — and the CLI therefore
observes live ticks without normally completing a live candle. The sustained
validation was done with a longer-running harness, not with the CLI.

**The scanner sits at "tested offline", and cannot climb higher yet.** Its code
exists and its unit tests pass, including against snapshots produced by the real
`FeatureEngine` rather than only by fixtures. Since 2026-09-24 it has also run
against a live feed: `check_scanner --live` scanned eleven live-built candles on
`RELIANCE`, screened every one of them as reachable, and produced one candidate
— with the cost screen pricing each name's hurdle off its own close, exactly as
it does over history. That is worth separating carefully from validation. What
it establishes is that the **mechanism** survives live input: the same scan runs
on a candle assembled from ticks seconds earlier as on one read from the
historical endpoint, and neither the screen nor the rules behave differently for
knowing which. What it does not establish is that any **threshold** in the layer
is right. A scanner is not validated by
running without erroring — it is validated by its candidates being measured, and
the engine that measures them, section 4.5, now exists and has run. **The first
answer was negative.** Over 61 sessions of 15 cached NSE symbols the default
rules lose at every stop multiple swept, and removing friction entirely leaves a
gross hit rate near 47%. Every threshold in `rules.py` is a conventional level
chosen so the layer could be built, not a number measured on this market, and
section 7 applies to all of them — including to that result, which measures
those conventions over that universe and that window rather than delivering a
verdict on the pipeline that computed it. "The scanner works" still means only
that the rules read what they claim to read and decline when they should. What
has changed is that the hypotheses inside it now carry a number, and the number
says tune them before anything downstream is asked to rank their output.
`backtests/README.md` records the sweep.

**What the scanner *has* established is the precondition replay depends on.**
Replay is only evidence about the live system if a scan replayed at time *t*
equals the scan that would have been live at *t*. That is no longer an
assumption. One session expressed twice — as ticks through `CandleBuilder` and
as candles through `MarketState.backfill` — produces identical candles,
snapshots and candidates; the snapshot at step *k* is unchanged whether the
engine was fed *k* candles or the whole session, checked at every step; and an
AST test forbids `features/`, `scanner/` and `costs/` from reading the wall
clock at all, so elapsed session time can only come from the candle's own
timestamp (`tests/test_source_independence.py`, section 9). One asymmetry
survives and is deliberate: a failed volume poll leaves a live minute's volume
*unknown*, a state history never reports, and every feature depending on it is
then **withheld rather than estimated**. A live scan therefore sees *less* than
a replay of the same minute, never something different — which means replay
overstates `VwapReversionRule`'s contribution, and that bias is one-directional
and known rather than lurking.

The cost screen sits one rung lower still, and for a different reason. Its
*arithmetic* is tested exactly — every charge reconciled in rupees against a
schedule worked by hand, for both of the schedules it models — but neither has
been checked against a real contract note, so the rates are published tables
transcribed rather than a measurement. Two other gaps are known and open:
`max_atr_multiple` is a placeholder like any threshold in `rules.py`, and the
spread, which at this horizon is frequently the largest cost of all, is absent
entirely because no layer produces one. A hurdle computed without it is a
floor, not an estimate.

The sizing layer inherits all of that and adds one gap of its own: the tick.
`NSE_EQUITY_TICK` is the common cash-segment value, but the authoritative tick
is a per-instrument attribute and `Instrument` carries only an exchange and a
trading symbol, so nothing in this system can look up the real one. On a name
whose true tick is finer, the exits it prices are further from the entry than
they need to be — conservative rather than wrong, but conservative in a way
that shows up as an overshoot rather than a loss, and therefore easy to mistake
for the strategy working better than it does.

**Broker calls are unreliable and must be retried — including
authentication.** Measured live on 2026-09-21 over 200 raw quote calls sampled
twice a second, 94 failed — every one because Groww answered a valid request
with a plain-text `404 page not found` body the SDK could not decode. Those
failures clustered into 44 short outages, the longest 3.6 seconds and most no
longer than one, with the failure rate staying high in between. `GrowwBroker`
therefore wraps each raw call in `_retry_broker_call`: eight attempts, the delay
doubling from 0.5s until it clears the longest observed outage and then held at
2s. Delays shorter than an outage put every attempt inside it; beyond that
length, more attempts are worth more than longer waits, because an attempt
landing between outages still fails about half the time. Only the raw call is
retried; normalization stays outside it so a genuine schema change surfaces at
once. Re-measured live against that budget, 50 broker reads all succeeded, at an
average of 1.3 seconds each.

The same fault hits the token endpoint, and authentication was for a while the
one entry point without a retry: 4 of 8 raw `get_access_token` calls failed the
same way on the same day, so every CLI had a coin-flip chance of dying before it
did any work. It is not TOTP reuse — failures landed on freshly generated codes,
and a deliberately reused code succeeded. `authenticate` now goes through the
same helper, generating the TOTP *inside* the retried call so an attempt that
crosses a 30-second window uses the code belonging to the window it lands in.
Because `GrowwAuthenticationError` subclasses `GrowwBrokerError`, a caller that
catches only the base class blames its own operation for a failure that happened
before that operation began; every entry point must catch the authentication
error first, and `tests/test_cli_auth_failures.py` pins that for all seven
broker-facing CLIs. `cli/backtest.py` is the deliberate exception: it catches
the base class, because its one authentication call site is already reporting
authentication and so has nothing else to blame.

Any future broker adapter needs the equivalent, and any supervisor must treat a
call failure as expected rather than exceptional — retries are not free latency,
so a per-cycle budget has to assume some calls take seconds rather than
milliseconds, and that a process restart pays that cost at startup too.

**Reserved: an orchestration layer.** Every entry point today is a diagnostic
CLI under `src/ai_trader/cli/`, and that is the right shape for a smoke check.
It is not the shape of a trading process. The live system needs one component
that owns the session: start-of-day warm-up and universe selection, the decision
cycle itself, ordered shutdown, and the degradation responses in section 6.
That logic must not accumulate inside a CLI module — the CLIs are tests, and a
test that grows into a trader is a trader nobody reviewed.

Mapped to the README roadmap, items 1–7 are complete and item 8 is next. The
transaction cost model is not a roadmap item of its own; it was built as a
precondition for item 6's feasibility screen and is the thing item 7 scores net
expectancy with.


## 12. Document map

| Document | Holds | Answers | Location fixed by |
|---|---|---|---|
| `AGENTS.md` | mandatory engineering/safety rules | what may not be done, ever | agent tooling reads it at repo root |
| `docs/ARCHITECTURE.md` | long-term system intent | what is being built and why this shape | — |
| `docs/handover.txt` | current implementation state / next task | where the work stands and what bit next | — |
| `README.md` | setup and usage | how to run it | `pyproject.toml` `readme` key; GitHub |
| `docs/LIVE_API_SAMPLES.md` | captured Groww request/response samples | what the broker actually returns | — |
| `docs/FEATURE_VALIDATION.md` | original 47-feature reference check, 16 new feature formulas/tests, and charting conventions | whether the numbers are right | — |
| `backtests/README.md` | published replay runs and the tape they read | what the scanner's thresholds actually earn | `ai_trader.cli.backtest` writes its artefacts beside it |

Each document is the single source for its column. Where two would otherwise
overlap: this one holds **intent that outlives the current state**, so a design
decision belongs here even before it is built; `handover.txt` holds **state and
hard-won reasons**, so a trap discovered while implementing belongs there even
if it changes nothing architecturally. The reading order for a fresh session is
at the top of this document.

`LIVE_API_SAMPLES.md` and `FEATURE_VALIDATION.md` exist for the same reason:
both capture something that can otherwise only be observed during market hours.
Consult them before waiting for an open market to answer a question one of them
has already answered.

`backtests/README.md` is the odd one out, and deliberately so: it is the only
document here that records a *result* rather than an intent, a state or an
observation of the vendor. Nothing in the code reads it. It lives beside the
artefacts it describes so that a claim about what the strategy earns can be
traced to the run, the tape and the flags that produced it — which is also why
every sweep it publishes is reproducible with `--offline` against the frozen
cache rather than against whatever the broker serves today.

`AGENTS.md` and `README.md` stay at the repository root for functional reasons,
not stylistic ones: agent tooling loads `AGENTS.md` from the root, and
`pyproject.toml` references `README.md` by path.

</details>
