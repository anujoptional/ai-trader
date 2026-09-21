# AI Trader — Target System Architecture

Canonical description of the intended system. Read this before making
architectural or cross-module changes.

Companion documents: [`handover.txt`](handover.txt) for current implementation
state and next task, [`../AGENTS.md`](../AGENTS.md) for mandatory engineering
and safety rules, [`../README.md`](../README.md) for setup and usage.


## 1. Goal and operating frequency

Build a robust AI-assisted intraday trading research and execution system for
highly liquid Indian markets.

Initial broker: Groww. Future broker support may include Zerodha Kite.

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
   replay, deterministically, offline.
2. **Does the AI's selection improve on simply taking every candidate?**
   Answered only by forward shadow trading. Replay cannot answer it — see
   section 7.1.

The deterministic baseline is therefore a **control**, not a gate the AI must
clear before being built. But if the AI cannot beat it in shadow mode, that is
a finding, and it should be acted on rather than explained away.


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

Two structural details carry most of this diagram's weight.

**The fork after the Scanner.** Historical replay and live/shadow trading
consume the *same* candidate stream from the *same* deterministic code. Replay
is not a separate offline tool bolted on later; it is a first-class consumer
sitting at the same level as live trading.

**The AI is sandwiched.** See section 3.

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

### 2.3 Layer status

| Layer | Module | Status |
|---|---|---|
| Broker adapters | `broker/` | built, read-only |
| Market layer | `market/` | built |
| Feature engine | `features/` | built |
| Deterministic scanner | — | **next** |
| Replay / research | — | not built |
| AI decision layer | — | not built |
| Deterministic risk | — | not built |
| Position manager | — | not built |
| Execution engine | — | deferred by AGENTS.md |
| Journal / observability | — | not built |


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

Convert completed candles into quantitative features: returns, EMA, EMA slope,
RSI, MACD, ATR, rolling highs/lows, VWAP, relative volume.

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

Computed: returns (1/5/15), EMA 9/21/50 and 5-period slopes, RSI14, MACD
(12/26/9) with histogram change, ATR14, rolling 20 high/low with distances,
session VWAP with `price_vs_vwap`, `volume_ratio_20`, `candle_range_pct`,
`true_range`.

Four properties matter more than the indicator list:

**Warm-up and live are the same code path.** `warm_up` is literally a loop over
`update`. An engine fed 60 candles in one batch and an engine fed 30 then 30
produce equal snapshots — asserted by test. This is what makes replay a valid
proxy for live behaviour below the scanner; without it, a backtest and a live
run could diverge for purely structural reasons.

**Readiness is mechanically checkable.** Every flag on `FeatureReadiness`
equals `getattr(snapshot, name) is not None`. `core_ready` deliberately
excludes `vwap` and `volume_ratio_20`, since both depend on broker volume that
may be absent.

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

**Memory per instrument is constant** — roughly 16 closes, two slope deques of
6, 20 highs, 20 lows, 21 volumes, and some scalars. EMA seed buffers are
discarded once seeded.

At the intended scale — order 200 liquid names — per-minute feature computation
is under two milliseconds and retained session state is tens of megabytes.
Throughput is not a design constraint at one-minute resolution. Tick-rate
contention on `CandleBuilder`'s lock is the thing to measure before scaling far
past that, not feature cost.

### 4.4 Deterministic scanner — NEXT

Filter the market and generate a small number of candidate setups.

Inputs are `FeatureSnapshot` and `PortfolioState` — **not** raw broker data and
**not** raw ticks. It must respect readiness flags rather than treating `None`
as zero.

It should encode measurable trading hypotheses such as: trend, momentum,
breakout, mean reversion, VWAP relationship, volatility, volume confirmation,
liquidity, and time-of-day constraints.

**A scanner result is a research hypothesis, not permission to trade.** This is
the sentence that resolves what would otherwise be a chicken-and-egg problem:
the scanner does not need the single correct strategy decided in advance. It
needs candidate hypotheses that the replay engine can then measure. Strategy
selection is an empirical *output* of section 4.5, not a prerequisite for this
layer.

**Candidate budget.** The scanner emits at most **N candidates per decision
cycle**, ranked by score, with N explicit and tunable. Start at 3–5.

That single number is the most consequential tuning knob in the system: it sets
LLM cost per session, rate-limit exposure, and how much attention the AI can
give each candidate. An unbounded scanner makes the AI layer simultaneously
expensive and shallow. A hard cap on LLM calls per session backs it up as a
cost circuit breaker (section 4.6).

**Suppression from `PortfolioState`.** Do not emit candidates for names already
at their position limit, inside a cooldown window, or flagged stale (section 6).

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

### 4.5 Replay / research engine — NOT BUILT

Run the exact deterministic pipeline over historical candles:

```
Candle → FeatureEngine → Scanner → Candidate → simulated outcome
```

Measure: forward returns, MFE / MAE, hit rate, expectancy, drawdown, turnover,
transaction costs, slippage sensitivity, time-of-day performance, regime
sensitivity.

Its correctness below the scanner rests on a property already established and
tested in the feature engine — batch and incremental feeding produce identical
snapshots. Replay is therefore not an approximation of live behaviour at the
feature level; it is the same computation over the same objects.

**Decision latency must be modelled explicitly.** This is where a naive replay
will lie.

Replay naturally fills at the signal candle's close. Live, the LLM sits in the
middle, so the order reaches the market several seconds later at a different
price. For momentum setups that bias runs in the *favourable* direction, which
is the worst kind: it makes a strategy look profitable in backtest and bleed in
production.

So the replay engine takes a **decision-latency parameter** and fills at the
price N seconds after the signal bar, not at its close. N should be the latency
**measured** in shadow mode and recorded in the journal (section 4.10), not a
guess. Until shadow data exists, sweep N across a range and report the
sensitivity — a strategy whose edge disappears between 2 and 10 seconds of
latency has no edge.

Fill assumptions — spread, queue position, slippage — are modelled here too,
and are a larger source of replay error than anything in the feature layer.

Build this immediately after the scanner and before the AI layer.

### 4.6 AI decision layer — NOT BUILT

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
| `check_groww` | authentication and profile |
| `check_historical_data` | historical candle retrieval |
| `check_market_data` | quotes and LTP |
| `check_stream` | live tick streaming |
| `check_market_state` | backfill → `MarketState` |
| `check_features` | backfill → `MarketState` → `FeatureEngine` |

Uniform exit codes across all six: **0** success, **1** broker/session failure,
**2** configuration error. Each backfills the most recent completed NSE
session, so they work outside market hours and are the fastest end-to-end smoke
test.

`check_features --export-csv PATH` exists so feature values can be compared
against a trusted external implementation before a scanner is built on them.


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

The overnight gap is real information, but the right way to express it is an
explicit `gap_pct` feature a scanner can reason about — not crippling every
other indicator for the first fifty minutes. Add `gap_pct` when the scanner
needs it.

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
in this architecture. Evaluate on **net expectancy per trade after realistic
costs**; a strategy can win most of its trades and still lose money, and at this
horizon that outcome is common enough to be the default suspicion.

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

**Time.** Timestamps are UTC internally. IST (`Asia/Kolkata`) is used only for
session boundaries and the VWAP session key. NSE regular session is
09:15–15:30 IST = 03:45–10:00 UTC: 375 one-minute slots, plus a closing-auction
print at 15:30 IST. Real sessions run short of 375 because of genuine gaps.

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
| `OPENAI_API_KEY` | reserved for the AI layer; unused today |
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
| Volume | per-candle, as given | **absent** — see below |
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

The consequence is architectural, not incidental: **any volume-derived feature
is historical-only until a separate volume source is wired in.** That rules out
VWAP, `volume_ratio_20`, and anything a scanner would build on them, for the
entire live session rather than just the seam. The one live volume Groww does
serve is the running session total on the REST quote endpoint, which was
confirmed monotonic across eight polls in the same run (5,951,700 → 6,051,078).
Differencing that at each minute boundary would restore live volume, but no
layer consumes it yet. Until one does, never fabricate a volume, and never
substitute zero for an unavailable one.

At the handoff — backfill up to the present, then attach a live stream — the
straddled minute is lost. That is one minute, and it is the correct trade
against emitting a fragment whose open, high and low are all wrong. Both
`MarketState` and `FeatureEngine` already tolerate gaps, so nothing downstream
needs to special-case it.

**This handoff has now run against a live market and held.** Joining mid-minute
at 06:53:59 UTC, the partial minute was discarded as designed and the first live
candle opened at 06:55:00; 188 backfilled candles plus 7 live ones left 195
retained, with `late_tick_count`, `duplicate_candle_count`, and the engine's
duplicate and out-of-order counters all zero, and the state's last price equal to
the last candle's close.

One live-only hazard the handoff must respect: when the backfill runs against
**today's in-progress session**, the final candle Groww returns is partial and
keeps mutating — the 12:22 candle read volume 1799, then 2181 seconds later. A
live pre-warm must request candles up to the current minute and discard the last
one. The existing CLIs backfill completed prior sessions, so none of them are
exposed to this, but any live warm-up path is.


## 10. Known limitations

Deliberate, recorded, non-blocking. The two with architectural consequences:

**`volume_ratio_20` is unavailable for the first 20 minutes of every session**
(09:15–09:35 IST) because the baseline is same-session scoped. That is prime
scanning time, and unlike the `ema50` cold start (section 5.1) it is *not* fixed
by pre-warming, since the feature is deliberately session-scoped. The real fix
is a time-of-day seasonality-adjusted RVOL — comparing 09:20 volume against
*historical 09:20s* rather than against the previous 20 minutes. Any scanner
leaning on relative volume at the open needs this first.

**Session VWAP is strict.** One candle with `volume=None` disables VWAP for the
remainder of that session rather than reporting a subtly wrong number. Correct,
but section 9's live measurement makes the consequence much larger than a
broker gap: because Groww's live feed carries no volume at all, the *first* live
candle disables VWAP and `volume_ratio_20` for the rest of the day. In live
operation both features are effectively historical-only. A scanner must not be
designed around them until a volume source is wired in. This was confirmed at
the feature layer, not merely inferred: a live run of the whole chain on
2026-09-21 produced four live candles, every one with `vwap`, `price_vs_vwap`
and `volume_ratio_20` `None`, while every price and momentum feature carried
across the seam unbroken. The remedy is proven available — the REST quote's
running session total differenced cleanly at a 15-second cadence in the same
window — so this is wiring that nobody has done, not a missing capability.

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
```

Next, in order:

```
Deterministic Scanner          + PortfolioState contract (section 2.2)
Historical Replay / Strategy Evaluation
Shadow Trading + Journal
AI Decision Layer
Deterministic Risk
Position Management
Execution
```

**No layer should be skipped merely to reach live trading faster.**

Two notes on that ordering.

`PortfolioState` should be defined — even as an empty or stubbed snapshot —
when the scanner is written. A scanner built as a pure function of
`FeatureSnapshot` alone is architecturally unable to suppress a name already at
its position limit, and retrofitting that parameter later is more disruptive
than accepting it from the start.

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
(section 9). What that run also established is that the live path delivers no
volume, so live VWAP and relative volume are not validated; they are *absent*,
and that is a data limitation rather than a code defect.

What remains short of validated is everything around the stream rather than in
it. A bounded diagnostic subscription is not a resilient stream supervisor, and
**no stream supervisor exists** — nothing today detects a silently dead
subscription, resubscribes after a disconnect, or reconciles what was missed.
Section 6 specifies the *policy* for a feed disconnect; the machinery that would
enforce it is not built. Note also that `check_market_state` bounds its tick
window at fifteen seconds, which cannot span a minute boundary, so that CLI
observes live ticks without normally completing a live candle — the sustained
validation was done with a longer-running harness, not with the CLI.

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
error first, and `tests/test_cli_auth_failures.py` pins that for all six CLIs.

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

Mapped to the README roadmap, items 1–5 are complete, and item 6 is next.


## 12. Document map

| Document | Holds | Location fixed by |
|---|---|---|
| `docs/ARCHITECTURE.md` | long-term system intent | — |
| `AGENTS.md` | mandatory engineering/safety rules | agent tooling reads it at repo root |
| `README.md` | setup and usage | `pyproject.toml` `readme` key; GitHub |
| `docs/handover.txt` | current implementation state / next task | — |

`AGENTS.md` and `README.md` stay at the repository root for functional reasons,
not stylistic ones: agent tooling loads `AGENTS.md` from the root, and
`pyproject.toml` references `README.md` by path.
