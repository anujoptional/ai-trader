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

**An unavailable AI never falls back to trading deterministically.**

Worth stating explicitly, because the tempting default is wrong. The scanner's
candidates were never validated as a standalone strategy — they are hypotheses
(section 4.4), and the AI's selection is part of the system being evaluated.
Trading them unfiltered because the AI timed out means running an unvalidated
strategy at the exact moment the system is already degraded.

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
| **AI selection adds value** | **forward shadow trading only** |
| Execution and reconciliation | shadow, then small-capital live |

**The AI layer cannot be backtested.** Three independent reasons:

1. LLM output is non-deterministic. Temperature 0 narrows the distribution; it
   does not collapse it.
2. Model endpoints drift. A validation run is only valid for the pinned model ID
   it ran against (section 4.6).
3. Replaying months of candidates through an LLM API is slow and expensive
   enough to distort how often it actually gets done, which is its own failure
   mode.

So replay establishes whether the candidates are worth anything, and shadow
trading establishes whether the AI picks the right ones. Different questions,
different instruments — and the build order (scanner, replay, shadow and
journal, *then* AI) exists to answer them in that order.


## 8. Cross-cutting concerns

**Exact arithmetic.** Every price and indicator is a `Decimal` under
`FEATURE_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)`, installed
per-computation via `localcontext`. Binary floats accumulate error differently
depending on operation order, which would make a replay and a live run disagree
for reasons unrelated to the market. The context inherits the default traps, so
`DivisionByZero`, `InvalidOperation` and `Overflow` **raise** rather than
producing NaN; consequently every division is explicitly guarded against a zero
or `None` denominator before it executes. `localcontext` installs a copy, so
sharing the context across threads is safe. Measured cost is ~8.3 µs per
update — about 4.2 ms per minute across 500 instruments, irrelevant at
one-minute resolution.

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
| Volume | per-candle, as given | differenced from cumulative |
| First minute | present | **discarded** (section 4.2) |
| Ordering | ascending, as returned | arbitrary; guarded |
| Gaps | real, from the exchange | real, plus thin-tick minutes |

At the handoff — backfill up to the present, then attach a live stream — the
straddled minute is lost. That is one minute, and it is the correct trade
against emitting a fragment whose open, high and low are all wrong. Both
`MarketState` and `FeatureEngine` already tolerate gaps, so nothing downstream
needs to special-case it.

**This handoff has never run against a live market.** It is the largest
untested surface in the system.


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
but it means a single broker gap costs a feature for the day.

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

One qualification on "implemented": the live market feed and tick normalization
exist and are exercised by `check_stream`, but the sustained
live-tick → candle → state path has not yet been run during market hours. See
section 9.

Mapped to the README roadmap, items 1–3 and 5 are complete, item 4 awaits a
weekday market-hours run, and item 6 is next.


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
