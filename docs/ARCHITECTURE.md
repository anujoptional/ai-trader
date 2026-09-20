# AI Trader — Target System Architecture

Canonical description of the intended system. Read this before making
architectural or cross-module changes.

Companion documents: [`handover.txt`](handover.txt) for current implementation
state and next task, [`../AGENTS.md`](../AGENTS.md) for mandatory engineering
and safety rules, [`../README.md`](../README.md) for setup and usage.


## 1. Goal

Build a robust AI-assisted intraday trading research and execution system for
highly liquid Indian markets.

Initial broker: Groww. Future broker support may include Zerodha Kite.

The system must remain testable, deterministic where possible,
broker-independent above the adapter layer, and safe by construction.

The LLM is an **advisory decision layer**. It must never control position
sizing, override hard risk rules, or communicate directly with the broker
execution API.

The system exists to answer one question: **does the AI layer add measurable
value over a deterministic baseline?** That question is only answerable if the
deterministic baseline exists, is trustworthy, and has been measured first.
That is the entire reason for the build order below.


## 2. Target architecture

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

Two structural details in that diagram carry most of its weight.

**The fork after the Scanner.** Historical replay and live/shadow trading
consume the *same* candidate stream from the *same* deterministic code. Replay
is not a separate offline tool bolted on later; it is a first-class consumer
sitting at the same level as live trading. This is what makes "did the AI
help?" a measurable question rather than an opinion.

**The AI is sandwiched.** See section 3.

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
convenience.


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
proxy for live behaviour; without it, a backtest and a live run could diverge
for purely structural reasons.

**Readiness is mechanically checkable.** Every flag on `FeatureReadiness`
equals `getattr(snapshot, name) is not None`. `core_ready` deliberately
excludes `vwap` and `volume_ratio_20`, since both depend on broker volume that
may be absent.

**Ordering is guarded before any state mutates.** A duplicate or out-of-order
candle increments a counter and returns `None`, touching nothing. A retry or a
replayed message cannot corrupt an EMA.

**Memory per instrument is constant** — roughly 16 closes, two slope deques of
6, 20 highs, 20 lows, 21 volumes, and some scalars. EMA seed buffers are
discarded once seeded. Tracking 500 symbols is bounded and flat.

### 4.4 Deterministic scanner — NEXT

Filter the market and generate a small number of candidate setups.

The scanner operates on `FeatureSnapshot` — **not** raw broker data and **not**
raw ticks. It must respect readiness flags rather than treating `None` as zero.

It should encode measurable trading hypotheses such as: trend, momentum,
breakout, mean reversion, VWAP relationship, volatility, volume confirmation,
liquidity, and time-of-day constraints.

**A scanner result is a research hypothesis, not permission to trade.** This is
the sentence that resolves what would otherwise be a chicken-and-egg problem:
the scanner does not need the single correct strategy decided in advance. It
needs candidate hypotheses that the replay engine can then measure. Strategy
selection is an empirical output of section 4.5, not a prerequisite for
section 4.4.

Still to be decided when this is built: the instrument universe and how it is
chosen and refreshed; what a `Candidate` carries (triggering features,
direction, score, validity window); and how time-of-day constraints interact
with `volume_ratio_20` being unavailable for the first 20 minutes of every
session (section 8), which is prime scanning time.

### 4.5 Replay / research engine — NOT BUILT

Run the exact deterministic pipeline over historical candles:

```
Candle → FeatureEngine → Scanner → Candidate → simulated outcome
```

Measure: forward returns, MFE / MAE, hit rate, expectancy, drawdown, turnover,
transaction costs, slippage sensitivity, time-of-day performance, regime
sensitivity.

**This becomes the baseline against which any AI contribution is measured.**

Its correctness rests on a property already established and tested in the
feature engine — that batch and incremental feeding produce identical
snapshots. Replay is therefore not an approximation of live behaviour; below
the scanner it is the same computation over the same objects.

Build this immediately after the scanner and before the AI layer. An AI layer
added before there is a measured deterministic baseline cannot be evaluated,
which makes it indistinguishable from decoration.

### 4.6 AI decision layer — NOT BUILT

GPT receives only structured candidate context, never raw tick streams. Its
purpose is higher-level synthesis: candidate quality, conflicting signals,
market context, regime context, trade / no-trade judgement.

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

Every request and response is journaled (AGENTS.md rule 9), both for audit and
because that record is the raw material for answering whether the layer adds
value.

Still to be decided: model choice; token and latency budget; and behaviour on
timeout or malformed output — where the safe default is to proceed
deterministically without it, which should be stated explicitly rather than
left to implementation accident.

### 4.7 Deterministic risk engine — NOT BUILT

**Absolute veto layer.** Max risk per trade, max daily loss, max open
positions, max exposure, max trades per symbol, spread and liquidity limits,
cooldown, stale-data checks, disconnect handling, kill switch.

**Position sizing must be deterministic** (AGENTS.md rule 8). No LLM output
bypasses this layer.

The control categories are settled; their numeric thresholds are not, and
should be set from replay evidence rather than intuition.

### 4.8 Position manager — NOT BUILT

Own the lifecycle of an accepted trade: entry state, stop loss, take profit,
time exit, partial fills, position reconciliation, forced close.

**It must continue functioning even when the AI layer is unavailable.** An open
position with a live stop cannot depend on a network call to a language model.
This is a hard availability boundary, not a preference: it means the position
manager holds its own complete exit logic rather than asking anything upstream
what to do.

### 4.9 Execution engine — DEFERRED

Translate approved deterministic trade instructions into broker orders. Order
type, limit price logic, slippage protection, partial fills, retries,
idempotency, broker reconciliation.

**Live execution remains disabled until research, replay and shadow validation
are complete.** AGENTS.md: *"Do not implement live order placement during the
initial development phase."* The broker abstraction currently exposes no order
method, which is the point.

### 4.10 Journal / observability — NOT BUILT

Every important event should eventually be persisted: market snapshot,
features, scanner decision, AI request/response, risk decision, orders, fills,
position state, PnL, errors, latency.

SQLite is sufficient initially.

**This data is required to determine whether each layer actually adds value.**
The journal is not an operational nicety added at the end — it is the
measurement instrument for the project's central question, and every layer
above the feature engine should be built with its journal records defined
alongside it.

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


## 5. Research principle

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


## 6. Cross-cutting concerns

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
a malformed tick is a bug, a duplicate is a fact of network life.

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


## 7. The two data paths

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


## 8. Known limitations

Deliberate, recorded, non-blocking. The two with architectural consequences:

**`volume_ratio_20` is unavailable for the first 20 minutes of every session**
(09:15–09:35 IST) because the baseline is same-session scoped. That is prime
scanning time. The real fix is a time-of-day seasonality-adjusted RVOL —
comparing 09:20 volume against *historical 09:20s* rather than against the
previous 20 minutes. Any scanner leaning on relative volume at the open needs
this first.

**Session VWAP is strict.** One candle with `volume=None` disables VWAP for the
remainder of that session rather than reporting a subtly wrong number. Correct,
but it means a single broker gap costs a feature for the day.

The full list, including minor items, is section 10 of
[`handover.txt`](handover.txt).


## 9. Current build boundary

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
Deterministic Scanner
Historical Replay / Strategy Evaluation
Shadow Trading + Journal
AI Decision Layer
Deterministic Risk
Position Management
Execution
```

**No layer should be skipped merely to reach live trading faster.**

One qualification on "implemented": the live market feed and tick normalization
exist and are exercised by `check_stream`, but the sustained
live-tick → candle → state path has not yet been run during market hours. See
section 7.

Mapped to the README roadmap, items 1–3 and 5 are complete, item 4 awaits a
weekday market-hours run, and item 6 is next.


## 10. Document map

| Document | Holds | Location fixed by |
|---|---|---|
| `docs/ARCHITECTURE.md` | long-term system intent | — |
| `AGENTS.md` | mandatory engineering/safety rules | agent tooling reads it at repo root |
| `README.md` | setup and usage | `pyproject.toml` `readme` key; GitHub |
| `docs/handover.txt` | current implementation state / next task | — |

`AGENTS.md` and `README.md` stay at the repository root for functional reasons,
not stylistic ones: agent tooling loads `AGENTS.md` from the root, and
`pyproject.toml` references `README.md` by path.
