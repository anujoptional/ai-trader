# AI Trader — Architecture

Status: the upper half of this pipeline is built and validated; the lower half
is designed only as far as its invariants. This document states both, and is
explicit about which is which, so nothing below gets built on an assumption
that was never actually decided.

Companion documents: [`handover.txt`](handover.txt) for current state and
operational detail, [`../AGENTS.md`](../AGENTS.md) for the binding safety
rules, [`../README.md`](../README.md) for setup and CLI usage.


## 1. Purpose

An AI-assisted intraday trading research system for Indian equities (NSE).
Broker today is Groww; Zerodha Kite is a plausible future swap.

The system exists to answer one question: **does an LLM add measurable value
over a deterministic baseline?** That question is only answerable if the
deterministic baseline exists and is trustworthy first. Hence the build order,
and hence the heavy investment in exactness at the bottom of the stack.

Nothing here places an order. Nothing should until the whole pipeline has been
proven end to end.


## 2. Design principles

These are the constraints that shaped every layer. They are not preferences.

**Determinism below the LLM, and above it.** The LLM sits between two
deterministic layers. The scanner decides what it is allowed to see; risk
decides what its output is allowed to become. It can rank and reject. It can
never size a position, invent a candidate, or widen a stop. This is
AGENTS.md rules 7 and 8, expressed structurally rather than by convention.

**Exact arithmetic, not fast arithmetic.** Every price and indicator is a
`Decimal` under a pinned context. Binary floats accumulate error differently
depending on operation order, which makes a backtest and a live run disagree
for reasons unrelated to the market. Measured cost is ~8.3 µs per update —
about 4.2 ms per minute across 500 instruments — which is irrelevant at
one-minute resolution.

**Reproducible by hand.** Any number a strategy might act on should be
derivable on paper from the candles that produced it. This is why indicators
are hand-written rather than pulled from TA-Lib or pandas-ta: not
invented-here, but because a scanner firing on an opaque number is a scanner
nobody can debug at 09:20 with money at stake.

**Broker specifics stay behind an abstraction.** The broker layer is the only
code that knows Groww exists. Everything above it consumes normalized types.
AGENTS.md rule 10.

**Layers own their own concurrency.** Broker SDKs deliver ticks on their own
feed threads. Every stateful layer takes its own `Lock` and invokes callbacks
*outside* that lock, so no layer may assume its caller serializes it.

**Unavailable beats approximately right.** A feature that cannot be computed
honestly reports `None`, never a plausible substitute. Consumers check
readiness flags rather than treating `None` as zero. This costs coverage and
buys the ability to trust a number when it does appear.


## 3. The pipeline

```
   Groww  (broker/)
     │
     ├── Historical candles ──────────────┐
     │                                    │
     └── Live ticks                       │
             │                            │
        CandleBuilder  (market/)          │     BUILT
             │                            │
             ▼                            ▼
          MarketState  (market/)  ◄───────┘     BUILT
             │
             ▼
        FeatureEngine  (features/)              BUILT
             │
             ▼
   Deterministic Scanner                        NOT BUILT — not specified
             │
             ▼
        Candidates
             │
             ▼
         GPT Brain                              NOT BUILT — not specified
             │
             ▼
   Deterministic Risk                           NOT BUILT — not specified
             │
             ▼
         Execution                              NOT BUILT — explicitly deferred
```

Note the join: historical candles reach `MarketState` **directly**, bypassing
`CandleBuilder` entirely. Only live ticks are aggregated. Section 7 covers why
that distinction matters more than it looks.

| Layer | Module | Status | Tests |
|---|---|---|---|
| Broker abstraction | `broker/__init__.py` | built | via CLI + mocks |
| Groww client | `broker/groww.py`, `groww_stream.py` | built, read-only | live-verified |
| Candle aggregation | `market/candles.py` | built | yes |
| Volume differencing | `market/volume.py` | built | yes |
| Market state | `market/state.py` | built | yes |
| Feature engine | `features/` | built | yes |
| Scanner | — | **not started** | — |
| GPT brain | — | **not started** | — |
| Risk | — | **not started** | — |
| Execution | — | **deferred by AGENTS.md** | — |


## 4. Built layers

### 4.1 Configuration — `config.py`

```python
class ConfigurationError(RuntimeError): ...
class GrowwSettings(BaseModel):          # frozen
    totp_token: SecretStr
    totp_secret: SecretStr

def load_groww_settings(environ: Mapping[str, str] | None = None) -> GrowwSettings
```

Credentials are `SecretStr`, so `repr()` masks them and an accidental log line
or traceback cannot leak a token — AGENTS.md rule 5 enforced by the type rather
than by reviewer discipline. `load_groww_settings` names only the *missing*
variables in its error, never a value.

Environment variables (see `.env.example`):

| Variable | Used by |
|---|---|
| `GROWW_TOTP_TOKEN`, `GROWW_TOTP_SECRET` | broker authentication |
| `OPENAI_API_KEY` | reserved for the GPT layer; unused today |
| `GITHUB_PAT`, `GITHUB_USERNAME` | `scripts/sync.sh` only |

`.env` is gitignored and must never be committed, printed, or inspected —
including by agents (AGENTS.md rule 11).

### 4.2 Broker abstraction — `broker/`

`ReadOnlyBroker` is a `Protocol`, not a base class, so a Kite implementation
never imports Groww code. The normalized types are frozen slots dataclasses:

```
Instrument  LastTradedPrice  MarketTick  MarketQuote  OHLCVCandle
BrokerProfile  CandleInterval(StrEnum)  ReadOnlyBroker(Protocol)
```

"Read-only" is a property of the abstraction itself: there is no order-placing
method to call. That is the structural half of AGENTS.md rules 1 and 2 — the
other half is that no such method gets added until live trading is explicitly
enabled.

Groww quirks absorbed here rather than leaked upward:

- Historical candle volumes are **per-candle**; the live LTP feed reports
  **cumulative** day volume. `market/volume.py` reconciles them.
- The historical endpoint is **end-inclusive**.
- Sessions have genuine gaps — a minute with no prints yields no candle. Never
  assume contiguity anywhere above this layer.

### 4.3 Candle aggregation — `market/candles.py`

Turns a live tick stream into independent one-minute `Candle` objects.

**The first minute observed for each instrument is discarded.** A stream is
joined at an arbitrary moment, so that minute is a fragment: its open is
whichever tick happened to arrive first, and its high/low span only the watched
portion. Downstream it would be indistinguishable from a real candle, which
makes it worse than no candle at all. It is also the one minute that could
never carry volume, since cumulative differencing needs a prior reading — so
one rule fixes both defects.

The discarded minute is still recorded as finalized. Omitting that would let a
late tick reopen it as a second, smaller fragment, costing a real candle.

This applies **per instrument**, and to the live path only. A mid-session
subscription costs that one symbol its first minute; a restart at 14:05
discards 14:05. Historical backfill never touches this code.

### 4.4 Market state — `market/state.py`

Normalized rolling history per instrument, fed from both paths.

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
broker's feed thread. That callback is the intended integration point for
`FeatureEngine` — the two layers are deliberately not coupled directly.

### 4.5 Feature engine — `features/`

Incremental, bounded-memory, `Decimal`-exact indicators.

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
produce equal snapshots — asserted by test. Without this, a backtest and a live
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
discarded once seeded. Tracking 500 symbols is bounded and flat.

### 4.6 Diagnostic CLIs — `cli/`

| Command | Checks |
|---|---|
| `check_groww` | authentication and profile |
| `check_historical_data` | historical candle retrieval |
| `check_market_data` | quotes and LTP |
| `check_stream` | live tick streaming |
| `check_market_state` | backfill → `MarketState` |
| `check_features` | backfill → `MarketState` → `FeatureEngine` |

Uniform exit codes: **0** success, **1** broker/session failure, **2**
configuration error. Each backfills the most recent completed NSE session, so
they work outside market hours and are the fastest end-to-end smoke test.

`check_features --export-csv PATH` exists so feature values can be compared
against a trusted external implementation before a scanner is built on them.


## 5. Unspecified layers

Everything below is **shape without specification**. The invariants are fixed;
the content is not. Anyone building here should write a spec first — the
FeatureEngine was built from a detailed written spec in a single pass, and that
was not a coincidence.

### 5.1 Deterministic scanner — blocked

Fixed: consumes `FeatureSnapshot`, never raw candles. Respects readiness flags
rather than treating `None` as zero. Fully deterministic and reproducible.
Emits `Candidate` objects. Is the sole gate on what the LLM may see.

Open, and blocking:

- **What strategy is this?** The feature set is deliberately generic and would
  serve momentum, mean-reversion, or opening-range-breakout equally well. The
  scanner cannot be designed without this answer, and it is the single most
  important undecided question in the system.
- Universe: which instruments, chosen how, refreshed when?
- What a `Candidate` carries — the triggering features, a direction, a score?
- Time-of-day handling. `volume_ratio_20` is unavailable for the first 20
  minutes of every session, which is prime scanning time (see section 8).

### 5.2 GPT brain — not specified

Fixed: receives only scanner-approved candidates. Output is advisory. Cannot
size, cannot invent a candidate, cannot alter risk parameters. Every call and
response is journaled (AGENTS.md rule 9).

Open: request/response schema; whether it ranks, filters, or annotates; model
choice; token and latency budget; behaviour on timeout or malformed output
(the safe default — proceed without it — should be explicit); how its
contribution is isolated for the "does it add value?" measurement, which is the
whole point of the project.

### 5.3 Deterministic risk — not specified

Fixed: position sizing is deterministic code, never the LLM (AGENTS.md rule 8).
Applies after the brain. No LLM output bypasses it.

Open: sizing model; per-trade and daily loss limits; max concurrent positions;
stop and target placement; correlation limits; what happens when a limit binds.

### 5.4 Shadow trading and journaling — not specified

Fixed: shadow/paper is the default mode (AGENTS.md rule 1). Every decision is
journaled (rule 9). SQLite if a database is needed; nothing heavier without a
demonstrated requirement.

Open: schema, granularity, whether the journal is the backtest substrate.

### 5.5 Execution — explicitly deferred

AGENTS.md: *"Do not implement live order placement during the initial
development phase."* Roadmap item 10 places it after backtest validation. The
broker abstraction currently has no order method, which is the point.


## 6. Cross-cutting concerns

**Time.** Timestamps are UTC internally. IST (`Asia/Kolkata`) is used only for
session boundaries and the VWAP session key. NSE regular session is
09:15–15:30 IST = 03:45–10:00 UTC: 375 one-minute slots, plus a closing-auction
print at 15:30 IST. Real sessions run short of 375 because of genuine gaps.

**Numerics.** `FEATURE_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)`,
installed per-computation via `localcontext`. It inherits the default traps, so
`DivisionByZero`, `InvalidOperation` and `Overflow` **raise** rather than
producing NaN. Consequently every division is explicitly guarded against a zero
or `None` denominator before it executes. `localcontext` installs a copy, so
sharing the context across threads is safe.

**Concurrency.** `CandleBuilder`, `MarketState` and `FeatureEngine` each hold
their own `Lock`. Callbacks fire outside the lock that protects the state they
report on, which avoids holding a lock across user code.

**Errors.** Invalid input raises (`InvalidTickError`, `ConfigurationError`);
*stale or repeated* input is counted and dropped. The distinction is
deliberate: a malformed tick is a bug, a duplicate is a fact of network life.

**Secrets.** Environment only, `SecretStr` at the boundary, never logged, never
committed, never inspected by agents. `scripts/sync.sh` passes the GitHub PAT
through `GIT_ASKPASS` — never into `.git/config`, a remote URL, or a credential
helper — and refuses to run against a non-GitHub remote.

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
| First minute | present | **discarded** |
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


## 9. Build order

From `README.md`, items 1–5 complete:

1. ✅ Read-only Groww connection
2. ✅ Account/profile retrieval
3. ✅ Historical market data
4. ⏳ Live market data — **needs a weekday market-hours run**
5. ✅ Normalized market-state snapshots
6. ⬜ Deterministic strategy/scanner layer ← next, blocked on section 5.1
7. ⬜ Shadow trading and journaling
8. ⬜ OpenAI decision engine
9. ⬜ Backtest and evaluate whether the AI adds measurable value
10. ⬜ Only after validation, consider live execution

Item 9 is the actual deliverable. Items 1–8 exist to make it answerable.


## 10. Document map

| Document | Holds | Location is fixed by |
|---|---|---|
| `AGENTS.md` | binding safety rules | agent tooling discovers it at repo root |
| `README.md` | setup, CLI usage, roadmap | `pyproject.toml` `readme` key; GitHub |
| `docs/ARCHITECTURE.md` | this — intent, contracts, open questions | — |
| `docs/handover.txt` | current state, gotchas, debt, next steps | — |

`AGENTS.md` and `README.md` stay at the repository root for functional
reasons, not stylistic ones: agent tooling loads `AGENTS.md` from the root, and
`pyproject.toml` references `README.md` by path.
