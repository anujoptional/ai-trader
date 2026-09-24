# AI Trader — Target System Architecture

Canonical description of the intended system. Read this before making
architectural or cross-module changes.

**Starting a fresh session? Read in this order.** The system is described across
five documents, each answering a different question. Reading the wrong one first
is how an agent rebuilds something that already exists, or breaks a contract it
never saw.

| Question | Where it is answered |
|---|---|
| What are we building, and why this shape? | this document, sections 1–4 |
| What am I forbidden to do? | [`../AGENTS.md`](../AGENTS.md) — binding, short, read in full |
| What exists today, and what is next? | sections 2.3 and 11 here, then [`handover.txt`](handover.txt) section 11 |
| Why is the code like *that*? | [`handover.txt`](handover.txt) — the traps, the reasons, the debt |
| How do I run it? | [`../README.md`](../README.md) |
| Is a broker response shaped how I think? | [`LIVE_API_SAMPLES.md`](LIVE_API_SAMPLES.md) — real captures, answerable with the market closed |
| Are the indicators numerically right? | [`FEATURE_VALIDATION.md`](FEATURE_VALIDATION.md) |

**Current versus final, at a glance.** Section 2.1 draws the finished pipeline.
Section 2.3 says which parts of it exist. Section 4 gives every layer's
responsibility whether or not it is built, with the status in its heading. The
gap between today and the target is the difference between those, and section 11
states it in build order along with what "built" is and is not claiming. None of
that has to be inferred by reading the code.

A standing rule, from `handover.txt` section 12: **when a change falsifies a
passage in this document or the README, fix it in the same change.** Stale
architecture text is worse than none, because it is trusted.


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
| Broker adapters | `broker/` | [4.1](#41-broker-adapters--broker--built-read-only) | validated live | read-only; no order path exists |
| Market layer | `market/` | [4.2](#42-market-layer--market--built) | validated live | supervisor's failure path proven live, its recovery path not yet |
| Feature engine | `features/` | [4.3](#43-feature-engine--features--built) | validated live | 47 features cross-checked against TradingView by hand |
| Transaction costs + sizing | `costs/` | [4.4](#44-deterministic-scanner--scanner--built) | tested offline | two schedules modelled; neither reconciled against a real contract note; spread not modelled |
| Deterministic scanner | `scanner/` | [4.4](#44-deterministic-scanner--scanner--built) | tested offline | thresholds are convention, not measurement; no `check_*` CLI yet |
| Diagnostic CLIs | `cli/` | [4.11](#411-diagnostic-clis--cli--built) | validated live | one per built layer except `scanner/` |
| Replay / research | — | [4.5](#45-replay--research-engine--next) | **next** | source-independence, its precondition, is pinned by tests |
| AI decision layer | — | [4.6](#46-ai-decision-layer--not-built) | not built | must not be started before replay; see section 1.3 |
| Deterministic risk | — | [4.7](#47-deterministic-risk-engine--not-built) | not built | AGENTS.md rules 7 and 8 live here |
| Position manager | — | [4.8](#48-position-manager--not-built) | not built | owns `PortfolioState` construction |
| Execution engine | — | [4.9](#49-execution-engine--deferred) | deferred | by AGENTS.md, not by sequencing |
| Journal / observability | — | [4.10](#410-journal--observability--not-built) | not built | AGENTS.md rule 9 |

Read down the Maturity column and the current state of the project is the
answer: everything from the broker to the scanner exists, nothing downstream of
it does, and replay is the next thing to build. Section 11 says the same in
build order and explains why that order is not negotiable.


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
is the common cash-segment value and is overridable. Defaulting it is legitimate
on the same grounds as the fee schedule — a published market fact, not an
opinion — which is exactly the distinction that denies `max_atr_multiple` a
default.

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
placeholder awaiting section 4.5 rather than a measurement (section 7).
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

### 4.5 Replay / research engine — NEXT

Run the exact deterministic pipeline over historical candles:

```
Candle → FeatureEngine → Scanner → Candidate → simulated outcome
```

Measure: forward returns, MFE / MAE, hit rate, expectancy, drawdown, turnover,
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
| `check_scanner` | backfill or live → `FeatureEngine` → `Scanner` |
| `check_costs` | round-trip cost, fixed-clip sizing, exit prices |

The seven broker-facing checks share exit codes: **0** success, **1**
broker/session failure, **2** configuration error. Each backfills the most
recent completed NSE session, so they work outside market hours and are the
fastest end-to-end smoke test.

`check_costs` is the exception and deliberately so. It touches no broker and
backfills nothing — the cost model is arithmetic over a published schedule — so
it has no configuration to get wrong and no exit code 2, and it prints the same
figures at midnight on a Sunday as at 09:20 on a Tuesday. It exits **1** only
on an unusable price argument. The numbers that decide whether a strategy is
viable at a given clip should not be reachable only while the market is open.

`check_features --export-csv PATH` exists so feature values can be compared
against a trusted external implementation before a scanner is built on them.

**`check_scanner` is the only check with a live mode, and the only one that
runs the full stack.** By default it backfills the most recent completed
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
is a time-of-day seasonality-adjusted RVOL — comparing 09:20 volume against
*historical 09:20s* rather than against the previous 20 minutes. Any scanner
leaning on relative volume at the open needs this first.

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
```

Next, in order:

```
Historical Replay / Strategy Evaluation
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
the engine that measures them is section 4.5. Every threshold in `rules.py` is a
conventional level chosen so the layer could be built, not a number measured on
this market, and section 7 applies to all of them. Until replay exists, "the
scanner works" means the rules read what they claim to read and decline when
they should; it does not mean any hypothesis in it is worth acting on.

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

Mapped to the README roadmap, items 1–6 are complete and item 7 is next. The
transaction cost model is not a roadmap item of its own; it was built as a
precondition for item 6's feasibility screen and will be the thing item 7 scores
net expectancy with.


## 12. Document map

| Document | Holds | Answers | Location fixed by |
|---|---|---|---|
| `AGENTS.md` | mandatory engineering/safety rules | what may not be done, ever | agent tooling reads it at repo root |
| `docs/ARCHITECTURE.md` | long-term system intent | what is being built and why this shape | — |
| `docs/handover.txt` | current implementation state / next task | where the work stands and what bit next | — |
| `README.md` | setup and usage | how to run it | `pyproject.toml` `readme` key; GitHub |
| `docs/LIVE_API_SAMPLES.md` | captured Groww request/response samples | what the broker actually returns | — |
| `docs/FEATURE_VALIDATION.md` | indicator cross-check against TradingView | whether the numbers are right | — |

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

`AGENTS.md` and `README.md` stay at the repository root for functional reasons,
not stylistic ones: agent tooling loads `AGENTS.md` from the root, and
`pyproject.toml` references `README.md` by path.
