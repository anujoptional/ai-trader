# AI Trader

AI-assisted intraday trading research system for Indian equities.

**The goal is medium-frequency intraday trading: many small round trips per
session on liquid names, each exited as soon as it is a little ahead of what the
round trip cost.** Both directions are in scope. A trade buys the smallest whole
number of shares worth at least ₹1,00,000 and targets **0.2% above the buy
price** — gross, so after a round trip costing roughly 0.0827% at that size it
keeps about 0.1173%, near ₹117.32. Small per trade, and the thesis is that it
repeats often enough across a session to matter. Whether it actually does is
what the replay engine (roadmap item 7) is being built to measure; nothing here
has traded.

**New to this repository?** Read
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first — its opening gives the
reading order for everything else, and its sections 2.3 and 11 tell you what
exists today versus what is still to build.

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the intended system in
full: the pipeline, each layer's contract, and which downstream decisions are
still open. [`docs/handover.txt`](docs/handover.txt) carries the current state,
environment gotchas, and known debt.
[`docs/LIVE_API_SAMPLES.md`](docs/LIVE_API_SAMPLES.md) records a real
request/response sample for every Groww API this system calls, captured against
a live session, so broker work is not blocked when the market is closed.
[`docs/FEATURE_VALIDATION.md`](docs/FEATURE_VALIDATION.md) records the
independent accuracy check of the 47 derived features and the conventions that
differ from a charting package.
[`AGENTS.md`](AGENTS.md) holds the binding safety rules.

## Development setup

Python 3.12 is required — the code uses PEP 695 generic syntax, and
`pyproject.toml` pins `>=3.12,<3.13`. The environment is managed by
[uv](https://docs.astral.sh/uv/), which reads the interpreter version from
`.python-version` and fetches it if the machine does not already have it.

One command builds everything:

```bash
scripts/bootstrap.sh
```

That creates `.venv`, installs the project in editable mode with its dev
extras, and bounds every version by `constraints.txt` so a fresh machine gets
the package set the gates were last validated against rather than whatever PyPI
is serving today. It is safe to re-run.

### The three gates

```bash
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

On Windows the interpreter is `.venv/Scripts/python.exe`. Invoke it by path on
both platforms rather than activating the venv and typing `python`: on Windows a
bare `python` is intercepted by the Store alias and will not be this
interpreter.

Note that `.venv` is created by uv and therefore has **no `pip` module** —
`python -m pip install ...` fails with `No module named pip`. Use
`uv pip install --python .venv/bin/python ...` if you need to add something by
hand.

The command examples further down this file abbreviate the interpreter to
`python`. Each one means the venv interpreter above.

### Reproducibility

`constraints.txt` records the exact versions the gates were last run against,
regenerated with `uv pip freeze`. It is a constraints file, not a requirements
file: `pyproject.toml` still decides *what* is installed, and this only bounds
*which version* of whatever the resolver picks. Entries for packages that are
not selected are ignored, which is what makes a set captured on Windows safe to
apply unchanged on Linux.

A `uv.lock` would be stricter and is the intended endpoint, but generating one
needs network access to `files.pythonhosted.org`, which the current development
machine cannot reach.

### Continuous integration

`.github/workflows/ci.yml` runs the same three gates on every push and pull
request, across both `ubuntu-latest` and `windows-latest`. The two-OS matrix is
not ceremony: this codebase resolves `Asia/Kolkata` to find NSE session
boundaries, and `zoneinfo` reads that from the operating system on Linux but
from the `tzdata` package on Windows. No CI step supplies broker credentials,
and none needs to — every test stubs the broker, so a test that reaches the
network fails there.

Copy `.env.example` to `.env` for local configuration. Never commit `.env` or
place real credentials in `.env.example`.

## Check Groww read-only access

After adding the required Groww TOTP environment variables locally, run the
profile check manually:

```bash
python -m ai_trader.cli.check_groww
```

This command authenticates with Groww and retrieves only the user profile. Its
output is restricted to exchange enablement, active segments, and DDPI status.

## Check Groww market data

Run the live market-data check manually:

```bash
python -m ai_trader.cli.check_market_data
```

This prints latest prices for NSE RELIANCE and NIFTY, plus a normalized detailed
quote for RELIANCE.

Any of these checks may pause for a few seconds before printing. Groww answers
roughly half of all requests with a malformed body that is not a real error —
the identical call succeeds moments later — so every broker call is retried
automatically, including the login that each check performs first. A check that
ultimately fails has exhausted eight attempts over about twelve seconds, which
is a genuine outage rather than the usual flakiness. A failure during login
prints `Groww authentication failed.`, which is distinct from the per-check
messages below and points at the broker or your credentials, not at the data.

Run the historical market-data check manually:

```bash
python -m ai_trader.cli.check_historical_data
```

This searches backward for a recent completed NSE session, requesting one-minute
RELIANCE CASH candles from 09:15 to 09:30 Asia/Kolkata time. Weekends are skipped
locally, and empty weekday results are treated as potential exchange holidays.
The search stops after at most 10 weekdays. Output contains only the selected
trading date, candle count, and the first and last candles; normalized candle
timestamps are printed in UTC.

## Check Groww streaming data

Run the bounded RELIANCE LTP stream check manually:

```bash
python -m ai_trader.cli.check_stream
```

The command resolves the NSE RELIANCE exchange token, subscribes only to its
CASH LTP feed, and prints normalized price ticks with UTC timestamps. It stops
after five ticks or 30 seconds and always unsubscribes. If no ticks arrive, it
reports that the market may be closed and exits normally.

If the feed cannot be reached at all, the connect is abandoned after 30 seconds
and the command exits 1 with a single line: `Groww live feed unreachable; the
stream connection failed.` As of 2026-09-22 this is the only outcome, in or out
of market hours — Groww's tick transport accepts the socket but never completes
its handshake, which is a server-side fault rather than anything this code can
retry around. See the market-open addendum in
[docs/LIVE_API_SAMPLES.md](docs/LIVE_API_SAMPLES.md) for the signature and for
what remains validatable meanwhile.

## Check Groww market state

Run the market-state check manually:

```bash
python -m ai_trader.cli.check_market_state
```

This exercises the whole market-state component end to end. It backfills a
recent completed NSE session of one-minute RELIANCE candles, then folds live LTP
ticks into the same state object, aggregating them into further one-minute
candles. Output is a JSON summary of the resulting state: the backfilled trading
date and candle count, the live tick count, the number of retained candles,
late-tick and duplicate-candle counters, the latest price, and the first and
last candles. Outside market hours no ticks arrive and the check still passes,
reporting the backfilled state alone.

Groww's LTP stream carries no volume. It advertises an optional volume field
but does not populate it — across 114 ticks measured during live trading on
2026-09-21 it was absent from every one. The only live volume Groww serves is
the running session total on the REST quote endpoint, so a `VolumePoller` reads
that total on a background thread and stamps it onto each bare tick before the
tick reaches the candle builder, which differences consecutive totals into a
per-minute figure. The check reports the outcome under `volume_source`: how many
ticks were stamped, the latest polled total, and the poll failure and regression
counts.

Volume is an enrichment, never a precondition. A failing quote endpoint costs
volume and nothing else — prices keep flowing, the check still exits 0, and the
failure is reported rather than swallowed. The check's 15-second tick window is
shorter than a minute, so it normally proves that ticks are being stamped
rather than that a stamped candle was emitted.

## Check feature engine

Run the feature-engine check manually:

```bash
python -m ai_trader.cli.check_features
```

Optionally export every candle's features for offline comparison:

```bash
python -m ai_trader.cli.check_features --export-csv features.csv
```

This backfills a recent completed NSE session of one-minute RELIANCE candles and
folds each one through the feature engine, computing 47 derived features:
returns, EMAs and their slopes, RSI, MACD, true range and ATR, the directional
movement index and ADX, rolling extremes and the distances to them, a simple
moving average with its Bollinger band, bandwidth and percent-B, VWAP with its
turnover-weighted deviation, a relative volume ratio, on-balance volume, and the
session frame — open, high, low, range position, elapsed minutes, cumulative
volume and the opening range.
Output is a JSON summary of the latest snapshot alone: the trading date, candle
counts, duplicate and out-of-order candle counters, the candle itself, every
derived value rounded to six decimal places, and a readiness flag per feature.
A feature without enough history behind it is reported as null rather than
guessed. The exported CSV holds one row per candle at full precision, and an
existing file is refused rather than replaced unless `--overwrite` is passed.

Read the per-feature readiness flags, not just `core_ready`. `core_ready`
covers price and momentum only; it says nothing about the volume-derived
fields. Before the volume poller existed, a live run on 2026-09-21 showed
exactly why that separation matters: `core_ready` stayed true across the
historical-to-live seam while `vwap`, `price_vs_vwap` and `volume_ratio_20`
went null and stayed null for want of volume. A later run the same day, with
the poller stamping ticks, carried all three across the seam populated. A live
consumer must still check each flag it depends on rather than trusting
`core_ready` to cover them.

The check exits 0 on success, 1 when the broker, the session lookup or the
export fails, and 2 on a configuration or usage error.

All 47 features have been checked for numerical accuracy against an
independently written reference over a real session, and all 47 agree. Before
comparing the exported CSV against TradingView, read
[`docs/FEATURE_VALIDATION.md`](docs/FEATURE_VALIDATION.md): the conventions
match a charting package almost everywhere, but `volume_ratio_20` deliberately
excludes the current candle from its own baseline and will therefore differ
from TradingView's relative volume on any spike.

## Check trade costs and sizing

Run the cost check manually:

```bash
python -m ai_trader.cli.check_costs             # a default spread of quotes
python -m ai_trader.cli.check_costs 2450 100    # specific quotes
python -m ai_trader.cli.check_costs --broker zerodha
```

Unlike every other check, this one talks to no broker and needs no market
session — the cost model is arithmetic over a published schedule, so it prints
the same figures at midnight on a Sunday as at 09:20 on a Tuesday. It exits 0
on success and 1 on an unusable price argument or an unknown broker name; it
has no configuration and therefore no exit code 2.

`--broker` takes `groww`, `zerodha` or `kite` and defaults to Groww, which is
the broker this system connects to and not the cheaper of the two. Two
schedules are published and both are modelled, because they differ in exactly
one line — brokerage — and agree on every statutory and exchange charge.

Output is a JSON summary with four parts. The round-trip cost by clip size
shows where the curve bends — brokerage caps at ₹20 *per leg*, so at Groww's
0.1% the cost is a flat 0.2715% of turnover up to ₹20,000 and then falls away,
reaching 0.0827% at ₹1,00,000. The schedule comparison prints both brokers at
every clip size whichever one was selected. The two interpretations block
prices both readings of a stated target. The estimates block sizes the fixed
₹1,00,000 clip at each quote and prints the exits.

**At the configured clip the broker choice costs nothing.** Zerodha charges
0.03% where Groww charges 0.1%, but both cap at ₹20 a leg, so Zerodha's cap
does not bind until ₹66,666.67 and above that figure the two schedules are
identical to the paisa. Below it Zerodha is cheaper and never dearer: a 0.2%
gross capture is a loss at Groww's rates below about ₹28,690 a clip and a gain
at Zerodha's at every size. Since this system trades a one-lakh clip, the swap
named in `AGENTS.md` rule 10 is a configuration change rather than a change of
strategy — which is why the comparison is printed rather than asserted.

Read the estimates with the integral quantity in mind: the clip is a floor, so
a one-lakh clip of a stock at ₹2,450 is forty-one shares and ₹1,00,450 rather
than ₹1,00,000, and every fraction is a fraction of what was actually filled.
Both exits are printed because cost is direction-symmetric and nothing here
picks a side; long exits round up to a tick and short exits round down, always
away from the entry, because rounding to the nearer tick fills a trade that
looks like a win but returns less than the margin that justified it.

Watch `tick_fraction`. At ₹2,450 a five-paisa tick is 0.002% of price and
rounding is a rounding error; at ₹100 it is 0.05% — half of a 0.1% target — and
past that point the exit can only land on the tick grid, so what the trade asks
for depends on where the entry sits in that grid rather than on the target. At
the 0.1% target an entry at ₹100.00 sits exactly on a tick and returns the
₹17.32 it asked for; an entry one tick higher at ₹100.05 must reach ₹100.20 and
returns ₹67.30, roughly four times the target off a five-paisa difference. That
is not a windfall — the trade is asking for a move four times larger and will
fill far less often. Cheap stocks are a coarser instrument for this strategy
than expensive ones.

Every figure carries the same caveat: the rates of both schedules are
transcribed from published tables and neither has been reconciled against a
real contract note, and the spread is not modelled at all. Treat the exit
prices as a floor, not a forecast.

## Sync with GitHub

Set `GITHUB_PAT` in `.env` to a personal access token with `repo` scope, then
pull and push with:

```bash
scripts/sync.sh
```

The token is read from `.env` at invocation time and passed to git through
`GIT_ASKPASS`. It is never written into `.git/config`, a remote URL, or a
credential helper, so running this leaves no token on disk. The script refuses
to run unless `origin` begins with `https://github.com/`, and both commands name
`origin` and the current branch explicitly, so the remote that is checked is the
remote that is contacted even if the branch is configured to track another one.
Any extra arguments are forwarded to `git push` after that pinned remote and
branch.

## Roadmap

Initial milestones:

1. Connect read-only to Groww.
2. Retrieve account/profile information safely.
3. Retrieve historical market data.
4. Retrieve live market data.
5. Build normalized market-state snapshots.
6. Build deterministic strategy/scanner layer.
7. Build the historical replay / strategy evaluation engine.
8. Add shadow trading and journaling.
9. Add the OpenAI decision engine.
10. Add deterministic risk and position management.
11. Evaluate whether the AI adds measurable value, by forward shadow trading
    against the control of taking every candidate.
12. Only after validation, small-capital live; production execution last.

Replay (7) comes before the AI (9) deliberately: it establishes whether the
scanner's candidates have edge at all, which is the control the AI is later
measured against. The ordering here is the same one in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) section 11; keep them in sync.

Items 1–6 are in place. Item 4 was validated against a live market on
2026-09-21: ticks streamed, aggregated into contiguous one-minute candles, and
joined the backfilled history with no late ticks and no duplicates. The same day
the whole chain was run live through the feature engine for the first time,
confirming that price and momentum features cross the historical-to-live seam
intact. The one thing the live stream does not deliver on its own is volume; a
`VolumePoller` now supplies it from the REST quote endpoint, as described under
"Check Groww market state", and a later run the same day carried the
volume-derived features across the seam populated.

The scanner (6) is the exception to all of that. It is tested offline and
deliberately unproven: every threshold in it is a conventional level rather than
a measurement, and item 7 is what turns any of them into evidence.

Alongside the scanner sits a transaction cost model (`costs/`), which is not a
roadmap item but a precondition for one. The objective is many small round trips
per session — long or short — each closed as soon as it is a little ahead of
what the round trip cost, so the first question about any candidate is whether a
move that size is available at all. It cannot be answered with a target
percentage: at Groww's rates the same round trip costs 0.2715% of a ₹20,000
clip and 0.0827% of a ₹1,00,000 one, because brokerage is capped per leg, so
0.2% gross is a loss at the first size and a profit at the second. The scanner
therefore computes the hurdle — `round-trip cost at the configured size + the
margin asked for` — rather than storing one. Two schedules are modelled, Groww
and Zerodha/Kite, and the hurdle is computed from whichever one the caller
passes; at the configured clip they agree to the paisa.

The buying model that settles the size is deliberately simple: one fixed clip
of ₹1,00,000 per entry, and a sell is the whole position. Nothing in the code
can express a partial exit — a single quantity serves both legs — so scaling
out would arrive as a visible change rather than quietly as a new code path.
The clip is a *floor* on turnover, not a budget: shares are indivisible, so an
entry buys the fewest whole shares worth at least a lakh — forty-one shares and
₹1,00,450 of a stock at ₹2,450, one share of anything dearer than the clip. The
scanner prices each name's hurdle on the notional that would actually fill,
which is never below the clip and therefore never costlier, as a fraction, than
the clip figure suggests.

The margin follows from a stated target: sell **0.2% above the buy price**.
That is a gross move, so what it keeps is 0.2% minus the round trip — about
0.1173%, or ₹117.32 on a lakh. `STATED_GROSS_TARGET` is the one place the
figure appears and `SizingPolicy.from_gross_target` does the conversion,
refusing a target its own costs would consume rather than clamping it: at
Groww's rates 0.2% at a ₹20,000 clip is a loss, and the constructor raises.
(At Zerodha's it is not — 0.2% clears at every clip size there — which is why
the schedule is an argument to the conversion rather than an import inside it.)
Converting once at the clip is safe in the only direction that matters — the
clip is the smallest permitted fill and therefore the most expensive as a
fraction, so every real name faces a hurdle at or *below* the stated figure and
keeps at or above the implied margin. The screen that applies it stays off
until a caller also states a `max_atr_multiple`, which has no measured value
and is not given one here. Neither fee schedule has been reconciled against a
real contract note, and the spread is not in either of them at all, so the
hurdle is a floor rather than an estimate.
