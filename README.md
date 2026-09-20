# AI Trader

AI-assisted intraday trading research system.

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the intended system in
full: the pipeline, each layer's contract, and which downstream decisions are
still open. [`docs/handover.txt`](docs/handover.txt) carries the current state,
environment gotchas, and known debt. [`AGENTS.md`](AGENTS.md) holds the binding
safety rules.

## Development setup

Python 3.12 is required.

Create and activate the virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the project and its development dependencies:

```bash
python -m pip install --editable '.[dev]'
```

Run the tests:

```bash
python -m pytest
```

Run Ruff checks:

```bash
python -m ruff check .
```

Run Ruff formatting verification:

```bash
python -m ruff format --check .
```

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

## Check Groww market state

Run the market-state check manually:

```bash
python -m ai_trader.cli.check_market_state
```

This exercises the whole market-state component end to end. It backfills a
recent completed NSE session of one-minute RELIANCE candles, then folds live LTP
ticks into the same state object, aggregating them into further one-minute
candles and differencing the feed's cumulative volume into per-minute volume.
Output is a JSON summary of the resulting state: the backfilled trading date and
candle count, the live tick count, the number of retained candles, late-tick and
duplicate-candle counters, the latest price, and the first and last candles.
Outside market hours no ticks arrive and the check still passes, reporting the
backfilled state alone.

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
folds each one through the feature engine, computing returns, EMAs and their
slopes, RSI, MACD, ATR, rolling extremes, VWAP and a relative volume ratio.
Output is a JSON summary of the latest snapshot alone: the trading date, candle
counts, duplicate and out-of-order candle counters, the candle itself, every
derived value rounded to six decimal places, and a readiness flag per feature.
A feature without enough history behind it is reported as null rather than
guessed. The exported CSV holds one row per candle at full precision, and an
existing file is refused rather than replaced unless `--overwrite` is passed.

The check exits 0 on success, 1 when the broker, the session lookup or the
export fails, and 2 on a configuration or usage error.

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

The deterministic feature layer that item 6 builds on is in place; see
"Check feature engine" above.
