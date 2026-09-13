# AI Trader

AI-assisted intraday trading research system.

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

## Roadmap

Initial milestones:

1. Connect read-only to Groww.
2. Retrieve account/profile information safely.
3. Retrieve historical market data.
4. Retrieve live market data.
5. Build normalized market-state snapshots.
6. Build deterministic strategy/scanner layer.
7. Add shadow trading and journaling.
8. Add OpenAI decision engine.
9. Backtest and evaluate whether the AI adds measurable value.
10. Only after validation, consider live execution.
