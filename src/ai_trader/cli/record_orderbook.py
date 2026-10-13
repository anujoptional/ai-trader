"""Record live five-level depth (and futures OI) to a daily SQLite file.

Read-only Kite quotes: cash equities from the universe plus their near and next
month futures, polled every ``--interval`` seconds until 15:30 IST. Requires a Kite
browser login each day. Stop any time with Ctrl+C; rows already written are kept.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, time
from pathlib import Path

from ai_trader.broker import Instrument
from ai_trader.broker.kite import KiteBroker, KiteBrokerError
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import ConfigurationError, load_kite_login_settings
from ai_trader.orderbook.recorder import DepthRecorder, record


def futures_for(broker: KiteBroker, symbols: set[str], months: int) -> list[Instrument]:
    by_name: dict[str, list[dict]] = {}
    for row in broker.nfo_futures():
        if row["name"] in symbols:
            by_name.setdefault(row["name"], []).append(row)
    chosen = []
    for rows in by_name.values():
        for row in sorted(rows, key=lambda r: r["expiry"])[:months]:
            chosen.append(Instrument("NFO", row["tradingsymbol"]))
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/orderbook"))
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--futures-months", type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument("--until", default="15:30")
    args = parser.parse_args(argv)
    universe = json.loads(args.universe.read_text("utf-8"))
    symbols = {
        e["symbol"] for e in universe["instruments"] if e["role"] == "equity_target"
    }
    try:
        broker = KiteBroker.authenticate(load_kite_login_settings())
    except (ConfigurationError, KiteBrokerError) as error:
        print(f"Kite login failed: {error}")
        return 2
    instruments = [Instrument("NSE", s) for s in sorted(symbols)]
    if args.futures_months:
        instruments += futures_for(broker, symbols, args.futures_months)
    now = datetime.now(INDIA_TIMEZONE)
    until = datetime.combine(now.date(), time.fromisoformat(args.until), INDIA_TIMEZONE)
    recorder = DepthRecorder(args.output_dir / f"{now.date().isoformat()}.sqlite")
    print(
        f"recording {len(instruments)} instruments every {args.interval}s until {until}"
    )
    try:
        stats = record(
            broker,
            instruments,
            recorder,
            interval_seconds=args.interval,
            until=until,
            on_error=lambda error: print(f"poll failed: {type(error).__name__}"),
        )
    except KeyboardInterrupt:
        stats = {"stopped": "keyboard interrupt"}
    finally:
        recorder.close()
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
