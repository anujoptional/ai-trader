"""Download Kite's continuous near-month futures day bars with open interest.

Read-only and resumable per symbol. Requires a Kite browser login. Output: one CSV
per underlying (session_date, open, high, low, close, volume, open_interest) and a
manifest with SHA-256 digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

from ai_trader.broker.kite import KiteBroker, KiteBrokerError
from ai_trader.config import ConfigurationError, load_kite_login_settings

HEADER = ["session_date", "open", "high", "low", "close", "volume", "open_interest"]


def download(
    broker: KiteBroker, symbols: list[str], start: date, end: date, output: Path
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for symbol in symbols:
        path = output / f"{symbol}.csv"
        if path.exists():
            results[symbol] = {"status": "existing"}
            continue
        try:
            bars = broker.get_continuous_futures_daily(symbol, start, end)
        except KiteBrokerError as error:
            results[symbol] = {"status": "failed", "error": str(error)}
            continue
        partial = path.with_suffix(".partial")
        with partial.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for bar in bars:
                writer.writerow(
                    [
                        bar.session_date.isoformat(),
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                        bar.open_interest,
                    ]
                )
        partial.replace(path)
        results[symbol] = {
            "status": "downloaded",
            "bars": len(bars),
            "first": bars[0].session_date.isoformat() if bars else None,
            "last": bars[-1].session_date.isoformat() if bars else None,
        }
    for symbol, result in results.items():
        path = output / f"{symbol}.csv"
        if path.exists():
            with path.open("rb") as handle:
                result["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        broker = KiteBroker.authenticate(load_kite_login_settings())
    except (ConfigurationError, KiteBrokerError) as error:
        print(f"Kite login failed: {error}")
        return 2
    return run(broker, args.universe, args.start, args.end, args.output)


def run(
    broker: KiteBroker, universe_path: Path, start: date, end: date, output: Path
) -> int:
    universe = json.loads(universe_path.read_text("utf-8"))
    symbols = [
        entry["symbol"]
        for entry in universe["instruments"]
        if entry["role"] == "equity_target"
    ]
    results = download(broker, symbols, start, end, output)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provider": "kite",
        "series": "continuous near-month futures, day interval, with open interest",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": results,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    failed = [s for s, r in results.items() if r["status"] == "failed"]
    print(f"symbols {len(results)}; failed {len(failed)}: {failed[:10]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
