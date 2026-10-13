"""Current NSE F&O stock universe from Kite's public instrument lists (research)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from ai_trader.broker import Instrument
from ai_trader.broker.kite import kite_symbol
from ai_trader.cli.download_history import CONTEXT_SYMBOLS, write_report
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.history import CandleStore

SOURCES = {
    "fno_underlyings": "https://api.kite.trade/instruments/NFO",
    "instrument_master": "https://api.kite.trade/instruments/NSE",
    "industry": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(nfo: Path, nse: Path, industries: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError("The universe snapshot already exists; use a new path")
    underlyings = sorted(
        {row["name"] for row in read_rows(nfo) if row["segment"] == "NFO-FUT"}
    )
    cash = [row for row in read_rows(nse) if row["exchange"] == "NSE"]
    equities: dict[str, list[dict[str, str]]] = {}
    for row in cash:
        if row["segment"] == "NSE" and row["instrument_type"] == "EQ":
            equities.setdefault(row["tradingsymbol"], []).append(row)
    stocks = [name for name in underlyings if name in equities]
    if not stocks or any(len(equities[name]) != 1 for name in stocks):
        raise ValueError("F&O underlyings must map to one NSE cash instrument each")
    sectors = {row["Symbol"]: row for row in read_rows(industries)}
    entries = []
    for name in stocks:
        item, reference = equities[name][0], sectors.get(name, {})
        entries.append(
            {
                "exchange": "NSE",
                "symbol": name,
                "role": "equity_target",
                "name": reference.get("Company Name") or item["name"],
                "industry": reference.get("Industry"),
                "isin": reference.get("ISIN Code"),
                "instrument_type": item["instrument_type"],
                "instrument_token": int(item["instrument_token"]),
                "kite_symbol": item["tradingsymbol"],
                "tick_size": item["tick_size"],
            }
        )
    aliases = {
        kite_symbol(Instrument("NSE", symbol)): symbol for symbol in CONTEXT_SYMBOLS
    }
    context = [
        row
        for row in cash
        if row["segment"] in ("NSE", "INDICES") and row["tradingsymbol"] in aliases
    ]
    if sorted(aliases[row["tradingsymbol"]] for row in context) != sorted(
        CONTEXT_SYMBOLS
    ):
        raise ValueError("Context instruments are missing or ambiguous")
    for row in sorted(
        context, key=lambda row: CONTEXT_SYMBOLS.index(aliases[row["tradingsymbol"]])
    ):
        entries.append(
            {
                "exchange": "NSE",
                "symbol": aliases[row["tradingsymbol"]],
                "role": "context_only",
                "name": row["name"],
                "instrument_type": row["instrument_type"],
                "instrument_token": int(row["instrument_token"]),
                "kite_symbol": row["tradingsymbol"],
                "tick_size": row["tick_size"],
            }
        )
    validator = CandleStore(destination.parent)
    for entry in entries:
        validator.path_for(Instrument(entry["exchange"], entry["symbol"]))
    document = {
        "schema_version": 2,
        "provider": "kite",
        "retrieved_at": datetime.now(INDIA_TIMEZONE).isoformat(),
        "universe_definition": (
            "Current NSE stock-futures underlyings (Kite NFO list) "
            "plus context instruments"
        ),
        "membership_warning": (
            "Current F&O list, not point-in-time 2020-2026 membership; "
            "survivorship and symbol/corporate-action effects are not corrected"
        ),
        "sources": SOURCES,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (nfo, nse, industries)
        },
        "excluded_underlyings": [name for name in underlyings if name not in equities],
        "instruments": entries,
    }
    write_report(destination, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfo", type=Path, required=True)
    parser.add_argument("--nse", type=Path, required=True)
    parser.add_argument("--industries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = prepare(args.nfo, args.nse, args.industries, args.output)
    roles = [entry["role"] for entry in document["instruments"]]
    print(
        json.dumps(
            {
                "equity_targets": roles.count("equity_target"),
                "context": roles.count("context_only"),
                "excluded": document["excluded_underlyings"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
