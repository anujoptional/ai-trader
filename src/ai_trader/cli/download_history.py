"""Download a stated universe into the existing candle cache, with checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ai_trader.broker import Instrument, ReadOnlyBroker
from ai_trader.broker.kite import (
    KiteAuthenticationError,
    KiteBroker,
    KiteBrokerError,
    kite_symbol,
)
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import ConfigurationError, load_kite_login_settings
from ai_trader.history import (
    CandleStore,
    CandleStoreError,
    last_completed_session_close,
)

CONTEXT_SYMBOLS = (
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "NIFTYIT",
    "NIFTYMIDCAP150",
    "NIFTY500",
    "INDIAVIX",
    "NIFTYAUTO",
    "NIFTYFMCG",
    "NIFTYPHARMA",
    "NIFTYMETAL",
    "NIFTYREALTY",
    "GOLDBEES",
    "SILVERBEES",
    "NIFTYBEES",
    "BANKBEES",
)


def prepare_universe(
    nifty_path: Path, master_path: Path, destination: Path
) -> dict[str, Any]:
    with nifty_path.open(encoding="utf-8-sig", newline="") as handle:
        members = list(csv.DictReader(handle))
    if len(members) != 50 or len({row["Symbol"] for row in members}) != 50:
        raise ValueError(
            "Official Nifty constituent input must contain 50 unique symbols"
        )
    wanted = {row["Symbol"] for row in members} | set(CONTEXT_SYMBOLS)
    aliases = {kite_symbol(Instrument("NSE", symbol)): symbol for symbol in wanted}
    with master_path.open(encoding="utf-8-sig", newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row["exchange"] == "NSE"
            and row["segment"] in ("NSE", "INDICES")
            and row["tradingsymbol"] in aliases
        ]
    master = {aliases[row["tradingsymbol"]]: row for row in selected}
    if len(master) != len(selected):
        raise ValueError("Kite instrument master contains ambiguous symbols")
    if wanted - master.keys():
        raise ValueError(
            "Some requested symbols are absent from the cash instrument master"
        )
    entries = []
    for row in members:
        symbol = row["Symbol"]
        item = master[symbol]
        entries.append(
            {
                "exchange": "NSE",
                "symbol": symbol,
                "role": "equity_target",
                "name": row["Company Name"],
                "industry": row["Industry"],
                "isin": row.get("ISIN Code"),
                "instrument_type": item["instrument_type"],
                "instrument_token": int(item["instrument_token"]),
                "kite_symbol": item["tradingsymbol"],
                "tick_size": item["tick_size"],
            }
        )
    for symbol in CONTEXT_SYMBOLS:
        item = master[symbol]
        entries.append(
            {
                "exchange": "NSE",
                "symbol": symbol,
                "role": "context_only",
                "name": item["name"],
                "instrument_type": item["instrument_type"],
                "instrument_token": int(item["instrument_token"]),
                "kite_symbol": item["tradingsymbol"],
                "tick_size": item["tick_size"],
            }
        )
    document = {
        "schema_version": 2,
        "provider": "kite",
        "retrieved_at": datetime.now(INDIA_TIMEZONE).isoformat(),
        "universe_definition": (
            "Current official Nifty50 snapshot plus context instruments"
        ),
        "membership_warning": (
            "This is not point-in-time 2024-2026 index membership; "
            "survivorship and symbol/corporate-action effects are not corrected"
        ),
        "sources": {
            "nifty50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
            "instrument_master": "https://api.kite.trade/instruments",
        },
        "source_sha256": {
            nifty_path.name: hashlib.sha256(nifty_path.read_bytes()).hexdigest(),
            master_path.name: hashlib.sha256(master_path.read_bytes()).hexdigest(),
        },
        "instruments": entries,
    }
    if destination.exists():
        raise ValueError("The universe snapshot already exists; use a new path")
    write_report(destination, document)
    return document


def download_windows(
    start: datetime,
    end: datetime,
    held: tuple[datetime, datetime] | None,
) -> tuple[tuple[datetime, datetime], ...]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Download bounds must be ordered and timezone-aware")
    step = timedelta(days=28)
    windows = []
    if held is None:
        cursor = start
        while cursor < end:
            stop = min(cursor + step, end)
            windows.append((cursor, stop))
            cursor = stop
        return tuple(windows)
    cursor = held[0]
    while start < cursor:
        beginning = max(start, cursor - step)
        windows.append((beginning, cursor))
        cursor = beginning
    cursor = held[1]
    while cursor < end:
        stop = min(cursor + step, end)
        windows.append((cursor, stop))
        cursor = stop
    return tuple(windows)


def read_universe(path: Path) -> tuple[dict[str, Any], tuple[Instrument, ...]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    instruments = tuple(
        Instrument(item["exchange"], item["symbol"]) for item in document["instruments"]
    )
    if not instruments or len(set(instruments)) != len(instruments):
        raise ValueError("Universe must contain unique instruments")
    validator = CandleStore(Path("data/candles"))
    for instrument in instruments:
        validator.path_for(instrument)
    return document, instruments


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def download(
    broker: ReadOnlyBroker,
    instruments: Sequence[Instrument],
    start: datetime,
    end: datetime,
    cache_dir: Path,
    report_path: Path,
    *,
    now: datetime,
    universe_sha256: str,
    emit: Callable[[str], None] = print,
    max_failures: int = 3,
    refresh: bool = False,
) -> dict[str, Any]:
    if end > last_completed_session_close(now):
        raise ValueError("The requested range includes an unfinished session")
    if max_failures < 1:
        raise ValueError("Failure limit must be positive")
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": now.isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "interval": "1minute",
        "universe_sha256": universe_sha256,
        "quality_scope": "Basic validity and coverage only; no live-fill simulation",
        "coverage_note": (
            "Downloaded means data returned, not complete coverage; "
            "boundary warnings do not distinguish holidays from missing data"
        ),
        "refresh": refresh,
        "instruments": {},
    }
    store = CandleStore(cache_dir, broker)
    failures = 0
    write_report(report_path, report)
    for instrument in instruments:
        symbol = f"{instrument.exchange}:{instrument.trading_symbol}"
        detail: dict[str, Any] = {"status": "running", "completed_windows": []}
        report["instruments"][symbol] = detail
        write_report(report_path, report)
        try:
            windows = download_windows(
                start, end, None if refresh else store.coverage(instrument)
            )
            for index, (beginning, stop) in enumerate(windows, 1):
                bars = store.load(instrument, beginning, stop, now=now, refresh=refresh)
                detail["completed_windows"].append(
                    {
                        "start": beginning.isoformat(),
                        "end": stop.isoformat(),
                        "returned_candles": len(bars),
                    }
                )
                write_report(report_path, report)
                emit(f"{symbol} window {index}/{len(windows)}: {len(bars)} candles")
            bars = store.load(instrument, start, end, now=now)
            regular = [
                bar
                for bar in bars
                if time(9, 15) <= bar.start_time.time() < time(15, 30)
            ]
            counts = Counter(bar.start_time.date() for bar in regular)
            boundary_warnings = []
            if not regular or regular[0].start_time > start:
                boundary_warnings.append("requested_start_not_observed")
            if not regular or regular[-1].end_time < end:
                boundary_warnings.append("requested_end_not_observed")
            detail.update(
                {
                    "status": "downloaded" if bars else "empty",
                    "candles": len(bars),
                    "regular_candles": len(regular),
                    "sessions_observed": len(counts),
                    "first_candle": bars[0].start_time.isoformat() if bars else None,
                    "last_candle_end": bars[-1].end_time.isoformat() if bars else None,
                    "coverage_boundary_warnings": boundary_warnings,
                    "unknown_volume_candles": sum(bar.volume is None for bar in bars),
                    "observed_sessions_not_375_minutes": {
                        str(day): count
                        for day, count in sorted(counts.items())
                        if count != 375
                    },
                    "candle_sha256": hashlib.sha256(
                        store.path_for(instrument).read_bytes()
                    ).hexdigest()
                    if bars
                    else None,
                }
            )
            emit(
                f"{symbol}: {detail['status']}, {len(bars)} candles, "
                f"{len(counts)} sessions"
            )
            if boundary_warnings:
                emit(f"{symbol}: coverage warning: {', '.join(boundary_warnings)}")
            failures = 0
        except (KiteBrokerError, CandleStoreError, ValueError, OSError) as error:
            detail.update({"status": "failed", "error_type": type(error).__name__})
            emit(f"{symbol}: failed ({type(error).__name__}); cached windows retained")
            failures += 1
        write_report(report_path, report)
        if failures >= max_failures:
            emit("Stopping after repeated instrument failures; rerun to resume")
            break
    report["instrument_status_counts"] = dict(
        Counter(detail["status"] for detail in report["instruments"].values())
    )
    report["not_attempted"] = [
        f"{instrument.exchange}:{instrument.trading_symbol}"
        for instrument in instruments
        if f"{instrument.exchange}:{instrument.trading_symbol}"
        not in report["instruments"]
    ]
    write_report(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/candles"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--nifty50-source", type=Path)
    parser.add_argument("--instrument-source", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch only the requested range, including previously empty responses",
    )
    args = parser.parse_args(argv)
    try:
        if args.nifty50_source or args.instrument_source:
            if not args.nifty50_source or not args.instrument_source:
                raise ValueError("Both public source files are required")
            prepare_universe(args.nifty50_source, args.instrument_source, args.universe)
        _, instruments = read_universe(args.universe)
        if args.prepare_only:
            print(
                json.dumps(
                    {"universe": str(args.universe), "instruments": len(instruments)}
                )
            )
            return 0
        if args.symbols:
            requested = set(args.symbols)
            instruments = tuple(
                item
                for item in instruments
                if f"{item.exchange}:{item.trading_symbol}" in requested
            )
            if len(instruments) != len(requested):
                raise ValueError("Requested symbols must be in the stated universe")
        start = datetime.combine(args.start, time(9, 15), INDIA_TIMEZONE)
        end = datetime.combine(args.end, time(15, 30), INDIA_TIMEZONE)
        download_windows(start, end, None)
        settings = load_kite_login_settings()
        broker = KiteBroker.authenticate(settings)
        report = download(
            broker,
            instruments,
            start,
            end,
            args.cache_dir,
            args.report,
            now=datetime.now(INDIA_TIMEZONE),
            universe_sha256=hashlib.sha256(args.universe.read_bytes()).hexdigest(),
            refresh=args.refresh,
        )
    except ConfigurationError:
        print(
            "Kite configuration is unavailable; no credentials were displayed",
            file=sys.stderr,
        )
        return 2
    except KiteAuthenticationError:
        print(
            "Kite authentication failed; no credentials were displayed",
            file=sys.stderr,
        )
        return 1
    except (ValueError, OSError, KeyError, TypeError):
        print("Download configuration or local storage is invalid", file=sys.stderr)
        return 2
    print(json.dumps(report["instrument_status_counts"], sort_keys=True))
    return int(
        bool(report["not_attempted"])
        or any(
            detail["status"] != "downloaded"
            for detail in report["instruments"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
