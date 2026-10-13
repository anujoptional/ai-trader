"""Build immutable TP-only shards from local minutes and vendor daily closes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

from ai_trader.broker import DailyClose, Instrument, ReadOnlyBroker
from ai_trader.cli.download_history import read_universe, write_report
from ai_trader.clock import INDIA_TIMEZONE, session_minute_offset
from ai_trader.features.tp_only import (
    CUTOFF_MINUTES,
    FEATURE_COLUMNS,
    HORIZON_MINUTES,
    MODEL_COLUMNS,
    TP_FRACTION,
    CompletedSession,
    PriceContext,
    TpFeatureState,
    touch_label,
)
from ai_trader.history import CandleStore, last_completed_session_close
from ai_trader.market import Candle

CONTEXT_SYMBOLS = ("NIFTY", "BANKNIFTY", "GOLDBEES", "INDIAVIX")
BANK_SYMBOLS = frozenset(
    ("AXISBANK", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "INDUSINDBK")
)
PHASE_ENDS = {
    "train": date(2025, 6, 30),
    "validation": date(2025, 12, 31),
    "calibration": date(2026, 3, 31),
    "evaluation": date(2026, 9, 28),
}
FEATURE_DEFINITIONS = {
    "log_return_5m": "ln(close[t] / close[t-5m]), contiguous same-session minutes",
    "log_return_previous_10m": "ln(close[t-5m] / close[t-15m]), contiguous minutes",
    "day_return": "close[t] / actual 09:15 open - 1",
    "gap_from_vendor_daily_close": "actual 09:15 open / prior vendor daily close - 1",
    "realized_volatility_20m": "sqrt(sum of 20 squared one-minute log returns)",
    "prior_5_daily_close_return": (
        "prior session vendor close / six-sessions-back vendor close - 1"
    ),
    "prior_5_cutoff_volatility": (
        "sqrt(mean prior five 09:15-15:15 realized variances), "
        "including open-to-first-close"
    ),
    "vwap_z": "(close - session typical-price VWAP) / weighted price deviation",
    "same_minute_rvol_5": (
        "current minute volume / mean same-minute volume on five prior market sessions"
    ),
    "minutes_from_open": "elapsed completed minutes since 09:15 IST",
    "benchmark_log_return_15m": (
        "strict 15-minute log return: BANKNIFTY for named banks, NIFTY otherwise"
    ),
    "gold_log_return_15m": "strict 15-minute GOLDBEES log return",
    "vix_level": "INDIAVIX close at exactly the decision minute",
    "side": "+1 LONG, -1 SHORT; probabilities are not complements",
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def daily_closes(
    instrument: Instrument,
    start: date,
    end: date,
    root: Path,
    broker: ReadOnlyBroker | None,
    *,
    store: CandleStore | None = None,
) -> dict[date, Decimal]:
    path = root / f"{instrument.exchange}_{instrument.trading_symbol}.json"
    verified_records = None
    source_fingerprint = None
    if store is not None:
        source = store.verified_path(
            f"daily/{instrument.exchange}_{instrument.trading_symbol}.csv"
        )
        source_fingerprint = sha256(source)
        with source.open(encoding="utf-8", newline="") as handle:
            verified_records = [
                DailyClose(
                    datetime.fromisoformat(row["timestamp"]).date(),
                    Decimal(row["close"]),
                )
                for row in csv.DictReader(handle)
                if start <= datetime.fromisoformat(row["timestamp"]).date() <= end
            ]
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            document["symbol"] != instrument.trading_symbol
            or document["start"] != start.isoformat()
            or document["end"] != end.isoformat()
        ):
            raise ValueError(
                "Daily-close snapshot bounds differ; use a new daily directory"
            )
        records = [
            DailyClose(date.fromisoformat(row["date"]), Decimal(row["close"]))
            for row in document["closes"]
        ]
        if verified_records is not None and (
            document.get("provider") != "kite"
            or document.get("source_sha256") != source_fingerprint
            or records != verified_records
        ):
            raise ValueError("Daily snapshot differs from the validated Kite source")
    else:
        if broker is None and verified_records is None:
            raise ValueError("Daily history is absent; use --fetch-daily once")
        records = verified_records if verified_records is not None else []
        cursor = start
        while verified_records is None and cursor <= end:
            stop = min(cursor + timedelta(days=179), end)
            records.extend(
                broker.get_daily_closes(
                    instrument,
                    datetime.combine(cursor, time.min, INDIA_TIMEZONE),
                    datetime.combine(stop, time(15, 30), INDIA_TIMEZONE),
                )
            )
            cursor = stop + timedelta(days=1)
        if not records:
            raise ValueError("Daily-close endpoint returned no rows")
        write_report(
            path,
            {
                "schema_version": 1,
                "provider": "kite",
                "source_sha256": source_fingerprint,
                "symbol": instrument.trading_symbol,
                "exchange": instrument.exchange,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "source": "Kite historical_data day, close field only",
                "retrieved_at": datetime.now(INDIA_TIMEZONE).isoformat(),
                "closes": [
                    {
                        "date": record.session_date.isoformat(),
                        "close": str(record.close),
                    }
                    for record in records
                ],
            },
        )
    if any(
        left.session_date >= right.session_date
        for left, right in zip(records, records[1:])
    ) or any(not start <= record.session_date <= end for record in records):
        raise ValueError("Daily-close dates are invalid or unordered")
    return {record.session_date: record.close for record in records}


def sessions(
    store: CandleStore, instrument: Instrument, start: date, end: date
) -> dict[date, list[Candle]]:
    grouped: dict[date, list[Candle]] = defaultdict(list)
    for bar in store.load(
        instrument,
        datetime.combine(start, time(9, 15), INDIA_TIMEZONE),
        datetime.combine(end, time(15, 30), INDIA_TIMEZONE),
        allow_incomplete=True,
    ):
        if 0 <= session_minute_offset(bar.start_time) < 375:
            grouped[bar.start_time.date()].append(bar)
    return dict(grouped)


def phase_for(day: date) -> str:
    for phase, end in PHASE_ENDS.items():
        if day <= end:
            return phase
    raise ValueError("Date falls outside the declared research protocol")


def instrument_rows(
    instrument: Instrument,
    calendar: Sequence[date],
    grouped: Mapping[date, Sequence[Candle]],
    closes: Mapping[date, Decimal],
    context: Mapping[str, Mapping[datetime, PriceContext]],
    stride: int = 5,
) -> tuple[dict[str, list[tuple]], dict]:
    if stride <= 0 or 60 % stride:
        raise ValueError("Decision stride must be a positive divisor of 60")
    rows: dict[str, list[tuple]] = {phase: [] for phase in PHASE_ENDS}
    records: list[CompletedSession] = []
    missing: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    benchmark = "BANKNIFTY" if instrument.trading_symbol in BANK_SYMBOLS else "NIFTY"
    for day in calendar:
        state = TpFeatureState(instrument, day, records)
        bars = grouped.get(day, ())
        vendor_close = closes.get(day)
        if vendor_close is None:
            quality["missing_daily_close"] += 1
        if bars and vendor_close is not None:
            if (
                min(bar.low for bar in bars)
                <= vendor_close
                <= max(bar.high for bar in bars)
            ):
                quality["daily_close_inside_observed_range"] += 1
            else:
                quality["daily_close_outside_observed_range"] += 1
            if vendor_close == bars[-1].close:
                quality["daily_close_equals_last_observed_minute"] += 1
        for index, bar in enumerate(bars):
            if session_minute_offset(bar.start_time) >= CUTOFF_MINUTES:
                break
            state.update(bar)
            minute = session_minute_offset(bar.end_time)
            if not 25 <= minute <= CUTOFF_MINUTES - HORIZON_MINUTES or minute % stride:
                continue
            counts["candidate_decisions"] += 1
            values = state.values(
                {
                    name: table[bar.end_time]
                    for name, table in context.items()
                    if bar.end_time in table
                },
                benchmark=benchmark,
            )
            unavailable = [
                name
                for name, value in values.items()
                if value is None or not value.is_finite()
            ]
            if unavailable:
                missing.update(unavailable)
                counts["excluded_missing_features"] += 1
                continue
            try:
                labels = [
                    touch_label(
                        bar, bars[index + 1 : index + 1 + HORIZON_MINUTES], side
                    )
                    for side in (1, -1)
                ]
            except ValueError:
                counts["excluded_incomplete_future"] += 1
                continue
            vector = tuple(float(values[name]) for name in FEATURE_COLUMNS)
            for side, label in zip((1, -1), labels):
                rows[phase_for(day)].append(
                    (
                        (*vector, float(side)),
                        int(label.hit),
                        int(bar.end_time.timestamp()),
                        label.first_hit_minute or 0,
                        float(label.adverse_excursion),
                        float(label.terminal_return),
                    )
                )
            counts["included_decisions"] += 1
        record = state.finish(vendor_close)
        quality[
            "complete_cutoff_sessions"
            if record.cutoff_variance is not None
            else "incomplete_cutoff_sessions"
        ] += 1
        records.append(record)
        records = records[-6:]
    return rows, {
        "counts": dict(counts),
        "missing_features": dict(missing),
        "quality": dict(quality),
        "benchmark": benchmark,
    }


def build_dataset(
    universe: Path,
    cache: Path,
    daily_root: Path,
    output: Path,
    start: date,
    end: date,
    *,
    symbols: Sequence[str] | None = None,
    fetch_daily: bool = False,
) -> dict:
    if start >= end or end > PHASE_ENDS["evaluation"]:
        raise ValueError("Invalid dates for this fixed protocol")
    if end > last_completed_session_close(datetime.now(INDIA_TIMEZONE)).date():
        raise ValueError("Cannot build unfinished market sessions")
    if output.exists():
        raise ValueError("Dataset directory already exists; use a new version")
    document, _ = read_universe(universe)
    targets = [
        Instrument(item["exchange"], item["symbol"])
        for item in document["instruments"]
        if item["role"] == "equity_target"
        and (symbols is None or item["symbol"] in symbols)
    ]
    if not targets or (
        symbols is not None
        and set(symbols) != {item.trading_symbol for item in targets}
    ):
        raise ValueError("Requested targets are not in the declared universe")
    store = CandleStore(cache)
    provenance = store.dataset
    if provenance is None or document.get("provider") != "kite" or fetch_daily:
        raise ValueError(
            "Build from a promoted Kite dataset; "
            "refresh history by staging a new version"
        )
    if sha256(universe) != sha256(store.verified_path("universe.json")):
        raise ValueError("Universe differs from the validated Kite dataset")
    context: dict[str, dict[datetime, PriceContext]] = {}
    source_hashes = {
        str(universe): sha256(universe),
        str(cache / "dataset.json"): sha256(cache / "dataset.json"),
    }
    calendar = [
        date.fromisoformat(day)
        for day in provenance["session_dates"]
        if start.isoformat() <= day <= end.isoformat()
    ]
    for symbol in CONTEXT_SYMBOLS:
        instrument = Instrument("NSE", symbol)
        source_path = store.path_for(instrument)
        source_hashes[str(source_path)] = sha256(source_path)
        grouped = sessions(store, instrument, start, end)
        table = {}
        for day, bars in grouped.items():
            state = TpFeatureState(instrument, day)
            for bar in bars:
                if session_minute_offset(bar.start_time) >= CUTOFF_MINUTES:
                    break
                state.update(bar)
                minute = session_minute_offset(bar.end_time)
                if 25 <= minute <= 300 and minute % 5 == 0:
                    table[bar.end_time] = state.price_context()
        context[symbol] = table
        print(json.dumps({"context": symbol, "decisions": len(table)}), flush=True)
    if not calendar:
        raise ValueError("No benchmark sessions available")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "provider": "kite",
        "objective": "directional_price_touch_without_stop_loss",
        "tp_fraction": str(TP_FRACTION),
        "horizon_minutes": HORIZON_MINUTES,
        "reference": "last completed one-minute close; first future minute included",
        "cutoff_ist": "15:15",
        "decision_stride_minutes": 5,
        "first_decision_ist": "09:40",
        "last_decision_ist": "14:15",
        "model_columns": list(MODEL_COLUMNS),
        "feature_definitions": FEATURE_DEFINITIONS,
        "stock_identity_input": False,
        "missing_policy": (
            "exclude decision; never fill a missing target window with a negative"
        ),
        "phase_ends": {phase: end.isoformat() for phase, end in PHASE_ENDS.items()},
        "calendar": [day.isoformat() for day in calendar],
        "calendar_source": provenance["calendar_basis"],
        "universe_warning": document.get(
            "membership_warning", "Current membership, not point-in-time history"
        ),
        "execution_contract": (
            "TP or 15:00 IST only; no stop loss; "
            "the 60-minute label is not a forced exit"
        ),
        "limitations": [
            "Kite vendor-adjusted history is not historical cash execution prices",
            "Daily consistency and sampled NSE checks do not certify every minute",
            "A daily official close can lie outside continuous-minute prices "
            "because of auctions",
            "The sixty-minute touch label can extend past "
            "the separate 15:00 execution deadline",
            "Current index membership introduces survivorship bias",
            "Overlapping targets and stocks are dependent; "
            "evaluate by whole-session blocks",
            "Evaluation is retrospective; July-September 2026 "
            "have previously been researched",
            "Price touch is neither an executable fill nor net profit",
        ],
        "source_sha256": source_hashes,
        "code_sha256": {
            name: sha256(Path(name))
            for name in (
                "scripts/prepare_tp_only.py",
                "src/ai_trader/features/tp_only.py",
                "src/ai_trader/features/indicators.py",
                "src/ai_trader/clock.py",
                "src/ai_trader/broker/kite.py",
            )
        },
        "instruments": {},
        "shards": [],
    }
    sample = []
    for instrument in targets:
        source_path = store.path_for(instrument)
        source_hashes[str(source_path)] = sha256(source_path)
        closes = daily_closes(instrument, start, end, daily_root, None, store=store)
        close_path = (
            daily_root / f"{instrument.exchange}_{instrument.trading_symbol}.json"
        )
        source_hashes[str(close_path)] = sha256(close_path)
        grouped = sessions(store, instrument, start, end)
        rows, report = instrument_rows(instrument, calendar, grouped, closes, context)
        manifest["instruments"][instrument.trading_symbol] = report
        for phase, values in rows.items():
            if not values:
                continue
            path = output / phase / f"{instrument.trading_symbol}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            matrix = np.asarray([row[0] for row in values], dtype=np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError("Feature matrix contains nonfinite numbers")
            timestamps = np.asarray([row[2] for row in values], dtype=np.int64)
            np.savez_compressed(
                path,
                features=matrix,
                labels=np.asarray([row[1] for row in values], dtype=np.uint8),
                timestamps=timestamps,
                first_hit_minutes=np.asarray(
                    [row[3] for row in values], dtype=np.uint8
                ),
                adverse_excursion=np.asarray(
                    [row[4] for row in values], dtype=np.float32
                ),
                terminal_return=np.asarray(
                    [row[5] for row in values], dtype=np.float32
                ),
            )
            manifest["shards"].append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "symbol": instrument.trading_symbol,
                    "phase": phase,
                    "rows": len(values),
                    "sha256": sha256(path),
                    "first_decision_epoch": int(timestamps[0]),
                    "last_decision_epoch": int(timestamps[-1]),
                }
            )
            if len(sample) < 6:
                for row in values[: 6 - len(sample)]:
                    sample.append(
                        {
                            "symbol": instrument.trading_symbol,
                            "decision_time_ist": datetime.fromtimestamp(
                                row[2], INDIA_TIMEZONE
                            ).isoformat(),
                            **dict(zip(MODEL_COLUMNS, row[0])),
                            "label_touch": row[1],
                            "label_first_hit_minute": row[3],
                        }
                    )
        print(
            json.dumps({"symbol": instrument.trading_symbol, **report["counts"]}),
            flush=True,
        )
    for name, expected in source_hashes.items():
        if sha256(Path(name)) != expected:
            raise ValueError("A source file changed during dataset creation")
    manifest["phase_rows"] = {
        phase: sum(
            shard["rows"] for shard in manifest["shards"] if shard["phase"] == phase
        )
        for phase in PHASE_ENDS
    }
    with (output / "sample.csv").open("w", encoding="utf-8", newline="") as handle:
        if sample:
            writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
            writer.writeheader()
            writer.writerows(sample)
    write_report(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/candles"))
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 28))
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--fetch-daily", action="store_true")
    args = parser.parse_args()
    try:
        manifest = build_dataset(
            args.universe,
            args.cache,
            args.daily_dir,
            args.output,
            args.start,
            args.end,
            symbols=args.symbols,
            fetch_daily=args.fetch_daily,
        )
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error_type": type(error).__name__}),
            flush=True,
        )
        return 1
    print(
        json.dumps({"status": "complete", "phase_rows": manifest["phase_rows"]}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
