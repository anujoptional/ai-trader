"""Stage and validate a Kite-only dataset before replacing any existing data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from time import sleep

import numpy as np

from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle
from ai_trader.broker.kite import KiteAuthenticationError, KiteBroker, KiteBrokerError
from ai_trader.cli.compare_broker_history import (
    _digest,
    _read_snapshot,
    _write_snapshot,
)
from ai_trader.cli.download_history import read_universe, write_report
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE
from ai_trader.config import ConfigurationError, load_kite_login_settings
from ai_trader.history import CandleStore, last_completed_session_close
from ai_trader.market import Candle, to_candle

MUHURAT_2020_2023 = (
    date(2020, 11, 14),
    date(2021, 11, 4),
    date(2022, 10, 24),
    date(2023, 11, 12),
)
RETRY_DELAYS_SECONDS = (2, 5, 15, 30, 60)


def windows_for(day: date, cas: bool) -> tuple[tuple[time, time], ...]:
    if day in MUHURAT_2020_2023:
        return ((time(18, 15), time(19, 15)),)
    if day == date(2021, 2, 24):
        # NSE outage: halted 11:40, resumed 15:45 and extended to 17:00.
        return ((time(9, 15), time(11, 45)), (time(15, 45), time(17)))
    if day in (date(2024, 3, 2), date(2024, 5, 18)):
        return ((time(9, 15), time(10)), (time(11, 30), time(12, 30)))
    if day == date(2024, 11, 1):
        return ((time(18), time(19)),)
    if day == date(2025, 10, 21):
        return ((time(13, 45), time(14, 45)),)
    return (
        (
            time(9, 15),
            time(15, 15) if cas and day >= date(2026, 8, 3) else time(15, 30),
        ),
    )


def validate_history(
    minute: tuple[OHLCVCandle, ...],
    daily: tuple[OHLCVCandle, ...],
    *,
    cas: bool,
) -> dict:
    for candles in (minute, daily):
        if not candles or any(
            left.timestamp >= right.timestamp
            for left, right in zip(candles, candles[1:])
        ):
            raise ValueError("History must be nonempty, unique and ordered.")
        for bar in candles:
            prices = (bar.open, bar.high, bar.low, bar.close)
            if any(not value.is_finite() or value <= 0 for value in prices):
                raise ValueError("History contains invalid prices.")
            if (
                not bar.low
                <= min(bar.open, bar.close)
                <= max(bar.open, bar.close)
                <= bar.high
            ):
                raise ValueError("History contains inconsistent OHLC.")
            if bar.volume is not None and (
                type(bar.volume) is not int or bar.volume < 0
            ):
                raise ValueError("History contains invalid volume.")
    if any(bar.timestamp.second or bar.timestamp.microsecond for bar in minute):
        raise ValueError("Minute history is not minute-aligned.")
    if len({bar.timestamp.date() for bar in daily}) != len(daily):
        raise ValueError("Daily history contains duplicate session dates.")
    by_day = {bar.timestamp.date(): bar for bar in daily}
    grouped = defaultdict(list)
    for bar in minute:
        grouped[bar.timestamp.date()].append(bar)
    missing, price_warnings, severe, outside, outside_dates = {}, [], [], 0, []
    for day in sorted(set(grouped) | set(by_day)):
        rows = grouped[day]
        expected = set()
        for opening, closing in windows_for(day, cas):
            current = datetime.combine(day, opening, INDIA_TIMEZONE)
            end = datetime.combine(day, closing, INDIA_TIMEZONE)
            while current < end:
                expected.add(current)
                current += ONE_MINUTE
        held = {bar.timestamp for bar in rows}
        gaps = sorted(expected - held)
        if gaps:
            missing[day.isoformat()] = [stamp.strftime("%H:%M") for stamp in gaps]
        if held - expected:
            outside += len(held - expected)
            outside_dates.append(day.isoformat())
        if day not in by_day:
            severe.append({"date": day.isoformat(), "reason": "no_daily_reference"})
            continue
        reference = by_day[day]
        for bar in rows:
            if bar.high > reference.high + Decimal(
                "0.10"
            ) or bar.low < reference.low - Decimal("0.10"):
                issue = {
                    "timestamp": bar.timestamp.isoformat(),
                    "minute_low": str(bar.low),
                    "minute_high": str(bar.high),
                    "daily_low": str(reference.low),
                    "daily_high": str(reference.high),
                }
                price_warnings.append(issue)
                if bar.high > reference.high * Decimal(
                    "1.02"
                ) or bar.low < reference.low * Decimal("0.98"):
                    severe.append(issue)
    return {
        "status": "blocked"
        if severe
        else "validated_with_gaps"
        if missing
        else "validated",
        "minute_rows": len(minute),
        "daily_rows": len(daily),
        "first": minute[0].timestamp.isoformat(),
        "last": minute[-1].timestamp.isoformat(),
        "unknown_volume": sum(bar.volume is None for bar in minute),
        "missing_minutes": sum(len(values) for values in missing.values()),
        "gaps": missing,
        "outside_session_rows": outside,
        "outside_session_dates": outside_dates,
        "daily_range_warnings": price_warnings,
        "critical_issues": severe,
        "bounds_note": (
            "Same-provider daily bounds are consistency checks, "
            "not independent accuracy proof"
        ),
        "missing_policy": (
            "Absent bars remain absent; affected feature/label windows must be excluded"
        ),
        "calendar": (
            "Kite daily dates and minute dates; documented special sessions; "
            "no inferred candles"
        ),
    }


def _capture(
    broker: KiteBroker,
    instrument: Instrument,
    start: datetime,
    end: datetime,
    interval: CandleInterval,
):
    """One checkpoint request; transient failures retry, an invalid session does not."""
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            return broker.capture_historical_candles(instrument, start, end, interval)
        except KiteAuthenticationError:
            raise
        except KiteBrokerError:
            if delay is None:
                raise
            sleep(delay)
    raise AssertionError("unreachable")


def _fetch_parts(
    broker: KiteBroker,
    instrument: Instrument,
    start: date,
    end: date,
    interval: CandleInterval,
    root: Path,
    minute_days: int = 28,
) -> tuple[OHLCVCandle, ...]:
    cursor = datetime.combine(start, time.min, INDIA_TIMEZONE)
    ending = datetime.combine(end + timedelta(days=1), time.min, INDIA_TIMEZONE)
    days = minute_days if interval == CandleInterval.ONE_MINUTE else 170
    rows = []
    while cursor < ending:
        boundary = min(cursor + timedelta(days=days), ending)
        path = root / f"{interval.value}_{cursor:%Y%m%d}_{boundary:%Y%m%d}.csv"
        metadata = path.with_suffix(".json")
        if path.exists() or metadata.exists():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            if record["sha256"] != _digest(path):
                raise ValueError("Staged history fingerprint mismatch.")
            batch = _read_snapshot(path)
        else:
            capture = _capture(
                broker, instrument, cursor, boundary - ONE_MINUTE, interval
            )
            batch = capture.candles
            root.mkdir(parents=True, exist_ok=True)
            _write_snapshot(path, batch)
            write_report(
                metadata,
                {
                    "provider": "kite",
                    "start": cursor.isoformat(),
                    "end_exclusive": boundary.isoformat(),
                    "sha256": _digest(path),
                    "rejected": capture.rejected,
                },
            )
        rows.extend(batch)
        cursor = boundary
    return tuple(rows)


def stage_instrument(
    broker: KiteBroker,
    entry: dict,
    start: date,
    end: date,
    root: Path,
    minute_days: int = 28,
) -> dict:
    instrument = Instrument(entry["exchange"], entry["symbol"])
    stem = f"{instrument.exchange}_{instrument.trading_symbol}"
    finished = root / "quality" / f"{stem}.json"
    if finished.exists():
        # Written last; matching outputs mean this resume would recompute the same.
        previous = json.loads(finished.read_text(encoding="utf-8"))
        minute_path = CandleStore(root / "candles").path_for(instrument)
        daily_path = root / "daily" / f"{stem}.csv"
        if (
            minute_path.exists()
            and daily_path.exists()
            and previous.get("minute_sha256") == _digest(minute_path)
            and previous.get("daily_sha256") == _digest(daily_path)
        ):
            return previous
    minute = _fetch_parts(
        broker,
        instrument,
        start,
        end,
        CandleInterval.ONE_MINUTE,
        root / "parts" / stem,
        minute_days,
    )
    daily = _fetch_parts(
        broker, instrument, start, end, CandleInterval.ONE_DAY, root / "parts" / stem
    )
    quality = validate_history(minute, daily, cas=entry["role"] == "equity_target")
    rejected = [
        row
        for path in sorted((root / "parts" / stem).glob("*.json"))
        for row in json.loads(path.read_text(encoding="utf-8")).get("rejected", [])
    ]
    quarantined_dates = {
        datetime.fromisoformat(row["timestamp"]).date()
        for row in (*rejected, *quality["daily_range_warnings"])
        if "timestamp" in row
    }
    quarantined_dates.update(
        date.fromisoformat(row["date"])
        for row in quality["critical_issues"]
        if "date" in row
    )
    quarantined_dates.update(
        date.fromisoformat(day) for day in quality["outside_session_dates"]
    )
    usable_minute = tuple(
        bar for bar in minute if bar.timestamp.date() not in quarantined_dates
    )
    usable_daily = tuple(
        bar for bar in daily if bar.timestamp.date() not in quarantined_dates
    )
    usable_quality = validate_history(
        usable_minute, usable_daily, cas=entry["role"] == "equity_target"
    )
    quality["rejected_rows"] = rejected
    quality["quarantined_dates"] = [
        day.isoformat() for day in sorted(quarantined_dates)
    ]
    quality["usable"] = usable_quality
    quality["status"] = (
        "validated_with_quarantine" if quarantined_dates else usable_quality["status"]
    )
    store = CandleStore(root / "candles")
    store._write(instrument, (to_candle(instrument, bar) for bar in usable_minute))
    store._write_fetched(
        instrument,
        (
            datetime.combine(start, time.min, INDIA_TIMEZONE),
            datetime.combine(end, time(23, 59), INDIA_TIMEZONE),
        ),
    )
    daily_path = root / "daily" / f"{stem}.csv"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    if daily_path.exists():
        previous_daily = _read_snapshot(daily_path)
        if previous_daily not in (daily, usable_daily):
            raise ValueError("Existing daily snapshot does not match staged data.")
        if previous_daily != usable_daily:
            temporary = daily_path.with_suffix(".validated.csv")
            _write_snapshot(temporary, usable_daily)
            temporary.replace(daily_path)
    else:
        _write_snapshot(daily_path, usable_daily)
    resolved = broker.resolve_instrument(instrument)
    result = {
        **entry,
        "provider": "kite",
        "instrument_token": resolved.instrument_token,
        "kite_symbol": resolved.trading_symbol,
        "tick_size": str(resolved.tick_size),
        "price_basis": (
            "Kite vendor-adjusted historical prices; "
            "not historical cash execution prices"
        ),
        "minute_sha256": _digest(store.path_for(instrument)),
        "daily_sha256": _digest(daily_path),
        "quality": quality,
    }
    write_report(root / "quality" / f"{stem}.json", result)
    return result


def stage_all(
    broker: KiteBroker,
    universe: Path,
    root: Path,
    start: date,
    end: date,
    *,
    workers: int = 4,
    minute_days: int = 28,
) -> dict:
    if not 1 <= workers <= 4 or not 1 <= minute_days <= 59 or start > end:
        raise ValueError("Use 1-4 workers, 1-59 minute-chunk days and ordered dates.")
    if (
        datetime.combine(end, time(23, 59), INDIA_TIMEZONE).date()
        > last_completed_session_close(datetime.now(INDIA_TIMEZONE)).date()
    ):
        raise ValueError("Only completed dates may be downloaded.")
    document, instruments = read_universe(universe)
    plan = {
        "provider": "kite",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "minute_chunk_days": minute_days,
        "universe_sha256": _digest(universe),
        "instruments": [asdict(item) for item in instruments],
    }
    plan_path = root / "plan.json"
    if root.exists():
        if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Staging directory belongs to another migration.")
    else:
        root.mkdir(parents=True)
        write_report(plan_path, plan)
    for instrument in instruments:
        broker.resolve_instrument(instrument)
    report = {
        "provider": "kite",
        "status": "staging",
        "workers": workers,
        "rate_limit": "One shared 2.5 requests/second history budget",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "instruments": {},
        "failures": {},
    }
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="kite-history"
    ) as pool:
        tasks = {
            pool.submit(
                stage_instrument,
                broker.fork_history(),
                entry,
                start,
                end,
                root,
                minute_days,
            ): entry
            for entry in document["instruments"]
        }
        for future in as_completed(tasks):
            entry = tasks[future]
            key = f"{entry['exchange']}:{entry['symbol']}"
            try:
                result = future.result()
                report["instruments"][key] = result
                print(
                    json.dumps(
                        {
                            "instrument": key,
                            "rows": result["quality"]["minute_rows"],
                            "gaps": result["quality"]["missing_minutes"],
                            "status": result["quality"]["status"],
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                report["failures"][key] = type(error).__name__
                print(
                    json.dumps(
                        {
                            "instrument": key,
                            "status": "failed",
                            "error_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
            write_report(root / "report.json", report)
    blocked = report["failures"] or any(
        value["quality"]["usable"]["critical_issues"]
        for value in report["instruments"].values()
    )
    report["status"] = "blocked" if blocked else "staged_and_validated"
    report["completed_at"] = datetime.now(INDIA_TIMEZONE).isoformat()
    write_report(root / "report.json", report)
    write_report(
        root / "universe.json",
        {
            "schema_version": 2,
            "provider": "kite",
            "sources": {"instruments": "Kite instrument master"},
            "membership_warning": document.get(
                "membership_warning", "Current membership, not point-in-time history"
            ),
            "instruments": [
                {key: value for key, value in entry.items() if key != "quality"}
                for entry in report["instruments"].values()
            ],
        },
    )
    return report


def promote_dataset(root: Path, destination: Path, *, replace: bool = False) -> dict:
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    universe, instruments = read_universe(root / "universe.json")
    expected = {f"{item.exchange}:{item.trading_symbol}" for item in instruments}
    planned = {
        f"{item['exchange']}:{item['trading_symbol']}" for item in plan["instruments"]
    }
    if (
        report.get("provider") != "kite"
        or universe.get("provider") != "kite"
        or plan.get("provider") != "kite"
        or report.get("status") != "staged_and_validated"
        or report.get("failures")
        or set(report["instruments"]) != expected
        or expected != planned
    ):
        raise ValueError("Incomplete or inconsistent Kite staging manifest.")
    source_store = CandleStore(root / "candles")
    files: dict[str, tuple[Path, str]] = {}
    calendar: set[str] = set()
    for instrument in instruments:
        entry = report["instruments"][
            f"{instrument.exchange}:{instrument.trading_symbol}"
        ]
        quality = entry["quality"]["usable"]
        if (
            entry["provider"] != "kite"
            or quality["critical_issues"]
            or quality["daily_range_warnings"]
            or quality["outside_session_rows"]
        ):
            raise ValueError("Unresolved data-quality errors block promotion.")
        minute = source_store.path_for(instrument)
        daily = root / "daily" / minute.name
        fetched = source_store.fetched_path_for(instrument)
        for relative, path, fingerprint in (
            (minute.name, minute, entry["minute_sha256"]),
            (f"daily/{daily.name}", daily, entry["daily_sha256"]),
            (fetched.name, fetched, _digest(fetched)),
        ):
            if path.is_symlink() or _digest(path) != fingerprint:
                raise ValueError("Staged data fingerprint mismatch.")
            files[relative] = (path, fingerprint)
        parts = sorted((root / "parts" / minute.stem).glob("1d_*.csv"))
        if not parts:
            raise ValueError("Original daily calendar checkpoints are missing.")
        for path in parts:
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if metadata["sha256"] != _digest(path):
                raise ValueError("Daily calendar checkpoint fingerprint mismatch.")
            calendar.update(
                bar.timestamp.date().isoformat() for bar in _read_snapshot(path)
            )
        calendar.update(entry["quality"]["quarantined_dates"])
    for name in ("universe.json", "report.json"):
        files[name] = (root / name, _digest(root / name))
    temporary = destination.with_name(f".{destination.name}.kite-staged")
    retired = destination.with_name(f".{destination.name}.retired-provider")
    if (
        destination.resolve() == (root / "candles").resolve()
        or root.resolve().is_relative_to(destination.resolve())
        or destination.is_symlink()
        or temporary.exists()
        or retired.exists()
        or (destination.exists() and not replace)
    ):
        raise ValueError("Unsafe or already occupied promotion destination.")
    manifest = {
        "schema_version": 1,
        "provider": "kite",
        "start": report["requested_start"],
        "end": report["requested_end"],
        "price_basis": "Vendor-adjusted history; not historical cash execution prices",
        "session_dates": sorted(calendar),
        "calendar_basis": (
            "Union of captured Kite daily dates, including quarantined days; "
            "not an independent holiday calendar"
        ),
        "files": {name: fingerprint for name, (_, fingerprint) in files.items()},
        "minute_rows": sum(
            entry["quality"]["usable"]["minute_rows"]
            for entry in report["instruments"].values()
        ),
        "daily_rows": sum(
            entry["quality"]["usable"]["daily_rows"]
            for entry in report["instruments"].values()
        ),
        "missing_policy": (
            "No imputation; rolling features reset after intraday gaps; "
            "quarantined dates remain in the calendar"
        ),
    }
    temporary.mkdir(parents=True)
    try:
        for name, (source, fingerprint) in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if _digest(target) != fingerprint:
                raise ValueError("Promoted copy fingerprint mismatch.")
        write_report(temporary / "dataset.json", manifest)
        if destination.exists():
            destination.rename(retired)
        try:
            temporary.rename(destination)
        except BaseException:
            if retired.exists():
                retired.rename(destination)
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    result = {
        "provider": "kite",
        "status": "promoted",
        "destination": str(destination),
        "dataset_sha256": _digest(destination / "dataset.json"),
        "retired_cache": str(retired) if retired.exists() else None,
        "instruments": len(instruments),
        "minute_rows": manifest["minute_rows"],
        "daily_rows": manifest["daily_rows"],
    }
    write_report(root / "promotion.json", result)
    return result


def check_labels(bars: tuple[Candle, ...], arrays: dict[str, np.ndarray]) -> int:
    moments = arrays["timestamps"]
    matrix = arrays["features"]
    count = len(moments)
    if (
        count == 0
        or count % 2
        or matrix.shape != (count, 14)
        or not np.isfinite(matrix).all()
        or not np.array_equal(moments[::2], moments[1::2])
        or not np.all(np.diff(moments[::2]) > 0)
        or not np.array_equal(matrix[::2, :-1], matrix[1::2, :-1])
        or not np.all(matrix[::2, -1] == 1)
        or not np.all(matrix[1::2, -1] == -1)
    ):
        raise ValueError("Invalid paired research feature rows.")
    endings = np.asarray(
        [int(bar.end_time.timestamp()) for bar in bars], dtype=np.int64
    )
    signals = moments[::2]
    offsets = np.searchsorted(endings, signals)
    if np.any(offsets >= len(bars)) or not np.array_equal(endings[offsets], signals):
        raise ValueError("Research decision is absent from the candle tape.")
    future = offsets[:, None] + np.arange(1, 61, dtype=np.int64)
    if np.any(future >= len(bars)) or not np.array_equal(
        endings[future],
        signals[:, None] + 60 * np.arange(1, 61, dtype=np.int64),
    ):
        raise ValueError("Research labels bridge a missing future minute.")
    local_minutes = ((signals + 19800) % 86400) // 60
    if (
        np.any(local_minutes < 580)
        or np.any(local_minutes > 855)
        or np.any(local_minutes % 5)
    ):
        raise ValueError("Decision timestamps violate the fixed research cadence.")
    prices = [(bar.high, bar.low, bar.close) for bar in bars]
    scale = max(0, max(-value.as_tuple().exponent for row in prices for value in row))
    multiplier = Decimal(10) ** scale
    integers = [[int(value * multiplier) for value in row] for row in prices]
    if max(value for row in integers for value in row) > np.iinfo(np.int64).max // 1002:
        raise ValueError("Prices exceed the exact integer label-check range.")
    values = np.asarray(integers, dtype=np.int64)
    closes = values[offsets, 2]
    highs, lows = values[future, 0], values[future, 1]
    for side_index, side in enumerate((1, -1)):
        touches = (
            highs * 1000 >= closes[:, None] * 1002
            if side == 1
            else lows * 1000 <= closes[:, None] * 998
        )
        hit = touches.any(axis=1)
        first_hit = np.where(hit, touches.argmax(axis=1) + 1, 0)
        if not np.array_equal(
            arrays["labels"][side_index::2], hit
        ) or not np.array_equal(
            arrays["first_hit_minutes"][side_index::2],
            first_hit,
        ):
            raise ValueError("Independent TP label or first-touch mismatch.")
        adverse = np.maximum(
            0,
            1 - lows.min(axis=1) / closes
            if side == 1
            else highs.max(axis=1) / closes - 1,
        )
        terminal = side * (values[future[:, -1], 2] / closes - 1)
        for name, expected in (
            ("adverse_excursion", adverse),
            ("terminal_return", terminal),
        ):
            if not np.allclose(
                arrays[name][side_index::2], expected, rtol=1e-6, atol=1e-8
            ):
                raise ValueError("Independent research outcome mismatch.")
    return count


def validate_dataset(
    cache: Path, research: Path, output: Path, nse: Path | None = None
) -> dict:
    store = CandleStore(cache)
    provenance = store.dataset
    if provenance is None:
        raise ValueError("Only promoted Kite data can be validated.")
    for name in provenance["files"]:
        store.verified_path(name)
    manifest = json.loads((research / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("provider") != "kite":
        raise ValueError("Research provider is not Kite.")
    for name, expected in {
        **manifest["source_sha256"],
        **manifest["code_sha256"],
    }.items():
        if _digest(Path(name)) != expected:
            raise ValueError("Research source or executed-code fingerprint changed.")
    report = json.loads(store.verified_path("report.json").read_text(encoding="utf-8"))
    _, instruments = read_universe(store.verified_path("universe.json"))
    targets = {
        entry["symbol"]
        for entry in report["instruments"].values()
        if entry["role"] == "equity_target"
    }
    if set(manifest["instruments"]) != targets:
        raise ValueError("Research target universe is incomplete.")
    checks, phase_rows, total_labels = {}, defaultdict(int), 0
    benchmarks: dict[str, dict] = {}
    benchmark_paths: dict[str, Path] = {}
    if nse is not None:
        for day in ("2026-09-01", "2026-09-30"):
            path = nse / f"{day}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = [
                    {key.strip(): value.strip() for key, value in row.items()}
                    for row in csv.DictReader(handle)
                ]
            eligible = [row for row in rows if row["SERIES"] == "EQ"]
            if any(
                datetime.strptime(row["DATE1"], "%d-%b-%Y").date().isoformat() != day
                for row in eligible
            ):
                raise ValueError("NSE benchmark date mismatch.")
            benchmarks[day] = {row["SYMBOL"]: row for row in eligible}
            if len(benchmarks[day]) != len(eligible):
                raise ValueError("NSE benchmark contains ambiguous symbols.")
            benchmark_paths[day] = path
    comparisons = []
    for instrument in instruments:
        key = f"{instrument.exchange}:{instrument.trading_symbol}"
        entry = report["instruments"][key]
        bars = store._read(instrument)
        if any(
            left.start_time >= right.start_time for left, right in zip(bars, bars[1:])
        ):
            raise ValueError("Candle timestamps are not unique and ordered.")
        daily = _read_snapshot(
            store.verified_path(f"daily/{store.path_for(instrument).name}")
        )
        actual = validate_history(
            tuple(
                OHLCVCandle(
                    bar.start_time, bar.open, bar.high, bar.low, bar.close, bar.volume
                )
                for bar in bars
            ),
            daily,
            cas=entry["role"] == "equity_target",
        )
        if actual != entry["quality"]["usable"]:
            raise ValueError("Repeated data-quality validation differs from staging.")
        held = {bar.start_time.date().isoformat() for bar in bars}
        quarantined = set(entry["quality"]["quarantined_dates"])
        if (
            held & quarantined
            or {bar.timestamp.date().isoformat() for bar in daily} & quarantined
        ):
            raise ValueError("Quarantined prices entered the usable dataset.")
        absent = sorted(set(provenance["session_dates"]) - held - quarantined)
        checks[key] = {
            "minute_rows": len(bars),
            "daily_rows": len(daily),
            "missing_minutes_on_observed_dates": actual["missing_minutes"],
            "quarantined_dates": sorted(quarantined),
            "other_absent_dates": absent,
        }
        daily_by_day = {bar.timestamp.date().isoformat(): bar for bar in daily}
        for day, reference in benchmarks.items():
            if instrument.trading_symbol not in reference:
                continue
            candle = daily_by_day[day]
            official = reference[instrument.trading_symbol]
            differences = [
                abs(getattr(candle, name) - Decimal(official[f"{name.upper()}_PRICE"]))
                for name in ("open", "high", "low", "close")
            ]
            volume_matches = candle.volume == int(official["TTL_TRD_QNTY"])
            if max(differences) > Decimal("0.10") or not volume_matches:
                raise ValueError("Independent NSE benchmark differs from Kite.")
            comparisons.append(
                {
                    "date": day,
                    "symbol": instrument.trading_symbol,
                    "maximum_ohlc_difference": str(max(differences)),
                    "volume_exact": volume_matches,
                }
            )
        for shard in manifest["shards"]:
            if shard["symbol"] != instrument.trading_symbol:
                continue
            path = (research / shard["path"]).resolve()
            if (
                not path.is_relative_to(research.resolve())
                or _digest(path) != shard["sha256"]
            ):
                raise ValueError("Research shard path or fingerprint mismatch.")
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
            count = check_labels(bars, arrays)
            if count != shard["rows"]:
                raise ValueError("Research shard row-count mismatch.")
            phase = shard["phase"]
            phases = sorted(manifest["phase_ends"], key=manifest["phase_ends"].get)
            index = phases.index(phase)
            lower = (
                manifest["calendar"][0]
                if index == 0
                else (
                    date.fromisoformat(manifest["phase_ends"][phases[index - 1]])
                    + timedelta(days=1)
                ).isoformat()
            )
            lower_epoch = datetime.combine(
                date.fromisoformat(lower), time.min, INDIA_TIMEZONE
            ).timestamp()
            upper_epoch = datetime.combine(
                date.fromisoformat(manifest["phase_ends"][phase]) + timedelta(days=1),
                time.min,
                INDIA_TIMEZONE,
            ).timestamp()
            if np.any(arrays["timestamps"] < lower_epoch) or np.any(
                arrays["timestamps"] >= upper_epoch
            ):
                raise ValueError(
                    "Research shard crosses a chronological phase boundary."
                )
            phase_rows[phase] += count
            total_labels += count
        print(
            json.dumps({"validated": key, "labels_checked": total_labels}), flush=True
        )
    if dict(phase_rows) != manifest["phase_rows"]:
        raise ValueError("Research phase totals do not reconcile.")
    for day, source in benchmark_paths.items():
        target = output.parent / "nse" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        if _digest(source) != _digest(target):
            raise ValueError("NSE benchmark copy failed verification.")
    result = {
        "status": "passed",
        "provider": "kite",
        "dataset_sha256": _digest(cache / "dataset.json"),
        "research_sha256": _digest(research / "manifest.json"),
        "instruments": checks,
        "independent_labels_checked": total_labels,
        "phase_rows": dict(phase_rows),
        "nse_comparisons": comparisons,
        "nse_sources": {
            day: {
                "sha256": _digest(path),
                "url": (
                    "https://nsearchives.nseindia.com/products/content/"
                    f"sec_bhavdata_full_{date.fromisoformat(day):%d%m%Y}.csv"
                ),
            }
            for day, path in benchmark_paths.items()
        },
        "limits": (
            "Sampled daily benchmarks do not prove every minute; "
            "no live execution or model approval"
        ),
    }
    write_report(output, result)
    return result


def retirement_plan(workspace: Path) -> tuple[Path, ...]:
    selected = [
        workspace / relative
        for relative in (
            "data/.candles.retired-provider",
            "data/azure",
            "data/downloads",
            "data/history_probe_20261003",
            "data/history_tail_probe_20261003",
            "data/groww_instruments_20261003.csv",
            "data/nifty50_context_20261003.json",
            "data/research/feature_audit_20261003",
            "data/research/reliance_sep2026",
            "data/research/scanner_comparison_20261004",
            "data/research/signed_score_v1",
            "data/research/tp_catboost_followup_20261004",
            "data/research/tp_horizon_symmetric_v1",
            "data/research/tp_no_stop_1500_20261005",
            "data/research/tp_only_20261003",
            "data/research/tp_only_daily_20261003",
            "data/research/tp_probability_2026q3",
            "data/research/tp_probability_month_split",
            "data/research/reliance_yahoo_20260923_24.comparison.csv",
            "backtests/reliance_sep2026",
            "backtests/tp_horizon_symmetric_v1",
            "backtests/tp_probability_2026q3",
            "backtests/tp_probability_month_split",
        )
    ]
    for directory in (workspace / "backtests", workspace / "data/regrid"):
        selected.extend(
            path
            for path in directory.glob("*")
            if (
                path.is_file()
                and path.suffix in (".tsv", ".txt")
                and path.name != "universe.txt"
            )
        )
    comparisons = workspace / "data/broker_comparisons"
    selected.extend(
        path
        for path in comparisons.rglob("*")
        if path.is_file()
        and ("groww" in path.name.lower() or path.name == "differences.csv")
    )
    paths = tuple(sorted(set(path for path in selected if path.exists())))
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("Unsafe retired-data path.")
        if path.is_dir() and any(child.is_symlink() for child in path.rglob("*")):
            raise ValueError("Retired-data directory contains a link.")
    return paths


def retire_data(
    workspace: Path, stage: Path, research: Path, *, apply: bool = False
) -> dict:
    validation = json.loads((stage / "validation.json").read_text(encoding="utf-8"))
    cache = workspace / "data/candles"
    if (
        validation.get("status") != "passed"
        or validation.get("provider") != "kite"
        or validation["dataset_sha256"] != _digest(cache / "dataset.json")
        or validation["research_sha256"] != _digest(research / "manifest.json")
        or len(validation["nse_comparisons"]) != 108
        or validation["independent_labels_checked"] <= 0
    ):
        raise ValueError("Passing Kite validation is required before retirement.")
    store = CandleStore(cache)
    for name in store.dataset["files"]:
        store.verified_path(name)
    manifest = json.loads((research / "manifest.json").read_text(encoding="utf-8"))
    for shard in manifest["shards"]:
        if _digest(research / shard["path"]) != shard["sha256"]:
            raise ValueError("Replacement research changed after validation.")
    paths = retirement_plan(workspace)
    records = []
    for path in paths:
        files = (
            sorted(child for child in path.rglob("*") if child.is_file())
            if path.is_dir()
            else [path]
        )
        records.append(
            {
                "path": path.relative_to(workspace).as_posix(),
                "directory": path.is_dir(),
                "files": [
                    {
                        "path": child.relative_to(workspace).as_posix(),
                        "bytes": child.stat().st_size,
                    }
                    for child in files
                ],
            }
        )
    result = {
        "status": "planned",
        "provider": "kite",
        "validation_sha256": _digest(stage / "validation.json"),
        "research_sha256": _digest(research / "manifest.json"),
        "targets": records,
        "files": sum(len(record["files"]) for record in records),
        "bytes": sum(item["bytes"] for record in records for item in record["files"]),
        "scope": (
            "Local retired datasets, trained artifacts and derived outputs only; "
            "cloud assets unchanged"
        ),
    }
    target = stage / ("retirement.json" if apply else "retirement_plan.json")
    if apply and target.exists():
        raise ValueError("Retirement was already recorded; preserve its receipt.")
    write_report(target, result)
    if apply:
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        if any(path.exists() for path in paths):
            raise ValueError("Retired data remains after deletion.")
        result["status"] = "deleted"
        result["completed_at"] = datetime.now(INDIA_TIMEZONE).isoformat()
        write_report(target, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minute-chunk-days", type=int, default=28)
    parser.add_argument("--promote-to", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--validate-cache", type=Path)
    parser.add_argument("--research", type=Path)
    parser.add_argument("--nse-reference", type=Path)
    parser.add_argument("--retire-workspace", type=Path)
    parser.add_argument("--apply-retirement", action="store_true")
    args = parser.parse_args()
    try:
        if args.retire_workspace is not None:
            if args.research is None:
                parser.error("--retire-workspace requires --research")
            result = retire_data(
                args.retire_workspace,
                args.stage,
                args.research,
                apply=args.apply_retirement,
            )
            print(
                json.dumps(
                    {name: result[name] for name in ("status", "files", "bytes")}
                ),
                flush=True,
            )
            return 0
        if args.validate_cache is not None:
            if args.research is None:
                parser.error("--validate-cache requires --research")
            result = validate_dataset(
                args.validate_cache,
                args.research,
                args.stage / "validation.json",
                args.nse_reference,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "labels_checked": result["independent_labels_checked"],
                        "nse_comparisons": len(result["nse_comparisons"]),
                    }
                ),
                flush=True,
            )
            return 0
        if args.promote_to is not None:
            result = promote_dataset(args.stage, args.promote_to, replace=args.replace)
            print(json.dumps(result), flush=True)
            return 0
        if (
            args.universe is None
            or args.start is None
            or args.end is None
            or args.replace
        ):
            parser.error(
                "Downloads require --universe, --start and --end; "
                "--replace is only for promotion."
            )
        print(
            "Opening private Kite browser login for the staged migration.", flush=True
        )
        broker = KiteBroker.authenticate(load_kite_login_settings())
        report = stage_all(
            broker,
            args.universe,
            args.stage,
            args.start,
            args.end,
            workers=args.workers,
            minute_days=args.minute_chunk_days,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "instruments": len(report["instruments"]),
                    "failed": len(report["failures"]),
                }
            ),
            flush=True,
        )
        return int(report["status"] == "blocked")
    except (ConfigurationError, KiteBrokerError, ValueError, OSError):
        print(
            "Kite migration failed validation or access; no existing data was deleted.",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
