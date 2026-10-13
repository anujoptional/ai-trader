"""Reproducible RELIANCE feed audit against public NSE daily archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, time
from decimal import Decimal
from math import ceil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ai_trader.broker import Instrument, OHLCVCandle
from ai_trader.cli.compare_broker_history import (
    _digest,
    _read_snapshot,
    _verified_report,
)
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE

RELIANCE = Instrument("NSE", "RELIANCE")
BONUS_DATE = date(2024, 10, 28)
CAS_DATE = date(2026, 8, 3)
PRICE_FIELDS = ("open", "high", "low", "close")
NSE_FIELDS = {
    "open": "OPEN_PRICE",
    "high": "HIGH_PRICE",
    "low": "LOW_PRICE",
    "close": "CLOSE_PRICE",
    "last": "LAST_PRICE",
    "volume": "TTL_TRD_QNTY",
}


def native_factor(provider: str, day: date) -> Decimal:
    return Decimal(2) if provider == "kite" and day < BONUS_DATE else Decimal(1)


def session_windows(day: date) -> tuple[tuple[time, time], ...]:
    if day in (date(2024, 3, 2), date(2024, 5, 18)):
        return ((time(9, 15), time(10)), (time(11, 30), time(12, 30)))
    if day == date(2024, 11, 1):
        return ((time(18), time(19)),)
    if day == date(2025, 10, 21):
        return ((time(13, 45), time(14, 45)),)
    return ((time(9, 15), time(15, 15) if day >= CAS_DATE else time(15, 30)),)


def is_continuous(candle: OHLCVCandle) -> bool:
    return any(
        opening <= candle.timestamp.time() < closing
        for opening, closing in session_windows(candle.timestamp.date())
    )


def parse_nse_day(payload: bytes, expected: date) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    rows = [
        {key.strip(): value.strip() for key, value in row.items()} for row in reader
    ]
    matches = [
        row
        for row in rows
        if row.get("SYMBOL") == "RELIANCE" and row.get("SERIES") == "EQ"
    ]
    if len(matches) != 1:
        raise ValueError("NSE archive must contain exactly one RELIANCE EQ record.")
    row = matches[0]
    if datetime.strptime(row["DATE1"], "%d-%b-%Y").date() != expected:
        raise ValueError("NSE archive date does not match the request.")
    values = {field: Decimal(row[name]) for field, name in NSE_FIELDS.items()}
    if any(not value.is_finite() or value < 0 for value in values.values()):
        raise ValueError("NSE archive contains invalid values.")
    if (
        not values["low"]
        <= min(values["open"], values["close"])
        <= max(values["open"], values["close"])
        <= values["high"]
    ):
        raise ValueError("NSE daily prices are inconsistent.")
    if values["volume"] != values["volume"].to_integral_value():
        raise ValueError("NSE daily volume must be integral.")
    return {
        "date": expected.isoformat(),
        **{field: str(value) for field, value in values.items()},
    }


def select_benchmark_dates(
    groww: Sequence[OHLCVCandle],
    kite: Sequence[OHLCVCandle],
    daily: Sequence[OHLCVCandle],
) -> tuple[date, ...]:
    first_by_month: dict[str, date] = {}
    for candle in daily:
        first_by_month.setdefault(
            candle.timestamp.strftime("%Y-%m"), candle.timestamp.date()
        )
    selected = set(first_by_month.values())
    left = {bar.timestamp: bar for bar in groww if is_continuous(bar)}
    right = {bar.timestamp: bar for bar in kite if is_continuous(bar)}
    selected.update(stamp.date() for stamp in left.keys() ^ right.keys())
    for stamp in left.keys() & right.keys():
        reference = right[stamp].close * native_factor("kite", stamp.date())
        if abs(left[stamp].close / reference - 1) > Decimal("0.1"):
            selected.add(stamp.date())
    selected.update((date(2024, 6, 4), date(2024, 10, 25), BONUS_DATE, CAS_DATE))
    return tuple(sorted(selected))


def capture_nse(dates: Sequence[date], output: Path) -> dict:
    directory = output / "nse"
    directory.mkdir(parents=True, exist_ok=True)
    records, failures, sources = [], [], []
    for day in dates:
        path = directory / f"{day.isoformat()}.csv"
        url = (
            "https://nsearchives.nseindia.com/products/content/"
            f"sec_bhavdata_full_{day:%d%m%Y}.csv"
        )
        try:
            if path.exists():
                payload = path.read_bytes()
            else:
                with urlopen(url, timeout=40) as response:
                    payload = response.read()
                parse_nse_day(payload, day)
                with path.open("xb") as handle:
                    handle.write(payload)
            record = parse_nse_day(payload, day)
            records.append(record)
            sources.append(
                {"date": day.isoformat(), "url": url, "sha256": _digest(path)}
            )
        except (HTTPError, URLError, OSError, ValueError, KeyError) as error:
            failures.append(
                {
                    "date": day.isoformat(),
                    "url": url,
                    "error_type": type(error).__name__,
                    "http_status": getattr(error, "code", None),
                }
            )
        print(
            json.dumps(
                {
                    "nse_date": day.isoformat(),
                    "verified": len(records),
                    "failed": len(failures),
                }
            ),
            flush=True,
        )
    return {"records": records, "failures": failures, "sources": sources}


def capture_groww_probes(output: Path) -> dict:
    raise ValueError(
        "Groww access was retired; this audit supports saved evidence only."
    )


def _pair_metrics(
    pairs: Sequence[tuple[Decimal, Decimal]], *, volume: bool = False
) -> dict:
    if not pairs:
        return {"count": 0}
    differences = sorted(abs(left - right) for left, right in pairs)
    relative = [abs(left / right - 1) * 10000 for left, right in pairs if right > 0]
    return {
        "count": len(pairs),
        "exact": sum(left == right for left, right in pairs),
        "mean_absolute": str(sum(differences) / len(pairs)),
        "median_absolute": str(differences[(len(pairs) - 1) // 2]),
        "p99_absolute": str(differences[ceil(len(pairs) * 0.99) - 1]),
        "maximum_absolute": str(differences[-1]),
        "within_native_010": None
        if volume
        else sum(value <= Decimal("0.10") for value in differences),
        "relative_compared": len(relative),
        "within_1bp": sum(value <= 1 for value in relative),
        "over_20bp": sum(value > 20 for value in relative),
        "over_1000bp": sum(value > 1000 for value in relative),
    }


def normalized_agreement(
    groww: Sequence[OHLCVCandle],
    kite: Sequence[OHLCVCandle],
) -> dict:
    left = {bar.timestamp: bar for bar in groww if is_continuous(bar)}
    right = {bar.timestamp: bar for bar in kite if is_continuous(bar)}
    common = sorted(left.keys() & right.keys())
    groups = {
        "all": common,
        "pre_bonus": [stamp for stamp in common if stamp.date() < BONUS_DATE],
        "post_bonus": [stamp for stamp in common if stamp.date() >= BONUS_DATE],
        "from_2025_04": [stamp for stamp in common if stamp.date() >= date(2025, 4, 1)],
    }
    metrics = {}
    for name, stamps in groups.items():
        fields = {}
        for field in (*PRICE_FIELDS, "volume"):
            pairs = []
            for stamp in stamps:
                first, second = (
                    getattr(left[stamp], field),
                    getattr(right[stamp], field),
                )
                if first is None or second is None:
                    continue
                factor = native_factor("kite", stamp.date())
                native = (
                    Decimal(second) / factor if field == "volume" else second * factor
                )
                pairs.append((Decimal(first), native))
            fields[field] = _pair_metrics(pairs, volume=field == "volume")
            fields[field]["unknown_pairs"] = len(stamps) - len(pairs)
        metrics[name] = {"common": len(stamps), "fields": fields}
    severe = []
    for stamp in common:
        factor = native_factor("kite", stamp.date())
        changed = [
            field
            for field in PRICE_FIELDS
            if abs(
                getattr(left[stamp], field) / (getattr(right[stamp], field) * factor)
                - 1
            )
            > Decimal("0.1")
        ]
        if changed:
            severe.append(
                {
                    "timestamp": stamp.isoformat(),
                    "fields_over_10pct": changed,
                    "groww": {
                        field: str(getattr(left[stamp], field))
                        for field in PRICE_FIELDS
                    },
                    "kite_native": {
                        field: str(getattr(right[stamp], field) * factor)
                        for field in PRICE_FIELDS
                    },
                }
            )
    return {
        "basis": (
            "Kite pre-2024-10-28 prices multiplied by2 and volumes divided by2; "
            "source files unchanged"
        ),
        "metrics": metrics,
        "severe_price_discrepancies": severe,
        "volume_note": (
            "Unknown volume is excluded, never treated as zero; "
            "native-equivalent volume may be fractional after adjustment"
        ),
    }


def audit_coverage(
    groww: Sequence[OHLCVCandle],
    kite: Sequence[OHLCVCandle],
    daily: Sequence[OHLCVCandle],
    nse: Sequence[dict],
) -> dict:
    providers = {"groww": groww, "kite": kite}
    maps = {
        name: {bar.timestamp: bar for bar in bars} for name, bars in providers.items()
    }
    days = sorted(
        {bar.timestamp.date() for bars in providers.values() for bar in bars}
        | {bar.timestamp.date() for bar in daily}
        | {date.fromisoformat(row["date"]) for row in nse}
    )
    totals = {
        name: {
            "missing_minutes": 0,
            "missing_before_1500": 0,
            "complete_sessions": 0,
            "empty_sessions": 0,
        }
        for name in providers
    }
    issues = []
    expected_total = 0
    for day in days:
        expected = set()
        for opening, closing in session_windows(day):
            cursor = datetime.combine(day, opening, INDIA_TIMEZONE)
            ending = datetime.combine(day, closing, INDIA_TIMEZONE)
            while cursor < ending:
                expected.add(cursor)
                cursor += ONE_MINUTE
        expected_total += len(expected)
        details = {"date": day.isoformat(), "expected_minutes": len(expected)}
        for name, bars in maps.items():
            missing = sorted(expected - bars.keys())
            totals[name]["missing_minutes"] += len(missing)
            totals[name]["missing_before_1500"] += sum(
                stamp.time() < time(15) for stamp in missing
            )
            totals[name]["complete_sessions"] += not missing
            totals[name]["empty_sessions"] += len(missing) == len(expected)
            details[name] = {
                "present": len(expected & bars.keys()),
                "missing": len(missing),
                "missing_times": [stamp.strftime("%H:%M") for stamp in missing],
            }
        if any(details[name]["missing"] for name in providers):
            issues.append(details)
    for name, bars in providers.items():
        totals[name]["downloaded_rows"] = len(bars)
        totals[name]["continuous_rows"] = sum(is_continuous(bar) for bar in bars)
        totals[name]["outside_continuous"] = sum(not is_continuous(bar) for bar in bars)
        totals[name]["unknown_volume_continuous"] = sum(
            is_continuous(bar) and bar.volume is None for bar in bars
        )
    return {
        "observed_sessions": len(days),
        "expected_minutes": expected_total,
        "providers": totals,
        "incomplete_sessions": issues,
        "calendar_limit": (
            f"Union of both minute feeds, Kite daily dates and {len(nse)} NSE samples; "
            "not a fully independent exchange-calendar audit"
        ),
    }


def benchmark_nse(
    groww: Sequence[OHLCVCandle],
    kite: Sequence[OHLCVCandle],
    daily: Sequence[OHLCVCandle],
    nse: Sequence[dict],
) -> tuple[dict, list[dict]]:
    grouped = {name: defaultdict(list) for name in ("groww", "kite")}
    for name, bars in (("groww", groww), ("kite", kite)):
        for bar in bars:
            if is_continuous(bar):
                grouped[name][bar.timestamp.date()].append(bar)
    vendor_daily = {bar.timestamp.date(): bar for bar in daily}
    rows, bounds = [], []
    daily_pairs = {field: [] for field in (*PRICE_FIELDS, "volume")}
    for reference in nse:
        day = date.fromisoformat(reference["date"])
        row = {
            "date": reference["date"],
            **{
                f"nse_{field}": value
                for field, value in reference.items()
                if field != "date"
            },
        }
        for name in ("groww", "kite"):
            factor = native_factor(name, day)
            bars = grouped[name].get(day, [])
            row[f"{name}_minute_count"] = len(bars)
            if not bars:
                continue
            row.update(
                {
                    f"{name}_first_open_native": str(bars[0].open * factor),
                    f"{name}_minute_high_native": str(
                        max(bar.high for bar in bars) * factor
                    ),
                    f"{name}_minute_low_native": str(
                        min(bar.low for bar in bars) * factor
                    ),
                    f"{name}_last_close_native": str(bars[-1].close * factor),
                    f"{name}_known_minute_volume_native": str(
                        sum(
                            Decimal(bar.volume) / factor
                            for bar in bars
                            if bar.volume is not None
                        )
                    ),
                    f"{name}_unknown_volume_minutes": sum(
                        bar.volume is None for bar in bars
                    ),
                }
            )
            official_low, official_high = (
                Decimal(reference["low"]),
                Decimal(reference["high"]),
            )
            for bar in bars:
                failures = [
                    field
                    for field in PRICE_FIELDS
                    if not (
                        official_low - Decimal("0.10")
                        <= getattr(bar, field) * factor
                        <= official_high + Decimal("0.10")
                    )
                ]
                if failures:
                    bounds.append(
                        {
                            "provider": name,
                            "timestamp": bar.timestamp.isoformat(),
                            "fields": failures,
                            "official_low": str(official_low),
                            "official_high": str(official_high),
                            "native_prices": {
                                field: str(getattr(bar, field) * factor)
                                for field in failures
                            },
                        }
                    )
        if day in vendor_daily:
            bar = vendor_daily[day]
            factor = native_factor("kite", day)
            for field in (*PRICE_FIELDS, "volume"):
                value = getattr(bar, field)
                if value is None:
                    continue
                native = (
                    Decimal(value) / factor if field == "volume" else value * factor
                )
                row[f"kite_daily_{field}_native"] = str(native)
                daily_pairs[field].append((native, Decimal(reference[field])))
        rows.append(row)
    return {
        "sample_days": len(nse),
        "kite_daily_vs_nse": {
            field: _pair_metrics(pairs, volume=field == "volume")
            for field, pairs in daily_pairs.items()
        },
        "minute_outside_official_daily_range": bounds,
        "range_tolerance_native_inr": "0.10",
        "limitations": (
            "Daily bounds can disprove impossible minute prices, not verify exact "
            "intraminute opens/highs/lows. Official close is not last trade. Minute "
            "volumes may exclude auctions; benchmark daily volume separately."
        ),
    }, rows


def analyze_saved(source: Path, evidence_path: Path, output: Path) -> dict:
    original = _verified_report(source / "combined")
    evidence = json.loads((evidence_path / "evidence.json").read_text())
    if evidence["source_report_sha256"] != _digest(source / "combined" / "report.json"):
        raise ValueError("Benchmark evidence belongs to another capture.")
    daily_metadata = json.loads((source / "kite_daily.json").read_text())
    if _digest(source / "kite_daily.csv") != daily_metadata["sha256"]:
        raise ValueError("Daily reference fingerprint mismatch.")
    groww, kite = (
        _read_snapshot(source / "combined" / name) for name in ("groww.csv", "kite.csv")
    )
    daily = _read_snapshot(source / "kite_daily.csv")
    nse, fingerprints = [], {}
    for day_text in evidence["selected_dates"]:
        path = evidence_path / "nse" / f"{day_text}.csv"
        nse.append(parse_nse_day(path.read_bytes(), date.fromisoformat(day_text)))
        fingerprints[day_text] = _digest(path)
    benchmarks, rows = benchmark_nse(groww, kite, daily, nse)
    probes = []
    for item in evidence.get("groww_probes", {}).get("windows", []):
        if item["status"] != "retrieved":
            probes.append(item)
            continue
        path = evidence_path / "groww_probes" / f"{item['date']}.csv"
        if _digest(path) != item["sha256"]:
            raise ValueError("Fresh Groww probe fingerprint mismatch.")
        candles = _read_snapshot(path)
        benchmark = next(row for row in nse if row["date"] == item["date"])
        violations = [
            bar.timestamp.isoformat()
            for bar in candles
            if bar.high > Decimal(benchmark["high"]) + Decimal("0.10")
            or bar.low < Decimal(benchmark["low"]) - Decimal("0.10")
        ]
        probes.append({**item, "outside_nse_daily_range": violations})
    report = {
        "schema_version": 1,
        "instrument": original["instrument"],
        "status": "quality_audit_complete",
        "source": str(source.resolve()),
        "source_report_sha256": evidence["source_report_sha256"],
        "evidence_sha256": _digest(evidence_path / "evidence.json"),
        "nse_sha256": fingerprints,
        "analysis_source_sha256": _digest(Path(__file__)),
        "corporate_action": {
            "ex_date": BONUS_DATE.isoformat(),
            "bonus": "1:1",
            "factor": 2,
            "source": "https://nsearchives.nseindia.com/content/circulars/FAOP64621.pdf",
        },
        "coverage": audit_coverage(groww, kite, daily, nse),
        "normalized_agreement": normalized_agreement(groww, kite),
        "exchange_benchmark": benchmarks,
        "fresh_groww_probes": probes,
    }
    output.mkdir(parents=True)
    with (output / "nse_comparison.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report["benchmark_csv_sha256"] = _digest(output / "nse_comparison.csv")
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-nse", action="store_true")
    parser.add_argument("--probe-groww", action="store_true")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new audit output directory.")
    if args.evidence:
        if args.capture_nse or args.probe_groww:
            raise ValueError("Saved-evidence analysis cannot request network capture.")
        report = analyze_saved(args.source, args.evidence, args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    "nse_samples": report["exchange_benchmark"]["sample_days"],
                }
            )
        )
        return 0
    original = _verified_report(args.source / "combined")
    daily_metadata = json.loads((args.source / "kite_daily.json").read_text())
    if _digest(args.source / "kite_daily.csv") != daily_metadata["sha256"]:
        raise ValueError("Daily reference fingerprint mismatch.")
    groww = _read_snapshot(args.source / "combined" / "groww.csv")
    kite = _read_snapshot(args.source / "combined" / "kite.csv")
    daily = _read_snapshot(args.source / "kite_daily.csv")
    dates = select_benchmark_dates(groww, kite, daily)
    args.output.mkdir(parents=True)
    evidence = {
        "source_report_sha256": _digest(args.source / "combined" / "report.json"),
        "groww_source_sha256": original["groww_source_sha256"],
        "selected_dates": [day.isoformat() for day in dates],
        "selection": (
            "First observed daily date each month, provider-only continuous dates, "
            ">10% native-price discrepancies, election/bonus/CAS boundary dates"
        ),
        "retrieved_at_ist": datetime.now(INDIA_TIMEZONE).isoformat(),
    }
    if args.capture_nse:
        evidence["nse"] = capture_nse(dates, args.output)
    if args.probe_groww:
        evidence["groww_probes"] = capture_groww_probes(args.output)
    with (args.output / "evidence.json").open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2)
    print(
        json.dumps(
            {
                "status": "evidence_captured",
                "output": str(args.output),
                "selected_dates": len(dates),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
