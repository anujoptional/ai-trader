"""Download or recompare Kite history against an untouched Groww snapshot."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, time, timedelta
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle
from ai_trader.broker.kite import (
    KiteAuthenticationError,
    KiteBrokerError,
    KiteHistoricalBroker,
)
from ai_trader.broker.kite_auth import REDIRECT_URL, login_browser
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE
from ai_trader.config import (
    ConfigurationError,
    load_kite_login_settings,
    load_kite_settings,
)
from ai_trader.history import CandleStore

_FIELDS = ("open", "high", "low", "close", "volume")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def month_bounds(month: str) -> tuple[datetime, datetime]:
    beginning = datetime.strptime(month, "%Y-%m").replace(
        tzinfo=INDIA_TIMEZONE, hour=9, minute=15
    )
    if beginning.strftime("%Y-%m") != month:
        raise ValueError("Month must be YYYY-MM.")
    ending = beginning.replace(
        day=calendar.monthrange(beginning.year, beginning.month)[1], hour=15, minute=30
    )
    if ending >= datetime.now(INDIA_TIMEZONE):
        raise ValueError("Comparison requires a fully completed month.")
    return beginning, ending


def _regular(
    candles: Sequence[OHLCVCandle], session_close: time
) -> dict[datetime, OHLCVCandle]:
    if any(
        left.timestamp >= right.timestamp for left, right in zip(candles, candles[1:])
    ):
        raise ValueError("Snapshot contains duplicate or unordered timestamps.")
    return {
        candle.timestamp: candle
        for candle in candles
        if time(9, 15) <= candle.timestamp.time() < session_close
    }


def compare_candles(
    groww: Sequence[OHLCVCandle],
    kite: Sequence[OHLCVCandle],
    *,
    session_close: time = time(15, 30),
) -> tuple[dict, list[dict]]:
    if session_close not in (time(15, 15), time(15, 30)):
        raise ValueError("Select a verified 15:15 or 15:30 continuous-session close.")
    session_minutes = session_close.hour * 60 + session_close.minute - (9 * 60 + 15)
    left, right = _regular(groww, session_close), _regular(kite, session_close)
    common = sorted(left.keys() & right.keys())
    field_stats = {}
    for field in _FIELDS:
        pairs = [
            (
                Decimal(getattr(left[stamp], field)),
                Decimal(getattr(right[stamp], field)),
            )
            for stamp in common
            if getattr(left[stamp], field) is not None
            and getattr(right[stamp], field) is not None
        ]
        absolute = [abs(kite_value - groww_value) for groww_value, kite_value in pairs]
        relative = [
            abs(kite_value / groww_value - 1) * 10000
            for groww_value, kite_value in pairs
            if groww_value != 0
        ]
        field_stats[field] = {
            "compared": len(pairs),
            "exact": sum(first == second for first, second in pairs),
            "missing_value_pairs": len(common) - len(pairs),
            "mean_absolute_difference": None
            if not pairs
            else str(sum(absolute) / len(pairs)),
            "maximum_absolute_difference": None if not pairs else str(max(absolute)),
            "mean_absolute_difference_bps_of_groww": None
            if not relative
            else str(sum(relative) / len(relative)),
        }
    differences = []
    for stamp in sorted(left.keys() | right.keys()):
        groww_bar, kite_bar = left.get(stamp), right.get(stamp)
        changed = [
            field
            for field in _FIELDS
            if groww_bar is None
            or kite_bar is None
            or getattr(groww_bar, field) != getattr(kite_bar, field)
        ]
        if not changed:
            continue
        differences.append(
            {
                "timestamp": stamp.isoformat(),
                "status": "kite_only"
                if groww_bar is None
                else "groww_only"
                if kite_bar is None
                else "different",
                "changed_fields": ",".join(changed),
                **{
                    f"{feed}_{field}": ""
                    if bar is None or getattr(bar, field) is None
                    else str(getattr(bar, field))
                    for feed, bar in (("groww", groww_bar), ("kite", kite_bar))
                    for field in _FIELDS
                },
            }
        )
    days = sorted({stamp.date() for stamp in left.keys() | right.keys()})
    daily = {}
    for day in days:
        opening = datetime.combine(day, time(9, 15), INDIA_TIMEZONE)
        expected = {opening + offset * ONE_MINUTE for offset in range(session_minutes)}
        daily[day.isoformat()] = {}
        for name, bars in (("groww", left), ("kite", right)):
            session = [bars[stamp] for stamp in sorted(expected & bars.keys())]
            daily[day.isoformat()][name] = {
                "regular_bars": len(session),
                "first_bar": None if not session else session[0].timestamp.isoformat(),
                "last_bar": None if not session else session[-1].timestamp.isoformat(),
                "missing_regular_minutes": len(expected - bars.keys()),
                "missing_before_1500": sum(
                    stamp.time() < time(15) for stamp in expected - bars.keys()
                ),
                "volume_sum_known_minutes": sum(
                    bar.volume for bar in session if bar.volume is not None
                ),
                "unknown_volume_minutes": sum(bar.volume is None for bar in session),
            }
    offsets = {}
    for offset in (-1, 0, 1):
        pairs = [
            (bar.close, right[stamp + offset * ONE_MINUTE].close)
            for stamp, bar in left.items()
            if stamp + offset * ONE_MINUTE in right
            and (stamp + offset * ONE_MINUTE).date() == stamp.date()
        ]
        offsets[str(offset)] = {
            "compared": len(pairs),
            "close_exact": sum(first == second for first, second in pairs),
            "close_mean_absolute_difference": None
            if not pairs
            else str(sum(abs(first - second) for first, second in pairs) / len(pairs)),
        }
    report = {
        "session": {
            "open": "09:15",
            "close_exclusive": session_close.strftime("%H:%M"),
            "expected_minutes_per_observed_date": session_minutes,
            "basis": "Explicit selection; verify instrument/date eligibility",
        },
        "groww_regular_bars": len(left),
        "kite_regular_bars": len(right),
        "groww_outside_regular_session": len(groww) - len(left),
        "kite_outside_regular_session": len(kite) - len(right),
        "common_timestamps": len(common),
        "groww_only_timestamps": len(left.keys() - right.keys()),
        "kite_only_timestamps": len(right.keys() - left.keys()),
        "all_ohlc_exact_bars": sum(
            all(
                getattr(left[stamp], field) == getattr(right[stamp], field)
                for field in _FIELDS[:-1]
            )
            for stamp in common
        ),
        "all_ohlcv_exact_bars": sum(
            left[stamp].volume is not None
            and right[stamp].volume is not None
            and all(
                getattr(left[stamp], field) == getattr(right[stamp], field)
                for field in _FIELDS
            )
            for stamp in common
        ),
        "fields": field_stats,
        "daily": daily,
        "offset_check_groww_t_vs_kite_t_plus_minutes": offsets,
        "interpretation": (
            "Feed agreement is not exchange-ground-truth accuracy. Differences can "
            "reflect coverage, aggregation, revisions or adjustments; "
            "no feed is assumed correct."
        ),
        "calendar": (
            f"Union of observed feed dates; {session_minutes} continuous-session "
            "minutes per observed date, not a verified holiday calendar"
        ),
    }
    return report, differences


def _write_snapshot(path: Path, candles: Sequence[OHLCVCandle]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", *_FIELDS))
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "timestamp": candle.timestamp.isoformat(),
                    **{
                        field: ""
                        if getattr(candle, field) is None
                        else str(getattr(candle, field))
                        for field in _FIELDS
                    },
                }
            )


def _save_comparison(
    output: Path,
    groww: Sequence[OHLCVCandle],
    kite: Sequence[OHLCVCandle],
    differences: list[dict],
    report: dict,
) -> dict:
    if output.exists():
        raise ValueError("Comparison output already exists; choose a new directory.")
    package = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        name: _digest(package / name)
        for name in (
            "cli/compare_broker_history.py",
            "broker/kite.py",
            "history/store.py",
        )
    }
    output.mkdir(parents=True)
    _write_snapshot(output / "groww.csv", groww)
    _write_snapshot(output / "kite.csv", kite)
    with (output / "differences.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = (
            "timestamp",
            "status",
            "changed_fields",
            *(f"{feed}_{field}" for feed in ("groww", "kite") for field in _FIELDS),
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(differences)
    report["snapshots_sha256"] = {
        name: _digest(output / name)
        for name in ("groww.csv", "kite.csv", "differences.csv")
    }
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def _read_snapshot(path: Path) -> tuple[OHLCVCandle, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["timestamp", *_FIELDS]:
            raise ValueError("Saved candle snapshot has an unexpected schema.")
        return tuple(
            OHLCVCandle(
                datetime.fromisoformat(row["timestamp"]),
                *(Decimal(row[field]) for field in _FIELDS[:-1]),
                None if row["volume"] == "" else int(row["volume"]),
            )
            for row in reader
        )


def _verified_report(source: Path) -> dict:
    original = json.loads((source / "report.json").read_text(encoding="utf-8"))
    for name in ("groww.csv", "kite.csv", "differences.csv"):
        if _digest(source / name) != original["snapshots_sha256"][name]:
            raise ValueError("Saved comparison snapshot fingerprint mismatch.")
    return original


def recompare_saved(
    source: Path,
    output: Path,
    instrument: Instrument,
    month: str,
    *,
    session_close: time,
) -> dict:
    if output.exists():
        raise ValueError("Comparison output already exists; choose a new directory.")
    original_path = source / "report.json"
    original = _verified_report(source)
    if original["instrument"] != asdict(instrument) or original["month"] != month:
        raise ValueError(
            "Saved comparison does not match the requested instrument/month."
        )
    groww, kite = (_read_snapshot(source / name) for name in ("groww.csv", "kite.csv"))
    statistics, differences = compare_candles(groww, kite, session_close=session_close)
    report = {
        **original,
        "schema_version": 2,
        "status": "read_only_saved_comparison_complete",
        "recomputed_at_ist": datetime.now(INDIA_TIMEZONE).isoformat(),
        "parent_report": str(original_path.resolve()),
        "parent_report_sha256": _digest(original_path),
        "verification_scope": (
            "Saved snapshot hashes verified; no broker, credentials or source cache "
            "accessed. Download metadata is inherited from the parent report."
        ),
        "comparison": statistics,
    }
    return _save_comparison(output, groww, kite, differences, report)


def run_comparison(
    broker: KiteHistoricalBroker,
    instrument: Instrument,
    month: str,
    groww_cache: Path,
    output: Path,
    *,
    session_close: time = time(15, 30),
) -> dict:
    if output.exists():
        raise ValueError("Comparison output already exists; choose a new directory.")
    beginning, ending = month_bounds(month)
    store = CandleStore(groww_cache)
    if store.dataset is not None:
        raise ValueError("A Kite dataset cannot be labeled as a retired Groww source.")
    source = store.path_for(instrument)
    source_hash = _digest(source)
    coverage = store.coverage(instrument)
    if coverage is None:
        raise ValueError("No Groww source candles are cached.")
    lower, upper = max(beginning, coverage[0]), min(ending, coverage[1])
    cached = store.load(instrument, lower, upper) if lower < upper else ()
    groww = tuple(
        OHLCVCandle(bar.start_time, bar.open, bar.high, bar.low, bar.close, bar.volume)
        for bar in cached
    )
    resolved = broker.resolve_instrument(instrument)
    cursor = beginning
    kite_rows = []
    while cursor < ending:
        stop = min(cursor + timedelta(days=28) - ONE_MINUTE, ending - ONE_MINUTE)
        kite_rows.extend(
            broker.get_historical_candles(
                instrument, cursor, stop, CandleInterval.ONE_MINUTE
            )
        )
        cursor = stop + ONE_MINUTE
    kite = tuple(kite_rows)
    if not kite:
        raise KiteBrokerError("Kite returned no candles for the requested month.")
    statistics, differences = compare_candles(groww, kite, session_close=session_close)
    if _digest(source) != source_hash:
        raise ValueError("Groww source changed during comparison.")
    report = {
        "schema_version": 2,
        "status": "read_only_comparison_complete",
        "retrieved_at_ist": datetime.now(INDIA_TIMEZONE).isoformat(),
        "instrument": asdict(instrument),
        "month": month,
        "interval": "1m",
        "requested_start": beginning.isoformat(),
        "requested_end": ending.isoformat(),
        "groww_source_sha256": source_hash,
        "groww_source_unchanged": True,
        "kite_instrument_token": resolved.instrument_token,
        "kite_current_tick_size": str(resolved.tick_size),
        "tick_note": "Current master metadata, not a verified historical tick schedule",
        "kite_sdk_version": version("kiteconnect"),
        "comparison": statistics,
    }
    return _save_comparison(output, groww, kite, differences, report)


def months_inclusive(first: str, last: str) -> tuple[str, ...]:
    beginning, _ = month_bounds(first)
    final, _ = month_bounds(last)
    if beginning > final:
        raise ValueError("The first month must not follow the last month.")
    months = []
    cursor = beginning
    while cursor <= final:
        months.append(cursor.strftime("%Y-%m"))
        cursor = cursor.replace(
            year=cursor.year + (cursor.month == 12), month=cursor.month % 12 + 1
        )
    return tuple(months)


def run_monthly_comparisons(
    broker: KiteHistoricalBroker,
    instrument: Instrument,
    first: str,
    last: str,
    groww_cache: Path,
    output: Path,
    *,
    session_closes: Mapping[str, time],
) -> dict[str, dict]:
    months = months_inclusive(first, last)
    if set(session_closes) != set(months) or any(
        close not in (time(15, 15), time(15, 30)) for close in session_closes.values()
    ):
        raise ValueError("Every requested month needs a verified session close.")
    store = CandleStore(groww_cache)
    if store.dataset is not None:
        raise ValueError("A Kite dataset cannot be labeled as a retired Groww source.")
    source = store.path_for(instrument)
    source_hash = _digest(source)
    plan = {
        "schema_version": 1,
        "instrument": asdict(instrument),
        "first_month": first,
        "last_month": last,
        "groww_source_sha256": source_hash,
        "session_closes": {
            month: session_closes[month].strftime("%H:%M") for month in months
        },
    }
    plan_path = output / "range_plan.json"
    if output.exists():
        if not plan_path.is_file() or json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing range plan does not match this request.")
    else:
        output.mkdir(parents=True)
        with plan_path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
    reports = {}
    for month in months:
        destination = output / "months" / month
        if destination.exists():
            report = _verified_report(destination)
            if (
                report["month"] != month
                or report["instrument"] != asdict(instrument)
                or report["groww_source_sha256"] != source_hash
                or report["comparison"]["session"]["close_exclusive"]
                != plan["session_closes"][month]
            ):
                raise ValueError("Saved month does not match the range plan.")
        else:
            report = run_comparison(
                broker,
                instrument,
                month,
                groww_cache,
                destination,
                session_close=session_closes[month],
            )
        reports[month] = report
        print(
            json.dumps(
                {
                    "month": month,
                    "kite_bars": report["comparison"]["kite_regular_bars"],
                    "common_bars": report["comparison"]["common_timestamps"],
                }
            ),
            flush=True,
        )
    if _digest(source) != source_hash:
        raise ValueError("Groww source changed during the range comparison.")
    return reports


def download_daily_reference(
    broker: KiteHistoricalBroker,
    instrument: Instrument,
    first: str,
    last: str,
    output: Path,
) -> dict:
    path = output / "kite_daily.csv"
    metadata_path = output / "kite_daily.json"
    beginning = month_bounds(first)[0].replace(hour=0, minute=0)
    ending = month_bounds(last)[1].replace(hour=23, minute=59)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata["instrument"] != asdict(instrument)
            or metadata["start"] != beginning.isoformat()
            or metadata["end"] != ending.isoformat()
            or metadata["sha256"] != _digest(path)
        ):
            raise ValueError("Saved daily reference does not match the request.")
        return metadata
    if path.exists():
        raise ValueError("Daily reference exists without its metadata.")
    candles = []
    cursor = beginning
    while cursor < ending:
        stop = min(cursor + timedelta(days=170) - ONE_MINUTE, ending)
        candles.extend(
            broker.get_historical_candles(
                instrument, cursor, stop, CandleInterval.ONE_DAY
            )
        )
        cursor = stop + ONE_MINUTE
    if not candles or any(
        left.timestamp >= right.timestamp for left, right in zip(candles, candles[1:])
    ):
        raise ValueError("Daily reference is empty or has invalid ordering.")
    _write_snapshot(path, candles)
    metadata = {
        "instrument": asdict(instrument),
        "start": beginning.isoformat(),
        "end": ending.isoformat(),
        "interval": "1d",
        "count": len(candles),
        "retrieved_at_ist": datetime.now(INDIA_TIMEZONE).isoformat(),
        "sha256": _digest(path),
        "interpretation": "Vendor daily OHLCV; not an aggregation of minute candles",
    }
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def combine_monthly_comparisons(output: Path, reports: Mapping[str, dict]) -> dict:
    destination = output / "combined"
    parents = {
        month: _digest(output / "months" / month / "report.json") for month in reports
    }
    if destination.exists():
        result = _verified_report(destination)
        if result["monthly_report_sha256"] != parents:
            raise ValueError("Combined report does not match the monthly captures.")
        return result
    groww, kite, differences = [], [], []
    for month in reports:
        directory = output / "months" / month
        _verified_report(directory)
        groww.extend(_read_snapshot(directory / "groww.csv"))
        kite.extend(_read_snapshot(directory / "kite.csv"))
        with (directory / "differences.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            differences.extend(csv.DictReader(handle))
    for candles in (groww, kite):
        if any(
            left.timestamp >= right.timestamp
            for left, right in zip(candles, candles[1:])
        ):
            raise ValueError("Monthly captures overlap or are unordered.")
    comparisons = [report["comparison"] for report in reports.values()]
    counts = (
        "groww_regular_bars",
        "kite_regular_bars",
        "groww_outside_regular_session",
        "kite_outside_regular_session",
        "common_timestamps",
        "groww_only_timestamps",
        "kite_only_timestamps",
        "all_ohlc_exact_bars",
        "all_ohlcv_exact_bars",
    )
    fields = {}
    for field in _FIELDS:
        parts = [comparison["fields"][field] for comparison in comparisons]
        compared = sum(part["compared"] for part in parts)
        valid = [part for part in parts if part["compared"]]
        fields[field] = {
            "compared": compared,
            "exact": sum(part["exact"] for part in parts),
            "missing_value_pairs": sum(part["missing_value_pairs"] for part in parts),
            "mean_absolute_difference": None
            if not compared
            else str(
                sum(
                    Decimal(part["mean_absolute_difference"]) * part["compared"]
                    for part in valid
                )
                / compared
            ),
            "maximum_absolute_difference": None
            if not valid
            else str(
                max(Decimal(part["maximum_absolute_difference"]) for part in valid)
            ),
        }
    first = next(iter(reports.values()))
    report = {
        "schema_version": 3,
        "status": "read_only_range_comparison_complete",
        "instrument": first["instrument"],
        "first_month": next(iter(reports)),
        "last_month": next(reversed(reports)),
        "groww_source_sha256": first["groww_source_sha256"],
        "kite_sdk_version": first["kite_sdk_version"],
        "monthly_report_sha256": parents,
        "monthly": {month: value["comparison"] for month, value in reports.items()},
        "comparison": {
            **{name: sum(part[name] for part in comparisons) for name in counts},
            "fields": fields,
            "daily": {
                day: value
                for part in comparisons
                for day, value in part["daily"].items()
            },
            "interpretation": (
                "As-returned provider values; no corporate-action normalization. "
                "Raw differences are not an accuracy score. See monthly reports "
                "for explicit session bounds and timestamp-offset diagnostics."
            ),
        },
    }
    return _save_comparison(destination, groww, kite, differences, report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument("--month")
    period.add_argument("--start-month")
    parser.add_argument("--end-month")
    parser.add_argument("--session-change-month")
    parser.add_argument("--changed-session-close", choices=("15:15", "15:30"))
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--exchange", choices=("NSE", "BSE"), default="NSE")
    parser.add_argument(
        "--session-close",
        choices=("15:15", "15:30"),
        required=True,
        help="Verified continuous-session close for this instrument and month (IST)",
    )
    parser.add_argument("--groww-cache", type=Path, default=Path("data/retired_groww"))
    parser.add_argument("--output", type=Path, required=True)
    input_mode = parser.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--browser-login", action="store_true")
    input_mode.add_argument("--existing-session", action="store_true")
    input_mode.add_argument("--saved-comparison", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.start_month:
            if not args.end_month or args.saved_comparison:
                raise ValueError("A range requires an end month and a broker session.")
            months = months_inclusive(args.start_month, args.end_month)
        else:
            if (
                args.end_month
                or args.session_change_month
                or args.changed_session_close
            ):
                raise ValueError("Range options require a start month.")
            month_bounds(args.month)
            months = ()
        if bool(args.session_change_month) != bool(args.changed_session_close):
            raise ValueError(
                "Session change month and close must be supplied together."
            )
        if args.session_change_month and args.session_change_month not in months:
            raise ValueError("Session change month must be inside the requested range.")
        if args.output.exists() and not months:
            raise ValueError
        instrument = Instrument(args.exchange, args.symbol)
        session_close = time.fromisoformat(args.session_close)
        if args.saved_comparison is not None:
            report = recompare_saved(
                args.saved_comparison,
                args.output,
                instrument,
                args.month,
                session_close=session_close,
            )
        else:
            if args.browser_login:
                settings = load_kite_login_settings()
                print(
                    f"Opening Zerodha sign-in. Expected redirect: {REDIRECT_URL}",
                    flush=True,
                )
                print(
                    "Sign in privately in the browser. Do not paste tokens into chat.",
                    flush=True,
                )
                broker = login_browser(settings)
            else:
                broker = KiteHistoricalBroker.from_session(load_kite_settings())
            if months:
                session_closes = {
                    month: time.fromisoformat(args.changed_session_close)
                    if args.session_change_month and month >= args.session_change_month
                    else session_close
                    for month in months
                }
                reports = run_monthly_comparisons(
                    broker,
                    instrument,
                    args.start_month,
                    args.end_month,
                    args.groww_cache,
                    args.output,
                    session_closes=session_closes,
                )
                download_daily_reference(
                    broker, instrument, args.start_month, args.end_month, args.output
                )
                report = combine_monthly_comparisons(args.output, reports)
            else:
                report = run_comparison(
                    broker,
                    instrument,
                    args.month,
                    args.groww_cache,
                    args.output,
                    session_close=session_close,
                )
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except KiteAuthenticationError:
        print(
            "Kite sign-in did not complete or the session expired. "
            "Check the redirect URL and retry.",
            file=sys.stderr,
        )
        return 1
    except KiteBrokerError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print(
            "Comparison failed local input or data validation; "
            "check dates, paths and available cache. Existing files were preserved.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Kite sign-in or comparison cancelled.", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "comparison": {
                    name: value
                    for name, value in report["comparison"].items()
                    if name != "daily"
                },
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
