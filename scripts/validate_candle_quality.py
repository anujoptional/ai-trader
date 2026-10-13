"""Independent quality audit of a promoted Kite candle cache (data readiness).

Re-checks every verified file without reusing the staging validator: hashes, schema,
session windows, OHLC/volume sanity, minute jumps, minute-vs-daily agreement, and
daily prices/volumes against NSE's official bhavcopy on sampled dates, where Kite's
adjusted history must differ from NSE only by a factor that is constant between
corporate actions. Writes a report and an instrument-day exclusion list.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from ai_trader.cli.download_history import write_report
from ai_trader.history import CandleStore
from ai_trader.news.nse import parse_nse_date

MINUTE_HEADER = ["start_time", "open", "high", "low", "close", "volume"]
DAILY_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
NORMAL = (((9, 15), (15, 30)),)
CAS = (((9, 15), (15, 15)),)
CAS_START = date(2026, 8, 3)
# Official NSE special sessions (Muhurat, DR drills, the 2021-02-24 outage).
SPECIAL = {
    date(2020, 11, 14): (((18, 15), (19, 15)),),
    date(2021, 11, 4): (((18, 15), (19, 15)),),
    date(2022, 10, 24): (((18, 15), (19, 15)),),
    date(2023, 11, 12): (((18, 15), (19, 15)),),
    date(2024, 11, 1): (((18, 0), (19, 0)),),
    date(2025, 10, 21): (((13, 45), (14, 45)),),
    date(2024, 3, 2): (((9, 15), (10, 0)), ((11, 30), (12, 30))),
    date(2024, 5, 18): (((9, 15), (10, 0)), ((11, 30), (12, 30))),
    date(2021, 2, 24): (((9, 15), (11, 45)), ((15, 45), (17, 0))),
}
JUMP_LEVELS = (0.02, 0.05, 0.10)
GAP_SUSPECT = 0.08
EX_DATE_EXCESS_GAP = 0.03
STRUCTURAL = ("split", "bonus", "demerger", "rights", "capital_reduction", "scheme")
ILLIQUID_ZERO_SHARE = 0.2
RATIO_TOLERANCE = 0.002
PRICE_TOLERANCE_INR = 0.10


def expected_minutes(day: date, cas: bool) -> set[int]:
    windows = SPECIAL.get(day) or (CAS if cas and day >= CAS_START else NORMAL)
    minutes: set[int] = set()
    for (open_h, open_m), (close_h, close_m) in windows:
        minutes.update(range(open_h * 60 + open_m, close_h * 60 + close_m))
    return minutes


def paise(text: str) -> bool:
    whole, _, fraction = text.partition(".")
    return len(fraction) <= 2 and whole.lstrip("-").isdigit()


def action_kind(subject: str) -> str:
    text = subject.lower()
    for kind, words in (
        ("split", ("split", "sub-division", "sub division")),
        ("bonus", ("bonus",)),
        ("demerger", ("demerger", "de-merger", "spin off", "spin-off")),
        ("rights", ("rights",)),
        ("capital_reduction", ("reduction",)),
        ("scheme", ("amalgamation", "merger", "scheme of arrangement")),
        ("buyback", ("buy back", "buyback")),
        ("dividend", ("dividend",)),
    ):
        if any(word in text for word in words):
            return kind
    return "other"


def read_minutes(path: Path) -> tuple[dict[str, dict], Counter]:
    """Per-day arrays plus schema counters; rows are validated as they are read."""
    issues: Counter = Counter()
    days: dict[str, dict] = {}
    previous = ""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != MINUTE_HEADER:
            raise ValueError(f"Unexpected minute header in {path.name}")
        for stamp, open_, high, low, close, volume in reader:
            if len(stamp) != 25 or not stamp.endswith(":00+05:30"):
                issues["bad_timestamp"] += 1
                continue
            if stamp <= previous:
                issues["unordered_or_duplicate"] += 1
            previous = stamp
            if not all(paise(value) for value in (open_, high, low, close)):
                issues["non_paise_price"] += 1
            o, h, lo, c = float(open_), float(high), float(low), float(close)
            if min(o, h, lo, c) <= 0 or not lo <= min(o, c) <= max(o, c) <= h:
                issues["invalid_ohlc"] += 1
            if volume == "":
                issues["unknown_volume"] += 1
                v = -1
            else:
                v = int(volume)
                if v < 0:
                    issues["negative_volume"] += 1
            record = days.setdefault(stamp[:10], {"minute": [], "values": []})
            record["minute"].append(int(stamp[11:13]) * 60 + int(stamp[14:16]))
            record["values"].append((o, h, lo, c, v))
    return days, issues


def read_daily(path: Path) -> dict[str, tuple[float, ...]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != DAILY_HEADER:
            raise ValueError(f"Unexpected daily header in {path.name}")
        for stamp, open_, high, low, close, volume in reader:
            rows[stamp[:10]] = tuple(float(x) for x in (open_, high, low, close)) + (
                float(volume) if volume else math.nan,
            )
    return rows


def factor_segments(samples: list[tuple[str, float]]) -> list[dict]:
    """Group consecutive NSE/Kite close ratios into constant-factor segments."""
    segments: list[dict] = []
    for day, ratio in samples:
        if segments and abs(ratio / segments[-1]["ratio"] - 1) <= RATIO_TOLERANCE:
            segments[-1]["last"] = day
            segments[-1]["samples"] += 1
        else:
            segments.append({"first": day, "last": day, "ratio": ratio, "samples": 1})
    return segments


def audit_instrument(task: tuple) -> dict:
    symbol, role, candles, nse_rows, actions, market_gaps = task
    store = CandleStore(candles)
    minute_path = store.verified_path(f"NSE_{symbol}.csv")
    daily_path = store.verified_path(f"daily/NSE_{symbol}.csv")
    days, issues = read_minutes(minute_path)
    daily = read_daily(daily_path)
    cas = role == "equity_target"
    stock = role == "equity_target" or symbol.endswith("BEES")
    excluded: dict[str, set[str]] = {}
    jumps = Counter()
    worst_jumps: list[tuple[float, str]] = []
    missing_total = outside_total = complete_days = 0
    zero_minutes = minutes_total = 0
    volume_ratios, open_offsets, close_offsets = [], [], []
    containment_breaks = 0
    for day_text, record in days.items():
        day = date.fromisoformat(day_text)
        minutes = np.asarray(record["minute"])
        values = np.asarray(record["values"])
        expected = expected_minutes(day, cas)
        present = set(minutes.tolist())
        missing = len(expected - present)
        outside = len(present - expected)
        missing_total += missing
        outside_total += outside
        if outside:
            excluded.setdefault(day_text, set()).add("outside_session")
        if missing:
            excluded.setdefault(day_text, set()).add("incomplete_session")
        else:
            complete_days += 1
        close = values[:, 3]
        moves = np.abs(np.diff(np.log(close)))
        consecutive = np.diff(minutes) == 1
        moves = moves[consecutive]
        for level in JUMP_LEVELS:
            jumps[f"over_{int(level * 100)}pct"] += int((moves > level).sum())
        if moves.size and moves.max() > JUMP_LEVELS[1]:
            excluded.setdefault(day_text, set()).add("minute_jump_over_5pct")
            worst_jumps.append((float(moves.max()), day_text))
        if stock:
            traded = values[:, 4]
            zero = int((traded == 0).sum())
            zero_minutes += zero
            minutes_total += len(traded)
            if len(traded) and zero / len(traded) > ILLIQUID_ZERO_SHARE:
                excluded.setdefault(day_text, set()).add("illiquid_session")
        reference = daily.get(day_text)
        if reference is None:
            excluded.setdefault(day_text, set()).add("no_daily_bar")
            continue
        d_open, d_high, d_low, d_close, d_volume = reference
        if (
            values[:, 1].max() > d_high + PRICE_TOLERANCE_INR
            or values[:, 2].min() < d_low - PRICE_TOLERANCE_INR
        ):
            containment_breaks += 1
            excluded.setdefault(day_text, set()).add("minute_outside_daily_range")
        open_offsets.append(abs(values[0, 0] / d_open - 1))
        close_offsets.append(abs(values[-1, 3] / d_close - 1))
        if stock and d_volume > 0 and (values[:, 4] >= 0).all():
            volume_ratios.append(values[:, 4].sum() / d_volume)
    ordered = sorted(daily)
    gaps = []
    session_gap = {}
    for previous, current in zip(ordered, ordered[1:], strict=False):
        gap = math.log(daily[current][0] / daily[previous][3])
        session_gap[current] = gap
        if abs(gap) > GAP_SUSPECT:
            gaps.append({"date": current, "overnight_gap": round(gap, 4)})
    breaks = []
    for ex_date, subject, kind in actions:
        if kind not in STRUCTURAL or ex_date not in session_gap:
            continue
        excess = session_gap[ex_date] - market_gaps.get(ex_date, 0.0)
        if abs(excess) > EX_DATE_EXCESS_GAP:
            breaks.append(
                {"date": ex_date, "subject": subject, "excess_gap": round(excess, 4)}
            )
            excluded.setdefault(ex_date, set()).add("corporate_action_break")
    comparison = {"dates": 0}
    if nse_rows:
        ratios, exact_close, exact_volume, volume_checked = [], 0, 0, 0
        for day_text in sorted(nse_rows):
            kite = daily.get(day_text)
            if kite is None:
                continue
            n_open, n_high, n_low, n_close, n_volume = nse_rows[day_text]
            ratios.append((day_text, n_close / kite[3]))
            if abs(n_close - kite[3]) <= 0.05:
                exact_close += 1
                volume_checked += 1
                exact_volume += int(n_volume == kite[4])
        segments = factor_segments(ratios)
        for before, after in zip(segments, segments[1:], strict=False):
            kinds = sorted(
                {
                    kind
                    for ex_date, _, kind in actions
                    if before["last"] < ex_date <= after["first"]
                }
            )
            after["change_from_previous"] = round(before["ratio"] / after["ratio"], 6)
            after["actions_in_interval"] = kinds
            after["explained"] = bool(set(kinds) & {*STRUCTURAL, "dividend"})
        comparison = {
            "dates": len(ratios),
            "close_exact_within_5_paise": exact_close,
            "volume_exact_on_unadjusted_dates": [exact_volume, volume_checked],
            "factor_segments": segments,
            "latest_factor": round(ratios[-1][1], 6) if ratios else None,
        }
    return {
        "symbol": symbol,
        "role": role,
        "schema_issues": dict(issues),
        "sessions": len(days),
        "first": min(days) if days else None,
        "last": max(days) if days else None,
        "complete_sessions": complete_days,
        "missing_minutes": missing_total,
        "outside_session_minutes": outside_total,
        "minute_jumps": dict(jumps),
        "worst_jumps": sorted(worst_jumps, reverse=True)[:5],
        "zero_volume_minute_share": zero_minutes / minutes_total
        if minutes_total
        else None,
        "minute_outside_daily_range_days": containment_breaks,
        "first_minute_open_vs_daily_open": percentiles(open_offsets),
        "last_minute_close_vs_daily_close": percentiles(close_offsets),
        "minute_volume_over_daily_volume": percentiles(volume_ratios, exact=1.0),
        "daily_sessions": len(daily),
        "sessions_without_minutes": sorted(set(daily) - set(days)),
        "overnight_gaps_over_8pct": gaps,
        "corporate_action_breaks": breaks,
        "nse": comparison,
        "exclusions": {
            day: sorted(reasons) for day, reasons in sorted(excluded.items())
        },
    }


def percentiles(values: list[float], exact: float = 0.0) -> dict | None:
    if not values:
        return None
    array = np.asarray(values)
    return {
        "p01": float(np.percentile(array, 1)),
        "median": float(np.median(array)),
        "p99": float(np.percentile(array, 99)),
        "exact_share": float(np.isclose(array, exact, rtol=0, atol=1e-12).mean()),
    }


def read_nse(directory: Path) -> dict[str, dict[str, tuple[float, ...]]]:
    """Official EQ-series rows by symbol and date from sec_bhavdata_full files."""
    rows: dict[str, dict[str, tuple[float, ...]]] = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = [name.strip() for name in next(reader)]
            index = {name: i for i, name in enumerate(header)}
            for row in reader:
                row = [value.strip() for value in row]
                if row[index["SERIES"]] not in ("EQ", "BE"):
                    continue
                rows.setdefault(row[index["SYMBOL"]], {})[path.stem] = tuple(
                    float(row[index[name]])
                    for name in (
                        "OPEN_PRICE",
                        "HIGH_PRICE",
                        "LOW_PRICE",
                        "CLOSE_PRICE",
                        "TTL_TRD_QNTY",
                    )
                )
    return rows


def read_actions(path: Path | None) -> dict[str, list[tuple[str, str, str]]]:
    if path is None:
        return {}
    actions: dict[str, list[tuple[str, str, str]]] = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
        ex_date = parse_nse_date(row.get("exDate"))
        subject = str(row.get("subject") or "").strip()
        if ex_date is None or not subject:
            continue
        actions.setdefault(str(row.get("symbol")).strip(), []).append(
            (ex_date.isoformat(), subject, action_kind(subject))
        )
    return actions


def market_gaps(store: CandleStore) -> dict[str, float]:
    daily = read_daily(store.verified_path("daily/NSE_NIFTY.csv"))
    ordered = sorted(daily)
    return {
        current: math.log(daily[current][0] / daily[previous][3])
        for previous, current in zip(ordered, ordered[1:], strict=False)
    }


def run(
    candles: Path, nse: Path, output: Path, workers: int, actions_path: Path | None
) -> dict:
    if output.exists() and any(output.glob("*.json")):
        raise ValueError("Output already holds a report; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    store = CandleStore(candles)
    manifest = store.dataset
    if manifest is None:
        raise ValueError("Not a promoted Kite dataset")
    for name in manifest["files"]:
        store.verified_path(name)
    universe = json.loads(store.verified_path("universe.json").read_text("utf-8"))
    references = read_nse(nse)
    actions = read_actions(actions_path)
    gaps = market_gaps(store)
    tasks = [
        (
            entry["symbol"],
            entry["role"],
            candles,
            references.get(entry.get("kite_symbol", entry["symbol"]))
            or references.get(entry["symbol"], {}),
            actions.get(entry["symbol"], []),
            gaps,
        )
        for entry in universe["instruments"]
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(audit_instrument, tasks))
    exclusions = {result["symbol"]: result.pop("exclusions") for result in results}
    reasons = Counter(
        reason
        for days in exclusions.values()
        for listed in days.values()
        for reason in listed
    )
    schema = Counter()
    for result in results:
        schema.update(result["schema_issues"])
    stocks = [r for r in results if r["role"] == "equity_target"]
    nse_checked = [r for r in results if r["nse"]["dates"]]
    multi_segment = {
        r["symbol"]: r["nse"]["factor_segments"]
        for r in nse_checked
        if len(r["nse"]["factor_segments"]) > 1
    }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": str(candles),
        "dataset_files_hash_verified": len(manifest["files"]),
        "instruments": len(results),
        "equity_targets": len(stocks),
        "nse_reference_files": len(list(nse.glob("*.csv"))),
        "schema_issues_total": dict(schema),
        "outside_session_minutes_total": sum(
            r["outside_session_minutes"] for r in results
        ),
        "missing_minutes_total": sum(r["missing_minutes"] for r in results),
        "sessions_total": sum(r["sessions"] for r in results),
        "complete_sessions_total": sum(r["complete_sessions"] for r in results),
        "minute_outside_daily_range_days_total": sum(
            r["minute_outside_daily_range_days"] for r in results
        ),
        "exclusion_reasons_instrument_days": dict(reasons),
        "nse": {
            "instruments_compared": len(nse_checked),
            "comparisons": sum(r["nse"]["dates"] for r in nse_checked),
            "close_exact_within_5_paise": sum(
                r["nse"]["close_exact_within_5_paise"] for r in nse_checked
            ),
            "volume_exact_on_unadjusted_dates": [
                sum(
                    r["nse"]["volume_exact_on_unadjusted_dates"][i] for r in nse_checked
                )
                for i in (0, 1)
            ],
            "latest_factor_not_one": {
                r["symbol"]: r["nse"]["latest_factor"]
                for r in nse_checked
                if abs(r["nse"]["latest_factor"] - 1) > RATIO_TOLERANCE
            },
            "instruments_with_factor_changes": len(multi_segment),
            "unexplained_factor_changes": {
                symbol: [
                    {k: s[k] for k in ("first", "change_from_previous")}
                    for s in segments[1:]
                    if not s["explained"]
                ]
                for symbol, segments in multi_segment.items()
                if any(not s["explained"] for s in segments[1:])
            },
        },
        "corporate_action_breaks": {
            r["symbol"]: r["corporate_action_breaks"]
            for r in results
            if r["corporate_action_breaks"]
        },
        "corporate_actions_file": str(actions_path) if actions_path else None,
        "instruments_detail": results,
        "factor_changes": multi_segment,
    }
    write_report(output / "report.json", report)
    write_report(
        output / "exclusions.json",
        {
            "dataset": str(candles),
            "dataset_sha256": manifest_sha(candles),
            "reasons": {
                "incomplete_session": "expected session minutes are missing",
                "outside_session": "candles outside the official session windows",
                "minute_jump_over_5pct": "a one-minute close-to-close move above 5%",
                "illiquid_session": "more than 20% zero-volume minutes",
                "minute_outside_daily_range": "minute extremes beyond the daily bar",
                "no_daily_bar": "minutes without a daily bar",
                "corporate_action_break": (
                    "split/bonus/demerger/rights/scheme ex-date whose Kite overnight "
                    "gap differs from NIFTY's by over 3%: adjustment likely broken; "
                    "treat as a series break for multi-day features"
                ),
            },
            "instrument_days": exclusions,
        },
    )
    return report


def manifest_sha(candles: Path) -> str:
    import hashlib

    return hashlib.sha256((candles / "dataset.json").read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--nse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--actions", type=Path)
    args = parser.parse_args()
    report = run(args.candles, args.nse, args.output, args.workers, args.actions)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "instruments",
                    "schema_issues_total",
                    "outside_session_minutes_total",
                    "missing_minutes_total",
                    "exclusion_reasons_instrument_days",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
