"""Rebuild a Kite cache on NSE's official daily price basis (research data repair).

Kite back-adjusts history for splits/bonuses/demergers and for dividends, the latter
sometimes additively and sometimes multiplicatively, and rounds adjusted prices to a
vendor quantum that can be as coarse as one rupee. Percentage moves on adjusted days
are therefore distorted. For every session with an NSE bhavcopy row this fits Kite's
daily O/H/L/C as ``kite = a * nse + c`` (per day), checks the residual against the
day's vendor quantum, and maps every minute back to NSE's raw traded-price basis with
``(kite - c) / a``. Raw prices are then multiplied by ``S``, the product of later
structural ratios: a split/bonus ratio is Kite's own regime change at the NSE ex-date
snapped to a real split/bonus ratio (within 0.5%); rights issues, demergers, schemes
and actions Kite left unadjusted use NSE's ex-date open over the prior close and the
ex-date is flagged as a structural break. History is therefore continuous across
structural actions and dividends are not adjusted. Volumes use NSE's daily
total on the same share basis. Days without an NSE anchor or failing the fit are
dropped; coarse vendor rounding and structural breaks are flagged.
Indices are copied unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np

from ai_trader.cli.download_history import write_report
from ai_trader.history import CandleStore
from ai_trader.news.nse import parse_nse_date
from scripts.validate_candle_quality import STRUCTURAL, action_kind

MINUTE_HEADER = ["start_time", "open", "high", "low", "close", "volume"]
DAILY_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
MIN_SPAN_FOR_FIT = 0.003
MIN_SPAN_FOR_EVENT = 0.01
EVENT_WINDOWS = (15, 10, 6, 3)
SNAP_TOLERANCE = 0.005
# Price ratios real splits (face value F1 -> F2) and bonuses (a new for b held:
# b / (a + b)) produce; demergers are measured, never snapped.
SPLIT_BONUS_RATIOS = frozenset(
    {Fraction(f2, f1) for f1 in (10, 5, 2) for f2 in (1, 2, 5) if f2 < f1}
    | {Fraction(b, a + b) for a in range(1, 11) for b in range(1, 6)}
    | {Fraction(10, 11)}
)
UNADJUSTED_TOLERANCE = 0.005
COARSE_RELATIVE = 0.001
IMPRECISE_RELATIVE = 0.001
RAW_TICK = 0.05
QUANTUM_SAMPLE = 90
# Renames missing from NSE's symbolchange.csv, verified in the bhavcopies.
EXTRA_SYMBOL_CHANGES = {"PATANJALI": [(date(2022, 7, 13), "RUCHI")]}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def symbol_history(path: Path) -> dict[str, list[tuple[date, str]]]:
    """For each current symbol, (change date, previous symbol) pairs, newest first."""
    changes: dict[str, tuple[date, str]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            old, new = row[1].strip(), row[2].strip()
            changed = parse_nse_date(row[3].strip().title())
            if old and new and changed:
                changes[new] = (changed, old)
    history: dict[str, list[tuple[date, str]]] = {}
    for symbol in changes:
        chain, current, seen = [], symbol, set()
        while current in changes and current not in seen:
            seen.add(current)
            changed, old = changes[current]
            chain.append((changed, old))
            current = old
        history[symbol] = chain
    for symbol, extra in EXTRA_SYMBOL_CHANGES.items():
        history.setdefault(symbol, list(extra))
    return history


def nse_symbol(symbol: str, day: date, history: dict) -> str:
    current = symbol
    for changed, old in history.get(symbol, []):
        if day < changed:
            current = old
        else:
            break
    return current


def read_nse(
    directory: Path, wanted: set[str] | None = None
) -> dict[str, dict[str, tuple[float, ...]]]:
    """NSE EQ/BE rows keyed by date then symbol (raw traded prices and volume)."""
    rows: dict[str, dict[str, tuple[float, ...]]] = {}
    for path in sorted(directory.glob("*.csv")):
        day_rows: dict[str, tuple[float, ...]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = [name.strip() for name in next(reader)]
            index = {name: i for i, name in enumerate(header)}
            for row in reader:
                row = [value.strip() for value in row]
                series = row[index["SERIES"]]
                symbol = row[index["SYMBOL"]]
                if series not in ("EQ", "BE") or (
                    symbol in day_rows and series == "BE"
                ):
                    continue
                if wanted is not None and symbol not in wanted:
                    continue
                day_rows[symbol] = tuple(
                    float(row[index[name]])
                    for name in (
                        "OPEN_PRICE",
                        "HIGH_PRICE",
                        "LOW_PRICE",
                        "CLOSE_PRICE",
                        "TTL_TRD_QNTY",
                    )
                )
        rows[path.stem] = day_rows
    return rows


def quantum(values: list[str]) -> float:
    """The coarsest price step consistent with all values (0 when unquantized)."""
    decimals = max((len(v.split(".")[1]) for v in values if "." in v), default=0)
    if decimals > 4:
        return 0.0
    scale = 10**decimals
    step = 0
    for value in values:
        step = gcd(step, round(float(value) * scale))
    return step / scale


def fit_day(
    kite: np.ndarray, nse: np.ndarray, fallback_slope: float | None
) -> tuple[float, float, float]:
    """Affine map kite = a*nse + c for one day's O/H/L/C, and its max residual."""
    if (nse.max() - nse.min()) / nse[3] >= MIN_SPAN_FOR_FIT:
        a, c = np.polyfit(nse, kite, 1)
    else:
        a = fallback_slope if fallback_slope is not None else kite[3] / nse[3]
        c = float(np.mean(kite - a * nse))
    return float(a), float(c), float(np.max(np.abs(a * nse + c - kite)))


def tolerance(step: float, price: float) -> float:
    """Kite re-rounds at every adjustment step, so allow 2.5 quanta, 1 paisa and 2 bp.

    Real misalignments (wrong day, bad bar) are 50+ bp and stay far outside this.
    """
    return 2.5 * step + 0.01 + 0.0002 * price


def pooled_slope(
    days: list[str],
    kite: dict[str, np.ndarray],
    nse: dict[str, np.ndarray],
    steps: dict[str, float],
) -> float | None:
    """One affine fit over several days; None unless every point fits."""
    if not days:
        return None
    x = np.concatenate([nse[d] for d in days])
    y = np.concatenate([kite[d] for d in days])
    if (x.max() - x.min()) / np.median(x) < MIN_SPAN_FOR_EVENT:
        return None
    a, c = np.polyfit(x, y, 1)
    for day in days:
        if np.max(np.abs(a * nse[day] + c - kite[day])) > tolerance(
            steps[day], kite[day][3]
        ):
            return None
    return float(a)


def snap(ratio: float) -> tuple[float, bool]:
    best = min(SPLIT_BONUS_RATIOS, key=lambda f: abs(float(f) / ratio - 1))
    if abs(float(best) / ratio - 1) <= SNAP_TOLERANCE:
        return float(best), True
    return ratio, False


def event_ratio(
    ex_date: str,
    days: list[str],
    kite: dict[str, np.ndarray],
    nse: dict[str, np.ndarray],
    steps: dict[str, float],
) -> dict:
    """Kite's adjustment ratio at an NSE ex-date: slope before / slope after."""
    before = [d for d in days if d < ex_date]
    after = [d for d in days if d >= ex_date]
    for width in EVENT_WINDOWS:
        left = pooled_slope(before[-width:], kite, nse, steps)
        right = pooled_slope(after[:width], kite, nse, steps)
        if left is not None and right is not None:
            measured = left / right
            if abs(measured - 1) <= UNADJUSTED_TOLERANCE:
                return {"ex_date": ex_date, "status": "kite_unadjusted", "width": width}
            value, snapped = snap(measured)
            return {
                "ex_date": ex_date,
                "status": "adjusted",
                "measured": measured,
                "ratio": value,
                "snapped": snapped,
                "width": width,
            }
    return {"ex_date": ex_date, "status": "unresolved"}


def read_csv_rows(path: Path, header: list[str]) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != header:
            raise ValueError(f"Unexpected header in {path.name}")
        return list(reader)


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def day_quanta(minute_path: Path) -> dict[str, float]:
    samples: dict[str, list[str]] = {}
    with minute_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != MINUTE_HEADER:
            raise ValueError(f"Unexpected minute header in {minute_path.name}")
        for row in reader:
            bucket = samples.setdefault(row[0][:10], [])
            if len(bucket) < 4 * QUANTUM_SAMPLE:
                bucket.extend(row[1:5])
    return {day: quantum(values) for day, values in samples.items()}


def rebuild_instrument(task: tuple) -> dict:
    symbol, role, source, target, nse_rows, ex_dates = task
    store = CandleStore(source)
    minute_path = store.verified_path(f"NSE_{symbol}.csv")
    daily_path = store.verified_path(f"daily/NSE_{symbol}.csv")
    fetched_name = f"NSE_{symbol}.fetched.csv"
    out_minute = target / minute_path.name
    out_daily = target / "daily" / daily_path.name
    if role != "equity_target" and not symbol.endswith("BEES"):
        shutil.copyfile(minute_path, out_minute)
        shutil.copyfile(daily_path, out_daily)
        shutil.copyfile(store.verified_path(fetched_name), target / fetched_name)
        return {"symbol": symbol, "role": role, "basis": "copied_unchanged_index"}
    daily_rows = read_csv_rows(daily_path, DAILY_HEADER)
    kite_daily = {row[0][:10]: row for row in daily_rows}
    steps = day_quanta(minute_path)
    kite, nse = {}, {}
    for day, row in kite_daily.items():
        if day in nse_rows:
            kite[day] = np.array([float(x) for x in row[1:5]])
            nse[day] = np.array(nse_rows[day][:4])
            steps.setdefault(day, quantum(row[1:5]))
    days = sorted(kite)
    fits, dropped, residuals, imprecise = {}, Counter(), [], []
    last_slope = None
    for day in days:
        a, c, residual = fit_day(kite[day], nse[day], last_slope)
        if a <= 0 or residual > tolerance(steps[day], kite[day][3]):
            dropped["affine_misfit"] += 1
            continue
        if (nse[day].max() - nse[day].min()) / nse[day][3] >= MIN_SPAN_FOR_FIT:
            last_slope = a
        fits[day] = (a, c)
        residuals.append(residual / kite[day][3])
        if residual / kite[day][3] > IMPRECISE_RELATIVE:
            imprecise.append(day)
    dropped["no_nse_anchor"] = len(kite_daily) - len(days)
    fitted_days = sorted(fits)
    events = (
        [
            event_ratio(e, fitted_days, kite, nse, steps)
            for e in ex_dates
            if fitted_days[0] < e <= fitted_days[-1]
        ]
        if fitted_days
        else []
    )
    factor = dict.fromkeys(fits, 1.0)
    for event in events:
        before = [d for d in fitted_days if d < event["ex_date"]]
        after = [d for d in fitted_days if d >= event["ex_date"]]
        event["nse_open_over_prev_close"] = (
            float(nse[after[0]][0] / nse[before[-1]][3]) if before and after else None
        )
        if event["status"] == "adjusted" and event["snapped"]:
            event["ratio_used"] = event["ratio"]
        else:
            # Rights, demergers, schemes and anything Kite left unadjusted: assume
            # the whole ex-date opening move is the action; the date is flagged.
            event["ratio_used"] = event["nse_open_over_prev_close"] or 1.0
        if event["nse_open_over_prev_close"]:
            event["output_ex_date_gap"] = (
                event["nse_open_over_prev_close"] / event["ratio_used"] - 1
            )
        for day in factor:
            if day < event["ex_date"]:
                factor[day] *= event["ratio_used"]
    volume_scale, coarse, volume_basis = {}, [], 0
    for day, (a, _c) in fits.items():
        nse_volume = nse_rows[day][4] / factor[day]
        kite_volume = float(kite_daily[day][5] or 0)
        if kite_volume > 0:
            volume_scale[day] = nse_volume / kite_volume
            volume_basis += abs(volume_scale[day] - 1) > 0.02
        elif nse_volume == 0:
            volume_scale[day] = 1.0
        if steps[day] / kite[day][3] > COARSE_RELATIVE and steps[day] / a > RAW_TICK:
            coarse.append(day)
    keep = {day for day in fits if day in volume_scale}
    dropped["volume_mismatch"] = len(fits) - len(keep)
    with out_daily.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DAILY_HEADER)
        for row in daily_rows:
            day = row[0][:10]
            if day in keep:
                o, h, lo, c, v = nse_rows[day]
                s = factor[day]
                writer.writerow(
                    [row[0], *(fmt(x * s) for x in (o, h, lo, c)), round(v / s)]
                )
    minutes, outside = 0, Counter()
    with (
        minute_path.open(newline="", encoding="utf-8") as source_handle,
        out_minute.open("w", newline="", encoding="utf-8") as handle,
    ):
        reader = csv.reader(source_handle)
        next(reader)
        writer = csv.writer(handle)
        writer.writerow(MINUTE_HEADER)
        for stamp, open_, high, low, close, volume in reader:
            day = stamp[:10]
            if day not in keep:
                continue
            a, c = fits[day]
            s = factor[day]
            prices = [(float(x) - c) / a * s for x in (open_, high, low, close)]
            n_low, n_high = nse[day][2] * s, nse[day][1] * s
            slack = tolerance(steps[day], kite[day][3]) / a * s
            if prices[1] > n_high + slack or prices[2] < n_low - slack:
                outside[day] += 1
            traded = round(int(volume) * volume_scale[day]) if volume else ""
            writer.writerow([stamp, *(fmt(x) for x in prices), traded])
            minutes += 1
    shutil.copyfile(store.verified_path(fetched_name), target / fetched_name)
    identity = sum(1 for d in keep if np.allclose(kite[d], nse[d]) and factor[d] == 1.0)
    return {
        "symbol": symbol,
        "role": role,
        "basis": "nse_daily_anchor",
        "kite_sessions": len(kite_daily),
        "kept_sessions": len(keep),
        "identity_sessions": identity,
        "dropped": {k: v for k, v in dropped.items() if v},
        "minute_rows": minutes,
        "structural_events": events,
        "coarse_vendor_rounding_days": coarse,
        "imprecise_fit_days": len(set(imprecise) & keep),
        "volume_basis_rescaled_days": volume_basis,
        "minutes_outside_nse_range_days": dict(outside),
        "relative_residual_p99": float(np.percentile(residuals, 99))
        if residuals
        else None,
        "relative_residual_max": float(max(residuals)) if residuals else None,
        "exclusions": {
            "dropped": sorted(set(kite_daily) - keep),
            "coarse_vendor_rounding": coarse,
            "imprecise_fit": sorted(set(imprecise) & keep),
            "structural_break": [
                e["ex_date"]
                for e in events
                if not (e["status"] == "adjusted" and e["snapped"])
            ],
            "minutes_outside_nse_range": sorted(outside),
        },
    }


def run(
    source: Path,
    nse_dir: Path,
    changes: Path,
    actions_path: Path,
    target: Path,
    workers: int,
) -> dict:
    if target.exists():
        raise ValueError("Target dataset already exists; use a new directory")
    store = CandleStore(source)
    manifest = store.dataset
    if manifest is None:
        raise ValueError("Source is not a promoted Kite dataset")
    universe = json.loads(store.verified_path("universe.json").read_text("utf-8"))
    history = symbol_history(changes)
    wanted = {
        name
        for entry in universe["instruments"]
        for name in (
            entry["symbol"],
            *(old for _, old in history.get(entry["symbol"], [])),
        )
    }
    nse = read_nse(nse_dir, wanted)
    structural: dict[str, set[str]] = {}
    for row in json.loads(actions_path.read_text("utf-8"))["rows"]:
        ex_date = parse_nse_date(row.get("exDate"))
        if ex_date and action_kind(str(row.get("subject") or "")) in STRUCTURAL:
            structural.setdefault(str(row.get("symbol")).strip(), set()).add(
                ex_date.isoformat()
            )
    (target / "daily").mkdir(parents=True)
    tasks = []
    for entry in universe["instruments"]:
        symbol = entry["symbol"]
        names = {symbol} | {old for _, old in history.get(symbol, [])}
        anchors = {}
        for day_text, rows in nse.items():
            name = nse_symbol(symbol, date.fromisoformat(day_text), history)
            if name in rows:
                anchors[day_text] = rows[name]
        ex_dates = sorted({d for name in names for d in structural.get(name, set())})
        tasks.append((symbol, entry["role"], source, target, anchors, ex_dates))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(rebuild_instrument, tasks))
    shutil.copyfile(store.verified_path("universe.json"), target / "universe.json")
    exclusions = {
        r["symbol"]: r.pop("exclusions") for r in results if "exclusions" in r
    }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": str(source),
        "source_dataset_sha256": digest(source / "dataset.json"),
        "nse_reference_files": len(nse),
        "method": __doc__.strip(),
        "parameters": {
            "tolerance": "2.5*vendor_quantum + 0.01 + 0.0002*price (kite units)",
            "imprecise_relative_residual": IMPRECISE_RELATIVE,
            "coarse_relative_quantum": COARSE_RELATIVE,
            "snap_tolerance": SNAP_TOLERANCE,
            "unadjusted_tolerance": UNADJUSTED_TOLERANCE,
            "extra_symbol_changes": {
                k: [(d.isoformat(), old) for d, old in v]
                for k, v in EXTRA_SYMBOL_CHANGES.items()
            },
        },
        "instruments": results,
        "source_sha256": digest(Path(__file__)),
    }
    write_report(target / "report.json", report)
    write_report(target / "exclusions.json", exclusions)
    files = {
        path.relative_to(target).as_posix(): digest(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name not in ("report.json", "exclusions.json")
    }
    write_report(
        target / "dataset.json",
        {
            "schema_version": 1,
            "provider": "kite",
            "derived_from": str(source),
            "start": manifest["start"],
            "end": manifest["end"],
            "price_basis": (
                "Kite minutes mapped per day onto NSE official raw prices, then "
                "multiplied by later split/bonus/demerger ratios; dividends NOT "
                "adjusted; intraday percentages equal NSE's within the vendor "
                "rounding quantum. See report.json and exclusions.json."
            ),
            "session_dates": manifest["session_dates"],
            "files": files,
            "missing_policy": manifest["missing_policy"],
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--nse", type=Path, required=True)
    parser.add_argument("--symbol-changes", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    report = run(
        args.candles,
        args.nse,
        args.symbol_changes,
        args.actions,
        args.output,
        args.workers,
    )
    stocks = [r for r in report["instruments"] if r["basis"] == "nse_daily_anchor"]
    events = Counter(
        e["status"] + ("_snapped" if e.get("snapped") else "")
        for r in stocks
        for e in r["structural_events"]
    )
    print(
        json.dumps(
            {
                "anchored_instruments": len(stocks),
                "kite_sessions": sum(r["kite_sessions"] for r in stocks),
                "kept_sessions": sum(r["kept_sessions"] for r in stocks),
                "identity_sessions": sum(r["identity_sessions"] for r in stocks),
                "dropped": sum((Counter(r["dropped"]) for r in stocks), Counter()),
                "coarse_days": sum(
                    len(r["coarse_vendor_rounding_days"]) for r in stocks
                ),
                "outside_range_days": sum(
                    len(r["minutes_outside_nse_range_days"]) for r in stocks
                ),
                "structural_events": events,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
