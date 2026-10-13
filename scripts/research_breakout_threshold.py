"""NIFTY-direction opening-range breakout over the top-500 universe (research only).

The October 11 lead (``research_intraday_strategy.py``,
``breakout_0940_move5_to1130_range_market``) with the user's changes of
October 14: every stock whose 09:15-09:40 move is at least a threshold trades,
instead of the 5 biggest movers of the Nifty 50; Rs 4 lakh per trade; no limit on
concurrent positions (one per stock). Unchanged, through the same code: the side
is NIFTY's 09:15-09:40 direction; entry on the first break of the stock's
09:15-09:40 range that way until 11:30 (at the range level, or a worse candle
open); stop at the opposite end of the range; otherwise out at the 14:59-15:00
close; Zerodha intraday charges plus 0.03% slippage.

The threshold is chosen on 2024-2025 only, as the original study chose; the frozen
choice is then tested once on 2020-2023, never used for this family, and on
January-September 2026. Minutes are on NSE's price basis
(``data/candles_fno_basis_20261011``) and exist only for the universe's F&O stocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.costs import ZERODHA_INTRADAY_EQUITY
from scripts import research_intraday_strategy as base
from scripts.research_exit_policy import block_interval
from scripts.train_tp_only import file_hash, write_json

UNIVERSE = Path("data/universe_mcap_20261012/universe.json")
CANDLES = Path("data/candles_fno_basis_20261011")
UNIVERSE_SIZE = 500
TRADE_INR = 400_000
RANGE_END = base.RANGE_ENDS[0]  # the 09:39-09:40 bar
ENTRY_LIMIT = base.ENTRY_LIMITS[0]  # the 11:29-11:30 bar
DEADLINE = base.DEADLINE
WIDTH = DEADLINE + 1
THRESHOLDS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03)
THRESHOLDS += (0.04, 0.05)
FIRST_DAY, LAST_DAY = "2020-01-01", "2026-09-30"
TUNE = ("2024-2025", ("2024-01-01", "2025-12-31"))
TESTS = {"2020-2023": ("2020-01-01", "2023-12-31"), "2026": ("2026-01-01", LAST_DAY)}
YEARS = tuple(str(year) for year in range(2020, 2027))
SIDES = {"both": (0, 1), "long": (0,), "short": (1,)}
REPEATS = 20
CHARGES = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(Decimal(TRADE_INR)))
SOURCES = ("research_breakout_threshold.py", *base.SOURCES)


def top_universe(path: Path, size: int = UNIVERSE_SIZE) -> list[str]:
    instruments = json.loads(path.read_text("utf-8"))["instruments"]
    stocks = sorted(
        (item for item in instruments if item.get("role") == "equity_target"),
        key=lambda item: item["rank"],
    )
    return [item["symbol"] for item in stocks[:size]]


def excluded_days(path: Path) -> dict[str, frozenset[str]]:
    flags = json.loads(path.read_text("utf-8"))
    return {
        symbol: frozenset(day for days in kinds.values() for day in days)
        for symbol, kinds in flags.items()
    }


def read_days(
    path: Path,
    skip: frozenset[str] = frozenset(),
    first: str = FIRST_DAY,
    last: str = LAST_DAY,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Complete 09:15-15:00 sessions: (days, prices [n, 345, 4], days present)."""
    columns = ["start_time", "open", "high", "low", "close"]
    frame = pd.read_csv(path, usecols=columns, dtype={"start_time": str})
    stamp = frame["start_time"]
    if not stamp.str.endswith(":00+05:30").all():
        raise ValueError(f"Unexpected timestamps in {path.name}")
    day = stamp.str.slice(0, 10)
    minute = (
        stamp.str.slice(11, 13).astype(int) * 60
        + stamp.str.slice(14, 16).astype(int)
        - base.SESSION_OPEN
    )
    inside = (day >= first) & (day <= last)
    present = int(day[inside].nunique())
    keep = inside & minute.between(0, DEADLINE) & ~day.isin(skip)
    frame = frame.loc[keep, columns[1:]].assign(day=day[keep], minute=minute[keep])
    counts = frame["day"].value_counts()
    frame = frame[frame["day"].isin(counts.index[counts == WIDTH])]
    frame = frame.sort_values(["day", "minute"], kind="stable")
    minutes = frame["minute"].to_numpy().reshape(-1, WIDTH)
    prices = frame[columns[1:]].to_numpy(float).reshape(-1, WIDTH, 4)
    days = frame["day"].to_numpy().reshape(-1, WIDTH)[:, 0].astype(str)
    open_, high, low, close = np.moveaxis(prices, -1, 0)
    good = (
        (minutes == np.arange(WIDTH)).all(axis=1)
        & (low > 0).all(axis=1)
        & (high >= np.maximum(open_, close) - 1e-6).all(axis=1)
        & (low <= np.minimum(open_, close) + 1e-6).all(axis=1)
    )
    return days[good], prices[good], present


def stock_outcomes(task: tuple[str, str, frozenset[str]]) -> dict:
    """Both sides' breakout outcomes for every complete day of one stock."""
    symbol, path, skip = task
    days, prices, present = read_days(Path(path), skip)
    result = {"symbol": symbol, "days": days, "present": present}
    if not len(days):
        return result
    gross, exits, trigger = base.breakout_outcomes(prices, RANGE_END)
    close = prices[..., 3]
    generator = np.random.default_rng([base.SEED, zlib.crc32(symbol.encode())])
    moments = generator.integers(RANGE_END + 1, ENTRY_LIMIT + 1, (len(days), REPEATS))
    sides = generator.choice((-1.0, 1.0), size=(len(days), REPEATS))
    entry = np.take_along_axis(close, moments, axis=1)
    return result | {
        "move": np.abs(close[:, RANGE_END] / prices[:, 0, 0] - 1),
        "trigger": trigger,
        "gross": gross[:, :, 0, :],  # [day, side, fill], stop at the far end
        "exit": exits[:, :, 0, :],
        "control": (sides * (close[:, DEADLINE, None] / entry - 1)).astype(np.float32),
    }


def nifty_direction(path: Path, skip: frozenset[str] = frozenset()) -> dict[str, int]:
    days, prices, _ = read_days(path, skip)
    signs = np.sign(prices[:, RANGE_END, 3] - prices[:, 0, 0]).astype(int)
    return dict(zip(days.tolist(), signs.tolist(), strict=True))


def build_table(results: list[dict], direction: dict[str, int]) -> dict:
    parts = [result for result in results if len(result["days"])]
    day = np.concatenate([part["days"] for part in parts])
    table = {
        "day": day,
        "symbol": np.concatenate([np.full(len(p["days"]), p["symbol"]) for p in parts]),
    }
    for name in ("move", "trigger", "gross", "exit", "control"):
        table[name] = np.concatenate([part[name] for part in parts])
    sign = np.asarray([direction.get(label, 0) for label in day.tolist()])
    side = np.where(sign > 0, 0, np.where(sign < 0, 1, -1))
    rows, pick = np.arange(len(day)), np.maximum(side, 0)
    entry = table["trigger"][rows, pick].astype(np.int64)
    table |= {
        "side": side,
        "entry": entry,
        "valid": (side >= 0) & (entry > RANGE_END) & (entry <= ENTRY_LIMIT),
        "gross_touch": table["gross"][rows, pick, 0],
        "gross_close": table["gross"][rows, pick, 1],
        "exit_touch": table["exit"][rows, pick, 0].astype(np.int64),
    }
    order = np.lexsort((table["symbol"], day))
    return {name: values[order] for name, values in table.items()}


def chosen_rows(table: dict, threshold: float, sides: tuple = (0, 1)) -> np.ndarray:
    mask = table["valid"] & (table["move"] >= threshold) & np.isin(table["side"], sides)
    return np.flatnonzero(mask)


def sessions_between(direction: dict[str, int], first: str, last: str) -> np.ndarray:
    return np.asarray(sorted(day for day in direction if first <= day <= last))


def summarize(
    table: dict,
    rows: np.ndarray,
    sessions: np.ndarray,
    gross: str = "gross_touch",
    slippage: float = base.SLIPPAGE,
) -> dict:
    """Net results of trading ``rows`` (Rs 4 lakh each) over ``sessions``."""
    rows = rows[np.isin(table["day"][rows], sessions)]
    index = np.searchsorted(sessions, table["day"][rows])
    net = table[gross][rows] - CHARGES - slippage
    daily = np.bincount(index, weights=TRADE_INR * net, minlength=len(sessions))
    counts = np.bincount(index, minlength=len(sessions))
    result = {
        "sessions": len(sessions),
        "trades": len(rows),
        "trades_per_day": float(len(rows) / max(len(sessions), 1)),
        "max_trades_in_a_day": int(counts.max()) if len(sessions) else 0,
        "net_inr_per_day": float(daily.mean()) if len(sessions) else 0.0,
    }
    if not len(rows):
        return result
    equity = np.cumsum(daily)
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    deviation = daily.std(ddof=1)
    low, high = block_interval(daily)
    held = np.zeros((len(sessions), WIDTH + 1), dtype=np.int64)
    np.add.at(held, (index, table["entry"][rows]), 1)
    np.add.at(held, (index, table["exit_touch"][rows] + 1), -1)
    held = np.cumsum(held, axis=1)
    result.update(
        mean_gross_percent=100 * float(table[gross][rows].mean()),
        mean_net_percent=100 * float(net.mean()),
        win_rate_percent=100 * float((net > 0).mean()),
        net_inr_per_day_ci95=[low / 100, high / 100],
        daily_t_stat=float(daily.mean() / deviation * np.sqrt(len(daily)))
        if deviation > 0
        else None,
        positive_days_percent=100 * float((daily > 0).mean()),
        max_drawdown_inr=float((peak - equity).max()),
        max_open_positions=int(held.max()),
        mean_daily_max_open_positions=float(held.max(axis=1).mean()),
    )
    return result


def compact(summary: dict) -> dict:
    keys = (
        "trades_per_day",
        "mean_net_percent",
        "win_rate_percent",
        "net_inr_per_day",
        "daily_t_stat",
        "max_open_positions",
    )
    return {key: summary.get(key) for key in keys}


def choose(curve: list[dict]) -> tuple[float, bool]:
    """Highest 2024-2025 net Rs per day among thresholds positive in every half."""
    robust = [entry for entry in curve if min(entry["tune_halves_net_inr_per_day"]) > 0]
    best = max(
        robust or curve, key=lambda entry: entry["both"][TUNE[0]]["net_inr_per_day"]
    )
    return best["threshold_percent"] / 100, bool(robust)


def random_control(table: dict, rows: np.ndarray, sessions: np.ndarray) -> dict:
    rows = rows[np.isin(table["day"][rows], sessions)]
    if not len(rows):
        return {}
    repeats = table["control"][rows].astype(np.float64).mean(axis=0)
    actual = float(table["gross_touch"][rows].mean())
    return {
        "actual_mean_gross_percent": 100 * actual,
        "random_mean_gross_percent": 100 * float(repeats.mean()),
        "random_max_gross_percent": 100 * float(repeats.max()),
        "share_of_random_at_least_actual": float((repeats >= actual).mean()),
    }


def clock(minute: int) -> str:
    total = base.SESSION_OPEN + int(minute)
    return f"{total // 60:02d}:{total % 60:02d}"


def protocol() -> dict:
    here = Path(__file__).parent
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "universe": f"top {UNIVERSE_SIZE} of {UNIVERSE} (market cap, 12 Oct 2026) "
        f"that have minutes in {CANDLES} (the F&O stocks); today's list applied "
        "to every year (survivorship)",
        "rules": "side = sign of NIFTY's 09:40 close minus its 09:15 open (none if "
        "flat); a stock trades when |its 09:40 close / 09:15 open - 1| >= the "
        "threshold and it first breaks its 09:15-09:40 range on that side by "
        "11:30; entry at the range level or a worse candle open; stop at the "
        "opposite end of the range (a breakout candle reaching it is stopped); "
        "otherwise exit at the 14:59-15:00 close; one trade per stock per day; "
        f"Rs {TRADE_INR:,} per trade; no limit on concurrent positions",
        "thresholds_percent": [100 * value for value in THRESHOLDS],
        "selection_rule": "on 2024-2025 only: among thresholds with positive mean "
        "net Rs per day in each half-year, the highest mean net Rs per day (else "
        "the highest overall, flagged not robust); the choice is then frozen",
        "tests": TESTS,
        "costs": {
            "charges_round_trip": CHARGES,
            "slippage_round_trip": base.SLIPPAGE,
            "stress_slippage_round_trip": base.STRESS,
        },
        "controls": f"{REPEATS} random entry times (09:41-11:30) and sides on the "
        "chosen stock-days, held to 15:00; close-of-trigger-candle fills; doubled "
        "slippage",
        "exclusions": f"{CANDLES / 'exclusions.json'} stock-days and days without "
        "all 345 minutes 09:15-14:59",
        "source_sha256": {name: file_hash(here / name) for name in SOURCES},
    }


def run(output: Path, workers: int) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol())
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(Path(__file__).parent / name, output / "code" / name)

    symbols = top_universe(UNIVERSE)
    skip = excluded_days(CANDLES / "exclusions.json")
    available = [s for s in symbols if (CANDLES / f"NSE_{s}.csv").exists()]
    direction = nifty_direction(
        CANDLES / "NSE_NIFTY.csv", skip.get("NIFTY", frozenset())
    )
    tasks = [
        (s, str(CANDLES / f"NSE_{s}.csv"), skip.get(s, frozenset())) for s in available
    ]
    print(
        f"{len(available)} of {len(symbols)} stocks have minutes; reading", flush=True
    )
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number, result in enumerate(pool.map(stock_outcomes, tasks), 1):
            results.append(result)
            if number % 25 == 0 or number == len(tasks):
                print(f"  {number}/{len(tasks)} stocks read", flush=True)
    table = build_table(results, direction)

    periods = {TUNE[0]: sessions_between(direction, *TUNE[1])}
    periods |= {
        name: sessions_between(direction, *bounds) for name, bounds in TESTS.items()
    }
    halves = [sessions_between(direction, *bounds) for bounds in base.HALVES]
    curve = []
    for threshold in THRESHOLDS:
        entry: dict = {"threshold_percent": 100 * threshold}
        for label, sides in SIDES.items():
            rows = chosen_rows(table, threshold, sides)
            entry[label] = {
                name: compact(summarize(table, rows, sessions))
                for name, sessions in periods.items()
            }
        rows = chosen_rows(table, threshold)
        entry["tune_halves_net_inr_per_day"] = [
            summarize(table, rows, half)["net_inr_per_day"] for half in halves
        ]
        curve.append(entry)
        print(f"  threshold {100 * threshold:.2f}% done", flush=True)
    threshold, robust = choose(curve)
    rows = chosen_rows(table, threshold)
    evaluation = {}
    for name, sessions in periods.items():
        evaluation[name] = {
            label: summarize(table, chosen_rows(table, threshold, sides), sessions)
            for label, sides in SIDES.items()
        } | {
            "stress_slippage": summarize(table, rows, sessions, slippage=base.STRESS),
            "close_of_trigger_fill": summarize(table, rows, sessions, "gross_close"),
            "random_control": random_control(table, rows, sessions),
        }
    years = {
        year: compact(
            summarize(
                table,
                rows,
                sessions_between(direction, f"{year}-01-01", f"{year}-12-31"),
            )
        )
        for year in YEARS
    }
    with (output / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "day",
                "symbol",
                "side",
                "move_0940_pct",
                "entry",
                "exit",
                "gross_pct",
                "net_pct",
            ]
        )
        for row in rows:
            gross = float(table["gross_touch"][row])
            writer.writerow(
                [
                    table["day"][row],
                    table["symbol"][row],
                    ("LONG", "SHORT")[table["side"][row]],
                    f"{100 * table['move'][row]:.3f}",
                    clock(table["entry"][row]),
                    clock(table["exit_touch"][row]),
                    f"{100 * gross:.4f}",
                    f"{100 * (gross - CHARGES - base.SLIPPAGE):.4f}",
                ]
            )
    present = sum(result["present"] for result in results)
    report = {
        "status": "complete",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "coverage": {
            "universe": len(symbols),
            "with_minutes": len(available),
            "without_minutes": [s for s in symbols if s not in available],
            "stock_days_present": present,
            "stock_days_used": int(len(table["day"])),
            "nifty_days": len(direction),
        },
        "curve": curve,
        "chosen_threshold_percent": 100 * threshold,
        "chosen_robust_in_every_half": robust,
        "evaluation": evaluation,
        "years": years,
    }
    write_json(output / "report.json", report)
    return report


def show(report: dict) -> None:
    def cell(values: dict) -> str:
        net, t = values.get("mean_net_percent"), values.get("daily_t_stat")
        if net is None or t is None:
            return "no trades".ljust(36)
        rate = f"{values['trades_per_day']:5.1f}/day"
        return f"{rate} {net:+.3f}% Rs{values['net_inr_per_day']:+8,.0f} t{t:+.1f}"

    print("move >= | 2024-25 (chooses)  | 2020-23 (test)  | 2026 (test)")
    for entry in report["curve"]:
        cells = " | ".join(cell(entry["both"][name]) for name in (TUNE[0], *TESTS))
        print(f"{entry['threshold_percent']:5.2f}%  | {cells}")
    print(
        f"chosen {report['chosen_threshold_percent']:.2f}% "
        f"(positive in every 2024-25 half: {report['chosen_robust_in_every_half']})"
    )
    for name, result in report["evaluation"].items():
        print(
            f"{name}: " + " | ".join(f"{side} {cell(result[side])}" for side in SIDES)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    show(run(args.output, args.workers))


if __name__ == "__main__":
    main()
