"""Opening-range intraday strategy study on Kite minutes (research only, no orders).

Develop on January 2024 only, freeze the best predeclared variant, then evaluate it
unchanged on February 2024-December 2025 and January-September 2026.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ai_trader.costs import FIXED_CLIP_NOTIONAL, ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore
from scripts.research_exit_policy import block_interval, epoch_day
from scripts.train_tp_only import file_hash, write_json

FIRST_DAY, LAST_DAY = "2024-01-01", "2026-09-28"
SESSION_OPEN = 555
OR_LAST = 24
LAST_ENTRY = 254
DEADLINE = 344
RVOL_SESSIONS = 10
RVOL_MINIMUM = 3
STOP_FRACTION = 0.005
STRESS = 0.0004
CLIP_INR = 100_000
RANDOM_REPEATS = 200
SEED = 20261011
HEADER = ["start_time", "open", "high", "low", "close", "volume"]
DAILY_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
DEV = "dev_jan_2024"
PERIODS = {
    DEV: ("2024-01-01", "2024-01-31"),
    "oos_feb_2024_dec_2025": ("2024-02-01", "2025-12-31"),
    "oos_jan_sep_2026": ("2026-01-01", "2026-09-28"),
}
SELECTIONS = ("rvol", "gap", "move")
COUNTS = (5, 10)
DIRECTIONS = ("follow", "fade")
ENTRIES = ("at_0940", "breakout")
STOPS = ("range", "50bp", "none")
SOURCES = ("research_orb_strategy.py", "research_exit_policy.py", "train_tp_only.py")


def read_symbols(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted(row["Symbol"] for row in csv.DictReader(handle))


def read_minutes(
    path: Path, quarantined: set[str]
) -> dict[str, tuple[np.ndarray, int]]:
    rows: dict[str, list] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != HEADER:
            raise ValueError(f"Unexpected candle header in {path.name}")
        for start, open_, high, low, close, volume in reader:
            day = start[:10]
            if not FIRST_DAY <= day <= LAST_DAY or day in quarantined:
                continue
            if start[16:] != ":00+05:30":
                raise ValueError(f"Unexpected timestamp {start} in {path.name}")
            minute = int(start[11:13]) * 60 + int(start[14:16]) - SESSION_OPEN
            if 0 <= minute <= DEADLINE:
                rows[day].append((minute, open_, high, low, close, volume))
    sessions = {}
    for day, values in rows.items():
        values.sort()
        if [value[0] for value in values] != list(range(DEADLINE + 1)):
            continue
        scaled = np.asarray([value[1:5] for value in values], dtype=np.float64) * 100
        prices = np.rint(scaled).astype(np.int64)
        open_, high, low, close = prices.T
        if (
            np.abs(scaled - prices).max() > 1e-6
            or (low <= 0).any()
            or (high < np.maximum(open_, close)).any()
            or (low > np.minimum(open_, close)).any()
        ):
            raise ValueError(f"Invalid or non-paise prices on {day} in {path.name}")
        volume = sum(int(value[5]) for value in values[: OR_LAST + 1])
        sessions[day] = (prices, volume)
    return sessions


def read_previous_closes(path: Path) -> dict[str, float]:
    previous: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != DAILY_HEADER:
            raise ValueError(f"Unexpected daily header in {path.name}")
        last_day, last_close = "", None
        for timestamp, _, _, _, close, _ in reader:
            day = timestamp[:10]
            if day <= last_day:
                raise ValueError(f"Daily rows out of order in {path.name}")
            if last_close is not None:
                previous[day] = last_close
            last_day, last_close = day, float(close)
    return previous


def relative_volume(volume: np.ndarray) -> np.ndarray:
    result = np.full(len(volume), np.nan)
    for index in range(len(volume)):
        prior = volume[max(0, index - RVOL_SESSIONS) : index]
        if len(prior) >= RVOL_MINIMUM and prior.mean() > 0:
            result[index] = volume[index] / prior.mean()
    return result


def session_outcomes(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    open_, high, low, close = (prices[..., i].astype(np.float64) for i in range(4))
    count = len(prices)
    rows = np.arange(count)
    columns = np.arange(DEADLINE + 1)[None, :]
    range_high = high[:, : OR_LAST + 1].max(axis=1)
    range_low = low[:, : OR_LAST + 1].min(axis=1)
    gross = np.full((count, 2, len(ENTRIES), len(STOPS)), np.nan)
    stopped = np.zeros(gross.shape, dtype=bool)
    window = (columns > OR_LAST) & (columns <= LAST_ENTRY)
    for side, sign in enumerate((1, -1)):
        level = range_high if sign == 1 else range_low
        crossed = window & (
            high > level[:, None] if sign == 1 else low < level[:, None]
        )
        trigger = crossed.argmax(axis=1)
        gap_open = open_[rows, trigger]
        plans = (
            (close[:, OR_LAST], np.full(count, OR_LAST + 1), np.ones(count, bool)),
            (
                np.maximum(level, gap_open)
                if sign == 1
                else np.minimum(level, gap_open),
                trigger,
                crossed.any(axis=1),
            ),
        )
        for entry, (price, start, valid) in enumerate(plans):
            levels = (
                range_low if sign == 1 else range_high,
                price * (1 - sign * STOP_FRACTION),
                None,
            )
            for stop, stop_level in enumerate(levels):
                exit_price = close[:, DEADLINE].copy()
                hit = np.zeros(count, dtype=bool)
                if stop_level is not None:
                    beyond = (columns >= start[:, None]) & (
                        low <= stop_level[:, None]
                        if sign == 1
                        else high >= stop_level[:, None]
                    )
                    hit = beyond.any(axis=1) & valid
                    first = beyond.argmax(axis=1)
                    candle_open = open_[rows, first]
                    fill = (
                        np.minimum(stop_level, candle_open)
                        if sign == 1
                        else np.maximum(stop_level, candle_open)
                    )
                    if entry == 1:
                        fill = np.where(first == start, stop_level, fill)
                    exit_price = np.where(hit, fill, exit_price)
                gross[:, side, entry, stop] = np.where(
                    valid, sign * (exit_price / price - 1), np.nan
                )
                stopped[:, side, entry, stop] = hit
    return gross, stopped


def load_table(candles: Path, symbols: list[str]) -> dict[str, np.ndarray]:
    store = CandleStore(candles)
    quality = json.loads(store.verified_path("report.json").read_text("utf-8"))
    parts: dict[str, list] = defaultdict(list)
    for symbol in symbols:
        quarantined = set(
            quality["instruments"][f"NSE:{symbol}"]["quality"]["quarantined_dates"]
        )
        sessions = read_minutes(store.verified_path(f"NSE_{symbol}.csv"), quarantined)
        closes = read_previous_closes(store.verified_path(f"daily/NSE_{symbol}.csv"))
        days = sorted(sessions)
        prices = np.stack([sessions[day][0] for day in days])
        volume = np.asarray([sessions[day][1] for day in days], dtype=np.float64)
        previous = np.asarray([closes.get(day, np.nan) for day in days])
        gross, stopped = session_outcomes(prices)
        rvol = relative_volume(volume)
        gap = prices[:, 0, 0] / (100 * previous) - 1
        parts["day"].append(np.asarray([epoch_day(day) for day in days]))
        parts["symbol"].append(np.full(len(days), symbol))
        parts["or_return"].append(prices[:, OR_LAST, 3] / prices[:, 0, 0] - 1)
        parts["gap"].append(gap)
        parts["rvol"].append(rvol)
        parts["eligible"].append(np.isfinite(rvol) & np.isfinite(gap))
        parts["gross"].append(gross)
        parts["stopped"].append(stopped)
    table = {name: np.concatenate(values) for name, values in parts.items()}
    order = np.lexsort((table["symbol"], table["day"]))
    return {name: values[order] for name, values in table.items()}


def rank_rows(table: dict, selection: str) -> tuple[np.ndarray, np.ndarray]:
    if selection == "gap":
        signal, score = np.sign(table["gap"]), np.abs(table["gap"])
    else:
        signal = np.sign(table["or_return"])
        score = table["rvol"] if selection == "rvol" else np.abs(table["or_return"])
    eligible = table["eligible"] & (signal != 0)
    rank = np.full(len(score), np.iinfo(np.int64).max)
    bounds = np.flatnonzero(np.diff(table["day"])) + 1
    for rows in np.split(np.arange(len(score)), bounds):
        rows = rows[eligible[rows]]
        ordered = rows[np.argsort(-score[rows], kind="stable")]
        rank[ordered] = np.arange(len(ordered))
    return rank, np.nan_to_num(signal).astype(np.int64)


def variant_trades(
    table: dict,
    rank: np.ndarray,
    signal: np.ndarray,
    count: int,
    direction: str,
    entry: int,
    stop: int,
    flips: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(rank < count)
    sign = signal[rows] * (1 if direction == "follow" else -1)
    if flips is not None:
        sign = sign * flips
    side = np.where(sign == 1, 0, 1)
    gross = table["gross"][rows, side, entry, stop]
    stopped = table["stopped"][rows, side, entry, stop]
    traded = np.isfinite(gross)
    return rows[traded], gross[traded], stopped[traded]


def metrics(
    gross: np.ndarray, stopped: np.ndarray, day_ids: np.ndarray, sessions: int
) -> dict:
    charges = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(FIXED_CLIP_NOTIONAL))
    net = gross - charges
    daily = np.bincount(day_ids, weights=net, minlength=sessions)
    inr = CLIP_INR * daily
    result = {
        "sessions": sessions,
        "trades": len(gross),
        "trades_per_session": len(gross) / sessions,
        "net_inr_per_session": float(inr.mean()),
        "stress_inr_per_session": float(
            inr.mean() - CLIP_INR * STRESS * len(gross) / sessions
        ),
        "positive_sessions_percent": 100 * float(np.mean(inr > 0)),
        "total_net_inr": float(inr.sum()),
    }
    if not len(gross):
        return result
    equity = np.cumsum(inr)
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    deviation = inr.std(ddof=1)
    low, high = block_interval(daily)
    result.update(
        mean_gross_percent=100 * float(gross.mean()),
        mean_net_percent=100 * float(net.mean()),
        win_rate_percent=100 * float(np.mean(net > 0)),
        stop_rate_percent=100 * float(stopped.mean()),
        net_inr_per_session_ci95=[CLIP_INR / 100 * low, CLIP_INR / 100 * high],
        daily_t_stat=float(inr.mean() / deviation * np.sqrt(sessions))
        if deviation > 0
        else None,
        annualized_sharpe=float(inr.mean() / deviation * np.sqrt(250))
        if deviation > 0
        else None,
        max_drawdown_inr=float((peak - equity).max()),
    )
    return result


def period_days(table: dict) -> dict[str, tuple[int, int, np.ndarray]]:
    result = {}
    for name, (first, last) in PERIODS.items():
        start, end = epoch_day(first), epoch_day(last)
        inside = (table["day"] >= start) & (table["day"] <= end) & table["eligible"]
        result[name] = (start, end, np.unique(table["day"][inside]))
    return result


def evaluate(table: dict, rows, gross, stopped, periods: dict) -> dict:
    days = table["day"][rows]
    result = {}
    for name, (start, end, sessions) in periods.items():
        inside = (days >= start) & (days <= end)
        day_ids = np.searchsorted(sessions, days[inside])
        result[name] = metrics(gross[inside], stopped[inside], day_ids, len(sessions))
    return result


def run(candles: Path, constituents: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    charges = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(FIXED_CLIP_NOTIONAL))
    here = Path(__file__).parent
    sources = {name: file_hash(here / name) for name in SOURCES}
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "universe": f"{constituents.name} symbols, complete 09:15-15:00 minutes",
            "opening_range": "09:15-09:40 candles; signals known at 09:40",
            "selection": {
                "rvol": "top-k by opening-range volume / mean of up to 10 prior "
                "complete sessions (at least 3); direction = opening-range sign",
                "gap": "top-k by |open / previous vendor daily close - 1|; "
                "direction = gap sign",
                "move": "top-k by |09:40 close / 09:15 open - 1|; "
                "direction = opening-range sign",
            },
            "grid": {
                "k": COUNTS,
                "direction": "follow or fade the selection's sign",
                "entry": "at_0940 = 09:40 close; breakout = stop order at the "
                "opening-range high (LONG) or low (SHORT) from 09:40 to 13:30, "
                "filled at the level or a worse candle open",
                "stop": "range = opposite opening-range extreme; 50bp from entry; "
                "none. Fill at the stop or a worse candle open; a breakout's "
                "entry candle counts as stopped if it also reaches the stop",
                "exit": "14:59-15:00 candle close",
            },
            "sizing": "one Rs 1 lakh clip per selected stock, one trade per stock-day",
            "periods": PERIODS,
            "selection_rule": "highest mean net Rs per session in dev_jan_2024 "
            "only; evaluate the frozen variant unchanged on both OOS periods",
            "charges_round_trip_fraction": charges,
            "stress": f"extra {100 * STRESS:.2f}% round-trip slippage",
            "controls": f"opposite direction (in grid); {RANDOM_REPEATS} random-side "
            "repeats on the chosen variant's stock selection",
            "versions": {name: version(name) for name in ("numpy", "scipy")},
            "source_sha256": sources,
        },
    )
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)

    table = load_table(candles, read_symbols(constituents))
    periods = period_days(table)
    ranks = {selection: rank_rows(table, selection) for selection in SELECTIONS}
    records = []
    grid = product(SELECTIONS, COUNTS, DIRECTIONS, ENTRIES, STOPS)
    for selection, count, direction, entry, stop in grid:
        rows, gross, stopped = variant_trades(
            table,
            *ranks[selection],
            count,
            direction,
            ENTRIES.index(entry),
            STOPS.index(stop),
        )
        records.append(
            {
                "name": f"{selection}{count}_{direction}_{entry}_{stop}",
                "parameters": [selection, count, direction, entry, stop],
                **evaluate(table, rows, gross, stopped, periods),
            }
        )
    chosen = max(records, key=lambda record: record[DEV]["net_inr_per_session"])
    selection, count, direction, entry, stop = chosen["parameters"]
    generator = np.random.default_rng(SEED)
    rank, signal = ranks[selection]
    candidates = int((rank < count).sum())
    random_values = defaultdict(list)
    for _ in range(RANDOM_REPEATS):
        flips = generator.choice((-1, 1), size=candidates)
        result = evaluate(
            table,
            *variant_trades(
                table,
                rank,
                signal,
                count,
                direction,
                ENTRIES.index(entry),
                STOPS.index(stop),
                flips,
            ),
            periods,
        )
        for name in PERIODS:
            random_values[name].append(result[name]["net_inr_per_session"])
    random_control = {
        name: {
            "mean_net_inr_per_session": float(np.mean(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
            "share_at_or_above_chosen": float(
                np.mean(np.asarray(values) >= chosen[name]["net_inr_per_session"])
            ),
        }
        for name, values in random_values.items()
    }
    values = {
        name: np.asarray([record[name]["net_inr_per_session"] for record in records])
        for name in PERIODS
    }
    names = list(PERIODS)
    correlations = {
        f"{left}_vs_{right}": float(spearmanr(values[left], values[right]).statistic)
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    oos = names[1:]
    both_positive = sorted(
        (
            {
                "name": record["name"],
                **{name: record[name]["net_inr_per_session"] for name in names},
            }
            for record in records
            if all(record[name]["net_inr_per_session"] > 0 for name in oos)
        ),
        key=lambda item: -min(item[name] for name in oos),
    )
    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "coverage": {
            "symbols": int(len(np.unique(table["symbol"]))),
            "stock_sessions": int(len(table["day"])),
            "eligible_stock_sessions": int(table["eligible"].sum()),
            "sessions": {name: len(days) for name, (_, _, days) in periods.items()},
        },
        "charges_round_trip_fraction": charges,
        "chosen_on_dev": chosen,
        "random_side_control": random_control,
        "rank_correlation_of_variants": correlations,
        "post_hoc_positive_in_both_oos_periods": both_positive,
        "variants": records,
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--constituents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.candles, args.constituents, args.output)
    print(json.dumps({"chosen": report["chosen_on_dev"]["name"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
