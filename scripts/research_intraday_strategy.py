"""Intraday strategy search on Kite minutes (research only, no orders).

Select on 2024-2025 (every half-year must be profitable), then evaluate the frozen
choice once on January-September 2026. Sizing: Rs 1 lakh margin at 5x MIS leverage,
split across at most 1, 2 or 5 concurrent positions.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from functools import cache
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ai_trader.costs import ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore
from scripts.research_exit_policy import block_interval, epoch_day
from scripts.research_orb_strategy import HEADER, read_symbols, relative_volume
from scripts.train_tp_only import file_hash, write_json

FIRST_DAY, LAST_DAY = "2024-01-01", "2026-09-28"
SESSION_OPEN = 555
DEADLINE = 344
RANGE_ENDS = (24, 59)
RANGE_NAMES = ("0940", "1015")
ENTRY_LIMITS = (134, 254)
LIMIT_NAMES = ("1130", "1330")
AFTERNOON = 254
STOPS = ("range", "mid", "none")
FILLS = ("touch", "close")
SCORES = ("rvol", "move", "width")
COUNTS = (3, 5, 10, 0)
MODES = ("oco", "market")
SLOTS = (1, 2, 5)
MARGIN_INR = 100_000
LEVERAGE = 5
SLIPPAGE = 0.0003
STRESS = 0.0006
STOP_FRACTION = 0.005
RANDOM_REPEATS = 200
SEED = 20261011
TRAIN = ("2024-01-01", "2025-12-31")
TEST = ("2026-01-01", "2026-09-28")
HALVES = (
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
)
SOURCES = (
    "research_intraday_strategy.py",
    "research_orb_strategy.py",
    "research_exit_policy.py",
    "train_tp_only.py",
)


def clip_inr(slots: int) -> int:
    return MARGIN_INR * LEVERAGE // slots


@cache
def charges(slots: int) -> float:
    return float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(Decimal(clip_inr(slots))))


def read_sessions(
    path: Path, quarantined: set[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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
        volume = np.asarray([int(value[5] or 0) for value in values], dtype=np.int64)
        sessions[day] = (prices, volume)
    return sessions


def breakout_outcomes(
    prices: np.ndarray, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    open_, high, low, close = (prices[..., i].astype(np.float64) for i in range(4))
    count = len(prices)
    rows = np.arange(count)
    columns = np.arange(DEADLINE + 1)[None, :]
    top = high[:, : end + 1].max(axis=1)
    bottom = low[:, : end + 1].min(axis=1)
    middle = (top + bottom) / 2
    window = (columns > end) & (columns <= ENTRY_LIMITS[-1])
    gross = np.full((count, 2, len(STOPS), len(FILLS)), np.nan)
    exits = np.full(gross.shape, DEADLINE, dtype=np.int16)
    trigger = np.full((count, 2), -1, dtype=np.int16)
    for side, sign in enumerate((1, -1)):
        level = top if sign == 1 else bottom
        crossed = window & (
            high > level[:, None] if sign == 1 else low < level[:, None]
        )
        valid = crossed.any(axis=1)
        first = crossed.argmax(axis=1)
        trigger[:, side] = np.where(valid, first, -1)
        candle_open = open_[rows, first]
        touch = (
            np.maximum(level, candle_open)
            if sign == 1
            else np.minimum(level, candle_open)
        )
        for fill, (price, start) in enumerate(
            ((touch, first), (close[rows, first], first + 1))
        ):
            levels = (bottom if sign == 1 else top, middle, None)
            for stop, stop_level in enumerate(levels):
                exit_price = close[:, DEADLINE].copy()
                exit_at = np.full(count, DEADLINE)
                if stop_level is not None:
                    beyond = (columns >= start[:, None]) & (
                        low <= stop_level[:, None]
                        if sign == 1
                        else high >= stop_level[:, None]
                    )
                    hit = beyond.any(axis=1) & valid
                    hit_at = beyond.argmax(axis=1)
                    hit_open = open_[rows, hit_at]
                    stopped = (
                        np.minimum(stop_level, hit_open)
                        if sign == 1
                        else np.maximum(stop_level, hit_open)
                    )
                    if fill == 0:
                        stopped = np.where(hit_at == start, stop_level, stopped)
                    exit_price = np.where(hit, stopped, exit_price)
                    exit_at = np.where(hit, hit_at, DEADLINE)
                gross[:, side, stop, fill] = np.where(
                    valid, sign * (exit_price / price - 1), np.nan
                )
                exits[:, side, stop, fill] = exit_at
    return gross, exits, trigger


def afternoon_outcomes(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    open_, high, low, close = (prices[..., i].astype(np.float64) for i in range(4))
    count = len(prices)
    rows = np.arange(count)
    columns = np.arange(DEADLINE + 1)[None, :]
    entry = close[:, AFTERNOON]
    trend = np.sign(entry - open_[:, 0])
    gross = np.full((count, 2, 2), np.nan)
    exits = np.full(gross.shape, DEADLINE, dtype=np.int16)
    for direction, flip in enumerate((1, -1)):
        side = trend * flip
        for stop, fraction in enumerate((None, STOP_FRACTION)):
            exit_price = close[:, DEADLINE].copy()
            exit_at = np.full(count, DEADLINE)
            if fraction is not None:
                level = entry * (1 - side * fraction)
                beyond = (columns > AFTERNOON) & np.where(
                    side[:, None] == 1, low <= level[:, None], high >= level[:, None]
                )
                hit = beyond.any(axis=1) & (side != 0)
                hit_at = beyond.argmax(axis=1)
                hit_open = open_[rows, hit_at]
                stopped = np.where(
                    side == 1, np.minimum(level, hit_open), np.maximum(level, hit_open)
                )
                exit_price = np.where(hit, stopped, exit_price)
                exit_at = np.where(hit, hit_at, DEADLINE)
            gross[:, direction, stop] = np.where(
                side != 0, side * (exit_price / entry - 1), np.nan
            )
            exits[:, direction, stop] = exit_at
    return gross, exits


def load_table(candles: Path, symbols: list[str]) -> dict[str, np.ndarray]:
    store = CandleStore(candles)
    quality = json.loads(store.verified_path("report.json").read_text("utf-8"))

    def quarantined(name: str) -> set[str]:
        return set(
            quality["instruments"][f"NSE:{name}"]["quality"]["quarantined_dates"]
        )

    market = {}
    index = read_sessions(store.verified_path("NSE_NIFTY.csv"), quarantined("NIFTY"))
    for day, (prices, _) in index.items():
        marks = [prices[end, 3] for end in (*RANGE_ENDS, AFTERNOON)]
        market[epoch_day(day)] = np.sign(np.asarray(marks) - prices[0, 0])
    parts: dict[str, list] = defaultdict(list)
    for symbol in symbols:
        sessions = read_sessions(
            store.verified_path(f"NSE_{symbol}.csv"), quarantined(symbol)
        )
        days = sorted(sessions)
        prices = np.stack([sessions[day][0] for day in days])
        volume = np.stack([sessions[day][1] for day in days]).astype(np.float64)
        epoch = np.asarray([epoch_day(day) for day in days])
        start = prices[:, 0, 0].astype(np.float64)
        outcomes = [breakout_outcomes(prices, end) for end in RANGE_ENDS]
        afternoon = afternoon_outcomes(prices)
        rvol = np.stack(
            [relative_volume(volume[:, : end + 1].sum(axis=1)) for end in RANGE_ENDS],
            axis=1,
        )
        parts["day"].append(epoch)
        parts["symbol"].append(np.full(len(days), symbol))
        parts["rvol"].append(rvol)
        parts["move"].append(
            np.stack([np.abs(prices[:, end, 3] / start - 1) for end in RANGE_ENDS], 1)
        )
        parts["width"].append(
            np.stack(
                [
                    (prices[:, : end + 1, 1].max(1) - prices[:, : end + 1, 2].min(1))
                    / start
                    for end in RANGE_ENDS
                ],
                axis=1,
            )
        )
        parts["pm_move"].append(np.abs(prices[:, AFTERNOON, 3] / start - 1))
        parts["pm_sign"].append(np.sign(prices[:, AFTERNOON, 3] - prices[:, 0, 0]))
        parts["gross_a"].append(np.stack([item[0] for item in outcomes], axis=1))
        parts["exit_a"].append(np.stack([item[1] for item in outcomes], axis=1))
        parts["trigger"].append(np.stack([item[2] for item in outcomes], axis=1))
        parts["gross_b"].append(afternoon[0])
        parts["exit_b"].append(afternoon[1])
        parts["closes"].append(prices[..., 3].astype(np.int32))
        parts["market"].append(
            np.stack([market.get(day, np.zeros(3)) for day in epoch]).astype(np.int8)
        )
        parts["eligible"].append(np.isfinite(rvol).all(axis=1))
    table = {name: np.concatenate(values) for name, values in parts.items()}
    order = np.lexsort((table["symbol"], table["day"]))
    return {name: values[order] for name, values in table.items()}


def day_ranks(day: np.ndarray, score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    rank = np.full(len(day), np.iinfo(np.int64).max)
    bounds = np.flatnonzero(np.diff(day)) + 1
    for rows in np.split(np.arange(len(day)), bounds):
        rows = rows[eligible[rows] & np.isfinite(score[rows])]
        ordered = rows[np.argsort(-score[rows], kind="stable")]
        rank[ordered] = np.arange(len(ordered))
    return rank


def breakout_trades(
    table: dict, end: int, limit: int, stop: int, mode: str, fill: int = 0
) -> tuple[np.ndarray, ...]:
    trigger = table["trigger"][:, end].astype(np.int64)
    valid = (trigger >= 0) & (trigger <= ENTRY_LIMITS[limit])
    if mode == "oco":
        long_first = valid[:, 0] & (~valid[:, 1] | (trigger[:, 0] < trigger[:, 1]))
        short_first = valid[:, 1] & (~valid[:, 0] | (trigger[:, 1] < trigger[:, 0]))
        side = np.where(long_first, 0, np.where(short_first, 1, -1))
    else:
        sign = table["market"][:, end]
        side = np.where(sign > 0, 0, np.where(sign < 0, 1, -1))
        side = np.where(
            (side >= 0) & valid[np.arange(len(side)), np.maximum(side, 0)], side, -1
        )
    rows = np.flatnonzero(side >= 0)
    chosen = side[rows]
    return (
        rows,
        trigger[rows, chosen],
        table["exit_a"][rows, end, chosen, stop, fill].astype(np.int64),
        table["gross_a"][rows, end, chosen, stop, fill],
    )


def afternoon_trades(
    table: dict, direction: int, stop: int, aligned: bool
) -> tuple[np.ndarray, ...]:
    gross = table["gross_b"][:, direction, stop]
    usable = np.isfinite(gross)
    if aligned:
        side = table["pm_sign"] * (1 if direction == 0 else -1)
        usable &= side == table["market"][:, 2]
    rows = np.flatnonzero(usable)
    return (
        rows,
        np.full(len(rows), AFTERNOON),
        table["exit_b"][rows, direction, stop].astype(np.int64),
        gross[rows],
    )


def capacity(
    day: np.ndarray, rank: np.ndarray, entry: np.ndarray, exit_: np.ndarray, slots: int
) -> np.ndarray:
    accepted = np.zeros(len(day), dtype=bool)
    if not len(day):
        return accepted
    order = np.lexsort((rank, entry, day))
    bounds = np.flatnonzero(np.diff(day[order])) + 1
    for chunk in np.split(order, bounds):
        if len(chunk) <= slots:
            accepted[chunk] = True
            continue
        holding: list[int] = []
        for row in chunk:
            holding = [ending for ending in holding if ending >= entry[row]]
            if len(holding) < slots:
                holding.append(int(exit_[row]))
                accepted[row] = True
    return accepted


def period_sessions(table: dict, first: str, last: str) -> np.ndarray:
    start, end = epoch_day(first), epoch_day(last)
    inside = (table["day"] >= start) & (table["day"] <= end) & table["eligible"]
    return np.unique(table["day"][inside])


def daily_inr(
    days: np.ndarray,
    gross: np.ndarray,
    sessions: np.ndarray,
    slots: int,
    slippage: float,
) -> np.ndarray:
    inside = np.isin(days, sessions)
    weights = clip_inr(slots) * (gross[inside] - charges(slots) - slippage)
    return np.bincount(
        np.searchsorted(sessions, days[inside]),
        weights=weights,
        minlength=len(sessions),
    )


def summary(
    days: np.ndarray,
    gross: np.ndarray,
    sessions: np.ndarray,
    slots: int,
    slippage: float,
) -> dict:
    inside = np.isin(days, sessions)
    values = gross[inside]
    daily = daily_inr(days, gross, sessions, slots, slippage)
    result = {
        "sessions": len(sessions),
        "trades": int(inside.sum()),
        "trades_per_session": float(inside.sum() / len(sessions)),
        "net_inr_per_session": float(daily.mean()),
        "annual_return_on_margin_percent": float(100 * 250 * daily.mean() / MARGIN_INR),
    }
    if not len(values):
        return result
    equity = np.cumsum(daily)
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    deviation = daily.std(ddof=1)
    low, high = block_interval(daily)
    result.update(
        mean_gross_percent=100 * float(values.mean()),
        mean_net_percent=100 * float(values.mean() - charges(slots) - slippage),
        win_rate_percent=100 * float(np.mean(values > charges(slots) + slippage)),
        positive_sessions_percent=100 * float(np.mean(daily > 0)),
        net_inr_per_session_ci95=[low / 100, high / 100],
        daily_t_stat=float(daily.mean() / deviation * np.sqrt(len(daily)))
        if deviation > 0
        else None,
        annualized_sharpe=float(daily.mean() / deviation * np.sqrt(250))
        if deviation > 0
        else None,
        max_drawdown_inr=float((peak - equity).max()),
    )
    return result


def variant_grid() -> list[dict]:
    grid = []
    for end, score, count, limit, stop, mode, slots in product(
        range(2), SCORES, COUNTS, range(2), range(3), MODES, SLOTS
    ):
        if count == 0 and score != SCORES[0]:
            continue
        label = "all" if count == 0 else f"{score}{count}"
        grid.append(
            {
                "name": f"breakout_{RANGE_NAMES[end]}_{label}_"
                f"to{LIMIT_NAMES[limit]}_{STOPS[stop]}_{mode}_p{slots}",
                "family": "breakout",
                "end": end,
                "score": score,
                "count": count,
                "limit": limit,
                "stop": stop,
                "mode": mode,
                "slots": slots,
            }
        )
    for count, direction, stop, aligned, slots in product(
        COUNTS[:3], range(2), range(2), (False, True), SLOTS
    ):
        grid.append(
            {
                "name": f"afternoon_{('follow', 'fade')[direction]}{count}_"
                f"{('none', '50bp')[stop]}_{'market' if aligned else 'any'}_p{slots}",
                "family": "afternoon",
                "count": count,
                "direction": direction,
                "stop": stop,
                "aligned": aligned,
                "slots": slots,
            }
        )
    return grid


def variant_trades(table: dict, ranks: dict, variant: dict, fill: int = 0) -> tuple:
    if variant["family"] == "breakout":
        rows, entry, exit_, gross = breakout_trades(
            table,
            variant["end"],
            variant["limit"],
            variant["stop"],
            variant["mode"],
            fill,
        )
        rank = ranks[(variant["score"], variant["end"])][rows]
    else:
        rows, entry, exit_, gross = afternoon_trades(
            table, variant["direction"], variant["stop"], variant["aligned"]
        )
        rank = ranks[("pm", 0)][rows]
    count = variant["count"] or np.iinfo(np.int64).max
    keep = (rank < count) & table["eligible"][rows]
    rows, entry, exit_, gross, rank = (
        values[keep] for values in (rows, entry, exit_, gross, rank)
    )
    taken = capacity(table["day"][rows], rank, entry, exit_, variant["slots"])
    return rows[taken], entry[taken], gross[taken]


def random_control(
    table: dict, rows: np.ndarray, first: np.ndarray, last: int, repeats: int
) -> np.ndarray:
    generator = np.random.default_rng(SEED)
    closes = table["closes"][rows].astype(np.float64)
    exit_price = closes[:, DEADLINE]
    means = []
    for _ in range(repeats):
        moment = generator.integers(first, last + 1)
        side = generator.choice((-1, 1), size=len(rows))
        entry = closes[np.arange(len(rows)), moment]
        means.append(float(np.mean(side * (exit_price / entry - 1))))
    return np.asarray(means)


def run(candles: Path, constituents: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    here = Path(__file__).parent
    sources = {name: file_hash(here / name) for name in SOURCES}
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "universe": f"{constituents.name} symbols with complete 09:15-15:00 "
            "minutes and at least 3 prior sessions for relative volume",
            "families": {
                "breakout": "opening range 09:15-09:40 or 09:15-10:15; stocks ranked "
                "by relative range volume, |range move| or range width (or all); "
                "oco = first break of range high (LONG) or low (SHORT), market = "
                "only the side of NIFTY's move over the same range; triggers until "
                "11:30 or 13:30; stop at opposite range end, range middle or none",
                "afternoon": "at 13:30 rank by |move since 09:15 open|; follow or "
                "fade it at the 13:30 close; stop 0.5% or none; optional NIFTY "
                "direction filter",
            },
            "exit": "stop or the 14:59-15:00 candle close",
            "fills": "touch at the trigger level or a worse candle open (a "
            "breakout candle reaching its stop counts as stopped); robustness: "
            "close of the trigger candle",
            "sizing": f"Rs {MARGIN_INR} margin x {LEVERAGE} leverage split into "
            f"{SLOTS} max concurrent positions; first triggers taken, ties by rank",
            "costs": {
                "charges_round_trip": {str(s): charges(s) for s in SLOTS},
                "slippage_round_trip": SLIPPAGE,
                "stress_round_trip": STRESS,
            },
            "selection_rule": "highest mean net Rs per session over 2024-2025 among "
            "variants with positive mean net Rs in each of the four half-years; "
            "evaluate the frozen choice once on Jan-Sep 2026",
            "periods": {"train": TRAIN, "test": TEST, "train_halves": HALVES},
            "controls": f"{RANDOM_REPEATS} random entry times/directions on the "
            "chosen trades, close-of-trigger fills, doubled slippage",
            "versions": {name: version(name) for name in ("numpy", "scipy")},
            "source_sha256": sources,
        },
    )
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)

    table = load_table(candles, read_symbols(constituents))
    eligible = table["eligible"]
    ranks = {
        (score, end): day_ranks(table["day"], table[score][:, end], eligible)
        for score, end in product(SCORES, range(2))
    }
    ranks[("pm", 0)] = day_ranks(table["day"], table["pm_move"], eligible)
    train = period_sessions(table, *TRAIN)
    test = period_sessions(table, *TEST)
    halves = [period_sessions(table, *bounds) for bounds in HALVES]
    records = []
    for variant in variant_grid():
        rows, _, gross = variant_trades(table, ranks, variant)
        days = table["day"][rows]
        slots = variant["slots"]
        records.append(
            {
                "variant": variant,
                "train": summary(days, gross, train, slots, SLIPPAGE),
                "test": summary(days, gross, test, slots, SLIPPAGE),
                "train_halves_inr_per_session": [
                    float(daily_inr(days, gross, half, slots, SLIPPAGE).mean())
                    for half in halves
                ],
            }
        )
    robust = [r for r in records if min(r["train_halves_inr_per_session"]) > 0]
    pool = robust or records
    chosen = max(pool, key=lambda record: record["train"]["net_inr_per_session"])
    variant = chosen["variant"]
    rows, entry, gross = variant_trades(table, ranks, variant)
    days = table["day"][rows]
    close_rows, _, close_gross = variant_trades(table, ranks, variant, fill=1)
    close_days = table["day"][close_rows]
    first = (
        RANGE_ENDS[variant["end"]] + 1 if variant["family"] == "breakout" else AFTERNOON
    )
    last = (
        ENTRY_LIMITS[variant["limit"]] if variant["family"] == "breakout" else AFTERNOON
    )
    robustness = {}
    for name, sessions in (("train", train), ("test", test)):
        inside = np.isin(days, sessions)
        control = random_control(
            table, rows[inside], np.full(int(inside.sum()), first), last, RANDOM_REPEATS
        )
        robustness[name] = {
            "stress_slippage": summary(days, gross, sessions, variant["slots"], STRESS),
            "close_fills": summary(
                close_days, close_gross, sessions, variant["slots"], SLIPPAGE
            ),
            "random_time_side_gross_percent": {
                "mean": 100 * float(control.mean()),
                "p95": 100 * float(np.quantile(control, 0.95)),
                "share_at_or_above_actual": float(
                    np.mean(control >= gross[inside].mean())
                ),
            },
        }
    train_values = np.asarray([r["train"]["net_inr_per_session"] for r in records])
    test_values = np.asarray([r["test"]["net_inr_per_session"] for r in records])
    ordered = sorted(records, key=lambda r: -r["train"]["net_inr_per_session"])
    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "coverage": {
            "symbols": int(len(np.unique(table["symbol"]))),
            "stock_sessions": int(len(table["day"])),
            "eligible_stock_sessions": int(eligible.sum()),
            "train_sessions": len(train),
            "test_sessions": len(test),
        },
        "variants": len(records),
        "robust_variants": len(robust),
        "chosen": chosen,
        "chosen_robustness": robustness,
        "train_test_rank_correlation": float(
            spearmanr(train_values, test_values).statistic
        ),
        "test_positive_share_all": float(np.mean(test_values > 0)),
        "test_positive_share_robust": float(
            np.mean([r["test"]["net_inr_per_session"] > 0 for r in robust])
        )
        if robust
        else None,
        "top_train": ordered[:15],
        "best_test_hindsight": max(
            records, key=lambda r: r["test"]["net_inr_per_session"]
        ),
        "all": [
            {
                "name": r["variant"]["name"],
                "train": r["train"]["net_inr_per_session"],
                "test": r["test"]["net_inr_per_session"],
                "halves": r["train_halves_inr_per_session"],
                "train_gross": r["train"].get("mean_gross_percent"),
                "test_gross": r["test"].get("mean_gross_percent"),
            }
            for r in records
        ],
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
    print(json.dumps({"chosen": report["chosen"]["variant"]["name"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
