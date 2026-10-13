"""Exit-policy research: TP, SL, TH and CatBoost scorer by gross gain per session."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ai_trader.costs import FIXED_CLIP_NOTIONAL, ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore
from scripts.train_tp_only import (
    EXPECTED_COLUMNS,
    file_hash,
    load_phase,
    read_manifest,
    write_json,
)

IST_OFFSET = 19800
EPOCH = date(1970, 1, 1).toordinal()
SESSION_OPEN = 555
FIRST_DECISION = 24
SLOTS = 47
STEP = 5
DEADLINE = 344
NONE = 10_000
LOW_SENTINEL = 10**12
LEVELS = (20, 30, 40, 50, 60, 75, 80, 100)
TP_BPS = (20, 30, 40, 50, 60, 80, 100, None)
SL_BPS = (20, 30, 50, 75, 100, None)
TH_MINUTES = (15, 30, 60, 120, None)
QUANTILES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
WINSOR = 0.02
SEED = 20261011
THREADS = 12
FIT_LIMIT_SECONDS = 900
ROUND_TWO_CONFIGS = 5
BOOTSTRAP_REPEATS = 1000
BLOCK_SESSIONS = 5
PHASES = ("train", "validation", "calibration", "evaluation")
FIRST_DAY, LAST_DAY = "2024-01-01", "2026-09-28"
PERIODS = {
    "dev_fit": ("2024-01-01", "2024-09-30"),
    "dev_stop": ("2024-10-01", "2024-12-31"),
    "select": ("2025-01-01", "2025-12-31"),
    "final_fit": ("2024-01-01", "2025-09-30"),
    "final_stop": ("2025-10-01", "2025-12-31"),
    "confirm_q1_2026": ("2026-01-01", "2026-03-31"),
    "confirm_apr_sep_2026": ("2026-04-01", "2026-09-28"),
}
CONFIRM = ("confirm_q1_2026", "confirm_apr_sep_2026")
HEADER = ["start_time", "open", "high", "low", "close", "volume"]
SOURCES = ("research_exit_policy.py", "train_tp_only.py")


@dataclass
class Universe:
    days: np.ndarray
    symbols: np.ndarray
    features: np.ndarray
    available: np.ndarray
    labels: np.ndarray
    up_minutes: np.ndarray
    down_minutes: np.ndarray
    up_open: np.ndarray
    down_open: np.ndarray
    time_return: np.ndarray
    time_minutes: np.ndarray


@dataclass(frozen=True)
class Calendar:
    day_ids: np.ndarray
    sessions: int
    quarter_of_day: np.ndarray
    month_of_day: np.ndarray
    months: tuple[str, ...]


def epoch_day(label: str) -> int:
    return date.fromisoformat(label).toordinal() - EPOCH


def config_name(tp: int | None, sl: int | None, th: int | None) -> str:
    return f"tp{tp or 'none'}_sl{sl or 'none'}_th{th or '1500'}"


def all_configs() -> list[tuple]:
    return [(tp, sl, th) for tp in TP_BPS for sl in SL_BPS for th in TH_MINUTES]


def round_one_specs() -> dict[str, dict]:
    specs = {"touch_tp20_th60": {"kind": "classifier", "target": None}}
    for th in (30, 60, 120, None):
        specs[f"return_{config_name(None, None, th)}"] = {
            "kind": "regressor",
            "target": (None, None, th),
        }
    for tp in (20, 40, 60, 100):
        for sl in (tp, None):
            for th in (60, None):
                specs[f"return_{config_name(tp, sl, th)}"] = {
                    "kind": "regressor",
                    "target": (tp, sl, th),
                }
    return specs


def read_paths(path: Path, quarantined: set[str]) -> dict[int, np.ndarray]:
    rows: dict[str, list] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != HEADER:
            raise ValueError(f"Unexpected candle header in {path.name}")
        for start, open_, high, low, close, _ in reader:
            day = start[:10]
            if not FIRST_DAY <= day <= LAST_DAY or day in quarantined:
                continue
            if start[16:] != ":00+05:30":
                raise ValueError(f"Unexpected timestamp {start} in {path.name}")
            minute = int(start[11:13]) * 60 + int(start[14:16]) - SESSION_OPEN
            if 0 <= minute <= DEADLINE:
                rows[day].append((minute, open_, high, low, close))
    sessions = {}
    for day, values in rows.items():
        values.sort()
        if [value[0] for value in values] != list(range(DEADLINE + 1)):
            continue
        scaled = np.asarray([value[1:] for value in values], dtype=np.float64) * 100
        prices = np.rint(scaled).astype(np.int64)
        open_, high, low, close = prices.T
        if (
            np.abs(scaled - prices).max() > 1e-6
            or (low <= 0).any()
            or (high < np.maximum(open_, close)).any()
            or (low > np.minimum(open_, close)).any()
        ):
            raise ValueError(f"Invalid or non-paise prices on {day} in {path.name}")
        sessions[epoch_day(day)] = prices
    return sessions


def session_primitives(prices: np.ndarray) -> dict[str, np.ndarray]:
    """First touches of every level, stop-gap opens and time-exit returns."""
    open_, high, low, close = prices.T
    decisions = FIRST_DECISION + STEP * np.arange(SLOTS)
    future = np.arange(DEADLINE + 1)[None, :] > decisions[:, None]
    high_run = (
        np.maximum.accumulate(np.where(future, high[None, :], 0), axis=1) * 10_000
    )
    low_run = (
        np.minimum.accumulate(np.where(future, low[None, :], LOW_SENTINEL), axis=1)
        * 10_000
    )
    base = close[decisions]
    # Rows are monotone, so the count of untouched columns is the first touch.
    up = np.stack(
        [
            (high_run < (base * (10_000 + level))[:, None]).sum(axis=1)
            for level in LEVELS
        ],
        axis=1,
    )
    down = np.stack(
        [
            (low_run > (base * (10_000 - level))[:, None]).sum(axis=1)
            for level in LEVELS
        ],
        axis=1,
    )
    exits = np.stack(
        [
            np.minimum(decisions + (minutes or DEADLINE), DEADLINE)
            for minutes in TH_MINUTES
        ],
        axis=1,
    )
    offsets = decisions[:, None]
    return {
        "up_minutes": np.where(up <= DEADLINE, up - offsets, NONE),
        "down_minutes": np.where(down <= DEADLINE, down - offsets, NONE),
        "up_open": open_[np.minimum(up, DEADLINE)] / base[:, None] - 1,
        "down_open": open_[np.minimum(down, DEADLINE)] / base[:, None] - 1,
        "time_return": close[exits] / base[:, None] - 1,
        "time_minutes": exits - offsets,
    }


def build_universe(
    data: Path, candles: Path, limit: int | None
) -> tuple[Universe, dict]:
    manifest = read_manifest(data)
    parts = [load_phase(data, manifest, phase) for phase in PHASES]
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    local = merged["timestamps"] + IST_OFFSET
    minute = local % 86400 // 60 - SESSION_OPEN
    keep = (minute >= FIRST_DECISION + 1) & (minute <= FIRST_DECISION + 1 + 230)
    names = sorted(set(merged["symbols"][keep].tolist()))[:limit]
    keep &= np.isin(merged["symbols"], names)
    store = CandleStore(candles)
    quality = json.loads(store.verified_path("report.json").read_text("utf-8"))
    sessions = []
    for name in names:
        quarantined = set(
            quality["instruments"][f"NSE:{name}"]["quality"]["quarantined_dates"]
        )
        paths = read_paths(store.verified_path(f"NSE_{name}.csv"), quarantined)
        sessions.extend((day, name, prices) for day, prices in paths.items())
    sessions.sort(key=lambda item: (item[0], item[1]))
    count = len(sessions)
    columns = len(EXPECTED_COLUMNS)
    universe = Universe(
        days=np.asarray([item[0] for item in sessions], dtype=np.int64),
        symbols=np.asarray([item[1] for item in sessions]),
        features=np.full((count, SLOTS, 2, columns), np.nan, dtype=np.float32),
        available=np.zeros((count, SLOTS), dtype=bool),
        labels=np.full((count, SLOTS, 2), -1, dtype=np.int8),
        up_minutes=np.empty((count, SLOTS, len(LEVELS)), dtype=np.int16),
        down_minutes=np.empty((count, SLOTS, len(LEVELS)), dtype=np.int16),
        up_open=np.empty((count, SLOTS, len(LEVELS))),
        down_open=np.empty((count, SLOTS, len(LEVELS))),
        time_return=np.empty((count, SLOTS, len(TH_MINUTES))),
        time_minutes=np.empty((count, SLOTS, len(TH_MINUTES)), dtype=np.int16),
    )
    for index, (_, _, prices) in enumerate(sessions):
        for name, values in session_primitives(prices).items():
            getattr(universe, name)[index] = values
    lookup = {(name, day): index for index, (day, name, _) in enumerate(sessions)}
    selected = np.flatnonzero(keep)
    row_days = (local[selected] // 86400).tolist()
    row_symbols = merged["symbols"][selected].tolist()
    targets = np.fromiter(
        (lookup.get(key, -1) for key in zip(row_symbols, row_days)),
        dtype=np.int64,
        count=len(selected),
    )
    mapped = targets >= 0
    rows, sessions_index = selected[mapped], targets[mapped]
    slots = (minute[rows] - FIRST_DECISION - 1) // STEP
    sides = np.where(merged["features"][rows, -1] == 1, 0, 1)
    if np.any((minute[rows] - FIRST_DECISION - 1) % STEP):
        raise ValueError("Decision outside the five-minute grid")
    universe.features[sessions_index, slots, sides] = merged["features"][rows]
    universe.labels[sessions_index, slots, sides] = merged["labels"][rows]
    filled = np.zeros((count, SLOTS), dtype=np.int8)
    np.add.at(filled, (sessions_index, slots), 1)
    if np.any((filled != 0) & (filled != 2)):
        raise ValueError("Expected LONG/SHORT pairs per decision")
    universe.available = filled == 2
    level = LEVELS.index(20)
    derived = np.stack(
        (
            universe.up_minutes[..., level] <= 60,
            universe.down_minutes[..., level] <= 60,
        ),
        axis=2,
    )
    if not np.array_equal(
        derived[universe.available], universe.labels[universe.available] == 1
    ):
        raise ValueError("Path touches disagree with the frozen 60-minute labels")
    coverage = {
        "symbols": names,
        "stock_sessions_with_paths": count,
        "window_rows": int(keep.sum()),
        "rows_mapped": int(mapped.sum()),
        "rows_without_complete_path_to_1500": int((~mapped).sum()),
        "label_cross_check": "all mapped 60-minute labels equal path-derived touches",
    }
    return universe, coverage


def period_rows(universe: Universe, first: str, last: str) -> slice:
    return slice(
        int(np.searchsorted(universe.days, epoch_day(first), side="left")),
        int(np.searchsorted(universe.days, epoch_day(last), side="right")),
    )


def calendar(universe: Universe, rows: slice) -> Calendar:
    unique, day_ids = np.unique(universe.days[rows], return_inverse=True)
    dates = [date.fromordinal(EPOCH + int(day)) for day in unique]
    _, quarters = np.unique(
        [day.year * 4 + (day.month - 1) // 3 for day in dates], return_inverse=True
    )
    labels = [f"{day.year}-{day.month:02d}" for day in dates]
    months, month_ids = np.unique(labels, return_inverse=True)
    return Calendar(day_ids, len(unique), quarters, month_ids, tuple(months))


def config_outcome(
    universe: Universe, rows: slice, tp: int | None, sl: int | None, th: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-side return, exit minute and exit kind (0 TP, 1 SL, 2 time)."""
    column = TH_MINUTES.index(th)
    limit = universe.time_minutes[rows, :, column].astype(np.int32)
    long_time = universe.time_return[rows, :, column]
    up, down = universe.up_minutes[rows], universe.down_minutes[rows]

    def first(values: np.ndarray, bps: int | None) -> np.ndarray:
        if bps is None:
            return np.full(limit.shape, NONE, dtype=np.int32)
        return values[..., LEVELS.index(bps)].astype(np.int32)

    def stop_fill(opens: np.ndarray, direction: int) -> np.ndarray | float:
        if sl is None:
            return 0.0
        return np.minimum(direction * opens[rows][..., LEVELS.index(sl)], -sl / 1e4)

    outcomes = []
    for tp_at, sl_at, fill, timed in (
        (first(up, tp), first(down, sl), stop_fill(universe.down_open, 1), long_time),
        (first(down, tp), first(up, sl), stop_fill(universe.up_open, -1), -long_time),
    ):
        take = (tp_at <= limit) & (tp_at < sl_at)
        stop = ~take & (sl_at <= limit)
        outcomes.append(
            (
                np.where(take, (tp or 0) / 1e4, np.where(stop, fill, timed)),
                np.where(take, tp_at, np.where(stop, sl_at, limit)),
                np.where(take, 0, np.where(stop, 1, 2)).astype(np.int8),
            )
        )
    return tuple(np.stack(values, axis=2) for values in zip(*outcomes))


def matrices(
    universe: Universe, rows: slice, target: tuple | None
) -> tuple[np.ndarray, np.ndarray]:
    mask = universe.available[rows]
    features = universe.features[rows][mask].reshape(-1, len(EXPECTED_COLUMNS))
    if target is None:
        values = universe.labels[rows].astype(np.int64)
    else:
        values = np.clip(config_outcome(universe, rows, *target)[0], -WINSOR, WINSOR)
    return features, values[mask].reshape(-1)


class TimeLimit:
    def __init__(self, seconds: float) -> None:
        self.deadline = time.perf_counter() + seconds
        self.timed_out = False

    def after_iteration(self, info) -> bool:
        self.timed_out = time.perf_counter() >= self.deadline
        return not self.timed_out


def fit_model(kind: str, train: tuple, stop: tuple):
    from catboost import CatBoostClassifier, CatBoostRegressor

    common = {
        "iterations": 3000,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 20.0,
        "border_count": 254,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "random_strength": 1.0,
        "random_seed": SEED,
        "thread_count": THREADS,
        "early_stopping_rounds": 200,
        "use_best_model": True,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = (
        CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", **common)
        if kind == "classifier"
        else CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", **common)
    )
    limit = TimeLimit(FIT_LIMIT_SECONDS)
    began = time.perf_counter()
    model.fit(*train, eval_set=stop, callbacks=[limit])
    return model, {
        "tree_count": model.tree_count_,
        "fit_seconds": time.perf_counter() - began,
        "timed_out": limit.timed_out,
        "stop_metric": float(next(iter(model.get_best_score()["validation"].values()))),
        "fit_rows": len(train[1]),
        "stop_rows": len(stop[1]),
    }


def predict_scores(model, kind: str, universe: Universe, rows: slice) -> np.ndarray:
    mask = universe.available[rows]
    features = universe.features[rows][mask].reshape(-1, len(EXPECTED_COLUMNS))
    values = (
        model.predict_proba(features)[:, 1]
        if kind == "classifier"
        else model.predict(features)
    )
    scores = np.full(mask.shape + (2,), -np.inf)
    scores[mask] = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    return scores


def side_choice(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side = scores.argmax(axis=2)
    return side, np.take_along_axis(scores, side[..., None], axis=2)[..., 0]


def cutoffs(kind: str, best: np.ndarray, available: np.ndarray) -> dict[str, float]:
    values = best[available]
    rules = {
        f"top{round(100 * (1 - level))}": float(np.quantile(values, level))
        if level
        else -np.inf
        for level in QUANTILES
    }
    if kind == "regressor":
        rules["positive"] = 0.0
    return rules


def simulate(eligible: np.ndarray, exits: np.ndarray) -> np.ndarray:
    """Greedy entries with at most one open position per stock-session."""
    count = len(eligible)
    jump = np.minimum(np.arange(SLOTS) + (exits + STEP - 1) // STEP, SLOTS)
    first = np.full((count, SLOTS + 1), SLOTS, dtype=np.int64)
    for slot in range(SLOTS - 1, -1, -1):
        first[:, slot] = np.where(eligible[:, slot], slot, first[:, slot + 1])
    taken = np.zeros((count, SLOTS), dtype=bool)
    rows = np.arange(count)
    pointer = first[:, 0].copy()
    active = pointer < SLOTS
    while active.any():
        index, at = rows[active], pointer[active]
        taken[index, at] = True
        pointer[index] = first[index, jump[index, at]]
        active = pointer < SLOTS
    return taken


def summarize(
    taken: np.ndarray,
    returns: np.ndarray,
    exits: np.ndarray,
    kinds: np.ndarray,
    days: Calendar,
    charges: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    trades_row = taken.sum(axis=1)
    gross_row = np.where(taken, returns, 0.0).sum(axis=1)
    daily_gross = np.bincount(days.day_ids, gross_row, minlength=days.sessions)
    daily_trades = np.bincount(days.day_ids, trades_row, minlength=days.sessions)
    daily_net = daily_gross - charges * daily_trades
    trades = int(trades_row.sum())
    counts = np.bincount(days.quarter_of_day)

    def quarterly(values: np.ndarray) -> list[float]:
        return (100 * np.bincount(days.quarter_of_day, values) / counts).tolist()

    def share(kind: int) -> float | None:
        return 100 * int((taken & (kinds == kind)).sum()) / trades if trades else None

    summary = {
        "sessions": days.sessions,
        "mean_daily_gross_percent": 100 * float(daily_gross.mean()),
        "mean_daily_net_percent": 100 * float(daily_net.mean()),
        "quarter_gross_percent": quarterly(daily_gross),
        "quarter_net_percent": quarterly(daily_net),
        "trades_per_session": trades / days.sessions,
        "mean_trade_gross_percent": 100 * float(gross_row.sum()) / trades
        if trades
        else None,
        "tp_exit_percent": share(0),
        "sl_exit_percent": share(1),
        "time_exit_percent": share(2),
        "mean_hold_minutes": float(exits[taken].mean()) if trades else None,
        "positive_session_percent": 100 * float((daily_gross > 0).mean()),
    }
    return summary, daily_gross, daily_net


def block_interval(daily: np.ndarray) -> list[float]:
    sessions = len(daily)
    block = min(BLOCK_SESSIONS, sessions)
    generator = np.random.default_rng(SEED)
    starts = generator.integers(
        sessions, size=(BOOTSTRAP_REPEATS, -(-sessions // block))
    )
    sampled = ((starts[:, :, None] + np.arange(block)) % sessions).reshape(
        BOOTSTRAP_REPEATS, -1
    )[:, :sessions]
    return (100 * np.quantile(daily[sampled].mean(axis=1), (0.025, 0.975))).tolist()


def concurrency(taken: np.ndarray, exits: np.ndarray, days: Calendar) -> dict:
    rows, slots = np.nonzero(taken)
    entry = FIRST_DECISION + 1 + STEP * slots
    grid = np.zeros((days.sessions, DEADLINE + 2), dtype=np.int64)
    np.add.at(grid, (days.day_ids[rows], entry), 1)
    np.add.at(grid, (days.day_ids[rows], entry + exits[rows, slots]), -1)
    held = np.cumsum(grid, axis=1)
    return {
        "max_open_positions": int(held.max()),
        "mean_daily_max_open_positions": float(held.max(axis=1).mean()),
        "mean_open_positions_0940_1500": float(
            held[:, FIRST_DECISION + 1 : DEADLINE + 1].mean()
        ),
    }


def chosen(values: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, side[..., None], axis=2)[..., 0]


def evaluate_policy(
    universe: Universe,
    rows: slice,
    scores: np.ndarray,
    config: tuple,
    threshold: float,
    charges: float,
    detail: bool = False,
) -> dict:
    side, best = side_choice(scores)
    returns, exits, kinds = (
        chosen(values, side) for values in config_outcome(universe, rows, *config)
    )
    eligible = universe.available[rows] & (best >= threshold)
    taken = simulate(eligible, exits)
    days = calendar(universe, rows)
    summary, daily_gross, daily_net = summarize(
        taken, returns, exits, kinds, days, charges
    )
    if detail:
        month_counts = np.bincount(days.month_of_day)
        summary.update(
            gross_95pct_session_block_interval=block_interval(daily_gross),
            net_95pct_session_block_interval=block_interval(daily_net),
            monthly_gross_percent=dict(
                zip(
                    days.months,
                    (
                        100 * np.bincount(days.month_of_day, daily_gross) / month_counts
                    ).tolist(),
                )
            ),
            long_trade_percent=100
            * int((taken & (side == 0)).sum())
            / max(int(taken.sum()), 1),
            selected_candle_time_percent=100
            * float(eligible.sum() / max(universe.available[rows].sum(), 1)),
            **concurrency(taken, exits, days),
        )
    return summary


def train_scorer(
    universe: Universe, fit: slice, stop: slice, spec: dict, path: Path
) -> dict:
    model, info = fit_model(
        spec["kind"],
        matrices(universe, fit, spec["target"]),
        matrices(universe, stop, spec["target"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    _, best = side_choice(predict_scores(model, spec["kind"], universe, stop))
    return {
        "model": model,
        "spec": spec,
        "info": {**info, "weights_sha256": file_hash(path)},
        "rules": cutoffs(spec["kind"], best, universe.available[stop]),
    }


def screen(
    universe: Universe,
    rows: slice,
    scorers: dict[str, dict],
    configs: list[tuple],
    charges: float,
) -> list[dict]:
    days = calendar(universe, rows)
    available = universe.available[rows]
    choices = {
        name: side_choice(
            predict_scores(item["model"], item["spec"]["kind"], universe, rows)
        )
        for name, item in scorers.items()
    }
    records = []
    for config in configs:
        outcome = config_outcome(universe, rows, *config)
        for name, (side, best) in choices.items():
            returns, exits, kinds = (chosen(values, side) for values in outcome)
            for rule, threshold in scorers[name]["rules"].items():
                taken = simulate(available & (best >= threshold), exits)
                summary, _, _ = summarize(taken, returns, exits, kinds, days, charges)
                records.append(
                    {
                        "model": name,
                        "tp": config[0],
                        "sl": config[1],
                        "th": config[2],
                        "rule": rule,
                        **summary,
                    }
                )
    return records


def robust(record: dict, metric: str) -> bool:
    return all(value > 0 for value in record[f"quarter_{metric}_percent"])


def choose(records: list[dict], metric: str) -> tuple[dict, bool]:
    pool = [record for record in records if robust(record, metric)]
    best = max(
        pool or records,
        key=lambda record: (
            record[f"mean_daily_{metric}_percent"],
            -record["trades_per_session"],
        ),
    )
    return best, bool(pool)


def protocol_document(sources: dict, charges: float, limit: int | None) -> dict:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "objective": (
            "maximize mean gross gain per session: sum of per-trade percentage "
            "returns over all trades entered that session; costs ignored for "
            "selection and reported separately"
        ),
        "symbol_limit_for_smoke_test_only": limit,
        "periods": PERIODS,
        "decisions": "five-minute grid 09:40-13:30 IST, entry at decision close",
        "trading_rule": (
            "per stock and decision, take the higher-scored side if its score >= "
            "threshold and no position in that stock is open; re-entry from the "
            "first decision at or after the exit"
        ),
        "exits": {
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "th_minutes": TH_MINUTES,
            "tp_fill": "exactly entry*(1+/-TP)",
            "sl_fill": "stop price, or the trigger candle open when it gaps through",
            "same_candle": "stop first",
            "time": "close at entry+TH, never after the 14:59-15:00 candle",
            "none": "None means no TP, no SL or a 15:00 time exit",
        },
        "complete_paths": "every minute 09:15 through the 14:59-15:00 candle",
        "scorers": {
            "round_one": {
                name: {"kind": spec["kind"], "target": spec["target"]}
                for name, spec in round_one_specs().items()
            },
            "round_two": (
                f"return regressors for the top {ROUND_TWO_CONFIGS} robust round-one "
                "configs lacking a dedicated regressor"
            ),
            "inputs": list(EXPECTED_COLUMNS),
            "touch_target": "frozen +/-0.20% touch within 60 minutes",
            "return_target": f"exit-rule gross return clipped to +/-{WINSOR:.0%}",
            "catboost": (
                "depth 4, learning rate 0.05, L2 20, Bernoulli 0.8, 254 borders, "
                f"3000 trees, early stopping 200, seed {SEED}"
            ),
        },
        "thresholds": (
            "quantiles of higher-side scores on the early-stopping slice "
            f"{QUANTILES}; regressors also use predicted gain > 0"
        ),
        "selection": (
            "highest 2025 mean daily gross among records positive in all four "
            "2025 quarters; ties prefer fewer trades. The same rule on net gain "
            "after charges gives a second, cost-aware policy"
        ),
        "final": (
            "refit the selected scorer on final_fit with early stopping on "
            "final_stop, recompute its threshold rule there, lock, then confirm "
            "once on both 2026 periods"
        ),
        "charges_round_trip_fraction": charges,
        "versions": {name: version(name) for name in ("catboost", "numpy")},
        "source_sha256": sources,
    }


def run_study(
    data: Path,
    candles: Path,
    previous: Path,
    output: Path,
    limit: int | None = None,
) -> dict:
    from catboost import CatBoostClassifier

    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    charges = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(FIXED_CLIP_NOTIONAL))
    here = Path(__file__).parent
    sources = {name: file_hash(here / name) for name in SOURCES}
    previous_lock = json.loads((previous / "selection_lock.json").read_text("utf-8"))
    if file_hash(previous / "best.cbm") != previous_lock["weights_sha256"]:
        raise ValueError("Previous scanner model changed after its lock")
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol_document(sources, charges, limit))
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)

    universe, coverage = build_universe(data, candles, limit)
    rows = {name: period_rows(universe, *bounds) for name, bounds in PERIODS.items()}
    configs = all_configs()
    scorers = {
        name: train_scorer(
            universe,
            rows["dev_fit"],
            rows["dev_stop"],
            spec,
            output / "dev_models" / f"{name}.cbm",
        )
        for name, spec in round_one_specs().items()
    }
    records = screen(universe, rows["select"], scorers, configs, charges)
    dedicated = {item["spec"]["target"] for item in scorers.values()}
    ranked = sorted(
        [record for record in records if robust(record, "gross")] or records,
        key=lambda record: record["mean_daily_gross_percent"],
        reverse=True,
    )
    extra = []
    for record in ranked:
        config = (record["tp"], record["sl"], record["th"])
        if config not in dedicated and config not in extra:
            extra.append(config)
        if len(extra) == ROUND_TWO_CONFIGS:
            break
    round_two = {
        f"return_{config_name(*config)}": train_scorer(
            universe,
            rows["dev_fit"],
            rows["dev_stop"],
            {"kind": "regressor", "target": config},
            output / "dev_models" / f"return_{config_name(*config)}.cbm",
        )
        for config in extra
    }
    records += screen(universe, rows["select"], round_two, configs, charges)
    scorers.update(round_two)
    with (output / "screen.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in records[0] if not key.startswith("quarter_")]
        writer = csv.DictWriter(handle, fieldnames=[*fields, "robust_gross"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record[key] for key in fields},
                    "robust_gross": robust(record, "gross"),
                }
            )

    finals = {}
    for metric in ("gross", "net"):
        record, is_robust = choose(records, metric)
        if record[f"mean_daily_{metric}_percent"] <= 0:
            finals[metric] = {"selected": None, "development": record}
            continue
        spec = scorers[record["model"]]["spec"]
        final = train_scorer(
            universe,
            rows["final_fit"],
            rows["final_stop"],
            spec,
            output / "final" / metric / "best.cbm",
        )
        lock = {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": file_hash(output / "protocol.json"),
            "development_record": record,
            "robust_in_all_2025_quarters": is_robust,
            "config": {
                "tp_bps": record["tp"],
                "sl_bps": record["sl"],
                "th": record["th"],
            },
            "scorer": spec,
            "rule": record["rule"],
            "threshold": None
            if np.isinf(final["rules"][record["rule"]])
            else final["rules"][record["rule"]],
            "final_model": final["info"],
            "confirmation_opened": False,
        }
        write_json(output / "final" / metric / "selection_lock.json", lock)
        finals[metric] = {**final, "record": record, "lock": lock}

    oct10 = CatBoostClassifier()
    oct10.load_model(str(previous / "best.cbm"))
    results = {}
    for metric, final in finals.items():
        if final.get("selected", True) is None:
            results[metric] = final
            continue
        record = final["record"]
        config = (record["tp"], record["sl"], record["th"])
        threshold = final["rules"][record["rule"]]
        kind = final["spec"]["kind"]
        stop = rows["final_stop"]
        _, stop_best = side_choice(predict_scores(final["model"], kind, universe, stop))
        selectivity = float((stop_best[universe.available[stop]] >= threshold).mean())
        confirmations = {}
        for period in CONFIRM:
            scores = predict_scores(final["model"], kind, universe, rows[period])
            confirmations[period] = {
                "selected_policy": evaluate_policy(
                    universe, rows[period], scores, config, threshold, charges, True
                ),
                "same_model_all_calls": evaluate_policy(
                    universe, rows[period], scores, config, -np.inf, charges
                ),
            }
        q1, later = rows["confirm_q1_2026"], rows["confirm_apr_sep_2026"]
        _, q1_best = side_choice(predict_scores(oct10, "classifier", universe, q1))
        oct10_threshold = float(
            np.quantile(q1_best[universe.available[q1]], 1 - selectivity)
        )
        confirmations["confirm_apr_sep_2026"]["oct10_model_same_selectivity"] = (
            evaluate_policy(
                universe,
                later,
                predict_scores(oct10, "classifier", universe, later),
                config,
                oct10_threshold,
                charges,
            )
        )
        results[metric] = {
            "development_record": record,
            "lock": final["lock"],
            "selectivity_on_final_stop": selectivity,
            "confirmations": confirmations,
        }

    def top(metric: str, count: int = 15) -> list[dict]:
        pool = [record for record in records if robust(record, metric)]
        return sorted(
            pool,
            key=lambda record: record[f"mean_daily_{metric}_percent"],
            reverse=True,
        )[:count]

    def best_by(key: str) -> dict:
        groups: dict = {}
        for record in records:
            value = record[key]
            current = groups.get(value)
            if current is None or (
                record["mean_daily_gross_percent"] > current["mean_daily_gross_percent"]
            ):
                groups[value] = record
        return {str(value): record for value, record in groups.items()}

    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "coverage": coverage,
        "period_stock_sessions": {
            name: value.stop - value.start for name, value in rows.items()
        },
        "scorers": {
            name: {
                "spec": item["spec"],
                "rules": {
                    rule: None if np.isinf(value) else value
                    for rule, value in item["rules"].items()
                },
                **item["info"],
            }
            for name, item in scorers.items()
        },
        "round_two_configs": [config_name(*config) for config in extra],
        "screen_records": len(records),
        "top_robust_gross": top("gross"),
        "top_robust_net": top("net"),
        "best_gross_by_model": best_by("model"),
        "best_gross_by_tp": best_by("tp"),
        "best_gross_by_sl": best_by("sl"),
        "best_gross_by_th": best_by("th"),
        "original_contract_best": max(
            (
                record
                for record in records
                if (record["tp"], record["sl"], record["th"]) == (20, None, None)
            ),
            key=lambda record: record["mean_daily_gross_percent"],
        ),
        "results": results,
        "charges_round_trip_fraction": charges,
        "limitations": [
            "Touch is not a fill; TP fills exactly, stops at stop or gapped open",
            "Gross per-trade percentages summed per session; no capital limit",
            "Decision grid every five minutes; one open position per stock",
            "2025 selection examined thousands of combinations; trust the 2026 "
            "confirmations, not the 2025 maxima",
            "April-September 2026 was examined by earlier research",
        ],
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-symbols", type=int)
    args = parser.parse_args()
    report = run_study(
        args.data, args.candles, args.previous, args.output, args.limit_symbols
    )
    print(json.dumps({"results": report["results"]}, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
