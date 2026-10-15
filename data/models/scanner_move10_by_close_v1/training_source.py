"""Scanner research: P(the price moves X% either way before the session ends).

User goal (2026-10-14): surface stocks with at least an 80% probability of touching
+0.8% or -0.8% of the buy price in the same session (the defaults, ``--move-bps 80
--threshold 0.80``); the follow-up asked for 1% at 0.90 (``--move-bps 100
--threshold 0.90``). Research only, on the NSE-basis corrected Kite dataset (213
F&O stocks, January 2020 to September 2026), with the live scanner's checks: every
5 minutes, decision closes 09:40 to 14:55, buy = open of the next minute.

Label: from the buy minute to the session's last continuous minute (15:29; 15:14
for F&O stocks from 2026-08-03, NSE's closing auction) some minute's high reaches
buy x (1 + move) or its low reaches buy x (1 - move). A session missing any minute
has no label. Also kept for analysis: which level was touched, which came first (a
minute reaching both is resolved by its open, else unknown) and the minutes to the
first touch.

Inputs: the production scanner's 31 inputs (``research_delivery_pipeline`` features
after the same pruning), and an extended set adding five two-sided movement inputs
(distance above the day's low, volatility since the open, and the move over
three volatility estimates scaled to the minutes left), pruned the same way.

Choices use 2020-2024 only: fit 2020-2023 (3 million sampled rows), early stopping
2024 H1, candidates ranked by raw log loss on 2024 H2. Stage 1 fits both input sets
with the production settings; the extended set is kept only if it is 0.0005
better. Stage 2 fits four more CatBoost settings on the kept set. Calibration (raw,
Platt or isotonic) is fitted on 2024 H2 and chosen on 2024 H1. The choice is
written to ``selection.json`` before the test (January 2025 to September 2026) is
scored. The chosen recipe is then refitted on newer data (fit to 2025-06, early
stopping 2025 H2, calibration 2026 H1) and scored on July-September 2026, as the
production scanner was.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time as clock
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai_trader.history import CandleStore
from scripts.e2e_research import apply_exclusions
from scripts.research_delivery_pipeline import (
    FEATURES,
    LOOKBACK,
    MARKET,
    OPTIONAL,
    PERCENT,
    SESSION_MINUTES,
    STOCK_FEATURES,
    as_date,
    daily_columns,
    logit,
    market_context,
    period_mask,
    read_path,
    session_features,
    wilder_rsi,
)
from scripts.research_fno_scanner import OPEN_MINUTE, SEED, read_daily, read_sessions
from scripts.research_scanner_goals import (
    AUCTION_FROM,
    AUCTION_MINUTES,
    DECISIONS,
    SLOTS,
    log_loss_of,
    session_minutes,
)
from scripts.train_tp_only import file_hash, write_json

MOVE_BPS = 80
THRESHOLD = 0.80
THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
BANDS = 10
BOOTSTRAP = 1000
RAW_EXTRA = ("dist_low", "rv_open")
EXTRA = (*RAW_EXTRA, "z_hist", "z_open", "z_atr")
NAMES = FEATURES + EXTRA
OUTCOMES = ("touched", "up", "down", "up_first", "minutes")
TOUCHED, UP, DOWN, UP_FIRST, MINUTES = range(len(OUTCOMES))
FIT = ("2020-01-01", "2023-12-31")
EARLY_STOP = ("2024-01-01", "2024-06-30")
SELECT = ("2024-07-01", "2024-12-31")  # ranks the candidates; calibration fitted here
TEST = ("2025-01-01", "2026-09-30")
RESEARCH = {"fit": FIT, "early_stop": EARLY_STOP, "calibration": SELECT}
PRODUCTION = {
    "fit": ("2020-01-01", "2025-06-30"),
    "early_stop": ("2025-07-01", "2025-12-31"),
    "calibration": ("2026-01-01", "2026-06-30"),
    "holdout": ("2026-07-01", "2026-09-30"),
}
PERIODS = (
    ("test_2025H1", "2025-01-01", "2025-06-30"),
    ("test_2025H2", "2025-07-01", "2025-12-31"),
    ("test_2026", "2026-01-01", "2026-09-30"),
    ("test_closing_auction", "2026-08-03", "2026-09-30"),
    ("test_all", *TEST),
)
HOLDOUT_PERIODS = (
    ("holdout_2026Q3", *PRODUCTION["holdout"]),
    ("holdout_closing_auction", "2026-08-03", "2026-09-30"),
)
BASELINE = "d6_lr0.08_l2_10"  # the production scanner's settings
SETTINGS = {
    BASELINE: {"depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 10},
    "d4_lr0.08_l2_10": {"depth": 4, "learning_rate": 0.08, "l2_leaf_reg": 10},
    "d8_lr0.05_l2_10": {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10},
    "d6_lr0.04_l2_10": {"depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 10},
    "d6_lr0.08_l2_30": {"depth": 6, "learning_rate": 0.08, "l2_leaf_reg": 30},
}
EXTENDED_MARGIN = 0.0005  # log loss the extended inputs must gain to be kept
FIT_ROWS, STOP_ROWS = 3_000_000, 600_000
TIMES = (("09:40-10:55", 0, 16), ("11:00-12:55", 16, 40), ("13:00-14:55", 40, 64))
PRODUCTION_BUNDLE = Path("data/models/scanner_up1_by_close_v1/bundle.json")


def percent(move_bps: int) -> str:
    return f"{move_bps / 100:g}%"


def label_text(move_bps: int) -> str:
    return (
        "some minute from the buy (open of the minute after the decision close) to "
        "the session's last continuous minute (15:29; 15:14 for F&O stocks from "
        f"2026-08-03) has high >= buy x {1 + move_bps / 10_000:g} or low <= buy x "
        f"{1 - move_bps / 10_000:g}; a session missing any minute has no label"
    )


def version(move_bps: int) -> str:
    return f"scanner-move{move_bps}bps-by-close-v1-2026-10-14"


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def move_labels(
    ohlc: np.ndarray,
    days: np.ndarray,
    minutes: np.ndarray,
    day: int,
    counts: Counter,
    move_bps: int = MOVE_BPS,
) -> np.ndarray:
    """Outcome of a buy at every decision of ``day``: [SLOTS, OUTCOMES], NaN when
    unknown. ``ohlc`` holds the stock's minutes in paise, sorted by day and minute."""
    out = np.full((SLOTS, len(OUTCOMES)), np.nan)
    first = int(np.searchsorted(days, day))
    end = int(np.searchsorted(days, day, side="right"))
    length = session_minutes(day)
    if not np.array_equal(minutes[first:end], np.arange(length)):
        counts["sessions_incomplete"] += 1
        return out
    block = ohlc[first:end] * 10_000
    buy = DECISIONS + 1
    entry = ohlc[first + buy, 0]
    up = entry * (10_000 + move_bps)
    down = entry * (10_000 - move_bps)
    after = np.arange(length)[None, :] >= buy[:, None]
    hit_up = after & (block[None, :, 1] >= up[:, None])
    hit_down = after & (block[None, :, 2] <= down[:, None])
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), length)
    first_down = np.where(hit_down.any(axis=1), hit_down.argmax(axis=1), length)
    touch = np.minimum(first_up, first_down)
    touched = touch < length
    up_first = (first_up < first_down).astype(float)
    same = touched & (first_up == first_down)
    if same.any():
        opened = block[touch[same], 0]
        up_first[same] = np.where(
            opened >= up[same], 1.0, np.where(opened <= down[same], 0.0, np.nan)
        )
        counts["same_minute_order_unknown"] += int(np.isnan(up_first[same]).sum())
    up_first[~touched] = np.nan
    out[:, TOUCHED] = touched
    out[:, UP] = first_up < length
    out[:, DOWN] = first_down < length
    out[:, UP_FIRST] = up_first
    out[:, MINUTES] = np.where(touched, touch - buy + 1, np.nan)
    counts["sessions_labelled"] += 1
    return out


def extra_columns(prices: np.ndarray, decisions: np.ndarray = DECISIONS) -> np.ndarray:
    """[slots, 2] at each decision close from that session's minutes up to it:
    distance above the day's low so far, and realised volatility since the open on
    the 20-minute scale of ``rv20`` and ``hist_rv``."""
    low, close = prices[:, 2], prices[:, 3]
    squares = np.concatenate(([0.0], np.cumsum(np.diff(np.log(close)) ** 2)))
    bottom = np.minimum.accumulate(low)
    d = decisions
    return np.column_stack((close[d] / bottom[d] - 1, np.sqrt(20 * squares[d] / d)))


def stock_rows(task: tuple[str, Path, np.ndarray, int]) -> dict:
    symbol, candles, calendar, move_bps = task
    store = CandleStore(candles)
    path = store.verified_path(f"NSE_{symbol}.csv")
    sessions = read_sessions(path)
    path_days, minutes, ohlc = read_path(path)
    daily_days, daily = read_daily(store.verified_path(f"daily/NSE_{symbol}.csv"))
    rsi = wilder_rsi(daily[:, 3])
    days = sorted(sessions)
    width = len(STOCK_FEATURES) + len(RAW_EXTRA)
    features = np.full((len(days), SLOTS, width), np.nan, np.float32)
    outcomes = np.full((len(days), SLOTS, len(OUTCOMES)), np.nan, np.float32)
    counts: Counter = Counter()
    day_rv: list[float] = []
    volumes: list[np.ndarray] = []
    for row, day in enumerate(days):
        part = session_features(sessions[day], DECISIONS)
        position = int(np.searchsorted(daily_days, day))
        market = int(np.searchsorted(calendar, day))
        if (
            len(day_rv) >= LOOKBACK
            and position > LOOKBACK
            and market > 0
            and daily_days[position - 1] == calendar[market - 1]
        ):
            features[row, :, : len(STOCK_FEATURES)] = daily_columns(
                daily, position, rsi, day_rv, volumes, part, SLOTS
            )
            features[row, :, len(STOCK_FEATURES) :] = extra_columns(sessions[day])
        outcomes[row] = move_labels(ohlc, path_days, minutes, day, counts, move_bps)
        day_rv.append(part["day_rv"])
        volumes.append(part["cumvol"])
    return {
        "days": np.asarray(days, dtype=np.int64),
        "features": features,
        "outcomes": outcomes,
        "counts": counts,
    }


def movement_scales(features: np.ndarray, move: float = MOVE_BPS / 10_000) -> None:
    """The move over three volatility estimates scaled to the minutes left
    (the ``slot`` input), in place. A random walk's chance of the move falls as
    the ratio rises; trees see that through one input instead of an interaction."""
    col = {name: i for i, name in enumerate(NAMES)}
    left = features[..., col["slot"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, volatility, unit in (
            ("z_hist", "hist_rv", 20),
            ("z_open", "rv_open", 20),
            ("z_atr", "atr20", SESSION_MINUTES),
        ):
            scale = features[..., col[volatility]] * np.sqrt(left / unit)
            features[..., col[name]] = move / np.maximum(scale, 1e-9)


def assemble(
    candles: Path, workers: int, limit: int | None = None, move_bps: int = MOVE_BPS
) -> dict:
    store = CandleStore(candles)
    universe = json.loads(store.verified_path("universe.json").read_text("utf-8"))
    stocks = [
        entry["symbol"]
        for entry in universe["instruments"]
        if entry["role"] == "equity_target"
    ][:limit]
    calendar, days, market = market_context(candles, DECISIONS)
    shape = (len(days), SLOTS, len(stocks))
    features = np.full((*shape, len(NAMES)), np.nan, np.float32)
    outcomes = np.full((*shape, len(OUTCOMES)), np.nan, np.float32)
    session = np.zeros(shape, bool)
    counts: Counter = Counter()
    index = {int(day): row for row, day in enumerate(days)}
    stock_width, extra_at = len(STOCK_FEATURES), len(FEATURES)
    tasks = [(symbol, candles, calendar, move_bps) for symbol in stocks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for column, result in enumerate(pool.map(stock_rows, tasks)):
            rows = np.asarray(
                [index.get(int(day), -1) for day in result["days"]], dtype=np.int64
            )
            keep = rows >= 0
            values = result["features"][keep]
            features[rows[keep], :, column, :stock_width] = values[..., :stock_width]
            features[rows[keep], :, column, extra_at : extra_at + len(RAW_EXTRA)] = (
                values[..., stock_width:]
            )
            outcomes[rows[keep], :, column] = result["outcomes"][keep]
            session[rows[keep], :, column] = True
            counts.update(result["counts"])
    col = {name: i for i, name in enumerate(NAMES)}
    for i, name in enumerate(MARKET):
        features[..., col[name]] = market[:, :, None, i]
    left = (
        np.array([session_minutes(int(day)) for day in days])[:, None] - 1 - DECISIONS
    )
    features[..., col["slot"]] = np.where(
        np.isfinite(features[..., col["slot"]]), left[:, :, None], np.nan
    )
    day_return = (1 + features[..., col["gap"]]) * (1 + features[..., col["ret_open"]])
    features[..., col["rel_day"]] = day_return - 1 - features[..., col["nifty_day"]]
    features[..., col["rel_20d"]] = (
        features[..., col["ret20d"]] - features[..., col["nifty_ret20d"]]
    )
    movement_scales(features, move_bps / 10_000)
    for name in FEATURES:
        if name not in OPTIONAL:
            session &= np.isfinite(features[..., col[name]])
    return {
        "days": days,
        "stocks": stocks,
        "features": features,
        "outcomes": outcomes,
        "available": session,
        "label_counts": counts,
    }


def prune_inputs(
    data: dict, names: tuple[str, ...], limit: float = 0.9
) -> tuple[list[str], list[dict]]:
    """``research_delivery_pipeline.prune`` over ``names``: keep inputs in declared
    order unless one already kept ranks the same way (|Spearman| > limit) on
    fit-period rows. Over ``FEATURES`` it reproduces that function's choice."""
    columns = [NAMES.index(name) for name in names]
    days = np.broadcast_to(data["days"][:, None, None], data["available"].shape)
    mask = data["available"] & period_mask(days, *FIT)
    for column in columns:
        mask &= np.isfinite(data["features"][..., column])
    rows = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(rows, size=min(300_000, len(rows)), replace=False))
    values = data["features"].reshape(-1, len(NAMES))[rows][:, columns]
    ranks = values.argsort(axis=0).argsort(axis=0).astype(np.float64)
    rho = np.corrcoef(ranks, rowvar=False)
    kept: list[int] = []
    dropped: list[dict] = []
    for i, name in enumerate(names):
        clash = [k for k in kept if abs(rho[i, k]) > limit]
        if clash:
            dropped.append(
                {
                    "feature": name,
                    "duplicates": names[clash[0]],
                    "spearman": round(float(rho[i, clash[0]]), 3),
                }
            )
        else:
            kept.append(i)
    return [names[i] for i in kept], dropped


def sample_rows(
    known: np.ndarray, day: np.ndarray, period: tuple[str, str], size: int, rng
) -> np.ndarray:
    rows = np.flatnonzero(known & period_mask(day, *period))
    return np.sort(rng.choice(rows, size=min(size, len(rows)), replace=False))


def fit_candidate(
    data: dict,
    kept: list[str],
    setting: str,
    periods: dict,
    threads: int,
    iterations: int,
):
    """CatBoost on sampled fit rows with early stopping; the same rows for every
    candidate of one period plan."""
    from catboost import CatBoostClassifier

    columns = [NAMES.index(name) for name in kept]
    flat = data["features"].reshape(-1, len(NAMES))
    y = data["outcomes"][..., TOUCHED].reshape(-1)
    known = data["available"].reshape(-1) & np.isfinite(y)
    rng = np.random.default_rng(SEED)
    fit_rows = sample_rows(known, data["day_flat"], periods["fit"], FIT_ROWS, rng)
    stop_rows = sample_rows(
        known, data["day_flat"], periods["early_stop"], STOP_ROWS, rng
    )
    model = CatBoostClassifier(
        **SETTINGS[setting],
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="Logloss",
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        random_seed=SEED,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    started = clock.monotonic()
    model.fit(
        flat[fit_rows][:, columns],
        y[fit_rows].astype(np.int8),
        eval_set=(flat[stop_rows][:, columns], y[stop_rows].astype(np.int8)),
    )
    info = {
        "setting": setting,
        "inputs_count": len(kept),
        "fit_rows": len(fit_rows),
        "early_stop_rows": len(stop_rows),
        "trees": int(model.tree_count_),
        "fit_seconds": round(clock.monotonic() - started, 1),
        "early_stop_log_loss": round(
            float(model.get_best_score()["validation"]["Logloss"]), 6
        ),
    }
    return model, info


def predict(model, data: dict, kept: list[str], rows: np.ndarray) -> np.ndarray:
    columns = [NAMES.index(name) for name in kept]
    flat = data["features"].reshape(-1, len(NAMES))
    out = np.empty(len(rows), np.float32)
    for start in range(0, len(rows), 2_000_000):
        chunk = rows[start : start + 2_000_000]
        out[start : start + len(chunk)] = model.predict_proba(flat[chunk][:, columns])[
            :, 1
        ]
    return out


def calibrate(raw: np.ndarray, data: dict, periods: dict) -> tuple[np.ndarray, dict]:
    """Raw, Platt or isotonic: fitted on the calibration period, chosen by log loss
    on the early-stopping period. Returns probabilities on every row with a raw
    estimate and the mapping's description."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y = data["outcomes"][..., TOUCHED].reshape(-1)
    flat_raw = raw.reshape(-1)
    known = np.isfinite(flat_raw) & np.isfinite(y)
    fitted = np.flatnonzero(
        known & period_mask(data["day_flat"], *periods["calibration"])
    )
    check = np.flatnonzero(
        known & period_mask(data["day_flat"], *periods["early_stop"])
    )
    platt = LogisticRegression(C=1e6).fit(logit(flat_raw[fitted])[:, None], y[fitted])
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
        flat_raw[fitted], y[fitted]
    )
    maps = {
        "raw": lambda p: p,
        "platt": lambda p: platt.predict_proba(logit(p)[:, None])[:, 1],
        "isotonic": isotonic.predict,
    }
    scores = {
        name: round(log_loss_of(y[check], f(flat_raw[check])), 6)
        for name, f in maps.items()
    }
    choice = min(scores, key=scores.get)
    rows = np.flatnonzero(np.isfinite(flat_raw))
    calibrated = np.full(flat_raw.shape, np.nan, np.float32)
    for start in range(0, len(rows), 4_000_000):
        chunk = rows[start : start + 4_000_000]
        calibrated[chunk] = maps[choice](flat_raw[chunk])
    info = {
        "fitted_on": list(periods["calibration"]),
        "chosen_on": list(periods["early_stop"]),
        "rows": len(fitted),
        "log_loss_choice_period": scores,
        "chosen": choice,
        "platt_slope": float(platt.coef_[0, 0]),
        "platt_intercept": float(platt.intercept_[0]),
        "isotonic_x": isotonic.X_thresholds_.tolist() if choice == "isotonic" else None,
        "isotonic_y": isotonic.y_thresholds_.tolist() if choice == "isotonic" else None,
    }
    return calibrated.reshape(raw.shape), info


def climatology(data: dict, period: tuple[str, str]) -> np.ndarray:
    """Base rate at each decision time over a period: the time-of-day baseline."""
    y = data["outcomes"][..., TOUCHED]
    inside = data["available"] & period_mask(data["days"], *period)[:, None, None]
    known = inside & np.isfinite(y)
    return np.where(known, y, 0).sum(axis=(0, 2)) / known.sum(axis=(0, 2))


def day_bootstrap(
    counts: np.ndarray, hits: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """95% range of hits / counts per column when whole sessions (rows) are drawn
    with replacement; outcomes on one day move together, so days are the unit."""
    low = np.full(counts.shape[1], np.nan)
    high = np.full(counts.shape[1], np.nan)
    used = counts.sum(axis=0) > 0
    if not used.any():
        return low, high
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(counts), size=(BOOTSTRAP, len(counts)))
    total = counts[:, used][picks].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = hits[:, used][picks].sum(axis=1) / total
    rates = np.where(total > 0, rates, np.nan)
    low[used] = np.nanpercentile(rates, 2.5, axis=0)
    high[used] = np.nanpercentile(rates, 97.5, axis=0)
    return low, high


def candidate_stats(
    cand: np.ndarray, p: np.ndarray, outcomes: np.ndarray, sessions: np.ndarray
) -> dict:
    """Estimates at or above a threshold: how many per session, how many came true
    (from the buy, by the session end) and how."""
    n = len(sessions)
    y = outcomes[..., TOUCHED]
    per_session = cand.sum(axis=(1, 2))[sessions]
    stocks = cand.any(axis=1).sum(axis=1)[sessions]
    known = cand & np.isfinite(y)
    result = {
        "estimates": int(cand.sum()),
        "per_session_mean": rounded(per_session.mean(), 1),
        "per_session_median": float(np.median(per_session)),
        "per_session_p10_p90": [
            float(np.percentile(per_session, 10)),
            float(np.percentile(per_session, 90)),
        ],
        "per_session_max": int(per_session.max()),
        "per_check_mean": rounded(per_session.mean() / SLOTS, 2),
        "sessions_with_any": rounded((per_session > 0).mean()),
        "stocks_per_session_mean": rounded(stocks.mean(), 1),
        "stocks_per_session_median": float(np.median(stocks)),
        "known_outcomes": int(known.sum()),
    }
    if not known.any():
        return result
    yk = y[known].astype(np.float64)
    sid = np.searchsorted(sessions, np.nonzero(known)[0])
    count = np.bincount(sid, minlength=n).astype(float)[:, None]
    hits = np.bincount(sid, weights=yk, minlength=n)[:, None]
    low, high = day_bootstrap(count, hits)
    daily = hits[count > 0] / count[count > 0]
    moved = known & (y == 1)
    up_first = outcomes[..., UP_FIRST][moved]
    minutes = outcomes[..., MINUTES][moved]
    has = cand.any(axis=1)
    dd, kk = np.nonzero(has)
    first = y[dd, cand.argmax(axis=1)[dd, kk], kk]
    first = first[np.isfinite(first)]
    times = []
    for label, a, b in TIMES:
        part = cand[:, a:b]
        part_known = part & np.isfinite(y[:, a:b])
        times.append(
            {
                "checks": label,
                "share_of_estimates": rounded(part.sum() / cand.sum()),
                "hit_rate": rounded(y[:, a:b][part_known].mean())
                if part_known.any()
                else None,
            }
        )
    result.update(
        {
            "mean_probability": rounded(p[known].astype(np.float64).mean()),
            "hit_rate": rounded(yk.mean()),
            "hit_rate_95": [rounded(low[0]), rounded(high[0])],
            "daily_hit_rate_p10_median_p90": [
                rounded(np.percentile(daily, q)) for q in (10, 50, 90)
            ],
            "sessions_below_70pct": rounded((daily < 0.7).mean()),
            "up_touched": rounded(outcomes[..., UP][known].mean()),
            "down_touched": rounded(outcomes[..., DOWN][known].mean()),
            "both_touched": rounded(
                (outcomes[..., UP][known] * outcomes[..., DOWN][known]).mean()
            ),
            "up_first_share_of_moves": rounded(np.nanmean(up_first))
            if np.isfinite(up_first).any()
            else None,
            "minutes_to_move_median_p75": [
                rounded(np.percentile(minutes, q), 1) for q in (50, 75)
            ]
            if minutes.size
            else None,
            "first_surfacing_hit_rate": rounded(first.mean()) if first.size else None,
            "by_time": times,
        }
    )
    return result


def evaluate(
    p: np.ndarray,
    data: dict,
    periods: tuple[tuple[str, str, str], ...],
    climate: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
    threshold: float = THRESHOLD,
) -> dict:
    """Probability accuracy by period: bands of 10 points (mean estimate vs actual,
    session-resampled 95% ranges), proper scores and their decomposition, ranking,
    calibration slope, day-level calibration and the candidates at each threshold.
    Estimates count every scored row; accuracy uses the rows whose outcome is known."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    outcomes = data["outcomes"]
    y = outcomes[..., TOUCHED]
    band = np.clip(np.nan_to_num(p * BANDS).astype(np.int64), 0, BANDS - 1)
    result = {}
    for name, first, last in periods:
        inside = (
            data["available"]
            & period_mask(data["days"], first, last)[:, None, None]
            & np.isfinite(p)
        )
        sessions = np.flatnonzero(inside.any(axis=(1, 2)))
        if not len(sessions):
            continue
        n = len(sessions)
        known = inside & np.isfinite(y)
        d_idx, s_idx, _ = np.nonzero(known)
        sid = np.searchsorted(sessions, d_idx)
        yk = y[known].astype(np.float64)
        pk = p[known].astype(np.float64)
        cell = sid * BANDS + band[known]
        counts = np.bincount(cell, minlength=n * BANDS).reshape(n, BANDS).astype(float)
        hits = np.bincount(cell, weights=yk, minlength=n * BANDS).reshape(n, BANDS)
        mass = np.bincount(cell, weights=pk, minlength=n * BANDS).reshape(n, BANDS)
        low, high = day_bootstrap(counts, hits)
        count, hit, total = counts.sum(axis=0), hits.sum(axis=0), mass.sum(axis=0)
        estimates = np.bincount(band[inside], minlength=BANDS)
        bands = []
        for b in range(BANDS):
            stock_days = int((inside & (band == b)).any(axis=1).sum())
            bands.append(
                {
                    "band": f"{10 * b}-{10 * b + 10}%",
                    "estimates": int(estimates[b]),
                    "per_session": round(estimates[b] / n, 1),
                    "stocks_per_session": round(stock_days / n, 1),
                    "known_outcomes": int(count[b]),
                    "mean_probability": rounded(total[b] / count[b])
                    if count[b]
                    else None,
                    "hit_rate": rounded(hit[b] / count[b]) if count[b] else None,
                    "hit_rate_95": [rounded(low[b]), rounded(high[b])]
                    if count[b]
                    else None,
                }
            )
        size, base = len(yk), float(yk.mean())
        used = count > 0
        gaps = np.abs(total[used] - hit[used]) / count[used]
        loss = log_loss_of(yk, pk)
        brier = float(np.mean((pk - yk) ** 2))
        uncertainty = base * (1 - base)
        slope = LogisticRegression(C=1e6).fit(logit(pk)[:, None], yk)
        per_day = counts.sum(axis=1)
        daily_p = mass.sum(axis=1)[per_day > 0] / per_day[per_day > 0]
        daily_y = hits.sum(axis=1)[per_day > 0] / per_day[per_day > 0]
        populated = count >= 100
        slot_known = known.sum(axis=(0, 2))
        slot_scored = inside.sum(axis=(0, 2))
        result[name] = {
            "sessions": n,
            "estimates": int(inside.sum()),
            "known_outcomes": size,
            "base_rate": rounded(base),
            "mean_probability": rounded(pk.mean()),
            "auc": rounded(roc_auc_score(yk, pk)),
            "log_loss": rounded(loss, 5),
            "log_loss_skill_vs_base_rate": rounded(
                1 - loss / log_loss_of(yk, np.full(size, base))
            ),
            "log_loss_skill_vs_time_of_day": rounded(
                1 - loss / log_loss_of(yk, climate[s_idx])
            ),
            "brier": rounded(brier, 5),
            "brier_skill": rounded(1 - brier / uncertainty),
            "brier_reliability": rounded(
                float(((total - hit)[used] ** 2 / count[used]).sum() / size), 5
            ),
            "brier_resolution": rounded(
                float(
                    (count[used] * (hit[used] / count[used] - base) ** 2).sum() / size
                ),
                5,
            ),
            "brier_uncertainty": rounded(uncertainty, 5),
            "calibration_error": rounded(float(np.abs(total - hit).sum() / size)),
            "max_band_gap_100plus": rounded(
                float(np.max(np.abs(total - hit)[populated] / count[populated]))
            )
            if populated.any()
            else None,
            "max_band_gap": rounded(float(gaps.max())),
            "calibration_slope": rounded(float(slope.coef_[0, 0]), 3),
            "calibration_intercept": rounded(float(slope.intercept_[0]), 3),
            "daily_calibration": {
                "correlation": rounded(float(np.corrcoef(daily_p, daily_y)[0, 1])),
                "mean_abs_gap": rounded(float(np.mean(np.abs(daily_p - daily_y)))),
                "p90_abs_gap": rounded(
                    float(np.percentile(np.abs(daily_p - daily_y), 90))
                ),
            },
            "max_probability": rounded(float(pk.max())),
            "bands": bands,
            "candidates": {
                f"{t:.2f}": candidate_stats(inside & (p >= t), p, outcomes, sessions)
                for t in thresholds
            },
            "by_decision_time": {
                "base_rate": [
                    rounded(v)
                    for v in np.where(known, y, 0).sum(axis=(0, 2))
                    / np.maximum(slot_known, 1)
                ],
                "mean_probability": [
                    rounded(v)
                    for v in np.where(inside, p, 0).sum(axis=(0, 2))
                    / np.maximum(slot_scored, 1)
                ],
                "share_at_threshold": [
                    rounded(v)
                    for v in (inside & (p >= threshold)).sum(axis=(0, 2))
                    / np.maximum(slot_scored, 1)
                ],
            },
        }
    return result


def label_sample(data: dict, size: int = 1000) -> list[dict]:
    """Random test rows with known outcomes, for an independent recheck."""
    rng = np.random.default_rng(SEED)
    y = data["outcomes"][..., TOUCHED]
    test = data["available"] & period_mask(data["days"], *TEST)[:, None, None]
    picks = np.flatnonzero((test & np.isfinite(y)).reshape(-1))
    picks = np.sort(rng.choice(picks, size=min(size, len(picks)), replace=False))
    rows = []
    for d, s, k in zip(*np.unravel_index(picks, y.shape), strict=True):
        day = int(data["days"][d])
        minute = OPEN_MINUTE + int(DECISIONS[s]) + 1
        last = OPEN_MINUTE + session_minutes(day) - 1
        values = data["outcomes"][d, s, k]
        rows.append(
            {
                "symbol": data["stocks"][k],
                "date": as_date(day).isoformat(),
                "buy_minute": f"{minute // 60:02d}:{minute % 60:02d}",
                "last_minute": f"{last // 60:02d}:{last % 60:02d}",
                "touched": int(values[TOUCHED]),
                "up": int(values[UP]),
                "down": int(values[DOWN]),
                "up_first": "" if np.isnan(values[UP_FIRST]) else int(values[UP_FIRST]),
                "minutes": "" if np.isnan(values[MINUTES]) else int(values[MINUTES]),
            }
        )
    return rows


def save_test_rows(path: Path, data: dict, raw: np.ndarray, p: np.ndarray) -> None:
    """Every scored test row (indices into ``days``/``stocks``), for later checks."""
    test = data["available"] & period_mask(data["days"], *TEST)[:, None, None]
    d, s, k = np.nonzero(test)

    def coded(values: np.ndarray, dtype) -> np.ndarray:
        return np.where(np.isfinite(values), values, -1).astype(dtype)

    o = data["outcomes"][test]
    np.savez_compressed(
        path,
        days=data["days"],
        stocks=np.asarray(data["stocks"]),
        day=d.astype(np.int16),
        slot=s.astype(np.int8),
        stock=k.astype(np.int16),
        probability=p[test].astype(np.float32),
        raw=raw[test].astype(np.float32),
        touched=coded(o[:, TOUCHED], np.int8),
        up=coded(o[:, UP], np.int8),
        down=coded(o[:, DOWN], np.int8),
        up_first=coded(o[:, UP_FIRST], np.int8),
        minutes=coded(o[:, MINUTES], np.int16),
    )


def bundle_document(
    model_path: Path,
    kept: list[str],
    calibration: dict,
    trees: int,
    metrics: dict,
    importance: dict,
    args: argparse.Namespace,
    excluded: dict,
) -> dict:
    """The live scanner's bundle format (``scanner/upside.py`` ``ScannerBundle``)."""
    move = percent(args.move_bps)
    return {
        "version": version(args.move_bps),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": f"the price touches +{move} or -{move} of the buy price before the "
        "session's last continuous minute (buy = open of the minute after the "
        "decision close)",
        "candidate_threshold": args.threshold,
        "features": kept,
        "percent_features": sorted(set(kept) & (PERCENT | {"dist_low", "rv_open"})),
        "optional_features": sorted(set(kept) & OPTIONAL),
        "new_inputs_not_in_live_code": sorted(set(kept) & set(EXTRA)),
        "decision_minutes": [int(d) for d in DECISIONS],
        "session": {
            "open": "09:15",
            "minutes": SESSION_MINUTES,
            "closing_auction_from": as_date(AUCTION_FROM).isoformat(),
            "closing_auction_minutes": AUCTION_MINUTES,
            "closing_auction_applies_to": "F&O stocks",
            "slot_feature": "minutes from the buy minute to the session's last "
            "continuous minute: session minutes - 1 - decision minute",
        },
        "calibration": calibration,
        "trees": trees,
        "periods": PRODUCTION,
        "metrics": metrics,
        "importance_top10": importance,
        "excluded_stock_days": excluded,
        "dataset": str(args.candles),
        "dataset_sha256": file_hash(args.candles / "dataset.json"),
        "model_sha256": file_hash(model_path),
        "source_sha256": file_hash(Path(__file__)),
    }


def importance_of(model, kept: list[str], top: int = 10) -> dict:
    values = model.get_feature_importance().round(2).tolist()
    return dict(sorted(zip(kept, values, strict=True), key=lambda kv: -kv[1])[:top])


def run(args: argparse.Namespace) -> dict:
    output: Path = args.output
    if output.exists():
        raise ValueError("Output exists; keep earlier studies")
    output.mkdir(parents=True)
    (output / "candidates").mkdir()
    shutil.copyfile(Path(__file__), output / "analysis_source.py")
    started = clock.monotonic()
    move = percent(args.move_bps)
    write_json(
        output / "protocol.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "goal": f"P(+{move} or -{move} of the buy price touched before the "
            f"session ends); candidates at >= {args.threshold:.2f}",
            "move_bps": args.move_bps,
            "threshold": args.threshold,
            "label": label_text(args.move_bps),
            "periods": {"research": RESEARCH, "test": TEST, "production": PRODUCTION},
            "settings": SETTINGS,
            "baseline_setting": BASELINE,
            "extended_inputs": list(EXTRA),
            "extended_margin": EXTENDED_MARGIN,
            "selection": "lowest raw log loss on 2024 H2 over all known rows",
            "fit_rows": FIT_ROWS,
            "early_stop_rows": STOP_ROWS,
            "iterations": args.iterations,
            "seed": SEED,
            "bands": BANDS,
            "bootstrap": f"{BOOTSTRAP} session resamples",
            "thresholds": list(THRESHOLDS),
            "stocks_limit": args.stocks,
            "source_sha256": file_hash(Path(__file__)),
        },
    )
    log("assembling features and labels")
    data = assemble(args.candles, args.workers, args.stocks, args.move_bps)
    excluded = apply_exclusions(data, args.candles, args.audit)
    base, base_dropped = prune_inputs(data, FEATURES)
    production_inputs = json.loads(PRODUCTION_BUNDLE.read_text("utf-8"))["features"]
    if args.stocks is None and base != production_inputs:
        raise ValueError("Pruned inputs differ from the production scanner's.")
    extended, extended_dropped = prune_inputs(data, NAMES)
    if [name for name in extended if name in FEATURES] != base:
        raise ValueError("The extended inputs must contain exactly the base inputs.")
    extras_ok = np.ones(data["available"].shape, bool)
    for name in EXTRA:
        extras_ok &= np.isfinite(data["features"][..., NAMES.index(name)])
    extras_missing = int((data["available"] & ~extras_ok).sum())
    data["available"] &= extras_ok
    data["day_flat"] = np.broadcast_to(
        data["days"][:, None, None], data["available"].shape
    ).reshape(-1)
    y_flat = data["outcomes"][..., TOUCHED].reshape(-1)
    known_flat = data["available"].reshape(-1) & np.isfinite(y_flat)
    select_rows = np.flatnonzero(known_flat & period_mask(data["day_flat"], *SELECT))
    log(
        f"assembled {len(data['stocks'])} stocks x {len(data['days'])} days; "
        f"scored rows {int(data['available'].sum()):,}, base inputs {len(base)}, "
        f"extended {len(extended)}"
    )

    inputs = {"base": base, "extended": extended}
    candidates: list[dict] = []
    models: dict[tuple[str, str], object] = {}

    def try_candidate(kind: str, setting: str) -> None:
        model, info = fit_candidate(
            data, inputs[kind], setting, RESEARCH, args.workers, args.iterations
        )
        raw = predict(model, data, inputs[kind], select_rows)
        info = {
            "inputs": kind,
            **info,
            "select_log_loss": round(log_loss_of(y_flat[select_rows], raw), 6),
        }
        model.save_model(str(output / "candidates" / f"{kind}_{setting}.cbm"))
        candidates.append(info)
        models[(kind, setting)] = model
        log(f"candidate {kind} {setting}: {info}")

    try_candidate("base", BASELINE)
    try_candidate("extended", BASELINE)
    score = {c["inputs"]: c["select_log_loss"] for c in candidates}
    kind = (
        "extended" if score["extended"] <= score["base"] - EXTENDED_MARGIN else "base"
    )
    for setting in SETTINGS:
        if setting != BASELINE:
            try_candidate(kind, setting)
    best = min(
        (c for c in candidates if c["inputs"] == kind),
        key=lambda c: c["select_log_loss"],
    )
    model, kept = models[(kind, best["setting"])], inputs[kind]
    model_path = output / f"scanner_move{args.move_bps}bps.cbm"
    model.save_model(str(model_path))

    log("scoring every row with the chosen model")
    rows = np.flatnonzero(data["available"].reshape(-1))
    raw = np.full(data["available"].shape, np.nan, np.float32)
    raw.reshape(-1)[rows] = predict(model, data, kept, rows)
    p, calibration = calibrate(raw, data, RESEARCH)
    climate = climatology(data, SELECT)
    selection = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": kind,
        "setting": best["setting"],
        "features": kept,
        "trees": best["trees"],
        "calibration": calibration,
        "candidates": candidates,
        "model_sha256": file_hash(model_path),
        "note": "written before any test-period outcome was scored",
    }
    write_json(output / "selection.json", selection)

    log("scoring the test period")
    test = evaluate(p, data, PERIODS, climate, threshold=args.threshold)
    save_test_rows(output / "test_predictions.npz", data, raw, p)
    sample = label_sample(data)
    with (output / "label_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)

    log("production refit on newer data")
    production_model, production_info = fit_candidate(
        data, kept, best["setting"], PRODUCTION, args.workers, args.iterations
    )
    late = np.flatnonzero(
        data["available"].reshape(-1)
        & period_mask(data["day_flat"], PRODUCTION["early_stop"][0], TEST[1])
    )
    production_raw = np.full(data["available"].shape, np.nan, np.float32)
    production_raw.reshape(-1)[late] = predict(production_model, data, kept, late)
    production_p, production_calibration = calibrate(production_raw, data, PRODUCTION)
    production_climate = climatology(data, PRODUCTION["calibration"])
    holdout = {
        "production": evaluate(
            production_p,
            data,
            HOLDOUT_PERIODS,
            production_climate,
            threshold=args.threshold,
        ),
        "research_model_same_rows": evaluate(
            p, data, HOLDOUT_PERIODS, production_climate, threshold=args.threshold
        ),
    }
    production_dir = output / "production"
    production_dir.mkdir()
    production_model.save_model(str(production_dir / "model.cbm"))
    write_json(
        production_dir / "bundle.json",
        bundle_document(
            production_dir / "model.cbm",
            kept,
            production_calibration,
            production_info["trees"],
            holdout,
            importance_of(production_model, kept),
            args,
            excluded,
        ),
    )

    labels = data["outcomes"][..., TOUCHED]
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "research_only_scanner_evaluation",
        "goal": f"candidates with >= {args.threshold:.0%} probability of touching "
        f"+{move} or -{move} of the buy price before the session ends",
        "design": {
            "decisions": "every 5 minutes, closes 09:40 to 14:55 (64 per session)",
            "label": label_text(args.move_bps),
            "estimates": "every scored stock at every check; accuracy counts the "
            "rows whose outcome is known",
            "periods": {"research": RESEARCH, "test": TEST},
            "bands": "10-point bands; 95% ranges resample whole sessions "
            f"({BOOTSTRAP} times)",
        },
        "dataset": str(args.candles),
        "dataset_sha256": file_hash(args.candles / "dataset.json"),
        "universe": {"stocks": len(data["stocks"]), "days": len(data["days"])},
        "excluded_stock_days": excluded,
        "extras_missing_rows": extras_missing,
        "label_counts": dict(sorted(data["label_counts"].items())),
        "rows": {
            "scored": int(data["available"].sum()),
            "known_outcomes": int((data["available"] & np.isfinite(labels)).sum()),
        },
        "features": {
            "base": base,
            "base_dropped": base_dropped,
            "extended": extended,
            "extended_dropped": extended_dropped,
        },
        "candidates": candidates,
        "chosen": {
            "inputs": kind,
            "setting": best["setting"],
            "trees": best["trees"],
            "calibration": {
                k: v
                for k, v in calibration.items()
                if k not in ("isotonic_x", "isotonic_y")
            },
            "importance_top10": importance_of(model, kept),
            "selection_sha256": file_hash(output / "selection.json"),
        },
        "time_of_day_baseline": [rounded(v) for v in climate],
        "test": test,
        "production_refit": {
            "periods": PRODUCTION,
            "model": production_info,
            "calibration": {
                k: v
                for k, v in production_calibration.items()
                if k not in ("isotonic_x", "isotonic_y")
            },
            "holdout": holdout,
        },
        "runtime_minutes": round((clock.monotonic() - started) / 60, 1),
        "source_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--stocks", type=int, help="first N stocks only (smoke runs)")
    parser.add_argument("--move-bps", type=int, default=MOVE_BPS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    if not 0 < args.move_bps < 10_000 or args.threshold not in THRESHOLDS:
        parser.error("--move-bps must be 1-9999 and --threshold one of the studied")
    report = run(args)
    for name, metrics in report["test"].items():
        top = metrics["candidates"][f"{args.threshold:.2f}"]
        print(
            f"{name:22s} AUC {metrics['auc']} ECE {metrics['calibration_error']} "
            f"base {metrics['base_rate']} | >= {args.threshold}: "
            f"{top['per_session_mean']}/session, {top['stocks_per_session_mean']} "
            f"stocks, hit {top.get('hit_rate')}"
        )
    print("chosen", report["chosen"]["inputs"], report["chosen"]["setting"])
    print("runtime_minutes", report["runtime_minutes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
