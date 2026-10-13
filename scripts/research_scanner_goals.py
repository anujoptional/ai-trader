"""Scanner-only study: four buy-side goals, each trained, calibrated and tested alone.

Research only, on the NSE-basis corrected Kite dataset (213 F&O stocks, January 2020
to September 2026). For a buy at the open of the minute after each decision close
(every 5 minutes, 09:40 to 14:55), the goals are:

a. +1% above the buy price is touched before the session ends: the 15:29 minute, or
   15:14 from 2026-08-03, when NSE's closing auction (15:15-15:35) replaced the last
   15 minutes of continuous trading in F&O stocks.
b. +1% is touched before -1%, with no time limit.
c. +2% is touched before -2%, with no time limit.
d. +5% is touched before -5%, with no time limit.

A minute that opens beyond a level reaches it first (overnight gaps included); a
minute that reaches both levels counts as the down move, because the order inside
one minute is unknown. An outcome still open when the data ends, or one that needs
a session the stock is missing or that is flagged for unreliable prices, is unknown:
it still counts as an estimate but is left out of accuracy (counts are reported).

Each goal has its own CatBoost model on the delivery scanner's indicator features,
except that the time input (``slot``) holds the minutes left from the buy to the
session's last continuous minute, so the models see the shorter auction-era
sessions: fit 2020-2023, early stopping 2024 H1, calibration fitted on 2024 H2
(raw, Platt or isotonic, whichever has the lowest log loss on 2024 H1), test
2025-01 to 2026-09.
"""

from __future__ import annotations

import argparse
import csv
import json
import time as clock
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai_trader.history import CandleStore
from scripts.e2e_research import apply_exclusions
from scripts.research_delivery_pipeline import (
    CALIBRATION,
    EARLY_STOP,
    FEATURES,
    FIT,
    LOOKBACK,
    MARKET,
    OPTIONAL,
    SESSION_MINUTES,
    STOCK_FEATURES,
    TEST,
    as_date,
    complete_holding,
    daily_columns,
    first_exit,
    flagged_days,
    logit,
    market_context,
    period_mask,
    prune,
    read_path,
    session_features,
    wilder_rsi,
)
from scripts.research_fno_scanner import (
    LAST,
    OPEN_MINUTE,
    SEED,
    epoch_day,
    read_daily,
    read_sessions,
)
from scripts.train_tp_only import file_hash, write_json

DECISIONS = 24 + 5 * np.arange(64)  # decision closes 09:40 to 14:55
SLOTS = len(DECISIONS)
GOALS = (
    ("a_up1_by_close", "+1% touched before the session ends"),
    ("b_up1_before_down1", "+1% touched before -1%, no time limit"),
    ("c_up2_before_down2", "+2% touched before -2%, no time limit"),
    ("d_up5_before_down5", "+5% touched before -5%, no time limit"),
)
CLOSE_BPS = 100
BRACKETS = (100, 200, 500)  # goals b, c and d
BANDS = 10  # probability bands 0-10%, 10-20%, ..., 90-100%
AUCTION_FROM = epoch_day("2026-08-03")  # NSE closing auction for F&O stocks
AUCTION_MINUTES = 360  # continuous trading 09:15-15:14 from then on
PERIODS = (
    ("validation_2024", "2024-01-01", "2024-12-31"),
    ("test_2025H1", "2025-01-01", "2025-06-30"),
    ("test_2025H2", "2025-07-01", "2025-12-31"),
    ("test_2026", "2026-01-01", "2026-09-30"),
    ("test_closing_auction", "2026-08-03", "2026-09-30"),
    ("test_all", *TEST),
)


def session_minutes(day: int) -> int:
    """Minutes of continuous trading in an F&O stock's session on ``day``."""
    return AUCTION_MINUTES if day >= AUCTION_FROM else SESSION_MINUTES


def day_labels(
    ohlc: np.ndarray,
    days: np.ndarray,
    minutes: np.ndarray,
    held_days: np.ndarray,
    calendar: np.ndarray,
    flagged: list[int],
    day: int,
    counts: Counter,
) -> np.ndarray:
    """Outcome of a buy at every decision of ``day``: [SLOTS, goals] of 1, 0 or NaN
    (unknown). ``ohlc`` holds the stock's minutes in paise, sorted by day and minute;
    ``held_days`` are the days it has minutes on."""
    out = np.full((SLOTS, len(GOALS)), np.nan)
    first = int(np.searchsorted(days, day))
    end = int(np.searchsorted(days, day, side="right"))
    if end - first <= LAST or not np.array_equal(
        minutes[first : first + LAST + 1], np.arange(LAST + 1)
    ):
        counts["sessions_incomplete_before_1500"] += 1
        return out
    offsets = DECISIONS + 1
    block = ohlc[first:end] * 10_000
    entry = ohlc[first + offsets, 0]
    if np.array_equal(minutes[first:end], np.arange(session_minutes(day))):
        best = np.maximum.accumulate(block[::-1, 1])[::-1]
        out[:, 0] = best[offsets] >= entry * (10_000 + CLOSE_BPS)
    else:
        counts["a_unknown_close_incomplete"] += SLOTS
    size = end - first
    after = np.arange(size)[None, :] >= offsets[:, None]
    for goal, bps in enumerate(BRACKETS, start=1):
        tag = GOALS[goal][0][0]
        up, down = entry * (10_000 + bps), entry * (10_000 - bps)
        hit_up = after & (block[None, :, 1] >= up[:, None])
        hit_down = after & (block[None, :, 2] <= down[:, None])
        first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), size)
        first_down = np.where(hit_down.any(axis=1), hit_down.argmax(axis=1), size)
        label = (first_up < first_down).astype(float)
        both = (first_up == first_down) & (first_up < size)
        if both.any():
            opened = block[first_up[both], 0]
            label[both] = opened >= up[both]
            counts[f"{tag}_same_minute_counted_down"] += int(
                ((opened < up[both]) & (opened > down[both])).sum()
            )
        for k in np.flatnonzero((first_up == size) & (first_down == size)):
            row, _, reason = first_exit(ohlc, end, int(entry[k]), bps, bps)
            if reason == "open":
                label[k] = np.nan
                counts[f"{tag}_unknown_open_at_data_end"] += 1
                continue
            exit_day = int(days[row])
            held = int(
                np.searchsorted(held_days, exit_day) - np.searchsorted(held_days, day)
            )
            if not complete_holding(day, exit_day, held, calendar, flagged):
                label[k] = np.nan
                counts[f"{tag}_unknown_missing_or_flagged_session"] += 1
                continue
            label[k] = reason in ("tp", "tp_gap")
            counts[f"{tag}_resolved_after_the_session"] += 1
            counts[f"{tag}_same_minute_counted_down"] += reason == "sl_same_minute"
        out[:, goal] = label
    return out


def stock_rows(task: tuple[str, Path, np.ndarray, list[int]]) -> dict:
    symbol, candles, calendar, flagged = task
    store = CandleStore(candles)
    path = store.verified_path(f"NSE_{symbol}.csv")
    sessions = read_sessions(path)
    path_days, minutes, ohlc = read_path(path)
    held_days = np.unique(path_days)
    daily_days, daily = read_daily(store.verified_path(f"daily/NSE_{symbol}.csv"))
    rsi = wilder_rsi(daily[:, 3])
    days = sorted(sessions)
    features = np.full((len(days), SLOTS, len(STOCK_FEATURES)), np.nan, np.float32)
    labels = np.full((len(days), SLOTS, len(GOALS)), np.nan, np.float32)
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
            features[row] = daily_columns(
                daily, position, rsi, day_rv, volumes, part, SLOTS
            )
        labels[row] = day_labels(
            ohlc, path_days, minutes, held_days, calendar, flagged, day, counts
        )
        day_rv.append(part["day_rv"])
        volumes.append(part["cumvol"])
    return {
        "days": np.asarray(days, dtype=np.int64),
        "features": features,
        "labels": labels,
        "counts": counts,
    }


def assemble(candles: Path, audit: Path | None, workers: int) -> dict:
    store = CandleStore(candles)
    universe = json.loads(store.verified_path("universe.json").read_text("utf-8"))
    stocks = [
        entry["symbol"]
        for entry in universe["instruments"]
        if entry["role"] == "equity_target"
    ]
    calendar, days, market = market_context(candles, DECISIONS)
    flags = flagged_days(candles, audit)
    shape = (len(days), SLOTS, len(stocks))
    data = {
        "days": days,
        "stocks": stocks,
        "features": np.full((*shape, len(FEATURES)), np.nan, np.float32),
        "labels": np.full((*shape, len(GOALS)), np.nan, np.float32),
        "label_counts": Counter(),
    }
    session = np.zeros(shape, bool)
    index = {int(day): row for row, day in enumerate(days)}
    tasks = [(s, candles, calendar, sorted(flags.get(s, ()))) for s in stocks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for column, result in enumerate(pool.map(stock_rows, tasks)):
            rows = np.asarray([index.get(int(day), -1) for day in result["days"]])
            keep = rows >= 0
            data["features"][rows[keep], :, column, : len(STOCK_FEATURES)] = result[
                "features"
            ][keep]
            data["labels"][rows[keep], :, column] = result["labels"][keep]
            session[rows[keep], :, column] = True
            data["label_counts"].update(result["counts"])
    features = data["features"]
    col = {name: i for i, name in enumerate(FEATURES)}
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
    for name in FEATURES:
        if name not in OPTIONAL:
            session &= np.isfinite(features[..., col[name]])
    data["available"] = session
    return data


def log_loss_of(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_goal(
    data: dict,
    kept: list[str],
    goal: int,
    threads: int,
    *,
    fit: tuple[str, str] = FIT,
    early_stop: tuple[str, str] = EARLY_STOP,
    calibration: tuple[str, str] = CALIBRATION,
):
    """CatBoost for one goal; returns calibrated probabilities on every available
    row (NaN elsewhere), the model and its fitting details."""
    from catboost import CatBoostClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    columns = [FEATURES.index(name) for name in kept]
    shape = data["available"].shape
    flat = data["features"].reshape(-1, len(FEATURES))
    y = data["labels"][..., goal].reshape(-1)
    day = np.broadcast_to(data["days"][:, None, None], shape).reshape(-1)
    available = data["available"].reshape(-1)
    known = available & np.isfinite(y)
    rng = np.random.default_rng(SEED + goal)

    def sample(first: str, last: str, size: int) -> np.ndarray:
        rows = np.flatnonzero(known & period_mask(day, first, last))
        return np.sort(rng.choice(rows, size=min(size, len(rows)), replace=False))

    fit_rows = sample(*fit, 3_000_000)
    stop_rows = sample(*early_stop, 600_000)
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.08,
        iterations=2000,
        l2_leaf_reg=10,
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
    raw = np.full(available.shape, np.nan, np.float32)
    rows = np.flatnonzero(available)
    for start in range(0, len(rows), 2_000_000):
        chunk = rows[start : start + 2_000_000]
        raw[chunk] = model.predict_proba(flat[chunk][:, columns])[:, 1]
    fitted = np.flatnonzero(known & period_mask(day, *calibration))
    platt = LogisticRegression(C=1e6).fit(logit(raw[fitted])[:, None], y[fitted])
    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
        raw[fitted], y[fitted]
    )
    maps = {
        "raw": lambda p: p,
        "platt": lambda p: platt.predict_proba(logit(p)[:, None])[:, 1],
        "isotonic": isotonic.predict,
    }
    check = np.flatnonzero(known & period_mask(day, *early_stop))
    scores = {
        name: round(log_loss_of(y[check], f(raw[check])), 6) for name, f in maps.items()
    }
    choice = min(scores, key=scores.get)
    calibrated = np.full(available.shape, np.nan, np.float32)
    for start in range(0, len(rows), 4_000_000):
        chunk = rows[start : start + 4_000_000]
        calibrated[chunk] = maps[choice](raw[chunk])
    info = {
        "fit_rows": len(fit_rows),
        "early_stop_rows": len(stop_rows),
        "trees": int(model.tree_count_),
        "fit_seconds": round(clock.monotonic() - started, 1),
        "calibration": {
            "fitted_on": f"{calibration[0]} to {calibration[1]}",
            "rows": len(fitted),
            "log_loss_early_stop_period": scores,
            "chosen": choice,
            "platt_slope": float(platt.coef_[0, 0]),
            "platt_intercept": float(platt.intercept_[0]),
            "isotonic_x": isotonic.X_thresholds_.tolist()
            if choice == "isotonic"
            else None,
            "isotonic_y": isotonic.y_thresholds_.tolist()
            if choice == "isotonic"
            else None,
        },
        "importance_top10": dict(
            sorted(
                zip(
                    kept, model.get_feature_importance().round(2).tolist(), strict=True
                ),
                key=lambda item: -item[1],
            )[:10]
        ),
    }
    return calibrated.reshape(shape), model, info


def goal_metrics(
    p: np.ndarray,
    y: np.ndarray,
    data: dict,
    periods: tuple[tuple[str, str, str], ...] = PERIODS,
) -> dict:
    """Accuracy and probability bands of one goal by period. Estimates count every
    available row; accuracy uses the rows whose outcome is known."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    available = data["available"]
    days = np.broadcast_to(data["days"][:, None, None], available.shape)
    band = np.clip(np.nan_to_num(p * BANDS).astype(np.int64), 0, BANDS - 1)
    result = {}
    for name, first, last in periods:
        inside = available & period_mask(days, first, last)
        if not inside.any():
            continue
        sessions = int(inside.any(axis=(1, 2)).sum())
        known = inside & np.isfinite(y)
        yk, pk, bk = y[known].astype(np.int8), p[known].astype(np.float64), band[known]
        base = float(yk.mean())
        loss = log_loss_of(yk, pk)
        base_loss = log_loss_of(yk, np.full(len(yk), base))
        slope = LogisticRegression(C=1e6).fit(logit(pk)[:, None], yk)
        estimates = np.bincount(band[inside], minlength=BANDS)
        resolved = np.bincount(bk, minlength=BANDS)
        hits = np.bincount(bk, weights=yk, minlength=BANDS)
        mass = np.bincount(bk, weights=pk, minlength=BANDS)
        bands = []
        for b in range(BANDS):
            stock_days = int((inside & (band == b)).any(axis=1).sum())
            bands.append(
                {
                    "band": f"{10 * b}-{10 * b + 10}%",
                    "estimates": int(estimates[b]),
                    "estimates_per_session": round(estimates[b] / sessions, 2),
                    "stocks_per_session": round(stock_days / sessions, 2),
                    "known_outcomes": int(resolved[b]),
                    "mean_probability": round(mass[b] / resolved[b], 4)
                    if resolved[b]
                    else None,
                    "hit_rate": round(hits[b] / resolved[b], 4)
                    if resolved[b]
                    else None,
                }
            )
        ece = float(np.abs(hits - mass).sum() / len(yk))
        result[name] = {
            "sessions": sessions,
            "estimates": int(inside.sum()),
            "known_outcomes": int(known.sum()),
            "base_rate": round(base, 4),
            "mean_probability": round(float(pk.mean()), 4),
            "auc": round(float(roc_auc_score(yk, pk)), 4),
            "log_loss": round(loss, 5),
            "log_loss_base_rate": round(base_loss, 5),
            "log_loss_skill": round(1 - loss / base_loss, 4),
            "brier": round(float(np.mean((pk - yk) ** 2)), 5),
            "calibration_error": round(ece, 4),
            "calibration_slope": round(float(slope.coef_[0, 0]), 3),
            "calibration_intercept": round(float(slope.intercept_[0]), 3),
            "max_probability": round(float(pk.max()), 4),
            "bands": bands,
        }
    return result


def label_sample(data: dict, size: int = 250) -> list[dict]:
    """Random test rows with known outcomes, for an independent recheck."""
    rng = np.random.default_rng(SEED)
    days = np.broadcast_to(data["days"][:, None, None], data["available"].shape)
    test = data["available"] & period_mask(days, *TEST)
    rows = []
    for goal, (name, _) in enumerate(GOALS):
        y = data["labels"][..., goal]
        picks = np.flatnonzero((test & np.isfinite(y)).reshape(-1))
        picks = rng.choice(picks, size=min(size, len(picks)), replace=False)
        for d, s, k in zip(*np.unravel_index(np.sort(picks), test.shape), strict=True):
            minute = OPEN_MINUTE + int(DECISIONS[s]) + 1
            rows.append(
                {
                    "goal": name,
                    "symbol": data["stocks"][k],
                    "date": as_date(data["days"][d]).isoformat(),
                    "buy_minute": f"{minute // 60:02d}:{minute % 60:02d}",
                    "label": int(y[d, s, k]),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict:
    output: Path = args.output
    if output.exists():
        raise ValueError("Output exists; keep earlier studies")
    output.mkdir(parents=True)
    started = clock.monotonic()
    data = assemble(args.candles, args.audit, args.workers)
    excluded = apply_exclusions(data, args.candles, args.audit)
    kept, dropped = prune(data)
    goals = {}
    for goal, (name, definition) in enumerate(GOALS):
        probability, model, info = fit_goal(data, kept, goal, args.workers)
        model.save_model(str(output / f"scanner_{name}.cbm"))
        goals[name] = {
            "definition": definition,
            "model": info,
            "metrics": goal_metrics(probability, data["labels"][..., goal], data),
        }
        del probability
    sample = label_sample(data)
    with (output / "label_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "research_only_scanner_evaluation",
        "design": {
            "decisions": "every 5 minutes, closes 09:40 to 14:55 (64 per session)",
            "buy": "open of the minute after the decision close",
            "session_end": "last continuous minute 15:29, or 15:14 from 2026-08-03 "
            "(NSE closing auction for F&O stocks); the slot feature holds the minutes "
            "from the buy to it",
            "levels": "a minute opening beyond a level reaches it first; a minute "
            "reaching both counts as the down move",
            "unknown": "open at the data end (2026-09-30), or needing a missing or "
            "flagged session; counted as estimates, left out of accuracy",
            "periods": {
                "fit": FIT,
                "early_stop": EARLY_STOP,
                "calibration": CALIBRATION,
                "test": TEST,
            },
            "note": "validation_2024 contains the early-stopping and calibration "
            "periods, so it is not out of sample",
        },
        "dataset": str(args.candles),
        "dataset_sha256": file_hash(args.candles / "dataset.json"),
        "excluded_stock_days": excluded,
        "universe": {"stocks": len(data["stocks"]), "days": len(data["days"])},
        "features": {"kept": kept, "dropped": dropped},
        "label_counts": dict(sorted(data["label_counts"].items())),
        "goals": goals,
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
    report = run(parser.parse_args())
    summary = {
        name: {
            k: goal["metrics"]["test_all"][k]
            for k in ("base_rate", "auc", "log_loss_skill", "calibration_error")
        }
        for name, goal in report["goals"].items()
    }
    print(json.dumps(summary, indent=1))
    print("runtime_minutes", report["runtime_minutes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
