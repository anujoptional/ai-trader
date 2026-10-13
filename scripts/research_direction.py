"""Direction study: green (+1) or red (-1) from a decision time to 15:00 IST.

Research only, on the verified 50-stock Kite cache (2024-01..2026-09). Decisions
every 5 minutes 09:30-14:45; entry at the next minute's open, exit at the 15:00
price (close of the 14:59 candle). Flat outcomes are excluded from accuracy.
Discovery uses 2024 only (model fit Jan-Oct, early stop Nov-Dec); 2025 fixes the
confidence thresholds; 2026 is the untouched test. Rows within a day share the
market move, so confidence intervals resample whole days.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from ai_trader.history import CandleStore
from scripts.research_fno_scanner import EPOCH, epoch_day, read_daily, read_sessions
from scripts.train_tp_only import file_hash, write_json

DECISIONS = 14 + 5 * np.arange(64)
SLOTS = len(DECISIONS)
EXIT = 344
LOOKBACK = 20
SEED = 20261011
BOOTSTRAP = 1000
PERIODS = {
    "fit_2024_01_10": ("2024-01-01", "2024-10-31"),
    "stop_2024_11_12": ("2024-11-01", "2024-12-31"),
    "discovery_2024": ("2024-01-01", "2024-12-31"),
    "validation_2025": ("2025-01-01", "2025-12-31"),
    "test_2026": ("2026-01-01", "2026-09-30"),
}
REPORTED = ("discovery_2024", "validation_2025", "test_2026")
COVERAGES = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
SECTORS = {
    "Financial Services": "FINNIFTY",
    "Information Technology": "NIFTYIT",
    "Automobile and Auto Components": "NIFTYAUTO",
    "Fast Moving Consumer Goods": "NIFTYFMCG",
    "Healthcare": "NIFTYPHARMA",
    "Metals & Mining": "NIFTYMETAL",
}
INDICES = ("NIFTY", "BANKNIFTY", "INDIAVIX", *sorted(set(SECTORS.values())))
STOCK = (
    "r1",
    "r5",
    "r15",
    "r30",
    "r60",
    "ropen",
    "rday",
    "gap",
    "rv30",
    "rvday",
    "up30",
    "rangepos",
    "dist_hi",
    "dist_lo",
    "vwap",
    "imb15",
    "imb60",
    "clv15",
    "vol15",
    "rvol",
    "d_r1",
    "d_r5",
    "d_r20",
    "d_ma20",
    "d_clv",
    "d_atr",
    "d_volr",
    "beta",
)
MARKET = (
    "n_r5",
    "n_r15",
    "n_r30",
    "n_r60",
    "n_ropen",
    "n_rday",
    "n_gap",
    "n_rv30",
    "n_rangepos",
    "n_twap",
    "n_d_r1",
    "n_d_r5",
    "b_r30",
    "b_rday",
    "v_level",
    "v_chg",
    "v_r30",
    "horizon",
    "weekday",
)
CROSS = (
    "sec_r30",
    "sec_rday",
    "rel_day",
    "rel_30",
    "rel_sec_day",
    "cs_rday",
    "cs_r30",
    "cs_vwap",
    "cs_rvol",
    "cs_imb60",
    "breadth_day",
    "breadth_vwap",
    "xs_r30",
)
FEATURES = STOCK + MARKET + CROSS


def path_features(prices: np.ndarray, volume: bool) -> dict[str, np.ndarray]:
    """Inputs at each decision close; nothing after candle ``m`` is read."""
    open_, high, low, close = (prices[:, i] for i in range(4))
    m = DECISIONS
    path = np.log(np.concatenate(([open_[0]], close)))
    steps = np.diff(path)
    squares = np.concatenate(([0.0], np.cumsum(steps**2)))
    ups = np.concatenate(([0.0], np.cumsum(steps > 0)))
    start30 = np.maximum(m + 1 - 30, 0)
    top = np.maximum.accumulate(high)[m]
    bottom = np.minimum.accumulate(low)[m]
    span = top - bottom

    def back(k: int) -> np.ndarray:
        return path[m + 1] - path[np.maximum(m + 1 - k, 0)]

    out = {
        "r1": back(1),
        "r5": back(5),
        "r15": back(15),
        "r30": back(30),
        "r60": back(60),
        "ropen": path[m + 1] - path[0],
        "level": path[m + 1],
        "open": np.full(SLOTS, path[0]),
        "rv30": np.sqrt(squares[m + 1] - squares[start30]),
        "rvday": np.sqrt(squares[m + 1] / (m + 1)),
        "up30": (ups[m + 1] - ups[start30]) / (m + 1 - start30),
        "rangepos": np.divide(
            close[m] - bottom, span, out=np.full(SLOTS, 0.5), where=span > 0
        ),
        "dist_hi": np.log(close[m] / top),
        "dist_lo": np.log(close[m] / bottom),
    }
    if not volume:
        twap = np.cumsum(close) / np.arange(1, len(close) + 1)
        out["twap"] = np.log(close[m] / twap[m])
        return out
    traded = prices[:, 4]
    cumulative = np.concatenate(([0.0], np.cumsum(traded)))
    priced = np.cumsum((high + low + close) / 3 * traded)
    signed = np.concatenate(([0.0], np.cumsum(np.sign(close - open_) * traded)))
    spread = high - low
    location = np.divide(
        2 * close - high - low, spread, out=np.zeros_like(spread), where=spread > 0
    )
    locations = np.concatenate(([0.0], np.cumsum(location)))

    def window(series: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        first = np.maximum(m + 1 - k, 0)
        return series[m + 1] - series[first], m + 1 - first

    flow15, volume15 = window(signed, 15)[0], window(cumulative, 15)
    flow60, volume60 = window(signed, 60)[0], window(cumulative, 60)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        out |= {
            "vwap": np.log(close[m] / (priced[m] / cumulative[m + 1])),
            "imb15": flow15 / volume15[0],
            "imb60": flow60 / volume60,
            "clv15": window(locations, 15)[0] / window(locations, 15)[1],
            "vol15": volume15[0] / (cumulative[m + 1] / (m + 1) * volume15[1]),
            "cumvol": cumulative[m + 1],
        }
    return out


def forward(prices: np.ndarray) -> np.ndarray:
    """Log return from the next minute's open to the 15:00 price."""
    return np.log(prices[EXIT, 3] / prices[DECISIONS + 1, 0])


def prior_daily(
    days: np.ndarray,
    values: np.ndarray,
    day: int,
    previous: int,
    market: dict[int, float],
) -> dict[str, float] | None:
    """Features from completed daily bars; the last one must be the prior session."""
    position = int(np.searchsorted(days, day))
    if position < 61 or days[position - 1] != previous:
        return None
    close = values[:position, 3]
    last = close[-1]
    high, low = values[position - 1, 1], values[position - 1, 2]
    window = values[position - 20 : position]
    true_range = np.maximum(window[:, 1], close[-21:-1]) - np.minimum(
        window[:, 2], close[-21:-1]
    )
    index = np.asarray(
        [market.get(int(d), np.nan) for d in days[position - 61 : position]]
    )
    own, mine = np.diff(np.log(index)), np.diff(np.log(close[-61:]))
    usable = np.isfinite(own)
    beta = (
        float(np.cov(mine[usable], own[usable])[0, 1] / np.var(own[usable], ddof=1))
        if usable.sum() >= 40
        else np.nan
    )
    return {
        "d_r1": float(np.log(last / close[-2])),
        "d_r5": float(np.log(last / close[-6])),
        "d_r20": float(np.log(last / close[-21])),
        "d_ma20": float(np.log(last / close[-20:].mean())),
        "d_clv": float((2 * last - high - low) / (high - low)) if high > low else 0.0,
        "d_atr": float(true_range.mean() / last),
        "d_volr": float(
            values[position - 1, 4] / values[position - 20 : position, 4].mean()
        ),
        "beta": beta,
        "prev_close": float(last),
    }


def stock_block(task: tuple[str, Path, np.ndarray, dict[int, float]]) -> dict:
    symbol, candles, calendar, market = task
    store = CandleStore(candles)
    sessions = read_sessions(store.verified_path(f"NSE_{symbol}.csv"))
    days, daily = read_daily(store.verified_path(f"daily/NSE_{symbol}.csv"))
    kept, rows, labels, volumes = [], [], [], []
    for day in sorted(sessions):
        prices = sessions[day]
        features = path_features(prices, volume=True)
        position = int(np.searchsorted(calendar, day))
        prior = (
            prior_daily(days, daily, day, int(calendar[position - 1]), market)
            if position
            else None
        )
        if prior is not None and len(volumes) >= LOOKBACK:
            with np.errstate(divide="ignore", invalid="ignore"):
                features["rvol"] = features["cumvol"] / np.mean(
                    volumes[-LOOKBACK:], axis=0
                )
            close = np.log(prior["prev_close"])
            features["rday"] = features["level"] - close
            features["gap"] = features["open"] - close
            for name in STOCK[20:]:
                features[name] = np.full(SLOTS, prior[name])
            rows.append(np.column_stack([features[name] for name in STOCK]))
            labels.append(forward(prices))
            kept.append(day)
        volumes.append(features["cumvol"])
    return {
        "symbol": symbol,
        "days": np.asarray(kept, dtype=np.int64),
        "features": np.asarray(rows, np.float32).reshape(-1, SLOTS, len(STOCK)),
        "labels": np.asarray(labels, np.float64).reshape(-1, SLOTS),
    }


def index_blocks(store: CandleStore) -> tuple[np.ndarray, dict, dict]:
    calendar, nifty_daily = read_daily(store.verified_path("daily/NSE_NIFTY.csv"))
    closes = {}
    for name in INDICES:
        days, values = read_daily(store.verified_path(f"daily/NSE_{name}.csv"))
        closes[name] = dict(zip(days.tolist(), values[:, 3].tolist(), strict=True))
    blocks: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for name in INDICES:
        blocks[name] = {}
        for day, prices in read_sessions(
            store.verified_path(f"NSE_{name}.csv")
        ).items():
            position = int(np.searchsorted(calendar, day))
            previous = (
                closes[name].get(int(calendar[position - 1])) if position else None
            )
            if previous is None:
                continue
            features = path_features(prices, volume=False)
            features["rday"] = features["level"] - np.log(previous)
            features["gap"] = features["open"] - np.log(previous)
            features["label"] = forward(prices)
            blocks[name][day] = features
    nifty = dict(zip(calendar.tolist(), nifty_daily[:, 3].tolist(), strict=True))
    return calendar, blocks, nifty


def market_rows(
    blocks: dict, nifty: dict, calendar: np.ndarray, day: int
) -> np.ndarray:
    n, b, v = blocks["NIFTY"][day], blocks["BANKNIFTY"][day], blocks["INDIAVIX"][day]
    position = int(np.searchsorted(calendar, day))
    history = [nifty[int(d)] for d in calendar[position - 6 : position]]
    weekday = date.fromordinal(EPOCH + day).weekday()
    values = {
        "n_r5": n["r5"],
        "n_r15": n["r15"],
        "n_r30": n["r30"],
        "n_r60": n["r60"],
        "n_ropen": n["ropen"],
        "n_rday": n["rday"],
        "n_gap": n["gap"],
        "n_rv30": n["rv30"],
        "n_rangepos": n["rangepos"],
        "n_twap": n["twap"],
        "n_d_r1": np.log(history[-1] / history[-2]),
        "n_d_r5": np.log(history[-1] / history[0]),
        "b_r30": b["r30"],
        "b_rday": b["rday"],
        "v_level": np.exp(v["level"]),
        "v_chg": v["rday"],
        "v_r30": v["r30"],
        "horizon": (EXIT - DECISIONS).astype(float),
        "weekday": float(weekday),
    }
    return np.column_stack([np.broadcast_to(values[name], SLOTS) for name in MARKET])


def ranks(values: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile (0..1) within each day-slot among available stocks."""
    filled = np.where(available, values, np.nan)
    order = np.argsort(np.argsort(np.where(available, values, np.inf), axis=2), axis=2)
    count = available.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = order / np.maximum(count - 1, 1)
    return np.where(np.isfinite(filled), result, np.nan)


def assemble(candles: Path, workers: int) -> dict:
    store = CandleStore(candles)
    universe = json.loads(store.verified_path("universe.json").read_text("utf-8"))
    targets = [e for e in universe["instruments"] if e["role"] == "equity_target"]
    calendar, blocks, nifty = index_blocks(store)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                stock_block,
                [(e["symbol"], candles, calendar, nifty) for e in targets],
            )
        )
    days = np.asarray(
        sorted(
            day
            for day in blocks["NIFTY"]
            if all(day in blocks[name] for name in INDICES)
            and int(np.searchsorted(calendar, day)) >= 6
        ),
        dtype=np.int64,
    )
    shape = (len(days), SLOTS, len(targets))
    features = np.full((*shape, len(FEATURES)), np.nan, np.float32)
    labels = np.full(shape, np.nan)
    index = {int(day): row for row, day in enumerate(days)}
    for column, result in enumerate(results):
        rows = np.asarray(
            [index.get(int(day), -1) for day in result["days"]], dtype=int
        )
        keep = rows >= 0
        features[rows[keep], :, column, : len(STOCK)] = result["features"][keep]
        labels[rows[keep], :, column] = result["labels"][keep]
    offset = len(STOCK)
    for row, day in enumerate(days):
        features[row, :, :, offset : offset + len(MARKET)] = market_rows(
            blocks, nifty, calendar, int(day)
        )[:, None, :]
    position = {name: i for i, name in enumerate(FEATURES)}
    sector_of = [SECTORS.get(e.get("industry"), "NIFTY") for e in targets]
    for column, sector in enumerate(sector_of):
        for row, day in enumerate(days):
            features[row, :, column, position["sec_r30"]] = blocks[sector][int(day)][
                "r30"
            ]
            features[row, :, column, position["sec_rday"]] = blocks[sector][int(day)][
                "rday"
            ]

    def get(name: str) -> np.ndarray:
        return features[..., position[name]]

    features[..., position["rel_day"]] = get("rday") - get("n_rday")
    features[..., position["rel_30"]] = get("r30") - get("n_r30")
    features[..., position["rel_sec_day"]] = get("rday") - get("sec_rday")
    present = np.isfinite(features[..., : len(STOCK)]).all(axis=3) & np.isfinite(labels)
    for name, source in (
        ("cs_rday", "rday"),
        ("cs_r30", "r30"),
        ("cs_vwap", "vwap"),
        ("cs_rvol", "rvol"),
        ("cs_imb60", "imb60"),
    ):
        features[..., position[name]] = ranks(get(source), present)
    with np.errstate(invalid="ignore"):
        count = present.sum(axis=2, keepdims=True)
        for name, values in (
            ("breadth_day", get("rday") > 0),
            ("breadth_vwap", get("vwap") > 0),
        ):
            share = np.where(present, values, False).sum(axis=2, keepdims=True) / count
            features[..., position[name]] = np.broadcast_to(share, shape)
        mean = np.where(present, get("r30"), 0).sum(axis=2, keepdims=True) / count
        features[..., position["xs_r30"]] = np.broadcast_to(mean, shape)
    nifty_label = np.stack([blocks["NIFTY"][int(day)]["label"] for day in days])
    available = np.isfinite(features).all(axis=3) & np.isfinite(labels)
    return {
        "days": days,
        "symbols": [e["symbol"] for e in targets],
        "features": features,
        "labels": labels,
        "nifty_label": nifty_label,
        "available": available,
    }


def in_period(days: np.ndarray, name: str) -> np.ndarray:
    first, last = PERIODS[name]
    return (days >= epoch_day(first)) & (days <= epoch_day(last))


def interval(correct: np.ndarray, total: np.ndarray, rng) -> list[float]:
    """Day-resampled 95% interval for pooled accuracy; arrays are per day."""
    picks = rng.integers(0, len(total), size=(BOOTSTRAP, len(total)))
    samples = correct[picks].sum(axis=1) / np.maximum(total[picks].sum(axis=1), 1)
    return [float(v) for v in np.percentile(samples, (2.5, 97.5))]


def accuracy(predicted: np.ndarray, truth: np.ndarray, day: np.ndarray, rng) -> dict:
    """Accuracy of nonzero predictions on nonflat outcomes, with day-level CI."""
    used = (predicted != 0) & (truth != 0)
    if not used.any():
        return {"rows": 0}
    hits = (predicted == truth) & used
    unique, inverse = np.unique(day[used], return_inverse=True)
    correct = np.bincount(inverse, weights=hits[used].astype(float))
    total = np.bincount(inverse).astype(float)
    return {
        "rows": int(used.sum()),
        "days": len(unique),
        "accuracy": float(hits.sum() / used.sum()),
        "ci95": interval(correct, total, rng),
        "predicted_green_share": float((predicted[used] > 0).mean()),
    }


def univariate(data: dict, rng) -> dict:
    """Decile green rates per feature, with 2024 thresholds reused out of sample."""
    feature, available = data["features"], data["available"]
    truth = np.sign(data["labels"])
    day = np.broadcast_to(data["days"][:, None, None], truth.shape)
    found = in_period(day, "discovery_2024") & available & (truth != 0)
    result = {}
    for i, name in enumerate(FEATURES):
        values = feature[..., i]
        low, high = np.quantile(values[found], (0.1, 0.9))
        result[name] = {}
        for period in REPORTED:
            inside = in_period(day, period) & available & (truth != 0)
            green = truth[inside] > 0
            top = values[inside] >= high
            bottom = values[inside] <= low
            prediction = np.where(values >= high, 1, np.where(values <= low, -1, 0))
            result[name][period] = {
                "base_green": float(green.mean()),
                "top_decile_green": float(green[top].mean()) if top.any() else None,
                "bottom_decile_green": float(green[bottom].mean())
                if bottom.any()
                else None,
                "momentum_tail_accuracy": accuracy(
                    np.where(inside, prediction, 0), truth, day, rng
                ).get("accuracy"),
            }
    return result


def rules(data: dict, rng) -> dict:
    feature = data["features"]
    position = {name: i for i, name in enumerate(FEATURES)}
    truth = np.sign(data["labels"])
    day = np.broadcast_to(data["days"][:, None, None], truth.shape)
    candidates = {
        "always_green": np.ones_like(truth),
        "stock_since_open": np.sign(feature[..., position["ropen"]]),
        "stock_day": np.sign(feature[..., position["rday"]]),
        "stock_last_30m": np.sign(feature[..., position["r30"]]),
        "nifty_day": np.sign(feature[..., position["n_rday"]]),
        "nifty_since_open": np.sign(feature[..., position["n_ropen"]]),
        "vwap_reversion": -np.sign(feature[..., position["vwap"]]),
        "gap_fade": -np.sign(feature[..., position["gap"]]),
        "stock_and_nifty_agree": np.where(
            np.sign(feature[..., position["rday"]])
            == np.sign(feature[..., position["n_rday"]]),
            np.sign(feature[..., position["rday"]]),
            0,
        ),
    }
    result = {}
    for name, prediction in candidates.items():
        result[name] = {}
        for period in REPORTED:
            inside = in_period(day, period) & data["available"]
            result[name][period] = accuracy(
                np.where(inside, prediction, 0), truth, day, rng
            )
    return result


def co_movement(data: dict) -> dict:
    truth = np.sign(data["labels"])
    market = np.sign(data["nifty_label"])[:, :, None]
    day = np.broadcast_to(data["days"][:, None, None], truth.shape)
    result = {}
    for period in REPORTED:
        inside = in_period(day, period) & data["available"] & (truth != 0)
        both = inside & (market != 0)
        result[period] = {
            "flat_share": float(
                (np.sign(data["labels"]) == 0)[
                    in_period(day, period) & data["available"]
                ].mean()
            ),
            "green_share": float((truth[inside] > 0).mean()),
            "same_sign_as_nifty": float((truth == market)[both].mean()),
        }
    return result


def fit_model(
    features: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    days: np.ndarray,
    columns: tuple[str, ...],
    threads: int,
):
    from catboost import CatBoostClassifier

    chosen_columns = [FEATURES.index(name) for name in columns]
    flat = features.reshape(-1, len(FEATURES))[:, chosen_columns]
    goal = target.reshape(-1)
    day = np.broadcast_to(
        days.reshape(-1, *([1] * (target.ndim - 1))), target.shape
    ).reshape(-1)
    usable = rows.reshape(-1) & (goal != 0)
    fit = np.flatnonzero(usable & in_period(day, "fit_2024_01_10"))
    stop = np.flatnonzero(usable & in_period(day, "stop_2024_11_12"))
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=1500,
        l2_leaf_reg=10,
        eval_metric="Logloss",
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        random_seed=SEED,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(flat[fit], goal[fit] > 0, eval_set=(flat[stop], goal[stop] > 0))
    probability = np.full(goal.shape, np.nan)
    chosen = np.flatnonzero(rows.reshape(-1))
    probability[chosen] = model.predict_proba(flat[chosen])[:, 1]
    importance = sorted(
        zip(columns, model.get_feature_importance().tolist(), strict=True),
        key=lambda item: -item[1],
    )
    return probability.reshape(target.shape), {
        "trees": int(model.tree_count_),
        "fit_rows": len(fit),
        "stop_rows": len(stop),
        "top_features": [[name, round(value, 2)] for name, value in importance[:15]],
    }


def selective(probability, truth, rows, days, rng) -> dict:
    """Confidence cut-offs fixed on 2025, then applied unchanged to 2026."""
    day = np.broadcast_to(days.reshape(-1, *([1] * (truth.ndim - 1))), truth.shape)
    confidence = np.abs(probability - 0.5)
    prediction = np.where(probability >= 0.5, 1, -1)
    validation = rows & in_period(day, "validation_2025") & (truth != 0)
    result = {}
    for coverage in COVERAGES:
        cut = float(np.quantile(confidence[validation], 1 - coverage))
        result[str(coverage)] = {"confidence_cut": cut}
        for period in REPORTED:
            inside = rows & in_period(day, period)
            picked = inside & (confidence >= cut)
            score = accuracy(np.where(picked, prediction, 0), truth, day, rng)
            score["coverage"] = float(picked.sum() / max(inside.sum(), 1))
            result[str(coverage)][period] = score
    return result


def auc(probability: np.ndarray, truth: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    used = np.isfinite(probability) & (truth != 0)
    return float(roc_auc_score(truth[used] > 0, probability[used]))


def run(candles: Path, output: Path, workers: int) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    data = assemble(candles, workers)
    truth = np.sign(data["labels"])
    available = data["available"]
    day = np.broadcast_to(data["days"][:, None, None], truth.shape)
    stock_probability, stock_model = fit_model(
        data["features"], truth, available, data["days"], FEATURES, workers
    )
    market_truth = np.sign(data["nifty_label"])
    market_rows = available.any(axis=2)
    market_probability, market_model = fit_model(
        data["features"][:, :, :1, :],
        market_truth[:, :, None],
        market_rows[:, :, None],
        data["days"],
        MARKET + ("breadth_day", "breadth_vwap", "xs_r30"),
        workers,
    )
    position = {name: i for i, name in enumerate(FEATURES)}
    beta = data["features"][..., position["beta"]]
    relative_truth = np.sign(data["labels"] - beta * data["nifty_label"][:, :, None])
    relative_probability, relative_model = fit_model(
        data["features"], relative_truth, available, data["days"], FEATURES, workers
    )
    models = {}
    for name, probability, target, rows, meta in (
        ("stock_direction", stock_probability, truth, available, stock_model),
        (
            "nifty_direction",
            market_probability[:, :, 0],
            market_truth,
            market_rows,
            market_model,
        ),
        (
            "stock_vs_nifty",
            relative_probability,
            relative_truth,
            available,
            relative_model,
        ),
    ):
        period_day = np.broadcast_to(
            data["days"].reshape(-1, *([1] * (target.ndim - 1))), target.shape
        )
        models[name] = {
            **meta,
            "auc": {
                period: auc(
                    np.where(rows & in_period(period_day, period), probability, np.nan),
                    target,
                )
                for period in REPORTED
            },
            "selective": selective(probability, target, rows, data["days"], rng),
        }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "research_complete_not_promoted",
        "universe": {
            "stocks": len(data["symbols"]),
            "days": len(data["days"]),
            "rows": int(available.sum()),
        },
        "label": "sign(15:00 price / next-minute open after decision); flat excluded",
        "decisions": "every 5 minutes 09:30-14:45 IST",
        "periods": PERIODS,
        "co_movement": co_movement(data),
        "rules": rules(data, rng),
        "models": models,
        "univariate": univariate(data, rng),
        "rows_by_period": {
            period: int((available & in_period(day, period)).sum())
            for period in PERIODS
        },
        "source_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "report.json", report)
    np.savez_compressed(
        output / "predictions.npz",
        days=data["days"],
        stock_probability=stock_probability.astype(np.float32),
        nifty_probability=market_probability[:, :, 0].astype(np.float32),
        labels=data["labels"].astype(np.float32),
        nifty_label=data["nifty_label"].astype(np.float32),
        available=available,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = run(args.candles, args.output, args.workers)
    print(json.dumps(report["universe"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
