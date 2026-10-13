"""F&O scanner study: recall-first LONG candidate filters, 213 stocks, 2020-2026.

Research only, on a verified Kite cache. Decisions every 5 minutes 09:40-13:30.
Outcomes from the decision close: +0.3% within 60 minutes (quick), +1% within 60
minutes (big), +0.3% before -0.3% by 15:00 (clean) and a same-day proxy (+0.3% else
the 15:00 close). Charges: Zerodha intraday rates at Rs 2 lakh, which apply to a
same-session CNC buy and sell. Thresholds and the CatBoost ranker are fitted on the
training period (early stopping on the next); later periods are out of sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from ai_trader.costs import ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore
from scripts.research_delivery_strategy import DP_INR, side_costs
from scripts.train_tp_only import file_hash, write_json

EPOCH = date(1970, 1, 1).toordinal()
OPEN_MINUTE = 9 * 60 + 15
LAST = 344
DECISIONS = 24 + 5 * np.arange(47)
SLOTS = len(DECISIONS)
LEVEL_BPS = 30
BIG_BPS = 100
QUICK = 60
LOOKBACK = 20
NONE = 10_000
CLIP_INR = 200_000
LIQUID_LOG10_INR = 9.0
BUDGETS = (1, 2, 3, 5, 10, 20)
SEED = 20261011
HEADER = ["start_time", "open", "high", "low", "close", "volume"]
DAILY_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
STOCK_FEATURES = (
    "rv20",
    "rv_ratio",
    "rvol",
    "range_atr",
    "day_ret",
    "gap",
    "ret15",
    "ret_open",
    "vwap_dev",
    "log_turnover20",
    "atr20",
    "hist_rv",
    "log_price",
    "dist_high",
    "dist_low",
)
CONTEXT_FEATURES = ("nifty_day", "nifty_ret15", "vix", "rel_strength", "slot")
FEATURES = STOCK_FEATURES + CONTEXT_FEATURES


def half_years(first: int, last: int) -> tuple[tuple[str, str, str], ...]:
    periods = []
    for year in range(first, last + 1):
        periods.append((f"{year}H1", f"{year}-01-01", f"{year}-06-30"))
        periods.append((f"{year}H2", f"{year}-07-01", f"{year}-12-31"))
    return tuple(periods)


def epoch_day(label: str) -> int:
    return date.fromisoformat(label).toordinal() - EPOCH


def read_sessions(path: Path) -> dict[int, np.ndarray]:
    """Complete 09:15-14:59 sessions as [345, 5] open/high/low/close/volume."""
    rows: dict[str, list[list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != HEADER:
            raise ValueError(f"Unexpected candle header in {path.name}")
        for start, open_, high, low, close, volume in reader:
            if start[16:] != ":00+05:30":
                raise ValueError(f"Unexpected timestamp {start} in {path.name}")
            minute = int(start[11:13]) * 60 + int(start[14:16]) - OPEN_MINUTE
            if 0 <= minute <= LAST:
                rows.setdefault(start[:10], []).append(
                    [minute, float(open_), float(high), float(low), float(close)]
                    + [float(volume or "nan")]
                )
    sessions = {}
    for day, values in rows.items():
        array = np.asarray(values)
        array = array[np.argsort(array[:, 0], kind="stable")]
        if np.array_equal(array[:, 0], np.arange(LAST + 1)):
            sessions[epoch_day(day)] = array[:, 1:]
    return sessions


def read_daily(path: Path) -> tuple[np.ndarray, np.ndarray]:
    days, values = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != DAILY_HEADER:
            raise ValueError(f"Unexpected daily header in {path.name}")
        for stamp, open_, high, low, close, volume in reader:
            days.append(epoch_day(stamp[:10]))
            values.append(
                [float(open_), float(high), float(low), float(close)]
                + [float(volume or "nan")]
            )
    return np.asarray(days, dtype=np.int64), np.asarray(values).reshape(-1, 5)


def session_outcomes(prices: np.ndarray) -> dict[str, np.ndarray]:
    """First-touch minutes after each decision close, in exact paise arithmetic."""
    paise = np.rint(prices[:, :4] * 100).astype(np.int64)
    high, low, close = paise[:, 1], paise[:, 2], paise[:, 3]
    future = np.arange(LAST + 1)[None, :] > DECISIONS[:, None]
    high_run = np.maximum.accumulate(np.where(future, high[None, :], 0), axis=1)
    low_run = np.minimum.accumulate(np.where(future, low[None, :], 10**12), axis=1)
    base = close[DECISIONS][:, None]

    def first(untouched: np.ndarray) -> np.ndarray:
        # Rows are monotone, so the count of untouched columns is the first touch.
        index = untouched.sum(axis=1)
        return np.where(index <= LAST, index - DECISIONS, NONE).astype(np.int16)

    return {
        "up": first(high_run * 10_000 < base * (10_000 + LEVEL_BPS)),
        "down": first(low_run * 10_000 > base * (10_000 - LEVEL_BPS)),
        "big": first(high_run * 10_000 < base * (10_000 + BIG_BPS)),
        "big_down": first(low_run * 10_000 > base * (10_000 - BIG_BPS)),
        "ret1500": (close[LAST] / base[:, 0] - 1).astype(np.float32),
        # Realistic fill: the open of the minute after the decision close.
        "ret_entry": (close[LAST] / paise[DECISIONS + 1, 0] - 1).astype(np.float32),
    }


def session_features(prices: np.ndarray) -> dict[str, np.ndarray | float]:
    """Same-session inputs at each decision close; no future minute is read."""
    open_, high, low, close, volume = prices.T
    squares = np.concatenate(([0.0], np.cumsum(np.diff(np.log(close)) ** 2)))
    d = DECISIONS
    cumulative = np.cumsum(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.cumsum((high + low + close) / 3 * volume) / cumulative
    top, bottom = np.maximum.accumulate(high), np.minimum.accumulate(low)
    return {
        "rv20": np.sqrt(squares[d] - squares[d - 20]),
        "cumvol": cumulative[d],
        "range": (top[d] - bottom[d]) / close[d],
        "ret15": close[d] / close[d - 15] - 1,
        "ret_open": close[d] / open_[0] - 1,
        "vwap_dev": close[d] / vwap[d] - 1,
        "log_price": np.log10(close[d]),
        "dist_high": close[d] / top[d] - 1,
        "dist_low": close[d] / bottom[d] - 1,
        "close": close[d],
        "open": float(open_[0]),
        "day_rv": math.sqrt(20 * squares[LAST] / LAST),
    }


def stock_rows(task: tuple[str, Path, np.ndarray]) -> dict:
    symbol, candles, calendar = task
    store = CandleStore(candles)
    sessions = read_sessions(store.verified_path(f"NSE_{symbol}.csv"))
    daily_days, daily = read_daily(store.verified_path(f"daily/NSE_{symbol}.csv"))
    days = sorted(sessions)
    features = np.full((len(days), SLOTS, len(STOCK_FEATURES)), np.nan, np.float32)
    outcomes = {
        name: np.empty((len(days), SLOTS), dtype)
        for name, dtype in (
            ("up", np.int16),
            ("down", np.int16),
            ("big", np.int16),
            ("big_down", np.int16),
            ("ret1500", np.float32),
            ("ret_entry", np.float32),
        )
    }
    day_rv: list[float] = []
    volumes: list[np.ndarray] = []
    for row, day in enumerate(days):
        part = session_features(sessions[day])
        for name, values in session_outcomes(sessions[day]).items():
            outcomes[name][row] = values
        position = int(np.searchsorted(daily_days, day))
        market = int(np.searchsorted(calendar, day))
        ready = (
            len(day_rv) >= LOOKBACK
            and position > LOOKBACK
            and market > 0
            and daily_days[position - 1] == calendar[market - 1]
        )
        if ready:
            window = daily[position - LOOKBACK : position]
            prior = daily[position - LOOKBACK - 1 : position - 1, 3]
            previous = window[-1, 3]
            true_range = np.maximum(window[:, 1], prior) - np.minimum(
                window[:, 2], prior
            )
            atr = float(np.mean(true_range) / previous)
            history = float(np.median(day_rv[-LOOKBACK:]))
            with np.errstate(divide="ignore", invalid="ignore"):
                rvol = part["cumvol"] / np.mean(volumes[-LOOKBACK:], axis=0)
                turnover = np.log10(np.mean(window[:, 3] * window[:, 4]))
            features[row] = np.column_stack(
                (
                    part["rv20"],
                    part["rv20"] / history,
                    rvol,
                    part["range"] / atr,
                    part["close"] / previous - 1,
                    np.full(SLOTS, part["open"] / previous - 1),
                    part["ret15"],
                    part["ret_open"],
                    part["vwap_dev"],
                    np.full(SLOTS, turnover),
                    np.full(SLOTS, atr),
                    np.full(SLOTS, history),
                    part["log_price"],
                    part["dist_high"],
                    part["dist_low"],
                )
            )
        day_rv.append(part["day_rv"])
        volumes.append(part["cumvol"])
    return {
        "symbol": symbol,
        "days": np.asarray(days, dtype=np.int64),
        "features": features,
        **outcomes,
    }


def context(candles: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Market days with complete NIFTY/VIX sessions and their decision-time context."""
    store = CandleStore(candles)
    calendar, nifty_daily = read_daily(store.verified_path("daily/NSE_NIFTY.csv"))
    nifty = read_sessions(store.verified_path("NSE_NIFTY.csv"))
    vix = read_sessions(store.verified_path("NSE_INDIAVIX.csv"))
    days, values = [], []
    for day in sorted(set(nifty) & set(vix)):
        position = int(np.searchsorted(calendar, day))
        if position == 0:
            continue
        close = nifty[day][DECISIONS, 3]
        values.append(
            np.column_stack(
                (
                    close / nifty_daily[position - 1, 3] - 1,
                    close / nifty[day][DECISIONS - 15, 3] - 1,
                    vix[day][DECISIONS, 3],
                )
            )
        )
        days.append(day)
    return calendar, np.asarray(days, dtype=np.int64), np.asarray(values, np.float32)


def assemble(candles: Path, workers: int) -> dict:
    store = CandleStore(candles)
    universe = json.loads(store.verified_path("universe.json").read_text("utf-8"))
    stocks = [
        entry["symbol"]
        for entry in universe["instruments"]
        if entry["role"] == "equity_target"
    ]
    calendar, days, market = context(candles)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(stock_rows, [(s, candles, calendar) for s in stocks]))
    shape = (len(days), SLOTS, len(stocks))
    data = {
        "days": days,
        "stocks": stocks,
        "features": np.full((*shape, len(FEATURES)), np.nan, np.float32),
        "up": np.full(shape, NONE, np.int16),
        "down": np.full(shape, NONE, np.int16),
        "big": np.full(shape, NONE, np.int16),
        "big_down": np.full(shape, NONE, np.int16),
        "ret1500": np.full(shape, np.nan, np.float32),
        "ret_entry": np.full(shape, np.nan, np.float32),
    }
    index = {int(day): row for row, day in enumerate(days)}
    for column, result in enumerate(results):
        rows = np.asarray([index.get(int(day), -1) for day in result["days"]])
        keep = rows >= 0
        data["features"][rows[keep], :, column, : len(STOCK_FEATURES)] = result[
            "features"
        ][keep]
        for name in ("up", "down", "big", "big_down", "ret1500", "ret_entry"):
            data[name][rows[keep], :, column] = result[name][keep]
    offset = len(STOCK_FEATURES)
    data["features"][..., offset] = market[:, :, None, 0]
    data["features"][..., offset + 1] = market[:, :, None, 1]
    data["features"][..., offset + 2] = market[:, :, None, 2]
    data["features"][..., offset + 3] = (
        data["features"][..., STOCK_FEATURES.index("day_ret")] - market[:, :, None, 0]
    )
    data["features"][..., offset + 4] = np.arange(SLOTS)[None, :, None]
    data["available"] = np.isfinite(data["features"]).all(axis=3) & np.isfinite(
        data["ret1500"]
    )
    return data


def top_k(score: np.ndarray, available: np.ndarray, k: int) -> np.ndarray:
    ranked = np.where(available, score, -np.inf)
    order = np.argsort(-ranked, axis=2, kind="stable")[..., :k]
    mask = np.zeros(available.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=2)
    return mask & available


def fit_ranker(data: dict, train: tuple[str, str], stop: tuple[str, str], threads: int):
    from catboost import CatBoostClassifier

    flat = data["features"].reshape(-1, len(FEATURES))
    target = (data["up"] <= QUICK).reshape(-1)
    day = np.broadcast_to(data["days"][:, None, None], data["up"].shape).reshape(-1)
    available = data["available"].reshape(-1)
    rng = np.random.default_rng(SEED)

    def sample(first: str, last: str, size: int) -> np.ndarray:
        rows = np.flatnonzero(
            available & (day >= epoch_day(first)) & (day <= epoch_day(last))
        )
        return np.sort(rng.choice(rows, size=min(size, len(rows)), replace=False))

    fit_rows, stop_rows = sample(*train, 2_000_000), sample(*stop, 500_000)
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.08,
        iterations=800,
        l2_leaf_reg=10,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=50,
        use_best_model=True,
        random_seed=SEED,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        flat[fit_rows],
        target[fit_rows],
        eval_set=(flat[stop_rows], target[stop_rows]),
    )
    score = np.full(available.shape, np.nan, np.float32)
    rows = np.flatnonzero(available)
    for start in range(0, len(rows), 2_000_000):
        chunk = rows[start : start + 2_000_000]
        score[chunk] = model.predict_proba(flat[chunk])[:, 1]
    importance = dict(
        zip(FEATURES, model.get_feature_importance().round(3).tolist(), strict=True)
    )
    return score.reshape(data["up"].shape), {
        "trees": int(model.tree_count_),
        "fit_rows": len(fit_rows),
        "stop_rows": len(stop_rows),
        "importance": importance,
    }


def outcome_labels(data: dict) -> dict[str, np.ndarray]:
    return {
        "quick": data["up"] <= QUICK,
        "big": data["big"] <= QUICK,
        "clean": data["up"] < data["down"],
        "proxy": np.where(data["up"] < NONE, LEVEL_BPS / 10_000, data["ret1500"]),
    }


def score_rule(
    mask: np.ndarray, available: np.ndarray, labels: dict, cost: float
) -> dict:
    chosen = mask & available
    sessions = int(available.any(axis=(1, 2)).sum())
    if not chosen.any() or sessions == 0:
        return {"candidates": 0}
    quick, big, clean, proxy = (
        labels[name] for name in ("quick", "big", "clean", "proxy")
    )
    distinct = chosen.any(axis=1).sum(axis=1)
    return {
        "candidates": int(chosen.sum()),
        "candidates_per_day": float(chosen.sum() / sessions),
        "distinct_stocks_per_day": float(distinct[distinct > 0].mean()),
        "quick_percent": 100 * float(quick[chosen].mean()),
        "quick_lift": float(quick[chosen].mean() / quick[available].mean()),
        "quick_recall_percent": 100
        * float(quick[chosen].sum() / quick[available].sum()),
        "big_percent": 100 * float(big[chosen].mean()),
        "big_recall_percent": 100 * float(big[chosen].sum() / big[available].sum()),
        "clean_percent": 100 * float(clean[chosen].mean()),
        "clean_base_percent": 100 * float(clean[available].mean()),
        "proxy_gross_percent": 100 * float(proxy[chosen].mean()),
        "proxy_net_percent": 100 * float(proxy[chosen].mean() - cost),
    }


def run(
    candles: Path,
    output: Path,
    train: tuple[str, str],
    stop: tuple[str, str],
    years: tuple[int, int],
    workers: int,
) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    output.mkdir(parents=True)
    data = assemble(candles, workers)
    column = {name: i for i, name in enumerate(FEATURES)}
    feature = data["features"]
    available = data["available"]
    day = np.broadcast_to(data["days"][:, None, None], available.shape)
    fit = available & (day >= epoch_day(train[0])) & (day <= epoch_day(train[1]))
    rv20 = feature[..., column["rv20"]]
    low, high = (float(v) for v in np.quantile(rv20[fit], (0.3, 0.7)))
    score, ranker = fit_ranker(data, train, stop, workers)
    liquid = feature[..., column["log_turnover20"]] >= LIQUID_LOG10_INR
    nifty_up = feature[..., column["nifty_day"]] > 0
    aligned = (
        nifty_up
        & (feature[..., column["rel_strength"]] > 0)
        & (feature[..., column["vwap_dev"]] > 0)
    )
    rvol = feature[..., column["rvol"]]
    model5 = top_k(score, available, 5)
    rules = {
        "all": available,
        "liquid_100cr": available & liquid,
        "low_move_bottom30": available & (rv20 <= low),
        "move_top30": available & (rv20 >= high),
        "rv_ratio_1_5": available & (feature[..., column["rv_ratio"]] >= 1.5),
        "rvol_1_5": available & (rvol >= 1.5),
        "move_top30_rvol_1_5": available & (rv20 >= high) & (rvol >= 1.5),
        "top5_rv_ratio": top_k(feature[..., column["rv_ratio"]], available, 5),
        "top5_model_liquid": top_k(score, available & liquid, 5),
        "top5_model_nifty_up": model5 & nifty_up,
        "top5_model_aligned": model5 & aligned,
    }
    for k in BUDGETS:
        rules[f"top{k}_rv20"] = top_k(rv20, available, k)
        rules[f"top{k}_model"] = model5 if k == 5 else top_k(score, available, k)
    cost = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(Decimal(CLIP_INR)))
    buy, sell = side_costs("stock")
    periods = half_years(*years)
    labels = outcome_labels(data)
    results = {}
    for name, mask in rules.items():
        results[name] = {}
        for period, first, last in periods:
            inside = (day >= epoch_day(first)) & (day <= epoch_day(last))
            results[name][period] = score_rule(mask, available & inside, labels, cost)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "research_complete_not_promoted",
        "universe": {"stocks": len(data["stocks"]), "days": len(data["days"])},
        "slots": "every 5 minutes, decision closes 09:40-13:30 IST, LONG only",
        "outcomes": {
            "quick": "+0.3% within 60 minutes",
            "big": "+1% within 60 minutes",
            "clean": "+0.3% before -0.3% by 15:00 (same-minute touches fail)",
            "proxy": "+0.3% if touched by 15:00, else the 15:00 close",
        },
        "charges_fraction_at_rs_2_lakh": {
            "same_day_cnc_or_intraday": cost,
            "overnight_delivery": buy + sell + DP_INR / CLIP_INR,
        },
        "train": list(train),
        "early_stop": list(stop),
        "rv20_thresholds_fit_on_train": [low, high],
        "liquidity_log10_inr": LIQUID_LOG10_INR,
        "ranker": ranker,
        "available_slots": int(available.sum()),
        "rules": results,
        "source_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", default="2020-01-01:2022-12-31")
    parser.add_argument("--stop", default="2023-01-01:2023-06-30")
    parser.add_argument("--years", default="2020:2026")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    train = tuple(args.train.split(":"))
    stop = tuple(args.stop.split(":"))
    first, last = (int(value) for value in args.years.split(":"))
    report = run(args.candles, args.output, train, stop, (first, last), args.workers)
    print(json.dumps({"rules": len(report["rules"]), **report["universe"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
