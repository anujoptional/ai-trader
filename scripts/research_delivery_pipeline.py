"""Delivery (CNC) pipeline research: +1%-in-2h scanner, components, AI BUY / NO_BUY.

Research only, on the NSE-basis corrected Kite dataset (213 F&O stocks, January 2020
to September 2026). Design (user, 2026-10-12):

A. Scanner: calibrated CatBoost probability that the price touches +1% above the buy
   price within the next 120 minutes, fitted fresh on intraday (micro), daily (macro)
   and market indicators; candidates are probabilities above 0.75. Direction
   components, each evaluated on its own: futures open interest as of the previous
   session, NSE filings, and the indicator features. The order book has no history
   and web news cannot be replayed without seeing the future, so neither is part of
   this evaluation.
B. AI: BUY or NO_BUY per candidate (delivery only), on rounds after the model's
   knowledge cutoff.

Trades: buy at the open of the minute after the decision close; sell when the price
first reaches +1% or -1% of the buy price, with no time limit; Rs 2 lakh in whole
shares; one buy per stock per day and none while that stock is held; Zerodha
intraday charges when sold in the same session, delivery charges otherwise.

Periods: scanner fit 2020-2023, early stopping 2024 H1, probability calibration
2024 H2, test 2025-01 to 2026-09 (AI from 2026-06, after gpt-6-luna's cutoff).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time as clock
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ai_trader.ai.decision import DELIVERY, DecisionRequest, request_json
from ai_trader.ai.packet import candidate, delivery_block
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.costs import ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore
from ai_trader.news.features import event_context
from ai_trader.news.store import NewsStore
from scripts.e2e_research import OI_FEATURES, apply_exclusions, load_oi
from scripts.research_delivery_strategy import DP_INR, side_costs
from scripts.research_fno_scanner import (
    EPOCH,
    HEADER,
    LAST,
    OPEN_MINUTE,
    SEED,
    epoch_day,
    read_daily,
    read_sessions,
)
from scripts.train_tp_only import file_hash, write_json

DECISIONS = 24 + 5 * np.arange(41)  # decision closes 09:40 to 13:00
SLOTS = len(DECISIONS)
HORIZON = 120  # minutes after the decision in which the scanner's +1% must print
TARGET_BPS = 100  # +1%: the scanner target and the take-profit
STOP_BPS = 100  # -1% stop-loss
THRESHOLD = 0.75
NOTIONAL_INR = 200_000
LOOKBACK = 20
NONE = 10_000
SESSION_MINUTES = 375  # 09:15 to 15:29
FIT = ("2020-01-01", "2023-12-31")
EARLY_STOP = ("2024-01-01", "2024-06-30")
CALIBRATION = ("2024-07-01", "2024-12-31")
TEST = ("2025-01-01", "2026-09-30")
AI_FROM = "2026-06-01"  # after gpt-6-luna's 2026-05-18 knowledge cutoff
NEWS_FROM = "2024-12-08"  # first stored NSE filings
PERIODS = (
    ("validation_2024", "2024-01-01", "2024-12-31"),
    ("test_2025H1", "2025-01-01", "2025-06-30"),
    ("test_2025H2", "2025-07-01", "2025-12-31"),
    ("test_2026", "2026-01-01", "2026-09-30"),
    ("ai_window", AI_FROM, TEST[1]),
    ("test_all", *TEST),
)
INTRADAY = (
    "ret5",
    "ret15",
    "ret60",
    "ret_open",
    "gap",
    "vwap_dev",
    "dist_high",
    "range_pos",
    "rv20",
    "rv_ratio",
    "rvol",
    "range_atr",
    "slot",
)
DAILY = (
    "atr20",
    "hist_rv",
    "ret5d",
    "ret20d",
    "ret60d",
    "dist_sma50",
    "dist_sma200",
    "dist_high250",
    "rsi14",
    "vol_trend",
    "log_turnover20",
    "log_price",
)
STOCK_FEATURES = INTRADAY + DAILY
RELATIVE = ("rel_day", "rel_20d")
MARKET = ("nifty_day", "nifty_ret15", "vix", "vix_chg", "nifty_ret20d", "nifty_sma50")
FEATURES = STOCK_FEATURES + RELATIVE + MARKET
PERCENT = frozenset(
    {
        "ret5",
        "ret15",
        "ret60",
        "ret_open",
        "gap",
        "vwap_dev",
        "dist_high",
        "rv20",
        "atr20",
        "hist_rv",
        "ret5d",
        "ret20d",
        "ret60d",
        "dist_sma50",
        "dist_sma200",
        "dist_high250",
        "rel_day",
        "rel_20d",
        "nifty_day",
        "nifty_ret15",
        "vix_chg",
        "nifty_ret20d",
        "nifty_sma50",
    }
)
# Long look-backs are missing early in a listing; the model accepts them as missing.
OPTIONAL = frozenset(
    {"ret60", "ret60d", "dist_sma50", "dist_sma200", "dist_high250", "nifty_sma50"}
)
BUY_SIDE, SELL_SIDE = side_costs("stock")


def as_date(epoch: int) -> date:
    return date.fromordinal(int(epoch) + EPOCH)


def decision_time(epoch: int, slot: int) -> datetime:
    minute = OPEN_MINUTE + int(DECISIONS[slot]) + 1
    return datetime.combine(
        as_date(epoch), time(minute // 60, minute % 60), INDIA_TIMEZONE
    )


def period_mask(days: np.ndarray, first: str, last: str) -> np.ndarray:
    return (days >= epoch_day(first)) & (days <= epoch_day(last))


def wilder_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI at each daily row from that row's close and earlier ones only."""
    rsi = np.full(len(closes), np.nan)
    change = np.diff(closes)
    if len(change) < period:
        return rsi
    gain, loss = np.clip(change, 0, None), np.clip(-change, 0, None)
    average_gain, average_loss = gain[:period].mean(), loss[:period].mean()
    for i in range(period, len(closes)):
        if i > period:
            average_gain = (average_gain * (period - 1) + gain[i - 1]) / period
            average_loss = (average_loss * (period - 1) + loss[i - 1]) / period
        rsi[i] = (
            100.0
            if average_loss == 0
            else 100 - 100 / (1 + average_gain / average_loss)
        )
    return rsi


def session_features(
    prices: np.ndarray, decisions: np.ndarray = DECISIONS
) -> dict[str, np.ndarray | float]:
    """Same-session inputs at each decision close; no later minute is read."""
    open_, high, low, close, volume = prices.T
    squares = np.concatenate(([0.0], np.cumsum(np.diff(np.log(close)) ** 2)))
    d = decisions
    cumulative = np.cumsum(volume)
    top, bottom = np.maximum.accumulate(high), np.minimum.accumulate(low)
    span = top[d] - bottom[d]
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.cumsum((high + low + close) / 3 * volume) / cumulative
        position = np.where(span > 0, (close[d] - bottom[d]) / span, np.nan)
    return {
        "ret5": close[d] / close[d - 5] - 1,
        "ret15": close[d] / close[d - 15] - 1,
        "ret60": np.where(d >= 60, close[d] / close[np.maximum(d - 60, 0)] - 1, np.nan),
        "ret_open": close[d] / open_[0] - 1,
        "vwap_dev": close[d] / vwap[d] - 1,
        "dist_high": close[d] / top[d] - 1,
        "range_pos": position,
        "rv20": np.sqrt(squares[d] - squares[d - 20]),
        "range": span / close[d],
        "cumvol": cumulative[d],
        "close": close[d],
        "open": float(open_[0]),
        "day_rv": math.sqrt(20 * squares[LAST] / LAST),
    }


def target_outcomes(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Buy price in paise (open of the minute after each decision) and the first
    minute after the decision, within 120, whose high reaches +1% of it."""
    paise = np.rint(prices[:, :4] * 100).astype(np.int64)
    entry = paise[DECISIONS + 1, 0]
    highs = sliding_window_view(paise[:, 1], HORIZON)[DECISIONS + 1]
    touched = highs * 10_000 >= entry[:, None] * (10_000 + TARGET_BPS)
    first = np.where(touched.any(axis=1), touched.argmax(axis=1) + 1, NONE)
    return entry, first.astype(np.int16)


def daily_columns(
    daily: np.ndarray,
    position: int,
    rsi: np.ndarray,
    day_rv: list[float],
    volumes: list[np.ndarray],
    part: dict,
    slots: int = SLOTS,
) -> np.ndarray:
    """Stock features at every decision of one session; daily rows before it only."""
    window = daily[position - LOOKBACK : position]
    prior = daily[position - LOOKBACK - 1 : position - 1, 3]
    previous = window[-1, 3]
    true_range = np.maximum(window[:, 1], prior) - np.minimum(window[:, 2], prior)
    atr = float(np.mean(true_range) / previous)
    history = float(np.median(day_rv[-LOOKBACK:]))
    closes = daily[:position, 3]

    def back(k: int) -> float:
        return closes[-1] / closes[-1 - k] - 1 if len(closes) > k else np.nan

    def above_average(k: int) -> float:
        return closes[-1] / closes[-k:].mean() - 1 if len(closes) >= k else np.nan

    high250 = daily[position - 250 : position, 1].max() if position >= 250 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        columns = {
            "ret5": part["ret5"],
            "ret15": part["ret15"],
            "ret60": part["ret60"],
            "ret_open": part["ret_open"],
            "gap": part["open"] / previous - 1,
            "vwap_dev": part["vwap_dev"],
            "dist_high": part["dist_high"],
            "range_pos": part["range_pos"],
            "rv20": part["rv20"],
            "rv_ratio": part["rv20"] / history,
            "rvol": part["cumvol"] / np.mean(volumes[-LOOKBACK:], axis=0),
            "range_atr": part["range"] / atr,
            "slot": np.arange(slots),
            "atr20": atr,
            "hist_rv": history,
            "ret5d": back(5),
            "ret20d": back(20),
            "ret60d": back(60),
            "dist_sma50": above_average(50),
            "dist_sma200": above_average(200),
            "dist_high250": closes[-1] / high250 - 1,
            "rsi14": rsi[position - 1],
            "vol_trend": np.mean(window[-5:, 4]) / np.mean(window[:, 4]),
            "log_turnover20": np.log10(np.mean(window[:, 3] * window[:, 4])),
            "log_price": np.log10(part["close"]),
        }
    return np.column_stack(
        [
            np.broadcast_to(np.asarray(columns[name], dtype=float), (slots,))
            for name in STOCK_FEATURES
        ]
    )


def stock_rows(task: tuple[str, Path, np.ndarray]) -> dict:
    symbol, candles, calendar = task
    store = CandleStore(candles)
    sessions = read_sessions(store.verified_path(f"NSE_{symbol}.csv"))
    daily_days, daily = read_daily(store.verified_path(f"daily/NSE_{symbol}.csv"))
    rsi = wilder_rsi(daily[:, 3])
    days = sorted(sessions)
    features = np.full((len(days), SLOTS, len(STOCK_FEATURES)), np.nan, np.float32)
    entry = np.zeros((len(days), SLOTS), np.int64)
    first = np.full((len(days), SLOTS), NONE, np.int16)
    day_rv: list[float] = []
    volumes: list[np.ndarray] = []
    for row, day in enumerate(days):
        prices = sessions[day]
        entry[row], first[row] = target_outcomes(prices)
        part = session_features(prices)
        position = int(np.searchsorted(daily_days, day))
        market = int(np.searchsorted(calendar, day))
        if (
            len(day_rv) >= LOOKBACK
            and position > LOOKBACK
            and market > 0
            and daily_days[position - 1] == calendar[market - 1]
        ):
            features[row] = daily_columns(daily, position, rsi, day_rv, volumes, part)
        day_rv.append(part["day_rv"])
        volumes.append(part["cumvol"])
    return {
        "symbol": symbol,
        "days": np.asarray(days, dtype=np.int64),
        "features": features,
        "entry": entry,
        "first": first,
    }


def market_context(
    candles: Path, decisions: np.ndarray = DECISIONS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sessions with complete NIFTY and VIX minutes, and their market features."""
    store = CandleStore(candles)
    calendar, nifty_daily = read_daily(store.verified_path("daily/NSE_NIFTY.csv"))
    vix_days, vix_daily = read_daily(store.verified_path("daily/NSE_INDIAVIX.csv"))
    nifty = read_sessions(store.verified_path("NSE_NIFTY.csv"))
    vix = read_sessions(store.verified_path("NSE_INDIAVIX.csv"))
    slots = len(decisions)
    days, values = [], []
    for day in sorted(set(nifty) & set(vix)):
        position = int(np.searchsorted(calendar, day))
        vix_position = int(np.searchsorted(vix_days, day))
        if position == 0 or vix_position == 0:
            continue
        closes = nifty_daily[:position, 3]
        close = nifty[day][decisions, 3]
        level = vix[day][decisions, 3]
        trend = closes[-1] / closes[-21] - 1 if len(closes) > 20 else np.nan
        average = closes[-1] / closes[-50:].mean() - 1 if len(closes) >= 50 else np.nan
        values.append(
            np.column_stack(
                (
                    close / closes[-1] - 1,
                    close / nifty[day][decisions - 15, 3] - 1,
                    level,
                    level / vix_daily[vix_position - 1, 3] - 1,
                    np.full(slots, trend),
                    np.full(slots, average),
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
    calendar, days, market = market_context(candles)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(stock_rows, [(s, candles, calendar) for s in stocks]))
    shape = (len(days), SLOTS, len(stocks))
    data = {
        "days": days,
        "stocks": stocks,
        "calendar": calendar,
        "features": np.full((*shape, len(FEATURES)), np.nan, np.float32),
        "entry": np.zeros(shape, np.int64),
        "first": np.full(shape, NONE, np.int16),
    }
    index = {int(day): row for row, day in enumerate(days)}
    for column, result in enumerate(results):
        rows = np.asarray([index.get(int(day), -1) for day in result["days"]])
        keep = rows >= 0
        data["features"][rows[keep], :, column, : len(STOCK_FEATURES)] = result[
            "features"
        ][keep]
        data["entry"][rows[keep], :, column] = result["entry"][keep]
        data["first"][rows[keep], :, column] = result["first"][keep]
    features = data["features"]
    col = {name: i for i, name in enumerate(FEATURES)}
    for i, name in enumerate(MARKET):
        features[..., col[name]] = market[:, :, None, i]
    day_return = (1 + features[..., col["gap"]]) * (1 + features[..., col["ret_open"]])
    features[..., col["rel_day"]] = day_return - 1 - features[..., col["nifty_day"]]
    features[..., col["rel_20d"]] = (
        features[..., col["ret20d"]] - features[..., col["nifty_ret20d"]]
    )
    available = data["entry"] > 0
    for name in FEATURES:
        if name not in OPTIONAL:
            available &= np.isfinite(features[..., col[name]])
    data["available"] = available
    return data


def flagged_days(candles: Path, audit: Path | None) -> dict[str, set[int]]:
    """Stock-days whose prices are unreliable (excluded from scanning and holding)."""
    flags: dict[str, set[int]] = {}
    for symbol, reasons in json.loads(
        (candles / "exclusions.json").read_text("utf-8")
    ).items():
        for reason in ("coarse_vendor_rounding", "imprecise_fit", "structural_break"):
            for day in reasons.get(reason, []):
                flags.setdefault(symbol, set()).add(epoch_day(day))
    if audit is not None:
        audited = json.loads(audit.read_text("utf-8"))["instrument_days"]
        for symbol, by_day in audited.items():
            for day, reasons in by_day.items():
                if "illiquid_session" in reasons:
                    flags.setdefault(symbol, set()).add(epoch_day(day))
    return flags


def prune(data: dict, limit: float = 0.9) -> tuple[list[str], list[dict]]:
    """Keep features in declared order unless one already kept ranks the same way
    (|Spearman| > limit) on fit-period rows."""
    days = np.broadcast_to(data["days"][:, None, None], data["available"].shape)
    flat = data["features"].reshape(-1, len(FEATURES))
    rows = np.flatnonzero(
        (data["available"] & period_mask(days, *FIT)).reshape(-1)
        & np.isfinite(flat).all(axis=1)
    )
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(rows, size=min(300_000, len(rows)), replace=False))
    ranks = flat[rows].argsort(axis=0).argsort(axis=0).astype(np.float64)
    rho = np.corrcoef(ranks, rowvar=False)
    kept: list[str] = []
    dropped: list[dict] = []
    for i, name in enumerate(FEATURES):
        clash = [k for k in kept if abs(rho[i, FEATURES.index(k)]) > limit]
        if clash:
            dropped.append(
                {
                    "feature": name,
                    "duplicates": clash[0],
                    "spearman": round(float(rho[i, FEATURES.index(clash[0])]), 3),
                }
            )
        else:
            kept.append(name)
    return kept, dropped


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_scanner(data: dict, kept: list[str], threads: int):
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression

    columns = [FEATURES.index(name) for name in kept]
    shape = data["available"].shape
    flat = data["features"].reshape(-1, len(FEATURES))
    labels = (data["first"] <= HORIZON).reshape(-1)
    day = np.broadcast_to(data["days"][:, None, None], shape).reshape(-1)
    available = data["available"].reshape(-1)
    rng = np.random.default_rng(SEED)

    def sample(first: str, last: str, size: int) -> np.ndarray:
        rows = np.flatnonzero(available & period_mask(day, first, last))
        return np.sort(rng.choice(rows, size=min(size, len(rows)), replace=False))

    fit_rows = sample(*FIT, 3_000_000)
    stop_rows = sample(*EARLY_STOP, 600_000)
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
        labels[fit_rows],
        eval_set=(flat[stop_rows][:, columns], labels[stop_rows]),
    )
    raw = np.full(available.shape, np.nan, np.float32)
    rows = np.flatnonzero(available)
    for start in range(0, len(rows), 2_000_000):
        chunk = rows[start : start + 2_000_000]
        raw[chunk] = model.predict_proba(flat[chunk][:, columns])[:, 1]
    calibration_rows = np.flatnonzero(available & period_mask(day, *CALIBRATION))
    platt = LogisticRegression(C=1e6).fit(
        logit(raw[calibration_rows])[:, None], labels[calibration_rows]
    )
    calibrated = np.full(available.shape, np.nan, np.float32)
    calibrated[rows] = platt.predict_proba(logit(raw[rows])[:, None])[:, 1]
    info = {
        "target": "high reaches +1% of the buy price within 120 minutes of the "
        "decision (buy = open of the next minute)",
        "features": kept,
        "trees": int(model.tree_count_),
        "fit_rows": len(fit_rows),
        "early_stop_rows": len(stop_rows),
        "calibration": {
            "method": "Platt: logistic regression on the log-odds, fitted on 2024 H2",
            "rows": len(calibration_rows),
            "slope": float(platt.coef_[0, 0]),
            "intercept": float(platt.intercept_[0]),
            "why_not_isotonic": "on 2024 H1, isotonic fitted on 2024 H2 put 2,023 "
            "rows at 0.955 that hit +1% only 79.3% of the time; Platt stayed within "
            "4 points in every bin above 0.5 (the test period was also seen while "
            "comparing the two)",
        },
        "fit_seconds": round(clock.monotonic() - started, 1),
        "importance": dict(
            sorted(
                zip(
                    kept, model.get_feature_importance().round(3).tolist(), strict=True
                ),
                key=lambda item: -item[1],
            )
        ),
    }
    return raw.reshape(shape), calibrated.reshape(shape), model, info


def scanner_metrics(
    raw: np.ndarray, calibrated: np.ndarray, data: dict, threshold: float
) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    labels = data["first"] <= HORIZON
    days = np.broadcast_to(data["days"][:, None, None], labels.shape)
    edges = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0001)
    result = {}
    for name, first, last in PERIODS:
        inside = data["available"] & period_mask(days, first, last)
        if not inside.any():
            continue
        y, p, r = labels[inside], calibrated[inside], raw[inside]
        base = float(y.mean())
        bins = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            chosen = (p >= low) & (p < high)
            bins.append(
                {
                    "from": low,
                    "to": min(high, 1.0),
                    "rows": int(chosen.sum()),
                    "mean_probability": float(p[chosen].mean())
                    if chosen.any()
                    else None,
                    "hit_rate": float(y[chosen].mean()) if chosen.any() else None,
                }
            )
        above = p > threshold
        sessions = np.unique(days[inside])
        per_day = np.bincount(
            np.searchsorted(sessions, days[inside][above]), minlength=len(sessions)
        )
        day_index, _, stock_index = np.nonzero(inside & (calibrated > threshold))
        result[name] = {
            "rows": int(inside.sum()),
            "base_rate": base,
            "auc": float(roc_auc_score(y, p)),
            "log_loss_raw": float(log_loss(y, np.clip(r, 1e-6, 1 - 1e-6))),
            "log_loss_calibrated": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "log_loss_base_rate": float(log_loss(y, np.full(len(y), base))),
            "brier_calibrated": float(brier_score_loss(y, p)),
            "max_probability": float(p.max()),
            "reliability": bins,
            "candidates": {
                "rows": int(above.sum()),
                "share_of_rows": float(above.mean()),
                "hit_rate": float(y[above].mean()) if above.any() else None,
                "mean_probability": float(p[above].mean()) if above.any() else None,
                "stock_days": int(np.unique(day_index * 1000 + stock_index).size),
                "days_with_candidates": int((per_day > 0).sum()),
                "sessions": len(per_day),
                "per_day_median": float(np.median(per_day)),
                "per_day_max": int(per_day.max()),
            },
        }
    return result


def read_path(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every regular-session minute (09:15 to 15:29) as day, minute and paise OHLC."""
    days: list[int] = []
    minutes: list[int] = []
    prices: list[tuple[float, float, float, float]] = []
    seen: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != HEADER:
            raise ValueError(f"Unexpected candle header in {path.name}")
        for start, open_, high, low, close, _volume in reader:
            minute = int(start[11:13]) * 60 + int(start[14:16]) - OPEN_MINUTE
            if 0 <= minute < SESSION_MINUTES:
                label = start[:10]
                if label not in seen:
                    seen[label] = epoch_day(label)
                days.append(seen[label])
                minutes.append(minute)
                prices.append((float(open_), float(high), float(low), float(close)))
    day_array = np.asarray(days, dtype=np.int64)
    minute_array = np.asarray(minutes, dtype=np.int64)
    order = np.lexsort((minute_array, day_array))
    ohlc = np.rint(np.asarray(prices)[order] * 100).astype(np.int64)
    return day_array[order], minute_array[order], ohlc


def first_exit(
    ohlc: np.ndarray,
    start: int,
    entry: int,
    target_bps: int = TARGET_BPS,
    stop_bps: int = STOP_BPS,
) -> tuple[int, float, str]:
    """The first +target / -stop exit from buy row ``start`` (prices in paise).

    Fills at the target or stop price, or at a minute's open when the price opens
    past either (overnight gaps included). A minute that reaches both is a stop,
    because the order inside one minute is unknown. Returns the exit row, the fill
    in rupees and the reason; ``open`` marks a trade still open when data ends."""
    up = entry * (10_000 + target_bps)
    down = entry * (10_000 - stop_bps)
    count = len(ohlc)
    row, low, size = -1, start, 256
    while low < count:
        high = min(count, low + size)
        touched = np.flatnonzero(
            (ohlc[low:high, 1] * 10_000 >= up) | (ohlc[low:high, 2] * 10_000 <= down)
        )
        if touched.size:
            row = low + int(touched[0])
            break
        low, size = high, size * 4
    if row < 0:
        return count - 1, ohlc[count - 1, 3] / 100, "open"
    opened = ohlc[row, 0] * 10_000
    if opened <= down:
        return row, ohlc[row, 0] / 100, "sl_gap"
    if opened >= up:
        return row, ohlc[row, 0] / 100, "tp_gap"
    reached_target = ohlc[row, 1] * 10_000 >= up
    reached_stop = ohlc[row, 2] * 10_000 <= down
    stop = down / 1_000_000
    if reached_target and reached_stop:
        return row, stop, "sl_same_minute"
    if reached_target:
        return row, up / 1_000_000, "tp"
    return row, stop, "sl"


def charges(buy_value: float, sell_value: float, same_session: bool) -> float:
    """Zerodha charges in rupees; a CNC buy sold in the same session is intraday."""
    if same_session:
        notional = Decimal(str(round(buy_value, 2)))
        return float(ZERODHA_INTRADAY_EQUITY.round_trip(notional).total)
    return BUY_SIDE * buy_value + SELL_SIDE * sell_value + DP_INR


def complete_holding(
    entry_day: int,
    exit_day: int,
    held: int,
    calendar: np.ndarray,
    flagged: Iterable[int],
) -> bool:
    """True when the stock has every market session of the holding and none of
    them is flagged for unreliable prices."""
    market_sessions = int(
        np.searchsorted(calendar, exit_day, side="right")
        - np.searchsorted(calendar, entry_day, side="right")
    )
    return held >= market_sessions and not any(
        entry_day < flag <= exit_day for flag in flagged
    )


@dataclass(frozen=True)
class Trade:
    day: int  # row in data["days"]
    slot: int
    stock: int
    symbol: str
    entry_day: int  # epoch days
    entry_minute: int
    exit_day: int
    exit_minute: int
    entry_price: float
    exit_price: float
    reason: str
    sessions: int
    quantity: int
    charges: float
    pnl: float
    net_return: float
    reliable: bool


def simulate_stock(task: tuple) -> list[Trade]:
    """Outcomes of every signal for one stock, each as if it were the only trade."""
    symbol, candles, signals, flagged, calendar = task
    days, minutes, ohlc = read_path(
        CandleStore(candles).verified_path(f"NSE_{symbol}.csv")
    )
    keys = days * 1000 + minutes
    sessions = np.unique(days)
    trades = []
    for day_row, slot, stock, epoch, entry in signals:
        minute = int(DECISIONS[slot]) + 1
        start = int(np.searchsorted(keys, epoch * 1000 + minute))
        if start >= len(keys) or keys[start] != epoch * 1000 + minute:
            raise ValueError(f"{symbol} has no buy minute on {as_date(epoch)}")
        if ohlc[start, 0] != entry:
            raise ValueError(f"{symbol} buy price differs on {as_date(epoch)}")
        row, price, reason = first_exit(ohlc, start, int(entry))
        exit_day = int(days[row])
        held = int(
            np.searchsorted(sessions, exit_day) - np.searchsorted(sessions, epoch)
        )
        reliable = reason != "open" and complete_holding(
            int(epoch), exit_day, held, calendar, flagged
        )
        quantity = int(NOTIONAL_INR * 100 // entry)
        buy_value = quantity * entry / 100
        sell_value = quantity * price
        cost = charges(buy_value, sell_value, held == 0)
        pnl = sell_value - buy_value - cost
        trades.append(
            Trade(
                day_row,
                slot,
                stock,
                symbol,
                int(epoch),
                minute,
                exit_day,
                int(minutes[row]),
                entry / 100,
                float(price),
                reason,
                held,
                quantity,
                cost,
                pnl,
                pnl / buy_value,
                reliable,
            )
        )
    return trades


def simulate(
    keys: Iterable[tuple[int, int, int]],
    data: dict,
    candles: Path,
    flags: dict[str, set[int]],
    workers: int,
) -> dict[tuple[int, int, int], Trade]:
    by_stock: dict[int, list] = {}
    for day, slot, stock in sorted(set(keys)):
        entry = int(data["entry"][day, slot, stock])
        if entry <= 0 or NOTIONAL_INR * 100 // entry == 0:
            continue
        by_stock.setdefault(stock, []).append(
            (day, slot, stock, int(data["days"][day]), entry)
        )
    tasks = [
        (
            data["stocks"][stock],
            candles,
            signals,
            sorted(flags.get(data["stocks"][stock], set())),
            data["calendar"],
        )
        for stock, signals in by_stock.items()
    ]
    outcomes: dict[tuple[int, int, int], Trade] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for trades in pool.map(simulate_stock, tasks):
            for trade in trades:
                outcomes[(trade.day, trade.slot, trade.stock)] = trade
    return outcomes


def execute(
    signals: Iterable[tuple[int, int, int]], outcomes: dict[tuple[int, int, int], Trade]
) -> list[Trade]:
    """First buy per stock per day, in time order, never while that stock is held."""
    taken: list[Trade] = []
    held_until: dict[int, tuple[int, int]] = {}
    bought: set[tuple[int, int]] = set()
    for key in sorted(set(signals)):
        trade = outcomes.get(key)
        day, _, stock = key
        if trade is None or (day, stock) in bought:
            continue
        if held_until.get(stock, (-1, -1)) >= (trade.entry_day, trade.entry_minute):
            continue
        bought.add((day, stock))
        held_until[stock] = (trade.exit_day, trade.exit_minute)
        taken.append(trade)
    return taken


def summarize(trades: list[Trade]) -> dict:
    valid = [t for t in trades if t.reliable]
    result: dict = {
        "excluded_open_at_data_end": sum(t.reason == "open" for t in trades),
        "excluded_unreliable_prices": sum(
            not t.reliable and t.reason != "open" for t in trades
        ),
    }
    if not valid:
        return {"trades": 0, **result}
    net = np.array([t.net_return for t in valid])
    pnl = np.array([t.pnl for t in valid])
    entry_days = np.array([t.entry_day for t in valid])
    unique_days, inverse = np.unique(entry_days, return_inverse=True)
    day_sum = np.bincount(inverse, weights=net)
    day_count = np.bincount(inverse)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(unique_days), size=(2000, len(unique_days)))
    means = day_sum[picks].sum(axis=1) / day_count[picks].sum(axis=1)
    events = sorted(
        [((t.entry_day, t.entry_minute), 1) for t in trades]
        + [((t.exit_day, t.exit_minute), -1) for t in trades]
    )
    running = peak = 0
    for _, change in events:
        running += change
        peak = max(peak, running)
    held = np.array([t.sessions for t in valid])
    reasons: dict[str, int] = {}
    for trade in valid:
        reasons[trade.reason] = reasons.get(trade.reason, 0) + 1
    return {
        "trades": len(valid),
        "trading_days": len(unique_days),
        "win_rate": float((pnl > 0).mean()),
        "target_first_rate": float(np.mean([t.reason.startswith("tp") for t in valid])),
        "exit_reasons": dict(sorted(reasons.items())),
        "mean_net_return_pct": 100 * float(net.mean()),
        "mean_net_return_pct_ci95": [
            100 * float(np.percentile(means, 2.5)),
            100 * float(np.percentile(means, 97.5)),
        ],
        "share_of_resamples_above_zero": float((means > 0).mean()),
        "mean_pnl_inr": float(pnl.mean()),
        "total_pnl_inr": float(pnl.sum()),
        "mean_charges_inr": float(np.mean([t.charges for t in valid])),
        "same_session_exit_share": float((held == 0).mean()),
        "sessions_held_median": float(np.median(held)),
        "sessions_held_p90": float(np.percentile(held, 90)),
        "sessions_held_max": int(held.max()),
        "max_open_positions": peak,
        **result,
    }


def by_period(trades: list[Trade]) -> dict:
    return {
        name: summarize(
            [t for t in trades if epoch_day(first) <= t.entry_day <= epoch_day(last)]
        )
        for name, first, last in PERIODS
    }


def auc_band(scores: np.ndarray, wins: np.ndarray) -> dict | None:
    """ROC AUC of one input for winning trades, with a Hanley-McNeil 95% band."""
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(scores)
    outcome = wins[finite]
    positives, negatives = int(outcome.sum()), int((~outcome).sum())
    if positives < 15 or negatives < 15:
        return None
    area = float(roc_auc_score(outcome, scores[finite]))
    q1, q2 = area / (2 - area), 2 * area * area / (1 + area)
    error = math.sqrt(
        (
            area * (1 - area)
            + (positives - 1) * (q1 - area * area)
            + (negatives - 1) * (q2 - area * area)
        )
        / (positives * negatives)
    )
    return {
        "auc": round(area, 4),
        "ci95": [round(area - 1.96 * error, 4), round(area + 1.96 * error, 4)],
        "trades": positives + negatives,
    }


def brief(trades: list[Trade]) -> dict:
    full = summarize(trades)
    return {
        key: full.get(key)
        for key in (
            "trades",
            "win_rate",
            "mean_net_return_pct",
            "mean_net_return_pct_ci95",
            "total_pnl_inr",
        )
    }


def keys_of(mask: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(d), int(s), int(k)) for d, s, k in zip(*np.nonzero(mask), strict=True)]


def in_period(keys: list[tuple[int, int, int]], data: dict, first: str, last: str):
    low, high = epoch_day(first), epoch_day(last)
    return [key for key in keys if low <= data["days"][key[0]] <= high]


def random_entries(candidates: np.ndarray, data: dict) -> list[tuple[int, int, int]]:
    """For each day, as many random available stock-slots as that day's candidates."""
    rng = np.random.default_rng(SEED)
    keys = []
    for day in np.flatnonzero(candidates.any(axis=(1, 2))):
        pool = np.flatnonzero(data["available"][day].reshape(-1))
        count = min(int(candidates[day].sum()), len(pool))
        for flat in rng.choice(pool, size=count, replace=False):
            keys.append(
                (
                    int(day),
                    int(flat // len(data["stocks"])),
                    int(flat % len(data["stocks"])),
                )
            )
    return keys


def oi_component(trades: list[Trade], oi_raw: dict) -> dict:
    groups: dict[str, list[Trade]] = {}
    for trade in trades:
        values = oi_raw.get((trade.symbol, as_date(trade.entry_day))) or {}
        groups.setdefault(values.get("buildup_1d") or "no_data", []).append(trade)
    reliable = [t for t in trades if t.reliable]
    wins = np.array([t.pnl > 0 for t in reliable])
    signals = {}
    for name in OI_FEATURES:
        values = np.array(
            [
                (oi_raw.get((t.symbol, as_date(t.entry_day))) or {}).get(name)
                for t in reliable
            ],
            dtype=float,
        )
        signals[name] = auc_band(values, wins)
    return {
        "by_previous_session_buildup": {k: brief(v) for k, v in sorted(groups.items())},
        "auc_for_winning_trades": signals,
    }


def oi_rule(keys: list[tuple[int, int, int]], data: dict, oi_raw: dict) -> list:
    """BUY when the previous session showed long buildup or short covering."""
    return [
        key
        for key in keys
        if (
            oi_raw.get((data["stocks"][key[2]], as_date(data["days"][key[0]]))) or {}
        ).get("buildup_1d")
        in ("long_buildup", "short_covering")
    ]


def since_previous_close(data: dict, day: int) -> datetime:
    return decision_time(int(data["days"][day - 1]), 0).replace(hour=15, minute=30)


def news_component(trades: list[Trade], news: NewsStore, data: dict) -> dict:
    groups: dict[str, list[Trade]] = {"material_filing": [], "none_or_routine": []}
    categories: dict[str, list[Trade]] = {}
    for trade in trades:
        if trade.entry_day < epoch_day(NEWS_FROM):
            continue
        context = event_context(
            news,
            trade.symbol,
            decision_time(trade.entry_day, trade.slot),
            since_previous_close(data, trade.day),
        )
        material = context["material_since_prev_close"] > 0
        groups["material_filing" if material else "none_or_routine"].append(trade)
        for name in context["categories"]:
            if name != "routine":
                categories.setdefault(name, []).append(trade)
    return {
        "filings_since_previous_close": {k: brief(v) for k, v in groups.items()},
        "by_category": {
            k: brief(v) for k, v in sorted(categories.items()) if len(v) >= 10
        },
    }


def feature_component(trades: list[Trade], data: dict, kept: list[str]) -> dict:
    reliable = [t for t in trades if t.reliable]
    wins = np.array([t.pnl > 0 for t in reliable])
    result = {}
    for name in kept:
        column = FEATURES.index(name)
        values = np.array(
            [data["features"][t.day, t.slot, t.stock, column] for t in reliable],
            dtype=float,
        )
        result[name] = auc_band(values, wins)
    return dict(
        sorted(
            result.items(),
            key=lambda item: -abs((item[1] or {"auc": 0.5})["auc"] - 0.5),
        )
    )


def fit_features_decider(
    data: dict,
    kept: list[str],
    candles: Path,
    flags: dict[str, set[int]],
    workers: int,
):
    """CatBoost P(+1% before -1%) from the indicator features alone, fitted on
    random stock-slots (2020-2023, early stopping on 2024 H1)."""
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(SEED + 1)
    days = np.broadcast_to(data["days"][:, None, None], data["available"].shape)
    width = len(data["stocks"])

    def sample(first: str, last: str, size: int) -> list[tuple[int, int, int]]:
        flat = np.flatnonzero(
            (data["available"] & period_mask(days, first, last)).reshape(-1)
        )
        chosen = np.sort(rng.choice(flat, size=min(size, len(flat)), replace=False))
        day, rest = np.divmod(chosen, SLOTS * width)
        slot, stock = np.divmod(rest, width)
        return list(zip(day.tolist(), slot.tolist(), stock.tolist(), strict=True))

    fit_keys, stop_keys = sample(*FIT, 300_000), sample(*EARLY_STOP, 60_000)
    outcomes = simulate(fit_keys + stop_keys, data, candles, flags, workers)
    columns = [FEATURES.index(name) for name in kept]

    def matrix(keys):
        usable = [k for k in keys if k in outcomes and outcomes[k].reliable]
        x = np.array([data["features"][d, s, k][columns] for d, s, k in usable])
        return x, np.array([outcomes[k].pnl > 0 for k in usable])

    x_fit, y_fit = matrix(fit_keys)
    x_stop, y_stop = matrix(stop_keys)
    model = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=1500,
        l2_leaf_reg=20,
        loss_function="Logloss",
        eval_metric="Logloss",
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        random_seed=SEED,
        thread_count=workers,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_fit, y_fit, eval_set=(x_stop, y_stop))
    return model, {
        "target": "+1% reached before -1% (trade P&L after charges > 0)",
        "fit_trades": len(y_fit),
        "fit_win_rate": float(y_fit.mean()),
        "early_stop_trades": len(y_stop),
        "early_stop_auc": auc_band(model.predict_proba(x_stop)[:, 1], y_stop),
        "trees": int(model.tree_count_),
    }


def choose_threshold(
    probabilities: dict[tuple[int, int, int], float],
    keys: list[tuple[int, int, int]],
    outcomes: dict,
) -> dict:
    """The P(win) cut with the best mean net return on validation candidates."""
    best: dict = {"threshold": None, "mean_net_return_pct": None}
    for cut in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
        stats = brief(execute([k for k in keys if probabilities[k] >= cut], outcomes))
        if stats["trades"] >= 30 and (
            best["threshold"] is None
            or stats["mean_net_return_pct"] > best["mean_net_return_pct"]
        ):
            best = {"threshold": cut, **stats}
    return best


def market_block(data: dict, day: int, slot: int) -> dict[str, float | None]:
    values = data["features"][day, slot]
    stock = int(np.flatnonzero(data["available"][day, slot])[0])
    block: dict[str, float | None] = {}
    for name in MARKET:
        value = float(values[stock, FEATURES.index(name)])
        key = (
            "india_vix" if name == "vix" else f"{name}_pct" if name in PERCENT else name
        )
        scaled = 100 * value if name in PERCENT else value
        block[key] = round(scaled, 3) if math.isfinite(scaled) else None
    return block


def build_request(
    data: dict,
    calibrated: np.ndarray,
    kept: list[str],
    oi_raw: dict,
    news: NewsStore | None,
    day: int,
    slot: int,
    stocks: list[int],
) -> DecisionRequest:
    epoch = int(data["days"][day])
    as_of = decision_time(epoch, slot)
    since = since_previous_close(data, day)
    names = [name for name in kept if name not in MARKET]
    candidates = []
    for stock in sorted(stocks, key=lambda s: -float(calibrated[day, slot, s])):
        symbol = data["stocks"][stock]
        values = {
            name: float(data["features"][day, slot, stock, FEATURES.index(name)])
            for name in names
        }
        news_context = (
            event_context(news, symbol, as_of, since)
            if news is not None and epoch >= epoch_day(NEWS_FROM)
            else None
        )
        candidates.append(
            candidate(
                symbol,
                delivery_block(float(calibrated[day, slot, stock]), values, PERCENT),
                oi=oi_raw.get((symbol, as_date(epoch))),
                news=news_context,
            )
        )
    return DecisionRequest(as_of, market_block(data, day, slot), tuple(candidates))


def evaluate_ai(
    args: argparse.Namespace,
    model: str,
    effort: str | None,
    keys: list[tuple[int, int, int]],
    data: dict,
    calibrated: np.ndarray,
    kept: list[str],
    oi_raw: dict,
    news: NewsStore | None,
    output: Path,
) -> tuple[dict, list[tuple[int, int, int]]]:
    """gpt BUY / NO_BUY on every post-cutoff round with candidates, journaled."""
    from concurrent.futures import as_completed

    from ai_trader.ai.decision import OpenAIDecider
    from ai_trader.ai.journal import DecisionJournal
    from ai_trader.ai.openai_client import OpenAIResponses
    from ai_trader.config import ConfigurationError, load_openai_settings

    try:
        settings = load_openai_settings()
    except ConfigurationError as error:
        return {"status": f"skipped: {error}"}, []
    effort = effort or settings.reasoning_effort
    rounds: dict[tuple[int, int], list[int]] = {}
    for day, slot, stock in keys:
        rounds.setdefault((day, slot), []).append(stock)
    requests = []
    for (day, slot), stocks in sorted(rounds.items()):
        request = build_request(data, calibrated, kept, oi_raw, news, day, slot, stocks)
        requests.append((f"{request.as_of:%Y%m%d-%H%M}", request, day, slot))
    journal = DecisionJournal(output / f"openai_journal_{model}_{effort}.sqlite")
    results = {}
    pending = []
    for cycle_id, request, _, _ in requests:
        done = journal.completed(cycle_id, request_json(request), DELIVERY.version)
        if done is None:
            pending.append((cycle_id, request))
        else:
            results[cycle_id] = done
    reused = len(results)
    spent = [0.0]
    client = OpenAIResponses(
        settings.api_key.get_secret_value(),
        timeout_seconds=settings.timeout_seconds,
        attempts=6,
    )
    decider = OpenAIDecider(
        client,
        model,
        reasoning_effort=effort,
        web_search=False,  # never in replay: search would see the future
        budget_left=lambda: args.openai_budget_usd - spent[0],
        policy=DELIVERY,
    )
    with ThreadPoolExecutor(max_workers=args.openai_workers) as pool:
        futures = {
            pool.submit(decider.decide, request): (cycle_id, request)
            for cycle_id, request in pending
        }
        for future in as_completed(futures):
            cycle_id, request = futures[future]
            result = future.result()
            spent[0] += result.cost_usd
            journal.record(cycle_id, request.as_of, "openai", result)
            results[cycle_id] = result
    journal.close()
    buys, statuses, verdicts, usage, latency = [], {}, {}, [], []
    for cycle_id, request, day, slot in requests:
        result = results[cycle_id]
        statuses[result.status[:40]] = statuses.get(result.status[:40], 0) + 1
        if result.usage:
            usage.append(result.usage)
            latency.append(result.latency_seconds)
        by_symbol = {data["stocks"][s]: s for s in rounds[(day, slot)]}
        for verdict in result.verdicts:
            verdicts[verdict.verdict] = verdicts.get(verdict.verdict, 0) + 1
            if verdict.verdict == "BUY":
                buys.append((day, slot, by_symbol[verdict.symbol]))
    return {
        "status": "completed",
        "model": model,
        "reasoning_effort": effort,
        "prompt_version": DELIVERY.version,
        "rounds": len(requests),
        "candidate_decisions": len(keys),
        "max_candidates_in_a_round": max((len(v) for v in rounds.values()), default=0),
        "statuses": statuses,
        "verdicts": verdicts,
        "reused_rounds": reused,
        "cost_usd": float(sum(r.cost_usd for r in results.values())),
        "cost_usd_this_run": spent[0],
        "latency_seconds_mean": float(np.mean(latency)) if latency else None,
        "tokens": {
            "input": int(sum(u.input_tokens for u in usage)),
            "output": int(sum(u.output_tokens for u in usage)),
            "reasoning": int(sum(u.reasoning_tokens for u in usage)),
        },
    }, buys


def random_same_rate(
    keys: list[tuple[int, int, int]],
    rate: float,
    outcomes: dict,
    observed: float | None,
    draws: int = 500,
) -> dict:
    """Buy each candidate at random with the AI's buy probability, many times."""
    rng = np.random.default_rng(SEED + 2)
    means = []
    for _ in range(draws):
        picks = rng.random(len(keys)) < rate
        stats = summarize(
            execute([k for k, pick in zip(keys, picks, strict=True) if pick], outcomes)
        )
        if stats["trades"]:
            means.append(stats["mean_net_return_pct"])
    if not means:
        return {"draws": 0}
    return {
        "draws": len(means),
        "mean_net_return_pct_median": float(np.median(means)),
        "mean_net_return_pct_p05": float(np.percentile(means, 5)),
        "mean_net_return_pct_p95": float(np.percentile(means, 95)),
        "share_of_draws_below_ai": float(np.mean(np.array(means) < observed))
        if observed is not None
        else None,
    }


def write_trades(path: Path, trades: list[Trade]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(Trade.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=["decision_time", *fields])
        writer.writeheader()
        for trade in trades:
            row = asdict(trade)
            row["entry_day"] = as_date(trade.entry_day).isoformat()
            row["exit_day"] = as_date(trade.exit_day).isoformat()
            writer.writerow(
                {
                    "decision_time": decision_time(
                        trade.entry_day, trade.slot
                    ).isoformat(),
                    **row,
                }
            )


def run(args: argparse.Namespace) -> dict:
    output: Path = args.output
    if output.exists() and not args.resume:
        raise ValueError("Output exists; keep earlier studies (or --resume)")
    output.mkdir(parents=True, exist_ok=True)
    started = clock.monotonic()
    data = assemble(args.candles, args.workers)
    excluded = apply_exclusions(data, args.candles, args.audit)
    flags = flagged_days(args.candles, args.audit)
    kept, dropped = prune(data)
    raw, calibrated, model, scanner_info = fit_scanner(data, kept, args.workers)
    model.save_model(str(output / "scanner_up1pct_2h.cbm"))
    scanner = scanner_metrics(raw, calibrated, data, args.threshold)
    days = np.broadcast_to(data["days"][:, None, None], data["available"].shape)
    candidates = data["available"] & (calibrated > args.threshold)
    candidates &= days >= epoch_day("2024-01-01")
    candidate_keys = keys_of(candidates)
    test_candidates = candidates & period_mask(days, *TEST)
    random_keys = random_entries(test_candidates, data)
    outcomes = simulate(
        candidate_keys + random_keys, data, args.candles, flags, args.workers
    )
    test_keys = in_period(candidate_keys, data, *TEST)
    buy_all = execute(candidate_keys, outcomes)
    oi_raw: dict = {}
    oi_info: dict = {"symbols_with_oi": 0}
    if args.oi and args.oi.exists():
        _, oi_info = load_oi(args.oi, args.candles, data)
        oi_raw = oi_info.pop("raw")
    news = NewsStore(args.news) if args.news and args.news.exists() else None
    decider, decider_info = fit_features_decider(
        data, kept, args.candles, flags, args.workers
    )
    columns = [FEATURES.index(name) for name in kept]
    p_win: dict[tuple[int, int, int], float] = {}
    if candidate_keys:
        rows = np.array(
            [data["features"][d, s, k][columns] for d, s, k in candidate_keys]
        )
        p_win = dict(
            zip(candidate_keys, decider.predict_proba(rows)[:, 1].tolist(), strict=True)
        )
    decider_info["validation_choice"] = choose_threshold(
        p_win, in_period(candidate_keys, data, "2024-01-01", "2024-12-31"), outcomes
    )
    cut = decider_info["validation_choice"]["threshold"]
    feature_buys = [k for k in candidate_keys if cut is not None and p_win[k] >= cut]
    pipelines = {
        "random_entries_no_scanner": execute(random_keys, outcomes),
        "scanner_buy_all": buy_all,
        "scanner_oi_rule": execute(oi_rule(candidate_keys, data, oi_raw), outcomes),
        "scanner_features_model": execute(feature_buys, outcomes),
    }
    test_buy_all = [t for t in buy_all if epoch_day(TEST[0]) <= t.entry_day]
    components = {
        "open_interest": oi_component(test_buy_all, oi_raw) if oi_raw else None,
        "news_filings": news_component(test_buy_all, news, data) if news else None,
        "features": {
            "univariate_auc_for_winning_trades": feature_component(
                test_buy_all, data, kept
            ),
            "features_model": decider_info,
        },
        "order_book": "not evaluated: Kite keeps no order-book history",
        "web_news": "not evaluated: a search today cannot be limited to the past",
    }
    ai: dict = {}
    if args.openai_models:
        ai_keys = in_period(test_keys, data, AI_FROM, TEST[1])
        for item in args.openai_models.split(","):
            if not item:
                continue
            model_name, _, effort = item.partition(":")
            result, buys = evaluate_ai(
                args,
                model_name,
                effort or None,
                ai_keys,
                data,
                calibrated,
                kept,
                oi_raw,
                news,
                output,
            )
            label = f"{model_name}:{result.get('reasoning_effort', 'skipped')}"
            ai[label] = result
            if result.get("status") != "completed":
                continue
            trades = execute(buys, outcomes)
            pipelines[f"scanner_ai_{label}"] = trades
            write_trades(
                output / f"trades_ai_{model_name}_{result['reasoning_effort']}.csv",
                trades,
            )
            result["random_buys_at_ai_rate"] = random_same_rate(
                ai_keys,
                len(buys) / max(len(ai_keys), 1),
                outcomes,
                summarize(trades).get("mean_net_return_pct"),
            )
    if news:
        news.close()
    write_trades(output / "trades_scanner_buy_all.csv", buy_all)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "research_only_not_promoted",
        "design": {
            "scanner": f"calibrated P(high >= +1% of buy within {HORIZON} min) > "
            f"{args.threshold}; decisions every 5 min 09:40-13:00",
            "buy": "open of the minute after the decision close",
            "sell": "first of +1% / -1% of the buy price, no time limit; fills at the "
            "level, or at the minute's open when it opens beyond; a minute reaching "
            "both counts as the stop",
            "size": f"Rs {NOTIONAL_INR:,} in whole shares",
            "charges": "Zerodha intraday when sold the same session; delivery (STT "
            "0.1% each leg, stamp duty, exchange, SEBI, GST, Rs 15.34 DP) otherwise",
            "rules": "one buy per stock per day, none while the stock is held; no "
            "capital limit (max_open_positions shows what was needed)",
            "not_modelled": "spread and slippage beyond the next-minute open, "
            "dividends received while holding, tick rounding of the exit levels",
            "periods": {
                "fit": FIT,
                "early_stop": EARLY_STOP,
                "calibration": CALIBRATION,
                "test": TEST,
                "ai": (AI_FROM, TEST[1]),
            },
        },
        "dataset": str(args.candles),
        "dataset_sha256": file_hash(args.candles / "dataset.json"),
        "excluded_stock_days": excluded,
        "universe": {"stocks": len(data["stocks"]), "days": len(data["days"])},
        "features": {"declared": list(FEATURES), "kept": kept, "dropped": dropped},
        "scanner": {"model": scanner_info, "metrics": scanner},
        "open_interest_data": oi_info,
        "components": components,
        "ai": ai,
        "pipelines": {name: by_period(trades) for name, trades in pipelines.items()},
        "runtime_minutes": round((clock.monotonic() - started) / 60, 1),
        "source_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--oi", type=Path)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument(
        "--openai-models", default=None, help="comma-separated model[:effort]"
    )
    parser.add_argument("--openai-budget-usd", type=float, default=3.0)
    parser.add_argument("--openai-workers", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse journaled OpenAI answers in an existing output",
    )
    report = run(parser.parse_args())
    summary = {
        "scanner_test": {
            k: report["scanner"]["metrics"]["test_all"].get(k)
            for k in ("auc", "base_rate", "candidates")
        },
        "pipelines_test_all": {
            name: brief_from(result["test_all"])
            for name, result in report["pipelines"].items()
        },
        "runtime_minutes": report["runtime_minutes"],
    }
    print(json.dumps(summary, indent=1))
    return 0


def brief_from(stats: dict) -> dict:
    return {
        k: stats.get(k)
        for k in ("trades", "win_rate", "mean_net_return_pct", "total_pnl_inr")
    }


if __name__ == "__main__":
    raise SystemExit(main())
