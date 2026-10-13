"""Delivery CatBoost: P(+1.25% before -0.75%) for LONG 15:00 entries held across days.

Research only, no orders. Kite candles back-adjust splits, bonuses, rights and >=2%
dividends. This study also total-return adjusts the remaining (ordinary) dividends
from NSE's corporate-action list, repairs the three broken demerger adjustments and
excludes outcome windows that cross them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ai_trader.costs import FIXED_CLIP_NOTIONAL, ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore
from scripts.report_tp_catboost import independent_metrics
from scripts.research_delivery_strategy import DP_INR, side_costs
from scripts.train_tp_only import file_hash, write_json

TP, SL = 0.0125, 0.0075
DECISION = 344
SESSION_OPEN = 555
HORIZON = 10
WARMUP = 63
SLIPPAGE = 0.0002
CLIP_INR = 100_000
SEED = 20261011
THREADS = 12
ALIASES = {"ZOMATO": "ETERNAL", "TATAMOTORS": "TMPV"}
SECTOR_INDEX = {
    "Automobile and Auto Components": "NIFTYAUTO",
    "Financial Services": "FINNIFTY",
    "Healthcare": "NIFTYPHARMA",
    "Information Technology": "NIFTYIT",
    "Metals & Mining": "NIFTYMETAL",
    "Fast Moving Consumer Goods": "NIFTYFMCG",
}
INDEXES = ("NIFTY", "INDIAVIX", *sorted(set(SECTOR_INDEX.values())))
EXPIRY_SWITCH = "2025-09-01"
PHASES = (
    ("train", "2024-01-01", "2025-06-30"),
    ("validation", "2025-07-01", "2025-12-31"),
    ("calibration", "2026-01-01", "2026-03-31"),
    ("test", "2026-04-01", "2026-12-31"),
)
GROUPS = {
    "trend": (
        "ret_today",
        "gap_today",
        "intraday_today",
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "ret_126_21",
        "dist_high_20",
        "dist_low_20",
        "dist_high_126",
        "dist_low_126",
        "ma_dist_20",
        "ma_dist_50",
        "rsi_14",
        "clv_today",
        "clv_prev",
        "overnight_21",
        "intraday_21",
    ),
    "volatility": (
        "atr_14",
        "rv_21",
        "vol_ratio_5_21",
        "range_today",
        "max_ret_21",
        "min_ret_21",
        "beta_63",
        "idio_63",
    ),
    "volume": ("rvol_today", "volume_ratio_5_63", "log_turnover_21"),
    "market": (
        "nifty_ret_today",
        "nifty_ret_5d",
        "nifty_ret_21d",
        "nifty_ma_dist_50",
        "vix_1500",
        "vix_change_5d",
        "rel_ret_5d",
        "rel_ret_21d",
        "sector_rel_5d",
    ),
    "breadth": (
        "breadth_ma20",
        "breadth_up",
        "dispersion",
        "rank_ret_5d",
        "rank_ret_21d",
        "rank_rvol",
        "rank_atr",
    ),
    "calendar": ("weekday", "sessions_to_month_end", "sessions_to_expiry"),
}
FEATURES = tuple(name for names in GROUPS.values() for name in names)
CONFIGS = (
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
    {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 30.0},
    {"depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 30.0},
    {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 10.0, "rsm": 0.5},
    {"depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
)
SOURCES = (
    "research_delivery_catboost.py",
    "research_delivery_strategy.py",
    "report_tp_catboost.py",
    "train_tp_only.py",
)


@dataclass
class Bars:
    days: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def find(self, day: int) -> int | None:
        position = bisect_left(self.days, day)
        found = position < len(self.days) and self.days[position] == day
        return position if found else None


@dataclass
class Stock:
    name: str
    day: np.ndarray
    minute: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    row_days: np.ndarray
    row_start: np.ndarray
    row_end: np.ndarray
    daily: Bars
    raw_close: np.ndarray
    raw_volume: np.ndarray
    breaks: tuple[int, ...]

    def span(self, day: int) -> tuple[int, int] | None:
        position = bisect_left(self.row_days, day)
        if position < len(self.row_days) and self.row_days[position] == day:
            return int(self.row_start[position]), int(self.row_end[position])
        return None


def iso(text: str) -> str:
    return datetime.strptime(text, "%d-%b-%Y").date().isoformat()


def dividend_amount(subject: str) -> float:
    return sum(float(value) for value in re.findall(r"R[se]\.?\s*([0-9.]+)", subject))


def actions_for(records: list[dict], symbol: str) -> dict[str, list]:
    found: dict[str, list] = {"dividends": [], "demergers": [], "other": []}
    for record in records:
        if ALIASES.get(record["symbol"], record["symbol"]) != symbol:
            continue
        subject, ex = record["subject"], iso(record["exDate"])
        if re.search("dividend", subject, re.I):
            found["dividends"].append((ex, dividend_amount(subject)))
        elif re.search("demerger", subject, re.I):
            found["demergers"].append(ex)
        else:
            found["other"].append((ex, subject))
    return {key: sorted(set(values)) for key, values in found.items()}


def read_daily(path: Path) -> dict[str, tuple[float, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["timestamp"][:10]: tuple(
                float(row[key]) for key in ("open", "high", "low", "close", "volume")
            )
            for row in csv.DictReader(handle)
        }


def read_minutes(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    days, minutes, prices, volumes = [], [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["start_time", "open", "high", "low", "close", "volume"]:
            raise ValueError(f"Unexpected candle header in {path.name}")
        for start, open_, high, low, close, volume in reader:
            days.append(start[:10])
            minutes.append(int(start[11:13]) * 60 + int(start[14:16]) - SESSION_OPEN)
            prices.append((float(open_), float(high), float(low), float(close)))
            volumes.append(int(volume or 0))
    return (
        days,
        np.asarray(minutes, dtype=np.int32),
        np.asarray(prices, dtype=np.float64).reshape(-1, 4),
        np.asarray(volumes, dtype=np.int64),
    )


def load_bhavcopies(roots: list[Path]) -> dict[str, dict[str, float]]:
    closes: dict[str, dict[str, float]] = {}
    for root in roots:
        for path in sorted(root.glob("*.csv")):
            if path.stem in closes:
                continue
            with path.open(encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = [value.strip() for value in next(reader)]
                day = {}
                for values in reader:
                    row = dict(zip(header, (v.strip() for v in values), strict=False))
                    if row["SERIES"] == "EQ":
                        day[ALIASES.get(row["SYMBOL"], row["SYMBOL"])] = float(
                            row["CLOSE_PRICE"]
                        )
                closes[path.stem] = day
    return closes


def bhavcopy_ratios(
    bhavcopies: dict[str, dict[str, float]], symbol: str, daily: dict
) -> dict[str, float]:
    return {
        day: values[symbol] / daily[day][3]
        for day, values in sorted(bhavcopies.items())
        if symbol in values and day in daily
    }


def classify_dividends(
    dividends: list[tuple[str, float]],
    daily: dict[str, tuple[float, ...]],
    ratios: dict[str, float],
    confounders: list[str],
) -> list[dict]:
    days = sorted(daily)
    samples = sorted(ratios)
    events = sorted([ex for ex, _ in dividends] + confounders)
    result = []
    for ex, amount in dividends:
        position = bisect_left(days, ex)
        if position == 0 or ex > days[-1]:
            continue
        previous = days[position - 1]
        earlier = [s for s in samples if s <= previous]
        later = [s for s in samples if s >= ex]
        factor = ratios[earlier[-1]] if earlier else ratios[samples[0]]
        dividend_yield = amount / (daily[previous][3] * factor)
        record = {"ex_date": ex, "amount": amount, "yield": dividend_yield}
        others = (
            [e for e in events if earlier[-1] < e <= later[0] and e != ex]
            if earlier and later
            else None
        )
        if others == []:
            change = float(np.log(ratios[earlier[-1]] / ratios[later[0]]))
            record["kite_adjusted"] = bool(change > -np.log(1 - dividend_yield) / 2)
            record["evidence"] = "bhavcopy_ratio"
            record["ratio_log_change"] = change
        else:
            record["kite_adjusted"] = bool(dividend_yield >= 0.02)
            record["evidence"] = "two_percent_rule"
        record["rule_agrees"] = record["kite_adjusted"] == (dividend_yield >= 0.02)
        result.append(record)
    return result


def factor_events(
    dividends: list[dict], demergers: list[str], daily: dict[str, tuple[float, ...]]
) -> list[tuple[str, float, str]]:
    days = sorted(daily)
    events = [
        (item["ex_date"], 1 - item["yield"], "dividend")
        for item in dividends
        if not item["kite_adjusted"]
    ]
    for ex in demergers:
        position = bisect_left(days, ex)
        if 0 < position < len(days):
            gap = daily[days[position]][0] / daily[days[position - 1]][3]
            events.append((ex, gap, "demerger"))
    return sorted(events)


def day_factors(events: list[tuple[str, float, str]], days: list[str]) -> np.ndarray:
    factors = np.ones(len(days))
    for ex, value, _ in events:
        factors[: bisect_left(days, ex)] *= value
    return factors


def build_stock(
    symbol: str,
    minutes: tuple[list[str], np.ndarray, np.ndarray, np.ndarray],
    daily: dict[str, tuple[float, ...]],
    calendar: dict[str, int],
    events: list[tuple[str, float, str]],
) -> Stock:
    days, minute, prices, volumes = minutes
    unique = sorted(set(days) | set(daily))
    lookup = dict(zip(unique, day_factors(events, unique), strict=True))
    day_index = np.asarray([calendar[day] for day in days], dtype=np.int32)
    prices = prices * np.asarray([lookup[day] for day in days])[:, None]
    bounds = np.flatnonzero(np.diff(day_index)) + 1
    starts = np.concatenate(([0], bounds)).astype(np.int64)
    ends = np.concatenate((bounds, [len(days)])).astype(np.int64)
    ordered = sorted(daily)
    values = np.asarray([daily[day] for day in ordered], dtype=np.float64)
    scale = np.asarray([lookup[day] for day in ordered])[:, None]
    adjusted = values[:, :4] * scale
    breaks = []
    for ex, _, kind in events:
        if kind == "demerger":
            position = bisect_left(unique, ex)
            breaks.append(calendar[unique[min(position, len(unique) - 1)]])
    return Stock(
        name=symbol,
        day=day_index,
        minute=minute,
        open=prices[:, 0],
        high=prices[:, 1],
        low=prices[:, 2],
        close=prices[:, 3],
        volume=volumes,
        row_days=day_index[starts],
        row_start=starts,
        row_end=ends,
        daily=Bars(
            days=np.asarray([calendar[day] for day in ordered], dtype=np.int64),
            open=adjusted[:, 0],
            high=adjusted[:, 1],
            low=adjusted[:, 2],
            close=adjusted[:, 3],
        ),
        raw_close=values[:, 3],
        raw_volume=values[:, 4],
        breaks=tuple(breaks),
    )


def decision_row(stock: Stock, day: int) -> int | None:
    span = stock.span(day)
    if span is None:
        return None
    start, end = span
    if end - start <= DECISION or stock.minute[start] != 0:
        return None
    if stock.minute[start + DECISION] != DECISION:
        return None
    return start + DECISION


def label_entry(stock: Stock, day: int, entry: int, last_day: int) -> dict:
    price = stock.close[entry]
    take, stop = price * (1 + TP), price * (1 - SL)
    final = day + HORIZON
    last_row_day = bisect_right(stock.row_days, final) - 1
    end = int(stock.row_end[last_row_day])
    high, low = stock.high[entry + 1 : end], stock.low[entry + 1 : end]
    touched = (high >= take) | (low <= stop)
    first = int(np.argmax(touched)) if touched.any() else None
    touch_day = int(stock.day[entry + 1 + first]) if first is not None else None
    stop_scan = touch_day if touch_day is not None else min(final, last_day) + 1
    outcome = None
    for gap_day in range(day + 1, stop_scan):
        if gap_day > last_day or stock.span(gap_day) is not None:
            continue
        position = stock.daily.find(gap_day)
        if position is None:
            return {"status": "missing_day"}
        open_, high_ = stock.daily.open[position], stock.daily.high[position]
        low_ = stock.daily.low[position]
        if open_ >= take:
            outcome = ("tp", gap_day, open_)
        elif open_ <= stop:
            outcome = ("sl", gap_day, open_)
        elif high_ >= take and low_ <= stop:
            return {"status": "daily_ambiguous"}
        elif high_ >= take:
            outcome = ("tp", gap_day, take)
        elif low_ <= stop:
            outcome = ("sl", gap_day, stop)
        if outcome:
            outcome = (*outcome, False, True)
            break
    if outcome is None and first is not None:
        row = entry + 1 + first
        open_ = stock.open[row]
        hit_take, hit_stop = stock.high[row] >= take, stock.low[row] <= stop
        if hit_take and hit_stop:
            if open_ >= take:
                outcome = ("tp", touch_day, open_, False, False)
            elif open_ <= stop:
                outcome = ("sl", touch_day, open_, False, False)
            else:
                outcome = ("sl", touch_day, stop, True, False)
        elif hit_take:
            outcome = ("tp", touch_day, max(take, open_), False, False)
        else:
            outcome = ("sl", touch_day, min(stop, open_), False, False)
    if outcome is None:
        if final > last_day:
            return {"status": "censored"}
        span = stock.span(final)
        if span is not None:
            exit_price = stock.close[span[1] - 1]
        else:
            position = stock.daily.find(final)
            if position is None:
                return {"status": "missing_day"}
            exit_price = stock.daily.close[position]
        outcome = ("timeout", final, exit_price, False, False)
    kind, exit_day, exit_price, ambiguous, fallback = outcome
    if any(day < moment <= exit_day for moment in stock.breaks):
        return {"status": "demerger_window"}
    return {
        "status": "ok",
        "label": int(kind == "tp"),
        "kind": kind,
        "exit_day": int(exit_day),
        "gross": float(exit_price / price - 1),
        "same_day": bool(exit_day == day),
        "ambiguous": bool(ambiguous),
        "daily_fallback": bool(fallback),
        "sessions": int(exit_day - day),
    }


def log_ratio(a: float, b: float) -> float:
    return float(np.log(a / b)) if a > 0 and b > 0 else np.nan


def rsi(closes: np.ndarray, period: int = 14) -> float:
    changes = np.diff(closes)
    gains, losses = np.clip(changes, 0, None), np.clip(-changes, 0, None)
    gain, loss = gains[:period].mean(), losses[:period].mean()
    for up, down in zip(gains[period:], losses[period:], strict=True):
        gain = (gain * (period - 1) + up) / period
        loss = (loss * (period - 1) + down) / period
    return 100.0 if loss == 0 else float(100 - 100 / (1 + gain / loss))


def index_features(series: dict, day: int) -> dict:
    now = series["at_1500"].get(day)
    position = bisect_left(series["days"], day)
    closes = series["closes"][max(0, position - 63) : position]
    if now is None or len(closes) < 50:
        return {}
    return {
        "now": now,
        "ret_today": log_ratio(now, closes[-1]),
        "ret_5d": log_ratio(now, closes[-5]),
        "ret_21d": log_ratio(now, closes[-21]),
        "ma_dist_50": log_ratio(now, np.append(closes[-49:], now).mean()),
    }


def stock_features(
    stock: Stock,
    day: int,
    entry: int,
    market: dict,
    sector: str,
    volume_1500: dict[int, float],
) -> dict | None:
    k = bisect_left(stock.daily.days, day)
    if k < WARMUP:
        return None
    start = stock.span(day)[0]
    price = stock.close[entry]
    open_ = stock.open[start]
    high = stock.high[start : entry + 1].max()
    low = stock.low[start : entry + 1].min()
    o, h = stock.daily.open[:k], stock.daily.high[:k]
    lo, c = stock.daily.low[:k], stock.daily.close[:k]
    history = stock.daily.days[:k]
    previous = c[-1]
    returns = np.diff(np.log(c))
    true_range = np.maximum(h[1:], c[:-1]) - np.minimum(lo[1:], c[:-1])
    atr = true_range[-14:].mean() / price
    recent = returns[-21:]
    nifty_close = market["NIFTY"]["close_by_day"]
    window = history[-64:]
    stock_returns = returns[-(len(window) - 1) :]
    index_returns = np.asarray(
        [
            np.log(nifty_close[b] / nifty_close[a])
            if a in nifty_close and b in nifty_close
            else np.nan
            for a, b in zip(window[:-1], window[1:], strict=True)
        ]
    )
    usable = np.isfinite(index_returns)
    beta = idio = np.nan
    if usable.sum() >= 40 and index_returns[usable].var() > 0:
        x, y = index_returns[usable], stock_returns[usable]
        beta = float(np.cov(y, x)[0, 1] / x.var(ddof=1))
        idio = float(np.std(y - beta * x, ddof=1))
    prior = [volume_1500[d] for d in history[-20:] if d in volume_1500]
    volumes = stock.raw_volume[:k]
    turnover = stock.raw_close[:k] * volumes
    long_ok = k >= 126
    features = {
        "ret_today": log_ratio(price, previous),
        "gap_today": log_ratio(open_, previous),
        "intraday_today": log_ratio(price, open_),
        "ret_5d": log_ratio(price, c[-5]),
        "ret_21d": log_ratio(price, c[-21]),
        "ret_63d": log_ratio(price, c[-63]),
        "ret_126_21": log_ratio(c[-21], c[-126]) if long_ok else np.nan,
        "dist_high_20": log_ratio(price, max(h[-19:].max(), high)),
        "dist_low_20": log_ratio(price, min(lo[-19:].min(), low)),
        "dist_high_126": log_ratio(price, max(h[-125:].max(), high))
        if long_ok
        else np.nan,
        "dist_low_126": log_ratio(price, min(lo[-125:].min(), low))
        if long_ok
        else np.nan,
        "ma_dist_20": log_ratio(price, np.append(c[-19:], price).mean()),
        "ma_dist_50": log_ratio(price, np.append(c[-49:], price).mean()),
        "rsi_14": rsi(np.append(c[-60:], price)),
        "clv_today": float((price - low) / (high - low)) if high > low else 0.5,
        "clv_prev": float((c[-1] - lo[-1]) / (h[-1] - lo[-1]))
        if h[-1] > lo[-1]
        else 0.5,
        "overnight_21": float(
            np.log(o[-20:] / c[-21:-1]).sum() + np.log(open_ / previous)
        ),
        "intraday_21": float(np.log(c[-20:] / o[-20:]).sum() + np.log(price / open_)),
        "atr_14": float(atr),
        "rv_21": float(recent.std(ddof=1)),
        "vol_ratio_5_21": float(returns[-5:].std(ddof=1) / recent.std(ddof=1)),
        "range_today": float(np.log(high / low) / atr) if atr > 0 else np.nan,
        "max_ret_21": float(recent.max()),
        "min_ret_21": float(recent.min()),
        "beta_63": beta,
        "idio_63": idio,
        "rvol_today": float(volume_1500[day] / np.mean(prior))
        if len(prior) >= 10 and np.mean(prior) > 0
        else np.nan,
        "volume_ratio_5_63": float(volumes[-5:].mean() / volumes[-63:].mean()),
        "log_turnover_21": float(np.log(turnover[-21:].mean())),
    }
    nifty = index_features(market["NIFTY"], day)
    peer = index_features(market[sector], day)
    vix = market["INDIAVIX"]
    vix_now = vix["at_1500"].get(day)
    vix_position = bisect_left(vix["days"], day)
    features.update(
        nifty_ret_today=nifty.get("ret_today", np.nan),
        nifty_ret_5d=nifty.get("ret_5d", np.nan),
        nifty_ret_21d=nifty.get("ret_21d", np.nan),
        nifty_ma_dist_50=nifty.get("ma_dist_50", np.nan),
        vix_1500=float(vix_now) if vix_now else np.nan,
        vix_change_5d=log_ratio(vix_now, vix["closes"][vix_position - 5])
        if vix_now and vix_position >= 5
        else np.nan,
    )
    features["rel_ret_5d"] = features["ret_5d"] - features["nifty_ret_5d"]
    features["rel_ret_21d"] = features["ret_21d"] - features["nifty_ret_21d"]
    features["sector_rel_5d"] = features["ret_5d"] - peer.get("ret_5d", np.nan)
    return features


def monthly_expiry(year: int, month: int, sessions: set[str]) -> str:
    weekday = 3 if f"{year:04d}-{month:02d}-01" < EXPIRY_SWITCH else 1
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    nominal = last - timedelta(days=(last.weekday() - weekday) % 7)
    candidate = nominal
    while candidate.isoformat() not in sessions and candidate.day > 1:
        candidate -= timedelta(days=1)
    return (candidate if candidate.isoformat() in sessions else nominal).isoformat()


def calendar_table(sessions: list[str]) -> dict[str, dict]:
    session_set = set(sessions)
    table = {}
    for position, day in enumerate(sessions):
        year, month = int(day[:4]), int(day[5:7])
        expiry = monthly_expiry(year, month, session_set)
        if expiry < day:
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
            expiry = monthly_expiry(year, month, session_set)
        month_end = bisect_left(sessions, f"{day[:7]}-32") - position - 1
        table[day] = {
            "weekday": float(date.fromisoformat(day).weekday()),
            "sessions_to_month_end": float(month_end),
            "sessions_to_expiry": float(bisect_left(sessions, expiry) - position),
        }
    return table


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) > 1:
        order = np.argsort(np.argsort(values[finite], kind="stable"), kind="stable")
        result[finite] = order / (len(finite) - 1)
    return result


def cross_sectional(rows: list[dict]) -> None:
    by_day: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_day[row["day"]].append(row)
    for group in by_day.values():
        today = np.asarray([r["features"]["ret_today"] for r in group])
        trend = np.asarray([r["features"]["ma_dist_20"] for r in group])
        common = {
            "breadth_ma20": float(np.mean(trend > 0)),
            "breadth_up": float(np.mean(today > 0)),
            "dispersion": float(np.std(today)),
        }
        ranks = {
            name: percentile_ranks(np.asarray([r["features"][source] for r in group]))
            for name, source in (
                ("rank_ret_5d", "ret_5d"),
                ("rank_ret_21d", "ret_21d"),
                ("rank_rvol", "rvol_today"),
                ("rank_atr", "atr_14"),
            )
        }
        for index, row in enumerate(group):
            row["features"].update(common)
            row["features"].update({k: float(v[index]) for k, v in ranks.items()})


def load_index(store: CandleStore, name: str, calendar: dict[str, int]) -> dict:
    daily = read_daily(store.verified_path(f"daily/NSE_{name}.csv"))
    days, minutes, prices, _ = read_minutes(store.verified_path(f"NSE_{name}.csv"))
    at_1500 = {
        calendar[d]: float(p[3])
        for d, m, p in zip(days, minutes, prices, strict=True)
        if m == DECISION and d in calendar
    }
    closes = {calendar[d]: v[3] for d, v in daily.items() if d in calendar}
    ordered = sorted(closes)
    return {
        "days": np.asarray(ordered),
        "closes": np.asarray([closes[d] for d in ordered]),
        "close_by_day": closes,
        "at_1500": at_1500,
    }


def build_dataset(
    candles: Path, constituents: Path, actions_path: Path, nse_roots: list[Path]
) -> tuple[list[dict], dict]:
    store = CandleStore(candles)
    with constituents.open(encoding="utf-8") as handle:
        sectors = {row["Symbol"]: row["Industry"] for row in csv.DictReader(handle)}
    records = json.loads(actions_path.read_text(encoding="utf-8"))
    dailies = {
        symbol: read_daily(store.verified_path(f"daily/NSE_{symbol}.csv"))
        for symbol in sectors
    }
    sessions_all = sorted(set().union(*(set(d) for d in dailies.values())))
    calendar = {day: index for index, day in enumerate(sessions_all)}
    market = {name: load_index(store, name, calendar) for name in INDEXES}
    regular = [sessions_all[g] for g in sorted(market["NIFTY"]["at_1500"])]
    regular_set = set(regular)
    calendar_values = calendar_table(regular)
    bhavcopies = load_bhavcopies(nse_roots)
    audit, rows, statuses = {}, [], defaultdict(int)
    last_day = len(sessions_all) - 1
    for symbol in sorted(sectors):
        actions = actions_for(records, symbol)
        ratios = bhavcopy_ratios(bhavcopies, symbol, dailies[symbol])
        confounders = [
            ex
            for ex, subject in actions["other"]
            if not re.search("buy", subject, re.I)
        ] + actions["demergers"]
        dividends = classify_dividends(
            actions["dividends"], dailies[symbol], ratios, confounders
        )
        events = factor_events(dividends, actions["demergers"], dailies[symbol])
        audit[symbol] = {
            "dividends": dividends,
            "demergers": actions["demergers"],
            "other_actions": actions["other"],
            "applied_factors": events,
            "bhavcopy_ratio_samples": len(ratios),
        }
        minutes = read_minutes(store.verified_path(f"NSE_{symbol}.csv"))
        stock = build_stock(symbol, minutes, dailies[symbol], calendar, events)
        decisions = {}
        for day in stock.row_days.tolist():
            entry = decision_row(stock, day)
            if entry is not None and sessions_all[day] in regular_set:
                decisions[day] = entry
        volume_1500 = {
            day: float(stock.volume[stock.span(day)[0] : entry + 1].sum())
            for day, entry in decisions.items()
        }
        sector = SECTOR_INDEX.get(sectors[symbol], "NIFTY")
        for day, entry in decisions.items():
            features = stock_features(stock, day, entry, market, sector, volume_1500)
            if features is None:
                statuses["warmup"] += 1
                continue
            outcome = label_entry(stock, day, entry, last_day)
            statuses[outcome["status"]] += 1
            if outcome["status"] != "ok":
                continue
            features.update(calendar_values[sessions_all[day]])
            rows.append(
                {
                    "symbol": symbol,
                    "day": day,
                    "date": sessions_all[day],
                    "exit_date": sessions_all[outcome["exit_day"]],
                    "entry_price": float(stock.close[entry]),
                    "features": features,
                    **{k: v for k, v in outcome.items() if k != "status"},
                }
            )
        print(json.dumps({"symbol": symbol, "rows": len(rows)}), flush=True)
    cross_sectional(rows)
    meta = {
        "sessions": len(sessions_all),
        "regular_sessions": len(regular),
        "statuses": dict(statuses),
        "corporate_actions": audit,
    }
    return rows, meta


def assign_phases(dates: np.ndarray, exits: np.ndarray) -> np.ndarray:
    phase = np.full(len(dates), "", dtype=object)
    for number, (name, first, last) in enumerate(PHASES):
        inside = (dates >= first) & (dates <= last)
        if number + 1 < len(PHASES):
            inside &= exits < PHASES[number + 1][1]
        phase[inside] = name
    return phase


def trade_costs(same_day: np.ndarray) -> np.ndarray:
    buy, sell = side_costs("stock")
    delivery = buy + sell + DP_INR / CLIP_INR
    intraday = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(FIXED_CLIP_NOTIONAL))
    return np.where(same_day, intraday, delivery) + 2 * SLIPPAGE


def fit_model(config: dict, train, validation, iterations: int | None = None):
    from catboost import CatBoostClassifier

    stopping = (
        {} if iterations else {"early_stopping_rounds": 300, "use_best_model": True}
    )
    model = CatBoostClassifier(
        iterations=iterations or 5000,
        loss_function="Logloss",
        eval_metric="Logloss",
        border_count=254,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=SEED,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=False,
        **stopping,
        **config,
    )
    model.fit(*train, eval_set=None if iterations else validation)
    return model


def bands(probability, labels, net, gross) -> list[dict]:
    table = []
    ids = np.searchsorted(np.arange(1, 10) / 10, probability, side="right")
    for band in range(10):
        chosen = ids == band
        row = {"band": f"{10 * band}-{10 * band + 10}%", "calls": int(chosen.sum())}
        if chosen.any():
            row.update(
                mean_probability_percent=100 * float(probability[chosen].mean()),
                tp_first_percent=100 * float(labels[chosen].mean()),
                mean_gross_percent=100 * float(gross[chosen].mean()),
                mean_net_percent=100 * float(net[chosen].mean()),
            )
        table.append(row)
    return table


def thresholds(probability, labels, net, gross, sessions: int, cutoffs: dict) -> list:
    table = []
    for name, cutoff in cutoffs.items():
        chosen = probability >= cutoff
        row = {"rule": name, "min_probability": cutoff, "calls": int(chosen.sum())}
        if chosen.any():
            row.update(
                calls_per_session=float(chosen.sum() / sessions),
                tp_first_percent=100 * float(labels[chosen].mean()),
                mean_gross_percent=100 * float(gross[chosen].mean()),
                mean_net_percent=100 * float(net[chosen].mean()),
                net_inr_per_session=float(CLIP_INR * net[chosen].sum() / sessions),
            )
        table.append(row)
    return table


def run(
    candles: Path,
    constituents: Path,
    actions: Path,
    nse_roots: list[Path],
    output: Path,
) -> dict:
    from catboost import Pool
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression

    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    here = Path(__file__).parent
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "target": f"LONG entry at the 14:59-15:00 candle close; label 1 if +{TP} "
            f"is touched before -{SL} within {HORIZON} sessions on the minute path "
            "stitched across days (same-minute touches count as the stop unless the "
            "candle opened beyond a barrier); 0 for stop or timeout",
            "adjustments": "Kite back-adjusts splits, bonuses, rights and >=2% "
            "dividends (verified against NSE bhavcopies); remaining dividends are "
            "total-return adjusted from NSE's corporate-action list (classified by "
            "bhavcopy ratio evidence, else the 2% rule); demerger breaks repaired by "
            "removing the ex-date gap and outcome windows crossing them excluded",
            "gaps": "missing minute days use the daily bar; both barriers inside "
            "it, missing data or censoring at the data end exclude the decision",
            "features": GROUPS,
            "phases": PHASES,
            "purge": "a decision is dropped when its exit date reaches the next "
            "phase's first date",
            "selection": "configs fit on train with early stopping on validation; "
            "lowest validation log loss wins; then greedy backward removal of "
            "feature groups while validation log loss improves by more than 0.0002",
            "final_model": "refit on train+validation with the selected tree count; "
            "Platt calibration fit on Jan-Feb 2026 adopted only if it lowers March "
            "log loss, then refit on Jan-Mar; test evaluated once",
            "costs": "delivery round trip incl. DP Rs 15.34 at Rs 1L, intraday "
            f"charges for same-day exits, {SLIPPAGE} slippage per side",
            "configs": CONFIGS,
            "versions": {n: version(n) for n in ("catboost", "numpy", "scikit-learn")},
            "source_sha256": {name: file_hash(here / name) for name in SOURCES},
        },
    )
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)
    began = time.perf_counter()
    rows, meta = build_dataset(candles, constituents, actions, nse_roots)
    features = np.asarray(
        [[row["features"][name] for name in FEATURES] for row in rows], dtype=np.float32
    )

    def column(key: str, dtype=None) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=dtype)

    labels, gross = column("label", np.int64), column("gross", np.float64)
    same_day, dates, exits = (
        column("same_day", bool),
        column("date"),
        column("exit_date"),
    )
    symbols, kinds, held = (
        column("symbol"),
        column("kind"),
        column("sessions", np.int64),
    )
    ambiguous, fallback = column("ambiguous", bool), column("daily_fallback", bool)
    net = gross - trade_costs(same_day)
    phase = assign_phases(dates, exits)
    np.savez_compressed(
        output / "dataset.npz",
        features=features,
        feature_names=np.asarray(FEATURES),
        labels=labels,
        gross=gross,
        net=net,
        same_day=same_day,
        dates=dates,
        exit_dates=exits,
        symbols=symbols,
        phase=phase.astype(str),
        kind=kinds,
        sessions=held,
        ambiguous=ambiguous,
        daily_fallback=fallback,
        entry_price=column("entry_price", np.float64),
    )
    write_json(output / "corporate_actions_audit.json", meta.pop("corporate_actions"))
    masks = {name: phase == name for name, *_ in PHASES}
    summary = {
        "build_seconds": time.perf_counter() - began,
        **meta,
        "rows": len(rows),
        "purged": int((phase == "").sum()),
        "phases": {
            name: {
                "rows": int(mask.sum()),
                "sessions": int(len(np.unique(dates[mask]))),
                "first": min(dates[mask]) if mask.any() else None,
                "last": max(dates[mask]) if mask.any() else None,
                "tp_first_rate": float(labels[mask].mean()) if mask.any() else None,
                "kinds": {
                    k: int((kinds[mask] == k).sum()) for k in ("tp", "sl", "timeout")
                },
                "same_day_exits": int(same_day[mask].sum()),
                "sessions_to_exit_quantiles": np.quantile(
                    held[mask], (0.5, 0.9, 0.99)
                ).tolist()
                if mask.any()
                else None,
                "ambiguous_minutes": int(ambiguous[mask].sum()),
                "daily_fallbacks": int(fallback[mask].sum()),
                "mean_gross_percent": 100 * float(gross[mask].mean())
                if mask.any()
                else None,
            }
            for name, mask in masks.items()
        },
        "feature_missing_share": {
            name: float(np.isnan(features[:, i]).mean())
            for i, name in enumerate(FEATURES)
        },
    }
    write_json(output / "dataset_summary.json", summary)

    def subset(mask: np.ndarray, names: tuple[str, ...]):
        columns = [FEATURES.index(n) for n in names]
        return features[mask][:, columns], labels[mask]

    train, validation = masks["train"], masks["validation"]
    trials = []
    for config in CONFIGS:
        model = fit_model(config, subset(train, FEATURES), subset(validation, FEATURES))
        probability = model.predict_proba(subset(validation, FEATURES)[0])[:, 1]
        trials.append(
            {
                "config": config,
                "trees": model.tree_count_,
                "validation": independent_metrics(labels[validation], probability),
            }
        )
    best = min(trials, key=lambda trial: trial["validation"]["log_loss"])
    config = best["config"]
    groups = list(GROUPS)
    current = best["validation"]["log_loss"]
    ablation = []
    while len(groups) > 1:
        options = []
        for group in groups:
            names = tuple(n for g in groups if g != group for n in GROUPS[g])
            model = fit_model(config, subset(train, names), subset(validation, names))
            loss = independent_metrics(
                labels[validation],
                model.predict_proba(subset(validation, names)[0])[:, 1],
            )["log_loss"]
            options.append((loss, group))
        ablation.append(
            {
                "kept": list(groups),
                "loss": current,
                "drop_one": {g: v for v, g in options},
            }
        )
        loss, group = min(options)
        if current - loss <= 0.0002:
            break
        groups.remove(group)
        current = loss
    names = tuple(n for g in groups for n in GROUPS[g])
    selector = fit_model(config, subset(train, names), subset(validation, names))
    selected_validation = selector.predict_proba(subset(validation, names)[0])[:, 1]
    importance = selector.get_feature_importance(
        type="LossFunctionChange", data=Pool(*subset(validation, names))
    )
    combined = train | validation
    final = fit_model(config, subset(combined, names), None, selector.tree_count_)
    final.save_model(str(output / "best.cbm"))
    calibration = masks["calibration"]
    raw_calibration = final.predict_proba(subset(calibration, names)[0])[:, 1]
    fit_part = dates[calibration] < "2026-03-01"

    def platt(probability, target):
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(logit(np.clip(probability, 1e-6, 1 - 1e-6))[:, None], target)
        return float(model.coef_[0, 0]), float(model.intercept_[0])

    def apply(mapping, probability):
        slope, intercept = mapping
        return expit(slope * logit(np.clip(probability, 1e-6, 1 - 1e-6)) + intercept)

    cal_labels = labels[calibration]
    trial_mapping = platt(raw_calibration[fit_part], cal_labels[fit_part])
    raw_check = independent_metrics(cal_labels[~fit_part], raw_calibration[~fit_part])
    platt_check = independent_metrics(
        cal_labels[~fit_part], apply(trial_mapping, raw_calibration[~fit_part])
    )
    adopt = platt_check["log_loss"] < raw_check["log_loss"]
    mapping = platt(raw_calibration, cal_labels) if adopt else None
    lock = {
        "config": config,
        "trees": selector.tree_count_,
        "feature_groups": groups,
        "features": names,
        "calibration": {
            "adopted": adopt,
            "mapping": mapping,
            "march_raw": raw_check,
            "march_platt": platt_check,
        },
        "weights_sha256": file_hash(output / "best.cbm"),
    }
    write_json(output / "selection_lock.json", lock)

    test = masks["test"]
    raw_test = final.predict_proba(subset(test, names)[0])[:, 1]
    selected = apply(mapping, raw_test) if adopt else raw_test
    prior = float(labels[combined].mean())
    control_names = ("atr_14", "rv_21", "range_today", "vix_1500")
    control_columns = [FEATURES.index(n) for n in control_names]
    filler = np.nanmedian(features[combined][:, control_columns], axis=0)

    def prepared(mask):
        values = features[mask][:, control_columns]
        return np.where(np.isnan(values), filler, values)

    control = LogisticRegression(max_iter=1000).fit(
        prepared(combined), labels[combined]
    )
    sessions = len(np.unique(dates[test]))
    t_labels, t_net, t_gross = labels[test], net[test], gross[test]
    wins = gross[combined & (labels == 1)].mean()
    losses = gross[combined & (labels == 0)].mean()
    cost = float(trade_costs(same_day[combined]).mean())
    break_even = float((cost - losses) / (wins - losses))
    months = defaultdict(list)
    for index, day in enumerate(dates[test]):
        months[day[:7]].append(index)
    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "selection_lock_sha256": file_hash(output / "selection_lock.json"),
        "dataset": {k: summary[k] for k in ("rows", "purged", "phases", "statuses")},
        "trials": trials,
        "selected_config": config,
        "group_ablation": ablation,
        "final_groups": groups,
        "validation_selected": independent_metrics(
            labels[validation], selected_validation
        ),
        "validation_importance": sorted(
            zip(names, map(float, importance), strict=True), key=lambda item: -item[1]
        ),
        "calibration": lock["calibration"],
        "economics_train_validation": {
            "mean_gross_win_percent": 100 * float(wins),
            "mean_gross_loss_percent": 100 * float(losses),
            "mean_cost_percent": 100 * cost,
            "net_break_even_probability": break_even,
            "gross_break_even_probability": float(-losses / (wins - losses)),
        },
        "test": {
            "sessions": sessions,
            "selected": independent_metrics(t_labels, selected),
            "raw": independent_metrics(t_labels, raw_test),
            "base_rate_control": independent_metrics(
                t_labels, np.full(len(t_labels), prior)
            ),
            "volatility_only_control": independent_metrics(
                t_labels, control.predict_proba(prepared(test))[:, 1]
            ),
            "probability_quantiles_percent": {
                str(q): 100 * float(np.quantile(selected, q))
                for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
            },
            "bands": bands(selected, t_labels, t_net, t_gross),
            "thresholds": thresholds(
                selected,
                t_labels,
                t_net,
                t_gross,
                sessions,
                {
                    "all": 0.0,
                    "0.40": 0.40,
                    "0.45": 0.45,
                    "net_break_even": break_even,
                    "0.55": 0.55,
                    "0.60": 0.60,
                },
            ),
            "by_month": {
                month: {
                    "rows": len(index),
                    "tp_first_rate": float(t_labels[index].mean()),
                    "mean_probability": float(selected[index].mean()),
                    "auc": independent_metrics(t_labels[index], selected[index])[
                        "roc_auc"
                    ],
                }
                for month, index in sorted(months.items())
            },
        },
    }
    np.savez_compressed(
        output / "test_predictions.npz",
        raw=raw_test,
        selected=selected,
        labels=t_labels,
        dates=dates[test],
        symbols=symbols[test],
        gross=t_gross,
        net=t_net,
    )
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--constituents", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--nse", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.candles, args.constituents, args.actions, args.nse, args.output)
    print(
        json.dumps(
            {"config": report["selected_config"], "groups": report["final_groups"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
