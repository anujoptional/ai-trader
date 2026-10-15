"""Live scanner: the production model's probability for every stock at a check.

For one stock at one decision minute this computes the same inputs the bundle's
model was trained on (``scripts/research_delivery_pipeline.py``:
``session_features``, ``daily_columns``, ``market_context``; session end and the
minutes-left input from ``scripts/research_scanner_goals.py``; the movement inputs
from ``scripts/research_scanner_move.py``), then applies the bundle's CatBoost
model and calibration. A parity test pins the two together.

Inputs per stock: today's minute bars (rows 0..minute, 09:15 = row 0), the
previous complete 09:15-14:59 sessions (at least 20) and the previous daily bars.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

from ai_trader.clock import INDIA_TIMEZONE

OPEN_MINUTE = 9 * 60 + 15
LAST = 344  # 14:59, the end of the frame history sessions are measured on
LOOKBACK = 20
DECISIONS = 24 + 5 * np.arange(64)  # decision bar closes 09:40 to 14:55
SESSION_MINUTES = 375  # continuous trading 09:15-15:29
AUCTION_MINUTES = 360  # 09:15-15:14 for F&O stocks once the closing auction began
AUCTION_FROM = date(2026, 8, 3)
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
MARKET = ("nifty_day", "nifty_ret15", "vix", "vix_chg", "nifty_ret20d", "nifty_sma50")
MAX_QUIET_SHARE = 0.10  # more no-trade minutes than this: too illiquid to score


def fill_quiet_minutes(
    bars: np.ndarray, upto: int, max_share: float = MAX_QUIET_SHARE
) -> np.ndarray | None:
    """Rows 0..upto with no-trade minutes filled; None when too many had no trade.

    Kite has no bar for a minute without trades: the last traded price did not
    change and nothing traded, so the minute is the previous close with zero
    volume (leading minutes take the first trade's open).
    """
    rows = np.array(bars[: upto + 1], dtype=float)
    missing = ~np.isfinite(rows[:, 3])
    if not missing.any():
        return rows
    if missing.all() or missing.sum() > max_share * len(rows):
        return None
    first = int(np.argmin(missing))
    rows[:first, :4] = rows[first, 0]
    rows[:first, 4] = 0.0
    for i in np.flatnonzero(missing):
        if i > first:
            rows[i, :4] = rows[i - 1, 3]
            rows[i, 4] = 0.0
    return rows


def session_minutes(day: date, closing_auction: bool) -> int:
    """Continuous-trading minutes of a stock's session (F&O stocks use the auction)."""
    return (
        AUCTION_MINUTES if closing_auction and day >= AUCTION_FROM else SESSION_MINUTES
    )


def minute_index(stamp: datetime) -> int:
    local = stamp.astimezone(INDIA_TIMEZONE)
    return local.hour * 60 + local.minute - OPEN_MINUTE


def decision_close(day: date, minute: int) -> datetime:
    """When decision bar ``minute`` has closed (the buy is from the next minute)."""
    return datetime.combine(day, time(9, 15), INDIA_TIMEZONE) + timedelta(
        minutes=minute + 1
    )


def wilder_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
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


def _squares(close: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.diff(np.log(close)) ** 2)))


def history_day_rv(session: np.ndarray) -> float:
    """Realised volatility of a complete 09:15-14:59 session (as research)."""
    return math.sqrt(20 * _squares(session[: LAST + 1, 3])[LAST] / LAST)


def cumulative_volume(session: np.ndarray, minute: int) -> float:
    return float(np.cumsum(session[: minute + 1, 4])[minute])


@dataclass(frozen=True)
class StockHistory:
    """What a stock's live features need from before today, oldest first."""

    sessions: tuple[np.ndarray, ...]  # complete 09:15-14:59 sessions, [345, 5]
    daily: np.ndarray  # [n, 5] open/high/low/close/volume of earlier sessions
    daily_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if len(self.daily) != len(self.daily_dates):
            raise ValueError("Daily bars and dates differ in length.")


@dataclass(frozen=True)
class MarketHistory:
    nifty_closes: np.ndarray  # NIFTY daily closes before today, oldest first
    nifty_dates: tuple[date, ...]
    vix_previous_close: float


def stock_features(
    session_bars: np.ndarray,
    minute: int,
    history: StockHistory,
    previous_session: date,
    minutes_in_session: int,
) -> dict[str, float] | None:
    """The stock inputs at the close of bar ``minute`` (rows after it ignored).

    None when research would not have produced the row: fewer than 20 earlier
    complete sessions or daily bars, or the stock missed the previous session.
    """
    if (
        len(history.sessions) < LOOKBACK
        or len(history.daily) <= LOOKBACK
        or history.daily_dates[-1] != previous_session
    ):
        return None
    bars = np.asarray(session_bars[: minute + 1], dtype=float)
    open_, high, low, close, volume = bars.T
    d = minute
    squares = _squares(close)
    cumulative = np.cumsum(volume)
    top, bottom = np.max(high), np.min(low)
    span = top - bottom
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.cumsum((high + low + close) / 3 * volume)[d] / cumulative[d]
    daily = history.daily
    position = len(daily)
    window = daily[position - LOOKBACK : position]
    prior = daily[position - LOOKBACK - 1 : position - 1, 3]
    previous = window[-1, 3]
    true_range = np.maximum(window[:, 1], prior) - np.minimum(window[:, 2], prior)
    atr = float(np.mean(true_range) / previous)
    sessions = history.sessions[-LOOKBACK:]
    hist_rv = float(np.median([history_day_rv(s) for s in sessions]))
    closes = daily[:, 3]
    rsi = wilder_rsi(closes)

    def back(k: int) -> float:
        return closes[-1] / closes[-1 - k] - 1 if len(closes) > k else np.nan

    def above_average(k: int) -> float:
        return closes[-1] / closes[-k:].mean() - 1 if len(closes) >= k else np.nan

    high250 = daily[position - 250 : position, 1].max() if position >= 250 else np.nan
    rv20 = math.sqrt(squares[d] - squares[d - 20])
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = cumulative[d] / np.mean([cumulative_volume(s, d) for s in sessions])
        values = {
            "ret5": close[d] / close[d - 5] - 1,
            "ret15": close[d] / close[d - 15] - 1,
            "ret60": close[d] / close[d - 60] - 1 if d >= 60 else np.nan,
            "ret_open": close[d] / open_[0] - 1,
            "gap": open_[0] / previous - 1,
            "vwap_dev": close[d] / vwap - 1,
            "dist_high": close[d] / top - 1,
            "range_pos": (close[d] - bottom) / span if span > 0 else np.nan,
            "rv20": rv20,
            "rv_ratio": rv20 / hist_rv,
            "rvol": rvol,
            "range_atr": span / close[d] / atr,
            "slot": float(minutes_in_session - 1 - d),
            "atr20": atr,
            "hist_rv": hist_rv,
            "ret5d": back(5),
            "ret20d": back(20),
            "ret60d": back(60),
            "dist_sma50": above_average(50),
            "dist_sma200": above_average(200),
            "dist_high250": closes[-1] / high250 - 1,
            "rsi14": rsi[-1],
            "vol_trend": np.mean(window[-5:, 4]) / np.mean(window[:, 4]),
            "log_turnover20": np.log10(np.mean(window[:, 3] * window[:, 4])),
            "log_price": np.log10(close[d]),
            # research_scanner_move.extra_columns
            "dist_low": close[d] / bottom - 1,
            "rv_open": math.sqrt(20 * squares[d] / d),
        }
    return {name: float(value) for name, value in values.items()}


def market_features(
    nifty_today: np.ndarray,
    vix_today: np.ndarray,
    minute: int,
    market: MarketHistory,
) -> dict[str, float]:
    closes = market.nifty_closes
    close = nifty_today[minute, 3]
    level = vix_today[minute, 3]
    return {
        "nifty_day": float(close / closes[-1] - 1),
        "nifty_ret15": float(close / nifty_today[minute - 15, 3] - 1),
        "vix": float(level),
        "vix_chg": float(level / market.vix_previous_close - 1),
        "nifty_ret20d": float(closes[-1] / closes[-21] - 1)
        if len(closes) > 20
        else math.nan,
        "nifty_sma50": float(closes[-1] / closes[-50:].mean() - 1)
        if len(closes) >= 50
        else math.nan,
    }


def all_features(
    stock: Mapping[str, float], market: Mapping[str, float], move: float | None = None
) -> dict[str, float]:
    """Every input, rounded to float32 and combined as research stored them. With
    ``move`` (a fraction), also the move over three volatility estimates scaled to
    the minutes left, in float32 as ``research_scanner_move.movement_scales``."""
    values = {k: np.float32(v) for k, v in {**stock, **market}.items()}
    one = np.float32(1)
    day_return = (one + values["gap"]) * (one + values["ret_open"])
    values["rel_day"] = day_return - one - values["nifty_day"]
    values["rel_20d"] = values["ret20d"] - values["nifty_ret20d"]
    if move is not None:
        left = values["slot"]
        with np.errstate(divide="ignore", invalid="ignore"):
            for name, volatility, unit in (
                ("z_hist", "hist_rv", 20),
                ("z_open", "rv_open", 20),
                ("z_atr", "atr20", SESSION_MINUTES),
            ):
                scale = values[volatility] * np.sqrt(left / unit)
                values[name] = move / np.maximum(scale, 1e-9)
    return {name: float(value) for name, value in values.items()}


@dataclass(frozen=True)
class ScannerBundle:
    """The frozen production model, its calibration and its input contract."""

    version: str
    features: tuple[str, ...]
    optional: frozenset[str]
    percent: frozenset[str]
    calibration: Mapping[str, object]
    model: object
    move: float | None = None  # the +-move of a movement target, as a fraction
    target: str = ""

    @classmethod
    def load(cls, directory: Path) -> ScannerBundle:
        from catboost import CatBoostClassifier

        bundle = json.loads((directory / "bundle.json").read_text("utf-8"))
        model = CatBoostClassifier()
        model.load_model(str(directory / "model.cbm"))
        if model.tree_count_ != bundle["trees"]:
            raise ValueError("Scanner model does not match its bundle.")
        if [int(d) for d in DECISIONS] != bundle["decision_minutes"]:
            raise ValueError("Scanner decision grid differs from the bundle.")
        move = bundle.get("move_bps")
        if move is None and {"z_hist", "z_open", "z_atr"} & set(bundle["features"]):
            raise ValueError("Scanner bundle uses movement inputs but has no move.")
        return cls(
            bundle["version"],
            tuple(bundle["features"]),
            frozenset(bundle["optional_features"]),
            frozenset(bundle["percent_features"]),
            bundle["calibration"],
            model,
            None if move is None else move / 10_000,
            bundle.get("target", ""),
        )

    def vector(self, features: Mapping[str, float]) -> np.ndarray | None:
        """Model inputs in bundle order, or None when a required one is missing."""
        row = np.array([features.get(name, np.nan) for name in self.features], float)
        required = np.array([name not in self.optional for name in self.features])
        return row if np.isfinite(row[required]).all() else None

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        method = self.calibration["chosen"]
        if method == "raw":
            return raw
        if method == "platt":
            p = np.clip(raw, 1e-6, 1 - 1e-6)
            score = self.calibration["platt_slope"] * np.log(p / (1 - p))
            return 1 / (1 + np.exp(-(score + self.calibration["platt_intercept"])))
        if method == "isotonic":
            return np.interp(
                raw, self.calibration["isotonic_x"], self.calibration["isotonic_y"]
            )
        raise ValueError(f"Unknown calibration {method}")

    def predict(self, rows: np.ndarray) -> np.ndarray:
        if not len(rows):
            return np.zeros(0)
        return self.calibrate(self.model.predict_proba(rows)[:, 1])


@dataclass(frozen=True)
class Estimate:
    symbol: str
    probability: float
    features: Mapping[str, float]


def scan(
    bundle: ScannerBundle,
    minute: int,
    session_day: date,
    stocks: Mapping[str, np.ndarray],
    histories: Mapping[str, StockHistory],
    closing_auction: set[str],
    nifty_today: np.ndarray,
    vix_today: np.ndarray,
    market: MarketHistory,
    previous_session: date,
) -> tuple[list[Estimate], dict[str, int]]:
    """Calibrated probability for every stock with complete inputs, highest first,
    and counts of stocks skipped by reason."""
    skipped = {"no_history": 0, "too_few_trades": 0, "missing_input": 0}
    if not (
        np.isfinite(nifty_today[: minute + 1, 3]).all()
        and np.isfinite(vix_today[minute, 3])
    ):
        raise ValueError("Index bars are incomplete for this decision.")
    context = market_features(nifty_today, vix_today, minute, market)
    names, rows, kept = [], [], []
    for symbol, bars in stocks.items():
        history = histories.get(symbol)
        if history is None:
            skipped["no_history"] += 1
            continue
        filled = fill_quiet_minutes(bars, minute)
        if filled is None:
            skipped["too_few_trades"] += 1
            continue
        stock = stock_features(
            filled,
            minute,
            history,
            previous_session,
            session_minutes(session_day, symbol in closing_auction),
        )
        features = all_features(stock, context, bundle.move) if stock else None
        vector = bundle.vector(features) if features else None
        if vector is None:
            skipped["no_history" if stock is None else "missing_input"] += 1
            continue
        names.append(symbol)
        rows.append(vector)
        kept.append(features)
    probabilities = bundle.predict(np.vstack(rows)) if rows else []
    estimates = [
        Estimate(symbol, float(p), features)
        for symbol, p, features in zip(names, probabilities, kept, strict=True)
    ]
    return sorted(estimates, key=lambda e: -e.probability), skipped


def session_array(
    bars: Sequence[tuple[int, float, float, float, float, float]], length: int
) -> np.ndarray:
    """Minute rows (index, open, high, low, close, volume) as a NaN-padded array."""
    array = np.full((length, 5), np.nan)
    for index, *values in bars:
        if 0 <= index < length:
            array[index] = values
    return array
