"""Live movement-scanner features, identical to the research assembly.

``scripts/research_fno_scanner.py`` builds the training rows; this module computes
the same 20 inputs for one stock at one decision minute from partial-session data,
so the frozen CatBoost model sees in live exactly what it saw in training. A parity
test pins the two implementations together.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

OPEN_MINUTE = 9 * 60 + 15
LAST = 344
DECISIONS = 24 + 5 * np.arange(47)
LOOKBACK = 20
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


def slot_of(minute: int) -> int | None:
    """Decision slot for a bar index (minutes after 09:15), or None off-grid."""
    matches = np.flatnonzero(DECISIONS == minute)
    return int(matches[0]) if len(matches) else None


def day_rv(session: np.ndarray) -> float:
    close = session[:, 3]
    return math.sqrt(20 * float(np.sum(np.diff(np.log(close)) ** 2)) / LAST)


def cumulative_volume_at(session: np.ndarray, minute: int) -> float:
    return float(np.sum(session[: minute + 1, 4]))


def stock_features(
    session: np.ndarray,
    minute: int,
    history: Sequence[np.ndarray],
    daily: np.ndarray,
) -> dict[str, float] | None:
    """Features at the close of bar ``minute``.

    ``session`` holds the current session's bars 0..minute as rows of open/high/
    low/close/volume (later rows are ignored); ``history`` the previous complete
    sessions (oldest first, at least 20, each 345 rows); ``daily`` the previous
    daily bars (oldest first, ending with the prior session, at least 21 rows).
    Returns None when the history is insufficient, as research does.
    """
    if minute < 20 or len(history) < LOOKBACK or len(daily) < LOOKBACK + 1:
        return None
    bars = np.asarray(session[: minute + 1], dtype=float)
    open_, high, low, close, volume = bars.T
    squares = np.concatenate(([0.0], np.cumsum(np.diff(np.log(close)) ** 2)))
    cumulative = np.cumsum(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.cumsum((high + low + close) / 3 * volume) / cumulative
    d = minute
    top, bottom = np.max(high), np.min(low)
    window = daily[-LOOKBACK:]
    prior = daily[-LOOKBACK - 1 : -1, 3]
    previous = window[-1, 3]
    true_range = np.maximum(window[:, 1], prior) - np.minimum(window[:, 2], prior)
    atr = float(np.mean(true_range) / previous)
    hist_rv = float(np.median([day_rv(s) for s in history[-LOOKBACK:]]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = cumulative[d] / np.mean(
            [cumulative_volume_at(s, d) for s in history[-LOOKBACK:]]
        )
        turnover = np.log10(np.mean(window[:, 3] * window[:, 4]))
    rv20 = math.sqrt(squares[d] - squares[d - 20])
    values = {
        "rv20": rv20,
        "rv_ratio": rv20 / hist_rv if hist_rv else math.nan,
        "rvol": float(rvol),
        "range_atr": (top - bottom) / close[d] / atr,
        "day_ret": close[d] / previous - 1,
        "gap": open_[0] / previous - 1,
        "ret15": close[d] / close[d - 15] - 1,
        "ret_open": close[d] / open_[0] - 1,
        "vwap_dev": close[d] / vwap[d] - 1,
        "log_turnover20": float(turnover),
        "atr20": atr,
        "hist_rv": hist_rv,
        "log_price": math.log10(close[d]),
        "dist_high": close[d] / top - 1,
        "dist_low": close[d] / bottom - 1,
    }
    return {k: float(v) for k, v in values.items()}


def context_features(
    nifty_today: np.ndarray,
    nifty_previous_close: float,
    vix_today: np.ndarray,
    minute: int,
    stock_day_ret: float,
) -> dict[str, float]:
    slot = slot_of(minute)
    nifty_day = float(nifty_today[minute, 3] / nifty_previous_close - 1)
    return {
        "nifty_day": nifty_day,
        "nifty_ret15": float(nifty_today[minute, 3] / nifty_today[minute - 15, 3] - 1),
        "vix": float(vix_today[minute, 3]),
        "rel_strength": stock_day_ret - nifty_day,
        "slot": float(slot if slot is not None else math.nan),
    }


def feature_vector(stock: dict[str, float], context: dict[str, float]) -> np.ndarray:
    merged = stock | context
    return np.asarray([merged[name] for name in FEATURES], dtype=np.float32)
