"""Build the per-candidate input packet the decision layer sees.

One place defines what the AI layer is shown, so the historical replay and the live
loop send identical structures. Return-like inputs are percentages (``*_pct``).

The v3 packet holds facts as of the request: multi-day changes and levels are
measured at the current price, not the last close, and a value from the previous
session says so in its name (``*_prev_session``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ai_trader.ai.decision import Candidate

SCANNER_PERCENT = (
    "day_ret",
    "gap",
    "ret15",
    "ret_open",
    "vwap_dev",
    "dist_high",
    "dist_low",
    "rel_strength",
    "rv20",
    "atr20",
)
SCANNER_RATIO = ("rvol", "rv_ratio", "range_atr", "log_turnover20")
OI_FIELDS = (
    "oi_change_1d",
    "oi_change_5d",
    "fut_return_1d",
    "buildup_1d",
    "basis_pct",
    "calendar_days_to_expiry",
    "volume_to_oi",
    "oi_change_since_prev_close",
    "intraday_buildup",
)
OI_PERCENT = (
    "oi_change_1d",
    "oi_change_5d",
    "fut_return_1d",
    "oi_change_since_prev_close",
)
BOOK_FIELDS = (
    "spread_bps",
    "imbalance_l1",
    "imbalance_l5",
    "imbalance_total",
    "microprice_offset_bps",
    "buy_slippage_bps",
    "sell_slippage_bps",
    "to_upper_circuit_pct",
    "to_lower_circuit_pct",
)


def _num(value: Any, digits: int = 3) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def scanner_block(
    probability: float, rank: int, features: Mapping[str, float]
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "movement_probability": _num(probability, 4),
        "rank": rank,
    }
    for name in SCANNER_PERCENT:
        block[f"{name}_pct"] = _num(100 * features[name]) if name in features else None
    for name in SCANNER_RATIO:
        block[name] = _num(features.get(name))
    return block


def delivery_block(
    probability: float,
    features: Mapping[str, float],
    percent: frozenset[str],
    key: str = "p_up1pct_2h",
    rename: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Delivery scanner probability and every model feature, fractions in percent."""
    block: dict[str, Any] = {key: _num(probability, 4)}
    for name, value in features.items():
        label = (rename or {}).get(name, name)
        if name in percent:
            block[f"{label}_pct"] = _num(100 * value)
        else:
            block[label] = _num(value)
    return block


# Delivery v3: facts only, plain names: (label, source field, scale; None = text).
# Sources are scanner features or, for values at the current price, ``daily_now``.
V3_MARKET = (
    ("nifty_change_today_pct", "nifty_day", 100),
    ("nifty_change_15min_pct", "nifty_ret15", 100),
    ("nifty_change_20d_pct", "nifty_change_20d", 100),
    ("nifty_vs_50d_avg_pct", "nifty_vs_50d_avg", 100),
    ("india_vix", "vix", 1),
    ("india_vix_change_today_pct", "vix_chg", 100),
)
V3_TODAY = (
    ("minutes_to_close", "slot", 1),
    ("change_today_pct", "change_today", 100),
    ("gap_at_open_pct", "gap", 100),
    ("change_since_open_pct", "ret_open", 100),
    ("change_5min_pct", "ret5", 100),
    ("change_15min_pct", "ret15", 100),
    ("change_60min_pct", "ret60", 100),
    ("vs_nifty_today_pct", "rel_day", 100),
    ("vs_vwap_pct", "vwap_dev", 100),
    ("vs_day_high_pct", "dist_high", 100),
    ("position_in_day_range", "range_pos", 1),
    ("day_range_vs_avg_range", "range_atr", 1),
    ("volatility_20min_pct", "rv20", 100),
    ("volatility_vs_usual", "rv_ratio", 1),
    ("volume_vs_usual", "rvol", 1),
)
V3_DAILY = (
    ("change_5d_pct", "change_5d", 100),
    ("change_20d_pct", "change_20d", 100),
    ("change_60d_pct", "change_60d", 100),
    ("vs_nifty_20d_pct", "vs_nifty_20d", 100),
    ("vs_50d_avg_pct", "vs_50d_avg", 100),
    ("vs_200d_avg_pct", "vs_200d_avg", 100),
    ("vs_250d_high_pct", "vs_250d_high", 100),
    ("avg_daily_range_20d_pct", "atr20", 100),
    ("volume_5d_vs_20d", "vol_trend", 1),
)
V3_BOOK = (
    ("spread_bps", "spread_bps", 1),
    ("imbalance_best_level", "imbalance_l1", 1),
    ("imbalance_top5", "imbalance_l5", 1),
    ("imbalance_total", "imbalance_total", 1),
    ("microprice_vs_mid_bps", "microprice_offset_bps", 1),
    ("impact_buy_2L_bps", "buy_slippage_bps", 1),
    ("impact_sell_2L_bps", "sell_slippage_bps", 1),
    ("to_upper_circuit_pct", "to_upper_circuit_pct", 1),
    ("to_lower_circuit_pct", "to_lower_circuit_pct", 1),
)
V3_OI = (
    ("days_to_expiry", "days_to_expiry", 1),
    ("premium_now_pct", "premium_now", 100),
    ("oi_change_today_pct", "oi_change_today", 100),
    ("buildup_today", "buildup_today", None),
    ("oi_change_prev_session_pct", "oi_change_prev_session", 100),
    ("futures_change_prev_session_pct", "futures_change_prev_session", 100),
    ("buildup_prev_session", "buildup_prev_session", None),
    ("oi_change_5_sessions_pct", "oi_change_5_sessions", 100),
)
V3_FILINGS = (
    ("filings_since_prev_close", "since_prev_close"),
    ("material_since_prev_close", "material_since_prev_close"),
    ("filings_last_7_days", "last_7_days"),
    ("subjects_since_prev_close", "categories"),
    ("material_filings", "items"),
    ("board_meetings_next_3_days", "board_meetings_next_3_days"),
    ("corporate_actions_today_or_tomorrow", "corporate_actions_today_or_next"),
)


def _scaled(
    values: Mapping[str, Any], fields: tuple[tuple[str, str, int | None], ...]
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for label, name, scale in fields:
        value = values.get(name)
        if scale is None:
            block[label] = value if isinstance(value, str) else None
        else:
            block[label] = None if value is None else _num(scale * value)
    return block


def daily_now(
    closes: np.ndarray,
    price: float,
    highs: np.ndarray | None = None,
    high: float | None = None,
    nifty: Mapping[str, float | None] | None = None,
) -> dict[str, float | None]:
    """Multi-day changes and levels at ``price`` (now), as fractions: against the
    completed sessions' ``closes`` (oldest first) and, given ``highs``, the higher of
    their last 250 highs and today's ``high`` so far. With NIFTY's own values
    (``nifty``), also the stock's 20-day change minus NIFTY's."""

    def change(back: int) -> float | None:
        return float(price / closes[-back] - 1) if len(closes) >= back else None

    def above_average(days: int) -> float | None:
        return float(price / closes[-days:].mean() - 1) if len(closes) >= days else None

    top = (
        max(float(highs[-250:].max()), high)
        if highs is not None and high is not None and len(highs) >= 250
        else None
    )
    values = {
        "change_today": change(1),
        "change_5d": change(5),
        "change_20d": change(20),
        "change_60d": change(60),
        "vs_50d_avg": above_average(50),
        "vs_200d_avg": above_average(200),
        "vs_250d_high": price / top - 1 if top else None,
    }
    if nifty is not None:
        mine, market = values["change_20d"], nifty.get("change_20d")
        values["vs_nifty_20d"] = (
            mine - market if mine is not None and market is not None else None
        )
    return values


def market_v3(
    features: Mapping[str, float], nifty: Mapping[str, float | None] | None = None
) -> dict[str, Any]:
    """``nifty``: NIFTY's ``daily_now`` values (multi-day fields are None without)."""
    now = {f"nifty_{name}": value for name, value in (nifty or {}).items()}
    return _scaled({**features, **now}, V3_MARKET)


def stock_v3(
    symbol: str,
    features: Mapping[str, float],
    last_price: float,
    *,
    now: Mapping[str, float | None] | None = None,
    profile: Mapping[str, str] | None = None,
    book: Mapping[str, Any] | None = None,
    oi: Mapping[str, Any] | None = None,
    filings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One stock as the v3 AI sees it: factual inputs only, no scanner output.
    ``now``: its ``daily_now`` values at ``last_price``."""
    values = {**features, **(now or {})}
    daily = _scaled(values, V3_DAILY)
    turnover = _num(features.get("log_turnover20"), 9)
    daily["avg_daily_turnover_20d_cr"] = (
        None if turnover is None else _num(10**turnover / 1e7, 2)
    )
    return {
        "symbol": symbol,
        "company": (profile or {}).get("company"),
        "industry": (profile or {}).get("industry"),
        "today": {"last_price": _num(last_price, 2)} | _scaled(values, V3_TODAY),
        "daily": daily,
        "order_book": _scaled(book, V3_BOOK) if book else None,
        "futures_oi": _scaled(oi, V3_OI) if oi else None,
        "nse_filings": {label: filings.get(name) for label, name in V3_FILINGS}
        if filings
        else None,
    }


def oi_block(features: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not features:
        return None
    block = {}
    for name in OI_FIELDS:
        if name not in features:
            continue
        value = features.get(name)
        if name in OI_PERCENT:
            block[f"{name}_pct"] = _num(100 * value) if value is not None else None
        elif isinstance(value, str) or value is None:
            block[name] = value
        else:
            block[name] = _num(value)
    return block


def book_block(features: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not features:
        return None
    return {name: _num(features.get(name)) for name in BOOK_FIELDS}


def candidate(
    symbol: str,
    scanner: dict[str, Any],
    *,
    oi: Mapping[str, Any] | None = None,
    news: Mapping[str, Any] | None = None,
    book: Mapping[str, Any] | None = None,
) -> Candidate:
    items = list((news or {}).get("items") or [])
    return Candidate(
        symbol=symbol,
        scanner=scanner,
        orderbook=book_block(book),
        open_interest=oi_block(oi),
        announcements=tuple(items),
        news={k: v for k, v in (news or {}).items() if k != "items"} or None,
    )
