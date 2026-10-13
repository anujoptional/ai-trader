"""One decision cycle, assembled from already-fetched inputs (pure and testable).

The CLI does the I/O (Kite, NSE, OpenAI); this module turns minute bars, the frozen
movement model, depth snapshots, OI and filings into a ``DecisionRequest`` exactly
as the historical replay does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from ai_trader.ai.decision import Candidate, DecisionRequest
from ai_trader.ai.packet import candidate, scanner_block
from ai_trader.broker import DepthSnapshot, OHLCVCandle
from ai_trader.orderbook.features import depth_features
from ai_trader.scanner.movement import (
    LAST,
    OPEN_MINUTE,
    context_features,
    feature_vector,
    stock_features,
)


def minute_index(stamp: datetime) -> int:
    return stamp.hour * 60 + stamp.minute - OPEN_MINUTE


def session_arrays(candles: Sequence[OHLCVCandle]) -> dict[date, np.ndarray]:
    """Bars by session as [345, 5] arrays; missing minutes (or later ones) are NaN."""
    sessions: dict[date, np.ndarray] = {}
    for bar in candles:
        index = minute_index(bar.timestamp)
        if not 0 <= index <= LAST:
            continue
        array = sessions.setdefault(
            bar.timestamp.date(), np.full((LAST + 1, 5), np.nan)
        )
        array[index] = [
            float(bar.open),
            float(bar.high),
            float(bar.low),
            float(bar.close),
            np.nan if bar.volume is None else float(bar.volume),
        ]
    return sessions


def complete(sessions: Mapping[date, np.ndarray]) -> list[np.ndarray]:
    """Complete sessions only (as research requires), oldest first."""
    return [
        sessions[day]
        for day in sorted(sessions)
        if np.isfinite(sessions[day][:, :4]).all()
    ]


def daily_array(candles: Sequence[OHLCVCandle]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(c.open),
                float(c.high),
                float(c.low),
                float(c.close),
                float(c.volume or 0),
            ]
            for c in sorted(candles, key=lambda c: c.timestamp)
        ]
    ).reshape(-1, 5)


@dataclass(frozen=True)
class StockInputs:
    today: np.ndarray
    history: list[np.ndarray]
    daily: np.ndarray


def score_universe(
    predict: Callable[[np.ndarray], np.ndarray],
    stocks: Mapping[str, StockInputs],
    nifty_today: np.ndarray,
    nifty_previous_close: float,
    vix_today: np.ndarray,
    minute: int,
) -> list[tuple[str, float, dict[str, float]]]:
    """(symbol, movement probability, features) for every stock with valid inputs."""
    rows, names, kept = [], [], []
    for symbol, inputs in stocks.items():
        if not np.isfinite(inputs.today[: minute + 1, :4]).all():
            continue
        stock = stock_features(inputs.today, minute, inputs.history, inputs.daily)
        if stock is None:
            continue
        context = context_features(
            nifty_today, nifty_previous_close, vix_today, minute, stock["day_ret"]
        )
        vector = feature_vector(stock, context)
        if not np.isfinite(vector).all():
            continue
        rows.append(vector)
        names.append(symbol)
        kept.append(stock | context)
    if not rows:
        return []
    probabilities = predict(np.vstack(rows))
    return sorted(
        zip(names, (float(p) for p in probabilities), kept, strict=True),
        key=lambda item: -item[1],
    )


def build_request(
    as_of: datetime,
    ranked: Sequence[tuple[str, float, dict[str, float]]],
    *,
    top_k: int,
    depth: Mapping[str, DepthSnapshot],
    oi: Mapping[str, Mapping] | None = None,
    news: Mapping[str, Mapping] | None = None,
    exclude: set[str] | None = None,
    notional: float = 200_000.0,
) -> DecisionRequest:
    chosen = [r for r in ranked if r[0] not in (exclude or set())][:top_k]
    packets: list[Candidate] = []
    for rank, (symbol, probability, features) in enumerate(chosen, start=1):
        snapshot = depth.get(symbol)
        book = (
            depth_features(snapshot, notional=notional, as_of=as_of)
            if snapshot
            else None
        )
        packets.append(
            candidate(
                symbol,
                scanner_block(probability, rank, features),
                oi=(oi or {}).get(symbol),
                news=(news or {}).get(symbol),
                book=book,
            )
        )
    first = chosen[0][2] if chosen else {}
    market = {
        "nifty_day_pct": round(100 * first.get("nifty_day", float("nan")), 3)
        if first
        else None,
        "nifty_ret15_pct": round(100 * first.get("nifty_ret15", float("nan")), 3)
        if first
        else None,
        "india_vix": round(first.get("vix", float("nan")), 2) if first else None,
    }
    return DecisionRequest(as_of, market, tuple(packets))
