"""Features of one five-level depth snapshot.

Imbalances lie in [-1, 1]; positive means more resting buy interest. Slippage is
the cost of an immediate market order of ``notional`` rupees against visible depth,
relative to the mid, and is ``None`` when the five levels cannot fill it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from ai_trader.broker import DepthLevel, DepthSnapshot


def _imbalance(buy: float, sell: float) -> float | None:
    total = buy + sell
    return (buy - sell) / total if total > 0 else None


def fill_slippage_bps(
    levels: Sequence[DepthLevel], mid: float, notional: float
) -> float | None:
    remaining, cost, quantity = notional, 0.0, 0
    for level in levels:
        price = float(level.price)
        take = min(level.quantity, int(remaining // price) + 1)
        cost += take * price
        quantity += take
        remaining -= take * price
        if remaining <= 0:
            return abs(cost / quantity - mid) / mid * 1e4
    return None


def depth_features(
    snapshot: DepthSnapshot, *, notional: float = 200_000, as_of: datetime | None = None
) -> dict[str, float | None]:
    last = float(snapshot.last_price)
    features: dict[str, float | None] = {
        "day_change_pct": (last / float(snapshot.previous_close) - 1) * 100,
        "imbalance_total": _imbalance(
            snapshot.total_buy_quantity, snapshot.total_sell_quantity
        ),
        "to_upper_circuit_pct": (float(snapshot.upper_circuit) / last - 1) * 100
        if snapshot.upper_circuit
        else None,
        "to_lower_circuit_pct": (1 - float(snapshot.lower_circuit) / last) * 100
        if snapshot.lower_circuit
        else None,
        "seconds_since_trade": (as_of - snapshot.last_trade_at).total_seconds()
        if as_of and snapshot.last_trade_at
        else None,
        "two_sided": float(bool(snapshot.bids and snapshot.asks)),
    }
    if not (snapshot.bids and snapshot.asks):
        return features | dict.fromkeys(
            (
                "spread_bps",
                "imbalance_l1",
                "imbalance_l5",
                "microprice_offset_bps",
                "depth5_bid_inr",
                "depth5_ask_inr",
                "buy_slippage_bps",
                "sell_slippage_bps",
            )
        )
    bid, ask = snapshot.bids[0], snapshot.asks[0]
    best_bid, best_ask = float(bid.price), float(ask.price)
    mid = (best_bid + best_ask) / 2
    microprice = (best_bid * ask.quantity + best_ask * bid.quantity) / (
        bid.quantity + ask.quantity
    )
    return features | {
        "spread_bps": (best_ask - best_bid) / mid * 1e4,
        "imbalance_l1": _imbalance(bid.quantity, ask.quantity),
        "imbalance_l5": _imbalance(
            sum(level.quantity for level in snapshot.bids),
            sum(level.quantity for level in snapshot.asks),
        ),
        "microprice_offset_bps": (microprice - mid) / mid * 1e4,
        "depth5_bid_inr": sum(float(lv.price) * lv.quantity for lv in snapshot.bids),
        "depth5_ask_inr": sum(float(lv.price) * lv.quantity for lv in snapshot.asks),
        "buy_slippage_bps": fill_slippage_bps(snapshot.asks, mid, notional),
        "sell_slippage_bps": fill_slippage_bps(snapshot.bids, mid, notional),
    }
