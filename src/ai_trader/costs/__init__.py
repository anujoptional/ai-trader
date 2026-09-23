"""Transaction costs: the floor every hypothesis has to clear.

A small, frequently repeated profit is the stated objective, and at that size
the cost of trading is not a rounding error on the edge — it *is* most of the
edge. This package computes what a round trip costs so that no layer above has
to guess, and so that "a small amount above the cost of trading" can be
expressed as arithmetic rather than as a number somebody picked.

It is stdlib-only and depends on nothing else in this system, which is
deliberate: the scanner screens with it, replay will score with it, and the risk
engine will size with it. Anything it imported would become a dependency of all
three.

Two layers sit inside it. ``model.py`` is pure fraction arithmetic — what a
round trip costs as a fraction of notional, and the gross move that clears it —
and it is what the scanner screens with, because screening compares against
``atr_pct`` and never needs a price. ``sizing.py`` is the decision-time layer:
given a quote, it turns the fixed clip into whole shares and the required
fraction into an exit price the exchange will accept. Keeping them apart is what
lets the scanner stay free of position sizing, which under ``AGENTS.md`` rule 8
belongs to deterministic risk code rather than to anything upstream of it.
"""

from ai_trader.costs.model import (
    BROKERAGE_CAP,
    BROKERAGE_FRACTION,
    EXCHANGE_TRANSACTION_FRACTION,
    GOODS_AND_SERVICES_TAX_FRACTION,
    GROWW_INTRADAY_EQUITY,
    REGULATOR_FEE_FRACTION,
    SECURITIES_TRANSACTION_TAX_FRACTION,
    STAMP_DUTY_FRACTION,
    CostModel,
    RoundTripCost,
)
from ai_trader.costs.sizing import (
    FIXED_CLIP_NOTIONAL,
    NSE_EQUITY_TICK,
    STATED_GROSS_TARGET,
    SizingPolicy,
    TradeCostEstimate,
    round_down_to_tick,
    round_up_to_tick,
)

__all__ = [
    "BROKERAGE_CAP",
    "BROKERAGE_FRACTION",
    "EXCHANGE_TRANSACTION_FRACTION",
    "FIXED_CLIP_NOTIONAL",
    "GOODS_AND_SERVICES_TAX_FRACTION",
    "GROWW_INTRADAY_EQUITY",
    "NSE_EQUITY_TICK",
    "REGULATOR_FEE_FRACTION",
    "SECURITIES_TRANSACTION_TAX_FRACTION",
    "STAMP_DUTY_FRACTION",
    "STATED_GROSS_TARGET",
    "CostModel",
    "RoundTripCost",
    "SizingPolicy",
    "TradeCostEstimate",
    "round_down_to_tick",
    "round_up_to_tick",
]
