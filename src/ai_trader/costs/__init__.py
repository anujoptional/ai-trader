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

__all__ = [
    "BROKERAGE_CAP",
    "BROKERAGE_FRACTION",
    "EXCHANGE_TRANSACTION_FRACTION",
    "GOODS_AND_SERVICES_TAX_FRACTION",
    "GROWW_INTRADAY_EQUITY",
    "REGULATOR_FEE_FRACTION",
    "SECURITIES_TRANSACTION_TAX_FRACTION",
    "STAMP_DUTY_FRACTION",
    "CostModel",
    "RoundTripCost",
]
