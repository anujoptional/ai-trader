"""Zerodha equity charges for one closed paper trade.

A CNC buy sold in the same session is billed as intraday; held overnight it is a
delivery trade (Zerodha support: "When you buy and sell a share on the same day
using Longterm (CNC), your trade will still be treated as an intraday trade").
Rates from zerodha.com/charges (October 2026). Each leg is charged on its own value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal

from ai_trader.costs.model import (
    EXCHANGE_TRANSACTION_FRACTION,
    GOODS_AND_SERVICES_TAX_FRACTION,
    INVESTOR_PROTECTION_FUND_FRACTION,
    REGULATOR_FEE_FRACTION,
    ZERODHA_BROKERAGE_CAP,
    ZERODHA_BROKERAGE_FRACTION,
)

INTRADAY_STT_SELL = Decimal("0.00025")
INTRADAY_STAMP_BUY = Decimal("0.00003")
DELIVERY_STT = Decimal("0.001")  # on the buy and on the sell
DELIVERY_STAMP_BUY = Decimal("0.00015")
DP_CHARGE = Decimal("15.34")  # per scrip on the day it is sold, GST included


@dataclass(frozen=True, slots=True)
class Charges:
    kind: str  # "intraday" or "delivery"
    brokerage: Decimal
    stt: Decimal
    exchange: Decimal
    sebi: Decimal
    stamp: Decimal
    gst: Decimal
    dp: Decimal

    @property
    def total(self) -> Decimal:
        return (
            self.brokerage
            + self.stt
            + self.exchange
            + self.sebi
            + self.stamp
            + self.gst
            + self.dp
        )

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()} | {"total": str(self.total)}


def trade_charges(
    buy_value: Decimal, sell_value: Decimal, same_session: bool
) -> Charges:
    turnover = buy_value + sell_value
    exchange = turnover * (
        EXCHANGE_TRANSACTION_FRACTION + INVESTOR_PROTECTION_FUND_FRACTION
    )
    sebi = turnover * REGULATOR_FEE_FRACTION
    if same_session:
        brokerage = sum(
            (min(ZERODHA_BROKERAGE_CAP, leg * ZERODHA_BROKERAGE_FRACTION))
            for leg in (buy_value, sell_value)
        )
        stt = sell_value * INTRADAY_STT_SELL
        stamp = buy_value * INTRADAY_STAMP_BUY
        dp = Decimal(0)
        kind = "intraday"
    else:
        brokerage = Decimal(0)
        stt = turnover * DELIVERY_STT
        stamp = buy_value * DELIVERY_STAMP_BUY
        dp = DP_CHARGE
        kind = "delivery"
    gst = (brokerage + exchange + sebi) * GOODS_AND_SERVICES_TAX_FRACTION
    return Charges(kind, brokerage, stt, exchange, sebi, stamp, gst, dp)
