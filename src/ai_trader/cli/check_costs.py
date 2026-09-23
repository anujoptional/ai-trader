"""Show what a round trip costs and where a trade would have to exit.

Unlike every other check in this package, this one talks to no broker and needs
no market session. The cost model is pure arithmetic over a published schedule,
so this runs at midnight on a Sunday and prints the same thing it would print at
09:20 on a Tuesday. That is the point: the numbers that decide whether a
strategy is viable at a given clip size should not be reachable only when the
market happens to be open.

Two things are worth reading in the output.

The first is the **bend in the cost curve**. Brokerage is capped at twenty
rupees *per leg*, so below about twenty thousand a leg the cap does not bind and
cost is a flat fraction of turnover; above it, the fraction falls away towards
an asymptote. A round trip is roughly 0.27% of a twenty-thousand clip and 0.08%
of a one-lakh clip. The same strategy is therefore a loser at one size and a
winner at the other, which is why the clip is stated before anything else.

The second is the **gap between the exit and what it keeps**. The stated aim is
0.2% above the buy price, which is a gross move; after charges at a one-lakh
clip it leaves about 0.12%. The two differ by roughly a third of the larger, so
which one a figure refers to has to be said every time. The ``interpretations``
block prices both readings side by side, and the estimate rows carry the stated
gross target and the margin it implies together rather than either alone.

Every figure inherits the caveat from ``costs/model.py``: the rates are
transcribed from published tables and have never been reconciled against a real
contract note, and the spread — frequently the largest cost at this horizon — is
not modelled at all. Treat the exit prices as a floor, not a forecast.

Usage::

    python -m ai_trader.cli.check_costs             # a default spread of quotes
    python -m ai_trader.cli.check_costs 2450 100    # specific quotes
"""

import json
import sys
from decimal import Decimal

from ai_trader.costs import (
    FIXED_CLIP_NOTIONAL,
    GROWW_INTRADAY_EQUITY,
    RoundTripCost,
    SizingPolicy,
    TradeCostEstimate,
)

_CLIP_SIZES = (
    Decimal(10_000),
    Decimal(20_000),
    Decimal(50_000),
    Decimal(100_000),
    Decimal(500_000),
)
"""Spanning the brokerage cap, which starts binding at twenty thousand a leg."""

_DEFAULT_PRICES = (
    Decimal("2450"),
    Decimal("1234.55"),
    Decimal("512.30"),
    Decimal("100"),
)
"""A spread of quotes, chosen so the tick is coarse on one and fine on another."""

_GROSS_TARGETS = (Decimal("0.001"), Decimal("0.002"))
"""The stated aim, read as a move from the entry: "0.2% ... or maybe even 0.1%".

Gross, not kept. The rows below convert each into the margin it leaves once the
round trip is paid for, which is the number the rest of the system works in.
"""


def _rupees(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.0001")), "f")


def _fraction(value: Decimal) -> str:
    """Render a fraction with its percentage beside it.

    Both, always. A fraction and a percentage of the same quantity differ by a
    hundred, and printing one alone is how that error survives a code review.
    """
    percent = (value * 100).quantize(Decimal("0.0001"))
    return f"{value.quantize(Decimal('0.00000001'))} ({percent}%)"


def _cost_row(cost: RoundTripCost) -> dict[str, object]:
    return {
        "notional": _rupees(cost.notional),
        "brokerage": _rupees(cost.brokerage),
        "securities_transaction_tax": _rupees(cost.securities_transaction_tax),
        "exchange_transaction_charge": _rupees(cost.exchange_transaction_charge),
        "regulator_fee": _rupees(cost.regulator_fee),
        "stamp_duty": _rupees(cost.stamp_duty),
        "goods_and_services_tax": _rupees(cost.goods_and_services_tax),
        "total": _rupees(cost.total),
        "round_trip_fraction": _fraction(cost.fraction),
    }


def _estimate_row(estimate: TradeCostEstimate) -> dict[str, object]:
    return {
        "entry_price": _rupees(estimate.entry_price),
        "quantity": estimate.quantity,
        "notional": _rupees(estimate.notional),
        "excess_notional": _rupees(estimate.excess_notional),
        "round_trip_cost": _rupees(estimate.cost.total),
        "cash_outlay": _rupees(estimate.cash_outlay),
        "required_gross_fraction": _fraction(estimate.required_gross_fraction),
        "tick_fraction": _fraction(estimate.tick_fraction),
        "long_exit_price": _rupees(estimate.long_exit_price),
        "short_exit_price": _rupees(estimate.short_exit_price),
        "target_net_rupees": _rupees(estimate.target_net_rupees),
        "long_net_rupees": _rupees(estimate.long_net_rupees),
        "short_net_rupees": _rupees(estimate.short_net_rupees),
    }


def _interpretations(notional: Decimal, stated: Decimal) -> dict[str, object]:
    """The two readings of a stated small target, priced side by side."""
    costs = GROWW_INTRADAY_EQUITY
    return {
        "stated": _fraction(stated),
        "as_net_kept_after_costs": {
            "gross_move_required": _fraction(
                costs.required_gross_fraction(notional, stated)
            ),
            "net_kept": _fraction(stated),
        },
        "as_the_gross_move": {
            "gross_move_required": _fraction(stated),
            "net_kept": _fraction(costs.net_fraction(notional, stated)),
        },
    }


def _parse_prices(argv: list[str]) -> tuple[Decimal, ...]:
    if not argv:
        return _DEFAULT_PRICES
    return tuple(Decimal(argument) for argument in argv)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        prices = _parse_prices(arguments)
    except ArithmeticError:
        print("Prices must be decimal numbers.", file=sys.stderr)
        return 1

    estimates: list[dict[str, object]] = []
    for gross_target in _GROSS_TARGETS:
        policy = SizingPolicy.from_gross_target(
            target_notional=FIXED_CLIP_NOTIONAL,
            gross_target_fraction=gross_target,
        )
        rows: list[dict[str, object]] = []
        for price in prices:
            if price <= 0:
                print(f"Price must be positive, got {price}.", file=sys.stderr)
                return 1
            rows.append(_estimate_row(policy.estimate(price)))
        estimates.append(
            {
                "gross_target_fraction": _fraction(gross_target),
                "net_margin_fraction": _fraction(policy.net_margin_fraction),
                "rows": rows,
            }
        )

    summary = {
        "schedule": "Groww retail intraday equity, transcribed and unverified",
        "clip_notional": _rupees(FIXED_CLIP_NOTIONAL),
        "clip_model": "a floor on turnover; whole shares round it up, never down",
        "exit_model": "whole position, one exit, no scaling out",
        "round_trip_cost_by_notional": [
            _cost_row(GROWW_INTRADAY_EQUITY.round_trip(size)) for size in _CLIP_SIZES
        ],
        "interpretations_at_the_clip": [
            _interpretations(FIXED_CLIP_NOTIONAL, target) for target in _GROSS_TARGETS
        ],
        "estimates": estimates,
        "not_modelled": ["spread", "slippage", "price movement between the legs"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
