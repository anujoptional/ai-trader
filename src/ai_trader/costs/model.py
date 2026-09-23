"""What one intraday round trip costs, and the gross move that clears it.

This layer exists because of a single sentence in section 7: *"Gross hit rate is
not edge."* The system must clear brokerage, exchange fees, STT, stamp duty and
GST before anything is left, and at a one-minute horizon those charges are a
large fraction of the move being captured. A strategy aiming at a small,
frequently-repeated profit is therefore not describable as a fixed target
percentage at all — the same 0.2% move is a loss at one position size and a
profit at another.

So nothing here hard-codes a target. It computes what a round trip costs, and
the caller states the margin it wants left over; the required gross move falls
out of the two. That is the only way to honour both the goal ("sell as soon as
it is a small amount above the cost of trading") and section 7's prohibition on
designing around an invented number.

**Units.** Every rate and every result in this module is a **fraction of
notional, not a percentage**: ``Decimal("0.001")`` is 0.1%, ten basis points.
That is deliberate, and it is the one thing to get right when reading this
alongside the feature engine — ``FeatureSnapshot.atr_pct`` is named as a
percentage but is also a fraction (it is ``atr14 / close``, with no
multiplication by a hundred). Comparing a fraction against a percentage is a
hundredfold error that still type-checks and still produces plausible-looking
output, so the names here say ``fraction`` everywhere rather than trusting the
reader to remember.

**These rates are estimates, not measurements.** They are the published
retail-intraday equity rates as generally understood, transcribed by hand; not
one of them has been reconciled against a Groww contract note. Section 7 applies
to them exactly as it applies to a scanner threshold. Until a real contract note
is parsed, treat every figure this module produces as an order of magnitude.

**What is deliberately not modelled.**

- **Spread and slippage**, which at this horizon are frequently larger than
  every charge below combined. A candidate whose theoretical edge is smaller
  than its spread is not a candidate — but no layer in this system produces a
  spread yet, so this module cannot include one and does not pretend to. The
  reserved microstructure fields on ``MarketContext`` are where that arrives.
- **Price movement between the legs.** Both legs are charged on the same
  notional. The real exit notional differs by the size of the move itself, so
  this understates the sell-side charges on a winning long and overstates them
  on a winning short, by roughly the move times those rates — a few thousandths
  of the total cost, far smaller than the uncertainty in the rates themselves.
- **NSE's investor-protection-fund charge** and **depository charges**, the
  latter because they apply to delivery rather than to an intraday round trip.

**Cost is direction-symmetric, which is why shorts need no separate model.**
STT falls on the sell leg and stamp duty on the buy leg. A long round trip buys
then sells; a short round trip sells then buys. Either way there is exactly one
buy and exactly one sell, so both pay the same. The asymmetry people expect from
"STT is on the sell side" is an asymmetry between *legs*, not between
*directions*.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

BROKERAGE_FRACTION = Decimal("0.001")
"""Brokerage per executed leg, as a fraction of that leg's turnover."""

BROKERAGE_CAP = Decimal("20")
"""Rupee ceiling on brokerage per leg. The charge is the lower of the two."""

SECURITIES_TRANSACTION_TAX_FRACTION = Decimal("0.00025")
"""STT on the sell leg only, intraday equity."""

EXCHANGE_TRANSACTION_FRACTION = Decimal("0.0000297")
"""NSE cash-segment transaction charge, both legs."""

REGULATOR_FEE_FRACTION = Decimal("0.000001")
"""SEBI turnover fee, both legs. Published as Rs 10 per crore."""

STAMP_DUTY_FRACTION = Decimal("0.00003")
"""Stamp duty on the buy leg only."""

GOODS_AND_SERVICES_TAX_FRACTION = Decimal("0.18")
"""GST, levied on brokerage plus exchange and regulator charges.

Not on STT or stamp duty, which are themselves taxes.
"""


@dataclass(frozen=True, slots=True)
class RoundTripCost:
    """One round trip's charges, itemised.

    Itemised rather than totalled because every rate above is an unverified
    estimate. A single number would be impossible to reconcile against a real
    contract note; these components can be checked one at a time, and the one
    that is wrong can be found.
    """

    notional: Decimal
    brokerage: Decimal
    securities_transaction_tax: Decimal
    exchange_transaction_charge: Decimal
    regulator_fee: Decimal
    stamp_duty: Decimal
    goods_and_services_tax: Decimal

    @property
    def total(self) -> Decimal:
        """Every charge for both legs, in rupees."""
        return (
            self.brokerage
            + self.securities_transaction_tax
            + self.exchange_transaction_charge
            + self.regulator_fee
            + self.stamp_duty
            + self.goods_and_services_tax
        )

    @property
    def fraction(self) -> Decimal:
        """The total as a fraction of notional. ``0.001`` is 0.1%."""
        return self.total / self.notional


@dataclass(frozen=True, slots=True)
class CostModel:
    """A charge schedule, and the arithmetic that turns it into a hurdle.

    Every rate is a field rather than a constant read directly, so replay can
    hold the strategy fixed and vary the schedule — which is the experiment that
    answers "would this have worked at a different broker's rates", and the one
    that will matter when the Kite swap contemplated in ``AGENTS.md`` happens.
    """

    brokerage_fraction: Decimal = BROKERAGE_FRACTION
    brokerage_cap: Decimal = BROKERAGE_CAP
    securities_transaction_tax_fraction: Decimal = SECURITIES_TRANSACTION_TAX_FRACTION
    exchange_transaction_fraction: Decimal = EXCHANGE_TRANSACTION_FRACTION
    regulator_fee_fraction: Decimal = REGULATOR_FEE_FRACTION
    stamp_duty_fraction: Decimal = STAMP_DUTY_FRACTION
    goods_and_services_tax_fraction: Decimal = GOODS_AND_SERVICES_TAX_FRACTION

    def round_trip(self, notional: Decimal) -> RoundTripCost:
        """Itemise both legs of a round trip of this size.

        ``notional`` is one leg's turnover — price times quantity — not the sum
        of both legs.
        """
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")

        # Capped per leg, not on the round trip. A cap applied once would halve
        # the brokerage on every position large enough to reach it, which is
        # every position where the cap binds at all.
        brokerage_leg = min(self.brokerage_cap, notional * self.brokerage_fraction)
        brokerage = brokerage_leg * 2
        exchange = notional * self.exchange_transaction_fraction * 2
        regulator = notional * self.regulator_fee_fraction * 2

        return RoundTripCost(
            notional=notional,
            brokerage=brokerage,
            securities_transaction_tax=(
                notional * self.securities_transaction_tax_fraction
            ),
            exchange_transaction_charge=exchange,
            regulator_fee=regulator,
            stamp_duty=notional * self.stamp_duty_fraction,
            goods_and_services_tax=(
                (brokerage + exchange + regulator)
                * self.goods_and_services_tax_fraction
            ),
        )

    def round_trip_fraction(self, notional: Decimal) -> Decimal:
        """Round-trip cost as a fraction of notional.

        This falls as notional rises while the brokerage cap binds, then flattens
        onto an asymptote once it does. The shape is the whole reason position
        size is a precondition for a small-target strategy rather than a detail
        of it.
        """
        return self.round_trip(notional).fraction

    def required_gross_fraction(
        self, notional: Decimal, net_margin_fraction: Decimal
    ) -> Decimal:
        """The gross move that leaves ``net_margin_fraction`` after all charges.

        The caller states what it wants to keep; this says what the price has to
        do. There is no default margin on purpose — picking one here would
        reintroduce exactly the hard-coded target section 7 forbids, one layer
        further down where it would be harder to see.

        A non-positive margin is refused. Zero is break-even, which is not a
        trading objective, and a negative one is a request to lose money slowly.
        """
        if net_margin_fraction <= 0:
            raise ValueError(
                f"net_margin_fraction must be positive, got {net_margin_fraction}"
            )
        return self.round_trip_fraction(notional) + net_margin_fraction

    def net_fraction(self, notional: Decimal, gross_fraction: Decimal) -> Decimal:
        """What is left of a gross move after charges. Negative when it loses.

        The inverse of ``required_gross_fraction``, and the one replay will use:
        given what the price actually did, say what the account actually kept.
        """
        return gross_fraction - self.round_trip_fraction(notional)


GROWW_INTRADAY_EQUITY = CostModel()
"""The default schedule: Groww's retail intraday equity rates, unverified.

Named for what it claims to be so that a future ``KITE_INTRADAY_EQUITY`` sits
beside it rather than replacing it, and so a reader is never left guessing which
broker a number came from.
"""

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
