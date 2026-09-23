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

**Two schedules, and only one line differs between them.** ``AGENTS.md`` rule 10
requires that Groww be replaceable by Kite, so both brokers' rates live here
side by side rather than one being the truth and the other a future edit. Every
charge below except brokerage is set by statute or by the exchange and is
therefore identical whoever executes the order; they are module constants, and
each schedule takes them as defaults. Brokerage is the only thing a broker
chooses, so it is the only thing each schedule states — which is why
``brokerage_fraction`` and ``brokerage_cap`` are the two fields with no default
at all. There is no universal brokerage rate to default them to, and a default
would silently make one broker's commercial terms look like a law of nature.

**These rates are estimates, not measurements.** They are the published
retail-intraday equity rates, read off each broker's own charges page and
cross-checked against NSE's published levies; not one of them has been
reconciled against a real contract note. Section 7 applies to them exactly as it
applies to a scanner threshold. Until a real contract note is parsed, treat
every figure this module produces as an order of magnitude.

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
- **Depository charges**, because they apply to delivery rather than to an
  intraday round trip.

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

# --- what the broker charges --------------------------------------------------

GROWW_BROKERAGE_FRACTION = Decimal("0.001")
"""Groww brokerage per executed leg, as a fraction of that leg's turnover."""

GROWW_BROKERAGE_CAP = Decimal("20")
"""Rupee ceiling on Groww brokerage per leg. The charge is the lower of the two."""

GROWW_BROKERAGE_FLOOR = Decimal("5")
"""Groww's rupee minimum per leg, which binds below a five-thousand-rupee leg.

Published as: if percentage brokerage would come to less than Rs 5, the charge
becomes Rs 5 — or 2.5% of turnover if that is lower still, which only happens
under Rs 200. Far beneath the clip this system trades, and modelled anyway
because a schedule that quietly under-reports on small sizes is a schedule that
cannot be checked against a contract note.
"""

GROWW_BROKERAGE_FLOOR_FRACTION = Decimal("0.025")
"""The 2.5% ceiling on Groww's rupee minimum. Binds only under Rs 200 a leg."""

ZERODHA_BROKERAGE_FRACTION = Decimal("0.0003")
"""Zerodha brokerage per executed leg — 0.03%, against Groww's 0.1%."""

ZERODHA_BROKERAGE_CAP = Decimal("20")
"""Rupee ceiling on Zerodha brokerage per leg. The same Rs 20 Groww caps at.

The equal caps are why the two brokers cost almost exactly the same at the size
this system trades and materially different amounts below it: the cap binds
above Rs 20,000 a leg at Groww's rate but only above Rs 66,666.67 at Zerodha's.
"""

ZERODHA_BROKERAGE_FLOOR = Decimal("0")
"""Zerodha publishes no rupee minimum. Zero, so the floor never binds."""

# --- what everyone charges, whoever the broker is -----------------------------

SECURITIES_TRANSACTION_TAX_FRACTION = Decimal("0.00025")
"""STT on the sell leg only, intraday equity."""

EXCHANGE_TRANSACTION_FRACTION = Decimal("0.000030699")
"""NSE cash-segment transaction charge, both legs. Rs 306.99 per crore.

Worth knowing why this does not match Groww's charges page, which says
0.00297%. NSE bills Rs 307 per crore per side and has re-split it between this
line and the investor-protection-fund line below: it was Rs 297 + Rs 10, and it
is now Rs 306.99 + Rs 0.01. Groww still quotes the old split and Zerodha quotes
the new one, so the two look like different rates and are not. The total is
what is charged, the total is what this module gets right, and the split is
carried separately only so a contract note printing either one can be
reconciled line by line.
"""

INVESTOR_PROTECTION_FUND_FRACTION = Decimal("0.000000001")
"""NSE's IPFT levy, both legs. Rs 0.01 per crore under the revised split.

Formerly Rs 10 per crore, which is the figure Groww's charges page still
itemises. Taken together with the transaction charge above it comes to Rs 307
per crore per side either way.
"""

REGULATOR_FEE_FRACTION = Decimal("0.000001")
"""SEBI turnover fee, both legs. Published as Rs 10 per crore."""

STAMP_DUTY_FRACTION = Decimal("0.00003")
"""Stamp duty on the buy leg only."""

GOODS_AND_SERVICES_TAX_FRACTION = Decimal("0.18")
"""GST, levied on brokerage plus the exchange and regulator charges.

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
    investor_protection_fund_charge: Decimal
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
            + self.investor_protection_fund_charge
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

    The two brokerage fields are required and the rest are defaulted, because
    that is the real division: brokerage is a commercial term and everything
    else is statute or exchange tariff. Constructing a schedule therefore forces
    a decision about the only figure a broker actually sets.
    """

    brokerage_fraction: Decimal
    brokerage_cap: Decimal
    brokerage_floor: Decimal = Decimal(0)
    brokerage_floor_fraction: Decimal = Decimal(0)
    securities_transaction_tax_fraction: Decimal = SECURITIES_TRANSACTION_TAX_FRACTION
    exchange_transaction_fraction: Decimal = EXCHANGE_TRANSACTION_FRACTION
    investor_protection_fund_fraction: Decimal = INVESTOR_PROTECTION_FUND_FRACTION
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

        # The floor is a second, lower bend in the same curve: below it the
        # charge stops following the rate. Its own percentage ceiling keeps the
        # smallest legs from paying more in brokerage than they turn over.
        if brokerage_leg < self.brokerage_floor:
            brokerage_leg = min(
                self.brokerage_floor, notional * self.brokerage_floor_fraction
            )

        brokerage = brokerage_leg * 2
        exchange = notional * self.exchange_transaction_fraction * 2
        protection = notional * self.investor_protection_fund_fraction * 2
        regulator = notional * self.regulator_fee_fraction * 2

        return RoundTripCost(
            notional=notional,
            brokerage=brokerage,
            securities_transaction_tax=(
                notional * self.securities_transaction_tax_fraction
            ),
            exchange_transaction_charge=exchange,
            investor_protection_fund_charge=protection,
            regulator_fee=regulator,
            stamp_duty=notional * self.stamp_duty_fraction,
            goods_and_services_tax=(
                (brokerage + exchange + protection + regulator)
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


GROWW_INTRADAY_EQUITY = CostModel(
    brokerage_fraction=GROWW_BROKERAGE_FRACTION,
    brokerage_cap=GROWW_BROKERAGE_CAP,
    brokerage_floor=GROWW_BROKERAGE_FLOOR,
    brokerage_floor_fraction=GROWW_BROKERAGE_FLOOR_FRACTION,
)
"""Groww's retail intraday equity rates, unverified. The default schedule.

Default because Groww is the broker this system is connected to, not because
its terms are better. They are not: at every size below Rs 66,666.67 a leg this
schedule is the dearer of the two.
"""

ZERODHA_INTRADAY_EQUITY = CostModel(
    brokerage_fraction=ZERODHA_BROKERAGE_FRACTION,
    brokerage_cap=ZERODHA_BROKERAGE_CAP,
    brokerage_floor=ZERODHA_BROKERAGE_FLOOR,
)
"""Zerodha's retail intraday equity rates, unverified.

This is the schedule behind the Kite swap ``AGENTS.md`` rule 10 contemplates;
Kite is Zerodha's platform, and the charges are published under the broker's
name, so the table is named for the broker.

**At the size this system trades, the choice of broker is not an economic
one.** Both schedules cap brokerage at Rs 20 a leg, and a one-lakh clip is far
past both caps, so a round trip costs the same Rs 82.68 either way — the
schedules differ by less than a paisa, and that only through the rounding in
NSE's re-split of its transaction charge. The difference is real below the cap
and large there: at a Rs 20,000 clip Groww costs about 0.2715% and Zerodha
about 0.1063%, which is the difference between a 0.2% gross target being a loss
and it being a profit. Worth knowing when the clip changes, and worth not
claiming as a reason to switch while it does not.
"""

__all__ = [
    "EXCHANGE_TRANSACTION_FRACTION",
    "GOODS_AND_SERVICES_TAX_FRACTION",
    "GROWW_BROKERAGE_CAP",
    "GROWW_BROKERAGE_FLOOR",
    "GROWW_BROKERAGE_FLOOR_FRACTION",
    "GROWW_BROKERAGE_FRACTION",
    "GROWW_INTRADAY_EQUITY",
    "INVESTOR_PROTECTION_FUND_FRACTION",
    "REGULATOR_FEE_FRACTION",
    "SECURITIES_TRANSACTION_TAX_FRACTION",
    "STAMP_DUTY_FRACTION",
    "ZERODHA_BROKERAGE_CAP",
    "ZERODHA_BROKERAGE_FLOOR",
    "ZERODHA_BROKERAGE_FRACTION",
    "ZERODHA_INTRADAY_EQUITY",
    "CostModel",
    "RoundTripCost",
]
