"""From a price to a costed decision: how many shares, and where to exit.

``model.py`` works entirely in fraction space. It never sees a price or a
quantity, which is right for screening — the scanner compares a required move
against ``atr_pct`` and neither side needs to know what a share costs. But a
*decision* is taken holding a quote, and at that moment three things are wanted
that a fraction cannot supply: how many shares to buy, what that actually costs
in rupees, and the price to leave at. This module is that step.

**The buying model is deliberately simple: one fixed rupee clip per entry, and
exits are always the whole position.** Nothing here can express a partial exit,
and that is structural rather than a default — ``TradeCostEstimate`` carries a
single ``quantity`` used by both legs, and ``CostModel.round_trip`` takes one
notional for the same reason. Scaling out would need a different object, so the
day it is wanted it will be visible as a change rather than arriving quietly.

**The clip is a floor on turnover, not a ceiling.** Quantity is the fewest whole
shares worth at least ``target_notional``, so what fills is the clip or a little
more, never less: a one-lakh clip of a stock at Rs 2,450 is forty-one shares and
Rs 1,00,450. The difference matters more than it looks, because round-trip cost
is a *fraction of notional* — feeding the target into the cost model instead of
the filled value prices a trade that was never placed. Every figure below is
computed from ``notional``, which is ``quantity * entry_price``.

**Every name is sizeable, including one quoted above the whole clip.** NSE lists
names above a lakh; at those, a single share already clears the floor, and a
single share is what is bought. The overshoot is real and worth reading —
``excess_notional`` reports it — but it is a known consequence of a fixed clip
rather than a reason to refuse the trade.

**Exit prices are rounded to a tick, and always away from the entry.** The
arithmetic gives an exact price that the exchange will not accept, so it has to
be moved to a tick boundary, and the direction of that move is a correctness
question rather than a rounding preference: rounding a long's exit *down* to the
nearer tick leaves it a shade under the price that pays for the round trip, so
the trade fills, looks like a win, and quietly returns less than the margin that
justified taking it. Long exits round up, short exits round down. The realised
figures below are computed at the rounded price, not the exact one, so what they
report is what the exchange can actually deliver.

**Both directions are priced, and neither is selected here.** A ``Direction``
enum exists in ``scanner/models.py``, and importing it would invert the
``scanner -> costs`` dependency and break the stdlib-only contract that
``costs/__init__.py`` makes. Exposing both exits costs one extra field and keeps
this module free to be imported by the scanner, by replay, and by the risk
engine without any of them dragging in the others. It also mirrors a fact from
``model.py``: cost itself is direction-symmetric, so there is nothing to choose
between until a rule has picked a side.

**What these numbers are worth.** Everything inherits the standing caveat from
``model.py``: the rates are transcribed, not reconciled against a contract note,
and the spread is not modelled at all. The exit price is therefore a *floor* —
the price below which the trade certainly does not pay — rather than a
prediction of what will be realised.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from ai_trader.costs.model import GROWW_INTRADAY_EQUITY, CostModel, RoundTripCost

FIXED_CLIP_NOTIONAL = Decimal(100_000)
"""The stated clip: at least one lakh of turnover per entry.

A floor rather than a target. Whole shares rarely divide it exactly, so a real
entry turns over this or a little more; see ``SizingPolicy.quantity_for``.

A stated strategy parameter rather than a measured one, which is why it is a
named constant a caller passes in and not a default buried in ``SizingPolicy``.
"""

STATED_GROSS_TARGET = Decimal("0.002")
"""The stated exit: 0.2% above the buy price, before charges.

A *gross* move measured from the entry, not the profit that survives it. What
is kept is this less the round-trip cost at whatever actually fills, which at a
one-lakh clip is roughly 0.12%. Stating it this way round is the caller's
choice and it is the one the strategy was described in; ``from_gross_target``
converts it into the net margin the rest of this module works in.

Stated, not measured — the same standing as ``FIXED_CLIP_NOTIONAL``, and named
here for the same reason: so that the one place it can be changed is obvious.
"""

NSE_EQUITY_TICK = Decimal("0.05")
"""The common NSE cash-segment tick.

A published market fact, so defaulting it is legitimate on the same grounds as
the fee schedule in ``model.py`` — unlike a strategy threshold, it is not an
opinion. It is still not universally right: NSE quotes a finer tick on some
scrips, and the authoritative value is a per-instrument attribute. ``Instrument``
carries only an exchange and a trading symbol today, so nothing in this system
can look it up. Override the field when the real tick is known.
"""


def round_up_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """The lowest tick boundary at or above ``price``."""
    if tick <= 0:
        raise ValueError(f"tick must be positive, got {tick}")
    return (price / tick).to_integral_value(rounding=ROUND_CEILING) * tick


def round_down_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    """The highest tick boundary at or below ``price``."""
    if tick <= 0:
        raise ValueError(f"tick must be positive, got {tick}")
    return (price / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


@dataclass(frozen=True, slots=True)
class TradeCostEstimate:
    """One sized round trip, costed, with both exits priced.

    Every field is retained rather than derived on demand because this is the
    object a journal entry will be written from. A decision that recorded only
    its conclusion could never be audited afterwards: the question asked of a
    losing trade is not "what did it decide" but "what did it believe when it
    decided", and that is the entry price, the clip, the schedule and the margin
    that were in force.
    """

    entry_price: Decimal
    quantity: int
    notional: Decimal
    target_notional: Decimal
    tick_size: Decimal
    net_margin_fraction: Decimal
    required_gross_fraction: Decimal
    cost: RoundTripCost
    long_exit_price: Decimal
    short_exit_price: Decimal

    @property
    def target_net_rupees(self) -> Decimal:
        """What the margin is worth on this notional, before tick rounding."""
        return self.notional * self.net_margin_fraction

    @property
    def excess_notional(self) -> Decimal:
        """Turnover above the clip, forced by whole shares. Never negative.

        Zero only when the price divides the clip exactly. It is a few hundred
        rupees on an ordinary name and unbounded on a dear one: a share quoted
        at Rs 1,40,000 against a one-lakh clip deploys forty per cent more than
        the clip asked for, because one share is the smallest trade there is.
        That is a fact a position limit should be shown rather than one this
        module should decide about.
        """
        return self.notional - self.target_notional

    @property
    def tick_fraction(self) -> Decimal:
        """One tick as a fraction of the entry price.

        Worth reading beside ``net_margin_fraction``, because a tick is the
        smallest move that exists and a target finer than one is unreachable.
        The ratio moves inversely with price: at Rs 2,450 a five-paisa tick is
        two thousandths of a percent and rounding is a rounding error, while at
        Rs 100 it is a twentieth of a percent — half of a 0.1% target, so such a
        name cannot be exited anywhere near the intended margin and will
        systematically overshoot it. Cheap stocks are a coarser instrument for
        this strategy than expensive ones, and this is the number that says so.
        """
        return self.tick_size / self.entry_price

    @property
    def cash_outlay(self) -> Decimal:
        """Rupees committed at entry, charges included.

        Intraday positions are leveraged, so this is not margin required; it is
        the figure a position-limit rule should compare against, and the one
        that answers "how much of the book does this consume".
        """
        return self.notional + self.cost.total

    @property
    def long_net_rupees(self) -> Decimal:
        """Rupees kept if a long fills at ``long_exit_price``, after charges."""
        return self.quantity * (self.long_exit_price - self.entry_price) - (
            self.cost.total
        )

    @property
    def short_net_rupees(self) -> Decimal:
        """Rupees kept if a short covers at ``short_exit_price``, after charges."""
        return self.quantity * (self.entry_price - self.short_exit_price) - (
            self.cost.total
        )


@dataclass(frozen=True, slots=True)
class SizingPolicy:
    """A fixed clip, a target margin, a fee schedule and a tick.

    The four things a trade decision cannot be made without, held together so
    that a result can be reproduced from the policy that produced it. Sizing is
    deterministic here and nowhere else: ``AGENTS.md`` rule 8 puts position
    sizing in code rather than in the model, and an LLM that is handed an
    already-sized candidate has no way to size it differently.
    """

    target_notional: Decimal
    net_margin_fraction: Decimal
    costs: CostModel = GROWW_INTRADAY_EQUITY
    tick_size: Decimal = NSE_EQUITY_TICK

    def __post_init__(self) -> None:
        if self.target_notional <= 0:
            raise ValueError(
                f"target_notional must be positive, got {self.target_notional}"
            )
        if self.net_margin_fraction <= 0:
            raise ValueError(
                f"net_margin_fraction must be positive, got {self.net_margin_fraction}"
            )
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")

    @classmethod
    def from_gross_target(
        cls,
        *,
        target_notional: Decimal,
        gross_target_fraction: Decimal,
        costs: CostModel = GROWW_INTRADAY_EQUITY,
        tick_size: Decimal = NSE_EQUITY_TICK,
    ) -> SizingPolicy:
        """State the exit as a move from the entry, not as profit kept.

        "Sell at 0.2% above the buy price" and "keep 0.2% after charges" are
        different instructions — at a one-lakh clip they differ by about 40% of
        the smaller one — and this module works in the second. A caller who
        thinks in the first should convert here rather than subtracting a cost
        fraction by hand at the call site, where the notional it was computed at
        would go unrecorded.

        The conversion is done at ``target_notional``, which is the *smallest*
        fill the clip permits and therefore the most expensive one as a
        fraction. Every real fill rounds up to a whole share, turns over at
        least this, and pays at most this fraction, so the hurdle a name
        actually faces comes out at or below the stated target and what it keeps
        comes out at or above the implied margin. The error is in the safe
        direction by construction rather than by luck.

        A target that does not survive its own costs is refused. At small clips
        that is a real case and not an edge one: a 0.2% target on a
        twenty-thousand-rupee clip is a 0.07% loss, and silently continuing with
        a negative margin would price every exit below the entry.
        """
        net_margin = costs.net_fraction(target_notional, gross_target_fraction)
        if net_margin <= 0:
            raise ValueError(
                f"gross target {gross_target_fraction} does not clear costs at "
                f"notional {target_notional}: it leaves {net_margin}"
            )
        return cls(
            target_notional=target_notional,
            net_margin_fraction=net_margin,
            costs=costs,
            tick_size=tick_size,
        )

    def quantity_for(self, entry_price: Decimal) -> int:
        """The fewest whole shares worth at least the clip. Never zero.

        Rounded up, never down. The clip is the minimum a leg should turn over,
        so a price that does not divide it leaves the trade slightly larger than
        stated rather than slightly smaller — and undershooting is the worse of
        the two errors, because cost is a fraction of what fills and the reason
        a clip is stated at all is that the fraction is only tolerable at size.

        The ceiling is taken by comparison rather than by a rounding mode, so the
        postcondition ``quantity * entry_price >= target_notional`` holds exactly
        under whatever decimal context the caller installed. A division rounded a
        hair low cannot silently return a share count that falls short.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        quantity = int(self.target_notional // entry_price)
        if quantity * entry_price < self.target_notional:
            quantity += 1
        return quantity

    def estimate(self, entry_price: Decimal) -> TradeCostEstimate:
        """Size a trade at this price and cost both of its legs.

        Sizing cannot fail. ``quantity_for`` always answers at least one share,
        so the notional that every fraction below divides by is always positive
        and a stock quoted above the whole clip is an ordinary case rather than
        an error several fields away from its cause.
        """
        quantity = self.quantity_for(entry_price)
        notional = entry_price * quantity
        required = self.costs.required_gross_fraction(
            notional, self.net_margin_fraction
        )

        # Away from entry in both cases. See the module docstring: rounding to
        # the nearer tick is what silently under-delivers the margin.
        long_exit = round_up_to_tick(entry_price * (1 + required), self.tick_size)
        short_exit = round_down_to_tick(entry_price * (1 - required), self.tick_size)
        if short_exit <= 0:
            raise ValueError(
                f"required gross move {required} leaves no positive short exit "
                f"at {entry_price}"
            )

        return TradeCostEstimate(
            entry_price=entry_price,
            quantity=quantity,
            notional=notional,
            target_notional=self.target_notional,
            tick_size=self.tick_size,
            net_margin_fraction=self.net_margin_fraction,
            required_gross_fraction=required,
            cost=self.costs.round_trip(notional),
            long_exit_price=long_exit,
            short_exit_price=short_exit,
        )


__all__ = [
    "FIXED_CLIP_NOTIONAL",
    "NSE_EQUITY_TICK",
    "STATED_GROSS_TARGET",
    "SizingPolicy",
    "TradeCostEstimate",
    "round_down_to_tick",
    "round_up_to_tick",
]
