"""The cost screen: can this name pay for the round trip at all?

Every rule in ``rules.py`` answers "is this name set up to move?". None of them
answers "is the move big enough to be worth the fees?", and at a one-minute
horizon that second question disqualifies more names than the first. A stock
that is beautifully poised but whose whole day's range is narrower than the
round-trip cost is not a candidate, however many rules fire on it.

So this runs **before** the rules, not after. A name that cannot clear the
hurdle is dropped without evaluating anything, which is both cheaper and more
honest: the alternative is scoring a setup that could never have been taken and
letting it occupy part of the candidate budget.

**It is a filter, never a term in the score.** Folding reachability into the
ranking would mean choosing how many points a spare basis point of ATR is worth
against a point of trend strength, and there is nothing behind such a number.
Whether headroom *should* also influence rank is a real question — it is a
question for replay, which can measure it, rather than for a constant invented
here.

**The two multiples below are placeholders.** Section 7 is explicit that an
invented threshold is not evidence, so ``max_atr_multiple`` has no default at
all: a caller must state the assumption it is making. ``min_minutes_remaining``
defaults to ``None``, meaning the time gate is off, on the same reasoning that
leaves the session window unbounded in ``ScannerConfig``.

**The hurdle is sized per name, not once per cycle.** A fixed clip buys whole
shares, so what fills is the clip rounded up to the next whole share — never
exactly the clip, and on a dear name substantially more. Every cost here is a
fraction *of what filled*, and brokerage is capped per leg, so a larger fill
pays a smaller fraction: a share quoted at Rs 1,40,000 turns over forty per cent
above a one-lakh clip and clears a materially lower hurdle than the clip figure
says. Screening every name against the clip figure would therefore be too
strict rather than too loose — rejecting names the buying model can in fact
afford. So the screen sizes at the snapshot's close, through the same
``SizingPolicy`` the decision will use, and prices the hurdle on the notional
that would actually fill.

**There is no reachability case about price.** The clip is a floor on turnover,
so a single share always clears it and every name is sizeable. The screen never
refuses a name for being too expensive to buy; what it still refuses is a name
that cannot plausibly move far enough, which is a judgement about the market
rather than about the configured size.

What this screen still cannot see is the spread, and the spread is frequently
the largest cost of all at this horizon. No layer produces one yet. When the
reserved microstructure fields on ``MarketContext`` arrive, they belong here —
a candidate whose required move is smaller than its spread should be dropped by
this function, for the same reason and in the same place.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ai_trader.costs import GROWW_INTRADAY_EQUITY, CostModel, SizingPolicy
from ai_trader.features import FeatureSnapshot
from ai_trader.scanner.models import FeasibilityCheck, FeasibilityReason
from ai_trader.scanner.rules import available

SESSION_MINUTES = Decimal(375)
"""Minutes from 09:15 to 15:30, the full NSE equity session.

Duplicated deliberately rather than imported from ``market.state``, where the
same figure bounds a candle ring buffer. That one is a storage limit and this
one is a trading horizon; they agree today, and coupling them would mean a
change to either silently moves the other.
"""


@dataclass(frozen=True, slots=True)
class FeasibilityPolicy:
    """What the caller is trying to earn, and what it will assume to get there.

    Every field is a strategy parameter the caller owns. Section 7 says the
    universe is a strategy parameter and must be stated explicitly rather than
    implied; the same is true of the clip size and the margin, and for the same
    reason — a result is meaningless unless the conditions that produced it were
    recorded alongside it.

    ``target_notional`` is one leg's intended turnover, and it is a precondition
    rather than a detail. At Groww's rates, round-trip cost is about 0.27% of a
    twenty-thousand-rupee clip and about 0.08% of a one-lakh clip, so the very
    same strategy is a loser at one size and a winner at the other. The scanner
    cannot choose this, because position sizing is the risk engine's job under
    ``AGENTS.md``; what it can do is refuse to show the AI names that the
    configured size cannot pay for. It is a *floor*: whole shares rarely consume
    a clip exactly, so the notional a trade fills is this figure or a little
    more, never less.

    ``costs`` is which broker's schedule those figures come from, and the sizes
    above are why it is a field rather than an import. It defaults to Groww
    because that is the broker this system connects to; the other published
    schedule is ``ZERODHA_INTRADAY_EQUITY``. At a one-lakh clip the two agree to
    the paisa — both cap brokerage at Rs 20 a leg — so the choice changes
    nothing at the configured size and a great deal below it.

    ``max_atr_multiple`` is the assumption with the least evidence behind it, so
    it is required rather than defaulted. It reads: *the required move is
    plausible if it is no more than this many one-minute ATRs.* Note the unit —
    ``atr_pct`` is a fourteen-period average true range over **one-minute**
    candles, so a multiple of three is a small intrabar move and a multiple of
    thirty is most of a session. Replay is what will replace the guess.
    """

    target_notional: Decimal
    net_margin_fraction: Decimal
    max_atr_multiple: Decimal
    costs: CostModel = GROWW_INTRADAY_EQUITY
    min_minutes_remaining: Decimal | None = None
    square_off_minutes_since_open: Decimal = SESSION_MINUTES

    def __post_init__(self) -> None:
        if self.target_notional <= 0:
            raise ValueError(
                f"target_notional must be positive, got {self.target_notional}"
            )
        if self.net_margin_fraction <= 0:
            raise ValueError(
                f"net_margin_fraction must be positive, got {self.net_margin_fraction}"
            )
        if self.max_atr_multiple <= 0:
            raise ValueError(
                f"max_atr_multiple must be positive, got {self.max_atr_multiple}"
            )
        if self.min_minutes_remaining is not None and self.min_minutes_remaining < 0:
            raise ValueError(
                "min_minutes_remaining cannot be negative, got "
                f"{self.min_minutes_remaining}"
            )
        if self.square_off_minutes_since_open <= 0:
            raise ValueError(
                "square_off_minutes_since_open must be positive, got "
                f"{self.square_off_minutes_since_open}"
            )

    @classmethod
    def from_gross_target(
        cls,
        *,
        target_notional: Decimal,
        gross_target_fraction: Decimal,
        max_atr_multiple: Decimal,
        costs: CostModel = GROWW_INTRADAY_EQUITY,
        min_minutes_remaining: Decimal | None = None,
        square_off_minutes_since_open: Decimal = SESSION_MINUTES,
    ) -> FeasibilityPolicy:
        """Configure the screen by the exit, not by the profit kept.

        "Sell 0.2% above the buy price" is how the objective was stated, and it
        is not the same instruction as "keep 0.2% after charges" — the first is
        a gross move and the second a net one, differing by the whole round-trip
        cost. ``SizingPolicy.from_gross_target`` performs the conversion and
        refuses a target its own costs consume; this exists so the scanner can
        be configured in the terms the strategy was written in rather than in
        the terms the arithmetic happens to use.

        Note what does *not* follow. Converting once at the clip fixes the
        margin, not the hurdle: the screen still prices per name, and a name
        that fills above the clip pays a smaller cost fraction and therefore
        clears a hurdle slightly under the stated target. The stated figure is
        an upper bound on what any name here is asked to move.
        """
        implied = SizingPolicy.from_gross_target(
            target_notional=target_notional,
            gross_target_fraction=gross_target_fraction,
            costs=costs,
        )
        return cls(
            target_notional=target_notional,
            net_margin_fraction=implied.net_margin_fraction,
            max_atr_multiple=max_atr_multiple,
            costs=costs,
            min_minutes_remaining=min_minutes_remaining,
            square_off_minutes_since_open=square_off_minutes_since_open,
        )

    @property
    def sizing(self) -> SizingPolicy:
        """The decision-time sizer this screen is obliged to agree with.

        Derived from the fields already here rather than held beside them. The
        screen exists to reject what a trade could not do, so a screen that
        sized differently from the trade would be measuring something else;
        deriving it makes disagreement impossible rather than merely unlikely.
        The tick is left at its default because the screen never prices an exit
        — it needs the share count and nothing further.
        """
        return SizingPolicy(
            target_notional=self.target_notional,
            net_margin_fraction=self.net_margin_fraction,
            costs=self.costs,
        )

    @property
    def required_gross_fraction(self) -> Decimal:
        """The hurdle at exactly the clip — the strictest case.

        A ceiling rather than the figure any particular name faces, and reached
        only when the price happens to divide the clip exactly. Every other
        quote rounds up to the next whole share, turns over slightly more than
        the clip, and therefore pays a slightly smaller fraction; use
        ``required_gross_fraction_at`` for a name. This is kept because it is
        the one number that describes the policy itself, and reporting a policy
        needs one.

        Computed on access rather than cached at construction so that it is
        evaluated under whatever decimal context the caller installed — the
        scanner runs the whole cycle inside ``FEATURE_CONTEXT``, and a value
        cached outside it could have been rounded differently from every
        feature it is compared against.
        """
        return self.costs.required_gross_fraction(
            self.target_notional, self.net_margin_fraction
        )

    def required_gross_fraction_at(self, price: Decimal) -> Decimal:
        """The hurdle a trade in this name actually faces.

        Always at or below ``required_gross_fraction``, never above it: the clip
        is a floor, so the fill is never smaller than the clip and the fraction
        is never larger. The share count comes from ``SizingPolicy`` rather than
        from arithmetic repeated here, so the screen and the trade cannot
        disagree about what a quote buys.
        """
        quantity = self.sizing.quantity_for(price)
        return self.costs.required_gross_fraction(
            price * quantity, self.net_margin_fraction
        )

    def evaluate(self, snapshot: FeatureSnapshot) -> FeasibilityCheck:
        """Screen one name, recording the working whether it passes or fails.

        Fails closed on every unknown. A missing ``atr_pct`` is not treated as
        "probably fine" — the entire point of reading through the readiness flag
        is that an absent feature is absent, and a screen that waves through
        what it could not measure is not a screen.

        The close is used as the entry price for sizing. It is the last traded
        price of the candle the rules are about to read, so it is the same
        instant they are reasoning about; anything else would screen one moment
        and score another. It is sized unguarded because ``Candle`` already
        refuses a non-positive price on construction, so a zero close cannot
        reach here through any path that begins at a real candle; re-checking it
        would be a second implementation of that rule, free to disagree.
        """
        minutes = available(snapshot, "minutes_since_session_open")
        remaining = (
            None if minutes is None else self.square_off_minutes_since_open - minutes
        )

        required = self.required_gross_fraction_at(snapshot.close)

        atr_fraction = available(snapshot, "atr_pct")
        if atr_fraction is None:
            return FeasibilityCheck(
                required_gross_fraction=required,
                minutes_remaining=remaining,
                reason=FeasibilityReason.VOLATILITY_UNKNOWN,
            )

        # Multiplied, never divided: a zero ATR is a real reading on a stock
        # that has not moved, and dividing by it to get "how many ATRs away"
        # would raise on exactly the names this test is meant to reject.
        if required > self.max_atr_multiple * atr_fraction:
            return FeasibilityCheck(
                required_gross_fraction=required,
                atr_fraction=atr_fraction,
                minutes_remaining=remaining,
                reason=FeasibilityReason.VOLATILITY_TOO_LOW,
            )

        if self.min_minutes_remaining is not None:
            if remaining is None:
                return FeasibilityCheck(
                    required_gross_fraction=required,
                    atr_fraction=atr_fraction,
                    reason=FeasibilityReason.CLOCK_UNKNOWN,
                )
            if remaining < self.min_minutes_remaining:
                return FeasibilityCheck(
                    required_gross_fraction=required,
                    atr_fraction=atr_fraction,
                    minutes_remaining=remaining,
                    reason=FeasibilityReason.SESSION_TOO_SHORT,
                )

        return FeasibilityCheck(
            required_gross_fraction=required,
            atr_fraction=atr_fraction,
            minutes_remaining=remaining,
        )


__all__ = ["SESSION_MINUTES", "FeasibilityPolicy"]
