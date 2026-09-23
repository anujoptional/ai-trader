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

What this screen still cannot see is the spread, and the spread is frequently
the largest cost of all at this horizon. No layer produces one yet. When the
reserved microstructure fields on ``MarketContext`` arrive, they belong here —
a candidate whose required move is smaller than its spread should be dropped by
this function, for the same reason and in the same place.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ai_trader.costs import GROWW_INTRADAY_EQUITY, CostModel
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

    ``notional`` is one leg's turnover, and it is a precondition rather than a
    detail. Round-trip cost is about 0.27% of a twenty-thousand-rupee clip and
    about 0.08% of a one-lakh clip, so the very same strategy is a loser at one
    size and a winner at the other. The scanner cannot choose this, because
    position sizing is the risk engine's job under ``AGENTS.md``; what it can do
    is refuse to show the AI names that the configured size cannot pay for.

    ``max_atr_multiple`` is the assumption with the least evidence behind it, so
    it is required rather than defaulted. It reads: *the required move is
    plausible if it is no more than this many one-minute ATRs.* Note the unit —
    ``atr_pct`` is a fourteen-period average true range over **one-minute**
    candles, so a multiple of three is a small intrabar move and a multiple of
    thirty is most of a session. Replay is what will replace the guess.
    """

    notional: Decimal
    net_margin_fraction: Decimal
    max_atr_multiple: Decimal
    costs: CostModel = GROWW_INTRADAY_EQUITY
    min_minutes_remaining: Decimal | None = None
    square_off_minutes_since_open: Decimal = SESSION_MINUTES

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError(f"notional must be positive, got {self.notional}")
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

    @property
    def required_gross_fraction(self) -> Decimal:
        """The move a round trip of this size must make to be worth taking.

        Cycle-constant: it depends on the policy alone, not on any instrument.
        Computed on access rather than cached at construction so that it is
        evaluated under whatever decimal context the caller installed — the
        scanner runs the whole cycle inside ``FEATURE_CONTEXT``, and a value
        cached outside it could have been rounded differently from every
        feature it is compared against.
        """
        return self.costs.required_gross_fraction(
            self.notional, self.net_margin_fraction
        )

    def evaluate(self, snapshot: FeatureSnapshot) -> FeasibilityCheck:
        """Screen one name, recording the working whether it passes or fails.

        Fails closed on every unknown. A missing ``atr_pct`` is not treated as
        "probably fine" — the entire point of reading through the readiness flag
        is that an absent feature is absent, and a screen that waves through
        what it could not measure is not a screen.
        """
        required = self.required_gross_fraction

        minutes = available(snapshot, "minutes_since_session_open")
        remaining = (
            None if minutes is None else self.square_off_minutes_since_open - minutes
        )

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
