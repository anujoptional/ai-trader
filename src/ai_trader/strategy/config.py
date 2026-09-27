"""One object that says what the strategy is, so two runs cannot disagree.

**The problem this exists to solve.** ``target_notional`` could be set in three
places, ``costs`` in four, and ``rules`` in two, and nothing checked that they
matched. The concrete failure is not hypothetical: a feasibility screen
configured at a one-lakh clip prices the round-trip hurdle at about 0.083%,
while a sizer configured at twenty thousand really pays about 0.272%. A scanner
wired that way shows the AI names it has screened as affordable and then takes
them at a size that cannot pay for them, three times over. The report would
blame the strategy. The cause would be two config sites disagreeing.

**So the fork happens after the strategy, not before it.** Replay, shadow and
live all build their scanner from one ``StrategyConfig`` by calling one
constructor -- not by building equal configs, which is a property somebody has
to keep true, but by calling the same code, which is a property that cannot
come apart. ``ReplayConfig`` adds a ``FillModel``, a universe and a date range,
and those really are replay-only: a fill model exists because replay has to
guess what live trading observes.

**Every field has a named default, and that is the safety argument.** A required
argument looks stricter and is in fact weaker for this purpose, because a
required argument can be handed two different values at two call sites while a
shared constant cannot drift. The defaults are conventions, not measurements,
and each one says so at its definition. The point of naming them is that the
number appears once, in a place that explains itself, instead of appearing
twice in two callers' keyword arguments.

**Derived, not restated.** ``gross_target_fraction`` is the term the strategy
was described in -- "sell 0.2% above the buy" -- and the net margin the sizer
and the screen both need is computed from it exactly once, here, and handed to
both. Neither can be given a different one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from ai_trader.costs import (
    FIXED_CLIP_NOTIONAL,
    GROWW_INTRADAY_EQUITY,
    NSE_EQUITY_TICK,
    STATED_GROSS_TARGET,
    ZERODHA_INTRADAY_EQUITY,
    CostModel,
    SizingPolicy,
)
from ai_trader.market import ONE_MINUTE, exact_timedelta
from ai_trader.scanner import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RULES,
    FeasibilityPolicy,
    Rule,
    Scanner,
    ScannerConfig,
)
from ai_trader.strategy.exits import DEFAULT_EXIT_POLICY, ExitPolicy

DEFAULT_MAX_ATR_MULTIPLE = Decimal(3)
"""How many one-minute ATRs the required move may be before a name is dropped.

Reads as: *the hurdle should be reachable in about three minutes of this name's
typical movement.* At a one-lakh clip the round-trip hurdle is roughly 0.083% of
notional, and a liquid NSE large cap's one-minute ATR sits in the region of
0.05%--0.15% of price, so a multiple of three puts the cut somewhere inside that
band rather than above or below all of it -- the screen bites on the quietest
names and passes the rest. A multiple that rejected everything or nothing would
be a screen in name only, and that is the whole of the justification: it is
calibration against the cost arithmetic, not evidence about what trades well.

``FeasibilityPolicy`` deliberately gives this field no default of its own, on
the grounds that a caller must state the assumption. That is still right for a
caller reaching for the screen directly. Stating it *once here* is the stronger
version of the same rule: the assumption is named, explained, and impossible to
set differently in replay than in live.
"""

DEFAULT_MAX_OPEN_POSITIONS = 3
"""Held positions the book will carry at once.

Smaller than the five-candidate budget on purpose. If the book could hold every
candidate the scanner returns, the ranking would never be consulted and the
score would be decoration; making capacity the binding constraint is what turns
rank into a decision the replay can be asked about. Three at a one-lakh clip is
about three lakh of gross intraday exposure.
"""

DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN = Decimal(360)
"""15:15 -- six hours after the 09:15 open, fifteen minutes before the close.

Intraday positions are closed by the broker if the account does not close them
first, and a broker's own auto square-off runs somewhere in the last part of the
session at a price the account does not choose. Being early is therefore the
safe side of the only question this number really asks. It is a parameter to
confirm against the broker's published cutoff, not a researched quantity, and it
is set here rather than left unset because ``None`` -- running to the last bar --
is the one setting that is definitely wrong for an intraday strategy.

When it is set, entries stop at the same moment as well, so that the last
minutes of a session are not spent opening positions that will be closed
minutes later at market.
"""


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """What the strategy is: size, costs, rules, screen, exits, book limits.

    Everything downstream is derived from this and nothing downstream may state
    any of it again. A ``ReplayResult`` cites the instance that produced it, so
    the run is reconstructible in the sense Section 7.2 requires.

    ``screen_feasibility`` exists so that the sweep can measure what the cost
    screen costs. Turning it off shows the scanner names it believes cannot pay
    for themselves, which is not a way to trade but is the only way to find out
    whether the screen is rejecting names that would have worked.
    """

    target_notional: Decimal = FIXED_CLIP_NOTIONAL
    gross_target_fraction: Decimal = STATED_GROSS_TARGET
    costs: CostModel = GROWW_INTRADAY_EQUITY
    tick_size: Decimal = NSE_EQUITY_TICK

    rules: Sequence[Rule] = DEFAULT_RULES
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    earliest_minutes_since_open: Decimal | None = None
    latest_minutes_since_open: Decimal | None = None

    screen_feasibility: bool = True
    max_atr_multiple: Decimal = DEFAULT_MAX_ATR_MULTIPLE
    min_minutes_remaining: Decimal | None = None

    exit_policy: ExitPolicy = DEFAULT_EXIT_POLICY
    max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS
    cooldown_minutes: Decimal = Decimal(0)
    square_off_minutes_since_open: Decimal = DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError(
                f"max_candidates must be positive, got {self.max_candidates}"
            )
        if self.max_open_positions <= 0:
            raise ValueError(
                f"max_open_positions must be positive, got {self.max_open_positions}"
            )
        if self.cooldown_minutes < 0:
            raise ValueError(
                f"cooldown_minutes cannot be negative, got {self.cooldown_minutes}"
            )
        if self.square_off_minutes_since_open <= 0:
            raise ValueError(
                "square_off_minutes_since_open must be positive, got "
                f"{self.square_off_minutes_since_open}"
            )
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        # Same discard-the-result move as the sizer below, for the same reason.
        # A cooldown that is not a whole number of microseconds cannot be the
        # duration the run reports, and this object is the one both paths read,
        # so refusing it here refuses it for the live path too rather than only
        # for whichever replay happens to convert it first.
        exact_timedelta(self.cooldown_minutes, ONE_MINUTE, name="cooldown_minutes")
        # Build the sizer now and discard it. ``from_gross_target`` refuses a
        # target its own costs would consume, and that refusal belongs at the
        # moment the configuration is written rather than several steps into a
        # run that has already fetched a year of candles.
        self.sizing_policy()

    @property
    def net_margin_fraction(self) -> Decimal:
        """What the strategy keeps, after charges, on a clip-sized round trip.

        Derived from ``gross_target_fraction`` rather than stated beside it.
        Stating both would allow a pair that does not satisfy its own arithmetic,
        and would leave no answer to the question of which one the strategy
        actually meant.
        """
        return self.sizing_policy().net_margin_fraction

    def sizing_policy(self) -> SizingPolicy:
        """The sizer the book uses to buy shares and place targets."""
        return SizingPolicy.from_gross_target(
            target_notional=self.target_notional,
            gross_target_fraction=self.gross_target_fraction,
            costs=self.costs,
            tick_size=self.tick_size,
        )

    def feasibility_policy(self) -> FeasibilityPolicy | None:
        """The cost screen, sized by the same sizer the trade will use.

        ``net_margin_fraction`` is taken from ``sizing_policy`` rather than
        recomputed, which is the specific guarantee this module exists to give:
        the screen's hurdle and the trade's target are two readings of one
        number. ``FeasibilityPolicy.from_gross_target`` would derive the same
        value today; going through the sizer means it still would if either
        conversion were ever changed.

        Returns ``None`` when the screen is off, which is what ``ScannerConfig``
        already understands to mean "do not screen".
        """
        if not self.screen_feasibility:
            return None
        return FeasibilityPolicy(
            target_notional=self.target_notional,
            net_margin_fraction=self.sizing_policy().net_margin_fraction,
            max_atr_multiple=self.max_atr_multiple,
            costs=self.costs,
            min_minutes_remaining=self.min_minutes_remaining,
            square_off_minutes_since_open=self.square_off_minutes_since_open,
        )

    def scanner_config(self) -> ScannerConfig:
        return ScannerConfig(
            max_candidates=self.max_candidates,
            earliest_minutes_since_open=self.earliest_minutes_since_open,
            latest_minutes_since_open=self.latest_minutes_since_open,
            feasibility=self.feasibility_policy(),
        )

    def scanner(self) -> Scanner:
        """The scanner itself, so that every caller builds it the same way.

        This method is the load-bearing one. Replay and live could each build a
        ``ScannerConfig`` from the fields above and would almost certainly build
        equal ones -- but "almost certainly equal" is a property somebody has to
        keep true as the code changes. Calling one constructor is a property that
        cannot come apart, and it is what makes the claim in Section 7.1 --
        replay sees exactly what the AI sees -- checkable rather than aspirational.
        """
        return Scanner(self.scanner_config(), self.rules)

    def _costs_label(self) -> str:
        """Which schedule, by name where the name is one this system publishes.

        ``CostModel`` carries no name field, deliberately -- it is a bundle of
        rates, and two schedules that agree on every rate are the same schedule
        whatever either is called. So the label is recovered by comparison, and
        anything unrecognised is described by the only terms that actually vary
        between brokers: brokerage is commercial, the rest is statute or tariff.
        """
        if self.costs == GROWW_INTRADAY_EQUITY:
            return "Groww intraday equity"
        if self.costs == ZERODHA_INTRADAY_EQUITY:
            return "Zerodha intraday equity"
        return (
            f"custom, brokerage {self.costs.brokerage_fraction:.4%} "
            f"capped at Rs {self.costs.brokerage_cap}"
        )

    def describe(self) -> tuple[str, ...]:
        """The resolved configuration, one line each, for a report header.

        Resolved rather than declared. The net margin and the hurdle are the
        derived figures the run truly used, so a reader comparing two reports
        does not have to redo either conversion to see what differed -- which is
        the whole use of a header on a sweep.
        """
        sizing = self.sizing_policy()
        hurdle = self.gross_target_fraction - sizing.net_margin_fraction
        screen = (
            f"{self.max_atr_multiple}x one-minute ATR"
            if self.screen_feasibility
            else "off"
        )
        return (
            f"clip                 Rs {self.target_notional:,}",
            f"gross target         {self.gross_target_fraction:.3%}",
            f"cost hurdle          {hurdle:.3%} (derived)",
            f"net margin           {sizing.net_margin_fraction:.3%} (derived)",
            f"costs                {self._costs_label()}",
            f"rules                {len(self.rules)}, top {self.max_candidates}",
            f"cost screen          {screen}",
            f"exit                 {self.exit_policy.description}",
            f"book                 {self.max_open_positions} positions",
            f"square-off           {self.square_off_minutes_since_open} min after open",
        )


__all__ = [
    "DEFAULT_MAX_ATR_MULTIPLE",
    "DEFAULT_MAX_OPEN_POSITIONS",
    "DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN",
    "StrategyConfig",
]
