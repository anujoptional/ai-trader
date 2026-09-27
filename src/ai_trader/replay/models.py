"""What a simulated round trip is, and what a run of them reports.

Three objects, in the order a trade meets them. ``FillModel`` is every
assumption about *how* an order becomes a price — how late it arrives, what the
spread takes, what the size costs. ``SimulatedTrade`` is one completed round
trip with its working shown. ``ReplayResult`` is a whole run, and the metrics
Section 4.5 asks for are properties on it rather than a separate report, so that
no summary can be computed from anything other than the trades it claims to
summarise.

**Nothing here is defaulted to zero.** ``FillModel`` has three required fields,
following the precedent ``FeasibilityPolicy.max_atr_multiple`` set: an
assumption with no measurement behind it must be stated by the caller rather
than inherited from a constructor. Zero friction is obtainable — it is
``FRICTIONLESS`` — but only by naming it, because a silent zero is the single
most flattering thing a backtest can assume and the one a reader is least
likely to notice. Section 7.2 calls an invented threshold "not evidence"; an
*unstated* one is worse.

**Friction always moves against the trade, and which way that is flips between
the legs.** A long pays up to get in and sells down to get out; a short is the
mirror. The same rule governs tick rounding, and for the reason
``costs/sizing.py`` gives at length: rounding to the *nearer* tick leaves a
target a shade under the price that pays for the round trip, so the trade fills,
looks like a win, and returns less than the margin that justified it. One
principle covers both — every rounding in replay moves the price the way that
costs the trade money.

**The exception is the target, and it is an exception for a reason.** A target
is a resting limit order: it fills at its own price or not at all, so it pays no
spread and needs no rounding, having been placed on a tick boundary by
``SizingPolicy.estimate``. A stop and a square-off are market orders and pay
everything. Modelling all three the same way would either charge a limit order
for liquidity it supplied or let a market order cross the spread for free.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType

from ai_trader.broker import Instrument
from ai_trader.costs import RoundTripCost, round_down_to_tick, round_up_to_tick
from ai_trader.market import INDIA_TIMEZONE
from ai_trader.scanner import Direction, SuppressionReason


class ExitReason(StrEnum):
    """Why a simulated position was closed.

    Three exits and no fourth. There is no discretionary close, because there is
    nothing in this system that could exercise discretion, and no trailing stop,
    because a trailing stop is a risk policy and the risk layer does not exist
    yet. When it does, its exits belong here and the replay that measured them
    will be comparable with this one only if the addition is visible.
    """

    TARGET = "target"
    STOP = "stop"
    SESSION_END = "session_end"


@dataclass(frozen=True, slots=True)
class FillModel:
    """The gap between the price a signal saw and the price a trade got.

    Section 4.5 is blunt about this: fill assumptions are "a larger source of
    replay error than anything in the feature layer". A one-minute strategy
    aiming to keep about 0.12% of a 0.2% move has roughly twelve basis points of
    room, and a single basis point of spread plus a basis point of slippage is a
    sixth of it. The feature arithmetic is exact to twenty-eight significant
    figures; this object is where the real uncertainty lives, and it is a
    parameter precisely so that a run can be repeated across a sweep of it and
    the result read as a surface rather than a number.

    ``latency_seconds`` is the delay between a candle closing and an order
    filling — the scan, the model call, the risk check, the round trip to the
    exchange. Zero means the fill happens at the signal candle's close, which is
    the assumption every naive backtest makes and no live system achieves.

    ``half_spread_fraction`` is what crossing the book costs one leg, as a
    fraction of price. Half, not whole, because a round trip crosses twice and
    this is charged on each leg. No layer in this system can measure it yet —
    ``MarketContext`` reserves the field and nothing fills it — so it is an
    assumption in the strict sense and stays the caller's to state.

    ``slippage_fraction`` is everything else that moves the price adversely
    between the decision and the fill: depth consumed by the order's own size,
    and the drift of a market that does not wait. Kept separate from the spread
    because they scale differently — the spread is a property of the book and
    slippage is a property of the order — and a sweep that moved them together
    could not tell which one the result was sensitive to.

    ``resolve_ambiguous_bar_as`` decides the classic backtest lie. When a bar's
    high reaches the target *and* its low reaches the stop, the candle cannot
    say which came first, and assuming the favourable one is how a losing
    strategy is made to look profitable. The default is the pessimistic reading;
    ``ReplayResult.ambiguous_exit_rate`` reports how often the choice was made
    at all, because at two per cent the assumption barely matters and at forty
    per cent the result is not a measurement of anything.
    """

    latency_seconds: Decimal
    half_spread_fraction: Decimal
    slippage_fraction: Decimal
    resolve_ambiguous_bar_as: ExitReason = ExitReason.STOP

    def __post_init__(self) -> None:
        if self.latency_seconds < 0:
            raise ValueError(
                f"latency_seconds cannot be negative, got {self.latency_seconds}"
            )
        if self.half_spread_fraction < 0:
            raise ValueError(
                "half_spread_fraction cannot be negative, got "
                f"{self.half_spread_fraction}"
            )
        if self.slippage_fraction < 0:
            raise ValueError(
                f"slippage_fraction cannot be negative, got {self.slippage_fraction}"
            )
        if self.adverse_fraction >= 1:
            raise ValueError(
                "half_spread_fraction plus slippage_fraction must be under 1, got "
                f"{self.adverse_fraction}"
            )
        if self.resolve_ambiguous_bar_as is ExitReason.SESSION_END:
            raise ValueError(
                "resolve_ambiguous_bar_as must be TARGET or STOP, got "
                f"{self.resolve_ambiguous_bar_as}"
            )

    @property
    def adverse_fraction(self) -> Decimal:
        """Everything that moves a market order against itself, per leg."""
        return self.half_spread_fraction + self.slippage_fraction

    def entry_price(
        self, reference: Decimal, direction: Direction, tick: Decimal
    ) -> Decimal:
        """What getting in actually costs, from the price the model quoted."""
        return self._against(reference, dearer=direction is Direction.LONG, tick=tick)

    def market_exit_price(
        self, reference: Decimal, direction: Direction, tick: Decimal
    ) -> Decimal:
        """What getting out actually yields, for a stop or a square-off.

        Not used for a target: see the module docstring. The direction test is
        inverted against ``entry_price`` — a long buys dear and sells cheap —
        which is the whole of the asymmetry, written once.
        """
        return self._against(reference, dearer=direction is Direction.SHORT, tick=tick)

    def _against(self, reference: Decimal, *, dearer: bool, tick: Decimal) -> Decimal:
        if dearer:
            return round_up_to_tick(reference * (1 + self.adverse_fraction), tick)
        adjusted = round_down_to_tick(reference * (1 - self.adverse_fraction), tick)
        if adjusted <= 0:
            raise ValueError(
                f"adverse fill of {self.adverse_fraction} leaves no positive price "
                f"at {reference}"
            )
        return adjusted


FRICTIONLESS = FillModel(
    latency_seconds=Decimal(0),
    half_spread_fraction=Decimal(0),
    slippage_fraction=Decimal(0),
)
"""No latency, no spread, no slippage — the upper bound on any result.

Named rather than defaulted so that a run which assumes it says so in its own
configuration. It is genuinely useful: it is the control a sweep is measured
against, and it is what the equivalence tests use, because two code paths can
only be compared for identity if nothing stochastic sits between them. It is not
a plausible trading assumption and nothing that reports a number from it should
describe that number as an expectation.
"""


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    """One completed round trip, with everything it believed at the time.

    Every input is retained rather than recomputed on demand, for the reason
    ``TradeCostEstimate`` gives: the question asked of a losing trade is not
    "what did it decide" but "what did it believe when it decided". A trade that
    stored only its outcome could never be re-examined under a different fill
    model, and re-examining trades under different fill models is most of what
    this layer is for.

    ``score`` and ``rules`` come from the ``Candidate`` and are carried through
    untouched, which is what makes ``ReplayResult.net_rupees_by_rule`` possible
    — and that mapping is the answer to the question ``DEFAULT_RULES`` asks
    about itself, namely which of the five earns its place.

    ``max_favourable_fraction`` and ``max_adverse_fraction`` are the excursions:
    the best and worst the position ever showed, measured on bar extremes rather
    than closes and signed so that favourable is always positive. They are how a
    target is judged as distinct from a strategy — a trade that reached 0.18% and
    stopped out is evidence about the exit, not about the entry.
    """

    instrument: Instrument
    direction: Direction
    session: date
    rules: tuple[str, ...]
    score: Decimal
    signal_time: datetime
    entry_time: datetime
    entry_price: Decimal
    quantity: int
    notional: Decimal
    target_price: Decimal
    stop_price: Decimal
    exit_time: datetime
    exit_price: Decimal
    exit_reason: ExitReason
    cost: RoundTripCost
    gross_fraction: Decimal
    net_fraction: Decimal
    net_rupees: Decimal
    max_favourable_fraction: Decimal
    max_adverse_fraction: Decimal
    bars_held: int
    ambiguous_exit: bool

    @property
    def is_win(self) -> bool:
        """Did the account keep anything? The only sense of "win" that pays."""
        return self.net_rupees > 0

    @property
    def is_gross_win(self) -> bool:
        """Did the price move the right way, charges ignored?

        Reported beside ``is_win`` rather than instead of it. Section 7.2 lists
        "a high hit rate that is gross rather than net" as a way this system
        could be wrong while looking right, and the cheapest guard against that
        is to publish both numbers so the gap between them is visible.
        """
        return self.gross_fraction > 0

    @property
    def gross_rupees(self) -> Decimal:
        """The price move in money, before any charge.

        Taken from ``gross_fraction`` rather than from the two prices, so that
        this and ``net_rupees`` are guaranteed to differ by exactly the cost.
        Recomputing it from the prices would be a second implementation of the
        same arithmetic, free to disagree with the first on a short.
        """
        return self.notional * self.gross_fraction

    @property
    def exit_notional(self) -> Decimal:
        """What the closing leg actually transacted.

        Not equal to ``notional`` unless the price finished where it started.
        The cost model charges both legs at the entry figure — deliberately, and
        for a reason it records — but turnover is not a cost estimate. It is a
        statement of what passed through the account, both legs are known exactly
        once the trade has closed, and a broker's own statement prices the
        closing leg at the closing price.
        """
        return self.exit_price * self.quantity

    @property
    def holding_minutes(self) -> Decimal:
        """Wall-clock minutes from fill to exit.

        Not the same as ``bars_held`` and the difference is informative: a live
        session has minutes with no ticks and therefore no candle, so a gap
        between the two counts is a gap in the tape.
        """
        return Decimal((self.exit_time - self.entry_time).total_seconds()) / 60


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """A whole run: its trades, its conditions, and what they add up to.

    The conditions are here because a result without them is not reproducible.
    Section 7.2 names the universe specifically — "the universe is a strategy
    parameter" and a result that does not record which names it ran over cannot
    be compared with any other result — so ``universe`` and ``sessions`` are
    fields rather than something the caller is trusted to remember.

    The counters beside the trades are the run's own accounting, and they should
    reconcile: every candidate either became a trade, was declined for a stated
    reason, or is still open at the end. A run whose numbers do not add up has a
    bug, and the arithmetic to notice that is available to anyone holding this
    object.

    ``unfilled_entries`` and ``unwound_entries`` are the two ways the tape
    running out swallows a decision, and they are separate because they mean
    different things. An unfilled entry was queued and never reached a bar to
    fill on — because the run ended, or because its *session* did, a queued
    entry being cancelled at the close rather than carried overnight. An unwound
    one *did* fill — on the very bar the session ended on — and so never lived
    through a bar it could be judged over; the book drops it rather than record
    a round trip whose entry and exit share a timestamp. A large
    ``unwound_entries`` says the run's square-off cutoff is missing or wrong,
    not that the strategy did anything.

    The two headline figures Section 4.5 asks for are
    ``net_expectancy_rupees`` and ``round_trips_per_session``. They are stated
    first among the properties because the strategy is explicitly a
    many-small-trades one, and at that design a fine expectancy on two trades a
    week is not a strategy and a hundred trades a day at a negative expectancy
    is a way to pay for someone else's holiday.
    """

    trades: tuple[SimulatedTrade, ...]
    universe: tuple[Instrument, ...]
    sessions: tuple[date, ...]
    fill: FillModel
    candles_replayed: int = 0
    candles_outside_session: int = 0
    """Bars skipped for starting outside 09:15-15:30, almost always the auction.

    Counted rather than silently dropped: a run that quietly replayed fewer bars
    than the fetch returned would hide a vendor change inside its own results.
    """
    cycles: int = 0
    candidates_seen: int = 0
    declined_book_full: int = 0
    declined_no_volatility: int = 0
    unfilled_entries: int = 0
    open_at_end: int = 0
    unwound_entries: int = 0
    suppressed: Mapping[SuppressionReason, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sealed = MappingProxyType(dict(self.suppressed))
        object.__setattr__(self, "suppressed", sealed)

    @property
    def round_trips(self) -> int:
        """How many completed trades. Read this before any average below."""
        return len(self.trades)

    @property
    def net_expectancy_rupees(self) -> Decimal:
        """Rupees kept per round trip, after every modelled charge.

        The first headline figure. Zero on an empty run rather than undefined,
        which is why ``round_trips`` is documented as the number to read first:
        an expectancy of zero from no trades and an expectancy of zero from four
        hundred trades are completely different findings and this property alone
        cannot distinguish them.
        """
        if not self.trades:
            return Decimal(0)
        return self.net_rupees / self.round_trips

    @property
    def round_trips_per_session(self) -> Decimal:
        """The second headline figure: how often the strategy finds anything."""
        if not self.sessions:
            return Decimal(0)
        return Decimal(self.round_trips) / Decimal(len(self.sessions))

    @property
    def net_rupees(self) -> Decimal:
        """What the account kept over the whole run."""
        return sum((trade.net_rupees for trade in self.trades), Decimal(0))

    @property
    def gross_rupees(self) -> Decimal:
        """What the price moves were worth before charges."""
        return sum((trade.gross_rupees for trade in self.trades), Decimal(0))

    @property
    def total_costs(self) -> Decimal:
        """Every charge the run paid. The gap between gross and net."""
        return sum((trade.cost.total for trade in self.trades), Decimal(0))

    @property
    def turnover(self) -> Decimal:
        """Rupees transacted across both legs of every trade.

        The denominator that makes costs comparable between runs of different
        sizes, and the figure a broker's own statement can be checked against.

        Each leg is priced at its own price, which is what makes that second
        claim true. The cost model charges both legs on the entry notional
        instead, and says why: it is invoked before the exit price is known, and
        in ``required_gross_fraction`` it is *solving for* the move, so the exit
        notional there depends on the answer. Neither applies here. A closed
        trade knows both prices exactly, nothing downstream feeds this figure
        back into cost arithmetic, and doubling the entry would report a number
        a statement would never show.
        """
        return sum(
            (trade.notional + trade.exit_notional for trade in self.trades),
            Decimal(0),
        )

    @property
    def net_hit_rate(self) -> Decimal:
        """Share of round trips that kept money."""
        return self._rate(sum(1 for trade in self.trades if trade.is_win))

    @property
    def gross_hit_rate(self) -> Decimal:
        """Share of round trips whose price moved the right way.

        Always at or above ``net_hit_rate``, and the distance between them is
        the share of trades that were right and still lost. On this strategy
        that distance is the whole design problem.
        """
        return self._rate(sum(1 for trade in self.trades if trade.is_gross_win))

    @property
    def mean_favourable_fraction(self) -> Decimal:
        """Average best-case excursion: how far trades got in their favour."""
        return self._mean(trade.max_favourable_fraction for trade in self.trades)

    @property
    def mean_adverse_fraction(self) -> Decimal:
        """Average worst-case excursion: how much heat trades took.

        Read against the stop multiple. If this sits far below the stop, the
        stop is not the thing ending trades and widening it will change little.
        """
        return self._mean(trade.max_adverse_fraction for trade in self.trades)

    @property
    def max_drawdown_rupees(self) -> Decimal:
        """Deepest fall from a peak on the cumulative net curve. Never negative.

        Ordered by exit time, which is when the money actually moved, rather
        than by entry. Positions that overlap in time are summed at their exits,
        so this understates the drawdown a live account would have shown on
        open positions — it is the realised curve, not the mark-to-market one.
        """
        peak = Decimal(0)
        running = Decimal(0)
        worst = Decimal(0)
        for trade in sorted(self.trades, key=lambda trade: trade.exit_time):
            running += trade.net_rupees
            peak = max(peak, running)
            worst = max(worst, peak - running)
        return worst

    @property
    def ambiguous_exits(self) -> int:
        """Trades whose exit bar touched the target and the stop together."""
        return sum(1 for trade in self.trades if trade.ambiguous_exit)

    @property
    def ambiguous_exit_rate(self) -> Decimal:
        """Share of exits decided by assumption rather than by the tape.

        The credibility figure for the whole run. Every other number here is
        conditional on it: a low rate means the fill model barely mattered, and
        a high one means most exits were assigned by a coin whose bias the
        caller chose.
        """
        return self._rate(self.ambiguous_exits)

    @property
    def exit_reasons(self) -> Mapping[ExitReason, int]:
        """How trades ended. A run that is mostly square-offs is not scalping."""
        counts: dict[ExitReason, int] = {}
        for trade in self.trades:
            counts[trade.exit_reason] = counts.get(trade.exit_reason, 0) + 1
        return MappingProxyType(counts)

    @property
    def round_trips_by_rule(self) -> Mapping[str, int]:
        """Trades each rule contributed to, counting every rule that fired.

        Counts overlap on purpose: a candidate that two rules agreed on appears
        under both, because the question being asked is "was this rule present
        on trades that worked", not "which rule owns this trade".
        """
        counts: dict[str, int] = {}
        for trade in self.trades:
            for rule in trade.rules:
                counts[rule] = counts.get(rule, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def net_rupees_by_rule(self) -> Mapping[str, Decimal]:
        """Rupees attributed to each rule, overlapping on the same terms.

        Sums across rules will exceed ``net_rupees`` wherever rules agreed. That
        is not double counting to be corrected; it is the price of asking a
        per-rule question of a multi-rule score, and dividing a trade between
        its rules would require weights nothing has measured.
        """
        totals: dict[str, Decimal] = {}
        for trade in self.trades:
            for rule in trade.rules:
                totals[rule] = totals.get(rule, Decimal(0)) + trade.net_rupees
        return MappingProxyType(dict(sorted(totals.items())))

    @property
    def round_trips_by_hour(self) -> Mapping[int, int]:
        """Entries per IST hour. The opening and closing hours trade differently."""
        counts: dict[int, int] = {}
        for trade in self.trades:
            hour = trade.entry_time.astimezone(INDIA_TIMEZONE).hour
            counts[hour] = counts.get(hour, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def net_rupees_by_hour(self) -> Mapping[int, Decimal]:
        """Rupees kept per IST hour of entry."""
        totals: dict[int, Decimal] = {}
        for trade in self.trades:
            hour = trade.entry_time.astimezone(INDIA_TIMEZONE).hour
            totals[hour] = totals.get(hour, Decimal(0)) + trade.net_rupees
        return MappingProxyType(dict(sorted(totals.items())))

    def _rate(self, count: int) -> Decimal:
        if not self.trades:
            return Decimal(0)
        return Decimal(count) / Decimal(self.round_trips)

    def _mean(self, values: Iterable[Decimal]) -> Decimal:
        if not self.trades:
            return Decimal(0)
        return sum(values, Decimal(0)) / Decimal(self.round_trips)


__all__ = [
    "FRICTIONLESS",
    "ExitReason",
    "FillModel",
    "ReplayResult",
    "SimulatedTrade",
]
