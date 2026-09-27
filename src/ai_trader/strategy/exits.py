"""Where a stop sits, and whether it moves.

``ReplayConfig`` used to carry a bare ``stop_atr_multiple: Decimal`` and the
book multiplied it by the snapshot's ATR once, at entry. That is enough for
exactly one exit rule. A trailing stop is not a different *number*, it is a
different *function*: it depends on the bars that printed after entry, which a
scalar cannot see. So the multiple becomes a policy object and the book asks it
a question every bar instead of once.

**The unit is a fraction of price, scaled by the one-minute ATR** -- the same
unit ``FeasibilityPolicy.max_atr_multiple`` screens on. Keeping the two in one
unit is what makes a sentence like "take nothing whose ATR is over three times
the cost hurdle, and stop out at two ATRs" internally consistent, and what lets
a sweep move one against the other and read the result.

**A stop is always one bar stale, deliberately.** The policy is asked for a new
level at the *end* of a bar, after that bar's exits have been tested, so the
level a bar is tested against was computed from bars strictly before it. Doing
it the other way -- trailing off this bar's high and then testing this bar's low
against the result -- reads an intrabar path nobody recorded, and would let a
trailing stop exit at a level that only existed after the move it claims to have
caught. The cost is that a trail lags by a bar. That direction is the safe one,
and it is the same trade-off ``ReplayPortfolio`` already makes by refusing to
exit on the entry bar.

**Nothing here is measured yet.** The multiples below are conventions with
citations, not findings from this system's own data, and the module says so at
the constant rather than burying it. Step 7 of the plan sweeps them; until then
a replay result quoting one of these is quoting an assumption.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from ai_trader.costs import round_down_to_tick, round_up_to_tick
from ai_trader.scanner import Direction

DEFAULT_STOP_ATR_MULTIPLE = Decimal(2)
"""Two ATRs, which is a convention rather than a measurement.

Two is the most-cited ATR stop distance there is: Wilder introduced the
volatility stop at 2--3 ATRs in *New Concepts in Technical Trading Systems*
(1978), and every later treatment sits in that band -- LeBeau's Chandelier Exit
uses 3 over a 22-bar window. Picking the low end of a published band is a
defensible starting point in a way that picking a number outright is not.

**What transfers and what does not.** The *multiple* is a statement about how
many typical bar ranges of noise a position should tolerate, and that survives a
change of timescale. The *unit* does not. Those sources all mean a daily ATR, and
a one-minute ATR is a far smaller fraction of a session, so two one-minute ATRs
is a much tighter stop in session terms than two daily ATRs is in swing terms. No
published number exists for one-minute intraday equity stops, which is not an
oversight -- it is that the answer depends on the entry rule, and the entry rules
here are five of this system's own.

The research that does exist is about the sign of the effect, not its size:
Kaminski and Lo (*Journal of Financial Markets*, 2014) show a stop **always**
lowers expected return under a random walk and **can raise** it under momentum,
so whether this constant helps at all is a question about whether the scanner's
edge is momentum-shaped. That is a question replay answers, and it is why this
value is a default to sweep from rather than a setting to trust.
"""


def stop_price_for(
    reference_price: Decimal,
    direction: Direction,
    distance_fraction: Decimal,
    tick: Decimal,
) -> Decimal:
    """Where a stop sits, rounded so that it is never harder to hit.

    The mirror of what ``SizingPolicy.estimate`` does to a target. A target
    rounds *away* from entry so that reaching it definitely pays for the round
    trip; a stop rounds *toward* its reference so that a run of bad luck is never
    quietly understated. Both roundings cost the trade money, which is the one
    principle every rounding in this layer follows.

    ``reference_price`` is the price the stop is measured from, which is the
    entry price for a fixed stop and the running extreme for a trailing one. It
    is a parameter rather than always the entry price precisely so that a trail
    can climb past entry: the clamp below says "at least one tick away from the
    anchor", and an anchor hard-coded to entry would have silently pinned every
    trailing stop below the entry price and turned it back into a fixed one.

    The clamp handles a real case rather than a theoretical one. A tick is five
    paise, so on a three-hundred-rupee name one tick is about 1.7 basis points; a
    stop placed a single one-minute ATR away can round onto the reference price
    itself, and a stop *at* the reference would fire on that bar's own noise.
    Pushed to one tick away instead -- still the tightest stop the price grid can
    express, and still unambiguously a stop.
    """
    if direction is Direction.LONG:
        stop = round_up_to_tick(reference_price * (1 - distance_fraction), tick)
        stop = min(stop, reference_price - tick)
        if stop <= 0:
            raise ValueError(
                f"stop distance {distance_fraction} leaves no positive price at "
                f"{reference_price}"
            )
        return stop
    stop = round_down_to_tick(reference_price * (1 + distance_fraction), tick)
    return max(stop, reference_price + tick)


class ExitPolicy(ABC):
    """How far a stop sits from its anchor, asked once per bar.

    One method, called in two places: at entry with ``favourable_fraction`` of
    zero, and at the end of every bar the position survives with the running
    maximum favourable excursion so far. A policy that ignores the second
    argument is a fixed stop; one that uses it trails. The book does not need to
    know which it is holding, which is the point -- ``ReplayPortfolio.advance``
    has no branch for trailing versus fixed, it just re-asks.

    Implementations must be immutable and comparable. They are carried on
    ``StrategyConfig``, which a ``ReplayResult`` cites, and a configuration you
    cannot compare for equality is a configuration two runs can disagree about
    without either of them noticing.
    """

    __slots__ = ()

    @abstractmethod
    def stop_price(
        self,
        *,
        entry_price: Decimal,
        direction: Direction,
        atr_fraction: Decimal,
        favourable_fraction: Decimal,
        tick: Decimal,
    ) -> Decimal:
        """The stop level for the *next* bar.

        ``atr_fraction`` is the one-minute ATR as a fraction of price, captured
        from the snapshot that produced the signal and never re-read, so the stop
        is placed using the volatility the decision saw rather than the
        volatility that followed it.

        ``favourable_fraction`` is the maximum favourable excursion since entry
        as a fraction of the entry price -- ``(high - entry) / entry`` for a long,
        ``(entry - low) / entry`` for a short -- and zero at entry. It is a
        running maximum, so it never decreases, which is what makes a trailing
        implementation monotonic without needing a ratchet.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """One line for a report header, so a run can say what it assumed."""


@dataclass(frozen=True, slots=True)
class FixedAtrStop(ExitPolicy):
    """A stop set once at entry and left there. The conservative baseline.

    Conservative in the sense that matters for a first measurement: it has one
    parameter, it cannot interact with the shape of the price path, and any edge
    a trailing variant shows over it in the Step 7 sweep is attributable to the
    trailing rather than to the two rules differing in a second way as well.
    """

    multiple: Decimal = DEFAULT_STOP_ATR_MULTIPLE

    def __post_init__(self) -> None:
        if self.multiple <= 0:
            raise ValueError(f"multiple must be positive, got {self.multiple}")

    def stop_price(
        self,
        *,
        entry_price: Decimal,
        direction: Direction,
        atr_fraction: Decimal,
        favourable_fraction: Decimal,
        tick: Decimal,
    ) -> Decimal:
        del favourable_fraction  # fixed by definition: the anchor never moves
        return stop_price_for(
            entry_price, direction, self.multiple * atr_fraction, tick
        )

    @property
    def description(self) -> str:
        return f"fixed stop at {self.multiple}x one-minute ATR from entry"


@dataclass(frozen=True, slots=True)
class ChandelierStop(ExitPolicy):
    """A stop that follows the best price the trade has seen.

    Long: ``highest_high_since_entry * (1 - multiple * atr)``. Short: the mirror.
    Identical to ``FixedAtrStop`` on the entry bar, where the best price seen is
    the entry price itself, and diverging from it only as the trade goes the
    right way -- so the two policies differ in exactly one respect and a sweep
    comparing them is comparing the trail and nothing else.

    **Monotonic by construction rather than by clamp.**
    ``favourable_fraction`` is a running maximum so the anchor never retreats;
    ``round_up_to_tick`` is non-decreasing in its argument; and the one-tick
    clamp moves with the anchor. A stop produced here can therefore only ever
    tighten, which is what a trailing stop is for. The classical Chandelier
    Exit anchors on a rolling *N*-bar window instead, which lets the stop loosen
    again when an old high falls out of the window -- defensible over 22 daily
    bars, indefensible for a position that lives five minutes, and it would also
    need the book to retain a bar history that nothing else here wants.

    The trail can lift the stop above the entry price, which is the whole reason
    ``stop_price_for`` takes an anchor rather than assuming entry. It can in
    principle lift it above the target too; that needs a favourable excursion
    several times the target distance without the target having been touched,
    and if it ever happens the bar resolves as an ambiguous one rather than
    silently preferring either leg.
    """

    multiple: Decimal = DEFAULT_STOP_ATR_MULTIPLE

    def __post_init__(self) -> None:
        if self.multiple <= 0:
            raise ValueError(f"multiple must be positive, got {self.multiple}")

    def stop_price(
        self,
        *,
        entry_price: Decimal,
        direction: Direction,
        atr_fraction: Decimal,
        favourable_fraction: Decimal,
        tick: Decimal,
    ) -> Decimal:
        if direction is Direction.LONG:
            anchor = entry_price * (1 + favourable_fraction)
        else:
            anchor = entry_price * (1 - favourable_fraction)
        return stop_price_for(anchor, direction, self.multiple * atr_fraction, tick)

    @property
    def description(self) -> str:
        return (
            f"trailing stop at {self.multiple}x one-minute ATR "
            "from the best price since entry"
        )


DEFAULT_EXIT_POLICY = FixedAtrStop(DEFAULT_STOP_ATR_MULTIPLE)
"""The fixed stop, because it is the one whose behaviour is fully predictable.

A default that trails would make every unswept run a measurement of two
untested choices at once. This way the baseline has one.
"""


__all__ = [
    "DEFAULT_EXIT_POLICY",
    "DEFAULT_STOP_ATR_MULTIPLE",
    "ChandelierStop",
    "ExitPolicy",
    "FixedAtrStop",
    "stop_price_for",
]
