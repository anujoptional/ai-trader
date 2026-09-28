"""A stop has to be an order somebody could place.

Everything here is one postcondition seen from three sides: whatever
``ai_trader.strategy.exits`` returns is a price on the exchange's tick grid, on
the losing side of its anchor, and never looser than the last one. A level that
fails any of those is not a stop -- it is a number the book would compare bars
against and the broker would reject, and replay would go on scoring exits at it
for as long as nobody looked.

**Why a whole file for arithmetic this small.** A review found two ways it broke
at once, and the two have nothing to do with each other beyond both being
invisible in a P&L column.

The first was the one-tick clamp measuring from the anchor: ``anchor - tick`` is
a grid price only when the anchor is one, and a trailing anchor frequently is
not. The second was the trailing anchor itself. ``favourable_fraction`` is
``(high - entry) / entry``, so ``entry * (1 + fraction)`` reconstructs the
running high *exactly* -- except that the round trip through a twenty-eight-digit
fraction leaves a residue of about 1e-25, positive about a quarter of the time,
and the rounding downstream turns that into a whole tick. Two orders of magnitude
short of a paisa becoming five paise of stop is not a rounding error, it is a
different trade.

So the tests below mostly do not compute a stop and compare it to a constant.
They compute it twice by different routes and insist the routes agree, and they
sweep rather than sample, because a defect that fires on a quarter of the inputs
is one a hand-picked triple can miss.
"""

from decimal import Decimal

import pytest

from ai_trader.costs import NSE_EQUITY_TICK
from ai_trader.scanner import Direction
from ai_trader.strategy import ChandelierStop, FixedAtrStop, stop_price_for

_TICK = NSE_EQUITY_TICK

_ENTRIES = tuple(
    Decimal(ticks) * _TICK for ticks in (1_763, 2_204, 6_007, 24_701, 49_000)
)
"""Grid-aligned entry prices from 88 rupees to 2,450: the traded range.

Prime-ish tick counts rather than round rupee figures. An entry at exactly 100
divides the arithmetic too cleanly and would hide the residue this file is about.
"""

_EXCURSIONS = (1, 2, 3, 7, 23, 101)
"""Ticks of favourable excursion, from the smallest move there is upward."""

_DISTANCES = tuple(
    Decimal(text) for text in ("0.00002", "0.0001", "0.0005", "0.002", "0.01", "0.05")
)
"""Stop distances as fractions of price.

The small end is load-bearing: a distance under one tick's worth of the price
rounds the stop onto its own anchor, which is the only time the clamp runs. Left
out, every clamp test here would pass on an unexercised branch.
"""


def _on_grid(price: Decimal) -> bool:
    return price % _TICK == 0


def _cases() -> tuple[tuple[Decimal, Decimal, Decimal, Direction], ...]:
    """Every (entry, extreme, distance, direction) the sweeps below share.

    The extreme is a grid price a whole number of ticks from a grid entry, which
    is what a high or a low actually is. Deriving the fraction from it here --
    rather than passing a fraction in -- is the point: this is the round trip
    the book performs, so it is the round trip under test.
    """
    cases = []
    for entry in _ENTRIES:
        for ticks in _EXCURSIONS:
            offset = Decimal(ticks) * _TICK
            for distance in _DISTANCES:
                cases.append((entry, entry + offset, distance, Direction.LONG))
                cases.append((entry, entry - offset, distance, Direction.SHORT))
    return tuple(cases)


_CASES = _cases()


def _fraction(entry: Decimal, extreme: Decimal, direction: Direction) -> Decimal:
    """What ``OpenTrade.observe`` records, computed the same way it computes it."""
    if direction is Direction.LONG:
        return (extreme - entry) / entry
    return (entry - extreme) / entry


# --- the grid ------------------------------------------------------------------


def test_a_fixed_stop_lands_on_the_grid() -> None:
    policy = FixedAtrStop(Decimal(2))

    for entry, _, distance, direction in _CASES:
        stop = policy.stop_price(
            entry_price=entry,
            direction=direction,
            atr_fraction=distance,
            favourable_fraction=Decimal(0),
            tick=_TICK,
        )
        assert _on_grid(stop), f"{stop} off the grid from {entry} {direction}"


def test_a_trailing_stop_lands_on_the_grid() -> None:
    policy = ChandelierStop(Decimal(2))

    for entry, extreme, distance, direction in _CASES:
        stop = policy.stop_price(
            entry_price=entry,
            direction=direction,
            atr_fraction=distance,
            favourable_fraction=_fraction(entry, extreme, direction),
            tick=_TICK,
        )
        assert _on_grid(stop), f"{stop} off the grid trailing {extreme} {direction}"


def test_an_anchor_off_the_grid_still_produces_a_stop_on_it() -> None:
    # ``stop_price_for`` is public and takes whatever reference it is handed, so
    # the guarantee has to hold for a caller that has not rounded first rather
    # than only for the two policies above. A quarter-paisa anchor is not a real
    # price; the stop derived from it still has to be one.
    for entry, _, distance, direction in _CASES:
        for drift in (Decimal("0.0025"), Decimal("-0.0025"), Decimal("1E-25")):
            stop = stop_price_for(entry + drift, direction, distance, _TICK)
            assert _on_grid(stop), f"{stop} off the grid from {entry + drift}"


# --- which side of the anchor --------------------------------------------------


def test_a_stop_never_sits_at_the_price_it_is_measured_from() -> None:
    # A stop *at* its anchor fires on that bar's own noise, so the clamp holds it
    # a tick away. The tightest distances in the sweep are the ones that would
    # otherwise round onto the anchor, which is what makes this more than a
    # restatement of "a stop is below the entry".
    policy = ChandelierStop(Decimal(2))

    for entry, extreme, distance, direction in _CASES:
        stop = policy.stop_price(
            entry_price=entry,
            direction=direction,
            atr_fraction=distance,
            favourable_fraction=_fraction(entry, extreme, direction),
            tick=_TICK,
        )
        if direction is Direction.LONG:
            assert stop < extreme, f"long stop {stop} not below its anchor {extreme}"
        else:
            assert stop > extreme, f"short stop {stop} not above its anchor {extreme}"


def test_a_stop_that_would_go_negative_is_refused_rather_than_floored() -> None:
    # A distance over one hundred percent has no price to name, and answering
    # with zero -- or with one tick -- would be a stop the book could never hit,
    # which reads in a report as a strategy that simply does not use stops.
    with pytest.raises(ValueError, match="leaves no positive price"):
        stop_price_for(Decimal("100"), Direction.LONG, Decimal("1.5"), _TICK)


# --- the trail agrees with the price it is trailing ----------------------------


def test_the_trail_anchors_on_the_running_extreme_itself() -> None:
    # The docstring's claim, stated the way it is written there: the trailing
    # stop is the stop measured from the best price since entry. The policy gets
    # there through an excursion fraction and this gets there from the price, so
    # any residue the fraction leaves behind shows up as a disagreement.
    #
    # It does show up. Roughly a quarter of grid-aligned (entry, high) pairs
    # reconstruct to the high plus 1e-25, and a positive residue crossing a tick
    # boundary moves the stop a whole tick -- or, where the clamp binds, leaves
    # it sitting exactly on the high it is supposed to be protecting.
    policy = ChandelierStop(Decimal(2))

    for entry, extreme, distance, direction in _CASES:
        trailed = policy.stop_price(
            entry_price=entry,
            direction=direction,
            atr_fraction=distance,
            favourable_fraction=_fraction(entry, extreme, direction),
            tick=_TICK,
        )
        direct = stop_price_for(extreme, direction, policy.multiple * distance, _TICK)
        assert trailed == direct, f"trailing {extreme} from {entry}: {trailed}/{direct}"


def test_the_trail_starts_where_the_fixed_stop_starts() -> None:
    # On the entry bar the best price seen is the entry price, so the two
    # policies have to agree exactly. This is what makes a sweep of fixed
    # against trailing a measurement of the trailing and nothing else.
    fixed = FixedAtrStop(Decimal(2))
    trailing = ChandelierStop(Decimal(2))

    for entry, _, distance, direction in _CASES:
        arguments = {
            "entry_price": entry,
            "direction": direction,
            "atr_fraction": distance,
            "favourable_fraction": Decimal(0),
            "tick": _TICK,
        }
        assert trailing.stop_price(**arguments) == fixed.stop_price(**arguments)


# --- monotonicity --------------------------------------------------------------


def test_a_trailing_stop_only_ever_tightens() -> None:
    # Claimed in the class docstring as holding "by construction rather than by
    # clamp", which was true of the arithmetic it described and is the kind of
    # claim a rounding change quietly falsifies: rounding to nearest is
    # non-decreasing, but that is an argument rather than a test.
    policy = ChandelierStop(Decimal(2))

    for entry in _ENTRIES:
        for distance in _DISTANCES:
            for direction in Direction:
                previous: Decimal | None = None
                for ticks in range(0, 120):
                    offset = Decimal(ticks) * _TICK
                    if direction is Direction.LONG:
                        extreme = entry + offset
                    else:
                        extreme = entry - offset
                    stop = policy.stop_price(
                        entry_price=entry,
                        direction=direction,
                        atr_fraction=distance,
                        favourable_fraction=_fraction(entry, extreme, direction),
                        tick=_TICK,
                    )
                    if previous is not None:
                        if direction is Direction.LONG:
                            assert stop >= previous, f"long stop loosened at {extreme}"
                        else:
                            assert stop <= previous, f"short stop loosened at {extreme}"
                    previous = stop
