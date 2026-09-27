"""Two layers read the session clock, and they must read the same time.

The scanner learns how far into the day it is from a feature,
``minutes_since_session_open``. The replay engine works it out for itself, from
the cycle's end time, and uses it to decide when to flatten the book. Those are
two computations of one fact, and the fact is a *decision clock*: both numbers
exist to be compared against ``square_off_minutes_since_open`` -- the engine
directly, the scanner through an entry window or through
``FeasibilityPolicy``'s ``square_off - minutes`` runway.

A candle covers ``[start, end)``, so reading that clock from a bar's start
rather than its close puts the two layers a minute apart on the same bar, and
always in the permissive direction: a cost screen credits a name with a minute
of trading that has already elapsed, and a ``latest_minutes_since_open`` bound
admits a decision taken a minute after the edge it names. Neither shows up in a
trade record -- the trade looks ordinary, it was simply allowed when it should
not have been.

**Both files' worth of claim, in two shapes.** The first test pins the
mechanism: on a tape where every name prints every minute, the feature and the
engine agree exactly, on every cycle. The second pins the consequence a caller
would actually configure, and establishes its own precondition first: it finds a
minute at which candidates appear on both that bar and the next, so that a bound
placed there has something to admit *and* something to reject. Without that, a
window test passes on a tape that had nothing to say near the edge.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, minutes_since_open
from ai_trader.market import Candle
from ai_trader.replay import FillModel, ReplayConfig, ReplayCycle, ReplayEngine
from ai_trader.strategy import FixedAtrStop, StrategyConfig

_ONE_MINUTE = timedelta(minutes=1)

_OPEN = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)
"""09:15 IST on Tuesday 22 September 2026."""

_MINUTES = 240
"""Four hours, as elsewhere: long enough to warm the indicators and to trade."""

_NAMES = (
    (Instrument(exchange="NSE", trading_symbol="ALPHA"), 0, Decimal("2450")),
    (Instrument(exchange="NSE", trading_symbol="BRAVO"), 17, Decimal("1310")),
    (Instrument(exchange="NSE", trading_symbol="DELTA"), 31, Decimal("880")),
)

_UNIVERSE = tuple(instrument for instrument, _, _ in _NAMES)

_FILL = FillModel(
    latency_seconds=Decimal(60),
    half_spread_fraction=Decimal("0.0002"),
    slippage_fraction=Decimal("0.0001"),
)


def _triangle(step: int, half: int) -> int:
    """A triangular wave, up to ``half`` then back down. Deliberately not random."""
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _tape() -> tuple[Candle, ...]:
    """A continuous session: every name prints every minute.

    Continuity is a precondition of the first test rather than decoration. The
    feature engine publishes the latest snapshot it holds per instrument, so a
    name that went quiet would carry a clock reading from the bar it last
    printed -- correct for that bar, and not the cycle's. The test asserts the
    continuity it relies on rather than assuming it.
    """
    out: list[Candle] = []
    for instrument, phase, base in _NAMES:
        for minute_index in range(_MINUTES):
            start = _OPEN + minute_index * _ONE_MINUTE
            open_price = _close_price(base, minute_index, phase)
            close_price = _close_price(base, minute_index + 1, phase)
            out.append(
                Candle(
                    instrument=instrument,
                    start_time=start,
                    end_time=start + _ONE_MINUTE,
                    open=open_price,
                    high=max(open_price, close_price) + Decimal("0.85"),
                    low=min(open_price, close_price) - Decimal("0.75"),
                    close=close_price,
                    volume=4_000 + (minute_index * 137 + phase * 29) % 2_600,
                )
            )
    return tuple(out)


def _config(latest: Decimal | None = None) -> ReplayConfig:
    """A run, optionally refusing entries after ``latest`` minutes.

    ``max_open_positions`` is the size of the universe so the book never fills.
    A full book suppresses every name for a reason that has nothing to do with
    the clock, which would let the window assertions hold while testing nothing.
    """
    return ReplayConfig(
        universe=_UNIVERSE,
        fill=_FILL,
        strategy=StrategyConfig(
            exit_policy=FixedAtrStop(Decimal("1.5")),
            max_open_positions=len(_UNIVERSE),
            latest_minutes_since_open=latest,
        ),
    )


def _cycles(latest: Decimal | None = None) -> tuple[ReplayCycle, ...]:
    collected: list[ReplayCycle] = []
    ReplayEngine(_config(latest)).run(_tape(), on_cycle=collected.append)
    return tuple(collected)


def _minutes_with_candidates(cycles: tuple[ReplayCycle, ...]) -> set[Decimal]:
    """The session minutes at which the scanner named at least one candidate."""
    return {
        minutes_since_open(cycle.as_of) for cycle in cycles if cycle.result.candidates
    }


def test_the_feature_clock_and_the_engine_clock_are_the_same_clock() -> None:
    """The mechanism, on every cycle of a continuous tape.

    This is the assertion the off-by-one cannot survive: keyed on a bar's start
    the feature reads one less than the engine on every single cycle, so the
    first minute of the tape fails it.
    """
    cycles = _cycles()

    assert len(cycles) == _MINUTES

    for cycle in cycles:
        elapsed = minutes_since_open(cycle.as_of)
        # Stated, not assumed: the agreement below is between a snapshot and
        # *its own* cycle, which only holds while no name has gone quiet.
        assert len(cycle.snapshots) == len(_UNIVERSE)
        for snapshot in cycle.snapshots:
            assert snapshot.candle_end_time == cycle.as_of
            assert snapshot.minutes_since_session_open == elapsed, (
                f"{snapshot.instrument.trading_symbol} at {cycle.as_of}: the "
                f"feature says {snapshot.minutes_since_session_open} minutes "
                f"into the session, the engine says {elapsed}."
            )


def test_an_entry_window_closes_on_the_minute_it_names() -> None:
    """The consequence: ``latest`` means the decision moment, not the bar's start.

    The edge is chosen from the tape rather than picked in advance. A bound is
    only tested by a minute where the scanner had something to say on both sides
    of it -- a candidate at the edge, which must still be admitted, and one a
    minute later, which must not be. Hard-coding the minute would leave the test
    passing on a tape that had fallen silent there.
    """
    unbounded = _cycles()
    speaking = _minutes_with_candidates(unbounded)

    edge = next(
        (minute for minute in sorted(speaking) if minute + 1 in speaking),
        None,
    )
    assert edge is not None, "no adjacent minutes produced candidates to bound"

    bounded = _minutes_with_candidates(_cycles(latest=edge))

    # The minute after the edge spoke in the unbounded run and must be silent
    # here; under a bar-start clock it is admitted, because the scanner reads
    # that bar as belonging to the edge minute itself.
    assert edge + 1 not in bounded
    assert max(bounded) <= edge
    # And the bound closed the door rather than bricking it up: the edge minute
    # itself is still allowed through.
    assert edge in bounded
