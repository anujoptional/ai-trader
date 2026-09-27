"""A decision does not survive the night that ended the day it was taken on.

This system squares off before every close and holds nothing overnight. Open
positions were already handled: the engine flushes the book at a session
rollover. *Queued* ones were not. A pending entry is a decision waiting for a
price, and the price it was waiting for stops existing when the session does --
so an entry left queued across midnight meets the next morning's first bar
instead. What comes out is not merely a late fill. The engine fills a pending
entry at ``fill_at``, which is yesterday's clock, while the price comes from
the bar that minute lands in, which is today's open: the trade is booked to
yesterday's session, stamped with yesterday's entry time, at a price that had
not printed then, and it exits on a day it is not recorded as belonging to.
Two quieter consequences ride along: the stale entry keeps consuming a book
slot, and keeps its name ``is_committed``, so it suppresses that name on the new
day too.

**Two tests, because there are two separate claims.** That ``reset_session``
empties the queue and hands back what it dropped is a property of the book, and
is checked directly on the book -- if the engine later grows a second caller,
the invariant is already enforced where the state lives. That no trade ends up
spanning a boundary is a property of a run, and needs a tape with a boundary in
it.

**The engine test cannot be allowed to pass vacuously**, which is the whole
difficulty: a two-session run with nothing queued at the rollover would satisfy
"no trade spans a session" without exercising anything. So the first day's tape
is replayed *alone* first, and its flush is required to report entries still
queued when it ran out. The engine only ever moves forward, so bars appended
after those minutes cannot change what happened during them: if day one alone
ends with a queue, the two-day run reaches the boundary with the same queue.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, trading_session_date
from ai_trader.market import Candle
from ai_trader.replay import (
    FillModel,
    ReplayConfig,
    ReplayEngine,
    ReplayPortfolio,
    ReplayResult,
    SimulatedTrade,
)
from ai_trader.scanner import Candidate, Direction
from ai_trader.strategy import FixedAtrStop, StrategyConfig

_ONE_MINUTE = timedelta(minutes=1)

_DAY_ONE = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)
_DAY_TWO = datetime(2026, 9, 23, 9, 15, tzinfo=INDIA_TIMEZONE)
"""Consecutive weekdays: Tuesday and Wednesday, 22 and 23 September 2026."""

_MINUTES = 240
"""Four hours per session, as in the session-hours file: long enough to trade."""

_NAMES = (
    (Instrument(exchange="NSE", trading_symbol="ALPHA"), 0, Decimal("2450")),
    (Instrument(exchange="NSE", trading_symbol="BRAVO"), 17, Decimal("1310")),
    (Instrument(exchange="NSE", trading_symbol="DELTA"), 31, Decimal("880")),
)

_UNIVERSE = tuple(instrument for instrument, _, _ in _NAMES)

_STRATEGY = StrategyConfig(
    exit_policy=FixedAtrStop(Decimal("1.5")),
    max_open_positions=len(_UNIVERSE),
)
"""A slot per name, so the position limit never binds.

Elsewhere a small book is the point; here it is in the way. A book at its limit
makes the scanner suppress every name, so no candidate is queued and the tape
runs out with an empty queue -- the one state in which this file tests nothing.
Sizing the book to the universe removes that interference without tuning
anything towards a wanted answer: the capacity rule is simply not what is under
test on this tape.
"""

_CONFIG = ReplayConfig(
    universe=_UNIVERSE,
    fill=FillModel(
        latency_seconds=Decimal(300),
        half_spread_fraction=Decimal("0.0002"),
        slippage_fraction=Decimal("0.0001"),
    ),
    strategy=_STRATEGY,
)
"""Five minutes of latency, rather than the one minute used elsewhere.

Latency is the width of the window in which a decision is queued but unfilled,
and that window is exactly what has to straddle the boundary for this file to
be testing anything. At one minute the test would hang on whether a candidate
happened to fire in the tape's final minute; at five it survives any single
quiet minute. It is not so wide that the book stops turning over, which the
round-trip assertion below checks rather than assumes.
"""


def _triangle(step: int, half: int) -> int:
    """A triangular wave, up to ``half`` then back down. Deliberately not random."""
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _session_tape(open_time: datetime, *, drift: int = 0) -> tuple[Candle, ...]:
    """One session's continuous bars for every name.

    ``drift`` shifts where each name sits in its own wave, so the second day is
    not a copy of the first. A second day that repeated the first would still
    have a boundary in it, but a trade wrongly carried across one would land on
    prices identical to the ones it was carried from, and the least visible
    failure is the one this file exists to catch.
    """
    out: list[Candle] = []
    for instrument, phase, base in _NAMES:
        for minute_index in range(_MINUTES):
            start = open_time + minute_index * _ONE_MINUTE
            open_price = _close_price(base, minute_index + drift, phase)
            close_price = _close_price(base, minute_index + drift + 1, phase)
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


def _run(candles: tuple[Candle, ...]) -> ReplayResult:
    return ReplayEngine(_CONFIG).run(candles)


def _sessions_of(trade: SimulatedTrade) -> tuple[date, date, date]:
    """The trading day each of a trade's three moments belongs to."""
    return (
        trading_session_date(trade.signal_time),
        trading_session_date(trade.entry_time),
        trading_session_date(trade.exit_time),
    )


def _book() -> ReplayPortfolio:
    """A book built exactly as ``ReplayEngine`` builds one, from the defaults."""
    return ReplayPortfolio(
        universe=_UNIVERSE,
        max_open_positions=_STRATEGY.max_open_positions,
        sizing=_STRATEGY.sizing_policy(),
        costs=_STRATEGY.costs,
        fill=_CONFIG.fill,
        exit_policy=_STRATEGY.exit_policy,
        cooldown_minutes=_STRATEGY.cooldown_minutes,
    )


def test_reset_session_drops_the_queue_and_says_what_it_dropped() -> None:
    """The invariant, checked on the object that owns the state.

    Returning the dropped entries is half the fix and not decoration. Clearing
    the queue silently would trade one wrong number for another: the run would
    report fewer entries the tape never priced than it actually had, and the
    honesty block of the report is where that number is read.
    """
    instrument = _UNIVERSE[0]
    book = _book()
    pending = book.queue(
        Candidate(
            instrument=instrument,
            direction=Direction.LONG,
            score=Decimal("0.8"),
            rules=("momentum",),
            as_of=_DAY_ONE + 200 * _ONE_MINUTE,
            reference_price=Decimal("2450"),
        ),
        atr_fraction=Decimal("0.004"),
    )

    assert book.committed() == 1
    assert book.is_committed(instrument)

    dropped = book.reset_session()

    assert dropped == (pending,)
    assert book.pending_entries() == ()
    # A slot the book still thinks is spoken for is a slot the new day cannot
    # use, and a name it still thinks is committed is a name the new day cannot
    # trade -- both silent, neither visible in any trade record.
    assert book.committed() == 0
    assert not book.is_committed(instrument)


def test_day_one_alone_ends_with_entries_still_queued() -> None:
    """Non-vacuity for the test below, established on day one's bars alone.

    If this run ends with nothing queued then the two-session run never reaches
    its boundary with anything to carry, and the spanning assertions there would
    hold for a reason that has nothing to do with the fix. The engine moves
    forward only, so what these bars do here is what they do there.
    """
    result = _run(_session_tape(_DAY_ONE))

    assert result.unfilled_entries >= 1
    # And the tape trades, so the book is genuinely turning over rather than
    # queueing into a book that never fills anything.
    assert result.round_trips > 10
    assert len(result.sessions) == 1


def test_no_trade_spans_a_session_boundary() -> None:
    """The money claim: nothing is signalled, entered or exited across a night.

    Under the bug it is the *exit* check that fires, not the entry one, and the
    reason is worth stating because it is the opposite of the obvious guess.
    The engine fills a pending entry at ``fill_at`` -- a timestamp fixed
    yesterday -- so the stale entry is stamped into yesterday's session even
    though the price it gets is this morning's open. ``signal_time`` and
    ``entry_time`` therefore agree, and the fabricated position gives itself
    away only when it closes, on a day its own record says it does not belong
    to. Both checks are asserted because they are the same invariant read from
    two ends: this one catches an entry carried forward, and the other would
    catch a future change that dropped the rollover flush instead of the queue.
    """
    both = (*_session_tape(_DAY_ONE), *_session_tape(_DAY_TWO, drift=113))

    result = _run(both)

    assert len(result.sessions) == 2
    assert result.round_trips > 10

    for trade in result.trades:
        signalled, entered, exited = _sessions_of(trade)
        assert signalled == entered, (
            f"{trade.instrument.trading_symbol} was signalled on {signalled} "
            f"and entered on {entered}: a queued entry survived the close."
        )
        assert entered == exited, (
            f"{trade.instrument.trading_symbol} entered on {entered} and exited "
            f"on {exited}: a position was held overnight."
        )

    # Both days must contribute, or "no trade spans a boundary" is a statement
    # about one day with some spare bars after it.
    traded_on = {trading_session_date(trade.entry_time) for trade in result.trades}
    assert traded_on == set(result.sessions)


def test_the_boundary_drop_is_counted_not_swallowed() -> None:
    """What day one left queued is reported, not quietly forgotten.

    A two-session run reports the entries dropped at the boundary *plus* those
    still queued when the tape finally ran out, so it can never report fewer
    than day one alone did. Weaker than the assertions above -- carrying the
    entries into day two changes day two's whole trajectory, so this could
    survive the bug by luck -- and kept because the counter is what a reader of
    the report actually sees.
    """
    day_one = _run(_session_tape(_DAY_ONE))
    both = _run((*_session_tape(_DAY_ONE), *_session_tape(_DAY_TWO, drift=113)))

    assert both.unfilled_entries >= day_one.unfilled_entries
