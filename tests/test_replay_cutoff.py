"""The intraday cutoff must not turn a queued decision into a different kind of
event depending on how long the order was going to take.

A review found that the engine called ``_fill_pending`` twice per cycle: once
unguarded, and once inside the ``if not past_cutoff:`` block. On the cutoff bar
the unguarded call still ran, so a queued entry whose fill moment had arrived
opened a position on the very bar the square-off below it closes. The book
refuses that -- ``ReplayPortfolio.square_off`` drops a position that has not
lived through a bar and hands back ``None`` -- and the engine reads that
``None`` as *unwound*.

**Nothing was lost, and the P&L is untouched**: a dropped position records no
costs, and the decision was still counted. It was counted as the wrong thing.
``unwound_entries`` means a position that filled on the bar the tape ran out
under, and the tape has not run out at an intraday cutoff -- it runs on to the
close. The engine's own square-off comment says a large ``unwound`` figure
means a run whose cutoff is misconfigured, so the misfiling turns a correct run
into a false alarm about its own configuration.

The second half is worse than the first. Entries are blocked from the cutoff to
the bell, so a queued entry that the cutoff overtakes is a decision no bar will
ever price -- which is what *unfilled* means, and where the session rollover
already puts the identical event. At zero latency the same decision on the same
tape fills on the previous bar and becomes an ordinary trade. A flag that models
how long an order takes to reach the exchange was deciding whether a decision
got evaluated at all, and which counter reported it.

**Why a band of cutoffs rather than one.** Whether any entry is queued at a
given minute is a property of this price fixture, not of the engine: sweeping
minute by minute, 73 of 86 candidate cutoffs strand at least one decision and
13 strand none. Picking the best single minute would tune the test to the
fixture and leave it one wave-phase edit away from passing vacuously. Sweeping
a stretch and asserting over the aggregate is both more robust and the stronger
claim, and the floors below are set well under what the band measures so that
ordinary drift moves the number without breaking the test.

**Why ``unwound_entries == 0`` is exact rather than a floor.** On a single
session with a cutoff inside it, no entry can fill at or after the cutoff, so no
open position can share a bar with the square-off that closes it -- and every
name here prints every minute, so each one's last seen bar at the cutoff is the
cutoff bar itself rather than something staler. Zero is therefore a theorem
about the fixed engine that this tape demonstrates, and it was precisely
non-zero under the defect.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, minutes_since_open
from ai_trader.market import Candle
from ai_trader.replay import (
    FillModel,
    ReplayConfig,
    ReplayEngine,
    ReplayResult,
)
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

_BAND = range(190, 221)
"""Cutoff minutes to sweep, as minutes since the open.

Inside the tape with room either side: late enough that the indicators have
warmed and the book is turning over, early enough that a five-minute fill window
still fits before the bars stop. The default cutoff of 360 is outside a
four-hour tape entirely and would never fire, so this file has to name one --
and naming a *range* is what keeps it from being tuned.
"""

_LATENCY = Decimal(300)
"""Five minutes, for the reason ``test_replay_session_boundary`` gives.

Latency is the width of the window in which a decision is queued but unfilled,
and that window is what has to straddle the cutoff for this file to test
anything. There the window straddles a session boundary; here it straddles the
intraday cutoff, which is the boundary every run now has.
"""


def _triangle(step: int, half: int) -> int:
    """A triangular wave, up to ``half`` then back down. Deliberately not random."""
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _tape() -> tuple[Candle, ...]:
    """One continuous session, every name printing every minute.

    No gaps, by design: a name that stopped printing would leave its last seen
    bar behind the cutoff bar, which is the other way a square-off can land on a
    position's own entry bar and is not what this file is about.
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


_TAPE = _tape()


def _run(cutoff: int, latency: Decimal) -> ReplayResult:
    config = ReplayConfig(
        universe=_UNIVERSE,
        fill=FillModel(
            latency_seconds=latency,
            half_spread_fraction=Decimal("0.0002"),
            slippage_fraction=Decimal("0.0001"),
        ),
        strategy=StrategyConfig(
            exit_policy=FixedAtrStop(Decimal("1.5")),
            max_open_positions=len(_UNIVERSE),
            square_off_minutes_since_open=Decimal(cutoff),
        ),
    )
    return ReplayEngine(config).run(_TAPE)


@pytest.fixture(scope="module")
def swept() -> tuple[dict[int, ReplayResult], dict[int, ReplayResult]]:
    """The band replayed twice, delayed and immediate. Built once: it is slow.

    A slot per name, so the position limit never binds. A book at its limit
    makes the scanner suppress every name, which would empty the queue for a
    reason that has nothing to do with the cutoff.
    """
    delayed = {cutoff: _run(cutoff, _LATENCY) for cutoff in _BAND}
    immediate = {cutoff: _run(cutoff, Decimal(0)) for cutoff in _BAND}
    return delayed, immediate


# --- the band has to strand something before the rest means anything ---------


def test_the_band_leaves_entries_queued_when_the_cutoff_arrives(swept) -> None:
    """Without this, every assertion below passes over a run that never queued.

    Both floors are far under what the band measures -- 35 stranded decisions
    across 23 of its 31 minutes -- so this reports a fixture that has gone quiet
    rather than one that has drifted.
    """
    delayed, _ = swept

    stranded = sum(result.unfilled_entries for result in delayed.values())
    minutes = sum(1 for result in delayed.values() if result.unfilled_entries)
    assert stranded >= 10, f"only {stranded} entries stranded across the band"
    assert minutes >= 6, f"only {minutes} of {len(_BAND)} cutoffs stranded anything"

    assert all(result.round_trips > 0 for result in delayed.values())


# --- the claim ---------------------------------------------------------------


def test_an_entry_the_cutoff_overtakes_is_unfilled_rather_than_unwound(swept) -> None:
    """A decision no bar priced is not a position the tape ran out under.

    This is the counter the defect moved. Filling on the cutoff bar handed the
    square-off a position it had to drop, and the drop was reported as
    ``unwound`` -- so the number that is supposed to say "your cutoff is
    misconfigured" rose on runs whose cutoff was fine.
    """
    delayed, _ = swept

    unwound = {cutoff: r.unwound_entries for cutoff, r in delayed.items()}
    assert not any(unwound.values()), f"unwound at {sorted(k for k in unwound if k)}"


def test_latency_moves_when_a_decision_fills_not_whether_it_is_evaluated(
    swept,
) -> None:
    """The same tape and the same cutoff, with the delay flag as the only change.

    At zero latency an entry fills on its own signal bar, so nothing is ever
    still queued when the cutoff arrives and the band strands exactly nothing.
    Every stranded decision the previous test counted therefore exists because
    of the delay -- which is the flag's job. Deciding which counter reports it
    was not, and that is the half this file pins.
    """
    _, immediate = swept

    assert not any(result.unfilled_entries for result in immediate.values())
    assert not any(result.unwound_entries for result in immediate.values())


# --- the same claim at the engine's level rather than the counters' ----------


def test_no_trade_opens_on_or_after_the_cutoff_it_ran_under(swept) -> None:
    """Stated on the trades, which is where a wrong fill would be visible.

    The counters above are how the defect showed, but they are bookkeeping: this
    is the behaviour underneath them. It is the weaker of the two, because a
    position opened on the cutoff bar was dropped rather than recorded and so
    never reached this list either way. It guards the other direction -- an edit
    that let a late fill survive the square-off would put an entry past the
    cutoff into the results, and nothing else here would notice.
    """
    delayed, immediate = swept

    for runs in (delayed, immediate):
        for cutoff, result in runs.items():
            late = [
                trade
                for trade in result.trades
                if minutes_since_open(trade.entry_time) >= cutoff
            ]
            assert not late, f"cutoff {cutoff} opened {len(late)} trades at or past it"
