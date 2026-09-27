"""The pre-open auction is not the continuous session, and must not move a run.

NSE collects orders from 09:00 and matches them around 09:08. Groww publishes
those minutes as ordinary candles. The unmatched ones carry a real volume
against four null prices and are dropped at the broker, because there is no
"unknown price" a later layer could carry. The *matched* ones carry a genuine
price, survive that drop, and arrive here looking like any other bar.

They are not like any other bar. An auction print is a single equilibrium quote
struck once, not a minute of trading, and a decision cycle does more than
compute features from it: it tests every open position's stop against the bar,
fills resting entries at its price, and runs the scanner. A stop "hit" at an
auction price is an exit the market never offered.

**The method is equivalence.** Run a tape with the out-of-session bars and again
without them, and require the two runs to agree trade for trade and cycle for
cycle. That is stronger than checking any one consequence: it says the bars
changed no feature, no stop, no fill and no decision, rather than that they
failed to change the two or three things a test happened to look at.

The boundary cases are pinned separately, because both are one minute wide and
neither is obvious. A candle covers ``[start, end)``, so the 09:14 auction bar
*ends* at 09:15 -- and the engine stamps each cycle with its end. The session's
last minute starts at 15:29 and ends at the close, so a bar stamped 15:30 starts
after it.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument
from ai_trader.market import INDIA_TIMEZONE, SESSION_MINUTES, Candle
from ai_trader.replay import (
    FillModel,
    ReplayConfig,
    ReplayCycle,
    ReplayEngine,
    ReplayResult,
)
from ai_trader.strategy import FixedAtrStop, StrategyConfig

_ONE_MINUTE = timedelta(minutes=1)

_OPEN = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)
"""09:15 IST on Tuesday 22 September 2026, the session this file replays."""

_MINUTES = 240
"""Four hours: long enough for the indicators to warm and the book to turn over."""

_AUCTION_PRICE = Decimal("400")
"""Far below every price in the tape, and deliberately so.

An auction print that merely resembled the session would let a run agree with
its own control by luck. At this distance any leak is loud: it would cross every
stop in the book at once and reset each session feature to a price the
continuous market never saw.
"""

_NAMES = (
    (Instrument(exchange="NSE", trading_symbol="ALPHA"), 0, Decimal("2450")),
    (Instrument(exchange="NSE", trading_symbol="BRAVO"), 17, Decimal("1310")),
    (Instrument(exchange="NSE", trading_symbol="DELTA"), 31, Decimal("880")),
)

_UNIVERSE = tuple(instrument for instrument, _, _ in _NAMES)

_CONFIG = ReplayConfig(
    universe=_UNIVERSE,
    fill=FillModel(
        latency_seconds=Decimal(60),
        half_spread_fraction=Decimal("0.0002"),
        slippage_fraction=Decimal("0.0001"),
    ),
    strategy=StrategyConfig(
        exit_policy=FixedAtrStop(Decimal("1.5")),
        max_open_positions=2,
    ),
)
"""A minute of latency, a tight stop and a small book, as elsewhere.

Latency separates the bar a decision is taken on from the bar it fills on, so a
stray auction bar has somewhere to land. The stop and the book keep positions
opening and closing throughout rather than a couple surviving the whole tape.
Everything else is ``StrategyConfig``'s default, which is the arrangement under
test as much as the filter is.
"""


def _triangle(step: int, half: int) -> int:
    """A triangular wave, up to ``half`` then back down. Deliberately not random.

    A failure here has to be reproducible from this file alone.
    """
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _bar(
    instrument: Instrument,
    start: datetime,
    open_price: Decimal,
    close_price: Decimal,
    volume: int,
) -> Candle:
    return Candle(
        instrument=instrument,
        start_time=start,
        end_time=start + _ONE_MINUTE,
        open=open_price,
        high=max(open_price, close_price) + Decimal("0.85"),
        low=min(open_price, close_price) - Decimal("0.75"),
        close=close_price,
        volume=volume,
    )


def _session_tape() -> tuple[Candle, ...]:
    """The continuous session, and nothing else: 09:15 to 13:15."""
    out: list[Candle] = []
    for instrument, phase, base in _NAMES:
        for minute_index in range(_MINUTES):
            out.append(
                _bar(
                    instrument,
                    _OPEN + minute_index * _ONE_MINUTE,
                    _close_price(base, minute_index, phase),
                    _close_price(base, minute_index + 1, phase),
                    4_000 + (minute_index * 137 + phase * 29) % 2_600,
                )
            )
    return tuple(out)


def _outside_bars(offsets: tuple[int, ...]) -> tuple[Candle, ...]:
    """One bar per instrument at each minute offset from the open.

    Negative offsets are pre-open. Volumes are the size a real auction prints,
    which is the reason these bars cannot simply be recognised by looking small.
    """
    return tuple(
        _bar(
            instrument,
            _OPEN + offset * _ONE_MINUTE,
            _AUCTION_PRICE,
            _AUCTION_PRICE,
            28_002 + index * 4_525,
        )
        for index, (instrument, _, _) in enumerate(_NAMES)
        for offset in offsets
    )


def _run(candles: tuple[Candle, ...]) -> tuple[ReplayResult, tuple[ReplayCycle, ...]]:
    cycles: list[ReplayCycle] = []
    result = ReplayEngine(_CONFIG).run(candles, on_cycle=cycles.append)
    return result, tuple(cycles)


def test_the_tape_trades_enough_for_agreement_to_mean_something() -> None:
    # Every comparison below is between two runs of this tape. A tape that did
    # nothing would let them agree without the filter doing any work at all.
    result, cycles = _run(_session_tape())

    assert result.round_trips > 10
    assert len(cycles) == _MINUTES
    assert result.candles_outside_session == 0


def test_bars_outside_the_session_change_neither_a_trade_nor_a_decision() -> None:
    """The whole claim, in one comparison.

    Matched-auction bars reach the engine with real prices. If any of them were
    folded into a feature, tested against a stop, or used to fill a resting
    entry, some trade or some cycle in the polluted run would differ from the
    control -- at a price 400 rupees from the tape, unmistakably.
    """
    session = _session_tape()
    pre_open = _outside_bars((-15, -7, -1))
    post_close = _outside_bars((SESSION_MINUTES, SESSION_MINUTES + 30))

    clean, clean_cycles = _run(session)
    polluted, polluted_cycles = _run((*pre_open, *session, *post_close))

    assert polluted.trades == clean.trades
    assert polluted_cycles == clean_cycles
    # A cycle is emitted once per minute the engine decided in, so an equal
    # count is also the statement that no extra minute was decided in.
    assert len(polluted_cycles) == len(clean_cycles)


def test_skipped_bars_are_reported_rather_than_quietly_dropped() -> None:
    # A run that replayed fewer bars than the fetch returned, and said nothing,
    # would hide a vendor change inside its own results.
    session = _session_tape()
    outside = _outside_bars((-15, -7, -1, SESSION_MINUTES))

    result, _ = _run((*outside, *session))

    assert result.candles_replayed == len(session)
    assert result.candles_outside_session == len(outside)


def test_the_bar_ending_at_the_open_is_still_the_auction() -> None:
    """09:14 ends at 09:15, and the engine stamps each cycle with its end.

    So this bar reads as minute zero of the session while holding nothing but
    auction activity. It is the single case that decides whether the filter is
    keyed on a candle's start or its end, and the only one where the two differ
    at the opening boundary.

    Nothing is open this early, so the damage here is not a trade: it is an
    extra decision cycle, in which the scanner is asked what to buy at a price
    no continuous session ever traded. The cycle stream is asserted first for
    that reason -- the counter below would fail on its own and hide it.
    """
    session = _session_tape()

    clean, clean_cycles = _run(session)
    polluted, polluted_cycles = _run((*_outside_bars((-1,)), *session))

    assert polluted_cycles == clean_cycles
    assert polluted.trades == clean.trades
    assert polluted.candles_outside_session == len(_NAMES)


def test_the_last_minute_of_the_session_is_replayed_not_skipped() -> None:
    """The guard must not have bought its correctness by losing a real minute.

    15:29 is the last minute that trades and its bar ends at the close, so a
    filter keyed on end times would drop it -- and an off-by-one in the other
    direction would admit 15:30, which starts after the session is over.
    """
    instrument, phase, base = _NAMES[0]
    last = SESSION_MINUTES - 1
    bars = tuple(
        _bar(
            instrument,
            _OPEN + offset * _ONE_MINUTE,
            _close_price(base, offset, phase),
            _close_price(base, offset + 1, phase),
            5_000,
        )
        for offset in (last - 1, last, SESSION_MINUTES)
    )

    result, cycles = _run(bars)

    assert result.candles_replayed == 2
    assert result.candles_outside_session == 1
    assert [cycle.as_of for cycle in cycles] == [
        _OPEN + (last - 1) * _ONE_MINUTE + _ONE_MINUTE,
        _OPEN + last * _ONE_MINUTE + _ONE_MINUTE,
    ]
