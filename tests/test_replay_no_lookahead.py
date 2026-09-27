"""A decision at *t* cannot depend on anything after *t*. Proved by truncation.

The second of the two validation routes. The first, in
``tests/test_replay_equivalence.py``, shows that a replay's output is a function
of its recorded inputs. This one shows that those inputs only ever contained the
past.

**The method.** Run the full session, then run it again over the first *k*
minutes, and compare. If the engine were reading ahead -- picking an exit from a
bar it had not reached, sizing against a later price, letting a stop be placed
using volatility that had not printed -- then the truncated run would disagree
with the full one somewhere in the overlap, because in the truncated run the
future it was reading simply is not there. Cut at several points rather than one,
because a single cut can be passed by accident.

**Two strengths of the same claim.** The trade-level check is the legible one:
every round trip the full run had closed by the cut comes back from the prefix
run byte for byte. The cycle-level check is the strict one: every *decision*, its
features, and the book it was taken against are identical minute for minute over
the whole overlap -- not just the trades that happened to result. The second
subsumes the first and is what the file really asserts; the first is kept because
when something breaks, "trade 34 differs" is a better starting point than "cycle
118 differs".

**Why session-end exits are excluded from the trade comparison.** Truncating the
tape forces a square-off that the full run never performs at that point, so a
position open at the cut closes in the prefix run and stays open in the full one.
That is the truncation talking, not the engine, and it is the one difference
between the two runs that is expected. Everything else must match.

The engine's own docstring makes the structural argument -- the loop never holds
a bar it has not reached -- and that argument is the reason this passes. These
tests are the consequence, not the proof.
"""

from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, ONE_SECOND, exact_timedelta
from ai_trader.market import Candle
from ai_trader.replay import (
    ExitReason,
    FillModel,
    ReplayConfig,
    ReplayCycle,
    ReplayEngine,
    ReplayResult,
    SimulatedTrade,
)
from ai_trader.strategy import FixedAtrStop, StrategyConfig

_ONE_MINUTE = timedelta(minutes=1)

_ORIGIN = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)

_MINUTES = 240
"""Four hours. Shorter than a full session because this file runs the tape eight
times over, and long enough that every cut below lands well past the point where
the indicators are warm and the book is busy."""

_CUTS = (60, 90, 120, 150, 180, 210, 240)
"""Where to truncate, in minutes from the open.

Several rather than one: a single cut can pass by luck, especially if it happens
to land where no position is open. The last cut equals the whole tape, which
checks the degenerate case -- truncating at the end must change nothing at all.
"""

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
"""A minute of latency specifically. Zero latency would fill every signal on its
own bar and the interesting case -- a decision taken on one bar and filled on the
next, with a cut able to fall between them -- would never arise."""

_CONFIG = ReplayConfig(
    universe=_UNIVERSE,
    fill=_FILL,
    strategy=StrategyConfig(
        exit_policy=FixedAtrStop(Decimal("1.5")),
        max_open_positions=2,
    ),
)
"""A tighter stop and a smaller book than the defaults, on purpose.

Both make the fixture busier: a 1.5-ATR stop is hit more often than a 2-ATR one,
and a two-position book runs out of room, so positions open and close repeatedly
across the cuts instead of a handful surviving the whole tape. Everything else --
clip, costs, rules, the cost screen -- comes from ``StrategyConfig``'s defaults,
which is the arrangement being tested as much as the prefix property is: there is
no longer a way for this file to configure a scanner the live path would not.
"""


# --- the tape ----------------------------------------------------------------


def _triangle(step: int, half: int) -> int:
    """A triangular wave, up to ``half`` then back down. Deliberately not random.

    A failure here has to be reproducible from this file alone. Two triangles of
    coprime periods superposed give trends and reversals without repetition over
    a session, and without any rule needing to be nudged into firing.
    """
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    """Every increment is a whole number of five-paise ticks, so prices land on
    the exchange's grid and the fill model's rounding is not a no-op."""
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _candles() -> tuple[Candle, ...]:
    out: list[Candle] = []
    for instrument, phase, base in _NAMES:
        for minute_index in range(_MINUTES):
            start = _ORIGIN + minute_index * _ONE_MINUTE
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


def _moment(cut: int) -> datetime:
    """The close of the last minute a ``cut``-minute prefix contains."""
    return _ORIGIN + cut * _ONE_MINUTE


def _prefix(candles: tuple[Candle, ...], cut: int) -> tuple[Candle, ...]:
    return tuple(candle for candle in candles if candle.end_time <= _moment(cut))


def _run(candles: tuple[Candle, ...]) -> tuple[ReplayResult, tuple[ReplayCycle, ...]]:
    cycles: list[ReplayCycle] = []
    result = ReplayEngine(_CONFIG).run(candles, on_cycle=cycles.append)
    return result, tuple(cycles)


def _settled(trades: tuple[SimulatedTrade, ...]) -> list[SimulatedTrade]:
    """Trades that ended on their own terms rather than because the tape did.

    A square-off at the end of the input is an artefact of where the input was
    cut, so it is the one difference between a prefix run and the full run that
    is expected and must be filtered out before comparing.
    """
    return [
        trade for trade in trades if trade.exit_reason is not ExitReason.SESSION_END
    ]


# --- the fixture has to do something before agreement means anything ---------


def test_the_tape_produces_trades_at_every_cut() -> None:
    """Non-vacuity, per cut rather than once.

    Every check below compares two runs. Two runs that both did nothing agree
    perfectly, so each cut has to be shown to have settled trades in its overlap
    before its agreement counts for anything.
    """
    candles = _candles()
    full, _ = _run(candles)

    assert len(_settled(full.trades)) > 10
    for cut in _CUTS:
        closed_by_cut = [
            trade for trade in _settled(full.trades) if trade.exit_time <= _moment(cut)
        ]
        assert closed_by_cut, f"nothing had settled by minute {cut}"


# --- the claim ---------------------------------------------------------------


def test_truncating_the_tape_does_not_change_a_settled_trade() -> None:
    """Every round trip already closed at the cut survives the cut, byte for byte.

    This is the legible form of the claim. If any part of a trade -- its entry
    price, its size, its stop, its exit, its costs -- had been computed using a
    bar after the cut, that bar is absent from the prefix run and the trade would
    come back different.
    """
    candles = _candles()
    full, _ = _run(candles)

    for cut in _CUTS:
        prefix, _ = _run(_prefix(candles, cut))
        expected = [
            trade for trade in _settled(full.trades) if trade.exit_time <= _moment(cut)
        ]
        assert _settled(prefix.trades) == expected, f"diverged at minute {cut}"


def test_truncating_the_tape_does_not_change_a_single_decision() -> None:
    """The strict form: every cycle in the overlap is identical, not just trades.

    A trade is the visible residue of a decision, and decisions that differ can
    still produce the same trades. Comparing the cycle stream compares the
    features, the book and the scan verdict at every minute, so a difference has
    nowhere to hide.

    The prefix run's cycle stream must be an exact prefix of the full run's,
    which is a statement about the loop rather than about the data: the extra
    work a truncated run does -- squaring off, abandoning queued entries -- all
    happens after the loop, so it cannot reach back into a cycle.
    """
    candles = _candles()
    _, full_cycles = _run(candles)

    for cut in _CUTS:
        _, prefix_cycles = _run(_prefix(candles, cut))
        assert len(prefix_cycles) == cut
        assert prefix_cycles == full_cycles[: len(prefix_cycles)], (
            f"diverged inside minute {cut}"
        )


def test_truncating_at_the_end_changes_nothing() -> None:
    """The degenerate cut, stated separately because it is the control.

    If this failed, ``_prefix`` would be dropping bars and every other cut in
    this file would be comparing the wrong things.
    """
    candles = _candles()
    full, full_cycles = _run(candles)
    whole, whole_cycles = _run(_prefix(candles, _MINUTES))

    assert whole == full
    assert whole_cycles == full_cycles


def test_the_order_candles_arrive_in_does_not_reach_the_result() -> None:
    """Reversed input, identical output. The engine sorts before it reads.

    Not a restatement of the truncation tests: those show the engine does not
    read *forward*, this shows it does not depend on the order the caller
    happened to assemble the tape in. A loop that consumed candles as given
    would fold a later bar into the features before an earlier one and would
    disagree here.
    """
    candles = _candles()
    forward, forward_cycles = _run(candles)
    backward, backward_cycles = _run(tuple(reversed(candles)))

    assert backward == forward
    assert backward_cycles == forward_cycles


# --- per-trade causality, which needs no second run --------------------------


def test_no_trade_acts_before_it_was_decided() -> None:
    """Signal, then fill, then exit -- and the fill lands where latency puts it.

    The truncation tests would catch a trade that read a later bar. They would
    not catch one whose own timestamps were incoherent, because the same
    incoherence would appear in both runs. This checks the ordering directly,
    including that the entry is one latency after the signal and never earlier.
    """
    full, _ = _run(_candles())
    latency = exact_timedelta(_FILL.latency_seconds, ONE_SECOND, name="latency_seconds")

    assert full.trades, "no trades to check"
    for trade in full.trades:
        assert trade.signal_time < trade.entry_time
        assert trade.entry_time == trade.signal_time + latency
        assert trade.exit_time > trade.entry_time
        assert trade.bars_held >= 1


def _entry_bar_end(entry_time: datetime) -> datetime:
    """The close of the bar a fill at ``entry_time`` happened inside.

    Ceiling, not ``floor + one minute``. A fill exactly on a minute boundary is
    the close of the bar that has just ended, so that bar is its entry bar and
    the *next* one is fully post-entry and legitimately exitable. Flooring
    would push the threshold a whole bar later and forbid an exit the position
    really was on for -- and at the sixty seconds of latency this file
    configures, every fill lands on a boundary, so flooring would be wrong for
    every trade rather than for an occasional one.
    """
    floored = entry_time.replace(second=0, microsecond=0)
    return floored if floored == entry_time else floored + _ONE_MINUTE


def test_no_position_exits_on_the_bar_it_entered_on() -> None:
    """A fill happened somewhere inside its bar, so that bar's extremes include
    a part of the minute the position was not in yet.

    Letting a trade exit on its entry bar would let it take a target from a move
    that finished before it was on. The book refuses, at the cost of exits being
    late by up to a bar, which is the safe direction.

    Session-end square-offs are included rather than excused. They are the one
    exit that does not go through ``advance``, so they are the one place the
    rule could be enforced in a docstring and quietly skipped in the code.
    """
    full, _ = _run(_candles())

    assert full.trades, "no trades to check"
    for trade in full.trades:
        assert trade.exit_time > _entry_bar_end(trade.entry_time)


def test_an_entry_filled_on_the_last_bar_is_unwound_rather_than_round_tripped() -> None:
    """The other half of the rule above: what happens to the refused position.

    It is dropped, not closed, and counted. Recording it instead would add a
    round trip whose entry and exit share a timestamp -- two legs of costs
    against a price move that is zero by construction -- and every session would
    carry that guaranteed loss into the expectancy.

    The count is asserted to be non-zero first, for the reason every comparison
    in this file is preceded by a non-vacuity check: this tape sets no
    square-off cutoff, so the case is reachable by design, and if a future edit
    to the fixture stopped reaching it the assertions below would pass over a
    situation that no longer occurs. The upper bound is the structural claim --
    the case needs a fill on the very last bar, and a name can hold at most one
    position -- so anything above it would mean the guard is firing on ordinary
    positions rather than on the one the tape ran out underneath.
    """
    full, _ = _run(_candles())

    assert full.unwound_entries >= 1, "the tape never reached the case under test"
    assert full.unwound_entries <= len(_UNIVERSE), (
        "at most one position per name can be filled on the final bar"
    )
    for trade in full.trades:
        assert trade.bars_held >= 1
        assert trade.exit_time > trade.entry_time


# --- the fill price, which is where the claim used to be relaxed -------------


_MID_BAR = FillModel(
    latency_seconds=Decimal(30),
    half_spread_fraction=Decimal(0),
    slippage_fraction=Decimal(0),
)
"""Half a minute of latency, and nothing else charged on top of it.

Half a minute because every other run in this file uses sixty seconds, which
lands each fill exactly on a bar's end, where the close genuinely has printed.
The mid-bar case -- the only one where the question below arises at all --
never comes up at that latency, which is why it went unasserted while the
engine was reading the close. Thirty seconds lands squarely inside a bar, and
far enough in that a price drawn on a line from the open to the close would sit
halfway between them rather than a rounding away from the open.

No spread and no slippage because the assertions read an entry price back and
compare it against a candle. Friction is applied to that price after the tape
is read, so leaving it on would force these tests to re-derive it, and a test
that recomputes the code's own arithmetic checks that it is repeatable rather
than that it is right.
"""

_MID_BAR_CONFIG = replace(_CONFIG, fill=_MID_BAR)

_BAR_END = replace(_MID_BAR, latency_seconds=Decimal(60))
"""The other side of the boundary, at the same zero friction.

A whole minute of latency puts the fill exactly on a bar's end, which is the
one instant in the minute where the close *has* printed and is the last price
rather than a future one. Kept as a separate model because the interesting
property is the discontinuity between the two: a hair inside the bar is the
open, the end of it is the close, and a rule that got the boundary backwards
would read a price a full minute stale without reading the future at all --
which is why the truncation tests above cannot see it. They run two mutated
engines against each other and both agree.
"""

_BAR_END_CONFIG = replace(_CONFIG, fill=_BAR_END)


def _fill_bar(candles: tuple[Candle, ...], trade: SimulatedTrade) -> Candle:
    """The bar the tape was inside when the order filled."""
    for candle in candles:
        if (
            candle.instrument == trade.instrument
            and candle.start_time <= trade.entry_time < candle.end_time
        ):
            return candle
    raise AssertionError(f"no bar covers a fill at {trade.entry_time}")


def _bar_ending_at(candles: tuple[Candle, ...], trade: SimulatedTrade) -> Candle:
    """The bar that had just finished when the order filled on its boundary."""
    for candle in candles:
        if (
            candle.instrument == trade.instrument
            and candle.end_time == trade.entry_time
        ):
            return candle
    raise AssertionError(f"no bar ends at {trade.entry_time}")


def _with_close(
    candles: tuple[Candle, ...], target: Candle, close: Decimal
) -> tuple[Candle, ...]:
    """The same tape with one bar's close moved, and its range opened to fit."""
    replacement = Candle(
        instrument=target.instrument,
        start_time=target.start_time,
        end_time=target.end_time,
        open=target.open,
        high=max(target.high, close),
        low=min(target.low, close),
        close=close,
        volume=target.volume,
    )
    return tuple(replacement if candle is target else candle for candle in candles)


def test_a_fill_inside_a_bar_is_priced_at_the_last_print_before_it() -> None:
    """A one-minute bar publishes two prices whose time is known -- the open, at
    the start, and the close, at the end -- so a fill thirty seconds in can only
    have been handed the open.

    The count afterwards is what stops this passing vacuously. A bar that opened
    and closed at the same price is priced identically by the open and by any
    line drawn to the close, so it cannot tell the two rules apart; some
    material number of the fill bars have to have travelled far enough that a
    halfway price would be several ticks from the open. A count rather than a
    per-trade threshold because this tape turns, and a bar that happens to be
    flat at a turning point is a fact about the fixture rather than a fault.
    """
    candles = _candles()
    result = ReplayEngine(_MID_BAR_CONFIG).run(candles)

    assert result.trades, "no trades to check"
    discriminating = 0
    for trade in result.trades:
        bar = _fill_bar(candles, trade)
        assert trade.entry_price == bar.open, (
            f"{trade.instrument.trading_symbol} filled at {trade.entry_price} "
            f"inside a bar that opened at {bar.open} and closed at {bar.close}"
        )
        if abs(bar.close - bar.open) >= Decimal("0.40"):
            discriminating += 1

    assert discriminating >= 3, (
        f"only {discriminating} fill bars moved enough to tell the open apart "
        "from a price interpolated toward the close"
    )


def test_a_fill_on_a_bar_s_boundary_is_priced_at_the_close_that_just_printed() -> None:
    """The far side of the same edge, and the reason it needs saying.

    At the instant a bar ends its close is the last price, not a future one, so
    a whole minute of latency fills at the close and nothing is being read
    early. Getting this boundary backwards costs nothing in lookahead -- it
    makes the fill a full minute *staler* instead -- and that is exactly why
    nothing else in this file constrains it: every check above runs two engines
    against each other, and two engines wrong in the same direction agree.

    It also happens to be the claim ``replay/engine.py``'s own docstring makes
    about what latency means, which ought not to live only in prose.
    """
    candles = _candles()
    result = ReplayEngine(_BAR_END_CONFIG).run(candles)

    assert result.trades, "no trades to check"
    discriminating = 0
    for trade in result.trades:
        bar = _bar_ending_at(candles, trade)
        assert trade.entry_price == bar.close, (
            f"{trade.instrument.trading_symbol} filled at {trade.entry_price} "
            f"on the boundary of a bar that opened at {bar.open} and closed at "
            f"{bar.close}"
        )
        if abs(bar.close - bar.open) >= Decimal("0.40"):
            discriminating += 1

    assert discriminating >= 3, (
        f"only {discriminating} of these bars moved enough to tell their close "
        "apart from their open"
    )


def test_moving_the_close_of_the_bar_a_fill_landed_in_does_not_move_the_fill() -> None:
    """The claim without reference to any particular rule: a price that prints
    at the end of a minute cannot reach a fill that happened inside it.

    Stronger than the test above, which names the open and so would survive any
    edit that kept naming it. This one changes only the future -- one bar's
    close, and the range widened to admit it -- and requires the fill to come
    back unchanged. Every rule that reads the close fails it, whether the read
    is a straight line, a weighted average, or the close outright; the drift of
    a market that does not wait belongs to ``FillModel.slippage_fraction``,
    which the caller states in advance rather than recovering from the answer.

    Only the entry is compared. The exit legitimately moves: by the end of that
    minute the close has printed, and a position already on is entitled to see
    it.
    """
    candles = _candles()
    baseline = ReplayEngine(_MID_BAR_CONFIG).run(candles)

    assert baseline.trades, "no trades to check"
    first = baseline.trades[0]
    bar = _fill_bar(candles, first)

    moved = _with_close(candles, bar, bar.close + Decimal(25))
    assert moved != candles, "the edit did not change the tape"

    after = ReplayEngine(_MID_BAR_CONFIG).run(moved)
    same = [
        trade
        for trade in after.trades
        if trade.instrument == first.instrument
        and trade.signal_time == first.signal_time
    ]
    assert len(same) == 1, "the trade under test did not survive the edit"
    assert same[0].entry_price == first.entry_price
    assert same[0].entry_time == first.entry_time
