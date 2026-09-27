"""Replay must be reproducible from what it recorded, and blind to its source.

Two independent claims, which together are the first of the two validation
routes this layer needs. The second is in ``tests/test_replay_no_lookahead.py``.

**Reproducibility.** Every minute, ``ReplayEngine`` hands an observer a
``ReplayCycle`` carrying the scan's two inputs and its output. An outsider
holding only the candles must be able to rebuild those inputs and re-derive that
output. If it can, then the numbers a replay reports are a function of the
recorded tape and nothing else -- not of the engine's private state, not of the
order a sweep happened to iterate in, not of anything that leaked in sideways.
The feature rebuild here is deliberately *incremental*: a fresh
``FeatureEngine`` is fed cycle by cycle and compared at every step, which proves
the engine folded exactly the candles it recorded, in the order it recorded
them, and nothing else. Rebuilding from scratch at each cycle would prove less
and cost quadratically more.

**Source independence.** The user-facing claim is that at any historical time
*t* the scanner presents what it would if *t* were now, whatever fed it.
``tests/test_source_independence.py`` proves that for a single candle and a
single snapshot. This file carries it through the whole layer: a session
assembled from ticks by ``CandleBuilder`` and the same session delivered whole
by the broker's historical endpoint must produce the same trades, rupee for
rupee. The candles are compared first so that a divergence localises to the
market layer instead of surfacing as a mystery in the P&L.

``CandleBuilder`` discards each instrument's first minute -- a live subscription
can join mid-minute, so that minute would be a fragment -- and leaves the last
one open. The historical side is therefore restated over the minutes the live
builder actually emitted, not over the minutes the generator produced. The
values are regenerated rather than copied, so a builder that mis-assembled a
minute cannot make both sides agree by supplying its own answer to both.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument, MarketTick, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.features import FeatureEngine
from ai_trader.market import Candle, CandleBuilder, MarketState
from ai_trader.replay import (
    ExitReason,
    FillModel,
    ReplayConfig,
    ReplayCycle,
    ReplayEngine,
    ReplayResult,
)
from ai_trader.scanner import MarketContext
from ai_trader.strategy import FixedAtrStop, StrategyConfig

_ONE_MINUTE = timedelta(minutes=1)

_SESSION_MINUTES = 375
"""A whole NSE session. Long enough that every indicator is warm for most of the
run, that the book fills and empties repeatedly, and that the square-off at the
end has something to square off."""

_PARITY_MINUTES = 120
"""Shorter, because the tick path costs four ticks a minute per name and the
property being checked -- that two sources agree -- does not need a full day to
show itself. Still long enough for ATR, ADX and the 20-period bands to warm."""

_NAMES = (
    (Instrument(exchange="NSE", trading_symbol="ALPHA"), 0, Decimal("2450")),
    (Instrument(exchange="NSE", trading_symbol="BRAVO"), 17, Decimal("1310")),
    (Instrument(exchange="NSE", trading_symbol="DELTA"), 31, Decimal("880")),
)
"""Three names at different price levels, each phase-shifted so they do not peak
together. A universe that moved in lockstep would fill the book with one idea
wearing three hats and would never exercise the ranking."""

_UNIVERSE = tuple(instrument for instrument, _, _ in _NAMES)

_STRATEGY = StrategyConfig(
    exit_policy=FixedAtrStop(Decimal("1.5")),
    max_open_positions=2,
    square_off_minutes_since_open=Decimal(327),
)
"""Three overrides, each so that the fixture reaches something worth comparing.

A 1.5-ATR stop is hit often enough that ``ExitReason.STOP`` appears, and a
two-position book turns the scanner's ranking into a decision rather than a
formality.

The cutoff is the awkward one and is worth stating plainly. Whether an intraday
square-off fires at all depends on whether a position happens to be open at the
cutoff minute, which is a property of the price fixture rather than of the
engine -- and at the default 360 this tape's book is empty, because it drains at
355 and stays drained. So the cutoff is placed inside a stretch where the book
*is* occupied, with a few minutes' slack either side. That is a number tuned to
a fixture, and the only honest defence of it is that the set equality in
``test_the_fixture_exercises_what_the_other_tests_claim_to_check`` is what keeps
it honest: if a future edit drains the book here too, ``SESSION_END`` stops
appearing and that test fails loudly rather than quietly checking less.

It has to be made to fire somewhere, because the cutoff square-off is the one
exit that does not go through ``ReplayPortfolio.advance`` -- and, now that every
run has a cutoff, it is how a real session ends rather than an edge case.

Everything else the two paths are compared on -- clip, costs, rules, the cost
screen -- is left at its default, which is the arrangement under test as much as
the agreement is: this file can no longer describe a scanner the live path would
not build.
"""

_FILL = FillModel(
    latency_seconds=Decimal(60),
    half_spread_fraction=Decimal("0.0002"),
    slippage_fraction=Decimal("0.0001"),
)
"""A minute of latency and three basis points against the trade. Friction is
switched *on* for these tests on purpose: a frictionless run would pass an
equivalence check that a run charging spread on both legs could still fail."""


# --- a deterministic session, expressible as ticks or as broker candles -------


def _triangle(step: int, half: int) -> int:
    """A triangular wave: up to ``half``, then back down. Never random.

    A failure here has to be reproducible from this file alone, so nothing is
    drawn from a generator -- not even a seeded one. Two triangles of coprime
    periods are superposed below, which gives a series with trends, reversals
    and no repetition over a session, without any rule needing to be nudged.
    """
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    """The price at the close of ``minute_index``.

    Every increment is a whole number of ticks -- 0.35 is seven, 0.25 is five --
    so the generated prices land on the exchange's grid. A fixture that produced
    off-grid prices would make the fill model's rounding untestable, because
    every rounding would be a no-op.
    """
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _ohlc(
    base: Decimal, minute_index: int, phase: int
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """One minute as open, high, low, close, in that order.

    The order matters: it is also the order the ticks are emitted in, and the
    extremes are placed outside the open/close pair so that the highest of the
    four prices really is the high and the lowest really is the low.
    """
    open_price = _close_price(base, minute_index, phase)
    close_price = _close_price(base, minute_index + 1, phase)
    high = max(open_price, close_price) + Decimal("0.85")
    low = min(open_price, close_price) - Decimal("0.75")
    return open_price, high, low, close_price


def _cumulative_volume(minute_index: int, phase: int) -> int:
    """Session-to-date volume at the close of ``minute_index``.

    Cumulative rather than per-minute because that is what the tick feed
    carries; the builder differences consecutive readings to get a minute's
    volume, and the historical side below has to difference the same series to
    agree with it.

    The wobble term is bounded *below* the per-minute growth on purpose. Both
    sides difference this series, and a difference that came out negative would
    be rejected by ``Candle`` before either path could disagree about anything.
    At ``% 900`` against growth of 1,000 the increment is 1,137 on an ordinary
    minute and 237 on the minute the term wraps -- uneven, which is the point,
    and never negative, which is the constraint.
    """
    if minute_index < 0:
        return 0
    return 5_000 + minute_index * 1_000 + (minute_index * 137 + phase * 29) % 900


def _candles(day: int, minutes: int) -> tuple[Candle, ...]:
    """The session as the historical endpoint would deliver it: whole candles."""
    origin = datetime(2026, 9, day, 9, 15, tzinfo=INDIA_TIMEZONE)
    out: list[Candle] = []
    for instrument, phase, base in _NAMES:
        for minute_index in range(minutes):
            start = origin + minute_index * _ONE_MINUTE
            open_price, high, low, close_price = _ohlc(base, minute_index, phase)
            out.append(
                Candle(
                    instrument=instrument,
                    start_time=start,
                    end_time=start + _ONE_MINUTE,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=(
                        _cumulative_volume(minute_index, phase)
                        - _cumulative_volume(minute_index - 1, phase)
                    ),
                )
            )
    return tuple(out)


def _ticks(day: int, minutes: int) -> tuple[MarketTick, ...]:
    """The same session as the live feed would deliver it: four ticks a minute.

    Ordered open, high, low, close within each minute so that the builder's
    first tick sets the open and its last sets the close. Cumulative volume is
    distributed across the four so that the reading on the closing tick is
    exactly the session total for that minute.
    """
    origin = datetime(2026, 9, day, 9, 15, tzinfo=INDIA_TIMEZONE)
    ticks: list[MarketTick] = []
    for minute_index in range(minutes):
        start = origin + minute_index * _ONE_MINUTE
        for instrument, phase, base in _NAMES:
            previous_total = _cumulative_volume(minute_index - 1, phase)
            step = _cumulative_volume(minute_index, phase) - previous_total
            for offset, price in enumerate(_ohlc(base, minute_index, phase)):
                ticks.append(
                    MarketTick(
                        instrument=instrument,
                        price=price,
                        timestamp=start + timedelta(seconds=5 + offset * 15),
                        cumulative_volume=previous_total + step * (offset + 1) // 4,
                    )
                )
    return tuple(ticks)


def _live_candles(day: int, minutes: int) -> tuple[Candle, ...]:
    """Assemble the session the way the live path does, from ticks.

    ``flush`` is deliberately not called. The trailing minute is still open and
    flushing it would emit a candle covering part of a minute that looks exactly
    like one covering all of it -- the single way a partial candle can reach the
    feature engine.
    """
    builder = CandleBuilder()
    emitted: list[Candle] = []
    for tick in _ticks(day, minutes):
        candle = builder.add_tick(tick)
        if candle is not None:
            emitted.append(candle)
    return tuple(emitted)


def _historical_candles(day: int, live: tuple[Candle, ...]) -> tuple[Candle, ...]:
    """Restate the minutes ``live`` contains, as whole broker candles.

    ``live`` is passed in only to say *which* minutes to restate -- the values
    are regenerated from the same functions and pushed through the same
    normalization a backfill uses, so nothing is copied across from the tick
    path. That matters: if the values were copied, a builder that mis-assembled
    a minute would be supplying its own answer to both sides of the comparison.
    """
    origin = datetime(2026, 9, day, 9, 15, tzinfo=INDIA_TIMEZONE)
    phases = {instrument: (phase, base) for instrument, phase, base in _NAMES}
    sources: dict[Instrument, list[OHLCVCandle]] = {}
    for candle in live:
        phase, base = phases[candle.instrument]
        minute_index = int((candle.start_time - origin) / _ONE_MINUTE)
        open_price, high, low, close_price = _ohlc(base, minute_index, phase)
        sources.setdefault(candle.instrument, []).append(
            OHLCVCandle(
                timestamp=candle.start_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=(
                    _cumulative_volume(minute_index, phase)
                    - _cumulative_volume(minute_index - 1, phase)
                ),
            )
        )

    state = MarketState()
    rebuilt: list[Candle] = []
    for instrument, ohlcv in sources.items():
        state.backfill(instrument, ohlcv)
        snapshot = state.snapshot(instrument)
        assert snapshot is not None
        rebuilt.extend(snapshot.candles)
    return tuple(rebuilt)


_CONFIG = ReplayConfig(universe=_UNIVERSE, fill=_FILL, strategy=_STRATEGY)


def _run(candles: tuple[Candle, ...]) -> tuple[ReplayResult, tuple[ReplayCycle, ...]]:
    cycles: list[ReplayCycle] = []
    engine = ReplayEngine(_CONFIG)
    result = engine.run(candles, on_cycle=cycles.append)
    return result, tuple(cycles)


# --- the run is worth auditing -----------------------------------------------


def test_the_fixture_exercises_what_the_other_tests_claim_to_check() -> None:
    """Non-vacuity, asserted rather than assumed.

    Every check below is of the form "two paths agree". Two paths that both did
    nothing also agree, so the fixture has to be shown to do something first.
    """
    result, cycles = _run(_candles(22, _SESSION_MINUTES))

    assert len(cycles) == _SESSION_MINUTES
    assert result.candles_replayed == _SESSION_MINUTES * len(_NAMES)
    assert result.round_trips > 20
    assert result.candidates_seen > 0
    assert {trade.exit_reason for trade in result.trades} == {
        ExitReason.TARGET,
        ExitReason.STOP,
        ExitReason.SESSION_END,
    }
    assert {trade.direction for trade in result.trades} != set()
    assert any(cycle.result.candidates for cycle in cycles)


def test_two_runs_over_the_same_candles_agree_exactly() -> None:
    """The engine is stateless between runs, which a sweep depends on.

    ``ReplayEngine`` is built once and called repeatedly across a sweep of fill
    models and stop multiples. If a second run inherited anything from the
    first, the sweep would be measuring its own iteration order.
    """
    candles = _candles(22, _SESSION_MINUTES)
    engine = ReplayEngine(_CONFIG)

    assert engine.run(candles) == engine.run(candles)


# --- reproducibility: the recorded inputs really are the inputs ---------------


def test_every_snapshot_is_reproducible_from_the_candles_the_cycle_recorded() -> None:
    """Feed an independent engine only what each cycle says it was fed.

    Incremental on purpose. Comparing at every step proves the replay engine
    folded exactly the recorded candles, in the recorded order, and nothing
    else; a single comparison at the end would let a mis-ordering that cancels
    out slip through.

    No ``localcontext`` wrapper is needed here: ``FeatureEngine`` installs
    ``FEATURE_CONTEXT`` around its own arithmetic, so an outsider gets the same
    precision without having to know that.
    """
    _, cycles = _run(_candles(22, _SESSION_MINUTES))
    independent = FeatureEngine()

    for cycle in cycles:
        for candle in cycle.candles:
            independent.update(candle)
        assert independent.snapshots() == cycle.snapshots, f"diverged at {cycle.as_of}"


def test_every_scan_is_reproducible_from_its_recorded_inputs() -> None:
    """Re-run each scan against the inputs the cycle recorded, and compare.

    A fresh ``Scanner`` per cycle rather than one reused instance. Reusing one
    would mirror what the engine does and so would agree with it even if the
    scanner had accumulated state; building a new one each time means a scan
    that depended on its own history would show up as a mismatch.

    It is built the way the engine builds one -- ``StrategyConfig.scanner()``,
    the single constructor replay and live both go through. Assembling a
    ``ScannerConfig`` by hand here would make this test prove something weaker
    than it claims: that a scanner *this file* describes reproduces the scan,
    rather than that the strategy's own scanner does.

    The book is an input to both paths. That is deliberate: this checks the
    features and the scan, and the book is checked by the no-lookahead test.
    """
    _, cycles = _run(_candles(22, _SESSION_MINUTES))

    for cycle in cycles:
        scanner = _STRATEGY.scanner()
        rescanned = scanner.scan(
            cycle.snapshots,
            cycle.portfolio,
            context=MarketContext(as_of=cycle.as_of),
        )
        assert rescanned == cycle.result, f"diverged at {cycle.as_of}"


def test_the_cycle_stream_partitions_the_input_by_minute() -> None:
    """Every candle appears in exactly one cycle, stamped with its own close.

    The reproducibility checks above are only worth anything if the cycles
    account for the whole tape. A cycle that quietly dropped a candle would let
    an independent rebuild agree with an engine that had seen something else.
    """
    candles = _candles(22, _SESSION_MINUTES)
    _, cycles = _run(candles)

    seen: list[Candle] = []
    previous: datetime | None = None
    for cycle in cycles:
        assert cycle.candles, "a cycle with no candles has nothing to be about"
        assert cycle.as_of == cycle.candles[0].end_time
        assert cycle.result.as_of == cycle.as_of
        assert cycle.portfolio.as_of == cycle.as_of
        assert all(candle.end_time == cycle.as_of for candle in cycle.candles)
        if previous is not None:
            assert cycle.as_of > previous, "minutes must advance, never repeat"
        previous = cycle.as_of
        seen.extend(cycle.candles)

    assert sorted(seen, key=_ordering) == sorted(candles, key=_ordering)


def _ordering(candle: Candle) -> tuple[datetime, str, str]:
    return (
        candle.end_time,
        candle.instrument.exchange,
        candle.instrument.trading_symbol,
    )


# --- source independence: ticks and broker candles must agree ----------------


def test_the_two_sources_produce_the_same_candles() -> None:
    """Checked before the replay, so a divergence localises to the market layer.

    If this fails, the disagreement is in ``CandleBuilder`` or ``MarketState``
    and the replay test below would only report it as an unexplained difference
    in P&L.
    """
    live = _live_candles(22, _PARITY_MINUTES)
    historical = _historical_candles(22, live)

    assert live, "the builder emitted nothing; the fixture is broken"
    assert sorted(live, key=_ordering) == sorted(historical, key=_ordering)


def test_the_builder_discards_each_instruments_first_minute() -> None:
    """Stated as a test because the parity fixture is built around it.

    A live subscription can join mid-minute, so ``CandleBuilder`` throws away
    each instrument's first minute rather than emit a fragment, and it never
    emits the minute still in progress. That is why the historical side above is
    restated over the minutes the builder emitted rather than over the minutes
    the generator produced.
    """
    origin = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)
    live = _live_candles(22, _PARITY_MINUTES)

    for instrument, _, _ in _NAMES:
        minutes = [
            int((candle.start_time - origin) / _ONE_MINUTE)
            for candle in live
            if candle.instrument == instrument
        ]
        assert minutes == list(range(1, _PARITY_MINUTES - 1))


def test_a_session_built_from_ticks_replays_identically_to_one_delivered_whole() -> (
    None
):
    """The claim, end to end: the source does not reach the result.

    Same minutes, same configuration, two different transports into the system.
    Trades are compared before the whole result so that a mismatch names the
    trade that differs rather than printing two summary objects.
    """
    live = _live_candles(22, _PARITY_MINUTES)
    historical = _historical_candles(22, live)

    from_ticks, _ = _run(live)
    from_history, _ = _run(historical)

    assert from_ticks.round_trips > 0, "a session with no trades proves nothing"
    assert from_ticks.trades == from_history.trades
    assert from_ticks.net_rupees == from_history.net_rupees
    assert from_ticks == from_history


def test_the_two_sources_agree_minute_by_minute_inside_the_run() -> None:
    """Not just the same trades -- the same decisions, in the same order.

    Two runs could in principle reach identical trades through different scans,
    so the cycle streams are compared directly. This is the strongest form of
    the "at time *t*, whatever the source" claim: at every *t*, both paths held
    the same features, the same book, and reached the same verdict.
    """
    live = _live_candles(22, _PARITY_MINUTES)
    historical = _historical_candles(22, live)

    _, tick_cycles = _run(live)
    _, history_cycles = _run(historical)

    assert len(tick_cycles) == len(history_cycles)
    for from_ticks, from_history in zip(tick_cycles, history_cycles, strict=True):
        assert from_ticks == from_history, f"diverged at {from_ticks.as_of}"
