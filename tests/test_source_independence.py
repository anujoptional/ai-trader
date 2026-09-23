"""The scanner must not be able to tell where its candles came from.

Two properties are asserted here, and they are the ones that make a replay
result mean anything about a live session.

**Source independence.** A minute assembled from ticks by ``CandleBuilder`` and
the same minute delivered whole by the broker's historical endpoint must be the
same ``Candle``, and must therefore produce the same ``FeatureSnapshot`` and the
same ``ScanResult``. If they diverged, every threshold replay established would
be a threshold for a market the live system never sees.

**No lookahead.** The snapshot at step *k* must be identical whether the engine
was fed *k* candles or the whole session. That is what licenses the sentence
"at any historical time *t*, the scanner presents what it would if *t* was now":
folding more history afterwards cannot reach back and change what step *k* said.

Both are properties of the whole chain rather than of one function, so they are
tested through the chain — ticks in at one end, candidates out at the other —
rather than by inspecting the pieces. The third test is structural and covers
what the first two cannot: a wall-clock read inside the decision path would make
a snapshot depend on *when* it was computed, which no fixture can detect because
every fixture computes it now.
"""

import ast
from dataclasses import fields, replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from ai_trader.broker import Instrument, MarketTick, OHLCVCandle
from ai_trader.features import FeatureEngine, FeatureReadiness, FeatureSnapshot
from ai_trader.market import INDIA_TIMEZONE, Candle, CandleBuilder, MarketState
from ai_trader.scanner import (
    MarketContext,
    PortfolioState,
    Scanner,
    ScannerConfig,
    ScanResult,
)

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")

_SESSION_OPEN = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)

_ONE_MINUTE = timedelta(minutes=1)

_MINUTES = 60
"""Long enough that ATR, ADX, the 20-period bands and the volume ratio are all
warm by the tail, so the comparison covers populated features rather than a run
of matching nulls."""


# --- a deterministic session, expressible as ticks or as broker candles -------


def _base_price(minute_index: int) -> Decimal:
    """A reproducible zig-zag with drift.

    Deliberately not random, not even seeded: a failure here has to be
    reproducible from the file alone, and a drifting series with a short cycle
    exercises the trend, breakout and reversion rules without any of them
    needing to be nudged.
    """
    drift = Decimal(minute_index) * Decimal("0.35")
    wobble = Decimal((minute_index * 37) % 11) - Decimal(5)
    return Decimal("2450") + drift + wobble


def _cumulative_volume(minute_index: int) -> int:
    """Session-to-date volume at the close of ``minute_index``.

    Strictly increasing, because the exchange's running total is, and a total
    that went backwards would exercise the regression guard rather than the
    equivalence this file is about.
    """
    if minute_index < 0:
        return 0
    return 5_000 + minute_index * 1_000 + (minute_index * 137) % 400


def _ohlc(minute_index: int) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    open_price = _base_price(minute_index)
    close_price = _base_price(minute_index + 1)
    high = max(open_price, close_price) + Decimal("1.05")
    low = min(open_price, close_price) - Decimal("0.95")
    return open_price, high, low, close_price


def _ticks() -> tuple[MarketTick, ...]:
    """Four ticks a minute, ordered open, high, low, close.

    That ordering is what makes the aggregate exactly predictable: the first
    tick is the open, the last is the close, and the extremes are the extremes.
    Cumulative volume ramps within the minute and lands exactly on the session
    total at the last tick, which is the reading the tracker differences
    against.
    """
    ticks: list[MarketTick] = []
    for minute_index in range(_MINUTES):
        start = _SESSION_OPEN + minute_index * _ONE_MINUTE
        previous_total = _cumulative_volume(minute_index - 1)
        minute_total = _cumulative_volume(minute_index)
        step = minute_total - previous_total
        for offset, price in enumerate(_ohlc(minute_index)):
            ticks.append(
                MarketTick(
                    instrument=_RELIANCE,
                    price=price,
                    timestamp=start + timedelta(seconds=5 + offset * 15),
                    cumulative_volume=previous_total + step * (offset + 1) // 4,
                )
            )
    return tuple(ticks)


def _live_candles() -> tuple[Candle, ...]:
    """Assemble the session the way the live path does, from ticks.

    ``flush`` is deliberately not called. The trailing minute is still open, and
    flushing it would emit a candle covering part of a minute that looks exactly
    like one covering all of it — the single way a partial candle can reach the
    feature engine.
    """
    builder = CandleBuilder()
    emitted: list[Candle] = []
    for tick in _ticks():
        candle = builder.add_tick(tick)
        if candle is not None:
            emitted.append(candle)
    return tuple(emitted)


def _historical_candles(live: tuple[Candle, ...]) -> tuple[Candle, ...]:
    """Deliver the same minutes the way the broker's REST endpoint does.

    The broker hands over whole candles, so this restates each minute as an
    ``OHLCVCandle`` from the same generator and pushes it through the same
    normalization a backfill uses. The live tuple is passed in only to say which
    minutes to restate — the values are regenerated, not copied, so a builder
    that mis-assembled a minute cannot make both sides agree by supplying its
    own answer to both.
    """
    state = MarketState()
    sources: list[OHLCVCandle] = []
    for candle in live:
        minute_index = int((candle.start_time - _SESSION_OPEN) / _ONE_MINUTE)
        open_price, high, low, close_price = _ohlc(minute_index)
        sources.append(
            OHLCVCandle(
                timestamp=candle.start_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=(
                    _cumulative_volume(minute_index)
                    - _cumulative_volume(minute_index - 1)
                ),
            )
        )
    accepted = state.backfill(_RELIANCE, sources)
    assert accepted == len(sources)
    snapshot = state.snapshot(_RELIANCE)
    assert snapshot is not None
    return snapshot.candles


def _fold(candles: tuple[Candle, ...]) -> tuple[FeatureSnapshot, ...]:
    engine = FeatureEngine()
    snapshots: list[FeatureSnapshot] = []
    for candle in candles:
        snapshot = engine.update(candle)
        assert snapshot is not None
        snapshots.append(snapshot)
    return tuple(snapshots)


def _scan(snapshot: FeatureSnapshot) -> ScanResult:
    """Scan one instrument as of the close of the candle that produced it."""
    scanner = Scanner(ScannerConfig())
    as_of = snapshot.candle_end_time
    return scanner.scan(
        (snapshot,),
        PortfolioState.empty(as_of),
        context=MarketContext(as_of=as_of),
    )


# --- the session itself, before anything is compared --------------------------


def test_the_fixture_is_a_real_session_rather_than_a_run_of_nulls() -> None:
    """Guard the guard.

    Every assertion below compares two pipelines against each other, so a
    fixture that produced nothing would pass all of them. This pins that the
    session is long enough to warm the slowest features and that the tail
    actually carries values.
    """
    live = _live_candles()

    # The opening minute is discarded as a fragment and the last is still open,
    # so a 60-minute tick stream yields 58 candles.
    assert len(live) == _MINUTES - 2
    assert live[0].start_time == _SESSION_OPEN + _ONE_MINUTE
    assert all(candle.volume is not None for candle in live)

    latest = _fold(live)[-1]
    assert latest.atr14 is not None
    assert latest.adx14 is not None
    assert latest.vwap is not None
    assert latest.volume_ratio_20 is not None


# --- source independence ------------------------------------------------------


def test_a_minute_built_from_ticks_equals_the_same_minute_from_the_broker() -> None:
    live = _live_candles()
    historical = _historical_candles(live)

    # Field-for-field, including volume: the live figure is differenced out of
    # consecutive session totals and the historical one is served per candle,
    # which is the most plausible place for the two sources to disagree.
    assert live == historical


def test_the_same_session_scans_identically_from_either_source() -> None:
    """The property the user asked for, end to end.

    Not merely the candles: the snapshots they fold into and the candidates
    those produce. A difference anywhere between the tick and the candidate
    would surface here even if the candles themselves matched.
    """
    live = _live_candles()
    historical = _historical_candles(live)

    live_snapshots = _fold(live)
    historical_snapshots = _fold(historical)

    assert live_snapshots == historical_snapshots
    assert [_scan(item) for item in live_snapshots] == [
        _scan(item) for item in historical_snapshots
    ]


# --- no lookahead -------------------------------------------------------------


def test_the_snapshot_at_step_k_does_not_depend_on_what_came_after_it() -> None:
    """A fresh engine fed a prefix agrees with the running fold at that point.

    This is what "at time *t* the scanner presents what it would if *t* was now"
    reduces to mechanically. Every step is checked rather than a sample: a
    lookahead that only appeared once the 20-period window filled would sit
    exactly in the middle of the session, where sampling the ends would miss it.
    """
    live = _live_candles()
    running = _fold(live)

    for step in range(1, len(live) + 1):
        prefix = _fold(live[:step])
        assert prefix[-1] == running[step - 1], f"step {step} disagreed"


def test_a_scan_replayed_at_time_t_matches_the_scan_that_was_live_at_t() -> None:
    live = _live_candles()
    running = _fold(live)

    for step in range(1, len(live) + 1):
        replayed = _fold(live[:step])[-1]
        assert _scan(replayed) == _scan(running[step - 1]), f"step {step} disagreed"


def test_elapsed_session_minutes_come_from_the_candle_not_from_the_count() -> None:
    """The one session feature that could have been a counter, and is not.

    ``minutes_since_session_open`` is arithmetic against 09:15 on the candle's
    own timestamp, so an engine started mid-session reports the true elapsed
    minutes rather than however many candles it happens to have seen. Counting
    candles instead would make a replay that began at 09:15 and a live process
    that attached at 11:00 disagree about the same minute.
    """
    live = _live_candles()
    full = _fold(live)[-1]

    # Attach late: a fresh engine that never saw the first forty minutes.
    late = _fold(live[40:])[-1]

    assert late.minutes_since_session_open == full.minutes_since_session_open
    expected = (live[-1].start_time - _SESSION_OPEN) / _ONE_MINUTE
    assert full.minutes_since_session_open == Decimal(int(expected))


# --- the one asymmetry that is real -------------------------------------------


def _blank_volume(candles: tuple[Candle, ...], index: int) -> tuple[Candle, ...]:
    """The same session with one minute's volume unknown, as a failed poll leaves it."""
    gapped = list(candles)
    gapped[index] = replace(gapped[index], volume=None)
    return tuple(gapped)


def test_a_withheld_volume_reading_narrows_the_scan_without_corrupting_it() -> None:
    """The single thing a live session can know less about than a replay.

    Groww's tick stream carries no volume, so a poller reads the running session
    total over REST and stamps it onto each tick. When a poll fails the minute's
    volume is *unknown* — a state the historical endpoint never reports, since it
    always serves a figure. This is therefore the one way the two sources can
    legitimately disagree, and what matters is the direction.

    Every affected feature is withheld rather than estimated, and no price or
    momentum feature moves at all. A live scan consequently sees less than a
    replay of the same minutes would, never something different about the same
    question — which is the difference between a replay that is optimistic and
    one that is wrong.
    """
    live = _live_candles()
    full = _fold(live)
    gapped = _fold(_blank_volume(live, 25))

    withheld: set[str] = set()
    for whole, holed in zip(full, gapped, strict=True):
        lost: set[str] = set()
        for field in fields(FeatureSnapshot):
            if field.name == "readiness":
                continue
            intact = getattr(whole, field.name)
            observed = getattr(holed, field.name)
            if intact == observed:
                continue
            assert observed is None, (
                f"{field.name} became {observed!r} rather than being withheld"
            )
            lost.add(field.name)

        # Readiness is the snapshot's own account of what it holds, so the two
        # have to agree. A flag that stayed true over a withheld value would be
        # the worst failure available here: a consumer that checks the flag
        # before reading — which is exactly what the README tells it to do —
        # would be told the number was there.
        fell = {
            flag.name
            for flag in fields(FeatureReadiness)
            if getattr(whole.readiness, flag.name)
            != getattr(holed.readiness, flag.name)
        }
        assert all(getattr(holed.readiness, name) is False for name in fell)
        assert fell <= lost, f"readiness dropped {fell - lost} that still had values"
        # ``volume`` is the raw candle figure rather than a derived feature, so
        # it is the one withheld value with no readiness flag behind it.
        assert lost - fell <= {"volume"}, f"{lost - fell} went null while ready"

        withheld |= lost

    # Not vacuous: an engine that ignored volume entirely would pass everything
    # above. The gap has to actually cost the features that depend on it.
    assert {"volume", "volume_ratio_20", "vwap"} <= withheld

    # And no price or momentum feature is collateral damage.
    assert not withheld & {"close", "rsi14", "atr14", "adx14", "macd", "ema20"}


def test_a_candidate_never_cites_a_value_its_snapshot_did_not_carry() -> None:
    """Evidence is a record of what was read, so it cannot outrun the snapshot.

    This is what makes a withheld reading safe. A rule that defaulted an unknown
    ratio to zero would still produce a plausible-looking candidate, and the only
    place that substitution would be visible is here — in the gap between what
    the evidence claims was seen and what the snapshot actually held.
    """
    live = _live_candles()

    for snapshot in _fold(_blank_volume(live, 25)):
        for candidate in _scan(snapshot).candidates:
            for name, value in candidate.evidence.items():
                assert value is not None, f"{name} cited as evidence while unknown"
                assert value == getattr(snapshot, name), (
                    f"{name} was recorded as {value!r} but the snapshot held "
                    f"{getattr(snapshot, name)!r}"
                )


# --- structural: the decision path cannot ask what time it is -----------------


def test_the_decision_path_never_reads_the_wall_clock() -> None:
    """Checked by walking the source, because no fixture can catch this.

    A snapshot that consulted the clock would be reproducible in a test — the
    test also runs now — and wrong in replay, where "now" is years after the
    candle. The only defensible clock in the decision path is the candle's own
    timestamp, so the packages that turn candles into candidates are held to
    reading nothing else.

    ``market`` is excluded on purpose: the volume poller stamps its readings
    from the clock to measure their staleness, and the stream transport uses a
    monotonic deadline. Neither reaches a candle's values — a stale reading
    withholds volume rather than changing it — and both are live-only plumbing
    that no historical candle passes through.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "ai_trader"
    banned_calls = {"now", "utcnow", "today", "monotonic", "perf_counter"}
    banned_modules = {"time"}
    offences: list[str] = []

    for package in ("features", "scanner", "costs"):
        for module in sorted((root / package).glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in banned_calls:
                    offences.append(
                        f"{package}/{module.name}:{node.lineno} .{node.attr}"
                    )
                elif isinstance(node, ast.Name) and node.id in banned_calls:
                    offences.append(f"{package}/{module.name}:{node.lineno} {node.id}")
                elif isinstance(node, ast.ImportFrom) and node.module in banned_modules:
                    offences.append(f"{package}/{module.name}:{node.lineno} from time")
                elif isinstance(node, ast.Import):
                    offences.extend(
                        f"{package}/{module.name}:{node.lineno} import {alias.name}"
                        for alias in node.names
                        if alias.name in banned_modules
                    )

    assert offences == []
