"""End-to-end checks spanning the market and feature layers.

Every other test in this suite stops at one layer's edge: the market tests end
at a candle, and the feature tests begin from candles built by hand. The live
path runs through both -- ticks arrive from a broker feed, become candles,
become features -- and until this file nothing crossed that seam.

The ticks here are shaped the way Groww's stream actually delivers them, which
is to say with no volume at all. Volume arrives through the stamper seam, so
these tests also exercise the arrangement that makes live volume possible
rather than a convenient fiction in which the stream supplies it.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument, MarketTick, OHLCVCandle
from ai_trader.features import FeatureEngine
from ai_trader.market import Candle, MarketState

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_SESSION_OPEN = datetime(2026, 9, 14, 3, 45, tzinfo=UTC)

_BACKFILL_MINUTES = range(30)
_LIVE_MINUTES = range(30, 80)
"""Enough live candles after the handoff to fill every 20-candle window."""

_VOLUME_FEATURES = (
    "vwap",
    "price_vs_vwap",
    "vwap_deviation",
    "price_vs_vwap_sigma",
    "volume_ratio_20",
    "obv",
    "session_volume",
)
"""Every derived feature that cannot be computed without a candle's volume."""


def _ohlcv(minute: int) -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=_SESSION_OPEN + timedelta(minutes=minute),
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=5_000 + minute,
    )


def _bare_ticks(minutes: range) -> tuple[MarketTick, ...]:
    """Two ticks a minute, carrying no volume, as Groww's stream delivers them.

    Two rather than one so each candle has a high and a low that differ, and so
    the minute's volume is a difference between two readings inside it rather
    than a single point.
    """
    return tuple(
        MarketTick(
            instrument=_RELIANCE,
            timestamp=_SESSION_OPEN + timedelta(minutes=minute, seconds=30 * half),
            price=Decimal(100 + minute % 7) + (Decimal("0.5") if half else Decimal(0)),
        )
        for minute in minutes
        for half in (0, 1)
    )


def _stamper() -> Callable[[MarketTick], MarketTick]:
    """Stand in for ``VolumePoller.stamp``: attach the running session total.

    The total only ever grows, the way an exchange's session counter does, and
    grows by a different amount each minute so that ``volume_ratio_20`` is
    measured against a moving baseline rather than a constant.
    """
    running = 1_000_000

    def stamp(tick: MarketTick) -> MarketTick:
        nonlocal running
        minute = (tick.timestamp - _SESSION_OPEN) // timedelta(minutes=1)
        running += 400 + (minute % 5) * 50
        return replace(tick, cumulative_volume=running)

    return stamp


def _run_live_session(
    tick_stamper: Callable[[MarketTick], MarketTick] | None,
) -> tuple[FeatureEngine, MarketState]:
    """Wire a state object to an engine the way the production path does.

    ``backfill`` deliberately does not fire ``on_candle``, so history is handed
    to the engine explicitly and only live candles arrive through the hook.
    That is the sequence ``check_features`` follows, and getting it wrong in
    either direction -- replaying history twice, or never seeding it -- is the
    mistake this helper exists to make impossible in these tests.
    """
    engine = FeatureEngine()

    def fold(candle: Candle) -> None:
        engine.update(candle)

    state = MarketState(on_candle=fold, tick_stamper=tick_stamper)
    state.backfill(_RELIANCE, tuple(_ohlcv(minute) for minute in _BACKFILL_MINUTES))
    seeded = state.snapshot(_RELIANCE)
    assert seeded is not None
    engine.warm_up(seeded.candles)

    for tick in _bare_ticks(_LIVE_MINUTES):
        state.record_tick(tick)
    state.flush()
    return engine, state


def test_streaming_ticks_and_replaying_candles_land_on_identical_features() -> None:
    """The live path must compute what a replay of the same candles computes.

    ``warm_up`` is a loop over ``update``, so the engine cannot diverge from
    itself -- but that only covers the engine's own boundary. It says nothing
    about whether the candles ``MarketState`` builds from a tick stream are the
    candles a replay would hand over. Driving both and comparing the result is
    what makes a historical replay admissible as evidence about live behaviour,
    which the entire validation plan rests on.
    """
    live, state = _run_live_session(_stamper())
    market = state.snapshot(_RELIANCE)
    assert market is not None

    replayed = FeatureEngine()
    replayed.warm_up(market.candles)

    streamed = live.snapshot(_RELIANCE)
    assert streamed is not None
    assert streamed == replayed.snapshot(_RELIANCE)
    assert streamed.candle_start_time == _SESSION_OPEN + timedelta(minutes=79)
    assert streamed.readiness.core_ready
    assert live.duplicate_candle_count == 0
    assert live.out_of_order_candle_count == 0


def test_the_handoff_from_history_to_live_ticks_costs_exactly_one_candle() -> None:
    """A stream joined mid-minute yields a fragment, so that minute is dropped.

    The cost is worth naming precisely: one candle, at the join, once per
    instrument per run. Everything either side of it is continuous, and the
    gap is visible rather than papered over with a partial minute that would
    understate volume and misplace the open.
    """
    _, state = _run_live_session(_stamper())
    market = state.snapshot(_RELIANCE)
    assert market is not None

    starts = [candle.start_time for candle in market.candles]
    assert starts == [
        _SESSION_OPEN + timedelta(minutes=minute)
        for minute in (*_BACKFILL_MINUTES, *_LIVE_MINUTES[1:])
    ]
    assert state.duplicate_candle_count == 0
    assert state.late_tick_count == 0


def test_a_stamped_stream_carries_volume_all_the_way_into_the_features() -> None:
    """Polled totals must survive differencing, aggregation and derivation.

    Groww's stream has no volume, so every volume feature in the system depends
    on a chain no single layer's tests can see end to end: poll, stamp,
    difference across a minute boundary, fold into a candle, derive. The pinned
    numbers below are the cheapest way to prove the chain arrived intact rather
    than merely produced something non-null.
    """
    engine, _ = _run_live_session(_stamper())
    snapshot = engine.snapshot(_RELIANCE)
    assert snapshot is not None

    # Minute 79 takes two increments of 400 + (79 % 5) * 50 = 600.
    assert snapshot.volume == 1_200
    # Against a 20-candle baseline averaging 1_000, that is exactly 1.2 -- an
    # exact Decimal rather than a tolerance, so a wrong answer fails loudly.
    assert snapshot.volume_ratio_20 == Decimal("1.2")
    # 150_435 backfilled plus 49_200 differenced from the live stream. The
    # session aggregate spanning the handoff is what proves the two sources
    # combine rather than one of them quietly resetting the other.
    assert snapshot.session_volume == 199_635
    for name in _VOLUME_FEATURES:
        assert getattr(snapshot, name) is not None, name
        assert getattr(snapshot.readiness, name), name


def test_an_unstamped_stream_withholds_volume_features_and_nothing_else() -> None:
    """Losing volume must cost volume features only, and must cost them fully.

    This is the system-level statement of the rule that a feature which cannot
    be computed honestly reports nothing. A broker outage, or simply forgetting
    to wire the poller, leaves every volume feature unavailable -- not zero,
    which claims nothing traded, and not a stale figure carried forward, which
    claims a VWAP that no longer describes the session. Price and momentum are
    untouched, so an instrument stays tradeable on the features that survive.
    """
    engine, _ = _run_live_session(None)
    snapshot = engine.snapshot(_RELIANCE)
    assert snapshot is not None

    assert snapshot.volume is None
    for name in _VOLUME_FEATURES:
        assert getattr(snapshot, name) is None, name
        assert not getattr(snapshot.readiness, name), name

    # The surviving features are not merely present, they are the ones a
    # scanner gates on, and they are identical to the stamped run's.
    assert snapshot.readiness.core_ready
    stamped, _ = _run_live_session(_stamper())
    reference = stamped.snapshot(_RELIANCE)
    assert reference is not None
    for name in ("ema9", "ema21", "ema50", "rsi14", "atr14", "adx14", "return_15"):
        assert getattr(snapshot, name) == getattr(reference, name), name
