from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument, MarketTick, OHLCVCandle
from ai_trader.market import Candle, MarketState

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_NIFTY = Instrument(exchange="NSE", trading_symbol="NIFTY")
_SESSION_OPEN = datetime(2026, 9, 14, 3, 45, tzinfo=UTC)


def _ohlcv(minutes: int, close: str = "101") -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=_SESSION_OPEN + timedelta(minutes=minutes),
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal(close),
        volume=5_000 + minutes,
    )


def _tick(
    minutes: int,
    price: str,
    seconds: int = 0,
    instrument: Instrument = _RELIANCE,
    cumulative_volume: int | None = None,
) -> MarketTick:
    return MarketTick(
        instrument=instrument,
        timestamp=_SESSION_OPEN + timedelta(minutes=minutes, seconds=seconds),
        price=Decimal(price),
        cumulative_volume=cumulative_volume,
    )


def test_backfill_converts_broker_candles_into_retained_history() -> None:
    state = MarketState()

    accepted = state.backfill(_RELIANCE, (_ohlcv(0), _ohlcv(1), _ohlcv(2)))
    snapshot = state.snapshot(_RELIANCE)

    assert accepted == 3
    assert snapshot is not None
    assert len(snapshot.candles) == 3
    assert snapshot.candles[0].start_time == _SESSION_OPEN
    assert snapshot.candles[0].end_time == _SESSION_OPEN + timedelta(minutes=1)
    assert snapshot.candles[0].volume == 5_000
    assert snapshot.latest_candle == snapshot.candles[-1]
    # Backfill carries no prices of its own; only ticks set the latest price.
    assert snapshot.last_price is None
    assert snapshot.last_tick_at is None


def test_backfill_rejects_replayed_candles_without_corrupting_history() -> None:
    state = MarketState()
    candles = (_ohlcv(0), _ohlcv(1), _ohlcv(2))
    state.backfill(_RELIANCE, candles)

    replayed = state.backfill(_RELIANCE, candles)
    snapshot = state.snapshot(_RELIANCE)

    assert replayed == 0
    assert state.duplicate_candle_count == 3
    assert snapshot is not None
    assert len(snapshot.candles) == 3


def test_rolling_window_evicts_the_oldest_candles() -> None:
    state = MarketState(max_candles=3)

    state.backfill(_RELIANCE, tuple(_ohlcv(minutes) for minutes in range(5)))
    snapshot = state.snapshot(_RELIANCE)

    assert snapshot is not None
    assert [candle.start_time for candle in snapshot.candles] == [
        _SESSION_OPEN + timedelta(minutes=minutes) for minutes in (2, 3, 4)
    ]


def test_max_candles_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_candles"):
        MarketState(max_candles=0)


def test_live_ticks_extend_backfilled_history_with_derived_volume() -> None:
    """The historical-to-live handoff costs one minute and no correctness.

    A live stream is joined partway through a minute, so the builder discards
    that minute instead of extending history with a fragment of it. Minute two
    is therefore absent, and every live candle that does arrive carries real
    volume — which is what keeps session VWAP alive across the seam.
    """
    emitted: list[Candle] = []
    state = MarketState(on_candle=emitted.append)
    state.backfill(_RELIANCE, (_ohlcv(0), _ohlcv(1)))

    for minutes, price, volume in (
        (2, "150", 9_000),
        (3, "151", 9_400),
        (4, "152", 9_900),
        (5, "153", 10_500),
    ):
        state.record_tick(_tick(minutes, price, cumulative_volume=volume))
    state.flush()
    snapshot = state.snapshot(_RELIANCE)

    assert snapshot is not None
    assert [candle.start_time for candle in snapshot.candles] == [
        _SESSION_OPEN + timedelta(minutes=minutes) for minutes in (0, 1, 3, 4, 5)
    ]
    assert [candle.volume for candle in snapshot.candles[2:]] == [400, 500, 600]
    assert [candle.start_time for candle in emitted] == [
        _SESSION_OPEN + timedelta(minutes=minutes) for minutes in (3, 4, 5)
    ]
    assert state.duplicate_candle_count == 0


def test_a_tick_stamper_supplies_volume_a_stream_omits() -> None:
    """Volume arrives through the state object, not through each call site.

    Groww's stream carries no volume, so a live candle can only get one if
    something attaches the running session total on the way in. Holding that
    seam here means a caller wires it once; stamping at each call site instead
    would lose a whole session's volume to one forgotten path.
    """
    totals = iter((9_000, 9_400, 9_900, 10_500))
    stamped: list[MarketTick] = []

    def stamp(tick: MarketTick) -> MarketTick:
        stamped.append(tick)
        return replace(tick, cumulative_volume=next(totals))

    state = MarketState(tick_stamper=stamp)
    for minutes, price in ((2, "150"), (3, "151"), (4, "152"), (5, "153")):
        state.record_tick(_tick(minutes, price))
    snapshot = state.snapshot(_RELIANCE)

    assert snapshot is not None
    # Minute two is the straddled first minute and is discarded as always.
    assert [candle.volume for candle in snapshot.candles] == [400, 500]
    # The stamper saw the ticks as the stream delivered them, unstamped.
    assert all(tick.cumulative_volume is None for tick in stamped)
    # Price tracking reads the same tick the builder aggregated.
    assert snapshot.last_price == Decimal("153")


def test_without_a_stamper_a_volumeless_stream_reports_no_volume() -> None:
    state = MarketState()
    for minutes, price in ((2, "150"), (3, "151"), (4, "152")):
        state.record_tick(_tick(minutes, price))
    snapshot = state.snapshot(_RELIANCE)

    assert snapshot is not None
    # No stamper and no volume on the tick leaves the candle honestly unset,
    # rather than defaulting to a zero that claims nothing traded.
    assert [candle.volume for candle in snapshot.candles] == [None]


def test_live_candle_for_a_backfilled_minute_is_rejected_as_duplicate() -> None:
    state = MarketState()
    state.backfill(_RELIANCE, (_ohlcv(0), _ohlcv(1), _ohlcv(2)))

    # Minute one is straddled and discarded; minute two is the first live
    # candle, and backfill already covers it.
    state.record_tick(_tick(1, "150", seconds=30))
    state.record_tick(_tick(2, "151"))
    state.record_tick(_tick(3, "152"))
    snapshot = state.snapshot(_RELIANCE)

    assert state.duplicate_candle_count == 1
    assert snapshot is not None
    assert len(snapshot.candles) == 3
    assert snapshot.candles[-1].close == Decimal("101")


def test_latest_price_follows_the_newest_tick_only() -> None:
    state = MarketState()

    state.record_tick(_tick(0, "150", seconds=30))
    state.record_tick(_tick(0, "90", seconds=10))
    snapshot = state.snapshot(_RELIANCE)

    assert snapshot is not None
    assert snapshot.last_price == Decimal("150")
    assert snapshot.last_tick_at == _SESSION_OPEN + timedelta(seconds=30)


def test_unknown_instrument_has_no_snapshot() -> None:
    state = MarketState()
    state.backfill(_RELIANCE, (_ohlcv(0),))

    assert state.snapshot(_NIFTY) is None


def test_instruments_and_snapshots_are_returned_in_a_stable_order() -> None:
    state = MarketState()
    state.backfill(_RELIANCE, (_ohlcv(0),))
    state.record_tick(_tick(0, "25000", instrument=_NIFTY))

    assert state.instruments() == (_NIFTY, _RELIANCE)
    assert [snapshot.instrument for snapshot in state.snapshots()] == [
        _NIFTY,
        _RELIANCE,
    ]


def test_late_ticks_are_counted_and_ignored() -> None:
    state = MarketState()
    # Minute zero is the straddled one the builder discards, so minute one is
    # the first candle history actually keeps.
    state.record_tick(_tick(0, "100"))
    state.record_tick(_tick(1, "101"))
    state.record_tick(_tick(2, "102"))

    state.record_tick(_tick(1, "999", seconds=30))

    assert state.late_tick_count == 1
    snapshot = state.snapshot(_RELIANCE)
    assert snapshot is not None
    assert snapshot.candles[0].high == Decimal("101")


def test_forget_drops_an_instrument_and_reports_whether_it_was_known() -> None:
    state = MarketState()
    state.backfill(_RELIANCE, (_ohlcv(0), _ohlcv(1)))
    state.record_tick(_tick(2, "102"))

    assert state.forget(_RELIANCE) is True
    assert state.snapshot(_RELIANCE) is None
    assert state.instruments() == ()
    # Reporting it the second time would make a caller's cleanup look like it
    # had found retained state to drop when there was none.
    assert state.forget(_RELIANCE) is False


def test_forgetting_one_instrument_leaves_the_others_retained() -> None:
    state = MarketState()
    state.backfill(_RELIANCE, (_ohlcv(0),))
    state.backfill(_NIFTY, (_ohlcv(0),))
    state.record_tick(_tick(1, "25000", instrument=_NIFTY))

    state.forget(_RELIANCE)

    assert state.instruments() == (_NIFTY,)
    snapshot = state.snapshot(_NIFTY)
    assert snapshot is not None
    assert len(snapshot.candles) == 1
    assert snapshot.last_price == Decimal("25000")


def test_a_forgotten_instrument_is_rebuilt_from_scratch_if_it_returns() -> None:
    """Eviction has to reach the aggregation state, not only the history.

    Dropping the retained candles while the builder still held an open minute
    and a finalized-minute marker would move the leak rather than close it: the
    instrument's return would emit a candle assembled before the gap, which is
    worse than the memory the eviction was meant to reclaim.
    """
    state = MarketState()
    state.record_tick(_tick(0, "100"))
    state.record_tick(_tick(1, "101"))
    state.record_tick(_tick(2, "102"))
    snapshot = state.snapshot(_RELIANCE)
    assert snapshot is not None
    assert [candle.start_time for candle in snapshot.candles] == [
        _SESSION_OPEN + timedelta(minutes=1)
    ]

    state.forget(_RELIANCE)

    # Without forgetting the builder's state too, this tick closes the minute
    # left open eight minutes ago and files it as current history.
    assert state.record_tick(_tick(10, "200")) is None
    assert state.record_tick(_tick(11, "201")) is None
    returned = state.snapshot(_RELIANCE)
    assert returned is not None
    assert returned.candles == ()

    # Aggregation resumes from the first minute watched end to end.
    assert state.record_tick(_tick(12, "202")) is not None
    resumed = state.snapshot(_RELIANCE)
    assert resumed is not None
    assert [candle.start_time for candle in resumed.candles] == [
        _SESSION_OPEN + timedelta(minutes=11)
    ]


def test_retain_drops_every_instrument_outside_the_declared_set() -> None:
    state = MarketState()
    state.backfill(_RELIANCE, (_ohlcv(0),))
    state.record_tick(_tick(0, "25000", instrument=_NIFTY))

    assert state.retain((_NIFTY,)) == (_RELIANCE,)
    assert state.instruments() == (_NIFTY,)
    # Naming an instrument it never watched must not conjure state for it, and
    # a second pass over the same set has nothing left to drop.
    assert state.retain((_NIFTY, _RELIANCE)) == ()
    assert state.instruments() == (_NIFTY,)


def test_retain_with_an_empty_set_drops_everything() -> None:
    state = MarketState()
    state.backfill(_RELIANCE, (_ohlcv(0),))
    state.record_tick(_tick(0, "25000", instrument=_NIFTY))

    assert state.retain(()) == (_NIFTY, _RELIANCE)
    assert state.snapshots() == ()
