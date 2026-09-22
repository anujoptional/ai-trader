from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from threading import Barrier, Lock, Thread

import pytest

from ai_trader.broker import Instrument, MarketTick
from ai_trader.market import Candle, CandleBuilder, InvalidTickError

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_NIFTY = Instrument(exchange="NSE", trading_symbol="NIFTY")
_ONE_MINUTE = timedelta(minutes=1)


def _tick(
    instrument: Instrument,
    timestamp: datetime,
    price: str,
    cumulative_volume: int | None = None,
) -> MarketTick:
    return MarketTick(
        instrument=instrument,
        timestamp=timestamp,
        price=Decimal(price),
        cumulative_volume=cumulative_volume,
    )


def _prime(builder: CandleBuilder, instrument: Instrument, minute: datetime) -> None:
    """Spend the opening minute the builder discards for ``instrument``.

    The builder never emits the first minute it observes, so a test that wants a
    real candle for ``minute`` must let an earlier minute take that place. The
    priming tick is consumed entirely by the discarded minute and never reaches
    the candles under assertion.
    """
    builder.add_tick(_tick(instrument, minute - _ONE_MINUTE, "1"))


def test_opening_partial_minute_is_never_emitted() -> None:
    """A stream is joined mid-minute, so its first minute is only a fragment."""
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=47), "100"))

    finalized = builder.add_tick(_tick(_RELIANCE, minute + _ONE_MINUTE, "101"))

    assert finalized is None
    assert emitted == []


def test_opening_minute_is_discarded_on_flush_too() -> None:
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=47), "100"))

    assert builder.flush() == ()
    assert emitted == []


def test_discarded_opening_minute_cannot_be_restarted_by_a_late_tick() -> None:
    """Discarded is not forgotten: the minute is still final and stays empty."""
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=47), "100"))
    builder.add_tick(_tick(_RELIANCE, minute + _ONE_MINUTE, "101"))

    late = _tick(_RELIANCE, minute + timedelta(seconds=55), "999")

    assert builder.add_tick(late) is None
    assert builder.late_tick_count == 1
    assert emitted == []


def test_discarded_opening_minute_stays_final_across_a_flush() -> None:
    """Flushing the discarded minute must still mark it final.

    Otherwise the builder forgets it ever saw the instrument: a late tick would
    open a fresh candle for the same minute, and the minute after it would be
    discarded as though it were the first, costing a second real candle.
    """
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=47), "100"))
    assert builder.flush() == ()

    late = _tick(_RELIANCE, minute + timedelta(seconds=55), "999")

    assert builder.add_tick(late) is None
    assert builder.late_tick_count == 1
    builder.add_tick(_tick(_RELIANCE, minute + _ONE_MINUTE, "101"))
    assert [candle.start_time for candle in builder.flush()] == [minute + _ONE_MINUTE]
    assert [candle.open for candle in emitted] == [Decimal("101")]


def test_first_emitted_candle_opens_at_its_minutes_true_open() -> None:
    """The fragment's prices must not leak into the candle that follows it."""
    builder = CandleBuilder()
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=47), "500"))
    second = minute + _ONE_MINUTE
    builder.add_tick(_tick(_RELIANCE, second, "100"))
    builder.add_tick(_tick(_RELIANCE, second + timedelta(seconds=30), "104"))

    finalized = builder.add_tick(_tick(_RELIANCE, minute + timedelta(minutes=2), "105"))

    assert finalized is not None
    assert finalized.start_time == second
    assert finalized.open == Decimal("100")
    assert finalized.high == Decimal("104")
    assert finalized.low == Decimal("100")


def test_each_instrument_discards_its_own_opening_minute() -> None:
    """Joining one instrument's stream must not cost another a real candle."""
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    builder.add_tick(_tick(_RELIANCE, minute, "100"))
    builder.add_tick(_tick(_RELIANCE, minute + _ONE_MINUTE, "101"))
    # NIFTY is subscribed a minute later, once RELIANCE already has one behind it.
    builder.add_tick(_tick(_NIFTY, minute + _ONE_MINUTE, "25000"))
    builder.add_tick(_tick(_NIFTY, minute + timedelta(minutes=2), "25010"))
    builder.flush()

    assert [(candle.instrument, candle.start_time) for candle in emitted] == [
        (_RELIANCE, minute + _ONE_MINUTE),
        (_NIFTY, minute + timedelta(minutes=2)),
    ]


def test_single_tick_produces_single_candle_on_flush() -> None:
    builder = CandleBuilder()
    timestamp = datetime(2026, 9, 14, 10, 0, 17, tzinfo=UTC)
    _prime(builder, _RELIANCE, datetime(2026, 9, 14, 10, 0, tzinfo=UTC))

    assert builder.add_tick(_tick(_RELIANCE, timestamp, "100.25")) is None
    assert builder.flush() == (
        Candle(
            instrument=_RELIANCE,
            start_time=datetime(2026, 9, 14, 10, 0, tzinfo=UTC),
            end_time=datetime(2026, 9, 14, 10, 1, tzinfo=UTC),
            open=Decimal("100.25"),
            high=Decimal("100.25"),
            low=Decimal("100.25"),
            close=Decimal("100.25"),
        ),
    )


def test_many_ticks_in_one_minute_calculate_ohlc() -> None:
    builder = CandleBuilder()
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, minute)
    for seconds, price in ((5, "100"), (15, "105.5"), (35, "95.25"), (59, "102")):
        assert (
            builder.add_tick(
                _tick(_RELIANCE, minute + timedelta(seconds=seconds), price)
            )
            is None
        )

    candle = builder.flush()[0]
    assert candle.open == Decimal("100")
    assert candle.high == Decimal("105.5")
    assert candle.low == Decimal("95.25")
    assert candle.close == Decimal("102")
    assert candle.volume is None


def test_new_minute_finalizes_previous_candle_and_callback_emits_it() -> None:
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    first_minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, first_minute)
    builder.add_tick(_tick(_RELIANCE, first_minute, "100"))

    finalized = builder.add_tick(
        _tick(_RELIANCE, first_minute + timedelta(minutes=1), "101")
    )

    assert finalized is not None
    assert finalized.start_time == first_minute
    assert finalized.close == Decimal("100")
    assert emitted == [finalized]
    assert builder.flush()[0].start_time == first_minute + timedelta(minutes=1)


def test_multiple_instruments_roll_over_independently() -> None:
    builder = CandleBuilder()
    first_minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, first_minute)
    _prime(builder, _NIFTY, first_minute)
    builder.add_tick(_tick(_RELIANCE, first_minute, "100"))
    builder.add_tick(_tick(_NIFTY, first_minute, "25000"))

    reliance_candle = builder.add_tick(
        _tick(_RELIANCE, first_minute + timedelta(minutes=1), "101")
    )

    assert reliance_candle is not None
    assert reliance_candle.instrument == _RELIANCE
    remaining = builder.flush()
    assert {candle.instrument for candle in remaining} == {_RELIANCE, _NIFTY}
    nifty_candle = next(candle for candle in remaining if candle.instrument == _NIFTY)
    assert nifty_candle.start_time == first_minute


def test_gap_does_not_manufacture_empty_candles() -> None:
    builder = CandleBuilder()
    first_minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, first_minute)
    builder.add_tick(_tick(_RELIANCE, first_minute, "100"))

    finalized = builder.add_tick(
        _tick(_RELIANCE, first_minute + timedelta(minutes=3), "103")
    )
    remaining = builder.flush()

    assert finalized is not None
    assert finalized.start_time == first_minute
    assert len(remaining) == 1
    assert remaining[0].start_time == first_minute + timedelta(minutes=3)


def test_duplicate_timestamps_use_first_arrival_for_open_and_last_for_close() -> None:
    builder = CandleBuilder()
    timestamp = datetime(2026, 9, 14, 10, 0, 15, tzinfo=UTC)
    _prime(builder, _RELIANCE, datetime(2026, 9, 14, 10, 0, tzinfo=UTC))
    for price in ("100", "105", "99"):
        builder.add_tick(_tick(_RELIANCE, timestamp, price))

    candle = builder.flush()[0]
    assert candle.open == Decimal("100")
    assert candle.high == Decimal("105")
    assert candle.low == Decimal("99")
    assert candle.close == Decimal("99")


def test_out_of_order_ticks_in_open_minute_use_event_time() -> None:
    builder = CandleBuilder()
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, minute)
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=20), "100"))
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=50), "110"))
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=10), "90"))

    candle = builder.flush()[0]
    assert candle.open == Decimal("90")
    assert candle.high == Decimal("110")
    assert candle.low == Decimal("90")
    assert candle.close == Decimal("110")


def test_tick_for_finalized_minute_is_ignored() -> None:
    builder = CandleBuilder()
    first_minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, first_minute)
    builder.add_tick(_tick(_RELIANCE, first_minute, "100"))
    finalized = builder.add_tick(
        _tick(_RELIANCE, first_minute + timedelta(minutes=1), "101")
    )

    assert (
        builder.add_tick(_tick(_RELIANCE, first_minute + timedelta(seconds=30), "1000"))
        is None
    )
    assert builder.late_tick_count == 1
    assert finalized is not None
    assert finalized.high == Decimal("100")
    assert builder.flush()[0].open == Decimal("101")


def test_timezone_aware_tick_is_normalized_to_utc() -> None:
    india_timezone = timezone(timedelta(hours=5, minutes=30))
    builder = CandleBuilder()
    _prime(builder, _RELIANCE, datetime(2026, 9, 14, 10, 0, tzinfo=UTC))
    builder.add_tick(
        _tick(
            _RELIANCE,
            datetime(2026, 9, 14, 15, 30, 42, tzinfo=india_timezone),
            "100",
        )
    )

    candle = builder.flush()[0]
    assert candle.start_time == datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    assert candle.end_time == datetime(2026, 9, 14, 10, 1, tzinfo=UTC)
    assert candle.start_time.tzinfo is UTC


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 9, 14, 10, 0),
        datetime(1900, 1, 1, tzinfo=UTC),
        datetime(2200, 1, 1, tzinfo=UTC),
    ],
)
def test_invalid_timestamp_is_rejected_without_corrupting_state(
    timestamp: datetime,
) -> None:
    builder = CandleBuilder()

    with pytest.raises(InvalidTickError):
        builder.add_tick(_tick(_RELIANCE, timestamp, "100"))

    assert builder.flush() == ()


def test_flush_keeps_finalized_minutes_final() -> None:
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, minute)
    builder.add_tick(_tick(_RELIANCE, minute, "100"))

    assert len(builder.flush()) == 1
    late = _tick(_RELIANCE, minute + timedelta(seconds=30), "1000")

    assert builder.add_tick(late) is None
    assert builder.late_tick_count == 1
    assert builder.flush() == ()
    assert len(emitted) == 1


def test_cumulative_volume_is_differenced_into_per_minute_volume() -> None:
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)

    builder.add_tick(_tick(_RELIANCE, minute, "100", 1_000))
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(seconds=30), "101", 1_500))
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(minutes=1), "102", 2_200))
    builder.add_tick(_tick(_RELIANCE, minute + timedelta(minutes=2), "103", 2_500))
    builder.flush()

    # The minute with no earlier reading to difference against is exactly the
    # opening minute, which is discarded anyway, so no emitted candle is left
    # without volume. Downstream consumers of volume — session VWAP above all —
    # therefore never see a hole at the start of a stream.
    assert [candle.start_time for candle in emitted] == [
        minute + timedelta(minutes=1),
        minute + timedelta(minutes=2),
    ]
    assert [candle.volume for candle in emitted] == [700, 300]


def test_forgetting_an_instrument_discards_the_minute_it_had_open() -> None:
    """The builder stopped watching, so the open minute is a fragment again.

    Emitting it would hand a consumer a candle covering only the part of the
    minute that was watched -- the same defect the opening minute is discarded
    for, arriving at the other end of the stream.
    """
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, minute)
    builder.add_tick(_tick(_RELIANCE, minute, "100"))
    builder.add_tick(_tick(_RELIANCE, minute + _ONE_MINUTE, "101"))
    assert [candle.start_time for candle in emitted] == [minute]

    builder.forget(_RELIANCE)

    # Without forgetting, this flush emits the minute that was open.
    assert builder.flush() == ()
    assert [candle.start_time for candle in emitted] == [minute]


def test_a_forgotten_instrument_returns_as_a_new_stream() -> None:
    """Its first minute back is a fragment for the reason its first one was."""
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    _prime(builder, _RELIANCE, minute)
    builder.add_tick(_tick(_RELIANCE, minute, "100"))
    builder.add_tick(_tick(_RELIANCE, minute + _ONE_MINUTE, "101"))

    builder.forget(_RELIANCE)

    later = minute + timedelta(minutes=10)
    # Were the open minute still held, this tick would close it and emit a
    # candle built from one tick ten minutes stale.
    assert builder.add_tick(_tick(_RELIANCE, later, "200")) is None
    # And were the finalized-minute record still held, this one would emit the
    # minute above rather than discard it as the fragment it is.
    assert builder.add_tick(_tick(_RELIANCE, later + _ONE_MINUTE, "201")) is None
    assert [candle.start_time for candle in emitted] == [minute]

    assert [candle.start_time for candle in builder.flush()] == [later + _ONE_MINUTE]


def test_forgetting_one_instrument_leaves_the_others_aggregating() -> None:
    emitted: list[Candle] = []
    builder = CandleBuilder(on_candle=emitted.append)
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    for instrument in (_RELIANCE, _NIFTY):
        _prime(builder, instrument, minute)
        builder.add_tick(_tick(instrument, minute, "100"))

    builder.forget(_RELIANCE)

    assert [candle.instrument for candle in builder.flush()] == [_NIFTY]
    assert [candle.instrument for candle in emitted] == [_NIFTY]


def test_forgetting_an_unknown_instrument_leaves_the_builder_untouched() -> None:
    builder = CandleBuilder()

    builder.forget(_RELIANCE)

    assert builder.flush() == ()
    assert builder.late_tick_count == 0


def _candle(**overrides: Decimal) -> Candle:
    """A valid candle, with individual prices replaced for rejection tests."""
    minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    prices = {
        "open": Decimal("100"),
        "high": Decimal("102"),
        "low": Decimal("99"),
        "close": Decimal("101"),
    }
    return Candle(
        instrument=_RELIANCE,
        start_time=minute,
        end_time=minute + _ONE_MINUTE,
        **(prices | overrides),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open", Decimal("0")),
        ("high", Decimal("0")),
        ("low", Decimal("0")),
        ("close", Decimal("0")),
        ("close", Decimal("-1")),
    ],
)
def test_a_non_positive_candle_price_is_refused(field: str, value: Decimal) -> None:
    """Every price field, not just the one an obvious bug would reach.

    A zero close is the dangerous one: the divisions downstream are all guarded
    and would report unavailable, but the trend features have no division to
    guard them and would absorb it as a real observation, staying dragged toward
    zero for their whole span while still reporting ready.
    """
    # Without this the cases could pass on a broken baseline rather than on the
    # price under test.
    assert _candle().close == Decimal("101")

    with pytest.raises(ValueError, match="must be positive"):
        _candle(**{field: value})


@pytest.mark.parametrize(
    ("price", "cumulative_volume"),
    [
        (100.0, None),
        (Decimal("NaN"), None),
        (Decimal("100"), -1),
        # A cash equity cannot trade at or below zero, so these are unset or
        # mis-parsed fields rather than prices. Refusing them at the tick
        # boundary is what keeps them from reaching the candle a minute later,
        # by which point the tick that caused it is unrecoverable.
        (Decimal("0"), None),
        (Decimal("-1"), None),
    ],
)
def test_unusable_tick_payload_is_rejected_without_corrupting_state(
    price: object,
    cumulative_volume: int | None,
) -> None:
    builder = CandleBuilder()
    tick = MarketTick(
        instrument=_RELIANCE,
        timestamp=datetime(2026, 9, 14, 10, 0, tzinfo=UTC),
        price=price,
        cumulative_volume=cumulative_volume,
    )

    with pytest.raises(InvalidTickError):
        builder.add_tick(tick)

    assert builder.flush() == ()


def test_concurrent_ticks_never_duplicate_or_corrupt_a_minute() -> None:
    thread_count = 8
    minutes = 60
    emitted: list[Candle] = []
    emitted_lock = Lock()
    failures: list[BaseException] = []
    ready = Barrier(thread_count)
    first_minute = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)

    def collect(candle: Candle) -> None:
        with emitted_lock:
            emitted.append(candle)

    builder = CandleBuilder(on_candle=collect)

    def feed(offset: int) -> None:
        # Any corruption surfaces as an exception from Candle validation.
        try:
            ready.wait()
            for index in range(minutes):
                builder.add_tick(
                    _tick(
                        _RELIANCE,
                        first_minute + timedelta(minutes=index, seconds=offset),
                        str(100 + offset),
                    )
                )
        except BaseException as error:
            failures.append(error)

    threads = [Thread(target=feed, args=(offset,)) for offset in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    builder.flush()

    start_times = [candle.start_time for candle in emitted]
    assert failures == []
    assert len(start_times) == len(set(start_times))
    assert len(start_times) <= minutes
