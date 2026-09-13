from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument, MarketTick
from ai_trader.market import Candle, CandleBuilder, InvalidTickError

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_NIFTY = Instrument(exchange="NSE", trading_symbol="NIFTY")


def _tick(
    instrument: Instrument,
    timestamp: datetime,
    price: str,
) -> MarketTick:
    return MarketTick(
        instrument=instrument,
        timestamp=timestamp,
        price=Decimal(price),
    )


def test_single_tick_produces_single_candle_on_flush() -> None:
    builder = CandleBuilder()
    timestamp = datetime(2026, 9, 14, 10, 0, 17, tzinfo=UTC)

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
