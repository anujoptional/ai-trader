from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from ai_trader.broker import Instrument, MarketTick
from ai_trader.broker.groww import GrowwBroker, GrowwInstrument, GrowwStreamError
from ai_trader.broker.groww_stream import GrowwLtpStream


@pytest.mark.parametrize("raw_timestamp", [1_789_122_509, 1_789_122_509_000])
def test_stream_normalizes_ticks_and_unsubscribes(raw_timestamp: int) -> None:
    feed = Mock()
    feed.get_ltp.return_value = {
        "NSE": {
            "CASH": {
                "2885": {
                    "tsInMillis": raw_timestamp,
                    "ltp": 1234.5,
                }
            }
        }
    }
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )

    def subscribe(
        instrument_list: list[dict[str, str]],
        on_data_received: Callable[[dict[str, object]], None],
    ) -> None:
        on_data_received(
            {
                "exchange": "NSE",
                "segment": "CASH",
                "feed_key": "2885",
            }
        )

    feed.subscribe_ltp.side_effect = subscribe
    received: list[MarketTick] = []
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    ticks = stream.collect(
        max_ticks=1,
        timeout_seconds=1,
        on_tick=received.append,
    )

    sdk_instruments = [{"exchange": "NSE", "segment": "CASH", "exchange_token": "2885"}]
    feed.subscribe_ltp.assert_called_once()
    feed.unsubscribe_ltp.assert_called_once_with(sdk_instruments)
    expected = MarketTick(
        instrument=resolved.instrument,
        price=Decimal("1234.5"),
        timestamp=datetime(2026, 9, 11, 10, 28, 29, tzinfo=UTC),
    )
    assert ticks == (expected,)
    assert received == [expected]


@pytest.mark.parametrize(
    ("volume_field", "expected_volume"),
    [
        ({"volume": 1_500_000}, 1_500_000),
        ({"volume": 1_500_000.0}, 1_500_000),
        ({"volume": 0.0}, None),
        ({}, None),
    ],
    ids=["integer", "double", "unset zero", "absent"],
)
def test_stream_normalizes_cumulative_volume(
    volume_field: dict[str, object],
    expected_volume: int | None,
) -> None:
    # Groww transports volume as a protobuf double, so an unset field arrives as
    # 0.0. Treating that as a real reading would make it a differencing baseline
    # and attribute a whole session's volume to a single minute.
    feed = Mock()
    raw_tick = {"tsInMillis": 1_789_122_509_000, "ltp": 1234.5, **volume_field}
    feed.get_ltp.return_value = {"NSE": {"CASH": {"2885": raw_tick}}}
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )

    def subscribe(
        instrument_list: list[dict[str, str]],
        on_data_received: Callable[[dict[str, object]], None],
    ) -> None:
        on_data_received({"exchange": "NSE", "segment": "CASH", "feed_key": "2885"})

    feed.subscribe_ltp.side_effect = subscribe
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    ticks = stream.collect(max_ticks=1, timeout_seconds=1)

    assert ticks[0].cumulative_volume == expected_volume


def test_stream_times_out_without_ticks_and_unsubscribes() -> None:
    feed = Mock()
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    ticks = stream.collect(max_ticks=5, timeout_seconds=0)

    assert ticks == ()
    feed.unsubscribe_ltp.assert_called_once_with(
        [{"exchange": "NSE", "segment": "CASH", "exchange_token": "2885"}]
    )


def test_stream_stops_after_maximum_tick_count() -> None:
    feed = Mock()
    feed.get_ltp.return_value = {
        "NSE": {"CASH": {"2885": {"tsInMillis": 1_789_122_509_000, "ltp": 1234.5}}}
    }
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )

    def subscribe(
        instrument_list: list[dict[str, str]],
        on_data_received: Callable[[dict[str, object]], None],
    ) -> None:
        metadata = {"exchange": "NSE", "segment": "CASH", "feed_key": "2885"}
        for _ in range(6):
            on_data_received(metadata)

    feed.subscribe_ltp.side_effect = subscribe
    received: list[MarketTick] = []

    ticks = GrowwLtpStream(feed=feed, instruments=(resolved,)).collect(
        max_ticks=5,
        timeout_seconds=1,
        on_tick=received.append,
    )

    assert len(ticks) == 5
    assert received == list(ticks)
    assert feed.get_ltp.call_count == 5
    feed.unsubscribe_ltp.assert_called_once()


def test_stream_rejects_invalid_tick_and_still_unsubscribes() -> None:
    feed = Mock()
    feed.get_ltp.return_value = {
        "NSE": {"CASH": {"2885": {"tsInMillis": 42, "ltp": 1234.5}}}
    }
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )

    def subscribe(
        instrument_list: list[dict[str, str]],
        on_data_received: Callable[[dict[str, object]], None],
    ) -> None:
        on_data_received({"exchange": "NSE", "segment": "CASH", "feed_key": "2885"})

    feed.subscribe_ltp.side_effect = subscribe
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    with pytest.raises(GrowwStreamError, match="stream failed"):
        stream.collect(max_ticks=1, timeout_seconds=1)

    feed.unsubscribe_ltp.assert_called_once()


def test_broker_resolves_groww_symbol_and_builds_stream_without_raw_metadata() -> None:
    client = Mock()
    client.get_instrument_by_groww_symbol.return_value = {
        "exchange": "NSE",
        "exchange_token": "2885",
        "trading_symbol": "RELIANCE",
        "groww_symbol": "NSE-RELIANCE",
        "segment": "CASH",
        "provider_only_field": "ignored",
    }
    feed = Mock()
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")

    with patch("ai_trader.broker.groww.GrowwFeed", return_value=feed) as feed_class:
        stream = GrowwBroker(client).create_ltp_stream((instrument,))

    client.get_instrument_by_groww_symbol.assert_called_once_with("NSE-RELIANCE")
    feed_class.assert_called_once_with(client)
    assert isinstance(stream, GrowwLtpStream)
