import asyncio
import logging
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from ai_trader.broker import Instrument, MarketTick
from ai_trader.broker.groww import (
    GrowwBroker,
    GrowwInstrument,
    GrowwStreamConnectionError,
    GrowwStreamError,
    _FeedLogSink,
)
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


def test_a_subscribe_failure_is_reported_as_a_transport_failure() -> None:
    """The class decides whether a supervisor reconnects, so it has to be right."""
    feed = Mock()
    feed.subscribe_ltp.side_effect = OSError("connection reset")
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    with pytest.raises(GrowwStreamConnectionError, match="stream failed"):
        stream.collect(max_ticks=1, timeout_seconds=0)

    # Nothing was subscribed, so there is nothing to unsubscribe from.
    feed.unsubscribe_ltp.assert_not_called()


def test_an_unsubscribe_failure_is_reported_as_a_transport_failure() -> None:
    """A subscription left behind is only clearable by a fresh connection."""
    feed = Mock()
    feed.unsubscribe_ltp.side_effect = OSError("connection reset")
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    with pytest.raises(GrowwStreamConnectionError, match="stream failed"):
        stream.collect(max_ticks=1, timeout_seconds=0)


def test_a_callback_failure_outranks_a_transport_failure() -> None:
    """Both can fail at once, and the deterministic one has to win.

    A callback that raises leaves the feed in a state the unsubscribe may then
    trip over, so the pair arrives together. Reporting that as retryable would
    have a supervisor reconnect into the same payload it cannot handle, once per
    backoff, until its failure budget runs out with the real cause unnamed.
    """
    feed = Mock()
    feed.get_ltp.return_value = {
        "NSE": {"CASH": {"2885": {"tsInMillis": 42, "ltp": 1234.5}}}
    }
    feed.unsubscribe_ltp.side_effect = OSError("connection reset")
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

    with pytest.raises(GrowwStreamError) as raised:
        stream.collect(max_ticks=1, timeout_seconds=1)

    assert not isinstance(raised.value, GrowwStreamConnectionError)


def _deliver_one_frame(
    feed: Mock,
    metadata: dict[str, object] | None = None,
) -> None:
    """Have ``subscribe_ltp`` push a single frame through the callback."""
    frame = metadata or {"exchange": "NSE", "segment": "CASH", "feed_key": "2885"}

    def subscribe(
        instrument_list: list[dict[str, str]],
        on_data_received: Callable[[dict[str, object]], None],
    ) -> None:
        on_data_received(frame)

    feed.subscribe_ltp.side_effect = subscribe


class _ConsumerBug(RuntimeError):
    """A failure originating in the caller's callback, not in the feed."""


def _valid_feed() -> Mock:
    feed = Mock()
    feed.get_ltp.return_value = {
        "NSE": {"CASH": {"2885": {"tsInMillis": 1_789_122_509_000, "ltp": 1234.5}}}
    }
    return feed


def test_a_consumer_callback_failure_reaches_the_caller_whole() -> None:
    """The callback runs on the vendor's thread, so it has to be carried back.

    Folding it into GrowwStreamError would leave a supervisor holding a
    deterministic bug wearing a transport failure's name, with no type, message
    or traceback to tell them apart by -- and reconnecting into the same
    payload it cannot handle until its budget runs out.
    """
    feed = _valid_feed()
    _deliver_one_frame(feed)
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )

    def explode(tick: MarketTick) -> None:
        raise _ConsumerBug("the consumer could not handle this tick")

    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    with pytest.raises(_ConsumerBug, match="could not handle this tick"):
        stream.collect(max_ticks=1, timeout_seconds=1, on_tick=explode)

    # The subscription is still released: the caller's bug is not a reason to
    # leak a connection.
    feed.unsubscribe_ltp.assert_called_once()


def test_a_consumer_failure_outranks_the_teardown_it_causes() -> None:
    """A callback that stops a collection also tends to trip the unsubscribe.

    Both then arrive together, and the one worth reporting is the cause.
    """
    feed = _valid_feed()
    feed.unsubscribe_ltp.side_effect = OSError("connection reset")
    _deliver_one_frame(feed)
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )

    def explode(tick: MarketTick) -> None:
        raise _ConsumerBug("the consumer could not handle this tick")

    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    with pytest.raises(_ConsumerBug):
        stream.collect(max_ticks=1, timeout_seconds=1, on_tick=explode)


def test_only_the_first_consumer_failure_is_reported() -> None:
    """Later callbacks failing the same way are symptoms of the first."""
    feed = _valid_feed()
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )

    def subscribe(
        instrument_list: list[dict[str, str]],
        on_data_received: Callable[[dict[str, object]], None],
    ) -> None:
        metadata = {"exchange": "NSE", "segment": "CASH", "feed_key": "2885"}
        for _ in range(3):
            on_data_received(metadata)

    feed.subscribe_ltp.side_effect = subscribe
    attempts: list[str] = []

    def explode(tick: MarketTick) -> None:
        attempts.append(f"failure {len(attempts) + 1}")
        raise _ConsumerBug(attempts[-1])

    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    with pytest.raises(_ConsumerBug, match="failure 1"):
        stream.collect(max_ticks=10, timeout_seconds=1, on_tick=explode)

    # All three frames still reached the callback -- the feed keeps pushing
    # until collect tears the subscription down -- so the point is the choice
    # of which failure surfaced, not that the later two never happened.
    assert attempts == ["failure 1", "failure 2", "failure 3"]


def test_a_frame_whose_quote_has_not_landed_yet_is_skipped() -> None:
    """get_ltp hands back the feed's own live dictionary, rewritten in place.

    Our entry can be absent even though the frame announcing it just arrived.
    Indexing would turn that ordinary race into a KeyError, which this module
    classes as unretryable -- ending the session over one dropped tick whose
    total the next frame carries anyway.
    """
    feed = Mock()
    feed.get_ltp.return_value = {"NSE": {"CASH": {}}}
    _deliver_one_frame(feed)
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    assert stream.collect(max_ticks=1, timeout_seconds=0) == ()
    feed.unsubscribe_ltp.assert_called_once()


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            {"exchange": "NSE", "segment": "CASH", "feed_key": 2885},
            "non-string feed_key: int",
        ),
        (
            {"exchange": "BSE", "segment": "CASH", "feed_key": "2885"},
            "arrived as 'BSE'/'CASH', expected 'NSE'/'CASH'",
        ),
    ],
    ids=["malformed key", "wrong exchange"],
)
def test_a_malformed_frame_names_what_was_wrong_with_it(
    frame: dict[str, object],
    expected: str,
) -> None:
    """collect raises a fixed message, so the detail rides on the cause.

    Without it a supervisor is told only that the stream failed, and cannot
    tell a schema change, which needs code, from a network fault, which needs
    patience. The exception is raised on the vendor's thread where nothing can
    catch it, so being chained is the only way it survives at all.
    """
    feed = _valid_feed()
    _deliver_one_frame(feed, frame)
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )
    stream = GrowwLtpStream(feed=feed, instruments=(resolved,))

    with pytest.raises(GrowwStreamError) as raised:
        stream.collect(max_ticks=1, timeout_seconds=1)

    assert not isinstance(raised.value, GrowwStreamConnectionError)
    assert expected in str(raised.value.__cause__)


class _FakeSocket:
    """The NATS socket, reduced to the one coroutine teardown calls."""

    def __init__(self) -> None:
        self.closes = 0

    async def close(self) -> None:
        self.closes += 1


class _FakeNatsClient:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._socket = _FakeSocket()


class _FeedWithTransport:
    """A ``GrowwFeed`` stand-in carrying a real loop on a real thread.

    A mocked loop would let a teardown that never reached one pass, and reaching
    it is the whole point: the vendor's socket can only be closed from the
    thread that owns it, and only a running loop can show whether it was.
    """

    _nats_clients: dict[object, object] = {}

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        running = threading.Event()

        def serve() -> None:
            self.loop.call_soon(running.set)
            self.loop.run_forever()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        running.wait(timeout=5.0)
        self._nats_client = _FakeNatsClient(self.loop)
        self._client_key = ("socket-jwt", "seed")
        self.registry[self._client_key] = self._nats_client

    @property
    def registry(self) -> dict[object, object]:
        """The vendor's process-global client dict, which only ever grows."""
        return type(self)._nats_clients

    @property
    def client_key(self) -> object:
        return self._client_key

    @property
    def socket(self) -> _FakeSocket:
        return self._nats_client._socket

    def dispose(self) -> None:
        """Undo whatever a test left, so loops and threads do not accumulate."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5.0)
        if not self.thread.is_alive():
            self.loop.close()
        self.registry.pop(self._client_key, None)


@pytest.fixture
def transport_feed() -> Iterator[_FeedWithTransport]:
    feed = _FeedWithTransport()
    try:
        yield feed
    finally:
        feed.dispose()


def _transport_stream(feed: object) -> GrowwLtpStream:
    """Build a stream over ``feed``, for tests that only exercise teardown."""
    resolved = GrowwInstrument(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        exchange_token="2885",
    )
    return GrowwLtpStream(feed=feed, instruments=(resolved,))


def test_close_hands_the_vendor_transport_back(
    transport_feed: _FeedWithTransport,
) -> None:
    """The SDK opens a websocket, a loop and a daemon thread and closes none.

    Streams are discarded routinely rather than exceptionally -- a supervisor
    rebuilds one after every silent session -- so without this a quiet afternoon
    strands dozens of live sockets, their threads, and registry entries that
    garbage collection can never reach because the dict holding them is
    class-level and keyed on a value that is fresh every time.
    """
    stream = _transport_stream(transport_feed)

    stream.close()

    transport_feed.thread.join(timeout=5.0)
    assert transport_feed.socket.closes == 1
    assert not transport_feed.thread.is_alive()
    assert transport_feed.client_key not in transport_feed.registry


def test_closing_twice_does_not_tear_down_twice(
    transport_feed: _FeedWithTransport,
) -> None:
    """Idempotence is not politeness here, it is the difference in latency.

    The first close stops the loop, so a second attempt to schedule a coroutine
    on it would be accepted, never run, and block the caller for the full
    five-second bound -- on a path a supervisor takes while already handling a
    failure.
    """
    stream = _transport_stream(transport_feed)
    stream.close()

    started = time.monotonic()
    stream.close()
    elapsed = time.monotonic() - started

    assert transport_feed.socket.closes == 1
    assert elapsed < 1.0


def test_a_stream_used_as_a_context_manager_releases_on_exit(
    transport_feed: _FeedWithTransport,
) -> None:
    with _transport_stream(transport_feed):
        assert transport_feed.socket.closes == 0

    transport_feed.thread.join(timeout=5.0)
    assert transport_feed.socket.closes == 1


def test_close_leaves_a_registry_entry_that_is_no_longer_ours(
    transport_feed: _FeedWithTransport,
) -> None:
    """The vendor means that dict to be shared; only a quirk of key generation
    stops it ever being so. If that is fixed upstream, evicting someone else's
    live client would be worse than leaking ours.
    """
    stream = _transport_stream(transport_feed)
    someone_else = object()
    transport_feed.registry[transport_feed.client_key] = someone_else

    stream.close()

    assert transport_feed.registry[transport_feed.client_key] is someone_else


@pytest.mark.parametrize(
    "feed",
    [object(), Mock()],
    ids=["nothing to release", "vendor internals renamed"],
)
def test_close_tolerates_a_feed_whose_internals_it_cannot_find(feed: object) -> None:
    """Every step of teardown is optional, because callers close on paths that
    are already handling a failure. A vendor upgrade that renames any of these
    attributes has to turn this back into today's leak, not into a crash that
    displaces the error being handled.
    """
    _transport_stream(feed).close()


@pytest.mark.usefixtures("isolated_feed_log")
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


def _stream_broker() -> GrowwBroker:
    client = Mock()
    client.get_instrument_by_groww_symbol.return_value = {
        "exchange": "NSE",
        "exchange_token": "2885",
        "trading_symbol": "RELIANCE",
        "groww_symbol": "NSE-RELIANCE",
        "segment": "CASH",
    }
    return GrowwBroker(client)


@pytest.fixture
def isolated_feed_log() -> Iterator[None]:
    """Undo the module's one-time global logging mutation after each test."""
    from ai_trader.broker import groww

    logger = logging.getLogger("growwapi")
    handlers = list(logger.handlers)
    propagate = logger.propagate
    yield
    logger.handlers = handlers
    logger.propagate = propagate
    groww._FEED_LOG_INSTALLED = False
    groww._FEED_LOG_SINK = _FeedLogSink()


@pytest.mark.usefixtures("isolated_feed_log")
def test_stream_connect_gives_up_at_the_timeout_instead_of_blocking() -> None:
    """A feed that connects forever must not outlive the caller's budget.

    The real library connects inside its constructor and retries for about four
    minutes against a server that accepts the socket but never completes the
    handshake. Without a bound, a ninety-second collection ran for 4m22s.
    """
    release = threading.Event()
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")

    def never_connects(_client: object) -> object:
        release.wait(30)
        raise AssertionError("the abandoned connect should never be awaited")

    try:
        with patch("ai_trader.broker.groww.GrowwFeed", side_effect=never_connects):
            started = time.monotonic()
            with pytest.raises(GrowwStreamConnectionError, match="timed out") as raised:
                _stream_broker().create_ltp_stream(
                    (instrument,), connect_timeout_seconds=0.2
                )
            elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 5
    assert "0.2 seconds" in str(raised.value)


@pytest.mark.usefixtures("isolated_feed_log")
def test_stream_connect_reports_contained_vendor_errors() -> None:
    """Vendor transport errors are folded into ours, not dumped to stderr."""
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")

    def fails_loudly(_client: object) -> object:
        vendor = logging.getLogger("growwapi.groww.nats_client")
        vendor.error("Error: %s", "")
        vendor.error("Error: %s", "nats: no servers available")
        raise OSError("socket closed")

    with patch("ai_trader.broker.groww.GrowwFeed", side_effect=fails_loudly):
        with pytest.raises(GrowwStreamConnectionError) as raised:
            _stream_broker().create_ltp_stream((instrument,))

    message = str(raised.value)
    assert "2 transport errors reported" in message
    assert "nats: no servers available" in message
    assert logging.getLogger("growwapi").propagate is False


@pytest.mark.usefixtures("isolated_feed_log")
def test_stream_connect_rejects_a_non_positive_timeout() -> None:
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")

    with pytest.raises(ValueError, match="timeout must be positive"):
        _stream_broker().create_ltp_stream((instrument,), connect_timeout_seconds=0)


def test_feed_log_sink_counts_empty_records_without_quoting_them() -> None:
    """The library's empty ``Error:`` lines are counted but carry no detail."""
    sink = _FeedLogSink()
    record = logging.LogRecord(
        name="growwapi",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Error: %s",
        args=("",),
        exc_info=None,
    )
    sink.emit(record)
    sink.emit(record)

    assert sink.count == 2
    assert sink.summarize_since(0) == "; 2 transport errors reported"


def test_feed_log_sink_counts_an_unformattable_record_without_raising() -> None:
    """A record whose arguments do not match its format string is still noise.

    Raising here would carry the failure back into the library's own logging
    call, on the library's own thread, which is precisely what a handler is
    contractually forbidden from doing.
    """
    sink = _FeedLogSink()
    record = logging.LogRecord(
        name="growwapi",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Error: %d",
        args=("not a number",),
        exc_info=None,
    )
    sink.emit(record)

    assert sink.count == 1
    assert sink.summarize_since(0) == "; 1 transport error reported"


def test_feed_log_sink_reports_nothing_when_no_records_are_new() -> None:
    sink = _FeedLogSink()
    record = logging.LogRecord(
        name="growwapi",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="stale",
        args=(),
        exc_info=None,
    )
    sink.emit(record)

    assert sink.summarize_since(sink.count) == ""
