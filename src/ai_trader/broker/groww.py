"""Read-only Groww authentication and profile access."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, Self

import pyotp
from growwapi import GrowwAPI, GrowwFeed
from pydantic import BaseModel, ConfigDict

from ai_trader.broker import (
    MAX_HISTORICAL_SPAN,
    BrokerProfile,
    CandleInterval,
    Instrument,
    LastTradedPrice,
    MarketQuote,
    OHLCVCandle,
)
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import GrowwSettings

if TYPE_CHECKING:
    from ai_trader.broker.groww_stream import GrowwLtpStream

_CASH_SEGMENT = "CASH"
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_CANDLE_INTERVALS = {CandleInterval.ONE_MINUTE: "1minute"}
_CALL_ATTEMPTS = 8
_RETRY_BASE_DELAY_SECONDS = 0.5
_MAX_RETRY_DELAY_SECONDS = 2.0
_STREAM_CONNECT_TIMEOUT_SECONDS = 30.0
_FEED_LOGGER_NAME = "growwapi"
_FEED_LOG_HISTORY = 64
_MIN_REASONABLE_EPOCH_SECONDS = int(
    datetime(2000, 1, 1, tzinfo=INDIA_TIMEZONE).timestamp()
)
_MAX_REASONABLE_EPOCH_SECONDS = int(
    datetime(2100, 1, 1, tzinfo=INDIA_TIMEZONE).timestamp()
)


class GrowwBrokerError(RuntimeError):
    """Base error for safe Groww broker failures."""


class GrowwAuthenticationError(GrowwBrokerError):
    """Raised when Groww authentication cannot be completed."""


class GrowwProfileError(GrowwBrokerError):
    """Raised when the Groww profile cannot be retrieved or validated."""


class GrowwMarketDataError(GrowwBrokerError):
    """Raised when Groww market data cannot be retrieved or validated."""


class GrowwStreamError(GrowwBrokerError):
    """Raised when the Groww market-data stream fails."""


class GrowwStreamConnectionError(GrowwStreamError):
    """Raised when a stream fails at the transport rather than the payload.

    The distinction exists so a supervisor can tell a dropped connection, which
    is worth reconnecting for, from a stream that failed while handling a tick.
    The latter is either a payload Groww no longer serves the way this module
    expects or a bug in the consumer's callback, and both are deterministic:
    reconnecting replays the same failure until something stops it.

    Only failures raised by the subscribe and unsubscribe calls themselves are
    classified here. Anything reaching the callback stays on the base class even
    where the underlying cause may have been the transport, because
    misclassifying a payload failure as retryable hides it behind an endless
    reconnect loop, while misclassifying a transport failure merely stops a run
    that then says exactly why it stopped.
    """


@dataclass(frozen=True, slots=True)
class GrowwInstrument:
    """A normalized instrument plus its Groww streaming token."""

    instrument: Instrument
    exchange_token: str


class _GrowwClient(Protocol):
    def get_user_profile(self) -> dict[str, Any]:
        """Return the raw Groww profile payload."""
        ...

    def get_ltp(
        self,
        exchange_trading_symbols: tuple[str, ...],
        segment: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Return raw latest prices."""
        ...

    def get_quote(
        self,
        trading_symbol: str,
        exchange: str,
        segment: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Return a raw detailed quote."""
        ...

    def get_historical_candles(
        self,
        exchange: str,
        segment: str,
        groww_symbol: str,
        start_time: str,
        end_time: str,
        candle_interval: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Return raw historical candles."""
        ...

    def get_instrument_by_groww_symbol(self, groww_symbol: str) -> dict[str, Any]:
        """Return raw Groww instrument metadata."""
        ...


class _GrowwProfilePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nse_enabled: bool
    bse_enabled: bool
    active_segments: tuple[str, ...]
    ddpi_enabled: bool


class _GrowwQuoteOHLC(BaseModel):
    model_config = ConfigDict(extra="ignore")

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


class _GrowwQuotePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    last_price: Decimal
    last_trade_time: int
    ohlc: _GrowwQuoteOHLC
    volume: int
    day_change: Decimal
    day_change_perc: Decimal


class _GrowwInstrumentPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    exchange: str
    exchange_token: str
    trading_symbol: str
    groww_symbol: str
    segment: str


class GrowwBroker:
    """A Groww adapter limited to read-only account and market data."""

    def __init__(self, client: _GrowwClient) -> None:
        self._client = client

    @classmethod
    def authenticate(cls, settings: GrowwSettings) -> Self:
        """Authenticate with TOTP and construct a read-only Groww adapter."""
        subject = "Groww authentication failed."
        # The TOTP is generated per attempt so a retry that crosses a
        # 30-second window uses the code belonging to the window it lands in.
        access_token = _request(
            lambda: GrowwAPI.get_access_token(
                api_key=settings.totp_token.get_secret_value(),
                totp=pyotp.TOTP(settings.totp_secret.get_secret_value()).now(),
            ),
            GrowwAuthenticationError,
            subject,
        )
        with _reading(GrowwAuthenticationError, subject):
            if not isinstance(access_token, str) or not access_token:
                raise TypeError

        # The SDK prints status text during construction. Suppress it so the
        # CLI emits only its explicitly allowlisted profile summary.
        with redirect_stdout(StringIO()):
            client = _request(
                lambda: GrowwAPI(access_token),
                GrowwAuthenticationError,
                subject,
            )

        return cls(client)

    def get_user_profile(self) -> BrokerProfile:
        """Retrieve and sanitize the Groww user profile."""
        subject = "Groww profile retrieval failed."
        raw = _request(self._client.get_user_profile, GrowwProfileError, subject)
        with _reading(GrowwProfileError, subject):
            payload = _GrowwProfilePayload.model_validate(raw)

        exchange_enablement = MappingProxyType(
            {
                "NSE": payload.nse_enabled,
                "BSE": payload.bse_enabled,
            }
        )
        return BrokerProfile(
            exchange_enablement=exchange_enablement,
            active_segments=payload.active_segments,
            ddpi_enabled=payload.ddpi_enabled,
        )

    def get_ltp(
        self,
        instruments: Sequence[Instrument],
    ) -> tuple[LastTradedPrice, ...]:
        """Retrieve normalized latest prices for one or more CASH instruments."""
        if not instruments:
            raise ValueError("At least one instrument is required.")
        if len(instruments) > 50:
            raise ValueError("Groww accepts at most 50 instruments per LTP request.")

        groww_symbols = tuple(_live_symbol(instrument) for instrument in instruments)
        subject = "Groww LTP retrieval failed."
        response = _request(
            lambda: self._client.get_ltp(
                exchange_trading_symbols=groww_symbols,
                segment=_CASH_SEGMENT,
            ),
            GrowwMarketDataError,
            subject,
        )
        with _reading(GrowwMarketDataError, subject):
            return tuple(
                LastTradedPrice(
                    instrument=instrument,
                    price=_decimal(response[groww_symbol]),
                )
                for instrument, groww_symbol in zip(
                    instruments, groww_symbols, strict=True
                )
            )

    def get_quote(self, instrument: Instrument) -> MarketQuote:
        """Retrieve and normalize a detailed CASH quote."""
        subject = "Groww quote retrieval failed."
        raw = _request(
            lambda: self._client.get_quote(
                trading_symbol=instrument.trading_symbol,
                exchange=instrument.exchange,
                segment=_CASH_SEGMENT,
            ),
            GrowwMarketDataError,
            subject,
        )
        with _reading(GrowwMarketDataError, subject):
            payload = _GrowwQuotePayload.model_validate(raw)
            last_trade_at = _groww_epoch_datetime(payload.last_trade_time)

        return MarketQuote(
            instrument=instrument,
            last_price=payload.last_price,
            last_trade_at=last_trade_at,
            open=payload.ohlc.open,
            high=payload.ohlc.high,
            low=payload.ohlc.low,
            previous_close=payload.ohlc.close,
            volume=payload.volume,
            day_change=payload.day_change,
            day_change_percent=payload.day_change_perc,
        )

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]:
        """Retrieve and normalize historical CASH candles.

        The failure names the instrument and window it was asked for. That is
        all caller-supplied, so it adds nothing to what a traceback could leak,
        and ``from None`` still holds: the SDK's own exception may carry the
        request that produced it, headers included. Without the identifiers a
        multi-instrument fetch reports only that something failed, which turns
        one malformed bar into a search of the whole universe.

        The fetch and the reading of it are separated so the message can say
        which of the two failed. A window this large is where that matters most:
        one unreachable minute is worth another run, while one bar Groww now
        serves differently means every run after this one fails the same way.
        """
        _validate_period(start, end, interval)
        subject = (
            "Groww historical data retrieval failed for "
            f"{instrument.exchange}:{instrument.trading_symbol} "
            f"({interval}) over {_groww_datetime(start)} .. "
            f"{_groww_datetime(end)}."
        )
        response = _request(
            lambda: self._client.get_historical_candles(
                exchange=instrument.exchange,
                segment=_CASH_SEGMENT,
                groww_symbol=_historical_symbol(instrument),
                start_time=_groww_datetime(start),
                end_time=_groww_datetime(end),
                candle_interval=_CANDLE_INTERVALS[interval],
            ),
            GrowwMarketDataError,
            subject,
        )
        with _reading(GrowwMarketDataError, subject):
            raw_candles = response["candles"]
            if not isinstance(raw_candles, list):
                raise TypeError
            normalized = (_normalize_candle(candle) for candle in raw_candles)
            return tuple(candle for candle in normalized if candle is not None)

    def resolve_instrument(self, groww_symbol: str) -> GrowwInstrument:
        """Resolve a Groww symbol to normalized metadata and a streaming token."""
        subject = "Groww instrument lookup failed."
        raw = _request(
            lambda: self._client.get_instrument_by_groww_symbol(groww_symbol),
            GrowwMarketDataError,
            subject,
        )
        with _reading(GrowwMarketDataError, subject):
            payload = _GrowwInstrumentPayload.model_validate(raw)
            if payload.groww_symbol != groww_symbol or payload.segment != _CASH_SEGMENT:
                raise ValueError
            if not payload.exchange_token.strip():
                raise ValueError

        return GrowwInstrument(
            instrument=Instrument(
                exchange=payload.exchange,
                trading_symbol=payload.trading_symbol,
            ),
            exchange_token=payload.exchange_token,
        )

    def create_ltp_stream(
        self,
        instruments: Sequence[Instrument],
        *,
        connect_timeout_seconds: float = _STREAM_CONNECT_TIMEOUT_SECONDS,
    ) -> GrowwLtpStream:
        """Create a read-only Groww LTP stream for CASH instruments.

        The connect is bounded by ``connect_timeout_seconds`` because the feed
        library connects inside its constructor and retries on a schedule this
        module cannot configure. A single attempt is made: reconnect policy
        belongs to the caller's supervisor, which already owns backoff, and
        stacking a retry here would multiply the library's own.
        """
        from ai_trader.broker.groww_stream import GrowwLtpStream

        if not instruments:
            raise ValueError("At least one streaming instrument is required.")
        if connect_timeout_seconds <= 0:
            raise ValueError("The stream connect timeout must be positive.")

        resolved = tuple(
            self.resolve_instrument(_historical_symbol(instrument))
            for instrument in instruments
        )
        sink = _install_feed_log_sink()
        baseline = sink.count
        try:
            feed = _call_with_timeout(
                lambda: GrowwFeed(self._client), connect_timeout_seconds
            )
        except TimeoutError:
            raise GrowwStreamConnectionError(
                "Groww stream connection timed out after "
                f"{connect_timeout_seconds:g} seconds"
                f"{sink.summarize_since(baseline)}."
            ) from None
        except Exception:
            raise GrowwStreamConnectionError(
                f"Groww stream connection failed{sink.summarize_since(baseline)}."
            ) from None
        return GrowwLtpStream(feed=feed, instruments=resolved)


def _live_symbol(instrument: Instrument) -> str:
    return f"{instrument.exchange}_{instrument.trading_symbol}"


def _historical_symbol(instrument: Instrument) -> str:
    return f"{instrument.exchange}-{instrument.trading_symbol}"


def _retry_broker_call[T](operation: Callable[[], T]) -> T:
    """Run a Groww call, retrying the transient failures it returns.

    Groww intermittently answers an otherwise valid request with a plain-text
    ``404 page not found`` body, which the SDK surfaces as a decode error. This
    is not a bad request: the identical call succeeds moments later. Measured
    against the live market on 2026-09-21, 94 of 200 raw quote calls failed this
    way, so without a retry a read is worse than a coin flip and the diagnostic
    CLIs fail for no real reason. Authentication is hit by the same fault at a
    similar rate -- 4 of 8 raw token requests, measured the same day -- which is
    why it is retried here too rather than being treated as a credential
    problem.

    The same measurement shaped the schedule. The failures cluster into short
    outages -- 44 of them across 200 calls sampled twice a second, the longest
    3.6 seconds and most no more than one -- but between outages the failure
    rate stays high. Two properties follow. Delays must exceed the longest
    outage, or every attempt lands inside one; and the attempt budget matters
    more than the delay length, because each attempt outside an outage still
    fails roughly half the time. So the delay doubles only until it passes the
    observed outage length and is then held there, which buys eight attempts
    across about twelve seconds rather than four across the same span.

    Only the raw broker call is retried. Normalization stays outside, so a
    genuine schema change surfaces immediately instead of being retried -- and,
    through ``_request`` and ``_reading`` below, is reported as a different
    failure rather than as another unlucky read.
    """
    for attempt in range(_CALL_ATTEMPTS):
        try:
            return operation()
        except Exception:
            if attempt == _CALL_ATTEMPTS - 1:
                raise
            time.sleep(
                min(
                    _RETRY_BASE_DELAY_SECONDS * 2**attempt,
                    _MAX_RETRY_DELAY_SECONDS,
                )
            )
    raise AssertionError("unreachable")


_NO_ANSWER = (
    f" The request did not get through across {_CALL_ATTEMPTS} attempts, so no reply"
    " was read and a later run may succeed."
)

_UNREADABLE_ANSWER = (
    " Groww answered and the reply is not the shape this module reads, so it is a"
    " schema change or a bug here and retrying will reproduce it."
)


def _request[T](
    operation: Callable[[], T],
    error: type[GrowwBrokerError],
    subject: str,
) -> T:
    """Make the raw Groww call, and report not reaching Groww as exactly that.

    ``_retry_broker_call`` already draws the line this pair exists to keep: the
    raw call is retried, because Groww intermittently refuses a request that is
    perfectly valid, and normalization is not, because a schema change repeats.
    The line was drawn for retrying and then discarded at the report -- a single
    ``except Exception`` spanning both halves gave a network outage and a changed
    payload the same sentence, so the run said which instrument failed and never
    which of the two things had gone wrong.

    They are not the same failure and the next move is not the same. An
    unanswered request may well be answered tomorrow. A reply this module cannot
    read will not read tomorrow either, and somebody has to change code before
    any amount of waiting helps.

    The split is structural rather than a list of exception types, and that is
    not a stylistic preference: a transport fault arrives here as ``ValueError``
    -- the SDK's failure to decode Groww's plain-text 404 -- and a rejected
    payload arrives as ``ValueError`` too, so no classification by type could
    tell the two apart. Where the failure happened is the only thing that can.

    Which is also why this half names no cause. Being structural, it knows only
    that nothing was read back, and that covers a request Groww refused, a
    request that never left -- ``authenticate`` mints its TOTP inside the
    operation, on purpose -- and a service that was simply down. Naming the
    connection would be guessing at three cases from a position that can only
    see one fact, and a message that guesses is the bug being fixed here.

    ``from None`` is unchanged, and so is the rule that only caller-supplied
    identifiers reach the message: the SDK's exception may carry the request that
    produced it, headers included.
    """
    try:
        return _retry_broker_call(operation)
    except Exception:
        raise error(subject + _NO_ANSWER) from None


@contextmanager
def _reading(error: type[GrowwBrokerError], subject: str) -> Iterator[None]:
    """Read a reply Groww has already given, and report a bad one as that.

    The other half of ``_request``, and it carries no judgement of its own.
    Everything inside this block runs after Groww has answered, so whatever is
    raised in it is about the answer rather than about reaching the service --
    which is the whole reason it can be reported as not worth retrying.
    """
    try:
        yield
    except Exception:
        raise error(subject + _UNREADABLE_ANSWER) from None


class _FeedLogSink(logging.Handler):
    """Hold the feed library's log records instead of letting them reach stderr.

    The Groww feed layer logs transport trouble through the standard library at
    ERROR level and provides no hook to redirect it. With no handler configured
    -- the normal state for a CLI that prints a JSON summary and nothing else --
    Python's last-resort handler writes those records straight to stderr. A
    single failed connection emits about sixty of them, each reading ``Error:``
    with an empty message because the exception stringifies to nothing, which
    buries the one line the operator actually needs.

    Capturing them is not hiding them. The records are kept here and folded into
    the error this module raises, so the count and the last useful message are
    reported rather than discarded, and the CLI's output contract survives.
    """

    def __init__(self) -> None:
        super().__init__()
        self._records: deque[str] = deque(maxlen=_FEED_LOG_HISTORY)
        self._lock = threading.Lock()
        self._count = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage().strip()
        except Exception:
            # A record whose arguments do not match its format string is still
            # transport noise and still worth counting. Letting the failure out
            # would raise it back into the library's own logging call, on the
            # library's own thread -- which is exactly what a handler is
            # contractually forbidden from doing.
            message = ""
        with self._lock:
            self._count += 1
            if message and message != "Error:":
                self._records.append(message)

    def summarize_since(self, baseline: int) -> str:
        """Describe the records seen since ``baseline``, for an error message.

        Returns an empty string when nothing was logged, so a caller can append
        it unconditionally and keep its plain message unchanged.
        """
        with self._lock:
            new = self._count - baseline
            detail = self._records[-1] if self._records else ""
        if new <= 0:
            return ""
        noun = "error" if new == 1 else "errors"
        if detail:
            return f"; {new} transport {noun} reported (last: {detail})"
        return f"; {new} transport {noun} reported"

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


_FEED_LOG_SINK = _FeedLogSink()
_FEED_LOG_INSTALLED = False
_FEED_LOG_LOCK = threading.Lock()


def _install_feed_log_sink() -> _FeedLogSink:
    """Route the feed library's logging into this module's sink, once.

    Installed lazily on first stream use rather than at import, so merely
    importing the broker leaves global logging state alone. Propagation is
    switched off permanently once installed, not restored per call, because a
    connect this module has abandoned keeps logging from the library's own
    daemon thread long after the call that started it returned.
    """
    global _FEED_LOG_INSTALLED
    with _FEED_LOG_LOCK:
        if not _FEED_LOG_INSTALLED:
            logger = logging.getLogger(_FEED_LOGGER_NAME)
            logger.addHandler(_FEED_LOG_SINK)
            logger.propagate = False
            _FEED_LOG_INSTALLED = True
    return _FEED_LOG_SINK


def _call_with_timeout[T](operation: Callable[[], T], timeout_seconds: float) -> T:
    """Run ``operation`` on a worker thread, giving up after ``timeout_seconds``.

    The feed library connects inside its constructor and drives that connect
    with its own retry schedule -- measured at sixty attempts roughly four
    seconds apart, so a connection to a server that accepts the socket but never
    completes the handshake blocks for about four and a half minutes. None of
    that is configurable through the public API, and a caller that asked for
    ninety seconds of ticks should not spend four minutes discovering it cannot
    have them.

    A timed-out worker is abandoned, not killed, because there is no way to
    interrupt a thread blocked in someone else's event loop. It is a daemon, so
    it cannot hold up interpreter exit, and the caller is answered on time. What
    it does cost is real and worth naming: the only operation passed here builds
    a ``GrowwFeed``, and a worker that connects after we stopped waiting leaves
    a live websocket, an event loop, a thread and a registry entry behind, owned
    by nobody. ``GrowwLtpStream.close`` exists to hand exactly that back, and a
    feed born this way is never handed to a stream, so it is never closed. The
    leak is bounded by how often connects time out, which is why the timeout is
    a ceiling on a rare path rather than a routine deadline. The other cost is
    the library's continued logging, which is why ``_install_feed_log_sink``
    stops that reaching stderr.
    """
    result: list[T] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(operation())
        except BaseException as error:  # noqa: BLE001 - relayed to the caller
            failure.append(error)

    worker = threading.Thread(target=run, name="groww-feed-connect", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(
            f"The operation did not finish within {timeout_seconds:g} seconds."
        )
    if failure:
        raise failure[0]
    return result[0]


def _validate_period(start: datetime, end: datetime, interval: CandleInterval) -> None:
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("The historical start time must be timezone-aware.")
    if end.tzinfo is None or end.utcoffset() is None:
        raise ValueError("The historical end time must be timezone-aware.")
    if start >= end:
        raise ValueError("The historical start time must be before the end time.")
    limit = MAX_HISTORICAL_SPAN.get(interval)
    if limit is not None and end - start > limit:
        # The span is printed whole rather than as ``.days``, which truncates:
        # a request six hours over the limit would otherwise be refused with
        # "at most 30 days; 30 days were requested", a sentence that refutes
        # itself and leaves the caller hunting for a bug that is really six
        # hours of overshoot.
        raise ValueError(
            f"Groww serves at most {limit.days} days of {interval.value} candles "
            f"per call; {end - start} was requested. Split the range."
        )


def _groww_datetime(value: datetime) -> str:
    return value.astimezone(INDIA_TIMEZONE).strftime(_DATETIME_FORMAT)


def _groww_epoch_datetime(value: int) -> datetime:
    """Normalize Groww epoch seconds or milliseconds to an IST datetime.

    IST because every timestamp this system holds is IST. An epoch is an instant
    and carries no zone, so the choice changes nothing about *when* this is -- it
    changes what ``.date()`` and ``.hour`` say about it downstream, and this is
    an Indian exchange, so the market's own zone is the one that makes the
    obvious reading of those correct.
    """
    if _MIN_REASONABLE_EPOCH_SECONDS <= value <= _MAX_REASONABLE_EPOCH_SECONDS:
        epoch_seconds = value
    elif (
        _MIN_REASONABLE_EPOCH_SECONDS * 1000
        <= value
        <= _MAX_REASONABLE_EPOCH_SECONDS * 1000
    ):
        epoch_seconds = value / 1000
    else:
        raise ValueError("Groww returned an invalid market timestamp.")

    return datetime.fromtimestamp(epoch_seconds, tz=INDIA_TIMEZONE)


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise TypeError
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError):
        # Defence in depth and nothing more. Every caller runs inside a
        # ``_reading`` block, whose own ``from None`` already suppresses this
        # whole chain, so no boundary can tell the two apart: removing this one
        # leaves the suite green, measured. That is a fact about where the
        # observable suppression lives, not a gap waiting for a test -- one
        # written here could only assert something nothing renders.
        raise TypeError from None
    # "NaN" and "Infinity" parse cleanly but are unusable as prices.
    if not price.is_finite():
        raise TypeError
    return price


def _normalize_candle(raw_candle: object) -> OHLCVCandle | None:
    """Normalize one raw row, or ``None`` for a row that is not a candle.

    ``None`` is returned only for the pre-open call auction. NSE collects orders
    from 09:00 and does not match them until roughly 09:08, so for those minutes
    Groww reports a real volume against four null prices -- 35 such rows per
    instrument per week, all before 09:10, never a partially null one. They are
    genuine market activity, but there is no traded price, and unlike a missing
    volume there is no "unknown" a downstream layer could carry: every consumer
    of a candle needs a price. Raising discards the whole fetch over an artifact
    that appears every single session; inventing one would put a number in the
    tape that the market never printed.

    Anything else still raises. A row missing *some* of its prices has not been
    observed and has no market-structure explanation, so it is a schema change
    or corruption and must surface rather than quietly shrink the tape.
    """
    if not isinstance(raw_candle, (list, tuple)) or len(raw_candle) < 6:
        raise TypeError

    raw_timestamp, raw_open, raw_high, raw_low, raw_close, raw_volume = raw_candle[:6]
    if raw_open is None and raw_high is None and raw_low is None and raw_close is None:
        return None
    if not isinstance(raw_timestamp, str):
        raise TypeError
    timestamp = datetime.fromisoformat(raw_timestamp)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=INDIA_TIMEZONE)
    timestamp = timestamp.astimezone(INDIA_TIMEZONE)

    # A null volume is data, not a fault: Groww leaves the field empty for the
    # occasional minute while still reporting a range that plainly moved. It is
    # carried through as ``None`` -- unknown -- because ``0`` would assert that
    # nobody traded, a measurement the vendor never made, and the feature engine
    # would then fold that assertion into ``volume_ratio_20`` as if it were one.
    if raw_volume is None:
        volume = None
    else:
        if isinstance(raw_volume, bool):
            raise TypeError
        volume = int(raw_volume)
        if volume < 0 or volume != raw_volume:
            raise TypeError

    open_price = _decimal(raw_open)
    high = _decimal(raw_high)
    low = _decimal(raw_low)
    close = _decimal(raw_close)
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        raise TypeError

    return OHLCVCandle(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


__all__ = [
    "GrowwAuthenticationError",
    "GrowwBroker",
    "GrowwBrokerError",
    "GrowwInstrument",
    "GrowwMarketDataError",
    "GrowwProfileError",
    "GrowwStreamConnectionError",
    "GrowwStreamError",
]
