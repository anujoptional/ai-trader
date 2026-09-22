"""Read-only Groww authentication and profile access."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, Self
from zoneinfo import ZoneInfo

import pyotp
from growwapi import GrowwAPI, GrowwFeed
from pydantic import BaseModel, ConfigDict

from ai_trader.broker import (
    BrokerProfile,
    CandleInterval,
    Instrument,
    LastTradedPrice,
    MarketQuote,
    OHLCVCandle,
)
from ai_trader.config import GrowwSettings

if TYPE_CHECKING:
    from ai_trader.broker.groww_stream import GrowwLtpStream

_CASH_SEGMENT = "CASH"
_INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_CANDLE_INTERVALS = {CandleInterval.ONE_MINUTE: "1minute"}
_CALL_ATTEMPTS = 8
_RETRY_BASE_DELAY_SECONDS = 0.5
_MAX_RETRY_DELAY_SECONDS = 2.0
_STREAM_CONNECT_TIMEOUT_SECONDS = 30.0
_FEED_LOGGER_NAME = "growwapi"
_FEED_LOG_HISTORY = 64
_MIN_REASONABLE_EPOCH_SECONDS = int(datetime(2000, 1, 1, tzinfo=UTC).timestamp())
_MAX_REASONABLE_EPOCH_SECONDS = int(datetime(2100, 1, 1, tzinfo=UTC).timestamp())


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
        try:
            # The TOTP is generated per attempt so a retry that crosses a
            # 30-second window uses the code belonging to the window it lands in.
            access_token = _retry_broker_call(
                lambda: GrowwAPI.get_access_token(
                    api_key=settings.totp_token.get_secret_value(),
                    totp=pyotp.TOTP(settings.totp_secret.get_secret_value()).now(),
                )
            )
            if not isinstance(access_token, str) or not access_token:
                raise TypeError

            # The SDK prints status text during construction. Suppress it so the
            # CLI emits only its explicitly allowlisted profile summary.
            with redirect_stdout(StringIO()):
                client = _retry_broker_call(lambda: GrowwAPI(access_token))
        except Exception:
            raise GrowwAuthenticationError("Groww authentication failed.") from None

        return cls(client)

    def get_user_profile(self) -> BrokerProfile:
        """Retrieve and sanitize the Groww user profile."""
        try:
            payload = _GrowwProfilePayload.model_validate(
                _retry_broker_call(self._client.get_user_profile)
            )
        except Exception:
            raise GrowwProfileError("Groww profile retrieval failed.") from None

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
        try:
            response = _retry_broker_call(
                lambda: self._client.get_ltp(
                    exchange_trading_symbols=groww_symbols,
                    segment=_CASH_SEGMENT,
                )
            )
            return tuple(
                LastTradedPrice(
                    instrument=instrument,
                    price=_decimal(response[groww_symbol]),
                )
                for instrument, groww_symbol in zip(
                    instruments, groww_symbols, strict=True
                )
            )
        except Exception:
            raise GrowwMarketDataError("Groww LTP retrieval failed.") from None

    def get_quote(self, instrument: Instrument) -> MarketQuote:
        """Retrieve and normalize a detailed CASH quote."""
        try:
            payload = _GrowwQuotePayload.model_validate(
                _retry_broker_call(
                    lambda: self._client.get_quote(
                        trading_symbol=instrument.trading_symbol,
                        exchange=instrument.exchange,
                        segment=_CASH_SEGMENT,
                    )
                )
            )
            last_trade_at = _groww_epoch_datetime(payload.last_trade_time)
        except Exception:
            raise GrowwMarketDataError("Groww quote retrieval failed.") from None

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
        """Retrieve and normalize historical CASH candles."""
        _validate_period(start, end)
        try:
            response = _retry_broker_call(
                lambda: self._client.get_historical_candles(
                    exchange=instrument.exchange,
                    segment=_CASH_SEGMENT,
                    groww_symbol=_historical_symbol(instrument),
                    start_time=_groww_datetime(start),
                    end_time=_groww_datetime(end),
                    candle_interval=_CANDLE_INTERVALS[interval],
                )
            )
            raw_candles = response["candles"]
            if not isinstance(raw_candles, list):
                raise TypeError
            return tuple(_normalize_candle(candle) for candle in raw_candles)
        except Exception:
            raise GrowwMarketDataError(
                "Groww historical data retrieval failed."
            ) from None

    def resolve_instrument(self, groww_symbol: str) -> GrowwInstrument:
        """Resolve a Groww symbol to normalized metadata and a streaming token."""
        try:
            payload = _GrowwInstrumentPayload.model_validate(
                _retry_broker_call(
                    lambda: self._client.get_instrument_by_groww_symbol(groww_symbol)
                )
            )
            if payload.groww_symbol != groww_symbol or payload.segment != _CASH_SEGMENT:
                raise ValueError
            if not payload.exchange_token.strip():
                raise ValueError
        except Exception:
            raise GrowwMarketDataError("Groww instrument lookup failed.") from None

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
    genuine schema change surfaces immediately instead of being retried.
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


def _validate_period(start: datetime, end: datetime) -> None:
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("The historical start time must be timezone-aware.")
    if end.tzinfo is None or end.utcoffset() is None:
        raise ValueError("The historical end time must be timezone-aware.")
    if start >= end:
        raise ValueError("The historical start time must be before the end time.")


def _groww_datetime(value: datetime) -> str:
    return value.astimezone(_INDIA_TIMEZONE).strftime(_DATETIME_FORMAT)


def _groww_epoch_datetime(value: int) -> datetime:
    """Normalize Groww epoch seconds or milliseconds to a UTC datetime."""
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

    return datetime.fromtimestamp(epoch_seconds, tz=UTC)


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise TypeError
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise TypeError from None
    # "NaN" and "Infinity" parse cleanly but are unusable as prices.
    if not price.is_finite():
        raise TypeError
    return price


def _normalize_candle(raw_candle: object) -> OHLCVCandle:
    if not isinstance(raw_candle, (list, tuple)) or len(raw_candle) < 6:
        raise TypeError

    raw_timestamp, raw_open, raw_high, raw_low, raw_close, raw_volume = raw_candle[:6]
    if not isinstance(raw_timestamp, str):
        raise TypeError
    timestamp = datetime.fromisoformat(raw_timestamp)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=_INDIA_TIMEZONE)
    timestamp = timestamp.astimezone(UTC)

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
