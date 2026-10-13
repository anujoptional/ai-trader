"""Read-only Kite profile, quotes, history and ticks; no order capabilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from threading import Lock
from time import monotonic, sleep
from typing import Any, Protocol, Self

from ai_trader.broker import (
    MAX_HISTORICAL_SPAN,
    BrokerProfile,
    CandleInterval,
    DailyClose,
    DepthLevel,
    DepthSnapshot,
    FuturesDay,
    Instrument,
    LastTradedPrice,
    MarketQuote,
    OHLCVCandle,
)
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import KiteLoginSettings, KiteSettings

_INTERVALS = {CandleInterval.ONE_MINUTE: "minute", CandleInterval.ONE_DAY: "day"}
_REQUEST_BOUNDS = MAX_HISTORICAL_SPAN

_INDEX_SYMBOLS = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "NIFTYIT": "NIFTY IT",
    "NIFTYMIDCAP150": "NIFTY MIDCAP 150",
    "NIFTY500": "NIFTY 500",
    "INDIAVIX": "INDIA VIX",
    "NIFTYAUTO": "NIFTY AUTO",
    "NIFTYFMCG": "NIFTY FMCG",
    "NIFTYPHARMA": "NIFTY PHARMA",
    "NIFTYMETAL": "NIFTY METAL",
    "NIFTYREALTY": "NIFTY REALTY",
}


def kite_symbol(instrument: Instrument) -> str:
    return _INDEX_SYMBOLS.get(instrument.trading_symbol, instrument.trading_symbol)


class HistoryRateLimiter:
    """One request-start budget shared across all workers using the same API key."""

    def __init__(self, spacing_seconds: float = 0.4) -> None:
        self._lock = Lock()
        self._last_request: float | None = None
        self._spacing = spacing_seconds

    def acquire(self) -> None:
        with self._lock:
            if self._last_request is not None:
                delay = self._spacing - (monotonic() - self._last_request)
                if delay > 0:
                    sleep(delay)
            self._last_request = monotonic()


class KiteBrokerError(RuntimeError):
    """A sanitized failure that contains no vendor response or credentials."""


class KiteAuthenticationError(KiteBrokerError):
    """The Kite session could not be created or has expired."""


class KiteStreamError(KiteBrokerError):
    """Non-retryable invalid stream data."""


class KiteStreamConnectionError(KiteStreamError):
    """Retryable stream transport failure."""


class _KiteHistoricalClient(Protocol):
    def profile(self) -> dict[str, Any]: ...

    def ltp(self, instruments: list[str]) -> dict[str, Any]: ...

    def quote(self, instruments: list[str]) -> dict[str, Any]: ...

    def instruments(self, exchange: str) -> list[dict[str, Any]]: ...

    def historical_data(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class KiteInstrument:
    instrument: Instrument
    instrument_token: int
    exchange_token: str
    tick_size: Decimal
    instrument_type: str = "EQ"
    trading_symbol: str = ""


@dataclass(frozen=True, slots=True)
class HistoricalCapture:
    candles: tuple[OHLCVCandle, ...]
    rejected: tuple[dict[str, Any], ...]


def _request[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except Exception as error:
        if type(error).__name__ == "TokenException":
            raise KiteAuthenticationError(
                "Kite session invalid or expired; sign in again."
            ) from None
        code = getattr(error, "code", None)
        if code == 429:
            raise KiteBrokerError("Kite rate limit reached; retry later.") from None
        if code == 403:
            raise KiteBrokerError(
                "Kite access denied; check the session and Connect data subscription."
            ) from None
        raise KiteBrokerError("Kite historical-data request failed.") from None


def _price(value: object) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise ValueError
    return result


def _history_client(api_key: str, access_token: str) -> Any:
    """A pinned-SDK client whose candle dates stay ISO strings for the adapter.

    The SDK parses every candle date with dateutil, about 60x slower than the
    adapter's own ``datetime.fromisoformat``; that made large downloads CPU-bound.
    """
    from kiteconnect import KiteConnect

    class RawDateKiteConnect(KiteConnect):
        def _format_historical(self, data: dict) -> list[dict[str, Any]]:
            fields = ("date", "open", "high", "low", "close", "volume", "oi")
            return [
                dict(zip(fields[: len(row)], row, strict=True))
                for row in data["candles"]
            ]

    return RawDateKiteConnect(
        api_key=api_key, access_token=access_token, timeout=30, debug=False
    )


def _candle(row: dict, interval: CandleInterval) -> OHLCVCandle:
    stamp = row["date"]
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    if (
        not isinstance(stamp, datetime)
        or stamp.tzinfo is None
        or stamp.utcoffset() is None
    ):
        raise ValueError
    stamp = stamp.astimezone(INDIA_TIMEZONE)
    if interval is CandleInterval.ONE_MINUTE and (stamp.second or stamp.microsecond):
        raise ValueError
    opening, high, low, close = (
        _price(row[name]) for name in ("open", "high", "low", "close")
    )
    if not low <= min(opening, close) <= max(opening, close) <= high:
        raise ValueError
    volume = row["volume"]
    if volume is not None and (type(volume) is not int or volume < 0):
        raise ValueError
    return OHLCVCandle(stamp, opening, high, low, close, volume)


class KiteHistoricalBroker:
    """Kite's history capability using the project's broker-neutral candle schema."""

    def __init__(
        self,
        client: _KiteHistoricalClient,
        *,
        limiter: HistoryRateLimiter | None = None,
    ) -> None:
        self._client = client
        self._instruments: dict[str, list[dict[str, Any]]] = {}
        self._history_limiter = limiter or HistoryRateLimiter()
        self._resolved: dict[Instrument, KiteInstrument] = {}

    @classmethod
    def from_session(cls, settings: KiteSettings) -> Self:
        try:
            import kiteconnect  # noqa: F401
        except ImportError:
            raise KiteBrokerError(
                "Install the project's declared Kite Connect dependency."
            ) from None
        client = _request(
            lambda: _history_client(
                settings.api_key.get_secret_value(),
                settings.access_token.get_secret_value(),
            )
        )
        return cls(client)

    def resolve_instrument(self, instrument: Instrument) -> KiteInstrument:
        if instrument in self._resolved:
            return self._resolved[instrument]
        if instrument.exchange not in ("NSE", "BSE") or not instrument.trading_symbol:
            raise ValueError(
                "Kite history currently supports NSE/BSE cash instruments only."
            )
        exchange = instrument.exchange
        if exchange not in self._instruments:
            self._instruments[exchange] = _request(
                lambda: self._client.instruments(exchange)
            )
        try:
            rows = [
                row
                for row in self._instruments[exchange]
                if row["exchange"] == exchange
                and row["tradingsymbol"] == kite_symbol(instrument)
                and row["segment"] in (exchange, "INDICES")
                and row["instrument_type"] in ("EQ", "INDEX")
            ]
            if len(rows) != 1:
                raise ValueError
            row = rows[0]
            token = row["instrument_token"]
            if type(token) is not int or token <= 0:
                raise ValueError
            tick_size = Decimal(str(row["tick_size"]))
            if not tick_size.is_finite() or tick_size < 0:
                raise ValueError
            if row["segment"] != "INDICES" and tick_size == 0:
                raise ValueError
            resolved = KiteInstrument(
                instrument,
                token,
                str(row["exchange_token"]),
                tick_size,
                "INDEX" if row["segment"] == "INDICES" else "EQ",
                row["tradingsymbol"],
            )
            self._resolved[instrument] = resolved
            return resolved
        except Exception:
            raise KiteBrokerError(
                "Kite equity instrument lookup was missing or ambiguous."
            ) from None

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]:
        capture = self.capture_historical_candles(instrument, start, end, interval)
        if capture.rejected:
            raise KiteBrokerError(
                "Kite historical candles failed schema or timestamp validation."
            )
        return capture.candles

    def capture_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> HistoricalCapture:
        _validate_period(start, end, interval)
        beginning, ending = (
            start.astimezone(INDIA_TIMEZONE),
            end.astimezone(INDIA_TIMEZONE),
        )
        resolved = self.resolve_instrument(instrument)
        self._history_limiter.acquire()
        rows = _request(
            lambda: self._client.historical_data(
                instrument_token=resolved.instrument_token,
                from_date=beginning.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=ending.strftime("%Y-%m-%d %H:%M:%S"),
                interval=_INTERVALS[interval],
                continuous=False,
                oi=False,
            )
        )
        try:
            if not isinstance(rows, list):
                raise ValueError
            candles, rejected = [], []
            previous: datetime | None = None
            for row in rows:
                stamp = row["date"]
                if isinstance(stamp, str):
                    stamp = datetime.fromisoformat(stamp)
                if (
                    not isinstance(stamp, datetime)
                    or stamp.tzinfo is None
                    or stamp.utcoffset() is None
                ):
                    raise ValueError
                stamp = stamp.astimezone(INDIA_TIMEZONE)
                if not beginning <= stamp <= ending:
                    raise ValueError
                try:
                    candle = _candle(row, interval)
                    if previous is not None and stamp <= previous:
                        raise ValueError
                    candles.append(candle)
                except Exception:
                    rejected.append(
                        {
                            "timestamp": stamp.isoformat(),
                            "reason": "invalid_ohlcv_or_ordering",
                            "values": {
                                name: str(row[name])
                                if type(row.get(name)) in (int, float, Decimal)
                                else None
                                for name in ("open", "high", "low", "close", "volume")
                            },
                        }
                    )
                previous = stamp
            return HistoricalCapture(tuple(candles), tuple(rejected))
        except Exception:
            raise KiteBrokerError(
                "Kite historical candles failed schema or timestamp validation."
            ) from None

    def get_daily_closes(
        self, instrument: Instrument, start: datetime, end: datetime
    ) -> tuple[DailyClose, ...]:
        if any(
            moment.tzinfo is None or moment.utcoffset() is None
            for moment in (start, end)
        ):
            raise ValueError("Kite historical bounds must be timezone-aware.")
        if start >= end:
            raise ValueError("Unordered daily history bounds.")
        beginning = datetime.combine(
            start.astimezone(INDIA_TIMEZONE).date(), time.min, INDIA_TIMEZONE
        )
        ending = datetime.combine(
            end.astimezone(INDIA_TIMEZONE).date(), time(23, 59), INDIA_TIMEZONE
        )
        candles = self.get_historical_candles(
            instrument, beginning, ending, CandleInterval.ONE_DAY
        )
        return tuple(
            DailyClose(candle.timestamp.date(), candle.close) for candle in candles
        )


def _validate_period(start: datetime, end: datetime, interval: CandleInterval) -> None:
    if any(
        moment.tzinfo is None or moment.utcoffset() is None for moment in (start, end)
    ):
        raise ValueError("Kite historical bounds must be timezone-aware.")
    if interval not in _INTERVALS or not start < end:
        raise ValueError("Unsupported interval or unordered history bounds.")
    if end - start > _REQUEST_BOUNDS[interval]:
        raise ValueError(
            f"Kite conservative window is {_REQUEST_BOUNDS[interval].days} days; "
            f"{end - start} was requested. Split the range."
        )


def _quote_timestamp(value: object) -> datetime:
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(stamp, datetime):
        raise ValueError
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=INDIA_TIMEZONE)
    return stamp.astimezone(INDIA_TIMEZONE)


class KiteBroker(KiteHistoricalBroker):
    """Read-only Zerodha market access; no order-placement capability."""

    def __init__(
        self,
        client: _KiteHistoricalClient,
        *,
        limiter: HistoryRateLimiter | None = None,
        quote_limiter: HistoryRateLimiter | None = None,
    ) -> None:
        super().__init__(client, limiter=limiter)
        # Kite allows one quote/LTP request per second per key, across all clients.
        self._quote_limiter = quote_limiter or HistoryRateLimiter(1.05)

    def fork_history(self) -> KiteBroker:
        client = _request(
            lambda: _history_client(self._client.api_key, self._client.access_token)
        )
        worker = KiteBroker(
            client, limiter=self._history_limiter, quote_limiter=self._quote_limiter
        )
        worker._instruments = dict(self._instruments)
        worker._resolved = dict(self._resolved)
        return worker

    @classmethod
    def authenticate(cls, settings: KiteLoginSettings | KiteSettings) -> KiteBroker:
        if isinstance(settings, KiteSettings):
            return cls.from_session(settings)
        from ai_trader.broker.kite_auth import login_browser

        return login_browser(settings)

    def get_user_profile(self) -> BrokerProfile:
        payload = _request(self._client.profile)
        try:
            exchanges = payload["exchanges"]
            if not isinstance(exchanges, list) or any(
                not isinstance(exchange, str) or not exchange for exchange in exchanges
            ):
                raise ValueError
            enabled = set(exchanges)
            segments = tuple(
                name
                for name, codes in (
                    ("CASH", {"NSE", "BSE"}),
                    ("FNO", {"NFO", "BFO"}),
                    ("CURRENCY", {"CDS", "BCD"}),
                    ("COMMODITY", {"MCX"}),
                )
                if enabled & codes
            )
            return BrokerProfile(
                exchange_enablement={exchange: True for exchange in sorted(enabled)},
                active_segments=segments,
                ddpi_enabled=None,
            )
        except Exception:
            raise KiteBrokerError("Kite profile failed schema validation.") from None

    def create_ltp_stream(
        self,
        instruments: Sequence[Instrument],
        *,
        connect_timeout_seconds: float = 30,
        strict: bool = True,
        queue_size: int = 10000,
    ):
        import logging

        from kiteconnect import KiteTicker
        from twisted.internet import reactor

        from ai_trader.broker.kite_stream import KiteLtpStream

        if not instruments or connect_timeout_seconds <= 0:
            raise ValueError("Instruments and a positive stream timeout are required.")
        resolved = tuple(self.resolve_instrument(item) for item in instruments)
        logger = logging.getLogger("kiteconnect.ticker")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        ticker = _request(
            lambda: KiteTicker(
                self._client.api_key,
                self._client.access_token,
                debug=False,
                reconnect=False,
                connect_timeout=connect_timeout_seconds,
            )
        )
        return KiteLtpStream(
            ticker,
            resolved,
            dispatch=reactor.callFromThread,
            strict=strict,
            queue_size=queue_size,
        ).connect(connect_timeout_seconds)

    def get_ltp(self, instruments: Sequence[Instrument]) -> tuple[LastTradedPrice, ...]:
        if not instruments:
            return ()
        if len(instruments) > 1000 or len(set(instruments)) != len(instruments):
            raise ValueError("Request at most 1000 unique instruments.")
        keys = [f"{item.exchange}:{kite_symbol(item)}" for item in instruments]
        self._quote_limiter.acquire()
        payload = _request(lambda: self._client.ltp(keys))
        try:
            return tuple(
                LastTradedPrice(instrument, _price(payload[key]["last_price"]))
                for instrument, key in zip(instruments, keys, strict=True)
            )
        except Exception:
            raise KiteBrokerError("Kite LTP response is missing or invalid.") from None

    def get_quote(self, instrument: Instrument) -> MarketQuote:
        key = f"{instrument.exchange}:{kite_symbol(instrument)}"
        payload = self._paced_quote([key])
        try:
            row = payload[key]
            opening, high, low, previous = (
                _price(row["ohlc"][field]) for field in ("open", "high", "low", "close")
            )
            last = _price(row["last_price"])
            volume = row["volume"]
            if type(volume) is not int or volume < 0 or not low <= opening <= high:
                raise ValueError
            change = last - previous
            return MarketQuote(
                instrument=instrument,
                last_price=last,
                last_trade_at=_quote_timestamp(row["last_trade_time"]),
                open=opening,
                high=high,
                low=low,
                previous_close=previous,
                volume=volume,
                day_change=change,
                day_change_percent=change / previous * 100,
            )
        except Exception:
            raise KiteBrokerError(
                "Kite quote response is missing or invalid."
            ) from None

    def _paced_quote(self, keys: list[str]) -> dict[str, Any]:
        """Kite's quote endpoint allows one request per second per key."""
        self._quote_limiter.acquire()
        return _request(lambda: self._client.quote(keys))

    def get_depth(
        self, instruments: Sequence[Instrument]
    ) -> tuple[tuple[DepthSnapshot, ...], tuple[Instrument, ...]]:
        """Five-level depth for up to 500 instruments: (snapshots, missing/invalid)."""
        if not instruments:
            return (), ()
        if len(instruments) > 500 or len(set(instruments)) != len(instruments):
            raise ValueError("Request at most 500 unique instruments.")
        keys = [f"{item.exchange}:{kite_symbol(item)}" for item in instruments]
        payload = self._paced_quote(keys)
        snapshots, missing = [], []
        for instrument, key in zip(instruments, keys, strict=True):
            try:
                snapshots.append(_depth_snapshot(instrument, payload[key]))
            except Exception:
                missing.append(instrument)
        return tuple(snapshots), tuple(missing)

    def nfo_futures(self) -> list[dict[str, Any]]:
        """Live NFO futures contracts from Kite's instrument dump (cached)."""
        if "NFO" not in self._instruments:
            self._instruments["NFO"] = _request(lambda: self._client.instruments("NFO"))
        return [row for row in self._instruments["NFO"] if row["segment"] == "NFO-FUT"]

    def get_continuous_futures_daily(
        self, underlying: str, start: date, end: date, *, window_days: int = 1000
    ) -> tuple[FuturesDay, ...]:
        """Kite's continuous near-month futures day bars with end-of-day OI."""
        contracts = sorted(
            (row for row in self.nfo_futures() if row["name"] == underlying),
            key=lambda row: row["expiry"],
        )
        if not contracts:
            raise KiteBrokerError(f"No live NFO future for {underlying}.")
        token = contracts[0]["instrument_token"]
        bars: dict[date, FuturesDay] = {}
        cursor = start
        while cursor <= end:
            stop = min(end, cursor + timedelta(days=window_days - 1))
            self._history_limiter.acquire()
            rows = _request(
                lambda c=cursor, s=stop: self._client.historical_data(
                    instrument_token=token,
                    from_date=c.strftime("%Y-%m-%d 00:00:00"),
                    to_date=s.strftime("%Y-%m-%d 23:59:59"),
                    interval="day",
                    continuous=True,
                    oi=True,
                )
            )
            for row in rows:
                bar = _futures_day(row)
                bars[bar.session_date] = bar
            cursor = stop + timedelta(days=1)
        return tuple(bars[day] for day in sorted(bars))


def _optional_price(value: object) -> Decimal | None:
    try:
        return _price(value)
    except Exception:
        return None


def _depth_snapshot(instrument: Instrument, row: dict[str, Any]) -> DepthSnapshot:
    def levels(side: str) -> tuple[DepthLevel, ...]:
        result = []
        for level in row.get("depth", {}).get(side, []):
            quantity, orders = level["quantity"], level["orders"]
            if type(quantity) is not int or type(orders) is not int:
                raise ValueError
            if quantity > 0:
                result.append(DepthLevel(_price(level["price"]), quantity, orders))
        return tuple(result)

    totals = [row["volume"], row["buy_quantity"], row["sell_quantity"]]
    if any(type(value) is not int or value < 0 for value in totals):
        raise ValueError
    oi = row.get("oi")
    trade_time = row.get("last_trade_time")
    return DepthSnapshot(
        instrument=instrument,
        timestamp=_quote_timestamp(row["timestamp"]),
        last_price=_price(row["last_price"]),
        last_trade_at=_quote_timestamp(trade_time) if trade_time else None,
        volume=totals[0],
        total_buy_quantity=totals[1],
        total_sell_quantity=totals[2],
        bids=levels("buy"),
        asks=levels("sell"),
        previous_close=_price(row["ohlc"]["close"]),
        lower_circuit=_optional_price(row.get("lower_circuit_limit")),
        upper_circuit=_optional_price(row.get("upper_circuit_limit")),
        open_interest=int(oi) if isinstance(oi, int | float) and oi > 0 else None,
    )


def _futures_day(row: dict[str, Any]) -> FuturesDay:
    stamp = row["date"]
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    opening, high, low, close = (
        _price(row[name]) for name in ("open", "high", "low", "close")
    )
    volume, oi = row["volume"], row.get("oi")
    if type(volume) is not int or volume < 0 or type(oi) is not int or oi < 0:
        raise KiteBrokerError("Kite futures bar failed schema validation.")
    if not low <= min(opening, close) <= max(opening, close) <= high:
        raise KiteBrokerError("Kite futures bar has an inconsistent range.")
    return FuturesDay(
        stamp.astimezone(INDIA_TIMEZONE).date() if stamp.tzinfo else stamp.date(),
        opening,
        high,
        low,
        close,
        volume,
        oi,
    )
