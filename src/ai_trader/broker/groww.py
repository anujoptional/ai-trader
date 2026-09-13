"""Read-only Groww authentication and profile access."""

from collections.abc import Sequence
from contextlib import redirect_stdout
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from types import MappingProxyType
from typing import Any, Protocol, Self
from zoneinfo import ZoneInfo

import pyotp
from growwapi import GrowwAPI
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

_CASH_SEGMENT = "CASH"
_INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_CANDLE_INTERVALS = {CandleInterval.ONE_MINUTE: "1minute"}
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


class GrowwBroker:
    """A Groww adapter limited to read-only account and market data."""

    def __init__(self, client: _GrowwClient) -> None:
        self._client = client

    @classmethod
    def authenticate(cls, settings: GrowwSettings) -> Self:
        """Authenticate with TOTP and construct a read-only Groww adapter."""
        try:
            totp = pyotp.TOTP(settings.totp_secret.get_secret_value()).now()
            access_token = GrowwAPI.get_access_token(
                api_key=settings.totp_token.get_secret_value(),
                totp=totp,
            )
            if not isinstance(access_token, str) or not access_token:
                raise TypeError

            # The SDK prints status text during construction. Suppress it so the
            # CLI emits only its explicitly allowlisted profile summary.
            with redirect_stdout(StringIO()):
                client = GrowwAPI(access_token)
        except Exception:
            raise GrowwAuthenticationError("Groww authentication failed.") from None

        return cls(client)

    def get_user_profile(self) -> BrokerProfile:
        """Retrieve and sanitize the Groww user profile."""
        try:
            payload = _GrowwProfilePayload.model_validate(
                self._client.get_user_profile()
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
            response = self._client.get_ltp(
                exchange_trading_symbols=groww_symbols,
                segment=_CASH_SEGMENT,
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
                self._client.get_quote(
                    trading_symbol=instrument.trading_symbol,
                    exchange=instrument.exchange,
                    segment=_CASH_SEGMENT,
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
            response = self._client.get_historical_candles(
                exchange=instrument.exchange,
                segment=_CASH_SEGMENT,
                groww_symbol=_historical_symbol(instrument),
                start_time=_groww_datetime(start),
                end_time=_groww_datetime(end),
                candle_interval=_CANDLE_INTERVALS[interval],
            )
            raw_candles = response["candles"]
            if not isinstance(raw_candles, list):
                raise TypeError
            return tuple(_normalize_candle(candle) for candle in raw_candles)
        except Exception:
            raise GrowwMarketDataError(
                "Groww historical data retrieval failed."
            ) from None


def _live_symbol(instrument: Instrument) -> str:
    return f"{instrument.exchange}_{instrument.trading_symbol}"


def _historical_symbol(instrument: Instrument) -> str:
    return f"{instrument.exchange}-{instrument.trading_symbol}"


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
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise TypeError from None


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

    return OHLCVCandle(
        timestamp=timestamp,
        open=_decimal(raw_open),
        high=_decimal(raw_high),
        low=_decimal(raw_low),
        close=_decimal(raw_close),
        volume=volume,
    )


__all__ = [
    "GrowwAuthenticationError",
    "GrowwBroker",
    "GrowwBrokerError",
    "GrowwMarketDataError",
    "GrowwProfileError",
]
