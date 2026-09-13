"""Broker-neutral interfaces and data structures."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Instrument:
    """A broker-independent exchange and trading-symbol pair."""

    exchange: str
    trading_symbol: str


@dataclass(frozen=True, slots=True)
class LastTradedPrice:
    """The latest available price for an instrument."""

    instrument: Instrument
    price: Decimal


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """A normalized detailed market quote."""

    instrument: Instrument
    last_price: Decimal
    last_trade_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    previous_close: Decimal
    volume: int
    day_change: Decimal
    day_change_percent: Decimal


@dataclass(frozen=True, slots=True)
class OHLCVCandle:
    """A normalized, timezone-aware OHLCV candle."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class CandleInterval(StrEnum):
    """Broker-independent candle intervals supported by the project."""

    ONE_MINUTE = "1m"


@dataclass(frozen=True)
class BrokerProfile:
    """Non-sensitive broker capabilities safe to display."""

    exchange_enablement: Mapping[str, bool]
    active_segments: tuple[str, ...]
    ddpi_enabled: bool


class ReadOnlyBroker(Protocol):
    """Read-only broker capabilities used by the project."""

    def get_user_profile(self) -> BrokerProfile:
        """Return a sanitized, non-sensitive broker profile."""
        ...

    def get_ltp(
        self,
        instruments: Sequence[Instrument],
    ) -> tuple[LastTradedPrice, ...]:
        """Return the latest price for each requested CASH instrument."""
        ...

    def get_quote(self, instrument: Instrument) -> MarketQuote:
        """Return a detailed quote for one CASH instrument."""
        ...

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]:
        """Return normalized historical CASH candles."""
        ...


__all__ = [
    "BrokerProfile",
    "CandleInterval",
    "Instrument",
    "LastTradedPrice",
    "MarketQuote",
    "OHLCVCandle",
    "ReadOnlyBroker",
]
