"""Broker-neutral interfaces and data structures."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
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
class MarketTick:
    """A normalized streaming market-price update.

    ``cumulative_volume`` is the exchange's running volume for the current
    trading session, when the broker reports it. It is cumulative rather than
    per-tick, so interval volume must be derived by differencing snapshots.
    """

    instrument: Instrument
    price: Decimal
    timestamp: datetime
    cumulative_volume: int | None = None


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


MAX_HISTORICAL_SPAN = MappingProxyType(
    {CandleInterval.ONE_MINUTE: timedelta(days=7)},
)
"""The longest period one historical call may cover, per interval.

Groww's published figure: seven days for one-minute candles. It sits here
rather than in ``broker/groww.py`` because it has two users that must not
disagree -- the Groww client refuses a longer request, and the historical store
splits long ranges so it never makes one -- and the store must be able to read
it without importing a broker SDK.

That placement is a compromise and worth naming as one. A limit is a property
of a particular broker, so a second broker with different figures would make
this a per-broker attribute rather than a module constant. Until there is a
second broker, inventing the abstraction would be guessing at its shape; one
honest constant with this note is better than a speculative interface.

Groww separately publishes that only the **last three months** of one-minute
data exists at all. That is the harder constraint and it is not expressible
here, because it bounds how far back a caller may ask rather than how much one
call may carry; ``history/store.py`` is where a caller meets it.
"""


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
    "MAX_HISTORICAL_SPAN",
    "BrokerProfile",
    "CandleInterval",
    "Instrument",
    "LastTradedPrice",
    "MarketQuote",
    "MarketTick",
    "OHLCVCandle",
    "ReadOnlyBroker",
]
