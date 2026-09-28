"""Broker-neutral interfaces and data structures."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from ai_trader.clock import INDIA_TIMEZONE


def _in_india_time(value: datetime, field_name: str) -> datetime:
    """The same instant, carried in IST, or a refusal if it names no instant.

    This is the boundary. A vendor sends whatever zone it likes -- Groww serves
    epoch seconds, a websocket frame may carry an offset of its own -- and
    everything inside this package then asks *local* questions of the result:
    which trading session a bar belongs to, how many minutes into the day it
    is, what date a report should print. Those questions have different answers
    in different zones for the same instant, so normalizing at the door rather
    than in each downstream type makes the guarantee structural: no aware
    datetime inside ``ai_trader`` carries a zone but this one.

    Converting changes nothing about *when* the moment is. An aware datetime
    names an instant, and the two spellings of one instant compare and hash
    equal; what changes is what ``.date()``, ``.hour`` and an ISO rendering say
    about it, which is exactly the set of questions this project asks.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(INDIA_TIMEZONE)


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp", _in_india_time(self.timestamp, "Tick timestamp")
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "last_trade_at",
            _in_india_time(self.last_trade_at, "Quote last_trade_at"),
        )


@dataclass(frozen=True, slots=True)
class OHLCVCandle:
    """An OHLCV candle, timestamped in IST.

    ``timestamp`` names the minute the bar opened, and it is carried in IST
    because the whole project reads local questions off it -- which session,
    how far into it, what date to print.

    ``volume`` is optional because the vendor genuinely omits it: a real NSE
    session returned one minute in 362 with a null volume and a price range that
    plainly moved, so the bar is not empty and reporting it as zero would state
    a fact nobody measured. ``None`` means unknown, and every layer downstream
    already reads it that way -- ``market.Candle`` accepts it, the feature
    engine withholds ``volume_ratio_20`` rather than averaging around it, and
    the candle store round-trips it as an empty field.
    """

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp", _in_india_time(self.timestamp, "Candle timestamp")
        )


class CandleInterval(StrEnum):
    """Broker-independent candle intervals supported by the project."""

    ONE_MINUTE = "1m"


MAX_HISTORICAL_SPAN = MappingProxyType(
    {CandleInterval.ONE_MINUTE: timedelta(days=30)},
)
"""The longest period one historical call may cover, per interval.

Groww's published figure for ``GET /v1/historical/candles``, the endpoint
``GrowwBroker.get_historical_candles`` actually calls: thirty days for
one-minute candles. The number is endpoint-specific and was wrong here for a
while -- the older ``/v1/historical/candle/range`` allowed seven, and this
constant still said seven after the call site had moved -- which is the reason
the endpoint is named rather than just the vendor. Groww marks that older
endpoint deprecated, and the SDK emits a deprecation warning when it is used.

It sits here rather than in ``broker/groww.py`` because it has two users that
must not disagree -- the Groww client refuses a longer request, and the
historical store splits long ranges so it never makes one -- and the store must
be able to read it without importing a broker SDK.

That placement is a compromise and worth naming as one. A limit is a property
of a particular broker, so a second broker with different figures would make
this a per-broker attribute rather than a module constant. Until there is a
second broker, inventing the abstraction would be guessing at its shape; one
honest constant with this note is better than a speculative interface.

Groww separately publishes how far back the data goes at all: the backtesting
endpoint serves **from 2020**, where its deprecated predecessor served only the
last three months. That is a different kind of constraint and is not
expressible here, because it bounds how far back a caller may ask rather than
how much one call may carry; ``history/store.py`` is where a caller meets it.
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
