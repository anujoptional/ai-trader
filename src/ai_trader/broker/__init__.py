"""Broker-neutral interfaces and data structures."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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


@dataclass(frozen=True, slots=True)
class DailyClose:
    """A positive vendor daily close for an IST session date, not a minute bar."""

    session_date: date
    close: Decimal

    def __post_init__(self) -> None:
        if type(self.session_date) is not date:
            raise ValueError("Daily close requires a session date.")
        if not self.close.is_finite() or self.close <= 0:
            raise ValueError("Daily close must be finite and positive.")


class CandleInterval(StrEnum):
    """Broker-independent candle intervals supported by the project."""

    ONE_MINUTE = "1m"
    ONE_DAY = "1d"


@dataclass(frozen=True, slots=True)
class DepthLevel:
    """One price level of the visible order book."""

    price: Decimal
    quantity: int
    orders: int


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    """A five-level order-book quote with session totals, timestamped in IST.

    ``total_buy_quantity``/``total_sell_quantity`` are the exchange's pending
    quantities across the whole book, not only the five visible levels.
    """

    instrument: Instrument
    timestamp: datetime
    last_price: Decimal
    last_trade_at: datetime | None
    volume: int
    total_buy_quantity: int
    total_sell_quantity: int
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    previous_close: Decimal
    lower_circuit: Decimal | None = None
    upper_circuit: Decimal | None = None
    open_interest: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp", _in_india_time(self.timestamp, "Depth timestamp")
        )


@dataclass(frozen=True, slots=True)
class FuturesDay:
    """A daily near-month futures bar with end-of-day open interest."""

    session_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    open_interest: int


MAX_HISTORICAL_SPAN = MappingProxyType(
    {
        CandleInterval.ONE_MINUTE: timedelta(days=60),
        CandleInterval.ONE_DAY: timedelta(days=400),
    },
)
"""Per-request windows shared by the Kite adapter and cache.

The minute bound is Kite's documented 60-day maximum; the daily bound stays well
under Kite's documented 2000 days. These are not claims about the maximum history
the vendor supports. The staged downloader uses smaller, checkpointed pages and one
shared rate limiter. Missing dates and revisions are recorded, never inferred away.
"""


@dataclass(frozen=True)
class BrokerProfile:
    """Non-sensitive broker capabilities safe to display."""

    exchange_enablement: Mapping[str, bool]
    active_segments: tuple[str, ...]
    ddpi_enabled: bool | None


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

    def get_daily_closes(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> tuple[DailyClose, ...]:
        """Return vendor daily closes independently of missing daily open prices."""
        ...


__all__ = [
    "MAX_HISTORICAL_SPAN",
    "BrokerProfile",
    "CandleInterval",
    "DailyClose",
    "DepthLevel",
    "DepthSnapshot",
    "FuturesDay",
    "Instrument",
    "LastTradedPrice",
    "MarketQuote",
    "MarketTick",
    "OHLCVCandle",
    "ReadOnlyBroker",
]
