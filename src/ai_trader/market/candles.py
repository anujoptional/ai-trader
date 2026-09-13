"""Real-time, broker-independent one-minute candle aggregation.

Ticks may arrive out of order within the currently open minute. In that case,
open and close are selected by tick timestamp while every tick contributes to
high and low. For duplicate timestamps, first arrival wins for open and last
arrival wins for close.

Once a later minute has started for an instrument, its prior candle is final.
Ticks for finalized minutes are ignored and counted by ``late_tick_count``.
This prevents late data from mutating candles that may already have consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument, MarketTick

_ONE_MINUTE = timedelta(minutes=1)
_MIN_REASONABLE_TIMESTAMP = datetime(2000, 1, 1, tzinfo=UTC)
_MAX_REASONABLE_TIMESTAMP = datetime(2100, 1, 1, tzinfo=UTC)


class InvalidTickError(ValueError):
    """Raised when a tick cannot safely be incorporated into a candle."""


@dataclass(frozen=True, slots=True)
class Candle:
    """An immutable one-minute OHLC candle using UTC timestamps."""

    instrument: Instrument
    start_time: datetime
    end_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None = None

    def __post_init__(self) -> None:
        start_time = _aware_utc(self.start_time, "start_time")
        end_time = _aware_utc(self.end_time, "end_time")
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)

        if end_time - start_time != _ONE_MINUTE:
            raise ValueError("A candle must span exactly one minute.")
        prices = (self.open, self.high, self.low, self.close)
        if not all(price.is_finite() for price in prices):
            raise ValueError("Candle prices must be finite.")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("Candle high is inconsistent with its prices.")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("Candle low is inconsistent with its prices.")
        if self.volume is not None and self.volume < 0:
            raise ValueError("Candle volume cannot be negative.")


@dataclass(slots=True)
class _WorkingCandle:
    instrument: Instrument
    start_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    first_tick_time: datetime
    last_tick_time: datetime

    @classmethod
    def from_tick(cls, tick: MarketTick, timestamp: datetime) -> _WorkingCandle:
        return cls(
            instrument=tick.instrument,
            start_time=_minute_start(timestamp),
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            first_tick_time=timestamp,
            last_tick_time=timestamp,
        )

    def add(self, tick: MarketTick, timestamp: datetime) -> None:
        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)

        if timestamp < self.first_tick_time:
            self.first_tick_time = timestamp
            self.open = tick.price

        if timestamp >= self.last_tick_time:
            self.last_tick_time = timestamp
            self.close = tick.price

    def finalize(self) -> Candle:
        return Candle(
            instrument=self.instrument,
            start_time=self.start_time,
            end_time=self.start_time + _ONE_MINUTE,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=None,
        )


class CandleBuilder:
    """Aggregate normalized ticks into independent one-minute candles."""

    def __init__(
        self,
        on_candle: Callable[[Candle], None] | None = None,
    ) -> None:
        self._on_candle = on_candle
        self._working: dict[Instrument, _WorkingCandle] = {}
        self._late_tick_count = 0

    @property
    def late_tick_count(self) -> int:
        """Number of ticks ignored because their candle was already final."""
        return self._late_tick_count

    def add_tick(self, tick: MarketTick) -> Candle | None:
        """Consume a tick and return a candle if this tick finalized one."""
        timestamp = _normalize_tick(tick)
        minute_start = _minute_start(timestamp)
        working = self._working.get(tick.instrument)

        if working is None:
            self._working[tick.instrument] = _WorkingCandle.from_tick(tick, timestamp)
            return None

        if minute_start < working.start_time:
            self._late_tick_count += 1
            return None

        if minute_start == working.start_time:
            working.add(tick, timestamp)
            return None

        finalized = working.finalize()
        self._working[tick.instrument] = _WorkingCandle.from_tick(tick, timestamp)
        self._emit(finalized)
        return finalized

    def flush(self) -> tuple[Candle, ...]:
        """Finalize all open candles without manufacturing missing minutes."""
        finalized = tuple(
            working.finalize()
            for working in sorted(
                self._working.values(),
                key=lambda item: (
                    item.start_time,
                    item.instrument.exchange,
                    item.instrument.trading_symbol,
                ),
            )
        )
        self._working.clear()
        for candle in finalized:
            self._emit(candle)
        return finalized

    def _emit(self, candle: Candle) -> None:
        if self._on_candle is not None:
            self._on_candle(candle)


def _normalize_tick(tick: MarketTick) -> datetime:
    timestamp = _aware_utc(tick.timestamp, "tick timestamp")
    if not _MIN_REASONABLE_TIMESTAMP <= timestamp < _MAX_REASONABLE_TIMESTAMP:
        raise InvalidTickError("Tick timestamp is outside the supported range.")
    if not tick.price.is_finite():
        raise InvalidTickError("Tick price must be finite.")
    return timestamp


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidTickError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _minute_start(timestamp: datetime) -> datetime:
    return timestamp.replace(second=0, microsecond=0)


__all__ = ["Candle", "CandleBuilder", "InvalidTickError"]
