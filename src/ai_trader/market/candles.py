"""Real-time, broker-independent one-minute candle aggregation.

Ticks may arrive out of order within the currently open minute. In that case,
open and close are selected by tick timestamp while every tick contributes to
high and low. For duplicate timestamps, first arrival wins for open and last
arrival wins for close.

Once a later minute has started for an instrument, its prior candle is final.
Ticks for finalized minutes are ignored and counted by ``late_tick_count``.
This prevents late data from mutating candles that may already have consumers,
and it holds across ``flush`` as well as ordinary minute roll-over.

Ticks that carry cumulative session volume are differenced into per-minute
volume, so a finalized candle carries real volume whenever the broker reports
it. Aggregation is thread-safe because broker SDKs deliver ticks on their own
feed threads; ``on_candle`` is invoked outside the internal lock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from threading import Lock

from ai_trader.broker import Instrument, MarketTick
from ai_trader.market._time import ONE_MINUTE, minute_start
from ai_trader.market.volume import (
    CumulativeVolumeSnapshot,
    CumulativeVolumeTracker,
    MinuteVolume,
    VolumeEnricher,
)

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

        if end_time - start_time != ONE_MINUTE:
            raise ValueError("A candle must span exactly one minute.")
        prices = (self.open, self.high, self.low, self.close)
        if not all(isinstance(price, Decimal) for price in prices):
            raise ValueError("Candle prices must be Decimal values.")
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
            start_time=minute_start(timestamp),
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
            end_time=self.start_time + ONE_MINUTE,
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
        volume_enricher: VolumeEnricher | None = None,
    ) -> None:
        self._on_candle = on_candle
        self._volume: VolumeEnricher = volume_enricher or CumulativeVolumeTracker()
        self._working: dict[Instrument, _WorkingCandle] = {}
        self._finalized_minute: dict[Instrument, datetime] = {}
        self._late_tick_count = 0
        self._lock = Lock()

    @property
    def late_tick_count(self) -> int:
        """Number of ticks ignored because their candle was already final."""
        with self._lock:
            return self._late_tick_count

    def add_tick(self, tick: MarketTick) -> Candle | None:
        """Consume a tick and return a candle if this tick finalized one."""
        timestamp = _normalize_tick(tick)
        start_time = minute_start(timestamp)

        with self._lock:
            finalized = self._add_tick_locked(tick, timestamp, start_time)

        if finalized is not None:
            self._emit(finalized)
        return finalized

    def flush(self) -> tuple[Candle, ...]:
        """Finalize all open candles without manufacturing missing minutes."""
        with self._lock:
            pending = sorted(
                self._working.values(),
                key=lambda item: (
                    item.start_time,
                    item.instrument.exchange,
                    item.instrument.trading_symbol,
                ),
            )
            closed = []
            for working in pending:
                minute_volume = self._volume.close_minute(working.instrument)
                closed.append(self._close_locked(working, minute_volume))
            self._working.clear()
            finalized = tuple(closed)

        for candle in finalized:
            self._emit(candle)
        return finalized

    def _add_tick_locked(
        self,
        tick: MarketTick,
        timestamp: datetime,
        start_time: datetime,
    ) -> Candle | None:
        last_finalized = self._finalized_minute.get(tick.instrument)
        if last_finalized is not None and start_time <= last_finalized:
            self._late_tick_count += 1
            return None

        working = self._working.get(tick.instrument)
        if working is None:
            self._working[tick.instrument] = _WorkingCandle.from_tick(tick, timestamp)
            self._observe_volume(tick, timestamp)
            return None

        if start_time < working.start_time:
            self._late_tick_count += 1
            return None

        if start_time == working.start_time:
            working.add(tick, timestamp)
            self._observe_volume(tick, timestamp)
            return None

        self._working[tick.instrument] = _WorkingCandle.from_tick(tick, timestamp)
        return self._close_locked(working, self._observe_volume(tick, timestamp))

    def _close_locked(
        self,
        working: _WorkingCandle,
        minute_volume: MinuteVolume | None,
    ) -> Candle:
        self._finalized_minute[working.instrument] = working.start_time
        candle = working.finalize()
        if minute_volume is None or minute_volume.start_time != candle.start_time:
            return candle
        return self._volume.enrich(candle, minute_volume)

    def _observe_volume(
        self,
        tick: MarketTick,
        timestamp: datetime,
    ) -> MinuteVolume | None:
        if tick.cumulative_volume is None:
            return None
        return self._volume.observe(
            CumulativeVolumeSnapshot(
                instrument=tick.instrument,
                timestamp=timestamp,
                cumulative_volume=tick.cumulative_volume,
            )
        )

    def _emit(self, candle: Candle) -> None:
        if self._on_candle is not None:
            self._on_candle(candle)


def _normalize_tick(tick: MarketTick) -> datetime:
    timestamp = _aware_utc(tick.timestamp, "tick timestamp")
    if not _MIN_REASONABLE_TIMESTAMP <= timestamp < _MAX_REASONABLE_TIMESTAMP:
        raise InvalidTickError("Tick timestamp is outside the supported range.")
    if not isinstance(tick.price, Decimal):
        raise InvalidTickError("Tick price must be a Decimal.")
    if not tick.price.is_finite():
        raise InvalidTickError("Tick price must be finite.")
    if tick.cumulative_volume is not None and tick.cumulative_volume < 0:
        raise InvalidTickError("Tick cumulative volume cannot be negative.")
    return timestamp


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidTickError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


__all__ = ["Candle", "CandleBuilder", "InvalidTickError"]
