"""Real-time, broker-independent one-minute candle aggregation.

Ticks may arrive out of order within the currently open minute. In that case,
open and close are selected by tick timestamp while every tick contributes to
high and low. For duplicate timestamps, first arrival wins for open and last
arrival wins for close.

Once a later minute has started for an instrument, its prior candle is final.
Ticks for finalized minutes are ignored and counted by ``late_tick_count``.
This prevents late data from mutating candles that may already have consumers,
and it holds across ``flush`` as well as ordinary minute roll-over.

A minute the builder cannot vouch for is discarded rather than emitted. The
first minute seen for an instrument is normally a fragment, because a stream is
normally joined partway through one: its open is the first tick that happened
to arrive rather than the minute's true open, and its high and low span only
the part that was watched. Such a candle is indistinguishable from a real one
downstream, which makes it worse than no candle at all.

Whether it is a fragment is the caller's fact rather than this module's, so a
caller may state it. ``watching_since`` is the instant ticks were known to be
arriving from, and a first minute beginning at or after it is kept. A live
session subscribes before the bell, so without that the 09:15 candle -- the one
the opening range is measured from -- was discarded every session for a reason
that describes only a mid-session join, while the same session replayed from
history kept it. Told nothing, the builder still assumes it joined late.

A kept first minute carries no volume: differencing a cumulative session total
needs an earlier reading, and the first minute has none. That is reported as
unknown rather than as zero, which is the distinction the layers above turn on.

Ticks that carry cumulative session volume are differenced into per-minute
volume, so every emitted candle carries real volume whenever the broker reports
it. Aggregation is thread-safe because broker SDKs deliver ticks on their own
feed threads; ``on_candle`` is invoked outside the internal lock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from threading import Lock

from ai_trader.broker import Instrument, MarketTick, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE, minute_start
from ai_trader.market.volume import (
    CumulativeVolumeSnapshot,
    CumulativeVolumeTracker,
    MinuteVolume,
    VolumeEnricher,
)

_MIN_REASONABLE_TIMESTAMP = datetime(2000, 1, 1, tzinfo=INDIA_TIMEZONE)
_MAX_REASONABLE_TIMESTAMP = datetime(2100, 1, 1, tzinfo=INDIA_TIMEZONE)
"""A century-wide sanity range, not a market calendar.

The zone moves each bound by the IST offset, which is immaterial at this width
and is not the reason for naming it: these are the only two datetimes this
module constructs, and constructing them in any zone but the one every other
timestamp here carries would put a second zone in the file for no gain.
"""


class InvalidTickError(ValueError):
    """Raised when a tick cannot safely be incorporated into a candle."""


@dataclass(frozen=True, slots=True)
class Candle:
    """An immutable one-minute OHLC candle timestamped in IST.

    The zone is normalized on construction rather than merely required, so a
    candle built from a broker payload and one built from live ticks answer
    ``.date()`` and ``.hour`` the same way. Both describe the same instant
    either way -- the normalization is about what the session-scoped layers
    above read out of that instant, and this is an Indian exchange.
    """

    instrument: Instrument
    start_time: datetime
    end_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None = None

    def __post_init__(self) -> None:
        start_time = _aware_ist(self.start_time, "start_time")
        end_time = _aware_ist(self.end_time, "end_time")
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)

        if end_time - start_time != ONE_MINUTE:
            raise ValueError("A candle must span exactly one minute.")
        prices = (self.open, self.high, self.low, self.close)
        if not all(isinstance(price, Decimal) for price in prices):
            raise ValueError("Candle prices must be Decimal values.")
        if not all(price.is_finite() for price in prices):
            raise ValueError("Candle prices must be finite.")
        # A cash equity never prints at or below zero, so a zero here is an
        # unpopulated field wearing the costume of a price: the same failure
        # mode as the broker's unset volume, and undetectable one step later.
        # Letting it through would feed a real-looking observation to the
        # trend features, which have no division to guard them and would stay
        # dragged toward zero for their whole span while reporting ready.
        if not all(price > 0 for price in prices):
            raise ValueError("Candle prices must be positive.")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("Candle high is inconsistent with its prices.")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("Candle low is inconsistent with its prices.")
        if self.volume is not None and self.volume < 0:
            raise ValueError("Candle volume cannot be negative.")


def to_candle(instrument: Instrument, source: OHLCVCandle) -> Candle:
    """Convert a broker OHLCV candle, whose timestamp starts the interval.

    Which end of the minute a broker's timestamp refers to is a fact about the
    broker, not about the caller, and getting it wrong shifts every candle by a
    minute without producing a single malformed one -- features would still
    compute, the suite would still pass, and replay would be reasoning about the
    bar after the one it thought it held.

    That is why this is public and lives here rather than being written a second
    time where it is next needed. It was private to ``state.py`` while live
    backfill was the only caller; the historical store is the second, and two
    copies of a convention only one of which gets corrected is precisely the
    failure this package is arranged to prevent.
    """
    start_time = minute_start(source.timestamp)
    return Candle(
        instrument=instrument,
        start_time=start_time,
        end_time=start_time + ONE_MINUTE,
        open=source.open,
        high=source.high,
        low=source.low,
        close=source.close,
        volume=source.volume,
    )


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
    """Aggregate normalized ticks into independent one-minute candles.

    The first minute seen for an instrument is finalized internally but not
    emitted unless ``watching_since`` shows ticks were already arriving when
    that minute began. Absent that evidence a stream joined mid-minute can only
    have observed a fragment of it, so every candle a caller receives is one
    the builder can say it watched from the start.

    That instant is supplied rather than observed. This class is driven
    entirely by tick timestamps, and reading a wall clock here would make the
    same tape produce different candles on different runs -- which is exactly
    the equivalence between a live session and its replay that the instant
    exists to restore.
    """

    def __init__(
        self,
        on_candle: Callable[[Candle], None] | None = None,
        volume_enricher: VolumeEnricher | None = None,
        watching_since: datetime | None = None,
    ) -> None:
        self._on_candle = on_candle
        self._volume: VolumeEnricher = volume_enricher or CumulativeVolumeTracker()
        self._working: dict[Instrument, _WorkingCandle] = {}
        self._finalized_minute: dict[Instrument, datetime] = {}
        self._watching_since = (
            None
            if watching_since is None
            else _aware_ist(watching_since, "watching_since")
        )
        self._watching: dict[Instrument, datetime | None] = {}
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
                candle = self._close_locked(working, minute_volume)
                if candle is not None:
                    closed.append(candle)
            self._working.clear()
            finalized = tuple(closed)

        for candle in finalized:
            self._emit(candle)
        return finalized

    def forget(self, instrument: Instrument, *, at: datetime | None = None) -> None:
        """Drop all aggregation state for an instrument.

        Any minute still open for it is discarded rather than emitted: the
        builder is being told it stopped watching, so that minute is a fragment
        for exactly the reason a mid-session first minute is. Forgetting also
        clears the record of which minutes were finalized, so the instrument's
        next minute is an opening one again.

        ``at`` says when watching resumed, and replaces whatever the builder
        was constructed believing about this instrument. A reconnect lands
        partway through a minute, so the usual answer keeps the existing
        behaviour -- that minute is discarded -- but a caller that resubscribed
        on a boundary, or before the bell, can say so and keep it. Omitting
        ``at`` is the conservative reading: watching resumed at a moment this
        builder cannot name, so the next minute is assumed to be a fragment.

        Forgetting an instrument whose ticks are still arriving is a caller
        error; the state simply rebuilds from the next tick.
        """
        watching = None if at is None else _aware_ist(at, "at")
        with self._lock:
            self._working.pop(instrument, None)
            self._finalized_minute.pop(instrument, None)
            self._watching[instrument] = watching
            self._volume.forget(instrument)

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
    ) -> Candle | None:
        """Finalize a minute, dropping a first one the builder may have missed.

        ``_finalized_minute`` is written for the discarded minute too. Without
        that, a late tick for it would look like the start of a fresh minute and
        a second, even smaller fragment of the same minute would be emitted.
        """
        first_minute = working.instrument not in self._finalized_minute
        self._finalized_minute[working.instrument] = working.start_time
        candle = working.finalize()
        if first_minute and not self._watched_from_start_locked(working):
            return None
        if minute_volume is None or minute_volume.start_time != candle.start_time:
            return candle
        return self._volume.enrich(candle, minute_volume)

    def _watched_from_start_locked(self, working: _WorkingCandle) -> bool:
        """Whether the builder can say it was receiving ticks when this minute began.

        Absence of an answer is not a yes. A builder told nothing, or told only
        that watching resumed at an unnamed moment, reports False and the
        minute is discarded -- the behaviour of every caller that says nothing.
        """
        if working.instrument in self._watching:
            watching_since = self._watching[working.instrument]
        else:
            watching_since = self._watching_since
        return watching_since is not None and watching_since <= working.start_time

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
    timestamp = _aware_ist(tick.timestamp, "tick timestamp")
    if not _MIN_REASONABLE_TIMESTAMP <= timestamp < _MAX_REASONABLE_TIMESTAMP:
        raise InvalidTickError("Tick timestamp is outside the supported range.")
    if not isinstance(tick.price, Decimal):
        raise InvalidTickError("Tick price must be a Decimal.")
    if not tick.price.is_finite():
        raise InvalidTickError("Tick price must be finite.")
    # Rejected here as well as on the candle so a bad tick is refused at the
    # boundary it entered by, rather than surfacing a minute later as a candle
    # whose own inputs are long gone.
    if tick.price <= 0:
        raise InvalidTickError("Tick price must be positive.")
    if tick.cumulative_volume is not None and tick.cumulative_volume < 0:
        raise InvalidTickError("Tick cumulative volume cannot be negative.")
    return timestamp


def _aware_ist(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidTickError(f"{field_name} must be timezone-aware.")
    return value.astimezone(INDIA_TIMEZONE)


__all__ = ["Candle", "CandleBuilder", "InvalidTickError", "to_candle"]
