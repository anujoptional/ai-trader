"""Retained, queryable market state for a set of instruments.

``CandleBuilder`` aggregates ticks but keeps no history. ``MarketState`` is the
view a strategy reads: a bounded rolling window of one-minute candles per
instrument plus the latest observed price, filled from historical candles at
startup and from live ticks thereafter.

Candles are appended in strictly increasing minute order. A candle for a minute
at or before the newest retained one is rejected as a duplicate rather than
corrupting history, which makes historical backfill and live streaming safe to
combine. The class is thread-safe; broker SDKs deliver ticks on feed threads.

Retention is bounded per instrument but unbounded across them, so a process that
watches a changing set has to say so: ``forget`` drops one instrument and
``retain`` drops everything outside a declared set. Both clear the aggregation
state alongside the history, so an instrument that comes back is rebuilt from
scratch rather than resuming against a stale minute or volume baseline.

A broker whose stream carries no volume can supply it separately, so ticks may
need a cumulative total attached before they are aggregated. ``tick_stamper``
is that seam. It is a plain callable rather than a named type so this module
keeps no dependency on whatever supplies the total: ``VolumePoller.stamp``
satisfies it, but this module must not import it, since that would drag the
broker into the aggregation layer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from threading import Lock

from ai_trader.broker import Instrument, MarketTick, OHLCVCandle
from ai_trader.market._time import ONE_MINUTE, minute_start
from ai_trader.market.candles import Candle, CandleBuilder

_SESSION_MINUTES = 375
"""One full NSE equity session: 09:15 to 15:30 IST inclusive of the open."""


@dataclass(frozen=True, slots=True)
class InstrumentState:
    """A point-in-time view of one instrument's market state."""

    instrument: Instrument
    last_price: Decimal | None
    last_tick_at: datetime | None
    candles: tuple[Candle, ...]

    @property
    def latest_candle(self) -> Candle | None:
        """The most recently completed candle, if any."""
        return self.candles[-1] if self.candles else None


class MarketState:
    """A thread-safe rolling window of completed candles and latest prices."""

    def __init__(
        self,
        max_candles: int = _SESSION_MINUTES,
        on_candle: Callable[[Candle], None] | None = None,
        tick_stamper: Callable[[MarketTick], MarketTick] | None = None,
    ) -> None:
        if max_candles <= 0:
            raise ValueError("max_candles must be positive.")
        self._max_candles = max_candles
        self._on_candle = on_candle
        self._tick_stamper = tick_stamper
        self._builder = CandleBuilder(on_candle=self._append_from_builder)
        self._candles: dict[Instrument, deque[Candle]] = {}
        self._last_price: dict[Instrument, Decimal] = {}
        self._last_tick_at: dict[Instrument, datetime] = {}
        self._duplicate_candle_count = 0
        self._lock = Lock()

    @property
    def late_tick_count(self) -> int:
        """Ticks ignored because their candle was already final."""
        return self._builder.late_tick_count

    @property
    def duplicate_candle_count(self) -> int:
        """Candles rejected for not advancing an instrument's history."""
        with self._lock:
            return self._duplicate_candle_count

    def record_tick(self, tick: MarketTick) -> Candle | None:
        """Consume a live tick, returning a candle if this tick finalized one.

        The tick is stamped before it is aggregated, so a broker whose stream
        omits volume still produces candles carrying it. Stamping here rather
        than at each call site means a caller cannot wire up a poller and then
        silently lose volume for a whole session by forgetting one path.
        """
        if self._tick_stamper is not None:
            tick = self._tick_stamper(tick)
        finalized = self._builder.add_tick(tick)
        timestamp = tick.timestamp
        with self._lock:
            previous = self._last_tick_at.get(tick.instrument)
            if previous is None or timestamp >= previous:
                self._last_tick_at[tick.instrument] = timestamp
                self._last_price[tick.instrument] = tick.price
        return finalized

    def record_candle(self, candle: Candle) -> bool:
        """Append a completed candle, returning False if it was a duplicate."""
        with self._lock:
            return self._record_candle_locked(candle)

    def backfill(
        self,
        instrument: Instrument,
        candles: Iterable[OHLCVCandle],
    ) -> int:
        """Seed history from broker candles, returning how many were accepted."""
        accepted = 0
        with self._lock:
            for source in candles:
                if self._record_candle_locked(_to_candle(instrument, source)):
                    accepted += 1
        return accepted

    def forget(self, instrument: Instrument) -> bool:
        """Drop everything retained for an instrument, reporting if it was known.

        Retention is bounded per instrument but not across them, so a process
        that watches a different set each session would otherwise accumulate
        both memory and stale instruments that ``snapshots`` keeps reporting
        long after their candles stopped meaning anything.

        The aggregation state goes with it, so an instrument that returns is
        rebuilt from scratch: its first minute back is discarded as a fragment
        and its first volume reading becomes a fresh baseline. Forgetting an
        instrument that is still being streamed is a caller error.
        """
        self._builder.forget(instrument)
        with self._lock:
            known = instrument in self._candles or instrument in self._last_price
            self._candles.pop(instrument, None)
            self._last_price.pop(instrument, None)
            self._last_tick_at.pop(instrument, None)
        return known

    def retain(self, instruments: Iterable[Instrument]) -> tuple[Instrument, ...]:
        """Forget every instrument outside ``instruments``, returning those dropped.

        Declaring the live set is what makes eviction correct by construction: a
        caller that had to name what to forget would need its own record of what
        it previously watched, which is the state this object already holds.
        """
        keep = set(instruments)
        dropped = tuple(
            instrument for instrument in self.instruments() if instrument not in keep
        )
        for instrument in dropped:
            self.forget(instrument)
        return dropped

    def flush(self) -> tuple[Candle, ...]:
        """Finalize every open candle and fold it into retained history."""
        return self._builder.flush()

    def snapshot(self, instrument: Instrument) -> InstrumentState | None:
        """Return the current state of one instrument, if it is known."""
        with self._lock:
            return self._snapshot_locked(instrument)

    def snapshots(self) -> tuple[InstrumentState, ...]:
        """Return the state of every known instrument, in a stable order."""
        with self._lock:
            instruments = self._instruments_locked()
            states = [self._snapshot_locked(instrument) for instrument in instruments]
        return tuple(state for state in states if state is not None)

    def instruments(self) -> tuple[Instrument, ...]:
        """Return every instrument with retained state, in a stable order."""
        with self._lock:
            return self._instruments_locked()

    def _append_from_builder(self, candle: Candle) -> None:
        with self._lock:
            self._record_candle_locked(candle)
        if self._on_candle is not None:
            self._on_candle(candle)

    def _record_candle_locked(self, candle: Candle) -> bool:
        history = self._candles.get(candle.instrument)
        if history is None:
            history = deque(maxlen=self._max_candles)
            self._candles[candle.instrument] = history
        elif history and candle.start_time <= history[-1].start_time:
            self._duplicate_candle_count += 1
            return False
        history.append(candle)
        return True

    def _snapshot_locked(self, instrument: Instrument) -> InstrumentState | None:
        history = self._candles.get(instrument)
        last_price = self._last_price.get(instrument)
        if history is None and last_price is None:
            return None
        return InstrumentState(
            instrument=instrument,
            last_price=last_price,
            last_tick_at=self._last_tick_at.get(instrument),
            candles=tuple(history) if history is not None else (),
        )

    def _instruments_locked(self) -> tuple[Instrument, ...]:
        known = set(self._candles) | set(self._last_price)
        return tuple(
            sorted(known, key=lambda item: (item.exchange, item.trading_symbol))
        )


def _to_candle(instrument: Instrument, source: OHLCVCandle) -> Candle:
    """Convert a broker OHLCV candle, whose timestamp starts the interval."""
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


__all__ = ["InstrumentState", "MarketState"]
