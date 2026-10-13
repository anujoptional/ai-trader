"""Observe native streamed cumulative volume without polling or imputing it."""

from collections.abc import Iterable
from threading import Lock

from ai_trader.broker import Instrument, MarketTick


class StreamVolume:
    poll_count = 0
    failure_count = 0
    regression_count = 0
    stale_stamp_count = 0
    last_error = None

    def __init__(self, instruments: Iterable[Instrument]) -> None:
        self._instruments = frozenset(instruments)
        self._latest: dict[Instrument, int | None] = {}
        self._lock = Lock()

    def stamp(self, tick: MarketTick) -> MarketTick:
        if tick.instrument in self._instruments:
            with self._lock:
                self._latest[tick.instrument] = tick.cumulative_volume
        return tick

    def latest(self, instrument: Instrument) -> int | None:
        with self._lock:
            return self._latest.get(instrument)

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        pass
