"""Broker-neutral derivation of per-minute volume from cumulative snapshots.

Exchanges publish a running session total rather than per-tick volume, so a
minute's volume is the difference between the last cumulative reading of that
minute and the last reading of the previous minute. The first observed minute
therefore has no baseline and reports no volume instead of guessing, and a
counter reset (a new session) restarts the baseline rather than producing a
negative count.

This module performs no broker access and no polling; it only consumes
snapshots supplied by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from ai_trader.broker import Instrument
from ai_trader.market._time import ONE_MINUTE, minute_start

if TYPE_CHECKING:  # Imported for typing only; importing it here would cycle.
    from ai_trader.market.candles import Candle


@dataclass(frozen=True, slots=True)
class CumulativeVolumeSnapshot:
    """Cumulative exchange volume observed at a point in time."""

    instrument: Instrument
    timestamp: datetime
    cumulative_volume: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Volume snapshot timestamp must be timezone-aware.")
        if self.cumulative_volume < 0:
            raise ValueError("Cumulative volume cannot be negative.")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class MinuteVolume:
    """Volume derived for one instrument and one minute interval."""

    instrument: Instrument
    start_time: datetime
    end_time: datetime
    volume: int

    def __post_init__(self) -> None:
        if self.start_time.tzinfo is None or self.start_time.utcoffset() is None:
            raise ValueError("Minute-volume start time must be timezone-aware.")
        if self.end_time.tzinfo is None or self.end_time.utcoffset() is None:
            raise ValueError("Minute-volume end time must be timezone-aware.")
        start_time = self.start_time.astimezone(UTC)
        end_time = self.end_time.astimezone(UTC)
        if end_time <= start_time:
            raise ValueError("Minute-volume end time must follow its start time.")
        if self.volume < 0:
            raise ValueError("Minute volume cannot be negative.")
        object.__setattr__(self, "start_time", start_time)
        object.__setattr__(self, "end_time", end_time)


class VolumeEnricher(Protocol):
    """Derives interval volume from snapshots and applies it to candles."""

    def observe(self, snapshot: CumulativeVolumeSnapshot) -> MinuteVolume | None:
        """Observe a snapshot and possibly derive a completed minute's volume."""
        ...

    def close_minute(self, instrument: Instrument) -> MinuteVolume | None:
        """Close the open minute early, returning its volume so far."""
        ...

    def enrich(self, candle: Candle, minute_volume: MinuteVolume) -> Candle:
        """Return a candle enriched with matching derived volume."""
        ...


@dataclass(slots=True)
class _VolumeState:
    """Per-instrument differencing state for the minute being accumulated."""

    minute: datetime
    latest: int
    baseline: int | None


class CumulativeVolumeTracker:
    """Derive per-minute volume by differencing cumulative snapshots."""

    def __init__(self) -> None:
        self._states: dict[Instrument, _VolumeState] = {}
        self._late_snapshot_count = 0

    @property
    def late_snapshot_count(self) -> int:
        """Number of snapshots ignored because their minute was already closed."""
        return self._late_snapshot_count

    def observe(self, snapshot: CumulativeVolumeSnapshot) -> MinuteVolume | None:
        """Record a snapshot, returning a minute's volume when one completes."""
        minute = minute_start(snapshot.timestamp)
        cumulative = snapshot.cumulative_volume
        state = self._states.get(snapshot.instrument)

        if state is None:
            self._states[snapshot.instrument] = _VolumeState(
                minute=minute,
                latest=cumulative,
                baseline=None,
            )
            return None

        if minute < state.minute:
            self._late_snapshot_count += 1
            return None

        if minute == state.minute:
            if cumulative < state.latest:
                # The session counter restarted; this minute is unattributable.
                state.baseline = None
            state.latest = cumulative
            return None

        completed = self._completed(snapshot.instrument, state)
        state.baseline = state.latest
        state.minute = minute
        if cumulative < state.latest:
            state.baseline = None
        state.latest = cumulative
        return completed

    def close_minute(self, instrument: Instrument) -> MinuteVolume | None:
        """Close the open minute without waiting for the next one to start."""
        state = self._states.get(instrument)
        if state is None:
            return None
        completed = self._completed(instrument, state)
        state.baseline = state.latest
        return completed

    def enrich(self, candle: Candle, minute_volume: MinuteVolume) -> Candle:
        """Return ``candle`` carrying the volume of the matching minute."""
        if candle.instrument != minute_volume.instrument:
            raise ValueError("Minute volume belongs to a different instrument.")
        if (
            candle.start_time != minute_volume.start_time
            or candle.end_time != minute_volume.end_time
        ):
            raise ValueError("Minute volume belongs to a different interval.")
        return replace(candle, volume=minute_volume.volume)

    def _completed(
        self,
        instrument: Instrument,
        state: _VolumeState,
    ) -> MinuteVolume | None:
        if state.baseline is None:
            return None
        return MinuteVolume(
            instrument=instrument,
            start_time=state.minute,
            end_time=state.minute + ONE_MINUTE,
            volume=state.latest - state.baseline,
        )


__all__ = [
    "CumulativeVolumeSnapshot",
    "CumulativeVolumeTracker",
    "MinuteVolume",
    "VolumeEnricher",
]
