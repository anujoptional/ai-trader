"""Broker-neutral contracts for future candle-volume enrichment.

A future implementation can derive interval volume from consecutive cumulative
exchange-volume snapshots. This module deliberately defines no broker access or
polling behavior.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from ai_trader.broker import Instrument
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
    """Future interface for cumulative-volume differencing and enrichment."""

    def observe(self, snapshot: CumulativeVolumeSnapshot) -> MinuteVolume | None:
        """Observe a snapshot and possibly derive a completed minute's volume."""
        ...

    def enrich(self, candle: Candle, minute_volume: MinuteVolume) -> Candle:
        """Return a candle enriched with matching derived volume."""
        ...


__all__ = ["CumulativeVolumeSnapshot", "MinuteVolume", "VolumeEnricher"]
