from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pytest

from ai_trader.broker import Instrument
from ai_trader.market import CumulativeVolumeSnapshot, MinuteVolume


def test_volume_models_are_immutable_and_normalize_timestamps_to_utc() -> None:
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")
    india_timezone = timezone(timedelta(hours=5, minutes=30))
    snapshot = CumulativeVolumeSnapshot(
        instrument=instrument,
        timestamp=datetime(2026, 9, 14, 15, 30, tzinfo=india_timezone),
        cumulative_volume=100_000,
    )
    minute_volume = MinuteVolume(
        instrument=instrument,
        start_time=datetime(2026, 9, 14, 15, 29, tzinfo=india_timezone),
        end_time=datetime(2026, 9, 14, 15, 30, tzinfo=india_timezone),
        volume=500,
    )

    assert snapshot.timestamp == datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    assert minute_volume.start_time == datetime(2026, 9, 14, 9, 59, tzinfo=UTC)
    assert minute_volume.end_time == datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, "cumulative_volume", 200_000)
