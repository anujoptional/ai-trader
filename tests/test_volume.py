from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument
from ai_trader.market import (
    Candle,
    CumulativeVolumeSnapshot,
    CumulativeVolumeTracker,
    MinuteVolume,
)

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_NIFTY = Instrument(exchange="NSE", trading_symbol="NIFTY")
_MINUTE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)


def _snapshot(
    minutes: int,
    cumulative_volume: int,
    seconds: int = 0,
    instrument: Instrument = _RELIANCE,
) -> CumulativeVolumeSnapshot:
    return CumulativeVolumeSnapshot(
        instrument=instrument,
        timestamp=_MINUTE + timedelta(minutes=minutes, seconds=seconds),
        cumulative_volume=cumulative_volume,
    )


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


def test_first_minute_reports_no_volume_for_lack_of_a_baseline() -> None:
    tracker = CumulativeVolumeTracker()

    assert tracker.observe(_snapshot(0, 1_000)) is None
    assert tracker.observe(_snapshot(0, 1_500, seconds=30)) is None
    assert tracker.observe(_snapshot(1, 2_200)) is None


def test_minute_volume_differences_consecutive_closing_readings() -> None:
    tracker = CumulativeVolumeTracker()
    for snapshot in (_snapshot(0, 1_000), _snapshot(1, 1_500), _snapshot(1, 2_200, 30)):
        tracker.observe(snapshot)

    completed = tracker.observe(_snapshot(2, 2_500))

    assert completed == MinuteVolume(
        instrument=_RELIANCE,
        start_time=_MINUTE + timedelta(minutes=1),
        end_time=_MINUTE + timedelta(minutes=2),
        volume=1_200,
    )


def test_instruments_are_tracked_independently() -> None:
    tracker = CumulativeVolumeTracker()
    tracker.observe(_snapshot(0, 1_000))
    tracker.observe(_snapshot(0, 50, instrument=_NIFTY))
    tracker.observe(_snapshot(1, 1_500))
    tracker.observe(_snapshot(1, 90, instrument=_NIFTY))

    reliance = tracker.observe(_snapshot(2, 1_900))
    nifty = tracker.observe(_snapshot(2, 140, instrument=_NIFTY))

    assert reliance is not None
    assert nifty is not None
    assert reliance.volume == 500
    assert nifty.volume == 40


@pytest.mark.parametrize("reset_seconds", [0, 30])
def test_counter_reset_suppresses_volume_until_a_baseline_returns(
    reset_seconds: int,
) -> None:
    tracker = CumulativeVolumeTracker()
    tracker.observe(_snapshot(0, 1_000))
    tracker.observe(_snapshot(1, 1_500))

    if reset_seconds:
        tracker.observe(_snapshot(1, 200, seconds=reset_seconds))
        assert tracker.observe(_snapshot(2, 600)) is None
    else:
        completed = tracker.observe(_snapshot(2, 200))
        assert completed is not None
        assert completed.volume == 500
        assert tracker.observe(_snapshot(3, 600)) is None

    recovered = tracker.observe(_snapshot(4, 900))

    assert recovered is not None
    assert recovered.volume == 400


def test_late_snapshot_is_counted_and_ignored() -> None:
    tracker = CumulativeVolumeTracker()
    tracker.observe(_snapshot(0, 1_000))
    tracker.observe(_snapshot(1, 1_500))

    assert tracker.observe(_snapshot(0, 9_999, seconds=45)) is None
    assert tracker.late_snapshot_count == 1
    completed = tracker.observe(_snapshot(2, 1_800))
    assert completed is not None
    assert completed.volume == 500


def test_a_gap_in_readings_reports_no_volume_for_the_minute_it_spans() -> None:
    tracker = CumulativeVolumeTracker()
    tracker.observe(_snapshot(0, 100_000))
    tracker.observe(_snapshot(1, 200_000))

    # Minutes 2 to 4 produce no snapshot at all, because no ticks printed or
    # because whatever supplies the total stalled.
    closed = tracker.observe(_snapshot(5, 600_000))
    spanning = tracker.observe(_snapshot(6, 700_000))
    resumed = tracker.observe(_snapshot(7, 800_000))

    # Minute 1 still closes normally: its baseline closed minute 0.
    assert closed is not None
    assert closed.volume == 100_000
    # Minute 5's baseline closed minute 1, so their difference spans the whole
    # gap. Reporting it would turn four minutes of missing data into a single
    # 400_000 spike -- worse than a missing figure, because a scanner triggers
    # on a spike where it would have suppressed on nothing.
    assert spanning is None
    # Minute 6's baseline closed minute 5, so differencing measures one minute
    # again and volume resumes without the gap having to be papered over.
    assert resumed is not None
    assert resumed.volume == 100_000


def test_close_minute_closes_the_open_minute_without_double_counting() -> None:
    tracker = CumulativeVolumeTracker()
    tracker.observe(_snapshot(0, 1_000))
    tracker.observe(_snapshot(1, 1_500))

    closed = tracker.close_minute(_RELIANCE)
    repeated = tracker.close_minute(_RELIANCE)

    assert closed is not None
    assert closed.start_time == _MINUTE + timedelta(minutes=1)
    assert closed.volume == 500
    # The repeated call reports nothing rather than a second, zero-volume copy
    # of a minute already closed. Zero is a claim that nothing traded, and a
    # caller enriching a candle with it would overwrite real volume.
    assert repeated is None
    assert tracker.close_minute(_NIFTY) is None


def test_enrich_applies_volume_only_to_the_matching_candle() -> None:
    tracker = CumulativeVolumeTracker()
    candle = Candle(
        instrument=_RELIANCE,
        start_time=_MINUTE,
        end_time=_MINUTE + timedelta(minutes=1),
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
    )
    matching = MinuteVolume(
        instrument=_RELIANCE,
        start_time=_MINUTE,
        end_time=_MINUTE + timedelta(minutes=1),
        volume=500,
    )

    assert tracker.enrich(candle, matching).volume == 500
    with pytest.raises(ValueError, match="different interval"):
        tracker.enrich(
            candle,
            MinuteVolume(
                instrument=_RELIANCE,
                start_time=_MINUTE + timedelta(minutes=1),
                end_time=_MINUTE + timedelta(minutes=2),
                volume=500,
            ),
        )
    with pytest.raises(ValueError, match="different instrument"):
        tracker.enrich(
            candle,
            MinuteVolume(
                instrument=_NIFTY,
                start_time=_MINUTE,
                end_time=_MINUTE + timedelta(minutes=1),
                volume=500,
            ),
        )


def test_forgetting_an_instrument_drops_its_baseline() -> None:
    """A stale baseline is worse than none: it spans however long the gap was."""
    tracker = CumulativeVolumeTracker()
    tracker.observe(_snapshot(0, 1_000))
    tracker.observe(_snapshot(1, 1_400))

    tracker.forget(_RELIANCE)

    # Without forgetting, this same call completes minute 1 at 400. The point of
    # forgetting is that the reading it would have differenced against is gone,
    # so the minute reports nothing rather than a number measured against a
    # total from before the instrument stopped being watched.
    assert tracker.observe(_snapshot(2, 1_900)) is None
    assert tracker.observe(_snapshot(3, 2_500)) is None
    # Differencing resumes once two adjacent minutes have been seen afresh.
    assert tracker.observe(_snapshot(4, 2_800)) == MinuteVolume(
        instrument=_RELIANCE,
        start_time=_MINUTE + timedelta(minutes=3),
        end_time=_MINUTE + timedelta(minutes=4),
        volume=600,
    )


def test_forgetting_one_instrument_leaves_the_others_differencing() -> None:
    tracker = CumulativeVolumeTracker()
    for instrument in (_RELIANCE, _NIFTY):
        tracker.observe(_snapshot(0, 1_000, instrument=instrument))
        tracker.observe(_snapshot(1, 1_400, instrument=instrument))

    tracker.forget(_RELIANCE)

    assert tracker.observe(_snapshot(2, 1_900)) is None
    completed = tracker.observe(_snapshot(2, 1_900, instrument=_NIFTY))
    assert completed is not None
    assert completed.instrument == _NIFTY
    assert completed.volume == 400


def test_forgetting_an_unknown_instrument_is_harmless() -> None:
    tracker = CumulativeVolumeTracker()

    tracker.forget(_RELIANCE)

    assert tracker.observe(_snapshot(0, 1_000)) is None
