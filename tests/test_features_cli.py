"""Tests for the feature-engine command-line check.

Every Groww call is mocked. That is not only for speed: the whole point of the
feature layer is that it is offline once candles exist, so a test that asserts
the broker is never touched after the session fetch is testing the design.

The fixture closes cycle through a small range, which makes the interesting
values exact. Sixty candles is the shortest run that leaves every feature
available, including the fifty-period EMA.
"""

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import fields
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytest import CaptureFixture

from ai_trader.broker import MarketQuote, MarketTick, OHLCVCandle
from ai_trader.broker.groww import GrowwBrokerError, GrowwStreamConnectionError
from ai_trader.cli.check_features import _RELIANCE, SessionNotFoundError, main
from ai_trader.config import ConfigurationError
from ai_trader.features import FeatureReadiness, FeatureSnapshot

_IDENTITY_FIELDS = frozenset(
    {"instrument", "candle_start_time", "candle_end_time", "readiness"}
)
"""Snapshot fields the CSV export replaces with its own columns."""

_TRADING_DATE = date(2026, 9, 11)

_SESSION_OPEN = datetime(2026, 9, 11, 3, 45, tzinfo=UTC)
"""09:15 IST on Friday 11 September 2026."""

_FULL_SESSION = 60
"""Candles enough for every feature, the fifty-period EMA included."""

_ONE = Decimal("1")


def _ohlcv(minutes: int) -> OHLCVCandle:
    """One broker candle ``minutes`` into the session, timestamped at its open."""
    close = Decimal(100 + minutes % 7)
    return OHLCVCandle(
        timestamp=_SESSION_OPEN + timedelta(minutes=minutes),
        open=close,
        high=close + _ONE,
        low=close - _ONE,
        close=close,
        volume=1_000 + minutes,
    )


def _session(count: int) -> tuple[OHLCVCandle, ...]:
    """A contiguous run of candles from the session open."""
    return tuple(_ohlcv(minute) for minute in range(count))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run(
    broker: Mock,
    candles: tuple[OHLCVCandle, ...],
    argv: Sequence[str] = (),
) -> int:
    with (
        patch("ai_trader.cli.check_features.load_groww_settings"),
        patch(
            "ai_trader.cli.check_features.GrowwBroker.authenticate",
            return_value=broker,
        ),
        patch(
            "ai_trader.cli.check_features._find_recent_completed_session",
            return_value=(_TRADING_DATE, candles),
        ),
    ):
        return main(argv)


_LIVE_OPEN = datetime(2026, 9, 14, 3, 45, tzinfo=UTC)
"""09:15 IST on Monday 14 September 2026: the session after the backfill.

Live candles have to start after every backfilled one, or the engine rejects
them as out of order and the test would be measuring the guard instead of the
live path. Using the next trading day is how live mode actually runs.
"""

_LIVE_MINUTES = 3
"""The shortest run that closes exactly one live candle.

The builder discards each instrument's first partial minute and only closes a
candle when the following minute's first tick arrives, so three ticks on three
consecutive minutes yield one candle: minute 0 discarded, minute 1 closed by the
minute-2 tick, minute 2 left open and never flushed.
"""


def _quote(volume: int) -> MarketQuote:
    return MarketQuote(
        instrument=_RELIANCE,
        last_price=Decimal("101"),
        last_trade_at=_LIVE_OPEN,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        previous_close=Decimal("100"),
        volume=volume,
        day_change=Decimal("1"),
        day_change_percent=Decimal("1"),
    )


def _ticks(
    open_time: datetime,
    totals: Sequence[int | None] = (None,) * _LIVE_MINUTES,
) -> tuple[MarketTick, ...]:
    """One tick per minute from ``open_time``, carrying ``totals`` if given.

    ``None`` is how Groww's stream really delivers a tick: the volume field is
    advertised but never populated, which is what the poller exists to fix.
    """
    return tuple(
        MarketTick(
            instrument=_RELIANCE,
            timestamp=open_time + timedelta(minutes=minute),
            price=Decimal(150 + minute),
            cumulative_volume=total,
        )
        for minute, total in enumerate(totals)
    )


def _collect(ticks: tuple[MarketTick, ...]) -> Callable[..., tuple[MarketTick, ...]]:
    def collect(
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None],
    ) -> tuple[MarketTick, ...]:
        assert max_ticks > 0
        assert timeout_seconds > 0
        for tick in ticks:
            on_tick(tick)
        return ticks

    return collect


def _streaming_broker(
    ticks: tuple[MarketTick, ...],
    *,
    quote_volume: int = 900_000,
) -> Mock:
    stream = Mock()
    stream.collect.side_effect = _collect(ticks)
    broker = Mock()
    broker.create_ltp_stream.return_value = stream
    # Live mode polls the quote endpoint for the session total, so the mock has
    # to answer with a real quote rather than another Mock.
    broker.get_quote.return_value = _quote(quote_volume)
    return broker


def _run_live(
    broker: Mock,
    candles: tuple[OHLCVCandle, ...],
    argv: Sequence[str] = (),
) -> int:
    return _run(broker, candles, ("--live", "--live-seconds", "1", *argv))


def test_a_completed_session_reports_its_latest_snapshot(
    capsys: CaptureFixture[str],
) -> None:
    exit_code = _run(Mock(), _session(_FULL_SESSION))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""

    summary = json.loads(captured.out)
    assert summary["trading_date"] == "2026-09-11"
    assert summary["backfilled_candles"] == _FULL_SESSION
    assert summary["feature_candles"] == _FULL_SESSION
    assert summary["duplicate_candles"] == 0
    assert summary["out_of_order_candles"] == 0

    latest = summary["latest"]
    assert latest["instrument"] == "RELIANCE"
    # The sixtieth minute after a 09:15 IST open.
    assert latest["candle_start_time"] == "2026-09-11T04:44:00+00:00"
    assert latest["candle_end_time"] == "2026-09-11T04:45:00+00:00"
    assert latest["close"] == "103"
    assert latest["volume"] == 1_059
    assert latest["core_ready"] is True


def test_only_the_latest_candle_is_printed(capsys: CaptureFixture[str]) -> None:
    _run(Mock(), _session(_FULL_SESSION))

    captured = capsys.readouterr()

    # Sixty candles go in and one snapshot comes out: this check is meant to be
    # read by a person, not to dump a session.
    assert captured.out.count('"candle_start_time"') == 1


def test_the_feature_layer_never_calls_the_broker() -> None:
    broker = Mock()

    exit_code = _run(broker, _session(_FULL_SESSION))

    assert exit_code == 0
    # The session fetch itself is mocked out, so any call recorded here would
    # mean a feature reached for the network instead of using its candles.
    assert broker.method_calls == []


def test_the_snapshot_reports_every_documented_field(
    capsys: CaptureFixture[str],
) -> None:
    _run(Mock(), _session(_FULL_SESSION))

    latest = json.loads(capsys.readouterr().out)["latest"]

    # Derived from ``FeatureSnapshot`` rather than from the CLI's own column
    # list, so adding a feature and forgetting to print it fails here. Checking
    # the summary against the same constant it is built from could not.
    assert set(latest) == {field.name for field in fields(FeatureSnapshot)} | {
        "core_ready"
    }
    assert set(latest["readiness"]) == {
        field.name for field in fields(FeatureReadiness)
    }
    assert all(latest["readiness"].values())


def test_the_export_carries_every_snapshot_field(tmp_path: Path) -> None:
    export = tmp_path / "features.csv"

    _run(Mock(), _session(_FULL_SESSION), ("--export-csv", str(export)))

    with export.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    # The export exists to be diffed against a trusted implementation, so a
    # feature missing from it is a silent hole in that comparison.
    expected = {field.name for field in fields(FeatureSnapshot)} - _IDENTITY_FIELDS
    assert set(header) == expected | {"start_time", "end_time"}


def test_derived_values_are_rounded_for_display(capsys: CaptureFixture[str]) -> None:
    _run(Mock(), _session(_FULL_SESSION))

    latest = json.loads(capsys.readouterr().out)["latest"]

    # Prices are printed as the broker gave them...
    assert latest["close"] == "103"
    # ...while everything derived is quantized to six decimal places.
    assert latest["rolling_high_20"] == "107.000000"
    assert latest["rolling_low_20"] == "99.000000"
    # 103 / 107 - 1 is -0.0373831..., which rounds half-even to six places.
    assert latest["distance_from_high_20"] == "-0.037383"


def test_a_short_session_still_reports_what_it_has(
    capsys: CaptureFixture[str],
) -> None:
    exit_code = _run(Mock(), _session(3))

    captured = capsys.readouterr()
    assert exit_code == 0

    latest = json.loads(captured.out)["latest"]
    # Three candles are enough for a one-candle return but nowhere near enough
    # for a fifty-period EMA; the check must still succeed and say which.
    assert latest["return_1"] is not None
    assert latest["ema50"] is None
    assert latest["readiness"]["ema50"] is False
    assert latest["core_ready"] is False


def test_live_ticks_become_features_on_the_same_engine(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    exit_code = _run_live(broker, _session(_FULL_SESSION))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""

    summary = json.loads(captured.out)
    live = summary["live"]
    assert live["live_ticks"] == _LIVE_MINUTES
    # Three ticks, one candle: the first minute is discarded and the last is
    # left open rather than flushed into a partial candle.
    assert live["live_candles"] == 1
    assert live["late_ticks"] == 0
    assert summary["backfilled_candles"] == _FULL_SESSION
    assert summary["feature_candles"] == _FULL_SESSION
    assert summary["duplicate_candles"] == 0
    assert summary["out_of_order_candles"] == 0

    # The whole point of the flag: the reported snapshot describes a candle
    # this process built from ticks, not one the broker handed over.
    assert summary["latest"]["candle_start_time"] == "2026-09-14T03:46:00+00:00"


def test_live_ticks_are_stamped_with_the_polled_session_volume(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN), quote_volume=987_654)

    _run_live(broker, _session(_FULL_SESSION))

    live = json.loads(capsys.readouterr().out)["live"]
    # Groww's stream carries no volume, so every one of these ticks arrived bare
    # and left the stamper carrying the polled total.
    assert live["stamped_ticks"] == _LIVE_MINUTES
    assert live["stale_stamps"] == 0
    assert live["session_volume"] == 987_654
    assert live["poll_failures"] == 0
    assert live["poll_error"] is None


def test_a_frozen_session_total_reports_no_volume_rather_than_guessing(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    _run_live(broker, _session(_FULL_SESSION))

    latest = json.loads(capsys.readouterr().out)["latest"]
    # The mock serves one unchanging total, so differencing it across the minute
    # boundary honestly reports that nothing traded. Zero here is a measurement,
    # not a fallback: it is what a real instrument that stopped printing looks
    # like, and the stale-reading bound is what keeps a stalled poller from
    # reaching this same number.
    assert latest["volume"] == 0
    assert latest["readiness"]["session_volume"] is True


def test_a_live_candle_carries_the_differenced_minute_volume(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN, (1_000_000, 1_000_400, 1_000_900)))

    _run_live(broker, _session(_FULL_SESSION))

    summary = json.loads(capsys.readouterr().out)
    # A tick carrying its own total is left alone by the stamper, and the closed
    # minute's volume is the difference across its own boundary: 1_000_400 less
    # the 1_000_000 that closed the minute before it.
    assert summary["live"]["stamped_ticks"] == _LIVE_MINUTES
    assert summary["latest"]["volume"] == 400


def test_out_of_order_live_candles_are_counted_rather_than_folded(
    capsys: CaptureFixture[str],
) -> None:
    # Ticks timestamped inside the session already backfilled. The candle
    # builder knows nothing of that history, so it emits a candle the engine
    # then has to refuse.
    broker = _streaming_broker(_ticks(_SESSION_OPEN + timedelta(minutes=10)))

    exit_code = _run_live(broker, _session(_FULL_SESSION))

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["live"]["live_candles"] == 1
    # Reached the engine and was rejected: before live mode existed this counter
    # was structurally pinned at zero, because backfill alone cannot produce a
    # candle that goes backwards.
    assert summary["out_of_order_candles"] == 1
    assert summary["duplicate_candles"] == 0
    # History is intact: the rejected candle did not rewind the engine.
    assert summary["latest"]["candle_start_time"] == "2026-09-11T04:44:00+00:00"


def test_the_stream_is_untouched_unless_live_is_asked_for() -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    exit_code = _run(broker, _session(_FULL_SESSION))

    assert exit_code == 0
    # Live mode reaches the network on every tick and polls a quote every two
    # seconds. A default run must do neither, so the backfill check stays usable
    # outside market hours and cannot be slowed down by a stream nobody wanted.
    broker.create_ltp_stream.assert_not_called()
    broker.get_quote.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [("--live-seconds", "0"), ("--live-ticks", "0"), ("--live-seconds", "-1")],
)
def test_a_non_positive_live_window_exits_two(
    option: str,
    value: str,
    capsys: CaptureFixture[str],
) -> None:
    # Both limits stop ``collect`` at whichever is reached first, so a zero
    # would end the window before a single tick and report an empty live run as
    # though the market had been quiet.
    with pytest.raises(SystemExit) as error:
        main(("--live", option, value))

    assert error.value.code == 2
    assert "must be positive" in capsys.readouterr().err


def test_the_export_includes_live_candles(tmp_path: Path) -> None:
    export = tmp_path / "features.csv"
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    exit_code = _run_live(
        broker,
        _session(_FULL_SESSION),
        ("--export-csv", str(export)),
    )

    rows = _read_csv(export)
    assert exit_code == 0
    # The export exists to be diffed offline, so a live run that dropped its own
    # candles from it would make the live path the one thing unverifiable.
    assert len(rows) == _FULL_SESSION + 1
    assert rows[-1]["start_time"] == "2026-09-14T03:46:00+00:00"


def test_the_export_writes_one_row_per_candle(tmp_path: Path) -> None:
    export = tmp_path / "features.csv"

    exit_code = _run(Mock(), _session(_FULL_SESSION), ("--export-csv", str(export)))

    rows = _read_csv(export)
    assert exit_code == 0
    assert len(rows) == _FULL_SESSION
    assert rows[0]["start_time"] == "2026-09-11T03:45:00+00:00"
    assert rows[-1]["close"] == "103"
    # The first candle has nothing behind it, so its unavailable features are
    # blank rather than zero.
    assert rows[0]["return_1"] == ""
    assert rows[0]["ema50"] == ""


def test_the_export_keeps_more_precision_than_the_summary(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    export = tmp_path / "features.csv"

    _run(Mock(), _session(_FULL_SESSION), ("--export-csv", str(export)))

    latest = json.loads(capsys.readouterr().out)["latest"]
    exported = _read_csv(export)[-1]["ema50"]

    # The fifty-period EMA seeds on an exact 5147 / 50 and then smooths by
    # 2 / 51, which does not terminate. The summary rounds that for reading;
    # the export must not, because it exists to be diffed against a trusted
    # implementation.
    assert len(latest["ema50"].split(".")[1]) == 6
    assert len(exported.split(".")[1]) > 6


def test_an_unwritable_export_path_exits_one(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    export = tmp_path / "missing" / "features.csv"

    exit_code = _run(Mock(), _session(_FULL_SESSION), ("--export-csv", str(export)))

    captured = capsys.readouterr()
    assert exit_code == 1
    # The export is written before the summary, so a failed write reports the
    # failure instead of a snapshot nobody asked for.
    assert captured.out == ""
    assert "Could not write" in captured.err


def test_an_existing_export_is_refused_rather_than_replaced(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    export = tmp_path / "features.csv"
    export.write_text("reference data", encoding="utf-8")

    exit_code = _run(Mock(), _session(_FULL_SESSION), ("--export-csv", str(export)))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "pass --overwrite" in captured.err
    # The file this export would have clobbered is the trusted comparison it
    # exists to be diffed against, so leaving it untouched is the whole point.
    assert export.read_text(encoding="utf-8") == "reference data"


def test_overwrite_replaces_an_existing_export(tmp_path: Path) -> None:
    export = tmp_path / "features.csv"
    export.write_text("stale data", encoding="utf-8")

    exit_code = _run(
        Mock(),
        _session(_FULL_SESSION),
        ("--export-csv", str(export), "--overwrite"),
    )

    assert exit_code == 0
    assert len(_read_csv(export)) == _FULL_SESSION


def test_an_unknown_option_exits_two(capsys: CaptureFixture[str]) -> None:
    # argparse exits 2 on a usage error, which collides with the exit code this
    # CLI uses for a configuration error. Both mean "you gave me bad input", so
    # the collision is tolerable, but it is asserted here so it stays a choice.
    with pytest.raises(SystemExit) as error:
        main(("--nonsense",))

    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_a_session_with_no_candles_exits_one(capsys: CaptureFixture[str]) -> None:
    exit_code = _run(Mock(), ())

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == "Market state produced no snapshot."


def test_missing_configuration_exits_two(capsys: CaptureFixture[str]) -> None:
    with patch(
        "ai_trader.cli.check_features.load_groww_settings",
        side_effect=ConfigurationError("GROWW_API_KEY is missing."),
    ):
        exit_code = main(())

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.strip() == "GROWW_API_KEY is missing."


@pytest.mark.parametrize(
    ("error", "expected_stderr"),
    [
        (SessionNotFoundError("No completed NSE trading session"), "No completed NSE"),
        (GrowwBrokerError("boom"), "Groww feature check failed."),
        (
            GrowwStreamConnectionError("boom"),
            "Groww live feed unreachable",
        ),
    ],
)
def test_broker_failures_exit_one_without_leaking_details(
    error: Exception,
    expected_stderr: str,
    capsys: CaptureFixture[str],
) -> None:
    with (
        patch("ai_trader.cli.check_features.load_groww_settings"),
        patch(
            "ai_trader.cli.check_features.GrowwBroker.authenticate",
            side_effect=error,
        ),
    ):
        exit_code = main(())

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert expected_stderr in captured.err
    # The name of this test is a contract: broker text must not reach stderr.
    if isinstance(error, GrowwBrokerError):
        assert "boom" not in captured.err
