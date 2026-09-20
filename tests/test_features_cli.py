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
from collections.abc import Sequence
from dataclasses import fields
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytest import CaptureFixture

from ai_trader.broker import OHLCVCandle
from ai_trader.broker.groww import GrowwBrokerError
from ai_trader.cli.check_features import SessionNotFoundError, main
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
