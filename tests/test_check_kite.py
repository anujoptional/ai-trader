import json
from unittest.mock import Mock, patch

from pytest import CaptureFixture

from ai_trader.broker import BrokerProfile
from ai_trader.broker.kite import KiteAuthenticationError, KiteBrokerError
from ai_trader.cli.check_kite import main


def test_main_prints_only_allowlisted_profile_fields(
    capsys: CaptureFixture[str],
) -> None:
    profile = BrokerProfile(
        exchange_enablement={"NSE": True, "BSE": False},
        active_segments=("CASH", "FNO"),
        ddpi_enabled=True,
    )
    broker = Mock()
    broker.get_user_profile.return_value = profile

    with (
        patch("ai_trader.cli.check_kite.load_kite_login_settings"),
        patch(
            "ai_trader.cli.check_kite.KiteBroker.authenticate",
            return_value=broker,
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "active_segments": ["CASH", "FNO"],
        "ddpi_status": "enabled",
        "exchange_enablement": {"BSE": False, "NSE": True},
    }


def test_main_prints_generic_error_without_exception_details(
    capsys: CaptureFixture[str],
) -> None:
    with (
        patch("ai_trader.cli.check_kite.load_kite_login_settings"),
        patch(
            "ai_trader.cli.check_kite.KiteBroker.authenticate",
            side_effect=KiteAuthenticationError("test-sensitive-detail"),
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Kite authentication failed.\n"
    assert "test-sensitive-detail" not in captured.err


def test_main_reports_a_profile_failure_distinctly_from_an_auth_failure(
    capsys: CaptureFixture[str],
) -> None:
    # Authentication and profile retrieval fail for unrelated reasons, and
    # KiteAuthenticationError is a KiteBrokerError, so a single handler would
    # send an operator to the wrong layer.
    broker = Mock()
    broker.get_user_profile.side_effect = KiteBrokerError("test-sensitive-detail")
    with (
        patch("ai_trader.cli.check_kite.load_kite_login_settings"),
        patch(
            "ai_trader.cli.check_kite.KiteBroker.authenticate",
            return_value=broker,
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Kite profile check failed.\n"
    assert "test-sensitive-detail" not in captured.err
