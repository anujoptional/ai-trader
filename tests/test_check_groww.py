import json
from unittest.mock import Mock, patch

from pytest import CaptureFixture

from ai_trader.broker import BrokerProfile
from ai_trader.broker.groww import GrowwAuthenticationError
from ai_trader.cli.check_groww import main


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
        patch("ai_trader.cli.check_groww.load_groww_settings"),
        patch(
            "ai_trader.cli.check_groww.GrowwBroker.authenticate",
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
        patch("ai_trader.cli.check_groww.load_groww_settings"),
        patch(
            "ai_trader.cli.check_groww.GrowwBroker.authenticate",
            side_effect=GrowwAuthenticationError("test-sensitive-detail"),
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Groww profile check failed.\n"
    assert "test-sensitive-detail" not in captured.err
