from unittest.mock import call, patch

import pytest

from ai_trader.broker import groww as groww_module
from ai_trader.broker.groww import GrowwAuthenticationError, GrowwBroker
from ai_trader.config import GrowwSettings


def _settings() -> GrowwSettings:
    return GrowwSettings(
        totp_token="test-token-value",
        totp_secret="test-secret-value",
    )


def test_authenticate_and_get_sanitized_profile_with_mocked_sdk() -> None:
    settings = _settings()

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.return_value = "654321"
        api_class.get_access_token.return_value = "test-access-token"
        client = api_class.return_value
        client.get_user_profile.return_value = {
            "vendor_user_id": "sensitive-vendor-id",
            "ucc": "sensitive-ucc",
            "nse_enabled": True,
            "bse_enabled": False,
            "ddpi_enabled": True,
            "active_segments": ["CASH", "FNO"],
        }

        profile = GrowwBroker.authenticate(settings).get_user_profile()

    totp_class.assert_called_once_with("test-secret-value")
    api_class.get_access_token.assert_called_once_with(
        api_key="test-token-value",
        totp="654321",
    )
    api_class.assert_called_once_with("test-access-token")
    assert client.method_calls == [call.get_user_profile()]
    assert dict(profile.exchange_enablement) == {"NSE": True, "BSE": False}
    assert profile.active_segments == ("CASH", "FNO")
    assert profile.ddpi_enabled is True


def test_authentication_retries_transient_broker_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The intermittent Groww 404 that spoils reads hits the token endpoint too:
    # 4 of 8 raw token requests failed when measured live on 2026-09-21. Without
    # a retry every CLI has a coin-flip chance of dying at startup.
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.return_value = "654321"
        api_class.get_access_token.side_effect = [
            ValueError("Extra data: line 1 column 5 (char 4)"),
            ValueError("Extra data: line 1 column 5 (char 4)"),
            "test-access-token",
        ]

        broker = GrowwBroker.authenticate(_settings())

    assert isinstance(broker, GrowwBroker)
    assert api_class.get_access_token.call_count == 3


def test_authentication_regenerates_the_totp_for_every_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A retry can cross a 30-second TOTP window, so the code has to be
    # generated inside the retried call rather than once before it.
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.side_effect = ["111111", "222222", "333333"]
        api_class.get_access_token.side_effect = [
            ValueError("Extra data: line 1 column 5 (char 4)"),
            ValueError("Extra data: line 1 column 5 (char 4)"),
            "test-access-token",
        ]

        GrowwBroker.authenticate(_settings())

    assert [
        keywords["totp"]
        for _args, keywords in api_class.get_access_token.call_args_list
    ] == ["111111", "222222", "333333"]


def test_authentication_retries_the_client_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.return_value = "654321"
        api_class.get_access_token.return_value = "test-access-token"
        api_class.side_effect = [
            ValueError("Extra data: line 1 column 5 (char 4)"),
            api_class.return_value,
        ]

        broker = GrowwBroker.authenticate(_settings())

    assert isinstance(broker, GrowwBroker)
    assert api_class.call_count == 2


def test_persistent_authentication_failures_surface_after_the_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.return_value = "654321"
        api_class.get_access_token.side_effect = ValueError(
            "Extra data: line 1 column 5 (char 4)"
        )

        with pytest.raises(GrowwAuthenticationError, match="authentication failed"):
            GrowwBroker.authenticate(_settings())

    assert api_class.get_access_token.call_count == groww_module._CALL_ATTEMPTS


def test_an_unusable_access_token_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The schema check sits outside the retry, matching every read path: a
    # genuine contract change must surface at once rather than be hammered.
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.return_value = "654321"
        api_class.get_access_token.return_value = {"access_token": "wrapped"}

        with pytest.raises(GrowwAuthenticationError, match="authentication failed"):
            GrowwBroker.authenticate(_settings())

    api_class.get_access_token.assert_called_once()
    api_class.assert_not_called()
