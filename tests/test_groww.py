from unittest.mock import call, patch

from ai_trader.broker.groww import GrowwBroker
from ai_trader.config import GrowwSettings


def test_authenticate_and_get_sanitized_profile_with_mocked_sdk() -> None:
    settings = GrowwSettings(
        totp_token="test-token-value",
        totp_secret="test-secret-value",
    )

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
