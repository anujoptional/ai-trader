from unittest.mock import patch

import pytest

from ai_trader.config import ConfigurationError, load_groww_settings


def test_load_groww_settings_uses_dotenv_and_redacts_repr() -> None:
    environment = {
        "GROWW_TOTP_TOKEN": "test-token-value",
        "GROWW_TOTP_SECRET": "test-secret-value",
    }

    with patch("ai_trader.config.load_dotenv") as load_dotenv:
        settings = load_groww_settings(environment)

    load_dotenv.assert_called_once_with(override=False)
    assert "test-token-value" not in repr(settings)
    assert "test-secret-value" not in repr(settings)


@pytest.mark.parametrize(
    ("environment", "missing_name"),
    [
        ({"GROWW_TOTP_SECRET": "test-secret-value"}, "GROWW_TOTP_TOKEN"),
        ({"GROWW_TOTP_TOKEN": "test-token-value"}, "GROWW_TOTP_SECRET"),
    ],
)
def test_load_groww_settings_reports_only_missing_variable_names(
    environment: dict[str, str],
    missing_name: str,
) -> None:
    with (
        patch("ai_trader.config.load_dotenv"),
        pytest.raises(ConfigurationError) as error,
    ):
        load_groww_settings(environment)

    assert missing_name in str(error.value)
    assert "test-token-value" not in str(error.value)
    assert "test-secret-value" not in str(error.value)
