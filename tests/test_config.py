from unittest.mock import patch

import pytest

from ai_trader.config import (
    ConfigurationError,
    load_kite_login_settings,
    load_kite_settings,
)


@pytest.mark.parametrize("login", [False, True])
def test_kite_settings_are_separate_and_redacted(login: bool) -> None:
    loader = load_kite_login_settings if login else load_kite_settings
    other_name = "KITE_API_SECRET" if login else "KITE_ACCESS_TOKEN"
    environment = {"KITE_API_KEY": "test-kite-key", other_name: "test-kite-secret"}
    with patch("ai_trader.config.load_dotenv") as dotenv:
        settings = loader(environment)
    dotenv.assert_not_called()
    assert "test-kite-key" not in repr(settings)
    assert "test-kite-secret" not in repr(settings)
    assert settings.api_key.get_secret_value() == environment["KITE_API_KEY"]
    with pytest.raises(ConfigurationError) as error:
        loader({"KITE_API_KEY": "test-kite-key"})
    assert other_name in str(error.value)
    assert "test-kite-key" not in str(error.value)


def test_kite_login_loads_environment_without_overriding_it() -> None:
    with (
        patch("ai_trader.config.load_dotenv") as dotenv,
        patch.dict(
            "os.environ", {"KITE_API_KEY": "fake-key", "KITE_API_SECRET": "fake-secret"}
        ),
    ):
        settings = load_kite_login_settings()
    dotenv.assert_called_once_with(override=False)
    assert settings.api_key.get_secret_value() == "fake-key"
    assert "fake-secret" not in repr(settings)
