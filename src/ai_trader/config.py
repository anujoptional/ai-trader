"""Application configuration loaded from environment variables."""

import os
from collections.abc import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


class GrowwSettings(BaseModel):
    """Credentials required for Groww's TOTP authentication flow."""

    model_config = ConfigDict(frozen=True)

    totp_token: SecretStr
    totp_secret: SecretStr


def load_groww_settings(
    environ: Mapping[str, str] | None = None,
) -> GrowwSettings:
    """Load Groww settings without including credential values in failures."""
    load_dotenv(override=False)
    source = os.environ if environ is None else environ

    variable_names = ("GROWW_TOTP_TOKEN", "GROWW_TOTP_SECRET")
    missing = [name for name in variable_names if not source.get(name, "").strip()]
    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(f"Missing required environment variables: {names}")

    return GrowwSettings(
        totp_token=source["GROWW_TOTP_TOKEN"],
        totp_secret=source["GROWW_TOTP_SECRET"],
    )
