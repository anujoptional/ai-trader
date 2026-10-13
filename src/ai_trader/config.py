"""Application configuration loaded from environment variables."""

import os
from collections.abc import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


class KiteSettings(BaseModel):
    """An existing Kite Connect session, never an order-placement capability."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    access_token: SecretStr


class KiteLoginSettings(BaseModel):
    """Credentials for Kite's user-authorized browser login."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    api_secret: SecretStr


def load_kite_settings(
    environ: Mapping[str, str] | None = None,
) -> KiteSettings:
    source = _kite_environment(environ, ("KITE_API_KEY", "KITE_ACCESS_TOKEN"))
    return KiteSettings(
        api_key=source["KITE_API_KEY"], access_token=source["KITE_ACCESS_TOKEN"]
    )


def load_kite_login_settings(
    environ: Mapping[str, str] | None = None,
) -> KiteLoginSettings:
    source = _kite_environment(environ, ("KITE_API_KEY", "KITE_API_SECRET"))
    return KiteLoginSettings(
        api_key=source["KITE_API_KEY"], api_secret=source["KITE_API_SECRET"]
    )


def _kite_environment(
    environ: Mapping[str, str] | None, variable_names: tuple[str, ...]
) -> Mapping[str, str]:
    if environ is None:
        load_dotenv(override=False)
    source = os.environ if environ is None else environ
    missing = [name for name in variable_names if not source.get(name, "").strip()]
    if missing:
        names = ", ".join(missing)
        raise ConfigurationError(f"Missing required environment variables: {names}")
    return source


class OpenAISettings(BaseModel):
    """OpenAI access for the decision layer; the key never leaves this object."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    model: str = "gpt-6-luna"
    reasoning_effort: str = "low"
    timeout_seconds: float = 45.0
    daily_budget_usd: float = 5.0
    web_search: bool = False


# The API rejects unknown values; each model supports a subset.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def load_openai_settings(environ: Mapping[str, str] | None = None) -> OpenAISettings:
    source = _kite_environment(environ, ("OPENAI_API_KEY",))
    effort = source.get("OPENAI_REASONING_EFFORT", "").strip().lower() or "low"
    if effort not in REASONING_EFFORTS:
        raise ConfigurationError(
            "OPENAI_REASONING_EFFORT must be one of " + ", ".join(REASONING_EFFORTS)
        )
    try:
        return OpenAISettings(
            api_key=source["OPENAI_API_KEY"],
            model=source.get("OPENAI_MODEL", "").strip() or "gpt-6-luna",
            reasoning_effort=effort,
            timeout_seconds=float(source.get("OPENAI_TIMEOUT_SECONDS") or 45),
            daily_budget_usd=float(source.get("AI_DAILY_BUDGET_USD") or 5),
            web_search=source.get("OPENAI_WEB_SEARCH", "").strip().lower()
            in ("1", "true", "yes"),
        )
    except ValueError as error:
        raise ConfigurationError("Invalid OpenAI numeric setting") from error


class AnthropicSettings(BaseModel):
    """Claude access for a comparison AI; the daily budget is the shared one."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    timeout_seconds: float = 90.0
    daily_budget_usd: float = 5.0


def load_anthropic_settings(
    environ: Mapping[str, str] | None = None,
) -> AnthropicSettings:
    source = _kite_environment(environ, ("ANTHROPIC_API_KEY",))
    try:
        return AnthropicSettings(
            api_key=source["ANTHROPIC_API_KEY"],
            timeout_seconds=float(source.get("ANTHROPIC_TIMEOUT_SECONDS") or 90),
            daily_budget_usd=float(source.get("AI_DAILY_BUDGET_USD") or 5),
        )
    except ValueError as error:
        raise ConfigurationError("Invalid Anthropic numeric setting") from error
