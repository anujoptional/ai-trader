"""Read-only Groww authentication and profile access."""

from contextlib import redirect_stdout
from io import StringIO
from types import MappingProxyType
from typing import Any, Protocol, Self

import pyotp
from growwapi import GrowwAPI
from pydantic import BaseModel, ConfigDict

from ai_trader.broker import BrokerProfile
from ai_trader.config import GrowwSettings


class GrowwBrokerError(RuntimeError):
    """Base error for safe Groww broker failures."""


class GrowwAuthenticationError(GrowwBrokerError):
    """Raised when Groww authentication cannot be completed."""


class GrowwProfileError(GrowwBrokerError):
    """Raised when the Groww profile cannot be retrieved or validated."""


class _ProfileClient(Protocol):
    def get_user_profile(self) -> dict[str, Any]:
        """Return the raw Groww profile payload."""
        ...


class _GrowwProfilePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nse_enabled: bool
    bse_enabled: bool
    active_segments: tuple[str, ...]
    ddpi_enabled: bool


class GrowwBroker:
    """A Groww adapter limited to retrieving the user profile."""

    def __init__(self, client: _ProfileClient) -> None:
        self._client = client

    @classmethod
    def authenticate(cls, settings: GrowwSettings) -> Self:
        """Authenticate with TOTP and construct a read-only Groww adapter."""
        try:
            totp = pyotp.TOTP(settings.totp_secret.get_secret_value()).now()
            access_token = GrowwAPI.get_access_token(
                api_key=settings.totp_token.get_secret_value(),
                totp=totp,
            )
            if not isinstance(access_token, str) or not access_token:
                raise TypeError

            # The SDK prints status text during construction. Suppress it so the
            # CLI emits only its explicitly allowlisted profile summary.
            with redirect_stdout(StringIO()):
                client = GrowwAPI(access_token)
        except Exception:
            raise GrowwAuthenticationError("Groww authentication failed.") from None

        return cls(client)

    def get_user_profile(self) -> BrokerProfile:
        """Retrieve and sanitize the Groww user profile."""
        try:
            payload = _GrowwProfilePayload.model_validate(
                self._client.get_user_profile()
            )
        except Exception:
            raise GrowwProfileError("Groww profile retrieval failed.") from None

        exchange_enablement = MappingProxyType(
            {
                "NSE": payload.nse_enabled,
                "BSE": payload.bse_enabled,
            }
        )
        return BrokerProfile(
            exchange_enablement=exchange_enablement,
            active_segments=payload.active_segments,
            ddpi_enabled=payload.ddpi_enabled,
        )


__all__ = [
    "GrowwAuthenticationError",
    "GrowwBroker",
    "GrowwBrokerError",
    "GrowwProfileError",
]
