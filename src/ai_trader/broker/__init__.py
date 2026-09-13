"""Broker-neutral interfaces and data structures."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BrokerProfile:
    """Non-sensitive broker capabilities safe to display."""

    exchange_enablement: Mapping[str, bool]
    active_segments: tuple[str, ...]
    ddpi_enabled: bool


class ReadOnlyBroker(Protocol):
    """The minimal broker behavior used by the profile check."""

    def get_user_profile(self) -> BrokerProfile:
        """Return a sanitized, non-sensitive broker profile."""
        ...


__all__ = ["BrokerProfile", "ReadOnlyBroker"]
