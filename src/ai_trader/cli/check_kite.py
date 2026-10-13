"""Check read-only Kite authentication and display safe profile fields."""

import json
import sys

from ai_trader.broker import BrokerProfile
from ai_trader.broker.kite import (
    KiteAuthenticationError,
    KiteBroker,
    KiteBrokerError,
)
from ai_trader.config import ConfigurationError, load_kite_login_settings


def _profile_summary(profile: BrokerProfile) -> dict[str, object]:
    return {
        "exchange_enablement": dict(profile.exchange_enablement),
        "active_segments": list(profile.active_segments),
        "ddpi_status": (
            "unknown"
            if profile.ddpi_enabled is None
            else "enabled"
            if profile.ddpi_enabled
            else "disabled"
        ),
    }


def main() -> int:
    """Run the read-only Kite profile check."""
    try:
        settings = load_kite_login_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        profile = KiteBroker.authenticate(settings).get_user_profile()
    except KiteAuthenticationError:
        print("Kite authentication failed.", file=sys.stderr)
        return 1
    except KiteBrokerError:
        print("Kite profile check failed.", file=sys.stderr)
        return 1

    print(json.dumps(_profile_summary(profile), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
