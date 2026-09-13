"""Check read-only Groww authentication and display safe profile fields."""

import json
import sys

from ai_trader.broker import BrokerProfile
from ai_trader.broker.groww import GrowwBroker, GrowwBrokerError
from ai_trader.config import ConfigurationError, load_groww_settings


def _profile_summary(profile: BrokerProfile) -> dict[str, object]:
    return {
        "exchange_enablement": dict(profile.exchange_enablement),
        "active_segments": list(profile.active_segments),
        "ddpi_status": "enabled" if profile.ddpi_enabled else "disabled",
    }


def main() -> int:
    """Run the read-only Groww profile check."""
    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        profile = GrowwBroker.authenticate(settings).get_user_profile()
    except GrowwBrokerError:
        print("Groww profile check failed.", file=sys.stderr)
        return 1

    print(json.dumps(_profile_summary(profile), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
