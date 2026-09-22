"""Check Groww historical market data using read-only broker operations."""

import json
import sys
from datetime import date, datetime, time
from decimal import Decimal

from ai_trader.broker import OHLCVCandle, ReadOnlyBroker
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBroker,
    GrowwBrokerError,
)
from ai_trader.cli._session import (
    INDIA_TIMEZONE,
    RELIANCE,
    SESSION_START,
    SessionNotFoundError,
    find_recent_completed_session,
)
from ai_trader.config import ConfigurationError, load_groww_settings

_INDIA_TIMEZONE = INDIA_TIMEZONE
_RELIANCE = RELIANCE
_SESSION_START = SESSION_START
_SESSION_END = time(hour=9, minute=30)
"""Only the opening fifteen minutes: this check proves the endpoint answers,
not that a whole session can be read, and a short window lets it run shortly
after the open rather than only after the close."""


def _price(value: Decimal) -> str:
    return format(value, "f")


def _candle_summary(candle: OHLCVCandle | None) -> dict[str, object] | None:
    if candle is None:
        return None
    return {
        "timestamp": candle.timestamp.isoformat(),
        "open": _price(candle.open),
        "high": _price(candle.high),
        "low": _price(candle.low),
        "close": _price(candle.close),
        "volume": candle.volume,
    }


def _find_recent_completed_session(
    broker: ReadOnlyBroker,
    now: datetime,
) -> tuple[date, tuple[OHLCVCandle, ...]]:
    """Find a recent session, reading only this check's opening window."""
    return find_recent_completed_session(
        broker,
        now,
        instrument=_RELIANCE,
        session_start=_SESSION_START,
        session_end=_SESSION_END,
    )


def main() -> int:
    """Find and summarize a recent completed NSE trading session."""
    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        broker = GrowwBroker.authenticate(settings)
        trading_date, candles = _find_recent_completed_session(
            broker,
            now=datetime.now(tz=_INDIA_TIMEZONE),
        )
    except SessionNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except GrowwAuthenticationError:
        print("Groww authentication failed.", file=sys.stderr)
        return 1
    except GrowwBrokerError:
        print("Groww historical data check failed.", file=sys.stderr)
        return 1

    summary = {
        "trading_date": trading_date.isoformat(),
        "candle_count": len(candles),
        "first_candle": _candle_summary(candles[0] if candles else None),
        "last_candle": _candle_summary(candles[-1] if candles else None),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
