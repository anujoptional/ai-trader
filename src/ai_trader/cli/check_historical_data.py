"""Check Groww historical market data using read-only broker operations."""

import json
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle
from ai_trader.broker.groww import GrowwBroker, GrowwBrokerError
from ai_trader.config import ConfigurationError, load_groww_settings

_INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_SESSION_START = time(hour=9, minute=15)
_SESSION_END = time(hour=9, minute=30)
_MAX_WEEKDAYS = 10


class HistoricalSessionNotFoundError(RuntimeError):
    """Raised when no completed trading session can be found for validation."""


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
    broker: GrowwBroker,
    now: datetime,
) -> tuple[date, tuple[OHLCVCandle, ...]]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("The current time must be timezone-aware.")

    local_now = now.astimezone(_INDIA_TIMEZONE)
    candidate = local_now.date()
    completed_today = datetime.combine(
        candidate,
        _SESSION_END,
        tzinfo=_INDIA_TIMEZONE,
    )
    if local_now < completed_today:
        candidate -= timedelta(days=1)

    weekdays_checked = 0
    while weekdays_checked < _MAX_WEEKDAYS:
        if candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
            continue

        weekdays_checked += 1
        start = datetime.combine(
            candidate,
            _SESSION_START,
            tzinfo=_INDIA_TIMEZONE,
        )
        end = datetime.combine(
            candidate,
            _SESSION_END,
            tzinfo=_INDIA_TIMEZONE,
        )
        candles = broker.get_historical_candles(
            instrument=_RELIANCE,
            start=start,
            end=end,
            interval=CandleInterval.ONE_MINUTE,
        )
        if candles:
            return candidate, candles

        candidate -= timedelta(days=1)

    raise HistoricalSessionNotFoundError(
        "No completed NSE trading session was found in the last 10 weekdays."
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
    except HistoricalSessionNotFoundError as error:
        print(str(error), file=sys.stderr)
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
