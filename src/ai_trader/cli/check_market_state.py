"""Check the market-state component against Groww historical and live data.

Backfills a recent completed NSE session, then folds live ticks into the same
state object. Outside market hours no ticks arrive and the check still passes,
reporting the backfilled state.
"""

import json
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from ai_trader.broker import CandleInterval, Instrument, MarketTick, OHLCVCandle
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBroker,
    GrowwBrokerError,
)
from ai_trader.config import ConfigurationError, load_groww_settings
from ai_trader.market import Candle, MarketState

_INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_SESSION_START = time(hour=9, minute=15)
_SESSION_END = time(hour=15, minute=30)
_MAX_WEEKDAYS = 10
_MAX_TICKS = 25
_TIMEOUT_SECONDS = 15.0


class SessionNotFoundError(RuntimeError):
    """Raised when no completed trading session can be found for validation."""


def _price(value: Decimal) -> str:
    return format(value, "f")


def _candle_summary(candle: Candle | None) -> dict[str, object] | None:
    if candle is None:
        return None
    return {
        "start_time": candle.start_time.isoformat(),
        "end_time": candle.end_time.isoformat(),
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
    local_now = now.astimezone(_INDIA_TIMEZONE)
    candidate = local_now.date()
    if local_now < datetime.combine(candidate, _SESSION_END, tzinfo=_INDIA_TIMEZONE):
        candidate -= timedelta(days=1)

    weekdays_checked = 0
    while weekdays_checked < _MAX_WEEKDAYS:
        if candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
            continue

        weekdays_checked += 1
        candles = broker.get_historical_candles(
            instrument=_RELIANCE,
            start=datetime.combine(candidate, _SESSION_START, tzinfo=_INDIA_TIMEZONE),
            end=datetime.combine(candidate, _SESSION_END, tzinfo=_INDIA_TIMEZONE),
            interval=CandleInterval.ONE_MINUTE,
        )
        if candles:
            return candidate, candles

        candidate -= timedelta(days=1)

    raise SessionNotFoundError(
        "No completed NSE trading session was found in the last 10 weekdays."
    )


def main() -> int:
    """Backfill and stream one instrument through the market-state component."""
    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    state = MarketState()
    tick_count = 0

    def record(tick: MarketTick) -> None:
        nonlocal tick_count
        tick_count += 1
        state.record_tick(tick)

    try:
        broker = GrowwBroker.authenticate(settings)
        trading_date, historical = _find_recent_completed_session(
            broker,
            now=datetime.now(tz=_INDIA_TIMEZONE),
        )
        backfilled = state.backfill(_RELIANCE, historical)
        stream = broker.create_ltp_stream((_RELIANCE,))
        stream.collect(
            max_ticks=_MAX_TICKS,
            timeout_seconds=_TIMEOUT_SECONDS,
            on_tick=record,
        )
    except SessionNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except GrowwAuthenticationError:
        print("Groww authentication failed.", file=sys.stderr)
        return 1
    except GrowwBrokerError:
        print("Groww market-state check failed.", file=sys.stderr)
        return 1

    state.flush()
    snapshot = state.snapshot(_RELIANCE)
    if snapshot is None:
        print("Market state produced no snapshot.", file=sys.stderr)
        return 1

    last_price = snapshot.last_price
    last_tick_at = snapshot.last_tick_at
    summary = {
        "backfill_trading_date": trading_date.isoformat(),
        "backfilled_candles": backfilled,
        "live_ticks": tick_count,
        "retained_candles": len(snapshot.candles),
        "late_ticks": state.late_tick_count,
        "duplicate_candles": state.duplicate_candle_count,
        "last_price": None if last_price is None else _price(last_price),
        "last_tick_at": None if last_tick_at is None else last_tick_at.isoformat(),
        "first_candle": _candle_summary(
            snapshot.candles[0] if snapshot.candles else None
        ),
        "last_candle": _candle_summary(snapshot.latest_candle),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
