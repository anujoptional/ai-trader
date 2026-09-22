"""Check the market-state component against Groww historical and live data.

Backfills a recent completed NSE session, then folds live ticks into the same
state object. Outside market hours no ticks arrive and the check still passes,
reporting the backfilled state.

Groww's tick stream carries no volume, so a volume poller runs alongside the
stream and stamps each tick with the running session total from the REST quote
endpoint. That total is the only live volume the broker serves, and stamping is
what lets the candle builder difference it into a per-minute figure.

The live window here is deliberately shorter than a minute, so the check
normally finishes before a live candle closes. It proves that ticks are being
stamped rather than that a stamped candle was emitted, which is what
``stamped_ticks`` in the summary reports.
"""

import json
import sys
from datetime import date, datetime
from decimal import Decimal

from ai_trader.broker import MarketTick, OHLCVCandle, ReadOnlyBroker
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBroker,
    GrowwBrokerError,
    GrowwStreamConnectionError,
)
from ai_trader.cli._session import (
    INDIA_TIMEZONE,
    RELIANCE,
    SessionNotFoundError,
    find_recent_completed_session,
)
from ai_trader.config import ConfigurationError, load_groww_settings
from ai_trader.market import Candle, MarketState, VolumePoller

_INDIA_TIMEZONE = INDIA_TIMEZONE
_RELIANCE = RELIANCE
_MAX_TICKS = 25
_TIMEOUT_SECONDS = 15.0


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
    broker: ReadOnlyBroker,
    now: datetime,
) -> tuple[date, tuple[OHLCVCandle, ...]]:
    """Find a recent session, reading the full NSE window for backfill."""
    return find_recent_completed_session(broker, now, instrument=_RELIANCE)


def main() -> int:
    """Backfill and stream one instrument through the market-state component."""
    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    tick_count = 0
    stamped_count = 0

    try:
        broker = GrowwBroker.authenticate(settings)
        poller = VolumePoller(broker, (_RELIANCE,))

        def stamp(tick: MarketTick) -> MarketTick:
            # Wrapping the poller's own stamper rather than stamping at the
            # call site keeps the counting here while the state object owns
            # the seam, so no live path can forget to stamp.
            nonlocal stamped_count
            stamped = poller.stamp(tick)
            if stamped.cumulative_volume is not None:
                stamped_count += 1
            return stamped

        def record(tick: MarketTick) -> None:
            nonlocal tick_count
            tick_count += 1
            state.record_tick(tick)

        state = MarketState(tick_stamper=stamp)
        trading_date, historical = _find_recent_completed_session(
            broker,
            now=datetime.now(tz=_INDIA_TIMEZONE),
        )
        backfilled = state.backfill(_RELIANCE, historical)
        with poller:
            stream = broker.create_ltp_stream((_RELIANCE,))
            stream.collect(
                max_ticks=_MAX_TICKS,
                timeout_seconds=_TIMEOUT_SECONDS,
                on_tick=record,
            )
        volume_source = {
            "stamped_ticks": stamped_count,
            "stale_stamps": poller.stale_stamp_count,
            "session_volume": poller.latest(_RELIANCE),
            "polls": poller.poll_count,
            "poll_failures": poller.failure_count,
            "poll_regressions": poller.regression_count,
            "poll_error": poller.last_error,
        }
    except SessionNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except GrowwAuthenticationError:
        print("Groww authentication failed.", file=sys.stderr)
        return 1
    except GrowwStreamConnectionError:
        # Authored literal, like the branch below: the cause is reported through
        # the exit code and this message, never by echoing broker text.
        print(
            "Groww live feed unreachable; the stream connection failed.",
            file=sys.stderr,
        )
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
        "volume_source": volume_source,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
