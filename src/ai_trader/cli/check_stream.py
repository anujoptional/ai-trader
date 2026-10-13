"""Run a bounded, read-only Kite LTP stream check."""

import json
import sys
from decimal import Decimal

from ai_trader.broker import Instrument, MarketTick
from ai_trader.broker.kite import (
    KiteAuthenticationError,
    KiteBroker,
    KiteBrokerError,
    KiteStreamConnectionError,
)
from ai_trader.config import ConfigurationError, load_kite_login_settings

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_MAX_TICKS = 5
_TIMEOUT_SECONDS = 30.0


def _price(value: Decimal) -> str:
    return format(value, "f")


def _print_tick(tick: MarketTick) -> None:
    print(
        json.dumps(
            {
                "exchange": tick.instrument.exchange,
                "trading_symbol": tick.instrument.trading_symbol,
                "price": _price(tick.price),
                "timestamp": tick.timestamp.isoformat(),
            },
            sort_keys=True,
        )
    )


def main() -> int:
    """Print up to five RELIANCE ticks, waiting at most 30 seconds."""
    try:
        settings = load_kite_login_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        broker = KiteBroker.authenticate(settings)
        # Through the context manager so the transport is handed back even when
        # the collection raises. A probe this short does not need a supervisor;
        # it needs to not leave a subscription open behind it.
        with broker.create_ltp_stream((_RELIANCE,)) as stream:
            ticks = stream.collect(
                max_ticks=_MAX_TICKS,
                timeout_seconds=_TIMEOUT_SECONDS,
                on_tick=_print_tick,
            )
    except KiteAuthenticationError:
        print("Kite authentication failed.", file=sys.stderr)
        return 1
    except KiteStreamConnectionError:
        print(
            "Kite live feed unreachable; the stream connection failed.",
            file=sys.stderr,
        )
        return 1
    except KiteBrokerError:
        print("Kite stream check failed.", file=sys.stderr)
        return 1

    if not ticks:
        print("No ticks received within 30 seconds; the market may be closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
