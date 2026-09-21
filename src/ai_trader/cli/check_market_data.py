"""Check Groww live market data using read-only broker operations."""

import json
import sys
from decimal import Decimal

from ai_trader.broker import Instrument, LastTradedPrice, MarketQuote
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBroker,
    GrowwBrokerError,
)
from ai_trader.config import ConfigurationError, load_groww_settings

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_NIFTY = Instrument(exchange="NSE", trading_symbol="NIFTY")


def _price(value: Decimal) -> str:
    return format(value, "f")


def _ltp_summary(ltp: LastTradedPrice) -> dict[str, str]:
    return {
        "exchange": ltp.instrument.exchange,
        "trading_symbol": ltp.instrument.trading_symbol,
        "price": _price(ltp.price),
    }


def _quote_summary(quote: MarketQuote) -> dict[str, object]:
    return {
        "exchange": quote.instrument.exchange,
        "trading_symbol": quote.instrument.trading_symbol,
        "last_price": _price(quote.last_price),
        "last_trade_at": quote.last_trade_at.isoformat(),
        "open": _price(quote.open),
        "high": _price(quote.high),
        "low": _price(quote.low),
        "previous_close": _price(quote.previous_close),
        "volume": quote.volume,
        "day_change": _price(quote.day_change),
        "day_change_percent": _price(quote.day_change_percent),
    }


def main() -> int:
    """Retrieve and print a small, non-sensitive live market-data summary."""
    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        broker = GrowwBroker.authenticate(settings)
        latest_prices = broker.get_ltp((_RELIANCE, _NIFTY))
        reliance_quote = broker.get_quote(_RELIANCE)
    except GrowwAuthenticationError:
        print("Groww authentication failed.", file=sys.stderr)
        return 1
    except GrowwBrokerError:
        print("Groww market data check failed.", file=sys.stderr)
        return 1

    summary = {
        "latest_prices": [_ltp_summary(ltp) for ltp in latest_prices],
        "reliance_quote": _quote_summary(reliance_quote),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
