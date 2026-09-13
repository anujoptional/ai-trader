import json
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

from pytest import CaptureFixture

from ai_trader.broker import Instrument, LastTradedPrice, MarketQuote
from ai_trader.cli.check_market_data import main


def test_main_prints_normalized_market_data(capsys: CaptureFixture[str]) -> None:
    reliance = Instrument(exchange="NSE", trading_symbol="RELIANCE")
    nifty = Instrument(exchange="NSE", trading_symbol="NIFTY")
    broker = Mock()
    broker.get_ltp.return_value = (
        LastTradedPrice(instrument=reliance, price=Decimal("1234.5")),
        LastTradedPrice(instrument=nifty, price=Decimal("25000.25")),
    )
    broker.get_quote.return_value = MarketQuote(
        instrument=reliance,
        last_price=Decimal("1234.5"),
        last_trade_at=datetime(2026, 9, 14, 4, 30, tzinfo=UTC),
        open=Decimal("1200"),
        high=Decimal("1250"),
        low=Decimal("1190"),
        previous_close=Decimal("1210"),
        volume=100_000,
        day_change=Decimal("24.5"),
        day_change_percent=Decimal("2.02"),
    )

    with (
        patch("ai_trader.cli.check_market_data.load_groww_settings"),
        patch(
            "ai_trader.cli.check_market_data.GrowwBroker.authenticate",
            return_value=broker,
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert output["latest_prices"] == [
        {"exchange": "NSE", "trading_symbol": "RELIANCE", "price": "1234.5"},
        {"exchange": "NSE", "trading_symbol": "NIFTY", "price": "25000.25"},
    ]
    assert output["reliance_quote"]["last_price"] == "1234.5"
    broker.get_ltp.assert_called_once_with((reliance, nifty))
    broker.get_quote.assert_called_once_with(reliance)
