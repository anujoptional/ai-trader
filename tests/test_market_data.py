from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from ai_trader.broker import CandleInterval, Instrument
from ai_trader.broker import groww as groww_module
from ai_trader.broker.groww import GrowwBroker, GrowwMarketDataError


def test_get_ltp_normalizes_prices_and_uses_cash_segment() -> None:
    client = Mock()
    client.get_ltp.return_value = {
        "NSE_RELIANCE": 1234.5,
        "NSE_NIFTY": 25000.25,
    }
    instruments = (
        Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        Instrument(exchange="NSE", trading_symbol="NIFTY"),
    )

    prices = GrowwBroker(client).get_ltp(instruments)

    client.get_ltp.assert_called_once_with(
        exchange_trading_symbols=("NSE_RELIANCE", "NSE_NIFTY"),
        segment="CASH",
    )
    assert [price.instrument for price in prices] == list(instruments)
    assert [price.price for price in prices] == [
        Decimal("1234.5"),
        Decimal("25000.25"),
    ]


@pytest.mark.parametrize("raw_timestamp", [1_789_122_509, 1_789_122_509_000])
def test_get_quote_normalizes_epoch_seconds_and_milliseconds(
    raw_timestamp: int,
) -> None:
    client = Mock()
    client.get_quote.return_value = {
        "last_price": 1234.5,
        "last_trade_time": raw_timestamp,
        "ohlc": {
            "open": 1200,
            "high": 1250.25,
            "low": 1190.5,
            "close": 1210,
        },
        "volume": 100_000,
        "day_change": 24.5,
        "day_change_perc": 2.02,
        "depth": {"ignored": "provider-specific"},
    }
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")

    quote = GrowwBroker(client).get_quote(instrument)

    client.get_quote.assert_called_once_with(
        trading_symbol="RELIANCE",
        exchange="NSE",
        segment="CASH",
    )
    assert quote.instrument == instrument
    assert quote.last_price == Decimal("1234.5")
    assert quote.last_trade_at == datetime(2026, 9, 11, 10, 28, 29, tzinfo=UTC)
    assert quote.open == Decimal("1200")
    assert quote.high == Decimal("1250.25")
    assert quote.low == Decimal("1190.5")
    assert quote.previous_close == Decimal("1210")
    assert quote.volume == 100_000
    assert quote.day_change == Decimal("24.5")
    assert quote.day_change_percent == Decimal("2.02")


def test_get_quote_rejects_nonsensical_timestamp() -> None:
    client = Mock()
    client.get_quote.return_value = {
        "last_price": 1234.5,
        "last_trade_time": 42,
        "ohlc": {"open": 1200, "high": 1250, "low": 1190, "close": 1210},
        "volume": 100_000,
        "day_change": 24.5,
        "day_change_perc": 2.02,
    }

    with pytest.raises(GrowwMarketDataError, match="quote retrieval failed"):
        GrowwBroker(client).get_quote(
            Instrument(exchange="NSE", trading_symbol="RELIANCE")
        )


def test_get_historical_candles_uses_replacement_sdk_method_and_normalizes() -> None:
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [
            ["2026-09-14T10:00:00", 100, 102.5, 99, 101.25, 5000, None],
            ["2026-09-14T10:01:00", 101.25, 103, 101, 102, 6000, None],
        ]
    }
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")
    india_timezone = ZoneInfo("Asia/Kolkata")
    start = datetime(2026, 9, 14, 10, 0, tzinfo=india_timezone)
    end = datetime(2026, 9, 14, 10, 2, tzinfo=india_timezone)

    candles = GrowwBroker(client).get_historical_candles(
        instrument=instrument,
        start=start,
        end=end,
        interval=CandleInterval.ONE_MINUTE,
    )

    client.get_historical_candles.assert_called_once_with(
        exchange="NSE",
        segment="CASH",
        groww_symbol="NSE-RELIANCE",
        start_time="2026-09-14 10:00:00",
        end_time="2026-09-14 10:02:00",
        candle_interval="1minute",
    )
    assert len(candles) == 2
    assert candles[0].timestamp == datetime(2026, 9, 14, 4, 30, tzinfo=UTC)
    assert candles[0].open == Decimal("100")
    assert candles[0].high == Decimal("102.5")
    assert candles[0].low == Decimal("99")
    assert candles[0].close == Decimal("101.25")
    assert candles[0].volume == 5000


def test_get_historical_candles_rejects_naive_datetimes_offline() -> None:
    client = Mock()
    broker = GrowwBroker(client)

    with pytest.raises(ValueError, match="timezone-aware"):
        broker.get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
            start=datetime(2026, 9, 14, 10, 0),
            end=datetime(2026, 9, 14, 10, 30),
            interval=CandleInterval.ONE_MINUTE,
        )

    client.get_historical_candles.assert_not_called()


@pytest.mark.parametrize("price", [float("nan"), float("inf"), float("-inf"), "NaN"])
def test_get_ltp_rejects_non_finite_prices(price: object) -> None:
    # "NaN" and "Infinity" survive Decimal parsing, so they need an explicit check.
    client = Mock()
    client.get_ltp.return_value = {"NSE_RELIANCE": price}

    with pytest.raises(GrowwMarketDataError, match="LTP retrieval failed"):
        GrowwBroker(client).get_ltp(
            (Instrument(exchange="NSE", trading_symbol="RELIANCE"),)
        )


@pytest.mark.parametrize(
    "raw_candle",
    [
        ["2026-09-14T10:00:00", float("nan"), 102.5, 99, 101.25, 5000],
        ["2026-09-14T10:00:00", 100, float("inf"), 99, 101.25, 5000],
        ["2026-09-14T10:00:00", 100, 100, 99, 101.25, 5000],
        ["2026-09-14T10:00:00", 100, 102.5, 101, 101.25, 5000],
        ["2026-09-14T10:00:00", 100, 99, 102.5, 101.25, 5000],
    ],
    ids=[
        "nan open",
        "inf high",
        "high below close",
        "low above open",
        "high below low",
    ],
)
def test_get_historical_candles_rejects_unusable_prices(
    raw_candle: list[object],
) -> None:
    client = Mock()
    client.get_historical_candles.return_value = {"candles": [raw_candle]}
    india_timezone = ZoneInfo("Asia/Kolkata")

    with pytest.raises(GrowwMarketDataError, match="historical data retrieval failed"):
        GrowwBroker(client).get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
            start=datetime(2026, 9, 14, 10, 0, tzinfo=india_timezone),
            end=datetime(2026, 9, 14, 10, 1, tzinfo=india_timezone),
            interval=CandleInterval.ONE_MINUTE,
        )


def test_get_historical_candles_accepts_a_flat_candle() -> None:
    # A minute with a single traded price is consistent, not corrupt.
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [["2026-09-14T10:00:00", 100, 100, 100, 100, 5000]]
    }
    india_timezone = ZoneInfo("Asia/Kolkata")

    candles = GrowwBroker(client).get_historical_candles(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        start=datetime(2026, 9, 14, 10, 0, tzinfo=india_timezone),
        end=datetime(2026, 9, 14, 10, 1, tzinfo=india_timezone),
        interval=CandleInterval.ONE_MINUTE,
    )

    assert len(candles) == 1
    assert candles[0].high == candles[0].low == Decimal("100")


def test_transient_broker_failures_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    # Groww intermittently answers a valid read with a plain-text "404 page not
    # found" body that the SDK cannot decode. Measured live on 2026-09-21, this
    # hit 94 of 200 reads, so a single attempt is worse than a coin flip.
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)
    client = Mock()
    client.get_ltp.side_effect = [
        ValueError("Extra data: line 1 column 5 (char 4)"),
        ValueError("Extra data: line 1 column 5 (char 4)"),
        {"NSE_RELIANCE": 1234.5},
    ]

    prices = GrowwBroker(client).get_ltp(
        (Instrument(exchange="NSE", trading_symbol="RELIANCE"),)
    )

    assert prices[0].price == Decimal("1234.5")
    assert client.get_ltp.call_count == 3


def test_persistent_broker_failures_surface_after_the_attempt_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)
    client = Mock()
    client.get_quote.side_effect = ValueError("Extra data: line 1 column 5 (char 4)")

    with pytest.raises(GrowwMarketDataError, match="quote retrieval failed"):
        GrowwBroker(client).get_quote(
            Instrument(exchange="NSE", trading_symbol="RELIANCE")
        )

    assert client.get_quote.call_count == groww_module._CALL_ATTEMPTS


def test_malformed_payloads_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only the raw broker call is retried. A schema change must surface at once
    # rather than being hammered five times.
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)
    client = Mock()
    client.get_historical_candles.return_value = {"candles": "not-a-list"}
    india_timezone = ZoneInfo("Asia/Kolkata")

    with pytest.raises(GrowwMarketDataError, match="historical data retrieval failed"):
        GrowwBroker(client).get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
            start=datetime(2026, 9, 14, 10, 0, tzinfo=india_timezone),
            end=datetime(2026, 9, 14, 10, 1, tzinfo=india_timezone),
            interval=CandleInterval.ONE_MINUTE,
        )

    client.get_historical_candles.assert_called_once()


def test_retries_back_off_until_the_delay_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # Delays must clear the longest observed outage (3.6s) but then stop
    # growing: past that point extra attempts buy more than extra waiting does.
    delays: list[float] = []
    monkeypatch.setattr(groww_module.time, "sleep", delays.append)
    client = Mock()
    client.get_quote.side_effect = ValueError("Extra data: line 1 column 5 (char 4)")

    with pytest.raises(GrowwMarketDataError, match="quote retrieval failed"):
        GrowwBroker(client).get_quote(
            Instrument(exchange="NSE", trading_symbol="RELIANCE")
        )

    base = groww_module._RETRY_BASE_DELAY_SECONDS
    cap = groww_module._MAX_RETRY_DELAY_SECONDS
    assert len(delays) == groww_module._CALL_ATTEMPTS - 1
    assert delays[:3] == [base, base * 2, base * 4]
    assert set(delays[3:]) == {cap}
    assert max(delays) == cap


def test_a_retried_call_succeeds_after_the_delay_cap_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The budget is only useful if a late attempt can still win.
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)
    client = Mock()
    failures = [ValueError("Extra data: line 1 column 5 (char 4)")] * (
        groww_module._CALL_ATTEMPTS - 1
    )
    client.get_instrument_by_groww_symbol.side_effect = [
        *failures,
        {
            "exchange": "NSE",
            "exchange_token": "2885",
            "trading_symbol": "RELIANCE",
            "groww_symbol": "NSE-RELIANCE",
            "segment": "CASH",
        },
    ]

    resolved = GrowwBroker(client).resolve_instrument("NSE-RELIANCE")

    assert resolved.exchange_token == "2885"
    assert (
        client.get_instrument_by_groww_symbol.call_count == groww_module._CALL_ATTEMPTS
    )
