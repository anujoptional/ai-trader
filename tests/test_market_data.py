from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from ai_trader.broker import CandleInterval, Instrument
from ai_trader.broker import groww as groww_module
from ai_trader.broker.groww import GrowwBroker, GrowwMarketDataError
from ai_trader.clock import INDIA_TIMEZONE


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
    start = datetime(2026, 9, 14, 10, 0, tzinfo=INDIA_TIMEZONE)
    end = datetime(2026, 9, 14, 10, 2, tzinfo=INDIA_TIMEZONE)

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

    with pytest.raises(GrowwMarketDataError, match="historical data retrieval failed"):
        GrowwBroker(client).get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
            start=datetime(2026, 9, 14, 10, 0, tzinfo=INDIA_TIMEZONE),
            end=datetime(2026, 9, 14, 10, 1, tzinfo=INDIA_TIMEZONE),
            interval=CandleInterval.ONE_MINUTE,
        )


def test_get_historical_candles_accepts_a_flat_candle() -> None:
    # A minute with a single traded price is consistent, not corrupt.
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [["2026-09-14T10:00:00", 100, 100, 100, 100, 5000]]
    }

    candles = GrowwBroker(client).get_historical_candles(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        start=datetime(2026, 9, 14, 10, 0, tzinfo=INDIA_TIMEZONE),
        end=datetime(2026, 9, 14, 10, 1, tzinfo=INDIA_TIMEZONE),
        interval=CandleInterval.ONE_MINUTE,
    )

    assert len(candles) == 1
    assert candles[0].high == candles[0].low == Decimal("100")


def test_get_historical_candles_carries_an_unreported_volume_as_unknown() -> None:
    """The bar below is verbatim from an NSE session: a real range, no volume.

    ``None`` rather than ``0``. The minute traded -- high and low differ -- so
    zero would be an assertion nobody measured, and the feature engine would
    average it into ``volume_ratio_20`` as though it had been.
    """
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [["2026-09-25T15:15:00", 1224.3, 1224.7, 1224.3, 1224.7, None, None]]
    }

    candles = GrowwBroker(client).get_historical_candles(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        start=datetime(2026, 9, 25, 15, 15, tzinfo=INDIA_TIMEZONE),
        end=datetime(2026, 9, 25, 15, 16, tzinfo=INDIA_TIMEZONE),
        interval=CandleInterval.ONE_MINUTE,
    )

    assert len(candles) == 1
    assert candles[0].volume is None
    assert candles[0].high == Decimal("1224.7")
    assert candles[0].low == Decimal("1224.3")


def test_one_unreported_volume_does_not_discard_the_bars_around_it() -> None:
    """The shape that actually failed: one null minute in a long fetch.

    Normalization is a generator inside a blanket ``except``, so a single bad
    bar aborted the whole call -- three months of history for an instrument
    thrown away over one minute of it.
    """
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [
            ["2026-09-25T15:13:00", 1224.0, 1224.5, 1223.8, 1224.3, 4100, None],
            ["2026-09-25T15:14:00", 1224.3, 1224.7, 1224.3, 1224.7, None, None],
            ["2026-09-25T15:15:00", 1224.7, 1225.0, 1224.4, 1224.9, 3800, None],
        ]
    }

    candles = GrowwBroker(client).get_historical_candles(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        start=datetime(2026, 9, 25, 15, 13, tzinfo=INDIA_TIMEZONE),
        end=datetime(2026, 9, 25, 15, 16, tzinfo=INDIA_TIMEZONE),
        interval=CandleInterval.ONE_MINUTE,
    )

    assert [candle.volume for candle in candles] == [4100, None, 3800]


def test_the_pre_open_auction_is_dropped_rather_than_failing_the_fetch() -> None:
    """The 09:00 row below is verbatim: real volume, not one price.

    NSE collects pre-open orders from 09:00 and matches them around 09:08, so
    until it matches there is a book but no trade. Groww publishes those minutes
    as ordinary rows -- 35 per instrument per week, every one before 09:10.
    Raising on them threw away 1,888 bars over an artifact of market structure.
    """
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [
            ["2026-09-22T09:00:00", None, None, None, None, 28002, None],
            ["2026-09-22T09:07:00", None, None, None, None, 41577, None],
            ["2026-09-22T09:15:00", 1224.0, 1224.5, 1223.8, 1224.3, 4100, None],
        ]
    }

    candles = GrowwBroker(client).get_historical_candles(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        start=datetime(2026, 9, 22, 9, 0, tzinfo=INDIA_TIMEZONE),
        end=datetime(2026, 9, 22, 9, 16, tzinfo=INDIA_TIMEZONE),
        interval=CandleInterval.ONE_MINUTE,
    )

    assert len(candles) == 1
    assert candles[0].timestamp == datetime(2026, 9, 22, 3, 45, tzinfo=UTC)
    assert candles[0].close == Decimal("1224.3")


def test_a_session_of_nothing_but_auction_rows_is_empty_not_an_error() -> None:
    # A holiday-shortened or halted day can leave only the auction. Empty is the
    # honest answer; the caller already treats no bars as no bars.
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [["2026-09-22T09:03:00", None, None, None, None, 28002, None]]
    }

    candles = GrowwBroker(client).get_historical_candles(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        start=datetime(2026, 9, 22, 9, 0, tzinfo=INDIA_TIMEZONE),
        end=datetime(2026, 9, 22, 9, 4, tzinfo=INDIA_TIMEZONE),
        interval=CandleInterval.ONE_MINUTE,
    )

    assert candles == ()


@pytest.mark.parametrize(
    "raw_candle",
    [
        ["2026-09-14T10:00:00", None, 102.5, 99, 101.25, 5000],
        ["2026-09-14T10:00:00", 100, None, 99, 101.25, 5000],
        ["2026-09-14T10:00:00", 100, 102.5, None, 101.25, 5000],
        ["2026-09-14T10:00:00", 100, 102.5, 99, None, 5000],
        ["2026-09-14T10:00:00", None, None, None, 101.25, 5000],
    ],
    ids=["no open", "no high", "no low", "no close", "close only"],
)
def test_a_partly_priced_bar_still_fails_rather_than_vanishing(
    raw_candle: list[object],
) -> None:
    """Dropping the auction must not have become "drop whatever looks odd".

    Every null-price row observed was null in all four fields at once. A row
    missing only some of them has no market-structure explanation, so it is a
    schema change or corruption -- and a fetch that silently returned fewer bars
    than the window holds would hide it inside a backtest's results.
    """
    client = Mock()
    client.get_historical_candles.return_value = {"candles": [raw_candle]}

    with pytest.raises(GrowwMarketDataError, match="historical data retrieval failed"):
        GrowwBroker(client).get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
            start=datetime(2026, 9, 14, 10, 0, tzinfo=INDIA_TIMEZONE),
            end=datetime(2026, 9, 14, 10, 1, tzinfo=INDIA_TIMEZONE),
            interval=CandleInterval.ONE_MINUTE,
        )


@pytest.mark.parametrize(
    "raw_volume",
    [True, False, -1, 5000.5, "5000"],
    ids=["true", "false", "negative", "fractional", "string"],
)
def test_get_historical_candles_still_rejects_an_unusable_volume(
    raw_volume: object,
) -> None:
    """Accepting ``None`` must not have widened the gate to anything else.

    ``True`` is the one worth naming: it is an ``int`` in Python, so a bare
    ``int()`` would record a minute that traded one share.
    """
    client = Mock()
    client.get_historical_candles.return_value = {
        "candles": [["2026-09-14T10:00:00", 100, 102.5, 99, 101.25, raw_volume]]
    }

    with pytest.raises(GrowwMarketDataError, match="historical data retrieval failed"):
        GrowwBroker(client).get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
            start=datetime(2026, 9, 14, 10, 0, tzinfo=INDIA_TIMEZONE),
            end=datetime(2026, 9, 14, 10, 1, tzinfo=INDIA_TIMEZONE),
            interval=CandleInterval.ONE_MINUTE,
        )


def test_a_failed_historical_fetch_names_what_it_asked_for() -> None:
    """Without this the message fits every instrument and every window.

    It cost five probes to find one malformed bar the first time, because the
    cause is deliberately suppressed -- the SDK exception can carry the request
    that produced it. The identifiers are the caller's own, so they leak
    nothing the traceback would have.

    The suppression is checked through ``__suppress_context__`` and not through
    ``__cause__``, which reads like the obvious test and is vacuous: ``from
    None`` and a bare ``raise`` inside an ``except`` both leave ``__cause__`` at
    ``None``, and only the second prints the chain. ``_shown`` in
    ``tests/test_failure_attribution.py`` has the longer version.

    **The window is spelled in UTC here and in ``INDIA_TIMEZONE`` everywhere
    else in this file, which is deliberate.** The asserted string is the
    caller's own ``start`` formatted back through ``_groww_datetime``, so if the
    test stated the window in the same constant the code converts with, a wrong
    constant would move both sides together and the substring would still match
    -- the assertion would read exactly as it does now and check nothing about
    the zone. Stating 04:30 UTC and expecting 10:00 makes the conversion itself
    the thing under test. Measured: with ``INDIA_TIMEZONE`` on both sides this
    test survived a mutation of that constant to ``America/New_York``; spelled
    this way it fails, as it did before the constant was shared.
    """
    client = Mock()
    client.get_historical_candles.return_value = {"candles": "not a list"}

    with pytest.raises(GrowwMarketDataError) as caught:
        GrowwBroker(client).get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="INFY"),
            start=datetime(2026, 9, 14, 4, 30, tzinfo=UTC),
            end=datetime(2026, 9, 14, 4, 31, tzinfo=UTC),
            interval=CandleInterval.ONE_MINUTE,
        )

    message = str(caught.value)
    assert "NSE:INFY" in message
    assert "2026-09-14 10:00:00" in message
    assert caught.value.__suppress_context__


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

    with pytest.raises(GrowwMarketDataError, match="historical data retrieval failed"):
        GrowwBroker(client).get_historical_candles(
            instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
            start=datetime(2026, 9, 14, 10, 0, tzinfo=INDIA_TIMEZONE),
            end=datetime(2026, 9, 14, 10, 1, tzinfo=INDIA_TIMEZONE),
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
