import json
from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest
from pytest import CaptureFixture

from ai_trader.broker import CandleInterval, OHLCVCandle
from ai_trader.cli.check_historical_data import (
    HistoricalSessionNotFoundError,
    _find_recent_completed_session,
    main,
)


def test_main_prints_only_count_and_boundary_candles(
    capsys: CaptureFixture[str],
) -> None:
    first = OHLCVCandle(
        timestamp=datetime(2026, 9, 11, 3, 45, tzinfo=UTC),
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=5000,
    )
    middle = OHLCVCandle(
        timestamp=datetime(2026, 9, 11, 3, 46, tzinfo=UTC),
        open=Decimal("101"),
        high=Decimal("103"),
        low=Decimal("100"),
        close=Decimal("102"),
        volume=6000,
    )
    last = OHLCVCandle(
        timestamp=datetime(2026, 9, 11, 3, 47, tzinfo=UTC),
        open=Decimal("102"),
        high=Decimal("104"),
        low=Decimal("101"),
        close=Decimal("103"),
        volume=7000,
    )
    broker = Mock()

    with (
        patch("ai_trader.cli.check_historical_data.load_groww_settings"),
        patch(
            "ai_trader.cli.check_historical_data.GrowwBroker.authenticate",
            return_value=broker,
        ),
        patch(
            "ai_trader.cli.check_historical_data._find_recent_completed_session",
            return_value=(date(2026, 9, 11), (first, middle, last)),
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert output["trading_date"] == "2026-09-11"
    assert output["candle_count"] == 3
    assert output["first_candle"]["timestamp"] == "2026-09-11T03:45:00+00:00"
    assert output["last_candle"]["timestamp"] == "2026-09-11T03:47:00+00:00"
    assert middle.timestamp.isoformat() not in captured.out


def test_session_search_skips_weekend_and_falls_back_from_empty_weekday() -> None:
    candle = OHLCVCandle(
        timestamp=datetime(2026, 8, 13, 3, 45, tzinfo=UTC),
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=5000,
    )
    broker = Mock()
    broker.get_historical_candles.side_effect = [(), (candle,)]
    india_timezone = ZoneInfo("Asia/Kolkata")

    trading_date, candles = _find_recent_completed_session(
        broker,
        now=datetime(2026, 8, 16, 12, 0, tzinfo=india_timezone),
    )

    assert trading_date == date(2026, 8, 13)
    assert candles == (candle,)
    assert broker.get_historical_candles.call_count == 2
    request_dates = [
        request.kwargs["start"].date()
        for request in broker.get_historical_candles.call_args_list
    ]
    assert request_dates == [date(2026, 8, 14), date(2026, 8, 13)]
    for request in broker.get_historical_candles.call_args_list:
        assert request.kwargs["start"].timetz().replace(tzinfo=None).isoformat() == (
            "09:15:00"
        )
        assert request.kwargs["end"].timetz().replace(tzinfo=None).isoformat() == (
            "09:30:00"
        )
        assert request.kwargs["interval"] is CandleInterval.ONE_MINUTE


def test_session_search_stops_after_ten_weekdays() -> None:
    broker = Mock()
    broker.get_historical_candles.return_value = ()
    india_timezone = ZoneInfo("Asia/Kolkata")

    with pytest.raises(HistoricalSessionNotFoundError, match="last 10 weekdays"):
        _find_recent_completed_session(
            broker,
            now=datetime(2026, 8, 17, 12, 0, tzinfo=india_timezone),
        )

    assert broker.get_historical_candles.call_count == 10
