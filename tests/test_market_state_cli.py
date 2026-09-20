import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from pytest import CaptureFixture

from ai_trader.broker import CandleInterval, MarketTick, OHLCVCandle
from ai_trader.broker.groww import GrowwBrokerError
from ai_trader.cli.check_market_state import (
    _RELIANCE,
    SessionNotFoundError,
    _find_recent_completed_session,
    main,
)
from ai_trader.config import ConfigurationError

_SESSION_OPEN = datetime(2026, 9, 11, 3, 45, tzinfo=UTC)


def _ohlcv(minutes: int) -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=_SESSION_OPEN + timedelta(minutes=minutes),
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=5_000 + minutes,
    )


def _broker_with_stream(
    collect: Callable[..., tuple[MarketTick, ...]] | None = None,
) -> Mock:
    stream = Mock()
    stream.collect.side_effect = collect
    if collect is None:
        stream.collect.return_value = ()
    broker = Mock()
    broker.create_ltp_stream.return_value = stream
    return broker


def _run(broker: Mock, candles: tuple[OHLCVCandle, ...]) -> int:
    with (
        patch("ai_trader.cli.check_market_state.load_groww_settings"),
        patch(
            "ai_trader.cli.check_market_state.GrowwBroker.authenticate",
            return_value=broker,
        ),
        patch(
            "ai_trader.cli.check_market_state._find_recent_completed_session",
            return_value=(date(2026, 9, 11), candles),
        ),
    ):
        return main()


def test_main_reports_backfilled_state_when_no_ticks_arrive(
    capsys: CaptureFixture[str],
) -> None:
    exit_code = _run(_broker_with_stream(), (_ohlcv(0), _ohlcv(1), _ohlcv(2)))

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert output["backfill_trading_date"] == "2026-09-11"
    assert output["backfilled_candles"] == 3
    assert output["live_ticks"] == 0
    assert output["retained_candles"] == 3
    assert output["late_ticks"] == 0
    assert output["duplicate_candles"] == 0
    assert output["last_price"] is None
    assert output["last_tick_at"] is None
    assert output["first_candle"]["start_time"] == "2026-09-11T03:45:00+00:00"
    assert output["last_candle"]["end_time"] == "2026-09-11T03:48:00+00:00"


def test_main_folds_live_ticks_into_the_backfilled_state(
    capsys: CaptureFixture[str],
) -> None:
    ticks = tuple(
        MarketTick(
            instrument=_RELIANCE,
            timestamp=_SESSION_OPEN + timedelta(minutes=minutes),
            price=Decimal(price),
            cumulative_volume=volume,
        )
        for minutes, price, volume in ((3, "150", 9_000), (4, "151", 9_400))
    )

    def collect(
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None],
    ) -> tuple[MarketTick, ...]:
        assert max_ticks > 0
        assert timeout_seconds > 0
        for tick in ticks:
            on_tick(tick)
        return ticks

    exit_code = _run(_broker_with_stream(collect), (_ohlcv(0), _ohlcv(1), _ohlcv(2)))

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert exit_code == 0
    assert output["live_ticks"] == 2
    assert output["retained_candles"] == 5
    assert output["duplicate_candles"] == 0
    assert output["last_price"] == "151"
    assert output["last_tick_at"] == "2026-09-11T03:49:00+00:00"
    assert output["last_candle"]["start_time"] == "2026-09-11T03:49:00+00:00"
    assert output["last_candle"]["close"] == "151"


def test_missing_configuration_exits_two(capsys: CaptureFixture[str]) -> None:
    with patch(
        "ai_trader.cli.check_market_state.load_groww_settings",
        side_effect=ConfigurationError("GROWW_API_KEY is missing."),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.strip() == "GROWW_API_KEY is missing."


@pytest.mark.parametrize(
    ("error", "expected_stderr"),
    [
        (SessionNotFoundError("No completed NSE trading session"), "No completed NSE"),
        (GrowwBrokerError("boom"), "Groww market-state check failed."),
    ],
)
def test_broker_failures_exit_one_without_leaking_details(
    error: Exception,
    expected_stderr: str,
    capsys: CaptureFixture[str],
) -> None:
    with (
        patch("ai_trader.cli.check_market_state.load_groww_settings"),
        patch(
            "ai_trader.cli.check_market_state.GrowwBroker.authenticate",
            side_effect=error,
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert expected_stderr in captured.err


def test_session_search_requests_the_full_session_and_skips_weekends() -> None:
    candle = _ohlcv(0)
    broker = Mock()
    broker.get_historical_candles.side_effect = [(), (candle,)]

    trading_date, candles = _find_recent_completed_session(
        broker,
        now=datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
    )

    assert trading_date == date(2026, 9, 11)
    assert candles == (candle,)
    request_dates = [
        request.kwargs["start"].date()
        for request in broker.get_historical_candles.call_args_list
    ]
    assert request_dates == [date(2026, 9, 14), date(2026, 9, 11)]
    for request in broker.get_historical_candles.call_args_list:
        assert request.kwargs["start"].timetz().replace(tzinfo=None).isoformat() == (
            "09:15:00"
        )
        assert request.kwargs["end"].timetz().replace(tzinfo=None).isoformat() == (
            "15:30:00"
        )
        assert request.kwargs["interval"] is CandleInterval.ONE_MINUTE


def test_session_search_stops_after_ten_weekdays() -> None:
    broker = Mock()
    broker.get_historical_candles.return_value = ()

    with pytest.raises(SessionNotFoundError, match="last 10 weekdays"):
        _find_recent_completed_session(
            broker,
            now=datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        )

    assert broker.get_historical_candles.call_count == 10
