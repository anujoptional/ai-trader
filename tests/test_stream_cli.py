import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

from pytest import CaptureFixture

from ai_trader.broker import Instrument, MarketTick
from ai_trader.cli.check_stream import main


def test_main_prints_normalized_ticks(capsys: CaptureFixture[str]) -> None:
    tick = MarketTick(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        price=Decimal("1234.5"),
        timestamp=datetime(2026, 9, 11, 10, 28, 29, tzinfo=UTC),
    )
    stream = Mock()

    def collect(
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None],
    ) -> tuple[MarketTick, ...]:
        on_tick(tick)
        return (tick,)

    stream.collect.side_effect = collect
    broker = Mock()
    broker.create_ltp_stream.return_value = stream

    with (
        patch("ai_trader.cli.check_stream.load_groww_settings"),
        patch(
            "ai_trader.cli.check_stream.GrowwBroker.authenticate",
            return_value=broker,
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "exchange": "NSE",
        "price": "1234.5",
        "timestamp": "2026-09-11T10:28:29+00:00",
        "trading_symbol": "RELIANCE",
    }
    instrument = Instrument(exchange="NSE", trading_symbol="RELIANCE")
    broker.create_ltp_stream.assert_called_once_with((instrument,))
    call_arguments = stream.collect.call_args.kwargs
    assert call_arguments["max_ticks"] == 5
    assert call_arguments["timeout_seconds"] == 30.0


def test_main_exits_normally_when_market_is_closed(
    capsys: CaptureFixture[str],
) -> None:
    stream = Mock()
    stream.collect.return_value = ()
    broker = Mock()
    broker.create_ltp_stream.return_value = stream

    with (
        patch("ai_trader.cli.check_stream.load_groww_settings"),
        patch(
            "ai_trader.cli.check_stream.GrowwBroker.authenticate",
            return_value=broker,
        ),
    ):
        exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == (
        "No ticks received within 30 seconds; the market may be closed.\n"
    )
