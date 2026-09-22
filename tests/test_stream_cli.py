import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

from pytest import CaptureFixture

from ai_trader.broker import Instrument, MarketTick
from ai_trader.broker.groww import GrowwStreamConnectionError
from ai_trader.cli.check_stream import main


def _stream() -> MagicMock:
    """A tick stream that can be used the way the CLI uses one.

    ``MagicMock`` rather than ``Mock`` because the check takes its stream
    through a ``with``, and a plain ``Mock`` has no context-manager protocol at
    all. ``__enter__`` returns the mock itself, exactly as ``GrowwLtpStream``
    does, so the calls these tests assert on land on the configured object
    rather than on an anonymous one the protocol invented.
    """
    stream = MagicMock()
    stream.__enter__.return_value = stream
    return stream


def test_main_prints_normalized_ticks(capsys: CaptureFixture[str]) -> None:
    tick = MarketTick(
        instrument=Instrument(exchange="NSE", trading_symbol="RELIANCE"),
        price=Decimal("1234.5"),
        timestamp=datetime(2026, 9, 11, 10, 28, 29, tzinfo=UTC),
    )
    stream = _stream()

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
    stream = _stream()
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


def test_the_stream_is_handed_back_when_the_collection_fails(
    capsys: CaptureFixture[str],
) -> None:
    stream = _stream()
    stream.collect.side_effect = GrowwStreamConnectionError("socket closed")
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
    assert exit_code == 1
    assert "Groww live feed unreachable" in captured.err
    # The check is short enough not to want a supervisor, but not so short that
    # it may leave a subscription open when it dies partway. ``__exit__`` is
    # where ``GrowwLtpStream`` closes, so asserting it ran is what pins this
    # CLI's half of the handover.
    stream.__exit__.assert_called_once()
