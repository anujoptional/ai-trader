from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import DepthLevel, DepthSnapshot, Instrument, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.paper.broker import PaperBroker, PaperConfig
from ai_trader.paper.charges import trade_charges
from ai_trader.paper.report import build_report, write_report
from ai_trader.paper.store import PaperStore

DAY = datetime(2026, 10, 12, tzinfo=INDIA_TIMEZONE)


def at(hour: int, minute: int, second: int = 0, days: int = 0) -> datetime:
    return DAY.replace(hour=hour, minute=minute, second=second) + timedelta(days=days)


def book(price: str, quantity: int = 10**6, *, levels=None) -> DepthSnapshot:
    asks = (DepthLevel(Decimal(price), quantity, 1),) if levels is None else levels
    return DepthSnapshot(
        instrument=Instrument("NSE", "ABC"),
        timestamp=at(9, 44),
        last_price=Decimal(price),
        last_trade_at=at(9, 44),
        volume=1,
        total_buy_quantity=1,
        total_sell_quantity=1,
        bids=(DepthLevel(Decimal(price) - Decimal("0.05"), 1000, 1),),
        asks=asks,
        previous_close=Decimal(price),
    )


def bar(moment: datetime, o: str, h: str, low: str, c: str) -> OHLCVCandle:
    return OHLCVCandle(moment, Decimal(o), Decimal(h), Decimal(low), Decimal(c), 100)


@pytest.fixture
def broker(tmp_path) -> PaperBroker:
    return PaperBroker(PaperStore(tmp_path / "paper.sqlite"), PaperConfig())


def buy(broker: PaperBroker, symbol="ABC", price="100", now=None, **kwargs):
    return broker.buy(
        symbol,
        kwargs.pop("depth", book(price)),
        Decimal("0.01"),
        now or at(9, 44, 30),
        cycle_id="c1",
        probability=0.65,
        confidence=0.6,
        reason="test",
    )


def test_charges_match_the_published_round_trips() -> None:
    same_day = trade_charges(Decimal(200_000), Decimal(202_000), same_session=True)
    assert same_day.kind == "intraday"
    assert same_day.brokerage == Decimal(40)  # Rs 20 cap on each leg
    assert same_day.stt == Decimal("50.500")  # 0.025% of the sell
    assert same_day.stamp == Decimal("6.00000")  # 0.003% of the buy
    assert same_day.dp == 0
    assert 118 < same_day.total < 120
    overnight = trade_charges(Decimal(200_000), Decimal(202_000), same_session=False)
    assert overnight.kind == "delivery"
    assert overnight.brokerage == 0
    assert overnight.stt == Decimal("402.000")  # 0.1% of both legs
    assert overnight.stamp == Decimal("30.00000")  # 0.015% of the buy
    assert overnight.dp == Decimal("15.34")
    assert 460 < overnight.total < 463


def test_buy_sizes_whole_shares_and_places_both_exits(broker) -> None:
    result = buy(broker, price="123.45")
    position = result.position
    assert result.reason == "filled"
    assert position.quantity == 1620  # floor(2,00,000 / 123.45)
    assert position.entry_price == Decimal("123.45")
    assert position.target_price == Decimal("124.69")  # 124.6845 rounded up
    assert position.stop_price == Decimal("122.21")  # 122.2155 rounded down
    assert broker.held() == {"ABC"}


def test_buy_walks_the_visible_book(broker) -> None:
    levels = (
        DepthLevel(Decimal("100"), 1000, 1),
        DepthLevel(Decimal("100.10"), 500, 1),
    )
    position = buy(broker, depth=book("100", levels=levels)).position
    # 2,000 shares: 1,000 @ 100, 500 @ 100.10, the last 500 at the deepest ask
    assert position.quantity == 2000
    assert position.buy_value == Decimal(1000 * 100) + Decimal("100.10") * 1000
    assert position.entry_price == position.buy_value / 2000


def test_limits_refuse_buys(broker, tmp_path) -> None:
    assert buy(broker).position is not None
    assert buy(broker).reason == "already_held"
    assert buy(broker, symbol="XYZ", depth=None).reason == "no_quote"
    empty = book("100", levels=())
    assert buy(broker, symbol="XYZ", depth=empty).reason == "no_sellers"
    assert buy(broker, symbol="XYZ", price="250000").reason == "price_above_per_trade"
    small = PaperBroker(
        PaperStore(tmp_path / "small.sqlite"),
        PaperConfig(capital=Decimal(300_000), max_positions=5),
    )
    assert buy(small, symbol="A1").position is not None
    assert buy(small, symbol="A2").reason == "capital_limit"
    capped = PaperBroker(
        PaperStore(tmp_path / "capped.sqlite"), PaperConfig(max_positions=1)
    )
    assert buy(capped, symbol="B1").position is not None
    assert buy(capped, symbol="B2").reason == "position_limit"


def test_target_fills_at_the_limit_and_is_intraday(broker) -> None:
    position = buy(broker).position  # 2,000 @ 100, target 101, stop 99
    bars = [
        bar(at(9, 44), "100", "102", "98", "100"),  # the buy minute is never checked
        bar(at(9, 45), "100", "100.5", "99.5", "100.2"),
        bar(at(9, 46), "100.2", "101", "100", "100.9"),
    ]
    assert broker.apply_bars(position, bars, at(9, 46, 30)) is None  # not final yet
    closed = broker.apply_bars(position, bars, at(9, 47, 20))
    assert (closed.exit_reason, closed.exit_price) == ("tp", Decimal("101"))
    assert closed.exit_at == at(9, 46)
    assert closed.gross_pnl == Decimal("2000.00")
    assert closed.charges_detail["kind"] == "intraday"
    assert closed.net_pnl == closed.gross_pnl - closed.charges
    assert broker.held() == set()


def test_stop_wins_a_minute_that_reaches_both(broker) -> None:
    position = buy(broker).position
    closed = broker.apply_bars(
        position, [bar(at(9, 45), "100", "101.2", "98.9", "100")], at(10, 0)
    )
    assert (closed.exit_reason, closed.exit_price) == ("sl_same_minute", Decimal("99"))


def test_overnight_gap_fills_at_the_open_with_delivery_charges(broker) -> None:
    position = buy(broker).position
    day_one = [
        bar(at(9, 45) + timedelta(minutes=i), "100", "100.5", "99.5", "100")
        for i in range(5)
    ]
    assert broker.apply_bars(position, day_one, at(15, 31)) is None
    reopened = broker.open_positions()[0]
    assert reopened.checked_through == at(9, 49)
    gap = [bar(at(9, 15, days=1), "97", "98", "96.5", "97.5")]
    closed = broker.apply_bars(reopened, gap, at(9, 17, days=1))
    assert (closed.exit_reason, closed.exit_price) == ("sl_gap", Decimal("97"))
    assert closed.charges_detail["kind"] == "delivery"
    up_gap = buy(broker, symbol="UP", now=at(10, 0)).position
    closed_up = broker.apply_bars(
        up_gap, [bar(at(9, 15, days=1), "103", "104", "102", "103")], at(9, 20, days=1)
    )
    assert (closed_up.exit_reason, closed_up.exit_price) == ("tp_gap", Decimal("103"))


def test_one_buy_per_stock_per_session_and_state_survives_restart(tmp_path) -> None:
    path = tmp_path / "paper.sqlite"
    first = PaperBroker(PaperStore(path), PaperConfig())
    position = buy(first).position
    first.apply_bars(position, [bar(at(9, 45), "100", "101", "100", "101")], at(10, 0))
    assert buy(first).reason == "already_bought_this_session"
    held = buy(first, symbol="HELD").position
    restarted = PaperBroker(PaperStore(path), PaperConfig())
    assert restarted.held() == {"HELD"}
    assert restarted.open_positions()[0].id == held.id
    assert buy(restarted, symbol="ABC", now=at(9, 45, days=1)).position is not None
    with pytest.raises(ValueError, match="different settings"):
        PaperBroker(PaperStore(path), PaperConfig(per_trade=Decimal(100_000)))


def test_report_lists_trades_cumulative_pnl_and_open_positions(
    broker, tmp_path
) -> None:
    win = buy(broker, symbol="WIN").position
    broker.apply_bars(win, [bar(at(9, 45), "100", "101", "100", "101")], at(10, 0))
    loss = buy(broker, symbol="LOSS", now=at(10, 0)).position
    broker.apply_bars(loss, [bar(at(10, 1), "100", "100", "99", "99")], at(10, 5))
    buy(broker, symbol="OPEN", now=at(11, 0))
    report = build_report(
        broker.store, {"OPEN": Decimal("100.5")}, at(12, 0), Decimal(5_000_000)
    )
    summary = report["summary"]
    assert summary["closed_trades"] == 2 and summary["wins"] == 1
    assert summary["target_exits"] == 1 and summary["stop_exits"] == 1
    trades = report["trades"]
    assert [t["symbol"] for t in trades] == ["WIN", "LOSS"]
    assert Decimal(trades[1]["cumulative_net_pnl"]) == Decimal(
        trades[0]["net_pnl"]
    ) + Decimal(trades[1]["net_pnl"])
    assert Decimal(summary["realised_net_pnl"]) == Decimal(
        trades[1]["cumulative_net_pnl"]
    )
    (held,) = report["open_positions"]
    assert held["symbol"] == "OPEN" and held["return_pct"] == 0.5
    assert Decimal(held["net_pnl_if_sold_now"]) < Decimal(held["gross_pnl"])
    write_report(report, tmp_path / "report")
    for name in ("trades.csv", "open_positions.csv", "daily.csv", "summary.json"):
        assert (tmp_path / "report" / name).exists()
    assert "Open portfolio:" in (tmp_path / "report" / "report.txt").read_text()
