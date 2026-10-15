from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import DepthLevel, DepthSnapshot, Instrument, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.paper.broker import PaperBroker, PaperConfig
from ai_trader.paper.charges import trade_charges
from ai_trader.paper.report import build_report, write_report
from ai_trader.paper.store import PaperStore

DAY = datetime(2026, 10, 14, tzinfo=INDIA_TIMEZONE)


def at(hour: int, minute: int, second: int = 0, days: int = 0) -> datetime:
    return DAY.replace(hour=hour, minute=minute, second=second) + timedelta(days=days)


def book(price: str, quantity: int = 10**6, *, levels=None, bids=None) -> DepthSnapshot:
    asks = (DepthLevel(Decimal(price), quantity, 1),) if levels is None else levels
    if bids is None:
        bids = (DepthLevel(Decimal(price) - Decimal("0.05"), quantity, 1),)
    return DepthSnapshot(
        instrument=Instrument("NSE", "ABC"),
        timestamp=at(9, 44),
        last_price=Decimal(price),
        last_trade_at=at(9, 44),
        volume=1,
        total_buy_quantity=1,
        total_sell_quantity=1,
        bids=bids,
        asks=asks,
        previous_close=Decimal(price),
    )


def bar(moment: datetime, o: str, h: str, low: str, c: str) -> OHLCVCandle:
    return OHLCVCandle(moment, Decimal(o), Decimal(h), Decimal(low), Decimal(c), 100)


def flat_bars(start: datetime, end: datetime, price: str) -> list[OHLCVCandle]:
    bars, moment = [], start
    while moment <= end:
        bars.append(bar(moment, price, price, price, price))
        moment += timedelta(minutes=1)
    return bars


@pytest.fixture
def broker(tmp_path) -> PaperBroker:
    return PaperBroker(PaperStore(tmp_path / "paper.sqlite"), PaperConfig())


def enter(
    broker: PaperBroker, symbol="ABC", price="100", now=None, side="LONG", **kwargs
):
    return broker.enter(
        symbol,
        side,
        kwargs.pop("depth", book(price)),
        Decimal("0.01"),
        now or at(9, 44, 30),
        cycle_id="c1",
        probability=0.92,
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


def test_long_sizes_whole_shares_with_a_target_and_no_stop(broker) -> None:
    result = enter(broker, price="123.45")
    position = result.position
    assert result.reason == "filled"
    assert (position.side, position.quantity) == ("LONG", 1620)  # 2,00,000 / 123.45
    assert position.entry_price == Decimal("123.45")
    assert position.target_price == Decimal("124.69")  # 124.6845 rounded up
    assert position.stop_price is None
    assert broker.held() == {"ABC"}
    assert broker.deadline(position) == at(15, 0)


def test_entries_walk_the_visible_book_on_their_side(broker) -> None:
    levels = (
        DepthLevel(Decimal("100"), 1000, 1),
        DepthLevel(Decimal("100.10"), 500, 1),
    )
    position = enter(broker, depth=book("100", levels=levels)).position
    # 2,000 shares: 1,000 @ 100, 500 @ 100.10, the last 500 at the deepest ask
    assert position.quantity == 2000
    assert position.entry_value == Decimal(1000 * 100) + Decimal("100.10") * 1000
    assert position.entry_price == position.entry_value / 2000
    bids = (DepthLevel(Decimal("100"), 1000, 1), DepthLevel(Decimal("99.9"), 500, 1))
    short = enter(
        broker, symbol="SHT", side="SHORT", depth=book("100", bids=bids)
    ).position
    # 2,000 shares sold: 1,000 @ 100, 500 @ 99.90, the last 500 at the deepest bid
    assert (short.side, short.quantity) == ("SHORT", 2000)
    assert short.entry_value == Decimal(100_000) + Decimal("99.9") * 1000
    assert short.entry_price == Decimal("99.95")
    assert short.target_price == Decimal("98.95")  # 98.9505 rounded down
    assert short.stop_price is None


def test_limits_refuse_entries(broker, tmp_path) -> None:
    assert enter(broker).position is not None
    assert enter(broker).reason == "already_held"
    assert enter(broker, side="SHORT").reason == "already_held"
    assert enter(broker, symbol="XYZ", depth=None).reason == "no_quote"
    empty = book("100", levels=(), bids=())
    assert enter(broker, symbol="XYZ", depth=empty).reason == "no_sellers"
    assert enter(broker, symbol="XYZ", side="SHORT", depth=empty).reason == (
        "no_buyers"
    )
    assert enter(broker, symbol="XYZ", price="250000").reason == (
        "price_above_per_trade"
    )
    late = enter(broker, symbol="XYZ", now=at(15, 0))
    assert late.reason == "after_exit_time"
    assert enter(broker, symbol="XYZ", now=at(14, 59, 59)).position is not None
    small = PaperBroker(
        PaperStore(tmp_path / "small.sqlite"),
        PaperConfig(capital=Decimal(300_000), max_positions=5),
    )
    assert enter(small, symbol="A1", side="SHORT").position is not None
    assert enter(small, symbol="A2").reason == "capital_limit"
    capped = PaperBroker(
        PaperStore(tmp_path / "capped.sqlite"), PaperConfig(max_positions=1)
    )
    assert enter(capped, symbol="B1").position is not None
    assert enter(capped, symbol="B2").reason == "position_limit"
    with pytest.raises(ValueError, match="Unknown side"):
        enter(broker, symbol="C1", side="BUY")


def test_long_target_fills_at_the_limit_and_is_intraday(broker) -> None:
    position = enter(broker).position  # 2,000 @ 100, target 101
    bars = [
        bar(at(9, 44), "100", "102", "98", "100"),  # the entry minute is not checked
        bar(at(9, 45), "100", "100.5", "95", "100.2"),  # no stop: a fall is held
        bar(at(9, 46), "100.2", "101", "100", "100.9"),
    ]
    assert broker.apply_bars(position, bars, at(9, 46, 30)) is None  # not final yet
    assert broker.open_positions()[0].checked_through == at(9, 45)
    closed = broker.apply_bars(position, bars, at(9, 47, 20))
    assert (closed.exit_reason, closed.exit_price) == ("tp", Decimal("101"))
    assert closed.exit_at == at(9, 46)
    assert closed.gross_pnl == Decimal("2000.00")
    assert closed.charges_detail["kind"] == "intraday"
    expected = trade_charges(Decimal(200_000), Decimal(202_000), same_session=True)
    assert closed.charges == expected.total.quantize(Decimal("0.01"))
    assert closed.net_pnl == closed.gross_pnl - closed.charges
    assert broker.held() == set()


def test_short_target_fills_below_and_charges_the_cover_as_the_buy(broker) -> None:
    position = enter(broker, side="SHORT", depth=book("100.05")).position
    assert position.entry_price == Decimal("100")  # sold at the bid
    assert position.target_price == Decimal("99")
    bars = [
        bar(at(9, 45), "100", "104", "99.5", "103"),  # no stop: a rise is held
        bar(at(9, 46), "99.6", "99.7", "98.9", "99.2"),
    ]
    closed = broker.apply_bars(position, bars, at(10, 0))
    assert (closed.exit_reason, closed.exit_price) == ("tp", Decimal("99"))
    assert closed.gross_pnl == Decimal("2000.00")  # sold 2,00,000, bought 1,98,000
    expected = trade_charges(Decimal(198_000), Decimal(200_000), same_session=True)
    assert closed.charges == expected.total.quantize(Decimal("0.01"))
    assert closed.exit_value == Decimal(198_000)


def test_a_minute_opening_through_the_target_fills_at_its_open(broker) -> None:
    long = enter(broker).position
    closed = broker.apply_bars(
        long, [bar(at(9, 45), "101.5", "102", "101", "101.8")], at(10, 0)
    )
    assert (closed.exit_reason, closed.exit_price) == ("tp_gap", Decimal("101.5"))
    short = enter(broker, symbol="S", side="SHORT", depth=book("100.05")).position
    closed = broker.apply_bars(
        short, [bar(at(9, 45), "98.5", "99", "98", "98.8")], at(10, 0)
    )
    assert (closed.exit_reason, closed.exit_price) == ("tp_gap", Decimal("98.5"))


def test_a_position_without_its_target_is_closed_at_the_15_00_price(broker) -> None:
    position = enter(broker, now=at(14, 0, 30)).position
    bars = flat_bars(at(14, 1), at(14, 58), "99") + [
        bar(at(14, 59), "99", "99.2", "98.4", "98.6"),
        bar(at(15, 0), "98.6", "102", "98.5", "101.5"),  # from 15:00: never checked
    ]
    assert broker.apply_bars(position, bars, at(15, 0, 10)) is None  # 14:59 not final
    closed = broker.apply_bars(position, bars, at(15, 0, 20))
    assert (closed.exit_reason, closed.exit_price) == ("time_1500", Decimal("98.6"))
    assert closed.exit_at == at(15, 0)
    assert closed.gross_pnl == Decimal("-2800.00")
    assert closed.charges_detail["kind"] == "intraday"
    short = enter(broker, symbol="S", side="SHORT", now=at(14, 0, 30)).position
    above_target = flat_bars(at(14, 1), at(14, 58), "99.5") + [
        bar(at(14, 59), "99.5", "99.6", "99.1", "99.2"),  # target 98.95 not reached
    ]
    closed = broker.apply_bars(short, above_target, at(15, 0, 20))
    # sold 2,001 (2,00,000 / 99.95) at the 99.95 bid, bought back at 99.20
    assert short.quantity == 2001
    assert (closed.exit_reason, closed.gross_pnl) == ("time_1500", Decimal("1500.75"))


def test_the_15_00_exit_waits_for_the_14_59_bar(broker) -> None:
    position = enter(broker, now=at(14, 50, 30)).position
    bars = flat_bars(at(14, 51), at(14, 58), "100.5")  # 14:59 not published (yet)
    assert broker.apply_bars(position, bars, at(15, 1, 20)) is None
    closed = broker.apply_bars(position, bars, at(15, 2, 0))
    assert (closed.exit_reason, closed.exit_price) == ("time_1500", Decimal("100.5"))
    later = enter(broker, symbol="L", now=at(14, 59, 30)).position
    assert broker.apply_bars(later, [], at(15, 5)) is None  # nothing to price it
    closed = broker.close_at(later, at(15, 0), Decimal("100.2"), "time_1500_ltp")
    assert (closed.exit_reason, closed.exit_at) == ("time_1500_ltp", at(15, 0))
    assert broker.close_at(later, at(15, 0), Decimal("1"), "again") is None


def test_a_position_left_open_overnight_closes_at_its_own_15_00(broker) -> None:
    position = enter(broker, now=at(14, 30, 30)).position
    bars = flat_bars(at(14, 31), at(15, 29), "100.4") + flat_bars(
        at(9, 15, days=1), at(9, 20, days=1), "103"
    )
    closed = broker.apply_bars(position, bars, at(9, 21, days=1))
    assert (closed.exit_reason, closed.exit_price) == ("time_1500", Decimal("100.4"))
    assert closed.exit_at == at(15, 0)
    assert closed.charges_detail["kind"] == "intraday"


def test_one_trade_per_stock_per_session_and_state_survives_restart(tmp_path) -> None:
    path = tmp_path / "paper.sqlite"
    first = PaperBroker(PaperStore(path), PaperConfig())
    position = enter(first).position
    first.apply_bars(position, [bar(at(9, 45), "100", "101", "100", "101")], at(10, 0))
    assert enter(first, side="SHORT").reason == "already_traded_this_session"
    held = enter(first, symbol="HELD", side="SHORT").position
    restarted = PaperBroker(PaperStore(path), PaperConfig())
    assert restarted.held() == {"HELD"}
    assert restarted.open_positions()[0].id == held.id
    assert restarted.open_positions()[0].side == "SHORT"
    assert enter(restarted, symbol="ABC", now=at(9, 45, days=1)).position is not None
    with pytest.raises(ValueError, match="different settings"):
        PaperBroker(PaperStore(path), PaperConfig(per_trade=Decimal(100_000)))


def test_saved_settings_of_older_portfolios_still_match() -> None:
    assert repr(PaperConfig()) == (
        "PaperConfig(capital=Decimal('5000000'), per_trade=Decimal('200000'), "
        "max_positions=25, target=Decimal('0.01'), exit_time=datetime.time(15, 0))"
    )
    free = PaperConfig(capital=None, max_positions=None, one_per_stock=False)
    assert repr(free).startswith("PaperConfig(capital=None, ")
    assert repr(free).endswith("one_per_stock=False)")


def test_a_portfolio_without_limits_can_hold_a_stock_more_than_once(
    tmp_path,
) -> None:
    path = tmp_path / "paper.sqlite"
    config = PaperConfig(
        capital=None,
        max_positions=None,
        target=Decimal("0.005"),
        one_per_stock=False,
    )
    broker = PaperBroker(PaperStore(path), config)
    first = enter(broker).position
    broker.apply_bars(first, [bar(at(9, 45), "100", "101", "100", "101")], at(10, 0))
    entries = [
        enter(broker, now=at(10, 5)),
        enter(broker, side="SHORT", now=at(10, 10)),
        enter(broker, now=at(10, 15)),
    ]
    assert [e.reason for e in entries] == ["filled"] * 3
    assert broker.excluded(at(10, 20)) == set()
    assert broker.available() is None
    restarted = PaperBroker(PaperStore(path), config)  # no one-per-symbol index
    assert len(restarted.open_positions()) == 3
    assert enter(restarted, now=at(10, 25)).reason == "filled"
    report = build_report(
        restarted.store, {"ABC": Decimal(100)}, at(11, 0), None, Decimal("0.005")
    )
    assert report["summary"]["starting_capital"] is None
    assert report["summary"]["equity"] is None
    write_report(report, tmp_path / "report")
    text = (tmp_path / "report" / "report.txt").read_text()
    assert "1 at the 0.5% target" in text and "Capital: no limit" in text
    limited = PaperBroker(PaperStore(tmp_path / "limited.sqlite"), PaperConfig())
    enter(limited)
    assert limited.excluded(at(10, 0)) == {"ABC"}
    assert enter(limited, now=at(10, 0)).reason == "already_held"


def test_long_only_stores_get_a_side_column(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE positions (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT "
        "NOT NULL, status TEXT NOT NULL, quantity INTEGER NOT NULL, entry_at TEXT "
        "NOT NULL, entry_price TEXT NOT NULL, buy_value TEXT NOT NULL, target_price "
        "TEXT NOT NULL, stop_price TEXT NOT NULL, tick TEXT NOT NULL, "
        "checked_through TEXT NOT NULL, cycle_id TEXT NOT NULL, scanner_probability "
        "REAL NOT NULL, ai_confidence REAL NOT NULL, ai_reason TEXT NOT NULL, exit_at "
        "TEXT, exit_price TEXT, exit_reason TEXT, sell_value TEXT, charges TEXT, "
        "charges_detail TEXT, gross_pnl TEXT, net_pnl TEXT);"
        "INSERT INTO positions (symbol, status, quantity, entry_at, entry_price, "
        "buy_value, target_price, stop_price, tick, checked_through, cycle_id, "
        "scanner_probability, ai_confidence, ai_reason) VALUES ('OLD', 'open', 10, "
        "'2026-10-12T10:00:00+05:30', '100', '1000', '101', '99', '0.05', "
        "'2026-10-12T10:00:00+05:30', 'c', 0.7, 0.6, 'r');"
    )
    db.close()
    (old,) = PaperStore(path).positions()
    assert (old.side, old.entry_value, old.stop_price) == (
        "LONG",
        Decimal(1000),
        Decimal(99),
    )


def test_report_lists_trades_cumulative_pnl_and_open_positions(
    broker, tmp_path
) -> None:
    win = enter(broker, symbol="WIN").position
    broker.apply_bars(win, [bar(at(9, 45), "100", "101", "100", "101")], at(10, 0))
    loss = enter(broker, symbol="LOSS", side="SHORT", now=at(14, 50)).position
    broker.apply_bars(loss, flat_bars(at(14, 51), at(14, 59), "100.5"), at(15, 0, 20))
    enter(broker, symbol="OPEN", side="SHORT", depth=book("100.05"), now=at(11, 0))
    report = build_report(
        broker.store, {"OPEN": Decimal("99.5")}, at(12, 0), Decimal(5_000_000)
    )
    summary = report["summary"]
    assert summary["closed_trades"] == 2 and summary["wins"] == 1
    assert (summary["long_trades"], summary["short_trades"]) == (1, 1)
    assert summary["target_exits"] == 1 and summary["time_exits"] == 1
    assert summary["stop_exits"] == 0
    trades = report["trades"]
    assert [t["symbol"] for t in trades] == ["WIN", "LOSS"]
    assert [t["side"] for t in trades] == ["LONG", "SHORT"]
    assert Decimal(trades[1]["gross_pnl"]) < 0  # short, price rose to 100.5
    assert Decimal(trades[1]["cumulative_net_pnl"]) == Decimal(
        trades[0]["net_pnl"]
    ) + Decimal(trades[1]["net_pnl"])
    assert Decimal(summary["realised_net_pnl"]) == Decimal(
        trades[1]["cumulative_net_pnl"]
    )
    (held,) = report["open_positions"]
    assert held["symbol"] == "OPEN" and held["side"] == "SHORT"
    assert held["return_pct"] == 0.5  # sold at 100, now 99.5
    assert Decimal(held["net_pnl_if_closed_now"]) < Decimal(held["gross_pnl"])
    write_report(report, tmp_path / "report")
    for name in ("trades.csv", "open_positions.csv", "daily.csv", "summary.json"):
        assert (tmp_path / "report" / name).exists()
    text = (tmp_path / "report" / "report.txt").read_text()
    assert "Open portfolio:" in text and "1 at 15:00" in text
