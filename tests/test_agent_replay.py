from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from test_autonomous_agent import TODAY, at, reply, use, write_features

from ai_trader.agent.account import PaperAccount
from ai_trader.agent.market import NseDaily, parse_quote
from ai_trader.agent.replay import (
    LIMITS,
    MinuteBars,
    ReplayClock,
    ReplayFeed,
    ReplayMarket,
    ReplayTools,
    TimedClient,
    replay_day,
    stored_filings,
)
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.news.nse import Announcement
from ai_trader.news.store import NewsStore

PREVIOUS = date(2026, 10, 16)


def stamp(day: date, hour: int, minute: int) -> int:
    return int(
        datetime(
            day.year, day.month, day.day, hour, minute, tzinfo=INDIA_TIMEZONE
        ).timestamp()
    )


def write_bars(folder: Path, symbol: str, previous_close: float, start: float) -> None:
    """Yesterday's last bar closes at ``previous_close``; today minute m opens at
    start + 0.01 m, high +0.05, low -0.05, close +0.01, 100 shares."""
    folder.mkdir(parents=True, exist_ok=True)
    t = [stamp(PREVIOUS, 15, 29)]
    ohlc = [[previous_close] * 4]
    for m in range(375):
        t.append(stamp(TODAY, 9, 15) + 60 * m)
        o = start + 0.01 * m
        ohlc.append([o, o + 0.05, o - 0.05, o + 0.01])
    np.savez_compressed(
        folder / f"{symbol}.npz",
        t=np.array(t, dtype=np.int64),
        ohlc=np.array(ohlc, dtype=np.float64),
        v=np.full(len(t), 100.0),
    )


@pytest.fixture
def setup(tmp_path: Path):
    write_features(tmp_path / "features")
    daily = NseDaily(tmp_path / "features", TODAY)
    minutes = tmp_path / "minutes"
    write_bars(minutes, "AAA", 64.5, 65.0)
    write_bars(minutes, "BBB", 25.0, 25.0)
    write_bars(minutes, "NIFTY", 25000.0, 25100.0)
    return tmp_path, daily, minutes


def test_minute_bars_show_completed_minutes_and_the_current_open(setup) -> None:
    _, _, minutes = setup
    bars = MinuteBars(minutes, TODAY, ["AAA"], {"AAA": 64.5})
    assert bars.state("AAA", at(9, 14, 59)) is None
    first = bars.state("AAA", at(9, 15, 30))
    assert first["open"] == first["last"] == 65.0 and first["volume"] == 0
    assert first["vwap"] is None
    later = bars.state("AAA", at(9, 20, 10))
    assert later["last"] == pytest.approx(65.05)
    assert later["high"] == pytest.approx(65.09) and later["low"] == pytest.approx(
        64.95
    )
    assert later["volume"] == 500
    assert later["vwap"] == pytest.approx(
        np.mean([65.0 + 0.01 * m + 1 / 300 for m in range(5)])
    )
    assert bars.previous_close("AAA") == 64.5 and bars.rescaled == {}
    candles = bars.candles("AAA", at(9, 0), at(9, 17, 30))
    assert [c.timestamp for c in candles] == [at(9, 15), at(9, 16)]


def test_bars_after_a_later_split_are_put_on_the_replay_days_basis(setup) -> None:
    _, _, minutes = setup
    bars = MinuteBars(minutes, TODAY, ["BBB"], {"BBB": 50.0})
    assert bars.rescaled == {"BBB": 2.0}
    state = bars.state("BBB", at(9, 16, 5))
    assert state["last"] == pytest.approx(50.02) and state["volume"] == 50


def test_feed_quotes_use_nse_previous_close_and_no_book(setup) -> None:
    _, daily, minutes = setup
    clock = ReplayClock(at(9, 5))
    bars = MinuteBars(minutes, TODAY, ["AAA", "NIFTY"], {"AAA": 64.5})
    feed = ReplayFeed(bars, {"AAA": 64.5}, clock.now)
    pre = parse_quote("AAA", feed.quote_rows(["NSE:AAA"])["NSE:AAA"])
    assert pre.last == Decimal("64.5") and pre.open is None and pre.bids == ()
    clock.move_to(at(9, 20, 10))
    rows = feed.quote_rows(["NSE:AAA", "NSE:NIFTY 50", "NSE:ZZZ"])
    assert set(rows) == {"NSE:AAA", "NSE:NIFTY 50"}
    quote = parse_quote("AAA", rows["NSE:AAA"])
    assert quote.prev_close == Decimal("64.5") and quote.open == Decimal("65.0")
    assert rows["NSE:NIFTY 50"]["ohlc"]["close"] == 25000.0


def make_tools(tmp_path, daily, minutes, clock):
    bars = MinuteBars(
        minutes, TODAY, ["AAA", "BBB", "NIFTY"], {"AAA": 64.5, "BBB": 50.0}
    )
    market = ReplayMarket(
        ReplayFeed(bars, {"AAA": 64.5, "BBB": 50.0}, clock.now),
        ["AAA", "BBB"],
        {},
        daily,
        clock.now,
    )
    account = PaperAccount(tmp_path / "replay.sqlite", Decimal(2_000_000))
    return ReplayTools(market, account, bars), account


def test_orders_cross_a_modelled_book_by_liquidity(setup) -> None:
    tmp_path, daily, minutes = setup
    clock = ReplayClock(at(9, 20, 10))
    tools, _ = make_tools(tmp_path, daily, minutes, clock)
    quote = tools.market.quotes(["AAA"])["AAA"]
    book = tools.market.book(quote)
    assert tools.replay.half_spread("AAA") == 0.0008
    assert book.asks[0] == (Decimal("65.10"), 1)
    assert book.bids[-1][0] == Decimal("64.63")
    details = tools.market.stock_details(["AAA"])[0]["live"]
    assert "bids" not in details and details["best_ask"] is None


def test_exits_and_alerts_run_on_minute_highs_and_lows(setup) -> None:
    tmp_path, daily, minutes = setup
    clock = ReplayClock(at(9, 20, 10))
    tools, account = make_tools(tmp_path, daily, minutes, clock)
    bought = tools.run(
        "place_order",
        {
            "symbol": "AAA",
            "side": "BUY",
            "quantity": 1,
            "target_price": 65.17,
            "reason": "x",
        },
    )
    assert bought["ok"]
    # BBB's minute high is 50.10 + 0.02 m (rescaled x2): 50.4 is reached at 09:30.
    assert tools.run("set_alert", {"symbol": "BBB", "above": 50.4})["ok"]
    clock.move_to(at(9, 26, 30))
    assert tools.watch() == []
    clock.move_to(at(9, 28, 30))
    [target] = tools.watch()
    assert (target["event"], target["time"], target["exit_price"]) == (
        "target_filled",
        "09:27",
        65.17,
    )
    clock.move_to(at(9, 32, 30))
    [alert] = tools.watch()
    assert (alert["event"], alert["symbol"], alert["time"]) == ("alert", "BBB", "09:30")
    assert account.open_positions() == [] and tools.alerts == {}


def test_a_gapped_stop_fills_at_the_open_plus_the_spread(setup) -> None:
    tmp_path, daily, minutes = setup
    clock = ReplayClock(at(9, 20, 10))
    tools, _ = make_tools(tmp_path, daily, minutes, clock)
    tools.run(
        "place_order",
        {
            "symbol": "AAA",
            "side": "SELL_SHORT",
            "quantity": 1,
            "stop_price": 65.055,
            "reason": "x",
        },
    )
    clock.move_to(at(9, 30, 30))
    [stop] = tools.watch()
    # The 09:21 minute opens at 65.06, above the stop: the short is covered there.
    assert stop["time"] == "09:21"
    assert stop["exit_price"] == pytest.approx(65.06 * 1.0008, abs=1e-4)


def test_timed_client_moves_the_clock_by_the_response_time(monkeypatch) -> None:
    clock = ReplayClock(at(10, 0))
    ticks = iter([100.0, 112.5])
    monkeypatch.setattr("ai_trader.agent.replay.wall.monotonic", lambda: next(ticks))

    class Echo:
        def send(self, body):
            return {"ok": True}

    assert TimedClient(Echo(), clock).send({}) == {"ok": True}
    assert clock.now() == at(10, 0, 12) + timedelta(seconds=0.5)


def test_stored_filings_are_read_by_dissemination_time(tmp_path) -> None:
    store = NewsStore(tmp_path / "news.sqlite")
    store.add_announcements(
        [
            Announcement(
                "k1", "AAA", "Aaa Ltd", "Outcome", "results", "", at(9, 40), {}
            ),
            Announcement(
                "k2",
                "BBB",
                "Bbb Ltd",
                "Press",
                "news",
                "",
                datetime(2026, 10, 18, 20, 0, tzinfo=INDIA_TIMEZONE),
                {},
            ),
        ]
    )
    store.close()
    found = stored_filings(tmp_path / "news.sqlite")(date(2026, 10, 18), TODAY)
    assert sorted((a.symbol, a.announced_at) for a in found) == [
        ("AAA", at(9, 40)),
        ("BBB", datetime(2026, 10, 18, 20, 0, tzinfo=INDIA_TIMEZONE)),
    ]


class ScriptedClaude:
    """Scripted replies, then finish_turn whenever the script is used up."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def send(self, body):
        self.bodies.append(copy.deepcopy(body))
        if self.replies:
            return self.replies.pop(0)
        return reply(
            use(
                f"f{len(self.bodies)}",
                "finish_turn",
                notes="wait",
                next_check_minutes=30,
            )
        )


def test_a_replayed_day_runs_the_live_timeline_without_web_search(setup) -> None:
    tmp_path, daily, minutes = setup
    news = NewsStore(tmp_path / "news.sqlite")
    news.close()
    claude = ScriptedClaude(
        [
            reply(
                use(
                    "o1",
                    "place_order",
                    symbol="AAA",
                    side="BUY",
                    quantity=10,
                    reason="open",
                ),
                use(
                    "f1",
                    "finish_turn",
                    notes="Long AAA at the open.",
                    next_check_minutes=30,
                ),
            )
        ]
    )
    summary = replay_day(
        TODAY,
        claude,
        tmp_path / "out",
        minutes=minutes,
        features=tmp_path / "features",
        scan=["AAA", "BBB"],
        profiles={},
        news_db=tmp_path / "news.sqlite",
        model="claude-opus-5-5",
        effort="medium",
        budget=5.0,
        wallet=Decimal(2_000_000),
        leverage=5,
        log=lambda _: None,
        stop_at=datetime(2026, 10, 19, 9, 30).time(),
        daily=daily,
    )
    first = claude.bodies[0]
    assert all(t.get("name") != "web_search" for t in first["tools"])
    assert LIMITS in first["system"][0]["text"]
    assert "Web search is not available today." in first["system"][0]["text"]
    brief = json.loads(first["messages"][0]["content"][0]["text"])
    assert brief["time"].startswith("2026-10-19 09:00:00")
    whys = [
        json.loads(b["messages"][-1]["content"][0]["text"])["why_now"]
        for b in claude.bodies[1:2]
    ]
    assert whys == ["your opening orders were filled"]
    [trade] = summary["closed"]
    assert trade["entry_price"] == 65.0 and trade["exit_reason"] == "time_1510"
    assert trade["exited"] == "09:30"
    assert summary["trades"] == 1 and summary["nifty_change_pct"] == pytest.approx(
        (25100 + 3.74 + 0.01) / 25000 * 100 - 100, abs=1e-3
    )
    saved = json.loads((tmp_path / "out" / "report.json").read_text("utf-8"))
    assert saved["net_rs"] == summary["net_rs"]
    with pytest.raises(FileExistsError):
        replay_day(
            TODAY,
            claude,
            tmp_path / "out",
            minutes=minutes,
            features=tmp_path / "features",
            scan=["AAA"],
            profiles={},
            news_db=tmp_path / "news.sqlite",
            model="claude-opus-5-5",
            effort="medium",
            budget=5.0,
            wallet=Decimal(2_000_000),
            leverage=5,
            log=lambda _: None,
            daily=daily,
        )
