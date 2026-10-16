from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ai_trader.agent.account import Book, PaperAccount, walk
from ai_trader.agent.brain import AgentTools, AutonomousAgent, TurnResult, spent_today
from ai_trader.agent.market import (
    STOCK_FACTS,
    Filings,
    MarketView,
    NseDaily,
    parse_quote,
)
from ai_trader.broker import OHLCVCandle
from ai_trader.cli.run_agent import SessionRunner, close_everything
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.news.nse import Announcement
from ai_trader.paper.charges import trade_charges

TODAY = date(2026, 10, 19)


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 10, 19, hour, minute, second, tzinfo=INDIA_TIMEZONE)


class Clock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def quote_row(
    last: float,
    *,
    prev: float | None = None,
    open_: float | None = None,
    vwap: float | None = None,
    volume: int = 1000,
    bids: tuple[tuple[float, int], ...] | None = None,
    asks: tuple[tuple[float, int], ...] | None = None,
) -> dict[str, Any]:
    bids = bids if bids is not None else ((round(last - 0.1, 2), 100),)
    asks = asks if asks is not None else ((round(last + 0.1, 2), 100),)
    return {
        "last_price": last,
        "ohlc": {
            "open": open_ or last,
            "high": max(last, open_ or last),
            "low": min(last, open_ or last),
            "close": prev or last,
        },
        "average_price": vwap or last,
        "volume": volume,
        "buy_quantity": 500,
        "sell_quantity": 700,
        "depth": {
            "buy": [{"price": p, "quantity": q, "orders": 1} for p, q in bids],
            "sell": [{"price": p, "quantity": q, "orders": 1} for p, q in asks],
        },
        "lower_circuit_limit": round(last * 0.8, 2),
        "upper_circuit_limit": round(last * 1.2, 2),
        "last_trade_time": at(10, 0, 5),
    }


class FakeKite:
    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows
        self.candles: dict[str, tuple[OHLCVCandle, ...]] = {}
        self.calls: list[list[str]] = []

    def quote_rows(self, keys):
        self.calls.append(list(keys))
        return {k: self.rows[k] for k in keys if k in self.rows}

    def get_historical_candles(self, instrument, start, end, interval):
        return self.candles.get(instrument.trading_symbol, ())


def write_features(folder: Path) -> None:
    """AAA: F&O, rises 100 -> 129 with a 1:2 split on day 20; BBB: flat 50."""
    folder.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(end=TODAY - timedelta(days=3), periods=30)
    rows = []
    for i, day in enumerate(days):
        factor = 0.5 if i >= 20 else 1.0
        close = (100 + i) * factor
        prev = (99 + i) * (0.5 if i >= 21 else 1.0) if i != 20 else 119 * 0.5
        if i == 0:
            prev = 99.0
        aaa = {
            "date": day.date().isoformat(),
            "symbol": "AAA",
            "series": "EQ",
            "cm_prev_close": prev,
            "cm_open": close,
            "cm_high": close * 1.01,
            "cm_low": close * 0.99,
            "cm_close": close,
            "cm_vwap": close,
            "cm_volume": 1000 / factor,
            "cm_turnover_lakh": 1.0,
            "cm_trades": 10,
            "cm_deliv_pct": 40.0,
        } | {name: 1.5 for name in STOCK_FACTS}
        aaa |= {"fno": True, "in_ban": False, "fut_buildup": "long_buildup"}
        bbb = dict(aaa) | {
            "symbol": "BBB",
            "cm_prev_close": 50.0,
            "cm_open": 50.0,
            "cm_high": 50.5,
            "cm_low": 49.5,
            "cm_close": 50.0,
            "cm_vwap": 50.0,
            "cm_volume": 500,
            "fno": False,
        }
        rows += [aaa, bbb]
    pd.DataFrame(rows).to_csv(folder / "stock_daily_2026.csv.gz", index=False)


@pytest.fixture
def daily(tmp_path: Path) -> NseDaily:
    write_features(tmp_path / "features")
    return NseDaily(tmp_path / "features", TODAY)


def make_market(daily: NseDaily, clock: Clock, rows=None, filings=None) -> tuple:
    kite = FakeKite(
        rows
        or {
            "NSE:AAA": quote_row(66.0, prev=64.5, open_=65.0, volume=4000),
            "NSE:BBB": quote_row(49.0, prev=50.0, open_=50.5, volume=100),
            "NSE:NIFTY 50": {"last_price": 25100, "ohlc": {"close": 25000}},
        }
    )
    market = MarketView(
        kite,
        ["AAA", "BBB"],
        {"AAA": {"company": "Aaa Ltd", "industry": "Cement"}},
        daily,
        clock,
        filings=filings,
        quote_ttl=0.0,
    )
    return kite, market


# --- account -------------------------------------------------------------------------
def book(last: str, bids=(), asks=()) -> Book:
    def levels(pairs):
        return tuple((Decimal(p), q) for p, q in pairs)

    return Book(Decimal(last), levels(bids), levels(asks))


def test_walk_takes_levels_in_order_and_beyond_the_book_at_the_last_level() -> None:
    levels = ((Decimal("10"), 5), (Decimal("11"), 5))
    assert walk(levels, 5) == Decimal("10")
    assert walk(levels, 10) == Decimal("10.5")
    assert walk(levels, 20) == Decimal("10.75")
    assert walk((), 1) is None


def test_open_fills_at_the_book_within_leveraged_buying_power(tmp_path) -> None:
    account = PaperAccount(tmp_path / "p.sqlite", Decimal(1000), leverage=5)
    quotes = book("100", bids=[("99", 100)], asks=[("100", 30), ("101", 100)])
    long = account.open("AAA", "LONG", 40, quotes, at(10, 0))
    assert long["ok"] and long["fill_price"] == pytest.approx(100.25)
    assert account.buying_power() == Decimal(5000) - Decimal("4010")
    refused = account.open("AAA", "LONG", 10, quotes, at(10, 1))
    assert not refused["ok"] and "buying power" in refused["error"]
    short = account.open("BBB", "SHORT", 9, quotes, at(10, 2))
    assert short["ok"] and short["fill_price"] == 99.0


def test_open_checks_whole_shares_side_and_exit_levels(tmp_path) -> None:
    account = PaperAccount(tmp_path / "p.sqlite", Decimal(100_000))
    quotes = book("100", bids=[("99.9", 100)], asks=[("100.1", 100)])
    assert not account.open("AAA", "LONG", 0, quotes, at(10, 0))["ok"]
    assert not account.open("AAA", "SIDEWAYS", 1, quotes, at(10, 0))["ok"]
    wrong = account.open("AAA", "LONG", 1, quotes, at(10, 0), stop=Decimal(101))
    assert not wrong["ok"]
    wrong = account.open("AAA", "SHORT", 1, quotes, at(10, 0), target=Decimal(101))
    assert not wrong["ok"]
    empty = account.open("AAA", "LONG", 1, book("100", bids=[("99", 1)]), at(10, 0))
    assert not empty["ok"]


def test_close_charges_zerodha_intraday_and_updates_equity(tmp_path) -> None:
    account = PaperAccount(tmp_path / "p.sqlite", Decimal(1_000_000))
    account.open("AAA", "SHORT", 100, book("100", bids=[("100", 1000)]), at(10, 0))
    closed = account.close(
        "AAA", book("95", asks=[("95", 1000)]), at(11, 0), "agent: cover"
    )
    charges = trade_charges(Decimal(9500), Decimal(10000), same_session=True).total
    assert closed[0]["net_rs"] == pytest.approx(float(500 - charges), abs=0.01)
    assert account.equity() == Decimal(1_000_000) + 500 - charges
    assert account.open_positions() == []


def test_check_exits_fills_targets_at_the_target_and_stops_at_the_book(
    tmp_path,
) -> None:
    account = PaperAccount(tmp_path / "p.sqlite", Decimal(1_000_000))
    entry = book("100", bids=[("100", 1000)], asks=[("100", 1000)])
    account.open("AAA", "LONG", 10, entry, at(10, 0), target=Decimal(105))
    account.open("AAA", "LONG", 10, entry, at(10, 0), stop=Decimal(90))
    account.open("BBB", "SHORT", 10, entry, at(10, 0), stop=Decimal(103))
    moved = {
        "AAA": book("106", bids=[("105.9", 1000)], asks=[("106.1", 1000)]),
        "BBB": book("104", bids=[("103.9", 1000)], asks=[("104.2", 1000)]),
    }
    closed = account.check_exits(moved, at(11, 0))
    assert [(c["symbol"], c["reason"], c["exit_price"]) for c in closed] == [
        ("AAA", "target", 105.0),
        ("BBB", "stop", 104.2),
    ]
    assert [(r["symbol"], r["stop"]) for r in account.open_positions()] == [
        ("AAA", "90")
    ]


def test_opening_orders_fill_at_the_opening_price_or_are_cancelled(tmp_path) -> None:
    account = PaperAccount(tmp_path / "p.sqlite", Decimal(1_000_000))
    account.queue_opening("AAA", "LONG", 10, at(9, 5), target=Decimal(120))
    account.queue_opening("BBB", "SHORT", 5, at(9, 6))
    account.queue_opening("CCC", "LONG", 5, at(9, 6))
    assert account.cancel_pending("BBB") == 1
    fills = account.fill_opening({"AAA": Decimal("101.5")}, at(9, 15, 20))
    assert [(f["symbol"], f["ok"]) for f in fills] == [("AAA", True), ("CCC", False)]
    [position] = account.open_positions()
    assert Decimal(position["entry_price"]) == Decimal("101.5")
    assert position["target"] == "120"
    assert account.pending_orders() == []


# --- market --------------------------------------------------------------------------
def test_parse_quote_reads_kite_rows() -> None:
    quote = parse_quote("AAA", quote_row(101.0, prev=100.0, open_=99.0, vwap=100.5))
    assert quote is not None
    assert quote.last == Decimal("101.0") and quote.prev_close == Decimal("100.0")
    assert quote.change_pct() == pytest.approx(1.0)
    assert quote.bids == ((Decimal("100.9"), 100),)
    facts = quote.facts()
    assert facts["gap_pct"] == pytest.approx(-1.0)
    assert facts["spread_pct"] == pytest.approx(0.198, abs=1e-3)
    assert facts["last_trade"] == "10:00:05"
    assert parse_quote("AAA", {"last_price": 0}) is None


def test_nse_daily_adjusts_for_a_split_with_nse_previous_close(daily) -> None:
    stats = daily.stats("AAA")
    assert stats["close"] == pytest.approx(64.5)
    assert stats["return_pct"]["20d"] == pytest.approx((129 / 109 - 1) * 100, abs=1e-3)
    assert stats["avg_volume_20d"] == pytest.approx(2000)
    bars = daily.bars("AAA", 30)
    assert bars[0]["close"] == pytest.approx(50.0)
    assert bars[0]["volume"] == 2000
    assert daily.facts("AAA")["fut_buildup"] == "long_buildup"
    assert daily.facts("BBB") == {"date": bars[-1]["date"], "fno": False}
    assert daily.stats("ZZZ") is None


def test_movers_rank_the_scan_list(daily) -> None:
    _, market = make_market(daily, Clock(at(10, 0)))
    gainers = market.movers("gainers", 5)["rows"]
    assert [r["symbol"] for r in gainers] == ["AAA", "BBB"]
    assert gainers[0]["volume_vs_20d_avg"] == pytest.approx(2.0)
    assert [r["symbol"] for r in market.movers("down_from_open")["rows"]] == [
        "BBB",
        "AAA",
    ]
    with pytest.raises(ValueError):
        market.movers("random")


def test_overview_has_indices_breadth_and_industries(daily) -> None:
    _, market = make_market(daily, Clock(at(10, 0)))
    view = market.overview()
    assert view["indices"]["NIFTY 50"]["change_pct"] == pytest.approx(0.4)
    assert view["breadth_scan_list"]["advancing"] == 1
    assert view["breadth_scan_list"]["declining"] == 1
    assert view["session"] == "open"


def test_stock_details_and_unknown_symbols(daily) -> None:
    _, market = make_market(daily, Clock(at(10, 0)))
    aaa, zzz = market.stock_details(["aaa", "ZZZ"])
    assert aaa["company"] == "Aaa Ltd" and aaa["live"]["last"] == 66.0
    assert aaa["daily_nse"]["close"] == pytest.approx(64.5)
    assert zzz == {"symbol": "ZZZ", "error": "unknown NSE symbol"}


def test_minute_bars_are_grouped_from_the_session_open(daily) -> None:
    kite, market = make_market(daily, Clock(at(9, 30)))
    kite.candles["AAA"] = tuple(
        OHLCVCandle(
            at(9, 15 + i),
            Decimal(100 + i),
            Decimal(101 + i),
            Decimal(99),
            Decimal(100 + i),
            10,
        )
        for i in range(12)
    )
    bars = market.bars("AAA", "5minute", 10)["bars"]
    assert [b["time"][-5:] for b in bars] == ["09:15", "09:20", "09:25"]
    assert bars[0] == {
        "time": "2026-10-19 09:15",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "volume": 50,
    }
    assert market.bars("AAA", "day", 2)["bars"][-1]["close"] == pytest.approx(64.5)


def test_filings_refresh_and_never_show_the_future(daily) -> None:
    clock = Clock(at(10, 0))
    calls = []

    def item(symbol: str, moment: datetime) -> Announcement:
        return Announcement(
            symbol, symbol, f"{symbol} Ltd", "Outcome", "text", "", moment, {}
        )

    def fetch(start, end):
        calls.append((start, end))
        return [item("AAA", at(9, 30)), item("BBB", at(10, 2))]

    filings = Filings(fetch, clock)
    _, market = make_market(daily, clock, filings=filings)
    assert [f["symbol"] for f in market.nse_filings(None, 24)["filings"]] == ["AAA"]
    assert market.nse_filings("BBB", 24)["count"] == 0
    clock.moment = at(10, 3)
    assert market.nse_filings("BBB", 24)["count"] == 1
    assert len(calls) == 1
    clock.moment = at(10, 6)
    market.nse_filings()
    assert len(calls) == 2


# --- the agent -----------------------------------------------------------------------
class FakeClaude:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []

    def send(self, body):
        self.bodies.append(copy.deepcopy(body))
        return self.responses.pop(0)


def reply(*blocks: dict[str, Any], stop: str = "tool_use") -> dict[str, Any]:
    return {
        "content": list(blocks),
        "stop_reason": stop,
        "usage": {"input_tokens": 1000, "output_tokens": 100},
    }


def use(ident: str, name: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": ident, "name": name, "input": args}


def make_agent(tmp_path, daily, clock, responses, *, budget=30.0, spent=0.0):
    _, market = make_market(daily, clock)
    account = PaperAccount(tmp_path / "day" / "paper.sqlite", Decimal(2_000_000))
    tools = AgentTools(market, account)
    claude = FakeClaude(responses)
    agent = AutonomousAgent(
        claude,
        "claude-opus-5-5",
        "medium",
        tools,
        budget_usd=budget,
        spent_usd=spent,
        journal_dir=tmp_path / "day",
        log=lambda _: None,
    )
    return agent, claude, tools, account


def test_a_turn_runs_tools_trades_and_compacts_its_history(tmp_path, daily) -> None:
    clock = Clock(at(10, 0))
    agent, claude, _, account = make_agent(
        tmp_path,
        daily,
        clock,
        [
            reply(
                {"type": "text", "text": "Scanning."},
                use("t1", "top_movers", sort="gainers", count=2),
            ),
            reply(
                use(
                    "t2",
                    "place_order",
                    symbol="AAA",
                    side="BUY",
                    quantity=10,
                    stop_price=60,
                    target_price=70,
                    reason="leader",
                ),
                use(
                    "t3",
                    "finish_turn",
                    notes="Long AAA, stop 60.",
                    next_check_minutes=7,
                ),
            ),
        ],
    )
    result = agent.turn("first turn", [])
    assert result.next_check_minutes == 7 and result.notes == "Long AAA, stop 60."
    assert result.tools == ["top_movers", "place_order", "finish_turn"]
    assert result.cost_usd == pytest.approx(2 * (1000 * 4 + 100 * 20) / 1e6)
    [position] = account.open_positions()
    assert (position["symbol"], position["quantity"], position["entry_price"]) == (
        "AAA",
        10,
        "66.1",
    )
    second = claude.bodies[1]
    assert second["output_config"] == {"effort": "medium"}
    assert second["thinking"] == {"type": "adaptive"}
    assert any(t.get("name") == "web_search" for t in second["tools"])
    results = second["messages"][-1]["content"]
    assert results[0]["tool_use_id"] == "t1"
    marked = [
        block
        for message in second["messages"]
        if message["role"] == "user"
        for block in message["content"]
        if "cache_control" in block
    ]
    assert marked == [results[-1]]
    brief = json.loads(second["messages"][0]["content"][0]["text"])
    assert brief["why_now"] == "first turn" and brief["account"]["equity_rs"] == 2e6
    assert [m["role"] for m in agent.history] == ["user", "assistant"]
    assert "Long AAA" in agent.history[1]["content"][0]["text"]
    journal = (tmp_path / "day" / "turns.jsonl").read_text("utf-8").splitlines()
    assert json.loads(journal[0])["trades"][0]["result"]["ok"] is True
    assert (tmp_path / "day" / "transcripts" / "turn_001.json").exists()
    assert spent_today(tmp_path / "day") == pytest.approx(result.cost_usd)


def test_paused_turns_resume_and_text_only_answers_end_the_turn(
    tmp_path, daily
) -> None:
    agent, claude, _, _ = make_agent(
        tmp_path,
        daily,
        Clock(at(10, 0)),
        [
            reply(
                {
                    "type": "server_tool_use",
                    "id": "s1",
                    "name": "web_search",
                    "input": {"query": "nifty today"},
                },
                stop="pause_turn",
            ),
            reply(
                {"type": "text", "text": "Nothing worth trading yet."}, stop="end_turn"
            ),
        ],
    )
    result = agent.turn("scheduled", [])
    assert result.searches == ["nifty today"]
    assert (
        result.notes == "Nothing worth trading yet." and result.next_check_minutes == 5
    )
    assert claude.bodies[1]["messages"][-1]["role"] == "assistant"


def test_no_request_once_the_budget_is_spent(tmp_path, daily) -> None:
    agent, claude, _, _ = make_agent(
        tmp_path, daily, Clock(at(10, 0)), [], budget=1.0, spent=1.0
    )
    assert agent.turn("scheduled", []).stopped == "budget"
    assert claude.bodies == []


def test_pre_open_orders_wait_for_the_opening_price(tmp_path, daily) -> None:
    clock = Clock(at(9, 5))
    _, _, tools, account = make_agent(tmp_path, daily, clock, [])
    queued = tools.run(
        "place_order", {"symbol": "AAA", "side": "BUY", "quantity": 100, "reason": "x"}
    )
    assert queued["ok"] and account.open_positions() == []
    too_big = tools.run(
        "place_order",
        {"symbol": "BBB", "side": "SELL_SHORT", "quantity": 10**6, "reason": "x"},
    )
    assert not too_big["ok"]
    clock.moment = at(9, 15, 20)
    [fill] = tools.fill_opening()
    assert fill["ok"] and fill["fill_price"] == 65.0


def test_watch_fills_stops_and_targets_and_fires_alerts(tmp_path, daily) -> None:
    clock = Clock(at(10, 0))
    _, _, tools, account = make_agent(tmp_path, daily, clock, [])
    tools.run(
        "place_order",
        {
            "symbol": "AAA",
            "side": "BUY",
            "quantity": 10,
            "target_price": 67,
            "reason": "x",
        },
    )
    assert tools.run(
        "set_alert", {"symbol": "BBB", "below": 48.5, "note": "breakdown"}
    )["ok"]
    assert not tools.run("set_alert", {"symbol": "AAA", "above": 65})["ok"]
    assert tools.watch() == []
    rows = tools.market._kite.rows
    rows["NSE:AAA"] = quote_row(67.2, prev=64.5, open_=65.0)
    rows["NSE:BBB"] = quote_row(48.4, prev=50.0, open_=50.5)
    events = tools.watch()
    assert [e["event"] for e in events] == ["target_filled", "alert"]
    assert events[0]["exit_price"] == 67.0
    assert tools.alerts == {} and account.open_positions() == []


def test_exits_must_be_on_the_right_side_and_trading_ends_at_1510(
    tmp_path, daily
) -> None:
    clock = Clock(at(10, 0))
    _, _, tools, _ = make_agent(tmp_path, daily, clock, [])
    tools.run(
        "place_order", {"symbol": "AAA", "side": "BUY", "quantity": 1, "reason": "x"}
    )
    assert not tools.run("set_exits", {"symbol": "AAA", "stop_price": 66.5})["ok"]
    assert tools.run("set_exits", {"symbol": "AAA", "stop_price": 64})["ok"]
    clock.moment = at(15, 10)
    late = tools.run(
        "place_order", {"symbol": "BBB", "side": "BUY", "quantity": 1, "reason": "x"}
    )
    assert not late["ok"] and "over" in late["error"]
    [closed] = tools.close_all("time_1510")
    assert closed["reason"] == "time_1510" and closed["exit_price"] == 65.9


# --- the session runner ------------------------------------------------------------
class StubAgent:
    def __init__(self) -> None:
        self.turns = 0
        self.spent = 0.0
        self.calls: list[tuple[str, list]] = []
        self.fail = False

    def budget_left(self) -> float:
        return 30.0 - self.spent

    def turn(self, why, events) -> TurnResult:
        if self.fail:
            raise RuntimeError("Claude is down")
        self.turns += 1
        self.calls.append((why, events))
        return TurnResult(notes="n", next_check_minutes=10, cost_usd=0.1, steps=1)


def test_the_runner_takes_turns_when_due_and_on_events(tmp_path, daily) -> None:
    clock = Clock(at(8, 50))
    _, market = make_market(daily, clock)
    tools = AgentTools(market, PaperAccount(tmp_path / "p.sqlite", Decimal(10**6)))
    agent = StubAgent()
    runner = SessionRunner(tools, agent, lambda _: None, clock())
    assert runner.step(clock()) == 5.0 and agent.turns == 0
    clock.moment = at(9, 0)
    assert runner.step(clock()) == 0.0
    assert agent.calls[0] == ("first turn of the day: pre-open, plan the session", [])
    clock.moment = at(9, 5)
    assert runner.step(clock()) == 5.0 and agent.turns == 1
    tools.run(
        "place_order", {"symbol": "AAA", "side": "BUY", "quantity": 5, "reason": "x"}
    )
    clock.moment = at(9, 15, 20)
    runner.step(clock())
    why, events = agent.calls[1]
    assert why == "your opening orders were filled"
    assert events[0]["event"] == "opening_fills" and events[0]["fills"][0]["ok"]
    clock.moment = at(9, 16)
    runner.add_events([{"event": "alert", "symbol": "BBB"}])
    runner.step(clock())
    assert agent.calls[2] == (
        "an alert, stop or target triggered",
        [{"event": "alert", "symbol": "BBB"}],
    )
    assert runner.next_turn == at(9, 26)
    agent.fail = True
    clock.moment = at(9, 26)
    runner.add_events([{"event": "stop_filled"}])
    assert runner.step(clock()) == 0.0
    assert runner.events == [{"event": "stop_filled"}]
    assert runner.next_turn == at(9, 28)
    agent.fail = False
    clock.moment = at(15, 7)
    runner.step(clock())
    assert agent.turns == 3
    clock.moment = at(15, 10)
    assert runner.step(clock()) is None


def test_no_turn_once_the_budget_is_spent_and_a_late_start_skips_the_auction(
    tmp_path, daily
) -> None:
    clock = Clock(at(11, 0))
    _, market = make_market(daily, clock)
    tools = AgentTools(market, PaperAccount(tmp_path / "p.sqlite", Decimal(10**6)))
    agent = StubAgent()
    runner = SessionRunner(tools, agent, lambda _: None, clock())
    assert runner.opened and runner.why == "first turn of the day"
    agent.spent = 30.0
    assert runner.step(clock()) == 1.0 and agent.turns == 0


def test_close_everything_retries_until_flat(tmp_path, daily) -> None:
    clock = Clock(at(15, 10))
    _, market = make_market(daily, clock)
    account = PaperAccount(tmp_path / "p.sqlite", Decimal(10**6))
    tools = AgentTools(market, account)
    account.open("AAA", "LONG", 3, market.quotes(["AAA"])["AAA"].book(), at(15, 0))
    real, calls = tools.close_all, []

    def flaky(reason):
        calls.append(reason)
        if len(calls) == 1:
            raise ConnectionError("quote failed")
        return real(reason)

    tools.close_all = flaky
    lines: list[str] = []
    closed = close_everything(tools, lines.append, pause=0)
    assert len(calls) == 2 and closed[0]["symbol"] == "AAA"
    assert account.open_positions() == []
    assert "15:10 close failed (ConnectionError: quote failed)" in lines
