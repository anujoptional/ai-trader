"""End-to-end paper session on a small synthetic market (offline replay).

Three stocks are bought at 09:40 by a BUY-everything stand-in: UPX then rises
past +1% (sold at the limit), DNX falls past -1% (stopped), FLT stays flat, is
carried overnight and gaps up at the next open (sold at the open, delivery
charges). The scanner model is a stub; its inputs are the real ones.
"""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from ai_trader.ai.decision import CycleResult, DecisionRequest, Verdict, request_json
from ai_trader.ai.journal import DecisionJournal
from ai_trader.broker import (
    CandleInterval,
    DepthLevel,
    DepthSnapshot,
    Instrument,
)
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.paper.broker import PaperBroker, PaperConfig
from ai_trader.paper.replay import ReplayData
from ai_trader.paper.session import (
    Book,
    PaperSession,
    ReplayClock,
    SessionSettings,
    _array,
)
from ai_trader.paper.store import PaperStore
from ai_trader.scanner.upside import ScannerBundle

STOCKS = ("UPX", "DNX", "FLT")
# The production model's input contract; the model itself is stubbed.
PRODUCTION = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "data/models/scanner_up1_by_close_v1/bundle.json"
    ).read_text("utf-8")
)


def stub_bundle() -> ScannerBundle:
    return ScannerBundle(
        "stub",
        tuple(PRODUCTION["features"]),
        frozenset(PRODUCTION["optional_features"]),
        frozenset(PRODUCTION["percent_features"]),
        {"chosen": "raw"},
        Model(),
    )


class BuyAll:
    def __init__(self, only: str | None = None) -> None:
        self.only = only
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> CycleResult:
        self.requests.append(request)
        return CycleResult(
            tuple(
                Verdict(
                    c.symbol,
                    "BUY" if self.only in (None, c.symbol) else "NO_BUY",
                    0.6,
                    "test",
                    "t",
                )
                for c in request.candidates
            ),
            "ok",
            "t",
            request_json=request_json(request),
            prompt_version="t",
        )


class Model:
    tree_count_ = 1

    def predict_proba(self, rows):
        return np.column_stack((np.full(len(rows), 0.3), np.full(len(rows), 0.7)))


class ReplayLive:
    """Live-bars stand-in backed by the replay's bars (always ticking)."""

    def __init__(self, data, day: date) -> None:
        self.data, self.day = data, day

    def bars(self, symbol: str, upto: int) -> np.ndarray:
        opening = datetime(*self.day.timetuple()[:3], 9, 15, tzinfo=INDIA_TIMEZONE)
        candles = self.data.get_historical_candles(
            Instrument("NSE", symbol),
            opening,
            opening + timedelta(minutes=upto + 1),
            CandleInterval.ONE_MINUTE,
        )
        return _array(candles, self.day, upto + 1)

    def complete(self, symbol: str, upto: int) -> bool:
        return bool(np.isfinite(self.bars(symbol, upto)[:, 3]).all())

    def silent_seconds(self) -> float:
        return 0.0

    def seed(self, bars, upto: int) -> None:
        raise AssertionError("live bars were usable; no full fetch expected")


def weekdays(start: date, count: int) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def minute_rows(day: date, path: np.ndarray) -> list[str]:
    rows = []
    for i, close in enumerate(path):
        opening = path[i - 1] if i else close
        moment = datetime(day.year, day.month, day.day, 9, 15) + timedelta(minutes=i)
        high, low = max(opening, close) * 1.0002, min(opening, close) * 0.9998
        rows.append(
            f"{moment:%Y-%m-%dT%H:%M:%S}+05:30,{opening:.2f},{high:.2f},{low:.2f},"
            f"{close:.2f},1000"
        )
    return rows


def write_market(root, days: list[date], test_days: list[date]) -> None:
    (root / "daily").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for symbol, level in (("NIFTY", 25000.0), ("INDIAVIX", 12.0)) + tuple(
        (s, 100.0) for s in STOCKS
    ):
        minutes, daily = [], []
        close = level
        for day in days:
            path = close * np.exp(np.cumsum(rng.normal(0, 0.0003, 375)))
            if day == test_days[0] and symbol in STOCKS:
                path = np.full(375, close)
                if symbol == "UPX":  # +2% from 10:00
                    path[45:] = close * 1.02
                if symbol == "DNX":  # -2% from 10:00
                    path[45:] = close * 0.98
            if day == test_days[1] and symbol == "FLT":
                path = np.full(375, close * 1.03)  # gaps above +1% at the open
            if day in test_days or day >= days[-45]:
                minutes += minute_rows(day, path)
            daily.append(
                f"{day:%Y-%m-%d}T00:00:00+05:30,{path[0]:.2f},{path.max():.2f},"
                f"{path.min():.2f},{path[-1]:.2f},{375000}"
            )
            close = float(path[-1])
        header = "start_time,open,high,low,close,volume\n"
        (root / f"NSE_{symbol}.csv").write_text(header + "\n".join(minutes) + "\n")
        (root / "daily" / f"NSE_{symbol}.csv").write_text(
            header.replace("start_time", "timestamp") + "\n".join(daily) + "\n"
        )


@pytest.mark.parametrize("mode", ["full", "shadow", "live"])
def test_paper_session_buys_sells_carries_and_reports(tmp_path, mode) -> None:
    with_shadow = mode == "shadow"
    days = weekdays(date(2025, 1, 1), 300)
    test_days = days[-2:]
    write_market(tmp_path / "candles", days, test_days)
    bundle = stub_bundle()
    store = PaperStore(tmp_path / "paper.sqlite")
    broker = PaperBroker(store, PaperConfig())
    decider = BuyAll()
    journal = DecisionJournal(tmp_path / "decisions.sqlite")
    shadow = BuyAll(only="UPX")
    shadows = (
        [
            Book(
                "upx-only",
                shadow,
                DecisionJournal(tmp_path / "shadow" / "decisions.sqlite"),
                PaperBroker(
                    PaperStore(tmp_path / "shadow" / "paper.sqlite"), PaperConfig()
                ),
                tmp_path / "shadow" / "report",
                "[upx] ",
            )
        ]
        if with_shadow
        else []
    )
    clock = ReplayClock(datetime(2025, 1, 1, tzinfo=INDIA_TIMEZONE))
    data = ReplayData(tmp_path / "candles", clock, first_day=days[-60])
    logs: list[str] = []
    sessions = []
    for day in test_days:
        clock.current = datetime(
            day.year, day.month, day.day, 9, 0, tzinfo=INDIA_TIMEZONE
        )
        session = PaperSession(
            data,
            clock,
            bundle,
            STOCKS,
            broker,
            decider,
            "buy-all",
            journal,
            report_dir=tmp_path / "report",
            settings=SessionSettings(workers=1),
            log=logs.append,
            shadows=shadows,
            capture_dir=tmp_path / "capture",
            live=ReplayLive(data, day) if mode == "live" else None,
        )
        session.prepare()
        sessions.append(session)
        session.run(
            datetime(day.year, day.month, day.day, 15, 31, tzinfo=INDIA_TIMEZONE)
        )

    first = decider.requests[0]
    assert first.as_of == datetime(
        *test_days[0].timetuple()[:3], 9, 40, 5, tzinfo=INDIA_TIMEZONE
    )
    assert {c.symbol for c in first.candidates} == set(STOCKS)
    stocks = first.document["stocks"]
    assert [s["symbol"] for s in stocks] == sorted(STOCKS)
    assert "minutes_to_close" in stocks[0]["today"] and stocks[0]["today"]["last_price"]
    sent = request_json(first)
    assert "p_up" not in sent and "probab" not in sent and "scanner" not in sent
    folder = tmp_path / "capture" / test_days[0].isoformat()
    scan_file = json.loads((folder / "0940.json").read_text("utf-8"))
    assert len(scan_file["stocks"]["UPX"]["bars"]) == 25
    # Multi-day values are at the decision price, not at the last close.
    daily = sessions[0].histories["UPX"].daily
    price = scan_file["stocks"]["UPX"]["bars"][24][3]
    upx = next(s for s in stocks if s["symbol"] == "UPX")
    assert upx["daily"]["change_5d_pct"] == pytest.approx(
        (price / daily[-5, 3] - 1) * 100, abs=6e-4
    )
    assert upx["today"]["change_today_pct"] == pytest.approx(
        (price / daily[-1, 3] - 1) * 100, abs=6e-4
    )
    assert "NSE:UPX" in scan_file["depth"] and scan_file["executions"]
    final = json.loads((folder / "final_bars.json").read_text("utf-8"))
    assert len(final["bars"]["UPX"]) == 375
    trades = {
        p.symbol: p
        for p in store.positions("closed")
        if p.entry_at.date() == test_days[0]
    }
    assert (
        trades["UPX"].exit_reason == "tp"
        and trades["UPX"].charges_detail["kind"] == "intraday"
    )
    assert trades["DNX"].exit_reason == "sl"
    assert trades["FLT"].exit_reason == "tp_gap"
    assert trades["FLT"].exit_at.date() == test_days[1]
    assert trades["FLT"].charges_detail["kind"] == "delivery"
    assert trades["FLT"].exit_price > trades["FLT"].target_price
    # day two: all three were out of the portfolio by 09:40 (FLT sold at the open)
    day_two = {
        p.symbol: p for p in store.positions() if p.entry_at.date() == test_days[1]
    }
    assert set(day_two) == set(STOCKS)
    assert day_two["FLT"].entry_at > trades["FLT"].exit_at
    summary = (tmp_path / "report" / "report.txt").read_text()
    assert f"Closed trades {len(store.positions('closed'))}:" in summary
    assert not store.events("error")
    verdicts = store.events("verdict")
    assert {v["symbol"] for v in verdicts} >= set(STOCKS)
    assert Decimal(trades["UPX"].net_pnl) > 0 > Decimal(trades["DNX"].net_pnl)
    scans = store.events("scan")
    assert {s["detail"]["mode"] for s in scans} == {
        "live" if mode == "live" else "full"
    }
    if not with_shadow:
        return
    # Same first question, then its own portfolio: asked again about DNX and FLT,
    # which only the main AI holds; bought nothing but UPX.
    assert shadow.requests[0] is decider.requests[0]
    assert any(c.symbol == "DNX" for r in shadow.requests[1:] for c in r.candidates)
    shadow_store = shadows[0].broker.store
    assert {p.symbol for p in shadow_store.positions()} == {"UPX"}
    assert shadow_store.positions("closed")[0].exit_reason == "tp"
    assert not shadow_store.events("error")
    assert (tmp_path / "shadow" / "report" / "report.txt").exists()


def test_reserves_replace_stocks_that_cannot_be_scored(tmp_path) -> None:
    days = weekdays(date(2025, 1, 1), 300)
    write_market(tmp_path / "candles", days, days[-2:])
    bundle = stub_bundle()
    day = days[-2]
    clock = ReplayClock(datetime(*day.timetuple()[:3], 9, 0, tzinfo=INDIA_TIMEZONE))
    data = ReplayData(tmp_path / "candles", clock, first_day=days[-60])
    decider = BuyAll()
    session = PaperSession(
        data,
        clock,
        bundle,
        STOCKS,  # ranks: UPX, DNX, FLT
        PaperBroker(PaperStore(tmp_path / "paper.sqlite"), PaperConfig()),
        decider,
        "buy-all",
        DecisionJournal(tmp_path / "decisions.sqlite"),
        report_dir=tmp_path / "report",
        settings=SessionSettings(workers=1, universe_target=2),
        log=lambda m: None,
    )
    session.prepare()
    clock.current = datetime(*day.timetuple()[:3], 9, 40, 5, tzinfo=INDIA_TIMEZONE)
    session.decision_cycle(24)
    assert {c.symbol for c in decider.requests[-1].candidates} == {"UPX", "DNX"}
    session.histories.pop("UPX")  # e.g. too few trades: the next-ranked stock steps in
    clock.current = datetime(*day.timetuple()[:3], 9, 45, 5, tzinfo=INDIA_TIMEZONE)
    session.decision_cycle(29)
    assert {c.symbol for c in decider.requests[-1].candidates} == {"FLT"}


class DownNse:
    def __init__(self) -> None:
        self.calls = 0

    def announcements(self, start, end):
        self.calls += 1
        raise TimeoutError("NSE down")


class News:
    def add_announcements(self, rows) -> None:
        raise AssertionError("nothing to add")


def test_filings_stop_after_three_consecutive_failures(tmp_path) -> None:
    clock = ReplayClock(datetime(2025, 1, 1, 10, tzinfo=INDIA_TIMEZONE))
    nse, logs = DownNse(), []
    session = PaperSession(
        None,
        clock,
        None,
        (),
        None,
        None,
        "none",
        None,
        report_dir=tmp_path,
        settings=SessionSettings(workers=1),
        news=News(),
        nse=nse,
        log=logs.append,
    )
    for _ in range(5):
        session._refresh_filings()
    assert nse.calls == 3
    assert logs[-1].endswith("filings off for this session")


def bare_session(tmp_path, moment: datetime, **kwargs) -> PaperSession:
    return PaperSession(
        None,
        ReplayClock(moment),
        None,
        (),
        None,
        None,
        "none",
        None,
        report_dir=tmp_path,
        settings=SessionSettings(workers=1),
        log=lambda m: None,
        **kwargs,
    )


class FlakyNse:
    """Records what was asked; the first request for one day times out."""

    def __init__(self, failing: date) -> None:
        self.calls: list[tuple[str, date]] = []
        self.failing: date | None = failing

    def board_meetings(self, start, end):
        self.calls.append(("meetings", start))
        return []

    def corporate_actions(self, start, end):
        self.calls.append(("actions", start))
        return []

    def announcements(self, start, end):
        assert start == end
        self.calls.append(("announcements", start))
        if start == self.failing:
            self.failing = None
            raise TimeoutError("NSE slow")
        return []


class Sink:
    def add_meetings(self, rows) -> None: ...

    def add_actions(self, rows) -> None: ...

    def add_announcements(self, rows) -> None: ...


def test_filings_cover_the_last_week_whole_and_resume_after_a_failure(
    tmp_path,
) -> None:
    today = date(2026, 10, 14)
    nse = FlakyNse(failing=date(2026, 10, 12))
    session = bare_session(
        tmp_path,
        datetime(2026, 10, 14, 8, 40, tzinfo=INDIA_TIMEZONE),
        news=Sink(),
        nse=nse,
    )
    session._refresh_filings(initial=True)  # stops at Oct 12
    session._refresh_filings()  # the next cycle resumes there
    days = [day for kind, day in nse.calls if kind == "announcements"]
    week = [today - timedelta(days=n) for n in range(7, 0, -1)]
    assert days == week[:6] + [date(2026, 10, 12), date(2026, 10, 13), today]
    assert [kind for kind, _ in nse.calls].count("meetings") == 1


def futures_quote(symbol: str, last: str, previous: str, oi=None) -> DepthSnapshot:
    price = Decimal(last)
    return DepthSnapshot(
        instrument=Instrument("NFO", symbol),
        timestamp=datetime(2026, 10, 13, 12, 45, tzinfo=INDIA_TIMEZONE),
        last_price=price,
        last_trade_at=None,
        volume=1,
        total_buy_quantity=1,
        total_sell_quantity=1,
        bids=(DepthLevel(price - Decimal("0.1"), 1, 1),),
        asks=(DepthLevel(price + Decimal("0.1"), 1, 1),),
        previous_close=Decimal(previous),
        open_interest=oi,
    )


def test_futures_facts_are_now_and_explicit_about_the_previous_session(
    tmp_path,
) -> None:
    session = bare_session(
        tmp_path, datetime(2026, 10, 13, 12, 45, tzinfo=INDIA_TIMEZONE)
    )
    session.near_next = {
        "ABC": [Instrument("NFO", "ABC26OCT"), Instrument("NFO", "ABC26NOV")]
    }
    session.expiries = {"ABC": date(2026, 10, 27)}
    session.oi_daily = {
        "ABC": {
            "previous_oi": {"ABC26OCT": 1000, "ABC26NOV": 100},
            "buildup_prev_session": "short_covering",
        }
    }
    depth = {
        "NFO:ABC26OCT": futures_quote("ABC26OCT", "202", "200", oi=1050),
        "NFO:ABC26NOV": futures_quote("ABC26NOV", "203", "201", oi=160),
        "NSE:ABC": futures_quote("ABC", "201", "199"),
    }
    facts = session._oi_context("ABC", depth)
    assert facts["days_to_expiry"] == 14  # from today, not from the last session
    assert facts["premium_now"] == pytest.approx(202 / 201 - 1)
    assert facts["oi_change_today"] == pytest.approx(1210 / 1100 - 1)
    assert facts["buildup_today"] == "long_buildup"
    assert facts["buildup_prev_session"] == "short_covering"
    json.dumps(facts)  # captured for validation as it is
    assert session._oi_context("XYZ", depth) is None
