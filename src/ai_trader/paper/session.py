"""One paper-trading session: scanner -> components -> AI -> paper execution.

Every 5 minutes from 09:40 to 14:55 the session fetches today's minute bars for the
whole universe (except stocks held or already bought today), scores each with the
production scanner, and sends every stock at or above the threshold, with its order
book, futures open interest, NSE filings and market context, to the AI. Each BUY
verdict becomes a paper buy; deterministic code sizes it and refuses it when a
limit is reached. Every minute, completed bars of held stocks are checked for +1%
or -1% and the trade report is rewritten.

The market-data source is anything with the read-only ``KiteBroker`` methods used
here (the live broker, or a historical replay), and the clock is injected, so the
same code runs live and in replay.
"""

from __future__ import annotations

import json
import threading
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from time import sleep
from typing import Any, Protocol

import numpy as np

from ai_trader.ai.decision import (
    Candidate,
    Decider,
    DecisionRequest,
    Verdict,
)
from ai_trader.ai.journal import DecisionJournal
from ai_trader.ai.packet import market_v3, stock_v3
from ai_trader.broker import CandleInterval, DepthSnapshot, Instrument, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.market.stream import StreamSupervisor
from ai_trader.news.features import event_context
from ai_trader.oi.features import daily_oi_features, live_oi_context, monthly_expiries
from ai_trader.orderbook.features import depth_features
from ai_trader.paper.broker import PaperBroker
from ai_trader.paper.live import LiveBars
from ai_trader.paper.report import build_report, write_report
from ai_trader.scanner.upside import (
    DECISIONS,
    LAST,
    Estimate,
    MarketHistory,
    ScannerBundle,
    StockHistory,
    decision_close,
    fill_quiet_minutes,
    minute_index,
    scan,
)

DAILY_CHUNK = timedelta(days=399)
MINUTE_CHUNK = timedelta(days=55)


class Clock(Protocol):
    simulated: bool

    def now(self) -> datetime: ...

    def sleep_until(self, moment: datetime) -> None: ...


class SystemClock:
    simulated = False

    def now(self) -> datetime:
        return datetime.now(INDIA_TIMEZONE)

    def sleep_until(self, moment: datetime) -> None:
        while (left := (moment - self.now()).total_seconds()) > 0:
            sleep(min(left, 15))


@dataclass(frozen=True)
class SessionSettings:
    threshold: float = 0.60
    batch_size: int = 10
    decision_delay: timedelta = timedelta(seconds=5)
    exit_delay: timedelta = timedelta(seconds=20)
    history_days: int = 45  # calendar days of minute history (>= 20 complete sessions)
    daily_days: int = 395  # calendar days of daily history (>= 251 sessions), 1 request
    workers: int = 3
    stale_decision: timedelta = timedelta(minutes=3)
    # Live pre-screen: re-score on historical bars everything within this margin
    # of the threshold, plus the top 10 and the best F&O stocks (live OI checks).
    prescreen_margin: float = 0.05
    oi_probe: int = 5
    universe_target: int | None = None  # highest-ranked scoreable stocks to scan


def ist(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, time(hour, minute), INDIA_TIMEZONE)


def _plain(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _filled(rows: np.ndarray) -> np.ndarray:
    """An index's minutes without an update keep the previous value."""
    filled = fill_quiet_minutes(rows, len(rows) - 1, max_share=1.0)
    return rows if filled is None else filled


def _rows(array: np.ndarray) -> list[list[float | None]]:
    return [[_plain(v) for v in row] for row in array]


def _show(value: Any, spec: str = "") -> str:
    return "n/a" if value is None else format(value, spec)


def _inputs_line(stock: Mapping[str, Any]) -> str:
    """The main facts the AI gets for one stock, on one line."""
    today = stock.get("today") or {}
    book = stock.get("order_book") or {}
    oi = stock.get("futures_oi") or {}
    filings = stock.get("nse_filings") or {}
    parts = [
        f"Rs {_show(today.get('last_price'), ',.2f')}, gap "
        f"{_show(today.get('gap_at_open_pct'), '+.2f')}%, since open "
        f"{_show(today.get('change_since_open_pct'), '+.2f')}%, vs VWAP "
        f"{_show(today.get('vs_vwap_pct'), '+.2f')}%, volume "
        f"{_show(today.get('volume_vs_usual'), '.1f')}x usual",
        f"book: spread {_show(book.get('spread_bps'), '.0f')} bps, buy impact "
        f"{_show(book.get('impact_buy_2L_bps'), '.0f')} bps, imbalance "
        f"{_show(book.get('imbalance_total'), '+.2f')}"
        if book
        else "book: none",
        f"futures: {oi.get('buildup_1d') or 'n/a'} yesterday, "
        f"{oi.get('buildup_today') or 'n/a'} today"
        if oi
        else "no futures",
        f"filings since yesterday: {filings.get('filings_since_prev_close')} "
        f"({filings.get('material_since_prev_close')} material)"
        if filings
        else "filings: n/a",
    ]
    return f"  {stock['symbol']}: " + " | ".join(parts)


def _search_line(search: Mapping[str, Any]) -> str:
    what = search.get("query") or search.get("url") or search.get("type")
    if isinstance(what, list):
        what = "; ".join(map(str, what))
    return f'searched "{what}" ({len(search.get("sources") or [])} sources)'


def _snapshot(snapshot: DepthSnapshot) -> dict[str, Any]:
    """A depth snapshot as plain JSON, every field the quote carried."""

    def levels(side: Sequence[Any]) -> list[list[Any]]:
        return [[str(lv.price), lv.quantity, lv.orders] for lv in side]

    return {
        "timestamp": snapshot.timestamp.isoformat(),
        "last_price": str(snapshot.last_price),
        "last_trade_at": snapshot.last_trade_at.isoformat()
        if snapshot.last_trade_at
        else None,
        "volume": snapshot.volume,
        "total_buy_quantity": snapshot.total_buy_quantity,
        "total_sell_quantity": snapshot.total_sell_quantity,
        "bids": levels(snapshot.bids),
        "asks": levels(snapshot.asks),
        "previous_close": str(snapshot.previous_close),
        "lower_circuit": str(snapshot.lower_circuit)
        if snapshot.lower_circuit
        else None,
        "upper_circuit": str(snapshot.upper_circuit)
        if snapshot.upper_circuit
        else None,
        "open_interest": snapshot.open_interest,
    }


def _array(candles: Sequence[OHLCVCandle], day: date, length: int) -> np.ndarray:
    rows = np.full((length, 5), np.nan)
    for bar in candles:
        if bar.timestamp.date() != day:
            continue
        index = minute_index(bar.timestamp)
        if 0 <= index < length:
            rows[index] = [
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                np.nan if bar.volume is None else float(bar.volume),
            ]
    return rows


def complete_sessions(candles: Sequence[OHLCVCandle]) -> list[tuple[date, np.ndarray]]:
    """09:15-14:59 sessions, oldest first; no-trade minutes filled, as live."""
    days = sorted({bar.timestamp.date() for bar in candles})
    by_day: dict[date, list[OHLCVCandle]] = {d: [] for d in days}
    for bar in candles:
        by_day[bar.timestamp.date()].append(bar)
    result = []
    for day in days:
        array = fill_quiet_minutes(_array(by_day[day], day, LAST + 1), LAST)
        if array is not None:
            result.append((day, array))
    return result


def daily_rows(
    candles: Sequence[OHLCVCandle], before: date
) -> tuple[list[date], np.ndarray]:
    rows = sorted(
        (c for c in candles if c.timestamp.date() < before), key=lambda c: c.timestamp
    )
    dates = [c.timestamp.date() for c in rows]
    values = np.array(
        [
            [float(c.open), float(c.high), float(c.low), float(c.close)]
            + [float(c.volume or 0)]
            for c in rows
        ]
    ).reshape(-1, 5)
    return dates, values


@dataclass(frozen=True)
class Book:
    """One AI and the paper portfolio that follows its verdicts."""

    name: str
    decider: Decider
    journal: DecisionJournal
    broker: PaperBroker
    report_dir: Path
    label: str = ""


class PaperSession:
    def __init__(
        self,
        data: Any,
        clock: Clock,
        bundle: ScannerBundle,
        universe: Sequence[str],
        broker: PaperBroker,
        decider: Decider,
        decider_name: str,
        journal: DecisionJournal,
        *,
        report_dir: Path,
        settings: SessionSettings = SessionSettings(),
        news: Any = None,
        nse: Any = None,
        cache_dir: Path | None = None,
        log: Callable[[str], None] = print,
        shadows: Sequence[Book] = (),
        profiles: Mapping[str, Mapping[str, str]] | None = None,
        capture_dir: Path | None = None,
        stream: Callable[[Sequence[Instrument]], Any] | None = None,
        live: Any = None,
        label: str = "",
    ) -> None:
        self.data = data
        self.clock = clock
        self.bundle = bundle
        self.universe = list(universe)
        self.broker = broker
        self.decider = decider
        self.decider_name = decider_name
        self.journal = journal
        self.report_dir = report_dir
        # Shadows see the same scan and requests as the main AI, never its portfolio.
        self.books = (
            Book(decider_name, decider, journal, broker, report_dir, label),
        ) + tuple(shadows)
        self.settings = settings
        self.news = news
        self.nse = nse
        self.log = log
        self.today = clock.now().date()
        self.cache_dir = cache_dir / self.today.isoformat() if cache_dir else None
        self.histories: dict[str, StockHistory] = {}
        self.ticks: dict[str, Decimal] = {}
        self.fno: set[str] = set()
        self.near_next: dict[str, list[Instrument]] = {}
        self.oi_daily: dict[str, dict] = {}
        self.market: MarketHistory | None = None
        self.previous_session: date | None = None
        self.marks: dict[str, Decimal] = {}
        self.profiles = dict(profiles or {})
        self.capture_dir = capture_dir
        self._captured: set[str] = set()
        self._stream = stream
        self.live = live
        self._supervisor: StreamSupervisor | None = None
        self.rank = {symbol: i for i, symbol in enumerate(self.universe)}
        self._filing_failures = 0
        self._clients = threading.local()
        self._client_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=settings.workers)
        self._exit_data: Any = None
        self._reports: list[dict[str, Any]] = []
        # Per AI: model calls, stock verdicts, BUYs, searches, paper buys this run;
        # spend starts from today's journal so a restart shows the day's total.
        self._stats = {
            book.name: dict.fromkeys(
                ("calls", "asked", "buy", "searches", "bought", "refused"), 0
            )
            | {
                "cost": book.journal.spent_usd(self.today)
                if book.journal is not None
                else 0.0
            }
            for book in self.books
        }

    # --- plumbing -------------------------------------------------------------
    @property
    def exit_data(self) -> Any:
        """The exit thread's own market-data client (shared rate limits)."""
        if self._exit_data is None:
            self._exit_data = (
                self.data.fork_history()
                if hasattr(self.data, "fork_history")
                else self.data
            )
        return self._exit_data

    def _client(self) -> Any:
        """One market-data client per worker thread (shared rate limits)."""
        client = getattr(self._clients, "value", None)
        if client is None:
            with self._client_lock:
                client = (
                    self.data.fork_history()
                    if hasattr(self.data, "fork_history")
                    else self.data
                )
            self._clients.value = client
        return client

    def _parallel(self, function: Callable[[str], Any], symbols: Iterable[str]):
        names = list(symbols)
        return dict(zip(names, self._pool.map(function, names), strict=True))

    def _event(
        self,
        kind: str,
        symbol: str | None,
        detail: Mapping[str, Any],
        broker: PaperBroker | None = None,
    ) -> None:
        (broker or self.broker).store.event(
            self.clock.now(), kind, symbol, dict(detail)
        )

    def _candles(
        self,
        client: Any,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> list[OHLCVCandle]:
        step = MINUTE_CHUNK if interval is CandleInterval.ONE_MINUTE else DAILY_CHUNK
        result: list[OHLCVCandle] = []
        cursor = start
        while cursor < end:
            stop = min(end, cursor + step)
            result.extend(
                client.get_historical_candles(
                    Instrument("NSE", symbol), cursor, stop, interval
                )
            )
            cursor = stop + timedelta(seconds=1)
        return result

    # --- preparation ------------------------------------------------------------
    def prepare(self) -> None:
        """History for every stock, index context, F&O list, OI and ticks."""
        today = self.today
        yesterday_end = ist(today - timedelta(days=1), 23, 59)
        # Resolve on the main client first so worker clients copy the instrument list.
        for symbol in list(self.universe):
            try:
                self.ticks[symbol] = self.data.resolve_instrument(
                    Instrument("NSE", symbol)
                ).tick_size
            except Exception:
                self.log(f"skipping {symbol}: not found in the instrument list")
                self.universe.remove(symbol)
        client = self.data
        nifty = self._candles(
            client,
            "NIFTY",
            ist(today - timedelta(days=self.settings.daily_days), 0, 0),
            yesterday_end,
            CandleInterval.ONE_DAY,
        )
        nifty_dates, nifty_daily = daily_rows(nifty, today)
        vix = self._candles(
            client,
            "INDIAVIX",
            ist(today - timedelta(days=30), 0, 0),
            yesterday_end,
            CandleInterval.ONE_DAY,
        )
        _vix_dates, vix_daily = daily_rows(vix, today)
        if not nifty_dates or not len(vix_daily):
            raise RuntimeError("NIFTY or INDIA VIX daily history is unavailable.")
        self.previous_session = nifty_dates[-1]
        self.market = MarketHistory(
            nifty_daily[:, 3], tuple(nifty_dates), float(vix_daily[-1, 3])
        )
        for row in self.data.nfo_futures():
            name = row.get("name")
            if name in self.universe:
                self.fno.add(name)
                self.near_next.setdefault(name, []).append(row)
        self.near_next = {
            name: [
                Instrument("NFO", r["tradingsymbol"])
                for r in sorted(rows, key=lambda r: r["expiry"])[:2]
            ]
            for name, rows in self.near_next.items()
        }
        self.log(
            f"loading history for {len(self.universe)} stocks "
            f"(~{len(self.universe) * 2 * 0.4 / 60:.0f} minutes at the Kite rate limit)"
        )
        loaded = self._parallel(self._load_history, self.universe)
        self.histories = {s: h for s, h in loaded.items() if h is not None}
        self.log(
            f"history ready for {len(self.histories)} of {len(self.universe)} stocks; "
            f"{len(self.fno)} are F&O (closing auction, futures OI)"
        )
        sessions = list(nifty_dates)
        expiries = monthly_expiries(sessions + [today])
        self.oi_daily = self._parallel(
            lambda s: self._load_oi(s, expiries),
            sorted(self.fno & set(self.histories)),
        )
        self._refresh_filings(initial=True)
        if self._stream is not None and self.live is None:
            self.live = LiveBars(
                today,
                list(self.histories) + ["NIFTY", "INDIAVIX"],
                from_open=self.clock.now() < ist(today, 9, 15),
            )
        for book in self.books:
            self._event(
                "prepared",
                None,
                {
                    "stocks": len(self.universe),
                    "with_history": len(self.histories),
                    "fno": len(self.fno),
                    "previous_session": self.previous_session.isoformat(),
                    "scanner": self.bundle.version,
                    "decider": book.name,
                },
                book.broker,
            )

    def _load_history(self, symbol: str) -> StockHistory | None:
        cache = self.cache_dir / f"{symbol}.npz" if self.cache_dir else None
        if cache is not None and cache.exists():
            with np.load(cache) as saved:
                return StockHistory(
                    tuple(saved["sessions"]),
                    saved["daily"],
                    tuple(date.fromordinal(int(d)) for d in saved["dates"]),
                )
        client = self._client()
        today = self.today
        end = ist(today - timedelta(days=1), 23, 59)
        try:
            minutes = self._candles(
                client,
                symbol,
                ist(today - timedelta(days=self.settings.history_days), 9, 0),
                end,
                CandleInterval.ONE_MINUTE,
            )
            daily = self._candles(
                client,
                symbol,
                ist(today - timedelta(days=self.settings.daily_days), 0, 0),
                end,
                CandleInterval.ONE_DAY,
            )
        except Exception as error:
            self.log(f"history failed for {symbol}: {type(error).__name__}")
            return None
        sessions = complete_sessions(minutes)
        dates, values = daily_rows(daily, today)
        if not dates:
            return None
        kept = [s for _, s in sessions[-25:]]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache,
                sessions=np.array(kept).reshape(-1, LAST + 1, 5),
                daily=values,
                dates=np.array([d.toordinal() for d in dates]),
            )
        return StockHistory(tuple(kept), values, tuple(dates))

    def _load_oi(self, symbol: str, expiries: list[date]) -> dict:
        history = self.histories[symbol]
        try:
            bars = self._client().get_continuous_futures_daily(
                symbol, self.today - timedelta(days=40), self.today - timedelta(days=1)
            )
        except Exception:
            return {}
        spot = dict(zip(history.daily_dates, history.daily[:, 3].tolist(), strict=True))
        return daily_oi_features(bars, spot, expiries)

    def _refresh_filings(self, *, initial: bool = False) -> None:
        if self.news is None or self.nse is None or self._filing_failures >= 3:
            return
        today = self.today
        try:
            if initial:
                self.news.add_meetings(
                    self.nse.board_meetings(today, today + timedelta(days=3))
                )
                self.news.add_actions(
                    self.nse.corporate_actions(today, today + timedelta(days=1))
                )
            self.news.add_announcements(self.nse.announcements(today, today))
            self._filing_failures = 0
        except Exception as error:
            self._filing_failures += 1
            stop = (
                "; filings off for this session" if self._filing_failures >= 3 else ""
            )
            self.log(f"NSE filings refresh failed: {type(error).__name__}{stop}")

    # --- decisions ----------------------------------------------------------------
    def decision_cycle(self, minute: int) -> dict[str, Any]:
        """Scan, ask the AI about every candidate, paper-buy the BUY verdicts."""
        today = self.today
        decision_at = decision_close(today, minute)
        skips = [
            book.broker.held() | book.broker.bought_on(decision_at)
            for book in self.books
        ]
        excluded = skips[0]
        everywhere = set.intersection(*skips)
        symbols = [s for s in self.histories if s not in everywhere]

        def bars(symbol: str) -> np.ndarray | None:
            try:
                candles = self._client().get_historical_candles(
                    Instrument("NSE", symbol),
                    ist(today, 9, 15),
                    decision_at,
                    CandleInterval.ONE_MINUTE,
                )
            except Exception:
                return None
            return _array(candles, today, minute + 1)

        fetch_started = self.clock.now()
        rough: list[Estimate] = []
        rough_skipped: dict[str, int] = {}
        live = self._live_ready(minute)
        if live:
            try:
                rough, rough_skipped = self._scan(
                    minute,
                    {s: self.live.bars(s, minute) for s in symbols},
                    self.live.bars("NIFTY", minute),
                    _filled(self.live.bars("INDIAVIX", minute)),
                )
            except ValueError:
                live = False
        if live:
            floor = self.settings.threshold - self.settings.prescreen_margin
            probe = [e for e in rough if e.symbol in self.fno][: self.settings.oi_probe]
            wanted = {e.symbol for e in rough if e.probability >= floor}
            wanted |= {e.symbol for e in rough[:10] + probe}
            fetch = sorted(wanted, key=self.rank.get)
        else:
            fetch = symbols
        fetched = self._parallel(bars, fetch + ["NIFTY", "INDIAVIX"])
        if live:
            # Seconds after the close Kite may not have published the decision
            # minute yet; ask again for any bar the stream shows but history lacks.
            for _ in range(3):
                late = [
                    s
                    for s, rows in fetched.items()
                    if rows is not None
                    and not np.isfinite(rows[minute, 3])
                    and np.isfinite(self.live.bars(s, minute)[minute, 3])
                ]
                if not late:
                    break
                self.clock.sleep_until(self.clock.now() + timedelta(seconds=2))
                fetched |= self._parallel(bars, late)
        fetch_finished = self.clock.now()
        nifty, vix = fetched.pop("NIFTY"), fetched.pop("INDIAVIX")
        stocks = {s: b for s, b in fetched.items() if b is not None}
        if nifty is None or vix is None:
            self.log(f"{decision_at:%H:%M} skipped: index bars unavailable")
            return {"status": "no_index_data"}
        if self.live is not None and not live:
            self.live.seed(stocks | {"NIFTY": nifty, "INDIAVIX": vix}, minute)
        try:
            estimates, skipped = self._scan(minute, stocks, nifty, vix)
        except ValueError as error:
            self.log(f"{decision_at:%H:%M} skipped: {error}")
            return {"status": "incomplete_index_data"}
        skipped["fetch_failed"] = len(fetch) - len(stocks)
        mode = "live" if live else "full"
        scored = len(rough) if live else len(estimates)
        if live:
            skipped = rough_skipped | {"fetch_failed": skipped["fetch_failed"]}
        ranked = [e for e in estimates if e.probability >= self.settings.threshold]
        candidates = [e for e in ranked if e.symbol not in excluded]
        market = market_v3(estimates[0].features) if estimates else {}
        self._event(
            "scan",
            None,
            {
                "decision_at": decision_at.isoformat(),
                "mode": mode,
                "scored": scored,
                "rescored": len(estimates),
                "seconds": round((fetch_finished - fetch_started).total_seconds(), 1),
                "excluded_held_or_bought": len(excluded),
                "skipped": skipped,
                "candidates": [
                    {"symbol": e.symbol, "probability": round(e.probability, 4)}
                    for e in candidates
                ],
                "top": [
                    {"symbol": e.symbol, "probability": round(e.probability, 4)}
                    for e in estimates[:5]
                ],
            },
        )
        self.log(
            f"===== {decision_at:%H:%M} scan: {scored} stocks ({mode} data, "
            f"{(fetch_finished - fetch_started).total_seconds():.0f} s"
            + (f", {len(estimates)} re-checked on exchange bars" if live else "")
            + ") | NIFTY "
            + _show(market.get("nifty_change_today_pct"), "+.2f")
            + "% today | VIX "
            + _show(market.get("india_vix"), ".2f")
            + " ====="
        )
        self.log(
            f"  scanner picks, P(+1% before close) >= {self.settings.threshold:.0%}: "
            + (
                ", ".join(
                    f"{e.symbol} {e.probability:.0%}"
                    + (" F&O" if e.symbol in self.fno else "")
                    for e in ranked[:25]
                )
                or "none"
            )
            + (f" ... +{len(ranked) - 25} more" if len(ranked) > 25 else "")
        )
        below = [e for e in estimates if e.probability < self.settings.threshold]
        self.log(
            "  next best: "
            + (", ".join(f"{e.symbol} {e.probability:.0%}" for e in below[:5]) or "-")
        )
        if len(ranked) > len(candidates):
            held = [e.symbol for e in ranked if e.symbol in excluded]
            self.log(
                f"  {self.books[0].label}already held or bought today: "
                + ", ".join(held)
            )
        capture = self._capture_scan(
            decision_at,
            minute,
            (fetch_started, fetch_finished),
            stocks,
            {"NIFTY": nifty, "INDIAVIX": vix},
            estimates,
            ranked,
            skipped,
        )
        if capture is not None:
            capture["mode"] = mode
            capture["prescreen"] = {
                e.symbol: round(e.probability, 6) for e in rough if e.symbol in stocks
            }
            # Live futures OI is requested for candidates; probe the best-scoring
            # F&O stocks every cycle too, so live OI is checked even when no
            # candidate has futures.
            chosen = {e.symbol for e in ranked}
            probe = [
                e
                for e in (rough if live else estimates)
                if e.symbol in self.fno and e.symbol not in chosen
            ][: self.settings.oi_probe]
            if probe:
                self._capture_books(capture, probe, self._depth(probe))
        if not ranked:
            self._save_capture(capture)
            return {"status": "no_candidates", "scored": scored}
        self._refresh_filings()
        depth = self._depth(ranked)
        closes = {e.symbol: float(stocks[e.symbol][minute, 3]) for e in ranked}
        self._capture_books(capture, ranked, depth)
        plans: dict[tuple[str, ...], list[tuple[str, list[Estimate], Any]]] = {}
        keys = []
        for skip in skips:
            mine = [e for e in ranked if e.symbol not in skip]
            key = tuple(e.symbol for e in mine)
            keys.append(key)
            if mine and key not in plans:
                as_of, size = self.clock.now(), self.settings.batch_size
                plans[key] = [
                    (
                        f"{today:%Y%m%d}-{decision_at:%H%M}-{number}",
                        mine[start : start + size],
                        self._request(as_of, mine[start : start + size], depth, closes),
                    )
                    for number, start in enumerate(range(0, len(mine), size), 1)
                ]
        stocks_seen = {
            stock["symbol"]: stock
            for plan in plans.values()
            for _, _, request in plan
            for stock in (request.document or {}).get("stocks", [])
        }
        self.log("  inputs sent to the AI (also searches the web itself):")
        for estimate in ranked:
            if estimate.symbol in stocks_seen:
                self.log(_inputs_line(stocks_seen[estimate.symbol]))
        bought: list[str] = []
        outcome = []
        for book, key in zip(self.books, keys, strict=True):
            done = []
            for cycle_id, batch, request in plans.get(key, []):
                done += self._ask(book, decision_at, cycle_id, batch, request, capture)
            if book is self.books[0]:
                bought += done
            outcome.append(
                f"{book.label.strip() or book.name} "
                + ("bought " + ", ".join(done) if done else "no buy")
            )
        self._save_capture(capture)
        self.write_report()
        self.log(f"----- {decision_at:%H:%M} result: " + " | ".join(outcome))
        self._log_books()
        return {
            "status": "ok",
            "scored": scored,
            "candidates": len(candidates),
            "bought": bought,
        }

    def _scan(
        self,
        minute: int,
        stocks: Mapping[str, np.ndarray],
        nifty: np.ndarray,
        vix: np.ndarray,
    ) -> tuple[list[Estimate], dict[str, int]]:
        """Score, then keep the ``universe_target`` highest-ranked scoreable stocks
        (lower-ranked reserves stand in for any that cannot be scored)."""
        estimates, skipped = scan(
            self.bundle,
            minute,
            self.today,
            stocks,
            self.histories,
            self.fno,
            nifty,
            vix,
            self.market,
            self.previous_session,
        )
        target = self.settings.universe_target
        if target is not None:
            kept = set(
                sorted((e.symbol for e in estimates), key=self.rank.get)[:target]
            )
            skipped["reserves_not_needed"] = len(estimates) - len(kept)
            estimates = [e for e in estimates if e.symbol in kept]
        return estimates, skipped

    def _live_ready(self, minute: int) -> bool:
        """Live bars can pre-screen: the stream is ticking and NIFTY (which trades
        every second) has every minute so far, so no stretch was missed."""
        if self.live is None:
            return False
        silent = self.live.silent_seconds()
        return (
            silent is not None and silent < 30 and self.live.complete("NIFTY", minute)
        )

    def _start_stream(self, until: datetime) -> None:
        if self._stream is None or self.live is None or self.clock.simulated:
            return
        instruments = [Instrument("NSE", s) for s in self.live.symbols]

        def run() -> None:
            self.clock.sleep_until(max(self.clock.now(), ist(self.today, 9, 14)))
            seconds = (until - self.clock.now()).total_seconds()
            if seconds <= 0:
                return
            self._supervisor = StreamSupervisor(
                lambda: self._stream(instruments),
                self.live.on_tick,
                retry_on=(RuntimeError, OSError),
            )
            self.log(f"live stream starting for {len(instruments)} instruments")
            try:
                report = self._supervisor.run(seconds)
                self.log(
                    f"live stream ended: {report.ticks} ticks, {report.reconnects} "
                    f"reconnects, {report.failures} failures ({report.stopped_because})"
                )
            except Exception as error:
                self.log(
                    f"live stream stopped ({type(error).__name__}); "
                    "scans fetch every stock instead"
                )

        threading.Thread(target=run, name="stream", daemon=True).start()

    def _ask(
        self,
        book: Book,
        decision_at: datetime,
        cycle_id: str,
        batch: Sequence[Estimate],
        request: DecisionRequest,
        capture: dict[str, Any] | None = None,
    ) -> list[str]:
        result = book.decider.decide(request)
        book.journal.record(cycle_id, decision_at, book.name, result)
        reply = json.loads(result.response_json or "{}")
        searches = reply.get("web_searches") or []
        stats = self._stats[book.name]
        stats["calls"] += result.usage is not None
        stats["asked"] += len(result.verdicts)
        stats["buy"] += sum(v.verdict == "BUY" for v in result.verdicts)
        stats["searches"] += len(searches)
        stats["cost"] += result.cost_usd
        self.log(
            f"{book.label}AI {result.status}: {len(result.verdicts)} stocks, "
            f"{len(searches)} web searches, ${result.cost_usd:.4f}, "
            f"{result.latency_seconds:.1f} s"
        )
        for search in searches:
            self.log(f"  {book.label}{_search_line(search)}")
        for verdict in result.verdicts:
            self.log(
                f"  {book.label}{verdict.symbol}: {verdict.verdict} "
                f"(confidence {verdict.confidence:.2f}) {verdict.reason}"
            )
            self._event(
                "verdict",
                verdict.symbol,
                {
                    "cycle_id": cycle_id,
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "reason": verdict.reason,
                    "status": result.status,
                },
                book.broker,
            )
        buys = [v for v in result.verdicts if v.verdict == "BUY"]
        return self._execute(buys, batch, cycle_id, book, capture)

    # --- input capture for validation ------------------------------------------
    def _capture_scan(
        self,
        decision_at: datetime,
        minute: int,
        fetch: tuple[datetime, datetime],
        stocks: Mapping[str, np.ndarray],
        index: Mapping[str, np.ndarray],
        estimates: Sequence[Estimate],
        ranked: Sequence[Estimate],
        skipped: Mapping[str, int],
    ) -> dict[str, Any] | None:
        """Raw inputs of one scan, kept so they can be checked independently."""
        if self.capture_dir is None:
            return None
        kept = {e.symbol: e for e in estimates[:10]} | {e.symbol: e for e in ranked}
        scored = {e.symbol for e in estimates}
        gaps = {
            symbol: np.flatnonzero(~np.isfinite(array[:, 3])).tolist()[:20]
            for symbol, array in stocks.items()
            if symbol not in scored and not np.isfinite(array[:, 3]).all()
        }
        self._captured |= set(kept)
        return {
            "decision_at": decision_at.isoformat(),
            "minute": minute,
            "fetch_started": fetch[0].isoformat(),
            "fetch_finished": fetch[1].isoformat(),
            "skipped": dict(skipped),
            "missing_minutes": gaps,
            "unscored_complete": sorted(set(stocks) - scored - set(gaps)),
            "index": {name: _rows(array) for name, array in index.items()},
            "stocks": {
                symbol: {
                    "probability": round(e.probability, 6),
                    "fno": symbol in self.fno,
                    "features": {k: _plain(v) for k, v in e.features.items()},
                    "bars": _rows(stocks[symbol]),
                }
                for symbol, e in kept.items()
            },
            "depth": {},
            "oi": {},
            "futures": {},
            "executions": [],
        }

    def _capture_books(
        self, capture: dict[str, Any] | None, estimates: Sequence[Estimate], depth
    ) -> None:
        if capture is None:
            return
        capture["depth"] |= {k: _snapshot(s) for k, s in depth.items()}
        capture["oi"] |= {
            e.symbol: self._oi_context(e.symbol, depth) for e in estimates
        }
        capture["futures"] |= {
            e.symbol: [f.trading_symbol for f in self.near_next.get(e.symbol, [])]
            for e in estimates
        }

    def _save_capture(self, capture: dict[str, Any] | None) -> None:
        if capture is None or self.capture_dir is None:
            return
        folder = self.capture_dir / self.today.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        name = datetime.fromisoformat(capture["decision_at"]).strftime("%H%M")
        (folder / f"{name}.json").write_text(json.dumps(capture), "utf-8")

    def capture_final_bars(self) -> None:
        """Complete minute bars of every captured stock, for revision checks."""
        if self.capture_dir is None or not self._captured:
            return
        today = self.today
        end = min(self.clock.now(), ist(today, 15, 30))
        rows = {}
        for symbol in sorted(self._captured | {"NIFTY", "INDIAVIX"}):
            try:
                candles = self.exit_data.get_historical_candles(
                    Instrument("NSE", symbol),
                    ist(today, 9, 15),
                    end,
                    CandleInterval.ONE_MINUTE,
                )
            except Exception as error:
                self.log(f"final bars failed for {symbol}: {type(error).__name__}")
                continue
            rows[symbol] = _rows(_array(candles, today, 375))
        folder = self.capture_dir / today.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "final_bars.json").write_text(
            json.dumps({"fetched_at": self.clock.now().isoformat(), "bars": rows}),
            "utf-8",
        )
        self.log(f"saved final bars of {len(rows)} instruments for validation")

    def _depth(self, estimates: Sequence[Estimate]) -> dict[str, DepthSnapshot]:
        instruments = [Instrument("NSE", e.symbol) for e in estimates]
        instruments += [f for e in estimates for f in self.near_next.get(e.symbol, [])]
        snapshots: list[DepthSnapshot] = []
        for start in range(0, len(instruments), 500):
            try:
                found, _missing = self._client().get_depth(
                    instruments[start : start + 500]
                )
                snapshots.extend(found)
            except Exception as error:
                self.log(f"order book request failed: {type(error).__name__}")
        return {
            f"{s.instrument.exchange}:{s.instrument.trading_symbol}": s
            for s in snapshots
        }

    def _request(
        self,
        as_of: datetime,
        batch: Sequence[Estimate],
        depth: Mapping[str, DepthSnapshot],
        closes: Mapping[str, float],
    ) -> DecisionRequest:
        """Facts only (v3): no scanner output, stocks in alphabetical order."""
        since = ist(self.previous_session, 15, 30)
        stocks = []
        for estimate in sorted(batch, key=lambda e: e.symbol):
            symbol = estimate.symbol
            snapshot = depth.get(f"NSE:{symbol}")
            stocks.append(
                stock_v3(
                    symbol,
                    estimate.features,
                    closes[symbol],
                    profile=self.profiles.get(symbol),
                    book=depth_features(
                        snapshot,
                        notional=float(self.broker.config.per_trade),
                        as_of=as_of,
                    )
                    if snapshot
                    else None,
                    oi=self._oi_context(symbol, depth),
                    filings=event_context(self.news, symbol, as_of, since)
                    if self.news is not None
                    else None,
                )
            )
        market = market_v3(batch[0].features)
        return DecisionRequest(
            as_of,
            market,
            tuple(Candidate(s["symbol"], {}) for s in stocks),
            document={"as_of": as_of.isoformat(), "market": market, "stocks": stocks},
        )

    def _oi_context(
        self, symbol: str, depth: Mapping[str, DepthSnapshot]
    ) -> dict | None:
        daily = self.oi_daily.get(symbol) or {}
        values = dict(daily.get(self.previous_session) or {})
        contracts = self.near_next.get(symbol, [])
        near = depth.get(f"NFO:{contracts[0].trading_symbol}") if contracts else None
        rolled = values.get("calendar_days_to_expiry") == 0
        if near is not None and values and not rolled:
            book = depth_features(near)
            values |= live_oi_context(
                near.open_interest,
                None,
                values.get("open_interest"),
                book["day_change_pct"] / 100,
            )
        return values or None

    def _execute(
        self,
        buys: Sequence[Verdict],
        batch: Sequence[Estimate],
        cycle_id: str,
        book: Book,
        capture: dict[str, Any] | None = None,
    ) -> list[str]:
        if not buys:
            return []
        probability = {e.symbol: e.probability for e in batch}
        fresh = self._depth([e for e in batch if e.symbol in {v.symbol for v in buys}])
        bought = []
        for verdict in buys:
            snapshot = fresh.get(f"NSE:{verdict.symbol}")
            result = book.broker.buy(
                verdict.symbol,
                snapshot,
                self.ticks[verdict.symbol],
                self.clock.now(),
                cycle_id=cycle_id,
                probability=probability[verdict.symbol],
                confidence=verdict.confidence,
                reason=verdict.reason,
            )
            if result.position:
                p = result.position
                bought.append(verdict.symbol)
                self._stats[book.name]["bought"] += 1
                self.log(
                    f"  {book.label}>>> PAPER BUY {p.symbol}: {p.quantity} shares @ "
                    f"Rs {p.entry_price:,.2f} = Rs {p.buy_value:,.0f}; sell at Rs "
                    f"{p.target_price:,.2f} (target) or Rs {p.stop_price:,.2f} (stop)"
                )
            else:
                self._stats[book.name]["refused"] += 1
                self.log(
                    f"  {book.label}>>> BUY {verdict.symbol} refused: {result.reason}"
                )
            if capture is not None:
                capture["executions"].append(
                    {
                        "book": book.name,
                        "symbol": verdict.symbol,
                        "cycle_id": cycle_id,
                        "depth": _snapshot(snapshot) if snapshot else None,
                        "position_id": result.position.id if result.position else None,
                        "refused": None if result.position else result.reason,
                    }
                )
        return bought

    # --- exits ----------------------------------------------------------------------
    def exit_cycle(self) -> list[str]:
        """Check held stocks' completed minutes for +1% / -1%; refresh marks."""
        now = self.clock.now()
        closed = []
        held = [(b, p) for b in self.books for p in b.broker.open_positions()]
        for book, position in held:
            start = position.checked_through
            try:
                bars = self._candles(
                    self.exit_data,
                    position.symbol,
                    start,
                    now,
                    CandleInterval.ONE_MINUTE,
                )
            except Exception as error:
                self.log(
                    f"{book.label}exit check failed for {position.symbol}: {error}"
                )
                continue
            result = book.broker.apply_bars(position, bars, now)
            if result is not None:
                closed.append(result.symbol)
                self.log(
                    f"{book.label}<<< PAPER SELL {result.symbol}: {result.quantity} "
                    f"shares @ Rs {result.exit_price:,.2f} at "
                    f"{result.exit_at:%H:%M} ({result.exit_reason}); bought "
                    f"{result.entry_at:%d %b %H:%M} @ Rs {result.entry_price:,.2f}; "
                    f"net Rs {result.net_pnl:+,.2f} after charges"
                )
        still_open = sorted(
            {p.symbol for b in self.books for p in b.broker.open_positions()}
        )
        if still_open:
            try:
                prices = self.exit_data.get_ltp(
                    [Instrument("NSE", s) for s in still_open]
                )
                self.marks |= {p.instrument.trading_symbol: p.price for p in prices}
            except Exception as error:
                self.log(f"price refresh failed: {type(error).__name__}")
        if closed or held:
            self.write_report()
        if closed:
            self._log_books()
        return closed

    def write_report(self) -> dict[str, Any]:
        reports = []
        for book in self.books:
            report = build_report(
                book.broker.store,
                self.marks,
                self.clock.now(),
                book.broker.config.capital,
            )
            write_report(report, book.report_dir)
            reports.append(report)
        self._reports = reports
        return reports[0]

    def _log_books(self) -> None:
        """Each AI's paper portfolio on one line, from the latest report."""
        today = self.today.isoformat()
        for book, report in zip(self.books, self._reports, strict=False):
            summary = report["summary"]
            day = next((d for d in report["daily"] if d["date"] == today), None)
            self.log(
                f"  {book.label}portfolio: {summary['open_positions']} open, "
                f"Rs {float(summary['deployed']):,.0f} invested, "
                f"Rs {float(summary['unrealised_net_pnl_if_sold_now']):+,.0f} if "
                f"sold now | closed today: {day['trades'] if day else 0}"
                + (
                    f" ({day['wins']} won), net Rs {float(day['net_pnl']):+,.0f}"
                    if day
                    else ""
                )
                + f" | AI spend today ${self._stats[book.name]['cost']:.2f}"
            )

    def _log_day(self) -> None:
        self.log("===== day summary (paper trading) =====")
        for book in self.books:
            stats = self._stats[book.name]
            self.log(
                f"  {book.label or book.name + ' '}AI: {stats['calls']} calls, "
                f"{stats['asked']} stock verdicts, {stats['buy']} BUY, "
                f"{stats['searches']} web searches | paper buys {stats['bought']}, "
                f"refused {stats['refused']}"
            )
        self._log_books()

    # --- the day --------------------------------------------------------------
    def schedule(
        self, start: datetime, until: datetime
    ) -> list[tuple[datetime, str, int]]:
        """(time, kind, minute) for exit checks every minute and each decision."""
        today = self.today
        events: list[tuple[datetime, str, int]] = []
        moment = max(start, ist(today, 9, 16)).replace(second=0, microsecond=0)
        while moment <= until:
            events.append((moment + self.settings.exit_delay, "exit", -1))
            moment += timedelta(minutes=1)
        for minute in DECISIONS:
            at = decision_close(today, int(minute)) + self.settings.decision_delay
            if start - self.settings.stale_decision <= at <= until:
                events.append((max(at, start), "decision", int(minute)))
        return sorted(events)

    def run(self, until: datetime) -> None:
        start = self.clock.now()
        events = self.schedule(start, until)
        self._start_stream(until)
        if self.clock.simulated:
            for at, kind, minute in events:
                self.clock.sleep_until(at)
                self._safely(
                    self.decision_cycle if kind == "decision" else self.exit_cycle,
                    minute,
                )
        else:
            exits = [e for e in events if e[1] == "exit"]
            decisions = [e for e in events if e[1] == "decision"]
            worker = threading.Thread(
                target=self._run_events, args=(exits,), name="exits", daemon=True
            )
            worker.start()
            self._run_events(decisions)
            worker.join()
        self._safely(self.exit_cycle, -1)
        self._safely(self.capture_final_bars, -1)
        if self._supervisor is not None:
            self._supervisor.stop()
        self.write_report()
        self._log_day()
        self._pool.shutdown(wait=False)
        self.log(f"session finished; report in {self.report_dir}")

    def _run_events(self, events: Sequence[tuple[datetime, str, int]]) -> None:
        for at, kind, minute in events:
            if self.clock.now() > at + timedelta(minutes=2 if kind == "exit" else 3):
                continue
            self.clock.sleep_until(at)
            self._safely(
                self.decision_cycle if kind == "decision" else self.exit_cycle, minute
            )

    def _safely(self, function: Callable, minute: int) -> None:
        try:
            function(minute) if minute >= 0 else function()
        except Exception as error:
            self.log(f"cycle failed: {type(error).__name__}: {error}")
            self._event(
                "error",
                None,
                {
                    "error": type(error).__name__,
                    "trace": traceback.format_exc()[-2000:],
                },
            )


@dataclass
class ReplayClock:
    """A simulated clock for replays and tests."""

    current: datetime
    simulated: bool = field(default=True, init=False)

    def now(self) -> datetime:
        return self.current

    def sleep_until(self, moment: datetime) -> None:
        if moment > self.current:
            self.current = moment
