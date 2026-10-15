"""One paper-trading session: scanner -> components -> AI -> paper execution.

Every 5 minutes from 09:40 to 14:55 the session scores the whole universe (except
stocks held or already traded today) with the production scanner and sends every
stock at or above the threshold (no limit on how many), with its order book,
futures open interest, NSE filings and market context, to each AI. A LONG verdict
becomes a paper buy, SHORT a paper short sale; deterministic code sizes them and
refuses them when a limit is reached. Every minute, completed bars of open
positions are checked for the 1% target; positions still open are closed at
15:00, and the trade report is rewritten.

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
from functools import partial
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
from ai_trader.ai.packet import daily_now, market_v3, stock_v3
from ai_trader.broker import CandleInterval, DepthSnapshot, Instrument, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.market.stream import StreamSupervisor
from ai_trader.news.features import WEEK_DAYS, event_context
from ai_trader.oi.features import (
    live_futures,
    monthly_expiries,
    previous_session_futures,
)
from ai_trader.orderbook.features import depth_features
from ai_trader.paper.broker import LAST_BAR_WAIT, PaperBroker
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
FUTURES_SESSIONS = 10  # of futures day bars: the 5-session OI change and spare


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
    threshold: float = 0.90
    batch_size: int = 10
    decision_delay: timedelta = timedelta(seconds=5)
    exit_delay: timedelta = timedelta(seconds=20)
    history_days: int = 45  # calendar days of minute history (>= 20 complete sessions)
    daily_days: int = 395  # calendar days of daily history (>= 251 sessions), 1 request
    workers: int = 3
    # AI calls in flight at once, over all AIs (batches of one AI run in parallel).
    ai_workers: int = 6
    stale_decision: timedelta = timedelta(minutes=3)
    # Live pre-screen: re-score on historical bars everything within this margin
    # of the threshold, plus the top 10 and the best F&O stocks (live OI checks).
    prescreen_margin: float = 0.05
    # Re-scoring stops this long after the bar closed (highest pre-screen first), so
    # the AI still gets fresh inputs; stocks not reached wait for the next scan.
    rescore_deadline: timedelta = timedelta(seconds=45)
    oi_probe: int = 5
    universe_target: int | None = None  # highest-ranked scoreable stocks to scan
    # The AI is asked only on bars this fresh: a decision starting later, or not
    # ready to ask by then, is skipped rather than decided on old prices.
    max_input_age: timedelta = timedelta(seconds=60)
    # A stream silent this long (all 652 instruments) is treated as down.
    stream_silence: timedelta = timedelta(seconds=10)
    # After a connection made during the session, minutes up to this long after it
    # are taken from historical bars (the subscription takes a moment to apply).
    stream_settle: timedelta = timedelta(seconds=5)
    # A decision waits (at most stream_silence) until the stream has processed a
    # tick this long after its bar closed: the whole minute has then been seen.
    stream_catch_up: timedelta = timedelta(seconds=2)
    # What the AI is given per stock: "facts" (prices, order book, futures OI, NSE
    # filings) or "identity" (symbol and company only; it researches the rest).
    ai_input: str = "facts"


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


def _scale_volume(rows: np.ndarray, factor: float) -> np.ndarray:
    scaled = np.array(rows, dtype=float)
    scaled[:, 4] *= factor
    return scaled


def _show(value: Any, spec: str = "") -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(value: Any, spec: str = "+.2f") -> str:
    return "n/a" if value is None else f"{value:{spec}}%"


def _inputs_line(stock: Mapping[str, Any]) -> str:
    """The main facts the AI gets for one stock, on one line."""
    if "today" not in stock:
        return f"  {stock['symbol']}: {stock.get('company') or ''}"
    today = stock.get("today") or {}
    book = stock.get("order_book") or {}
    oi = stock.get("futures_oi") or {}
    filings = stock.get("nse_filings") or {}
    parts = [
        f"Rs {_show(today.get('last_price'), ',.2f')}, today "
        f"{_pct(today.get('change_today_pct'))} (gap "
        f"{_pct(today.get('gap_at_open_pct'))}, since open "
        f"{_pct(today.get('change_since_open_pct'))}), vs VWAP "
        f"{_pct(today.get('vs_vwap_pct'))}, volume "
        f"{_show(today.get('volume_vs_usual'), '.1f')}x usual",
        f"book: spread {_show(book.get('spread_bps'), '.0f')} bps, buy impact "
        f"{_show(book.get('impact_buy_2L_bps'), '.0f')} bps, imbalance "
        f"{_show(book.get('imbalance_total'), '+.2f')}"
        if book
        else "book: none",
        f"futures: premium {_pct(oi.get('premium_now_pct'))}, OI today "
        f"{_pct(oi.get('oi_change_today_pct'), '+.1f')} "
        f"({oi.get('buildup_today') or 'n/a'}), yesterday "
        f"{oi.get('buildup_prev_session') or 'n/a'}"
        if oi
        else "no futures",
        f"filings since yesterday: {filings.get('filings_since_prev_close')} "
        f"({filings.get('material_since_prev_close')} material)"
        if filings
        else "filings: n/a",
    ]
    return f"  {stock['symbol']}: " + " | ".join(parts)


def _search_line(search: Mapping[str, Any]) -> str:
    if search.get("type") in ("open_page", "find_in_page"):
        return f"read page {search.get('url')}"
    what = search.get("query") or search.get("url")
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
        basis: Any = None,
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
        self.expiries: dict[str, date] = {}  # the near month's
        self.oi_daily: dict[str, dict] = {}
        self.market: MarketHistory | None = None
        self.previous_session: date | None = None
        self.marks: dict[str, Decimal] = {}
        self.profiles = dict(profiles or {})
        self.capture_dir = capture_dir
        self._captured: set[str] = set()
        self._stream = stream
        self.live = live
        self.basis = basis
        self._session_dates: dict[str, tuple[date, ...]] = {}
        self._supervisor: StreamSupervisor | None = None
        self._stream_stop = threading.Event()
        self._feed: Any = None  # the current live connection
        self._dropped = 0  # ticks it dropped (buffer full) already handled
        self._rejected = 0  # ticks it could not read, already logged
        self._checked = 0  # minute of the last stream check
        self.rank = {symbol: i for i, symbol in enumerate(self.universe)}
        self._filing_failures = 0
        self._filings_due: list[tuple[Callable, Callable, date, date]] = []
        self._clients = threading.local()
        self._client_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=settings.workers)
        self._ai_pool = ThreadPoolExecutor(max_workers=settings.ai_workers)
        self._exit_data: Any = None
        self._reports: list[dict[str, Any]] = []
        # Per AI: model calls, stock verdicts, LONG/SHORT verdicts, searches, paper
        # entries this run; spend starts from today's journal so a restart shows
        # the day's total.
        self._stats = {
            book.name: dict.fromkeys(
                ("calls", "asked", "long", "short", "searches", "entered", "refused"),
                0,
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
        contracts: dict[str, list[dict[str, Any]]] = {}
        for row in self.data.nfo_futures():
            name = row.get("name")
            if name in self.universe:
                self.fno.add(name)
                contracts.setdefault(name, []).append(row)
        for name, rows in contracts.items():
            near_next = sorted(rows, key=lambda r: r["expiry"])[:2]
            self.near_next[name] = [
                Instrument("NFO", r["tradingsymbol"]) for r in near_next
            ]
            self.expiries[name] = near_next[0]["expiry"]
        self.log(
            f"loading history for {len(self.universe)} stocks "
            f"(~{len(self.universe) * 2 * 0.4 / 60:.0f} minutes at the Kite rate limit)"
        )
        loaded = self._parallel(self._load_history, self.universe)
        self.histories = {s: h for s, h in loaded.items() if h is not None}
        if self.basis is not None:
            self.histories = self._rebase(self.histories)
        self.log(
            f"history ready for {len(self.histories)} of {len(self.universe)} stocks; "
            f"{len(self.fno)} are F&O (closing auction, futures OI)"
        )
        sessions = list(nifty_dates)
        expiries = monthly_expiries(sessions + [today])
        since = max((e for e in expiries if e < today), default=date.min)
        self.oi_daily = self._parallel(
            lambda s: self._load_oi(s, sessions[-FUTURES_SESSIONS:], since),
            sorted(self.fno & set(self.histories)),
        )
        self.log(
            f"futures OI ready for {sum(map(bool, self.oi_daily.values()))} of "
            f"{len(self.oi_daily)} F&O stocks"
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
                if "session_dates" in saved:
                    self._session_dates[symbol] = tuple(
                        date.fromordinal(int(d)) for d in saved["session_dates"]
                    )
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
        kept_dates = [d for d, _ in sessions[-25:]]
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache,
                sessions=np.array(kept).reshape(-1, LAST + 1, 5),
                daily=values,
                dates=np.array([d.toordinal() for d in dates]),
                session_dates=np.array([d.toordinal() for d in kept_dates]),
            )
        self._session_dates[symbol] = tuple(kept_dates)
        return StockHistory(tuple(kept), values, tuple(dates))

    def _rebase(self, histories: Mapping[str, StockHistory]) -> dict[str, StockHistory]:
        """Daily history on NSE's price basis (paper/basis.py), minute-history
        volume alike; a stock whose basis cannot be established is left out today.
        If the step itself fails, Kite's adjusted history is kept and logged."""
        sessions = sorted({d for h in histories.values() for d in h.daily_dates})
        try:
            notes = self.basis.load(sessions, list(histories), self.today)
            kept, left_out, record = {}, {}, {}
            for symbol, history in histories.items():
                result = self.basis.rebase(
                    symbol, history.daily_dates, history.daily, self.today
                )
                if result.daily is None:
                    left_out[symbol] = result.reason
                    continue
                minutes = history.sessions
                dates = self._session_dates.get(symbol, ())
                if result.volume_scale and len(dates) == len(minutes):
                    minutes = tuple(
                        _scale_volume(rows, result.volume_scale.get(day, 1.0))
                        for day, rows in zip(dates, minutes, strict=True)
                    )
                kept[symbol] = StockHistory(minutes, result.daily, history.daily_dates)
                if result.events or result.volume_scale:
                    record[symbol] = {
                        "events": result.events,
                        "volume_scale": {
                            day.isoformat(): scale
                            for day, scale in result.volume_scale.items()
                        },
                    }
        except Exception as error:
            self.log(
                f"price basis failed ({type(error).__name__}: {error}); "
                "using Kite's adjusted history today"
            )
            self._event("error", None, {"error": "price_basis", "detail": str(error)})
            return dict(histories)
        for note in notes:
            self.log(f"price basis: {note}")
        changed = sum(
            not np.array_equal(kept[s].daily, histories[s].daily) for s in kept
        )
        actions = sum(len(r["events"]) for r in record.values())
        named = [f"{s} ({why})" for s, why in sorted(left_out.items())]
        self.log(
            f"price basis: NSE's official daily prices for {len(kept)} stocks "
            f"({changed} differ from Kite's adjusted history; {actions} structural "
            "actions; dividends not adjusted)"
            + (
                f"; left out today ({len(named)}): "
                + ", ".join(named[:10])
                + (f" and {len(named) - 10} more" if len(named) > 10 else "")
                if named
                else ""
            )
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.cache_dir / "basis.npz", **{s: h.daily for s, h in kept.items()}
            )
            (self.cache_dir / "basis.json").write_text(
                json.dumps(
                    {"notes": notes, "left_out": left_out, "stocks": record},
                    indent=1,
                    default=str,
                ),
                "utf-8",
            )
        return kept

    def _load_oi(self, symbol: str, sessions: Sequence[date], since: date) -> dict:
        """The previous session's futures facts from the near and next contracts."""
        contracts = [c.trading_symbol for c in self.near_next.get(symbol, [])]
        if not contracts:
            return {}
        client = self._client()
        try:
            bars = {
                c: client.get_futures_daily(c, sessions[0], sessions[-1])
                for c in contracts
            }
        except Exception:
            return {}
        return previous_session_futures(bars, contracts[0], sessions, since)

    def _refresh_filings(self, *, initial: bool = False) -> None:
        if self.news is None or self.nse is None or self._filing_failures >= 3:
            return
        today = self.today
        if initial:
            # Once, retried until done; whole days, as filings go on after a run ends.
            self._filings_due = [
                (
                    self.nse.board_meetings,
                    self.news.add_meetings,
                    today,
                    today + timedelta(days=3),
                ),
                (
                    self.nse.corporate_actions,
                    self.news.add_actions,
                    today,
                    today + timedelta(days=1),
                ),
            ] + [
                (self.nse.announcements, self.news.add_announcements, day, day)
                for day in (today - timedelta(days=n) for n in range(WEEK_DAYS, 0, -1))
            ]
        try:
            while self._filings_due:
                fetch, add, start, end = self._filings_due[0]
                add(fetch(start, end))
                self._filings_due.pop(0)
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
        """Scan, ask the AIs about every candidate, paper-trade LONG / SHORT."""
        today = self.today
        decision_at = decision_close(today, minute)
        skips = [book.broker.excluded(decision_at) for book in self.books]
        excluded = skips[0]
        everywhere = set.intersection(*skips)
        symbols = [s for s in self.histories if s not in everywhere]

        def bars(symbol: str) -> np.ndarray | None:
            return self._today_bars(symbol, minute)

        fetch_started = self.clock.now()
        age = (fetch_started - decision_at).total_seconds()
        if age > self.settings.max_input_age.total_seconds():
            return self._skip(decision_at, f"started {age:.0f} s after the bar closed")
        if self._stream is not None:
            # Live run: only the stream's bars, complete and fresh; no slow fallback.
            problem = self._stream_problem(minute, decision_at, bars)
            if problem is not None:
                return self._skip(decision_at, problem)
            unrepaired = set(self.live.unrepaired())
            symbols = [s for s in symbols if s not in unrepaired]
            fetch_started = self.clock.now()
        rough: list[Estimate] = []
        rough_skipped: dict[str, int] = {}
        live = self._stream is not None or self._live_ready(minute)
        if live:
            try:
                rough, rough_skipped = self._scan(
                    minute,
                    {s: self.live.bars(s, minute) for s in symbols},
                    self.live.bars("NIFTY", minute),
                    _filled(self.live.bars("INDIAVIX", minute)),
                )
            except ValueError as error:
                if self._stream is not None:
                    return self._skip(decision_at, f"live bars: {error}")
                live = False
        not_reached: list[str] = []
        if live:
            floor = self.settings.threshold - self.settings.prescreen_margin
            probe = [e for e in rough if e.symbol in self.fno][: self.settings.oi_probe]
            # Highest pre-screen first: if time runs out, the margin is what waits.
            fetch = list(
                dict.fromkeys(
                    [e.symbol for e in rough if e.probability >= floor]
                    + [e.symbol for e in rough[:10] + probe]
                )
            )
            fetched = self._parallel(bars, ["NIFTY", "INDIAVIX"])
            deadline = decision_at + self.settings.rescore_deadline
            for start in range(0, len(fetch), 15):
                if self.clock.now() >= deadline:
                    not_reached = fetch[start:]
                    break
                fetched |= self._parallel(bars, fetch[start : start + 15])
            fetch = fetch[: len(fetch) - len(not_reached)]
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
                "not_rechecked_in_time": not_reached,
                "seconds": round((fetch_finished - fetch_started).total_seconds(), 1),
                "excluded_held_or_traded": len(excluded),
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
            + (
                f", {len(not_reached)} near the threshold not re-checked in time "
                "(next scan)"
                if not_reached
                else ""
            )
            + ") | NIFTY "
            + _show(market.get("nifty_change_today_pct"), "+.2f")
            + "% today | VIX "
            + _show(market.get("india_vix"), ".2f")
            + " ====="
        )
        self.log(
            f"  scanner picks ({len(ranked)}), {self._target_label()} >= "
            f"{self.settings.threshold:.0%}: "
            + (
                ", ".join(
                    f"{e.symbol} {e.probability:.0%}"
                    + (" F&O" if e.symbol in self.fno else "")
                    for e in ranked[:40]
                )
                or "none"
            )
            + (f" ... +{len(ranked) - 40} more" if len(ranked) > 40 else "")
        )
        below = [e for e in estimates if e.probability < self.settings.threshold]
        self.log(
            "  next best: "
            + (", ".join(f"{e.symbol} {e.probability:.0%}" for e in below[:5]) or "-")
        )
        if len(ranked) > len(candidates):
            held = [e.symbol for e in ranked if e.symbol in excluded]
            self.log(
                f"  {self.books[0].label}already held or traded today: "
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
        # As the scanner saw them: a minute without trades is the previous close.
        bars = {e.symbol: fill_quiet_minutes(stocks[e.symbol], minute) for e in ranked}
        nifty_now = float(nifty[minute, 3])
        self._capture_books(capture, ranked, depth)
        age = (self.clock.now() - decision_at).total_seconds()
        if age > self.settings.max_input_age.total_seconds():
            self._save_capture(capture)
            return self._skip(
                decision_at, f"inputs ready {age:.0f} s after the bar closed"
            )
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
                        self._request(
                            as_of, mine[start : start + size], depth, bars, nifty_now
                        ),
                    )
                    for number, start in enumerate(range(0, len(mine), size), 1)
                ]
        stocks_seen = {
            stock["symbol"]: stock
            for plan in plans.values()
            for _, _, request in plan
            for stock in (request.document or {}).get("stocks", [])
        }
        self.log(
            "  stocks sent to the AI (symbol and company only; it researches each):"
            if self.settings.ai_input == "identity"
            else "  inputs sent to the AI (also searches the web itself):"
        )
        for estimate in ranked:
            if estimate.symbol in stocks_seen:
                self.log(_inputs_line(stocks_seen[estimate.symbol]))
        # Every AI's batches go out at once; the answers are then handled AI by AI,
        # batch by batch, in a fixed order (a readable log, fills in a set order).
        calls = {
            (index, cycle_id): self._ai_pool.submit(book.decider.decide, request)
            for index, (book, key) in enumerate(zip(self.books, keys, strict=True))
            for cycle_id, _batch, request in plans.get(key, [])
        }
        entered: list[str] = []
        outcome = []
        for index, (book, key) in enumerate(zip(self.books, keys, strict=True)):
            done = []
            for cycle_id, batch, _request in plans.get(key, []):
                result = calls[(index, cycle_id)].result()
                done += self._ask(book, decision_at, cycle_id, batch, result, capture)
            if book is self.books[0]:
                entered += done
            outcome.append(
                f"{book.label.strip() or book.name} "
                + ("entered " + ", ".join(done) if done else "no trade")
            )
        self._save_capture(capture)
        self.write_report()
        self.log(f"----- {decision_at:%H:%M} result: " + " | ".join(outcome))
        self._log_books()
        return {
            "status": "ok",
            "scored": scored,
            "candidates": len(candidates),
            "entered": entered,
        }

    def _target_label(self) -> str:
        move = self.bundle.move
        return (
            f"P(+-{move:.0%} move before close)"
            if move is not None
            else "P(+1% before close)"
        )

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

    def _stream_problem(
        self,
        minute: int,
        decision_at: datetime,
        bars: Callable[[str], np.ndarray | None],
    ) -> str | None:
        """Why the live stream cannot serve this decision (None: it can).

        A gap (a connection after the open, or ticks dropped from a full buffer)
        is repaired by seeding every instrument from Kite's historical bars:
        normally while waiting for the decision (``_repair_before``), else here,
        when the decision goes ahead only if its inputs are still fresh. Then the
        stream must have processed a tick past the decision bar's close (so it
        has seen the whole minute) and NIFTY must have every minute.
        """
        live = self.live
        silent = live.silent_seconds() if live is not None else None
        if silent is None:
            return "the live stream has not started"
        if silent > self.settings.stream_silence.total_seconds():
            return f"the live stream has been silent for {silent:.0f} s"
        now_minute = minute_index(self.clock.now())
        dropped = getattr(self._feed, "dropped", 0)
        if dropped > self._dropped:
            # Lost after the last check: those minutes are repaired like a gap.
            self.log(
                f"live stream dropped {dropped - self._dropped} ticks (buffer full); "
                "its bars are repaired from history"
            )
            live.mark_gap(now_minute, first=self._checked)
            self._dropped = dropped
        rejected = getattr(self._feed, "rejected", 0)
        if rejected > self._rejected:
            # Only the pre-screen uses these bars; the AI's come from history.
            self.log(f"live stream could not read {rejected - self._rejected} ticks")
            self._rejected = rejected
        self._checked = now_minute
        gap = live.gap()
        if gap is not None:
            last = gap[1]
            if last is None:
                return "the live stream has not connected"
            if last >= minute:
                return (
                    "live bars are incomplete until "
                    f"{decision_close(self.today, last):%H:%M} (connection or lost "
                    "ticks); repaired before the next decision"
                )
            self._repair(live.unrepaired(), minute - 1, bars)
            if {"NIFTY", "INDIAVIX"} & set(live.unrepaired()):
                return "index bars could not be repaired"
            age = (self.clock.now() - decision_at).total_seconds()
            if age > self.settings.max_input_age.total_seconds():
                return f"live bars repaired; {age:.0f} s after the bar closed"
        # Every tick of the decision minute has been processed once a later one has.
        target = decision_at + self.settings.stream_catch_up
        give_up = self.clock.now() + self.settings.stream_silence
        while (latest := live.latest()) is None or latest < target:
            if self.clock.now() >= give_up:
                return "the live stream is behind" + (
                    f" (newest tick {latest:%H:%M:%S})" if latest else ""
                )
            self.clock.sleep_until(self.clock.now() + timedelta(seconds=0.5))
        if not live.complete("NIFTY", minute):
            return "NIFTY minutes are missing from the live stream"
        return None

    def _today_bars(self, symbol: str, minute: int) -> np.ndarray | None:
        """Today's minute bars 0..minute from Kite's historical API (None: failed)."""
        try:
            candles = self._client().get_historical_candles(
                Instrument("NSE", symbol),
                ist(self.today, 9, 15),
                decision_close(self.today, minute),
                CandleInterval.ONE_MINUTE,
            )
        except Exception:
            return None
        return _array(candles, self.today, minute + 1)

    def _repair_before(self, moment: datetime) -> None:
        """While waiting for the decision at ``moment``, repair a stream gap as
        soon as Kite has published its minutes, so that decision can use the
        stream (rather than repairing at it, which leaves its inputs stale)."""
        if self._stream is None or self.live is None:
            return
        while (now := self.clock.now()) < moment:
            gap = self.live.gap()
            due = None
            if gap is not None and gap[1] is not None:
                # A minute after the gap's last minute closed: Kite has its bars.
                due = decision_close(self.today, gap[1] + 1)
                due += self.settings.decision_delay
            if due is not None and now >= due:
                upto = minute_index(now) - 2  # closed at least a minute ago
                self._repair(
                    self.live.unrepaired(), upto, lambda s: self._today_bars(s, upto)
                )
                # Anything that failed is retried after a pause, or at the decision.
                wake = self.clock.now() + timedelta(seconds=15)
            else:
                wake = now + timedelta(seconds=5)
                if due is not None:
                    wake = min(wake, due)
            self.clock.sleep_until(min(moment, wake))

    def _repair(
        self, symbols: Sequence[str], upto: int, bars: Callable[[str], Any]
    ) -> None:
        """Seed the live bars of ``symbols`` through ``upto`` from historical bars."""
        started = self.clock.now()
        fetched = self._parallel(bars, symbols)
        self.live.seed({s: b for s, b in fetched.items() if b is not None}, upto)
        failed = sorted(s for s, b in fetched.items() if b is None)
        seconds = (self.clock.now() - started).total_seconds()
        self.log(
            f"live bars repaired from Kite's historical bars: "
            f"{len(fetched) - len(failed)} instruments through "
            f"{decision_close(self.today, upto) - timedelta(minutes=1):%H:%M} "
            f"in {seconds:.0f} s"
            + (f"; {len(failed)} failed, retried next cycle" if failed else "")
        )
        self._event(
            "live_repair",
            None,
            {"instruments": len(fetched), "failed": failed, "through_minute": upto},
        )

    def _skip(self, decision_at: datetime, reason: str) -> dict[str, Any]:
        self.log(f"{decision_at:%H:%M} skipped: {reason}")
        self._event(
            "skip", None, {"decision_at": decision_at.isoformat(), "reason": reason}
        )
        return {"status": "skipped", "reason": reason}

    def _connect(self, instruments: Sequence[Instrument]) -> Any:
        """Open the live stream (connected and subscribed when this returns)."""
        stream = self._stream(instruments)
        now = self.clock.now()
        if now >= ist(self.today, 9, 15):
            # Connected after the open (late start or a reconnect): ticks before
            # this connection may be missing. Before the open nothing is lost.
            last = minute_index(now + self.settings.stream_settle)
            self.live.mark_gap(last)
            self.log(
                f"live stream connected at {now:%H:%M:%S}; bars through "
                f"{decision_close(self.today, last) - timedelta(minutes=1):%H:%M} "
                "are taken from Kite's history once published"
            )
        self._feed, self._dropped, self._rejected = stream, 0, 0
        return stream

    def _start_stream(self, until: datetime) -> None:
        if self._stream is None or self.live is None or self.clock.simulated:
            return
        instruments = [Instrument("NSE", s) for s in self.live.symbols]

        def run() -> None:
            self.clock.sleep_until(max(self.clock.now(), ist(self.today, 9, 14)))
            while not self._stream_stop.is_set():
                seconds = (until - self.clock.now()).total_seconds()
                if seconds <= 0:
                    return
                self._supervisor = StreamSupervisor(
                    lambda: self._connect(instruments),
                    self.live.on_tick,
                    retry_on=(RuntimeError, OSError),
                )
                self.log(f"live stream starting for {len(instruments)} instruments")
                try:
                    report = self._supervisor.run(seconds)
                    self.log(
                        f"live stream ended: {report.ticks} ticks, "
                        f"{report.reconnects} reconnects, {report.failures} failures "
                        f"({report.stopped_because}); last connection dropped "
                        f"{getattr(self._feed, 'dropped', 0)} and could not read "
                        f"{getattr(self._feed, 'rejected', 0)} ticks"
                    )
                    if report.stopped_because in ("deadline", "stopped"):
                        return
                except Exception as error:
                    self.log(f"live stream stopped ({type(error).__name__}: {error})")
                self.log("decisions are skipped until the live stream is back")
                self._stream_stop.wait(30)

        threading.Thread(target=run, name="stream", daemon=True).start()

    def _ask(
        self,
        book: Book,
        decision_at: datetime,
        cycle_id: str,
        batch: Sequence[Estimate],
        result: Any,
        capture: dict[str, Any] | None = None,
    ) -> list[str]:
        """Record and log one AI answer, then paper-trade its LONG / SHORT verdicts."""
        book.journal.record(cycle_id, decision_at, book.name, result)
        reply = json.loads(result.response_json or "{}")
        searches = reply.get("web_searches") or []
        found = [s for s in searches if s.get("type") in (None, "search")]
        stats = self._stats[book.name]
        stats["calls"] += result.usage is not None
        stats["asked"] += len(result.verdicts)
        stats["long"] += sum(v.verdict == "LONG" for v in result.verdicts)
        stats["short"] += sum(v.verdict == "SHORT" for v in result.verdicts)
        stats["searches"] += len(found)
        stats["cost"] += result.cost_usd
        pages = len(searches) - len(found)
        self.log(
            f"{book.label}AI {result.status}: {len(result.verdicts)} stocks, "
            f"{len(found)} web searches"
            + (f", {pages} pages read" if pages else "")
            + f", ${result.cost_usd:.4f}, {result.latency_seconds:.1f} s"
        )
        for search in searches:
            self.log(f"  {book.label}{_search_line(search)}")
        for verdict in result.verdicts:
            self.log(
                f"  {book.label}{verdict.symbol}: {verdict.verdict}"
                + (
                    (f" ({verdict.reason})" if verdict.reason else "")
                    if self.settings.ai_input == "identity"
                    else f" (P(green at close) {verdict.confidence:.2f}) "
                    f"{verdict.reason}"
                )
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
        trades = [v for v in result.verdicts if v.verdict in ("LONG", "SHORT")]
        return self._execute(trades, batch, cycle_id, book, capture)

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
            "nifty_closes": self.market.nifty_closes[-60:].tolist(),
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
        """Complete minute bars of every stock captured today, by this run or by an
        earlier run of the same day (a restart), for revision checks and hindsight."""
        if self.capture_dir is None:
            return
        today = self.today
        folder = self.capture_dir / today.isoformat()
        for path in sorted(folder.glob("[0-9]*.json")):
            try:
                self._captured |= set(json.loads(path.read_text("utf-8"))["stocks"])
            except (OSError, ValueError, KeyError, TypeError):
                self.log(
                    f"capture {path.name} unreadable; its stocks may lack final bars"
                )
        if not self._captured:
            return
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
        bars: Mapping[str, np.ndarray],
        nifty_now: float,
    ) -> DecisionRequest:
        """Facts only (v3): no scanner output, stocks in alphabetical order.
        ``bars``: each stock's minute bars up to the decision minute. With
        ``ai_input`` "identity": only each stock's symbol and company."""
        if self.settings.ai_input == "identity":
            stocks = [
                {
                    "symbol": e.symbol,
                    "company": (self.profiles.get(e.symbol) or {}).get("company"),
                }
                for e in sorted(batch, key=lambda e: e.symbol)
            ]
            return DecisionRequest(
                as_of,
                {},
                tuple(Candidate(s["symbol"], {}) for s in stocks),
                document={"as_of": as_of.isoformat(), "stocks": stocks},
            )
        since = ist(self.previous_session, 15, 30)
        nifty = daily_now(self.market.nifty_closes, nifty_now)
        stocks = []
        for estimate in sorted(batch, key=lambda e: e.symbol):
            symbol = estimate.symbol
            snapshot = depth.get(f"NSE:{symbol}")
            intraday, daily = bars[symbol], self.histories[symbol].daily
            price = float(intraday[-1, 3])
            stocks.append(
                stock_v3(
                    symbol,
                    estimate.features,
                    price,
                    now=daily_now(
                        daily[:, 3],
                        price,
                        daily[:, 1],
                        float(np.nanmax(intraday[:, 1])),
                        nifty,
                    ),
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
        market = market_v3(batch[0].features, nifty)
        return DecisionRequest(
            as_of,
            market,
            tuple(Candidate(s["symbol"], {}) for s in stocks),
            document={"as_of": as_of.isoformat(), "market": market, "stocks": stocks},
        )

    def _oi_context(
        self, symbol: str, depth: Mapping[str, DepthSnapshot]
    ) -> dict | None:
        """Futures facts: the previous session's (loaded at the start) and now
        (this cycle's quotes of the same contracts and of the stock)."""
        previous = self.oi_daily.get(symbol)
        contracts = [c.trading_symbol for c in self.near_next.get(symbol, [])]
        if not previous or not contracts:
            return None
        quotes = {c: depth[f"NFO:{c}"] for c in contracts if f"NFO:{c}" in depth}
        days = (self.expiries[symbol] - self.today).days
        return (
            previous
            | {"days_to_expiry": float(days)}
            | live_futures(
                previous["previous_oi"],
                quotes,
                contracts[0],
                depth.get(f"NSE:{symbol}"),
            )
        )

    def _execute(
        self,
        trades: Sequence[Verdict],
        batch: Sequence[Estimate],
        cycle_id: str,
        book: Book,
        capture: dict[str, Any] | None = None,
    ) -> list[str]:
        if not trades:
            return []
        probability = {e.symbol: e.probability for e in batch}
        fresh = self._depth(
            [e for e in batch if e.symbol in {v.symbol for v in trades}]
        )
        entered = []
        for verdict in trades:
            snapshot = fresh.get(f"NSE:{verdict.symbol}")
            result = book.broker.enter(
                verdict.symbol,
                verdict.verdict,
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
                entered.append(verdict.symbol)
                self._stats[book.name]["entered"] += 1
                action = "BUY" if p.side == "LONG" else "SHORT SELL"
                self.log(
                    f"  {book.label}>>> PAPER {p.side} {p.symbol}: {action} "
                    f"{p.quantity} shares @ Rs {p.entry_price:,.2f} = Rs "
                    f"{p.entry_value:,.0f}; target Rs {p.target_price:,.2f}, else "
                    "out at 15:00"
                )
            else:
                self._stats[book.name]["refused"] += 1
                self.log(
                    f"  {book.label}>>> {verdict.verdict} {verdict.symbol} refused: "
                    f"{result.reason}"
                )
            if capture is not None:
                capture["executions"].append(
                    {
                        "book": book.name,
                        "symbol": verdict.symbol,
                        "side": verdict.verdict,
                        "cycle_id": cycle_id,
                        "depth": _snapshot(snapshot) if snapshot else None,
                        "position_id": result.position.id if result.position else None,
                        "refused": None if result.position else result.reason,
                    }
                )
        return entered

    # --- exits ----------------------------------------------------------------------
    def exit_cycle(self) -> list[str]:
        """Check open positions' completed minutes for the target, close what is
        still open at 15:00, refresh marks."""
        now = self.clock.now()
        closed = []
        held = [(b, p) for b in self.books for p in b.broker.open_positions()]
        # One request per stock, from its earliest unchecked minute: a stock can be
        # held by several AIs and, without the one-per-stock rule, more than once.
        starts: dict[str, datetime] = {}
        for _, p in held:
            starts[p.symbol] = min(
                starts.get(p.symbol, p.checked_through), p.checked_through
            )
        fetched: dict[str, Any] = {}
        for symbol, start in starts.items():
            try:
                fetched[symbol] = self._candles(
                    self.exit_data, symbol, start, now, CandleInterval.ONE_MINUTE
                )
            except Exception as error:
                fetched[symbol] = error
        for book, position in held:
            bars = fetched[position.symbol]
            if isinstance(bars, Exception):
                self.log(f"{book.label}exit check failed for {position.symbol}: {bars}")
                continue
            result = book.broker.apply_bars(position, bars, now)
            deadline = book.broker.deadline(position)
            if result is None and now >= deadline + LAST_BAR_WAIT:
                # No minute bar before 15:00 to price it: the last traded price.
                price = self._last_price(position.symbol)
                if price is not None:
                    result = book.broker.close_at(
                        position, deadline, price, "time_1500_ltp"
                    )
            if result is not None:
                closed.append(result.symbol)
                self.log(
                    f"{book.label}<<< PAPER EXIT {result.side} {result.symbol}: "
                    f"{result.quantity} shares @ Rs {result.exit_price:,.2f} at "
                    f"{result.exit_at:%H:%M} ({result.exit_reason}); entered "
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

    def _last_price(self, symbol: str) -> Decimal | None:
        try:
            prices = self.exit_data.get_ltp([Instrument("NSE", symbol)])
        except Exception as error:
            self.log(f"price request failed for {symbol}: {type(error).__name__}")
            return None
        return prices[0].price if prices else None

    def write_report(self) -> dict[str, Any]:
        reports = []
        for book in self.books:
            report = build_report(
                book.broker.store,
                self.marks,
                self.clock.now(),
                book.broker.config.capital,
                target=book.broker.config.target,
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
                f"Rs {float(summary['deployed']):,.0f} open value, "
                f"Rs {float(summary['unrealised_net_pnl_if_closed_now']):+,.0f} if "
                f"closed now | closed today: {day['trades'] if day else 0}"
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
                f"{stats['asked']} stock verdicts, {stats['long']} LONG, "
                f"{stats['short']} SHORT, {stats['searches']} web searches | paper "
                f"entries {stats['entered']}, refused {stats['refused']}"
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
        self._stream_stop.set()
        if self._supervisor is not None:
            self._supervisor.stop()
        self.write_report()
        self._log_day()
        self._pool.shutdown(wait=False)
        self._ai_pool.shutdown(wait=False)
        self.log(f"session finished; report in {self.report_dir}")

    def _run_events(self, events: Sequence[tuple[datetime, str, int]]) -> None:
        for at, kind, minute in events:
            if self.clock.now() > at + timedelta(minutes=2 if kind == "exit" else 3):
                continue
            if kind == "decision":
                self._safely(partial(self._repair_before, at), -1)
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
