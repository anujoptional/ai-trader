"""Replays of past sessions for the autonomous agent: the same prompt, tools, account
and session timeline as a live day, run on Kite minute bars, NSE daily data and NSE
filings as they stood at each simulated moment.

Nothing from after the simulated moment reaches the model: quotes and bars stop at
the clock (the current minute shows only its opening trade), daily data and filings
are as of that moment, and web search is off because a search would return later
news. The model's response time moves the clock, so prices move while it decides,
as live. The days replayed should be after the model's training data (June 2026 for
Claude Opus 5.5), so it cannot remember them.

What a replay cannot reproduce: the order book (orders fill against a modelled book:
a spread by liquidity plus levels sized from the stock's usual volume per minute),
news beyond NSE filings, and pre-open indicative prices. Stops, targets and alerts
are checked on each minute's high and low from the minute after they were set; a
stop gapped through fills at the minute's open less the spread.
"""

from __future__ import annotations

import json
import sqlite3
import time as wall
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.agent.account import Book, PaperAccount
from ai_trader.agent.brain import AgentTools, AutonomousAgent, MessagesClient
from ai_trader.agent.market import Filings, MarketView, NseDaily, Quote
from ai_trader.agent.session import (
    OPENING_FILL,
    REVIEW,
    SessionRunner,
    close_everything,
    report,
)
from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.news.nse import Announcement

SESSION_OPEN = time(9, 15)
MINUTES = 375
INDEX_FILES = {
    "NIFTY 50": "NIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "NIFTY FIN SERVICE": "FINNIFTY",
    "NIFTY MIDCAP 150": "NIFTYMIDCAP150",
    "NIFTY 500": "NIFTY500",
    "NIFTY IT": "NIFTYIT",
    "NIFTY AUTO": "NIFTYAUTO",
    "NIFTY FMCG": "NIFTYFMCG",
    "NIFTY PHARMA": "NIFTYPHARMA",
    "NIFTY METAL": "NIFTYMETAL",
    "NIFTY REALTY": "NIFTYREALTY",
    "INDIA VIX": "INDIAVIX",
}
LIMITS = (
    "Order books are not shown: a market order fills near the last price, paying a "
    "spread that is wider for less liquid stocks and a cost that grows with the "
    "order's size against the stock's usual volume per minute. Company news comes "
    "only from NSE filings."
)
# Half-spread by average daily traded value (Rs crore), as a fraction of the price.
SPREADS = ((1000.0, 0.0001), (200.0, 0.0002), (50.0, 0.0004), (0.0, 0.0008))
LEVEL_STEPS = (1, 2, 3, 5, 8)
LEVEL_SHARE = 0.1
# A split or bonus after the replay day: the previous close is off by more than this.
STRUCTURAL = 0.08


def half_spread(daily_value_cr: float | None) -> float:
    value = daily_value_cr or 0.0
    return next(half for floor, half in SPREADS if value >= floor)


# --- data ---------------------------------------------------------------------------
def save_minutes(path: Path, candles: Sequence[OHLCVCandle]) -> None:
    np.savez_compressed(
        path,
        t=np.array([int(c.timestamp.timestamp()) for c in candles], dtype=np.int64),
        ohlc=np.array(
            [
                [float(c.open), float(c.high), float(c.low), float(c.close)]
                for c in candles
            ],
            dtype=np.float64,
        ).reshape(-1, 4),
        v=np.array(
            [np.nan if c.volume is None else float(c.volume) for c in candles],
            dtype=np.float64,
        ),
    )


def download_minutes(
    kite: Any,
    symbols: Iterable[str],
    start: date,
    end: date,
    folder: Path,
    log: Callable[[str], None],
) -> dict[str, str]:
    """Kite minute bars from ``start`` to ``end`` per symbol (one request each);
    symbols already saved are skipped. What failed and why."""
    folder.mkdir(parents=True, exist_ok=True)
    begin = datetime.combine(start, SESSION_OPEN, INDIA_TIMEZONE)
    finish = datetime.combine(end, time(15, 30), INDIA_TIMEZONE)
    todo = [s for s in dict.fromkeys(symbols) if not (folder / f"{s}.npz").exists()]
    failed: dict[str, str] = {}
    for count, symbol in enumerate(todo, 1):
        try:
            candles = kite.get_historical_candles(
                Instrument("NSE", symbol), begin, finish, CandleInterval.ONE_MINUTE
            )
        except Exception as error:
            failed[symbol] = f"{type(error).__name__}: {error}"[:200]
            continue
        save_minutes(folder / f"{symbol}.npz", candles)
        if count % 50 == 0 or count == len(todo):
            log(f"minute bars: {count}/{len(todo)} ({len(failed)} failed)")
    return failed


class MinuteBars:
    """One session's minute bars per symbol (minute 0 = 09:15, NaN = no trade) and
    the bars of the days before. Stock prices are put on the replay day's NSE basis:
    when the previous session's last bar is more than 8% off NSE's previous close (a
    split or bonus after the replay day), the bars are rescaled."""

    def __init__(
        self,
        folder: Path,
        day: date,
        symbols: Iterable[str],
        reference: Mapping[str, float] | None = None,
    ) -> None:
        self.day = day
        self.open_at = datetime.combine(day, SESSION_OPEN, INDIA_TIMEZONE)
        base = int(self.open_at.timestamp())
        end = base + MINUTES * 60
        self.rescaled: dict[str, float] = {}
        self._all: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._day: dict[str, np.ndarray] = {}
        for symbol in dict.fromkeys(symbols):
            path = folder / f"{symbol}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                t, ohlc, volume = data["t"], data["ohlc"], data["v"]
            keep = t < end
            t, ohlc, volume = t[keep], ohlc[keep].copy(), volume[keep].copy()
            before = t < base
            nse = (reference or {}).get(symbol)
            if nse and before.any():
                factor = nse / ohlc[before][-1, 3]
                # Smaller gaps are NSE's closing price (auction or last-30-minute VWAP).
                if abs(factor - 1) > STRUCTURAL:
                    ohlc *= factor
                    volume /= factor
                    self.rescaled[symbol] = round(float(factor), 6)
            self._all[symbol] = (t, ohlc, volume)
            today = ~before
            if not today.any():
                continue
            grid = np.full((MINUTES, 5), np.nan)
            index = ((t[today] - base) // 60).astype(int)
            grid[index, :4] = ohlc[today]
            grid[index, 4] = volume[today]
            self._day[symbol] = grid

    def has(self, symbol: str) -> bool:
        return symbol in self._day

    def minute(self, moment: datetime) -> int:
        return int((moment - self.open_at).total_seconds() // 60)

    def bar(self, symbol: str, minute: int) -> np.ndarray | None:
        grid = self._day.get(symbol)
        if grid is None or not 0 <= minute < MINUTES or np.isnan(grid[minute, 0]):
            return None
        return grid[minute]

    def previous_close(self, symbol: str) -> float | None:
        data = self._all.get(symbol)
        if data is None:
            return None
        before = data[0] < int(self.open_at.timestamp())
        return float(data[1][before][-1, 3]) if before.any() else None

    def last_close(self, symbol: str) -> float | None:
        grid = self._day.get(symbol)
        if grid is None:
            return None
        closes = grid[~np.isnan(grid[:, 3]), 3]
        return float(closes[-1]) if len(closes) else None

    def state(self, symbol: str, now: datetime) -> dict[str, float | None] | None:
        """What a quote shows at ``now``: completed minutes plus the current minute's
        opening trade. None before 09:15 or without bars."""
        grid = self._day.get(symbol)
        minute = self.minute(now)
        if grid is None or minute < 0:
            return None
        done = grid[: min(minute, MINUTES)]
        done = done[~np.isnan(done[:, 0])]
        current = grid[minute] if minute < MINUTES else None
        live = current is not None and not np.isnan(current[0])
        if not len(done) and not live:
            return None
        prices = [done[:, 1], done[:, 2]] + ([current[:1]] if live else [])
        traded = done[:, 4] > 0
        weight = np.nansum(done[traded, 4])
        typical = done[traded, 1:4].mean(axis=1)
        return {
            "open": float(done[0, 0] if len(done) else current[0]),
            "high": float(np.nanmax(np.concatenate(prices))),
            "low": float(np.nanmin(np.concatenate(prices))),
            "last": float(current[0] if live else done[-1, 3]),
            "volume": float(np.nansum(done[:, 4])),
            "vwap": float((typical * done[traded, 4]).sum() / weight)
            if weight > 0
            else None,
        }

    def candles(
        self, symbol: str, start: datetime, now: datetime
    ) -> tuple[OHLCVCandle, ...]:
        """Completed minute bars from ``start`` up to ``now``."""
        data = self._all.get(symbol)
        if data is None:
            return ()
        t, ohlc, volume = data
        keep = (t >= int(start.timestamp())) & (t + 60 <= int(now.timestamp()))
        return tuple(
            OHLCVCandle(
                datetime.fromtimestamp(int(stamp), INDIA_TIMEZONE),
                *(Decimal(str(round(float(x), 2))) for x in prices),
                None if np.isnan(size) else int(size),
            )
            for stamp, prices, size in zip(
                t[keep], ohlc[keep], volume[keep], strict=True
            )
        )


class ReplayClock:
    def __init__(self, start: datetime) -> None:
        self.moment = start

    def now(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> None:
        self.moment += timedelta(seconds=seconds)

    def move_to(self, moment: datetime) -> None:
        self.moment = max(self.moment, moment)


class ReplayFeed:
    """Kite-shaped quote rows and candles from minute bars at the replay clock.
    Stocks' previous close is NSE's; indices' is their previous last bar."""

    def __init__(
        self,
        bars: MinuteBars,
        previous: Mapping[str, float],
        clock: Callable[[], datetime],
    ) -> None:
        self.bars = bars
        self.previous = dict(previous)
        self._clock = clock

    def quote_rows(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        now = self._clock()
        rows = {}
        for key in keys:
            name = key.removeprefix("NSE:")
            symbol = INDEX_FILES.get(name, name)
            if not self.bars.has(symbol):
                continue
            previous = self.previous.get(symbol) or self.bars.previous_close(symbol)
            state = self.bars.state(symbol, now) or {}
            last = state.get("last") or previous
            if not last:
                continue
            rows[key] = {
                "last_price": last,
                "ohlc": {
                    "open": state.get("open") or 0,
                    "high": state.get("high") or 0,
                    "low": state.get("low") or 0,
                    "close": previous or 0,
                },
                "average_price": state.get("vwap") or 0,
                "volume": int(state.get("volume") or 0),
                "depth": {"buy": [], "sell": []},
                "last_trade_time": now,
            }
        return rows

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]:
        return self.bars.candles(
            instrument.trading_symbol, start, min(end, self._clock())
        )


class ReplayMarket(MarketView):
    """The live MarketView over replayed quotes; orders cross a modelled book."""

    def __init__(
        self,
        feed: ReplayFeed,
        scan: Sequence[str],
        profiles: dict[str, dict[str, str]],
        daily: NseDaily,
        clock: Callable[[], datetime],
        *,
        filings: Filings | None = None,
    ) -> None:
        super().__init__(
            feed,
            scan,
            profiles,
            daily,
            clock,
            filings=filings,
            quote_ttl=0.0,
            show_depth=False,
        )

    def _stats(self, symbol: str) -> dict[str, Any]:
        stats = self._averages.get(symbol)
        if stats is None:
            stats = self._averages[symbol] = self.daily.stats(symbol) or {}
        return stats

    def half_spread(self, symbol: str) -> float:
        return half_spread(self._stats(symbol).get("avg_traded_value_20d_cr"))

    def book(self, quote: Quote) -> Book:
        stats = self._stats(quote.symbol)
        half = Decimal(str(self.half_spread(quote.symbol)))
        size = max(1, int((stats.get("avg_volume_20d") or 0) / MINUTES * LEVEL_SHARE))
        cent = Decimal("0.01")
        asks = tuple(
            ((quote.last * (1 + half * step)).quantize(cent), size)
            for step in LEVEL_STEPS
        )
        bids = tuple(
            ((quote.last * (1 - half * step)).quantize(cent), size)
            for step in LEVEL_STEPS
        )
        return Book(quote.last, bids, asks)


class ReplayTools(AgentTools):
    """The agent's tools on a replay: exits and alerts run on minute bars."""

    TRADING = ("place_order", "close_position", "set_exits")

    def __init__(
        self, market: ReplayMarket, account: PaperAccount, bars: MinuteBars
    ) -> None:
        super().__init__(market, account)
        self.replay = market
        self.bars = bars
        self._checked = -1
        self._pending: list[dict[str, Any]] = []
        self._alert_minute: dict[str, int] = {}

    def run(self, name: str, args: Mapping[str, Any]) -> Any:
        if name in self.TRADING:
            self._pending += self._catch_up()
        result = super().run(name, args)
        if name == "set_alert" and isinstance(result, dict) and result.get("ok"):
            self._alert_minute[str(result.get("symbol"))] = self.bars.minute(self.now())
        return result

    def watch(self) -> list[dict[str, Any]]:
        events = self._pending + self._catch_up()
        self._pending = []
        return events

    def _catch_up(self) -> list[dict[str, Any]]:
        last = min(self.bars.minute(self.now()) - 1, MINUTES - 1)
        events: list[dict[str, Any]] = []
        for minute in range(self._checked + 1, last + 1):
            events += self._minute(minute)
        self._checked = max(self._checked, last)
        return events

    def _minute(self, minute: int) -> list[dict[str, Any]]:
        moment = self.bars.open_at + timedelta(minutes=minute)
        events: list[dict[str, Any]] = []
        for row in self.account.open_positions():
            symbol = row["symbol"]
            bar = self.bars.bar(symbol, minute)
            entered = self.bars.minute(datetime.fromisoformat(row["entry_at"]))
            if bar is None or minute <= entered:
                continue
            open_, high, low = (Decimal(str(round(float(x), 4))) for x in bar[:3])
            long = row["side"] == "LONG"
            stop = Decimal(row["stop"]) if row["stop"] else None
            target = Decimal(row["target"]) if row["target"] else None
            if stop is not None and (low <= stop if long else high >= stop):
                half = Decimal(str(self.replay.half_spread(symbol)))
                price = (
                    min(stop, open_) * (1 - half)
                    if long
                    else max(stop, open_) * (1 + half)
                )
                closed = self.account.close_at(row["id"], price, moment, "stop")
            elif target is not None and (high >= target if long else low <= target):
                price = max(target, open_) if long else min(target, open_)
                closed = self.account.close_at(row["id"], price, moment, "target")
            else:
                continue
            events += [
                {"event": f"{c['reason']}_filled", "time": f"{moment:%H:%M}", **c}
                for c in closed
            ]
        for symbol, alert in list(self.alerts.items()):
            bar = self.bars.bar(symbol, minute)
            if bar is None or minute <= self._alert_minute.get(symbol, -1):
                continue
            high, low = float(bar[1]), float(bar[2])
            if (alert.above is not None and high >= float(alert.above)) or (
                alert.below is not None and low <= float(alert.below)
            ):
                del self.alerts[symbol]
                events.append(
                    {
                        "event": "alert",
                        "time": f"{moment:%H:%M}",
                        "symbol": symbol,
                        "high": high,
                        "low": low,
                        "above": float(alert.above) if alert.above else None,
                        "below": float(alert.below) if alert.below else None,
                        "note": alert.note,
                    }
                )
        return events


class TimedClient:
    """The model's response time moves the replay clock, as the market moves live."""

    def __init__(self, client: MessagesClient, clock: ReplayClock) -> None:
        self._client = client
        self._clock = clock

    def send(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        started = wall.monotonic()
        try:
            return self._client.send(body)
        finally:
            self._clock.advance(wall.monotonic() - started)


def stored_filings(path: Path) -> Callable[[date, date], list[Announcement]]:
    """NSE filings from the local archive, by dissemination time."""

    def fetch(start: date, end: date) -> list[Announcement]:
        with closing(sqlite3.connect(path)) as db:
            rows = db.execute(
                "SELECT key, symbol, company, subject, text, attachment, "
                "disseminated_at FROM announcements "
                "WHERE disseminated_at >= ? AND disseminated_at < ?",
                (start.isoformat(), (end + timedelta(days=1)).isoformat()),
            ).fetchall()
        return [
            Announcement(
                key,
                symbol,
                company,
                subject,
                text,
                attachment,
                datetime.fromisoformat(stamp),
                {},
            )
            for key, symbol, company, subject, text, attachment, stamp in rows
        ]

    return fetch


def replay_day(
    day: date,
    client: MessagesClient,
    out_dir: Path,
    *,
    minutes: Path,
    features: Path,
    scan: Sequence[str],
    profiles: dict[str, dict[str, str]],
    news_db: Path,
    model: str,
    effort: str,
    budget: float,
    wallet: Decimal,
    leverage: int,
    log: Callable[[str], None],
    thinking: bool = True,
    stop_at: time | None = None,
    daily: NseDaily | None = None,
) -> dict[str, Any]:
    """One full replayed session; the day's report."""
    daily = daily or NseDaily(features, day)
    previous = {}
    for symbol in scan:
        frame = daily.frame(symbol)
        if frame is not None and len(frame):
            previous[symbol] = float(frame["cm_close"].iloc[-1])
    bars = MinuteBars(minutes, day, [*scan, *INDEX_FILES.values()], previous)
    clock = ReplayClock(datetime.combine(day, time(9, 0), INDIA_TIMEZONE))

    def say(message: str) -> None:
        log(f"[{day:%m-%d} {clock.now():%H:%M:%S}] {message}")

    market = ReplayMarket(
        ReplayFeed(bars, previous, clock.now),
        [s for s in scan if bars.has(s)],
        profiles,
        daily,
        clock.now,
        filings=Filings(stored_filings(news_db), clock.now),
    )
    account = PaperAccount(out_dir / "paper.sqlite", wallet, leverage)
    if account.open_positions() or account.closed_positions():
        raise FileExistsError(f"{out_dir} already holds a replay; use a new folder")
    tools = ReplayTools(market, account, bars)
    agent = AutonomousAgent(
        TimedClient(client, clock),
        model,
        effort,
        tools,
        budget_usd=budget,
        journal_dir=out_dir,
        log=say,
        thinking=thinking,
        web_search=False,
        limits=LIMITS,
    )
    say(
        f"replay: {len(market.scan)} stocks with bars, NSE daily to {daily.latest}, "
        f"rescaled {bars.rescaled or 'none'}"
    )
    runner = SessionRunner(tools, agent, say, clock.now())
    end = stop_at and datetime.combine(day, stop_at, INDIA_TIMEZONE)
    opening = datetime.combine(day, OPENING_FILL, INDIA_TIMEZONE)
    while end is None or clock.now() < end:
        moment = clock.now()
        runner.watch(moment)
        pause = runner.step(moment)
        if pause is None:
            break
        if pause > 0:
            following = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
            for waypoint in (runner.next_turn, opening):
                if moment < waypoint < following:
                    following = waypoint
            clock.move_to(following)
    for event in tools.watch():
        say(f"agent {event['event']}: {json.dumps(event, default=str)}")
    closed = close_everything(tools, say, pause=0)
    if agent.budget_left() > 0:
        try:
            agent.turn(REVIEW, [{"event": "closed_at_1510", "closed": closed}])
        except Exception as error:
            say(f"review turn failed: {type(error).__name__}: {error}")
    nifty_prev, nifty_last = bars.previous_close("NIFTY"), bars.last_close("NIFTY")
    summary = {"day": day.isoformat()} | report(account, agent)
    summary |= {
        "nifty_change_pct": round((nifty_last / nifty_prev - 1) * 100, 3)
        if nifty_prev and nifty_last
        else None,
        "stocks": len(market.scan),
        "ended": f"{clock.now():%H:%M:%S}",
        "fills": {
            "half_spread_by_daily_value_cr": [list(s) for s in SPREADS],
            "level_steps": list(LEVEL_STEPS),
            "level_share_of_minute_volume": LEVEL_SHARE,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=1), "utf-8")
    say(
        f"replay day: {summary['trades']} trades, {summary['won']} won, net Rs "
        f"{summary['net_rs']:,.0f} (charges Rs {summary['charges_rs']:,.0f}); "
        f"AI ${summary['ai_spend_usd']:.2f} over {summary['turns']} turns; NIFTY "
        f"{summary['nifty_change_pct']}%"
    )
    return summary
