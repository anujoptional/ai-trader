"""Objective market data for the autonomous agent: live quotes and order books of any
NSE stock, the indices, breadth and movers over a scan list, NSE's official daily
data (prices adjusted for corporate actions, delivery, futures and options), intraday
bars and NSE filings as disseminated. Facts only: no estimates or signals."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ai_trader.agent.account import Book
from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.news.nse import Announcement

INDICES = {
    "NIFTY 50": "NSE:NIFTY 50",
    "NIFTY BANK": "NSE:NIFTY BANK",
    "NIFTY FIN SERVICE": "NSE:NIFTY FIN SERVICE",
    "NIFTY MIDCAP 150": "NSE:NIFTY MIDCAP 150",
    "NIFTY 500": "NSE:NIFTY 500",
    "NIFTY IT": "NSE:NIFTY IT",
    "NIFTY AUTO": "NSE:NIFTY AUTO",
    "NIFTY FMCG": "NSE:NIFTY FMCG",
    "NIFTY PHARMA": "NSE:NIFTY PHARMA",
    "NIFTY METAL": "NSE:NIFTY METAL",
    "NIFTY REALTY": "NSE:NIFTY REALTY",
    "INDIA VIX": "NSE:INDIA VIX",
}
MOVER_SORTS = (
    "gainers",
    "losers",
    "gap_up",
    "gap_down",
    "up_from_open",
    "down_from_open",
    "traded_value",
    "volume_vs_20d_avg",
    "near_52w_high",
    "near_52w_low",
)
BAR_INTERVALS = {"minute": 1, "5minute": 5, "15minute": 15, "day": 0}
STOCK_FACTS = (
    "fno",
    "in_ban",
    "mwpl_oi_pct",
    "fut_oi_change_pct",
    "fut_basis_pct",
    "fut_rollover_pct",
    "fut_buildup",
    "opt_pcr_oi",
    "opt_max_pain_near",
    "opt_max_pain_dist_pct",
    "opt_call_wall_near",
    "opt_call_wall_dist_pct",
    "opt_put_wall_near",
    "opt_put_wall_dist_pct",
    "iv_atm_near",
    "iv_atm_30d",
    "iv_skew_near",
    "iv_term",
)
PRICE_COLUMNS = (
    "cm_prev_close",
    "cm_open",
    "cm_high",
    "cm_low",
    "cm_close",
    "cm_vwap",
    "cm_volume",
    "cm_turnover_lakh",
    "cm_trades",
    "cm_deliv_pct",
)
MARKET_FACTS = (
    "oi_fii_idx_fut_long_pct",
    "oi_fii_stk_fut_long_pct",
    "oi_fii_idx_call_net",
    "oi_fii_idx_put_net",
    "oi_client_idx_fut_long_pct",
    "oi_pro_idx_fut_long_pct",
    "oi_dii_idx_fut_long_pct",
    "stocks_in_ban",
)
INDEX_FACTS = (
    "spot",
    "fut_oi_change_pct",
    "fut_basis_pct",
    "fut_buildup",
    "opt_pcr_oi",
    "opt_pcr_oi_near",
    "opt_max_pain_near",
    "opt_call_wall_near",
    "opt_put_wall_near",
    "iv_atm_near",
    "iv_atm_30d",
    "iv_skew_near",
    "iv_term",
)


class QuoteSource(Protocol):
    def quote_rows(self, keys: Sequence[str]) -> dict[str, dict[str, Any]]: ...

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]: ...


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _price(value: Any) -> Decimal | None:
    number = _number(value)
    return Decimal(str(value)) if number is not None and number > 0 else None


def _plain(value: Any) -> Any:
    """JSON-friendly scalar: floats rounded, NaN to None."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating, Decimal)):
        number = float(value)
        return round(number, 4) if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT:
        return None
    return value


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old <= 0:
        return None
    return round((new / old - 1) * 100, 3)


@dataclass(frozen=True)
class Quote:
    symbol: str
    last: Decimal
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    prev_close: Decimal | None
    vwap: Decimal | None
    volume: int
    buy_quantity: int
    sell_quantity: int
    bids: tuple[tuple[Decimal, int], ...]
    asks: tuple[tuple[Decimal, int], ...]
    lower_circuit: Decimal | None
    upper_circuit: Decimal | None
    last_trade: str | None

    def book(self) -> Book:
        return Book(self.last, self.bids, self.asks)

    def change_pct(self) -> float | None:
        return _pct(float(self.last), _number(self.prev_close))

    def facts(self, depth: bool = True) -> dict[str, Any]:
        last = float(self.last)
        bid = float(self.bids[0][0]) if self.bids else None
        ask = float(self.asks[0][0]) if self.asks else None
        out: dict[str, Any] = {
            "last": last,
            "change_pct": self.change_pct(),
            "open": _number(self.open),
            "high": _number(self.high),
            "low": _number(self.low),
            "prev_close": _number(self.prev_close),
            "gap_pct": _pct(_number(self.open), _number(self.prev_close)),
            "from_open_pct": _pct(last, _number(self.open)),
            "vwap": _number(self.vwap),
            "vs_vwap_pct": _pct(last, _number(self.vwap)),
            "volume": self.volume,
            "traded_value_cr": round(
                float(self.vwap or self.last) * self.volume / 1e7, 2
            ),
            "best_bid": bid,
            "best_ask": ask,
            "spread_pct": round((ask - bid) / last * 100, 3) if bid and ask else None,
            "lower_circuit": _number(self.lower_circuit),
            "upper_circuit": _number(self.upper_circuit),
            "last_trade": self.last_trade,
        }
        if depth:
            out["total_buy_qty"] = self.buy_quantity
            out["total_sell_qty"] = self.sell_quantity
            out["bids"] = [[float(p), q] for p, q in self.bids]
            out["asks"] = [[float(p), q] for p, q in self.asks]
        return out


def _levels(rows: Any) -> tuple[tuple[Decimal, int], ...]:
    levels = []
    for level in rows or []:
        price = _price(level.get("price"))
        quantity = int(level.get("quantity") or 0)
        if price is not None and quantity > 0:
            levels.append((price, quantity))
    return tuple(levels)


def parse_quote(symbol: str, row: dict[str, Any] | None) -> Quote | None:
    if not row:
        return None
    last = _price(row.get("last_price"))
    if last is None:
        return None
    ohlc = row.get("ohlc") or {}
    depth = row.get("depth") or {}
    stamp = row.get("last_trade_time")
    return Quote(
        symbol=symbol,
        last=last,
        open=_price(ohlc.get("open")),
        high=_price(ohlc.get("high")),
        low=_price(ohlc.get("low")),
        prev_close=_price(ohlc.get("close")),
        vwap=_price(row.get("average_price")),
        volume=int(row.get("volume") or 0),
        buy_quantity=int(row.get("buy_quantity") or 0),
        sell_quantity=int(row.get("sell_quantity") or 0),
        bids=_levels(depth.get("buy")),
        asks=_levels(depth.get("sell")),
        lower_circuit=_price(row.get("lower_circuit_limit")),
        upper_circuit=_price(row.get("upper_circuit_limit")),
        last_trade=stamp.strftime("%H:%M:%S")
        if isinstance(stamp, datetime)
        else (str(stamp)[11:19] if stamp else None),
    )


class NseDaily:
    """NSE's official daily data per stock from the evening task's feature files
    (bhavcopy, delivery, F&O), before ``today`` only. Prices are adjusted for
    corporate actions with NSE's own previous close (a split or bonus shows as a
    previous close different from the day before's close)."""

    def __init__(self, folder: Path, today: date) -> None:
        columns = ["date", "symbol", "series", *PRICE_COLUMNS, *STOCK_FACTS]
        first = (today - timedelta(days=430)).isoformat()
        frames = []
        for year in (today.year - 1, today.year):
            path = folder / f"stock_daily_{year}.csv.gz"
            if path.exists():
                part = pd.read_csv(path, usecols=columns, low_memory=False)
                frames.append(part[part["date"] >= first])
        if not frames:
            raise FileNotFoundError(f"no stock_daily files in {folder}")
        data = pd.concat(frames, ignore_index=True)
        data = data[data["date"] < today.isoformat()]
        data = data.assign(other_series=data["series"] != "EQ")
        data = data.sort_values(["symbol", "date", "other_series"]).drop_duplicates(
            ["symbol", "date"]
        )
        self.latest = str(data["date"].max())
        self._frames = {
            symbol: self._adjust(frame.reset_index(drop=True))
            for symbol, frame in data.groupby("symbol", sort=False)
        }
        self._market = self._last_rows(folder / "market_daily.csv.gz", today, None)
        self._indices = self._last_rows(folder / "index_daily.csv.gz", today, "symbol")

    @staticmethod
    def _adjust(frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["cm_close"].to_numpy(float)
        prev = frame["cm_prev_close"].to_numpy(float)
        factor = np.ones(len(frame))
        if len(frame) > 1:
            ratio = prev[1:] / close[:-1]
            ok = np.isfinite(ratio) & (ratio > 0) & (np.abs(ratio - 1) > 0.001)
            factor[1:] = np.where(ok, ratio, 1.0)
        # Multiplier for each day: the product of all later days' factors.
        later = np.append(factor[1:], 1.0)
        multiplier = np.cumprod(later[::-1])[::-1]
        out = frame.copy()
        for column in ("cm_prev_close", "cm_open", "cm_high", "cm_low", "cm_close"):
            out[column] = frame[column] * multiplier
        out["cm_vwap"] = frame["cm_vwap"] * multiplier
        out["cm_volume"] = frame["cm_volume"] / multiplier
        out.loc[0, "cm_prev_close"] = (
            frame.loc[0, "cm_prev_close"] * multiplier[0] * factor[0]
        )
        # Day t's previous close on the adjusted scale is day t-1's adjusted close.
        if len(frame) > 1:
            out.loc[1:, "cm_prev_close"] = out["cm_close"].to_numpy()[:-1]
        return out

    @staticmethod
    def _last_rows(path: Path, today: date, key: str | None) -> dict[str, Any]:
        if not path.exists():
            return {}
        data = pd.read_csv(path, low_memory=False)
        data = data[data["date"] < today.isoformat()].sort_values("date")
        if data.empty:
            return {}
        if key is None:
            return {"rows": data.tail(2).to_dict("records")}
        return {
            name: group.tail(2).to_dict("records") for name, group in data.groupby(key)
        }

    def symbols(self) -> set[str]:
        return set(self._frames)

    def has(self, symbol: str) -> bool:
        return symbol in self._frames

    def frame(self, symbol: str) -> pd.DataFrame | None:
        return self._frames.get(symbol)

    def stats(self, symbol: str) -> dict[str, Any] | None:
        frame = self._frames.get(symbol)
        if frame is None or frame.empty:
            return None
        close = frame["cm_close"].to_numpy(float)
        high = frame["cm_high"].to_numpy(float)
        low = frame["cm_low"].to_numpy(float)
        prev = frame["cm_prev_close"].to_numpy(float)
        volume = frame["cm_volume"].to_numpy(float)
        turnover = frame["cm_turnover_lakh"].to_numpy(float)
        delivery = frame["cm_deliv_pct"].to_numpy(float)

        def ret(days: int) -> float | None:
            if len(close) <= days:
                return None
            return _pct(close[-1], close[-1 - days])

        daily = close[1:] / close[:-1] - 1 if len(close) > 1 else np.array([])
        true_range = np.maximum(high, prev) - np.minimum(low, prev)
        year = slice(-250, None)
        return {
            "last_session": str(frame["date"].iloc[-1]),
            "close": _plain(close[-1]),
            "return_pct": {
                "1d": ret(1),
                "5d": ret(5),
                "20d": ret(20),
                "60d": ret(60),
                "250d": ret(250),
            },
            "volatility_20d_annual_pct": _plain(
                np.std(daily[-20:], ddof=1) * math.sqrt(252) * 100
            )
            if len(daily) >= 20
            else None,
            "atr14_pct": _plain(np.nanmean(true_range[-14:] / prev[-14:]) * 100),
            "high_52w": _plain(np.nanmax(high[year])),
            "low_52w": _plain(np.nanmin(low[year])),
            "avg_volume_20d": _plain(np.nanmean(volume[-20:])),
            "avg_traded_value_20d_cr": _plain(np.nanmean(turnover[-20:]) / 100),
            "delivery_pct_last": _plain(delivery[-1]),
            "delivery_pct_avg_20d": _plain(np.nanmean(delivery[-20:])),
        }

    def facts(self, symbol: str) -> dict[str, Any] | None:
        frame = self._frames.get(symbol)
        if frame is None or frame.empty:
            return None
        row = frame.iloc[-1]
        if str(row.get("fno")).strip().lower() not in ("true", "1", "1.0"):
            return {"date": str(row["date"]), "fno": False}
        return {"date": str(row["date"])} | {
            name: _plain(row.get(name)) for name in STOCK_FACTS
        }

    def bars(self, symbol: str, count: int) -> list[dict[str, Any]]:
        frame = self._frames.get(symbol)
        if frame is None:
            return []
        rows = frame.tail(count)
        return [
            {
                "date": str(r["date"]),
                "open": _plain(r["cm_open"]),
                "high": _plain(r["cm_high"]),
                "low": _plain(r["cm_low"]),
                "close": _plain(r["cm_close"]),
                "vwap": _plain(r["cm_vwap"]),
                "volume": int(v)
                if (v := _number(r["cm_volume"])) is not None
                else None,
                "delivery_pct": _plain(r["cm_deliv_pct"]),
            }
            for _, r in rows.iterrows()
        ]

    def market_facts(self) -> dict[str, Any]:
        rows = self._market.get("rows") or []
        if not rows:
            return {}
        last = rows[-1]
        out = {"date": last["date"]} | {k: _plain(last.get(k)) for k in MARKET_FACTS}
        if len(rows) == 2:
            out["oi_fii_idx_fut_long_pct_prev_day"] = _plain(
                rows[0].get("oi_fii_idx_fut_long_pct")
            )
        return out

    def index_facts(self) -> dict[str, Any]:
        out = {}
        for name in ("NIFTY", "BANKNIFTY"):
            rows = self._indices.get(name) or []
            if rows:
                last = rows[-1]
                out[name] = {"date": last["date"]} | {
                    k: _plain(last.get(k)) for k in INDEX_FACTS
                }
        return out


def session_phase(now: datetime) -> str:
    moment = now.time()
    if now.weekday() >= 5 or moment < clock_time(9, 0):
        return "closed (before the pre-open)"
    if moment < clock_time(9, 8):
        return (
            "pre-open order entry (09:00-09:08); the opening auction matches by 09:08"
        )
    if moment < clock_time(9, 15):
        return "pre-open matched: opening prices known; normal trading from 09:15"
    if moment < clock_time(15, 30):
        return "open"
    return "closed"


class Filings:
    """NSE company announcements of the last days, refreshed every few minutes."""

    def __init__(
        self,
        fetch: Callable[[date, date], list[Announcement]],
        clock: Callable[[], datetime],
        *,
        days: int = 4,
        refresh: timedelta = timedelta(minutes=5),
    ) -> None:
        self._fetch = fetch
        self._clock = clock
        self._days = days
        self._refresh = refresh
        self._items: list[Announcement] = []
        self._at: datetime | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def items(self) -> list[Announcement]:
        with self._lock:
            now = self._clock()
            if self._at is None or now - self._at >= self._refresh:
                try:
                    items = self._fetch(
                        now.date() - timedelta(days=self._days), now.date()
                    )
                    self._items = sorted(
                        items, key=lambda a: a.announced_at, reverse=True
                    )
                    self.error = None
                except Exception as error:
                    self.error = f"NSE filings unavailable ({type(error).__name__})"
                self._at = now
            return [a for a in self._items if a.announced_at <= now]


def filing_row(item: Announcement) -> dict[str, Any]:
    return {
        "time": item.announced_at.strftime("%Y-%m-%d %H:%M"),
        "symbol": item.symbol,
        "company": item.company,
        "subject": item.subject,
        "text": item.text[:400],
        "attachment": item.attachment,
    }


class MarketView:
    """What the agent can look up. Quotes are cached for ``quote_ttl`` seconds."""

    def __init__(
        self,
        kite: QuoteSource,
        scan: Sequence[str],
        profiles: dict[str, dict[str, str]],
        daily: NseDaily,
        clock: Callable[[], datetime],
        *,
        filings: Filings | None = None,
        quote_ttl: float = 10.0,
        monotonic: Callable[[], float] = time.monotonic,
        show_depth: bool = True,
    ) -> None:
        self._kite = kite
        self.scan = list(scan)
        self.profiles = profiles
        self.daily = daily
        self._clock = clock
        self.filings = filings
        self._ttl = quote_ttl
        self._monotonic = monotonic
        self.show_depth = show_depth
        self._quotes: dict[str, tuple[float, Quote]] = {}
        self._lock = threading.RLock()
        self._averages = {
            s: (self.daily.stats(s) or {}) for s in self.scan if self.daily.has(s)
        }

    def now(self) -> datetime:
        return self._clock()

    def quotes(
        self, symbols: Iterable[str], max_age: float | None = None
    ) -> dict[str, Quote]:
        ttl = self._ttl if max_age is None else max_age
        with self._lock:
            now = self._monotonic()
            found: dict[str, Quote] = {}
            missing = []
            for symbol in dict.fromkeys(symbols):
                cached = self._quotes.get(symbol)
                if cached and ttl > 0 and now - cached[0] <= ttl:
                    found[symbol] = cached[1]
                else:
                    missing.append(symbol)
            if missing:
                rows = self._kite.quote_rows([f"NSE:{s}" for s in missing])
                stamp = self._monotonic()
                for symbol in missing:
                    quote = parse_quote(symbol, rows.get(f"NSE:{symbol}"))
                    if quote is not None:
                        self._quotes[symbol] = (stamp, quote)
                        found[symbol] = quote
            return found

    def book(self, quote: Quote) -> Book:
        """The book an order in this stock fills against."""
        return quote.book()

    def books(self, symbols: Iterable[str], max_age: float = 0.0) -> dict[str, Book]:
        return {s: self.book(q) for s, q in self.quotes(symbols, max_age).items()}

    # --- the agent's tools ---------------------------------------------------------
    def overview(self) -> dict[str, Any]:
        with self._lock:
            rows = self._kite.quote_rows(list(INDICES.values()))
        indices = {}
        for name, key in INDICES.items():
            row = rows.get(key) or {}
            last = _number(row.get("last_price"))
            ohlc = row.get("ohlc") or {}
            if last is None:
                continue
            indices[name] = {
                "last": last,
                "change_pct": _pct(last, _number(ohlc.get("close"))),
                "open": _number(ohlc.get("open")),
                "high": _number(ohlc.get("high")),
                "low": _number(ohlc.get("low")),
                "prev_close": _number(ohlc.get("close")),
            }
        quotes = self.quotes(self.scan)
        changes = [c for q in quotes.values() if (c := q.change_pct()) is not None]
        above_vwap = [q.last > q.vwap for q in quotes.values() if q.vwap]
        by_industry: dict[str, list[float]] = {}
        for symbol, quote in quotes.items():
            change = quote.change_pct()
            industry = self.profiles.get(symbol, {}).get("industry") or ""
            if change is not None and industry:
                by_industry.setdefault(industry, []).append(change)
        industries = sorted(
            (
                {
                    "industry": k,
                    "stocks": len(v),
                    "median_change_pct": round(median(v), 2),
                }
                for k, v in by_industry.items()
                if len(v) >= 3
            ),
            key=lambda row: row["median_change_pct"],
        )
        now = self.now()
        return {
            "time": now.strftime("%H:%M:%S"),
            "session": session_phase(now),
            "indices": indices,
            "breadth_scan_list": {
                "stocks": len(quotes),
                "advancing": sum(c > 0 for c in changes),
                "declining": sum(c < 0 for c in changes),
                "median_change_pct": round(median(changes), 2) if changes else None,
                "above_vwap_pct": round(100 * sum(above_vwap) / len(above_vwap), 1)
                if above_vwap
                else None,
            },
            "weakest_industries": industries[:6],
            "strongest_industries": industries[::-1][:6],
            "previous_session_nse": {
                "market_positioning": self.daily.market_facts(),
                "index_derivatives": self.daily.index_facts(),
            },
        }

    def movers(self, sort: str, count: int = 15) -> dict[str, Any]:
        if sort not in MOVER_SORTS:
            raise ValueError(f"sort must be one of {MOVER_SORTS}")
        count = max(1, min(int(count), 40))
        rows = []
        for symbol, quote in self.quotes(self.scan).items():
            stats = self._averages.get(symbol) or {}
            avg_volume = stats.get("avg_volume_20d")
            last = float(quote.last)
            row = {
                "symbol": symbol,
                "industry": self.profiles.get(symbol, {}).get("industry"),
                "last": last,
                "change_pct": quote.change_pct(),
                "gap_pct": _pct(_number(quote.open), _number(quote.prev_close)),
                "from_open_pct": _pct(last, _number(quote.open)),
                "vs_vwap_pct": _pct(last, _number(quote.vwap)),
                "traded_value_cr": round(
                    float(quote.vwap or quote.last) * quote.volume / 1e7, 2
                ),
                "volume_vs_20d_avg": round(quote.volume / avg_volume, 2)
                if avg_volume
                else None,
                "vs_52w_high_pct": _pct(last, stats.get("high_52w")),
                "vs_52w_low_pct": _pct(last, stats.get("low_52w")),
            }
            rows.append(row)
        key, reverse = {
            "gainers": ("change_pct", True),
            "losers": ("change_pct", False),
            "gap_up": ("gap_pct", True),
            "gap_down": ("gap_pct", False),
            "up_from_open": ("from_open_pct", True),
            "down_from_open": ("from_open_pct", False),
            "traded_value": ("traded_value_cr", True),
            "volume_vs_20d_avg": ("volume_vs_20d_avg", True),
            "near_52w_high": ("vs_52w_high_pct", True),
            "near_52w_low": ("vs_52w_low_pct", False),
        }[sort]
        ranked = sorted(
            (r for r in rows if r[key] is not None),
            key=lambda r: r[key],
            reverse=reverse,
        )
        return {
            "time": self.now().strftime("%H:%M:%S"),
            "sort": sort,
            "scan_list": f"{len(rows)} largest NSE companies by market cap",
            "rows": ranked[:count],
        }

    def stock_details(self, symbols: Sequence[str]) -> list[dict[str, Any]]:
        wanted = [s.strip().upper() for s in symbols][:10]
        quotes = self.quotes(wanted)
        now = self.now()
        recent = []
        if self.filings is not None:
            since = now - timedelta(days=3)
            recent = [a for a in self.filings.items() if a.announced_at > since]
        out = []
        for symbol in wanted:
            quote = quotes.get(symbol)
            if quote is None and not self.daily.has(symbol):
                out.append({"symbol": symbol, "error": "unknown NSE symbol"})
                continue
            stats = self.daily.stats(symbol) or {}
            avg_volume = stats.get("avg_volume_20d")
            entry: dict[str, Any] = {
                "symbol": symbol,
                "company": self.profiles.get(symbol, {}).get("company"),
                "industry": self.profiles.get(symbol, {}).get("industry"),
                "live": quote.facts(self.show_depth) if quote else None,
                "volume_vs_20d_avg": round(quote.volume / avg_volume, 2)
                if quote and avg_volume
                else None,
                "daily_nse": stats or None,
                "previous_session_derivatives": self.daily.facts(symbol),
                "filings_last_3_days": [
                    filing_row(a) for a in recent if a.symbol == symbol
                ][:6],
            }
            out.append(entry)
        return out

    def bars(self, symbol: str, interval: str, count: int = 60) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        if interval not in BAR_INTERVALS:
            raise ValueError(f"interval must be one of {tuple(BAR_INTERVALS)}")
        count = max(1, min(int(count), 250))
        if interval == "day":
            rows = self.daily.bars(symbol, count)
            return {
                "symbol": symbol,
                "interval": "day",
                "note": "NSE official, adjusted for corporate actions; today's "
                "session is not included (see stock_details for live)",
                "bars": rows,
            }
        now = self.now()
        start = now - timedelta(days=5)
        with self._lock:
            candles = self._kite.get_historical_candles(
                Instrument("NSE", symbol), start, now, CandleInterval.ONE_MINUTE
            )
        size = BAR_INTERVALS[interval]
        bars: list[dict[str, Any]] = []
        for candle in candles:
            stamp = candle.timestamp.astimezone(INDIA_TIMEZONE)
            minute = stamp.hour * 60 + stamp.minute - (9 * 60 + 15)
            bucket = stamp.replace(
                hour=(9 * 60 + 15 + minute // size * size) // 60,
                minute=(9 * 60 + 15 + minute // size * size) % 60,
            )
            label = bucket.strftime("%Y-%m-%d %H:%M")
            volume = candle.volume or 0
            if bars and bars[-1]["time"] == label:
                bar = bars[-1]
                bar["high"] = max(bar["high"], float(candle.high))
                bar["low"] = min(bar["low"], float(candle.low))
                bar["close"] = float(candle.close)
                bar["volume"] += volume
            else:
                bars.append(
                    {
                        "time": label,
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "volume": volume,
                    }
                )
        return {"symbol": symbol, "interval": interval, "bars": bars[-count:]}

    def nse_filings(
        self, symbol: str | None = None, hours: float = 24, limit: int = 25
    ) -> dict[str, Any]:
        if self.filings is None:
            return {"error": "NSE filings are not available"}
        since = self.now() - timedelta(hours=max(0.25, min(float(hours), 96)))
        wanted = symbol.strip().upper() if symbol else None
        items = [
            filing_row(a)
            for a in self.filings.items()
            if a.announced_at > since and (wanted is None or a.symbol == wanted)
        ]
        out: dict[str, Any] = {
            "count": len(items),
            "filings": items[: max(1, min(int(limit), 60))],
        }
        if self.filings.error:
            out["warning"] = self.filings.error
        return out
