"""Historical replay of the read-only market-data calls a paper session makes.

Serves the local Kite candle files as if live at a simulated clock: minute bars
only once complete, daily bars only for earlier sessions, an order book whose ask
is the open of the minute in progress (the price research assumed for a buy) with
unlimited size, and futures OI from the downloaded daily files. For offline
end-to-end validation; it never touches the network.
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.broker import (
    CandleInterval,
    DepthLevel,
    DepthSnapshot,
    FuturesDay,
    Instrument,
    LastTradedPrice,
    OHLCVCandle,
)
from ai_trader.clock import INDIA_TIMEZONE

MINUTE = 60


def nse_tick(price: Decimal) -> Decimal:
    """NSE price-band tick sizes (CMTR67133, from April 2025)."""
    for limit, tick in (
        (250, "0.01"),
        (1000, "0.05"),
        (5000, "0.10"),
        (10000, "0.50"),
        (20000, "1.00"),
    ):
        if price <= limit:
            return Decimal(tick)
    return Decimal("5.00")


def _decimal(value: float) -> Decimal:
    return Decimal(repr(float(value)))


@dataclass(frozen=True)
class ReplayInstrument:
    instrument: Instrument
    tick_size: Decimal


@dataclass(frozen=True)
class _Series:
    seconds: np.ndarray  # bar start, epoch seconds
    values: np.ndarray  # [n, 5] open/high/low/close/volume (NaN = unknown volume)

    def candles(self, first: int, last: int) -> list[OHLCVCandle]:
        return [
            OHLCVCandle(
                datetime.fromtimestamp(int(self.seconds[i]), INDIA_TIMEZONE),
                _decimal(self.values[i, 0]),
                _decimal(self.values[i, 1]),
                _decimal(self.values[i, 2]),
                _decimal(self.values[i, 3]),
                None if np.isnan(self.values[i, 4]) else int(self.values[i, 4]),
            )
            for i in range(first, last)
        ]


def _epoch(moment: datetime) -> int:
    return int(moment.timestamp())


class ReplayLiveBars:
    """The live tick stream's bars in replay: the same minute files, as visible."""

    def __init__(self, data: ReplayData, day: date) -> None:
        self.data = data
        self.opening = _epoch(datetime.combine(day, time(9, 15), INDIA_TIMEZONE))

    def bars(self, symbol: str, upto: int) -> np.ndarray:
        series = self.data._series(symbol, CandleInterval.ONE_MINUTE)
        last = self.data._visible_end(series, CandleInterval.ONE_MINUTE)
        index = (series.seconds[:last] - self.opening) // MINUTE
        keep = (index >= 0) & (index <= upto)
        rows = np.full((upto + 1, 5), np.nan)
        rows[index[keep]] = series.values[:last][keep]
        return rows

    def complete(self, symbol: str, upto: int) -> bool:
        return bool(np.isfinite(self.bars(symbol, upto)[:, 3]).all())

    def silent_seconds(self) -> float:
        return 0.0

    def seed(self, bars: Any, upto: int) -> None:
        pass


class ReplayData:
    def __init__(
        self,
        candles: Path,
        clock: Any,
        *,
        oi: Path | None = None,
        first_day: date,
    ) -> None:
        self._candles = candles
        self._oi = oi
        self._clock = clock
        self._first = first_day.isoformat()
        self._cache: dict[tuple[str, CandleInterval], _Series] = {}
        self._lock = threading.Lock()

    def fork_history(self) -> ReplayData:
        return self

    # --- files ------------------------------------------------------------------
    @staticmethod
    def _read(path: Path, since: str) -> _Series:
        seconds, values = [], []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader)
                for stamp, open_, high, low, close, volume in reader:
                    if stamp[:10] < since:
                        continue
                    seconds.append(_epoch(datetime.fromisoformat(stamp)))
                    values.append(
                        (
                            float(open_),
                            float(high),
                            float(low),
                            float(close),
                            float(volume) if volume else np.nan,
                        )
                    )
        return _Series(
            np.asarray(seconds, dtype=np.int64),
            np.asarray(values, dtype=float).reshape(-1, 5),
        )

    def _series(self, symbol: str, interval: CandleInterval) -> _Series:
        key = (symbol, interval)
        with self._lock:
            if key not in self._cache:
                minute = interval is CandleInterval.ONE_MINUTE
                folder = self._candles if minute else self._candles / "daily"
                self._cache[key] = self._read(
                    folder / f"NSE_{symbol}.csv", self._first if minute else ""
                )
            return self._cache[key]

    def _visible_end(self, series: _Series, interval: CandleInterval) -> int:
        now = self._clock.now()
        if interval is CandleInterval.ONE_MINUTE:
            cutoff = _epoch(now) - MINUTE  # complete bars only
            return int(np.searchsorted(series.seconds, cutoff, side="right"))
        midnight = datetime.combine(now.date(), datetime.min.time(), INDIA_TIMEZONE)
        return int(np.searchsorted(series.seconds, _epoch(midnight), side="left"))

    # --- read-only broker surface ---------------------------------------------
    def resolve_instrument(self, instrument: Instrument) -> ReplayInstrument:
        series = self._series(instrument.trading_symbol, CandleInterval.ONE_DAY)
        end = self._visible_end(series, CandleInterval.ONE_DAY)
        if end == 0:
            raise KeyError(instrument.trading_symbol)
        return ReplayInstrument(
            instrument, nse_tick(_decimal(series.values[end - 1, 3]))
        )

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]:
        series = self._series(instrument.trading_symbol, interval)
        first = int(np.searchsorted(series.seconds, _epoch(start), side="left"))
        last = min(
            int(np.searchsorted(series.seconds, _epoch(end), side="right")),
            self._visible_end(series, interval),
        )
        return tuple(series.candles(first, last)) if last > first else ()

    def _current(self, symbol: str) -> tuple[Decimal, Decimal, datetime] | None:
        """(price a buy pays now, last traded price, last trade time)."""
        now = self._clock.now()
        series = self._series(symbol, CandleInterval.ONE_MINUTE)
        done = self._visible_end(series, CandleInterval.ONE_MINUTE)
        if done == 0:
            return None
        last_bar = datetime.fromtimestamp(int(series.seconds[done - 1]), INDIA_TIMEZONE)
        if last_bar.date() != now.date():
            return None
        last = _decimal(series.values[done - 1, 3])
        live = done < len(series.seconds) and series.seconds[done] <= _epoch(now)
        price = _decimal(series.values[done, 0]) if live else last
        return price, last, last_bar + timedelta(minutes=1)

    def get_depth(self, instruments):
        snapshots, missing = [], []
        now = self._clock.now()
        for instrument in instruments:
            current = (
                self._current(instrument.trading_symbol)
                if instrument.exchange == "NSE"
                else None
            )
            if current is None:
                missing.append(instrument)
                continue
            price, last, traded = current
            daily = self._series(instrument.trading_symbol, CandleInterval.ONE_DAY)
            previous = daily.values[
                self._visible_end(daily, CandleInterval.ONE_DAY) - 1
            ]
            tick = nse_tick(price)
            snapshots.append(
                DepthSnapshot(
                    instrument=instrument,
                    timestamp=now,
                    last_price=last,
                    last_trade_at=traded,
                    volume=0,
                    total_buy_quantity=10**9,
                    total_sell_quantity=10**9,
                    bids=(DepthLevel(price - tick, 10**9, 1),),
                    asks=(DepthLevel(price, 10**9, 1),),
                    previous_close=_decimal(previous[3]),
                )
            )
        return tuple(snapshots), tuple(missing)

    def get_ltp(self, instruments) -> tuple[LastTradedPrice, ...]:
        prices = []
        for instrument in instruments:
            current = self._current(instrument.trading_symbol)
            if current is not None:
                prices.append(LastTradedPrice(instrument, current[1]))
        return tuple(prices)

    def nfo_futures(self) -> list[dict[str, Any]]:
        if self._oi is None:
            return []
        expiry = self._clock.now().date() + timedelta(days=30)
        return [
            {
                "name": path.stem,
                "tradingsymbol": f"{path.stem}FUT",
                "expiry": expiry,
                "segment": "NFO-FUT",
                "instrument_type": "FUT",
            }
            for path in sorted(self._oi.glob("*.csv"))
        ]

    def get_continuous_futures_daily(
        self, underlying: str, start: date, end: date
    ) -> tuple[FuturesDay, ...]:
        bars = []
        with (self._oi / f"{underlying}.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                day = date.fromisoformat(row["session_date"])
                if start <= day <= end and day < self._clock.now().date():
                    bars.append(
                        FuturesDay(
                            day,
                            Decimal(row["open"]),
                            Decimal(row["high"]),
                            Decimal(row["low"]),
                            Decimal(row["close"]),
                            int(row["volume"]),
                            int(row["open_interest"]),
                        )
                    )
        return tuple(bars)

    def get_futures_daily(
        self, contract: str, start: date, end: date
    ) -> tuple[FuturesDay, ...]:
        """The one replayed contract per stock (see ``nfo_futures``): its
        continuous near-month series."""
        return self.get_continuous_futures_daily(
            contract.removesuffix("FUT"), start, end
        )
