"""Today's minute bars built from Kite's live tick stream.

The paper session uses these to pre-screen the whole universe in about a second
at each decision; the shortlisted stocks are then scored again on Kite's
historical bars, the inputs the scanner was trained and validated on. Each bar
takes the prices of the ticks whose exchange time falls in its minute, and its
volume from the rise of the exchange's cumulative volume since the previous tick.
Minutes taken from historical bars (``seed``) are never overwritten by ticks.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from datetime import date
from time import monotonic

import numpy as np

from ai_trader.broker import MarketTick
from ai_trader.scanner.upside import SESSION_MINUTES, minute_index


class LiveBars:
    def __init__(
        self,
        day: date,
        symbols: Iterable[str],
        *,
        from_open: bool = True,
        length: int = SESSION_MINUTES,
    ) -> None:
        self.day = day
        self.length = length
        # Started after the open, a stock's first tick only sets the volume baseline.
        self.from_open = from_open
        names = list(symbols)
        self.symbols = tuple(names)
        self._prices = {s: np.full((length, 4), np.nan) for s in names}
        self._volume = {s: np.zeros(length) for s in names}
        self._last_cumulative: dict[str, int] = {}
        self._seeded = dict.fromkeys(names, -1)
        self._lock = threading.Lock()
        self.ticks = 0
        self._last_tick: float | None = None

    def on_tick(self, tick: MarketTick) -> None:
        if tick.timestamp.date() != self.day:
            return
        index = minute_index(tick.timestamp)
        symbol = tick.instrument.trading_symbol
        price = float(tick.price)
        with self._lock:
            self.ticks += 1
            self._last_tick = monotonic()
            prices = self._prices.get(symbol)
            if prices is None or not 0 <= index < self.length:
                return
            fresh = index > self._seeded[symbol]
            cumulative = tick.cumulative_volume
            if cumulative is not None:
                last = self._last_cumulative.get(symbol)
                self._last_cumulative[symbol] = (
                    cumulative if last is None else max(last, cumulative)
                )
                if fresh and (last is not None or self.from_open):
                    self._volume[symbol][index] += max(cumulative - (last or 0), 0)
            if not fresh:
                return
            row = prices[index]
            if np.isnan(row[0]):
                row[:] = price
            else:
                row[1], row[2], row[3] = max(row[1], price), min(row[2], price), price

    def seed(self, bars: Mapping[str, np.ndarray], upto: int) -> None:
        """Take minutes 0..upto from historical bars (final, so they win)."""
        with self._lock:
            for symbol, rows in bars.items():
                prices = self._prices.get(symbol)
                if prices is None or rows is None:
                    continue
                part = rows[: upto + 1]
                prices[: upto + 1] = part[:, :4]
                self._volume[symbol][: upto + 1] = np.where(
                    np.isfinite(part[:, 4]), part[:, 4], 0.0
                )
                self._seeded[symbol] = max(self._seeded[symbol], upto)

    def bars(self, symbol: str, upto: int) -> np.ndarray:
        """Rows 0..upto as (open, high, low, close, volume); NaN where no trade."""
        with self._lock:
            rows = np.full((upto + 1, 5), np.nan)
            rows[:, :4] = self._prices[symbol][: upto + 1]
            volume = self._volume[symbol][: upto + 1]
        rows[:, 4] = np.where(np.isfinite(rows[:, 3]), volume, np.nan)
        return rows

    def complete(self, symbol: str, upto: int) -> bool:
        """Every minute 0..upto has a price (an index that ticks every second)."""
        with self._lock:
            return bool(np.isfinite(self._prices[symbol][: upto + 1, 3]).all())

    def silent_seconds(self) -> float | None:
        with self._lock:
            return None if self._last_tick is None else monotonic() - self._last_tick
