"""Paper execution: deterministic sizing, risk checks, simulated fills and exits.

Buy: an AI BUY verdict becomes a market buy of whole shares worth at most the
per-trade amount, filled against the visible asks of the live order book (best ask
first, walking deeper levels; any remainder at the deepest visible ask). A sell
limit at +1% (rounded up to the tick) and a stop at -1% (rounded down) are placed
at once; there is no time limit.

Exit: completed one-minute bars after the buy minute are checked in order, with
the research rules: a minute opening at or beyond a level fills at that open
(overnight gaps included); a minute reaching both levels counts as the stop; else
the target fills at its limit price and the stop at its trigger price.

Limits (never relaxed by the AI): one open position per stock, at most one buy per
stock per session, a maximum number of open positions, and capital: the cost of all
open positions may not exceed the starting capital plus realised P&L.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

from ai_trader.broker import DepthSnapshot, OHLCVCandle
from ai_trader.costs import round_down_to_tick, round_up_to_tick
from ai_trader.paper.charges import trade_charges
from ai_trader.paper.store import PaperStore, Position

PAISA = Decimal("0.01")
MINUTE = timedelta(minutes=1)
SETTLE = timedelta(
    seconds=15
)  # a minute bar is treated as final this long after it closes


@dataclass(frozen=True)
class PaperConfig:
    capital: Decimal = Decimal(5_000_000)
    per_trade: Decimal = Decimal(200_000)
    max_positions: int = 25
    target: Decimal = Decimal("0.01")
    stop: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        if self.capital <= 0 or self.per_trade <= 0 or self.max_positions < 1:
            raise ValueError(
                "Capital, per-trade amount and position limit must be > 0."
            )
        if not (0 < self.target < 1 and 0 < self.stop < 1):
            raise ValueError("Target and stop are fractions between 0 and 1.")


@dataclass(frozen=True)
class BuyResult:
    position: Position | None
    reason: str  # "filled" or why it was refused


def _rupees(value: Decimal) -> Decimal:
    return value.quantize(PAISA, rounding=ROUND_HALF_EVEN)


class PaperBroker:
    def __init__(self, store: PaperStore, config: PaperConfig) -> None:
        self._store = store
        self.config = config
        self._lock = threading.RLock()
        recorded = store.setting("config")
        current = repr(config)
        if recorded is None:
            store.set_setting("config", current)
        elif recorded != current:
            raise ValueError(
                "This paper portfolio was created with different settings "
                f"({recorded}); use a new store path to change them."
            )

    # --- portfolio ----------------------------------------------------------
    @property
    def store(self) -> PaperStore:
        return self._store

    def open_positions(self) -> list[Position]:
        return self._store.positions("open")

    def held(self) -> set[str]:
        return {p.symbol for p in self.open_positions()}

    def bought_on(self, moment: datetime) -> set[str]:
        return self._store.entered_on(moment.date())

    def realised(self) -> Decimal:
        return sum(
            (p.net_pnl for p in self._store.positions("closed") if p.net_pnl),
            Decimal(0),
        )

    def deployed(self) -> Decimal:
        return sum((p.buy_value for p in self.open_positions()), Decimal(0))

    def available(self) -> Decimal:
        return self.config.capital + self.realised() - self.deployed()

    # --- entries ------------------------------------------------------------
    def buy(
        self,
        symbol: str,
        depth: DepthSnapshot | None,
        tick: Decimal,
        now: datetime,
        *,
        cycle_id: str,
        probability: float,
        confidence: float,
        reason: str,
    ) -> BuyResult:
        with self._lock:
            refusal = self._refusal(symbol, depth, now)
            if refusal:
                return self._refused(symbol, now, refusal, cycle_id)
            asks = depth.asks
            quantity = int(self.config.per_trade // asks[0].price)
            if quantity < 1:
                return self._refused(symbol, now, "price_above_per_trade", cycle_id)
            remaining, cost = quantity, Decimal(0)
            for level in asks:
                take = min(remaining, level.quantity)
                cost += take * level.price
                remaining -= take
                if not remaining:
                    break
            thin_book = remaining > 0
            cost += remaining * asks[-1].price
            if cost > self.available():
                return self._refused(symbol, now, "capital_limit", cycle_id)
            price = cost / quantity
            position = self._store.open_position(
                Position(
                    id=0,
                    symbol=symbol,
                    status="open",
                    quantity=quantity,
                    entry_at=now,
                    entry_price=price,
                    buy_value=cost,
                    target_price=round_up_to_tick(
                        price * (1 + self.config.target), tick
                    ),
                    stop_price=round_down_to_tick(price * (1 - self.config.stop), tick),
                    tick=tick,
                    checked_through=now.replace(second=0, microsecond=0),
                    cycle_id=cycle_id,
                    scanner_probability=probability,
                    ai_confidence=confidence,
                    ai_reason=reason,
                )
            )
            self._store.event(
                now,
                "buy",
                symbol,
                {
                    "position_id": position.id,
                    "quantity": quantity,
                    "price": str(_rupees(price)),
                    "value": str(_rupees(cost)),
                    "best_ask": str(asks[0].price),
                    "last_price": str(depth.last_price),
                    "thin_book": thin_book,
                    "target": str(position.target_price),
                    "stop": str(position.stop_price),
                    "cycle_id": cycle_id,
                },
            )
            return BuyResult(position, "filled")

    def _refusal(
        self, symbol: str, depth: DepthSnapshot | None, now: datetime
    ) -> str | None:
        if symbol in self.held():
            return "already_held"
        if symbol in self.bought_on(now):
            return "already_bought_this_session"
        if len(self.open_positions()) >= self.config.max_positions:
            return "position_limit"
        if depth is None:
            return "no_quote"
        if not depth.asks:
            return "no_sellers"
        return None

    def _refused(self, symbol: str, now: datetime, reason: str, cycle_id: str):
        self._store.event(
            now, "buy_refused", symbol, {"reason": reason, "cycle_id": cycle_id}
        )
        return BuyResult(None, reason)

    # --- exits --------------------------------------------------------------
    def apply_bars(
        self, position: Position, bars: Sequence[OHLCVCandle], now: datetime
    ) -> Position | None:
        """Check completed minutes after ``checked_through``; the closed position if
        a level was reached, else None. Bars must be the stock's own minutes."""
        with self._lock:
            current = self._store.position(position.id)
            if current.status != "open":
                return None
            last_checked = current.checked_through
            for bar in sorted(bars, key=lambda b: b.timestamp):
                if (
                    bar.timestamp <= last_checked
                    or bar.timestamp + MINUTE + SETTLE > now
                ):
                    continue
                exit_price, exit_reason = self._touch(current, bar)
                if exit_price is not None:
                    return self._close(current, bar.timestamp, exit_price, exit_reason)
                last_checked = bar.timestamp
            if last_checked > current.checked_through:
                self._store.checked(current.id, last_checked)
            return None

    @staticmethod
    def _touch(position: Position, bar: OHLCVCandle) -> tuple[Decimal | None, str]:
        target, stop = position.target_price, position.stop_price
        if bar.open <= stop:
            return bar.open, "sl_gap"
        if bar.open >= target:
            return bar.open, "tp_gap"
        if bar.low <= stop:
            return stop, "sl_same_minute" if bar.high >= target else "sl"
        if bar.high >= target:
            return target, "tp"
        return None, ""

    def _close(
        self, position: Position, at: datetime, price: Decimal, reason: str
    ) -> Position:
        sell_value = price * position.quantity
        same_session = at.date() == position.entry_at.date()
        charges = trade_charges(position.buy_value, sell_value, same_session)
        gross = sell_value - position.buy_value
        closed = self._store.close_position(
            position.id,
            exit_at=at,
            exit_price=price,
            exit_reason=reason,
            sell_value=sell_value,
            charges=_rupees(charges.total),
            charges_detail=charges.as_dict(),
            gross_pnl=_rupees(gross),
            net_pnl=_rupees(gross - charges.total),
        )
        self._store.event(
            at,
            "sell",
            position.symbol,
            {
                "position_id": position.id,
                "price": str(price),
                "reason": reason,
                "charges": str(_rupees(charges.total)),
                "charges_kind": charges.kind,
                "net_pnl": str(closed.net_pnl),
            },
        )
        return closed
