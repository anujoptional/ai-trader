"""Paper execution: deterministic sizing, risk checks, simulated fills and exits.

Entry: an AI LONG verdict becomes a market buy and SHORT a market short sale of
whole shares worth at most the per-trade amount, filled against the visible order
book of a fresh quote (asks for a buy, bids for a short sale; best price first,
walking deeper levels; any remainder at the deepest visible price). A target is
placed at once (``target``, 1% by default): above the entry for a long (rounded up
to the tick), below it for a short (rounded down). There is no stop.

Exit: completed one-minute bars after the entry minute are checked in order: a
minute opening at or beyond the target fills at that open; else a minute reaching
it fills at the target. A position whose target is not reached by 15:00 is closed
at the 15:00 market price: the close of the last minute before 15:00 (14:59), the
last trade before 15:00:00. Minutes from 15:00 on are never checked.

Limits (never relaxed by the AI): one open position per stock, at most one entry
per stock per session, a maximum number of open positions, no entry at or after
the exit time, and capital: the entry value of all open positions may not exceed
the starting capital plus realised P&L. A portfolio can be created without the
capital, position and per-stock limits (``capital`` / ``max_positions`` None,
``one_per_stock`` False); the exit-time rule always applies. Every trade closes
the session it opened, so it is charged at Zerodha's intraday rates.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime, time, timedelta
from decimal import ROUND_HALF_EVEN, Decimal

from ai_trader.broker import DepthSnapshot, OHLCVCandle
from ai_trader.costs import round_down_to_tick, round_up_to_tick
from ai_trader.paper.charges import trade_charges
from ai_trader.paper.store import SIDES, PaperStore, Position

PAISA = Decimal("0.01")
MINUTE = timedelta(minutes=1)
SETTLE = timedelta(
    seconds=15
)  # a minute bar is treated as final this long after it closes
# The 14:59 bar decides the 15:00 exit; without it this long after 15:00, the last
# earlier bar does (a minute without trades has no bar).
LAST_BAR_WAIT = timedelta(minutes=2)
# Settings added later are saved only when changed, so the saved settings of older
# stores still match.
_FIRST_SETTINGS = ("capital", "per_trade", "max_positions", "target", "exit_time")


@dataclass(frozen=True, repr=False)
class PaperConfig:
    capital: Decimal | None = Decimal(5_000_000)  # None: no capital limit
    per_trade: Decimal = Decimal(200_000)
    max_positions: int | None = 25  # None: no limit on open positions
    target: Decimal = Decimal("0.01")
    exit_time: time = time(15, 0)
    # False: a stock may be entered again, and held more than once, in a session.
    one_per_stock: bool = True

    def __post_init__(self) -> None:
        if (
            (self.capital is not None and self.capital <= 0)
            or self.per_trade <= 0
            or (self.max_positions is not None and self.max_positions < 1)
        ):
            raise ValueError(
                "Capital, per-trade amount and position limit must be > 0."
            )
        if not 0 < self.target < 1:
            raise ValueError("The target is a fraction between 0 and 1.")

    def __repr__(self) -> str:
        shown = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if f.name in _FIRST_SETTINGS or getattr(self, f.name) != f.default
        ]
        return f"PaperConfig({', '.join(shown)})"


@dataclass(frozen=True)
class EntryResult:
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
        if not config.one_per_stock:
            store.allow_many_per_symbol()

    # --- portfolio ----------------------------------------------------------
    @property
    def store(self) -> PaperStore:
        return self._store

    def open_positions(self) -> list[Position]:
        return self._store.positions("open")

    def held(self) -> set[str]:
        return {p.symbol for p in self.open_positions()}

    def entered_on(self, moment: datetime) -> set[str]:
        return self._store.entered_on(moment.date())

    def excluded(self, moment: datetime) -> set[str]:
        """Stocks not to ask the AI about: held or entered today (one per stock)."""
        if not self.config.one_per_stock:
            return set()
        return self.held() | self.entered_on(moment)

    def realised(self) -> Decimal:
        return sum(
            (p.net_pnl for p in self._store.positions("closed") if p.net_pnl),
            Decimal(0),
        )

    def deployed(self) -> Decimal:
        return sum((p.entry_value for p in self.open_positions()), Decimal(0))

    def available(self) -> Decimal | None:
        """Cash for new entries; None without a capital limit."""
        if self.config.capital is None:
            return None
        return self.config.capital + self.realised() - self.deployed()

    def deadline(self, position: Position) -> datetime:
        """When a position still open is closed at the market: 15:00 of its day."""
        return datetime.combine(
            position.entry_at.date(), self.config.exit_time, position.entry_at.tzinfo
        )

    # --- entries ------------------------------------------------------------
    def enter(
        self,
        symbol: str,
        side: str,
        depth: DepthSnapshot | None,
        tick: Decimal,
        now: datetime,
        *,
        cycle_id: str,
        probability: float,
        confidence: float,
        reason: str,
    ) -> EntryResult:
        if side not in SIDES:
            raise ValueError(f"Unknown side {side}")
        with self._lock:
            refusal = self._refusal(symbol, side, depth, now)
            if refusal:
                return self._refused(symbol, side, now, refusal, cycle_id)
            levels = depth.asks if side == "LONG" else depth.bids
            quantity = int(self.config.per_trade // levels[0].price)
            if quantity < 1:
                return self._refused(
                    symbol, side, now, "price_above_per_trade", cycle_id
                )
            remaining, value = quantity, Decimal(0)
            for level in levels:
                take = min(remaining, level.quantity)
                value += take * level.price
                remaining -= take
                if not remaining:
                    break
            thin_book = remaining > 0
            value += remaining * levels[-1].price
            available = self.available()
            if available is not None and value > available:
                return self._refused(symbol, side, now, "capital_limit", cycle_id)
            price = value / quantity
            target = (
                round_up_to_tick(price * (1 + self.config.target), tick)
                if side == "LONG"
                else round_down_to_tick(price * (1 - self.config.target), tick)
            )
            position = self._store.open_position(
                Position(
                    id=0,
                    symbol=symbol,
                    status="open",
                    quantity=quantity,
                    entry_at=now,
                    entry_price=price,
                    entry_value=value,
                    target_price=target,
                    stop_price=None,
                    tick=tick,
                    checked_through=now.replace(second=0, microsecond=0),
                    cycle_id=cycle_id,
                    scanner_probability=probability,
                    ai_confidence=confidence,
                    ai_reason=reason,
                    side=side,
                )
            )
            self._store.event(
                now,
                "entry",
                symbol,
                {
                    "position_id": position.id,
                    "side": side,
                    "quantity": quantity,
                    "price": str(_rupees(price)),
                    "value": str(_rupees(value)),
                    "best_price": str(levels[0].price),
                    "last_price": str(depth.last_price),
                    "thin_book": thin_book,
                    "target": str(position.target_price),
                    "exit_by": self.deadline(position).isoformat(),
                    "cycle_id": cycle_id,
                },
            )
            return EntryResult(position, "filled")

    def _refusal(
        self, symbol: str, side: str, depth: DepthSnapshot | None, now: datetime
    ) -> str | None:
        if self.config.one_per_stock:
            if symbol in self.held():
                return "already_held"
            if symbol in self.entered_on(now):
                return "already_traded_this_session"
        limit = self.config.max_positions
        if limit is not None and len(self.open_positions()) >= limit:
            return "position_limit"
        if now >= datetime.combine(now.date(), self.config.exit_time, now.tzinfo):
            return "after_exit_time"
        if depth is None:
            return "no_quote"
        if side == "LONG" and not depth.asks:
            return "no_sellers"
        if side == "SHORT" and not depth.bids:
            return "no_buyers"
        return None

    def _refused(
        self, symbol: str, side: str, now: datetime, reason: str, cycle_id: str
    ) -> EntryResult:
        self._store.event(
            now,
            "entry_refused",
            symbol,
            {"side": side, "reason": reason, "cycle_id": cycle_id},
        )
        return EntryResult(None, reason)

    # --- exits --------------------------------------------------------------
    def apply_bars(
        self, position: Position, bars: Sequence[OHLCVCandle], now: datetime
    ) -> Position | None:
        """Check completed minutes after ``checked_through`` for the target, and
        close at 15:00 if it was not reached; the closed position, else None.
        Bars must be the stock's own minutes."""
        with self._lock:
            current = self._store.position(position.id)
            if current.status != "open":
                return None
            deadline = self.deadline(current)
            last_checked = current.checked_through
            last_bar: OHLCVCandle | None = None  # the latest final bar before 15:00
            for bar in sorted(bars, key=lambda b: b.timestamp):
                if bar.timestamp + MINUTE + SETTLE > now or bar.timestamp >= deadline:
                    continue
                if bar.timestamp > last_checked:
                    exit_price, exit_reason = self._touch(current, bar)
                    if exit_price is not None:
                        return self._close(
                            current, bar.timestamp, exit_price, exit_reason
                        )
                    last_checked = bar.timestamp
                last_bar = bar
            if (
                last_bar is not None
                and now >= deadline + SETTLE
                and (
                    last_bar.timestamp >= deadline - MINUTE
                    or now >= deadline + LAST_BAR_WAIT
                )
            ):
                return self._close(current, deadline, last_bar.close, "time_1500")
            if last_checked > current.checked_through:
                self._store.checked(current.id, last_checked)
            return None

    def close_at(
        self, position: Position, at: datetime, price: Decimal, reason: str
    ) -> Position | None:
        """Close at a given price (the 15:00 exit when no minute bar was found)."""
        with self._lock:
            current = self._store.position(position.id)
            if current.status != "open":
                return None
            return self._close(current, at, price, reason)

    @staticmethod
    def _touch(position: Position, bar: OHLCVCandle) -> tuple[Decimal | None, str]:
        target = position.target_price
        if position.side == "LONG":
            if bar.open >= target:
                return bar.open, "tp_gap"
            if bar.high >= target:
                return target, "tp"
        else:
            if bar.open <= target:
                return bar.open, "tp_gap"
            if bar.low <= target:
                return target, "tp"
        return None, ""

    def _close(
        self, position: Position, at: datetime, price: Decimal, reason: str
    ) -> Position:
        exit_value = price * position.quantity
        if position.side == "LONG":
            bought, sold = position.entry_value, exit_value
        else:
            bought, sold = exit_value, position.entry_value
        same_session = at.date() == position.entry_at.date()
        charges = trade_charges(bought, sold, same_session)
        gross = sold - bought
        closed = self._store.close_position(
            position.id,
            exit_at=at,
            exit_price=price,
            exit_reason=reason,
            exit_value=exit_value,
            charges=_rupees(charges.total),
            charges_detail=charges.as_dict(),
            gross_pnl=_rupees(gross),
            net_pnl=_rupees(gross - charges.total),
        )
        self._store.event(
            at,
            "exit",
            position.symbol,
            {
                "position_id": position.id,
                "side": position.side,
                "price": str(price),
                "reason": reason,
                "charges": str(_rupees(charges.total)),
                "charges_kind": charges.kind,
                "net_pnl": str(closed.net_pnl),
            },
        )
        return closed
