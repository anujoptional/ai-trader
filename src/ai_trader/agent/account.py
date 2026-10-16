"""Paper wallet of the autonomous agent: whole shares, fills at the order book, MIS
leverage up to 5x on the wallet, the agent's own stop and target per position
checked by code, Zerodha intraday charges, and everything closed at the session's
end. Paper only: no order ever reaches a broker."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_trader.paper.charges import trade_charges

SIDES = ("LONG", "SHORT")
SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL,
    entry_at TEXT NOT NULL, entry_price TEXT NOT NULL, entry_value TEXT NOT NULL,
    stop TEXT, target TEXT, reason TEXT NOT NULL, status TEXT NOT NULL,
    exit_at TEXT, exit_price TEXT, exit_value TEXT, exit_reason TEXT,
    charges TEXT, net TEXT
);
CREATE TABLE IF NOT EXISTS pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL,
    stop TEXT, target TEXT, reason TEXT NOT NULL, placed_at TEXT NOT NULL,
    status TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Book:
    """Order book levels (price, quantity), best first, and the last price."""

    last: Decimal
    bids: tuple[tuple[Decimal, int], ...]
    asks: tuple[tuple[Decimal, int], ...]


def walk(levels: tuple[tuple[Decimal, int], ...], quantity: int) -> Decimal | None:
    """Average price of ``quantity`` taken from the levels; what the book shows
    beyond its last level is filled at that level's price. None: no levels."""
    if not levels:
        return None
    remaining, value = quantity, Decimal(0)
    for price, size in levels:
        take = min(remaining, size)
        value += take * price
        remaining -= take
        if not remaining:
            break
    value += remaining * levels[-1][0]
    return value / quantity


class PaperAccount:
    def __init__(self, path: Path, wallet: Decimal, leverage: int = 5) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self.wallet = wallet
        self.leverage = leverage

    # --- state ----------------------------------------------------------------
    def open_positions(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM positions WHERE status = 'open' ORDER BY id"
            ).fetchall()

    def closed_positions(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM positions WHERE status = 'closed' ORDER BY id"
            ).fetchall()

    def pending_orders(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM pending WHERE status = 'pending' ORDER BY id"
            ).fetchall()

    def realised(self) -> Decimal:
        return sum((Decimal(r["net"]) for r in self.closed_positions()), Decimal(0))

    def equity(self) -> Decimal:
        return self.wallet + self.realised()

    def exposure(self) -> Decimal:
        return sum(
            (Decimal(r["entry_value"]) for r in self.open_positions()), Decimal(0)
        )

    def buying_power(self) -> Decimal:
        return self.equity() * self.leverage - self.exposure()

    def summary(self, prices: dict[str, Decimal]) -> dict[str, Any]:
        """The account as the agent sees it, open positions at ``prices``."""
        held = []
        unrealised = Decimal(0)
        for r in self.open_positions():
            entry = Decimal(r["entry_price"])
            last = prices.get(r["symbol"])
            sign = 1 if r["side"] == "LONG" else -1
            pnl = sign * (last - entry) * r["quantity"] if last is not None else None
            unrealised += pnl or 0
            held.append(
                {
                    "symbol": r["symbol"],
                    "side": r["side"],
                    "quantity": r["quantity"],
                    "entered": r["entry_at"][11:16],
                    "entry_price": float(entry),
                    "value_rs": round(float(r["entry_value"]), 2),
                    "last_price": float(last) if last is not None else None,
                    "pnl_rs_before_charges": round(float(pnl), 2)
                    if pnl is not None
                    else None,
                    "stop": float(r["stop"]) if r["stop"] else None,
                    "target": float(r["target"]) if r["target"] else None,
                }
            )
        closed = self.closed_positions()
        return {
            "wallet_rs": float(self.wallet),
            "equity_rs": round(float(self.equity()), 2),
            "in_positions_rs": round(float(self.exposure()), 2),
            "buying_power_rs": round(float(self.buying_power()), 2),
            "max_leverage": self.leverage,
            "open": held,
            "unrealised_rs_before_charges": round(float(unrealised), 2),
            "pending_opening_orders": [
                {"symbol": p["symbol"], "side": p["side"], "quantity": p["quantity"]}
                for p in self.pending_orders()
            ],
            "closed_today": [
                {
                    "symbol": r["symbol"],
                    "side": r["side"],
                    "quantity": r["quantity"],
                    "entered": r["entry_at"][11:16],
                    "exited": r["exit_at"][11:16],
                    "entry_price": float(r["entry_price"]),
                    "exit_price": float(r["exit_price"]),
                    "exit_reason": r["exit_reason"],
                    "charges_rs": round(float(r["charges"]), 2),
                    "net_rs": round(float(r["net"]), 2),
                }
                for r in closed
            ],
            "realised_net_rs": round(float(self.realised()), 2),
            "charges_today_rs": round(sum(float(r["charges"]) for r in closed), 2),
        }

    # --- orders -----------------------------------------------------------------
    def _check_exits(
        self, side: str, price: Decimal, stop: Decimal | None, target: Decimal | None
    ) -> str | None:
        sign = 1 if side == "LONG" else -1
        if stop is not None and sign * (stop - price) >= 0:
            return "the stop must be below the price for a long, above it for a short"
        if target is not None and sign * (target - price) <= 0:
            return "the target must be above the price for a long, below it for a short"
        return None

    def open(
        self,
        symbol: str,
        side: str,
        quantity: int,
        book: Book,
        now: datetime,
        *,
        stop: Decimal | None = None,
        target: Decimal | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        if side not in SIDES:
            return {"ok": False, "error": f"side must be one of {SIDES}"}
        if not isinstance(quantity, int) or quantity < 1:
            return {"ok": False, "error": "quantity must be a whole number of shares"}
        with self._lock:
            price = walk(book.asks if side == "LONG" else book.bids, quantity)
            if price is None:
                return {"ok": False, "error": "no one on the other side of the book"}
            value = price * quantity
            if value > self.buying_power():
                return {
                    "ok": False,
                    "error": f"needs Rs {value:,.0f}; buying power is "
                    f"Rs {self.buying_power():,.0f} (wallet x {self.leverage} "
                    "less what is in positions)",
                }
            problem = self._check_exits(side, price, stop, target)
            if problem:
                return {"ok": False, "error": problem}
            cursor = self._db.execute(
                "INSERT INTO positions (symbol, side, quantity, entry_at, entry_price,"
                " entry_value, stop, target, reason, status) "
                "VALUES (?,?,?,?,?,?,?,?,?, 'open')",
                (
                    symbol,
                    side,
                    quantity,
                    now.isoformat(timespec="seconds"),
                    str(price),
                    str(value),
                    str(stop) if stop is not None else None,
                    str(target) if target is not None else None,
                    reason[:300],
                ),
            )
            self._db.commit()
            return {
                "ok": True,
                "position_id": cursor.lastrowid,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "fill_price": round(float(price), 4),
                "value_rs": round(float(value), 2),
                "buying_power_left_rs": round(float(self.buying_power()), 2),
            }

    def queue_opening(
        self,
        symbol: str,
        side: str,
        quantity: int,
        now: datetime,
        *,
        stop: Decimal | None = None,
        target: Decimal | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """A pre-open order: filled at the session's official open (the auction)."""
        if side not in SIDES:
            return {"ok": False, "error": f"side must be one of {SIDES}"}
        if not isinstance(quantity, int) or quantity < 1:
            return {"ok": False, "error": "quantity must be a whole number of shares"}
        with self._lock:
            self._db.execute(
                "INSERT INTO pending (symbol, side, quantity, stop, target, reason, "
                "placed_at, status) VALUES (?,?,?,?,?,?,?, 'pending')",
                (
                    symbol,
                    side,
                    quantity,
                    str(stop) if stop is not None else None,
                    str(target) if target is not None else None,
                    reason[:300],
                    now.isoformat(timespec="seconds"),
                ),
            )
            self._db.commit()
        return {
            "ok": True,
            "queued": f"{side} {quantity} {symbol} at the 09:15 opening price",
        }

    def cancel_pending(self, symbol: str) -> int:
        """Cancel the pending opening orders in ``symbol``; how many."""
        with self._lock:
            cursor = self._db.execute(
                "UPDATE pending SET status = 'cancelled' "
                "WHERE symbol = ? AND status = 'pending'",
                (symbol,),
            )
            self._db.commit()
            return cursor.rowcount

    def fill_opening(
        self, opens: dict[str, Decimal], now: datetime
    ) -> list[dict[str, Any]]:
        """Fill every pending pre-open order at its stock's opening price, whole
        quantity (the auction is a single price), within buying power."""
        done = []
        with self._lock:
            for order in self.pending_orders():
                price = opens.get(order["symbol"])
                status, result = "rejected", None
                if price is None:
                    result = {"ok": False, "error": "no opening price"}
                else:
                    level = ((price, order["quantity"]),)
                    result = self.open(
                        order["symbol"],
                        order["side"],
                        order["quantity"],
                        Book(price, level, level),
                        now,
                        stop=Decimal(order["stop"]) if order["stop"] else None,
                        target=Decimal(order["target"]) if order["target"] else None,
                        reason=order["reason"],
                    )
                    status = "filled" if result.get("ok") else "rejected"
                self._db.execute(
                    "UPDATE pending SET status = ? WHERE id = ?", (status, order["id"])
                )
                done.append({"symbol": order["symbol"], **result})
            self._db.commit()
        return done

    def set_exits(
        self, symbol: str, stop: Decimal | None, target: Decimal | None
    ) -> dict[str, Any]:
        """Replace the stop and target of every open position in ``symbol``
        (None removes one); a level already passed acts at the next check."""
        with self._lock:
            rows = [r for r in self.open_positions() if r["symbol"] == symbol]
            if not rows:
                return {"ok": False, "error": f"no open position in {symbol}"}
            for r in rows:
                self._db.execute(
                    "UPDATE positions SET stop = ?, target = ? WHERE id = ?",
                    (
                        str(stop) if stop is not None else None,
                        str(target) if target is not None else None,
                        r["id"],
                    ),
                )
            self._db.commit()
        return {
            "ok": True,
            "positions": len(rows),
            "stop": float(stop) if stop is not None else None,
            "target": float(target) if target is not None else None,
        }

    def close(
        self,
        symbol: str,
        book: Book,
        now: datetime,
        reason: str,
        *,
        price: Decimal | None = None,
    ) -> list[dict[str, Any]]:
        """Close every open position in ``symbol``: at ``price`` when given (a target
        limit), else by walking the book (a market order)."""
        with self._lock:
            rows = [r for r in self.open_positions() if r["symbol"] == symbol]
            return self._close_rows(rows, book, now, reason, price)

    def _close_rows(
        self,
        rows: list[sqlite3.Row],
        book: Book,
        now: datetime,
        reason: str,
        price: Decimal | None,
    ) -> list[dict[str, Any]]:
        results = []
        with self._lock:
            for r in rows:
                quantity = r["quantity"]
                long = r["side"] == "LONG"
                fill = price or walk(book.bids if long else book.asks, quantity)
                if fill is None:
                    fill = book.last
                exit_value = fill * quantity
                entry_value = Decimal(r["entry_value"])
                buy, sell = (
                    (entry_value, exit_value) if long else (exit_value, entry_value)
                )
                charges = trade_charges(buy, sell, same_session=True).total
                net = sell - buy - charges
                self._db.execute(
                    "UPDATE positions SET status = 'closed', exit_at = ?, "
                    "exit_price = ?, exit_value = ?, exit_reason = ?, charges = ?, "
                    "net = ? WHERE id = ? AND status = 'open'",
                    (
                        now.isoformat(timespec="seconds"),
                        str(fill),
                        str(exit_value),
                        reason,
                        str(charges),
                        str(net),
                        r["id"],
                    ),
                )
                results.append(
                    {
                        "symbol": r["symbol"],
                        "side": r["side"],
                        "quantity": quantity,
                        "exit_price": round(float(fill), 4),
                        "reason": reason,
                        "charges_rs": round(float(charges), 2),
                        "net_rs": round(float(net), 2),
                    }
                )
            self._db.commit()
        return results

    def check_exits(
        self, books: dict[str, Book], now: datetime
    ) -> list[dict[str, Any]]:
        """Close positions whose stop or target the last price has reached: a
        target at its price (a resting limit), a stop by walking the book."""
        closed = []
        with self._lock:
            for r in self.open_positions():
                book = books.get(r["symbol"])
                if book is None:
                    continue
                sign = 1 if r["side"] == "LONG" else -1
                stop = Decimal(r["stop"]) if r["stop"] else None
                target = Decimal(r["target"]) if r["target"] else None
                if target is not None and sign * (book.last - target) >= 0:
                    closed += self._close_rows([r], book, now, "target", target)
                elif stop is not None and sign * (book.last - stop) <= 0:
                    closed += self._close_rows([r], book, now, "stop", None)
        return closed

    def close_all(
        self, books: dict[str, Book], now: datetime, reason: str
    ) -> list[dict[str, Any]]:
        closed = []
        with self._lock:
            for symbol in sorted({r["symbol"] for r in self.open_positions()}):
                book = books.get(symbol)
                if book is not None:
                    closed += self.close(symbol, book, now, reason)
        return closed
