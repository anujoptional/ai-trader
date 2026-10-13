"""SQLite portfolio and trade store for paper trading.

One row per position: open while held, closed with its exit, charges and P&L.
A partial unique index allows at most one open position per symbol, so the
portfolio can never hold a stock twice. Prices and rupees are stored as exact
decimal text. ``events`` is the audit trail of every scan, verdict, fill and skip.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import wraps
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    entry_at TEXT NOT NULL,
    entry_price TEXT NOT NULL,
    buy_value TEXT NOT NULL,
    target_price TEXT NOT NULL,
    stop_price TEXT NOT NULL,
    tick TEXT NOT NULL,
    checked_through TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    scanner_probability REAL NOT NULL,
    ai_confidence REAL NOT NULL,
    ai_reason TEXT NOT NULL,
    exit_at TEXT,
    exit_price TEXT,
    exit_reason TEXT,
    sell_value TEXT,
    charges TEXT,
    charges_detail TEXT,
    gross_pnl TEXT,
    net_pnl TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_position_per_symbol
    ON positions(symbol) WHERE status = 'open';
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT,
    detail TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True, slots=True)
class Position:
    id: int
    symbol: str
    status: str
    quantity: int
    entry_at: datetime
    entry_price: Decimal
    buy_value: Decimal
    target_price: Decimal
    stop_price: Decimal
    tick: Decimal
    checked_through: datetime
    cycle_id: str
    scanner_probability: float
    ai_confidence: float
    ai_reason: str
    exit_at: datetime | None = None
    exit_price: Decimal | None = None
    exit_reason: str | None = None
    sell_value: Decimal | None = None
    charges: Decimal | None = None
    charges_detail: dict[str, str] | None = None
    gross_pnl: Decimal | None = None
    net_pnl: Decimal | None = None


def _decimal(value: str | None) -> Decimal | None:
    return None if value is None else Decimal(value)


def _moment(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _position(row: sqlite3.Row) -> Position:
    return Position(
        id=row["id"],
        symbol=row["symbol"],
        status=row["status"],
        quantity=row["quantity"],
        entry_at=datetime.fromisoformat(row["entry_at"]),
        entry_price=Decimal(row["entry_price"]),
        buy_value=Decimal(row["buy_value"]),
        target_price=Decimal(row["target_price"]),
        stop_price=Decimal(row["stop_price"]),
        tick=Decimal(row["tick"]),
        checked_through=datetime.fromisoformat(row["checked_through"]),
        cycle_id=row["cycle_id"],
        scanner_probability=row["scanner_probability"],
        ai_confidence=row["ai_confidence"],
        ai_reason=row["ai_reason"],
        exit_at=_moment(row["exit_at"]),
        exit_price=_decimal(row["exit_price"]),
        exit_reason=row["exit_reason"],
        sell_value=_decimal(row["sell_value"]),
        charges=_decimal(row["charges"]),
        charges_detail=json.loads(row["charges_detail"])
        if row["charges_detail"]
        else None,
        gross_pnl=_decimal(row["gross_pnl"]),
        net_pnl=_decimal(row["net_pnl"]),
    )


def _locked[F: Callable[..., Any]](method: F) -> F:
    @wraps(method)
    def guarded(self: PaperStore, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return guarded  # type: ignore[return-value]


class PaperStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    @_locked
    def close(self) -> None:
        self._db.close()

    # --- settings ---------------------------------------------------------
    @_locked
    def setting(self, key: str) -> str | None:
        row = self._db.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row["value"]

    @_locked
    def set_setting(self, key: str, value: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value)
            )

    # --- positions --------------------------------------------------------
    @_locked
    def open_position(self, position: Position) -> Position:
        with self._db:
            cursor = self._db.execute(
                "INSERT INTO positions (symbol, status, quantity, entry_at, "
                "entry_price, buy_value, target_price, stop_price, tick, "
                "checked_through, cycle_id, scanner_probability, ai_confidence, "
                "ai_reason) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    position.symbol,
                    position.quantity,
                    position.entry_at.isoformat(),
                    str(position.entry_price),
                    str(position.buy_value),
                    str(position.target_price),
                    str(position.stop_price),
                    str(position.tick),
                    position.checked_through.isoformat(),
                    position.cycle_id,
                    position.scanner_probability,
                    position.ai_confidence,
                    position.ai_reason,
                ),
            )
        return self.position(int(cursor.lastrowid))

    @_locked
    def position(self, position_id: int) -> Position:
        row = self._db.execute(
            "SELECT * FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
        if row is None:
            raise KeyError(position_id)
        return _position(row)

    @_locked
    def checked(self, position_id: int, through: datetime) -> None:
        with self._db:
            self._db.execute(
                "UPDATE positions SET checked_through = ? WHERE id = ? "
                "AND status = 'open'",
                (through.isoformat(), position_id),
            )

    @_locked
    def close_position(
        self,
        position_id: int,
        *,
        exit_at: datetime,
        exit_price: Decimal,
        exit_reason: str,
        sell_value: Decimal,
        charges: Decimal,
        charges_detail: dict[str, str],
        gross_pnl: Decimal,
        net_pnl: Decimal,
    ) -> Position:
        with self._db:
            updated = self._db.execute(
                "UPDATE positions SET status = 'closed', exit_at = ?, exit_price = ?, "
                "exit_reason = ?, sell_value = ?, charges = ?, charges_detail = ?, "
                "gross_pnl = ?, net_pnl = ?, checked_through = ? "
                "WHERE id = ? AND status = 'open'",
                (
                    exit_at.isoformat(),
                    str(exit_price),
                    exit_reason,
                    str(sell_value),
                    str(charges),
                    json.dumps(charges_detail, sort_keys=True),
                    str(gross_pnl),
                    str(net_pnl),
                    exit_at.isoformat(),
                    position_id,
                ),
            ).rowcount
        if updated != 1:
            raise ValueError(f"Position {position_id} is not open.")
        return self.position(position_id)

    @_locked
    def positions(self, status: str | None = None) -> list[Position]:
        query = "SELECT * FROM positions"
        args: tuple = ()
        if status:
            query += " WHERE status = ?"
            args = (status,)
        return [_position(r) for r in self._db.execute(query + " ORDER BY id", args)]

    @_locked
    def entered_on(self, day: date) -> set[str]:
        return {
            row["symbol"]
            for row in self._db.execute(
                "SELECT symbol FROM positions WHERE substr(entry_at, 1, 10) = ?",
                (day.isoformat(),),
            )
        }

    # --- audit trail ------------------------------------------------------
    @_locked
    def event(
        self, at: datetime, kind: str, symbol: str | None, detail: dict[str, Any]
    ) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO events (at, kind, symbol, detail, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    at.isoformat(),
                    kind,
                    symbol,
                    json.dumps(detail, sort_keys=True, default=str),
                    datetime.now(UTC).isoformat(),
                ),
            )

    @_locked
    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT at, kind, symbol, detail FROM events"
        args: tuple = ()
        if kind:
            query += " WHERE kind = ?"
            args = (kind,)
        return [
            {
                "at": row["at"],
                "kind": row["kind"],
                "symbol": row["symbol"],
                "detail": json.loads(row["detail"]),
            }
            for row in self._db.execute(query + " ORDER BY id", args)
        ]
