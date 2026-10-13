"""SQLite recorder for depth snapshots, plus the polling loop and as-of reads."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from ai_trader.broker import DepthLevel, DepthSnapshot, Instrument
from ai_trader.clock import INDIA_TIMEZONE

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    captured_at TEXT NOT NULL, exchange TEXT NOT NULL, symbol TEXT NOT NULL,
    quote_time TEXT NOT NULL, last_price TEXT NOT NULL, last_trade_at TEXT,
    volume INTEGER NOT NULL, buy_quantity INTEGER NOT NULL,
    sell_quantity INTEGER NOT NULL, open_interest INTEGER,
    previous_close TEXT NOT NULL, lower_circuit TEXT, upper_circuit TEXT,
    bids TEXT NOT NULL, asks TEXT NOT NULL,
    PRIMARY KEY (exchange, symbol, captured_at)
);
"""


class DepthSource(Protocol):
    def get_depth(
        self, instruments: Sequence[Instrument]
    ) -> tuple[tuple[DepthSnapshot, ...], tuple[Instrument, ...]]: ...


def _levels(levels: tuple[DepthLevel, ...]) -> str:
    return json.dumps([[str(lv.price), lv.quantity, lv.orders] for lv in levels])


def _parse_levels(text: str) -> tuple[DepthLevel, ...]:
    return tuple(DepthLevel(Decimal(p), q, o) for p, q, o in json.loads(text))


def _optional(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


class DepthRecorder:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def write(self, captured_at: datetime, snapshots: Sequence[DepthSnapshot]) -> int:
        stamp = captured_at.astimezone(INDIA_TIMEZONE).isoformat(timespec="seconds")
        rows = [
            (
                stamp,
                s.instrument.exchange,
                s.instrument.trading_symbol,
                s.timestamp.isoformat(timespec="seconds"),
                str(s.last_price),
                s.last_trade_at.isoformat(timespec="seconds")
                if s.last_trade_at
                else None,
                s.volume,
                s.total_buy_quantity,
                s.total_sell_quantity,
                s.open_interest,
                str(s.previous_close),
                _optional(s.lower_circuit),
                _optional(s.upper_circuit),
                _levels(s.bids),
                _levels(s.asks),
            )
            for s in snapshots
        ]
        with self._db:
            self._db.executemany(
                f"INSERT OR IGNORE INTO snapshots VALUES ({','.join('?' * 15)})",
                rows,
            )
        return len(rows)

    def latest(self, instrument: Instrument, as_of: datetime) -> DepthSnapshot | None:
        """The newest snapshot captured at or before ``as_of`` (no look-ahead)."""
        row = self._db.execute(
            "SELECT * FROM snapshots WHERE exchange = ? AND symbol = ? "
            "AND captured_at <= ? ORDER BY captured_at DESC LIMIT 1",
            (
                instrument.exchange,
                instrument.trading_symbol,
                as_of.astimezone(INDIA_TIMEZONE).isoformat(timespec="seconds"),
            ),
        ).fetchone()
        if row is None:
            return None
        return DepthSnapshot(
            instrument=instrument,
            timestamp=datetime.fromisoformat(row[3]),
            last_price=Decimal(row[4]),
            last_trade_at=datetime.fromisoformat(row[5]) if row[5] else None,
            volume=row[6],
            total_buy_quantity=row[7],
            total_sell_quantity=row[8],
            open_interest=row[9],
            previous_close=Decimal(row[10]),
            lower_circuit=Decimal(row[11]) if row[11] else None,
            upper_circuit=Decimal(row[12]) if row[12] else None,
            bids=_parse_levels(row[13]),
            asks=_parse_levels(row[14]),
        )


def record(
    source: DepthSource,
    instruments: Sequence[Instrument],
    recorder: DepthRecorder,
    *,
    interval_seconds: float,
    until: datetime,
    now: Callable[[], datetime] = lambda: datetime.now(INDIA_TIMEZONE),
    sleep: Callable[[float], None] = time.sleep,
    on_error: Callable[[Exception], None] = lambda error: None,
) -> dict[str, int]:
    """Poll depth for all instruments every ``interval_seconds`` until ``until``."""
    batches = [instruments[i : i + 500] for i in range(0, len(instruments), 500)]
    stats = {"cycles": 0, "snapshots": 0, "missing": 0, "errors": 0}
    while (started := now()) < until:
        for batch in batches:
            try:
                snapshots, missing = source.get_depth(batch)
            except Exception as error:  # a failed poll must not end the session
                stats["errors"] += 1
                on_error(error)
                continue
            stats["snapshots"] += recorder.write(started, snapshots)
            stats["missing"] += len(missing)
        stats["cycles"] += 1
        remaining = interval_seconds - (now() - started).total_seconds()
        if remaining > 0:
            sleep(remaining)
    return stats
