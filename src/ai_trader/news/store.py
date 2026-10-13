"""SQLite store for NSE filings with as-of reads (no look-ahead by construction)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path

from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.news.nse import (
    Announcement,
    BoardMeeting,
    CorporateAction,
    parse_nse_time,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS announcements (
    key TEXT PRIMARY KEY, symbol TEXT NOT NULL, company TEXT NOT NULL,
    subject TEXT NOT NULL, text TEXT NOT NULL, attachment TEXT NOT NULL,
    disseminated_at TEXT NOT NULL, raw TEXT NOT NULL, fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS announcements_by_symbol
    ON announcements (symbol, disseminated_at);
CREATE TABLE IF NOT EXISTS corporate_actions (
    key TEXT PRIMARY KEY, symbol TEXT NOT NULL, series TEXT NOT NULL,
    subject TEXT NOT NULL, ex_date TEXT, record_date TEXT,
    raw TEXT NOT NULL, fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS actions_by_symbol ON corporate_actions (symbol, ex_date);
CREATE TABLE IF NOT EXISTS board_meetings (
    key TEXT PRIMARY KEY, symbol TEXT NOT NULL, meeting_date TEXT,
    purpose TEXT NOT NULL, description TEXT NOT NULL, announced_at TEXT,
    raw TEXT NOT NULL, fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS meetings_by_symbol ON board_meetings (symbol, meeting_date);
CREATE TABLE IF NOT EXISTS fetch_log (
    kind TEXT NOT NULL, day TEXT NOT NULL, rows INTEGER NOT NULL,
    fetched_at TEXT NOT NULL, PRIMARY KEY (kind, day)
);
"""


def _iso(stamp: datetime) -> str:
    return stamp.astimezone(INDIA_TIMEZONE).isoformat(timespec="seconds")


def disseminated_at(item: Announcement) -> datetime:
    """The later of NSE's announcement and dissemination times: when the market knew."""
    stamps = [item.announced_at]
    other = parse_nse_time(item.raw.get("exchdisstime"))
    if other is not None:
        stamps.append(other)
    return max(stamps)


class NewsStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def add_announcements(self, items: Iterable[Announcement]) -> int:
        now = self._now()
        rows = [
            (
                a.key,
                a.symbol,
                a.company,
                a.subject,
                a.text,
                a.attachment,
                _iso(disseminated_at(a)),
                json.dumps(a.raw, sort_keys=True, default=str),
                now,
            )
            for a in items
        ]
        with self._db:
            before = self._db.total_changes
            self._db.executemany(
                "INSERT OR IGNORE INTO announcements VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
            return self._db.total_changes - before

    def add_actions(self, items: Iterable[CorporateAction]) -> int:
        now = self._now()
        rows = [
            (
                a.key,
                a.symbol,
                a.series,
                a.subject,
                a.ex_date.isoformat() if a.ex_date else None,
                a.record_date.isoformat() if a.record_date else None,
                json.dumps(a.raw, sort_keys=True, default=str),
                now,
            )
            for a in items
        ]
        with self._db:
            before = self._db.total_changes
            self._db.executemany(
                "INSERT OR IGNORE INTO corporate_actions VALUES (?,?,?,?,?,?,?,?)", rows
            )
            return self._db.total_changes - before

    def add_meetings(self, items: Iterable[BoardMeeting]) -> int:
        now = self._now()
        rows = [
            (
                m.key,
                m.symbol,
                m.meeting_date.isoformat() if m.meeting_date else None,
                m.purpose,
                m.description,
                _iso(m.announced_at) if m.announced_at else None,
                json.dumps(m.raw, sort_keys=True, default=str),
                now,
            )
            for m in items
        ]
        with self._db:
            before = self._db.total_changes
            self._db.executemany(
                "INSERT OR IGNORE INTO board_meetings VALUES (?,?,?,?,?,?,?,?)", rows
            )
            return self._db.total_changes - before

    def mark_fetched(self, kind: str, day: date, rows: int) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?)",
                (kind, day.isoformat(), rows, self._now()),
            )

    def fetched_days(self, kind: str) -> set[date]:
        return {
            date.fromisoformat(row[0])
            for row in self._db.execute(
                "SELECT day FROM fetch_log WHERE kind = ?", (kind,)
            )
        }

    def announcements(
        self, symbol: str, after: datetime, until: datetime
    ) -> list[sqlite3.Row]:
        """Announcements disseminated in (after, until], newest first."""
        return self._db.execute(
            "SELECT * FROM announcements WHERE symbol = ? "
            "AND disseminated_at > ? AND disseminated_at <= ? "
            "ORDER BY disseminated_at DESC",
            (symbol, _iso(after), _iso(until)),
        ).fetchall()

    def actions(self, symbol: str, start: date, end: date) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM corporate_actions WHERE symbol = ? "
            "AND ex_date >= ? AND ex_date <= ? ORDER BY ex_date",
            (symbol, start.isoformat(), end.isoformat()),
        ).fetchall()

    def meetings(
        self, symbol: str, start: date, end: date, known_by: datetime
    ) -> list[sqlite3.Row]:
        """Board meetings scheduled in [start, end] whose intimation was public."""
        return self._db.execute(
            "SELECT * FROM board_meetings WHERE symbol = ? "
            "AND meeting_date >= ? AND meeting_date <= ? "
            "AND announced_at IS NOT NULL AND announced_at <= ? ORDER BY meeting_date",
            (symbol, start.isoformat(), end.isoformat(), _iso(known_by)),
        ).fetchall()
