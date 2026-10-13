"""SQLite journal for decision cycles, verdicts and later outcomes."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

from ai_trader.ai.decision import PROMPT_VERSION, CycleResult, Verdict, request_hash
from ai_trader.ai.openai_client import Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    cycle_id TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    decider TEXT NOT NULL,
    model TEXT,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER, cached_tokens INTEGER, cache_write_tokens INTEGER,
    output_tokens INTEGER, reasoning_tokens INTEGER, web_searches INTEGER,
    cost_usd REAL NOT NULL,
    latency_seconds REAL NOT NULL,
    request_sha256 TEXT NOT NULL,
    request_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    response_json TEXT
);
CREATE TABLE IF NOT EXISTS verdicts (
    cycle_id TEXT NOT NULL REFERENCES cycles(cycle_id),
    symbol TEXT NOT NULL,
    verdict TEXT NOT NULL
        CHECK (verdict IN ('LONG', 'SHORT', 'NO_TRADE', 'BUY', 'NO_BUY')),
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    source TEXT NOT NULL,
    outcome_return REAL,
    PRIMARY KEY (cycle_id, symbol)
);
"""


class DecisionJournal:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(cycles)")}
        if "response_json" not in columns:  # journals written before October 12
            self._db.execute("ALTER TABLE cycles ADD COLUMN response_json TEXT")

    def record(
        self, cycle_id: str, as_of: datetime, decider: str, result: CycleResult
    ) -> None:
        usage = result.usage
        counts = (
            (
                usage.input_tokens,
                usage.cached_tokens,
                usage.cache_write_tokens,
                usage.output_tokens,
                usage.reasoning_tokens,
                usage.web_searches,
            )
            if usage
            else (None,) * 6
        )
        with self._db:
            self._db.execute("DELETE FROM verdicts WHERE cycle_id = ?", (cycle_id,))
            self._db.execute(
                f"INSERT OR REPLACE INTO cycles VALUES ({','.join('?' * 18)})",
                (
                    cycle_id,
                    as_of.isoformat(),
                    decider,
                    result.model,
                    result.prompt_version,
                    result.status,
                    *counts,
                    result.cost_usd,
                    result.latency_seconds,
                    request_hash(result.request_json),
                    result.request_json,
                    datetime.now(UTC).isoformat(),
                    result.response_json or None,
                ),
            )
            self._db.executemany(
                "INSERT INTO verdicts VALUES (?,?,?,?,?,?,NULL)",
                [
                    (cycle_id, v.symbol, v.verdict, v.confidence, v.reason, v.source)
                    for v in result.verdicts
                ],
            )

    def completed(
        self, cycle_id: str, request_json: str, prompt_version: str = PROMPT_VERSION
    ) -> CycleResult | None:
        """A successful earlier result for exactly this input and prompt, else None."""
        row = self._db.execute(
            "SELECT model, status, input_tokens, cached_tokens, cache_write_tokens, "
            "output_tokens, reasoning_tokens, web_searches, cost_usd, "
            "latency_seconds, request_sha256, prompt_version, response_json "
            "FROM cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        if (
            row is None
            or row[1] != "ok"
            or row[10] != request_hash(request_json)
            or row[11] != prompt_version
        ):
            return None
        verdicts = tuple(
            Verdict(*v)
            for v in self._db.execute(
                "SELECT symbol, verdict, confidence, reason, source FROM verdicts "
                "WHERE cycle_id = ? ORDER BY rowid",
                (cycle_id,),
            )
        )
        usage = Usage(*(int(x or 0) for x in row[2:8]))
        return CycleResult(
            verdicts,
            "ok",
            row[0],
            usage,
            float(row[8]),
            float(row[9]),
            request_json,
            prompt_version,
            row[12] or "",
        )

    def set_outcome(self, cycle_id: str, symbol: str, outcome_return: float) -> None:
        with self._db:
            self._db.execute(
                "UPDATE verdicts SET outcome_return = ? "
                "WHERE cycle_id = ? AND symbol = ?",
                (outcome_return, cycle_id, symbol),
            )

    def spent_usd(self, day: date) -> float:
        row = self._db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM cycles "
            "WHERE substr(recorded_at, 1, 10) = ?",
            (day.isoformat(),),
        ).fetchone()
        return float(row[0])

    def close(self) -> None:
        self._db.close()
