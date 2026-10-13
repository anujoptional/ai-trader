"""Read-only NSE corporate-filings client: announcements, actions, board meetings.

NSE serves these JSON endpoints only to a browser-like session that first loads a
filings page for cookies. Requests are paced, retried a bounded number of times and
re-primed on 401/403. Records are normalized but keep NSE's original fields.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from ai_trader.clock import INDIA_TIMEZONE

BASE_URL = "https://www.nseindia.com"
PRIME_PATH = "/companies-listing/corporate-filings-announcements"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL + PRIME_PATH,
}
NSE_TIME_FORMATS = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y")


class NseError(RuntimeError):
    """An NSE request failed after bounded retries; no response body is kept."""


@dataclass(frozen=True, slots=True)
class Announcement:
    key: str
    symbol: str
    company: str
    subject: str
    text: str
    attachment: str
    announced_at: datetime
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CorporateAction:
    key: str
    symbol: str
    series: str
    subject: str
    ex_date: date | None
    record_date: date | None
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BoardMeeting:
    key: str
    symbol: str
    meeting_date: date | None
    purpose: str
    description: str
    announced_at: datetime | None
    raw: Mapping[str, Any]


def parse_nse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip() or value.strip() == "-":
        return None
    text = value.strip()
    for pattern in NSE_TIME_FORMATS:
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=INDIA_TIMEZONE)
        except ValueError:
            continue
    return None


def parse_nse_date(value: object) -> date | None:
    stamp = parse_nse_time(value)
    return stamp.date() if stamp else None


def _key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


def announcement(row: Mapping[str, Any]) -> Announcement | None:
    stamp = parse_nse_time(row.get("an_dt")) or parse_nse_time(row.get("exchdisstime"))
    symbol = str(row.get("symbol") or "").strip()
    if stamp is None or not symbol:
        return None
    subject = str(row.get("desc") or "").strip()
    attachment = str(row.get("attchmntFile") or "").strip()
    return Announcement(
        key=_key(symbol, stamp.isoformat(), subject, attachment),
        symbol=symbol,
        company=str(row.get("sm_name") or "").strip(),
        subject=subject,
        text=str(row.get("attchmntText") or "").strip(),
        attachment=attachment,
        announced_at=stamp,
        raw=dict(row),
    )


def corporate_action(row: Mapping[str, Any]) -> CorporateAction | None:
    symbol = str(row.get("symbol") or "").strip()
    subject = str(row.get("subject") or "").strip()
    if not symbol or not subject:
        return None
    ex_date = parse_nse_date(row.get("exDate"))
    return CorporateAction(
        key=_key(symbol, subject, ex_date, row.get("recDate")),
        symbol=symbol,
        series=str(row.get("series") or "").strip(),
        subject=subject,
        ex_date=ex_date,
        record_date=parse_nse_date(row.get("recDate")),
        raw=dict(row),
    )


def board_meeting(row: Mapping[str, Any]) -> BoardMeeting | None:
    symbol = str(row.get("bm_symbol") or "").strip()
    if not symbol:
        return None
    meeting = parse_nse_date(row.get("bm_date"))
    purpose = str(row.get("bm_purpose") or "").strip()
    return BoardMeeting(
        key=_key(symbol, meeting, purpose, row.get("bm_timestamp")),
        symbol=symbol,
        meeting_date=meeting,
        purpose=purpose,
        description=str(row.get("bm_desc") or "").strip(),
        announced_at=parse_nse_time(row.get("bm_timestamp")),
        raw=dict(row),
    )


class NseClient:
    """Paced, cookie-primed JSON client; one instance per process."""

    def __init__(
        self,
        session: Any | None = None,
        *,
        min_interval_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if session is None:
            import requests

            session = requests.Session()
        session.headers.update(HEADERS)
        self._session = session
        self._interval = min_interval_seconds
        self._timeout = timeout_seconds
        self._attempts = attempts
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self._primed = False

    def _pace(self) -> None:
        if self._last is not None:
            delay = self._interval - (self._clock() - self._last)
            if delay > 0:
                self._sleep(delay)
        self._last = self._clock()

    def _prime(self) -> None:
        self._pace()
        response = self._session.get(BASE_URL + PRIME_PATH, timeout=self._timeout)
        if response.status_code != 200:
            raise NseError(
                f"NSE session priming failed with HTTP {response.status_code}"
            )
        self._primed = True

    def get_json(self, path: str, params: Mapping[str, str]) -> Any:
        failure = "no attempt"
        for attempt in range(self._attempts):
            try:
                if not self._primed:
                    self._prime()
                self._pace()
                response = self._session.get(
                    BASE_URL + path, params=dict(params), timeout=self._timeout
                )
                if response.status_code in (401, 403):
                    self._primed = False
                    failure = f"HTTP {response.status_code}"
                elif response.status_code == 200:
                    return response.json()
                else:
                    failure = f"HTTP {response.status_code}"
            except NseError as error:
                failure = str(error)
            except Exception as error:
                failure = type(error).__name__
            self._sleep(min(30.0, 2.0 * 2**attempt))
        raise NseError(f"NSE request {path} failed: {failure}")

    @staticmethod
    def _range(start: date, end: date, symbol: str | None) -> dict[str, str]:
        if start > end:
            raise ValueError("NSE date range must be ordered")
        params = {
            "index": "equities",
            "from_date": start.strftime("%d-%m-%Y"),
            "to_date": end.strftime("%d-%m-%Y"),
        }
        if symbol:
            params["symbol"] = symbol
        return params

    @staticmethod
    def _rows(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        if not isinstance(payload, list):
            raise NseError("NSE response was not a list of records")
        return [row for row in payload if isinstance(row, Mapping)]

    def announcements(
        self, start: date, end: date, symbol: str | None = None
    ) -> list[Announcement]:
        rows = self._rows(
            self.get_json(
                "/api/corporate-announcements", self._range(start, end, symbol)
            )
        )
        return [item for item in map(announcement, rows) if item is not None]

    def corporate_actions(
        self, start: date, end: date, symbol: str | None = None
    ) -> list[CorporateAction]:
        rows = self._rows(
            self.get_json(
                "/api/corporates-corporateActions", self._range(start, end, symbol)
            )
        )
        return [item for item in map(corporate_action, rows) if item is not None]

    def board_meetings(
        self, start: date, end: date, symbol: str | None = None
    ) -> list[BoardMeeting]:
        rows = self._rows(
            self.get_json(
                "/api/corporate-board-meetings", self._range(start, end, symbol)
            )
        )
        return [item for item in map(board_meeting, rows) if item is not None]
