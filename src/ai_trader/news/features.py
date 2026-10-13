"""As-of company-event context for one symbol at one decision time.

A filing's category is NSE's own subject (a fixed list of about 140), not a guess
from its text: keywords in boilerplate text ("SEBI", "in order to", "results
of e-voting") mislabelled many filings.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from ai_trader.news.store import NewsStore

WEEK_DAYS = 7  # "last 7 days" counts reach back this far

# NSE subjects of periodic or procedural filings, with no new business information.
ROUTINE_SUBJECTS = (
    "address change",
    "analysts/institutional investor meet/con. call updates",
    "certificate under sebi (depositories and participants)",
    "change in company secretary/compliance officer",
    "copy of newspaper publication",
    "disclosure of all complaints including scores complaints",
    "esop/esos/esps",
    "integrated filing- governance",
    "issue of duplicate share certificate",
    "loss of share certificates",
    "monitoring agency report",
    "quarterly compliance report on corporate governance",
    "registrar & share transfer agent update",
    "shareholders meeting",
    "statement of deviation(s) or variation(s) under reg. 32",
    "structural digital database",
    "trading window",
)


def routine(subject: str) -> bool:
    return " ".join(subject.lower().split()).startswith(ROUTINE_SUBJECTS)


def event_context(
    store: NewsStore,
    symbol: str,
    as_of: datetime,
    since: datetime,
    *,
    max_items: int = 5,
    max_chars: int = 400,
) -> dict[str, Any]:
    """Filings public at ``as_of``: since ``since`` (normally the prior close) with
    their NSE subjects, the last 7 days, scheduled board meetings and nearby
    corporate actions. Material means not routine (``ROUTINE_SUBJECTS``)."""
    recent = store.announcements(symbol, since, as_of)
    week = store.announcements(symbol, as_of - timedelta(days=WEEK_DAYS), as_of)
    material = [row for row in recent if not routine(row["subject"])]
    today = as_of.date()
    meetings = store.meetings(symbol, today, today + timedelta(days=3), as_of)
    actions = store.actions(symbol, today, today + timedelta(days=1))
    return {
        "since_prev_close": len(recent),
        "material_since_prev_close": len(material),
        "last_7_days": len(week),
        "categories": dict(sorted(Counter(row["subject"] for row in recent).items())),
        "items": [
            {
                "time": row["disseminated_at"][11:16],
                "date": row["disseminated_at"][:10],
                "subject": row["subject"][:120],
                "text": row["text"][:max_chars],
            }
            for row in material[:max_items]
        ],
        "board_meetings_next_3_days": [
            {"date": row["meeting_date"], "purpose": row["purpose"][:120]}
            for row in meetings
        ],
        "corporate_actions_today_or_next": [
            {"ex_date": row["ex_date"], "subject": row["subject"][:120]}
            for row in actions
        ],
    }
