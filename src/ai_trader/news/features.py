"""As-of company-event context for one symbol at one decision time."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from ai_trader.news.store import NewsStore

# Ordered: the first matching rule wins. Subjects are NSE's own categories.
CATEGORIES = (
    ("results", ("financial result", "outcome of board meeting", "results")),
    ("dividend_or_capital", ("dividend", "bonus", "split", "buyback", "buy back")),
    ("deal", ("acquisition", "amalgamation", "merger", "scheme of arrangement")),
    ("order_or_contract", ("order", "contract", "bagging", "award", "receipt of")),
    ("fundraise", ("qip", "preferential", "rights issue", "allotment", "fund raising")),
    ("rating", ("credit rating",)),
    ("regulatory_or_legal", ("penalty", "sebi", "litigation", "show cause", "tax")),
    ("management", ("resignation", "appointment", "cessation", "change in director")),
    ("investor_meet", ("analyst", "investor meet", "con. call", "earnings call")),
    (
        "routine",
        ("trading window", "newspaper", "certificate", "compliance", "loss of share"),
    ),
)


def categorize(subject: str, text: str = "") -> str:
    lowered = f"{subject} {text[:200]}".lower()
    for name, needles in CATEGORIES:
        if any(needle in lowered for needle in needles):
            return name
    return "other"


def event_context(
    store: NewsStore,
    symbol: str,
    as_of: datetime,
    since: datetime,
    *,
    max_items: int = 5,
    max_chars: int = 400,
) -> dict[str, Any]:
    """Filings public at ``as_of``: since ``since`` (normally the prior close),
    the last 7 days, scheduled board meetings and nearby corporate actions."""
    recent = store.announcements(symbol, since, as_of)
    week = store.announcements(symbol, as_of - timedelta(days=7), as_of)
    categories = Counter(categorize(row["subject"], row["text"]) for row in recent)
    material = [
        row for row in recent if categorize(row["subject"], row["text"]) != "routine"
    ]
    today = as_of.date()
    meetings = store.meetings(symbol, today, today + timedelta(days=3), as_of)
    actions = store.actions(symbol, today, today + timedelta(days=1))
    return {
        "since_prev_close": len(recent),
        "material_since_prev_close": len(material),
        "last_7_days": len(week),
        "categories": dict(sorted(categories.items())),
        "items": [
            {
                "time": row["disseminated_at"][11:16],
                "date": row["disseminated_at"][:10],
                "category": categorize(row["subject"], row["text"]),
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
