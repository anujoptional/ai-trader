"""Shared lookup for the most recent completed NSE trading session.

Every check that needs real market data starts the same way: walk backwards
from today until a weekday returns candles. Three CLIs grew their own copy of
that walk and the copies drifted -- one lost the timezone guard the others
kept -- so the behaviour a test pinned in one place was not the behaviour that
ran in another. This module is the single copy.

The window is a parameter rather than a constant because the callers genuinely
differ: the historical check reads only the first fifteen minutes of a session
while the market-state and feature checks read all of it. ``session_end`` does
double duty as the end of the candle window and as the threshold for "has
today's window closed yet", which is what lets a short window find today's
session shortly after it opens instead of waiting for the closing bell.

The broker is typed as ``ReadOnlyBroker`` rather than ``GrowwBroker``: the walk
needs one read-only method, and naming the protocol keeps this module free of
any particular broker, which is what lets Groww be swapped for Kite later.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle, ReadOnlyBroker

INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
SESSION_START = time(hour=9, minute=15)
SESSION_END = time(hour=15, minute=30)
MAX_WEEKDAYS = 10


class SessionNotFoundError(RuntimeError):
    """Raised when no completed trading session can be found for validation."""


def find_recent_completed_session(
    broker: ReadOnlyBroker,
    now: datetime,
    *,
    instrument: Instrument = RELIANCE,
    session_start: time = SESSION_START,
    session_end: time = SESSION_END,
) -> tuple[date, tuple[OHLCVCandle, ...]]:
    """Return the newest completed session's date and candles.

    Walks back at most ``MAX_WEEKDAYS`` weekdays, skipping weekends and any
    weekday the broker returns nothing for, which is how exchange holidays are
    handled without shipping a holiday calendar.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("The current time must be timezone-aware.")

    local_now = now.astimezone(INDIA_TIMEZONE)
    candidate = local_now.date()
    if local_now < datetime.combine(candidate, session_end, tzinfo=INDIA_TIMEZONE):
        candidate -= timedelta(days=1)

    weekdays_checked = 0
    while weekdays_checked < MAX_WEEKDAYS:
        if candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
            continue

        weekdays_checked += 1
        candles = broker.get_historical_candles(
            instrument=instrument,
            start=datetime.combine(candidate, session_start, tzinfo=INDIA_TIMEZONE),
            end=datetime.combine(candidate, session_end, tzinfo=INDIA_TIMEZONE),
            interval=CandleInterval.ONE_MINUTE,
        )
        if candles:
            return candidate, candles

        candidate -= timedelta(days=1)

    raise SessionNotFoundError(
        "No completed NSE trading session was found in the last 10 weekdays."
    )


__all__ = [
    "INDIA_TIMEZONE",
    "MAX_WEEKDAYS",
    "RELIANCE",
    "SESSION_END",
    "SESSION_START",
    "SessionNotFoundError",
    "find_recent_completed_session",
]
