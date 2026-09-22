"""Time bucketing shared by the market-data and feature layers.

Candle building and volume differencing must agree exactly on where a minute
starts, and the volume poller and the feature engine must agree exactly on
where a session starts, so both definitions live here rather than being
repeated per module. A second definition of either would be a second source of
truth for where one bucket ends and the next begins.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")

ONE_MINUTE = timedelta(minutes=1)

_ONE_DAY = timedelta(days=1)

SESSION_OPEN_TIME = time(hour=9, minute=15)
"""When the NSE continuous session opens, in IST."""


def minute_start(timestamp: datetime) -> datetime:
    """Return the start of the minute containing ``timestamp``."""
    return timestamp.replace(second=0, microsecond=0)


def trading_session_date(timestamp: datetime) -> date:
    """Return the trading session ``timestamp`` falls in, as an IST date.

    A moment before 09:15 IST belongs to the session that opened on the
    previous calendar day, because that is the session whose figures the
    exchange is still serving: its running totals reset at the open, not at
    midnight. The invariant is that two moments share a label exactly when no
    reset has happened between them.

    Keying on the plain calendar date instead would file a pre-open reading
    under the session that has not begun yet, and the reset at 09:15 would then
    look like a total moving backwards inside one session rather than the
    session boundary it is.

    The label does not claim the market traded that day. A Saturday pre-open
    moment is labelled Friday, which is correct under the invariant above --
    the total being served is still Friday's, because no reset has intervened.
    """
    local = timestamp.astimezone(INDIA_TIMEZONE)
    if local.time() < SESSION_OPEN_TIME:
        return local.date() - _ONE_DAY
    return local.date()


__all__ = [
    "INDIA_TIMEZONE",
    "ONE_MINUTE",
    "SESSION_OPEN_TIME",
    "minute_start",
    "trading_session_date",
]
