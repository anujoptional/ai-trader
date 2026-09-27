"""The one clock every layer reads. This exchange trades in IST, so this is it.

Candle building and volume differencing must agree exactly on where a minute
starts, and the volume poller and the feature engine must agree exactly on
where a session starts, so both definitions live here rather than being
repeated per module. A second definition of either would be a second source of
truth for where one bucket ends and the next begins.

**Why this is a top-level leaf.** It imports nothing from ``ai_trader``, and
that is load-bearing rather than incidental: ``market`` is built on ``broker``,
so a clock owned by ``market`` is one ``broker`` cannot reach without inverting
the stack. ``broker/groww.py`` duly grew its own ``ZoneInfo("Asia/Kolkata")`` --
a second answer to what IST is, in the module that parses every vendor
timestamp the system ingests. The two agreed, which is exactly why nothing
caught it. Sitting below every package means there is one definition for
everyone to import, and no layer has an excuse to write its own.

**Everything is IST.** Timestamps are normalized to IST at every boundary, not
merely made timezone-aware. An aware UTC instant and an aware IST instant
compare and hash identically, so that choice is invisible to arithmetic -- but
it is not invisible to ``.date()``, ``.hour`` or a cached CSV, each of which
answers a *local* question. Normalizing to the market's own zone is what makes
the obvious reading of those the correct one.

The same argument covers the two converters at the bottom. Durations are stated
as ``Decimal`` because every other quantity in this system is, and ``timedelta``
is what the clock arithmetic needs; the two do not meet without a conversion,
and a conversion written per call site is a rounding rule written per call site.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
"""Indian Standard Time: the only zone this system reasons in.

Defined once, here, and imported everywhere else. IST observes no daylight
saving, so no local time this system constructs is ever nonexistent or
ambiguous -- a property several callers rely on when they replace the time of
day on a date.
"""

ONE_MINUTE = timedelta(minutes=1)

ONE_SECOND = timedelta(seconds=1)

_ONE_DAY = timedelta(days=1)

_MICROSECOND = timedelta(microseconds=1)
"""The finest interval a ``datetime`` can express, and so the limit below.

Not a policy anyone chose. A ``timedelta`` stores whole microseconds and
nothing smaller, which is why the converters below treat one as the floor
rather than picking a tolerance.
"""

SESSION_OPEN_TIME = time(hour=9, minute=15)
"""When the NSE continuous session opens, in IST."""

SESSION_CLOSE_TIME = time(hour=15, minute=30)
"""When the NSE continuous session closes, in IST."""

SESSION_MINUTES = (
    datetime.combine(date.min, SESSION_CLOSE_TIME)
    - datetime.combine(date.min, SESSION_OPEN_TIME)
) // ONE_MINUTE
"""How many one-minute bars a full session contains: 375.

Derived from the two times rather than written down, because it was previously
written down in three places -- ``market/state.py``, ``scanner/feasibility.py``
and ``cli/_session.py`` -- each of which had to be right independently. Deriving
it means a change to the exchange's hours moves one line and every consumer
follows, and it means the count and the clock cannot disagree.
"""


def minute_start(timestamp: datetime) -> datetime:
    """Return the start of the minute containing ``timestamp``."""
    return timestamp.replace(second=0, microsecond=0)


def session_open_moment(timestamp: datetime) -> datetime:
    """Return 09:15 IST on the calendar day ``timestamp`` falls on, in IST.

    The anchor every other session function measures from. It was previously
    written out three times -- in the feature layer's minute offset, in the
    replay engine's decision clock, and implicitly in the session label's
    time-of-day comparison -- as the same four-keyword ``replace``. Three copies
    of one anchor is three places to move an exchange's opening bell, and the
    project already keeps a test whose only job is to catch two of them drifting
    apart. Deriving them from one line makes that agreement structural instead.

    Note this answers a question about a *calendar day*, not about a session: it
    returns a moment for a Sunday too. ``trading_session_date`` is what decides
    which session a moment belongs to.
    """
    local = timestamp.astimezone(INDIA_TIMEZONE)
    return local.replace(
        hour=SESSION_OPEN_TIME.hour,
        minute=SESSION_OPEN_TIME.minute,
        second=0,
        microsecond=0,
    )


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
    if local < session_open_moment(local):
        return local.date() - _ONE_DAY
    return local.date()


def session_minute_offset(timestamp: datetime) -> int:
    """Return whole minutes from the NSE session open to ``timestamp``.

    Negative before 09:15 IST, which a pre-open candle would be, and which is
    deliberate: a pre-open bar should fail an "at least this far in" test rather
    than wrap around into passing one.

    Measured from the same anchor as ``trading_session_date``, which is what
    ties the two together: this is non-negative exactly when that function
    returns the plain IST calendar date. Where those two disagree is where a
    pre-open bar gets filed under a session that has not begun.

    The count comes from the clock rather than from how many candles have been
    seen, so it stays correct across the gaps a thinly traded instrument leaves
    in its session -- a name that prints nothing between 11:00 and 11:20 still
    reports the true elapsed minutes on its next candle.

    The division floors rather than truncating, so a moment a fraction of a
    second before the bell reports ``-1`` and not ``0``. Truncating gave the
    half-minute before every open an offset that claimed it was inside the
    session; no minute-aligned candle can land there, but the rule should be
    right about the boundary it exists to describe.
    """
    local = timestamp.astimezone(INDIA_TIMEZONE)
    return (local - session_open_moment(local)) // ONE_MINUTE


def exact_timedelta(count: Decimal, unit: timedelta, *, name: str) -> timedelta:
    """Turn a ``Decimal`` count of ``unit`` into a ``timedelta``, or refuse to.

    ``timedelta`` rejects a ``Decimal`` outright, so the obvious workaround is
    ``timedelta(seconds=float(count))`` — and it rounds twice. The float
    conversion rounds, and ``timedelta`` rounds again to the nearest microsecond.
    A duration finer than a microsecond therefore becomes *zero*, which is the
    frictionless assumption arrived at silently rather than stated; and one that
    is merely awkward becomes a neighbouring value nobody asked for. Either way
    the report still prints the ``Decimal`` the caller stated, so the conditions
    on the page are not the conditions that produced the numbers, which is the
    one thing Section 7.2 asks of a recorded result.

    One of those roundings is worse than a lost digit. A latency of
    ``59.9999999`` seconds rounds *up* to exactly sixty, and sixty seconds lands
    a fill on a bar's boundary instead of inside it, where the replay's fill rule
    hands it the close rather than the open — a price that had not printed at the
    stated fill moment. A rounding that crosses that boundary reintroduces the
    very lookahead the fill rule exists to prevent, and does it invisibly.

    So a count that is not a whole number of microseconds is refused rather than
    rounded, and ``name`` is required so the refusal says which stated duration
    was impossible. Refusing loses the caller who sweeps a parameter by dividing
    by three; that is the intended trade. They can round to microseconds
    themselves and have the report state what they actually ran.
    """
    if not count.is_finite():
        raise ValueError(f"{name} must be a finite duration, got {count}")
    scale = unit // _MICROSECOND
    if scale <= 0:
        raise ValueError(f"{name} needs a unit of at least a microsecond, got {unit}")

    # Widened past the ambient precision on purpose: at the default 28 digits a
    # count with a long enough tail multiplies *into* an integer, and the check
    # below would then wave through a value that is not one.
    with localcontext() as context:
        context.prec = len(count.as_tuple().digits) + 12
        microseconds = count * scale

    whole = microseconds.to_integral_value()
    if whole != microseconds:
        raise ValueError(
            f"{name} of {count} is {microseconds} microseconds, which is not a "
            "whole number of them. Rounding it would run at a duration the "
            "report does not state; round it yourself to say what you meant."
        )
    return timedelta(microseconds=int(whole))


def elapsed_minutes(delta: timedelta) -> Decimal:
    """How many minutes a ``timedelta`` spans, without binary rounding.

    ``delta.total_seconds()`` divides a whole microsecond count by a million in
    binary, so it is inexact for most sub-second durations: a tenth of a second
    comes back as 0.1000000000000000055511151231257827, and handing that to
    ``Decimal`` preserves the error rather than removing it. Dividing the
    microsecond count itself keeps every digit that was there.

    It matters where the result meets a threshold stated as a ``Decimal``.
    ``square_off_minutes_since_open`` is one, and a boundary compared against a
    value carrying binary noise can resolve one way in a replay and the other way
    live — the two agreeing is the whole premise of the fork after the scanner.

    The division by sixty can still repeat, because twenty seconds is a third of
    a minute, and that quotient is rounded at the ambient precision like every
    other one in this system. What is gone is the binary error underneath it.
    """
    return Decimal(delta // _MICROSECOND) / Decimal(ONE_MINUTE // _MICROSECOND)


def minutes_since_open(moment: datetime) -> Decimal:
    """How far into the session a moment is, in IST minutes, fraction kept.

    The same quantity ``session_minute_offset`` returns, before the floor: this
    is what a threshold stated as a ``Decimal`` gets compared against, and
    ``square_off_minutes_since_open`` is one. A boundary met by a figure carrying
    binary noise can resolve one way in a replay and the other way live, which is
    why the subtraction goes through ``elapsed_minutes`` rather than
    ``total_seconds``.

    Kept distinct from ``session_minute_offset`` because the two answer different
    questions and always did: which bar of the session is this (an index, and so
    an ``int``), versus how much of the session has elapsed (a duration, and so
    exact). What they must not do is measure from different opens, and now they
    cannot -- there is one anchor and both call it.
    """
    local = moment.astimezone(INDIA_TIMEZONE)
    return elapsed_minutes(local - session_open_moment(local))


__all__ = [
    "INDIA_TIMEZONE",
    "ONE_MINUTE",
    "ONE_SECOND",
    "SESSION_CLOSE_TIME",
    "SESSION_MINUTES",
    "SESSION_OPEN_TIME",
    "elapsed_minutes",
    "exact_timedelta",
    "minute_start",
    "minutes_since_open",
    "session_minute_offset",
    "session_open_moment",
    "trading_session_date",
]
