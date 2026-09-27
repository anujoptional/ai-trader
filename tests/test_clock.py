"""The clock's two readings of a moment, and the rule that has to link them.

``session_minute_offset`` answers "how far into the session is this?" and
``trading_session_date`` answers "which session is this?". They are separate
functions with separate callers -- the feature engine guards on the offset, the
session-scoped indicators key their resets on the label -- so nothing stops one
from drifting away from the other.

That drift is not hypothetical. The label rule used to be the plain calendar
date, so a bar timestamped before the bell was filed under a session that had
not opened yet, while the offset rule called the same bar pre-open. The cost
was not a mislabelled row: a pre-open bar seeded the session VWAP and OBV with
auction turnover, and because the label never changed at 09:15 no reset ever
fired to clear it. An outer guard in the feature engine kept that path
unreachable, but ``session_date`` was exported and answered the question
wrongly on its own.

So the link is worth stating outright and testing directly::

    session_minute_offset(t) >= 0  <->  trading_session_date(t) == local date

Both sides are the same fact -- has the bell rung? -- and the tests below pin
it at the boundary, on either side of it, and across a whole day at once.
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from ai_trader.clock import (
    INDIA_TIMEZONE,
    session_minute_offset,
    trading_session_date,
)

_OPEN = datetime(2026, 9, 14, 9, 15, tzinfo=INDIA_TIMEZONE)
"""09:15 IST on Monday 14 September 2026, the moment the NSE bell rings."""


def test_a_moment_before_the_bell_belongs_to_the_previous_session() -> None:
    """Midnight opens no session; the one it falls inside opened yesterday.

    This is the case the calendar date gets wrong, and it gets it wrong by a
    whole trading day rather than by a rounding error: under the old rule this
    instant was filed under the 15th, a session that would not begin for
    another nine and a quarter hours. The second assertion is what stops the
    first from passing by coincidence -- the two answers genuinely differ here.
    """
    midnight = datetime(2026, 9, 15, 0, 0, tzinfo=INDIA_TIMEZONE)

    assert trading_session_date(midnight) == date(2026, 9, 14)
    assert midnight.date() == date(2026, 9, 15)


@pytest.mark.parametrize(
    ("moment", "expected_offset", "expected_session"),
    [
        (_OPEN - timedelta(seconds=1), -1, date(2026, 9, 13)),
        (_OPEN, 0, date(2026, 9, 14)),
        (_OPEN + timedelta(seconds=59), 0, date(2026, 9, 14)),
    ],
)
def test_the_bell_is_the_boundary_for_both_readings(
    moment: datetime,
    expected_offset: int,
    expected_session: date,
) -> None:
    """One second either side of 09:15, both readings change together.

    The offset floors rather than truncates, which is the whole difference in
    the first row: a second before the bell is ``-1`` minute into the session,
    not ``0``. Truncating toward zero would report ``0`` there and quietly
    claim the session had started.

    The 13th is a Sunday, and that is correct rather than a slip. This clock is
    arithmetic over a day boundary drawn at 09:15; it answers which window an
    instant falls in, not whether the exchange opened that morning. Holidays
    and weekends are the calendar's business, and the store's.
    """
    assert session_minute_offset(moment) == expected_offset
    assert trading_session_date(moment) == expected_session


def test_the_offset_and_the_label_never_disagree_about_the_bell() -> None:
    """The invariant itself, checked minute by minute across a whole day."""
    midnight = datetime(2026, 9, 14, 0, 0, tzinfo=INDIA_TIMEZONE)
    moments = [midnight + timedelta(minutes=n) for n in range(24 * 60)]

    disagreements = [
        moment
        for moment in moments
        if (session_minute_offset(moment) >= 0)
        != (trading_session_date(moment) == moment.date())
    ]
    assert disagreements == []

    # Two constants agreeing is not an invariant. A day that never crossed the
    # bell would satisfy the comparison above under a rule that always returned
    # the calendar date, so these pin that the sweep straddles it: midnight is
    # 555 minutes short of 09:15, and 23:59 is 884 minutes past it.
    offsets = [session_minute_offset(moment) for moment in moments]
    assert min(offsets) == -555
    assert max(offsets) == 884


def test_the_carrying_zone_does_not_change_either_answer() -> None:
    """The same instant, spelled in UTC, is the same session and the same offset.

    An aware datetime names an instant and carries a zone only as a way of
    spelling it, so both functions convert before they read anything local.
    This instant is chosen to make that load-bearing: 07:00 IST on the 15th
    falls in the hour and a half where the UTC date has already rolled over but
    the bell has not rung, so reading ``.date()`` off the UTC spelling gives the
    15th and the correct answer is the 14th. Reading the offset in UTC would be
    wrong by the offset itself -- ``-465`` rather than ``-135``.
    """
    in_utc = datetime(2026, 9, 15, 1, 30, tzinfo=UTC)

    assert trading_session_date(in_utc) == date(2026, 9, 14)
    assert session_minute_offset(in_utc) == -135
    assert in_utc.date() == date(2026, 9, 15)
