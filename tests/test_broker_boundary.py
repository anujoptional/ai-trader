"""The door every vendor timestamp comes through, and what it guarantees.

Three types carry a moment in from outside: a streaming tick, a REST quote, and
a historical candle. Groww hands all three to us as epoch seconds, a websocket
frame may carry an offset of its own, and a hand-edited cache row may carry
none at all. Everything past this point asks *local* questions of what arrives
-- which trading session, how many minutes into it, what date to print -- and
those have different answers in different zones for the very same instant.

So the guarantee is made structurally rather than per consumer: normalize at
construction, and no aware datetime inside ``ai_trader`` carries a zone but
``INDIA_TIMEZONE``. A downstream type that forgot to convert is then not a bug
waiting to be found, it is a thing that cannot be built.

The tests are parametrized across all three rather than written out per type,
because the claim is about the boundary and not about any one of them -- a
fourth type added without a ``__post_init__`` should fail here, not quietly
inherit an exemption.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument, MarketQuote, MarketTick, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")

_Stamped = MarketTick | MarketQuote | OHLCVCandle


def _tick(timestamp: datetime) -> MarketTick:
    return MarketTick(instrument=_RELIANCE, price=Decimal("100"), timestamp=timestamp)


def _quote(last_trade_at: datetime) -> MarketQuote:
    return MarketQuote(
        instrument=_RELIANCE,
        last_price=Decimal("100"),
        last_trade_at=last_trade_at,
        open=Decimal("99"),
        high=Decimal("101"),
        low=Decimal("98"),
        previous_close=Decimal("99"),
        volume=1_000,
        day_change=Decimal("1"),
        day_change_percent=Decimal("1.01"),
    )


def _candle(timestamp: datetime) -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=timestamp,
        open=Decimal("99"),
        high=Decimal("101"),
        low=Decimal("98"),
        close=Decimal("100"),
        volume=1_000,
    )


_TYPES = (
    pytest.param(_tick, "timestamp", "Tick timestamp", id="tick"),
    pytest.param(_quote, "last_trade_at", "Quote last_trade_at", id="quote"),
    pytest.param(_candle, "timestamp", "Candle timestamp", id="candle"),
)


@pytest.mark.parametrize(("build", "attribute", "label"), _TYPES)
def test_a_foreign_zone_becomes_the_projects_own(
    build: Callable[[datetime], _Stamped],
    attribute: str,
    label: str,
) -> None:
    """The same instant, and a different answer to every local question.

    20:00 UTC on the 14th is 01:30 IST on the 15th, which is chosen so the two
    spellings disagree about the *date*. That is the whole cost of carrying a
    foreign zone: the equality on the first line holds either way, and the
    calendar day -- what a session label, a cache key and a printed report are
    all built on -- does not.
    """
    del label
    moment = datetime(2026, 9, 14, 20, 0, tzinfo=UTC)

    stamped = getattr(build(moment), attribute)

    assert stamped == moment
    assert stamped.tzinfo is INDIA_TIMEZONE
    assert stamped.date() == date(2026, 9, 15)
    # Non-vacuity: the input genuinely answered that question differently, so
    # the line above cannot be satisfied by handing the input straight back.
    assert moment.date() == date(2026, 9, 14)


@pytest.mark.parametrize(("build", "attribute", "label"), _TYPES)
def test_an_offset_equal_to_ist_is_still_replaced(
    build: Callable[[datetime], _Stamped],
    attribute: str,
    label: str,
) -> None:
    """A fixed +05:30 names exactly the same instants, and is still not IST.

    Every equality a test could write is already satisfied here, which is what
    makes the case worth having: only the zone identity distinguishes
    normalizing from passing through, and only normalizing keeps the guarantee
    the docstring above states. ``ZoneInfo`` is interned by key, so ``is`` is a
    real identity check rather than a coincidence of construction.
    """
    del label
    india_offset = timezone(timedelta(hours=5, minutes=30))
    moment = datetime(2026, 9, 15, 1, 30, tzinfo=india_offset)

    stamped = getattr(build(moment), attribute)

    assert stamped.tzinfo is INDIA_TIMEZONE
    assert stamped == moment
    assert moment.tzinfo is not INDIA_TIMEZONE


@pytest.mark.parametrize(("build", "attribute", "label"), _TYPES)
def test_a_naive_timestamp_is_refused_at_the_door(
    build: Callable[[datetime], _Stamped],
    attribute: str,
    label: str,
) -> None:
    """Refused rather than assumed, and the refusal says which field.

    ``astimezone`` on a naive datetime assumes the machine's local zone, so a
    moment that forgot its offset would silently acquire whichever zone the box
    happens to sit in -- and every session question asked of it downstream would
    be answered against a day that need not be the trading day. On a developer's
    laptop in IST that is invisible; on a CI runner in UTC it is five and a half
    hours of wrong answers. Refusing at construction means such a value cannot
    exist to be asked.
    """
    del attribute

    with pytest.raises(ValueError, match=f"{label} must be timezone-aware"):
        build(datetime(2026, 9, 14, 10, 0))
