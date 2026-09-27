"""A duration the report states must be the duration the run used.

Every timing parameter in this system is a ``Decimal`` -- latency, cooldown --
because every other quantity is. Every clock in this system is a ``timedelta``,
because that is what ``datetime`` arithmetic speaks. The two do not meet without
a conversion, and ``timedelta`` refuses a ``Decimal`` outright, so the obvious
bridge is ``timedelta(seconds=float(count))``. It rounds twice: once into a
float, once to the nearest microsecond. The ``Decimal`` still appears in the
report, so a rounded run publishes conditions that did not produce its numbers
-- the one thing Section 7.2 asks a recorded result not to do.

**One of those roundings is not a lost digit, it is lookahead.** A stated
latency of 59.9999999 seconds rounds *up* to exactly sixty, and sixty seconds
puts a fill on a bar's closing boundary rather than inside it. The fill rule
then hands it the bar's *close* -- a price that had not printed when the fill
was supposed to happen. The rounding walks back in through a side door the fill
rule cannot see, which is why the first test here is about that number and not
about precision in the abstract.

**And the reverse rounding turns a gate on and off at the same time.** A
cooldown of a ten-billionth of a minute is greater than zero, so the gate that
reads ``cooldown_minutes > 0`` opens; converted, it is a zero-length interval,
so the gate never blocks anything. The run would print a cooldown in its
conditions and apply it exactly never.

So conversions are refused rather than rounded, and refused at the moment the
configuration is written rather than partway through a run. The tests below are
in three groups: what the shared converter accepts and refuses, that the
minute-counting converter carries no binary residue, and that the three objects
a caller actually constructs refuse an impossible duration before any candle is
fetched.

``SimulatedTrade.holding_minutes`` is the one call site not exercised here. It
reads the same ``elapsed_minutes`` the second group pins, it is a reported
figure rather than a decision input, and reaching a sub-second holding period
through a replay means steering a fixture tape rather than stating a property.
"""

from datetime import datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from ai_trader.broker import Instrument
from ai_trader.costs import GROWW_INTRADAY_EQUITY
from ai_trader.market import (
    INDIA_TIMEZONE,
    ONE_MINUTE,
    ONE_SECOND,
    elapsed_minutes,
    exact_timedelta,
)
from ai_trader.replay import FillModel, minutes_since_open
from ai_trader.replay.portfolio import ReplayPortfolio
from ai_trader.strategy import StrategyConfig

_MICROSECOND = timedelta(microseconds=1)

_FRACTION = Decimal("0.0001")
"""A half-spread and slippage figure, used only so a FillModel can be built."""


def _fill(latency: Decimal) -> FillModel:
    """A fill model that differs from its neighbours only in latency."""
    return FillModel(
        latency_seconds=latency,
        half_spread_fraction=_FRACTION,
        slippage_fraction=_FRACTION,
    )


# --------------------------------------------------------------------------
# What the converter accepts, and what it refuses.
# --------------------------------------------------------------------------


def test_the_float_bridge_really_does_cross_the_bar_boundary() -> None:
    """The premise of everything below, established before it is relied on.

    Without this, the next test is a rule with no reason: it would show that a
    number is refused, not that refusing it prevents anything. This shows the
    old bridge turning a latency that lands *inside* a bar into one that lands
    exactly *on* its closing edge, which is where the fill rule switches from
    the open to the close.
    """
    assert timedelta(seconds=float(Decimal("59.9999999"))) == ONE_MINUTE


def test_a_latency_that_would_round_onto_a_bar_edge_is_refused() -> None:
    """The lookahead case. Refused, not rounded, and not silently accepted."""
    with pytest.raises(ValueError, match="latency_seconds"):
        _fill(Decimal("59.9999999"))


def test_a_sub_microsecond_duration_is_refused_rather_than_zeroed() -> None:
    """The frictionless assumption must be stated, never arrived at by rounding.

    The first assertion establishes what the old bridge did with this number,
    so the refusal below reads as preventing something rather than as pedantry.
    """
    assert timedelta(seconds=float(Decimal("1E-9"))) == timedelta(0)

    with pytest.raises(ValueError, match="latency_seconds"):
        _fill(Decimal("1E-9"))


def test_a_whole_number_of_microseconds_converts_exactly() -> None:
    """The accepting half. Without it the refusals could be a blanket no."""
    assert exact_timedelta(Decimal("0.5"), ONE_SECOND, name="x") == timedelta(
        microseconds=500_000
    )
    assert exact_timedelta(Decimal("0.0000005"), ONE_MINUTE, name="x") == timedelta(
        microseconds=30
    )
    assert exact_timedelta(Decimal(0), ONE_SECOND, name="x") == timedelta(0)


def test_the_smallest_accepted_cooldown_is_still_a_real_interval() -> None:
    """A gate that opens must be a gate that can close.

    Whatever survives conversion and is greater than zero has to be a non-zero
    interval, or the ``cooldown_minutes > 0`` check reports a gate that is on
    while it is off. Thirty microseconds is the smallest such cooldown that can
    be written down as a terminating decimal number of minutes.
    """
    smallest = Decimal("0.0000005")

    assert smallest > 0
    assert exact_timedelta(smallest, ONE_MINUTE, name="x") > timedelta(0)


def test_a_count_too_long_for_the_ambient_precision_is_still_refused() -> None:
    """The guard against a tail that multiplies itself into an integer.

    At the default twenty-eight digits, a count with a long enough fraction
    rounds *into* a whole number of microseconds during the conversion, and a
    whole-microsecond check performed at that precision would then wave through
    a duration that is not one. The first assertion shows the trap is real
    rather than theoretical; the second shows the converter is not in it.
    """
    count = Decimal("1.0000000000000000000000000001")

    with localcontext() as context:
        context.prec = 28
        product = count * 1_000_000
    assert product == product.to_integral_value()

    with pytest.raises(ValueError, match="whole number"):
        exact_timedelta(count, ONE_SECOND, name="x")


def test_the_refusal_says_which_duration_was_impossible() -> None:
    """Two parameters share the converter, so the message must name one.

    A caller reading "not a whole number of microseconds" with no parameter in
    it has to guess which of the durations they stated was the problem.
    """
    with pytest.raises(ValueError) as latency:
        exact_timedelta(Decimal("1E-9"), ONE_SECOND, name="latency_seconds")
    with pytest.raises(ValueError) as cooldown:
        exact_timedelta(Decimal("1E-10"), ONE_MINUTE, name="cooldown_minutes")

    assert "latency_seconds" in str(latency.value)
    assert "cooldown_minutes" in str(cooldown.value)
    assert "cooldown_minutes" not in str(latency.value)


def test_a_duration_that_is_not_a_number_is_refused() -> None:
    """``Decimal`` carries NaN and infinity; a clock cannot."""
    for impossible in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        with pytest.raises(ValueError, match="finite"):
            exact_timedelta(impossible, ONE_SECOND, name="latency_seconds")


# --------------------------------------------------------------------------
# Counting minutes without picking up binary noise.
# --------------------------------------------------------------------------


def test_elapsed_minutes_does_not_inherit_a_float_division() -> None:
    """Three tenths of a second is a clean two hundredths of a minute.

    ``total_seconds`` divides whole microseconds by a million in binary, so the
    0.3 it returns is not three tenths, and dividing that by sixty produces a
    figure that is wrong in its last dozen digits. The first assertion shows
    that error; the second shows this path does not have it.
    """
    delta = timedelta(microseconds=300_000)

    assert Decimal(delta.total_seconds()) / 60 != Decimal("0.005")
    assert elapsed_minutes(delta) == Decimal("0.005")


def test_elapsed_minutes_counts_whole_minutes_as_whole_numbers() -> None:
    """The ordinary case, so the test above is not the only shape covered."""
    assert elapsed_minutes(timedelta(minutes=375)) == Decimal(375)
    assert elapsed_minutes(-ONE_MINUTE) == Decimal(-1)
    assert elapsed_minutes(_MICROSECOND) * 60_000_000 == Decimal(1)


def test_the_decision_clock_is_exact_at_a_sub_second_offset() -> None:
    """The clock the square-off threshold is compared against.

    ``minutes_since_open`` meets ``square_off_minutes_since_open``, a
    ``Decimal``, and a boundary met by a figure carrying binary residue can
    resolve one way in a replay and the other way live. A fill lands mid-bar by
    design, so sub-second offsets are reachable here rather than hypothetical.
    """
    moment = datetime(2026, 9, 22, 9, 45, 0, 300_000, tzinfo=INDIA_TIMEZONE)

    assert minutes_since_open(moment) == Decimal("30.005")


# --------------------------------------------------------------------------
# Refused at the door, not partway through a run.
# --------------------------------------------------------------------------


def test_a_strategy_refuses_an_unrepresentable_cooldown() -> None:
    """The shared configuration, so the refusal covers the live path too."""
    with pytest.raises(ValueError, match="cooldown_minutes"):
        StrategyConfig(cooldown_minutes=Decimal("1E-10"))


def test_a_strategy_still_accepts_a_cooldown_it_can_honour() -> None:
    """Non-vacuity: the refusal above is about the value, not the field."""
    config = StrategyConfig(cooldown_minutes=Decimal(5))

    assert config.cooldown_minutes == Decimal(5)


def test_a_book_refuses_an_unrepresentable_cooldown() -> None:
    """The object that owns the gate, checked where the gate lives."""
    strategy = StrategyConfig()

    with pytest.raises(ValueError, match="cooldown_minutes"):
        ReplayPortfolio(
            universe=(Instrument(exchange="NSE", trading_symbol="ALPHA"),),
            max_open_positions=strategy.max_open_positions,
            sizing=strategy.sizing_policy(),
            costs=GROWW_INTRADAY_EQUITY,
            fill=_fill(Decimal(30)),
            cooldown_minutes=Decimal("1E-10"),
        )
