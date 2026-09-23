"""Tests for the transaction cost model.

Two kinds of error matter here and the rest are cosmetic. The first is a unit
error: every rate in the module is a fraction of notional, and comparing one
against a percentage is a hundredfold mistake that still type-checks and still
prints a plausible number, so the charges are reconciled in rupees rather than
only against each other. The second is the brokerage cap, which is per leg —
applying it once would halve the brokerage on every position large enough for
it to bind, which is every position where it binds at all.

The itemised charges are restated here by hand rather than recomputed from the
model's own constants. A test that multiplies the same constants in the same
order cannot fail when the order is wrong.
"""

from decimal import Decimal

import pytest

from ai_trader.costs import (
    BROKERAGE_CAP,
    BROKERAGE_FRACTION,
    GROWW_INTRADAY_EQUITY,
    CostModel,
)

_LAKH = Decimal(100_000)
_UNDER_CAP = Decimal(10_000)
"""A leg small enough that percentage brokerage is below the rupee cap."""

_FREE = CostModel(
    brokerage_fraction=Decimal(0),
    brokerage_cap=Decimal(0),
    securities_transaction_tax_fraction=Decimal(0),
    exchange_transaction_fraction=Decimal(0),
    regulator_fee_fraction=Decimal(0),
    stamp_duty_fraction=Decimal(0),
    goods_and_services_tax_fraction=Decimal(0),
)
"""A schedule that charges nothing, for isolating the arithmetic from the rates."""


# --- the charges themselves ---------------------------------------------------


def test_every_charge_is_itemised_and_the_total_is_their_sum() -> None:
    """Reconciled line by line, in rupees, against arithmetic done by hand.

    On a one-lakh leg: brokerage caps at Rs 20 twice, STT is 0.025% of one
    sell leg, exchange and SEBI charges fall on both legs, stamp duty on the
    one buy leg, and GST is 18% of brokerage plus the two fee lines.
    """
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)

    assert cost.notional == _LAKH
    assert cost.brokerage == Decimal("40")
    assert cost.securities_transaction_tax == Decimal("25")
    assert cost.exchange_transaction_charge == Decimal("5.94")
    assert cost.regulator_fee == Decimal("0.2")
    assert cost.stamp_duty == Decimal("3")
    assert cost.goods_and_services_tax == Decimal("8.3052")
    assert cost.total == Decimal("82.4452")


def test_rates_are_fractions_of_notional_rather_than_percentages() -> None:
    """The unit check, done in rupees where a hundredfold error is visible.

    ``BROKERAGE_FRACTION`` of ``0.001`` means a tenth of a percent, so a
    ten-thousand-rupee leg is charged ten rupees. Read as a percentage it would
    be a thousand, and every hurdle computed from it would be unreachable.
    """
    assert BROKERAGE_FRACTION * _UNDER_CAP == Decimal("10")
    assert GROWW_INTRADAY_EQUITY.round_trip(_UNDER_CAP).brokerage == Decimal("20")


def test_the_total_as_a_fraction_is_the_total_divided_by_one_leg() -> None:
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)
    assert cost.fraction == Decimal("0.000824452")
    assert cost.fraction == GROWW_INTRADAY_EQUITY.round_trip_fraction(_LAKH)


def test_brokerage_is_capped_on_each_leg_not_on_the_round_trip() -> None:
    """Two capped legs are Rs 40, not Rs 20.

    Capping the round trip instead would halve the brokerage on exactly the
    positions where the cap matters, and would do it silently: the total would
    still look like a plausible cost, just a wrong one.
    """
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)
    assert cost.brokerage == BROKERAGE_CAP * 2


def test_below_the_cap_brokerage_is_charged_at_the_rate() -> None:
    cost = GROWW_INTRADAY_EQUITY.round_trip(_UNDER_CAP)
    assert cost.brokerage == _UNDER_CAP * BROKERAGE_FRACTION * 2
    assert cost.brokerage < BROKERAGE_CAP * 2


def test_the_cap_starts_binding_at_twenty_thousand_a_leg() -> None:
    """Where the two brokerage regimes meet, and the cost curve bends.

    Below it the cost is a flat fraction of turnover; above it the fraction
    starts falling. That bend is the whole reason position size is a
    precondition for a small-target strategy rather than a detail of it.
    """
    boundary = BROKERAGE_CAP / BROKERAGE_FRACTION
    assert boundary == Decimal(20_000)

    below = GROWW_INTRADAY_EQUITY.round_trip_fraction(boundary - Decimal(1))
    at = GROWW_INTRADAY_EQUITY.round_trip_fraction(boundary)
    above = GROWW_INTRADAY_EQUITY.round_trip_fraction(boundary + Decimal(1))

    assert below == at
    assert above < at


def test_gst_is_levied_on_brokerage_and_fees_but_not_on_the_other_taxes() -> None:
    """STT and stamp duty are themselves taxes and are not taxed again."""
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)
    base = cost.brokerage + cost.exchange_transaction_charge + cost.regulator_fee

    assert cost.goods_and_services_tax == base * Decimal("0.18")
    assert cost.goods_and_services_tax < (
        base + cost.securities_transaction_tax + cost.stamp_duty
    ) * Decimal("0.18")


def test_stt_and_stamp_duty_are_charged_once_and_the_rest_twice() -> None:
    """The leg arithmetic that makes direction irrelevant.

    Doubling the notional doubles every line, so the per-leg multiplicities
    have to be read off a ratio between charges instead. STT sits on the sell
    leg and stamp duty on the buy leg; exchange and SEBI charges sit on both.
    """
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)

    # 0.00025 charged once against 0.0000297 charged twice.
    assert cost.securities_transaction_tax / cost.exchange_transaction_charge == (
        Decimal("0.00025") / (Decimal("0.0000297") * 2)
    )
    # 0.00003 charged once against 0.000001 charged twice.
    assert cost.stamp_duty / cost.regulator_fee == (
        Decimal("0.00003") / (Decimal("0.000001") * 2)
    )


def test_a_short_round_trip_costs_what_a_long_one_costs() -> None:
    """There is no direction parameter, and that is the claim being tested.

    A short sells then buys and a long buys then sells. Either way there is
    exactly one buy and exactly one sell, so the sell-side STT and the buy-side
    stamp duty are both paid once whichever way round they happen. A model that
    took a direction would imply the two differ; this asserts the signature
    never grows one by accident.
    """
    parameters = set(CostModel.round_trip.__annotations__) - {"return"}
    assert parameters == {"notional"}


def test_the_cost_fraction_falls_with_size_and_flattens_onto_an_asymptote() -> None:
    """Why the hurdle cannot be a constant.

    The same strategy needs a 0.27% move to break even on a twenty-thousand
    rupee clip and a 0.045% move on a five-lakh one. No single target
    percentage describes both, which is section 7's point about invented
    numbers made arithmetic.
    """
    ladder = [Decimal(size) for size in (20_000, 50_000, 100_000, 500_000)]
    fractions = [GROWW_INTRADAY_EQUITY.round_trip_fraction(size) for size in ladder]

    assert fractions == sorted(fractions, reverse=True)
    assert fractions[0] > Decimal("0.0027")
    assert fractions[-1] < Decimal("0.00045")

    # The floor once brokerage has stopped mattering: STT, stamp duty and the
    # fee lines are all proportional, so the curve never reaches zero.
    huge = GROWW_INTRADAY_EQUITY.round_trip_fraction(Decimal(100_000_000))
    assert huge > Decimal("0.0003")


# --- the hurdle ---------------------------------------------------------------


def test_required_gross_is_the_round_trip_cost_plus_the_margin_asked_for() -> None:
    margin = Decimal("0.001")
    required = GROWW_INTRADAY_EQUITY.required_gross_fraction(_LAKH, margin)

    assert required == Decimal("0.000824452") + margin


def test_the_same_margin_needs_a_bigger_move_on_a_smaller_clip() -> None:
    """The hurdle is a property of the position, not of the strategy."""
    margin = Decimal("0.001")
    small = GROWW_INTRADAY_EQUITY.required_gross_fraction(Decimal(20_000), margin)
    large = GROWW_INTRADAY_EQUITY.required_gross_fraction(Decimal(500_000), margin)

    assert small > large


def test_net_fraction_inverts_required_gross_fraction() -> None:
    """Ask what it takes to keep a margin, deliver exactly that move, keep it."""
    margin = Decimal("0.0005")
    required = GROWW_INTRADAY_EQUITY.required_gross_fraction(_LAKH, margin)

    assert GROWW_INTRADAY_EQUITY.net_fraction(_LAKH, required) == margin


def test_a_two_tenths_of_a_percent_move_loses_money_on_a_small_clip() -> None:
    """The finding that made the cost model a precondition for the scanner.

    A 0.2% gross capture is a net loss below roughly thirty thousand rupees a
    clip and a net gain above it. Stated as a test because it is the concrete
    reason no target percentage is hard-coded anywhere in this system.
    """
    move = Decimal("0.002")

    assert GROWW_INTRADAY_EQUITY.net_fraction(Decimal(20_000), move) < 0
    assert GROWW_INTRADAY_EQUITY.net_fraction(Decimal(100_000), move) > 0


def test_a_non_positive_notional_is_rejected() -> None:
    """Dividing by it would be the alternative, and zero cost is not a cost."""
    for notional in (Decimal(0), Decimal(-1)):
        with pytest.raises(ValueError, match="notional must be positive"):
            GROWW_INTRADAY_EQUITY.round_trip(notional)


def test_a_non_positive_margin_is_rejected() -> None:
    """Zero is break-even, which is not an objective, and below it is worse."""
    for margin in (Decimal(0), Decimal("-0.001")):
        with pytest.raises(ValueError, match="net_margin_fraction must be positive"):
            GROWW_INTRADAY_EQUITY.required_gross_fraction(_LAKH, margin)


# --- the schedule is data -----------------------------------------------------


def test_the_default_model_carries_the_published_rates() -> None:
    assert GROWW_INTRADAY_EQUITY == CostModel()
    assert GROWW_INTRADAY_EQUITY.brokerage_fraction == BROKERAGE_FRACTION
    assert GROWW_INTRADAY_EQUITY.brokerage_cap == BROKERAGE_CAP


def test_a_schedule_can_be_varied_so_replay_can_hold_the_strategy_fixed() -> None:
    """The experiment this shape exists for: same strategy, different broker.

    With every rate zeroed, the required gross move collapses onto the margin
    and a gross move is kept whole — which shows the arithmetic above the rates
    carries no charges of its own.
    """
    margin = Decimal("0.001")

    assert _FREE.round_trip_fraction(_LAKH) == 0
    assert _FREE.required_gross_fraction(_LAKH, margin) == margin
    assert _FREE.net_fraction(_LAKH, Decimal("0.002")) == Decimal("0.002")


def test_a_cheaper_schedule_lowers_the_hurdle() -> None:
    halved = CostModel(
        brokerage_fraction=BROKERAGE_FRACTION / 2, brokerage_cap=Decimal(10)
    )
    margin = Decimal("0.001")

    assert halved.required_gross_fraction(
        _LAKH, margin
    ) < GROWW_INTRADAY_EQUITY.required_gross_fraction(_LAKH, margin)
