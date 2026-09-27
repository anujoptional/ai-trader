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

Two schedules are published and both are exercised. The interesting property is
not that they differ — brokerage is a commercial term and of course it differs —
but *where* they differ: the statutory and exchange lines are identical by law,
and the brokerage lines converge once both caps bind. Tests that pinned only
Groww would let a future edit move a shared rate on one schedule alone.
"""

from dataclasses import fields
from decimal import Decimal

import pytest

from ai_trader.costs import (
    GROWW_BROKERAGE_CAP,
    GROWW_BROKERAGE_FLOOR,
    GROWW_BROKERAGE_FRACTION,
    GROWW_INTRADAY_EQUITY,
    ZERODHA_INTRADAY_EQUITY,
    CostModel,
)

_LAKH = Decimal(100_000)
_UNDER_CAP = Decimal(10_000)
"""A leg small enough that percentage brokerage is below the rupee cap."""

_FREE = CostModel(**{field.name: Decimal(0) for field in fields(CostModel)})
"""A schedule that charges nothing, for isolating the arithmetic from the rates.

Zeroed programmatically rather than field by field. An enumerated version of this
quietly stopped being free when the investor-protection line was added: the new
field took its live default, and a fixture whose whole purpose is to charge
nothing began charging. Built this way, a rate added tomorrow is zero on the day
it appears.
"""


# --- the charges themselves ---------------------------------------------------


def test_every_charge_is_itemised_and_the_total_is_their_sum() -> None:
    """Reconciled line by line, in rupees, against arithmetic done by hand.

    On a one-lakh leg: brokerage caps at Rs 20 twice, STT is 0.025% of one
    sell leg, exchange, IPFT and SEBI charges fall on both legs, stamp duty on
    the one buy leg, and GST is 18% of brokerage plus the three fee lines.
    """
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)

    assert cost.notional == _LAKH
    assert cost.brokerage == Decimal("40")
    assert cost.securities_transaction_tax == Decimal("25")
    assert cost.exchange_transaction_charge == Decimal("6.1398")
    assert cost.investor_protection_fund_charge == Decimal("0.0002")
    assert cost.regulator_fee == Decimal("0.2")
    assert cost.stamp_duty == Decimal("3")
    assert cost.goods_and_services_tax == Decimal("8.3412")
    assert cost.total == Decimal("82.6812")


def test_rates_are_fractions_of_notional_rather_than_percentages() -> None:
    """The unit check, done in rupees where a hundredfold error is visible.

    ``GROWW_BROKERAGE_FRACTION`` of ``0.001`` means a tenth of a percent, so a
    ten-thousand-rupee leg is charged ten rupees. Read as a percentage it would
    be a thousand, and every hurdle computed from it would be unreachable.
    """
    assert GROWW_BROKERAGE_FRACTION * _UNDER_CAP == Decimal("10")
    assert GROWW_INTRADAY_EQUITY.round_trip(_UNDER_CAP).brokerage == Decimal("20")


def test_the_total_as_a_fraction_is_the_total_divided_by_one_leg() -> None:
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)
    assert cost.fraction == Decimal("0.000826812")
    assert cost.fraction == GROWW_INTRADAY_EQUITY.round_trip_fraction(_LAKH)


def test_brokerage_is_capped_on_each_leg_not_on_the_round_trip() -> None:
    """Two capped legs are Rs 40, not Rs 20.

    Capping the round trip instead would halve the brokerage on exactly the
    positions where the cap matters, and would do it silently: the total would
    still look like a plausible cost, just a wrong one.
    """
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)
    assert cost.brokerage == GROWW_BROKERAGE_CAP * 2


def test_below_the_cap_brokerage_is_charged_at_the_rate() -> None:
    cost = GROWW_INTRADAY_EQUITY.round_trip(_UNDER_CAP)
    assert cost.brokerage == _UNDER_CAP * GROWW_BROKERAGE_FRACTION * 2
    assert cost.brokerage < GROWW_BROKERAGE_CAP * 2


def test_the_cap_starts_binding_at_twenty_thousand_a_leg() -> None:
    """Where the two brokerage regimes meet, and the cost curve bends.

    Below it the cost is a flat fraction of turnover; above it the fraction
    starts falling. That bend is the whole reason position size is a
    precondition for a small-target strategy rather than a detail of it.
    """
    boundary = GROWW_BROKERAGE_CAP / GROWW_BROKERAGE_FRACTION
    assert boundary == Decimal(20_000)

    below = GROWW_INTRADAY_EQUITY.round_trip_fraction(boundary - Decimal(1))
    at = GROWW_INTRADAY_EQUITY.round_trip_fraction(boundary)
    above = GROWW_INTRADAY_EQUITY.round_trip_fraction(boundary + Decimal(1))

    assert below == at
    assert above < at


def test_the_rupee_floor_is_a_second_bend_below_the_cap() -> None:
    """Groww's Rs 5 minimum, and the 2.5% ceiling on the minimum itself.

    Three regimes, not two. Above Rs 20,000 a leg the cap binds; between
    Rs 5,000 and Rs 20,000 the rate binds; below Rs 5,000 the floor binds; and
    below Rs 200 even the floor is capped at 2.5% of turnover. All far beneath
    the clip this system trades, and asserted anyway because a schedule that
    quietly under-reports at small sizes cannot be checked against a contract
    note at any size.
    """
    # 0.1% of Rs 5,000 is exactly Rs 5, so the floor does not bind there.
    assert GROWW_INTRADAY_EQUITY.round_trip(Decimal(5_000)).brokerage == Decimal(10)
    # 0.1% of Rs 4,000 is Rs 4, below the floor, so both legs pay Rs 5.
    assert GROWW_INTRADAY_EQUITY.round_trip(Decimal(4_000)).brokerage == (
        GROWW_BROKERAGE_FLOOR * 2
    )
    # 2.5% of Rs 150 is Rs 3.75, below the floor, so the floor is itself capped.
    assert GROWW_INTRADAY_EQUITY.round_trip(Decimal(150)).brokerage == Decimal("7.5")


def test_gst_is_levied_on_brokerage_and_fees_but_not_on_the_other_taxes() -> None:
    """STT and stamp duty are themselves taxes and are not taxed again."""
    cost = GROWW_INTRADAY_EQUITY.round_trip(_LAKH)
    base = (
        cost.brokerage
        + cost.exchange_transaction_charge
        + cost.investor_protection_fund_charge
        + cost.regulator_fee
    )

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

    # 0.00025 charged once against 0.000030699 charged twice.
    assert cost.securities_transaction_tax / cost.exchange_transaction_charge == (
        Decimal("0.00025") / (Decimal("0.000030699") * 2)
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


# --- the one simplification, and how big it is --------------------------------


def _leg(notional: Decimal, *, is_sell: bool) -> Decimal:
    """One leg's charges at the notional that leg actually transacted.

    Rates written out rather than read off the model, for the reason this file
    gives at the top. The floor is not reproduced because every size used below
    is above the cap; if that stopped being true the zero-move test would say
    so rather than quietly measuring the wrong thing.
    """
    brokerage = min(Decimal(20), notional * Decimal("0.001"))
    fees = (
        brokerage
        + notional * Decimal("0.000030699")
        + notional * Decimal("0.000000001")
        + notional * Decimal("0.000001")
    )
    levy = notional * (Decimal("0.00025") if is_sell else Decimal("0.00003"))
    return fees + levy + fees * Decimal("0.18")


def _both_legs(entry: Decimal, exit_notional: Decimal, *, long: bool) -> Decimal:
    """A round trip with each leg charged at its own notional.

    A long buys then sells, a short sells then buys, so which leg carries STT
    and which carries stamp duty follows from the direction alone.
    """
    return _leg(entry, is_sell=not long) + _leg(exit_notional, is_sell=long)


def test_the_hand_written_schedule_is_the_model_when_the_price_holds_still() -> None:
    """The precondition for measuring the simplification at all.

    ``round_trip`` charges both legs on one notional. Arranged per leg and fed
    that notional twice, the schedule above has to come back to exactly what the
    model charges -- otherwise the gaps measured below are gaps between two cost
    models rather than the one approximation being weighed.
    """
    for notional in (Decimal(50_000), _LAKH, Decimal(250_000)):
        charged = GROWW_INTRADAY_EQUITY.round_trip(notional).total
        assert _both_legs(notional, notional, long=True) == charged
        assert _both_legs(notional, notional, long=False) == charged


def test_direction_picks_the_charge_and_the_outcome_picks_the_sign() -> None:
    """Which leg is mispriced, and which way.

    The error sits entirely on the exit leg, since the entry is charged on its
    own notional and is therefore right. A long exits by selling, so what it
    gets wrong is STT; a short exits by buying, so what it gets wrong is stamp
    duty. Whether that is an overcharge or an undercharge is then decided by
    where the price finished, not by the direction.
    """
    move = Decimal("0.002")
    charged = GROWW_INTRADAY_EQUITY.round_trip(_LAKH).total

    # (went long, made money) -> does one notional overstate the true charge?
    cases = {
        (True, True): False,  # sold higher than it bought: STT understated
        (True, False): True,  # sold lower: STT overstated
        (False, True): True,  # bought back lower: stamp duty overstated
        (False, False): False,  # bought back higher: stamp duty understated
    }
    for (long, won), overstates in cases.items():
        # The exit leg is the larger one when a long wins or a short loses.
        up = long is won
        exit_notional = _LAKH * (1 + move if up else 1 - move)
        truth = _both_legs(_LAKH, exit_notional, long=long)
        assert (charged > truth) is overstates, (
            f"a {'long' if long else 'short'} that "
            f"{'won' if won else 'lost'} was charged {charged} against a true "
            f"{truth}."
        )


def test_the_error_is_a_few_hundredths_of_a_per_cent_of_the_charge() -> None:
    """The magnitude both docstrings quote, so neither can drift unnoticed.

    They quoted a fifth of a per cent and "a few thousandths of the total cost"
    before anything measured it, and both were pessimistic by three to four
    times. Pinned here as bands rather than exact figures: the point is the
    order of magnitude, and a band survives a rate being corrected.
    """
    move = Decimal("0.002")
    charged = GROWW_INTRADAY_EQUITY.round_trip(_LAKH).total
    assert charged.quantize(Decimal("0.01")) == Decimal("82.68")

    understated = _both_legs(_LAKH, _LAKH * (1 + move), long=True) - charged
    overstated = charged - _both_legs(_LAKH, _LAKH * (1 - move), long=False)

    # About six paise on a long and a little over one on a short.
    assert Decimal("0.055") < understated < Decimal("0.060")
    assert Decimal("0.012") < overstated < Decimal("0.015")

    # Both under a tenth of a per cent of what is charged, which is what the
    # module docstring and replay's exit path now claim.
    assert understated / charged < Decimal("0.001")
    assert overstated / charged < Decimal("0.001")

    # STT is 0.00025 against stamp duty's 0.00003 and the lines both legs pay
    # are common to the two, which leaves the long side the larger by about 4.3.
    assert Decimal("4.2") < understated / overstated < Decimal("4.4")


# --- the hurdle ---------------------------------------------------------------


def test_required_gross_is_the_round_trip_cost_plus_the_margin_asked_for() -> None:
    margin = Decimal("0.001")
    required = GROWW_INTRADAY_EQUITY.required_gross_fraction(_LAKH, margin)

    assert required == Decimal("0.000826812") + margin


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


def test_a_schedule_must_state_its_brokerage_because_nothing_can_default_it() -> None:
    """The division the field list encodes: commercial term versus statute.

    Every other rate is set by law or by the exchange and is the same whoever
    executes the order, so each is defaulted. Brokerage is the one figure a
    broker chooses, and there is no universal value to fall back on — a default
    would quietly present one broker's commercial terms as a law of nature. So
    a bare ``CostModel()`` does not construct, and that is the point.
    """
    with pytest.raises(TypeError):
        CostModel()  # type: ignore[call-arg]


def test_the_groww_schedule_carries_the_published_groww_rates() -> None:
    assert GROWW_INTRADAY_EQUITY.brokerage_fraction == GROWW_BROKERAGE_FRACTION
    assert GROWW_INTRADAY_EQUITY.brokerage_cap == GROWW_BROKERAGE_CAP
    assert GROWW_INTRADAY_EQUITY.brokerage_floor == GROWW_BROKERAGE_FLOOR


def test_the_two_schedules_differ_in_brokerage_and_in_nothing_else() -> None:
    """What "same charges, different broker" has to mean to be checkable.

    Asserted field by field rather than by eye. Every line below brokerage is
    statutory or an exchange tariff, so a future edit that moved one of them on
    one schedule alone would be wrong by construction — and would otherwise be
    invisible, because at the configured clip the two totals agree anyway.
    """
    shared = {
        "securities_transaction_tax_fraction",
        "exchange_transaction_fraction",
        "investor_protection_fund_fraction",
        "regulator_fee_fraction",
        "stamp_duty_fraction",
        "goods_and_services_tax_fraction",
    }
    for field in shared:
        assert getattr(GROWW_INTRADAY_EQUITY, field) == getattr(
            ZERODHA_INTRADAY_EQUITY, field
        )

    assert (
        GROWW_INTRADAY_EQUITY.brokerage_fraction
        != ZERODHA_INTRADAY_EQUITY.brokerage_fraction
    )


def test_both_schedules_pay_the_exchange_the_same_rupees_per_crore() -> None:
    """Why the two published rate cards look contradictory and are not.

    NSE bills Rs 307 per crore per side in the cash segment and has re-split it
    between the transaction line and the investor-protection line: it was
    Rs 297 + Rs 10 and is now Rs 306.99 + Rs 0.01. Groww's charges page still
    quotes the old split and Zerodha's the new one, so the two appear to
    disagree about an exchange-set charge, which is impossible. The total is
    what is billed, and it is the total this asserts.
    """
    crore = Decimal(10_000_000)
    for schedule in (GROWW_INTRADAY_EQUITY, ZERODHA_INTRADAY_EQUITY):
        cost = schedule.round_trip(crore)
        per_side = (
            cost.exchange_transaction_charge + cost.investor_protection_fund_charge
        ) / 2
        assert per_side == Decimal("307")


def test_at_the_configured_clip_the_broker_choice_costs_nothing() -> None:
    """The headline result, and the reason the swap is not an economic decision.

    Both schedules cap brokerage at Rs 20 a leg, and a one-lakh clip is well
    past both caps, so every line of the round trip is identical and the totals
    agree to the paisa. Worth pinning: it is what makes ``AGENTS.md`` rule 10's
    broker swap a configuration change rather than a change of strategy.
    """
    assert ZERODHA_INTRADAY_EQUITY.round_trip(_LAKH).total == Decimal("82.6812")
    assert ZERODHA_INTRADAY_EQUITY.round_trip_fraction(
        _LAKH
    ) == GROWW_INTRADAY_EQUITY.round_trip_fraction(_LAKH)


def test_the_zerodha_cap_binds_much_higher_and_below_it_zerodha_is_cheaper() -> None:
    """Equal caps at unequal rates, so the two curves meet at Rs 66,666.67.

    Groww's cap binds above Rs 20,000 a leg; Zerodha's 0.03% rate does not
    reach Rs 20 until Rs 66,666.67. Between those two figures Groww is paying a
    flat Rs 20 while Zerodha is still paying a percentage, which is the whole
    of the difference between the schedules. Bracketed rather than pinned at
    the exact boundary, which does not terminate.
    """
    below = Decimal(66_666)
    above = Decimal(66_667)

    assert ZERODHA_INTRADAY_EQUITY.round_trip_fraction(
        below
    ) < GROWW_INTRADAY_EQUITY.round_trip_fraction(below)
    assert ZERODHA_INTRADAY_EQUITY.round_trip_fraction(
        above
    ) == GROWW_INTRADAY_EQUITY.round_trip_fraction(above)


def test_zerodha_is_never_the_dearer_schedule() -> None:
    """The ordering holds at every size, which is why Groww is not the cheaper.

    Groww is the default because it is the broker this system connects to, and
    for no other reason. Stated as a test so the docstring that says so cannot
    drift away from the arithmetic.
    """
    ladder = [
        Decimal(size) for size in (150, 1_000, 5_000, 20_000, 66_666, 100_000, 500_000)
    ]
    for notional in ladder:
        assert ZERODHA_INTRADAY_EQUITY.round_trip_fraction(
            notional
        ) <= GROWW_INTRADAY_EQUITY.round_trip_fraction(notional)


def test_the_cheaper_schedule_clears_a_small_target_where_the_dearer_does_not() -> None:
    """Where the choice of broker stops being cosmetic.

    At a twenty-thousand-rupee clip a 0.2% gross capture is a net loss at
    Groww's rates and a net gain at Zerodha's — the same move, the same
    strategy, opposite signs. It is the concrete case behind treating the
    schedule as a parameter rather than as an import.
    """
    move = Decimal("0.002")
    small = Decimal(20_000)

    assert GROWW_INTRADAY_EQUITY.net_fraction(small, move) < 0
    assert ZERODHA_INTRADAY_EQUITY.net_fraction(small, move) > 0


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
    """Below the caps, where the schedules actually differ."""
    margin = Decimal("0.001")
    small = Decimal(20_000)

    assert ZERODHA_INTRADAY_EQUITY.required_gross_fraction(
        small, margin
    ) < GROWW_INTRADAY_EQUITY.required_gross_fraction(small, margin)
