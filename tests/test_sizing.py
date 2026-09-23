"""Tests for fixed-clip sizing and decision-time cost estimates.

Three failures matter here; the rest are cosmetic.

The first is **pricing a trade that was never placed**. Quantity is integral, so
a one-lakh clip almost never buys exactly a lakh of stock, and every fraction in
the estimate is a fraction *of notional*. Feeding the target in where the filled
value belongs is a mistake that shifts each number by under two percent — far
too small to look wrong and far too large to ignore on a target measured in
tenths of a percent. So the notional is reconciled by hand here rather than
recomputed from the policy.

The second is **rounding the exit the wrong way**. An exit rounded to the nearer
tick is under the paying price about half the time, and the trade still fills
and still looks like a win. The test for it does not hunt for a price where
nearest-rounding happens to fail; it asserts the stronger and exactly true
property that the chosen exit is the *tightest* tick that clears the margin —
it pays, and one tick closer to the entry does not.

The third is **a partial exit becoming expressible**. The stated model is that
sells are the whole position, and that holds today because a single quantity
serves both legs. It would stop holding the moment a second quantity appeared,
so the shape is pinned rather than trusted.
"""

import ast
from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import pytest

from ai_trader.costs import (
    FIXED_CLIP_NOTIONAL,
    GROWW_INTRADAY_EQUITY,
    NSE_EQUITY_TICK,
    STATED_GROSS_TARGET,
    CostModel,
    SizingPolicy,
    TradeCostEstimate,
    round_down_to_tick,
    round_up_to_tick,
)

_TICK = NSE_EQUITY_TICK
_MARGIN = Decimal("0.001")
"""A tenth of a percent kept after charges — the lower end of the stated aim."""

_POLICY = SizingPolicy(
    target_notional=FIXED_CLIP_NOTIONAL,
    net_margin_fraction=_MARGIN,
)

_PRICES = (
    Decimal("2450"),
    Decimal("100"),
    Decimal("1234.55"),
    Decimal("37.40"),
    Decimal("9999.95"),
    Decimal("512.30"),
)
"""A spread of quotes: a penny-ish name, one that divides the clip exactly, and
one dear enough that whole shares overshoot it by a tenth."""


# --- sizing -------------------------------------------------------------------


def test_the_clip_is_a_floor_so_quantity_rounds_up() -> None:
    # Forty shares is 98,000 and forty-one is 100,450. Only one of those is
    # worth at least a lakh.
    assert _POLICY.quantity_for(Decimal("2450")) == 41


def test_every_quote_buys_at_least_the_clip() -> None:
    """The postcondition the whole module rests on, checked at every price.

    Stated as the inequality rather than as six expected share counts, because
    the counts are what the arithmetic produces and the inequality is what the
    caller asked for. A rounding mode that lost a hair on the division would
    still satisfy a hand-written count and fail this.
    """
    for price in _PRICES:
        quantity = _POLICY.quantity_for(price)

        assert quantity * price >= FIXED_CLIP_NOTIONAL, price
        # And no more than necessary: one share fewer must fall short.
        assert (quantity - 1) * price < FIXED_CLIP_NOTIONAL, price


def test_notional_is_what_was_filled_rather_than_what_was_asked_for() -> None:
    estimate = _POLICY.estimate(Decimal("2450"))

    assert estimate.notional == Decimal("100450")
    assert estimate.target_notional == FIXED_CLIP_NOTIONAL
    assert estimate.notional > estimate.target_notional
    assert estimate.excess_notional == Decimal("450")


def test_charges_are_computed_on_the_filled_notional() -> None:
    estimate = _POLICY.estimate(Decimal("2450"))

    # Restated by hand on 100,450, not recomputed from the model's constants.
    # Brokerage is capped at 20 a leg; STT and stamp fall on one leg each.
    assert estimate.cost.brokerage == Decimal("40")
    assert estimate.cost.securities_transaction_tax == Decimal("25.1125")
    assert estimate.cost.exchange_transaction_charge == Decimal("6.1674291")
    assert estimate.cost.investor_protection_fund_charge == Decimal("0.0002009")
    assert estimate.cost.regulator_fee == Decimal("0.2009")
    assert estimate.cost.stamp_duty == Decimal("3.0135")
    assert estimate.cost.goods_and_services_tax == Decimal("8.3463354")
    assert estimate.cost.total == Decimal("82.8408654")


def test_a_stock_quoted_above_the_clip_buys_a_single_share() -> None:
    # Not a defensive branch: NSE lists names above a lakh a share. One share
    # already clears a floor, so there is nothing here to refuse.
    assert _POLICY.quantity_for(Decimal("150000")) == 1


def test_a_dear_name_overshoots_the_clip_and_says_so() -> None:
    """The cost of a fixed clip meeting an indivisible share, made visible.

    Fifty per cent above the clip is not an error and is not correctable — one
    share is the smallest trade there is. It is reported rather than hidden so
    that a position limit, when one exists, has a number to refuse.
    """
    estimate = _POLICY.estimate(Decimal("150000"))

    assert estimate.notional == Decimal("150000")
    assert estimate.excess_notional == Decimal("50000")


def test_a_price_that_divides_the_clip_exactly_leaves_no_excess() -> None:
    estimate = _POLICY.estimate(Decimal("100"))

    assert estimate.quantity == 1000
    assert estimate.notional == FIXED_CLIP_NOTIONAL
    assert estimate.excess_notional == 0


def test_cash_outlay_is_the_notional_plus_the_charges() -> None:
    estimate = _POLICY.estimate(Decimal("2450"))

    assert estimate.cash_outlay == Decimal("100450") + Decimal("82.8408654")


# --- the exit price -----------------------------------------------------------


def test_the_chosen_exits_pay_the_margin_asked_for() -> None:
    for price in _PRICES:
        estimate = _POLICY.estimate(price)

        assert estimate.long_net_rupees >= estimate.target_net_rupees, price
        assert estimate.short_net_rupees >= estimate.target_net_rupees, price


def test_one_tick_closer_to_the_entry_does_not_pay() -> None:
    """The exits are the tightest ticks that clear, not merely ticks that do.

    This is what rules out rounding to the nearer tick without needing to find
    a price where nearest-rounding misbehaves. A long exit one tick lower gives
    up ``quantity * tick``, and the surplus the rounding created is strictly
    less than that, so the shortfall is guaranteed rather than incidental.
    """
    for price in _PRICES:
        estimate = _POLICY.estimate(price)
        given_up = estimate.quantity * _TICK

        assert estimate.long_net_rupees - given_up < estimate.target_net_rupees, price
        assert estimate.short_net_rupees - given_up < estimate.target_net_rupees, price


def test_exits_land_on_tick_boundaries() -> None:
    for price in _PRICES:
        estimate = _POLICY.estimate(price)

        assert estimate.long_exit_price % _TICK == 0, price
        assert estimate.short_exit_price % _TICK == 0, price


def test_the_long_exit_is_above_the_entry_and_the_short_below_it() -> None:
    for price in _PRICES:
        estimate = _POLICY.estimate(price)

        assert estimate.long_exit_price > price, price
        assert estimate.short_exit_price < price, price


def test_both_directions_are_priced_so_neither_is_chosen_here() -> None:
    estimate = _POLICY.estimate(Decimal("2450"))

    # Cost is direction-symmetric, so on a tick-aligned entry the two exits sit
    # the same distance either side. Naming a side is the scanner's job.
    assert estimate.long_exit_price == Decimal("2454.50")
    assert estimate.short_exit_price == Decimal("2445.50")


def test_a_price_already_on_a_tick_is_left_alone_by_both_roundings() -> None:
    assert round_up_to_tick(Decimal("100.20"), _TICK) == Decimal("100.20")
    assert round_down_to_tick(Decimal("100.20"), _TICK) == Decimal("100.20")


def test_rounding_moves_to_the_next_boundary_in_the_named_direction() -> None:
    assert round_up_to_tick(Decimal("100.21"), _TICK) == Decimal("100.25")
    assert round_down_to_tick(Decimal("100.24"), _TICK) == Decimal("100.20")


def test_a_non_positive_tick_is_rejected() -> None:
    for rounding in (round_up_to_tick, round_down_to_tick):
        with pytest.raises(ValueError, match="tick must be positive"):
            rounding(Decimal("100"), Decimal(0))


# --- what the tick costs the strategy -----------------------------------------


def test_the_tick_is_a_larger_share_of_a_cheap_stock() -> None:
    dear = _POLICY.estimate(Decimal("2450")).tick_fraction
    cheap = _POLICY.estimate(Decimal("100")).tick_fraction

    assert cheap > dear
    assert cheap == Decimal("0.0005")


def test_a_coarse_tick_costs_more_than_the_margin_it_rounds() -> None:
    # At a hundred rupees one tick is half of a tenth-percent target, so the
    # grid is coarser than the quantity being measured on it. What that does to
    # a particular trade depends on where the entry sits — see below.
    estimate = _POLICY.estimate(Decimal("100"))

    assert estimate.tick_fraction >= estimate.net_margin_fraction / 2
    assert estimate.long_net_rupees > estimate.target_net_rupees


def test_what_the_tick_costs_depends_on_the_entry_rather_than_the_target() -> None:
    """The measured pair the documents quote, pinned so it cannot rot.

    An earlier version of this file claimed a coarse tick *systematically*
    overshoots the target. It does not: once the grid is coarser than the
    target, the exit can only land on the grid, so what the trade ends up
    asking for is set by where the entry sits in it. An entry already on a tick
    overshoots by nothing at all; an entry one tick above it overshoots by
    nearly four times. Both readings are below, because either alone is the
    misleading one.

    The overshoot is not a windfall. It means asking the market for a move four
    times larger, which fills correspondingly less often — a cost that shows up
    as trades that never happen rather than as trades that lose.
    """
    policy = SizingPolicy.from_gross_target(
        target_notional=FIXED_CLIP_NOTIONAL,
        gross_target_fraction=Decimal("0.001"),
    )

    on_the_grid = policy.estimate(Decimal("100"))
    one_tick_above = policy.estimate(Decimal("100.05"))

    # A hundred rupees is an exact multiple of a five-paisa tick, and so is the
    # exit it needs. Nothing is rounded away and nothing is overshot.
    assert on_the_grid.long_exit_price == Decimal("100.10")
    assert on_the_grid.long_net_rupees == on_the_grid.target_net_rupees

    # Five paisa higher, the exit cannot stop at 100.15 and must reach 100.20.
    assert one_tick_above.long_exit_price == Decimal("100.20")
    ratio = one_tick_above.long_net_rupees / one_tick_above.target_net_rupees
    assert ratio.quantize(Decimal("0.0001")) == Decimal("3.8841")


# --- the shape of the model ---------------------------------------------------


def test_only_one_quantity_exists_so_a_partial_exit_cannot_be_expressed() -> None:
    quantities = [name for name in TradeCostEstimate.__slots__ if "quantity" in name]

    assert quantities == ["quantity"]


def test_the_estimate_records_the_assumptions_that_produced_it() -> None:
    # A journal entry written from this has to answer "what did it believe",
    # which means the clip, the tick and the margin travel with the result.
    estimate = _POLICY.estimate(Decimal("2450"))

    assert estimate.target_notional == FIXED_CLIP_NOTIONAL
    assert estimate.tick_size == _TICK
    assert estimate.net_margin_fraction == _MARGIN
    assert estimate.entry_price == Decimal("2450")


def test_required_gross_is_the_round_trip_cost_plus_the_margin() -> None:
    estimate = _POLICY.estimate(Decimal("2450"))
    expected = GROWW_INTRADAY_EQUITY.required_gross_fraction(estimate.notional, _MARGIN)

    assert estimate.required_gross_fraction == expected


def test_a_cheaper_schedule_lowers_the_exit() -> None:
    # Zeroed programmatically rather than field by field, so that a rate added
    # to ``CostModel`` later cannot quietly make this schedule charge something.
    free = CostModel(**{field.name: Decimal(0) for field in fields(CostModel)})
    cheap = SizingPolicy(
        target_notional=FIXED_CLIP_NOTIONAL,
        net_margin_fraction=_MARGIN,
        costs=free,
    ).estimate(Decimal("2450"))

    assert cheap.long_exit_price < _POLICY.estimate(Decimal("2450")).long_exit_price


def test_costs_imports_nothing_from_the_rest_of_the_system() -> None:
    """The stdlib-only contract, checked rather than described.

    ``costs`` is imported by the scanner today and will be imported by replay
    and by the risk engine. A single import of ``Direction`` from the scanner
    would invert that and couple all three, which is why both exits are priced
    instead of one being selected.
    """
    package = Path(__file__).resolve().parents[1] / "src" / "ai_trader" / "costs"
    imported: set[str] = set()

    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)

    foreign = {
        name
        for name in imported
        if name.startswith("ai_trader") and not name.startswith("ai_trader.costs")
    }

    assert foreign == set()


# --- stating the target as a move rather than as profit kept ------------------


def test_a_gross_target_becomes_the_margin_it_leaves_after_costs() -> None:
    # Restated by hand at the clip: a round trip on a lakh is 82.6812, so
    # 0.000826812 of it. A 0.2% move therefore keeps 0.1173188%.
    policy = SizingPolicy.from_gross_target(
        target_notional=FIXED_CLIP_NOTIONAL,
        gross_target_fraction=STATED_GROSS_TARGET,
    )

    assert policy.net_margin_fraction == Decimal("0.001173188")


def test_the_stated_target_is_an_upper_bound_on_what_a_name_must_move() -> None:
    """Converting at the clip is safe in the one direction that matters.

    The clip is the *smallest* fill allowed, so it pays the *largest* cost
    fraction. Every real quote rounds up to a whole share, turns over more, and
    pays less — so the hurdle it actually faces lands at or under the figure the
    strategy was stated in. This is the property that makes stating the target
    gross legitimate rather than merely convenient, so it is asserted across the
    spread rather than argued for in a comment.
    """
    policy = SizingPolicy.from_gross_target(
        target_notional=FIXED_CLIP_NOTIONAL,
        gross_target_fraction=STATED_GROSS_TARGET,
    )

    for price in _PRICES:
        estimate = policy.estimate(price)

        assert estimate.required_gross_fraction <= STATED_GROSS_TARGET, price


def test_a_target_its_own_costs_consume_is_refused() -> None:
    # At a twenty-thousand clip the round trip is 0.2715%, so a 0.2% move is a
    # loss. Continuing with a negative margin would price every exit the wrong
    # side of the entry, which is why this raises instead of clamping.
    with pytest.raises(ValueError, match="does not clear costs"):
        SizingPolicy.from_gross_target(
            target_notional=Decimal(20_000),
            gross_target_fraction=STATED_GROSS_TARGET,
        )


def test_the_stated_target_is_named_rather_than_written_at_call_sites() -> None:
    # 0.2% above the buy price, before charges. Pinned so that the one place it
    # can be changed stays the only place it appears.
    assert STATED_GROSS_TARGET == Decimal("0.002")


# --- validation ---------------------------------------------------------------


def test_a_non_positive_clip_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_notional must be positive"):
        SizingPolicy(target_notional=Decimal(0), net_margin_fraction=_MARGIN)


def test_a_non_positive_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="net_margin_fraction must be positive"):
        SizingPolicy(
            target_notional=FIXED_CLIP_NOTIONAL,
            net_margin_fraction=Decimal(0),
        )


def test_a_non_positive_tick_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="tick_size must be positive"):
        SizingPolicy(
            target_notional=FIXED_CLIP_NOTIONAL,
            net_margin_fraction=_MARGIN,
            tick_size=Decimal(0),
        )


def test_a_non_positive_price_is_rejected() -> None:
    with pytest.raises(ValueError, match="entry_price must be positive"):
        _POLICY.quantity_for(Decimal(0))


def test_a_margin_that_would_drive_a_short_exit_negative_is_refused() -> None:
    # A short's whole notional is the most it can make, so a margin above one
    # is not a demanding target but an impossible one.
    absurd = SizingPolicy(
        target_notional=FIXED_CLIP_NOTIONAL,
        net_margin_fraction=Decimal("1.5"),
    )

    with pytest.raises(ValueError, match="no positive short exit"):
        absurd.estimate(Decimal("2450"))
