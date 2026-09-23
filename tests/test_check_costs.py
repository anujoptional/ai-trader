"""Tests for the cost and sizing check.

This is the one diagnostic that talks to no broker, so unlike its siblings it
has no configuration branch and no exit code 2 — the surface worth testing is
arithmetic rather than failure handling.

Two properties carry the weight. The first is that **the CLI invents nothing**:
every printed figure is reconciled against ``SizingPolicy`` directly, so a
rendering helper that quietly rounded or rescaled would show up here rather than
in a decision. The second is that **the gross target and the margin it leaves
stay distinguishable** — a 0.2% move from the buy price keeps about 0.12% once
charges are paid at a one-lakh clip, a difference of roughly a third, so a
figure that did not say which of the two it was would be worse than none.
"""

import json
from decimal import Decimal

from pytest import CaptureFixture

from ai_trader.cli.check_costs import main
from ai_trader.costs import FIXED_CLIP_NOTIONAL, SizingPolicy


def _run(capsys: CaptureFixture[str], *arguments: str) -> tuple[int, dict]:
    exit_code = main(list(arguments))
    captured = capsys.readouterr()
    assert captured.err == ""
    return exit_code, json.loads(captured.out)


def test_the_default_run_succeeds_without_a_broker_or_a_market_session(
    capsys: CaptureFixture[str],
) -> None:
    exit_code, output = _run(capsys)

    assert exit_code == 0
    assert output["clip_notional"] == "100000.0000"
    assert output["exit_model"] == "whole position, one exit, no scaling out"


def test_the_printed_estimate_matches_the_policy_it_claims_to_report(
    capsys: CaptureFixture[str],
) -> None:
    _, output = _run(capsys, "2450")

    row = output["estimates"][0]["rows"][0]
    expected = SizingPolicy.from_gross_target(
        target_notional=FIXED_CLIP_NOTIONAL,
        gross_target_fraction=Decimal("0.001"),
    ).estimate(Decimal("2450"))

    # Forty shares is 98,000 — under the clip. Forty-one is 100,450.
    assert row["quantity"] == expected.quantity == 41
    assert Decimal(row["notional"]) == expected.notional == Decimal("100450")
    assert Decimal(row["excess_notional"]) == expected.excess_notional == Decimal("450")
    assert Decimal(row["long_exit_price"]) == expected.long_exit_price
    assert Decimal(row["short_exit_price"]) == expected.short_exit_price
    # Charges carry more decimal places than money does, so the printed figure
    # is the exact one rounded for display — four places, two beyond a paisa.
    assert Decimal(row["round_trip_cost"]) == expected.cost.total.quantize(
        Decimal("0.0001")
    )


def test_each_block_names_the_target_and_the_margin_it_implies(
    capsys: CaptureFixture[str],
) -> None:
    """Both numbers together, because one cannot be read off the other.

    The target is a move from the buy price; the margin is what survives the
    round trip. Printing the margin alone would look like the strategy had been
    quietly restated downwards, and printing the target alone would hide that
    the charges have already been taken out of it.
    """
    _, output = _run(capsys, "2450")

    targets = [block["gross_target_fraction"] for block in output["estimates"]]
    margins = [block["net_margin_fraction"] for block in output["estimates"]]

    assert targets == ["0.00100000 (0.1000%)", "0.00200000 (0.2000%)"]
    assert margins == ["0.00017319 (0.0173%)", "0.00117319 (0.1173%)"]


def test_fractions_are_printed_beside_their_percentages(
    capsys: CaptureFixture[str],
) -> None:
    # A fraction and a percentage of the same quantity differ by a hundred, and
    # ``atr_pct`` is already a fraction despite its name. Printing one alone is
    # how that confusion reaches a decision.
    _, output = _run(capsys, "100")

    row = output["estimates"][0]["rows"][0]
    assert row["tick_fraction"] == "0.00050000 (0.0500%)"


def test_the_two_readings_of_the_stated_target_are_both_shown(
    capsys: CaptureFixture[str],
) -> None:
    _, output = _run(capsys)

    stated_two = output["interpretations_at_the_clip"][1]
    assert stated_two["stated"].startswith("0.00200000")
    # Keeping 0.2% after charges needs a 0.2827% move; moving 0.2% keeps only
    # 0.1173%. A third of the target rides on which one was meant.
    assert stated_two["as_net_kept_after_costs"]["gross_move_required"] == (
        "0.00282681 (0.2827%)"
    )
    assert stated_two["as_the_gross_move"]["net_kept"] == "0.00117319 (0.1173%)"


def test_the_cost_fraction_falls_as_the_clip_grows(
    capsys: CaptureFixture[str],
) -> None:
    _, output = _run(capsys)

    rows = output["round_trip_cost_by_notional"]
    fractions = [Decimal(row["round_trip_fraction"].split()[0]) for row in rows]

    # Flat while the twenty-rupee cap is slack, falling once it binds. This is
    # the whole reason the clip is stated before the strategy: the same trade
    # is a loser at twenty thousand and a winner at a lakh.
    assert fractions[0] == fractions[1]
    assert fractions[1] > fractions[2] > fractions[3] > fractions[4] > fractions[5]


# --- which broker's schedule --------------------------------------------------


def _compared(output: dict, notional: str) -> tuple[Decimal, Decimal]:
    """Both schedules' fractions at one row of the comparison block."""
    row = next(
        row
        for row in output["schedule_comparison"]
        if Decimal(row["notional"]) == Decimal(notional)
    )
    return Decimal(row["groww"].split()[0]), Decimal(row["zerodha"].split()[0])


def test_the_broker_flag_selects_the_other_published_schedule(
    capsys: CaptureFixture[str],
) -> None:
    _, groww = _run(capsys)
    _, zerodha = _run(capsys, "--broker", "zerodha")

    assert "Groww" in groww["schedule"]
    assert "Zerodha" in zerodha["schedule"]
    # Both keep the caveat: neither has been reconciled against a contract note.
    assert "unverified" in zerodha["schedule"]

    # And the selection reaches the arithmetic rather than only the label. The
    # cost table is priced per broker, and at a ten-thousand clip Groww's cap is
    # still slack while Zerodha charges a third of Groww's rate.
    assert (
        zerodha["round_trip_cost_by_notional"][0]["brokerage"]
        != groww["round_trip_cost_by_notional"][0]["brokerage"]
    )


def test_at_the_configured_clip_the_two_runs_are_the_same_run(
    capsys: CaptureFixture[str],
) -> None:
    """The headline, stated where someone comparing two runs would look.

    Every estimate is priced at the one-lakh clip and both brokerage caps bind
    well below it, so the exits, the outlays and the margins are identical to
    the paisa. That makes the broker swap in ``AGENTS.md`` rule 10 a
    configuration change rather than a change of strategy — and it is worth
    pinning, because a reader who saw only the differing cost table would
    reasonably conclude the opposite.
    """
    _, groww = _run(capsys)
    _, zerodha = _run(capsys, "--broker", "zerodha")

    assert zerodha["estimates"] == groww["estimates"]
    assert (
        zerodha["interpretations_at_the_clip"] == groww["interpretations_at_the_clip"]
    )


def test_kite_and_zerodha_name_the_same_schedule(
    capsys: CaptureFixture[str],
) -> None:
    # Rule 10 in ``AGENTS.md`` words the swap as Groww replaced by *Kite*, which
    # is the platform rather than the broker. Typing either has to work.
    _, by_broker = _run(capsys, "--broker", "zerodha")
    _, by_platform = _run(capsys, "--broker", "kite")

    assert by_broker == by_platform


def test_the_broker_flag_is_taken_out_before_the_prices_are_read(
    capsys: CaptureFixture[str],
) -> None:
    """The ordering bug this would otherwise have: a flag read as a quote.

    ``_parse_prices`` takes every remaining argument as a price, so a flag left
    in the list would be rejected for not being a decimal — a confusing way to
    be told about a typo, and an outright wrong answer for ``--broker=zerodha``
    if it ever parsed.
    """
    exit_code, output = _run(capsys, "--broker", "zerodha", "2450")

    assert exit_code == 0
    assert len(output["estimates"][0]["rows"]) == 1
    assert output["estimates"][0]["rows"][0]["quantity"] == 41


def test_the_flag_is_accepted_joined_to_its_value(
    capsys: CaptureFixture[str],
) -> None:
    _, spaced = _run(capsys, "--broker", "zerodha")
    _, joined = _run(capsys, "--broker=zerodha")

    assert spaced == joined


def test_both_schedules_are_printed_whichever_one_was_selected(
    capsys: CaptureFixture[str],
) -> None:
    """The comparison block, which is the only place the choice is visible.

    At the configured clip the two agree to the paisa — both caps bind — so a
    run that printed only the selected schedule would make the broker decision
    look consequential at a size where it is not, and inconsequential at the
    sizes where it is.
    """
    _, output = _run(capsys)

    groww_small, zerodha_small = _compared(output, "10000")
    groww_clip, zerodha_clip = _compared(output, "100000")

    assert zerodha_small < groww_small
    assert zerodha_clip == groww_clip


def test_the_comparison_does_not_change_with_the_broker_selected(
    capsys: CaptureFixture[str],
) -> None:
    _, groww = _run(capsys)
    _, zerodha = _run(capsys, "--broker", "zerodha")

    assert groww["schedule_comparison"] == zerodha["schedule_comparison"]


def test_an_unknown_broker_exits_one(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["--broker", "hdfc"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    # Naming the known values, because the alternative is reading the source.
    assert "groww" in captured.err and "zerodha" in captured.err


def test_a_broker_flag_without_a_name_exits_one(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["--broker"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "needs a name" in captured.err


def test_a_stock_quoted_above_the_clip_buys_one_share_and_says_by_how_much(
    capsys: CaptureFixture[str],
) -> None:
    # NSE lists names above a lakh a share. The clip is a floor on turnover, so
    # one share already clears it — there is nothing here to refuse, only an
    # overshoot to report to whatever eventually imposes a position limit.
    exit_code, output = _run(capsys, "150000")

    row = output["estimates"][0]["rows"][0]

    assert exit_code == 0
    assert row["quantity"] == 1
    assert Decimal(row["notional"]) == Decimal("150000")
    assert Decimal(row["excess_notional"]) == Decimal("50000")


def test_the_unmodelled_costs_are_named_in_the_output(
    capsys: CaptureFixture[str],
) -> None:
    # The exit price is a floor, not a forecast, and the output has to say so
    # where it is read rather than only in a docstring.
    _, output = _run(capsys)

    assert "spread" in output["not_modelled"]
    assert "unverified" in output["schedule"]


def test_an_unparseable_price_exits_one(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["not-a-price"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == "Prices must be decimal numbers."


def test_a_non_positive_price_exits_one(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["0"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "must be positive" in captured.err
