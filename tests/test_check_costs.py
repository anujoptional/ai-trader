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
    assert margins == ["0.00017555 (0.0176%)", "0.00117555 (0.1176%)"]


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
    # Keeping 0.2% after charges needs a 0.2824% move; moving 0.2% keeps only
    # 0.1176%. A third of the target rides on which one was meant.
    assert stated_two["as_net_kept_after_costs"]["gross_move_required"] == (
        "0.00282445 (0.2824%)"
    )
    assert stated_two["as_the_gross_move"]["net_kept"] == "0.00117555 (0.1176%)"


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
    assert fractions[1] > fractions[2] > fractions[3] > fractions[4]


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
