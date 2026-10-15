from datetime import date, datetime

from ai_trader.ai.decision import CycleResult, Verdict
from ai_trader.ai.journal import DecisionJournal
from ai_trader.clock import INDIA_TIMEZONE
from scripts.compare_ai_decisions import close_direction, compare_book

FLAT = [[100.0, 100.0, 100.0, 100.0, 1.0]] * 10


def closing_at(price: float) -> list[list[float | None]]:
    rows = [list(row) for row in FLAT]
    rows[-1] = [100.0, max(price, 100.0), min(price, 100.0), price, 1.0]
    return rows


def test_close_direction_compares_the_last_close_with_the_decision_close() -> None:
    assert close_direction(closing_at(100.5), 2) == "green"
    assert close_direction(closing_at(99.5), 2) == "red"
    assert close_direction(FLAT, 2) == "flat"
    gap = closing_at(101.0)
    gap[2] = [None] * 5
    assert close_direction(gap, 2) == "no_data"
    tail = closing_at(101.0)
    tail[-1] = [None] * 5  # the last minute missing: the latest close counts
    assert close_direction(tail, 2) == "flat"


def test_compare_book_scores_each_verdict(tmp_path) -> None:
    journal = DecisionJournal(tmp_path / "decisions.sqlite")
    # Cycle 09:18 decides on bar 2 (09:17-09:18), its close is the AI's last price.
    verdicts = (
        Verdict("UP", "LONG", 0.7, "r", "s"),
        Verdict("DOWN", "SHORT", 0.2, "r", "s"),
        Verdict("SAME", "SHORT", 0.82, "r", "s"),  # contradicts its own estimate
        Verdict("WAIT", "NO_TRADE", 0.5, "r", "s"),
    )
    journal.record(
        "20261014-0918-1",
        datetime(2026, 10, 14, 9, 18, tzinfo=INDIA_TIMEZONE),
        "anthropic:claude-sonnet-5-5:low",
        CycleResult(verdicts, "ok", "claude-sonnet-5-5", cost_usd=0.04),
    )
    bars = {
        "UP": closing_at(101.0),
        "DOWN": closing_at(99.0),
        "SAME": closing_at(100.5),
        "WAIT": FLAT,
    }
    result = compare_book(
        tmp_path / "decisions.sqlite",
        tmp_path / "none.sqlite",
        date(2026, 10, 14),
        bars,
    )
    assert result["ai"] == "anthropic:claude-sonnet-5-5:low"
    assert result["calls"] == 1 and result["cost_usd"] == 0.04
    assert result["verdicts"] == {"LONG": 1, "SHORT": 2, "NO_TRADE": 1}
    assert result["hindsight"] == {
        "LONG": {"green": 1},
        "SHORT": {"red": 1, "green": 1},
        "NO_TRADE": {"flat": 1},
    }
    expected = ((0.7 - 1) ** 2 + 0.2**2 + (0.82 - 1) ** 2) / 3
    assert result["brier"] == round(expected, 4) and result["resolved"] == 3
    assert result["disagree"] == 1
    assert "paper" not in result
