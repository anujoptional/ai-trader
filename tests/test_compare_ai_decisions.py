from datetime import date, datetime

from ai_trader.ai.decision import CycleResult, Verdict
from ai_trader.ai.journal import DecisionJournal
from ai_trader.clock import INDIA_TIMEZONE
from scripts.compare_ai_decisions import compare_book, first_touch

FLAT = [[100.0, 100.0, 100.0, 100.0, 1.0]] * 10


def with_bar(minute: int, high: float, low: float) -> list[list[float | None]]:
    rows = [list(row) for row in FLAT]
    rows[minute] = [100.0, high, low, 100.0, 1.0]
    return rows


def test_first_touch_buys_the_next_open_and_counts_ties_as_losses() -> None:
    assert first_touch(with_bar(3, 101.5, 99.5), 1) == "up"
    assert first_touch(with_bar(3, 100.5, 98.9), 1) == "down"
    assert first_touch(with_bar(3, 101.5, 98.9), 1) == "down"
    assert first_touch(FLAT, 1) == "neither"
    gap = [list(row) for row in FLAT]
    gap[2] = [None] * 5
    assert first_touch(gap, 1) == "no_entry"


def test_compare_book_scores_each_verdict(tmp_path) -> None:
    journal = DecisionJournal(tmp_path / "decisions.sqlite")
    # Cycle 09:18 decides on bar 2 (09:17-09:18); the buy is at bar 3's open.
    verdicts = (
        Verdict("UP", "BUY", 0.7, "r", "s"),
        Verdict("DOWN", "NO_BUY", 0.2, "r", "s"),
        Verdict("FLAT", "NO_BUY", 0.82, "r", "s"),  # contradicts its own estimate
    )
    journal.record(
        "20261013-0918-1",
        datetime(2026, 10, 13, 9, 18, tzinfo=INDIA_TIMEZONE),
        "anthropic:claude-sonnet-5-5:low",
        CycleResult(verdicts, "ok", "claude-sonnet-5-5", cost_usd=0.04),
    )
    bars = {
        "UP": with_bar(5, 101.5, 99.5),
        "DOWN": with_bar(5, 100.5, 98.0),
        "FLAT": FLAT,
    }
    result = compare_book(
        tmp_path / "decisions.sqlite",
        tmp_path / "none.sqlite",
        date(2026, 10, 13),
        bars,
    )
    assert result["ai"] == "anthropic:claude-sonnet-5-5:low"
    assert result["calls"] == 1 and result["cost_usd"] == 0.04
    assert result["hindsight"] == {
        "BUY": {"up": 1},
        "NO_BUY": {"down": 1, "neither": 1},
    }
    assert result["brier"] == round(((0.7 - 1) ** 2 + 0.2**2) / 2, 4)
    assert result["disagree"] == 1
    assert "paper" not in result
