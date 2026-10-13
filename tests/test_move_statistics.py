from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_move_statistics as moves


def flat(price: int = 100_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.full(moves.SESSION_MINUTES, price, dtype=np.int64) for _ in "hlc")


def test_sequential_moves_reset_to_the_touching_close_and_count_exact_touches():
    high, low, close = flat()
    high[5] = close[5] = 100_200
    high[6:9] = low[6:9] = close[6:9] = 100_200
    high[9], low[9], close[9] = 100_200, 99_999, 100_000
    assert moves.sequential_moves(high, low, close, 2) == [(0, 5, 1), (5, 9, -1)]
    assert moves.sequential_moves(high, low, close, 4) == []
    high, low, close = flat()
    high[3], low[3] = 100_300, 99_700
    assert moves.sequential_moves(high, low, close, 2) == [(0, 3, 0)]


def test_call_outcomes_measure_first_touch_adverse_excursion_and_deadline():
    high, low, close = flat()
    low[30] = 99_900
    high[40] = 100_200
    high[moves.DEADLINE + 1] = 101_000
    outcomes = moves.call_outcomes(high, low, close, (2,))
    long, short = outcomes[(2, 1)], outcomes[(2, -1)]
    first, at_39, at_40 = 0, 39 - moves.FIRST_CALL, 40 - moves.FIRST_CALL
    assert long["minutes"][first] == 16
    assert long["adverse"][first] == 0.001
    assert long["minutes"][at_39] == 1
    assert long["adverse"][at_39] == 0
    assert (long["minutes"][at_40:] == -1).all()
    assert (short["minutes"] == -1).all()
    assert (outcomes["long_return_at_1500"] == 0).all()
    assert len(long["minutes"]) == moves.LAST_CALL - moves.FIRST_CALL + 1


def write_day(handle, day: str, minutes: range, price: str = "1000.05") -> None:
    for minute in minutes:
        clock = 555 + minute
        handle.write(
            f"{day}T{clock // 60:02d}:{clock % 60:02d}:00+05:30,"
            f"{price},{price},{price},{price},10\n"
        )


def test_sessions_keep_only_complete_unquarantined_training_days(tmp_path: Path):
    path = tmp_path / "NSE_TEST.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(moves.HEADER) + "\n")
        write_day(handle, "2024-01-01", range(375))
        write_day(handle, "2024-01-02", range(374))
        write_day(handle, "2024-01-03", range(375))
        write_day(handle, "2024-01-04", range(-5, 380))
        write_day(handle, "2026-01-05", range(375))
    sessions, excluded = moves.read_sessions(path, {"2024-01-03"})
    assert sorted(sessions) == ["2024-01-01", "2024-01-04"]
    assert excluded == {"quarantined": 1, "incomplete_or_special": 1}
    assert sessions["2024-01-01"].shape == (375, 3)
    assert (sessions["2024-01-01"] == 100_005).all()


def test_histogram_quantiles_use_bin_values() -> None:
    summary = moves.quantiles(np.array([0, 2, 1, 0, 1]))
    assert summary["count"] == 4
    assert summary["mean"] == 2.0
    assert summary["median"] == 1
    assert summary["p90"] == 4


def test_call_summary_rates_and_cost_break_even() -> None:
    minutes = np.zeros(moves.DEADLINE + 1, dtype=np.int64)
    minutes[[10, 90]] = 1
    summary = moves.call_summary(minutes, np.array([2]), -0.012, 2, -0.008, 4, 2, 0.001)
    assert summary["tp_by_1500_percent"] == 50
    assert summary["tp_within_60m_percent"] == 25
    assert summary["no_touch_mean_return_at_1500_percent"] == pytest.approx(-0.6)
    assert summary["tp_or_1500_gross_mean_percent"] == pytest.approx(-0.2)
    assert summary["break_even_tp_by_1500_percent_after_charges"] == pytest.approx(87.5)
