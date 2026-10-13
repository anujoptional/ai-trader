import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("scipy")

from scripts import research_breakout_threshold as study  # noqa: E402


def write_minutes(path, days: dict[str, int]) -> None:
    """``days``: date -> minutes 09:15.. written (345 = complete to 14:59)."""
    lines = ["start_time,open,high,low,close,volume"]
    for day, count in days.items():
        for minute in range(count):
            hour, rest = divmod(555 + minute, 60)
            lines.append(f"{day}T{hour:02d}:{rest:02d}:00+05:30,100,101,99,100.5,10")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_only_complete_unflagged_days_inside_the_period_are_read(tmp_path) -> None:
    path = tmp_path / "NSE_ABC.csv"
    write_minutes(
        path,
        {"2023-12-29": 345, "2024-01-02": 345, "2024-01-03": 344, "2024-01-04": 375},
    )
    days, prices, present = study.read_days(
        path, frozenset({"2024-01-04"}), "2024-01-01", "2024-12-31"
    )
    assert days.tolist() == ["2024-01-02"] and present == 3
    assert prices.shape == (1, 345, 4) and prices[0, 0].tolist() == [
        100,
        101,
        99,
        100.5,
    ]


def results(symbol: str, move: list[float], trigger: list[list[int]], exits: list[int]):
    count = len(move)
    gross = np.zeros((count, 2, 2))
    gross[:, 0, 0], gross[:, 1, 0] = 0.01, -0.02  # long wins 1%, short loses 2%
    exit_ = np.zeros((count, 2, 2), dtype=np.int16)
    exit_[:, :, 0] = np.asarray(exits)[:, None]
    return {
        "symbol": symbol,
        "days": np.asarray(["2024-01-02", "2024-01-03"][:count]),
        "present": count,
        "move": np.asarray(move),
        "trigger": np.asarray(trigger, dtype=np.int16),
        "gross": gross,
        "exit": exit_,
        "control": np.zeros((count, study.REPEATS), dtype=np.float32),
    }


def test_side_entry_window_threshold_and_open_positions() -> None:
    direction = {"2024-01-02": 1, "2024-01-03": -1}
    table = study.build_table(
        [
            results("AAA", [0.02, 0.02], [[30, -1], [-1, 40]], [100, 344]),
            results("BBB", [0.004, 0.03], [[60, -1], [-1, 135]], [200, 344]),
        ],
        direction,
    )
    rows = study.chosen_rows(table, 0.0)
    picked = [(table["day"][r], table["symbol"][r], table["side"][r]) for r in rows]
    assert picked == [
        ("2024-01-02", "AAA", 0),
        ("2024-01-02", "BBB", 0),
        ("2024-01-03", "AAA", 1),  # BBB broke out after 11:30 that day
    ]
    assert len(study.chosen_rows(table, 0.01)) == 2  # BBB moved 0.4% by 09:40
    assert len(study.chosen_rows(table, 0.0, (0,))) == 2
    sessions = np.asarray(sorted(direction))
    summary = study.summarize(table, rows, sessions)
    net_long = 0.01 - study.CHARGES - study.base.SLIPPAGE
    net_short = -0.02 - study.CHARGES - study.base.SLIPPAGE
    assert summary["trades"] == 3 and summary["max_trades_in_a_day"] == 2
    assert summary["net_inr_per_day"] == pytest.approx(
        study.TRADE_INR * (2 * net_long + net_short) / 2
    )
    assert summary["win_rate_percent"] == pytest.approx(100 * 2 / 3)
    assert summary["max_open_positions"] == 2  # AAA 09:45-10:55 and BBB 10:15-12:35


def test_choice_needs_every_half_positive() -> None:
    def entry(threshold: float, per_day: float, halves: list[float]) -> dict:
        return {
            "threshold_percent": threshold,
            "both": {"2024-2025": {"net_inr_per_day": per_day}},
            "tune_halves_net_inr_per_day": halves,
        }

    curve = [entry(0.5, 900, [1, -1, 5, 5]), entry(1.0, 400, [1, 2, 3, 4])]
    assert study.choose(curve) == (0.01, True)
    curve = [entry(0.5, 900, [1, -1, 5, 5]), entry(1.0, 400, [1, -2, 3, 4])]
    assert study.choose(curve) == (0.005, False)
