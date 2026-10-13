from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")

from scripts import research_exit_policy as policy

BASE = 100_000


def flat_session() -> np.ndarray:
    return np.full((policy.DEADLINE + 1, 4), BASE, dtype=np.int64)


def scenario_paths() -> tuple[np.ndarray, np.ndarray]:
    first = flat_session()
    first[30, 1] = 100_200
    first[40, [0, 2]] = 99_600
    first[54, [1, 3]] = 100_100
    tie = flat_session()
    tie[30, 1], tie[30, 2] = 100_200, 99_800
    return first, tie


def universe_from(*sessions: np.ndarray) -> policy.Universe:
    parts = [policy.session_primitives(prices) for prices in sessions]
    count = len(sessions)
    columns = len(policy.EXPECTED_COLUMNS)
    return policy.Universe(
        days=np.arange(count, dtype=np.int64) + 20000,
        symbols=np.asarray(["TEST"] * count),
        features=np.zeros((count, policy.SLOTS, 2, columns), dtype=np.float32),
        available=np.ones((count, policy.SLOTS), dtype=bool),
        labels=np.zeros((count, policy.SLOTS, 2), dtype=np.int8),
        **{name: np.stack([part[name] for part in parts]) for name in parts[0]},
    )


def level(bps: int) -> int:
    return policy.LEVELS.index(bps)


def test_primitives_record_first_touches_gap_opens_and_time_exits() -> None:
    first, _ = scenario_paths()
    values = policy.session_primitives(first)
    assert values["up_minutes"][0, level(20)] == 6
    assert values["up_minutes"][0, level(30)] == policy.NONE
    assert values["down_minutes"][0, [level(20), level(30), level(40)]].tolist() == [
        16,
        16,
        16,
    ]
    assert values["down_minutes"][0, level(50)] == policy.NONE
    assert values["down_open"][0, level(30)] == pytest.approx(-0.004)
    assert values["time_minutes"][0].tolist() == [15, 30, 60, 120, 320]
    assert values["time_return"][0, policy.TH_MINUTES.index(30)] == pytest.approx(0.001)
    assert values["time_minutes"][policy.SLOTS - 1, -1] == policy.DEADLINE - 254


def test_exit_rules_take_profit_gap_stop_time_exit_and_stop_first_ties() -> None:
    universe = universe_from(*scenario_paths())
    rows = slice(0, 2)

    def outcome(config: tuple, session: int, side: int) -> tuple:
        returns, exits, kinds = policy.config_outcome(universe, rows, *config)
        return (
            float(returns[session, 0, side]),
            int(exits[session, 0, side]),
            int(kinds[session, 0, side]),
        )

    assert outcome((20, 30, 60), 0, 0) == (pytest.approx(0.002), 6, 0)
    assert outcome((None, 30, 60), 0, 0) == (pytest.approx(-0.004), 16, 1)
    assert outcome((None, None, 30), 0, 0) == (pytest.approx(0.001), 30, 2)
    assert outcome((30, 20, 60), 0, 1) == (pytest.approx(-0.002), 6, 1)
    assert outcome((30, None, 15), 0, 1) == (pytest.approx(0.0), 15, 2)
    assert outcome((20, 20, None), 1, 0) == (pytest.approx(-0.002), 6, 1)
    assert outcome((20, None, None), 1, 0) == (pytest.approx(0.002), 6, 0)


def test_greedy_simulation_keeps_one_position_per_stock() -> None:
    eligible = np.zeros((2, policy.SLOTS), dtype=bool)
    eligible[0, :4] = True
    eligible[1, [1, 46]] = True
    exits = np.full((2, policy.SLOTS), 300)
    exits[0, 0], exits[0, 2] = 7, 5
    exits[1, 1] = 3
    taken = policy.simulate(eligible, exits)
    assert np.flatnonzero(taken[0]).tolist() == [0, 2, 3]
    assert np.flatnonzero(taken[1]).tolist() == [1, 46]


def test_summary_counts_daily_quarterly_net_and_exit_kinds() -> None:
    days = policy.Calendar(
        day_ids=np.array([0, 0, 1]),
        sessions=2,
        quarter_of_day=np.array([0, 1]),
        month_of_day=np.array([0, 1]),
        months=("2025-03", "2025-04"),
    )
    taken = np.zeros((3, policy.SLOTS), dtype=bool)
    taken[0, 0] = taken[1, 0] = True
    taken[2, [0, 5]] = True
    returns = np.zeros((3, policy.SLOTS))
    returns[0, 0], returns[1, 0], returns[2, 0] = 0.004, -0.002, 0.003
    kinds = np.full((3, policy.SLOTS), 2, dtype=np.int8)
    kinds[0, 0], kinds[1, 0] = 0, 1
    exits = np.full((3, policy.SLOTS), 10)
    summary, daily_gross, daily_net = policy.summarize(
        taken, returns, exits, kinds, days, 0.001
    )
    assert daily_gross.tolist() == pytest.approx([0.002, 0.003])
    assert daily_net.tolist() == pytest.approx([0.0, 0.001])
    assert summary["mean_daily_gross_percent"] == pytest.approx(0.25)
    assert summary["quarter_net_percent"] == pytest.approx([0.0, 0.1])
    assert summary["trades_per_session"] == 2
    assert summary["tp_exit_percent"] == 25
    assert summary["sl_exit_percent"] == 25
    assert summary["mean_trade_gross_percent"] == pytest.approx(0.125)


def test_threshold_rules_come_from_stop_slice_quantiles() -> None:
    best = np.arange(100, dtype=float).reshape(1, 100)
    rules = policy.cutoffs("regressor", best, np.ones_like(best, dtype=bool))
    assert rules["top100"] == -np.inf
    assert rules["top10"] == pytest.approx(89.1)
    assert rules["positive"] == 0
    assert "positive" not in policy.cutoffs("classifier", best, best >= 0)


def write_day(handle, day: str, last_minute: int) -> None:
    for minute in range(last_minute + 1):
        clock = policy.SESSION_OPEN + minute
        handle.write(
            f"{day}T{clock // 60:02d}:{clock % 60:02d}:00+05:30,"
            "1000.05,1000.10,1000.00,1000.05,10\n"
        )


def test_paths_need_only_minutes_through_1500(tmp_path: Path) -> None:
    path = tmp_path / "NSE_TEST.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(policy.HEADER) + "\n")
        write_day(handle, "2026-08-04", 359)
        write_day(handle, "2026-08-05", 343)
        write_day(handle, "2026-08-06", 374)
    sessions = policy.read_paths(path, {"2026-08-06"})
    assert list(sessions) == [policy.epoch_day("2026-08-04")]
    assert sessions[policy.epoch_day("2026-08-04")].shape == (policy.DEADLINE + 1, 4)
