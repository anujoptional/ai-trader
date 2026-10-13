import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("scipy")

from scripts import research_intraday_strategy as intraday


def session() -> np.ndarray:
    prices = np.full((1, intraday.DEADLINE + 1, 4), 10_000, dtype=np.int64)
    prices[0, 5, 1] = 10_100
    prices[0, 10, 2] = 9_900
    return prices


def test_breakout_triggers_fills_and_stops() -> None:
    prices = session()
    prices[0, 50] = (10_050, 10_150, 10_050, 10_100)
    prices[0, 100] = (9_950, 9_950, 9_890, 9_900)
    prices[0, 101:] = 10_200
    gross, exits, trigger = intraday.breakout_outcomes(prices, 24)
    assert trigger[0].tolist() == [50, 100]
    assert gross[0, 0, 0, 0] == pytest.approx(9_900 / 10_100 - 1)
    assert exits[0, 0, 0, 0] == 100
    assert gross[0, 0, 1, 0] == pytest.approx(10_000 / 10_100 - 1)
    assert exits[0, 0, 1, 0] == 51
    assert gross[0, 0, 2, 0] == pytest.approx(10_200 / 10_100 - 1)
    assert exits[0, 0, 2, 0] == intraday.DEADLINE
    assert gross[0, 1, 0, 0] == pytest.approx(1 - 10_200 / 9_900)
    assert gross[0, 0, 2, 1] == pytest.approx(10_200 / 10_100 - 1)


def test_oco_and_market_modes_pick_sides_and_skip_ambiguity() -> None:
    gross = np.zeros((4, 2, 2, 3, 2))
    gross[..., 2, 0] = np.arange(4)[:, None, None] * 10 + np.arange(2)
    trigger = np.zeros((4, 2, 2), dtype=np.int16)
    trigger[:, 0] = [[50, 100], [80, 60], [70, 70], [200, -1]]
    table = {
        "trigger": trigger,
        "market": np.array([[1, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 0]]),
        "exit_a": np.full(gross.shape, intraday.DEADLINE, dtype=np.int16),
        "gross_a": gross,
    }
    rows, entry, _, values = intraday.breakout_trades(table, 0, 1, 2, "oco")
    assert rows.tolist() == [0, 1, 3]
    assert entry.tolist() == [50, 60, 200]
    assert values.tolist() == [0, 11, 30]
    rows, _, _, values = intraday.breakout_trades(table, 0, 1, 2, "market")
    assert values.tolist() == [0, 10, 21, 30]
    rows, _, _, _ = intraday.breakout_trades(table, 0, 0, 2, "oco")
    assert rows.tolist() == [0, 1]


def test_capacity_takes_first_triggers_and_reuses_stopped_slots() -> None:
    accepted = intraday.capacity(
        np.ones(4, dtype=np.int64),
        np.arange(4),
        np.array([30, 30, 40, 50]),
        np.array([344, 35, 344, 344]),
        2,
    )
    assert accepted.tolist() == [True, True, True, False]


def test_afternoon_follow_and_fade_with_stop() -> None:
    prices = np.full((1, intraday.DEADLINE + 1, 4), 10_000, dtype=np.int64)
    prices[0, intraday.AFTERNOON :] = 10_100
    prices[0, 300] = (10_120, 10_200, 10_110, 10_180)
    prices[0, intraday.DEADLINE] = 10_300
    gross, exits = intraday.afternoon_outcomes(prices)
    assert gross[0, 0, 0] == pytest.approx(10_300 / 10_100 - 1)
    assert gross[0, 0, 1] == pytest.approx(10_300 / 10_100 - 1)
    assert gross[0, 1, 0] == pytest.approx(1 - 10_300 / 10_100)
    assert gross[0, 1, 1] == pytest.approx(-0.005)
    assert exits[0, 1, 1] == 300
