import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("scipy")

from scripts import research_orb_strategy as orb


def session() -> np.ndarray:
    prices = np.full((1, orb.DEADLINE + 1, 4), 10_000, dtype=np.int64)
    prices[0, 5, 1] = 10_100
    prices[0, 10, 2] = 9_900
    return prices


def test_entries_stops_and_fills_follow_candle_order() -> None:
    prices = session()
    prices[0, 50] = (10_050, 10_150, 10_050, 10_100)
    prices[0, 100] = (9_950, 9_950, 9_890, 9_900)
    prices[0, 101:] = 10_200
    gross, stopped = orb.session_outcomes(prices)
    expected = {
        (0, 0, 0): -0.01,
        (0, 0, 1): -0.005,
        (0, 0, 2): 0.02,
        (0, 1, 0): 9_900 / 10_100 - 1,
        (0, 1, 2): 10_200 / 10_100 - 1,
        (1, 0, 0): -0.01,
        (1, 0, 2): -0.02,
        (1, 1, 0): 1 - 10_200 / 9_900,
    }
    for key, value in expected.items():
        assert gross[(0, *key)] == pytest.approx(value)
    assert stopped[0, 0, 0, 0] and stopped[0, 1, 1, 0]
    assert not stopped[0, 0, 0, 2]


def test_breakout_candle_reaching_the_stop_fills_at_the_stop() -> None:
    prices = session()
    prices[0, 30] = (10_000, 10_200, 10_000, 10_100)
    gross, stopped = orb.session_outcomes(prices)
    assert gross[0, 0, 1, 1] == pytest.approx(-0.005)
    assert stopped[0, 0, 1, 1]


def test_relative_volume_needs_three_prior_sessions() -> None:
    result = orb.relative_volume(np.array([10.0, 10.0, 10.0, 20.0, 5.0]))
    assert np.isnan(result[:3]).all()
    assert result[3:] == pytest.approx([2.0, 0.4])


def test_daily_ranking_and_direction_choose_the_expected_sides() -> None:
    gross = np.zeros((5, 2, 2, 3))
    gross[..., 0, 2] = np.arange(5)[:, None] * 10 + np.arange(2)
    table = {
        "day": np.array([1, 1, 1, 2, 2]),
        "rvol": np.array([3.0, 1.0, 2.0, 1.0, np.nan]),
        "or_return": np.array([0.01, -0.02, -0.005, 0.01, 0.02]),
        "gap": np.zeros(5),
        "eligible": np.array([True, True, True, True, False]),
        "gross": gross,
        "stopped": np.zeros(gross.shape, dtype=bool),
    }
    rank, signal = orb.rank_rows(table, "rvol")
    assert rank[:4].tolist() == [0, 2, 1, 0]
    rows, values, _ = orb.variant_trades(table, rank, signal, 2, "follow", 0, 2)
    assert rows.tolist() == [0, 2, 3]
    assert values.tolist() == [0, 21, 30]
    _, faded, _ = orb.variant_trades(table, rank, signal, 2, "fade", 0, 2)
    assert faded.tolist() == [1, 20, 31]
