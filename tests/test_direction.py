import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")

from scripts.research_direction import (  # noqa: E402
    DECISIONS,
    EXIT,
    forward,
    path_features,
    prior_daily,
    ranks,
)


def session(price: float = 100.0) -> np.ndarray:
    prices = np.full((EXIT + 1, 5), price)
    prices[:, 4] = 1000.0
    return prices


def test_label_runs_from_next_minute_open_to_the_1500_close() -> None:
    prices = session()
    first = int(DECISIONS[0])
    prices[first + 1, 0] = 99.0
    prices[EXIT, 3] = 101.0
    labels = forward(prices)
    assert labels[0] == pytest.approx(np.log(101 / 99))
    assert labels[1] == pytest.approx(np.log(101 / 100))


def test_features_ignore_minutes_after_the_decision() -> None:
    prices = session()
    slot = 3
    base = path_features(prices, volume=True)
    later = prices.copy()
    later[DECISIONS[slot] + 1 :, :4] = 150.0
    later[DECISIONS[slot] + 1 :, 4] = 9e9
    moved = path_features(later, volume=True)
    for name, values in base.items():
        assert values[slot] == pytest.approx(moved[name][slot], nan_ok=True), name


def test_prior_daily_requires_the_previous_session() -> None:
    days = np.arange(100, 170)
    values = np.column_stack(
        (np.full(70, 10.0), np.full(70, 11.0), np.full(70, 9.0), np.full(70, 10.0))
        + (np.full(70, 5.0),)
    )
    market = {int(day): 100.0 + day for day in days}
    assert prior_daily(days, values, 170, 169, market)["d_atr"] == pytest.approx(0.2)
    assert prior_daily(days, values, 171, 170, market) is None


def test_ranks_are_percentiles_among_available_stocks() -> None:
    values = np.array([[[3.0, 1.0, 2.0, 9.0]]])
    available = np.array([[[True, True, True, False]]])
    result = ranks(values, available)
    assert result[0, 0, :3].tolist() == [1.0, 0.0, 0.5]
    assert np.isnan(result[0, 0, 3])
