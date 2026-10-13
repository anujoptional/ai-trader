import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")

from scripts.research_fno_scanner import (  # noqa: E402
    DECISIONS,
    LAST,
    NONE,
    session_features,
    session_outcomes,
    top_k,
)


def flat_session(price: float = 100.0) -> np.ndarray:
    prices = np.full((LAST + 1, 5), price)
    prices[:, 4] = 1000
    return prices


def test_outcomes_count_only_later_minutes_and_exact_levels() -> None:
    prices = flat_session()
    first = int(DECISIONS[0])
    prices[first + 10, 1] = 100.3
    prices[first + 70, 2] = 99.7
    prices[first + 80, 1] = 100.99
    outcome = session_outcomes(prices)
    assert outcome["up"][0] == 10
    assert outcome["down"][0] == 70
    assert outcome["big"][0] == NONE
    assert outcome["up"][2] == 70
    assert outcome["down"][2] == 60
    assert outcome["ret1500"][0] == pytest.approx(0.0)


def test_features_ignore_future_minutes() -> None:
    prices = flat_session()
    prices[200:, :4] = 200.0
    features = session_features(prices)
    slot = list(DECISIONS).index(199)
    assert features["ret_open"][slot] == 0.0
    assert features["rv20"][slot] == 0.0
    assert features["ret_open"][slot + 1] == pytest.approx(1.0)
    assert features["dist_high"][slot + 1] == 0.0


def test_top_k_keeps_only_available_best_scores() -> None:
    score = np.array([[[3.0, 2.0, 1.0]]])
    available = np.array([[[False, True, True]]])
    assert top_k(score, available, 1).tolist() == [[[False, True, False]]]
    assert top_k(score, available, 5).tolist() == [[[False, True, True]]]
