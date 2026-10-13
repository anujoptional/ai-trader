import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")

from scripts import research_scanner_filters as filters


def test_rules_and_filter_scores() -> None:
    feature = {
        "nifty_day": np.array([0.01, 0.01, -0.01, 0.01]),
        "day_return": np.array([0.02, 0.0, 0.03, 0.02]),
        "vwap_z": np.array([1.0, 1.0, 1.0, -2.0]),
        "realized_volatility_20m": np.array([0.005, 0.001, 0.005, 0.002]),
        "same_minute_rvol_5": np.array([2.0, 2.0, 2.0, 2.0]),
    }
    masks = filters.rule_masks(feature, low=0.0015, high=0.004)
    assert masks["aligned"].tolist() == [True, False, False, False]
    assert masks["aligned_high_movement_rvol_1_5"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert masks["low_movement"].tolist() == [False, True, False, False]
    assert masks["dip_in_up_market"].tolist() == [False, False, False, True]
    label = np.array([True, False, True, False])
    outcome = np.array([0.003, -0.002, 0.003, np.nan])
    result = filters.score(
        masks["high_movement"],
        label,
        label,
        np.ones(4, dtype=bool),
        outcome,
        np.array([1, 1, 2, 2]),
        2,
    )
    assert result["candidates"] == 2 and result["precision_percent"] == 100
    assert result["lift"] == pytest.approx(1.5)
    assert result["recall_percent"] == 100
    assert result["proxy_gross_percent"] == pytest.approx(0.3)
