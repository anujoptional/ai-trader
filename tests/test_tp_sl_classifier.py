import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("sklearn")

from scripts import research_tp_sl_classifier as classifier


def test_bands_and_break_even_thresholds_use_realized_returns() -> None:
    probability = np.array([0.05, 0.15, 0.25, 0.35, 0.95, 1.0])
    labels = np.array([0, 0, 1, 0, 1, 1])
    gains = np.array([-0.002, -0.002, 0.006, -0.002, 0.006, 0.006])
    bands = {
        row["band"]: row
        for row in classifier.band_table(probability, labels, gains, 0.001)
    }
    assert [bands[name]["calls"] for name in ("0-10%", "20-30%", "90-100%")] == [
        1,
        1,
        2,
    ]
    assert bands["20-30%"]["tp_first_percent"] == 100
    assert bands["20-30%"]["mean_net_return_percent"] == pytest.approx(0.5)
    assert bands["50-60%"] == {"band": "50-60%", "calls": 0}
    rules = {
        row["rule"]: row
        for row in classifier.threshold_table(probability, labels, gains, 0.001, 2)
    }
    assert rules["gross_break_even"]["min_probability"] == pytest.approx(0.25)
    assert rules["gross_break_even"]["calls"] == 4
    assert rules["net_break_even"]["min_probability"] == pytest.approx(0.375)
    assert rules["net_break_even"]["calls_per_session"] == 1
    assert rules["net_break_even"]["mean_gross_return_percent"] == pytest.approx(0.6)
