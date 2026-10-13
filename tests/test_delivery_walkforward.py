import numpy as np
import pytest

pytest.importorskip("catboost")
pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("sklearn")

from scripts import research_delivery_walkforward as walk


def test_months_and_block_interval() -> None:
    assert walk.months_between("2025-11", "2026-02") == [
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
    ]
    low, high = walk.block_interval(np.full(50, 0.25))
    assert low == pytest.approx(0.25) and high == pytest.approx(0.25)


def test_daily_top_picks_highest_scores_each_day() -> None:
    days = np.asarray(["d1", "d1", "d1", "d2", "d2"])
    result = walk.daily_top(
        days,
        np.asarray(["A", "B", "C", "A", "B"]),
        np.asarray([0.2, 0.9, 0.5, 0.7, 0.1]),
        np.asarray([0, 1, 0, 1, 1]),
        np.asarray([-0.01, 0.02, 0.0, 0.02, 0.02]),
        np.asarray([-0.01, 0.02, 0.0, 0.02, 0.02]),
        1,
    )
    assert result["picks"] == 2 and result["tp_first_percent"] == 100
    assert result["day_average_tp_first_percent"] == pytest.approx(
        100 * (1 / 3 + 1) / 2
    )


def test_within_day_auc_ignores_single_class_days() -> None:
    auc = walk.within_day_auc(
        np.asarray(["d1", "d1", "d2", "d2"]),
        np.asarray([0, 1, 1, 1]),
        np.asarray([0.1, 0.9, 0.3, 0.4]),
    )
    assert auc == {"d1": 1.0}
