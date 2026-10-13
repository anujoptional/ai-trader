import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("sklearn")

from scripts import research_tp_scanner as scanner
from scripts.train_tp_only import EXPECTED_COLUMNS, file_hash


def ist(moment: str) -> int:
    return int(datetime.fromisoformat(moment + "+05:30").timestamp())


def test_call_window_is_0940_through_1330_inclusive() -> None:
    day = ist("2026-04-01T00:00:00")
    minutes = np.array([9 * 60 + 35, 9 * 60 + 40, 13 * 60 + 30, 13 * 60 + 35, 855])
    assert scanner.scanner_window(day + minutes * 60).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_band_report_counts_bands_15h_touches_and_high_score_misses() -> None:
    probability = np.array([0.0, 0.1, 0.75, 0.7499, 0.95, 1.0, 0.8, 0.2])
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype=np.float32)
    hits = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    returns = np.array([-0.001, 0.0, 0.0, 0.0, -0.012, 0.0, 0.0, 0.0005])
    timestamps = np.repeat(
        [ist("2026-04-01T10:00:00"), ist("2026-04-01T10:05:00")]
        + [ist("2026-04-02T10:00:00"), ist("2026-04-02T10:05:00")],
        2,
    )
    symbols = np.repeat(["A", "B", "A", "B"], 2)
    side = np.tile([1, -1], 4)
    charges = 0.0008
    report = scanner.scanner_report(
        probability, labels, hits, returns, timestamps, symbols, side, charges
    )
    bands = {row["band"]: row for row in report["bands"]}
    assert bands["0-10%"]["calls"] == 1
    assert bands["10-20%"]["calls"] == 1
    assert bands["70-80%"]["calls"] == 2
    assert bands["90-100%"]["calls"] == 2
    assert sum(row["calls"] for row in report["bands"]) == 8
    assert bands["90-100%"]["tp_within_60m_percent"] == 50
    assert bands["90-100%"]["tp_by_1500_percent"] == 50
    assert bands["90-100%"]["no_tp_mean_return_at_1500_percent"] == pytest.approx(-1.2)
    high = report["high_score"]
    assert high["calls"] == 4
    assert high["tp_within_60m_percent"] == 50
    assert high["tp_by_1500_percent"] == 75
    assert high["distinct_candle_times"] == 3
    assert high["candle_times_with_both_sides"] == 1
    misses = high["no_tp_by_1500_gross_return_at_1500"]
    assert misses["calls"] == 1
    assert misses["mean_percent"] == pytest.approx(-1.2)
    assert high["no_tp_by_1500_return_after_charges"]["mean_percent"] == pytest.approx(
        -1.28
    )
    gross = (3 * 0.002 - 0.012) / 4
    assert high["idealized_tp_or_1500_gross_mean_percent"] == pytest.approx(100 * gross)
    assert high["idealized_tp_or_1500_mean_after_charges_percent"] == pytest.approx(
        100 * (gross - charges)
    )
    assert high["break_even_tp_by_1500_percent_gross"] == pytest.approx(
        100 * 0.012 / 0.014
    )
    labels[4] = 1
    with pytest.raises(ValueError, match="must count by 15:00"):
        scanner.scanner_report(
            probability, labels, hits, returns, timestamps, symbols, side, charges
        )


def test_recency_weights_halve_each_half_life_and_average_one() -> None:
    timestamps = np.array([ist("2025-01-01T10:00:00"), ist("2026-01-01T10:00:00")])
    weights = scanner.recency_weights(timestamps, 365)
    assert weights.mean() == pytest.approx(1)
    assert weights[1] / weights[0] == pytest.approx(2)
    assert scanner.recency_weights(timestamps, None) is None


def test_selection_prefers_lower_loss_then_uniform_then_shallower() -> None:
    def trial(loss: float, depth: int, half_life: int | None) -> dict:
        return {
            "validation": {"log_loss": loss},
            "candidate": {"depth": depth, "half_life_days": half_life},
            "tree_count": 10,
        }

    reports = {
        "deep": trial(0.5, 6, None),
        "recent": trial(0.5, 4, 365),
        "uniform": trial(0.5, 4, None),
    }
    assert scanner.select_candidate(reports) == "uniform"
    reports["recent"]["validation"]["log_loss"] = 0.4
    assert scanner.select_candidate(reports) == "recent"


def test_feature_audit_exposes_a_derivable_feature() -> None:
    generator = np.random.default_rng(5)
    features = generator.normal(size=(400, len(EXPECTED_COLUMNS)))
    features[:, 2] = features[:, 0] + features[:, 1]
    features[:, -1] = np.tile([1, -1], 200)
    audit = scanner.feature_audit(features)["features"]
    assert audit["day_return"]["linear_r2_from_other_features"] > 0.999
    assert audit["vix_level"]["linear_r2_from_other_features"] < 0.2


def write_shard(data: Path, phase: str, moments: list[int], seed: int) -> dict:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(len(moments), len(EXPECTED_COLUMNS)))
    values = np.repeat(values.astype(np.float32), 2, axis=0)
    values[:, -1] = np.tile([1, -1], len(moments))
    labels = (values[:, 0] + 0.3 * values[:, -1] > 0).astype(np.float32)
    path = data / f"{phase}.npz"
    np.savez_compressed(
        path,
        features=values,
        labels=labels,
        timestamps=np.repeat(np.asarray(moments, dtype=np.int64), 2),
    )
    return {
        "phase": phase,
        "symbol": "TEST",
        "path": path.name,
        "rows": 2 * len(moments),
        "sha256": file_hash(path),
    }


def session_times(day: str) -> list[int]:
    times = ["10:00", "10:30", "11:00", "11:30", "12:00", "13:30", "13:35", "14:15"]
    return [ist(f"{day}T{clock}:00") for clock in times]


@pytest.fixture
def scanner_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    pytest.importorskip("catboost")
    from scripts import research_tp_catboost as study
    from scripts import train_tp_only as training

    data = tmp_path / "data"
    data.mkdir()
    phases = {
        "train": ("2025-03-03", "2025-03-04", "2025-06-02", "2025-06-03"),
        "validation": ("2025-07-01", "2025-07-02", "2025-12-01"),
        "calibration": ("2026-01-05", "2026-02-02", "2026-03-02", "2026-03-03"),
        "evaluation": ("2026-04-01", "2026-05-04", "2026-09-01"),
    }
    manifest = {
        "schema_version": 1,
        "objective": "directional_price_touch_without_stop_loss",
        "stock_identity_input": False,
        "model_columns": list(EXPECTED_COLUMNS),
        "tp_fraction": "0.002",
        "horizon_minutes": 60,
        "calendar": ["2025-01-01"],
        "phase_ends": {
            "train": "2025-06-30",
            "validation": "2025-12-31",
            "calibration": "2026-03-31",
            "evaluation": "2026-09-28",
        },
        "shards": [
            write_shard(
                data,
                phase,
                [moment for day in days for moment in session_times(day)],
                index,
            )
            for index, (phase, days) in enumerate(phases.items())
        ],
        "limitations": ["Synthetic scanner fixture, not market evidence"],
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = training.TrainConfig(
        model="catboost", tree_depth=2, tree_iterations=8, threads=1
    )
    training.run_training(data, tmp_path / "baseline", config, validation_only=True)
    monkeypatch.setattr(study, "variant_configurations", lambda: {"probe": config})
    study.run_study(data, tmp_path / "baseline", tmp_path / "study")
    current = tmp_path / "study" / "final"
    evaluation = training.load_phase(data, manifest, "evaluation")
    with np.load(current / "evaluation_predictions.npz") as saved:
        raw = saved["raw_probability"]
    generator = np.random.default_rng(9)
    outcomes = tmp_path / "outcomes"
    outcomes.mkdir()
    np.savez_compressed(
        outcomes / "returns.npz",
        timestamps=evaluation["timestamps"],
        symbols=evaluation["symbols"],
        side=evaluation["features"][:, -1],
        labels=evaluation["labels"],
        raw_probability=raw,
        hit_by_1500=np.maximum(
            evaluation["labels"], generator.integers(0, 2, len(raw))
        ).astype(np.int8),
        return_at_1500=generator.normal(scale=0.005, size=len(raw)),
    )
    (outcomes / "report.json").write_text(
        json.dumps({"returns_sha256": file_hash(outcomes / "returns.npz")}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scanner,
        "candidates",
        lambda: {
            "tiny": scanner.Candidate(depth=2, iterations=8, patience=5),
            "tiny_recent": scanner.Candidate(
                depth=2, half_life_days=30, iterations=8, patience=5
            ),
        },
    )
    return data, current, outcomes, evaluation


def test_scanner_study_locks_selection_before_opening_window_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scanner_inputs: tuple
) -> None:
    data, current, outcomes, evaluation = scanner_inputs
    output = tmp_path / "scanner"
    opened, fitted = [], []
    original_load = scanner.load_phase
    original_fit = scanner.fit_candidate

    def checked_load(root: Path, document: dict, phase: str) -> dict:
        assert (output / "protocol.json").is_file()
        if phase == "evaluation":
            assert (output / "final" / "selection_lock.json").is_file()
        else:
            assert "evaluation" not in opened
        opened.append(phase)
        return original_load(root, document, phase)

    def checked_fit(*args, **kwargs) -> dict:
        assert (output / "feature_audit.json").is_file()
        assert not (output / "final").exists()
        fitted.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(scanner, "load_phase", checked_load)
    monkeypatch.setattr(scanner, "fit_candidate", checked_fit)
    report = scanner.run_study(data, current, outcomes, output)
    assert opened == ["train", "validation", "calibration", "evaluation"]
    assert len(fitted) == 2
    window = scanner.scanner_window(evaluation["timestamps"])
    assert report["cohort"]["rows"] == window.sum() == 36
    assert report["cohort"]["excluded_outside_window"] == 12
    assert report["evaluation_used_for_selection"] is False
    assert isinstance(report["fresh_preferred_by_predeclared_rule"], bool)
    assert "decision_rule" in json.loads((output / "protocol.json").read_text())
    lock = json.loads((output / "final" / "selection_lock.json").read_text())
    assert lock["selected"] == report["fresh_selected"]
    assert lock["weights_sha256"] == file_hash(output / "final" / "best.cbm")
    for values in report["models"].values():
        assert sum(row["calls"] for row in values["bands"]) == 36
    with np.load(output / "evaluation_predictions.npz") as saved:
        assert scanner.scanner_window(saved["timestamps"]).all()
        np.testing.assert_array_equal(saved["labels"], evaluation["labels"][window])
    with pytest.raises(ValueError, match="already exists"):
        scanner.run_study(data, current, outcomes, output)
    with (outcomes / "returns.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="do not match"):
        scanner.run_study(data, current, outcomes, tmp_path / "tampered")
