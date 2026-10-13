import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")

from ai_trader.features.tp_only import MODEL_COLUMNS
from scripts.train_tp_only import (
    EXPECTED_COLUMNS,
    TrainConfig,
    build_model,
    fit_calibration,
    fit_model,
    fit_preprocessing,
    load_phase,
    predict_logits,
    transform,
)


def test_training_schema_has_no_symbol_or_target_leakage() -> None:
    assert EXPECTED_COLUMNS == MODEL_COLUMNS
    model = build_model(len(MODEL_COLUMNS), TrainConfig())
    assert (
        500_000 < sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
    )
    assert "symbol" not in EXPECTED_COLUMNS
    assert "label" not in EXPECTED_COLUMNS


def test_preprocessing_is_fit_only_on_training_and_preserves_direction() -> None:
    train = np.asarray([[0, 1], [1, -1], [2, 1], [3, -1]], dtype=np.float32)
    fitted = fit_preprocessing(train)
    original = json.dumps(fitted)
    later = transform(np.asarray([[1000000, -1]], dtype=np.float32), fitted)
    assert json.dumps(fitted) == original
    assert later[0, -1] == -1
    assert np.isfinite(later).all()
    assert later[0, 0] < 2


@pytest.mark.parametrize("model_name", ["ft_transformer", "resnet", "tabm", "catboost"])
def test_tiny_training_writes_reloadable_finite_checkpoint(
    tmp_path: Path, model_name: str
) -> None:
    if model_name in ("tabm", "catboost"):
        pytest.importorskip(model_name)
    generator = np.random.default_rng(12)
    features = generator.normal(size=(48, 4)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.float32)
    config = TrainConfig(
        model=model_name,
        n_blocks=1,
        d_block=32,
        batch_size=16,
        epochs=2,
        threads=1,
        ensemble_size=4,
        tree_iterations=8,
        tree_depth=3,
    )
    model, summary = fit_model(
        features[:33], labels[:33], features[33:], labels[33:], tmp_path, config
    )
    predicted = predict_logits(model, features[33:], 8)
    assert np.isfinite(predicted).all() and predicted.shape == (15,)
    if model_name == "catboost":
        assert 1 <= summary["best_iteration"] <= 8
        assert (tmp_path / "best.cbm").is_file()
        assert summary["effective_parameters"]["depth"] == 3
        return
    assert 1 <= summary["best_epoch"] <= 2
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert summary["parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert summary["parameter_count"] > 0
    assert all(epoch["train_rows_seen"] == 33 for epoch in summary["epochs"])


def test_tabm_optimizes_mean_member_loss_and_averages_probabilities() -> None:
    import torch

    from scripts.train_tp_only import binary_training_loss, ensemble_logit

    outputs = torch.tensor([[-4.0, 1.0, 2.0], [4.0, -1.0, -2.0]], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    loss = binary_training_loss(outputs, labels)
    individual = torch.stack(
        [
            torch.nn.functional.binary_cross_entropy_with_logits(
                outputs[:, member], labels
            )
            for member in range(3)
        ]
    ).mean()
    torch.testing.assert_close(loss, individual)
    averaged = ensemble_logit(outputs)
    torch.testing.assert_close(averaged.sigmoid(), outputs.sigmoid().mean(dim=1))
    assert not torch.allclose(averaged, outputs.mean(dim=1))
    loss.backward()
    assert outputs.grad is not None and torch.all(outputs.grad != 0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "unknown"},
        {"attention_dropout": 1.0},
        {"hidden_dropout": -0.1},
        {"tree_depth": 17},
        {"tree_l2": float("nan")},
        {"tree_subsample": 0},
    ],
)
def test_candidate_config_rejects_invalid_settings(overrides: dict) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**overrides)


def test_calibration_is_monotone_and_requires_actual_outcomes() -> None:
    logits = np.asarray([-3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.float64)
    mapping = fit_calibration(labels, logits)
    assert 0 <= mapping["slope"] <= 10
    assert np.isfinite(mapping["intercept"])
    with pytest.raises(ValueError, match="both outcomes"):
        fit_calibration(np.ones_like(labels), logits)


def test_catboost_refinement_is_bounded_and_has_distinct_hypotheses() -> None:
    from scripts.research_tp_catboost import variant_configurations

    variants = variant_configurations()
    assert len(variants) == 7
    assert {config.tree_depth for config in variants.values()} == {4, 6, 8}
    assert variants["slow_lr"].learning_rate == 0.015
    assert variants["strong_l2"].tree_l2 == 20
    assert variants["depth4_l2"].tree_depth == 4
    assert variants["depth4_l2"].tree_l2 == 20
    assert variants["depth4_slow"].tree_depth == 4
    assert variants["depth4_slow"].learning_rate == 0.015
    assert variants["depth4_slow"].tree_iterations == 4000
    assert all(config.max_seconds == 1200 for config in variants.values())
    assert all(
        config.threads == 8 and config.seed == 20261003 for config in variants.values()
    )


def test_azure_retirement_plan_matches_exact_jobs_and_preserves_other_data(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("azure.ai.ml")
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts import cleanup_azure_training as cleanup
    from scripts.azure_tp_job import RESOURCE_GROUP, SUBSCRIPTION, WORKSPACE
    from scripts.cleanup_azure_training import ACCOUNT, RETIRED_JOBS, build_cleanup_plan

    prefix = "LocalUpload/" + "a" * 64 + "/dataset/"
    inventory = {
        "subscription": SUBSCRIPTION,
        "resource_group": RESOURCE_GROUP,
        "workspace": WORKSPACE,
        "storage_account": "storage/" + ACCOUNT,
        "compute": {"nodes": 0, "min_instances": 0},
        "datastores": [
            {"container": "test", "name": "workspaceblobstore", "type": "AzureBlob"}
        ],
        "jobs": [
            {
                "name": name,
                "status": "Completed",
                "inputs": {
                    "dataset": "azureml://datastores/workspaceblobstore/paths/" + prefix
                },
            }
            for name in RETIRED_JOBS
        ],
    }
    storage = {
        "account": ACCOUNT,
        "policy": {},
        "containers": {
            "test": {
                "blobs": [
                    {
                        "name": name,
                        "size": 12,
                        "etag": '"original"',
                        "version_id": None,
                        "snapshot": None,
                    }
                    for name in (
                        prefix + "train.npz",
                        f"azureml/{RETIRED_JOBS[0]}/model/best.pt",
                        f"azureml/{RETIRED_JOBS[0]}-different/model",
                        "new-kite-dataset/train.npz",
                    )
                ]
            }
        },
    }
    plan = build_cleanup_plan(inventory, storage)
    assert len(plan["targets"]) == len(plan["retained"]) == 2
    assert plan["target_bytes"] == 24
    altered = deepcopy(inventory)
    altered["jobs"][0]["status"] = "Running"
    with pytest.raises(ValueError, match="terminal"):
        build_cleanup_plan(altered, storage)
    altered = deepcopy(inventory)
    altered["subscription"] = "another-subscription"
    with pytest.raises(ValueError, match="scope"):
        build_cleanup_plan(altered, storage)

    from azure.ai.ml.constants import ListViewType
    from azure.core import MatchConditions

    connection = Mock()
    archived = set()
    connection.jobs.list.side_effect = lambda *, list_view_type: [
        SimpleNamespace(name=name, status="Completed")
        for name in (
            archived if list_view_type == ListViewType.ARCHIVED_ONLY else RETIRED_JOBS
        )
    ]
    connection.jobs.get.side_effect = lambda name: SimpleNamespace(
        inputs={
            "dataset": SimpleNamespace(path=inventory["jobs"][0]["inputs"]["dataset"])
        }
    )
    connection.jobs.archive.side_effect = lambda *, name: archived.add(name)
    connection.compute.get.return_value = SimpleNamespace(min_instances=0)
    connection.compute.list_nodes.return_value = []
    service = Mock()
    changed = deepcopy(storage)
    changed["containers"]["test"]["blobs"][0]["etag"] = '"changed"'
    monkeypatch.setattr(cleanup, "blob_inventory", lambda *args: changed)
    with pytest.raises(ValueError, match="ETags"):
        cleanup.execute_cleanup(
            plan, inventory, connection, service, tmp_path / "blocked.json"
        )
    connection.jobs.archive.assert_not_called()
    service.get_container_client.assert_not_called()
    assert not (tmp_path / "blocked.json").exists()

    remaining = deepcopy(storage)
    monkeypatch.setattr(cleanup, "blob_inventory", lambda *args: deepcopy(remaining))

    def delete_batch(*blobs, **kwargs):
        assert kwargs == {"raise_on_any_failure": True}
        for blob in blobs:
            assert blob["etag"] == '"original"'
            assert blob["match_condition"] == MatchConditions.IfNotModified
            assert blob["delete_snapshots"] == "include"
        removed = {blob["name"] for blob in blobs}
        remaining["containers"]["test"]["blobs"] = [
            row
            for row in remaining["containers"]["test"]["blobs"]
            if row["name"] not in removed
        ]
        return [SimpleNamespace(status_code=202) for _ in blobs]

    service.get_container_client.return_value.delete_blobs.side_effect = delete_batch
    result = cleanup.execute_cleanup(
        plan, inventory, connection, service, tmp_path / "receipt.json"
    )
    assert result["status"] == "complete"
    assert result["deleted_blobs"] == result["remaining_blob_count"] == 2
    assert result["deleted_bytes"] == 24
    assert set(result["jobs_archived"]) == set(RETIRED_JOBS)
    assert result["compute_nodes"] == 0

    remaining = deepcopy(storage)
    removed_name = plan["targets"][0]["name"]
    remaining["containers"]["test"]["blobs"] = [
        row
        for row in remaining["containers"]["test"]["blobs"]
        if row["name"] != removed_name
    ]
    resume = tmp_path / "resume.json"
    resume.write_text(
        json.dumps(
            {
                "status": "deleting",
                "account": ACCOUNT,
                "jobs_archived": list(RETIRED_JOBS),
                "deleted_blobs": 1,
                "deleted_bytes": 12,
            }
        )
    )
    retried = cleanup.execute_cleanup(plan, inventory, connection, service, resume)
    assert retried["deleted_blobs"] == 2
    assert retried["attempts"] == 2
    assert retried["remaining_blob_count"] == 2


def test_cleanup_credential_caches_concurrent_requests_without_persisting_tokens():
    from concurrent.futures import ThreadPoolExecutor
    from time import time
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts.cleanup_azure_training import CachedCliCredential

    token = SimpleNamespace(token="synthetic", expires_on=time() + 3600)
    underlying = Mock()
    underlying.get_token.return_value = token
    cached = CachedCliCredential(underlying)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: cached.get_token("test-scope"), range(32)))
    assert all(result is token for result in results)
    underlying.get_token.assert_called_once_with("test-scope")
    token.expires_on = 0
    cached.get_token("test-scope")
    assert underlying.get_token.call_count == 2


def test_catboost_report_calibration_bins_cover_boundaries_exactly_once():
    from scripts.report_tp_catboost import independent_metrics

    probabilities = np.arange(11, dtype=np.float64) / 10
    labels = np.ones_like(probabilities)
    metrics = independent_metrics(labels, probabilities)
    assert metrics["ece_10_bins"] == pytest.approx(
        np.abs(labels - probabilities).mean()
    )
    assert metrics["hits"] == metrics["rows"] == 11


def test_calibration_selector_can_retain_identity_or_choose_improvement() -> None:
    from scripts.research_tp_catboost import apply_mapping, choose_calibration

    side = np.tile([-1, 1], 10)
    logits = np.zeros(20)
    mappings = {
        "identity": {"method": "identity"},
        "global_sigmoid": {
            "method": "global_sigmoid",
            "parameters": {"slope": 1, "intercept": 2},
        },
        "side_sigmoid": {
            "method": "side_sigmoid",
            "parameters": {
                "-1": {"slope": 1, "intercept": 2},
                "1": {"slope": 1, "intercept": -2},
            },
        },
    }
    balanced = np.tile([0, 0, 1, 1], 5)
    selected, _ = choose_calibration(balanced, logits, side, mappings)
    assert selected == "identity"
    high_rate = np.ones(20)
    high_rate[:2] = 0
    selected, report = choose_calibration(high_rate, logits, side, mappings)
    assert selected == "global_sigmoid"
    assert "side_sigmoid" not in report["eligible"]
    assert np.all(apply_mapping(mappings[selected], logits, side) > 0.5)
    with pytest.raises(ValueError, match="finite"):
        apply_mapping(mappings[selected], logits * np.nan, side)


def test_shard_loader_rejects_unhashed_or_cross_date_inputs(tmp_path: Path) -> None:
    from scripts.train_tp_only import file_hash

    path = tmp_path / "stock.npz"
    features = np.zeros((2, len(EXPECTED_COLUMNS)), dtype=np.float32)
    features[:, -1] = [1, -1]
    np.savez_compressed(
        path,
        features=features,
        labels=np.asarray([0, 1]),
        timestamps=np.asarray([1767225600, 1767225600]),
    )
    manifest = {
        "calendar": ["2024-01-01"],
        "phase_ends": {"train": "2025-06-30", "validation": "2025-12-31"},
        "shards": [
            {
                "path": "stock.npz",
                "symbol": "TEST",
                "rows": 2,
                "phase": "train",
                "sha256": file_hash(path),
            }
        ],
    }
    with pytest.raises(ValueError, match="chronological"):
        load_phase(tmp_path, manifest, "train")
    manifest["shards"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        load_phase(tmp_path, manifest, "train")


@pytest.fixture
def tp_training_data(tmp_path: Path) -> Path:
    from scripts import train_tp_only as training

    data = tmp_path / "data"
    data.mkdir()
    generator = np.random.default_rng(71)
    manifest = {
        "schema_version": 1,
        "objective": "directional_price_touch_without_stop_loss",
        "stock_identity_input": False,
        "model_columns": list(EXPECTED_COLUMNS),
        "tp_fraction": "0.002",
        "horizon_minutes": 60,
        "calendar": ["2024-01-01"],
        "phase_ends": {
            "train": "2025-06-30",
            "validation": "2025-12-31",
            "calibration": "2026-03-31",
            "evaluation": "2026-09-28",
        },
        "shards": [],
        "limitations": ["Synthetic integration fixture, not market evidence"],
    }
    for phase, day in (
        ("train", "2025-06-02"),
        ("validation", "2025-07-01"),
        ("calibration", "2026-02-02"),
        ("evaluation", "2026-05-01"),
    ):
        values = generator.normal(size=(32, len(EXPECTED_COLUMNS))).astype(np.float32)
        values = np.repeat(values, 2, axis=0)
        values[:, -1] = np.tile([1, -1], 32)
        epoch = int(datetime.fromisoformat(day + "T10:30:00+05:30").timestamp())
        path = data / f"{phase}.npz"
        np.savez_compressed(
            path,
            features=values,
            labels=(values[:, 0] + 0.2 * values[:, -1] > 0).astype(np.float32),
            timestamps=np.repeat(epoch + np.arange(32) * 60, 2),
        )
        manifest["shards"].append(
            {
                "phase": phase,
                "symbol": "TEST",
                "path": path.name,
                "rows": 64,
                "sha256": training.file_hash(path),
            }
        )
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data


@pytest.mark.parametrize(
    ("validation_only", "model_name"),
    [
        (False, "ft_transformer"),
        (True, "ft_transformer"),
        (True, "resnet"),
        (True, "tabm"),
        (True, "catboost"),
    ],
)
def test_main_training_path_locks_choices_before_reading_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_only: bool,
    model_name: str,
    tp_training_data: Path,
) -> None:
    import sys
    from dataclasses import asdict

    from scripts import train_tp_only as training

    if model_name in ("tabm", "catboost"):
        pytest.importorskip(model_name)
    data = tp_training_data
    output = tmp_path / "run"
    manifest = training.read_manifest(data)
    original = training.load_phase
    opened = []

    def checked_load(root: Path, document: dict, phase: str) -> dict:
        if validation_only:
            assert phase in ("train", "validation")
        if phase == "evaluation":
            assert (output / "selection_lock.json").is_file()
            assert (output / "calibration.json").is_file()
        opened.append(phase)
        return original(root, document, phase)

    monkeypatch.setattr(training, "load_phase", checked_load)
    config = TrainConfig(
        model=model_name,
        n_blocks=1,
        d_block=32,
        batch_size=16,
        epochs=2,
        threads=1,
        ensemble_size=4,
        tree_depth=3,
        tree_iterations=8,
    )
    if validation_only:
        config_path = tmp_path / "trial.json"
        config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_tp_only",
                "--data",
                str(data),
                "--output",
                str(output),
                "--config",
                str(config_path),
                "--validation-only",
            ],
        )
        assert training.main() == 0
        report = json.loads((output / "report.json").read_text())
        assert opened == ["train", "validation"]
        assert report["status"] == "validation_only_complete"
        assert report["calibration_and_evaluation_opened"] is False
        assert report["config"]["model"] == model_name
        assert (output / "validation_predictions.npz").is_file()
        assert not (output / "calibration.json").exists()
        assert not (output / "evaluation_predictions.npz").exists()
        if model_name == "catboost":
            preprocessing = json.loads((output / "preprocessing.json").read_text())
            assert preprocessing["kind"] == "identity"
            reloaded, _, restored = training.load_catboost_candidate(output, data)
            validation = original(data, manifest, "validation")
            with np.load(output / "validation_predictions.npz") as predictions:
                np.testing.assert_allclose(
                    training.expit(
                        training.predict_logits(reloaded, validation["features"], 16)
                    ),
                    predictions["probability"],
                    rtol=0,
                    atol=0,
                )
            assert restored["weights_sha256"] == report["weights_sha256"]
            with (output / "best.cbm").open("ab") as handle:
                handle.write(b"tampered")
            with pytest.raises(ValueError, match="hash mismatch"):
                training.load_catboost_candidate(output, data)
        return
    report = training.run_training(data, output, config)
    assert opened == ["train", "validation", "calibration", "evaluation"]
    assert report["evaluation"]["rows"] == 64
    assert report["status"] == "research_candidate_complete_not_promoted"
    assert (output / "evaluation_predictions.npz").is_file()


def test_catboost_study_locks_before_calibration_and_evaluates_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tp_training_data: Path
) -> None:
    pytest.importorskip("catboost")
    from scripts import research_tp_catboost as study
    from scripts import train_tp_only as training

    data = tp_training_data
    manifest = training.read_manifest(data)
    with np.load(data / "calibration.npz") as shard:
        features, labels, timestamps = (
            shard["features"],
            shard["labels"],
            shard["timestamps"],
        )
    np.savez_compressed(
        data / "calibration.npz",
        features=np.tile(features, (2, 1)),
        labels=np.tile(labels, 2),
        timestamps=np.concatenate((timestamps, timestamps + 28 * 86400)),
    )
    manifest["shards"][2]["rows"] = 128
    manifest["shards"][2]["sha256"] = training.file_hash(data / "calibration.npz")
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    baseline = tmp_path / "baseline"
    config = TrainConfig(model="catboost", tree_depth=2, tree_iterations=8, threads=1)
    training.run_training(data, baseline, config, validation_only=True)
    monkeypatch.setattr(study, "variant_configurations", lambda: {"probe": config})
    output = tmp_path / "study"
    original_load = study.load_phase
    original_fit = training.fit_model
    original_calibration = study.fit_calibration
    calibration_fit_sizes = []
    fitted = []
    opened = []

    def checked_fit(*args, **kwargs):
        assert "calibration" not in opened
        fitted.append(True)
        return original_fit(*args, **kwargs)

    def checked_load(root, document, phase):
        if phase == "calibration":
            assert (output / "model_selection.json").is_file()
            assert (output / "final" / "model_lock.json").is_file()
        if phase == "evaluation":
            assert (output / "final" / "selection_lock.json").is_file()
            assert (output / "final" / "calibration.json").is_file()
        opened.append(phase)
        return original_load(root, document, phase)

    def checked_calibration(labels, logits):
        assert "evaluation" not in opened
        calibration_fit_sizes.append(len(labels))
        return original_calibration(labels, logits)

    monkeypatch.setattr(training, "fit_model", checked_fit)
    monkeypatch.setattr(study, "fit_calibration", checked_calibration)
    monkeypatch.setattr(study, "load_phase", checked_load)
    result = study.run_study(data, baseline, output)
    assert len(fitted) == 2
    assert calibration_fit_sizes[:3] == [64, 32, 32]
    assert opened.count("calibration") == opened.count("evaluation") == 1
    assert result["seed_repeat_used_for_selection"] is False
    with np.load(output / "final" / "evaluation_predictions.npz") as predictions:
        evaluation = original_load(data, manifest, "evaluation")
        np.testing.assert_array_equal(
            study.predict_frozen(output / "final", evaluation["features"]),
            predictions["probability"],
        )
    with pytest.raises(ValueError, match="already exists"):
        study.run_study(data, baseline, output)
    from scripts.report_tp_catboost import audit_study

    audited = audit_study(data, output, baseline, output / "verification.json")
    assert audited["all_saved_predictions_reproduced"]
    assert audited["evaluation"]["selected"]["all"]["rows"] == 64
    altered = json.loads((output / "model_selection.json").read_text())
    altered["ranking"][0]["validation"]["log_loss"] += 0.1
    (output / "model_selection.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Independent"):
        audit_study(data, output, baseline, output / "bad-verification.json")
    with (output / "final" / "calibration.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        study.load_frozen_model(output / "final")


def test_session_block_uncertainty_is_paired_and_zero_for_identical_forecasts() -> None:
    from scripts.research_tp_catboost import block_loss_comparison

    labels = np.tile([0.0, 1.0], 10)
    probability = np.tile([0.2, 0.8], 10)
    timestamps = np.repeat(np.arange(10) * 86400 + 1767225600, 2)
    identical = block_loss_comparison(labels, probability, probability, timestamps)
    assert identical["sessions"] == 10
    assert identical["block_sessions"] == 5
    assert identical["log_loss"]["reduction_95pct_interval"] == [0.0, 0.0]
    better = block_loss_comparison(labels, probability, np.full(20, 0.5), timestamps)
    assert better["log_loss"]["reduction_95pct_interval"][0] > 0
    assert better["brier"]["reduction_95pct_interval"][0] > 0


@pytest.mark.parametrize("side", [1, -1])
def test_no_stop_calls_count_late_recovery_and_ignore_adverse_moves(side: int) -> None:
    from dataclasses import replace
    from datetime import timedelta
    from decimal import Decimal

    from ai_trader.broker import Instrument
    from ai_trader.market import Candle
    from scripts.analyze_tp_no_stop import (
        deadline_touch_labels,
        measure_call,
        touch_by_deadline,
    )

    decision = datetime.fromisoformat("2026-04-01T13:00:00+05:30")
    signal = Candle(
        instrument=Instrument("NSE", "TEST"),
        start_time=decision - timedelta(minutes=1),
        end_time=decision,
        open=Decimal(100),
        high=Decimal(100),
        low=Decimal(100),
        close=Decimal(100),
        volume=1000,
    )
    future = [
        replace(
            signal,
            start_time=decision + timedelta(minutes=offset),
            end_time=decision + timedelta(minutes=offset + 1),
        )
        for offset in range(120)
    ]
    future[0] = replace(
        future[0], **({"low": Decimal(95)} if side == 1 else {"high": Decimal(105)})
    )
    future[89] = replace(
        future[89],
        **({"high": Decimal("100.2")} if side == 1 else {"low": Decimal("99.8")}),
    )
    result = measure_call(signal, future, side)
    assert result["hit_60m"] is False
    assert result["hit_by_1500"] is True
    assert result["first_touch_by_1500"] == "2026-04-01T14:30:00+05:30"
    assert Decimal(result["tp_or_1500_return_fraction"]) == Decimal("0.002")
    assert (
        touch_by_deadline(signal, future, side, decision + timedelta(hours=1)) is False
    )
    assert (
        touch_by_deadline(signal, future, side, decision + timedelta(hours=2)) is True
    )
    assert (
        touch_by_deadline(
            signal, future[:50] + future[51:], side, decision + timedelta(hours=2)
        )
        is None
    )
    epochs = np.full(2, int(decision.timestamp()))
    deadlines = epochs + np.array([3600, 7200])
    directions = np.full(2, side)
    np.testing.assert_array_equal(
        deadline_touch_labels([signal, *future], epochs, directions, deadlines),
        [0, 1],
    )
    np.testing.assert_array_equal(
        deadline_touch_labels(
            [signal, *future[:50], *future[51:]], epochs, directions, deadlines
        ),
        [-1, -1],
    )
    future[0] = replace(future[0], high=Decimal(105), low=Decimal(95))
    assert measure_call(signal, future, side)["hit_60m_before_1500"] is True
    with pytest.raises(ValueError, match="Missing"):
        measure_call(signal, future[:50] + future[51:], side)


@pytest.mark.parametrize(("side", "expected"), [(1, "-0.001"), (-1, "0.001")])
def test_no_stop_calls_exclude_post_1500_touches_and_use_directional_return(
    side: int, expected: str
) -> None:
    from dataclasses import replace
    from datetime import timedelta
    from decimal import Decimal

    from ai_trader.broker import Instrument
    from ai_trader.market import Candle
    from scripts.analyze_tp_no_stop import (
        deadline_returns,
        measure_call,
        touch_by_deadline,
    )

    decision = datetime.fromisoformat("2026-04-01T14:15:00+05:30")
    signal = Candle(
        instrument=Instrument("NSE", "TEST"),
        start_time=decision - timedelta(minutes=1),
        end_time=decision,
        open=Decimal(100),
        high=Decimal(100),
        low=Decimal(100),
        close=Decimal(100),
        volume=1000,
    )
    future = [
        replace(
            signal,
            start_time=decision + timedelta(minutes=offset),
            end_time=decision + timedelta(minutes=offset + 1),
        )
        for offset in range(60)
    ]
    future[44] = replace(future[44], low=Decimal("99.9"), close=Decimal("99.9"))
    future[45] = replace(future[45], high=Decimal("100.2"), low=Decimal("99.8"))
    result = measure_call(signal, future, side)
    assert result["hit_60m"] is True
    assert result["hit_by_1500"] is False
    assert result["hit_60m_before_1500"] is False
    assert result["full_hour_before_1500"] is False
    assert result["square_off_time"] == "2026-04-01T15:00:00+05:30"
    assert Decimal(result["return_at_1500_fraction"]) == Decimal(expected)
    assert result["tp_or_1500_return_fraction"] == result["return_at_1500_fraction"]
    epochs = np.array([int(decision.timestamp())])
    deadlines = np.array([int(decision.replace(hour=15, minute=0).timestamp())])
    returns = deadline_returns([signal, *future], epochs, np.array([side]), deadlines)
    assert returns[0] == float(expected)
    assert np.isnan(
        deadline_returns(
            [signal, *future[:10], *future[11:]], epochs, np.array([side]), deadlines
        )[0]
    )
    assert (
        touch_by_deadline(signal, future, side, decision.replace(hour=15, minute=0))
        is False
    )
    assert (
        touch_by_deadline(signal, future, side, decision.replace(hour=15, minute=15))
        is True
    )
    future[44] = replace(future[44], high=Decimal("100.2"), low=Decimal("99.8"))
    assert measure_call(signal, future, side)["hit_by_1500"] is True
    assert (
        touch_by_deadline(signal, future, side, decision.replace(hour=15, minute=0))
        is True
    )


def test_no_stop_summary_keeps_unknown_days_out_of_failures() -> None:
    from scripts.analyze_tp_no_stop import summarize_calls

    success = {
        "side": 1,
        "hit_60m": True,
        "hit_60m_before_1500": True,
        "hit_by_1500": True,
        "full_hour_before_1500": True,
        "day_outcome_available": True,
        "return_at_1500_fraction": "-0.02",
        "tp_or_1500_return_fraction": "0.002",
    }
    late_success = {
        **success,
        "side": -1,
        "hit_60m": False,
        "hit_60m_before_1500": False,
    }
    failure = {
        **success,
        "hit_60m": False,
        "hit_60m_before_1500": False,
        "hit_by_1500": False,
        "return_at_1500_fraction": "-0.01",
        "tp_or_1500_return_fraction": "-0.01",
    }
    unknown = {**success, "day_outcome_available": False}
    summary = summarize_calls([success, late_success, failure, unknown])
    assert summary["calls"] == 4
    assert summary["hit_60m_rate"] == 0.5
    assert summary["hit_by_1500_rate"] == pytest.approx(2 / 3)
    assert summary["day_outcome_unavailable_calls"] == 1
    assert summary["never_hit_by_1500_count"] == 1
    assert summary["first_hit_after_60m_but_by_1500"] == 1
    assert summary["never_hit_return_at_1500"]["mean_fraction"] == -0.01
    assert summary["mean_tp_or_1500_gross_return_fraction"] == pytest.approx(-0.002)
    assert summarize_calls([]) == {"calls": 0}


def test_frequency_summary_preserves_pairs_denominators_and_conditional_returns():
    from scripts.analyze_tp_no_stop import frequency_and_return_summary

    arrays = {
        "timestamps": np.array([100, 100, 200, 200, 86500, 86500, 86600, 86600]),
        "symbols": np.array(["ALPHA", "ALPHA", "BRAVO", "BRAVO"] * 2),
        "side": np.tile([-1, 1], 4),
        "raw_probability": np.array([0.85, 0.82, 0.75, 0.35, 0.81, 0.45, 0.55, 0.25]),
        "hit_by_1500": np.array([1, 0, 0, 0, 1, 0, -1, 0]),
    }
    returns = np.array([0.01, -0.01, -0.02, -0.03, 0.01, -0.004, np.nan, -0.01])
    result = frequency_and_return_summary(arrays, returns)
    band = result["bins"][8]
    assert result["scored_candles"] == 4
    assert result["sessions"] == 2
    assert band["forecast_rows"] == 3
    assert band["percentage_of_all_forecasts"] == 37.5
    assert band["mean_forecasts_per_session"] == 1.5
    assert band["unique_scored_candles_with_either_side"] == 2
    assert band["scored_candles_with_both_sides"] == 1
    assert band["no_tp_by_1500"]["rows"] == 1
    assert band["no_tp_by_1500"]["mean_fraction"] == -0.01
    assert band["idealized_tp_or_1500_gross_return"]["mean_fraction"] == pytest.approx(
        -0.002
    )
    assert (
        result["scored_candle_bins_by_max_side_probability"][8][
            "percentage_of_scored_candles"
        ]
        == 50
    )
    assert result["all"]["unavailable_return_rows"] == 1
    assert sum(row["forecast_rows"] for row in result["bins"]) == 8
    assert (
        sum(
            row["scored_candles"]
            for row in result["scored_candle_bins_by_max_side_probability"]
        )
        == 4
    )
    malformed = {name: values.copy() for name, values in arrays.items()}
    malformed["side"][0] = 1
    with pytest.raises(ValueError, match="unique LONG/SHORT"):
        frequency_and_return_summary(malformed, returns)


def test_probability_reliability_separates_calibration_from_ranking_and_missing():
    from scripts.analyze_tp_no_stop import probability_reliability

    probabilities = np.array([0.2, 0.2, 0.8, 0.8, 0.8])
    labels = np.array([0, 0, 1, 1, -1])
    result = probability_reliability(labels, probabilities, np.arange(5))
    assert result["all"]["rows"] == 4
    assert result["all"]["unavailable_rows"] == 1
    assert result["all"]["ece_10_bins"] == pytest.approx(0.2)
    assert result["all"]["weighted_probability_bin_correlation"] == pytest.approx(1)
    assert result["bins"][8]["rows"] == 2
    assert result["bins"][8]["hit_rate"] == 1
    assert result["bins"][8]["unavailable_rows"] == 1
    with pytest.raises(ValueError, match="aligned"):
        probability_reliability(labels, probabilities[:-1], np.arange(5))


def test_quality_summary_counts_quarantine_minutes_not_dates():
    from scripts.analyze_tp_no_stop import data_quality_summary

    validation = {
        "independent_labels_checked": 20,
        "nse_comparisons": [],
        "instruments": {
            "NSE:TEST": {
                "minute_rows": 358,
                "missing_minutes_on_observed_dates": 2,
                "quarantined_dates": ["2026-08-04"],
                "other_absent_dates": [],
            },
            "NSE:NIFTY": {
                "minute_rows": 750,
                "missing_minutes_on_observed_dates": 0,
                "quarantined_dates": [],
                "other_absent_dates": [],
            },
        },
    }
    quality = {
        "usable": {
            "outside_session_rows": 0,
            "unknown_volume": 0,
            "critical_issues": [],
        }
    }
    capture = {
        "instruments": {
            "NSE:TEST": {"role": "equity_target", "quality": quality},
            "NSE:NIFTY": {"role": "context", "quality": quality},
        }
    }
    result = data_quality_summary(validation, capture, ["2026-08-04", "2026-08-05"])
    assert result["all_instruments"]["expected_slots"] == 1470
    assert result["all_instruments"]["quarantined_slots"] == 360
    assert result["all_instruments"]["excluded_slots"] == 362
    assert result["all_instruments"]["excluded_percentage"] == pytest.approx(
        362 / 1470 * 100
    )
    validation["instruments"]["NSE:TEST"]["minute_rows"] -= 1
    with pytest.raises(ValueError, match="reconcile"):
        data_quality_summary(validation, capture, ["2026-08-04", "2026-08-05"])


def test_matched_scanner_comparison_keeps_the_same_cycle_budget_and_eligibility() -> (
    None
):
    from scripts.compare_tp_scanners import matched_selections

    timestamps = np.asarray([10, 10, 10, 10, 20, 20, 30, 30])
    symbols = np.asarray(
        ["ALPHA", "ALPHA", "BRAVO", "BRAVO", "ALPHA", "ALPHA", "ALPHA", "ALPHA"]
    )
    sides = np.tile([1, -1], 4)
    rules = np.asarray([0.9, np.nan, 0.8, np.nan, 0.7, np.nan, np.nan, np.nan])
    raw = np.asarray([0.4, 0.8, 0.99, 0.7, 0.6, 0.5, 0.99, 0.99])
    eligible = np.asarray([True, True, False, True, True, True, True, True])
    selected = matched_selections(
        timestamps, symbols, sides, rules, {"raw": raw}, eligible
    )
    np.testing.assert_array_equal(np.flatnonzero(selected["rules"]), [0, 4])
    np.testing.assert_array_equal(np.flatnonzero(selected["raw"]), [1, 4])
    for moment in np.unique(timestamps):
        assert (
            selected["rules"][timestamps == moment].sum()
            == selected["raw"][timestamps == moment].sum()
        )
    permutation = np.asarray([7, 2, 4, 0, 6, 1, 3, 5])
    shuffled = matched_selections(
        timestamps[permutation],
        symbols[permutation],
        sides[permutation],
        rules[permutation],
        {"raw": raw[permutation]},
        eligible[permutation],
    )
    np.testing.assert_array_equal(shuffled["raw"], selected["raw"][permutation])
    with pytest.raises(ValueError, match="Duplicate"):
        matched_selections(
            np.asarray([10, 10]),
            np.asarray(["ALPHA", "ALPHA"]),
            np.asarray([1, 1]),
            np.asarray([0.5, 0.5]),
            {"raw": np.asarray([0.6, 0.7])},
            np.asarray([True, True]),
        )


def test_comparison_execution_preserves_entry_exclusion_stop_priority_horizon() -> None:
    from dataclasses import replace
    from datetime import timedelta
    from decimal import Decimal

    from ai_trader.broker import Instrument
    from ai_trader.market import Candle
    from ai_trader.replay import ExitReason, FillModel
    from ai_trader.scanner import Direction
    from ai_trader.strategy import StrategyConfig
    from scripts.compare_tp_scanners import isolated_case, simulate_ranked

    origin = datetime.fromisoformat("2026-04-01T10:00:00+05:30")
    signal = Candle(
        instrument=Instrument("NSE", "ALPHA"),
        start_time=origin,
        end_time=origin + timedelta(minutes=1),
        open=Decimal(100),
        high=Decimal("100.05"),
        low=Decimal("99.95"),
        close=Decimal(100),
        volume=1000,
    )
    future = [
        replace(
            signal,
            start_time=signal.end_time + timedelta(minutes=offset),
            end_time=signal.end_time + timedelta(minutes=offset + 1),
        )
        for offset in range(60)
    ]
    future[0] = replace(future[0], high=Decimal(102), low=Decimal(98))
    fill = FillModel(Decimal(1), Decimal("0.0002"), Decimal("0.0001"))
    strategy = StrategyConfig()
    horizon = isolated_case(
        signal, future, Decimal("0.001"), Direction.LONG, strategy, fill
    )
    assert horizon.exit_reason is ExitReason.HORIZON
    assert horizon.exit_time == signal.end_time + timedelta(minutes=60)
    assert horizon.entry_time == signal.end_time + timedelta(seconds=1)
    assert horizon.net_rupees < 0
    future[1] = replace(future[1], high=Decimal(102), low=Decimal(98))
    stopped = isolated_case(
        signal, future, Decimal("0.001"), Direction.LONG, strategy, fill
    )
    assert stopped.exit_reason is ExitReason.STOP
    assert stopped.ambiguous_exit is True
    with pytest.raises(ValueError, match="contiguous"):
        isolated_case(
            signal, future[:-1], Decimal("0.001"), Direction.LONG, strategy, fill
        )

    timestamp = int(signal.end_time.timestamp())
    timestamps = np.asarray([timestamp] * 4 + [timestamp + 300] * 4)
    symbols = np.asarray(["ALPHA", "ALPHA", "BRAVO", "CHARLIE"] * 2)
    sides = np.asarray([1, -1, 1, 1] * 2)
    scores = np.asarray([0.9, 0.8, 0.7, 0.6] * 2)
    requested = []

    def case_for(index):
        requested.append(index)
        return replace(horizon, instrument=Instrument("NSE", str(symbols[index])))

    chosen = simulate_ranked(
        timestamps, symbols, sides, scores, np.ones(8, dtype=bool), case_for
    )
    assert requested == [0, 2, 3]
    assert len(chosen) == 3


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
@pytest.mark.parametrize("scenario", ["standard", "adverse"])
def test_comparison_case_matches_replay_engine(side: str, scenario: str) -> None:
    from dataclasses import replace
    from datetime import timedelta
    from decimal import Decimal

    from ai_trader.broker import Instrument
    from ai_trader.market import Candle
    from ai_trader.replay import ExitReason, ReplayConfig, ReplayEngine
    from ai_trader.scanner import Direction
    from ai_trader.scanner.rules import RuleSignal
    from ai_trader.strategy import StrategyConfig
    from scripts.compare_tp_scanners import FILL_SCENARIOS, isolated_case

    direction = Direction(side)
    origin = datetime.fromisoformat("2026-04-01T10:00:00+05:30")
    signal = Candle(
        instrument=Instrument("NSE", "TEST"),
        start_time=origin,
        end_time=origin + timedelta(minutes=1),
        open=Decimal(100),
        high=Decimal("100.05"),
        low=Decimal("99.95"),
        close=Decimal(100),
        volume=1000,
    )
    warmup = tuple(
        replace(
            signal,
            start_time=signal.start_time - timedelta(minutes=offset),
            end_time=signal.end_time - timedelta(minutes=offset),
        )
        for offset in range(45, 0, -1)
    )
    future = [
        replace(
            signal,
            start_time=signal.end_time + timedelta(minutes=offset),
            end_time=signal.end_time + timedelta(minutes=offset + 1),
        )
        for offset in range(60)
    ]
    for index in (0, 1):
        future[index] = replace(future[index], high=Decimal(102), low=Decimal(98))

    class FixedSignalRule:
        name = "matched_comparison"
        required_features = ("atr_pct",)

        def evaluate(self, snapshot, portfolio, context):
            if snapshot.candle_end_time != signal.end_time:
                return None
            assert snapshot.atr_pct == Decimal("0.001")
            return RuleSignal(direction=direction, score=Decimal(1), evidence={})

    strategy = StrategyConfig(rules=(FixedSignalRule(),), screen_feasibility=False)
    fill = FILL_SCENARIOS[scenario]
    expected = isolated_case(
        signal, future, Decimal("0.001"), direction, strategy, fill
    )
    replay = ReplayEngine(
        ReplayConfig(universe=(signal.instrument,), strategy=strategy, fill=fill)
    ).run((signal, *future), warmup_candles=warmup)
    assert replay.trades == (expected,)
    assert expected.exit_reason is ExitReason.STOP
    assert expected.ambiguous_exit is True


def test_scanner_comparison_end_to_end_is_read_only_and_future_independent(
    tmp_path: Path, tp_training_data: Path
) -> None:
    from dataclasses import replace
    from datetime import timedelta
    from decimal import Decimal

    from ai_trader.broker import Instrument
    from ai_trader.features.tp_only import touch_label
    from ai_trader.history import CandleStore
    from ai_trader.market import Candle
    from ai_trader.strategy import StrategyConfig
    from scripts import compare_tp_scanners as comparison
    from scripts.train_tp_only import file_hash, load_phase, read_manifest

    data = tp_training_data
    manifest = read_manifest(data)
    warm_dates = ["2026-03-23", "2026-03-24", "2026-03-25", "2026-03-27", "2026-03-30"]
    manifest["calendar"] = ["2024-01-01", *warm_dates, "2026-05-01"]
    manifest["phase_ends"]["evaluation"] = "2026-05-01"
    cache = tmp_path / "candles"
    store = CandleStore(cache)
    instrument = Instrument("NSE", "TEST")
    candles = []
    for day in (*warm_dates, "2026-05-01"):
        beginning = datetime.fromisoformat(day + "T09:15:00+05:30")
        for minute in range(375):
            price = Decimal(100) + Decimal(minute) / 100
            candles.append(
                Candle(
                    instrument=instrument,
                    start_time=beginning + timedelta(minutes=minute),
                    end_time=beginning + timedelta(minutes=minute + 1),
                    open=price,
                    high=price + Decimal("0.10"),
                    low=price - Decimal("0.10"),
                    close=price + Decimal("0.01"),
                    volume=1000,
                )
            )
    store._write(instrument, tuple(candles))
    source = store.path_for(instrument)
    locations = {
        int(candle.end_time.timestamp()): index for index, candle in enumerate(candles)
    }
    with np.load(data / "evaluation.npz", allow_pickle=False) as shard:
        evaluation_values = {name: shard[name].copy() for name in shard.files}
    evaluation_values["labels"] = np.asarray(
        [
            int(
                touch_label(
                    candles[locations[int(epoch)]],
                    candles[locations[int(epoch)] + 1 : locations[int(epoch)] + 61],
                    int(side),
                ).hit
            )
            for epoch, side in zip(
                evaluation_values["timestamps"],
                evaluation_values["features"][:, -1],
                strict=True,
            )
        ],
        dtype=np.uint8,
    )
    np.savez_compressed(data / "evaluation.npz", **evaluation_values)
    for shard in manifest["shards"]:
        if shard["phase"] == "evaluation":
            shard["sha256"] = file_hash(data / shard["path"])
    manifest["source_sha256"] = {str(source): file_hash(source)}
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    phase = load_phase(data, manifest, "evaluation")
    probability = np.where(phase["features"][:, -1] == 1, 0.8, 0.3)
    model = tmp_path / "model"
    model.mkdir()
    (model / "best.cbm").write_bytes(b"fixed test artifact")
    (model / "calibration.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(
        model / "evaluation_predictions.npz",
        labels=phase["labels"],
        timestamps=phase["timestamps"],
        symbols=phase["symbols"],
        side=phase["features"][:, -1],
        raw_probability=probability,
        probability=probability,
    )
    lock = {
        "dataset_sha256": file_hash(data / "manifest.json"),
        "weights_sha256": file_hash(model / "best.cbm"),
        "calibration_sha256": file_hash(model / "calibration.json"),
    }
    (model / "selection_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    (model / "report.json").write_text(
        json.dumps(
            {
                "selection_lock": lock,
                "evaluation_predictions_sha256": file_hash(
                    model / "evaluation_predictions.npz"
                ),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison"
    result = comparison.run_comparison(data, model, cache, output)
    assert result["coverage"]["paired_action_rows"] == 64
    assert result["new_model_fits"] == result["live_orders"] == 0
    assert result["matched_tp_touch"]["rules"]["rows"] > 0
    assert len({values["rows"] for values in result["matched_tp_touch"].values()}) == 1
    assert set(result["execution"]) == {"standard", "adverse"}
    assert (output / "trades.csv").is_file()
    assert file_hash(source) == manifest["source_sha256"][str(source)]
    with pytest.raises(ValueError, match="already exists"):
        comparison.run_comparison(data, model, cache, output)

    from scripts.analyze_tp_no_stop import analyze_calls

    no_stop_output = tmp_path / "no_stop"
    no_stop = analyze_calls(data, output, cache, no_stop_output)
    for policy in ("catboost_raw", "catboost_calibrated"):
        summary = no_stop["policies"][policy]["all"]
        assert summary["calls"] == result["matched_tp_touch"][policy]["rows"]
        assert (
            summary["hit_60m_count"] == summary["hit_by_1500_count"] == summary["calls"]
        )
        assert summary["never_hit_by_1500_count"] == 0
    assert (no_stop_output / "calls.csv").is_file()
    assert file_hash(source) == manifest["source_sha256"][str(source)]
    with pytest.raises(ValueError, match="already exists"):
        analyze_calls(data, output, cache, no_stop_output)

    decision = int(phase["timestamps"][0])
    earlier = comparison.score_candles(candles, {decision}, StrategyConfig())
    altered = [
        replace(candle, high=candle.high * 2)
        if int(candle.start_time.timestamp()) >= decision
        else candle
        for candle in candles
    ]
    assert comparison.score_candles(altered, {decision}, StrategyConfig()) == earlier


def test_azure_job_has_cost_bounds_and_only_allowlisted_upload_roots(
    tmp_path: Path,
) -> None:
    pytest.importorskip("azure.ai.ml")
    from scripts.azure_tp_job import compute_spec, job_spec, stage_bundle

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = {
        "objective": "directional_price_touch_without_stop_loss",
        "stock_identity_input": False,
        "tp_fraction": "0.002",
        "horizon_minutes": 60,
        "shards": [{"path": "../forbidden.npz", "sha256": "invalid"}],
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="shard path"):
        stage_bundle(dataset, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
    (tmp_path / "code").mkdir()
    compute = compute_spec()
    assert compute.min_instances == 0 and compute.max_instances == 1
    assert compute.enable_node_public_ip is False
    assert compute.ssh_public_access_enabled is False
    job = job_spec(tmp_path, name="test-only-do-not-submit")
    assert job.resources.instance_count == 1
    assert job.limits.timeout == 144000
    assert job.code == str(tmp_path / "code")
    assert job.inputs["dataset"].path == str(dataset)
    assert "--max-seconds 129600" in job.command
    assert ".env" not in job.command


@pytest.mark.parametrize("validation_only", [False, True])
def test_staged_shards_survive_azure_url_encoding_without_changing_data(
    tmp_path: Path,
    validation_only: bool,
) -> None:
    from scripts.azure_tp_job import digest, stage_bundle

    dataset = tmp_path / "source"
    dataset.mkdir()
    manifest = {
        "objective": "directional_price_touch_without_stop_loss",
        "stock_identity_input": False,
        "tp_fraction": "0.002",
        "horizon_minutes": 60,
        "calendar": ["2024-01-01"],
        "phase_ends": {
            "train": "2025-06-30",
            "validation": "2025-12-31",
            "calibration": "2026-03-31",
            "evaluation": "2026-09-28",
        },
        "shards": [],
    }
    for phase, day in (
        ("train", "2025-06-02"),
        ("validation", "2025-07-01"),
        ("calibration", "2026-02-02"),
        ("evaluation", "2026-05-01"),
    ):
        (dataset / phase).mkdir()
        epoch = int(datetime.fromisoformat(day + "T10:30:00+05:30").timestamp())
        for index, symbol in enumerate(("M&M", "M_M", "M%26M")):
            path = dataset / phase / f"{symbol}.npz"
            values = np.full((2, len(EXPECTED_COLUMNS)), index, dtype=np.float32)
            values[:, -1] = [1, -1]
            np.savez_compressed(
                path,
                features=values,
                labels=np.asarray([0, 1]),
                timestamps=np.asarray([epoch, epoch]),
            )
            manifest["shards"].append(
                {
                    "phase": phase,
                    "symbol": symbol,
                    "rows": 2,
                    "path": path.relative_to(dataset).as_posix(),
                    "sha256": digest(path),
                }
            )
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_manifest = manifest_path.read_bytes()
    bundle = tmp_path / "bundle"
    result = stage_bundle(dataset, bundle, validation_only=validation_only)
    staged_root = bundle / "dataset"
    staged_manifest = json.loads((staged_root / "manifest.json").read_text())
    original_download = tmp_path / "original_download"
    safe_download = tmp_path / "safe_download"
    for root, downloaded in (
        (dataset, original_download),
        (staged_root, safe_download),
    ):
        for source in root.rglob("*"):
            if source.is_file():
                target = downloaded / quote(
                    source.relative_to(root).as_posix(), safe="/"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    with pytest.raises(FileNotFoundError):
        load_phase(original_download, manifest, "train")
    phases = (
        ("train", "validation") if validation_only else tuple(manifest["phase_ends"])
    )
    for phase in phases:
        expected = load_phase(dataset, manifest, phase)
        actual = load_phase(safe_download, staged_manifest, phase)
        for name in ("features", "labels", "timestamps", "symbols"):
            np.testing.assert_array_equal(actual[name], expected[name])
    assert (
        len({shard["path"] for shard in staged_manifest["shards"]}) == len(phases) * 3
    )
    for original, staged in zip(manifest["shards"], staged_manifest["shards"]):
        assert staged["source_path"] == original["path"]
        assert quote(staged["path"], safe="/") == staged["path"]
        assert (staged_root / staged["path"]).read_bytes() == (
            dataset / original["path"]
        ).read_bytes()
    assert manifest_path.read_bytes() == original_manifest
    assert staged_manifest["source_manifest_sha256"] == digest(manifest_path)
    assert result["source_dataset_sha256"] == digest(manifest_path)
    assert result["dataset_sha256"] == digest(staged_root / "manifest.json")
    if validation_only:
        assert not (staged_root / "calibration").exists()
        assert not (staged_root / "evaluation").exists()
        assert staged_manifest["available_phases"] == ["train", "validation"]


def test_fixed_trial_plan_uses_researched_values_and_bounded_validation_only_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("azure.ai.ml")
    from dataclasses import asdict
    from types import SimpleNamespace

    from azure.core.exceptions import ResourceNotFoundError

    from scripts import azure_tp_job as launcher
    from scripts.azure_tp_job import (
        stage_trials,
        trial_configurations,
        trial_job_spec,
        verify_trial_bundle,
    )

    dataset = tmp_path / "source"
    dataset.mkdir()
    manifest = {
        "objective": "directional_price_touch_without_stop_loss",
        "stock_identity_input": False,
        "tp_fraction": "0.002",
        "horizon_minutes": 60,
        "shards": [],
    }
    from scripts.azure_tp_job import digest

    for phase in ("train", "validation", "calibration", "evaluation"):
        (dataset / phase).mkdir()
        path = dataset / phase / "M&M.npz"
        np.savez_compressed(path, features=np.zeros((2, 14)))
        manifest["shards"].append(
            {
                "path": f"{phase}/M&M.npz",
                "phase": phase,
                "symbol": "M&M",
                "sha256": digest(path),
            }
        )
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = tmp_path / "batch"
    plan = stage_trials(dataset, bundle)
    assert verify_trial_bundle(bundle) == plan
    configurations = trial_configurations()
    assert len(configurations) == 6
    assert {config.model for config in configurations.values()} == {
        "ft_transformer",
        "resnet",
        "tabm",
        "catboost",
    }
    assert configurations["tabm"].learning_rate == 0.002
    assert configurations["tabm"].weight_decay == 0.0003
    assert configurations["catboost"].tree_depth == 6
    for trial in plan["trials"]:
        job = trial_job_spec(bundle, trial, plan)
        assert job.limits.timeout == 43200
        assert job.resources.instance_count == 1
        assert "--validation-only" in job.command
        assert f"--config {trial['config_file']}" in job.command
        config = json.loads((bundle / "code" / trial["config_file"]).read_text())
        assert config == asdict(configurations[trial["trial_id"]])
        assert config["max_seconds"] == 36000
        assert config["threads"] == 8
        assert config["seed"] == 20261003
    assert not (bundle / "dataset" / "calibration").exists()
    assert not (bundle / "dataset" / "evaluation").exists()
    compute = SimpleNamespace(
        name=launcher.COMPUTE,
        size="Standard_DS4_v2",
        min_instances=0,
        max_instances=1,
        ssh_public_access_enabled=False,
        enable_node_public_ip=False,
    )
    jobs = {}
    created = []
    updates = []

    def get_job(name):
        if name not in jobs:
            raise ResourceNotFoundError("No such fixture job")
        return jobs[name]

    def create_job(specification):
        created.append(specification.name)
        job = SimpleNamespace(
            name=specification.name,
            tags=specification.tags,
            status="Queued",
            studio_url="https://example.invalid/job",
        )
        jobs[job.name] = job
        return job

    def update_compute(specification):
        updates.append(specification.max_instances)
        return SimpleNamespace(result=lambda: specification)

    service = SimpleNamespace(
        compute=SimpleNamespace(get=lambda name: compute, begin_update=update_compute),
        jobs=SimpleNamespace(get=get_job, create_or_update=create_job),
    )
    monkeypatch.setattr(launcher, "client", lambda: service)
    first = launcher.submit_trials(bundle)
    second = launcher.submit_trials(bundle)
    assert len(first["jobs"]) == len(second["jobs"]) == 6
    assert len(created) == 6
    assert updates == [3]
    compute.max_instances = 4
    with pytest.raises(ValueError, match="parallel trial guard"):
        launcher.submit_trials(bundle)
    assert len(created) == 6
    (bundle / "code" / "unexpected.txt").write_text("not an allowed upload")
    with pytest.raises(ValueError, match="Non-allowlisted code"):
        verify_trial_bundle(bundle)
