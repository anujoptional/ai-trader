"""Independently verify and summarize a frozen TP-only CatBoost study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.research_tp_catboost import apply_mapping, load_frozen_model
from scripts.train_tp_only import (
    EXPECTED_COLUMNS,
    file_hash,
    load_catboost_candidate,
    load_phase,
    predict_logits,
    read_manifest,
    write_json,
)


def independent_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    labels = labels.astype(np.float64)
    if labels.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("Invalid forecast arrays")
    if not np.isin(labels, (0, 1)).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("Labels or probabilities outside their domain")
    if not len(labels):
        return {"rows": 0}
    calibration_error = 0.0
    bin_ids = np.searchsorted(np.arange(1, 10) / 10, probability, side="right")
    for bin_id in range(10):
        chosen = bin_ids == bin_id
        if chosen.any():
            calibration_error += (
                abs(probability[chosen].mean() - labels[chosen].mean())
                * chosen.sum()
                / len(labels)
            )
    return {
        "rows": len(labels),
        "hits": int(labels.sum()),
        "hit_rate": float(labels.mean()),
        "mean_probability": float(probability.mean()),
        "log_loss": float(
            log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])
        ),
        "brier": float(brier_score_loss(labels, probability)),
        "roc_auc": float(roc_auc_score(labels, probability))
        if len(np.unique(labels)) == 2
        else None,
        "ece_10_bins": float(calibration_error),
        "classification_accuracy_at_05": float(np.mean((probability >= 0.5) == labels)),
    }


def _verified_metrics(
    labels: np.ndarray, probability: np.ndarray, recorded: dict
) -> dict:
    computed = independent_metrics(labels, probability)
    if computed["rows"] != recorded["rows"]:
        raise ValueError("Recorded forecast count differs")
    if not computed["rows"]:
        return computed
    for name in ("log_loss", "brier", "mean_probability"):
        if not np.isclose(computed[name], recorded[name], rtol=0, atol=1e-10):
            raise ValueError(f"Independent {name} differs from recorded result")
    if not np.isclose(computed["hit_rate"], recorded["hit_rate"], rtol=0, atol=1e-7):
        raise ValueError("Independent hit rate differs from recorded result")
    return computed


def audit_study(data: Path, study: Path, baseline: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Verification output already exists")
    manifest = read_manifest(data)
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((study / "model_selection.json").read_text(encoding="utf-8"))
    final = json.loads((study / "final/report.json").read_text(encoding="utf-8"))
    top = json.loads((study / "report.json").read_text(encoding="utf-8"))
    if (
        file_hash(data / "manifest.json") != protocol["dataset_sha256"]
        or file_hash(study / "protocol.json") != selection["protocol_sha256"]
        or file_hash(study / "final/report.json") != top["final_report_sha256"]
        or selection["calibration_and_evaluation_opened"] is not False
        or final["evaluation_used_for_selection"] is not False
    ):
        raise ValueError("Study provenance or chronological selection changed")
    for name, expected in {
        **manifest.get("source_sha256", {}),
        **manifest.get("code_sha256", {}),
    }.items():
        if file_hash(Path(name)) != expected:
            raise ValueError("Dataset source fingerprint changed")
    for name, expected in protocol["source_sha256"].items():
        if file_hash(study / "code" / name) != expected:
            raise ValueError("Executed training source fingerprint changed")
    validation = load_phase(data, manifest, "validation")
    candidates = []
    for entry in selection["ranking"]:
        path = (
            baseline
            if entry["trial"] == "original"
            else study / "trials" / entry["trial"]
        )
        model, _, report = load_catboost_candidate(path, data)
        probabilities = expit(predict_logits(model, validation["features"], 1024))
        checked = _verified_metrics(
            validation["labels"], probabilities, entry["validation"]
        )
        if report["calibration_and_evaluation_opened"] is not False:
            raise ValueError("A model candidate accessed later phases")
        candidates.append(
            {
                "trial": entry["trial"],
                "tree_count": model.tree_count_,
                "depth": report["config"]["tree_depth"],
                "l2": report["config"]["tree_l2"],
                "learning_rate": report["config"]["learning_rate"],
                "stop_reason": report["stop_reason"],
                "training_seconds": report.get("training_seconds"),
                "validation": checked,
            }
        )
    ranked = sorted(
        selection["ranking"],
        key=lambda entry: (
            entry["validation"]["log_loss"],
            entry["config"]["tree_depth"],
            entry["tree_count"],
            entry["trial"],
        ),
    )
    if selection["selected"] != ranked[0]["trial"]:
        raise ValueError("Selected model is not the declared validation winner")
    repeat_model, _, repeat_report = load_catboost_candidate(
        study / "seed_repeat", data
    )
    repeat = _verified_metrics(
        validation["labels"],
        expit(predict_logits(repeat_model, validation["features"], 1024)),
        repeat_report["validation"],
    )
    del validation, repeat_model
    model, mapping, lock = load_frozen_model(study / "final")
    if selection["weights_sha256"] != lock["weights_sha256"]:
        raise ValueError("Selected model weights changed")
    evaluation = load_phase(data, manifest, "evaluation")
    logits = predict_logits(model, evaluation["features"], 1024)
    side = evaluation["features"][:, -1]
    predictions = {
        "selected": apply_mapping(mapping, logits, side),
        "raw": expit(logits),
    }
    saved_path = study / "final/evaluation_predictions.npz"
    if file_hash(saved_path) != final["evaluation_predictions_sha256"]:
        raise ValueError("Saved evaluation predictions changed")
    with np.load(saved_path, allow_pickle=False) as saved:
        for name in ("labels", "timestamps", "symbols"):
            if not np.array_equal(saved[name], evaluation[name]):
                raise ValueError("Saved evaluation metadata differs from the dataset")
        for name, values in (
            ("probability", predictions["selected"]),
            ("raw_probability", predictions["raw"]),
            ("side", side),
        ):
            if not np.array_equal(saved[name], values):
                raise ValueError("Reloaded model does not reproduce saved forecasts")
    predictions["constant_50pct"] = np.full(len(side), 0.5)
    for name, key in (
        ("training_prior", "training_side_priors"),
        ("calibration_prior", "calibration_side_priors"),
    ):
        predictions[name] = np.asarray(
            [lock[key][str(int(direction))] for direction in side]
        )
    days = (
        (evaluation["timestamps"] + 19800)
        .astype("datetime64[s]")
        .astype("datetime64[D]")
    )
    months = days.astype("datetime64[M]")
    results = {}
    for name, probability in predictions.items():
        recorded = final["evaluation"][name]
        results[name] = {
            "all": _verified_metrics(
                evaluation["labels"], probability, recorded["all"]
            ),
            "by_side": {
                str(direction): _verified_metrics(
                    evaluation["labels"][side == direction],
                    probability[side == direction],
                    recorded["by_side"][str(direction)],
                )
                for direction in (-1, 1)
            },
            "by_month": {
                str(month): _verified_metrics(
                    evaluation["labels"][months == month],
                    probability[months == month],
                    recorded["by_month"][str(month)]["all"],
                )
                for month in np.unique(months)
            },
            "thresholds": {},
        }
        for threshold in lock["fixed_probability_thresholds_not_trading_rules"]:
            chosen = probability >= threshold
            results[name]["thresholds"][str(threshold)] = {
                **_verified_metrics(
                    evaluation["labels"][chosen],
                    probability[chosen],
                    recorded["fixed_threshold_diagnostics_not_trades"][str(threshold)][
                        "all"
                    ],
                ),
                "sessions": len(np.unique(days[chosen])),
                "by_side": {
                    str(direction): independent_metrics(
                        evaluation["labels"][chosen & (side == direction)],
                        probability[chosen & (side == direction)],
                    )
                    for direction in (-1, 1)
                },
            }
    importance = model.get_feature_importance(type="PredictionValuesChange")
    result = {
        "status": "verified",
        "provider": manifest.get("provider", "synthetic"),
        "dataset_sha256": file_hash(data / "manifest.json"),
        "protocol_sha256": file_hash(study / "protocol.json"),
        "selection_lock_sha256": file_hash(study / "final/selection_lock.json"),
        "selected": selection["selected"],
        "candidates": candidates,
        "seed_repeat": repeat,
        "evaluation_sessions": len(np.unique(days)),
        "evaluation": results,
        "all_saved_predictions_reproduced": True,
        "feature_importance_not_causal": sorted(
            [
                {"feature": name, "importance_percent": float(value)}
                for name, value in zip(EXPECTED_COLUMNS, importance, strict=True)
            ],
            key=lambda item: item["importance_percent"],
            reverse=True,
        ),
        "method": (
            "Exact reloaded forecasts; independent scikit-learn loss, Brier "
            "and AUC; direct hit counts"
        ),
        "limits": (
            "Retrospective dependent actions; threshold diagnostics are not "
            "executed trades or net profits"
        ),
        "source_sha256": file_hash(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_study(args.data, args.study, args.baseline, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"],
                "rows": report["evaluation"]["selected"]["all"]["rows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
