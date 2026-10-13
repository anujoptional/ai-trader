"""Bounded CatBoost refinement and chronological probability calibration."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.train_tp_only import (
    EXPECTED_COLUMNS,
    TrainConfig,
    file_hash,
    fit_calibration,
    load_catboost_candidate,
    load_phase,
    metrics,
    predict_logits,
    read_manifest,
    run_training,
    write_json,
)

CALIBRATION_METHODS = ("identity", "global_sigmoid", "side_sigmoid")
CALIBRATION_CHECK_START = "2026-03-01"
FIT_LIMIT_SECONDS = 1200
REPEAT_SEED = 20261004
BOOTSTRAP_REPEATS = 1000
BLOCK_SESSIONS = 5


def variant_configurations() -> dict[str, TrainConfig]:
    base = TrainConfig(
        model="catboost", learning_rate=0.03, max_seconds=FIT_LIMIT_SECONDS
    )
    return {
        "depth4": replace(base, tree_depth=4),
        "depth4_l2": replace(base, tree_depth=4, tree_l2=20.0),
        "depth4_slow": replace(
            base,
            tree_depth=4,
            learning_rate=0.015,
            tree_iterations=4000,
            tree_patience=200,
        ),
        "depth8": replace(base, tree_depth=8),
        "slow_lr": replace(
            base, learning_rate=0.015, tree_iterations=4000, tree_patience=200
        ),
        "fast_lr": replace(base, learning_rate=0.06),
        "strong_l2": replace(base, tree_l2=20.0),
    }


def fit_mapping(
    method: str, labels: np.ndarray, logits: np.ndarray, side: np.ndarray
) -> dict:
    if method == "identity":
        return {"method": method}
    if method == "global_sigmoid":
        return {"method": method, "parameters": fit_calibration(labels, logits)}
    if method == "side_sigmoid":
        return {
            "method": method,
            "parameters": {
                str(direction): fit_calibration(
                    labels[side == direction], logits[side == direction]
                )
                for direction in (-1, 1)
            },
        }
    raise ValueError("Unknown calibration method")


def apply_mapping(mapping: dict, logits: np.ndarray, side: np.ndarray) -> np.ndarray:
    if logits.shape != side.shape or not np.isin(side, (-1, 1)).all():
        raise ValueError("Calibration requires paired directional logits")
    if not np.isfinite(logits).all():
        raise ValueError("Calibration logits must be finite")
    adjusted = np.asarray(logits, dtype=np.float64).copy()
    method = mapping["method"]
    if method not in CALIBRATION_METHODS:
        raise ValueError("Unknown calibration method")
    if method != "identity":
        for direction in (-1, 1):
            parameters = mapping["parameters"]
            if method == "side_sigmoid":
                parameters = parameters[str(direction)]
            slope, intercept = parameters["slope"], parameters["intercept"]
            if not np.isfinite((slope, intercept)).all() or not 0 <= slope <= 10:
                raise ValueError("Invalid monotone calibration parameters")
            selected = side == direction
            adjusted[selected] = slope * logits[selected] + intercept
    return expit(adjusted)


def probability_diagnostics(
    labels: np.ndarray, probability: np.ndarray, side: np.ndarray
) -> dict:
    return {
        "all": metrics(labels, probability),
        "by_side": {
            str(direction): metrics(
                labels[side == direction], probability[side == direction]
            )
            for direction in (-1, 1)
        },
    }


def choose_calibration(
    labels: np.ndarray, logits: np.ndarray, side: np.ndarray, mappings: dict
) -> tuple[str, dict]:
    reports = {
        method: probability_diagnostics(
            labels, apply_mapping(mapping, logits, side), side
        )
        for method, mapping in mappings.items()
    }
    raw = reports["identity"]
    eligible = [
        method
        for method in CALIBRATION_METHODS
        if reports[method]["all"]["brier"] <= raw["all"]["brier"]
        and all(
            reports[method]["by_side"][str(direction)]["log_loss"]
            <= raw["by_side"][str(direction)]["log_loss"]
            for direction in (-1, 1)
        )
    ]
    selected = min(eligible, key=lambda method: reports[method]["all"]["log_loss"])
    return selected, {"selected": selected, "eligible": eligible, "candidates": reports}


def dated_diagnostics(phase: dict, probability: np.ndarray) -> dict:
    labels, side = phase["labels"], phase["features"][:, -1]
    days = (phase["timestamps"] + 19800).astype("datetime64[s]").astype("datetime64[D]")
    months = days.astype("datetime64[M]")
    return {
        **probability_diagnostics(labels, probability, side),
        "sessions": len(np.unique(days)),
        "by_month": {
            str(month): probability_diagnostics(
                labels[months == month],
                probability[months == month],
                side[months == month],
            )
            for month in np.unique(months)
        },
        "fixed_threshold_diagnostics_not_trades": {
            str(threshold): {
                **probability_diagnostics(
                    labels[probability >= threshold],
                    probability[probability >= threshold],
                    side[probability >= threshold],
                ),
                "sessions": len(np.unique(days[probability >= threshold])),
            }
            for threshold in (0.6, 0.7, 0.8, 0.9)
        },
    }


def block_loss_comparison(
    labels: np.ndarray,
    probability: np.ndarray,
    reference: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    days, inverse = np.unique((timestamps + 19800) // 86400, return_inverse=True)
    counts = np.bincount(inverse)
    block_size = min(BLOCK_SESSIONS, len(days))
    generator = np.random.default_rng(REPEAT_SEED)
    starts = generator.integers(
        len(days), size=(BOOTSTRAP_REPEATS, (len(days) + block_size - 1) // block_size)
    )
    sampled = (starts[:, :, None] + np.arange(block_size)) % len(days)
    sampled = sampled.reshape(BOOTSTRAP_REPEATS, -1)[:, : len(days)]
    denominators = counts[sampled].sum(axis=1)
    result = {
        "method": (
            "paired circular moving blocks of trading dates; all assets together"
        ),
        "sessions": len(days),
        "block_sessions": block_size,
        "repeats": BOOTSTRAP_REPEATS,
        "seed": REPEAT_SEED,
        "warning": (
            "Descriptive retrospective uncertainty, not a stationarity guarantee"
        ),
    }
    for metric in ("log_loss", "brier"):
        losses = []
        for values in (probability, reference):
            clipped = np.clip(values, 1e-7, 1 - 1e-7)
            losses.append(
                -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
                if metric == "log_loss"
                else (values - labels) ** 2
            )
        model_losses, reference_losses = losses
        model_sums = np.bincount(inverse, weights=model_losses)
        reference_sums = np.bincount(inverse, weights=reference_losses)
        reduction = (reference_sums[sampled] - model_sums[sampled]).sum(
            axis=1
        ) / denominators
        result[metric] = {
            "reference_minus_model": float(np.mean(reference_losses - model_losses)),
            "reduction_95pct_interval": np.quantile(reduction, (0.025, 0.975)).tolist(),
        }
    return result


def calibration_protocol() -> dict:
    return {
        "fit": "2026-01-01 through 2026-02-28",
        "choose": "2026-03-01 through 2026-03-31",
        "methods": list(CALIBRATION_METHODS),
        "selection": (
            "minimum March log loss among mappings with no worse overall Brier "
            "or either direction's log loss than identity; ties prefer simplicity"
        ),
        "refit_chosen_mapping_on_all_calibration_rows": True,
        "model_refit": False,
        "sigmoid": (
            "monotone logit Platt, slope [0,10], identity-centered penalty 0.001"
        ),
        "evaluation": "2026-04-01 through 2026-09-28; no further selection",
    }


def load_frozen_model(output: Path):
    from catboost import CatBoostClassifier

    lock = json.loads((output / "selection_lock.json").read_text(encoding="utf-8"))
    for name, key in (
        ("best.cbm", "weights_sha256"),
        ("preprocessing.json", "preprocessing_sha256"),
        ("calibration.json", "calibration_sha256"),
        ("model_lock.json", "model_lock_sha256"),
    ):
        if file_hash(output / name) != lock[key]:
            raise ValueError("Locked CatBoost artifact hash mismatch")
    if (
        tuple(lock["columns"]) != EXPECTED_COLUMNS
        or lock["stock_identity_input"] is not False
    ):
        raise ValueError("Locked model input contract changed")
    model = CatBoostClassifier()
    model.load_model(str(output / "best.cbm"))
    if model.tree_count_ != lock["tree_count"]:
        raise ValueError("Locked tree count changed")
    mapping = json.loads((output / "calibration.json").read_text(encoding="utf-8"))[
        "mapping"
    ]
    return model, mapping, lock


def predict_frozen(output: Path, features: np.ndarray) -> np.ndarray:
    if (
        features.ndim != 2
        or features.shape[1] != len(EXPECTED_COLUMNS)
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            "Expected finite TP-only feature rows in the recorded column order"
        )
    model, mapping, _ = load_frozen_model(output)
    return apply_mapping(
        mapping, predict_logits(model, features, 1024), features[:, -1]
    )


def finalize_candidate(data: Path, candidate: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Final output already exists; preserve the frozen evaluation")
    model, _, candidate_report = load_catboost_candidate(candidate, data)
    manifest = read_manifest(data)
    output.mkdir(parents=True)
    for name in ("best.cbm", "preprocessing.json"):
        shutil.copyfile(candidate / name, output / name)
    model_lock = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": file_hash(data / "manifest.json"),
        "candidate_report_sha256": file_hash(candidate / "report.json"),
        "weights_sha256": file_hash(output / "best.cbm"),
        "preprocessing_sha256": file_hash(output / "preprocessing.json"),
        "config": candidate_report["config"],
        "tree_count": model.tree_count_,
        "columns": list(EXPECTED_COLUMNS),
        "stock_identity_input": False,
        "phase_ends": manifest["phase_ends"],
        "calibration_protocol": calibration_protocol(),
        "research_code_sha256": file_hash(Path(__file__)),
        "trainer_code_sha256": file_hash(Path(__file__).with_name("train_tp_only.py")),
        "versions": {name: version(name) for name in ("catboost", "numpy", "scipy")},
    }
    write_json(output / "model_lock.json", model_lock)
    train = load_phase(data, manifest, "train")
    training_priors = {
        str(side): float(train["labels"][train["features"][:, -1] == side].mean())
        for side in (-1, 1)
    }
    del train
    calibration = load_phase(data, manifest, "calibration")
    logits = predict_logits(model, calibration["features"], 1024)
    side = calibration["features"][:, -1]
    dates = (
        (calibration["timestamps"] + 19800)
        .astype("datetime64[s]")
        .astype("datetime64[D]")
    )
    fitting = dates < np.datetime64(CALIBRATION_CHECK_START)
    checking = ~fitting
    if not fitting.any() or not checking.any():
        raise ValueError("Calibration requires both January-February and March rows")
    mappings = {
        method: fit_mapping(
            method, calibration["labels"][fitting], logits[fitting], side[fitting]
        )
        for method in CALIBRATION_METHODS
    }
    selected, check_report = choose_calibration(
        calibration["labels"][checking], logits[checking], side[checking], mappings
    )
    mapping = fit_mapping(selected, calibration["labels"], logits, side)
    calibration_record = {
        "protocol": calibration_protocol(),
        "check": check_report,
        "january_february_mappings": mappings,
        "mapping": mapping,
        "fit_rows": int(fitting.sum()),
        "check_rows": int(checking.sum()),
        "fit_sessions": len(np.unique(dates[fitting])),
        "check_sessions": len(np.unique(dates[checking])),
        "refit_rows": len(side),
    }
    write_json(output / "calibration.json", calibration_record)
    np.savez_compressed(
        output / "calibration_check_predictions.npz",
        probability=apply_mapping(mappings[selected], logits[checking], side[checking]),
        raw_probability=expit(logits[checking]),
        labels=calibration["labels"][checking],
        timestamps=calibration["timestamps"][checking],
        side=side[checking],
    )
    calibration_priors = {
        str(direction): float(calibration["labels"][side == direction].mean())
        for direction in (-1, 1)
    }
    lock = {
        **model_lock,
        "model_lock_sha256": file_hash(output / "model_lock.json"),
        "calibration_sha256": file_hash(output / "calibration.json"),
        "training_side_priors": training_priors,
        "calibration_side_priors": calibration_priors,
        "bootstrap": {
            "block_sessions": BLOCK_SESSIONS,
            "repeats": BOOTSTRAP_REPEATS,
            "seed": REPEAT_SEED,
        },
        "fixed_probability_thresholds_not_trading_rules": [0.6, 0.7, 0.8, 0.9],
    }
    write_json(output / "selection_lock.json", lock)
    model, mapping, lock = load_frozen_model(output)
    if file_hash(data / "manifest.json") != lock["dataset_sha256"]:
        raise ValueError("Source dataset changed after model lock")
    evaluation = load_phase(data, manifest, "evaluation")
    raw_logits = predict_logits(model, evaluation["features"], 1024)
    evaluation_side = evaluation["features"][:, -1]
    probability = apply_mapping(mapping, raw_logits, evaluation_side)
    predictions = {
        "selected": probability,
        "raw": expit(raw_logits),
        "constant_50pct": np.full(len(probability), 0.5),
        **{
            name: np.asarray(
                [priors[str(int(direction))] for direction in evaluation_side]
            )
            for name, priors in (
                ("training_prior", training_priors),
                ("calibration_prior", calibration_priors),
            )
        },
    }
    np.savez_compressed(
        output / "evaluation_predictions.npz",
        probability=probability,
        raw_probability=predictions["raw"],
        labels=evaluation["labels"],
        timestamps=evaluation["timestamps"],
        symbols=evaluation["symbols"],
        side=evaluation_side,
    )
    report = {
        "status": "research_complete_not_promoted",
        "selection_lock": lock,
        "calibration": calibration_record,
        "evaluation": {
            name: dated_diagnostics(evaluation, values)
            for name, values in predictions.items()
        },
        "paired_session_uncertainty": {
            name: block_loss_comparison(
                evaluation["labels"], probability, values, evaluation["timestamps"]
            )
            for name, values in predictions.items()
            if name != "selected"
        },
        "evaluation_predictions_sha256": file_hash(
            output / "evaluation_predictions.npz"
        ),
        "limitations": manifest["limitations"],
        "model_refitted": False,
        "evaluation_used_for_selection": False,
    }
    write_json(output / "report.json", report)
    return report


def run_study(data: Path, baseline: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Study output already exists; use the saved result")
    baseline_model, _, baseline_report = load_catboost_candidate(baseline, data)
    configurations = variant_configurations()
    output.mkdir(parents=True)
    protocol = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": file_hash(data / "manifest.json"),
        "baseline_weights_sha256": baseline_report["weights_sha256"],
        "baseline_report_sha256": file_hash(baseline / "report.json"),
        "variants": {name: asdict(config) for name, config in configurations.items()},
        "candidate_design": (
            "CPU symmetric trees: depth 4/6/8 capacity controls, slower/faster "
            "learning, stronger L2, and shallow regularized alternatives. "
            "Natural class frequencies; no stock identity or target changes."
        ),
        "selection": (
            "minimum validation log loss; ties prefer shallower then fewer trees"
        ),
        "repeat_seed": REPEAT_SEED,
        "seed_repeat_used_for_selection": False,
        "maximum_new_fits": len(configurations) + 1,
        "per_fit_seconds": FIT_LIMIT_SECONDS,
        "concurrent_fits": 1,
        "execution": "local CPU; no Azure training jobs or live trading",
        "calibration": calibration_protocol(),
        "sources": [
            "https://catboost.ai/en/docs/concepts/parameter-tuning",
            "https://scikit-learn.org/stable/modules/calibration.html",
        ],
        "source_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("research_tp_catboost.py", "train_tp_only.py")
        },
    }
    write_json(output / "protocol.json", protocol)
    (output / "code").mkdir()
    for name in protocol["source_sha256"]:
        shutil.copyfile(Path(__file__).with_name(name), output / "code" / name)
    validation = load_phase(data, read_manifest(data), "validation")
    reproduced = metrics(
        validation["labels"],
        expit(predict_logits(baseline_model, validation["features"], 1024)),
    )
    if not np.isclose(
        reproduced["log_loss"],
        baseline_report["validation"]["log_loss"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Downloaded baseline does not reproduce its validation result")
    del baseline_model, validation
    reports = {"original": baseline_report}
    paths = {"original": baseline}
    for name, config in configurations.items():
        paths[name] = output / "trials" / name
        print(
            json.dumps({"trial": name, "status": "starting", "config": asdict(config)}),
            flush=True,
        )
        reports[name] = run_training(data, paths[name], config, validation_only=True)
    ranking = sorted(
        reports,
        key=lambda name: (
            reports[name]["validation"]["log_loss"],
            reports[name]["config"]["tree_depth"],
            reports[name]["tree_count"],
            name,
        ),
    )
    selected = ranking[0]
    selection = {
        "protocol_sha256": file_hash(output / "protocol.json"),
        "selected": selected,
        "weights_sha256": reports[selected]["weights_sha256"],
        "ranking": [
            {
                "trial": name,
                "validation": reports[name]["validation"],
                "config": reports[name]["config"],
                "tree_count": reports[name]["tree_count"],
            }
            for name in ranking
        ],
        "calibration_and_evaluation_opened": False,
    }
    write_json(output / "model_selection.json", selection)
    repeat_config = replace(
        TrainConfig(**reports[selected]["config"]),
        seed=REPEAT_SEED,
        max_seconds=FIT_LIMIT_SECONDS,
    )
    repeat = run_training(
        data, output / "seed_repeat", repeat_config, validation_only=True
    )
    if (
        file_hash(data / "manifest.json") != protocol["dataset_sha256"]
        or file_hash(paths[selected] / "best.cbm") != selection["weights_sha256"]
        or any(
            file_hash(Path(__file__).with_name(name)) != expected
            for name, expected in protocol["source_sha256"].items()
        )
    ):
        raise ValueError("Study inputs or source changed after the protocol was frozen")
    final = finalize_candidate(data, paths[selected], output / "final")
    report = {
        "status": final["status"],
        "model_selection": selection,
        "seed_repeat_validation": repeat["validation"],
        "seed_repeat_used_for_selection": False,
        "final_report_sha256": file_hash(output / "final" / "report.json"),
        "calibration_method": final["calibration"]["mapping"]["method"],
        "evaluation": {
            name: values["all"] for name, values in final["evaluation"].items()
        },
    }
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "selected": selected,
                "calibration_method": report["calibration_method"],
                "evaluation": report["evaluation"],
            }
        ),
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.data, args.baseline, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
