"""Standalone, checkpointed tabular research training; no broker access."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from rtdl_revisiting_models import FTTransformer, ResNet
from scipy.optimize import minimize
from scipy.special import expit

if TYPE_CHECKING:
    from catboost import CatBoostClassifier

EXPECTED_COLUMNS = (
    "log_return_5m",
    "log_return_previous_10m",
    "day_return",
    "gap_from_vendor_daily_close",
    "realized_volatility_20m",
    "prior_5_daily_close_return",
    "prior_5_cutoff_volatility",
    "vwap_z",
    "same_minute_rvol_5",
    "minutes_from_open",
    "benchmark_log_return_15m",
    "gold_log_return_15m",
    "vix_level",
    "side",
)
PRIOR_COLUMNS = frozenset(
    (
        "gap_from_vendor_daily_close",
        "prior_5_daily_close_return",
        "prior_5_cutoff_volatility",
    )
)


@dataclass(frozen=True)
class TrainConfig:
    model: str = "ft_transformer"
    seed: int = 20261003
    n_blocks: int = 3
    d_block: int = 192
    batch_size: int = 1024
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_seconds: int = 43200
    threads: int = 8
    attention_dropout: float = 0.2
    ffn_dropout: float = 0.1
    hidden_dropout: float = 0.1
    residual_dropout: float = 0.0
    hidden_multiplier: float = 2.0
    ensemble_size: int = 32
    tree_depth: int = 6
    tree_iterations: int = 2000
    tree_patience: int = 100
    tree_l2: float = 5.0
    tree_subsample: float = 0.8

    def __post_init__(self) -> None:
        if (
            min(
                self.n_blocks,
                self.d_block,
                self.batch_size,
                self.epochs,
                self.patience,
                self.max_seconds,
                self.threads,
                self.ensemble_size,
                self.tree_iterations,
                self.tree_patience,
            )
            <= 0
            or self.d_block % 8
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not np.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or self.model not in ("ft_transformer", "resnet", "tabm", "catboost")
            or any(
                not np.isfinite(value) or not 0 <= value < 1
                for value in (
                    self.attention_dropout,
                    self.ffn_dropout,
                    self.hidden_dropout,
                    self.residual_dropout,
                )
            )
            or not np.isfinite(self.hidden_multiplier)
            or self.hidden_multiplier <= 0
            or not 1 <= self.tree_depth <= 16
            or not np.isfinite(self.tree_l2)
            or self.tree_l2 <= 0
            or not np.isfinite(self.tree_subsample)
            or not 0 < self.tree_subsample <= 1
        ):
            raise ValueError("Invalid bounded training configuration")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_manifest(root: Path) -> dict:
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        document["schema_version"] != 1
        or document["objective"] != "directional_price_touch_without_stop_loss"
        or document["stock_identity_input"] is not False
        or tuple(document["model_columns"]) != EXPECTED_COLUMNS
        or float(document["tp_fraction"]) != 0.002
        or document["horizon_minutes"] != 60
    ):
        raise ValueError("Dataset does not match the fixed TP-only model contract")
    pairs = [(shard["phase"], shard["symbol"]) for shard in document["shards"]]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate phase/symbol shard")
    return document


def load_phase(root: Path, manifest: dict, phase: str) -> dict[str, np.ndarray]:
    matrices, labels, timestamps, symbols = [], [], [], []
    phase_names = list(manifest["phase_ends"])
    ordered_phases = sorted(phase_names, key=lambda name: manifest["phase_ends"][name])
    phase_index = ordered_phases.index(phase)
    lower = date.fromisoformat(manifest["calendar"][0]).toordinal()
    if phase_index:
        lower = (
            date.fromisoformat(
                manifest["phase_ends"][ordered_phases[phase_index - 1]]
            ).toordinal()
            + 1
        )
    upper = date.fromisoformat(manifest["phase_ends"][phase]).toordinal()
    for shard in manifest["shards"]:
        if shard["phase"] != phase:
            continue
        path = (root / shard["path"]).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or file_hash(path) != shard["sha256"]
        ):
            raise ValueError("Dataset shard path or hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            matrix = data["features"].astype(np.float32)
            target = data["labels"].astype(np.float32)
            moments = data["timestamps"]
        count = shard["rows"]
        days = (moments + 19800) // 86400 + date(1970, 1, 1).toordinal()
        if (
            matrix.shape != (count, len(EXPECTED_COLUMNS))
            or target.shape != (count,)
            or moments.shape != (count,)
            or count % 2
            or count == 0
            or not np.isfinite(matrix).all()
            or not np.isin(target, (0, 1)).all()
            or not np.equal(matrix[::2, -1], 1).all()
            or not np.equal(matrix[1::2, -1], -1).all()
            or not np.array_equal(matrix[::2, :-1], matrix[1::2, :-1])
            or not np.array_equal(moments[::2], moments[1::2])
            or not np.all(np.diff(moments[::2]) > 0)
            or np.any(days < lower)
            or np.any(days > upper)
        ):
            raise ValueError(
                "Invalid feature tensor, paired sides or chronological phase"
            )
        matrices.append(matrix)
        labels.append(target)
        timestamps.append(moments)
        symbols.append(np.full(count, shard["symbol"], dtype="U32"))
    if not matrices:
        raise ValueError(f"No eligible data in {phase}")
    return {
        "features": np.concatenate(matrices),
        "labels": np.concatenate(labels),
        "timestamps": np.concatenate(timestamps),
        "symbols": np.concatenate(symbols),
    }


def fit_preprocessing(features: np.ndarray) -> dict:
    if features.ndim != 2 or not np.isfinite(features).all() or len(features) < 2:
        raise ValueError("Preprocessing requires finite training rows")
    lower, upper = np.quantile(features, (0.005, 0.995), axis=0)
    clipped = np.clip(features, lower, upper)
    mean = clipped.mean(axis=0)
    scale = clipped.std(axis=0)
    scale[scale < 1e-12] = 1.0
    lower[-1], upper[-1], mean[-1], scale[-1] = -1.0, 1.0, 0.0, 1.0
    return {
        name: values.tolist()
        for name, values in (
            ("lower", lower),
            ("upper", upper),
            ("mean", mean),
            ("scale", scale),
        )
    }


def transform(features: np.ndarray, preprocessing: dict) -> np.ndarray:
    if preprocessing.get("kind") == "identity":
        return features.astype(np.float32)
    values = np.clip(features, preprocessing["lower"], preprocessing["upper"])
    return ((values - preprocessing["mean"]) / preprocessing["scale"]).astype(
        np.float32
    )


def build_model(feature_count: int, config: TrainConfig) -> torch.nn.Module:
    if config.model == "resnet":
        return ResNet(
            d_in=feature_count,
            d_out=1,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            d_hidden=None,
            d_hidden_multiplier=config.hidden_multiplier,
            dropout1=config.hidden_dropout,
            dropout2=config.residual_dropout,
        )
    if config.model == "tabm":
        from tabm import TabM

        return TabM.make(
            n_num_features=feature_count,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            dropout=config.hidden_dropout,
            k=config.ensemble_size,
            arch_type="tabm",
        )
    if config.model != "ft_transformer":
        raise ValueError("This model does not use the neural training path")
    return FTTransformer(
        n_cont_features=feature_count,
        cat_cardinalities=[],
        d_out=1,
        n_blocks=config.n_blocks,
        d_block=config.d_block,
        attention_n_heads=8,
        attention_dropout=config.attention_dropout,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=config.ffn_dropout,
        residual_dropout=config.residual_dropout,
    )


def neural_outputs(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    output = model(features) if isinstance(model, ResNet) else model(features, None)
    return output.squeeze(-1)


def binary_training_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 2:
        targets = targets[:, None].expand_as(outputs)
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)


def ensemble_logit(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 1:
        return outputs
    return torch.logit(outputs.sigmoid().mean(dim=1), eps=1e-7)


def predict_logits(
    model: torch.nn.Module | CatBoostClassifier, features: np.ndarray, batch_size: int
) -> np.ndarray:
    if not isinstance(model, torch.nn.Module):
        return np.asarray(
            model.predict(features, prediction_type="RawFormulaVal"), dtype=np.float64
        )
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size])
            predictions.append(
                ensemble_logit(neural_outputs(model, batch)).numpy().astype(np.float64)
            )
    return np.concatenate(predictions)


def log_loss_from_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, logits) - labels * logits))


def fit_model(
    train: np.ndarray,
    labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    output: Path,
    config: TrainConfig,
) -> tuple[torch.nn.Module | CatBoostClassifier, dict]:
    if config.model == "catboost":
        return fit_catboost(
            train, labels, validation, validation_labels, output, config
        )
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = build_model(train.shape[1], config)
    optimizer = torch.optim.AdamW(
        model.make_parameter_groups()
        if isinstance(model, FTTransformer)
        else model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    inputs = torch.from_numpy(train)
    targets = torch.from_numpy(labels.astype(np.float32))
    generator = torch.Generator().manual_seed(config.seed)
    best_loss, best_epoch = float("inf"), 0
    history = []
    began = time.perf_counter()
    stop_reason = "epoch_limit"
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss, seen = 0.0, 0
        batches = list(
            torch.randperm(len(train), generator=generator).split(config.batch_size)
        )
        if len(batches) > 1 and len(batches[-1]) == 1:
            final_batch = batches.pop()
            batches[-1] = torch.cat((batches[-1], final_batch))
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = binary_training_loss(
                neural_outputs(model, inputs[indices]), targets[indices]
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(indices)
            seen += len(indices)
            if time.perf_counter() - began >= config.max_seconds:
                stop_reason = "time_limit"
                break
        logits = predict_logits(model, validation, config.batch_size)
        validation_loss = log_loss_from_logits(validation_labels, logits)
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            torch.save(model.state_dict(), output / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "torch_rng_state": torch.get_rng_state(),
                "batch_rng_state": generator.get_state(),
            },
            output / "last.pt",
        )
        record = {
            "epoch": epoch,
            "train_rows_seen": seen,
            "train_log_loss": total_loss / seen,
            "validation_log_loss": validation_loss,
            "best_epoch": best_epoch,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "elapsed_seconds": time.perf_counter() - began,
        }
        history.append(record)
        write_json(output / "progress.json", {"epochs": history, "status": "running"})
        print(json.dumps(record), flush=True)
        if stop_reason == "time_limit":
            break
        if epoch - best_epoch >= config.patience:
            stop_reason = "early_stopping"
            break
    model.load_state_dict(
        torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    )
    summary = {
        "epochs": history,
        "best_epoch": best_epoch,
        "best_validation_log_loss": best_loss,
        "stop_reason": stop_reason,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.perf_counter() - began,
    }
    write_json(output / "progress.json", {**summary, "status": "fit_complete"})
    return model, summary


class CatBoostProgress:
    def __init__(self, output: Path, max_seconds: int) -> None:
        self.output = output
        self.max_seconds = max_seconds
        self.began = time.perf_counter()
        self.timed_out = False

    def after_iteration(self, info) -> bool:
        elapsed = time.perf_counter() - self.began
        self.timed_out = elapsed >= self.max_seconds
        if info.iteration == 1 or info.iteration % 25 == 0 or self.timed_out:
            values = info.metrics["validation"]["Logloss"]
            record = {
                "status": "running",
                "iteration": info.iteration,
                "train_log_loss": info.metrics["learn"]["Logloss"][-1],
                "validation_log_loss": values[-1],
                "best_validation_log_loss": min(values),
                "elapsed_seconds": elapsed,
            }
            write_json(self.output / "progress.json", record)
            print(json.dumps(record), flush=True)
        return not self.timed_out


def fit_catboost(
    train: np.ndarray,
    labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    output: Path,
    config: TrainConfig,
) -> tuple[CatBoostClassifier, dict]:
    from catboost import CatBoostClassifier

    progress = CatBoostProgress(output, config.max_seconds)
    model = CatBoostClassifier(
        iterations=config.tree_iterations,
        depth=config.tree_depth,
        learning_rate=config.learning_rate,
        l2_leaf_reg=config.tree_l2,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="CPU",
        boosting_type="Plain",
        grow_policy="SymmetricTree",
        border_count=254,
        bootstrap_type="Bernoulli",
        subsample=config.tree_subsample,
        random_strength=1.0,
        random_seed=config.seed,
        thread_count=config.threads,
        early_stopping_rounds=config.tree_patience,
        use_best_model=True,
        verbose=False,
        train_dir=str(output / "catboost"),
    )
    model.fit(
        train, labels, eval_set=(validation, validation_labels), callbacks=[progress]
    )
    model.save_model(str(output / "best.cbm"))
    reloaded = CatBoostClassifier()
    reloaded.load_model(str(output / "best.cbm"))
    losses = model.get_evals_result()["validation"]["Logloss"]
    best_iteration = model.get_best_iteration() + 1
    summary = {
        "best_epoch": None,
        "best_iteration": best_iteration,
        "iterations_completed": len(losses),
        "tree_count": model.tree_count_,
        "best_validation_log_loss": min(losses),
        "stop_reason": "time_limit"
        if progress.timed_out
        else (
            "early_stopping"
            if len(losses) < config.tree_iterations
            else "iteration_limit"
        ),
        "training_seconds": time.perf_counter() - progress.began,
        "effective_parameters": model.get_all_params(),
    }
    write_json(output / "progress.json", {**summary, "status": "fit_complete"})
    return reloaded, summary


def fit_calibration(labels: np.ndarray, logits: np.ndarray) -> dict:
    if not np.isfinite(logits).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires finite logits and both outcomes")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = parameters
        adjusted = slope * logits + intercept
        residual = expit(adjusted) - labels
        loss = log_loss_from_logits(labels, adjusted) + 0.001 * (
            (slope - 1) ** 2 + intercept**2
        )
        gradient = np.asarray(
            [
                np.mean(residual * logits) + 0.002 * (slope - 1),
                np.mean(residual) + 0.002 * intercept,
            ]
        )
        return loss, gradient

    fitted = minimize(
        objective,
        np.asarray([1.0, 0.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=((0, 10), (None, None)),
    )
    if not fitted.success:
        raise ValueError("Calibration did not converge")
    return {
        "slope": float(fitted.x[0]),
        "intercept": float(fitted.x[1]),
        "method": "monotone logit Platt; separate chronological calibration dates",
    }


def load_catboost_candidate(
    candidate: Path, data: Path
) -> tuple[CatBoostClassifier, dict, dict]:
    from catboost import CatBoostClassifier

    report = json.loads((candidate / "report.json").read_text(encoding="utf-8"))
    candidate_manifest = read_manifest(candidate)
    source_manifest = read_manifest(data)
    candidate_hash = file_hash(candidate / "manifest.json")
    source_hash = file_hash(data / "manifest.json")
    if (
        report["status"] != "validation_only_complete"
        or report["calibration_and_evaluation_opened"] is not False
        or report["config"]["model"] != "catboost"
        or tuple(report["columns"]) != EXPECTED_COLUMNS
        or report["dataset_sha256"] != candidate_hash
        or candidate_manifest.get("source_manifest_sha256", candidate_hash)
        != source_hash
        or report["phase_ends"] != source_manifest["phase_ends"]
        or report["weights_sha256"] != file_hash(candidate / "best.cbm")
        or report["preprocessing_sha256"] != file_hash(candidate / "preprocessing.json")
    ):
        raise ValueError("CatBoost candidate provenance or artifact hash mismatch")
    for phase in ("train", "validation"):
        expected = {
            shard["symbol"]: shard["sha256"]
            for shard in source_manifest["shards"]
            if shard["phase"] == phase
        }
        actual = {
            shard["symbol"]: shard["sha256"]
            for shard in candidate_manifest["shards"]
            if shard["phase"] == phase
        }
        if not expected or actual != expected:
            raise ValueError("CatBoost candidate training/validation shards differ")
    preprocessing = json.loads(
        (candidate / "preprocessing.json").read_text(encoding="utf-8")
    )
    if (
        preprocessing.get("kind") != "identity"
        or tuple(preprocessing["columns"]) != EXPECTED_COLUMNS
    ):
        raise ValueError("CatBoost continuation requires unchanged raw features")
    model = CatBoostClassifier()
    model.load_model(str(candidate / "best.cbm"))
    if model.tree_count_ != report["tree_count"]:
        raise ValueError("CatBoost checkpoint tree count differs from report")
    return model, preprocessing, report


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    if len(labels) == 0:
        return {"rows": 0}
    bins = []
    for index in range(10):
        selected = (probability >= index / 10) & (
            (probability < (index + 1) / 10) if index < 9 else (probability <= 1)
        )
        if np.any(selected):
            bins.append(
                {
                    "lower": index / 10,
                    "rows": int(selected.sum()),
                    "mean_probability": float(probability[selected].mean()),
                    "hit_rate": float(labels[selected].mean()),
                }
            )
    return {
        "rows": len(labels),
        "hit_rate": float(labels.mean()),
        "mean_probability": float(probability.mean()),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        ),
        "brier": float(np.mean((probability - labels) ** 2)),
        "reliability_bins": bins,
    }


def run_training(
    data: Path,
    output: Path,
    config: TrainConfig,
    omit_prior: bool = False,
    *,
    validation_only: bool = False,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output is not empty; use a new run")
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(data)
    columns = [
        name for name in EXPECTED_COLUMNS if not omit_prior or name not in PRIOR_COLUMNS
    ]
    indices = [EXPECTED_COLUMNS.index(name) for name in columns]
    train = load_phase(data, manifest, "train")
    validation = load_phase(data, manifest, "validation")
    preprocessing = (
        {"kind": "identity"}
        if config.model == "catboost"
        else fit_preprocessing(train["features"][:, indices])
    )
    write_json(output / "preprocessing.json", {"columns": columns, **preprocessing})
    train_values = transform(train["features"][:, indices], preprocessing)
    validation_values = transform(validation["features"][:, indices], preprocessing)
    priors = {
        str(side): float(train["labels"][train["features"][:, -1] == side].mean())
        for side in (-1, 1)
    }
    print(
        json.dumps(
            {
                "status": "starting",
                "train_rows": len(train_values),
                "validation_rows": len(validation_values),
                "columns": columns,
                "configuration": asdict(config),
            }
        ),
        flush=True,
    )
    model, summary = fit_model(
        train_values,
        train["labels"],
        validation_values,
        validation["labels"],
        output,
        config,
    )
    validation_logits = predict_logits(model, validation_values, config.batch_size)
    summary["validation"] = metrics(validation["labels"], expit(validation_logits))
    summary["validation_base_rate"] = metrics(
        validation["labels"],
        np.asarray([priors[str(int(side))] for side in validation["features"][:, -1]]),
    )
    weights_path = output / ("best.cbm" if config.model == "catboost" else "best.pt")
    summary["validation_by_side"] = {
        str(side): metrics(
            validation["labels"][validation["features"][:, -1] == side],
            expit(validation_logits[validation["features"][:, -1] == side]),
        )
        for side in (-1, 1)
    }
    months = (
        (validation["timestamps"] + 19800)
        .astype("datetime64[s]")
        .astype("datetime64[M]")
    )
    summary["validation_by_month"] = {
        str(month): metrics(
            validation["labels"][months == month],
            expit(validation_logits[months == month]),
        )
        for month in np.unique(months)
    }
    if validation_only:
        shutil.copyfile(data / "manifest.json", output / "manifest.json")
        summary.update(
            status="validation_only_complete",
            config=asdict(config),
            columns=columns,
            dataset_sha256=file_hash(data / "manifest.json"),
            trainer_sha256=file_hash(Path(__file__)),
            weights_sha256=file_hash(weights_path),
            preprocessing_sha256=file_hash(output / "preprocessing.json"),
            phase_ends=manifest["phase_ends"],
            training_side_priors=priors,
            calibration_and_evaluation_opened=False,
        )
        np.savez_compressed(
            output / "validation_predictions.npz",
            probability=expit(validation_logits),
            labels=validation["labels"],
            timestamps=validation["timestamps"],
            symbols=validation["symbols"],
            side=validation["features"][:, -1],
        )
        write_json(output / "report.json", summary)
        print(
            json.dumps(
                {"status": summary["status"], "validation": summary["validation"]}
            ),
            flush=True,
        )
        return summary
    calibration = load_phase(data, manifest, "calibration")
    calibration_logits = predict_logits(
        model,
        transform(calibration["features"][:, indices], preprocessing),
        config.batch_size,
    )
    mapping = fit_calibration(calibration["labels"], calibration_logits)
    write_json(output / "calibration.json", mapping)
    lock = {
        "dataset_sha256": file_hash(data / "manifest.json"),
        "trainer_sha256": file_hash(Path(__file__)),
        "config": asdict(config),
        "columns": columns,
        "stock_identity_input": False,
        "weights_sha256": file_hash(weights_path),
        "preprocessing_sha256": file_hash(output / "preprocessing.json"),
        "calibration_sha256": file_hash(output / "calibration.json"),
        "phase_ends": manifest["phase_ends"],
        "candidate": config.model,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    write_json(output / "selection_lock.json", lock)
    evaluation = load_phase(data, manifest, "evaluation")
    evaluation_logits = predict_logits(
        model,
        transform(evaluation["features"][:, indices], preprocessing),
        config.batch_size,
    )
    probability = expit(mapping["slope"] * evaluation_logits + mapping["intercept"])
    baseline = np.asarray(
        [priors[str(int(side))] for side in evaluation["features"][:, -1]]
    )
    summary["evaluation"] = metrics(evaluation["labels"], probability)
    summary["evaluation_raw"] = metrics(evaluation["labels"], expit(evaluation_logits))
    summary["evaluation_base_rate"] = metrics(evaluation["labels"], baseline)
    summary["evaluation_by_side"] = {
        str(side): metrics(
            evaluation["labels"][evaluation["features"][:, -1] == side],
            probability[evaluation["features"][:, -1] == side],
        )
        for side in (-1, 1)
    }
    summary["limitations"] = manifest["limitations"]
    summary["status"] = "research_candidate_complete_not_promoted"
    np.savez_compressed(
        output / "evaluation_predictions.npz",
        probability=probability,
        raw_probability=expit(evaluation_logits),
        labels=evaluation["labels"],
        timestamps=evaluation["timestamps"],
        symbols=evaluation["symbols"],
        side=evaluation["features"][:, -1],
    )
    write_json(output / "report.json", {**summary, "selection_lock": lock})
    print(
        json.dumps(
            {
                "status": summary["status"],
                "best_epoch": summary["best_epoch"],
                "evaluation_brier": summary["evaluation"]["brier"],
            }
        ),
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--omit-prior-history", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    overrides = {
        name: getattr(args, name)
        for name in ("epochs", "max_seconds", "threads", "batch_size")
        if getattr(args, name) is not None
    }
    if args.config and overrides:
        parser.error("Use a fixed --config or individual options, not both")
    config = TrainConfig(
        **(
            json.loads(args.config.read_text(encoding="utf-8"))
            if args.config
            else overrides
        )
    )
    run_training(
        args.data,
        args.output,
        config,
        args.omit_prior_history,
        validation_only=args.validation_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
