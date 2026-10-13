"""CatBoost TP scanner study: 09:40-13:30 calls, probability bands, 15:00 misses."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.special import expit

from ai_trader.costs import FIXED_CLIP_NOTIONAL, ZERODHA_INTRADAY_EQUITY
from scripts.report_tp_catboost import independent_metrics
from scripts.research_tp_catboost import (
    apply_mapping,
    block_loss_comparison,
    load_frozen_model,
)
from scripts.train_tp_only import (
    EXPECTED_COLUMNS,
    CatBoostProgress,
    file_hash,
    load_phase,
    predict_logits,
    read_manifest,
    write_json,
)

IST_OFFSET_SECONDS = 19800
FIRST_CALL_MINUTE = 9 * 60 + 40
LAST_CALL_MINUTE = 13 * 60 + 30
BAND_EDGES = np.arange(1, 10) / 10
HIGH_SCORE = 0.75
TP_RETURN = 0.002
SEED = 20261003
THREADS = 8
FIT_LIMIT_SECONDS = 1200
HALF_LIFE_DAYS = 365
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_REPEATS = 1000
BLOCK_SESSIONS = 5
FRESH_PHASES = {
    "train": ("train", "validation"),
    "validation": ("calibration",),
}
SOURCES = (
    "research_tp_scanner.py",
    "train_tp_only.py",
    "research_tp_catboost.py",
    "report_tp_catboost.py",
)
FEATURE_INPUTS = {
    "log_return_5m": "close[t], close[t-5]",
    "log_return_previous_10m": "close[t-5], close[t-15]",
    "day_return": "close[t], 09:15 open",
    "gap_from_vendor_daily_close": "09:15 open, prior vendor daily close",
    "realized_volatility_20m": "twenty squared one-minute log returns",
    "prior_5_daily_close_return": "prior daily close, six-sessions-back daily close",
    "prior_5_cutoff_volatility": "five prior sessions' 09:15-15:15 minute returns",
    "vwap_z": "close[t], session VWAP and its volume-weighted deviation",
    "same_minute_rvol_5": "current minute volume, five prior same-minute volumes",
    "minutes_from_open": "decision clock",
    "benchmark_log_return_15m": "NIFTY or BANKNIFTY closes",
    "gold_log_return_15m": "GOLDBEES closes",
    "vix_level": "INDIAVIX close",
    "side": "direction flag",
}


@dataclass(frozen=True)
class Candidate:
    depth: int
    half_life_days: int | None = None
    learning_rate: float = 0.03
    l2_leaf_reg: float = 20.0
    iterations: int = 2000
    patience: int = 100


def candidates() -> dict[str, Candidate]:
    return {
        "depth4": Candidate(depth=4),
        "depth4_recent": Candidate(depth=4, half_life_days=HALF_LIFE_DAYS),
        "depth6": Candidate(depth=6),
        "depth6_recent": Candidate(depth=6, half_life_days=HALF_LIFE_DAYS),
    }


def scanner_window(timestamps: np.ndarray) -> np.ndarray:
    minutes = (timestamps + IST_OFFSET_SECONDS) % 86400 // 60
    return (minutes >= FIRST_CALL_MINUTE) & (minutes <= LAST_CALL_MINUTE)


def load_window(data: Path, manifest: dict, phases: tuple[str, ...]) -> dict:
    parts = [load_phase(data, manifest, phase) for phase in phases]
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    keep = scanner_window(merged["timestamps"])
    return {name: values[keep] for name, values in merged.items()}


def recency_weights(
    timestamps: np.ndarray, half_life_days: int | None
) -> np.ndarray | None:
    if half_life_days is None:
        return None
    days = (timestamps + IST_OFFSET_SECONDS) // 86400
    weights = 0.5 ** ((days.max() - days) / half_life_days)
    return weights / weights.mean()


def feature_audit(features: np.ndarray) -> dict:
    values = features[features[:, -1] == 1, :-1].astype(np.float64)
    standardized = (values - values.mean(axis=0)) / values.std(axis=0)
    correlation = np.corrcoef(standardized, rowvar=False)
    np.fill_diagonal(correlation, 0.0)
    audit = {}
    for index, name in enumerate(EXPECTED_COLUMNS[:-1]):
        design = np.column_stack(
            (np.ones(len(standardized)), np.delete(standardized, index, axis=1))
        )
        target = standardized[:, index]
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        partner = int(np.argmax(np.abs(correlation[index])))
        audit[name] = {
            "inputs": FEATURE_INPUTS[name],
            "linear_r2_from_other_features": float(
                1 - np.var(target - design @ coefficients) / np.var(target)
            ),
            "strongest_correlation": float(correlation[index, partner]),
            "strongest_partner": EXPECTED_COLUMNS[partner],
        }
    return {
        "rows": len(values),
        "rule": (
            "Each feature uses an input no other feature carries, so none is an "
            "algebraic function of the rest; linear R2 measures overlap only"
        ),
        "features": audit,
    }


def fit_candidate(
    train: dict, validation: dict, candidate: Candidate, output: Path
) -> dict:
    from catboost import CatBoostClassifier

    output.mkdir(parents=True)
    progress = CatBoostProgress(output, FIT_LIMIT_SECONDS)
    model = CatBoostClassifier(
        iterations=candidate.iterations,
        depth=candidate.depth,
        learning_rate=candidate.learning_rate,
        l2_leaf_reg=candidate.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="CPU",
        boosting_type="Plain",
        grow_policy="SymmetricTree",
        border_count=254,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_strength=1.0,
        random_seed=SEED,
        thread_count=THREADS,
        early_stopping_rounds=candidate.patience,
        use_best_model=True,
        verbose=False,
        train_dir=str(output / "catboost"),
    )
    model.fit(
        train["features"],
        train["labels"],
        sample_weight=recency_weights(train["timestamps"], candidate.half_life_days),
        eval_set=(validation["features"], validation["labels"]),
        callbacks=[progress],
    )
    model.save_model(str(output / "best.cbm"))
    losses = model.get_evals_result()["validation"]["Logloss"]
    reloaded = CatBoostClassifier()
    reloaded.load_model(str(output / "best.cbm"))
    probability = expit(predict_logits(reloaded, validation["features"], 1024))
    side = validation["features"][:, -1]
    report = {
        "candidate": asdict(candidate),
        "tree_count": reloaded.tree_count_,
        "iterations_completed": len(losses),
        "best_catboost_validation_log_loss": min(losses),
        "stop_reason": "time_limit"
        if progress.timed_out
        else (
            "early_stopping"
            if len(losses) < candidate.iterations
            else "iteration_limit"
        ),
        "fit_seconds": time.perf_counter() - progress.began,
        "weights_sha256": file_hash(output / "best.cbm"),
        "validation": independent_metrics(validation["labels"], probability),
        "validation_by_side": {
            str(direction): independent_metrics(
                validation["labels"][side == direction], probability[side == direction]
            )
            for direction in (-1, 1)
        },
        "evaluation_opened": False,
    }
    write_json(output / "report.json", report)
    return report


def select_candidate(reports: dict[str, dict]) -> str:
    return min(
        reports,
        key=lambda name: (
            reports[name]["validation"]["log_loss"],
            reports[name]["candidate"]["half_life_days"] is not None,
            reports[name]["candidate"]["depth"],
            reports[name]["tree_count"],
            name,
        ),
    )


def session_block_interval(
    values: np.ndarray, day_ids: np.ndarray, sessions: int
) -> list[float] | None:
    if not len(values):
        return None
    sums = np.bincount(day_ids, weights=values, minlength=sessions)
    counts = np.bincount(day_ids, minlength=sessions)
    block = min(BLOCK_SESSIONS, sessions)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    starts = generator.integers(
        sessions, size=(BOOTSTRAP_REPEATS, -(-sessions // block))
    )
    sampled = ((starts[:, :, None] + np.arange(block)) % sessions).reshape(
        BOOTSTRAP_REPEATS, -1
    )[:, :sessions]
    totals = counts[sampled].sum(axis=1)
    usable = totals > 0
    means = sums[sampled].sum(axis=1)[usable] / totals[usable]
    return np.quantile(means, (0.025, 0.975)).tolist()


def return_statistics(values: np.ndarray) -> dict:
    if not len(values):
        return {"calls": 0}
    return {
        "calls": len(values),
        "mean_percent": 100 * float(values.mean()),
        "median_percent": 100 * float(np.median(values)),
        "p10_percent": 100 * float(np.quantile(values, 0.1)),
        "p90_percent": 100 * float(np.quantile(values, 0.9)),
        "worst_percent": 100 * float(values.min()),
        "best_percent": 100 * float(values.max()),
        "negative_calls": int((values < 0).sum()),
    }


def scanner_report(
    probability: np.ndarray,
    labels: np.ndarray,
    hit_by_1500: np.ndarray,
    return_at_1500: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    side: np.ndarray,
    charges: float,
) -> dict:
    arrays = (labels, hit_by_1500, return_at_1500, timestamps, symbols, side)
    if (
        probability.ndim != 1
        or not len(probability)
        or len(probability) % 2
        or any(values.shape != probability.shape for values in arrays)
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or not np.isin(labels, (0, 1)).all()
        or not np.isin(hit_by_1500, (0, 1)).all()
        or not np.isfinite(return_at_1500).all()
        or not np.isin(side, (-1, 1)).all()
    ):
        raise ValueError("Scanner evaluation needs complete aligned outcomes")
    if np.any((labels == 1) & (hit_by_1500 == 0)):
        raise ValueError("A 60-minute touch inside the window must count by 15:00")
    if (
        not np.all(side[::2] == 1)
        or not np.all(side[1::2] == -1)
        or not np.array_equal(timestamps[::2], timestamps[1::2])
        or not np.array_equal(symbols[::2], symbols[1::2])
    ):
        raise ValueError("Expected adjacent LONG/SHORT pairs per candle-time")
    days, day_ids = np.unique(
        (timestamps + IST_OFFSET_SECONDS) // 86400, return_inverse=True
    )
    hits = hit_by_1500 == 1
    gross = np.where(hits, TP_RETURN, return_at_1500)
    bins = np.searchsorted(BAND_EDGES, probability, side="right")

    def band(selected: np.ndarray) -> dict:
        calls = int(selected.sum())
        if not calls:
            return {"calls": 0}
        misses = selected & ~hits
        return {
            "calls": calls,
            "percent_of_calls": 100 * calls / len(probability),
            "long_calls": int((selected & (side == 1)).sum()),
            "short_calls": int((selected & (side == -1)).sum()),
            "mean_probability_percent": 100 * float(probability[selected].mean()),
            "tp_within_60m_percent": 100 * float(labels[selected].mean()),
            "tp_by_1500_percent": 100 * float(hits[selected].mean()),
            "no_tp_by_1500_calls": int(misses.sum()),
            "no_tp_mean_return_at_1500_percent": 100
            * float(return_at_1500[misses].mean())
            if misses.any()
            else None,
        }

    def high_score(selected: np.ndarray) -> dict:
        calls = int(selected.sum())
        if not calls:
            return {"threshold": HIGH_SCORE, "calls": 0}
        misses = selected & ~hits
        miss_returns = return_at_1500[misses]
        miss_mean = float(miss_returns.mean()) if len(miss_returns) else 0.0
        pairs = selected.reshape(-1, 2)
        selected_days = np.bincount(day_ids[selected], minlength=len(days))
        rate_interval = session_block_interval(
            hits[selected].astype(float), day_ids[selected], len(days)
        )
        gross_interval = session_block_interval(
            gross[selected], day_ids[selected], len(days)
        )
        return {
            "threshold": HIGH_SCORE,
            "calls": calls,
            "long_calls": int((selected & (side == 1)).sum()),
            "short_calls": int((selected & (side == -1)).sum()),
            "sessions": len(days),
            "sessions_with_calls": int((selected_days > 0).sum()),
            "mean_calls_per_session": calls / len(days),
            "distinct_candle_times": int(pairs.any(axis=1).sum()),
            "candle_times_with_both_sides": int(pairs.all(axis=1).sum()),
            "mean_probability_percent": 100 * float(probability[selected].mean()),
            "tp_within_60m_percent": 100 * float(labels[selected].mean()),
            "tp_by_1500_percent": 100 * float(hits[selected].mean()),
            "tp_by_1500_percent_95pct_session_block_interval": [
                100 * value for value in rate_interval
            ],
            "no_tp_by_1500_gross_return_at_1500": return_statistics(miss_returns),
            "no_tp_by_1500_return_after_charges": return_statistics(
                miss_returns - charges
            ),
            "idealized_tp_or_1500_gross_mean_percent": 100
            * float(gross[selected].mean()),
            "idealized_tp_or_1500_gross_mean_95pct_session_block_interval": [
                100 * value for value in gross_interval
            ],
            "idealized_tp_or_1500_mean_after_charges_percent": 100
            * (float(gross[selected].mean()) - charges),
            "break_even_tp_by_1500_percent_gross": 100
            * -miss_mean
            / (TP_RETURN - miss_mean),
            "break_even_tp_by_1500_percent_after_charges": 100
            * (charges - miss_mean)
            / (TP_RETURN - miss_mean),
            "by_side": {
                str(direction): band(selected & (side == direction))
                for direction in (1, -1)
            },
        }

    return {
        "overall_60m": independent_metrics(labels, probability),
        "overall_tp_by_1500_percent": 100 * float(hits.mean()),
        "auc_by_side": {
            str(direction): independent_metrics(
                labels[side == direction], probability[side == direction]
            )["roc_auc"]
            for direction in (1, -1)
        },
        "bands": [
            {"band": f"{10 * index}-{10 * (index + 1)}%", **band(bins == index)}
            for index in range(10)
        ],
        "high_score": high_score(probability >= HIGH_SCORE),
    }


def protocol_document(
    data: Path, current: Path, outcomes: Path, charges: float, sources: dict
) -> dict:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "TP-touch CatBoost scanner: band audit of the locked model and one "
            "fresh validation-selected model on the 09:40-13:30 call window"
        ),
        "inputs": {
            "dataset_manifest_sha256": file_hash(data / "manifest.json"),
            "current_selection_lock_sha256": file_hash(current / "selection_lock.json"),
            "outcome_report_sha256": file_hash(outcomes / "report.json"),
            "outcome_returns_sha256": file_hash(outcomes / "returns.npz"),
        },
        "call_window_ist": {
            "first": "09:40",
            "last": "13:30",
            "inclusive": True,
            "grid_minutes": 5,
            "reference": "close of the last completed one-minute candle",
        },
        "target": (
            "LONG close*1.002 or SHORT close*0.998 touched by a complete minute "
            "high/low within the next 60 minutes; no stop-loss"
        ),
        "model_inputs": list(EXPECTED_COLUMNS),
        "feature_audit": {
            "inputs": FEATURE_INPUTS,
            "empirical": "linear R2 of each feature from the other twelve, LONG rows",
        },
        "bands": "ten lower-inclusive 10% bands; probability 1.0 is in 90-100%",
        "outcomes": {
            "tp_within_60m": "frozen dataset label",
            "tp_by_1500": "touch in any complete minute through the 14:59-15:00 candle",
            "return_at_1500": (
                "side * (close of the 14:59-15:00 candle / decision close - 1), gross"
            ),
        },
        "high_score_threshold": HIGH_SCORE,
        "charges": {
            "model": "ZERODHA_INTRADAY_EQUITY estimate",
            "notional_inr": int(FIXED_CLIP_NOTIONAL),
            "round_trip_fraction": charges,
            "excludes": "spread, slippage and fill uncertainty",
        },
        "current_model": (
            "locked depth4_l2 study; calibrated selected output and raw comparator, "
            "no refit or reselection"
        ),
        "fresh_model": {
            "train": "frozen train+validation phases through 2025-12-31, window rows",
            "validation": (
                "2026-01-01 through 2026-03-31 window rows: early stopping and "
                "candidate selection"
            ),
            "evaluation": (
                "2026-04-01 through 2026-09-28 window rows, opened once after "
                "the selection lock"
            ),
            "candidates": {name: asdict(value) for name, value in candidates().items()},
            "fixed": {
                "loss": "Logloss",
                "boosting": "Plain SymmetricTree",
                "border_count": 254,
                "bootstrap": "Bernoulli subsample 0.8",
                "random_strength": 1.0,
                "seed": SEED,
                "threads": THREADS,
                "fit_limit_seconds": FIT_LIMIT_SECONDS,
                "use_best_model": True,
            },
            "recency_weights": (
                "0.5 ** (days before the last training session / half-life), "
                "normalized to mean one; validation loss is unweighted"
            ),
            "selection": (
                "minimum validation log loss; ties prefer uniform weights, "
                "shallower depth, fewer trees"
            ),
            "output": "raw sigmoid of the CatBoost margin; no post-hoc calibration",
        },
        "decision_rule": (
            "Prefer the fresh model only if its window evaluation log-loss reduction "
            "versus BOTH locked outputs has a paired 95% session-block interval "
            "entirely above zero; otherwise keep the locked model"
        ),
        "uncertainty": (
            f"circular {BLOCK_SESSIONS}-session blocks, {BOOTSTRAP_REPEATS} repeats; "
            "descriptive only"
        ),
        "limitations": [
            "April-September 2026 was examined by earlier research: retrospective, "
            "not a pristine holdout",
            "A touch is not a fill; 15:00 returns are gross close-to-close",
            "Calls overlap across consecutive times and both sides; not trades",
            "Current-membership universe and vendor-adjusted prices",
        ],
        "versions": {name: version(name) for name in ("catboost", "numpy", "scipy")},
        "source_sha256": sources,
    }


def run_study(data: Path, current: Path, outcomes: Path, output: Path) -> dict:
    from catboost import CatBoostClassifier

    if output.exists():
        raise ValueError("Scanner study output already exists; preserve it")
    manifest = read_manifest(data)
    dataset_sha256 = file_hash(data / "manifest.json")
    _, _, current_lock = load_frozen_model(current)
    outcome_report = json.loads((outcomes / "report.json").read_text(encoding="utf-8"))
    if current_lock["dataset_sha256"] != dataset_sha256:
        raise ValueError("Current model was not locked on this dataset")
    if file_hash(outcomes / "returns.npz") != outcome_report["returns_sha256"]:
        raise ValueError("15:00 outcome arrays do not match their report")
    charges = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(FIXED_CLIP_NOTIONAL))
    here = Path(__file__).parent
    sources = {name: file_hash(here / name) for name in SOURCES}
    output.mkdir(parents=True)
    protocol = protocol_document(data, current, outcomes, charges, sources)
    write_json(output / "protocol.json", protocol)
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)

    train = load_window(data, manifest, FRESH_PHASES["train"])
    validation = load_window(data, manifest, FRESH_PHASES["validation"])
    write_json(output / "feature_audit.json", feature_audit(train["features"]))
    reports = {}
    for name, candidate in candidates().items():
        print(json.dumps({"candidate": name, "status": "starting"}), flush=True)
        reports[name] = fit_candidate(
            train, validation, candidate, output / "trials" / name
        )
    del train
    selected = select_candidate(reports)
    final = output / "final"
    final.mkdir()
    shutil.copyfile(output / "trials" / selected / "best.cbm", final / "best.cbm")
    lock = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": file_hash(output / "protocol.json"),
        "selected": selected,
        "candidate": reports[selected]["candidate"],
        "weights_sha256": file_hash(final / "best.cbm"),
        "tree_count": reports[selected]["tree_count"],
        "columns": list(EXPECTED_COLUMNS),
        "output": "raw sigmoid probability",
        "validation_rows": len(validation["labels"]),
        "ranking": sorted(
            (
                {
                    "candidate": name,
                    "validation_log_loss": report["validation"]["log_loss"],
                    "tree_count": report["tree_count"],
                }
                for name, report in reports.items()
            ),
            key=lambda row: row["validation_log_loss"],
        ),
        "evaluation_opened": False,
    }
    write_json(final / "selection_lock.json", lock)
    if file_hash(data / "manifest.json") != dataset_sha256 or any(
        file_hash(here / name) != digest for name, digest in sources.items()
    ):
        raise ValueError("Inputs or source changed after the protocol was frozen")

    evaluation = load_phase(data, manifest, "evaluation")
    side = evaluation["features"][:, -1]
    with np.load(outcomes / "returns.npz", allow_pickle=False) as saved:
        outcome = {name: saved[name] for name in saved.files}
    with np.load(current / "evaluation_predictions.npz", allow_pickle=False) as saved:
        current_saved = {name: saved[name] for name in saved.files}
    if not all(
        np.array_equal(evaluation[name], outcome[name])
        for name in ("timestamps", "symbols", "labels")
    ) or not np.array_equal(side, outcome["side"]):
        raise ValueError("15:00 outcomes are not aligned with the evaluation shards")
    model, mapping, _ = load_frozen_model(current)
    logits = predict_logits(model, evaluation["features"], 1024)
    current_raw = expit(logits)
    current_calibrated = apply_mapping(mapping, logits, side)
    if not (
        np.array_equal(current_raw, outcome["raw_probability"])
        and np.array_equal(current_raw, current_saved["raw_probability"])
        and np.array_equal(current_calibrated, current_saved["probability"])
    ):
        raise ValueError("Current model no longer reproduces its saved forecasts")
    if file_hash(final / "best.cbm") != lock["weights_sha256"]:
        raise ValueError("Fresh model weights changed after the selection lock")
    fresh_model = CatBoostClassifier()
    fresh_model.load_model(str(final / "best.cbm"))
    fresh = expit(predict_logits(fresh_model, evaluation["features"], 1024))

    keep = scanner_window(evaluation["timestamps"])
    cohort = {
        "timestamps": evaluation["timestamps"][keep],
        "symbols": evaluation["symbols"][keep],
        "side": side[keep],
        "labels": evaluation["labels"][keep],
        "hit_by_1500": outcome["hit_by_1500"][keep],
        "return_at_1500": outcome["return_at_1500"][keep],
    }
    forecasts = {
        "fresh": fresh[keep],
        "current_calibrated": current_calibrated[keep],
        "current_raw": current_raw[keep],
    }
    np.savez_compressed(output / "evaluation_predictions.npz", **cohort, **forecasts)
    models = {
        name: scanner_report(
            values,
            cohort["labels"],
            cohort["hit_by_1500"],
            cohort["return_at_1500"],
            cohort["timestamps"],
            cohort["symbols"],
            cohort["side"],
            charges,
        )
        for name, values in forecasts.items()
    }
    paired = {
        reference: block_loss_comparison(
            cohort["labels"],
            forecasts["fresh"],
            forecasts[reference],
            cohort["timestamps"],
        )
        for reference in ("current_raw", "current_calibrated")
    }
    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "selection_lock_sha256": file_hash(final / "selection_lock.json"),
        "fresh_selected": selected,
        "fresh_preferred_by_predeclared_rule": all(
            values["log_loss"]["reduction_95pct_interval"][0] > 0
            for values in paired.values()
        ),
        "fresh_trials": reports,
        "cohort": {
            "rows": int(keep.sum()),
            "excluded_outside_window": int((~keep).sum()),
            "candle_times": int(keep.sum()) // 2,
            "sessions": len(
                np.unique((cohort["timestamps"] + IST_OFFSET_SECONDS) // 86400)
            ),
            "symbols": len(np.unique(cohort["symbols"])),
            "first_decision_ist": datetime.fromtimestamp(
                int(cohort["timestamps"].min()) + IST_OFFSET_SECONDS, UTC
            ).strftime("%Y-%m-%d %H:%M"),
            "last_decision_ist": datetime.fromtimestamp(
                int(cohort["timestamps"].max()) + IST_OFFSET_SECONDS, UTC
            ).strftime("%Y-%m-%d %H:%M"),
        },
        "charges_round_trip_fraction": charges,
        "models": models,
        "fresh_minus_current_paired_loss": paired,
        "evaluation_predictions_sha256": file_hash(
            output / "evaluation_predictions.npz"
        ),
        "evaluation_used_for_selection": False,
        "limitations": protocol["limitations"],
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study(args.data, args.current, args.outcomes, args.output)
    print(
        json.dumps(
            {
                "fresh_selected": report["fresh_selected"],
                "cohort": report["cohort"],
                "high_score": {
                    name: values["high_score"]
                    for name, values in report["models"].items()
                },
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
