"""CatBoost P(TP 0.6% before SL 0.2% by 15:00) with probability-band calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ai_trader.costs import FIXED_CLIP_NOTIONAL, ZERODHA_INTRADAY_EQUITY
from scripts.report_tp_catboost import independent_metrics
from scripts.research_exit_policy import build_universe, config_outcome, period_rows
from scripts.train_tp_only import EXPECTED_COLUMNS, file_hash, write_json

TP_BPS = 60
SL_BPS = 20
SEED = 20261011
THREADS = 12
BAND_EDGES = np.arange(1, 10) / 10
PERIODS = {
    "fit": ("2024-01-01", "2025-09-30"),
    "stop": ("2025-10-01", "2025-12-31"),
    "q1_2026": ("2026-01-01", "2026-03-31"),
    "apr_sep_2026": ("2026-04-01", "2026-09-28"),
}
EVALUATION = ("stop", "q1_2026", "apr_sep_2026")
SOURCES = (
    "research_tp_sl_classifier.py",
    "research_exit_policy.py",
    "train_tp_only.py",
    "report_tp_catboost.py",
)


def outcomes(universe, rows: slice) -> dict[str, np.ndarray]:
    returns, _, kinds = config_outcome(universe, rows, TP_BPS, SL_BPS, None)
    mask = universe.available[rows]
    return {
        "features": universe.features[rows][mask].reshape(-1, len(EXPECTED_COLUMNS)),
        "labels": (kinds[mask] == 0).reshape(-1).astype(np.int64),
        "gains": returns[mask].reshape(-1),
    }


def band_table(
    probability: np.ndarray, labels: np.ndarray, gains: np.ndarray, charges: float
) -> list[dict]:
    bins = np.searchsorted(BAND_EDGES, probability, side="right")
    table = []
    for index in range(10):
        chosen = bins == index
        calls = int(chosen.sum())
        row = {"band": f"{10 * index}-{10 * (index + 1)}%", "calls": calls}
        if calls:
            row.update(
                share_percent=100 * calls / len(probability),
                mean_probability_percent=100 * float(probability[chosen].mean()),
                tp_first_percent=100 * float(labels[chosen].mean()),
                mean_gross_return_percent=100 * float(gains[chosen].mean()),
                mean_net_return_percent=100 * (float(gains[chosen].mean()) - charges),
            )
        table.append(row)
    return table


def threshold_table(
    probability: np.ndarray,
    labels: np.ndarray,
    gains: np.ndarray,
    charges: float,
    sessions: int,
) -> list[dict]:
    move = (TP_BPS + SL_BPS) / 10_000
    cutoffs = {
        "gross_break_even": SL_BPS / 10_000 / move,
        "0.30": 0.30,
        "net_break_even": (SL_BPS / 10_000 + charges) / move,
        "0.40": 0.40,
        "0.50": 0.50,
    }
    table = []
    for name, cutoff in cutoffs.items():
        chosen = probability >= cutoff
        calls = int(chosen.sum())
        row = {"rule": name, "min_probability": cutoff, "calls": calls}
        if calls:
            row.update(
                calls_per_session=calls / sessions,
                tp_first_percent=100 * float(labels[chosen].mean()),
                mean_gross_return_percent=100 * float(gains[chosen].mean()),
                mean_net_return_percent=100 * (float(gains[chosen].mean()) - charges),
            )
        table.append(row)
    return table


def run(data: Path, candles: Path, output: Path) -> dict:
    from catboost import CatBoostClassifier

    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    charges = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(FIXED_CLIP_NOTIONAL))
    here = Path(__file__).parent
    sources = {name: file_hash(here / name) for name in SOURCES}
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "label": (
                f"1 if +/-{TP_BPS / 100}% TP is touched before the {SL_BPS / 100}% "
                "stop and by the 14:59-15:00 candle; stop first on a shared candle"
            ),
            "inputs": list(EXPECTED_COLUMNS),
            "calls": "five-minute grid 09:40-13:30 IST, entry at decision close",
            "periods": PERIODS,
            "model": (
                "CatBoostClassifier depth 4, learning rate 0.03, L2 20, Bernoulli "
                "0.8, 254 borders, 2000 trees, early stopping 100 on stop period"
            ),
            "break_even": (
                "gross P = SL/(TP+SL); net P = (SL+charges)/(TP+SL), ignoring "
                "time exits and gap fills, which are in the realized returns"
            ),
            "verdict_rule": (
                "tradable only if calls at or above the net break-even probability "
                "have positive mean net realized return in both 2026 periods"
            ),
            "charges_round_trip_fraction": charges,
            "versions": {name: version(name) for name in ("catboost", "numpy")},
            "source_sha256": sources,
        },
    )
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)

    universe, coverage = build_universe(data, candles, None)
    sets = {
        name: outcomes(universe, period_rows(universe, *bounds))
        for name, bounds in PERIODS.items()
    }
    fit, stop = sets["fit"], sets["stop"]
    began = time.perf_counter()
    model = CatBoostClassifier(
        iterations=2000,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=20.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        border_count=254,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_strength=1.0,
        random_seed=SEED,
        thread_count=THREADS,
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        fit["features"], fit["labels"], eval_set=(stop["features"], stop["labels"])
    )
    model.save_model(str(output / "best.cbm"))
    side = fit["features"][:, -1]
    priors = {
        direction: float(fit["labels"][side == direction].mean())
        for direction in (-1, 1)
    }
    importance = model.get_feature_importance()
    periods = {}
    arrays = {}
    for name in EVALUATION:
        values = sets[name]
        probability = model.predict_proba(values["features"])[:, 1]
        sides = values["features"][:, -1]
        prior = np.where(sides == 1, priors[1], priors[-1])
        sessions = len(np.unique(universe.days[period_rows(universe, *PERIODS[name])]))
        arrays[name] = probability
        periods[name] = {
            "calls": len(probability),
            "sessions": sessions,
            "model": independent_metrics(values["labels"], probability),
            "base_rate_control": independent_metrics(values["labels"], prior),
            "auc_by_side": {
                str(int(direction)): independent_metrics(
                    values["labels"][sides == direction],
                    probability[sides == direction],
                )["roc_auc"]
                for direction in (1, -1)
            },
            "probability_quantiles_percent": {
                str(level): 100 * float(np.quantile(probability, level))
                for level in (0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
            },
            "bands": band_table(
                probability, values["labels"], values["gains"], charges
            ),
            "thresholds": threshold_table(
                probability, values["labels"], values["gains"], charges, sessions
            ),
        }
    np.savez_compressed(output / "predictions.npz", **arrays)
    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "coverage": {k: v for k, v in coverage.items() if k != "symbols"},
        "model": {
            "tree_count": model.tree_count_,
            "fit_seconds": time.perf_counter() - began,
            "weights_sha256": file_hash(output / "best.cbm"),
            "fit_rows": len(fit["labels"]),
            "stop_rows": len(stop["labels"]),
            "fit_side_base_rates_percent": {str(k): 100 * v for k, v in priors.items()},
            "feature_importance": {
                name: float(value)
                for name, value in zip(EXPECTED_COLUMNS, importance, strict=True)
            },
        },
        "periods": periods,
        "charges_round_trip_fraction": charges,
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data, args.candles, args.output)
    print(json.dumps(report["model"], default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
