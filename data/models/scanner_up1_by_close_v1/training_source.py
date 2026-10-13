"""Production scanner: P(+1% above the buy price before the session ends).

Same recipe as goal (a) of ``research_scanner_goals`` (tested January 2025 to
September 2026: AUC 0.732, bands within 2.4 points), refitted on newer data:
fit 2020-01 to 2025-06, early stopping 2025 H2, calibration fitted on 2026 H1
(method chosen on 2025 H2), holdout 2026-07 to 2026-09, never used for fitting or
choices. The tested model is scored on the same holdout for comparison.

Writes the bundle the live scanner loads: ``model.cbm`` and ``bundle.json``
(feature order, calibration, decision grid, session rules, metrics, hashes).
"""

from __future__ import annotations

import argparse
import json
import time as clock
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.e2e_research import apply_exclusions
from scripts.research_delivery_pipeline import (
    FEATURES,
    OPTIONAL,
    PERCENT,
    SESSION_MINUTES,
    as_date,
    logit,
    period_mask,
    prune,
)
from scripts.research_scanner_goals import (
    AUCTION_FROM,
    AUCTION_MINUTES,
    DECISIONS,
    assemble,
    fit_goal,
    goal_metrics,
)
from scripts.train_tp_only import file_hash, write_json

FIT = ("2020-01-01", "2025-06-30")
EARLY_STOP = ("2025-07-01", "2025-12-31")
CALIBRATION = ("2026-01-01", "2026-06-30")
HOLDOUT = ("2026-07-01", "2026-09-30")
PERIODS = (
    ("calibration_2026H1", *CALIBRATION),
    ("holdout_2026Q3", *HOLDOUT),
    ("holdout_closing_auction", "2026-08-03", "2026-09-30"),
)
VERSION = "scanner-up1-by-close-v1-2026-10-12"


def tested_probabilities(data: dict, kept: list[str], study: Path) -> np.ndarray:
    """The tested goal-(a) model with its Platt calibration, on 2026 rows only."""
    from catboost import CatBoostClassifier

    report = json.loads((study / "report.json").read_text("utf-8"))
    calibration = report["goals"]["a_up1_by_close"]["model"]["calibration"]
    if calibration["chosen"] != "platt":
        raise ValueError("The tested model was expected to use Platt calibration.")
    model = CatBoostClassifier()
    model.load_model(str(study / "scanner_a_up1_by_close.cbm"))
    shape = data["available"].shape
    day = np.broadcast_to(data["days"][:, None, None], shape)
    rows = np.flatnonzero(
        (data["available"] & period_mask(day, "2026-01-01", HOLDOUT[1])).reshape(-1)
    )
    flat = data["features"].reshape(-1, len(FEATURES))
    columns = [FEATURES.index(name) for name in kept]
    raw = model.predict_proba(flat[rows][:, columns])[:, 1]
    probability = np.full(shape[0] * shape[1] * shape[2], np.nan, np.float32)
    probability[rows] = 1 / (
        1
        + np.exp(
            -(calibration["platt_slope"] * logit(raw) + calibration["platt_intercept"])
        )
    )
    return probability.reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--tested", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; production bundles are never overwritten")
    started = clock.monotonic()
    tested_report = json.loads((args.tested / "report.json").read_text("utf-8"))
    data = assemble(args.candles, args.audit, args.workers)
    excluded = apply_exclusions(data, args.candles, args.audit)
    kept, _dropped = prune(data)
    if kept != tested_report["features"]["kept"]:
        raise ValueError("Feature pruning differs from the tested study.")
    probability, model, info = fit_goal(
        data,
        kept,
        0,
        args.workers,
        fit=FIT,
        early_stop=EARLY_STOP,
        calibration=CALIBRATION,
    )
    labels = data["labels"][..., 0]
    production = goal_metrics(probability, labels, data, PERIODS)
    tested = goal_metrics(
        tested_probabilities(data, kept, args.tested), labels, data, PERIODS
    )
    args.output.mkdir(parents=True)
    model.save_model(str(args.output / "model.cbm"))
    bundle = {
        "version": VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": "the price touches +1% above the buy price before the session's "
        "last continuous minute (buy = open of the minute after the decision close)",
        "features": kept,
        "percent_features": sorted(set(kept) & PERCENT),
        "optional_features": sorted(set(kept) & OPTIONAL),
        "decision_minutes": [int(d) for d in DECISIONS],
        "session": {
            "open": "09:15",
            "minutes": SESSION_MINUTES,
            "closing_auction_from": as_date(AUCTION_FROM).isoformat(),
            "closing_auction_minutes": AUCTION_MINUTES,
            "closing_auction_applies_to": "F&O stocks",
            "slot_feature": "minutes from the buy minute to the session's last "
            "continuous minute: session minutes - 1 - decision minute",
        },
        "calibration": info["calibration"],
        "trees": info["trees"],
        "periods": {
            "fit": FIT,
            "early_stop": EARLY_STOP,
            "calibration": CALIBRATION,
            "holdout": HOLDOUT,
        },
        "metrics": {"production": production, "tested_model_same_rows": tested},
        "importance_top10": info["importance_top10"],
        "excluded_stock_days": excluded,
        "tested_study": str(args.tested),
        "dataset": str(args.candles),
        "dataset_sha256": file_hash(args.candles / "dataset.json"),
        "model_sha256": file_hash(args.output / "model.cbm"),
        "source_sha256": file_hash(Path(__file__)),
        "runtime_minutes": round((clock.monotonic() - started) / 60, 1),
    }
    write_json(args.output / "bundle.json", bundle)
    for name, metrics in (("production", production), ("tested", tested)):
        for period in ("holdout_2026Q3", "holdout_closing_auction"):
            m = metrics[period]
            print(
                f"{name:10s} {period:24s} AUC {m['auc']:.3f} ECE "
                f"{m['calibration_error']:.4f} base {m['base_rate']:.3f} "
                f"mean p {m['mean_probability']:.3f}"
            )
    print("calibration", info["calibration"]["chosen"], "trees", info["trees"])
    print("runtime_minutes", bundle["runtime_minutes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
