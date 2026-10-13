"""Walk-forward evaluation of the delivery TP-before-SL CatBoost (research only).

Uses the verified dataset from research_delivery_catboost.py. Each month from Jan 2025
is predicted by models trained only on decisions whose outcomes resolved before that
month began (inner time-split early stopping, then refit). Probability accuracy is
judged against causal base-rate controls; ranking skill within each day is reported
separately from day-level timing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from scripts.report_tp_catboost import independent_metrics
from scripts.research_delivery_catboost import GROUPS
from scripts.train_tp_only import file_hash, write_json

FIRST_MONTH, LAST_MONTH = "2025-01", "2026-09"
EVALUATE_FROM = "2025-04"
INNER_SESSIONS = 63
RECENT_SESSIONS = 63
PLATT_MONTHS = 3
BOOTSTRAP = 2000
BLOCK_DAYS = 5
SEED = 20261011
THREADS = 12
CONFIG = {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 30.0}
STOCK_ONLY = (
    GROUPS["trend"]
    + GROUPS["volatility"]
    + GROUPS["volume"]
    + (
        "rank_ret_5d",
        "rank_ret_21d",
        "rank_rvol",
        "rank_atr",
    )
)
CANDIDATES = {
    "all_features": tuple(n for names in GROUPS.values() for n in names),
    "stock_only": STOCK_ONLY,
    "volatility_only": ("atr_14", "rv_21", "range_today", "vix_1500"),
}
TOP_K = (1, 3, 5)
SCREEN_T = 2.0
SCREEN_MAX = 8
SCREENED = ("screened_catboost", "screened_logistic")
SOURCES = (
    "research_delivery_walkforward.py",
    "research_delivery_catboost.py",
    "report_tp_catboost.py",
    "train_tp_only.py",
)


def months_between(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    result = []
    while f"{year:04d}-{month:02d}" <= last:
        result.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def fit(features, labels, inner_mask, train_mask, iterations=None):
    from catboost import CatBoostClassifier

    common = {
        "loss_function": "Logloss",
        "border_count": 254,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "random_seed": SEED,
        "thread_count": THREADS,
        "allow_writing_files": False,
        "verbose": False,
        **CONFIG,
    }
    if iterations is None:
        model = CatBoostClassifier(
            iterations=3000, early_stopping_rounds=200, use_best_model=True, **common
        )
        model.fit(
            features[train_mask],
            labels[train_mask],
            eval_set=(features[inner_mask], labels[inner_mask]),
        )
        return model
    model = CatBoostClassifier(iterations=max(iterations, 1), **common)
    model.fit(features[train_mask], labels[train_mask])
    return model


def screen_features(
    features: np.ndarray, labels: np.ndarray, days: np.ndarray, names: list[str]
) -> list[tuple[str, float]]:
    from scipy.stats import rankdata

    groups = defaultdict(list)
    for index, day in enumerate(days):
        groups[day].append(index)
    kept = []
    for column, name in enumerate(names):
        values = []
        for rows in groups.values():
            x, y = features[rows, column], labels[rows]
            usable = np.isfinite(x)
            if usable.sum() >= 20 and 0 < y[usable].sum() < usable.sum():
                values.append(np.corrcoef(rankdata(x[usable]), y[usable])[0, 1])
        values = np.asarray(values)
        if len(values) > 20 and values.std(ddof=1) > 0:
            t = values.mean() / values.std(ddof=1) * np.sqrt(len(values))
            if abs(t) >= SCREEN_T:
                kept.append((name, float(t)))
    return sorted(kept, key=lambda item: -abs(item[1]))[:SCREEN_MAX]


def logistic_fit_predict(train_x, train_y, test_x) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    mean = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    scale[scale == 0] = 1

    def standard(values):
        return np.nan_to_num((values - mean) / scale)

    model = LogisticRegression(C=0.05, max_iter=2000).fit(standard(train_x), train_y)
    return model.predict_proba(standard(test_x))[:, 1]


def within_day_auc(days: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    values = {}
    for day in np.unique(days):
        chosen = days == day
        if 0 < labels[chosen].sum() < chosen.sum():
            values[day] = roc_auc_score(labels[chosen], scores[chosen])
    return values


def block_interval(values: np.ndarray) -> list[float]:
    generator = np.random.default_rng(SEED)
    count = len(values)
    blocks = -(-count // BLOCK_DAYS)
    starts = generator.integers(count, size=(BOOTSTRAP, blocks))
    index = (starts[:, :, None] + np.arange(BLOCK_DAYS)) % count
    means = values[index.reshape(BOOTSTRAP, -1)[:, :count]].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def daily_top(days, symbols, scores, labels, gross, net, count) -> dict:
    picks, baseline = [], []
    for day in np.unique(days):
        rows = np.flatnonzero(days == day)
        order = rows[np.lexsort((symbols[rows], -scores[rows]))][:count]
        picks.extend(order.tolist())
        baseline.append(labels[rows].mean())
    picks = np.asarray(picks)
    return {
        "picks": len(picks),
        "tp_first_percent": 100 * float(labels[picks].mean()),
        "day_average_tp_first_percent": 100 * float(np.mean(baseline)),
        "mean_gross_percent": 100 * float(gross[picks].mean()),
        "mean_net_percent": 100 * float(net[picks].mean()),
    }


def bands(probability, labels, gross, net) -> list[dict]:
    ids = np.searchsorted(np.arange(1, 10) / 10, probability, side="right")
    table = []
    for band in range(10):
        chosen = ids == band
        row = {"band": f"{10 * band}-{10 * band + 10}%", "calls": int(chosen.sum())}
        if chosen.any():
            row.update(
                mean_probability_percent=100 * float(probability[chosen].mean()),
                tp_first_percent=100 * float(labels[chosen].mean()),
                mean_gross_percent=100 * float(gross[chosen].mean()),
                mean_net_percent=100 * float(net[chosen].mean()),
            )
        table.append(row)
    return table


def platt_map(raw: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logit(np.clip(raw, 1e-6, 1 - 1e-6))[:, None], labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_map(mapping: tuple[float, float], raw: np.ndarray) -> np.ndarray:
    from scipy.special import expit, logit

    slope, intercept = mapping
    return expit(slope * logit(np.clip(raw, 1e-6, 1 - 1e-6)) + intercept)


def day_log_loss(labels, probability, daily, ordered) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    loss = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    return np.asarray([loss[daily[d]].mean() for d in ordered])


def evaluate(name, probability, labels, days, symbols, gross, net, controls) -> dict:
    metrics = independent_metrics(labels, probability)
    daily = defaultdict(list)
    for index, day in enumerate(days):
        daily[day].append(index)
    ordered = sorted(daily)
    loss = day_log_loss(labels, probability, daily, ordered)
    gains = {}
    for control, values in controls.items():
        gain = day_log_loss(labels, values, daily, ordered) - loss
        gains[control] = {"mean": float(gain.mean()), "ci95": block_interval(gain)}
    auc = within_day_auc(days, labels, probability)
    auc_values = np.asarray([auc[d] for d in ordered if d in auc])
    day_mean_p = np.asarray([probability[daily[d]].mean() for d in ordered])
    day_rate = np.asarray([labels[daily[d]].mean() for d in ordered])
    return {
        "candidate": name,
        **metrics,
        "log_loss_gain_per_day": gains,
        "within_day_auc_mean": float(auc_values.mean()),
        "within_day_auc_ci95": block_interval(auc_values),
        "day_timing_correlation": float(np.corrcoef(day_mean_p, day_rate)[0, 1])
        if day_mean_p.std() > 0
        else None,
        "top_k": {
            str(k): daily_top(days, symbols, probability, labels, gross, net, k)
            for k in TOP_K
        },
    }


def run(study: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    here = Path(__file__).parent
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": str(study / "dataset.npz"),
            "dataset_sha256": file_hash(study / "dataset.npz"),
            "walk_forward": f"months {FIRST_MONTH}..{LAST_MONTH}; train on decisions "
            "whose exit date precedes the month; last "
            f"{INNER_SESSIONS} sessions (purged) give early stopping, then refit on "
            "the whole training window with that tree count",
            "config": CONFIG,
            "candidates": CANDIDATES,
            "controls": "expanding training base rate; last "
            f"{RECENT_SESSIONS}-session base rate",
            "calibration": f"rolling Platt fit on the previous {PLATT_MONTHS} months "
            "of out-of-sample predictions (evaluation from "
            f"{EVALUATE_FROM})",
            "decision_rule": "a candidate has useful probability skill only if its "
            "log loss beats BOTH controls with a day-block 95% interval above zero; "
            "ranking skill needs a within-day AUC interval above 0.5",
            "config_note": "depth 4 / l2 30 was the best of six configs on the "
            "Jul-Dec 2025 validation of the single-split study (differences < 0.003 "
            "log loss), a mild overlap acknowledged here",
            "versions": {n: version(n) for n in ("catboost", "numpy", "scikit-learn")},
            "source_sha256": {name: file_hash(here / name) for name in SOURCES},
        },
    )
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)
    data = np.load(study / "dataset.npz", allow_pickle=False)
    names = [str(n) for n in data["feature_names"]]
    features, labels = data["features"], data["labels"].astype(np.int64)
    dates, exits = data["dates"].astype(str), data["exit_dates"].astype(str)
    symbols, gross, net = data["symbols"].astype(str), data["gross"], data["net"]
    sessions = np.unique(dates)
    predictions = {
        name: np.full(len(labels), np.nan) for name in (*CANDIDATES, *SCREENED)
    }
    controls = {
        "expanding_base_rate": np.full(len(labels), np.nan),
        "recent_base_rate": np.full(len(labels), np.nan),
    }
    trees = defaultdict(dict)
    screened_by_month = {}
    stock_columns = [names.index(c) for c in STOCK_ONLY]
    for month in months_between(FIRST_MONTH, LAST_MONTH):
        start = f"{month}-01"
        target = np.char.startswith(dates, month)
        if not target.any():
            continue
        train = exits < start
        known = sessions[sessions < start]
        inner_start = known[-INNER_SESSIONS]
        inner = train & (dates >= inner_start)
        early = train & (exits < inner_start)
        recent = train & (dates >= known[-RECENT_SESSIONS])
        controls["expanding_base_rate"][target] = labels[train].mean()
        controls["recent_base_rate"][target] = labels[recent].mean()
        for name, columns in CANDIDATES.items():
            matrix = features[:, [names.index(c) for c in columns]]
            stopper = fit(matrix, labels, inner, early)
            final = fit(matrix, labels, None, train, stopper.tree_count_)
            predictions[name][target] = final.predict_proba(matrix[target])[:, 1]
            trees[name][month] = stopper.tree_count_
        kept = screen_features(
            features[train][:, stock_columns],
            labels[train],
            dates[train],
            list(STOCK_ONLY),
        )
        screened_by_month[month] = kept
        if kept:
            matrix = features[:, [names.index(n) for n, _ in kept]]
            stopper = fit(matrix, labels, inner, early)
            final = fit(matrix, labels, None, train, stopper.tree_count_)
            predictions["screened_catboost"][target] = final.predict_proba(
                matrix[target]
            )[:, 1]
            trees["screened_catboost"][month] = stopper.tree_count_
            predictions["screened_logistic"][target] = logistic_fit_predict(
                matrix[train], labels[train], matrix[target]
            )
        else:
            for name in SCREENED:
                predictions[name][target] = labels[train].mean()
            trees["screened_catboost"][month] = 0
        print(
            json.dumps(
                {"month": month, "trees": {k: v[month] for k, v in trees.items()}}
            )
        )
    evaluated = np.char.startswith(dates, "2025") | np.char.startswith(dates, "2026")
    month_of = dates.astype("<U7")
    months = sorted(set(month_of[evaluated].tolist()))
    calibrated = {name: np.full(len(labels), np.nan) for name in predictions}
    mappings = defaultdict(dict)
    for index, month in enumerate(months):
        if month < EVALUATE_FROM or index < PLATT_MONTHS:
            continue
        window = np.isin(month_of, months[index - PLATT_MONTHS : index])
        window &= exits < f"{month}-01"
        target = month_of == month
        for name in predictions:
            mapping = platt_map(predictions[name][window], labels[window])
            calibrated[name][target] = apply_map(mapping, predictions[name][target])
            mappings[name][month] = mapping
    scope = (month_of >= EVALUATE_FROM) & evaluated
    days, syms = dates[scope], symbols[scope]
    y, g, n = labels[scope], gross[scope], net[scope]
    scores = {f"{k}_raw": v[scope] for k, v in predictions.items()}
    scores |= {f"{k}_platt": v[scope] for k, v in calibrated.items()}
    scores |= {k: v[scope] for k, v in controls.items()}
    reference = {k: scores[k] for k in controls}
    results = {}
    for key, values in scores.items():
        results[key] = evaluate(key, values, y, days, syms, g, n, reference)
        results[key]["bands"] = bands(values, y, g, n)
    halves = {
        label: {
            key: independent_metrics(y[part], values[part])
            for key, values in scores.items()
        }
        for label, part in (
            ("2025_apr_dec", np.char.startswith(days, "2025")),
            ("2026_jan_sep", np.char.startswith(days, "2026")),
        )
    }
    np.savez_compressed(
        output / "walk_forward_predictions.npz",
        dates=days,
        symbols=syms,
        labels=y,
        gross=g,
        net=n,
        **scores,
    )
    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "evaluated_rows": int(scope.sum()),
        "evaluated_sessions": int(len(np.unique(days))),
        "tp_first_rate": float(y.mean()),
        "trees_by_month": trees,
        "screened_features_by_month": screened_by_month,
        "results": results,
        "halves": halves,
        "platt_mappings": mappings,
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.study, args.output)
    print(
        json.dumps(
            {"rows": report["evaluated_rows"], "sessions": report["evaluated_sessions"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
