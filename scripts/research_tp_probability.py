"""Chronological research into calibrated, direction-signed TP probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, minutes_since_open
from ai_trader.features import FeatureEngine, FeatureSnapshot
from ai_trader.history import CandleStore
from ai_trader.market import Candle
from ai_trader.replay.models import ExitReason, ReplayResult, SimulatedTrade
from ai_trader.scanner import Direction
from ai_trader.strategy import FixedAtrStop, StrategyConfig
from scripts.research_signed_score import _hash
from scripts.research_tp_policy import (
    TP_CONTEXT_NAMES,
    TP_CURRENT_NAMES,
    build_cases,
    choose_policy,
    execute_cached,
    fit_tp,
    precision_summary,
    previous_sessions,
    probability,
    signed_tp_probability,
    tp_inputs,
)

ActionKey = tuple[datetime, Direction]


def _logit(probability: float) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    bounded = min(1 - 1e-9, max(1e-9, probability))
    return math.log(bounded / (1 - bounded))


@dataclass(frozen=True, slots=True)
class PlattCalibration:
    slope: float = 1.0
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.slope, self.intercept)):
            raise ValueError("Calibration parameters must be finite")
        if self.slope < 0:
            raise ValueError("Calibration must preserve probability ordering")

    def apply(self, probability: float) -> float:
        raw = self.slope * _logit(probability) + self.intercept
        return (1 + math.tanh(raw / 2)) / 2


def fit_calibration(
    samples: Sequence[tuple[date, float, int]],
    *,
    through: date,
) -> PlattCalibration:
    from scipy.optimize import minimize

    training = [
        (day, _logit(estimate), outcome)
        for day, estimate, outcome in samples
        if day <= through
    ]
    if not training or any(outcome not in (0, 1) for _, _, outcome in training):
        raise ValueError("Calibration requires past binary outcomes")
    counts = Counter(day for day, _, _ in training)

    def objective(parameters):
        slope, intercept = parameters
        loss = 0.005 * ((slope - 1) ** 2 + intercept**2)
        gradient = [0.01 * (slope - 1), 0.01 * intercept]
        for day, logit, outcome in training:
            weight = 1 / (len(counts) * counts[day])
            raw = slope * logit + intercept
            estimate = (1 + math.tanh(raw / 2)) / 2
            loss += weight * (
                max(raw, 0) - outcome * raw + math.log1p(math.exp(-abs(raw)))
            )
            residual = weight * (estimate - outcome)
            gradient[0] += residual * logit
            gradient[1] += residual
        return loss, gradient

    result = minimize(
        objective,
        [1.0, 0.0],
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, 10), (-20, 20)],
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success:
        raise ValueError(f"Calibration failed to converge: {result.message}")
    return PlattCalibration(*map(float, result.x))


def probability_metrics(pairs: Sequence[tuple[float, int]]) -> dict[str, Any]:
    for estimate, outcome in pairs:
        _logit(estimate)
        if outcome not in (0, 1):
            raise ValueError("Probability metrics require binary outcomes")
    if not pairs:
        return {
            "observations": 0,
            "mean_probability": None,
            "tp_rate": None,
            "gap": None,
            "brier": None,
            "log_loss": None,
            "ece": None,
            "roc_auc": None,
            "bins": [],
        }
    bins = []
    for index in range(10):
        bucket = [
            (estimate, outcome)
            for estimate, outcome in pairs
            if min(9, int(estimate * 10)) == index
        ]
        if bucket:
            bins.append(
                {
                    "lower": index / 10,
                    "upper": (index + 1) / 10,
                    "observations": len(bucket),
                    "mean_probability": statistics.fmean(value for value, _ in bucket),
                    "tp_rate": statistics.fmean(value for _, value in bucket),
                }
            )
    predicted = statistics.fmean(estimate for estimate, _ in pairs)
    actual = statistics.fmean(outcome for _, outcome in pairs)
    positive_count = sum(outcome for _, outcome in pairs)
    negative_count = len(pairs) - positive_count
    grouped: dict[float, list[int]] = {}
    for estimate, outcome in pairs:
        counts = grouped.setdefault(estimate, [0, 0])
        counts[outcome] += 1
    negatives_below = 0
    ordered_pairs = 0.0
    for _, (negatives, positives) in sorted(grouped.items()):
        ordered_pairs += positives * (negatives_below + negatives / 2)
        negatives_below += negatives
    return {
        "observations": len(pairs),
        "mean_probability": predicted,
        "tp_rate": actual,
        "gap": predicted - actual,
        "roc_auc": ordered_pairs / (positive_count * negative_count)
        if positive_count and negative_count
        else None,
        "brier": statistics.fmean(
            (estimate - outcome) ** 2 for estimate, outcome in pairs
        ),
        "log_loss": statistics.fmean(
            -math.log(max(1e-9, estimate if outcome else 1 - estimate))
            for estimate, outcome in pairs
        ),
        "ece": sum(
            bucket["observations"] * abs(bucket["mean_probability"] - bucket["tp_rate"])
            for bucket in bins
        )
        / len(pairs),
        "bins": bins,
    }


def phase_for(day: date, protocol: Mapping[str, Any]) -> str | None:
    matches = [
        name
        for name, (start, end) in protocol["phases"].items()
        if date.fromisoformat(start) <= day <= date.fromisoformat(end)
    ]
    if len(matches) > 1:
        raise ValueError("Research phases cannot overlap")
    return matches[0] if matches else None


def model_training_sessions(
    phases: Mapping[str, Sequence[date]], protocol: Mapping[str, Any]
) -> tuple[date, ...]:
    refit = protocol.get("refit_model_selection", True)
    if type(refit) is not bool:
        raise ValueError("refit_model_selection must be a boolean")
    return tuple(phases["fit"]) + (tuple(phases["model_selection"]) if refit else ())


def _write_json(path: Path, document: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_tape(
    root: Path, protocol: dict[str, Any]
) -> tuple[tuple[Candle, ...], dict[str, Any]]:
    repository = root.resolve().parents[1]
    data_path = repository / "data/research" / root.name / "NSE_RELIANCE.csv"
    manifest_path = root / "data_manifest.json"
    instrument = Instrument("NSE", "RELIANCE")
    first = datetime(2026, 7, 1, 9, 15, tzinfo=INDIA_TIMEZONE)
    boundary = datetime(2026, 8, 31, 9, 15, tzinfo=INDIA_TIMEZONE)
    last = datetime(2026, 9, 28, 15, 15, tzinfo=INDIA_TIMEZONE)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_sha256"] != _hash(root / "protocol.json"):
            raise ValueError("Frozen probability-study protocol changed")
        if manifest["candles_sha256"] != _hash(data_path):
            raise ValueError("Frozen probability-study candles changed")
        return CandleStore(data_path.parent).load(instrument, first, last), manifest
    if data_path.exists():
        raise ValueError("Unmanifested snapshot exists; do not overwrite it")
    frozen_root = repository / "data/research/reliance_sep2026"
    parent_manifest_path = root / protocol["frozen_september_manifest"]
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    for name, expected in parent["candle_files_sha256"].items():
        if _hash(frozen_root / name) != expected:
            raise ValueError("The frozen September source changed")
    provenance = {}
    if "source_snapshot_directory" in protocol:
        source = (root / protocol["source_snapshot_directory"]).resolve()
        source_manifest_path = root / protocol["source_data_manifest"]
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if _hash(source / "NSE_RELIANCE.csv") != source_manifest["candles_sha256"]:
            raise ValueError("The combined source snapshot changed")
        earlier = CandleStore(source).load(instrument, first, last)
        later = ()
        provenance["source_data_manifest_sha256"] = _hash(source_manifest_path)
    else:
        source = repository / "data/candles"
        earlier = CandleStore(source).load(instrument, first, boundary)
        later = CandleStore(frozen_root).load(instrument, boundary, last)
    regular = tuple(
        sorted(
            (
                bar
                for bar in (*earlier, *later)
                if time(9, 15) <= bar.start_time.time() < time(15, 30)
            ),
            key=lambda bar: bar.start_time,
        )
    )
    if len({bar.start_time for bar in regular}) != len(regular):
        raise ValueError("Duplicate minute in combined research tape")
    sessions = sorted({bar.start_time.date() for bar in regular})
    counts = Counter(phase_for(day, protocol) for day in sessions)
    if dict(counts) != protocol["expected_session_counts"]:
        raise ValueError(f"Research session coverage mismatch: {dict(counts)}")
    for day in sessions:
        bars = [bar for bar in regular if bar.start_time.date() == day]
        opening = datetime.combine(day, time(9, 15), tzinfo=INDIA_TIMEZONE)
        expected = {opening + timedelta(minutes=index) for index in range(360)}
        if not expected.issubset({bar.start_time for bar in bars}):
            raise ValueError(f"Incomplete pre-cutoff path on {day}")
        if any(
            price % Decimal("0.1")
            for bar in bars
            for price in (bar.open, bar.high, bar.low, bar.close)
        ):
            raise ValueError(f"Off-grid price on {day}")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("start_time", "open", "high", "low", "close", "volume"))
        writer.writerows(
            (
                bar.start_time.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume if bar.volume is not None else "",
            )
            for bar in regular
        )
    manifest = {
        **provenance,
        "protocol_sha256": _hash(root / "protocol.json"),
        "candles_sha256": _hash(data_path),
        "earlier_source_sha256": _hash(source / "NSE_RELIANCE.csv"),
        "september_manifest_sha256": _hash(parent_manifest_path),
        "baseline_strategy_sha256": StrategyConfig.load(
            root / protocol["baseline_strategy"]
        ).fingerprint,
        "sessions": [str(day) for day in sessions],
        "phases": {
            name: [str(day) for day in sessions if phase_for(day, protocol) == name]
            for name in protocol["phases"]
        },
        "regular_candles": len(regular),
        "pre_cutoff_minutes_per_session": 360,
        "unknown_volume_bars": sum(bar.volume is None for bar in regular),
    }
    _write_json(manifest_path, manifest, overwrite=False)
    return regular, manifest


def barrier_and_speed(trades: Sequence[SimulatedTrade]) -> dict[str, Any]:
    def distribution(values: Sequence[float]) -> dict[str, float | None]:
        return {
            "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "max": max(values) if values else None,
        }

    hits = [trade for trade in trades if trade.exit_reason is ExitReason.TARGET]
    return {
        "sl_percent": distribution(
            [
                100 * float(abs(trade.stop_price / trade.entry_price - 1))
                for trade in trades
            ]
        ),
        "tp_percent": distribution(
            [
                100 * float(abs(trade.target_price / trade.entry_price - 1))
                for trade in trades
            ]
        ),
        "successful_tp_minutes": distribution(
            [float(trade.holding_minutes) for trade in hits]
        ),
        "tp_within_minutes": {
            str(horizon): {
                "hits": sum(trade.holding_minutes <= horizon for trade in hits),
                "fraction_of_all_actions": sum(
                    trade.holding_minutes <= horizon for trade in hits
                )
                / len(trades)
                if trades
                else None,
            }
            for horizon in (5, 15, 30)
        },
    }


def _action_metrics(pairs: Sequence[tuple[float, SimulatedTrade]]) -> dict[str, Any]:
    result = probability_metrics(
        [
            (estimate, int(trade.exit_reason is ExitReason.TARGET))
            for estimate, trade in pairs
        ]
    )
    result["outcomes"] = barrier_and_speed([trade for _, trade in pairs])
    for bucket in result["bins"]:
        selected = [
            trade
            for estimate, trade in pairs
            if min(9, int(estimate * 10)) == round(bucket["lower"] * 10)
        ]
        bucket["outcomes"] = barrier_and_speed(selected)
    return result


def coefficient_profile(
    names: Sequence[str],
    coefficients: Sequence[float],
    training: Mapping[ActionKey, tuple[float, ...]],
) -> dict[str, Any]:
    sampled = {
        key: values
        for key, values in training.items()
        if (int(minutes_since_open(key[0])) - 1) % 5 == 0
    }
    if not sampled or len(names) != len(coefficients):
        raise ValueError("Coefficient profiling needs matching names and training data")
    counts = Counter(moment.date() for moment, _ in sampled)
    weights = {key: 1 / (len(counts) * counts[key[0].date()]) for key in sampled}
    result = []
    for index, (name, coefficient) in enumerate(
        zip(names[1:], coefficients[1:], strict=True)
    ):
        mean = sum(weights[key] * values[index] for key, values in sampled.items())
        effect = sum(
            weights[key] * abs(coefficient * (values[index] - mean))
            for key, values in sampled.items()
        )
        result.append(
            {
                "feature": name,
                "coefficient": coefficient,
                "training_mean": mean,
                "mean_absolute_centered_logit_effect": effect,
            }
        )
    return {
        "intercept": coefficients[0],
        "sampled_actions": len(sampled),
        "sessions": len(counts),
        "features": sorted(
            result, key=lambda item: -item["mean_absolute_centered_logit_effect"]
        ),
        "meaning": (
            "Training-distribution contribution scale, not causal importance. "
            "Correlated features may substitute; calibration can rescale logit effects."
        ),
    }


def _group_metrics(
    estimates: Mapping[ActionKey, float],
    cases: Mapping[ActionKey, SimulatedTrade],
) -> dict[str, Any]:
    pairs = {
        key: (value, int(cases[key].exit_reason is ExitReason.TARGET))
        for key, value in estimates.items()
        if key in cases
    }
    result = _action_metrics([(estimates[key], cases[key]) for key in pairs])
    result["by_direction"] = {
        direction.value: _action_metrics(
            [(estimates[key], cases[key]) for key in pairs if key[1] is direction]
        )
        for direction in Direction
    }
    days = sorted({key[0].date() for key in pairs})
    result["sessions"] = len(days)
    result["equal_day_log_loss"] = (
        statistics.fmean(
            probability_metrics(
                [pair for key, pair in pairs.items() if key[0].date() == day]
            )["log_loss"]
            for day in days
        )
        if days
        else None
    )
    return result


def _predict(
    inputs: Mapping[ActionKey, tuple[float, ...]],
    coefficients: Sequence[float],
    sessions: Sequence[date],
) -> dict[ActionKey, float]:
    return {
        key: probability(values, coefficients)
        for key, values in inputs.items()
        if key[0].date() in sessions
    }


def _correct(
    raw: Mapping[ActionKey, float],
    calibrators: Mapping[Direction, PlattCalibration],
) -> dict[ActionKey, float]:
    return {key: calibrators[key[1]].apply(value) for key, value in raw.items()}


def _signed(estimates: Mapping[ActionKey, float]) -> dict[datetime, Decimal]:
    return {
        moment: signed_tp_probability(
            estimates[(moment, Direction.LONG)], estimates[(moment, Direction.SHORT)]
        )
        for moment, _ in estimates
    }


def trade_diagnostics(
    result: ReplayResult, base_rates: Mapping[Direction, float]
) -> dict[str, Any]:
    summary = precision_summary(result)
    summary["outcomes"] = barrier_and_speed(result.trades)

    def diagnostics(trades: Sequence[SimulatedTrade]) -> dict[str, Any]:
        result = _action_metrics([(abs(float(trade.score)), trade) for trade in trades])
        result["constant_control"] = probability_metrics(
            [
                (
                    base_rates[trade.direction],
                    int(trade.exit_reason is ExitReason.TARGET),
                )
                for trade in trades
            ]
        )
        return result

    summary["probability"] = diagnostics(result.trades)
    summary["by_direction"] = {}
    for direction in Direction:
        side = tuple(trade for trade in result.trades if trade.direction is direction)
        details = precision_summary(replace(result, trades=side))
        details.pop("trades")
        details.pop("by_direction")
        details["probability"] = diagnostics(side)
        details["outcomes"] = barrier_and_speed(side)
        details["stop_fraction_range"] = (
            [
                min(
                    float(abs(trade.stop_price / trade.entry_price - 1))
                    for trade in side
                ),
                max(
                    float(abs(trade.stop_price / trade.entry_price - 1))
                    for trade in side
                ),
            ]
            if side
            else None
        )
        summary["by_direction"][direction.value] = details
    summary["by_day"] = {}
    for day in result.sessions:
        trades = tuple(trade for trade in result.trades if trade.session == day)
        details = probability_metrics(
            [
                (abs(float(trade.score)), int(trade.exit_reason is ExitReason.TARGET))
                for trade in trades
            ]
        )
        details["net_rupees"] = float(
            sum((trade.net_rupees for trade in trades), Decimal(0))
        )
        summary["by_day"][day.isoformat()] = details
    return summary


def _threshold_rows(
    estimates: Mapping[ActionKey, float],
    cases: Mapping[ActionKey, SimulatedTrade],
    snapshots: Mapping[datetime, FeatureSnapshot],
    strategy: StrategyConfig,
    sessions: Sequence[date],
    thresholds: Sequence[float],
    base_rates: Mapping[Direction, float],
) -> list[dict[str, Any]]:
    scores = _signed(estimates)
    rows = []
    for threshold in thresholds:
        result = execute_cached(
            cases, scores, Decimal(str(threshold)), snapshots, strategy, sessions
        )
        summary = trade_diagnostics(result, base_rates)
        summary.pop("trades")
        rows.append(
            {
                "threshold": str(threshold),
                "selection": summary,
                "tp_fraction": str(strategy.gross_target_fraction),
                "sl_atr": str(strategy.exit_policy.multiple),
            }
        )
    return rows


def _make_inputs(
    snapshots: Mapping[datetime, FeatureSnapshot],
    bars: Mapping[datetime, Candle],
    prior: Mapping[date, Any],
    target: Decimal,
) -> dict[str, dict[ActionKey, tuple[float, ...]]]:
    families: dict[str, dict[ActionKey, tuple[float, ...]]] = {
        "tp_current": {},
        "tp_context": {},
    }
    for moment, snapshot in snapshots.items():
        for direction in Direction:
            values = tp_inputs(
                snapshot,
                bars[moment],
                direction,
                prior[moment.date()],
                include_context=True,
                target_fraction=target,
            )
            side = (1.0 if direction is Direction.LONG else -1.0,)
            families["tp_context"][(moment, direction)] = (*values, *side)
            families["tp_current"][(moment, direction)] = (
                *values[: len(TP_CURRENT_NAMES)],
                *side,
            )
    return families


def _label_sessions(
    tapes: Mapping[date, tuple[Candle, ...]],
    snapshots: Mapping[datetime, FeatureSnapshot],
    strategy: StrategyConfig,
    sessions: Sequence[date],
) -> dict[ActionKey, SimulatedTrade]:
    policy = strategy.feasibility_policy()
    eligible = {
        moment: snapshot
        for moment, snapshot in snapshots.items()
        if moment.date() in sessions
        and (policy is None or policy.evaluate(snapshot).reachable)
    }
    cases = {}
    for day in sessions:
        cases.update(build_cases(tapes[day], eligible, strategy))
    return cases


def run_probability_study(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_path = root / "report.json"
    lock_path = root / "selection_lock.json"
    prediction_path = (
        root.resolve().parents[1] / "data/research" / root.name / "predictions.csv"
    )
    if not overwrite and any(
        path.exists()
        for path in (
            report_path,
            lock_path,
            root / "policies.csv",
            root / "trades.csv",
            prediction_path,
        )
    ):
        raise ValueError("Probability-study outputs exist; use --overwrite explicitly")
    regular, manifest = load_tape(root, protocol)
    phases = {
        name: tuple(date.fromisoformat(value) for value in values)
        for name, values in manifest["phases"].items()
    }
    training_sessions = model_training_sessions(phases, protocol)
    baseline = StrategyConfig.load(root / protocol["baseline_strategy"])
    if baseline.fingerprint != manifest["baseline_strategy_sha256"]:
        raise ValueError("Probability-study execution baseline changed")
    features = FeatureEngine()
    snapshots = {}
    for bar in regular:
        snapshot = features.update(bar)
        if phase_for(bar.start_time.date(), protocol) != "warmup" and (
            bar.end_time.time() < time(15, 15)
        ):
            if snapshot is None or not snapshot.readiness.core_ready:
                raise ValueError("Probability inputs did not warm up")
            snapshots[bar.end_time] = snapshot
    days = tuple(sorted({moment.date() for moment in snapshots}))
    tapes = {
        day: tuple(
            bar
            for bar in regular
            if bar.start_time.date() == day and bar.end_time.time() <= time(15, 15)
        )
        for day in days
    }
    bars = {bar.end_time: bar for bar in regular}
    prior = previous_sessions(regular)
    development = (
        *phases["fit"],
        *phases["model_selection"],
        *phases["calibration"],
        *phases["threshold_selection"],
    )
    inputs_by_target = {}
    policies = {}
    for target in map(Decimal, protocol["take_profit_fractions"]):
        families = _make_inputs(snapshots, bars, prior, target)
        inputs_by_target[target] = families
        for stop in map(Decimal, protocol["stop_atr_multiples"]):
            key = f"tp_{target}_sl_{stop}"
            print(f"Fitting {key} on development periods", file=sys.stderr, flush=True)
            strategy = replace(
                baseline, gross_target_fraction=target, exit_policy=FixedAtrStop(stop)
            )
            cases = _label_sessions(tapes, snapshots, strategy, development)
            eligible = {
                name: {
                    action: values
                    for action, values in inputs.items()
                    if action in cases
                }
                for name, inputs in families.items()
            }
            choices = []
            for family in protocol["model_families"]:
                fitting = {
                    action: values
                    for action, values in eligible[family].items()
                    if action[0].date() in phases["fit"]
                }
                for penalty in protocol["l2_penalties"]:
                    coefficients = fit_tp(
                        fitting,
                        cases,
                        before=phases["model_selection"][0],
                        penalty=penalty,
                    )
                    predicted = _predict(
                        eligible[family], coefficients, phases["model_selection"]
                    )
                    choices.append(
                        {
                            "family": family,
                            "penalty": penalty,
                            "metrics": _group_metrics(predicted, cases),
                        }
                    )
            chosen = min(
                choices,
                key=lambda option: (
                    option["metrics"]["equal_day_log_loss"],
                    len(
                        families[option["family"]][
                            next(iter(families[option["family"]]))
                        ]
                    ),
                    -option["penalty"],
                ),
            )
            family = chosen["family"]
            refit = {
                action: values
                for action, values in eligible[family].items()
                if action[0].date() in training_sessions
            }
            coefficients = fit_tp(
                refit, cases, before=phases["calibration"][0], penalty=chosen["penalty"]
            )
            calibration_predictions = _predict(
                eligible[family], coefficients, phases["calibration"]
            )
            calibrators = {}
            base_rates = {}
            for direction in Direction:
                samples = [
                    (
                        moment.date(),
                        estimate,
                        int(cases[(moment, side)].exit_reason is ExitReason.TARGET),
                    )
                    for (moment, side), estimate in calibration_predictions.items()
                    if side is direction
                    and (int(minutes_since_open(moment)) - 1) % 5 == 0
                ]
                calibrators[direction] = fit_calibration(
                    samples, through=phases["calibration"][-1]
                )
                sample_days = sorted({day for day, _, _ in samples})
                base_rates[direction] = statistics.fmean(
                    statistics.fmean(
                        outcome for item_day, _, outcome in samples if item_day == day
                    )
                    for day in sample_days
                )
            raw_selection = _predict(
                eligible[family], coefficients, phases["threshold_selection"]
            )
            calibrated_selection = _correct(raw_selection, calibrators)
            raw_metrics = _group_metrics(raw_selection, cases)
            calibrated_metrics = _group_metrics(calibrated_selection, cases)
            method = (
                "platt"
                if calibrated_metrics["equal_day_log_loss"]
                < raw_metrics["equal_day_log_loss"]
                else "identity"
            )
            estimates = calibrated_selection if method == "platt" else raw_selection
            rows = _threshold_rows(
                estimates,
                cases,
                snapshots,
                strategy,
                phases["threshold_selection"],
                protocol["thresholds"],
                base_rates,
            )
            selected = choose_policy(rows)
            threshold = selected["threshold"] if selected else "1"
            policies[key] = {
                "tp_fraction": str(target),
                "sl_atr": str(stop),
                "threshold": threshold,
                "family": family,
                "penalty": chosen["penalty"],
                "training_sessions": [str(day) for day in training_sessions],
                "coefficients": coefficients,
                "feature_names": [
                    "intercept",
                    *(TP_CONTEXT_NAMES if family == "tp_context" else TP_CURRENT_NAMES),
                    "side",
                ],
                "coefficient_profile": coefficient_profile(
                    [
                        "intercept",
                        *(
                            TP_CONTEXT_NAMES
                            if family == "tp_context"
                            else TP_CURRENT_NAMES
                        ),
                        "side",
                    ],
                    coefficients,
                    refit,
                ),
                "model_comparisons": choices,
                "calibrators": {
                    direction.value: asdict(calibrator)
                    for direction, calibrator in calibrators.items()
                },
                "base_rates": {
                    direction.value: value for direction, value in base_rates.items()
                },
                "calibration_method": method,
                "selection_hypothetical_raw": raw_metrics,
                "selection_hypothetical_platt": calibrated_metrics,
                "threshold_selection": rows,
                "selection": selected["selection"]
                if selected
                else rows[-1]["selection"],
                "execution_strategy": strategy.to_dict(),
            }
    fixed = choose_policy(list(policies.values()), target=Decimal("0.002"))
    overall = choose_policy(list(policies.values()))
    selected_ids = {
        "fixed_tp_0.2_percent": next(
            (key for key, policy in policies.items() if policy is fixed), None
        ),
        "overall": next(
            (key for key, policy in policies.items() if policy is overall), None
        ),
    }
    lock = {
        "schema_version": 1,
        "score_semantics": "signed_tp_probability",
        "status": "research_only_not_a_live_strategy_profile",
        "protocol_sha256": _hash(protocol_path),
        "data_manifest_sha256": _hash(root / "data_manifest.json"),
        "selected_policy_ids": selected_ids,
        "policies": policies,
        "model_fit_through": str(training_sessions[-1]),
        "calibration_fit_through": str(phases["calibration"][-1]),
        "selection_through": str(phases["threshold_selection"][-1]),
    }
    _write_json(lock_path, lock, overwrite=overwrite)
    policies = json.loads(lock_path.read_text(encoding="utf-8"))["policies"]
    report = {
        "protocol_sha256": _hash(protocol_path),
        "selection_lock_sha256": _hash(lock_path),
        "data_manifest_sha256": _hash(root / "data_manifest.json"),
        "phases": manifest["phases"],
        "selected_policy_ids": selected_ids,
        "test_policies": {},
        "status": "research_only_not_validated",
        "score_semantics": (
            "Absolute score is estimated TP probability; sign is direction; "
            "tie is abstention."
        ),
        "limitations": protocol["limits"],
    }
    trade_rows = []
    prediction_rows = []
    for key, policy in policies.items():
        print(
            f"Evaluating locked {key} on {len(phases['test'])} test sessions",
            file=sys.stderr,
            flush=True,
        )
        target = Decimal(policy["tp_fraction"])
        strategy = StrategyConfig.from_dict(policy["execution_strategy"])
        cases = _label_sessions(tapes, snapshots, strategy, phases["test"])
        inputs = {
            action: values
            for action, values in inputs_by_target[target][policy["family"]].items()
            if action in cases
        }
        raw = _predict(inputs, policy["coefficients"], phases["test"])
        calibrators = {
            Direction(direction): PlattCalibration(**values)
            for direction, values in policy["calibrators"].items()
        }
        calibrated = _correct(raw, calibrators)
        estimates = calibrated if policy["calibration_method"] == "platt" else raw
        scores = _signed(estimates)
        base_rates = {
            Direction(direction): value
            for direction, value in policy["base_rates"].items()
        }
        result = execute_cached(
            cases,
            scores,
            Decimal(policy["threshold"]),
            snapshots,
            strategy,
            phases["test"],
        )
        diagnostic = trade_diagnostics(result, base_rates)
        control = {action: base_rates[action[1]] for action in estimates}
        report["test_policies"][key] = {
            "tp_fraction": policy["tp_fraction"],
            "sl_atr": policy["sl_atr"],
            "threshold": policy["threshold"],
            "family": policy["family"],
            "calibration_method": policy["calibration_method"],
            "hypothetical_raw": _group_metrics(raw, cases),
            "hypothetical_platt": _group_metrics(calibrated, cases),
            "hypothetical_selected": _group_metrics(estimates, cases),
            "hypothetical_constant_control": _group_metrics(control, cases),
            "trading": diagnostic,
        }
        for action, estimate in estimates.items():
            trade = cases[action]
            prediction_rows.append(
                {
                    "policy": key,
                    "decision_time": action[0].isoformat(),
                    "direction": action[1].value,
                    "tp_fraction": policy["tp_fraction"],
                    "sl_atr": policy["sl_atr"],
                    "raw_probability": raw[action],
                    "platt_probability": calibrated[action],
                    "used_probability": estimate,
                    "tp_hit": int(trade.exit_reason is ExitReason.TARGET),
                    "exit_reason": trade.exit_reason.value,
                }
            )
        trade_rows.extend(
            {
                "policy": key,
                "tp_fraction": policy["tp_fraction"],
                "sl_atr": policy["sl_atr"],
                "probability": str(abs(trade.score)),
                "target_price": str(trade.target_price),
                "stop_price": str(trade.stop_price),
                **row,
            }
            for trade, row in zip(result.trades, diagnostic["trades"], strict=True)
        )
    code_root = root.resolve().parents[1] / "src/ai_trader"
    report["source_sha256"] = {
        "clock.py": _hash(code_root / "clock.py"),
        **{
            path.name: _hash(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("research_tp_policy.py"),
                Path(__file__).with_name("research_signed_score.py"),
            )
        },
        **{
            path.relative_to(code_root).as_posix(): _hash(path)
            for package in ("features", "scanner", "strategy", "costs", "replay")
            for path in sorted((code_root / package).glob("*.py"))
        },
    }
    report["python_version"] = sys.version.split()[0]
    report["scipy_version"] = version("scipy")
    mode = "w" if overwrite else "x"
    with prediction_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "policy",
                "decision_time",
                "direction",
                "tp_fraction",
                "sl_atr",
                "raw_probability",
                "platt_probability",
                "used_probability",
                "tp_hit",
                "exit_reason",
            ),
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    report["test_predictions_sha256"] = _hash(prediction_path)
    _write_json(report_path, report, overwrite=overwrite)
    with (root / "policies.csv").open(mode, encoding="utf-8", newline="") as handle:
        fields = (
            "policy",
            "direction",
            "tp_fraction",
            "sl_atr",
            "threshold",
            "trades",
            "tp_hits",
            "tp_rate",
            "mean_predicted_probability",
            "gap",
            "brier",
            "net_rupees",
            "mean_stop_fraction",
            "sl_median_percent",
            "sl_min_percent",
            "sl_max_percent",
            "median_successful_tp_minutes",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, policy in report["test_policies"].items():
            for direction in Direction:
                detail = policy["trading"]["by_direction"][direction.value]
                writer.writerow(
                    {
                        "policy": key,
                        "direction": direction.value,
                        "tp_fraction": policy["tp_fraction"],
                        "sl_atr": policy["sl_atr"],
                        "threshold": policy["threshold"],
                        "trades": detail["round_trips"],
                        "tp_hits": detail["tp_hits"],
                        "tp_rate": detail["tp_hit_rate"],
                        "mean_predicted_probability": detail["probability"][
                            "mean_probability"
                        ],
                        "gap": detail["probability"]["gap"],
                        "brier": detail["probability"]["brier"],
                        "net_rupees": detail["net_rupees"],
                        "mean_stop_fraction": detail["mean_stop_fraction"],
                        "sl_median_percent": detail["outcomes"]["sl_percent"]["median"],
                        "sl_min_percent": detail["outcomes"]["sl_percent"]["min"],
                        "sl_max_percent": detail["outcomes"]["sl_percent"]["max"],
                        "median_successful_tp_minutes": detail["outcomes"][
                            "successful_tp_minutes"
                        ]["median"],
                    }
                )
    with (root / "trades.csv").open(mode, encoding="utf-8", newline="") as handle:
        fields = (
            "policy",
            "tp_fraction",
            "sl_atr",
            "probability",
            "direction",
            "score",
            "signal_time",
            "entry_time",
            "exit_time",
            "quantity",
            "entry_price",
            "exit_price",
            "target_price",
            "stop_price",
            "exit_reason",
            "holding_minutes",
            "gross_rupees",
            "costs",
            "net_rupees",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trade_rows)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("backtests/tp_probability_2026q3")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_probability_study(args.root, overwrite=args.overwrite)
    except (ValueError, ArithmeticError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "selected_policy_ids": report["selected_policy_ids"],
                "policies": {
                    key: {
                        "tp": policy["tp_fraction"],
                        "sl_atr": policy["sl_atr"],
                        "threshold": policy["threshold"],
                        "family": policy["family"],
                        "calibration_method": policy["calibration_method"],
                        "trades": policy["trading"]["round_trips"],
                        "tp_hits": policy["trading"]["tp_hits"],
                        "tp_rate": policy["trading"]["tp_hit_rate"],
                        "mean_probability": policy["trading"]["probability"][
                            "mean_probability"
                        ],
                        "net_rupees": policy["trading"]["net_rupees"],
                    }
                    for key, policy in report["test_policies"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
