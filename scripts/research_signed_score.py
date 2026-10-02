"""Research fast, direction-symmetric intraday opportunities on frozen candles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal, localcontext
from itertools import product
from pathlib import Path
from typing import Any

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.costs import round_down_to_tick, round_up_to_tick
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine, FeatureSnapshot
from ai_trader.features.models import DERIVED_FEATURE_NAMES
from ai_trader.history import CandleStore
from ai_trader.market import Candle
from ai_trader.replay import ReplayConfig, ReplayEngine, ReplayResult
from ai_trader.replay.models import ExitReason, FillModel
from ai_trader.scanner import Direction, PortfolioState
from ai_trader.scanner.opportunity import COMPONENT_NAMES, TargetScoreConfig
from ai_trader.strategy import StrategyConfig

MODEL_SPECIFICATIONS = {
    "trend_3_ridge_0.1": ((0, 1, 2), 0.1),
    "compact_8_ridge_0.1": (tuple(range(8)), 0.1),
    "compact_8_ridge_1.0": (tuple(range(8)), 1.0),
}


@dataclass(frozen=True, slots=True)
class TargetOutcome:
    long_minutes: int | None
    short_minutes: int | None
    first_direction: int
    first_minutes: int | None
    utility: Decimal | None
    excluded_reason: str | None = None


def target_outcome(
    signal: Candle,
    subsequent: Sequence[Candle],
    cutoff: datetime,
    *,
    target_fraction: Decimal = Decimal("0.002"),
    tick_size: Decimal = Decimal("0.10"),
    decay_minutes: Decimal = Decimal(5),
) -> TargetOutcome:
    """First competing target, discounted by time; ambiguous or gapped data abstain."""
    if target_fraction <= 0 or tick_size <= 0 or decay_minutes <= 0:
        raise ValueError("Target, tick and time-decay parameters must be positive")
    if cutoff.date() != signal.end_time.date() or cutoff <= signal.end_time:
        raise ValueError("The cutoff must follow the decision in the same session")
    with localcontext(FEATURE_CONTEXT):
        upper = round_up_to_tick(signal.close * (1 + target_fraction), tick_size)
        lower = round_down_to_tick(signal.close * (1 - target_fraction), tick_size)
        expected_start = signal.end_time
        long_minutes = None
        short_minutes = None
        for bar in subsequent:
            if bar.instrument != signal.instrument:
                raise ValueError("Outcome candles must belong to the signal instrument")
            if bar.start_time < expected_start:
                raise ValueError("Outcome candles must be strictly after the signal")
            if bar.start_time >= cutoff:
                break
            if bar.start_time != expected_start or bar.end_time > cutoff:
                return TargetOutcome(None, None, 0, None, None, "missing_minute")
            elapsed = int((bar.end_time - signal.end_time) / timedelta(minutes=1))
            if long_minutes is None and bar.high >= upper:
                long_minutes = elapsed
            if short_minutes is None and bar.low <= lower:
                short_minutes = elapsed
            expected_start = bar.end_time
        if expected_start != cutoff:
            return TargetOutcome(None, None, 0, None, None, "incomplete_session")
        if long_minutes is None and short_minutes is None:
            return TargetOutcome(None, None, 0, None, Decimal(0))
        if long_minutes == short_minutes:
            return TargetOutcome(
                long_minutes, short_minutes, 0, long_minutes, None, "ambiguous_bar"
            )
        is_long = short_minutes is None or (
            long_minutes is not None and long_minutes < short_minutes
        )
        first = long_minutes if is_long else short_minutes
        direction = 1 if is_long else -1
        utility = Decimal(direction) * (-(Decimal(first) - 1) / decay_minutes).exp()
        return TargetOutcome(long_minutes, short_minutes, direction, first, utility)


@dataclass(frozen=True, slots=True)
class Observation:
    snapshot: FeatureSnapshot
    outcome: TargetOutcome
    components: tuple[float, ...]
    reachability: float
    legacy_score: float

    @property
    def session(self) -> date:
        return self.snapshot.candle_start_time.date()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(root: Path) -> tuple[list[Observation], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if _hash(root / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("The frozen parent plan has changed")
    cache = (root / plan["snapshot_directory"]).resolve()
    for filename, expected in manifest["candle_files_sha256"].items():
        if _hash(cache / filename) != expected:
            raise ValueError("Frozen candle data failed its integrity check")
    strategy = StrategyConfig.load(root / "strategy.json")
    if strategy.fingerprint != manifest["strategy_sha256"]:
        raise ValueError("Frozen baseline strategy has changed")
    instrument = Instrument("NSE", "RELIANCE")
    bars = CandleStore(cache).load(
        instrument,
        datetime(2026, 8, 31, 9, 15, tzinfo=INDIA_TIMEZONE),
        datetime(2026, 9, 4, 15, 30, tzinfo=INDIA_TIMEZONE),
    )
    regular = tuple(
        bar for bar in bars if time(9, 15) <= bar.start_time.time() < time(15, 30)
    )
    sessions: dict[date, list[Candle]] = defaultdict(list)
    snapshots: dict[datetime, FeatureSnapshot] = {}
    features = FeatureEngine()
    for bar in regular:
        snapshot = features.update(bar)
        if snapshot is not None:
            snapshots[bar.start_time] = snapshot
        if bar.start_time.date() >= date(2026, 9, 1):
            sessions[bar.start_time.date()].append(bar)
    expected_days = {date(2026, 9, day) for day in range(1, 5)}
    if set(sessions) != expected_days:
        raise ValueError("Research requires exactly September 1-4")
    model = TargetScoreConfig()
    rows: list[Observation] = []
    daily: dict[str, Any] = {}
    for day, tape in sorted(sessions.items()):
        cutoff = datetime.combine(day, time(15, 15), tzinfo=INDIA_TIMEZONE)
        active = [bar for bar in tape if bar.end_time <= cutoff]
        if len(active) != 360:
            raise ValueError(f"Incomplete pre-cutoff research session {day}")
        for index, bar in enumerate(active[:-1]):
            snapshot = snapshots[bar.start_time]
            outcome = target_outcome(bar, active[index + 1 :], cutoff)
            inputs = model.inputs(snapshot)
            if inputs.reason is not None:
                raise ValueError(f"Unready core inputs at {snapshot.candle_end_time}")
            scores = {Direction.LONG: Decimal(0), Direction.SHORT: Decimal(0)}
            with localcontext(FEATURE_CONTEXT):
                for rule in strategy.rules:
                    signal = rule.evaluate(
                        snapshot, PortfolioState.empty(bar.end_time), None
                    )
                    if signal is not None:
                        scores[signal.direction] = max(
                            scores[signal.direction], signal.score
                        )
            rows.append(
                Observation(
                    snapshot,
                    outcome,
                    tuple(float(value) for value in inputs.components),
                    float(inputs.reachability),
                    float(scores[Direction.LONG] - scores[Direction.SHORT]),
                )
            )
        daily[day.isoformat()] = {
            "open": str(active[0].open),
            "last_pre_cutoff_close": str(active[-1].close),
            "high": str(max(bar.high for bar in active)),
            "low": str(min(bar.low for bar in active)),
            "open_to_cutoff_fraction": str(active[-1].close / active[0].open - 1),
            "range_fraction": str(
                (max(bar.high for bar in active) - min(bar.low for bar in active))
                / active[0].open
            ),
            "mean_atr_fraction": statistics.fmean(
                float(snapshots[bar.start_time].atr_pct) for bar in active
            ),
        }
    return rows, {
        "parent_manifest_sha256": _hash(root / "manifest.json"),
        "parent_strategy_sha256": strategy.fingerprint,
        "data_sha256": manifest["candle_files_sha256"],
        "sessions": daily,
    }


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = (start + stop - 1) / 2
        for index in ordered[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(_ranks(left), _ranks(right))


def fit_coefficients(
    rows: Sequence[Observation],
    selected: Sequence[int],
    penalty: float,
) -> tuple[float, ...]:
    from scipy.optimize import minimize

    training = [
        row
        for row in rows
        if row.outcome.utility is not None
        and (int(row.snapshot.minutes_since_session_open) - 1) % 5 == 0
    ]
    if not training:
        raise ValueError("There are no eligible training observations")
    day_counts = Counter(row.session for row in training)
    weights = [1 / (len(day_counts) * day_counts[row.session]) for row in training]
    matrix = [
        tuple(row.components[index] * row.reachability for index in selected)
        for row in training
    ]
    targets = [(float(row.outcome.utility) + 1) / 2 for row in training]

    def objective(coefficients):
        loss = 0.0
        gradient = [0.0] * len(selected)
        for inputs, target, weight in zip(matrix, targets, weights, strict=True):
            raw = sum(
                coefficient * value
                for coefficient, value in zip(coefficients, inputs, strict=True)
            )
            probability = (
                1 / (1 + math.exp(-raw))
                if raw >= 0
                else math.exp(raw) / (1 + math.exp(raw))
            )
            loss += weight * (
                max(raw, 0) - target * raw + math.log1p(math.exp(-abs(raw)))
            )
            for index, value in enumerate(inputs):
                gradient[index] += weight * (probability - target) * value
        for index, coefficient in enumerate(coefficients):
            loss += penalty * coefficient * coefficient / 2
            gradient[index] += penalty * coefficient
        return loss, gradient

    fitted = minimize(
        objective,
        [0.0] * len(selected),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-20, 20)] * len(selected),
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not fitted.success:
        raise ValueError(f"Model fit did not converge: {fitted.message}")
    coefficients = [0.0] * len(COMPONENT_NAMES)
    for index, value in zip(selected, fitted.x, strict=True):
        coefficients[index] = float(value)
    return tuple(coefficients)


def predict(row: Observation, coefficients: Sequence[float]) -> float:
    raw = row.reachability * sum(
        value * coefficient
        for value, coefficient in zip(row.components, coefficients, strict=True)
    )
    return math.tanh(raw / 2)


def metrics(rows: Sequence[Observation], scores: Sequence[float]) -> dict[str, Any]:
    paired = [
        (row, score)
        for row, score in zip(rows, scores, strict=True)
        if row.outcome.utility is not None
    ]
    if not paired:
        return {"observations": 0}
    targets = [float(row.outcome.utility) for row, _ in paired]
    predicted = [score for _, score in paired]
    result: dict[str, Any] = {
        "observations": len(paired),
        "mse": statistics.fmean(
            (score - target) ** 2
            for score, target in zip(predicted, targets, strict=True)
        ),
        "neutral_mse": statistics.fmean(target**2 for target in targets),
        "rank_correlation": rank_correlation(predicted, targets),
        "mean_absolute_score": statistics.fmean(abs(score) for score in predicted),
        "score_min": min(predicted),
        "score_max": max(predicted),
    }
    nonzero = [(row, score) for row, score in paired if abs(score) > 1e-12]
    if nonzero:
        result["chosen_direction_mean_utility"] = statistics.fmean(
            float(row.outcome.utility) * (1 if score > 0 else -1)
            for row, score in nonzero
        )
        result["chosen_direction_first_hit_rate"] = statistics.fmean(
            row.outcome.first_direction == (1 if score > 0 else -1)
            for row, score in nonzero
        )
        result["chosen_direction_first_hit_within_5m"] = statistics.fmean(
            row.outcome.first_direction == (1 if score > 0 else -1)
            and row.outcome.first_minutes <= 5
            for row, score in nonzero
        )
    ranked = sorted(paired, key=lambda pair: -abs(pair[1]))
    top = ranked[: max(1, len(ranked) // 5)]
    result["top_magnitude_quintile"] = {
        "observations": len(top),
        "mean_score_magnitude": statistics.fmean(abs(score) for _, score in top),
        "mean_realized_directional_utility": statistics.fmean(
            float(row.outcome.utility) * (1 if score > 0 else -1 if score < 0 else 0)
            for row, score in top
        ),
        "first_target_within_5m": statistics.fmean(
            score != 0
            and row.outcome.first_direction == (1 if score > 0 else -1)
            and row.outcome.first_minutes <= 5
            for row, score in top
        ),
    }
    return result


def threshold_metrics(
    rows: Sequence[Observation], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    """Measure selective signals, not independent trades or executable returns."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The absolute score threshold must be finite and in [0, 1]")
    paired = list(zip(rows, scores, strict=True))
    if any(not math.isfinite(score) or abs(score) > 1 for _, score in paired):
        raise ValueError("Signed scores must be finite and in [-1, 1]")
    valid = [(row, score) for row, score in paired if row.outcome.utility is not None]
    selected = [
        (row, score) for row, score in valid if score != 0 and abs(score) >= threshold
    ]
    correct = [
        row
        for row, score in selected
        if row.outcome.first_direction == (1 if score > 0 else -1)
    ]
    wrong = sum(
        row.outcome.first_direction == (-1 if score > 0 else 1)
        for row, score in selected
    )
    fast_available = sum(
        row.outcome.first_minutes is not None and row.outcome.first_minutes <= 5
        for row, _ in valid
    )
    fast_correct = sum(row.outcome.first_minutes <= 5 for row in correct)
    utilities = [
        float(row.outcome.utility) * (1 if score > 0 else -1) for row, score in selected
    ]
    days = sorted({row.session for row, _ in valid})
    daily = {}
    for day in days:
        day_rows = [row for row, _ in valid if row.session == day]
        signals = [(row, score) for row, score in selected if row.session == day]
        daily[day.isoformat()] = {
            "signals": len(signals),
            "utility_per_decision": sum(
                float(row.outcome.utility) * (1 if score > 0 else -1)
                for row, score in signals
            )
            / len(day_rows),
        }
    hit_times = [row.outcome.first_minutes for row in correct]
    eventual = sum(
        (row.outcome.long_minutes if score > 0 else row.outcome.short_minutes)
        is not None
        for row, score in selected
    )
    return {
        "threshold": threshold,
        "observations": len(valid),
        "excluded_observations": len(paired) - len(valid),
        "signals": len(selected),
        "abstentions": len(valid) - len(selected),
        "coverage": len(selected) / len(valid) if valid else None,
        "long_signals": sum(score > 0 for _, score in selected),
        "short_signals": sum(score < 0 for _, score in selected),
        "correct_first_target": len(correct),
        "opposite_first_target": wrong,
        "neither_target": len(selected) - len(correct) - wrong,
        "accuracy": len(correct) / len(selected) if selected else None,
        "eventual_target_hit_rate": eventual / len(selected) if selected else None,
        "fast_correct": fast_correct,
        "fast_precision": fast_correct / len(selected) if selected else None,
        "fast_recall": fast_correct / fast_available if fast_available else None,
        "correct_first_target_within_minutes": {
            str(horizon): sum(row.outcome.first_minutes <= horizon for row in correct)
            / len(selected)
            if selected
            else None
            for horizon in (1, 3, 5, 15, 30)
        },
        "median_correct_minutes": statistics.median(hit_times) if hit_times else None,
        "mean_correct_minutes": statistics.fmean(hit_times) if hit_times else None,
        "mean_directional_utility": statistics.fmean(utilities) if utilities else None,
        "equal_day_utility_per_decision": statistics.fmean(
            day["utility_per_decision"] for day in daily.values()
        )
        if daily
        else 0.0,
        "daily": daily,
    }


def select_threshold(
    predictions: dict[str, list[tuple[Observation, float]]],
    thresholds: Sequence[float],
    *,
    selection_days: Sequence[date],
    minimum_signals: int = 30,
    minimum_signals_per_day: int = 5,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select on named past days only; an abstaining policy has utility zero."""
    expected_days = {day.isoformat() for day in selection_days}
    if not expected_days or minimum_signals < 1 or minimum_signals_per_day < 1:
        raise ValueError("Selection requires days and positive support limits")
    sweep = []
    for name, predicted in predictions.items():
        tuning = [
            (row, score) for row, score in predicted if row.session in selection_days
        ]
        for threshold in thresholds:
            measured = threshold_metrics(
                [row for row, _ in tuning], [score for _, score in tuning], threshold
            )
            eligible = (
                measured["signals"] >= minimum_signals
                and set(measured["daily"]) == expected_days
                and all(
                    day["signals"] >= minimum_signals_per_day
                    for day in measured["daily"].values()
                )
            )
            sweep.append({"model": name, "eligible": eligible, **measured})
    supported = [
        item
        for item in sweep
        if item["eligible"] and item["equal_day_utility_per_decision"] > 0
    ]
    winner = min(
        supported,
        key=lambda item: (
            -item["equal_day_utility_per_decision"],
            len(MODEL_SPECIFICATIONS[item["model"]][0]),
            -MODEL_SPECIFICATIONS[item["model"]][1],
            item["threshold"],
        ),
        default=None,
    )
    return winner, sweep


def descriptive(
    rows: Sequence[Observation], sessions: dict[str, Any]
) -> dict[str, Any]:
    for day, summary in sessions.items():
        observations_day = [row for row in rows if row.session.isoformat() == day]
        valid = [row for row in observations_day if row.outcome.utility is not None]
        summary["decisions"] = len(observations_day)
        summary["excluded"] = dict(
            Counter(
                row.outcome.excluded_reason
                for row in observations_day
                if row.outcome.excluded_reason
            )
        )
        summary["first_targets"] = dict(
            Counter(
                "LONG"
                if row.outcome.first_direction > 0
                else "SHORT"
                if row.outcome.first_direction < 0
                else "NONE"
                for row in valid
            )
        )
        for direction, field in (("LONG", "long_minutes"), ("SHORT", "short_minutes")):
            times = [getattr(row.outcome, field) for row in valid]
            reached = [value for value in times if value is not None]
            summary[direction] = {
                "same_day_hit_fraction": len(reached) / len(valid),
                "median_minutes_if_hit": statistics.median(reached)
                if reached
                else None,
                "hit_within_minutes": {
                    str(horizon): sum(
                        value is not None and value <= horizon for value in times
                    )
                    / len(valid)
                    for horizon in (1, 3, 5, 15, 30)
                },
            }
        summary["first_target_by_speed"] = dict(
            Counter(
                "none"
                if row.outcome.first_minutes is None
                else "1m"
                if row.outcome.first_minutes == 1
                else "2-5m"
                if row.outcome.first_minutes <= 5
                else "6-15m"
                if row.outcome.first_minutes <= 15
                else "16-30m"
                if row.outcome.first_minutes <= 30
                else ">30m"
                for row in valid
            )
        )
        summary["legacy_score"] = metrics(valid, [row.legacy_score for row in valid])
        summary["by_decision_hour"] = {}
        for hour in sorted({row.snapshot.candle_end_time.hour for row in valid}):
            bucket = [row for row in valid if row.snapshot.candle_end_time.hour == hour]
            summary["by_decision_hour"][str(hour)] = {
                "observations": len(bucket),
                "first_long": sum(row.outcome.first_direction == 1 for row in bucket),
                "first_short": sum(row.outcome.first_direction == -1 for row in bucket),
                "no_target": sum(row.outcome.first_direction == 0 for row in bucket),
                "any_first_hit_within_5m": sum(
                    row.outcome.first_minutes is not None
                    and row.outcome.first_minutes <= 5
                    for row in bucket
                )
                / len(bucket),
                "mean_signed_utility": statistics.fmean(
                    float(row.outcome.utility) for row in bucket
                ),
            }
    feature_audit: dict[str, Any] = {}
    for name in DERIVED_FEATURE_NAMES:
        daily = {}
        for day in sessions:
            pairs = [
                (float(getattr(row.snapshot, name)), float(row.outcome.utility))
                for row in rows
                if row.session.isoformat() == day
                and row.outcome.utility is not None
                and getattr(row.snapshot.readiness, name)
                and getattr(row.snapshot, name) is not None
            ]
            daily[day] = {
                "available": len(pairs),
                "rank_correlation": rank_correlation(
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                ),
            }
        feature_audit[name] = daily
    return {"sessions": sessions, "feature_audit": feature_audit}


def _forward_diagnostics(
    predicted: Sequence[tuple[Observation, float]],
) -> dict[str, Any]:
    valid = [
        (row, score) for row, score in predicted if row.outcome.utility is not None
    ]
    days = sorted({row.session for row, _ in valid})
    gains = [
        statistics.fmean(
            float(row.outcome.utility) ** 2 - (score - float(row.outcome.utility)) ** 2
            for row, score in valid
            if row.session == day
        )
        for day in days
    ]
    bootstrap = sorted(
        statistics.fmean(draw) for draw in product(gains, repeat=len(gains))
    )
    buckets = []
    for name, predicate in (
        ("negative_below_-0.02", lambda value: value < -0.02),
        ("negative_-0.02_to_0", lambda value: -0.02 <= value < 0),
        ("positive_0_to_0.02", lambda value: 0 <= value < 0.02),
        ("positive_at_least_0.02", lambda value: value >= 0.02),
    ):
        bucket = [(row, score) for row, score in valid if predicate(score)]
        if bucket:
            buckets.append(
                {
                    "bucket": name,
                    "sessions": len({row.session for row, _ in bucket}),
                    **metrics(
                        [row for row, _ in bucket], [score for _, score in bucket]
                    ),
                    "mean_signed_outcome": statistics.fmean(
                        float(row.outcome.utility) for row, _ in bucket
                    ),
                    "mean_signed_prediction": statistics.fmean(
                        score for _, score in bucket
                    ),
                }
            )
    sides = {}
    for side in (-1, 1):
        bucket = [(row, score) for row, score in valid if score * side > 0]
        hit_times = [
            row.outcome.first_minutes
            for row, _ in bucket
            if row.outcome.first_direction == side
        ]
        sides["LONG" if side > 0 else "SHORT"] = {
            **metrics([row for row, _ in bucket], [score for _, score in bucket]),
            "median_first_hit_minutes_when_correct": statistics.median(hit_times)
            if hit_times
            else None,
        }
    sampled = [
        (row, score)
        for row, score in valid
        if (int(row.snapshot.minutes_since_session_open) - 1) % 30 == 0
    ]
    examples = []
    for row, score in sorted(valid, key=lambda item: -abs(item[1]))[:8]:
        examples.append(
            {
                "decision_time": row.snapshot.candle_end_time.isoformat(),
                "score": score,
                "reference_close": str(row.snapshot.close),
                "first_direction": row.outcome.first_direction,
                "first_target_minutes": row.outcome.first_minutes,
                "realized_utility": str(row.outcome.utility),
            }
        )
    return {
        "daily_mse_improvement_over_zero": dict(
            zip((day.isoformat() for day in days), gains, strict=True)
        ),
        "equal_day_bootstrap_mse_improvement_95pct": [
            bootstrap[int(0.025 * (len(bootstrap) - 1))],
            bootstrap[int(0.975 * (len(bootstrap) - 1))],
        ],
        "uncertainty_caveat": (
            "Only three test sessions; exhaustive day-block resampling is "
            "descriptive, not a reliable population confidence interval."
        ),
        "signed_score_buckets": buckets,
        "by_direction": sides,
        "every_30_minutes_sensitivity": metrics(
            [row for row, _ in sampled], [score for _, score in sampled]
        ),
        "largest_magnitude_forward_examples": examples,
    }


def run_research(
    root: Path, output: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    from scipy import __version__ as scipy_version

    if output.exists() and not overwrite:
        raise ValueError(
            "Research report already exists; choose a new experiment output"
        )
    rows, provenance = observations(root)
    summary = descriptive(rows, provenance.pop("sessions"))
    specifications = MODEL_SPECIFICATIONS
    comparisons = {}
    all_predictions: dict[str, list[tuple[Observation, float]]] = {}
    for name, (selected, penalty) in specifications.items():
        folds = []
        predicted: list[tuple[Observation, float]] = []
        for day_number in (2, 3, 4):
            test_day = date(2026, 9, day_number)
            training = [row for row in rows if row.session < test_day]
            testing = [row for row in rows if row.session == test_day]
            coefficients = fit_coefficients(training, selected, penalty)
            scores = [predict(row, coefficients) for row in testing]
            folds.append(
                {
                    "test_session": test_day.isoformat(),
                    "training_sessions": sorted(
                        {row.session.isoformat() for row in training}
                    ),
                    "coefficients": dict(
                        zip(COMPONENT_NAMES, coefficients, strict=True)
                    ),
                    **metrics(testing, scores),
                }
            )
            predicted.extend(zip(testing, scores, strict=True))
        comparisons[name] = {
            "folds": folds,
            "forward_combined": metrics(
                [row for row, _ in predicted], [score for _, score in predicted]
            ),
        }
        all_predictions[name] = predicted
    winner = min(
        comparisons, key=lambda name: comparisons[name]["forward_combined"]["mse"]
    )
    selected, penalty = specifications[winner]
    coefficients = fit_coefficients(rows, selected, penalty)
    model = TargetScoreConfig(tuple(Decimal(str(value)) for value in coefficients))
    fitted_scores = [float(model.evaluate(row.snapshot).score) for row in rows]
    for row, score in zip(rows, fitted_scores, strict=True):
        if abs(score - predict(row, coefficients)) > 1e-12:
            raise ValueError("Runtime and research predictions differ")
    selected_forward = comparisons[winner]["forward_combined"]
    diagnostics = _forward_diagnostics(all_predictions[winner])
    report = {
        "provenance": provenance,
        "scipy_version": scipy_version,
        "protocol_sha256": _hash(output.parent / "protocol.json"),
        **summary,
        "model_comparisons": comparisons,
        "selected_research_model": winner,
        "coefficients": dict(zip(COMPONENT_NAMES, coefficients, strict=True)),
        "selection_uses_forward_days": True,
        "selection_caveat": (
            "These folds influenced model selection and are not an untouched "
            "final test. Only four correlated sessions were studied. "
            "No dates after September 4 were scored or labelled."
        ),
        "fitted_in_sample": metrics(rows, fitted_scores),
        "forward_diagnostics": diagnostics,
        "relative_mse_improvement_over_zero": 1
        - selected_forward["mse"] / selected_forward["neutral_mse"],
        "promotion_status": "research_only_not_validated",
        "promotion_reason": (
            "Four sessions and unstable forward-day direction quality do not "
            "establish a deployable edge; small scores must not be amplified "
            "to fill the range."
        ),
        "runtime_model": {
            "transform_version": model.transform_version,
            "component_names": COMPONENT_NAMES,
            "target_fraction": str(model.target_fraction),
            "decay_minutes": str(model.decay_minutes),
            "square_off_minutes": str(model.square_off_minutes),
            "formula": (
                "2*sigmoid(reachability * dot(coefficients, bounded_components))-1"
            ),
            "reachability": (
                "(1-exp(-5*(atr_fraction/0.002)^2)) * "
                "(1-exp(-minutes_remaining/5)); structural heuristic, "
                "not a first-passage probability"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    strategy = replace(StrategyConfig.load(root / "strategy.json"), target_score=model)
    strategy_path = output.parent / "strategy.json"
    with strategy_path.open(mode, encoding="utf-8", newline="") as handle:
        json.dump(strategy.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    if StrategyConfig.load(strategy_path) != strategy:
        raise ValueError("Saved research strategy did not round-trip")
    report["strategy_sha256"] = strategy.fingerprint
    report["research_script_sha256"] = _hash(Path(__file__))
    code_root = Path(__file__).parents[1] / "src/ai_trader"
    report["decision_code_sha256"] = {
        path.relative_to(code_root).as_posix(): _hash(path)
        for package in ("features", "scanner", "strategy", "costs")
        for path in sorted((code_root / package).glob("*.py"))
    }
    with output.open(mode, encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rows_path = Path("data/research") / output.parent.name / "observations.csv"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open(mode, encoding="utf-8", newline="") as handle:
        fields = [
            "decision_time",
            "close",
            "long_minutes",
            "short_minutes",
            "first_direction",
            "first_minutes",
            "utility",
            "excluded_reason",
            "legacy_score",
            "fitted_score",
            *COMPONENT_NAMES,
            *DERIVED_FEATURE_NAMES,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, score in zip(rows, fitted_scores, strict=True):
            writer.writerow(
                {
                    "decision_time": row.snapshot.candle_end_time.isoformat(),
                    "close": row.snapshot.close,
                    "long_minutes": row.outcome.long_minutes,
                    "short_minutes": row.outcome.short_minutes,
                    "first_direction": row.outcome.first_direction,
                    "first_minutes": row.outcome.first_minutes,
                    "utility": row.outcome.utility,
                    "excluded_reason": row.outcome.excluded_reason,
                    "legacy_score": row.legacy_score,
                    "fitted_score": score,
                    **dict(zip(COMPONENT_NAMES, row.components, strict=True)),
                    **{
                        name: getattr(row.snapshot, name)
                        for name in DERIVED_FEATURE_NAMES
                    },
                }
            )
    forward_path = rows_path.with_name("forward_predictions.csv")
    with forward_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "decision_time",
                "score",
                "outcome",
                "first_minutes",
                "first_direction",
            ),
        )
        writer.writeheader()
        for row, score in all_predictions[winner]:
            writer.writerow(
                {
                    "decision_time": row.snapshot.candle_end_time.isoformat(),
                    "score": score,
                    "outcome": row.outcome.utility,
                    "first_minutes": row.outcome.first_minutes,
                    "first_direction": row.outcome.first_direction,
                }
            )
    return report


def _threshold_breakdown(
    predicted: Sequence[tuple[Observation, float]], threshold: float
) -> dict[str, Any]:
    result = threshold_metrics(
        [row for row, _ in predicted], [score for _, score in predicted], threshold
    )
    groups = {
        "LONG": [(row, score) for row, score in predicted if score > 0],
        "SHORT": [(row, score) for row, score in predicted if score < 0],
    }
    result["by_direction"] = {
        name: threshold_metrics(
            [row for row, _ in group], [score for _, score in group], threshold
        )
        for name, group in groups.items()
    }
    result["by_day"] = {
        day.isoformat(): threshold_metrics(
            [row for row, _ in predicted if row.session == day],
            [score for row, score in predicted if row.session == day],
            threshold,
        )
        for day in sorted({row.session for row, _ in predicted})
    }
    return result


def _trade_summary(result: ReplayResult) -> dict[str, Any]:
    wins = [trade for trade in result.trades if trade.net_rupees > 0]
    losses = [trade for trade in result.trades if trade.net_rupees < 0]
    targets = [
        trade for trade in result.trades if trade.exit_reason is ExitReason.TARGET
    ]
    win_sum = sum((trade.net_rupees for trade in wins), Decimal(0))
    loss_sum = -sum((trade.net_rupees for trade in losses), Decimal(0))
    duration = [float(trade.holding_minutes) for trade in result.trades]
    return {
        "sessions": [day.isoformat() for day in result.sessions],
        "round_trips": result.round_trips,
        "tp_hits": len(targets),
        "tp_hit_rate": len(targets) / result.round_trips if result.trades else None,
        "tp_hits_within_5_minutes": sum(
            trade.holding_minutes <= 5 for trade in targets
        ),
        "median_tp_minutes": statistics.median(
            float(trade.holding_minutes) for trade in targets
        )
        if targets
        else None,
        "net_winners": len(wins),
        "net_losers": len(losses),
        "net_hit_rate": float(result.net_hit_rate) if result.trades else None,
        "gross_hit_rate": float(result.gross_hit_rate) if result.trades else None,
        "gross_rupees": float(result.gross_rupees),
        "total_costs": float(result.total_costs),
        "net_rupees": float(result.net_rupees),
        "net_expectancy_rupees": float(result.net_expectancy_rupees)
        if result.trades
        else None,
        "profit_factor": float(win_sum / loss_sum) if loss_sum else None,
        "average_net_win_rupees": float(win_sum / len(wins)) if wins else None,
        "average_net_loss_rupees": float(loss_sum / len(losses)) if losses else None,
        "realized_max_drawdown_rupees": float(result.max_drawdown_rupees),
        "mean_holding_minutes": statistics.fmean(duration) if duration else None,
        "median_holding_minutes": statistics.median(duration) if duration else None,
        "mean_favourable_fraction": float(result.mean_favourable_fraction),
        "mean_adverse_fraction": float(result.mean_adverse_fraction),
        "turnover_rupees": float(result.turnover),
        "exit_reasons": dict(result.exit_reasons),
        "ambiguous_exits": result.ambiguous_exits,
        "unfilled_entries": result.unfilled_entries,
        "unwound_entries": result.unwound_entries,
        "open_at_end": result.open_at_end,
        "by_direction": {
            direction.value: {
                "trades": sum(trade.direction == direction for trade in result.trades),
                "net_winners": sum(
                    trade.direction == direction and trade.is_win
                    for trade in result.trades
                ),
                "net_rupees": float(
                    sum(
                        (
                            trade.net_rupees
                            for trade in result.trades
                            if trade.direction == direction
                        ),
                        Decimal(0),
                    )
                ),
            }
            for direction in Direction
        },
        "trades": [
            {
                "direction": trade.direction.value,
                "score": str(trade.score),
                "signal_time": trade.signal_time.isoformat(),
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "quantity": trade.quantity,
                "entry_price": str(trade.entry_price),
                "exit_price": str(trade.exit_price),
                "exit_reason": trade.exit_reason.value,
                "holding_minutes": str(trade.holding_minutes),
                "gross_rupees": str(trade.gross_rupees),
                "costs": str(trade.cost.total),
                "net_rupees": str(trade.net_rupees),
            }
            for trade in result.trades
        ],
    }


def run_threshold_research(
    root: Path, output: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    experiment = root / "signed_score_v1"
    protocol_path = experiment / "threshold_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent_path = experiment / "report.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    strategy_path = output.with_name("threshold_strategy.json")
    accuracy_path = output.with_name("accuracy_strategy.json")
    sweep_path = output.with_name("threshold_sweep.csv")
    prediction_path = (
        Path("data/research") / experiment.name / "threshold_predictions.csv"
    )
    if not overwrite and any(
        path.exists()
        for path in (output, strategy_path, accuracy_path, sweep_path, prediction_path)
    ):
        raise ValueError("Threshold research outputs exist; use --overwrite explicitly")
    rows, provenance = observations(root)
    selection_days = tuple(
        date.fromisoformat(day) for day in protocol["selection_days"]
    )
    test_day = date.fromisoformat(protocol["locked_forward_day"])
    if selection_days != (date(2026, 9, 2), date(2026, 9, 3)) or test_day != date(
        2026, 9, 4
    ):
        raise ValueError("This threshold study is restricted to September 1-4")
    predictions: dict[str, list[tuple[Observation, float]]] = {}
    test_models = {}
    for name in protocol["candidate_models"]:
        predicted = []
        for fold in parent["model_comparisons"][name]["folds"]:
            day = date.fromisoformat(fold["test_session"])
            if day not in (*selection_days, test_day) or any(
                date.fromisoformat(training) >= day
                for training in fold["training_sessions"]
            ):
                raise ValueError("Saved model fold violates chronological ordering")
            model = TargetScoreConfig(
                tuple(
                    Decimal(str(fold["coefficients"][component]))
                    for component in COMPONENT_NAMES
                )
            )
            testing = [row for row in rows if row.session == day]
            scores = [float(model.evaluate(row.snapshot).score) for row in testing]
            if abs(metrics(testing, scores)["mse"] - fold["mse"]) > 1e-12:
                raise ValueError(
                    "Saved fold scores no longer reproduce; stop the sweep"
                )
            predicted.extend(zip(testing, scores, strict=True))
            if day == test_day:
                test_models[name] = model
        predictions[name] = predicted
    winner, sweep = select_threshold(
        predictions,
        protocol["threshold_grid"],
        selection_days=selection_days,
        minimum_signals=protocol["minimum_signals"],
        minimum_signals_per_day=protocol["minimum_signals_per_day"],
    )
    accuracy_leader = min(
        (item for item in sweep if item["eligible"]),
        key=lambda item: (-item["accuracy"], -item["signals"], item["threshold"]),
        default=None,
    )
    if winner is None:
        name = "no_trade"
        threshold = 1.0
        model = TargetScoreConfig(score_threshold=Decimal(1))
        chosen = [
            (row, 0.0) for row in rows if row.session in (*selection_days, test_day)
        ]
    else:
        name, threshold = winner["model"], winner["threshold"]
        model = replace(test_models[name], score_threshold=Decimal(str(threshold)))
        chosen = predictions[name]
    tuning = [(row, score) for row, score in chosen if row.session in selection_days]
    forward = [(row, score) for row, score in chosen if row.session == test_day]
    baseline = StrategyConfig.load(root / "strategy.json")
    strategy = replace(baseline, target_score=model)
    accuracy_strategy = None
    accuracy_diagnostic = None
    if accuracy_leader is not None:
        accuracy_name = accuracy_leader["model"]
        accuracy_threshold = accuracy_leader["threshold"]
        accuracy_strategy = replace(
            baseline,
            target_score=replace(
                test_models[accuracy_name],
                score_threshold=Decimal(str(accuracy_threshold)),
            ),
        )
        accuracy_diagnostic = {
            "selection": accuracy_leader,
            "forward": _threshold_breakdown(
                [
                    (row, score)
                    for row, score in predictions[accuracy_name]
                    if row.session == test_day
                ],
                accuracy_threshold,
            ),
            "strategy_sha256": accuracy_strategy.fingerprint,
            "caveat": (
                "Secondary accuracy-only diagnostic selected on Sep2-3, "
                "not a replacement for the speed-utility objective."
            ),
        }
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    instrument = Instrument("NSE", "RELIANCE")
    bars = CandleStore((root / plan["snapshot_directory"]).resolve()).load(
        instrument,
        datetime(2026, 8, 31, 9, 15, tzinfo=INDIA_TIMEZONE),
        datetime.combine(test_day, time(15, 30), tzinfo=INDIA_TIMEZONE),
    )
    regular = [
        bar for bar in bars if time(9, 15) <= bar.start_time.time() < time(15, 30)
    ]
    warmup = [bar for bar in regular if bar.start_time.date() < test_day]
    measured = [bar for bar in regular if bar.start_time.date() == test_day]
    fill = FillModel(Decimal(1), Decimal("0.0002"), Decimal("0.0001"))
    replay = {}
    replay_profiles = [
        ("selected_threshold", strategy),
        (
            "same_model_without_threshold",
            replace(strategy, target_score=replace(model, score_threshold=Decimal(0))),
        ),
    ]
    if accuracy_strategy is not None:
        replay_profiles.append(("accuracy_leader", accuracy_strategy))
    for label, profile in replay_profiles:
        result = ReplayEngine(
            ReplayConfig(universe=(instrument,), fill=fill, strategy=profile)
        ).run(measured, warmup_candles=warmup)
        replay[label] = {
            "strategy_sha256": profile.fingerprint,
            **_trade_summary(result),
        }
    controls = {
        "same_model_without_threshold": _threshold_breakdown(forward, 0),
        "always_long": _threshold_breakdown([(row, 1.0) for row, _ in forward], 0),
        "always_short": _threshold_breakdown([(row, -1.0) for row, _ in forward], 0),
        "never_trade": _threshold_breakdown([(row, 0.0) for row, _ in forward], 0),
    }
    report = {
        "protocol_sha256": _hash(protocol_path),
        "parent_report_sha256": _hash(parent_path),
        "provenance": provenance,
        "selected_model": name,
        "score_threshold": threshold,
        "coefficients": dict(
            zip(COMPONENT_NAMES, map(str, model.coefficients), strict=True)
        ),
        "model_fit_through": "2026-09-03",
        "selection_sessions": [day.isoformat() for day in selection_days],
        "forward_session": test_day.isoformat(),
        "selection": _threshold_breakdown(tuning, threshold),
        "forward": _threshold_breakdown(forward, threshold),
        "forward_controls": controls,
        "accuracy_diagnostic": accuracy_diagnostic,
        "threshold_sweep": sweep,
        "replay": replay,
        "replay_assumptions": {
            "clip_rupees": str(strategy.target_notional),
            "target_fraction": str(strategy.gross_target_fraction),
            "stop": strategy.exit_policy.description,
            "feasibility_screen": strategy.screen_feasibility,
            "latency_seconds": str(fill.latency_seconds),
            "half_spread_fraction": str(fill.half_spread_fraction),
            "slippage_fraction": str(fill.slippage_fraction),
            "ambiguous_exit": fill.resolve_ambiguous_bar_as.value,
        },
        "status": "research_only_not_validated",
        "limitations": [
            protocol["honesty"],
            "Threshold selection uses overlapping minute signals, "
            "not independent trades.",
            "Signal accuracy means chosen 0.2% barrier hit first; neither is a miss.",
            "Replay has 2-ATR stops, costs and modelled fills, "
            "unlike analytical labels.",
            "Realized drawdown omits open-position mark-to-market losses.",
            "No later sessions, new model families or exit policies were searched.",
        ],
        "strategy_sha256": strategy.fingerprint,
        "research_script_sha256": _hash(Path(__file__)),
    }
    code_root = Path(__file__).parents[1] / "src/ai_trader"
    report["decision_code_sha256"] = {
        path.relative_to(code_root).as_posix(): _hash(path)
        for package in ("features", "scanner", "strategy", "costs", "replay")
        for path in sorted((code_root / package).glob("*.py"))
    }
    mode = "w" if overwrite else "x"
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    documents = [(strategy_path, strategy.to_dict()), (output, report)]
    if accuracy_strategy is not None:
        documents.append((accuracy_path, accuracy_strategy.to_dict()))
    for path, document in documents:
        with path.open(mode, encoding="utf-8", newline="") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    if StrategyConfig.load(strategy_path) != strategy:
        raise ValueError("Threshold strategy failed to round-trip")
    fields = (
        "model",
        "threshold",
        "eligible",
        "signals",
        "coverage",
        "accuracy",
        "long_signals",
        "short_signals",
        "fast_precision",
        "fast_recall",
        "mean_directional_utility",
        "equal_day_utility_per_decision",
    )
    with sweep_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep)
    with prediction_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "decision_time",
                "phase",
                "score",
                "action",
                "first_direction",
                "first_minutes",
                "utility",
            ),
        )
        writer.writeheader()
        for row, score in chosen:
            writer.writerow(
                {
                    "decision_time": row.snapshot.candle_end_time.isoformat(),
                    "phase": "forward" if row.session == test_day else "selection",
                    "score": score,
                    "action": (1 if score > 0 else -1)
                    if score and abs(score) >= threshold
                    else 0,
                    "first_direction": row.outcome.first_direction,
                    "first_minutes": row.outcome.first_minutes,
                    "utility": row.outcome.utility,
                }
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("backtests/reliance_sep2026"))
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument("--threshold-study", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate this experiment's outputs; never alter the frozen parent.",
    )
    args = parser.parse_args(argv)
    output = args.output or args.root / "signed_score_v1" / (
        "threshold_report.json" if args.threshold_study else "report.json"
    )
    try:
        runner = run_threshold_research if args.threshold_study else run_research
        report = runner(args.root, output, overwrite=args.overwrite)
    except (ArithmeticError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.threshold_study:
        print(
            json.dumps(
                {
                    "selected": report["selected_model"],
                    "threshold": report["score_threshold"],
                    "accuracy_diagnostic": report["accuracy_diagnostic"],
                    "selection": {
                        key: value
                        for key, value in report["selection"].items()
                        if key not in ("by_direction", "by_day", "daily")
                    },
                    "forward": {
                        key: value
                        for key, value in report["forward"].items()
                        if key not in ("by_direction", "by_day", "daily")
                    },
                    "replay": {
                        name: {
                            key: value
                            for key, value in result.items()
                            if key != "trades"
                        }
                        for name, result in report["replay"].items()
                    },
                },
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "sessions": report["sessions"],
                "models": {
                    name: result["forward_combined"]
                    for name, result in report["model_comparisons"].items()
                },
                "selected": report["selected_research_model"],
                "coefficients": report["coefficients"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
