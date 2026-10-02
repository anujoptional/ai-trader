"""Research TP precision with shared execution and chronological selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from ai_trader.broker import Instrument
from ai_trader.clock import (
    INDIA_TIMEZONE,
    ONE_MINUTE,
    exact_timedelta,
    minutes_since_open,
)
from ai_trader.features import FEATURE_CONTEXT, FeatureSnapshot
from ai_trader.history import CandleStore
from ai_trader.market import Candle
from ai_trader.replay import ReplayConfig, ReplayEngine, ReplayResult
from ai_trader.replay.models import ExitReason, FillModel, SimulatedTrade
from ai_trader.replay.portfolio import ReplayPortfolio
from ai_trader.scanner import Candidate, Direction
from ai_trader.scanner.opportunity import COMPONENT_NAMES, TargetScoreConfig
from ai_trader.strategy import FixedAtrStop, StrategyConfig
from scripts.research_signed_score import _hash, _trade_summary, observations

RESEARCH_FILL = FillModel(Decimal(1), Decimal("0.0002"), Decimal("0.0001"))
TP_CURRENT_NAMES = (*COMPONENT_NAMES, "volatility_to_target", "time_remaining", "adx")
TP_CONTEXT_NAMES = (
    *TP_CURRENT_NAMES,
    "prior_cutoff_close_distance",
    "prior_range_position",
    "opening_from_prior_cutoff",
    "prior_pre_cutoff_return",
    "candle_body_atr",
    "candle_close_location",
)


@dataclass(frozen=True, slots=True)
class PriorFrame:
    session: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


def previous_sessions(tape: Sequence[Candle]) -> dict[date, PriorFrame | None]:
    grouped: dict[date, list[Candle]] = {}
    for bar in sorted(tape, key=lambda bar: bar.start_time):
        if time(9, 15) <= bar.start_time.time() < time(15, 15):
            grouped.setdefault(bar.start_time.date(), []).append(bar)
    previous = None
    frames = {}
    for day, bars in sorted(grouped.items()):
        frames[day] = previous
        if (
            len(bars) == 360
            and bars[0].start_time.time() == time(9, 15)
            and bars[-1].end_time.time() == time(15, 15)
            and all(
                left.end_time == right.start_time for left, right in zip(bars, bars[1:])
            )
        ):
            previous = PriorFrame(
                day,
                bars[0].open,
                max(bar.high for bar in bars),
                min(bar.low for bar in bars),
                bars[-1].close,
            )
        else:
            previous = None
    return frames


def tp_inputs(
    snapshot: FeatureSnapshot,
    bar: Candle,
    direction: Direction,
    prior: PriorFrame | None,
    *,
    include_context: bool,
    target_fraction: Decimal = Decimal("0.002"),
) -> tuple[float, ...]:
    if not target_fraction.is_finite() or not 0 < target_fraction < 1:
        raise ValueError("TP feature target must be finite and in (0, 1)")
    sign = 1 if direction is Direction.LONG else -1
    inputs = TargetScoreConfig().inputs(snapshot)
    if inputs.reason is not None:
        raise ValueError("TP features need ready core inputs")
    base = tuple(sign * float(value) for value in inputs.components) + (
        math.tanh(float(snapshot.atr_pct / target_fraction)),
        max(0.0, 1 - float(snapshot.minutes_since_session_open) / 360),
        min(1.0, float(snapshot.adx14) / 100),
    )
    if not include_context:
        return base
    if prior is None or prior.session >= snapshot.candle_start_time.date():
        raise ValueError("TP context requires a completed earlier session")
    if snapshot.session_open is None or not snapshot.readiness.session_open:
        raise ValueError("TP context requires a known session open")
    atr = float(snapshot.atr14)
    close = float(snapshot.close)
    prior_high, prior_low = float(prior.high), float(prior.low)
    span = float(bar.high - bar.low)
    return base + (
        sign * math.tanh((close - float(prior.close)) / (5 * atr)),
        sign
        * math.tanh(
            (2 * close - prior_high - prior_low) / (prior_high - prior_low + 2 * atr)
        ),
        sign * math.tanh(float(snapshot.session_open - prior.close) / (5 * atr)),
        sign * math.tanh(float(prior.close / prior.open - 1) / 0.01),
        sign * math.tanh(float(bar.close - bar.open) / atr),
        sign * (float(2 * bar.close - bar.high - bar.low) / span if span else 0),
    )


def probability(inputs: Sequence[float], coefficients: Sequence[float]) -> float:
    raw = coefficients[0] + sum(
        value * coefficient
        for value, coefficient in zip(inputs, coefficients[1:], strict=True)
    )
    return (math.tanh(raw / 2) + 1) / 2


def signed_tp_confidence(long_probability: float, short_probability: float) -> Decimal:
    if not all(
        math.isfinite(value) and 0 <= value <= 1
        for value in (long_probability, short_probability)
    ):
        raise ValueError("TP probabilities must be finite and in [0, 1]")
    if long_probability == short_probability:
        return Decimal(0)
    if max(long_probability, short_probability) <= 0.5:
        return Decimal(0)
    sign = 1 if long_probability > short_probability else -1
    confidence = Decimal(str(max(long_probability, short_probability)))
    return sign * max(Decimal(0), 2 * confidence - 1)


def signed_tp_probability(long_probability: float, short_probability: float) -> Decimal:
    """Attach direction without changing probability magnitude; ties abstain."""
    if not all(
        math.isfinite(value) and 0 <= value <= 1
        for value in (long_probability, short_probability)
    ):
        raise ValueError("TP probabilities must be finite and in [0, 1]")
    if long_probability == short_probability:
        return Decimal(0)
    if long_probability > short_probability:
        return Decimal(str(long_probability))
    return -Decimal(str(short_probability))


def fit_tp(
    inputs: Mapping[tuple[datetime, Direction], tuple[float, ...]],
    cases: Mapping[tuple[datetime, Direction], SimulatedTrade],
    *,
    before: date,
    penalty: float = 0.1,
) -> tuple[float, ...]:
    from scipy.optimize import minimize

    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("TP fitting requires positive finite regularization")
    keys = [
        key
        for key in inputs
        if key in cases
        and key[0].date() < before
        and (int(minutes_since_open(key[0])) - 1) % 5 == 0
    ]
    if not keys:
        raise ValueError("No earlier executable TP labels are available")
    day_counts = Counter(moment.date() for moment, _ in keys)
    weights = [1 / (len(day_counts) * day_counts[moment.date()]) for moment, _ in keys]
    targets = [float(cases[key].exit_reason is ExitReason.TARGET) for key in keys]
    width = len(inputs[keys[0]]) + 1

    def objective(coefficients):
        loss = 0.0
        gradient = [0.0] * width
        for key, weight, target in zip(keys, weights, targets, strict=True):
            values = (1.0, *inputs[key])
            raw = sum(
                coefficient * value
                for coefficient, value in zip(coefficients, values, strict=True)
            )
            estimate = (math.tanh(raw / 2) + 1) / 2
            loss += weight * (
                max(raw, 0) - target * raw + math.log1p(math.exp(-abs(raw)))
            )
            for index, value in enumerate(values):
                gradient[index] += weight * (estimate - target) * value
        for index in range(1, width):
            loss += penalty / 2 * coefficients[index] ** 2
            gradient[index] += penalty * coefficients[index]
        return loss, gradient

    fitted = minimize(
        objective,
        [0.0] * width,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-20, 20)] * width,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not fitted.success:
        raise ValueError(f"TP model did not converge: {fitted.message}")
    return tuple(map(float, fitted.x))


def thresholds() -> tuple[Decimal, ...]:
    return tuple(
        sorted(
            {Decimal(index) / 100 for index in range(101)}
            | {Decimal(index) / 1000 for index in range(5, 100, 10)}
            | {Decimal("0.0025"), Decimal("0.0075")}
        )
    )


def isolated_trade(
    signal: Candle,
    subsequent: Sequence[Candle],
    atr_fraction: Decimal,
    direction: Direction,
    strategy: StrategyConfig,
) -> SimulatedTrade | None:
    """Label one action with the real portfolio's execution, not a second engine."""
    cutoff = datetime.combine(
        signal.start_time.date(), time(9, 15), tzinfo=INDIA_TIMEZONE
    ) + exact_timedelta(
        strategy.square_off_minutes_since_open, ONE_MINUTE, name="square_off_minutes"
    )
    if not subsequent or subsequent[0].start_time != signal.end_time:
        raise ValueError("A label requires the next contiguous minute")
    future = [bar for bar in subsequent if bar.end_time <= cutoff]
    if not future or future[-1].end_time != cutoff:
        raise ValueError("A label requires a complete path through square-off")
    previous = signal.end_time
    for bar in future:
        if bar.instrument != signal.instrument or bar.start_time != previous:
            raise ValueError("A label cannot cross instruments or missing minutes")
        previous = bar.end_time
    if future[0].end_time >= cutoff:
        return None
    candidate = Candidate(
        instrument=signal.instrument,
        direction=direction,
        score=Decimal(1) if direction is Direction.LONG else Decimal(-1),
        rules=("signed_time_target",),
        as_of=signal.end_time,
        reference_price=signal.close,
    )
    with localcontext(FEATURE_CONTEXT):
        book = ReplayPortfolio(
            universe=(signal.instrument,),
            max_open_positions=1,
            sizing=strategy.sizing_policy(),
            costs=strategy.costs,
            fill=RESEARCH_FILL,
            exit_policy=strategy.exit_policy,
        )
        pending = book.queue(candidate, atr_fraction)
        book.enter(pending, future[0].open, pending.fill_at, future[0].end_time)
        for bar in future:
            closed = book.advance(bar)
            if closed is not None:
                return closed
        return book.square_off(signal.instrument, future[-1].close, cutoff)


def execute_cached(
    cases: Mapping[tuple[datetime, Direction], SimulatedTrade],
    scores: Mapping[datetime, Decimal],
    threshold: Decimal,
    snapshots: Mapping[datetime, FeatureSnapshot],
    strategy: StrategyConfig,
    sessions: Sequence[date],
) -> ReplayResult:
    """Schedule one instrument without overlap; outcomes came from ReplayPortfolio."""
    if not threshold.is_finite() or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    selected: list[SimulatedTrade] = []
    available_at: datetime | None = None
    active_session: date | None = None
    policy = strategy.feasibility_policy()
    instruments = {snapshot.instrument for snapshot in snapshots.values()}
    if len(instruments) != 1:
        raise ValueError("Cached scheduling supports exactly one instrument")
    for moment, score in sorted(scores.items()):
        if moment.date() not in sessions:
            continue
        if moment.date() != active_session:
            available_at = None
            active_session = moment.date()
        if not score.is_finite() or abs(score) > 1:
            raise ValueError("Signed scores must be finite and bounded")
        if not score or abs(score) < threshold:
            continue
        if available_at is not None and moment < available_at:
            continue
        snapshot = snapshots[moment]
        if policy is not None and not policy.evaluate(snapshot).reachable:
            continue
        minutes = snapshot.minutes_since_session_open
        if (
            minutes is None
            or minutes >= strategy.square_off_minutes_since_open
            or (
                strategy.earliest_minutes_since_open is not None
                and minutes < strategy.earliest_minutes_since_open
            )
            or (
                strategy.latest_minutes_since_open is not None
                and minutes > strategy.latest_minutes_since_open
            )
        ):
            continue
        direction = Direction.LONG if score > 0 else Direction.SHORT
        trade = cases.get((moment, direction))
        if trade is not None:
            selected.append(replace(trade, score=score))
            available_at = trade.exit_time + exact_timedelta(
                strategy.cooldown_minutes, ONE_MINUTE, name="cooldown_minutes"
            )
    return ReplayResult(
        trades=tuple(selected),
        universe=tuple(instruments),
        sessions=tuple(sessions),
        fill=RESEARCH_FILL,
    )


def precision_summary(result: ReplayResult) -> dict[str, Any]:
    summary = _trade_summary(result)
    hits = summary["tp_hits"]
    count = summary["round_trips"]
    interval = None
    if count:
        rate = hits / count
        zscore = 1.959963984540054
        denominator = 1 + zscore**2 / count
        center = (rate + zscore**2 / (2 * count)) / denominator
        radius = (
            zscore
            * (rate * (1 - rate) / count + zscore**2 / (4 * count**2)) ** 0.5
            / denominator
        )
        interval = [max(0.0, center - radius), min(1.0, center + radius)]
    summary["descriptive_wilson_interval"] = interval
    summary["represented_sessions"] = len({trade.session for trade in result.trades})
    summary["mean_stop_fraction"] = (
        sum(
            float(abs(trade.stop_price / trade.entry_price - 1))
            for trade in result.trades
        )
        / count
        if count
        else None
    )
    for direction in Direction:
        side = [trade for trade in result.trades if trade.direction is direction]
        hits = sum(trade.exit_reason.value == "target" for trade in side)
        summary["by_direction"][direction.value]["tp_hits"] = hits
        summary["by_direction"][direction.value]["tp_hit_rate"] = (
            hits / len(side) if side else None
        )
    return summary


def build_cases(
    tape: Sequence[Candle],
    snapshots: Mapping[datetime, FeatureSnapshot],
    strategy: StrategyConfig,
    scores: Mapping[datetime, Decimal] | None = None,
) -> dict[tuple[datetime, Direction], SimulatedTrade]:
    cases = {}
    for index, bar in enumerate(tape[:-1]):
        snapshot = snapshots.get(bar.end_time)
        if snapshot is None or snapshot.atr_pct is None:
            continue
        directions = tuple(Direction)
        if scores is not None:
            score = scores.get(bar.end_time, Decimal(0))
            if not score:
                continue
            directions = (Direction.LONG if score > 0 else Direction.SHORT,)
        for direction in directions:
            trade = isolated_trade(
                bar, tape[index + 1 :], snapshot.atr_pct, direction, strategy
            )
            if trade is not None:
                cases[(bar.end_time, direction)] = trade
    return cases


def choose_policy(
    rows: Sequence[dict[str, Any]],
    *,
    target: Decimal | None = None,
    stop: Decimal | None = None,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if row["selection"]["round_trips"]
        and (target is None or Decimal(row["tp_fraction"]) == target)
        and (stop is None or Decimal(row["sl_atr"]) == stop)
    ]
    return min(
        eligible,
        key=lambda row: (
            -row["selection"]["tp_hit_rate"],
            -row["selection"]["represented_sessions"],
            -row["selection"]["round_trips"],
            Decimal(row["sl_atr"]),
            -Decimal(row["tp_fraction"]),
            Decimal(row["threshold"]),
        ),
        default=None,
    )


def _compact(result: ReplayResult) -> dict[str, Any]:
    summary = precision_summary(result)
    del summary["trades"]
    return summary


def reliability(
    predicted: Mapping[tuple[datetime, Direction], float],
    cases: Mapping[tuple[datetime, Direction], SimulatedTrade],
    days: Sequence[date],
) -> dict[str, Any]:
    pairs = [
        (estimate, int(cases[key].exit_reason is ExitReason.TARGET))
        for key, estimate in predicted.items()
        if key in cases and key[0].date() in days
    ]
    if not pairs:
        return {"observations": 0, "brier": None, "bins": []}
    bins = []
    for index in range(5):
        bucket = [
            (estimate, outcome)
            for estimate, outcome in pairs
            if index / 5 <= estimate < (index + 1) / 5 or (index == 4 and estimate == 1)
        ]
        if bucket:
            bins.append(
                {
                    "lower": index / 5,
                    "upper": (index + 1) / 5,
                    "observations": len(bucket),
                    "mean_probability": statistics.fmean(value for value, _ in bucket),
                    "actual_tp_rate": statistics.fmean(value for _, value in bucket),
                }
            )
    return {
        "observations": len(pairs),
        "brier": statistics.fmean(
            (estimate - outcome) ** 2 for estimate, outcome in pairs
        ),
        "bins": bins,
    }


def new_model_study(
    regular: Sequence[Candle],
    snapshots: Mapping[datetime, FeatureSnapshot],
    baseline: StrategyConfig,
    days: Sequence[date],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strategy = replace(
        baseline,
        gross_target_fraction=Decimal("0.002"),
        exit_policy=FixedAtrStop(Decimal(2)),
    )
    cases = {}
    for day in sorted({moment.date() for moment in snapshots}):
        tape = tuple(
            bar
            for bar in regular
            if bar.start_time.date() == day and bar.end_time.time() <= time(15, 15)
        )
        cases.update(build_cases(tape, snapshots, strategy))
    prior = previous_sessions(regular)
    candles = {bar.end_time: bar for bar in regular}
    report = {}
    sweep = []
    for name, context in (("tp_current", False), ("tp_context", True)):
        inputs = {
            (moment, direction): tp_inputs(
                snapshot,
                candles[moment],
                direction,
                prior[moment.date()],
                include_context=context,
            )
            for moment, snapshot in snapshots.items()
            for direction in Direction
        }
        predicted = {}
        signed = {}
        fitted = {}
        for day in days:
            coefficients = fit_tp(inputs, cases, before=day)
            fitted[day.isoformat()] = {
                "training_sessions": sorted(
                    {key[0].date().isoformat() for key in inputs if key[0].date() < day}
                ),
                "coefficients": coefficients,
            }
            for moment in snapshots:
                if moment.date() != day:
                    continue
                estimates = {
                    direction: probability(inputs[(moment, direction)], coefficients)
                    for direction in Direction
                }
                for direction, estimate in estimates.items():
                    predicted[(moment, direction)] = estimate
                signed[moment] = signed_tp_confidence(
                    estimates[Direction.LONG], estimates[Direction.SHORT]
                )
        model_rows = []
        for threshold in thresholds():
            tuning = execute_cached(
                cases, signed, threshold, snapshots, strategy, days[:2]
            )
            check = execute_cached(
                cases, signed, threshold, snapshots, strategy, days[2:]
            )
            model_rows.append(
                {
                    "model": name,
                    "threshold": str(threshold),
                    "tp_fraction": "0.002",
                    "sl_atr": "2",
                    "selection": _compact(tuning),
                    "check": _compact(check),
                }
            )
        best = choose_policy(model_rows)
        if best is not None:
            best["check_trades"] = precision_summary(
                execute_cached(
                    cases,
                    signed,
                    Decimal(best["threshold"]),
                    snapshots,
                    strategy,
                    days[2:],
                )
            )["trades"]
        sweep.extend(model_rows)
        report[name] = {
            "features": (
                "intercept",
                *(TP_CONTEXT_NAMES if context else TP_CURRENT_NAMES),
            ),
            "fits": fitted,
            "best_on_selection_days": best,
            "selection_reliability": reliability(predicted, cases, days[:2]),
            "check_reliability": reliability(predicted, cases, days[2:]),
            "check_probability_range": [
                min(
                    value
                    for (moment, _), value in predicted.items()
                    if moment.date() == days[-1]
                ),
                max(
                    value
                    for (moment, _), value in predicted.items()
                    if moment.date() == days[-1]
                ),
            ],
            "check_score_range": [
                str(
                    min(
                        score
                        for moment, score in signed.items()
                        if moment.date() == days[-1]
                    )
                ),
                str(
                    max(
                        score
                        for moment, score in signed.items()
                        if moment.date() == days[-1]
                    )
                ),
            ],
        }
    report["selected_by_tp_rate"] = choose_policy(sweep)
    report["semantics"] = (
        "Direction times max(0, 2*p(TP)-1), with the higher estimated side chosen. "
        "This is an uncalibrated TP confidence estimate, not the earlier speed utility."
    )
    report["status"] = "Research-only; not a live StrategyConfig profile"
    return report, sweep


def run_study(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    output = root / "tp_precision_v1"
    report_path = output / "report.json"
    if not overwrite and any(
        (output / name).exists()
        for name in (
            "report.json",
            "threshold_curve.csv",
            "barrier_sweep.csv",
            "tp_model_sweep.csv",
            "threshold_only_strategy.json",
            "fixed_tp_0.2_percent_strategy.json",
            "tp_and_sl_strategy.json",
        )
    ):
        raise ValueError("TP research outputs exist; use --overwrite explicitly")
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["selection_days"] != ["2026-09-02", "2026-09-03"] or (
        protocol["check_day"] != "2026-09-04"
    ):
        raise ValueError("This TP study is restricted to September 1-4")
    rows, provenance = observations(root)
    snapshots = {row.snapshot.candle_end_time: row.snapshot for row in rows}
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    instrument = Instrument("NSE", "RELIANCE")
    regular = tuple(
        bar
        for bar in CandleStore((root / plan["snapshot_directory"]).resolve()).load(
            instrument,
            datetime(2026, 8, 31, 9, 15, tzinfo=INDIA_TIMEZONE),
            datetime(2026, 9, 4, 15, 30, tzinfo=INDIA_TIMEZONE),
        )
        if time(9, 15) <= bar.start_time.time() < time(15, 30)
    )
    days = tuple(date(2026, 9, number) for number in (2, 3, 4))
    tuning_days = days[:2]
    tapes = {
        day: tuple(
            bar
            for bar in regular
            if bar.start_time.date() == day and bar.end_time.time() <= time(15, 15)
        )
        for day in days
    }
    if any(len(tape) != 360 for tape in tapes.values()):
        raise ValueError("TP study requires complete pre-cutoff sessions")
    parent_path = root / "signed_score_v1/report.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    folds = parent["model_comparisons"]["compact_8_ridge_0.1"]["folds"]
    models = {}
    for fold in folds:
        day = date.fromisoformat(fold["test_session"])
        if day not in days or any(
            date.fromisoformat(train) >= day for train in fold["training_sessions"]
        ):
            raise ValueError("Saved fold violates chronological ordering")
        models[day] = TargetScoreConfig(
            tuple(Decimal(str(fold["coefficients"][name])) for name in COMPONENT_NAMES)
        )
    baseline = StrategyConfig.load(root / "strategy.json")
    scores_by_tp = {}
    for target in map(Decimal, protocol["take_profit_fractions"]):
        scores_by_tp[target] = {
            moment: replace(models[moment.date()], target_fraction=target)
            .evaluate(snapshot)
            .score
            for moment, snapshot in snapshots.items()
            if moment.date() in days
        }
    sweep = []
    all_cases = {}
    for target in map(Decimal, protocol["take_profit_fractions"]):
        for stop in map(Decimal, protocol["stop_atr_multiples"]):
            strategy = replace(
                baseline, gross_target_fraction=target, exit_policy=FixedAtrStop(stop)
            )
            scores = scores_by_tp[target]
            cases = {}
            for tape in tapes.values():
                cases.update(build_cases(tape, snapshots, strategy, scores))
            all_cases[(target, stop)] = cases
            for threshold in thresholds():
                tuning = execute_cached(
                    cases, scores, threshold, snapshots, strategy, tuning_days
                )
                check = execute_cached(
                    cases, scores, threshold, snapshots, strategy, days[2:]
                )
                sweep.append(
                    {
                        "threshold": str(threshold),
                        "tp_fraction": str(target),
                        "sl_atr": str(stop),
                        "selection": _compact(tuning),
                        "check": _compact(check),
                    }
                )
    selected = {
        "threshold_only": choose_policy(
            sweep, target=Decimal("0.002"), stop=Decimal(2)
        ),
        "fixed_tp_0.2_percent": choose_policy(sweep, target=Decimal("0.002")),
        "tp_and_sl": choose_policy(sweep),
    }
    parity_checks = 0
    profiles = {}
    for name, choice in selected.items():
        if choice is None:
            continue
        target, stop = Decimal(choice["tp_fraction"]), Decimal(choice["sl_atr"])
        threshold = Decimal(choice["threshold"])
        cases = all_cases[(target, stop)]
        for day in days:
            strategy = replace(
                baseline,
                gross_target_fraction=target,
                exit_policy=FixedAtrStop(stop),
                target_score=replace(
                    models[day], target_fraction=target, score_threshold=threshold
                ),
            )
            actual = ReplayEngine(
                ReplayConfig(
                    universe=(instrument,), fill=RESEARCH_FILL, strategy=strategy
                )
            ).run(
                tapes[day],
                warmup_candles=[bar for bar in regular if bar.start_time.date() < day],
            )
            cached = execute_cached(
                cases, scores_by_tp[target], threshold, snapshots, strategy, (day,)
            )
            if cached.trades != actual.trades:
                raise ValueError(f"Cached trades differ from ReplayEngine at {day}")
            parity_checks += 1
        check_strategy = replace(
            baseline,
            gross_target_fraction=target,
            exit_policy=FixedAtrStop(stop),
            target_score=replace(
                models[days[-1]], target_fraction=target, score_threshold=threshold
            ),
        )
        profiles[name] = check_strategy
        choice["strategy_sha256"] = check_strategy.fingerprint
        choice["check_trades"] = precision_summary(
            execute_cached(
                cases,
                scores_by_tp[target],
                threshold,
                snapshots,
                check_strategy,
                days[2:],
            )
        )["trades"]
    curve = [
        row
        for row in sweep
        if Decimal(row["tp_fraction"]) == Decimal("0.002")
        and Decimal(row["sl_atr"]) == 2
    ]
    new_models, model_rows = new_model_study(regular, snapshots, baseline, days)
    report = {
        "protocol_sha256": _hash(protocol_path),
        "parent_report_sha256": _hash(parent_path),
        "provenance": provenance,
        "selection_days": [day.isoformat() for day in tuning_days],
        "check_day": days[-1].isoformat(),
        "current_model_threshold_curve": curve,
        "combinations_evaluated": len(sweep),
        "selected": selected,
        "new_models": new_models,
        "execution_parity_checks": parity_checks,
        "raw_score_ranges": {
            day.isoformat(): [
                str(
                    min(
                        score
                        for moment, score in scores_by_tp[Decimal("0.002")].items()
                        if moment.date() == day
                    )
                ),
                str(
                    max(
                        score
                        for moment, score in scores_by_tp[Decimal("0.002")].items()
                        if moment.date() == day
                    )
                ),
            ]
            for day in days
        },
        "status": "exploratory_not_validated",
        "interval_caveat": (
            "Wilson intervals assume independent trials; these trades share "
            "sessions. They describe sample size, not a validated future "
            "hit-rate guarantee."
        ),
        "research_script_sha256": _hash(Path(__file__)),
        "shared_research_helpers_sha256": _hash(
            Path(__file__).with_name("research_signed_score.py")
        ),
    }
    code_root = Path(__file__).parents[1] / "src/ai_trader"
    report["decision_code_sha256"] = {
        path.relative_to(code_root).as_posix(): _hash(path)
        for package in ("features", "scanner", "strategy", "costs", "replay")
        for path in sorted((code_root / package).glob("*.py"))
    }
    mode = "w" if overwrite else "x"
    for name, profile in profiles.items():
        path = output / f"{name}_strategy.json"
        with path.open(mode, encoding="utf-8", newline="") as handle:
            json.dump(profile.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        if StrategyConfig.load(path) != profile:
            raise ValueError("Selected TP profile did not round-trip")
    with report_path.open(mode, encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    fields = (
        "round_trips",
        "tp_hits",
        "tp_hit_rate",
        "net_hit_rate",
        "net_rupees",
        "net_expectancy_rupees",
        "profit_factor",
        "mean_holding_minutes",
        "mean_stop_fraction",
        "represented_sessions",
    )
    for filename, records in (
        ("threshold_curve.csv", curve),
        ("barrier_sweep.csv", sweep),
        ("tp_model_sweep.csv", model_rows),
    ):
        with (output / filename).open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "model",
                    "threshold",
                    "tp_fraction",
                    "sl_atr",
                    *(
                        f"{phase}_{name}"
                        for phase in ("selection", "check")
                        for name in fields
                    ),
                ),
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "model": record.get("model", "existing_signed_utility"),
                        **{
                            key: record[key]
                            for key in ("threshold", "tp_fraction", "sl_atr")
                        },
                        **{
                            f"{phase}_{name}": record[phase][name]
                            for phase in ("selection", "check")
                            for name in fields
                        },
                    }
                )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("backtests/reliance_sep2026"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_study(args.root, overwrite=args.overwrite)
    except (ArithmeticError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "combinations": report["combinations_evaluated"],
                "parity_checks": report["execution_parity_checks"],
                "new_models": {
                    name: report["new_models"][name]["best_on_selection_days"]
                    for name in ("tp_current", "tp_context")
                },
                "selected": {
                    name: {
                        key: value
                        for key, value in row.items()
                        if key != "check_trades"
                    }
                    if row is not None
                    else None
                    for name, row in report["selected"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
