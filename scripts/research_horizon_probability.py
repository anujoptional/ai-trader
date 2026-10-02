"""Research symmetric, entry-anchored TP-before-SL probabilities by horizon."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import statistics
import sys
import warnings
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from decimal import Decimal, localcontext
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE, ONE_SECOND, exact_timedelta
from ai_trader.costs import SizingPolicy, TradeCostEstimate, round_up_to_tick
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine, FeatureSnapshot
from ai_trader.market import Candle
from ai_trader.replay.models import ExitReason, FillModel, ReplayResult, SimulatedTrade
from ai_trader.replay.portfolio import ReplayPortfolio
from ai_trader.scanner import Candidate, Direction
from ai_trader.scanner.opportunity import TargetScoreConfig
from ai_trader.strategy import ExitPolicy, StrategyConfig
from scripts.research_signed_score import _hash
from scripts.research_tp_policy import (
    RESEARCH_FILL,
    TP_CURRENT_NAMES,
    PriorFrame,
    execute_cached,
    previous_sessions,
    tp_inputs,
)
from scripts.research_tp_probability import (
    ActionKey,
    _signed,
    _write_json,
    load_tape,
    probability_metrics,
    trade_diagnostics,
)

WEEK_NAMES = (
    "prior_week_return",
    "prior_week_log_slope",
    "prior_week_return_volatility",
    "prior_week_daily_range",
    "price_vs_prior_week_range",
    "price_vs_prior_cutoff_close",
)
FEATURE_NAMES = {
    "time_volatility_control": ("volatility_to_target", "time_remaining", "side"),
    "intraday": (*TP_CURRENT_NAMES, "side"),
    "intraday_week": (*TP_CURRENT_NAMES, "side", *WEEK_NAMES),
}


def symmetric_prices(
    entry_price: Decimal, fraction: Decimal, tick: Decimal
) -> tuple[Decimal, Decimal]:
    if not fraction.is_finite() or not 0 < fraction < 1:
        raise ValueError("Symmetric barrier fraction must be finite and in (0, 1)")
    if not entry_price.is_finite() or entry_price <= 0:
        raise ValueError("Entry price must be finite and positive")
    if not tick.is_finite() or tick <= 0 or entry_price % tick:
        raise ValueError("Entry price must lie on a positive finite tick grid")
    distance = round_up_to_tick(entry_price * fraction, tick)
    if distance >= entry_price:
        raise ValueError("Symmetric barriers must leave a positive lower price")
    return entry_price + distance, entry_price - distance


@dataclass(frozen=True, slots=True, kw_only=True)
class SymmetricSizing(SizingPolicy):
    barrier_fraction: Decimal

    def __post_init__(self) -> None:
        SizingPolicy.__post_init__(self)
        if not self.barrier_fraction.is_finite() or not 0 < self.barrier_fraction < 1:
            raise ValueError("Symmetric barrier fraction must be finite and in (0, 1)")

    def estimate(self, entry_price: Decimal) -> TradeCostEstimate:
        estimate = SizingPolicy.estimate(self, entry_price)
        upper, lower = symmetric_prices(
            entry_price, self.barrier_fraction, self.tick_size
        )
        return replace(
            estimate,
            required_gross_fraction=self.barrier_fraction,
            net_margin_fraction=(
                self.barrier_fraction - estimate.cost.total / estimate.notional
            ),
            long_exit_price=upper,
            short_exit_price=lower,
        )


@dataclass(frozen=True, slots=True)
class SymmetricStop(ExitPolicy):
    barrier_fraction: Decimal

    def __post_init__(self) -> None:
        if not self.barrier_fraction.is_finite() or not 0 < self.barrier_fraction < 1:
            raise ValueError("Symmetric barrier fraction must be finite and in (0, 1)")

    def stop_price(
        self,
        *,
        entry_price: Decimal,
        direction: Direction,
        atr_fraction: Decimal,
        favourable_fraction: Decimal,
        tick: Decimal,
    ) -> Decimal:
        upper, lower = symmetric_prices(entry_price, self.barrier_fraction, tick)
        return lower if direction is Direction.LONG else upper

    @property
    def description(self) -> str:
        return f"symmetric TP/SL at {100 * self.barrier_fraction}% of executed entry"


def horizon_trade(
    signal: Candle,
    subsequent: Sequence[Candle],
    direction: Direction,
    strategy: StrategyConfig,
    fraction: Decimal,
    hours: Decimal,
    *,
    fill: FillModel = RESEARCH_FILL,
) -> SimulatedTrade | None:
    if not hours.is_finite() or hours * 60 < 2:
        raise ValueError("A candle-based horizon must be at least two minutes")
    duration = exact_timedelta(hours * 60, ONE_MINUTE, name="horizon_minutes")
    entered_at = signal.end_time + exact_timedelta(
        fill.latency_seconds, ONE_SECOND, name="latency_seconds"
    )
    deadline = entered_at + duration
    cutoff = datetime.combine(signal.start_time.date(), time(9, 15), INDIA_TIMEZONE)
    cutoff += exact_timedelta(
        strategy.square_off_minutes_since_open, ONE_MINUTE, name="square_off_minutes"
    )
    if deadline > cutoff:
        return None
    future = [bar for bar in subsequent if bar.end_time <= deadline]
    previous = signal.end_time
    for bar in future:
        if bar.instrument != signal.instrument or bar.start_time != previous:
            raise ValueError(
                "A horizon label cannot cross instruments or missing minutes"
            )
        previous = bar.end_time
    if not future or (deadline - future[-1].end_time).total_seconds() >= 60:
        raise ValueError(
            "A horizon label requires all completed minutes to its deadline"
        )
    if not future[0].start_time <= entered_at < future[0].end_time:
        raise ValueError("Horizon labeling requires a fill in the next candle")
    with localcontext(FEATURE_CONTEXT):
        base = strategy.sizing_policy()
        sizing = SymmetricSizing(
            target_notional=base.target_notional,
            net_margin_fraction=base.net_margin_fraction,
            costs=base.costs,
            tick_size=base.tick_size,
            barrier_fraction=fraction,
        )
        book = ReplayPortfolio(
            universe=(signal.instrument,),
            max_open_positions=1,
            sizing=sizing,
            costs=strategy.costs,
            fill=fill,
            exit_policy=SymmetricStop(fraction),
        )
        candidate = Candidate(
            instrument=signal.instrument,
            direction=direction,
            score=Decimal(1) if direction is Direction.LONG else Decimal(-1),
            rules=("symmetric_horizon_probability",),
            as_of=signal.end_time,
            reference_price=signal.close,
        )
        pending = book.queue(candidate, Decimal(0))
        reference = signal.close if not fill.latency_seconds else future[0].open
        book.enter(pending, reference, pending.fill_at, future[0].end_time)
        for bar in future:
            closed = book.advance(bar)
            if closed is not None:
                return closed
        return book.square_off(
            signal.instrument, future[-1].close, deadline, reason=ExitReason.HORIZON
        )


def weekly_context(tape: Sequence[Candle]) -> dict[date, tuple[PriorFrame, ...]]:
    recent: deque[PriorFrame] = deque(maxlen=5)
    result = {}
    for day, previous in sorted(previous_sessions(tape).items()):
        if previous is None:
            recent.clear()
        else:
            recent.append(previous)
        result[day] = tuple(recent) if len(recent) == 5 else ()
    return result


def horizon_inputs(
    snapshot: FeatureSnapshot,
    bar: Candle,
    direction: Direction,
    week: Sequence[PriorFrame],
    fraction: Decimal,
) -> dict[str, tuple[float, ...]]:
    if len(week) != 5 or any(frame.session >= bar.start_time.date() for frame in week):
        raise ValueError("Weekly context needs five complete earlier sessions")
    if len({frame.session for frame in week}) != 5:
        raise ValueError("Weekly context cannot repeat sessions")
    if list(week) != sorted(week, key=lambda frame: frame.session):
        raise ValueError("Weekly context must be chronological")
    sign = 1.0 if direction is Direction.LONG else -1.0
    current = tp_inputs(
        snapshot, bar, direction, None, include_context=False, target_fraction=fraction
    ) + (sign,)
    closes = [math.log(float(frame.close)) for frame in week]
    slope = sum((index - 2) * value for index, value in enumerate(closes)) / 10
    returns = [later - earlier for earlier, later in zip(closes, closes[1:])]
    high, low = max(frame.high for frame in week), min(frame.low for frame in week)
    atr = snapshot.atr14
    macro = (
        sign * math.tanh(float(week[-1].close / week[0].open - 1) / 0.02),
        sign * math.tanh(slope * 5 / 0.02),
        math.tanh(statistics.pstdev(returns) / 0.02),
        math.tanh(
            statistics.fmean(
                float((frame.high - frame.low) / frame.close) for frame in week
            )
            / 0.02
        ),
        sign
        * math.tanh(float((2 * snapshot.close - high - low) / (high - low + 2 * atr))),
        sign * math.tanh(float((snapshot.close - week[-1].close) / (5 * atr))),
    )
    return {
        "time_volatility_control": (current[8], current[9], sign),
        "intraday": current,
        "intraday_week": (*current, *macro),
    }


def outcome_class(trade: SimulatedTrade, horizons: Sequence[Decimal]) -> int:
    if not horizons or list(horizons) != sorted(set(horizons)):
        raise ValueError("Horizons must be nonempty, unique and increasing")
    if trade.exit_reason is ExitReason.HORIZON:
        required_end = trade.entry_time + exact_timedelta(
            horizons[-1] * 60, ONE_MINUTE, name="horizon_minutes"
        )
        if trade.exit_time != required_end:
            raise ValueError("A timeout label must cover the full largest horizon")
        return 2 * len(horizons)
    if trade.exit_reason not in (ExitReason.TARGET, ExitReason.STOP):
        raise ValueError("Only complete horizon outcomes can train the time model")
    for index, hours in enumerate(horizons):
        deadline = trade.entry_time + exact_timedelta(
            hours * 60, ONE_MINUTE, name="horizon_minutes"
        )
        if trade.exit_time <= deadline:
            return index + (
                len(horizons) if trade.exit_reason is ExitReason.STOP else 0
            )
    raise ValueError("A barrier outcome falls outside the largest horizon")


def cumulative_tp(mass: Sequence[float]) -> tuple[float, ...]:
    if len(mass) < 3 or len(mass) % 2 != 1:
        raise ValueError("Event mass must contain TP bins, SL bins and timeout")
    if any(not math.isfinite(value) or value < 0 for value in mass) or not math.isclose(
        sum(mass), 1.0, abs_tol=1e-9
    ):
        raise ValueError(
            "Event probabilities must be finite, nonnegative and sum to one"
        )
    total = 0.0
    result = []
    for value in mass[: len(mass) // 2]:
        total += value
        result.append(min(1.0, total))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MassCalibration:
    slope: float
    biases: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.slope)
            or self.slope < 0
            or any(not math.isfinite(value) for value in self.biases)
        ):
            raise ValueError(
                "Mass calibration parameters must be finite, slope nonnegative"
            )

    def apply(self, mass: Sequence[float]) -> tuple[float, ...]:
        cumulative_tp(mass)
        if len(mass) != len(self.biases):
            raise ValueError("Calibration and event distribution widths must agree")
        logits = [
            self.slope * math.log(max(1e-12, value)) + bias
            for value, bias in zip(mass, self.biases, strict=True)
        ]
        largest = max(logits)
        weights = [math.exp(value - largest) for value in logits]
        total = sum(weights)
        return tuple(value / total for value in weights)


def fit_mass_calibration(
    samples: Sequence[tuple[date, tuple[float, ...], int]], *, through: date
) -> MassCalibration:
    from scipy.optimize import minimize

    training = [sample for sample in samples if sample[0] <= through]
    if not training:
        raise ValueError("Calibration requires earlier observations")
    width = len(training[0][1])
    for _, mass, label in training:
        cumulative_tp(mass)
        if len(mass) != width or not 0 <= label < width:
            raise ValueError("Calibration outcomes and distributions must agree")
    counts = Counter(day for day, _, _ in training)
    prepared = [
        (
            tuple(math.log(max(1e-12, value)) for value in mass),
            label,
            1 / (len(counts) * counts[day]),
        )
        for day, mass, label in training
    ]

    def objective(parameters):
        slope, *biases = parameters
        loss = 0.01 * ((slope - 1) ** 2 + sum(value**2 for value in biases))
        gradient = [0.02 * (slope - 1), *(0.02 * value for value in biases)]
        for log_mass, label, weight in prepared:
            logits = [
                slope * value + bias
                for value, bias in zip(log_mass, biases, strict=True)
            ]
            largest = max(logits)
            exponentials = [math.exp(value - largest) for value in logits]
            normalizer = sum(exponentials)
            loss += weight * (largest + math.log(normalizer) - logits[label])
            for index, (value, log_probability) in enumerate(
                zip(exponentials, log_mass, strict=True)
            ):
                residual = weight * (value / normalizer - int(index == label))
                gradient[0] += residual * log_probability
                gradient[index + 1] += residual
        return loss, gradient

    fitted = minimize(
        objective,
        [1.0, *([0.0] * width)],
        method="L-BFGS-B",
        jac=True,
        bounds=[(0, 5), *([(-4, 4)] * width)],
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not fitted.success:
        raise ValueError(f"Mass calibration did not converge: {fitted.message}")
    return MassCalibration(float(fitted.x[0]), tuple(map(float, fitted.x[1:])))


def fit_time_model(
    inputs: Mapping[ActionKey, tuple[float, ...]],
    outcomes: Mapping[ActionKey, int],
    specification: Mapping[str, Any],
    sessions: Sequence[date],
    *,
    stride: int = 5,
    seed: int = 20261001,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    if stride <= 0:
        raise ValueError("Fitting stride must be positive")
    keys = sorted(
        key
        for key in inputs
        if key in outcomes
        and key[0].date() in sessions
        and ((key[0].hour * 60 + key[0].minute) - (9 * 60 + 16)) % stride == 0
    )
    if not keys or len({outcomes[key] for key in keys}) < 2:
        raise ValueError("Time models need earlier outcomes from at least two classes")
    matrix = [inputs[key] for key in keys]
    if any(not math.isfinite(value) for values in matrix for value in values):
        raise ValueError("Time model inputs must be finite")
    counts = Counter(key[0].date() for key in keys)
    weights = [len(keys) / (len(counts) * counts[key[0].date()]) for key in keys]
    kind = specification["kind"]
    if kind == "logistic":
        model = LogisticRegression(
            C=specification["C"],
            solver="lbfgs",
            max_iter=2000,
            tol=1e-7,
            random_state=seed,
        )
    elif kind == "trees":
        model = HistGradientBoostingClassifier(
            max_depth=specification["max_depth"],
            learning_rate=0.05,
            max_iter=80,
            min_samples_leaf=60,
            l2_regularization=5,
            early_stopping=False,
            random_state=seed,
            categorical_features=None,
        )
    elif kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(specification["width"],),
            activation="tanh",
            solver="lbfgs",
            alpha=specification["alpha"],
            max_iter=1500,
            max_fun=30000,
            tol=1e-6,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported research model: {kind}")
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        scaler = StandardScaler().fit(matrix, sample_weight=weights)
        model.fit(
            scaler.transform(matrix),
            [outcomes[key] for key in keys],
            sample_weight=weights,
        )
    return {
        "model": model,
        "scaler": scaler,
        "specification": dict(specification),
        "training_actions": len(keys),
        "training_sessions": [str(day) for day in sorted(counts)],
        "warnings": [str(warning.message) for warning in captured],
        "converged": not any(
            issubclass(warning.category, ConvergenceWarning) for warning in captured
        ),
    }


def predict_mass(
    fitted: Mapping[str, Any], inputs: Mapping[ActionKey, tuple[float, ...]], width: int
) -> dict[ActionKey, tuple[float, ...]]:
    from threadpoolctl import threadpool_limits

    if not inputs:
        return {}
    keys = sorted(inputs)
    with threadpool_limits(limits=1):
        values = fitted["scaler"].transform([inputs[key] for key in keys])
        predicted = fitted["model"].predict_proba(values).tolist()
    classes = list(map(int, fitted["model"].classes_))
    result = {}
    for key, estimates in zip(keys, predicted, strict=True):
        mass = [0.0] * width
        for label, estimate in zip(classes, estimates, strict=True):
            mass[label] = float(estimate)
        cumulative_tp(mass)
        result[key] = tuple(mass)
    return result


def action_metrics(
    probabilities: Mapping[ActionKey, float], labels: Mapping[ActionKey, int]
) -> dict[str, Any]:
    result = probability_metrics(
        [(value, labels[key]) for key, value in probabilities.items()]
    )
    grouped: dict[date, list[tuple[float, int]]] = {}
    for key, value in probabilities.items():
        grouped.setdefault(key[0].date(), []).append((value, labels[key]))
    result["by_day"] = {
        str(day): probability_metrics(pairs) for day, pairs in sorted(grouped.items())
    }
    result["sessions"] = len(grouped)
    for metric in ("brier", "log_loss"):
        result[f"equal_day_{metric}"] = (
            statistics.fmean(detail[metric] for detail in result["by_day"].values())
            if grouped
            else None
        )
    result["by_direction"] = {
        direction.value: probability_metrics(
            [
                (value, labels[key])
                for key, value in probabilities.items()
                if key[1] is direction
            ]
        )
        for direction in Direction
    }
    return result


def integrated_tp_brier(
    masses: Mapping[ActionKey, tuple[float, ...]], outcomes: Mapping[ActionKey, int]
) -> float:
    by_day: dict[date, list[float]] = {}
    for key, mass in masses.items():
        cumulative = cumulative_tp(mass)
        loss = statistics.fmean(
            (estimate - int(outcomes[key] <= index)) ** 2
            for index, estimate in enumerate(cumulative)
        )
        by_day.setdefault(key[0].date(), []).append(loss)
    if not by_day:
        raise ValueError("Probability scoring requires observations")
    return statistics.fmean(statistics.fmean(values) for values in by_day.values())


def constant_masses(
    outcomes: Mapping[ActionKey, int], width: int
) -> dict[Direction, tuple[float, ...]]:
    result = {}
    for direction in Direction:
        grouped: dict[date, Counter[int]] = {}
        for (moment, side), label in outcomes.items():
            if side is direction:
                grouped.setdefault(moment.date(), Counter())[label] += 1
        if not grouped:
            raise ValueError("The base-rate control requires both directions")
        result[direction] = tuple(
            statistics.fmean(
                (counts[label] + 0.5) / (sum(counts.values()) + 0.5 * width)
                for counts in grouped.values()
            )
            for label in range(width)
        )
    return result


def probability_skill(
    metrics: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, float]:
    return {
        metric: 1 - metrics[f"equal_day_{metric}"] / control[f"equal_day_{metric}"]
        for metric in ("brier", "log_loss")
    }


def select_event(policies: Mapping[str, Mapping[str, Any]]) -> str:
    if not policies:
        raise ValueError("Event selection requires validation results")
    return min(
        policies,
        key=lambda key: (
            -min(policies[key]["validation_skill"].values()),
            Decimal(policies[key]["horizon_hours"]),
            Decimal(policies[key]["barrier_fraction"]),
            key,
        ),
    )


def bootstrap_precision(
    trades: Sequence[SimulatedTrade], sessions: Sequence[date], *, seed: int
) -> dict[str, float | None]:
    if not trades:
        return {"lower_10_percentile": None, "upper_90_percentile": None}
    if not sessions or any(trade.session not in sessions for trade in trades):
        raise ValueError("Bootstrap sessions must cover all trades")
    counts = Counter(trade.session for trade in trades)
    hits = Counter(
        trade.session for trade in trades if trade.exit_reason is ExitReason.TARGET
    )
    generator = random.Random(seed)
    rates = []
    for _ in range(1000):
        sampled = generator.choices(sessions, k=len(sessions))
        total = sum(counts[day] for day in sampled)
        rates.append(sum(hits[day] for day in sampled) / total if total else 0.0)
    rates.sort()
    return {"lower_10_percentile": rates[100], "upper_90_percentile": rates[899]}


@dataclass(slots=True)
class TimeData:
    inputs: dict[str, dict[ActionKey, tuple[float, ...]]]
    cases: dict[ActionKey, SimulatedTrade]
    outcomes: dict[ActionKey, int]


def prepare_features(tape: Sequence[Candle]):
    engine = FeatureEngine()
    snapshots = {}
    by_day: dict[date, list[Candle]] = {}
    for bar in tape:
        snapshots[bar.end_time] = engine.update(bar)
        by_day.setdefault(bar.start_time.date(), []).append(bar)
    return snapshots, by_day, weekly_context(tape)


def build_time_data(
    snapshots: Mapping[datetime, FeatureSnapshot],
    by_day: Mapping[date, Sequence[Candle]],
    weeks: Mapping[date, tuple[PriorFrame, ...]],
    strategy: StrategyConfig,
    fraction: Decimal,
    horizons: Sequence[Decimal],
    sessions: Sequence[date],
) -> TimeData:
    inputs: dict[str, dict[ActionKey, tuple[float, ...]]] = {
        family: {} for family in FEATURE_NAMES
    }
    cases, outcomes = {}, {}
    duration = exact_timedelta(horizons[-1] * 60, ONE_MINUTE, name="horizon_minutes")
    latency = exact_timedelta(
        RESEARCH_FILL.latency_seconds, ONE_SECOND, name="latency_seconds"
    )
    scorer = TargetScoreConfig()
    for day in sessions:
        if not weeks.get(day):
            continue
        cutoff = datetime.combine(day, time(9, 15), INDIA_TIMEZONE) + exact_timedelta(
            strategy.square_off_minutes_since_open,
            ONE_MINUTE,
            name="square_off_minutes",
        )
        bars = by_day[day]
        for index, bar in enumerate(bars):
            if bar.end_time + latency + duration > cutoff:
                break
            snapshot = snapshots[bar.end_time]
            if scorer.inputs(snapshot).reason is not None:
                continue
            for direction in Direction:
                action = (bar.end_time, direction)
                features = horizon_inputs(
                    snapshot, bar, direction, weeks[day], fraction
                )
                trade = horizon_trade(
                    bar, bars[index + 1 :], direction, strategy, fraction, horizons[-1]
                )
                if trade is None:
                    raise ValueError("An eligible common-window action has no outcome")
                cases[action] = trade
                outcomes[action] = outcome_class(trade, horizons)
                for family, values in features.items():
                    inputs[family][action] = values
    return TimeData(inputs, cases, outcomes)


def phase_inputs(
    inputs: Mapping[ActionKey, tuple[float, ...]], sessions: Sequence[date]
) -> dict[ActionKey, tuple[float, ...]]:
    return {key: values for key, values in inputs.items() if key[0].date() in sessions}


def cases_for_horizon(
    cases: Mapping[ActionKey, SimulatedTrade],
    scores: Mapping[datetime, Decimal],
    by_day: Mapping[date, Sequence[Candle]],
    strategy: StrategyConfig,
    fraction: Decimal,
    hours: Decimal,
) -> dict[ActionKey, SimulatedTrade]:
    locations = {
        bar.end_time: (day, index)
        for day, bars in by_day.items()
        for index, bar in enumerate(bars)
    }
    result = {}
    duration = exact_timedelta(hours * 60, ONE_MINUTE, name="horizon_minutes")
    for moment, score in scores.items():
        if not score:
            continue
        key = (moment, Direction.LONG if score > 0 else Direction.SHORT)
        trade = cases[key]
        if trade.exit_time <= trade.entry_time + duration:
            result[key] = trade
        else:
            day, index = locations[moment]
            bars = by_day[day]
            closed = horizon_trade(
                bars[index], bars[index + 1 :], key[1], strategy, fraction, hours
            )
            if closed is None:
                raise ValueError("A complete longer horizon lost its shorter outcome")
            result[key] = closed
    return result


def threshold_study(
    cases: Mapping[ActionKey, SimulatedTrade],
    scores: Mapping[datetime, Decimal],
    snapshots: Mapping[datetime, FeatureSnapshot],
    strategy: StrategyConfig,
    sessions: Sequence[date],
    thresholds: Sequence[str],
    *,
    seed: int,
) -> tuple[str, list[dict[str, Any]], ReplayResult]:
    rows = []
    results = {}
    for text in thresholds:
        result = execute_cached(
            cases, scores, Decimal(text), snapshots, strategy, sessions
        )
        results[text] = result
        measured = probability_metrics(
            [
                (abs(float(trade.score)), int(trade.exit_reason is ExitReason.TARGET))
                for trade in result.trades
            ]
        )
        rows.append(
            {
                "threshold": text,
                "trades": len(result.trades),
                "sessions": len({trade.session for trade in result.trades}),
                "tp_hits": sum(
                    trade.exit_reason is ExitReason.TARGET for trade in result.trades
                ),
                "tp_rate": measured["tp_rate"],
                "mean_probability": measured["mean_probability"],
                "brier": measured["brier"],
                "net_rupees": float(result.net_rupees),
                **bootstrap_precision(result.trades, sessions, seed=seed),
            }
        )
    eligible = [row for row in rows if row["trades"]]
    chosen = (
        max(
            eligible,
            key=lambda row: (
                row["lower_10_percentile"],
                row["sessions"],
                row["trades"],
                -Decimal(row["threshold"]),
            ),
        )
        if eligible
        else rows[0]
    )
    return chosen["threshold"], rows, results[chosen["threshold"]]


def permutation_reliance(
    fitted: Mapping[str, Any],
    inputs: Mapping[ActionKey, tuple[float, ...]],
    outcomes: Mapping[ActionKey, int],
    names: Sequence[str],
    width: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    baseline = integrated_tp_brier(predict_mass(fitted, inputs, width), outcomes)
    keys = sorted(inputs)
    generator = random.Random(seed)
    result = []
    for column, name in enumerate(names):
        differences = []
        for _ in range(3):
            values = [inputs[key][column] for key in keys]
            generator.shuffle(values)
            changed = {
                key: (*inputs[key][:column], value, *inputs[key][column + 1 :])
                for key, value in zip(keys, values, strict=True)
            }
            loss = integrated_tp_brier(predict_mass(fitted, changed, width), outcomes)
            differences.append(loss - baseline)
        result.append(
            {
                "feature": name,
                "validation_brier_increase": statistics.fmean(differences),
                "repeats": differences,
            }
        )
    return sorted(result, key=lambda row: -row["validation_brier_increase"])


def model_description(fitted: Mapping[str, Any]) -> dict[str, Any]:
    model, scaler = fitted["model"], fitted["scaler"]
    description = {
        key: value for key, value in fitted.items() if key not in ("model", "scaler")
    }
    description.update(
        {
            "parameters": model.get_params(deep=False),
            "class_order": model.classes_.tolist(),
            "standardization_mean": scaler.mean_.tolist(),
            "standardization_scale": scaler.scale_.tolist(),
        }
    )
    if hasattr(model, "coef_"):
        description["coefficients"] = model.coef_.tolist()
        description["intercepts"] = model.intercept_.tolist()
    if hasattr(model, "coefs_"):
        description["layer_weights"] = [values.tolist() for values in model.coefs_]
        description["layer_biases"] = [values.tolist() for values in model.intercepts_]
    return description


def save_model(path: Path, fitted: Mapping[str, Any], *, overwrite: bool) -> str:
    with path.open("wb" if overwrite else "xb") as handle:
        pickle.dump(dict(fitted), handle, protocol=5)
    return _hash(path)


def load_model(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _hash(path) != expected_sha256:
        raise ValueError("Research model artifact changed; refusing to deserialize")
    with path.open("rb") as handle:
        return pickle.load(handle)


def paired_day_skill_interval(
    measured: Mapping[str, Any], control: Mapping[str, Any], *, seed: int
) -> dict[str, float]:
    days = sorted(measured["by_day"])
    generator = random.Random(seed)
    skills = []
    for _ in range(1000):
        sampled = generator.choices(days, k=len(days))
        loss = statistics.fmean(measured["by_day"][day]["brier"] for day in sampled)
        base = statistics.fmean(control["by_day"][day]["brier"] for day in sampled)
        skills.append(1 - loss / base)
    skills.sort()
    return {"lower_2.5_percentile": skills[25], "upper_97.5_percentile": skills[974]}


def _source_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), repository / "src/ai_trader/clock.py"]
    paths.extend(
        repository / "scripts" / name
        for name in (
            "research_tp_probability.py",
            "research_tp_policy.py",
            "research_signed_score.py",
        )
    )
    for package in ("features", "scanner", "strategy", "costs", "replay"):
        paths.extend(sorted((repository / "src/ai_trader" / package).glob("*.py")))
    return {path.relative_to(repository).as_posix(): _hash(path) for path in paths}


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if not rows:
        return
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_horizon_study(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if (root / "report.json").exists() and not overwrite:
        raise ValueError("Research output exists; use --overwrite to reproduce it")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    horizons = tuple(Decimal(value) for value in protocol["horizons_hours"])
    if list(horizons) != sorted(set(horizons)) or any(
        not hours.is_finite() or hours * 60 < 2 for hours in horizons
    ):
        raise ValueError("Horizon grid must be increasing and at least two minutes")
    width = 2 * len(horizons) + 1
    tape, manifest = load_tape(root, protocol)
    phases = {
        name: tuple(date.fromisoformat(value) for value in days)
        for name, days in manifest["phases"].items()
    }
    base = StrategyConfig.load(root / protocol["baseline_strategy"])
    if base.fingerprint != manifest["baseline_strategy_sha256"]:
        raise ValueError("The baseline strategy changed")
    snapshots, by_day, weeks = prepare_features(tape)
    development = tuple(
        day
        for name in ("fit", "model_selection", "calibration", "threshold_selection")
        for day in phases[name]
    )
    data_root = root.resolve().parents[1] / "data/research" / root.name
    seed = protocol["seed"]
    models, policies, trials = {}, {}, []
    for text in protocol["barrier_fractions"]:
        fraction = Decimal(text)
        strategy = replace(
            base,
            gross_target_fraction=fraction,
            screen_feasibility=False,
            target_score=None,
        )
        print(f"Building symmetric {text} development labels", flush=True)
        data = build_time_data(
            snapshots, by_day, weeks, strategy, fraction, horizons, development
        )
        chosen, chosen_row = None, None
        for family in protocol["feature_families"]:
            validation_inputs = phase_inputs(
                data.inputs[family], phases["model_selection"]
            )
            for specification in protocol["models"]:
                print(f"Fitting {text} {family} {specification['name']}", flush=True)
                fitted = fit_time_model(
                    data.inputs[family],
                    data.outcomes,
                    specification,
                    phases["fit"],
                    stride=protocol["fit_stride_minutes"],
                    seed=seed,
                )
                loss = integrated_tp_brier(
                    predict_mass(fitted, validation_inputs, width), data.outcomes
                )
                row = {
                    "barrier_fraction": text,
                    "feature_family": family,
                    "model": specification["name"],
                    "kind": specification["kind"],
                    "validation_integrated_tp_brier": loss,
                    "training_actions": fitted["training_actions"],
                    "converged": fitted["converged"],
                    "warnings": " | ".join(fitted["warnings"]),
                }
                trials.append(row)
                key = (loss, f"{family}/{specification['name']}")
                previous_key = (
                    (
                        chosen_row["validation_integrated_tp_brier"],
                        f"{chosen_row['feature_family']}/{chosen_row['model']}",
                    )
                    if chosen_row
                    else None
                )
                if fitted["converged"] and (previous_key is None or key < previous_key):
                    chosen, chosen_row = fitted, row
        if chosen is None:
            raise ValueError(f"No converged candidate for barrier {text}")
        family = chosen_row["feature_family"]
        calibration_inputs = phase_inputs(data.inputs[family], phases["calibration"])
        calibration_inputs = {
            key: values
            for key, values in calibration_inputs.items()
            if ((key[0].hour * 60 + key[0].minute) - (9 * 60 + 16))
            % protocol["fit_stride_minutes"]
            == 0
        }
        raw_calibration = predict_mass(chosen, calibration_inputs, width)
        calibrators = {
            direction: fit_mass_calibration(
                [
                    (key[0].date(), mass, data.outcomes[key])
                    for key, mass in raw_calibration.items()
                    if key[1] is direction
                ],
                through=phases["calibration"][-1],
            )
            for direction in Direction
        }
        controls = constant_masses(
            {key: data.outcomes[key] for key in raw_calibration}, width
        )
        selection_inputs = phase_inputs(
            data.inputs[family], phases["threshold_selection"]
        )
        raw_selection = predict_mass(chosen, selection_inputs, width)
        mapped_selection = {
            key: calibrators[key[1]].apply(mass) for key, mass in raw_selection.items()
        }
        raw_loss = integrated_tp_brier(raw_selection, data.outcomes)
        mapped_loss = integrated_tp_brier(mapped_selection, data.outcomes)
        method = "joint_softmax" if mapped_loss < raw_loss else "identity"
        selected_mass = mapped_selection if method == "joint_softmax" else raw_selection
        model_path = data_root / f"model_tp_{text}.pkl"
        metadata = model_description(chosen)
        metadata.update(
            {
                "feature_family": family,
                "feature_names": list(FEATURE_NAMES[family]),
                "calibration_method": method,
                "calibrators": {
                    direction.value: asdict(value)
                    for direction, value in calibrators.items()
                },
                "constant_mass": {
                    direction.value: value for direction, value in controls.items()
                },
                "calibration_comparison": {"raw": raw_loss, "mapped": mapped_loss},
                "artifact": model_path.name,
                "artifact_sha256": save_model(model_path, chosen, overwrite=overwrite),
                "permutation_reliance": permutation_reliance(
                    chosen,
                    phase_inputs(data.inputs[family], phases["model_selection"]),
                    data.outcomes,
                    FEATURE_NAMES[family],
                    width,
                    seed=seed,
                ),
            }
        )
        models[text] = metadata
        for index, hours in enumerate(horizons):
            key = f"tp_{text}_h_{hours}"
            probabilities = {
                action: cumulative_tp(mass)[index]
                for action, mass in selected_mass.items()
            }
            labels = {
                action: int(data.outcomes[action] <= index) for action in probabilities
            }
            constant = {
                action: cumulative_tp(controls[action[1]])[index]
                for action in probabilities
            }
            measured, control = (
                action_metrics(probabilities, labels),
                action_metrics(constant, labels),
            )
            scores = _signed(probabilities)
            cases = cases_for_horizon(
                data.cases, scores, by_day, strategy, fraction, hours
            )
            threshold, curve, result = threshold_study(
                cases,
                scores,
                snapshots,
                strategy,
                phases["threshold_selection"],
                protocol["thresholds"],
                seed=seed,
            )
            policies[key] = {
                "barrier_fraction": text,
                "horizon_hours": str(hours),
                "threshold": threshold,
                "model": chosen_row["model"],
                "feature_family": family,
                "calibration_method": method,
                "validation_probability": measured,
                "validation_constant": control,
                "validation_skill": probability_skill(measured, control),
                "threshold_curve": curve,
                "validation_trading": trade_diagnostics(
                    result,
                    {
                        direction: cumulative_tp(controls[direction])[index]
                        for direction in Direction
                    },
                ),
            }
        del data
    selected_id = select_event(policies)
    lock = {
        "protocol_sha256": _hash(root / "protocol.json"),
        "candles_sha256": manifest["candles_sha256"],
        "model_fit_through": str(phases["fit"][-1]),
        "calibration_fit_through": str(phases["calibration"][-1]),
        "selection_through": str(phases["threshold_selection"][-1]),
        "selected_policy_id": selected_id,
        "models": models,
        "policies": policies,
        "model_trials": trials,
        "source_sha256": _source_hashes(),
        "versions": {
            name: version(name)
            for name in ("scikit-learn", "scipy", "numpy", "joblib", "threadpoolctl")
        },
        "python_version": sys.version.split()[0],
    }
    _write_json(root / "selection_lock.json", lock, overwrite=overwrite)
    _csv(root / "model_trials.csv", trials, overwrite=overwrite)
    lock = json.loads((root / "selection_lock.json").read_text(encoding="utf-8"))
    report = {
        "selected_policy_id": lock["selected_policy_id"],
        "test_sessions": [str(day) for day in phases["test"]],
        "selection_lock_sha256": _hash(root / "selection_lock.json"),
        "test_policies": {},
    }
    prediction_rows, trade_rows, summary_rows = [], [], []
    for text in protocol["barrier_fractions"]:
        print(
            f"Evaluating locked symmetric {text} on {len(phases['test'])} sessions",
            flush=True,
        )
        fraction = Decimal(text)
        strategy = replace(
            base,
            gross_target_fraction=fraction,
            screen_feasibility=False,
            target_score=None,
        )
        metadata = lock["models"][text]
        chosen = load_model(
            data_root / metadata["artifact"], metadata["artifact_sha256"]
        )
        data = build_time_data(
            snapshots, by_day, weeks, strategy, fraction, horizons, phases["test"]
        )
        raw = predict_mass(chosen, data.inputs[metadata["feature_family"]], width)
        calibrators = {
            Direction(side): MassCalibration(values["slope"], tuple(values["biases"]))
            for side, values in metadata["calibrators"].items()
        }
        mapped = {key: calibrators[key[1]].apply(mass) for key, mass in raw.items()}
        used = mapped if metadata["calibration_method"] == "joint_softmax" else raw
        controls = {
            Direction(side): tuple(mass)
            for side, mass in metadata["constant_mass"].items()
        }
        for action, mass in used.items():
            trade = data.cases[action]
            row = {
                "barrier_fraction": text,
                "decision_time": action[0].isoformat(),
                "direction": action[1].value,
                "outcome_class": data.outcomes[action],
                "entry_price": str(trade.entry_price),
                "target_price": str(trade.target_price),
                "stop_price": str(trade.stop_price),
                "max_horizon_exit_reason": trade.exit_reason.value,
                "holding_minutes": float(trade.holding_minutes),
            }
            for index, hours in enumerate(horizons):
                row[f"raw_tp_{hours}h"] = cumulative_tp(raw[action])[index]
                row[f"mapped_tp_{hours}h"] = cumulative_tp(mapped[action])[index]
                row[f"used_tp_{hours}h"] = cumulative_tp(mass)[index]
                row[f"tp_hit_{hours}h"] = int(data.outcomes[action] <= index)
            prediction_rows.append(row)
        for index, hours in enumerate(horizons):
            key = f"tp_{text}_h_{hours}"
            policy = lock["policies"][key]
            probabilities = {
                action: cumulative_tp(mass)[index] for action, mass in used.items()
            }
            labels = {
                action: int(data.outcomes[action] <= index) for action in probabilities
            }
            constant = {
                action: cumulative_tp(controls[action[1]])[index]
                for action in probabilities
            }
            measured, control = (
                action_metrics(probabilities, labels),
                action_metrics(constant, labels),
            )
            scores = _signed(probabilities)
            cases = cases_for_horizon(
                data.cases, scores, by_day, strategy, fraction, hours
            )
            result = execute_cached(
                cases,
                scores,
                Decimal(policy["threshold"]),
                snapshots,
                strategy,
                phases["test"],
            )
            diagnostic = trade_diagnostics(
                result,
                {
                    direction: cumulative_tp(controls[direction])[index]
                    for direction in Direction
                },
            )
            report["test_policies"][key] = {
                name: policy[name]
                for name in (
                    "barrier_fraction",
                    "horizon_hours",
                    "threshold",
                    "model",
                    "feature_family",
                    "calibration_method",
                )
            } | {
                "probability": measured,
                "constant_control": control,
                "skill": probability_skill(measured, control),
                "trading": diagnostic,
                "paired_day_brier_skill_interval": paired_day_skill_interval(
                    measured, control, seed=seed
                ),
                "trading_session_bootstrap": bootstrap_precision(
                    result.trades, phases["test"], seed=seed
                ),
            }
            for direction in Direction:
                detail = diagnostic["by_direction"][direction.value]
                action_detail = measured["by_direction"][direction.value]
                summary_rows.append(
                    {
                        "policy": key,
                        "direction": direction.value,
                        "tp_fraction": text,
                        "sl_fraction": text,
                        "horizon_hours": str(hours),
                        "model": policy["model"],
                        "feature_family": policy["feature_family"],
                        "threshold": policy["threshold"],
                        "hypothetical_actions": action_detail["observations"],
                        "action_mean_probability": action_detail["mean_probability"],
                        "action_hit_rate": action_detail["tp_rate"],
                        "action_auc": action_detail["roc_auc"],
                        "trades": detail["round_trips"],
                        "tp_hits": detail["tp_hits"],
                        "trade_mean_probability": detail["probability"][
                            "mean_probability"
                        ],
                        "trade_tp_rate": detail["tp_hit_rate"],
                        "net_rupees": detail["net_rupees"],
                        "median_actual_sl_percent": detail["outcomes"]["sl_percent"][
                            "median"
                        ],
                        "median_successful_tp_minutes": detail["outcomes"][
                            "successful_tp_minutes"
                        ]["median"],
                    }
                )
            trade_rows.extend(
                {
                    "policy": key,
                    "tp_fraction": text,
                    "sl_fraction": text,
                    "horizon_hours": str(hours),
                    "probability": abs(float(trade.score)),
                    "target_price": str(trade.target_price),
                    "stop_price": str(trade.stop_price),
                    "tp_percent": 100
                    * float(abs(trade.target_price / trade.entry_price - 1)),
                    "sl_percent": 100
                    * float(abs(trade.stop_price / trade.entry_price - 1)),
                    **row,
                }
                for trade, row in zip(result.trades, diagnostic["trades"], strict=True)
            )
    _csv(data_root / "predictions.csv", prediction_rows, overwrite=overwrite)
    _csv(root / "policies.csv", summary_rows, overwrite=overwrite)
    _csv(root / "trades.csv", trade_rows, overwrite=overwrite)
    report["predictions_sha256"] = _hash(data_root / "predictions.csv")
    report["source_sha256"] = _source_hashes()
    if report["source_sha256"] != lock["source_sha256"]:
        raise ValueError("Research code changed during evaluation")
    _write_json(root / "report.json", report, overwrite=overwrite)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("backtests/tp_horizon_symmetric_v1")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_horizon_study(args.root, overwrite=args.overwrite)
    except (ValueError, ArithmeticError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    selected = report["test_policies"][report["selected_policy_id"]]
    print(
        json.dumps(
            {
                "selected_policy_id": report["selected_policy_id"],
                "model": selected["model"],
                "feature_family": selected["feature_family"],
                "tp_equals_sl_fraction": selected["barrier_fraction"],
                "horizon_hours": selected["horizon_hours"],
                "threshold": selected["threshold"],
                "test_probability_skill": selected["skill"],
                "trades": selected["trading"]["round_trips"],
                "tp_hits": selected["trading"]["tp_hits"],
                "mean_probability": selected["trading"]["probability"][
                    "mean_probability"
                ],
                "net_rupees": selected["trading"]["net_rupees"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
