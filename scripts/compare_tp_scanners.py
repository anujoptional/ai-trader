"""Retrospective matched-opportunity comparison of rules and frozen CatBoost."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, time
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE, session_minute_offset
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine
from ai_trader.history import CandleStore
from ai_trader.market import Candle
from ai_trader.replay import (
    ExitReason,
    FillModel,
    ReplayPortfolio,
    ReplayResult,
    SimulatedTrade,
)
from ai_trader.scanner import Candidate, Direction, PortfolioState
from ai_trader.strategy import StrategyConfig
from scripts.research_signed_score import _trade_summary
from scripts.train_tp_only import file_hash, load_phase, read_manifest, write_json

FILL_SCENARIOS = {
    "standard": FillModel(Decimal(1), Decimal("0.0002"), Decimal("0.0001")),
    "adverse": FillModel(Decimal(10), Decimal("0.0004"), Decimal("0.0003")),
}


def matched_selections(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    sides: np.ndarray,
    rule_scores: np.ndarray,
    forecasts: dict[str, np.ndarray],
    eligible: np.ndarray,
    *,
    max_candidates: int = 5,
) -> dict[str, np.ndarray]:
    count = len(timestamps)
    arrays = (symbols, sides, rule_scores, eligible, *forecasts.values())
    if (
        max_candidates < 1
        or any(values.shape != (count,) for values in arrays)
        or not np.isin(sides, (-1, 1)).all()
        or any(not np.isfinite(values).all() for values in forecasts.values())
        or np.isinf(rule_scores).any()
        or "rules" in forecasts
    ):
        raise ValueError("Invalid matched scanner inputs")
    identities = np.rec.fromarrays((timestamps, symbols, sides))
    if len(np.unique(identities)) != count:
        raise ValueError("Duplicate timestamp, symbol and side")
    direction_order = np.where(sides == 1, 0, 1)
    selections = {name: np.zeros(count, dtype=bool) for name in ("rules", *forecasts)}
    order = np.argsort(timestamps, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(timestamps[order])) + 1)
    for group in groups:
        pool = group[eligible[group]]
        firing = pool[np.isfinite(rule_scores[pool])]
        budget = min(max_candidates, len(firing))
        if not budget:
            continue
        ranked_rules = np.lexsort(
            (direction_order[firing], symbols[firing], -np.abs(rule_scores[firing]))
        )
        selections["rules"][firing[ranked_rules[:budget]]] = True
        for name, probability in forecasts.items():
            ranked = np.lexsort(
                (direction_order[pool], symbols[pool], -probability[pool])
            )
            selections[name][pool[ranked[:budget]]] = True
    return selections


def isolated_case(
    signal: Candle,
    subsequent: Sequence[Candle],
    atr_fraction: Decimal,
    direction: Direction,
    strategy: StrategyConfig,
    fill: FillModel,
) -> SimulatedTrade:
    future = tuple(subsequent[:60])
    if (
        len(future) != 60
        or fill.latency_seconds <= 0
        or fill.latency_seconds >= 60
        or atr_fraction <= 0
        or not atr_fraction.is_finite()
        or session_minute_offset(future[-1].end_time) > 360
        or any(
            candle.instrument != signal.instrument
            or candle.start_time != signal.end_time + offset * ONE_MINUTE
            or candle.end_time != candle.start_time + ONE_MINUTE
            for offset, candle in enumerate(future)
        )
    ):
        raise ValueError(
            "Execution case requires 60 contiguous minutes and subminute fill"
        )
    with localcontext(FEATURE_CONTEXT):
        book = ReplayPortfolio(
            universe=(signal.instrument,),
            max_open_positions=1,
            sizing=strategy.sizing_policy(),
            costs=strategy.costs,
            fill=fill,
            exit_policy=strategy.exit_policy,
        )
        candidate = Candidate(
            instrument=signal.instrument,
            direction=direction,
            score=Decimal(1),
            rules=("matched_comparison",),
            as_of=signal.end_time,
            reference_price=signal.close,
        )
        pending = book.queue(candidate, atr_fraction)
        book.enter(pending, future[0].open, pending.fill_at, future[0].end_time)
        for candle in future:
            closed = book.advance(candle)
            if closed is not None:
                return closed
        reason = (
            ExitReason.SESSION_END
            if session_minute_offset(future[-1].end_time) == 360
            else ExitReason.HORIZON
        )
        trade = book.square_off(
            signal.instrument, future[-1].close, future[-1].end_time, reason=reason
        )
        if trade is None:
            raise ValueError("Complete execution case did not close")
        return trade


def simulate_ranked(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    sides: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    case_for: Callable[[int], SimulatedTrade],
    *,
    max_candidates: int = 5,
    max_open_positions: int = 3,
) -> list[tuple[int, SimulatedTrade]]:
    active: dict[str, SimulatedTrade] = {}
    selected: list[tuple[int, SimulatedTrade]] = []
    direction_order = np.where(sides == 1, 0, 1)
    order = np.argsort(timestamps, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(timestamps[order])) + 1)
    for group in groups:
        if not len(group):
            continue
        moment = datetime.fromtimestamp(int(timestamps[group[0]]), INDIA_TIMEZONE)
        active = {
            symbol: trade
            for symbol, trade in active.items()
            if trade.exit_time > moment
        }
        if len(active) >= max_open_positions:
            continue
        pool = np.asarray(
            [
                index
                for index in group
                if eligible[index]
                and np.isfinite(scores[index])
                and symbols[index] not in active
            ],
            dtype=np.int64,
        )
        ranked = np.lexsort(
            (direction_order[pool], symbols[pool], -np.abs(scores[pool]))
        )
        for index in pool[ranked[:max_candidates]]:
            symbol = str(symbols[index])
            if symbol in active or len(active) >= max_open_positions:
                continue
            trade = case_for(int(index))
            direction = Direction.LONG if sides[index] == 1 else Direction.SHORT
            if (
                trade.signal_time != moment
                or trade.instrument.trading_symbol != symbol
                or trade.direction is not direction
                or trade.exit_time <= moment
            ):
                raise ValueError("Execution case does not match the chosen opportunity")
            active[symbol] = trade
            selected.append(
                (int(index), replace(trade, score=Decimal(str(scores[index]))))
            )
    return selected


def score_candles(
    candles: Sequence[Candle], decision_epochs: set[int], strategy: StrategyConfig
) -> dict[int, dict]:
    features = FeatureEngine()
    scanner = strategy.scanner()
    feasibility = strategy.feasibility_policy()
    result = {}
    for position, candle in enumerate(candles):
        snapshot = features.update(candle)
        epoch = int(candle.end_time.timestamp())
        if snapshot is None or epoch not in decision_epochs:
            continue
        with localcontext(FEATURE_CONTEXT):
            ready = snapshot.readiness.atr_pct and snapshot.atr_pct is not None
            reachable = ready and (
                feasibility is None or feasibility.evaluate(snapshot).reachable
            )
            scan = scanner.scan((snapshot,), PortfolioState.empty(candle.end_time))
            result[epoch] = {
                "position": position,
                "eligible": bool(reachable),
                "atr_fraction": str(snapshot.atr_pct) if ready else "",
                "rules": {
                    1 if candidate.direction is Direction.LONG else -1: float(
                        candidate.score
                    )
                    for candidate in scan.candidates
                },
            }
    return result


def build_surface(
    phase: dict, manifest: dict, cache: Path, strategy: StrategyConfig
) -> tuple[dict, dict]:
    count = len(phase["labels"])
    arrays = {
        "rule_scores": np.full(count, np.nan),
        "eligible": np.zeros(count, dtype=bool),
        "atr_fraction": np.full(count, "", dtype="U64"),
        "bar_position": np.full(count, -1, dtype=np.int64),
    }
    evaluation_start = date.fromisoformat(manifest["phase_ends"]["calibration"])
    warm_days = [
        day for day in manifest["calendar"] if day <= evaluation_start.isoformat()
    ][-5:]
    if len(warm_days) != 5:
        raise ValueError("Scanner comparison requires five earlier warm-up sessions")
    beginning = datetime.combine(
        date.fromisoformat(warm_days[0]), time(9, 15), INDIA_TIMEZONE
    )
    ending = datetime.combine(
        date.fromisoformat(manifest["phase_ends"]["evaluation"]),
        time(15, 30),
        INDIA_TIMEZONE,
    )
    expected_sources = {
        Path(name).resolve(): value for name, value in manifest["source_sha256"].items()
    }
    store = CandleStore(cache)
    tapes = {}
    for symbol in np.unique(phase["symbols"]):
        instrument = Instrument("NSE", str(symbol))
        source = store.path_for(instrument)
        if (
            source.resolve() not in expected_sources
            or file_hash(source) != expected_sources[source.resolve()]
        ):
            raise ValueError(
                "Scanner candle source differs from the frozen model dataset"
            )
        candles = tuple(
            candle
            for candle in store.load(instrument, beginning, ending)
            if 0 <= session_minute_offset(candle.start_time) < 375
        )
        rows = np.flatnonzero(phase["symbols"] == symbol)
        epochs = {int(epoch) for epoch in phase["timestamps"][rows]}
        scored = score_candles(candles, epochs, strategy)
        if set(scored) != epochs:
            raise ValueError("Scanner candle alignment is incomplete")
        for index in rows:
            values = scored[int(phase["timestamps"][index])]
            arrays["eligible"][index] = values["eligible"]
            arrays["atr_fraction"][index] = values["atr_fraction"]
            arrays["bar_position"][index] = values["position"]
            arrays["rule_scores"][index] = values["rules"].get(
                int(phase["features"][index, -1]), np.nan
            )
        if file_hash(source) != expected_sources[source.resolve()]:
            raise ValueError("Candle source changed while reading")
        tapes[str(symbol)] = candles
        print(
            json.dumps(
                {
                    "symbol": str(symbol),
                    "bars": len(candles),
                    "actions": len(rows),
                    "eligible_actions": int(arrays["eligible"][rows].sum()),
                }
            ),
            flush=True,
        )
    return arrays, tapes


def touch_summary(
    phase: dict, selected: np.ndarray, probability: np.ndarray | None
) -> dict:
    count = int(selected.sum())
    if not count:
        return {"rows": 0, "tp_touch_rate": None}
    moments = phase["timestamps"][selected]
    side = phase["features"][selected, -1]
    labels = phase["labels"][selected]
    months = (moments + 19800).astype("datetime64[s]").astype("datetime64[M]")
    return {
        "rows": count,
        "tp_touches": int(labels.sum()),
        "tp_touch_rate": float(labels.mean()),
        "cycles": len(np.unique(moments)),
        "sessions": len(np.unique((moments + 19800) // 86400)),
        "mean_forecast": None
        if probability is None
        else float(probability[selected].mean()),
        "by_side": {
            str(direction): {
                "rows": int((side == direction).sum()),
                "tp_touch_rate": float(labels[side == direction].mean())
                if np.any(side == direction)
                else None,
            }
            for direction in (-1, 1)
        },
        "by_month": {
            str(month): {
                "rows": int((months == month).sum()),
                "tp_touch_rate": float(labels[months == month].mean()),
            }
            for month in np.unique(months)
        },
    }


def matched_hit_uncertainty(
    phase: dict, reference: np.ndarray, selected: np.ndarray
) -> dict:
    days, inverse = np.unique(
        (phase["timestamps"] + 19800) // 86400, return_inverse=True
    )
    counts = np.bincount(inverse, weights=reference)
    other_counts = np.bincount(inverse, weights=selected)
    if not np.array_equal(counts, other_counts) or not counts.sum():
        raise ValueError(
            "Paired hit-rate comparison requires equal nonzero daily counts"
        )
    differences = np.bincount(
        inverse,
        weights=phase["labels"] * (selected.astype(int) - reference.astype(int)),
    )
    generator = np.random.default_rng(20261004)
    block_size = min(5, len(days))
    starts = generator.integers(
        len(days), size=(1000, (len(days) + block_size - 1) // block_size)
    )
    samples = ((starts[:, :, None] + np.arange(block_size)) % len(days)).reshape(
        1000, -1
    )[:, : len(days)]
    totals = counts[samples].sum(axis=1)
    valid = totals > 0
    estimates = differences[samples].sum(axis=1)[valid] / totals[valid]
    return {
        "difference_vs_rules": float(differences.sum() / counts.sum()),
        "descriptive_95pct_interval": np.quantile(estimates, (0.025, 0.975)).tolist(),
        "method": (
            "paired circular five-session blocks; all assets and directions together"
        ),
        "bootstrap_repeats": 1000,
        "nonempty_repeats": int(valid.sum()),
    }


def run_comparison(data: Path, model: Path, cache: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(
            "Comparison output already exists; preserve the previous result"
        )
    manifest = read_manifest(data)
    model_report = json.loads((model / "report.json").read_text(encoding="utf-8"))
    lock = json.loads((model / "selection_lock.json").read_text(encoding="utf-8"))
    if (
        file_hash(data / "manifest.json") != lock["dataset_sha256"]
        or file_hash(model / "best.cbm") != lock["weights_sha256"]
        or file_hash(model / "calibration.json") != lock["calibration_sha256"]
        or file_hash(model / "evaluation_predictions.npz")
        != model_report["evaluation_predictions_sha256"]
        or model_report["selection_lock"] != lock
    ):
        raise ValueError("Frozen model or prediction provenance differs")
    strategy = StrategyConfig()
    source_paths = [
        Path(__file__),
        Path("scripts/train_tp_only.py"),
        Path("scripts/research_signed_score.py"),
        Path("src/ai_trader/clock.py"),
    ]
    for directory in (
        "features",
        "scanner",
        "replay",
        "strategy",
        "costs",
        "history",
        "market",
        "broker",
    ):
        source_paths.extend(Path("src/ai_trader", directory).rglob("*.py"))
    protocol = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "retrospective fixed-policy comparison, "
            "not model tuning or production promotion"
        ),
        "dataset_sha256": lock["dataset_sha256"],
        "model_weights_sha256": lock["weights_sha256"],
        "model_selection_lock_sha256": file_hash(model / "selection_lock.json"),
        "predictions_sha256": file_hash(model / "evaluation_predictions.npz"),
        "source_sha256": {path.as_posix(): file_hash(path) for path in source_paths},
        "strategy": strategy.to_dict(),
        "fills": {
            name: {key: str(value) for key, value in asdict(fill).items()}
            for name, fill in FILL_SCENARIOS.items()
        },
        "policies": [
            "rules",
            "catboost_raw",
            "catboost_calibrated",
            "rules_reranked_raw",
        ],
        "cadence": "same eligible five-minute model decisions, 09:40-14:15 IST",
        "ranking_comparison": "equal rule-selected count per timestamp, maximum five",
        "execution_comparison": (
            "same five-candidate/three-position budgets; actual trade counts may differ"
        ),
        "execution_horizon": (
            "exit at latest decision+60 minutes or 15:15; 2ATR stop remains active"
        ),
        "tick_warning": (
            "inherited uniform 0.05 INR research grid, "
            "NOT exchange-verified per symbol/date"
        ),
        "limitations": [
            *manifest["limitations"],
            "Existing scanner restricted to model-eligible cohort/cadence, "
            "not its unrestricted deployment",
            "No newly unseen test data; benchmark choices made "
            "after prior evaluation inspection",
            "Fill/spread/slippage and uniform tick grid are scenarios, "
            "not observed execution",
            "Predicting a TP touch without SL is not predicting "
            "a target exit before a stop",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    phase = load_phase(data, manifest, "evaluation")
    with np.load(
        model / "evaluation_predictions.npz", allow_pickle=False
    ) as predictions:
        for name in ("labels", "timestamps", "symbols"):
            if not np.array_equal(predictions[name], phase[name]):
                raise ValueError("Predictions and comparison rows do not align")
        if not np.array_equal(predictions["side"], phase["features"][:, -1]):
            raise ValueError("Prediction directions do not align")
        forecasts = {
            "catboost_raw": predictions["raw_probability"].copy(),
            "catboost_calibrated": predictions["probability"].copy(),
        }
    surface, tapes = build_surface(phase, manifest, cache, strategy)
    eligible, rule_scores = surface["eligible"], surface["rule_scores"]
    sides = phase["features"][:, -1]
    selected = matched_selections(
        phase["timestamps"], phase["symbols"], sides, rule_scores, forecasts, eligible
    )
    reranked = matched_selections(
        phase["timestamps"],
        phase["symbols"],
        sides,
        rule_scores,
        {"raw": forecasts["catboost_raw"]},
        eligible & np.isfinite(rule_scores),
    )
    selected["rules_reranked_raw"] = reranked["raw"]
    touch_results = {
        name: touch_summary(
            phase,
            mask,
            None
            if name == "rules"
            else forecasts["catboost_raw" if name == "rules_reranked_raw" else name],
        )
        for name, mask in selected.items()
    }
    uncertainty = {
        name: matched_hit_uncertainty(phase, selected["rules"], mask)
        for name, mask in selected.items()
        if name != "rules"
    }
    np.savez_compressed(
        output / "opportunities.npz",
        timestamps=phase["timestamps"],
        symbols=phase["symbols"],
        side=sides,
        labels=phase["labels"],
        **surface,
        **forecasts,
        **{name + "_selected": values for name, values in selected.items()},
    )
    scores = {
        "rules": rule_scores,
        **forecasts,
        "rules_reranked_raw": np.where(
            np.isfinite(rule_scores), forecasts["catboost_raw"], np.nan
        ),
    }
    universe = tuple(
        Instrument("NSE", str(symbol)) for symbol in np.unique(phase["symbols"])
    )
    sessions = tuple(
        date.fromisoformat(str(day))
        for day in np.unique(
            (phase["timestamps"] + 19800)
            .astype("datetime64[s]")
            .astype("datetime64[D]")
        )
    )
    execution = {}
    fields = [
        "scenario",
        "policy",
        "symbol",
        "side",
        "signal_time",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "target_price",
        "stop_price",
        "quantity",
        "score",
        "exit_reason",
        "net_rupees",
    ]
    with (output / "trades.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario, fill in FILL_SCENARIOS.items():
            case_cache = {}

            def case_for(index: int) -> SimulatedTrade:
                if index not in case_cache:
                    symbol = str(phase["symbols"][index])
                    position = int(surface["bar_position"][index])
                    tape = tapes[symbol]
                    case_cache[index] = isolated_case(
                        tape[position],
                        tape[position + 1 : position + 61],
                        Decimal(str(surface["atr_fraction"][index])),
                        Direction.LONG if sides[index] == 1 else Direction.SHORT,
                        strategy,
                        fill,
                    )
                return case_cache[index]

            execution[scenario] = {}
            for policy, values in scores.items():
                chosen = simulate_ranked(
                    phase["timestamps"],
                    phase["symbols"],
                    sides,
                    values,
                    eligible,
                    case_for,
                    max_candidates=strategy.max_candidates,
                    max_open_positions=strategy.max_open_positions,
                )
                trades = tuple(
                    sorted(
                        (trade for _, trade in chosen),
                        key=lambda trade: (
                            trade.exit_time,
                            trade.instrument.trading_symbol,
                            trade.direction.value,
                        ),
                    )
                )
                replay = ReplayResult(
                    trades=trades, universe=universe, sessions=sessions, fill=fill
                )
                result = _trade_summary(replay)
                result["by_month"] = {
                    month: _trade_summary(
                        ReplayResult(
                            trades=tuple(
                                trade
                                for trade in trades
                                if trade.session.isoformat().startswith(month)
                            ),
                            universe=universe,
                            sessions=tuple(
                                day
                                for day in sessions
                                if day.isoformat().startswith(month)
                            ),
                            fill=fill,
                        )
                    )
                    for month in sorted({day.isoformat()[:7] for day in sessions})
                }
                result["mean_selected_tp_forecast"] = (
                    None
                    if policy == "rules"
                    else float(np.mean([values[index] for index, _ in chosen]))
                    if chosen
                    else None
                )
                execution[scenario][policy] = result
                for _, trade in chosen:
                    writer.writerow(
                        {
                            "scenario": scenario,
                            "policy": policy,
                            "symbol": trade.instrument.trading_symbol,
                            "side": trade.direction.value,
                            **{name: str(getattr(trade, name)) for name in fields[4:]},
                        }
                    )
                handle.flush()
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "policy": policy,
                            "trades": result["round_trips"],
                            "net_rupees": result["net_rupees"],
                            "tp_exit_rate": result["tp_hit_rate"],
                        }
                    ),
                    flush=True,
                )
    if any(
        file_hash(Path(name)) != expected
        for name, expected in protocol["source_sha256"].items()
    ):
        raise ValueError("Comparison source changed during the run")
    report = {
        "status": "retrospective_comparison_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "opportunities_sha256": file_hash(output / "opportunities.npz"),
        "trades_sha256": file_hash(output / "trades.csv"),
        "coverage": {
            "paired_action_rows": len(sides),
            "sessions": len(sessions),
            "symbols": len(universe),
            "common_eligible_action_rows": int(eligible.sum()),
            "rule_firing_action_rows": int(np.isfinite(rule_scores).sum()),
        },
        "matched_tp_touch": touch_results,
        "matched_session_uncertainty": uncertainty,
        "execution": execution,
        "limitations": protocol["limitations"],
        "new_model_fits": 0,
        "live_orders": 0,
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_comparison(args.data, args.model, args.cache, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "coverage": report["coverage"],
                "matched_tp_touch": report["matched_tp_touch"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
