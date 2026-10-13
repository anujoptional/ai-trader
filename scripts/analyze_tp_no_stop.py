"""Measure frozen CatBoost calls with only TP or a 15:00 time exit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from decimal import localcontext
from pathlib import Path

import numpy as np

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE
from ai_trader.features import FEATURE_CONTEXT
from ai_trader.features.tp_only import TP_FRACTION, touch_label
from ai_trader.history import CandleStore
from ai_trader.market import Candle
from scripts.train_tp_only import (
    file_hash,
    load_phase,
    predict_logits,
    read_manifest,
    write_json,
)

POLICIES = ("catboost_raw", "catboost_calibrated")


def touch_by_deadline(
    signal: Candle, subsequent: Sequence[Candle], side: int, deadline: datetime
) -> bool | None:
    decision = signal.end_time.astimezone(INDIA_TIMEZONE)
    if (
        side not in (-1, 1)
        or signal.close <= 0
        or deadline.tzinfo is None
        or deadline <= decision
        or deadline.astimezone(INDIA_TIMEZONE).date() != decision.date()
        or (deadline - decision) % ONE_MINUTE
    ):
        raise ValueError("Invalid same-session TP deadline or direction")
    future = tuple(candle for candle in subsequent if candle.start_time < deadline)
    if len(future) != (deadline - decision) // ONE_MINUTE or any(
        candle.instrument != signal.instrument
        or candle.start_time != decision + offset * ONE_MINUTE
        or candle.end_time != candle.start_time + ONE_MINUTE
        for offset, candle in enumerate(future)
    ):
        return None
    with localcontext(FEATURE_CONTEXT):
        target = signal.close * (1 + side * TP_FRACTION)
        return any(
            candle.high >= target if side == 1 else candle.low <= target
            for candle in future
        )


def deadline_touch_labels(
    candles: Sequence[Candle],
    timestamps: np.ndarray,
    sides: np.ndarray,
    deadlines: np.ndarray,
) -> np.ndarray:
    if (
        not candles
        or timestamps.ndim != 1
        or timestamps.shape != sides.shape
        or timestamps.shape != deadlines.shape
        or not np.isin(sides, (-1, 1)).all()
        or np.any(deadlines <= timestamps)
        or np.any((deadlines - timestamps) % 60)
    ):
        raise ValueError("Invalid aligned same-session deadlines")
    starts = np.asarray([int(bar.start_time.timestamp()) for bar in candles])
    ends = np.asarray([int(bar.end_time.timestamp()) for bar in candles])
    session = candles[0].start_time.astimezone(INDIA_TIMEZONE).date()
    if (
        np.any(starts[1:] <= starts[:-1])
        or np.any(ends != starts + 60)
        or any(
            bar.instrument != candles[0].instrument
            or bar.start_time.astimezone(INDIA_TIMEZONE).date() != session
            for bar in candles
        )
        or any(
            datetime.fromtimestamp(int(epoch), INDIA_TIMEZONE).date() != session
            for epoch in (*timestamps, *deadlines)
        )
    ):
        raise ValueError("Candles must be ordered within one instrument session")
    locations = np.searchsorted(ends, timestamps)
    if np.any(locations == len(candles)) or np.any(ends[locations] != timestamps):
        raise ValueError("Missing signal candle")
    prices = [(bar.close, bar.high, bar.low) for bar in candles]
    if any(not value.is_finite() or value <= 0 for row in prices for value in row):
        raise ValueError("Invalid candle price")
    scale = max(0, max(-value.as_tuple().exponent for row in prices for value in row))
    with localcontext(FEATURE_CONTEXT):
        scaled = [[int(value * 10**scale) for value in row] for row in prices]
        numerators = [int((1 + int(side) * TP_FRACTION) * 1000) for side in sides]
    if max(value for row in scaled for value in row) > np.iinfo(np.int64).max // 1002:
        raise ValueError("Prices exceed exact integer touch arithmetic range")
    values = np.asarray(scaled, dtype=np.int64)
    targets = values[locations, 0] * np.asarray(numerators, dtype=np.int64)
    future = (starts[None, :] >= timestamps[:, None]) & (
        ends[None, :] <= deadlines[:, None]
    )
    complete = future.sum(axis=1) == (deadlines - timestamps) // 60
    touches = np.where(
        sides[:, None] == 1,
        values[None, :, 1] * 1000 >= targets[:, None],
        values[None, :, 2] * 1000 <= targets[:, None],
    )
    return np.where(complete, (touches & future).any(axis=1), -1).astype(np.int8)


def deadline_returns(
    candles: Sequence[Candle],
    timestamps: np.ndarray,
    sides: np.ndarray,
    deadlines: np.ndarray,
) -> np.ndarray:
    complete = deadline_touch_labels(candles, timestamps, sides, deadlines) >= 0
    closes = {int(candle.end_time.timestamp()): candle.close for candle in candles}
    returns = np.full(len(timestamps), np.nan, dtype=np.float64)
    with localcontext(FEATURE_CONTEXT):
        for index in np.flatnonzero(complete):
            returns[index] = float(
                int(sides[index])
                * (closes[int(deadlines[index])] / closes[int(timestamps[index])] - 1)
            )
    return returns


def frequency_and_return_summary(arrays: dict, returns: np.ndarray) -> dict:
    timestamps, symbols, sides, probability, hits = (
        arrays[name]
        for name in ("timestamps", "symbols", "side", "raw_probability", "hit_by_1500")
    )
    if (
        not len(timestamps)
        or any(
            values.shape != timestamps.shape or values.ndim != 1
            for values in (symbols, sides, probability, hits, returns)
        )
        or not np.isin(sides, (-1, 1)).all()
        or not np.isin(hits, (-1, 0, 1)).all()
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
        or np.isinf(returns).any()
    ):
        raise ValueError("Invalid aligned frequency/return inputs")
    order = np.lexsort((sides, timestamps, symbols))
    if len(order) % 2:
        raise ValueError("Expected one LONG/SHORT pair per scored candle")
    pairs = order.reshape(-1, 2)
    if (
        not np.array_equal(sides[pairs], np.tile([-1, 1], (len(pairs), 1)))
        or np.any(timestamps[pairs[:, 0]] != timestamps[pairs[:, 1]])
        or np.any(symbols[pairs[:, 0]] != symbols[pairs[:, 1]])
        or np.any(
            (timestamps[pairs[1:, 0]] == timestamps[pairs[:-1, 0]])
            & (symbols[pairs[1:, 0]] == symbols[pairs[:-1, 0]])
        )
    ):
        raise ValueError("Expected unique LONG/SHORT pairs per scored candle")
    days, day_ids = np.unique((timestamps + 19800) // 86400, return_inverse=True)
    bins = np.searchsorted(np.arange(1, 10) / 10, probability, side="right")
    best_side_bins = bins[pairs].max(axis=1)

    def return_statistics(values: np.ndarray) -> dict:
        if not len(values):
            return {"rows": 0}
        return {
            "rows": len(values),
            "mean_fraction": float(values.mean()),
            "median_fraction": float(np.median(values)),
            "minimum_fraction": float(values.min()),
            "maximum_fraction": float(values.max()),
            "p10_fraction": float(np.quantile(values, 0.1)),
            "p90_fraction": float(np.quantile(values, 0.9)),
            "negative_rows": int((values < 0).sum()),
            "zero_rows": int((values == 0).sum()),
            "positive_rows": int((values > 0).sum()),
        }

    def summarize(selected: np.ndarray) -> dict:
        daily = np.bincount(day_ids[selected], minlength=len(days))
        chosen_pairs = selected[pairs]
        distinct = int(chosen_pairs.any(axis=1).sum())
        known = selected & (hits >= 0) & np.isfinite(returns)
        failures = known & (hits == 0)
        gross = np.where(hits[known] == 1, float(TP_FRACTION), returns[known])
        return {
            "forecast_rows": int(selected.sum()),
            "percentage_of_all_forecasts": float(100 * selected.mean()),
            "long_forecasts": int((selected & (sides == 1)).sum()),
            "short_forecasts": int((selected & (sides == -1)).sum()),
            "mean_forecasts_per_session": float(daily.mean()),
            "median_forecasts_per_session": float(np.median(daily)),
            "p90_forecasts_per_session": float(np.quantile(daily, 0.9)),
            "maximum_forecasts_per_session": int(daily.max()),
            "sessions_with_forecasts": int((daily > 0).sum()),
            "sessions_without_forecasts": int((daily == 0).sum()),
            "unique_scored_candles_with_either_side": distinct,
            "percentage_of_scored_candles_with_either_side": 100
            * distinct
            / len(pairs),
            "mean_unique_scored_candles_per_session": distinct / len(days),
            "scored_candles_with_both_sides": int(chosen_pairs.all(axis=1).sum()),
            "decision_cycles_with_forecasts": len(np.unique(timestamps[selected])),
            "complete_return_rows": int(known.sum()),
            "unavailable_return_rows": int((selected & ~known).sum()),
            "tp_hits_by_1500": int((known & (hits == 1)).sum()),
            "no_tp_by_1500": return_statistics(returns[failures]),
            "idealized_tp_or_1500_gross_return": return_statistics(gross),
        }

    return {
        "forecast_rows": len(timestamps),
        "scored_candles": len(pairs),
        "sessions": len(days),
        "symbols": len(np.unique(symbols)),
        "decision_cycles": len(np.unique(timestamps)),
        "first_decision_ist": datetime.fromtimestamp(
            int(timestamps.min()), INDIA_TIMEZONE
        ).isoformat(),
        "last_decision_ist": datetime.fromtimestamp(
            int(timestamps.max()), INDIA_TIMEZONE
        ).isoformat(),
        "all": summarize(np.ones(len(timestamps), dtype=bool)),
        "bins": [
            {
                "lower": bin_id / 10,
                "upper": (bin_id + 1) / 10,
                **summarize(bins == bin_id),
            }
            for bin_id in range(10)
        ],
        "scored_candle_bins_by_max_side_probability": [
            {
                "lower": bin_id / 10,
                "upper": (bin_id + 1) / 10,
                "scored_candles": int((best_side_bins == bin_id).sum()),
                "percentage_of_scored_candles": float(
                    100 * (best_side_bins == bin_id).mean()
                ),
            }
            for bin_id in range(10)
        ],
        "thresholds_not_trade_counts": {
            str(threshold): summarize(probability >= threshold)
            for threshold in (0.6, 0.7, 0.8, 0.9)
        },
        "by_side": {str(side): summarize(sides == side) for side in (-1, 1)},
    }


def probability_reliability(
    labels: np.ndarray, probability: np.ndarray, timestamps: np.ndarray
) -> dict:
    from scripts.report_tp_catboost import independent_metrics

    if (
        labels.shape != probability.shape
        or labels.shape != timestamps.shape
        or labels.ndim != 1
        or not np.isin(labels, (-1, 0, 1)).all()
        or not np.isfinite(probability).all()
        or np.any((probability < 0) | (probability > 1))
    ):
        raise ValueError("Invalid aligned probability/outcome arrays")
    known = labels >= 0
    observed, forecast = labels[known], probability[known]
    metrics = independent_metrics(observed, forecast)
    metrics.update(
        unavailable_rows=int((~known).sum()),
        sessions=len(np.unique((timestamps[known] + 19800) // 86400)),
        probability_outcome_correlation=(
            float(np.corrcoef(forecast, observed)[0, 1])
            if len(observed) > 1 and forecast.std() > 0 and observed.std() > 0
            else None
        ),
    )
    bin_ids = np.searchsorted(np.arange(1, 10) / 10, probability, side="right")
    bins = []
    for bin_id in range(10):
        selected = bin_ids == bin_id
        available = selected & known
        entry = {
            "lower": bin_id / 10,
            "upper": (bin_id + 1) / 10,
            "unavailable_rows": int((selected & ~known).sum()),
            **independent_metrics(labels[available], probability[available]),
        }
        bins.append(entry)
    populated = [entry for entry in bins if entry["rows"]]
    if len(populated) > 1:
        weights = np.asarray([entry["rows"] for entry in populated], dtype=float)
        means = np.asarray([entry["mean_probability"] for entry in populated])
        rates = np.asarray([entry["hit_rate"] for entry in populated])
        centered_means = means - np.average(means, weights=weights)
        centered_rates = rates - np.average(rates, weights=weights)
        denominator = np.sqrt(
            np.sum(weights * centered_means**2) * np.sum(weights * centered_rates**2)
        )
        metrics["weighted_probability_bin_correlation"] = (
            float(np.sum(weights * centered_means * centered_rates) / denominator)
            if denominator > 0
            else None
        )
    else:
        metrics["weighted_probability_bin_correlation"] = None
    return {"all": metrics, "bins": bins}


def data_quality_summary(validation: dict, capture: dict, session_dates: list) -> dict:
    from ai_trader.cli.migrate_kite_data import windows_for

    days = [date.fromisoformat(day) for day in session_dates]

    def minutes_on(day: date, cas: bool) -> int:
        return sum(
            (datetime.combine(day, closing) - datetime.combine(day, opening))
            // ONE_MINUTE
            for opening, closing in windows_for(day, cas)
        )

    rows = []
    for name, checked in validation["instruments"].items():
        source = capture["instruments"][name]
        cas = source["role"] == "equity_target"
        quality = source["quality"]["usable"]
        expected = sum(minutes_on(day, cas) for day in days)
        quarantined = sum(
            minutes_on(date.fromisoformat(day), cas)
            for day in checked["quarantined_dates"]
        )
        absent = sum(
            minutes_on(date.fromisoformat(day), cas)
            for day in checked["other_absent_dates"]
        )
        missing = checked["missing_minutes_on_observed_dates"]
        usable = checked["minute_rows"] - quality["outside_session_rows"]
        if expected != usable + missing + quarantined + absent:
            raise ValueError("Minute-slot quality accounting does not reconcile")
        rows.append(
            {
                "instrument": name,
                "role": source["role"],
                "expected_slots": expected,
                "usable_slots": usable,
                "missing_slots": missing,
                "quarantined_slots": quarantined,
                "quarantined_instrument_days": len(checked["quarantined_dates"]),
                "other_absent_slots": absent,
                "outside_session_rows": quality["outside_session_rows"],
                "unknown_volume_rows": quality["unknown_volume"],
                "critical_issues": len(quality["critical_issues"]),
            }
        )

    def totals(selected: list[dict]) -> dict:
        result = {
            key: sum(row[key] for row in selected)
            for key in rows[0]
            if key not in ("instrument", "role")
        }
        result["instruments"] = len(selected)
        result["excluded_slots"] = sum(
            result[key]
            for key in ("missing_slots", "quarantined_slots", "other_absent_slots")
        )
        result["excluded_percentage"] = (
            100 * result["excluded_slots"] / result["expected_slots"]
        )
        return result

    return {
        "calendar_sessions": len(days),
        "calendar_basis": "Captured union dates; not independent exchange coverage",
        "all_instruments": totals(rows),
        "equity_targets": totals(
            [row for row in rows if row["role"] == "equity_target"]
        ),
        "model_input_instruments": totals(
            [
                row
                for row in rows
                if row["role"] == "equity_target"
                or row["instrument"] in ("NSE:NIFTY", "NSE:GOLDBEES", "NSE:INDIAVIX")
            ]
        ),
        "by_instrument": rows,
        "independently_checked_labels": validation["independent_labels_checked"],
        "independent_daily_comparisons": len(validation["nse_comparisons"]),
    }


def measure_call(signal: Candle, subsequent: Sequence[Candle], side: int) -> dict:
    decision = signal.end_time.astimezone(INDIA_TIMEZONE)
    cutoff = datetime.combine(decision.date(), time(15), INDIA_TIMEZONE)
    hour_end = decision + 60 * ONE_MINUTE
    if side not in (-1, 1) or signal.close <= 0 or decision >= cutoff:
        raise ValueError("Call must have a valid direction and precede 15:00")
    horizon = max(cutoff, hour_end)
    future = tuple(candle for candle in subsequent if candle.start_time < horizon)
    count = (horizon - decision) // ONE_MINUTE
    if len(future) != count or any(
        candle.instrument != signal.instrument
        or candle.start_time != decision + offset * ONE_MINUTE
        or candle.end_time != candle.start_time + ONE_MINUTE
        for offset, candle in enumerate(future)
    ):
        raise ValueError("Missing or unordered minutes in the call outcome window")
    with localcontext(FEATURE_CONTEXT):
        target = signal.close * (1 + side * TP_FRACTION)
        touches = [
            candle.end_time
            for candle in future
            if (candle.high >= target if side == 1 else candle.low <= target)
        ]
        first_hour = next((moment for moment in touches if moment <= hour_end), None)
        first_day = next((moment for moment in touches if moment <= cutoff), None)
        square_off = next(
            candle.close for candle in future if candle.end_time == cutoff
        )
        gross_return = side * (square_off / signal.close - 1)
    return {
        "side": side,
        "decision_time": decision.isoformat(),
        "reference_price": str(signal.close),
        "target_price": str(target),
        "hit_60m": first_hour is not None,
        "hit_60m_before_1500": first_hour is not None and first_hour <= cutoff,
        "hit_by_1500": first_day is not None,
        "first_touch_60m": None if first_hour is None else first_hour.isoformat(),
        "first_touch_by_1500": None if first_day is None else first_day.isoformat(),
        "full_hour_before_1500": hour_end <= cutoff,
        "square_off_time": cutoff.isoformat(),
        "square_off_price": str(square_off),
        "return_at_1500_fraction": str(gross_return),
        "tp_or_1500_return_fraction": str(TP_FRACTION if first_day else gross_return),
    }


def summarize_calls(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"calls": 0}
    complete = [row for row in rows if row["day_outcome_available"]]
    failed = [row for row in complete if not row["hit_by_1500"]]
    full_hour = [row for row in rows if row["full_hour_before_1500"]]
    result = {
        "calls": len(rows),
        "long_calls": sum(row["side"] == 1 for row in rows),
        "short_calls": sum(row["side"] == -1 for row in rows),
        "hit_60m_count": sum(row["hit_60m"] for row in rows),
        "hit_60m_rate": sum(row["hit_60m"] for row in rows) / len(rows),
        "hit_60m_before_1500_count": sum(row["hit_60m_before_1500"] for row in rows),
        "hit_60m_before_1500_rate": sum(row["hit_60m_before_1500"] for row in rows)
        / len(rows),
        "calls_with_full_hour_before_1500": len(full_hour),
        "hit_60m_rate_on_full_hour_calls": (
            sum(row["hit_60m"] for row in full_hour) / len(full_hour)
            if full_hour
            else None
        ),
        "late_calls_with_less_than_60m_to_square_off": len(rows) - len(full_hour),
        "literal_60m_success_only_after_1500": sum(
            row["hit_60m"] and not row["hit_60m_before_1500"] for row in rows
        ),
        "day_outcome_available_calls": len(complete),
        "day_outcome_unavailable_calls": len(rows) - len(complete),
        "hit_by_1500_count": sum(row["hit_by_1500"] for row in complete),
        "hit_by_1500_rate": (
            sum(row["hit_by_1500"] for row in complete) / len(complete)
            if complete
            else None
        ),
        "first_hit_after_60m_but_by_1500": sum(
            row["hit_by_1500"] and not row["hit_60m"] for row in complete
        ),
        "never_hit_by_1500_count": len(failed),
        "never_hit_return_at_1500": {"calls": len(failed)},
    }
    if failed:
        returns = np.asarray([float(row["return_at_1500_fraction"]) for row in failed])
        result["never_hit_return_at_1500"].update(
            mean_fraction=float(returns.mean()),
            median_fraction=float(np.median(returns)),
            minimum_fraction=float(returns.min()),
            maximum_fraction=float(returns.max()),
            p10_fraction=float(np.quantile(returns, 0.1)),
            p90_fraction=float(np.quantile(returns, 0.9)),
            losing_calls=int((returns < 0).sum()),
            flat_calls=int((returns == 0).sum()),
            positive_below_tp_calls=int((returns > 0).sum()),
        )
    if complete:
        result["mean_tp_or_1500_gross_return_fraction"] = float(
            np.mean([float(row["tp_or_1500_return_fraction"]) for row in complete])
        )
    return result


def analyze_calls(data: Path, comparison: Path, cache: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Analysis output already exists; preserve earlier results")
    manifest = read_manifest(data)
    comparison_report = json.loads(
        (comparison / "report.json").read_text(encoding="utf-8")
    )
    comparison_protocol = json.loads(
        (comparison / "protocol.json").read_text(encoding="utf-8")
    )
    if (
        file_hash(data / "manifest.json") != comparison_protocol["dataset_sha256"]
        or file_hash(comparison / "protocol.json")
        != comparison_report["protocol_sha256"]
        or file_hash(comparison / "opportunities.npz")
        != comparison_report["opportunities_sha256"]
    ):
        raise ValueError(
            "Frozen signal provenance differs from the recorded comparison"
        )
    with np.load(comparison / "opportunities.npz", allow_pickle=False) as saved:
        arrays = {
            name: saved[name].copy()
            for name in (
                "timestamps",
                "symbols",
                "side",
                "labels",
                *POLICIES,
                *(policy + "_selected" for policy in POLICIES),
            )
        }
    count = len(arrays["timestamps"])
    if any(values.shape != (count,) for values in arrays.values()):
        raise ValueError("Misaligned frozen signals")
    if (
        not np.isin(arrays["side"], (-1, 1)).all()
        or not np.isin(arrays["labels"], (0, 1)).all()
    ):
        raise ValueError("Invalid frozen directions or labels")
    selected = np.zeros(count, dtype=bool)
    for policy in POLICIES:
        mask = arrays[policy + "_selected"]
        if mask.dtype != np.bool_ or not np.isfinite(arrays[policy]).all():
            raise ValueError("Invalid frozen selection mask or probability")
        recorded = comparison_report["matched_tp_touch"][policy]
        if (
            int(mask.sum()) != recorded["rows"]
            or int(arrays["labels"][mask].sum()) != recorded["tp_touches"]
        ):
            raise ValueError("Frozen signal count or hourly outcomes differ")
        selected |= mask
    if not selected.any():
        raise ValueError("No frozen CatBoost calls to analyze")
    source_files = [
        Path(__file__),
        Path("src/ai_trader/features/tp_only.py"),
        Path("src/ai_trader/clock.py"),
        Path("src/ai_trader/history/store.py"),
        Path(__file__).with_name("train_tp_only.py"),
    ]
    protocol = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": file_hash(data / "manifest.json"),
        "comparison_report_sha256": file_hash(comparison / "report.json"),
        "frozen_opportunities_sha256": comparison_report["opportunities_sha256"],
        "weights_sha256": comparison_protocol["model_weights_sha256"],
        "policies": list(POLICIES),
        "signal_cohort": (
            "Exactly the earlier frozen matched top-five-per-cycle calls; "
            "inherited eligibility and rule-derived count, "
            "no new threshold or selection"
        ),
        "tp_fraction": str(TP_FRACTION),
        "stop_loss": None,
        "reference": "last completed signal candle close",
        "square_off_ist": "15:00",
        "whole_remaining_day": (
            "all subsequent complete minute bars ending by 15:00 IST"
        ),
        "hourly_measurements": (
            "Literal next 60 minutes plus a separate min(60 minutes, 15:00) metric; "
            "post-15:00 hourly hits cannot count as realized TP exits"
        ),
        "first_future_minute_included": True,
        "square_off_price": "close of 14:59-15:00 candle, no next-candle lookahead",
        "return_formula": "side * (price_at_1500 / signal_close - 1)",
        "missing_minutes": (
            "Unknown whole-day outcomes excluded, never counted as failed TP"
        ),
        "fees_spread_slippage": (
            "excluded: price-touch accuracy and gross returns, not a fill simulation"
        ),
        "portfolio": (
            "independent signal outcomes, no position cap or overlap suppression"
        ),
        "source_sha256": {path.as_posix(): file_hash(path) for path in source_files},
        "limitations": [
            *manifest["limitations"],
            "15:00 is the newly requested time exit; "
            "old 15:15/2ATR studies remain separate",
            "Calls reuse the previous selection cohort; "
            "this is not a new all-market entry policy",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    store = CandleStore(cache)
    expected_sources = {
        Path(name).resolve(): digest
        for name, digest in manifest["source_sha256"].items()
    }
    results: dict[str, list[dict]] = {policy: [] for policy in POLICIES}
    fields = [
        "policy",
        "symbol",
        "side",
        "decision_time",
        "probability",
        "reference_price",
        "target_price",
        "hit_60m",
        "hit_60m_before_1500",
        "hit_by_1500",
        "first_touch_60m",
        "first_touch_by_1500",
        "full_hour_before_1500",
        "square_off_time",
        "square_off_price",
        "return_at_1500_fraction",
        "tp_or_1500_return_fraction",
        "day_outcome_available",
    ]
    with (output / "calls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for symbol in np.unique(arrays["symbols"][selected]):
            indices = np.flatnonzero(selected & (arrays["symbols"] == symbol))
            epochs = arrays["timestamps"][indices]
            beginning = datetime.fromtimestamp(
                int(epochs.min()), INDIA_TIMEZONE
            ).replace(hour=9, minute=15)
            ending = datetime.fromtimestamp(int(epochs.max()), INDIA_TIMEZONE).replace(
                hour=15, minute=15
            )
            instrument = Instrument("NSE", str(symbol))
            source = store.path_for(instrument)
            if (
                source.resolve() not in expected_sources
                or file_hash(source) != expected_sources[source.resolve()]
            ):
                raise ValueError("Candle data differs from the frozen model source")
            grouped = defaultdict(list)
            for candle in store.load(instrument, beginning, ending):
                grouped[candle.start_time.astimezone(INDIA_TIMEZONE).date()].append(
                    candle
                )
            locations = {
                int(candle.end_time.timestamp()): (bars, position)
                for bars in grouped.values()
                for position, candle in enumerate(bars)
            }
            unavailable = 0
            for index in indices:
                epoch = int(arrays["timestamps"][index])
                if epoch not in locations:
                    raise ValueError("Missing signal candle")
                bars, position = locations[epoch]
                signal = bars[position]
                future = bars[position + 1 :]
                direction = int(arrays["side"][index])
                hourly = touch_label(signal, future, direction)
                if hourly.hit != bool(arrays["labels"][index]):
                    raise ValueError(
                        "Recomputed hourly label differs from frozen outcome"
                    )
                try:
                    outcome = {
                        **measure_call(signal, future, direction),
                        "day_outcome_available": True,
                    }
                except ValueError as error:
                    if not str(error).startswith("Missing or unordered"):
                        raise
                    unavailable += 1
                    cutoff = datetime.combine(
                        signal.end_time.date(), time(15), INDIA_TIMEZONE
                    )
                    first_hour = (
                        (signal.end_time + hourly.first_hit_minute * ONE_MINUTE)
                        if hourly.hit
                        else None
                    )
                    outcome = {
                        "side": direction,
                        "decision_time": signal.end_time.isoformat(),
                        "reference_price": str(signal.close),
                        "target_price": str(
                            signal.close * (1 + direction * TP_FRACTION)
                        ),
                        "hit_60m": hourly.hit,
                        "hit_60m_before_1500": first_hour is not None
                        and first_hour <= cutoff,
                        "full_hour_before_1500": signal.end_time + 60 * ONE_MINUTE
                        <= cutoff,
                        "first_touch_60m": None
                        if first_hour is None
                        else first_hour.isoformat(),
                        "square_off_time": cutoff.isoformat(),
                        "day_outcome_available": False,
                    }
                for policy in POLICIES:
                    if arrays[policy + "_selected"][index]:
                        row = {
                            "policy": policy,
                            "symbol": str(symbol),
                            "probability": float(arrays[policy][index]),
                            **outcome,
                        }
                        results[policy].append(row)
                        writer.writerow(row)
            handle.flush()
            if file_hash(source) != expected_sources[source.resolve()]:
                raise ValueError("Candle data changed during analysis")
            print(
                json.dumps(
                    {
                        "symbol": str(symbol),
                        "unique_calls": len(indices),
                        "incomplete_whole_day_windows": unavailable,
                    }
                ),
                flush=True,
            )
    for source, digest in protocol["source_sha256"].items():
        if file_hash(Path(source)) != digest:
            raise ValueError("Analysis source changed during the run")
    report = {
        "status": "no_stop_tp_or_1500_analysis_complete",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "calls_sha256": file_hash(output / "calls.csv"),
        "policies": {
            policy: {
                "all": summarize_calls(rows),
                "by_side": {
                    str(side): summarize_calls(
                        [row for row in rows if row["side"] == side]
                    )
                    for side in (1, -1)
                },
                "by_month": {
                    month: summarize_calls(
                        [row for row in rows if row["decision_time"].startswith(month)]
                    )
                    for month in sorted({row["decision_time"][:7] for row in rows})
                },
            }
            for policy, rows in results.items()
        },
        "new_model_fits": 0,
        "live_orders": 0,
        "signal_first_date": datetime.fromtimestamp(
            int(arrays["timestamps"][selected].min()), INDIA_TIMEZONE
        )
        .date()
        .isoformat(),
        "signal_last_date": datetime.fromtimestamp(
            int(arrays["timestamps"][selected].max()), INDIA_TIMEZONE
        )
        .date()
        .isoformat(),
        "symbols": len(np.unique(arrays["symbols"][selected])),
        "sessions": len(np.unique((arrays["timestamps"][selected] + 19800) // 86400)),
        "limitations": protocol["limitations"],
    }
    write_json(output / "report.json", report)
    return report


def analyze_forecasts(
    data: Path, study: Path, cache: Path, validation_path: Path, output: Path
) -> dict:
    from scipy.special import expit

    from ai_trader.cli.migrate_kite_data import windows_for
    from scripts.research_tp_catboost import load_frozen_model

    if output.exists():
        raise ValueError("Analysis output already exists; preserve earlier results")
    manifest = read_manifest(data)
    dataset = json.loads((cache / "dataset.json").read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    final = json.loads((study / "final/report.json").read_text(encoding="utf-8"))
    top = json.loads((study / "report.json").read_text(encoding="utf-8"))
    model, _, lock = load_frozen_model(study / "final")
    predictions = study / "final/evaluation_predictions.npz"
    if (
        dataset["provider"] != "kite"
        or manifest["provider"] != "kite"
        or validation["dataset_sha256"] != file_hash(cache / "dataset.json")
        or validation["research_sha256"] != file_hash(data / "manifest.json")
        or lock["dataset_sha256"] != file_hash(data / "manifest.json")
        or final["evaluation_predictions_sha256"] != file_hash(predictions)
        or top["final_report_sha256"] != file_hash(study / "final/report.json")
    ):
        raise ValueError("Frozen Kite data/model/forecast provenance differs")
    for name, expected in {
        **manifest["source_sha256"],
        **manifest["code_sha256"],
    }.items():
        if file_hash(Path(name)) != expected:
            raise ValueError("Research source fingerprint changed")
    store = CandleStore(cache)
    for relative in dataset["files"]:
        store.verified_path(relative)
    capture = json.loads(store.verified_path("report.json").read_text(encoding="utf-8"))
    quality = data_quality_summary(validation, capture, dataset["session_dates"])
    with np.load(predictions, allow_pickle=False) as saved:
        arrays = {
            name: saved[name].copy()
            for name in ("timestamps", "symbols", "side", "labels", "raw_probability")
        }
    evaluation = load_phase(data, manifest, "evaluation")
    for name in ("timestamps", "symbols", "labels"):
        if not np.array_equal(arrays[name], evaluation[name]):
            raise ValueError("Saved forecast cohort differs from evaluation data")
    if not np.array_equal(
        arrays["side"], evaluation["features"][:, -1]
    ) or not np.array_equal(
        arrays["raw_probability"],
        expit(predict_logits(model, evaluation["features"], 1024)),
    ):
        raise ValueError("Raw forecasts did not reproduce from frozen weights")
    del evaluation, model
    sources = [
        Path(__file__),
        Path("scripts/report_tp_catboost.py"),
        Path("scripts/train_tp_only.py"),
        Path("scripts/research_tp_catboost.py"),
        Path("src/ai_trader/cli/migrate_kite_data.py"),
        Path("src/ai_trader/history/store.py"),
        Path("src/ai_trader/clock.py"),
    ]
    protocol = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": file_hash(cache / "dataset.json"),
        "research_manifest_sha256": file_hash(data / "manifest.json"),
        "selection_lock_sha256": file_hash(study / "final/selection_lock.json"),
        "predictions_sha256": file_hash(predictions),
        "data_validation_sha256": file_hash(validation_path),
        "source_sha256": {path.as_posix(): file_hash(path) for path in sources},
        "cohort": (
            "All frozen raw evaluation side-actions; "
            "no score filtering or new selection"
        ),
        "tp_fraction": str(TP_FRACTION),
        "reference": "Last completed close, first future minute included, no stop loss",
        "horizons": {
            "hit_60m": "Literal next 60 complete minutes",
            "hit_by_1500": "Remaining time through 15:00 IST, the user exit cutoff",
            "hit_by_session_close": (
                "Remaining continuous session: 15:30 before August 3, 2026; "
                "15:15 for equity targets thereafter; excludes closing auctions"
            ),
        },
        "unknown_outcomes": "Incomplete windows excluded, not labeled failures",
        "probability_bins": (
            "Ten disjoint equal-width bins; lower inclusive, final upper inclusive"
        ),
        "limits": [
            "Retrospective dependent forecasts, not independent trades or net profit",
            "60-minute forecasts are not recalibrated to the remaining-session target",
            "Session close follows recorded migration policy, "
            "not point-in-time CAS eligibility",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), output / "analysis_source.py")
    horizons = {
        name: np.full(len(arrays["timestamps"]), -1, dtype=np.int8)
        for name in protocol["horizons"]
    }
    closing_times = np.zeros(len(arrays["timestamps"]), dtype=np.int64)
    for symbol in np.unique(arrays["symbols"]):
        chosen = np.flatnonzero(arrays["symbols"] == symbol)
        epochs = arrays["timestamps"][chosen]
        beginning = datetime.fromtimestamp(int(epochs.min()), INDIA_TIMEZONE).replace(
            hour=9, minute=15
        )
        ending = datetime.fromtimestamp(int(epochs.max()), INDIA_TIMEZONE).replace(
            hour=15, minute=30
        )
        instrument = Instrument("NSE", str(symbol))
        source = store.path_for(instrument)
        original_hash = file_hash(source)
        grouped = defaultdict(list)
        for candle in store.load(instrument, beginning, ending, allow_incomplete=True):
            grouped[candle.start_time.astimezone(INDIA_TIMEZONE).date()].append(candle)
        local_days = (epochs + 19800) // 86400
        for day_id in np.unique(local_days):
            indices = chosen[local_days == day_id]
            decisions = arrays["timestamps"][indices]
            day = datetime.fromtimestamp(int(decisions[0]), INDIA_TIMEZONE).date()
            closing = windows_for(day, cas=True)[-1][1]
            session_end = int(
                datetime.combine(day, closing, INDIA_TIMEZONE).timestamp()
            )
            cutoff = int(datetime.combine(day, time(15), INDIA_TIMEZONE).timestamp())
            deadlines = {
                "hit_60m": decisions + 3600,
                "hit_by_1500": np.full(len(indices), cutoff),
                "hit_by_session_close": np.full(len(indices), session_end),
            }
            for name, limits in deadlines.items():
                horizons[name][indices] = deadline_touch_labels(
                    grouped[day], decisions, arrays["side"][indices], limits
                )
            closing_times[indices] = session_end
        if file_hash(source) != original_hash:
            raise ValueError("Candle source changed during outcome calculation")
        print(json.dumps({"symbol": str(symbol), "actions": len(chosen)}), flush=True)
    if not np.array_equal(horizons["hit_60m"], arrays["labels"]):
        raise ValueError("Recomputed 60-minute outcomes differ from frozen labels")
    timestamps = arrays["timestamps"]
    probability = arrays["raw_probability"]
    months = (timestamps + 19800).astype("datetime64[s]").astype("datetime64[M]")
    hours = ((timestamps + 19800) % 86400) // 3600
    common = np.logical_and.reduce([labels >= 0 for labels in horizons.values()])
    results = {}
    for name, labels in horizons.items():
        results[name] = {
            **probability_reliability(labels, probability, timestamps),
            "common_complete_cohort": probability_reliability(
                labels[common], probability[common], timestamps[common]
            ),
            "by_side": {
                str(side): probability_reliability(
                    labels[arrays["side"] == side],
                    probability[arrays["side"] == side],
                    timestamps[arrays["side"] == side],
                )
                for side in (-1, 1)
            },
            "by_month": {
                str(month): probability_reliability(
                    labels[months == month],
                    probability[months == month],
                    timestamps[months == month],
                )
                for month in np.unique(months)
            },
            "by_decision_hour_ist": {
                str(hour): probability_reliability(
                    labels[hours == hour],
                    probability[hours == hour],
                    timestamps[hours == hour],
                )
                for hour in np.unique(hours)
            },
            "fixed_thresholds_not_trades": {
                str(threshold): probability_reliability(
                    labels[probability >= threshold],
                    probability[probability >= threshold],
                    timestamps[probability >= threshold],
                )
                for threshold in (0.6, 0.7, 0.8, 0.9)
            },
        }
    np.savez_compressed(
        output / "outcomes.npz",
        **arrays,
        **horizons,
        session_close_timestamps=closing_times,
    )
    for name, expected in protocol["source_sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError("Analysis source changed during the run")
    report = {
        "status": "raw_probability_horizon_audit_complete",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "outcomes_sha256": file_hash(output / "outcomes.npz"),
        "quality": quality,
        "all_raw_predictions_reproduced": True,
        "all_60m_labels_recomputed": len(timestamps),
        "common_complete_rows": int(common.sum()),
        "horizons": results,
        "new_model_fits": 0,
        "live_orders": 0,
        "limits": protocol["limits"],
    }
    write_json(output / "report.json", report)
    return report


def analyze_frequency_and_returns(
    data: Path, cache: Path, horizon_audit: Path, output: Path
) -> dict:
    if output.exists():
        raise ValueError("Analysis output already exists; preserve earlier results")
    manifest = read_manifest(data)
    dataset_path = cache / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    previous = json.loads((horizon_audit / "report.json").read_text(encoding="utf-8"))
    previous_protocol = json.loads(
        (horizon_audit / "protocol.json").read_text(encoding="utf-8")
    )
    original_source = [
        expected
        for name, expected in previous_protocol["source_sha256"].items()
        if Path(name).name == "analyze_tp_no_stop.py"
    ]
    if (
        previous["status"] != "raw_probability_horizon_audit_complete"
        or previous["all_raw_predictions_reproduced"] is not True
        or previous["protocol_sha256"] != file_hash(horizon_audit / "protocol.json")
        or previous["outcomes_sha256"] != file_hash(horizon_audit / "outcomes.npz")
        or previous_protocol["dataset_sha256"] != file_hash(dataset_path)
        or previous_protocol["research_manifest_sha256"]
        != file_hash(data / "manifest.json")
        or original_source != [file_hash(horizon_audit / "analysis_source.py")]
        or dataset["provider"] != "kite"
        or manifest["provider"] != "kite"
    ):
        raise ValueError("Frozen horizon audit or Kite dataset provenance differs")
    with np.load(horizon_audit / "outcomes.npz", allow_pickle=False) as saved:
        arrays = {
            name: saved[name].copy()
            for name in (
                "timestamps",
                "symbols",
                "side",
                "raw_probability",
                "labels",
                "hit_60m",
                "hit_by_1500",
                "hit_by_session_close",
            )
        }
    evaluation = load_phase(data, manifest, "evaluation")
    for name in ("timestamps", "symbols", "labels"):
        if not np.array_equal(arrays[name], evaluation[name]):
            raise ValueError("Frozen outcomes differ from the evaluation cohort")
    if not np.array_equal(arrays["side"], evaluation["features"][:, -1]):
        raise ValueError("Frozen outcome directions differ from the evaluation cohort")
    del evaluation
    protocol = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "horizon_audit_report_sha256": file_hash(horizon_audit / "report.json"),
        "horizon_audit_protocol_sha256": file_hash(horizon_audit / "protocol.json"),
        "horizon_audit_outcomes_sha256": file_hash(horizon_audit / "outcomes.npz"),
        "dataset_sha256": file_hash(dataset_path),
        "research_manifest_sha256": file_hash(data / "manifest.json"),
        "selection_lock_sha256": previous_protocol["selection_lock_sha256"],
        "phase_ends": manifest["phase_ends"],
        "decision_stride_minutes": manifest["decision_stride_minutes"],
        "source_sha256": {
            name: file_hash(Path(name))
            for name in (
                Path(__file__).as_posix(),
                "scripts/train_tp_only.py",
                "src/ai_trader/history/store.py",
                "src/ai_trader/clock.py",
            )
        },
        "cohort": (
            "Same frozen raw evaluation actions, no new model or threshold selection"
        ),
        "failure_condition": (
            "No TP touch from the decision through 15:00, "
            "not merely no touch in 60 minutes"
        ),
        "return_formula": "side * (close of 14:59-15:00 candle / signal close - 1)",
        "units": (
            "Price-return fraction of entry notional, not leverage or capital return"
        ),
        "frequency_denominator": (
            "Two directions per scored symbol/candle; one decision every five minutes"
        ),
        "candle_bins": (
            "One bin per scored candle using max(LONG P, SHORT P), "
            "descriptive not a trading policy"
        ),
        "idealized_gross_outcome": (
            "Exactly 0.002 for TP by 15:00, otherwise cutoff price return; "
            "no fills or costs"
        ),
        "unknown_outcomes": "Incomplete windows excluded, never counted as failures",
        "limits": [
            "Repeated and opposite-direction opportunities are not executed trades",
            "Per-session frequency includes zero-opportunity days "
            "in the evaluation cohort",
            "No capital, concurrent-position, cooldown or transaction-cost simulation",
            "Retrospective evaluation, not new unseen or forward shadow evidence",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), output / "analysis_source.py")
    store = CandleStore(cache)
    returns = np.full(len(arrays["timestamps"]), np.nan, dtype=np.float64)
    verified_candles = {}
    for symbol in np.unique(arrays["symbols"]):
        chosen = np.flatnonzero(arrays["symbols"] == symbol)
        epochs = arrays["timestamps"][chosen]
        beginning = datetime.fromtimestamp(int(epochs.min()), INDIA_TIMEZONE).replace(
            hour=9, minute=15
        )
        ending = datetime.fromtimestamp(int(epochs.max()), INDIA_TIMEZONE).replace(
            hour=15, minute=30
        )
        instrument = Instrument("NSE", str(symbol))
        source = store.verified_path(store.path_for(instrument).name)
        expected = dataset["files"][source.name]
        grouped = defaultdict(list)
        for candle in store.load(instrument, beginning, ending, allow_incomplete=True):
            grouped[candle.start_time.astimezone(INDIA_TIMEZONE).date()].append(candle)
        days = (epochs + 19800) // 86400
        for day_id in np.unique(days):
            indices = chosen[days == day_id]
            decisions = arrays["timestamps"][indices]
            day = datetime.fromtimestamp(int(decisions[0]), INDIA_TIMEZONE).date()
            cutoff = int(datetime.combine(day, time(15), INDIA_TIMEZONE).timestamp())
            deadlines = np.full(len(indices), cutoff)
            hits = deadline_touch_labels(
                grouped[day], decisions, arrays["side"][indices], deadlines
            )
            if not np.array_equal(hits, arrays["hit_by_1500"][indices]):
                raise ValueError("Recomputed 15:00 hits differ from the frozen audit")
            returns[indices] = deadline_returns(
                grouped[day], decisions, arrays["side"][indices], deadlines
            )
        if file_hash(source) != expected:
            raise ValueError("Candle data changed during return calculation")
        verified_candles[source.name] = expected
        print(
            json.dumps({"symbol": str(symbol), "return_rows": len(chosen)}), flush=True
        )
    summary = frequency_and_return_summary(arrays, returns)
    np.savez_compressed(output / "returns.npz", **arrays, return_at_1500=returns)
    for name, expected in protocol["source_sha256"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError("Analysis source changed during the run")
    report = {
        "status": "raw_frequency_and_cutoff_returns_complete",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "returns_sha256": file_hash(output / "returns.npz"),
        "verified_candles_sha256": verified_candles,
        "all_1500_labels_recomputed": len(returns),
        "summary": summary,
        "new_model_fits": 0,
        "live_orders": 0,
        "limits": protocol["limits"],
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--comparison", type=Path)
    source.add_argument("--study", type=Path)
    source.add_argument("--horizon-audit", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon_audit:
        report = analyze_frequency_and_returns(
            args.data, args.cache, args.horizon_audit, args.output
        )
        print(
            json.dumps({"status": report["status"], "summary": report["summary"]}),
            flush=True,
        )
        return 0
    if args.study:
        if args.validation is None:
            parser.error("--study requires --validation for the Kite quality audit")
        report = analyze_forecasts(
            args.data, args.study, args.cache, args.validation, args.output
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "quality": report["quality"]["all_instruments"],
                    "horizons": {
                        name: result["all"]
                        for name, result in report["horizons"].items()
                    },
                }
            ),
            flush=True,
        )
        return 0
    report = analyze_calls(args.data, args.comparison, args.cache, args.output)
    print(
        json.dumps(
            {policy: values["all"] for policy, values in report["policies"].items()}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
