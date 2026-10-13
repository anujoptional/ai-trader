"""Capture bounded read-only Kite market evidence and audit the research path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, localcontext
from pathlib import Path

from ai_trader.broker import CandleInterval, Instrument, MarketTick, OHLCVCandle
from ai_trader.broker.kite import KiteBroker, KiteStreamConnectionError
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE
from ai_trader.config import load_kite_login_settings
from ai_trader.features import DERIVED_FEATURE_NAMES, FeatureEngine, FeatureSnapshot
from ai_trader.features.tp_only import (
    FEATURE_COLUMNS,
    CompletedSession,
    TpFeatureState,
)
from ai_trader.market import (
    Candle,
    MarketState,
    StreamSupervisor,
    StreamVolume,
    to_candle,
)

TARGETS = tuple(
    Instrument("NSE", symbol)
    for symbol in ("RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "SBIN")
)
INSTRUMENTS = TARGETS + tuple(
    Instrument("NSE", symbol) for symbol in ("NIFTY", "GOLDBEES", "INDIAVIX")
)


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_candles(path: Path, candles: Sequence[Candle]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("start_time", "open", "high", "low", "close", "volume"))
        for candle in candles:
            writer.writerow(
                (
                    candle.start_time.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    "" if candle.volume is None else candle.volume,
                )
            )


def _source_bar(candle: Candle) -> OHLCVCandle:
    return OHLCVCandle(
        candle.start_time,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
    )


def previous_sessions(
    instrument: Instrument,
    candles: Sequence[Candle],
    daily_closes: Mapping[date, Decimal],
    calendar: Sequence[date],
    today: date,
) -> tuple[CompletedSession, ...]:
    grouped = defaultdict(list)
    for candle in candles:
        local = candle.start_time.astimezone(INDIA_TIMEZONE)
        if local.date() < today and time(9, 15) <= local.time() < time(15, 15):
            grouped[local.date()].append(candle)
    result = []
    for day in sorted(day for day in calendar if day < today)[-6:]:
        state = TpFeatureState(instrument, day)
        for candle in grouped[day]:
            state.update(candle)
        result.append(state.finish(daily_closes.get(day)))
    return tuple(result)


def feature_record(snapshot: FeatureSnapshot) -> dict:
    return {
        "symbol": snapshot.instrument.trading_symbol,
        "end_time": snapshot.candle_end_time.isoformat(),
        "core_ready": snapshot.readiness.core_ready,
        "values": {
            name: None
            if getattr(snapshot, name) is None
            else str(getattr(snapshot, name))
            for name in DERIVED_FEATURE_NAMES
        },
        "readiness": {
            field.name: getattr(snapshot.readiness, field.name)
            for field in fields(snapshot.readiness)
        },
    }


class ResearchPath:
    def __init__(
        self,
        instruments: Sequence[Instrument],
        targets: Sequence[Instrument],
        prior: Mapping[Instrument, Sequence[CompletedSession]],
        today: date,
        *,
        predict: Callable[[list[list[float]]], dict] | None = None,
    ) -> None:
        self.instruments = tuple(instruments)
        self.targets = tuple(targets)
        self.generic = FeatureEngine()
        self.states = {
            item: TpFeatureState(item, today, prior.get(item, ()))
            for item in instruments
        }
        self.pending: dict[datetime, dict[Instrument, Candle]] = defaultdict(dict)
        self.predictions: list[dict] = []
        self.features: list[dict] = []
        self.predict = predict
        self.live_after: datetime | None = None
        self.completed_batches = 0
        self.data_quality_issues: set[str] = set()

    def invalidate(self, reason: str) -> None:
        self.data_quality_issues.add(reason)

    def add(self, candle: Candle) -> None:
        batch = self.pending[candle.end_time]
        if candle.instrument in batch:
            raise ValueError("Duplicate candle in validation input")
        batch[candle.instrument] = candle
        if len(batch) != len(self.instruments):
            return
        if any(moment < candle.end_time for moment in self.pending):
            raise ValueError("Missing aligned candle before the next feature batch")
        for instrument in self.instruments:
            snapshot = self.generic.update(batch[instrument])
            self.states[instrument].update(batch[instrument])
            if (
                snapshot is not None
                and self.live_after is not None
                and candle.end_time > self.live_after
            ):
                self.features.append(feature_record(snapshot))
        context = {
            item.trading_symbol: state.price_context()
            for item, state in self.states.items()
        }
        if self.live_after is not None and candle.end_time > self.live_after:
            self.completed_batches += 1
            if candle.end_time.minute % 5 == 0:
                for instrument in self.targets:
                    values = self.states[instrument].values(context, benchmark="NIFTY")
                    missing = [name for name in FEATURE_COLUMNS if values[name] is None]
                    record = {
                        "symbol": instrument.trading_symbol,
                        "end_time": candle.end_time.isoformat(),
                        "missing_inputs": missing,
                        "data_quality_issues": sorted(self.data_quality_issues),
                        "within_training_decision_window": (
                            time(9, 40)
                            <= candle.end_time.astimezone(INDIA_TIMEZONE).time()
                            <= time(14, 15)
                        ),
                        "inputs": {
                            name: None if value is None else str(value)
                            for name, value in values.items()
                        },
                    }
                    if not record["within_training_decision_window"]:
                        record["declined_reason"] = "outside_training_decision_window"
                    elif self.data_quality_issues:
                        record["declined_reason"] = "unreconciled_stream_discontinuity"
                    elif not missing and self.predict is not None:
                        base = [float(values[name]) for name in FEATURE_COLUMNS]
                        record["probabilities"] = self.predict(
                            [base + [1.0], base + [-1.0]]
                        )
                    self.predictions.append(record)
        del self.pending[candle.end_time]


def reconnect_probe_ready(
    collections: int, injected: bool, predictions: Sequence[dict]
) -> bool:
    return (
        collections >= 3
        and not injected
        and any("probabilities" in row for row in predictions)
    )


def compare_candles(live: Sequence[Candle], historical: Sequence[Candle]) -> dict:
    keys = [(bar.instrument, bar.start_time) for bar in historical]
    if len(set(keys)) != len(keys):
        raise ValueError("Historical comparison contains duplicate candles")
    reference = dict(zip(keys, historical, strict=True))
    results = []
    for candle in live:
        other = reference.get((candle.instrument, candle.start_time))
        row = {
            "symbol": candle.instrument.trading_symbol,
            "start_time": candle.start_time.isoformat(),
            "historical_present": other is not None,
        }
        if other is not None:
            row["end_time_equal"] = candle.end_time == other.end_time
            row["differences"] = {
                name: {
                    "live": str(getattr(candle, name)),
                    "historical": str(getattr(other, name)),
                }
                for name in ("open", "high", "low", "close", "volume")
                if getattr(candle, name) != getattr(other, name)
            }
            row["exact_match"] = row["end_time_equal"] and not row["differences"]
        results.append(row)
    return {
        "live_candles": len(live),
        "matched_timestamps": sum(row["historical_present"] for row in results),
        "exact_ohlcv_matches": sum(row.get("exact_match", False) for row in results),
        "rows": results,
    }


def _model_predictor(final: Path) -> tuple[Callable, dict]:
    import numpy as np
    from scipy.special import expit

    from scripts.research_tp_catboost import apply_mapping, load_frozen_model
    from scripts.train_tp_only import predict_logits

    model, mapping, lock = load_frozen_model(final)

    def predict(rows: list[list[float]]) -> dict:
        inputs = np.asarray(rows, dtype=np.float32)
        if inputs.shape[1] != len(lock["columns"]) or not np.isfinite(inputs).all():
            raise ValueError("Incomplete or nonfinite model inputs")
        logits = predict_logits(model, inputs, 1024)
        return {
            "sides": ["LONG", "SHORT"],
            "raw": expit(logits).tolist(),
            "calibrated": apply_mapping(mapping, logits, inputs[:, -1]).tolist(),
        }

    return predict, {
        "weights_sha256": lock["weights_sha256"],
        "selection_lock_sha256": _hash(final / "selection_lock.json"),
        "model_columns": lock["columns"],
    }


def _prime_path(
    histories: Mapping[Instrument, Sequence[Candle]],
    prior: Mapping[Instrument, Sequence[CompletedSession]],
    today: date,
    cutoff: datetime,
    predict: Callable,
) -> ResearchPath:
    path = ResearchPath(INSTRUMENTS, TARGETS, prior, today, predict=predict)
    for instrument in INSTRUMENTS:
        past = [bar for bar in histories[instrument] if bar.start_time.date() < today]
        if past:
            last_day = past[-1].start_time.date()
            path.generic.warm_up(
                [bar for bar in past if bar.start_time.date() == last_day]
            )
    warm = sorted(
        (
            bar
            for bars in histories.values()
            for bar in bars
            if bar.start_time.date() == today and bar.end_time <= cutoff
        ),
        key=lambda bar: (bar.end_time, bar.instrument.trading_symbol),
    )
    for bar in warm:
        path.add(bar)
    if path.pending:
        raise ValueError("Warm-up history has missing aligned current-session candles")
    path.live_after = cutoff
    return path


def _tick_statistics(ticks: Sequence[MarketTick]) -> dict:
    rows = {}
    for instrument in INSTRUMENTS:
        selected = [tick for tick in ticks if tick.instrument == instrument]
        volumes = [
            tick.cumulative_volume
            for tick in selected
            if tick.cumulative_volume is not None
        ]
        rows[instrument.trading_symbol] = {
            "ticks": len(selected),
            "first_exchange_time": selected[0].timestamp.isoformat()
            if selected
            else None,
            "last_exchange_time": selected[-1].timestamp.isoformat()
            if selected
            else None,
            "distinct_prices": len({tick.price for tick in selected}),
            "native_volume_ticks": len(volumes),
            "volume_regressions": sum(
                right < left for left, right in zip(volumes, volumes[1:])
            ),
            "timestamp_regressions": sum(
                right.timestamp < left.timestamp
                for left, right in zip(selected, selected[1:])
            ),
        }
    return rows


def run_validation(output: Path, duration_seconds: int, final: Path) -> dict:
    if output.exists():
        raise ValueError("Validation output exists; use a new evidence directory")
    now = datetime.now(INDIA_TIMEZONE)
    if now.weekday() >= 5 or not time(9, 20) <= now.time() <= time(14, 45):
        raise ValueError("Run this bounded check during the regular continuous session")
    output.mkdir(parents=True)
    report: dict = {
        "status": "in_progress",
        "phase": "model_preflight",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "orders_enabled": False,
        "live_orders": 0,
        "new_model_fits": 0,
    }

    def progress(phase: str) -> None:
        report["phase"] = phase
        _write_json(output / "progress.json", report)
        print(json.dumps({"phase": phase}), flush=True)

    stream = None
    try:
        predict, model_metadata = _model_predictor(final)
        protected = {
            str(path): _hash(path)
            for path in (
                Path("data/candles/dataset.json"),
                Path("data/research/tp_only_kite_20261005_final/manifest.json"),
                final / "best.cbm",
                final / "calibration.json",
                final / "selection_lock.json",
            )
        }
        report["model"] = model_metadata
        report["protected_sha256"] = protected
        source_paths = [
            Path(__file__),
            Path("src/ai_trader/broker/kite.py"),
            Path("src/ai_trader/broker/kite_stream.py"),
            Path("src/ai_trader/market/candles.py"),
            Path("src/ai_trader/market/state.py"),
            Path("src/ai_trader/market/stream.py"),
            Path("src/ai_trader/features/engine.py"),
            Path("src/ai_trader/features/tp_only.py"),
        ]
        report["source_sha256"] = {
            path.as_posix(): _hash(path) for path in source_paths
        }
        shutil.copyfile(Path(__file__), output / "validation_source.py")
        progress("private_browser_login")
        broker = KiteBroker.authenticate(load_kite_login_settings())
        progress("read_only_profile_quotes_and_history")
        profile = broker.get_user_profile()
        report["profile"] = {
            "exchange_enablement": dict(profile.exchange_enablement),
            "active_segments": list(profile.active_segments),
            "ddpi_enabled": profile.ddpi_enabled,
        }
        report["instruments"] = [
            {
                "symbol": item.trading_symbol,
                "token": broker.resolve_instrument(item).instrument_token,
                "tick_size": str(broker.resolve_instrument(item).tick_size),
            }
            for item in INSTRUMENTS
        ]
        report["initial_ltp"] = [
            {"symbol": value.instrument.trading_symbol, "price": str(value.price)}
            for value in broker.get_ltp(INSTRUMENTS)
        ]
        report["quotes"] = []
        for instrument in TARGETS:
            quote = broker.get_quote(instrument)
            report["quotes"].append(
                {
                    "symbol": instrument.trading_symbol,
                    "price": str(quote.last_price),
                    "last_trade_at": quote.last_trade_at.isoformat(),
                    "volume": quote.volume,
                }
            )
        today = datetime.now(INDIA_TIMEZONE).date()
        beginning = datetime.combine(
            today - timedelta(days=21), time(9, 15), INDIA_TIMEZONE
        )
        initial_cutoff = (
            datetime.now(INDIA_TIMEZONE).replace(second=0, microsecond=0) - ONE_MINUTE
        )
        histories, daily, calendar = {}, {}, set()
        for instrument in INSTRUMENTS:
            histories[instrument] = [
                to_candle(instrument, bar)
                for bar in broker.get_historical_candles(
                    instrument, beginning, initial_cutoff, CandleInterval.ONE_MINUTE
                )
                if bar.timestamp + ONE_MINUTE <= initial_cutoff
            ]
            prior_daily = broker.get_historical_candles(
                instrument,
                beginning.replace(hour=0, minute=0),
                datetime.combine(
                    today - timedelta(days=1), time(23, 59), INDIA_TIMEZONE
                ),
                CandleInterval.ONE_DAY,
            )
            daily[instrument] = {bar.timestamp.date(): bar.close for bar in prior_daily}
            calendar.update(daily[instrument])
            _write_json(
                output / f"{instrument.trading_symbol}_daily.json",
                {
                    day.isoformat(): str(close)
                    for day, close in daily[instrument].items()
                },
            )
        prior = {
            instrument: previous_sessions(
                instrument,
                histories[instrument],
                daily[instrument],
                sorted(calendar),
                today,
            )
            for instrument in INSTRUMENTS
        }
        report["prior_session_counts"] = {
            item.trading_symbol: len(prior[item]) for item in INSTRUMENTS
        }
        progress("buffered_stream_startup")
        stream = broker.create_ltp_stream(INSTRUMENTS)
        opened_at = datetime.now(INDIA_TIMEZONE)
        cutoff = opened_at.replace(second=0, microsecond=0) + ONE_MINUTE
        ticks: list[MarketTick] = []
        live_candles: list[Candle] = []
        with (output / "ticks.jsonl").open("x", encoding="utf-8") as tape:

            def capture(tick: MarketTick) -> None:
                ticks.append(tick)
                tape.write(
                    json.dumps(
                        {
                            "symbol": tick.instrument.trading_symbol,
                            "exchange_time": tick.timestamp.isoformat(),
                            "collected_at": datetime.now(INDIA_TIMEZONE).isoformat(),
                            "price": str(tick.price),
                            "cumulative_volume": tick.cumulative_volume,
                        }
                    )
                    + "\n"
                )

            stream.collect(max_ticks=200000, timeout_seconds=75, on_tick=capture)
            tape.flush()
            if set(tick.instrument for tick in ticks) != set(INSTRUMENTS):
                raise ValueError("Startup did not receive all required instruments")
            progress("overlap_backfill_and_feature_warmup")
            for instrument in INSTRUMENTS:
                extra = broker.get_historical_candles(
                    instrument,
                    initial_cutoff - ONE_MINUTE,
                    cutoff,
                    CandleInterval.ONE_MINUTE,
                )
                histories[instrument].extend(
                    to_candle(instrument, bar)
                    for bar in extra
                    if initial_cutoff <= bar.timestamp
                    and bar.timestamp + ONE_MINUTE <= cutoff
                )
                _write_candles(
                    output / f"{instrument.trading_symbol}_warmup.csv",
                    histories[instrument],
                )
            path = _prime_path(histories, prior, today, cutoff, predict)

            def on_candle(candle: Candle) -> None:
                if candle.end_time > cutoff:
                    live_candles.append(candle)
                    path.add(candle)

            state = MarketState(
                max_candles=10000,
                on_candle=on_candle,
                tick_stamper=StreamVolume(INSTRUMENTS).stamp,
            )
            for instrument in INSTRUMENTS:
                state.backfill(
                    instrument, (_source_bar(bar) for bar in histories[instrument])
                )
            for tick in ticks:
                state.record_tick(tick)

            def on_tick(tick: MarketTick) -> None:
                capture(tick)
                state.record_tick(tick)

            injected = False
            openings = []
            first_stream = stream

            class ObservedStream:
                def __init__(self, source):
                    self.source = source
                    self.collections = 0

                def collect(self, **kwargs):
                    nonlocal injected
                    if reconnect_probe_ready(
                        self.collections, injected, path.predictions
                    ):
                        injected = True
                        report["controlled_disconnect_at"] = datetime.now(
                            INDIA_TIMEZONE
                        ).isoformat()
                        self.source.close()
                        raise KiteStreamConnectionError(
                            "Controlled read-only reconnect probe"
                        )
                    result = self.source.collect(**kwargs)
                    self.collections += 1
                    tape.flush()
                    print(
                        json.dumps(
                            {
                                "phase": "live_capture",
                                "ticks": len(ticks),
                                "live_candles": len(live_candles),
                                "model_decisions": len(path.predictions),
                            }
                        ),
                        flush=True,
                    )
                    return result

                def close(self):
                    self.source.close()

            def open_stream():
                nonlocal first_stream
                current = first_stream
                first_stream = None
                if current is None:
                    reason = "stream_reconnected_without_history_reconciliation"
                    path.invalidate(reason)
                    tape.write(
                        json.dumps(
                            {
                                "event": "stream_discontinuity",
                                "reason": reason,
                                "collected_at": datetime.now(
                                    INDIA_TIMEZONE
                                ).isoformat(),
                            }
                        )
                        + "\n"
                    )
                    tape.flush()
                    current = broker.create_ltp_stream(INSTRUMENTS)
                openings.append(datetime.now(INDIA_TIMEZONE).isoformat())
                return ObservedStream(current)

            progress("supervised_live_capture")
            supervisor = StreamSupervisor(
                open_stream,
                on_tick,
                retry_on=(KiteStreamConnectionError,),
                session_seconds=60,
                max_consecutive_failures=3,
            )
            report["stream"] = asdict(
                supervisor.run(duration_seconds, max_ticks=200000)
            )
            report["stream_opened_at"] = openings
            report["controlled_disconnect_injected"] = injected
        stream.close()
        report["capture_ended_at"] = datetime.now(INDIA_TIMEZONE).isoformat()
        report["handoff_cutoff"] = cutoff.isoformat()
        report["tick_statistics"] = _tick_statistics(ticks)
        report["late_tick_count"] = state.late_tick_count
        report["duplicate_candle_count"] = state.duplicate_candle_count
        report["unknown_volume_candles"] = state.unknown_volume_candle_count
        report["pending_unaligned_minutes"] = [
            stamp.isoformat() for stamp in path.pending
        ]
        _write_json(output / "live_features.json", {"rows": path.features})
        _write_json(output / "live_predictions.json", {"rows": path.predictions})
        for instrument in INSTRUMENTS:
            _write_candles(
                output / f"{instrument.trading_symbol}_live.csv",
                [candle for candle in live_candles if candle.instrument == instrument],
            )
        progress("saved_tick_replay")
        replay = _prime_path(histories, prior, today, cutoff, predict)
        replay_candles = []

        def replay_candle(candle: Candle) -> None:
            if candle.end_time > cutoff:
                replay_candles.append(candle)
                replay.add(candle)

        replay_state = MarketState(max_candles=10000, on_candle=replay_candle)
        for instrument in INSTRUMENTS:
            replay_state.backfill(
                instrument, (_source_bar(bar) for bar in histories[instrument])
            )
        with (output / "ticks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") == "stream_discontinuity":
                    replay.invalidate(row["reason"])
                    continue
                replay_state.record_tick(
                    MarketTick(
                        Instrument("NSE", row["symbol"]),
                        Decimal(row["price"]),
                        datetime.fromisoformat(row["exchange_time"]),
                        row["cumulative_volume"],
                    )
                )
        report["replay"] = {
            "candles_exact": replay_candles == live_candles,
            "features_exact": replay.features == path.features,
            "model_inputs_and_predictions_exact": replay.predictions
            == path.predictions,
            "live_candles": len(live_candles),
            "feature_rows": len(path.features),
            "model_decisions": len(path.predictions),
            "predictions_with_complete_inputs": sum(
                "probabilities" in row for row in path.predictions
            ),
        }
        progress("subsequent_history_comparison")
        settled = (
            datetime.now(INDIA_TIMEZONE).replace(second=0, microsecond=0) - ONE_MINUTE
        )
        historical = {}
        for instrument in INSTRUMENTS:
            historical[instrument] = [
                to_candle(instrument, bar)
                for bar in broker.get_historical_candles(
                    instrument,
                    datetime.combine(today, time(9, 15), INDIA_TIMEZONE),
                    settled,
                    CandleInterval.ONE_MINUTE,
                )
                if bar.timestamp + ONE_MINUTE <= settled
            ]
            _write_candles(
                output / f"{instrument.trading_symbol}_subsequent_history.csv",
                historical[instrument],
            )
        comparison = compare_candles(
            [bar for bar in live_candles if bar.end_time <= settled],
            [bar for bars in historical.values() for bar in bars],
        )
        _write_json(output / "candle_comparison.json", comparison)
        historical_inputs = {
            item: [bar for bar in histories[item] if bar.start_time.date() < today]
            + historical[item]
            for item in INSTRUMENTS
        }
        history_path = _prime_path(historical_inputs, prior, today, cutoff, predict)
        for bar in sorted(
            (
                bar
                for bars in historical.values()
                for bar in bars
                if bar.end_time > cutoff
            ),
            key=lambda bar: (bar.end_time, bar.instrument.trading_symbol),
        ):
            history_path.add(bar)
        _write_json(
            output / "historical_features.json", {"rows": history_path.features}
        )
        _write_json(
            output / "historical_predictions.json", {"rows": history_path.predictions}
        )
        historical_features = {
            (row["symbol"], row["end_time"]): row for row in history_path.features
        }
        comparable = [
            row
            for row in path.features
            if (row["symbol"], row["end_time"]) in historical_features
        ]
        feature_differences = [
            {
                "symbol": row["symbol"],
                "end_time": row["end_time"],
                "different_fields": [
                    name
                    for name, value in row["values"].items()
                    if historical_features[row["symbol"], row["end_time"]]["values"][
                        name
                    ]
                    != value
                ],
            }
            for row in comparable
        ]
        _write_json(output / "feature_comparison.json", {"rows": feature_differences})
        report["historical_comparison"] = {
            key: value for key, value in comparison.items() if key != "rows"
        }
        report["historical_comparison"]["comparable_feature_rows"] = len(comparable)
        report["historical_comparison"]["exact_feature_rows"] = sum(
            not row["different_fields"] for row in feature_differences
        )
        for name, expected in {**protected, **report["source_sha256"]}.items():
            if _hash(Path(name)) != expected:
                raise ValueError(
                    "A protected artifact or executed source changed during capture"
                )
        report["protected_artifacts_unchanged"] = True
        report["gates"] = {
            "all_instruments_moving": all(
                row["distinct_prices"] > 1 for row in report["tick_statistics"].values()
            ),
            "equity_native_volume_present": all(
                report["tick_statistics"][item.trading_symbol]["native_volume_ticks"]
                == report["tick_statistics"][item.trading_symbol]["ticks"]
                for item in (*TARGETS, Instrument("NSE", "GOLDBEES"))
            ),
            "controlled_reconnect_observed": injected
            and report["stream"]["reconnects"] >= 1,
            "capture_reached_deadline": report["stream"]["stopped_because"]
            == "deadline",
            "saved_tick_replay_exact": all(
                report["replay"][name]
                for name in (
                    "candles_exact",
                    "features_exact",
                    "model_inputs_and_predictions_exact",
                )
            ),
            "complete_frozen_model_inference": report["replay"][
                "predictions_with_complete_inputs"
            ]
            > 0,
            "stream_continuity_reconciled": not path.data_quality_issues,
            "exact_vendor_candle_parity": comparison["live_candles"] > 0
            and comparison["exact_ohlcv_matches"] == comparison["live_candles"],
        }
        report["limitations"] = [
            "Bounded research-data-path validation, "
            "not an unattended trading certification",
            "Frozen inference is a diagnostic path, not runtime scanner promotion",
            "Controlled client reconnect does not certify arbitrary outages "
            "or loss recovery",
            "Sampled WebSocket ticks can differ from vendor historical "
            "candle construction",
            "No TP outcome, paper portfolio, LLM, production risk engine "
            "or order execution tested",
        ]
        report["status"] = (
            "bounded_validation_complete"
            if all(report["gates"].values())
            else "bounded_validation_with_gaps"
        )
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        report["evidence_sha256"] = {
            path.name: _hash(path)
            for path in output.iterdir()
            if path.is_file() and path.name not in ("progress.json", "report.json")
        }
        _write_json(output / "report.json", report)
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["failed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(output / "report.json", report)
        raise
    finally:
        if stream is not None:
            stream.close()


def _read_candles(path: Path, instrument: Instrument) -> list[Candle]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            Candle(
                instrument,
                datetime.fromisoformat(row["start_time"]),
                datetime.fromisoformat(row["start_time"]) + ONE_MINUTE,
                *(Decimal(row[name]) for name in ("open", "high", "low", "close")),
                None if row["volume"] == "" else int(row["volume"]),
            )
            for row in csv.DictReader(handle)
        ]


def reference_tp_inputs(
    instrument: Instrument,
    as_of: datetime,
    histories: Mapping[Instrument, Sequence[Candle]],
    daily: Mapping[Instrument, Mapping[date, Decimal]],
    calendar: Sequence[date],
) -> dict[str, Decimal | None]:
    with localcontext() as context:
        context.prec = 60
        local = as_of.astimezone(INDIA_TIMEZONE)
        opening = local.replace(hour=9, minute=15, second=0, microsecond=0)
        minutes = int((local - opening) // ONE_MINUTE)
        lookup = {}
        for item, bars in histories.items():
            indexed = {bar.start_time: bar for bar in bars}
            if len(indexed) != len(bars):
                raise ValueError("Duplicate TP reference candle")
            lookup[item.trading_symbol] = indexed

        def window(symbol: str, start: datetime, count: int) -> list[Candle]:
            bars = [
                lookup[symbol].get(start + offset * ONE_MINUTE)
                for offset in range(count)
            ]
            if any(
                bar is None or bar.end_time != bar.start_time + ONE_MINUTE
                for bar in bars
            ):
                return []
            return bars

        def log_change(symbol: str, length: int, lag: int = 0) -> Decimal | None:
            bars = window(
                symbol, as_of - (length + lag + 1) * ONE_MINUTE, length + lag + 1
            )
            return (bars[-lag - 1].close / bars[0].close).ln() if bars else None

        symbol = instrument.trading_symbol
        current = lookup[symbol].get(as_of - ONE_MINUTE)
        if current is None or not 1 <= minutes <= 360:
            raise ValueError("TP reference requires a completed cutoff-session candle")
        first = lookup[symbol].get(opening)
        prior_days = sorted(set(day for day in calendar if day < local.date()))[-6:]
        closes = [daily[instrument].get(day) for day in prior_days]
        prior_close = closes[-1] if closes else None
        variances, volumes = [], []
        for day in prior_days[-5:]:
            start = datetime.combine(day, time(9, 15), INDIA_TIMEZONE)
            bars = window(symbol, start, 360)
            variance = None
            if bars:
                previous = [bars[0].open] + [bar.close for bar in bars[:-1]]
                variance = sum(
                    (bar.close / price).ln() ** 2
                    for bar, price in zip(bars, previous, strict=True)
                )
            variances.append(variance)
            matched = lookup[symbol].get(start + (minutes - 1) * ONE_MINUTE)
            volumes.append(matched.volume if matched is not None else None)
        mean_volume = (
            Decimal(sum(volumes)) / 5
            if len(volumes) == 5 and all(value is not None for value in volumes)
            else None
        )
        recent = window(symbol, as_of - 21 * ONE_MINUTE, 21)
        session = window(symbol, opening, minutes)
        vwap_z = None
        if session and all(bar.volume is not None for bar in session):
            total = sum(bar.volume for bar in session)
            if total:
                prices = [(bar.high + bar.low + bar.close) / 3 for bar in session]
                mean = (
                    sum(
                        price * bar.volume
                        for price, bar in zip(prices, session, strict=True)
                    )
                    / total
                )
                variance = (
                    sum(
                        bar.volume * (price - mean) ** 2
                        for price, bar in zip(prices, session, strict=True)
                    )
                    / total
                )
                if len(set(prices)) > 1 and variance > 0:
                    vwap_z = (current.close - mean) / variance.sqrt()
        vix = lookup["INDIAVIX"].get(as_of - ONE_MINUTE)
        return {
            "log_return_5m": log_change(symbol, 5),
            "log_return_previous_10m": log_change(symbol, 10, 5),
            "day_return": current.close / first.open - 1 if first else None,
            "gap_from_vendor_daily_close": first.open / prior_close - 1
            if first and prior_close
            else None,
            "realized_volatility_20m": sum(
                (right.close / left.close).ln() ** 2
                for left, right in zip(recent, recent[1:])
            ).sqrt()
            if recent
            else None,
            "prior_5_daily_close_return": closes[-1] / closes[0] - 1
            if len(closes) == 6 and all(value is not None for value in closes)
            else None,
            "prior_5_cutoff_volatility": (sum(variances) / 5).sqrt()
            if len(variances) == 5 and all(value is not None for value in variances)
            else None,
            "vwap_z": vwap_z,
            "same_minute_rvol_5": Decimal(current.volume) / mean_volume
            if current.volume is not None and mean_volume
            else None,
            "minutes_from_open": Decimal(minutes),
            "benchmark_log_return_15m": log_change("NIFTY", 15),
            "gold_log_return_15m": log_change("GOLDBEES", 15),
            "vix_level": vix.close if vix else None,
        }


def audit_tp_inputs(
    histories: Mapping[Instrument, Sequence[Candle]],
    daily: Mapping[Instrument, Mapping[date, Decimal]],
    calendar: Sequence[date],
    predictions: Sequence[dict],
) -> dict:
    numeric, unavailable = 0, 0
    maximum = Decimal(0)
    tolerance = Decimal("1e-18")
    for row in predictions:
        expected = reference_tp_inputs(
            Instrument("NSE", row["symbol"]),
            datetime.fromisoformat(row["end_time"]),
            histories,
            daily,
            calendar,
        )
        if set(row["inputs"]) != set(expected):
            raise ValueError("Independent TP input schema differs")
        for name, reference in expected.items():
            actual = row["inputs"][name]
            if reference is None or actual is None:
                if reference is not None or actual is not None:
                    raise ValueError(
                        f"Independent TP input availability differs: {name}"
                    )
                unavailable += 1
                continue
            value = Decimal(actual)
            error = abs(value - reference)
            if not value.is_finite() or error > tolerance * max(
                Decimal(1), abs(reference)
            ):
                raise ValueError(f"Independent TP input arithmetic differs: {name}")
            maximum = max(maximum, error)
            numeric += 1
    return {
        "status": "passed" if predictions else "no_decision_rows",
        "rows": len(predictions),
        "numeric_comparisons": numeric,
        "unavailable_agreements": unavailable,
        "maximum_absolute_error": str(maximum),
        "reference_precision": 60,
        "absolute_relative_tolerance": str(tolerance),
    }


def compare_predictions(live: Sequence[dict], historical: Sequence[dict]) -> list[dict]:
    references = {(row["symbol"], row["end_time"]): row for row in historical}
    results = []
    for row in live:
        reference = references.get((row["symbol"], row["end_time"]))
        if reference is None:
            continue
        live_present = "probabilities" in row
        history_present = "probabilities" in reference
        results.append(
            {
                "symbol": row["symbol"],
                "end_time": row["end_time"],
                "changed_inputs": [
                    name
                    for name, value in row["inputs"].items()
                    if value != reference["inputs"][name]
                ],
                "live_prediction_present": live_present,
                "historical_prediction_present": history_present,
                "probabilities_exact": row["probabilities"]
                == reference["probabilities"]
                if live_present and history_present
                else None,
            }
        )
    return results


def review_capture(capture: Path, output: Path, final: Path) -> dict:
    from scripts.validate_scanner_features import snapshots, validate_arithmetic

    if output.exists():
        raise ValueError("Review output exists; preserve earlier evidence")
    report = json.loads((capture / "report.json").read_text(encoding="utf-8"))
    for name, expected in report["evidence_sha256"].items():
        if _hash(capture / name) != expected:
            raise ValueError("Captured evidence fingerprint mismatch")
    for name, expected in report["protected_sha256"].items():
        if _hash(Path(name)) != expected:
            raise ValueError("A protected research artifact changed")
    source_hash = next(
        expected
        for name, expected in report["source_sha256"].items()
        if Path(name).name == Path(__file__).name
    )
    if _hash(capture / "validation_source.py") != source_hash:
        raise ValueError("Executed capture source fingerprint mismatch")
    with (capture / "ticks.jsonl").open(encoding="utf-8") as handle:
        tape = [json.loads(line) for line in handle]
    groups = defaultdict(list)
    tick_rows = [row for row in tape if "symbol" in row]
    for row in tick_rows:
        minute = datetime.fromisoformat(row["exchange_time"]).replace(
            second=0, microsecond=0
        )
        groups[row["symbol"], minute].append(row)
    independent_candles = 0
    histories, lives, daily, calendar = {}, {}, {}, set()
    cutoff = datetime.fromisoformat(report["handoff_cutoff"])
    today = cutoff.date()
    arithmetic = {}
    saved_features = json.loads(
        (capture / "live_features.json").read_text(encoding="utf-8")
    )["rows"]
    for instrument in INSTRUMENTS:
        symbol = instrument.trading_symbol
        histories[instrument] = _read_candles(
            capture / f"{symbol}_warmup.csv", instrument
        )
        lives[instrument] = _read_candles(capture / f"{symbol}_live.csv", instrument)
        daily[instrument] = {
            date.fromisoformat(day): Decimal(close)
            for day, close in json.loads(
                (capture / f"{symbol}_daily.json").read_text(encoding="utf-8")
            ).items()
        }
        calendar.update(daily[instrument])
        for candle in lives[instrument]:
            observed = groups[symbol, candle.start_time]
            prices = [Decimal(row["price"]) for row in observed]
            if (prices[0], max(prices), min(prices), prices[-1]) != (
                candle.open,
                candle.high,
                candle.low,
                candle.close,
            ):
                raise ValueError("Independent sampled-tick OHLC differs")
            previous = groups.get((symbol, candle.start_time - ONE_MINUTE), [])
            volume = None
            if (
                previous
                and previous[-1]["cumulative_volume"] is not None
                and observed[-1]["cumulative_volume"] is not None
            ):
                volume = (
                    observed[-1]["cumulative_volume"]
                    - previous[-1]["cumulative_volume"]
                )
            if volume != candle.volume:
                raise ValueError("Independent cumulative-volume arithmetic differs")
            independent_candles += 1
        prior_days = sorted(
            {
                bar.start_time.date()
                for bar in histories[instrument]
                if bar.start_time.date() < today
            }
        )
        bars = [
            bar
            for bar in histories[instrument]
            if bar.start_time.date() in (prior_days[-1], today)
        ] + lives[instrument]
        arithmetic[symbol] = validate_arithmetic(bars, all_features=True)
        recalculated = [
            feature_record(snapshot)
            for snapshot in snapshots(bars)
            if snapshot.candle_end_time > cutoff
        ]
        if recalculated != [row for row in saved_features if row["symbol"] == symbol]:
            raise ValueError("Independent audit input does not reproduce live features")
    comparisons = json.loads(
        (capture / "candle_comparison.json").read_text(encoding="utf-8")
    )
    by_symbol = {}
    for instrument in INSTRUMENTS:
        rows = [
            row
            for row in comparisons["rows"]
            if row["symbol"] == instrument.trading_symbol
        ]
        changes = defaultdict(int)
        maximum = defaultdict(lambda: Decimal(0))
        for row in rows:
            for name, values in row.get("differences", {}).items():
                changes[name] += 1
                if values["live"] != "None" and values["historical"] != "None":
                    maximum[name] = max(
                        maximum[name],
                        abs(Decimal(values["live"]) - Decimal(values["historical"])),
                    )
        by_symbol[instrument.trading_symbol] = {
            "compared": len(rows),
            "exact_ohlcv": sum(row.get("exact_match", False) for row in rows),
            "exact_ohlc": sum(
                row["historical_present"]
                and not (set(row["differences"]) & {"open", "high", "low", "close"})
                for row in rows
            ),
            "changed_field_counts": dict(changes),
            "maximum_absolute_difference": {
                name: str(value) for name, value in maximum.items()
            },
        }
    live_predictions = json.loads(
        (capture / "live_predictions.json").read_text(encoding="utf-8")
    )["rows"]
    historical_predictions = json.loads(
        (capture / "historical_predictions.json").read_text(encoding="utf-8")
    )["rows"]
    prediction_differences = compare_predictions(
        live_predictions, historical_predictions
    )
    tp_arithmetic = audit_tp_inputs(
        {item: histories[item] + lives[item] for item in INSTRUMENTS},
        daily,
        sorted(calendar),
        live_predictions,
    )
    predict, _ = _model_predictor(final)
    prior = {
        item: previous_sessions(
            item, histories[item], daily[item], sorted(calendar), today
        )
        for item in INSTRUMENTS
    }
    guarded = _prime_path(histories, prior, today, cutoff, predict)
    replay_state = MarketState(
        max_candles=10000,
        on_candle=lambda candle: (
            guarded.add(candle) if candle.end_time > cutoff else None
        ),
    )
    gap = report.get("controlled_disconnect_at")
    gap_time = datetime.fromisoformat(gap) if gap else None
    invalidated = False
    for row in tape:
        if row.get("event") == "stream_discontinuity":
            guarded.invalidate(row["reason"])
            invalidated = True
            continue
        if (
            gap_time
            and not invalidated
            and datetime.fromisoformat(row["collected_at"]) >= gap_time
        ):
            guarded.invalidate("stream_reconnected_without_history_reconciliation")
            invalidated = True
        replay_state.record_tick(
            MarketTick(
                Instrument("NSE", row["symbol"]),
                Decimal(row["price"]),
                datetime.fromisoformat(row["exchange_time"]),
                row["cumulative_volume"],
            )
        )
    if guarded.features != saved_features:
        raise ValueError("Guarded replay changed the captured feature calculations")
    in_window = sum(
        row["within_training_decision_window"] for row in guarded.predictions
    )
    quality_preserved = all(
        not row["data_quality_issues"] for row in guarded.predictions
    )
    emitted = sum("probabilities" in row for row in guarded.predictions)
    remaining_gates = [
        "An explicit candle-source/parity policy before runtime model integration",
        "History reconciliation after reconnect, not only socket recovery",
        "Prospective shadow evaluation; no paper portfolio or order engine validated",
    ]
    if not emitted:
        remaining_gates.insert(
            0, "A live capture with eligible in-window frozen-model predictions"
        )
    result = {
        "status": "review_complete_not_end_to_end_approved",
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "capture_report_sha256": _hash(capture / "report.json"),
        "capture_source_sha256": source_hash,
        "review_source_sha256": _hash(Path(__file__)),
        "reference_source_sha256": _hash(Path("scripts/validate_scanner_features.py")),
        "verified_evidence_files": len(report["evidence_sha256"]),
        "protected_artifacts_unchanged": True,
        "tick_rows": len(tick_rows),
        "independent_sampled_tick_candles": independent_candles,
        "feature_arithmetic": arithmetic,
        "tp_feature_arithmetic": tp_arithmetic,
        "by_symbol_history_parity": by_symbol,
        "historical_prediction_comparison": prediction_differences,
        "guarded_saved_tape_replay": {
            "features_unchanged": True,
            "decision_attempts": len(guarded.predictions),
            "within_training_window": in_window,
            "predictions_emitted": emitted,
            "no_unreconciled_discontinuity": quality_preserved,
            "declined_rows": guarded.predictions,
        },
        "remaining_gates": remaining_gates,
        "broker_calls_in_review": 0,
        "live_orders": 0,
        "new_model_fits": 0,
    }
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / "review_source.py")
    _write_json(output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=480)
    parser.add_argument("--review-capture", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/research/tp_catboost_kite_20261005/final"),
    )
    args = parser.parse_args()
    if not 240 <= args.duration_seconds <= 900:
        parser.error("Use a bounded capture between 240 and 900 seconds")
    try:
        if args.review_capture:
            report = review_capture(args.review_capture, args.output, args.model)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "independent_candles": report[
                            "independent_sampled_tick_candles"
                        ],
                        "guarded_in_window_predictions": report[
                            "guarded_saved_tape_replay"
                        ]["predictions_emitted"],
                    }
                ),
                flush=True,
            )
            return 0
        report = run_validation(args.output, args.duration_seconds, args.model)
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error_type": type(error).__name__}),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "gates": report["gates"],
                "replay": report["replay"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
