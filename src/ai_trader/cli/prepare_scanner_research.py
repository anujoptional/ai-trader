"""Freeze a declared research sample, without looking at score outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import ai_trader
from ai_trader.broker import Instrument
from ai_trader.broker.groww import GrowwBroker, GrowwBrokerError
from ai_trader.clock import (
    INDIA_TIMEZONE,
    SESSION_CLOSE_TIME,
    SESSION_MINUTES,
    SESSION_OPEN_TIME,
    minutes_since_open,
)
from ai_trader.config import ConfigurationError, load_groww_settings
from ai_trader.features import FeatureEngine
from ai_trader.history import (
    CandleStore,
    CandleStoreError,
    last_completed_session_close,
)
from ai_trader.market import Candle
from ai_trader.strategy import StrategyConfig


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    path: Path
    instrument: Instrument
    warmup_start: date
    start: date
    end: date
    development_end: date
    validation_end: date
    expected_sessions: tuple[date, ...]
    horizons_minutes: tuple[int, ...]
    primary_horizon_minutes: int
    strategy_file: Path
    snapshot_directory: Path

    @classmethod
    def load(cls, path: Path) -> ResearchPlan:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "exchange",
            "symbol",
            "warmup_start",
            "start",
            "end",
            "development_end",
            "validation_end",
            "expected_sessions",
            "horizons_minutes",
            "primary_horizon_minutes",
            "strategy_file",
            "snapshot_directory",
            "protocol",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("Research plan fields do not match schema version 1")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("Unsupported research plan schema")
        dates = {
            name: date.fromisoformat(payload[name])
            for name in (
                "warmup_start",
                "start",
                "end",
                "development_end",
                "validation_end",
            )
        }
        if not (
            dates["warmup_start"]
            < dates["start"]
            <= dates["development_end"]
            < dates["validation_end"]
            < dates["end"]
        ):
            raise ValueError("Research windows must be chronological and disjoint")
        sessions = tuple(
            date.fromisoformat(value) for value in payload["expected_sessions"]
        )
        if not sessions or sessions != tuple(sorted(set(sessions))):
            raise ValueError("Expected sessions must be unique and ordered")
        if sessions[0] != dates["start"] or sessions[-1] != dates["end"]:
            raise ValueError("Expected sessions must span the evaluation window")
        horizons = tuple(payload["horizons_minutes"])
        if not horizons or any(
            type(value) is not int or value <= 0 for value in horizons
        ):
            raise ValueError("Horizons must be positive integer minutes")
        if len(set(horizons)) != len(horizons):
            raise ValueError("Horizons must be unique")
        if payload["primary_horizon_minutes"] not in horizons:
            raise ValueError("Primary horizon must be one of the declared horizons")
        return cls(
            path=path.resolve(),
            instrument=Instrument(payload["exchange"], payload["symbol"]),
            **dates,
            expected_sessions=sessions,
            horizons_minutes=horizons,
            primary_horizon_minutes=payload["primary_horizon_minutes"],
            strategy_file=(path.parent / payload["strategy_file"]).resolve(),
            snapshot_directory=(path.parent / payload["snapshot_directory"]).resolve(),
        )

    def split(self, day: date) -> str:
        if day < self.start:
            return "warmup"
        if day <= self.development_end:
            return "development"
        return "validation" if day <= self.validation_end else "holdout"

    @property
    def bounds(self) -> tuple[datetime, datetime]:
        return (
            datetime.combine(
                self.warmup_start, SESSION_OPEN_TIME, tzinfo=INDIA_TIMEZONE
            ),
            datetime.combine(self.end, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE),
        )

    @property
    def manifest_path(self) -> Path:
        return self.path.parent / "manifest.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_code_hashes() -> dict[str, str]:
    root = Path(ai_trader.__file__).parent
    paths = [root / "clock.py"]
    for package in ("features", "scanner", "costs", "strategy", "replay"):
        paths.extend(sorted((root / package).glob("*.py")))
    return {path.relative_to(root).as_posix(): file_hash(path) for path in paths}


def audit_candles(
    candles: Sequence[Candle], plan: ResearchPlan, strategy: StrategyConfig
) -> dict[str, Any]:
    regular = tuple(
        candle
        for candle in candles
        if 0 <= minutes_since_open(candle.start_time) < SESSION_MINUTES
    )
    if len({candle.start_time for candle in regular}) != len(regular):
        raise ValueError("Duplicate minute in research candles")
    measured = {
        candle.start_time.date()
        for candle in regular
        if candle.start_time.date() >= plan.start
    }
    if measured != set(plan.expected_sessions):
        raise ValueError(
            "Session coverage mismatch: "
            f"missing {sorted(set(plan.expected_sessions) - measured)}, "
            f"unexpected {sorted(measured - set(plan.expected_sessions))}"
        )
    cutoff = strategy.square_off_minutes_since_open
    if cutoff != int(cutoff) or not 0 < cutoff <= SESSION_MINUTES:
        raise ValueError("Research requires a minute-aligned intraday square-off")
    by_day: dict[date, list[Candle]] = defaultdict(list)
    for candle in regular:
        by_day[candle.start_time.date()].append(candle)
    sessions: list[dict[str, Any]] = []
    for day, bars in sorted(by_day.items()):
        opening = datetime.combine(day, SESSION_OPEN_TIME, tzinfo=INDIA_TIMEZONE)
        expected = {
            opening + timedelta(minutes=minute) for minute in range(int(cutoff))
        }
        missing = expected - {bar.start_time for bar in bars}
        if missing:
            raise ValueError(f"{day}: {len(missing)} missing minutes before square-off")
        if any(
            price % strategy.tick_size
            for bar in bars
            for price in (bar.open, bar.high, bar.low, bar.close)
        ):
            raise ValueError(f"{day}: prices do not lie on the configured tick grid")
        sessions.append(
            {
                "session": day.isoformat(),
                "split": plan.split(day),
                "candles": len(bars),
                "first_bar": bars[0].start_time.isoformat(),
                "last_bar": bars[-1].start_time.isoformat(),
                "minutes_before_square_off": len(expected),
                "unknown_volume": [
                    bar.start_time.isoformat() for bar in bars if bar.volume is None
                ],
            }
        )
    warmup = tuple(bar for bar in regular if bar.start_time.date() < plan.start)
    features = FeatureEngine()
    accepted = features.warm_up(warmup)
    if not accepted or not features.is_ready(plan.instrument):
        raise ValueError("Warm-up does not initialize the core feature set")
    return {
        "raw_candles": len(candles),
        "regular_candles": len(regular),
        "warmup_candles": accepted,
        "evaluation_candles": len(regular) - accepted,
        "sessions": sessions,
    }


def verify_snapshot(plan: ResearchPlan, strategy: StrategyConfig) -> dict[str, Any]:
    manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
    if manifest["plan_sha256"] != file_hash(plan.path):
        raise ValueError("Research plan changed after freezing")
    if manifest["strategy_sha256"] != strategy.fingerprint:
        raise ValueError("Baseline strategy changed after freezing")
    if manifest["decision_code_sha256"] != decision_code_hashes():
        raise ValueError("Decision code changed; record a new experiment version")
    for name, expected in manifest["candle_files_sha256"].items():
        if file_hash(plan.snapshot_directory / name) != expected:
            raise ValueError(f"Frozen candle file changed: {name}")
    candles = CandleStore(plan.snapshot_directory).load(plan.instrument, *plan.bounds)
    if audit_candles(candles, plan, strategy) != manifest["audit"]:
        raise ValueError("Frozen coverage no longer matches its audit")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/candles"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-query the requested source range; never overwrite a frozen snapshot.",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify only; never authenticate."
    )
    args = parser.parse_args(argv)
    if args.refresh and (args.offline or args.verify):
        parser.error("--refresh requires an online preparation run")
    try:
        plan = ResearchPlan.load(args.plan)
        strategy = StrategyConfig.load(plan.strategy_file)
        if args.verify:
            manifest = verify_snapshot(plan, strategy)
        else:
            if plan.manifest_path.exists():
                raise ValueError(
                    "Research is already frozen; use --verify or a new plan directory"
                )
            target = CandleStore(plan.snapshot_directory)
            if (
                target.path_for(plan.instrument).exists()
                or target.fetched_path_for(plan.instrument).exists()
            ):
                raise ValueError("Snapshot files already exist; refusing to overwrite")
            if plan.bounds[1] > last_completed_session_close(
                datetime.now(INDIA_TIMEZONE)
            ):
                raise ValueError("The research end session has not finished")
            broker = (
                None
                if args.offline
                else GrowwBroker.authenticate(load_groww_settings())
            )
            store = CandleStore(args.cache, broker)
            candles = store.load(plan.instrument, *plan.bounds, refresh=args.refresh)
            audit = audit_candles(candles, plan, strategy)
            frozen = store.freeze(
                plan.instrument, *plan.bounds, plan.snapshot_directory
            )
            if frozen != candles:
                raise ValueError("Source candles changed while freezing")
            manifest = {
                "schema_version": 1,
                "frozen_at": datetime.now(INDIA_TIMEZONE).isoformat(),
                "plan_sha256": file_hash(plan.path),
                "strategy_sha256": strategy.fingerprint,
                "decision_code_sha256": decision_code_hashes(),
                "candle_files_sha256": {
                    path.name: file_hash(path)
                    for path in (
                        target.path_for(plan.instrument),
                        target.fetched_path_for(plan.instrument),
                    )
                },
                "audit": audit,
                "outcomes_evaluated": False,
            }
            with plan.manifest_path.open("x", encoding="utf-8", newline="") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            verify_snapshot(plan, strategy)
    except (
        ArithmeticError,
        ConfigurationError,
        GrowwBrokerError,
        CandleStoreError,
        OSError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    counts: dict[str, int] = defaultdict(int)
    for session in manifest["audit"]["sessions"]:
        counts[session["split"]] += 1
    print(
        json.dumps(
            {
                "status": "verified" if args.verify else "frozen",
                "strategy_sha256": strategy.fingerprint,
                "sessions_by_split": dict(counts),
                "audit": {
                    key: value
                    for key, value in manifest["audit"].items()
                    if key != "sessions"
                },
                "outcomes_evaluated": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
