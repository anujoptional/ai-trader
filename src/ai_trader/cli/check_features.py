"""Check the feature engine against a recent completed Groww session.

Backfills one full NSE session through the existing market-data path and folds
every completed candle through ``FeatureEngine``. The candles are historical,
but they are the same ``Candle`` objects the live path produces, so this
exercises the production calculation path rather than a parallel one.

Only the latest snapshot is printed. ``--export-csv`` additionally writes one
exact, unrounded row per candle, which is what makes it practical to check a
real session against a trusted implementation before anything is built on these
numbers.
"""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import fields
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBroker,
    GrowwBrokerError,
)
from ai_trader.config import ConfigurationError, load_groww_settings
from ai_trader.features import (
    DERIVED_FEATURE_NAMES,
    FEATURE_CONTEXT,
    FeatureEngine,
    FeatureReadiness,
    FeatureSnapshot,
)
from ai_trader.market import MarketState

_INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_SESSION_START = time(hour=9, minute=15)
_SESSION_END = time(hour=15, minute=30)
_MAX_WEEKDAYS = 10

_DISPLAY_EXPONENT = Decimal("0.000001")
"""Display precision for derived values; the engine itself never rounds."""

_CANDLE_FIELDS = ("open", "high", "low", "close")


class SessionNotFoundError(RuntimeError):
    """Raised when no completed trading session can be found for validation."""


def _price(value: Decimal) -> str:
    return format(value, "f")


def _derived(value: Decimal | None) -> str | None:
    if value is None:
        return None
    # Pinned so the displayed number depends only on the value, never on
    # whatever precision the surrounding process happens to be running at.
    with localcontext(FEATURE_CONTEXT):
        rounded = value.quantize(_DISPLAY_EXPONENT, rounding=ROUND_HALF_EVEN)
    return format(rounded, "f")


def _exact(value: Decimal | None) -> str:
    return "" if value is None else format(value, "f")


def _readiness(readiness: FeatureReadiness) -> dict[str, bool]:
    return {field.name: getattr(readiness, field.name) for field in fields(readiness)}


def _summarize(snapshot: FeatureSnapshot) -> dict[str, object]:
    summary: dict[str, object] = {
        "instrument": snapshot.instrument.trading_symbol,
        "candle_start_time": snapshot.candle_start_time.isoformat(),
        "candle_end_time": snapshot.candle_end_time.isoformat(),
        "volume": snapshot.volume,
    }
    for name in _CANDLE_FIELDS:
        summary[name] = _price(getattr(snapshot, name))
    for name in DERIVED_FEATURE_NAMES:
        summary[name] = _derived(getattr(snapshot, name))
    summary["readiness"] = _readiness(snapshot.readiness)
    summary["core_ready"] = snapshot.readiness.core_ready
    return summary


def _export_csv(
    path: Path,
    snapshots: Sequence[FeatureSnapshot],
    *,
    overwrite: bool,
) -> None:
    """Write every snapshot at full precision for offline comparison.

    An existing file is refused rather than truncated. The whole point of this
    export is to compare a session against a trusted implementation, and a run
    that fails after opening the file would otherwise replace the reference data
    with an empty header and report success.
    """
    header = (
        "start_time",
        "end_time",
        *_CANDLE_FIELDS,
        "volume",
        *DERIVED_FEATURE_NAMES,
    )
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for snapshot in snapshots:
            writer.writerow(
                (
                    snapshot.candle_start_time.isoformat(),
                    snapshot.candle_end_time.isoformat(),
                    *(_exact(getattr(snapshot, name)) for name in _CANDLE_FIELDS),
                    "" if snapshot.volume is None else snapshot.volume,
                    *(
                        _exact(getattr(snapshot, name))
                        for name in DERIVED_FEATURE_NAMES
                    ),
                )
            )


def _find_recent_completed_session(
    broker: GrowwBroker,
    now: datetime,
) -> tuple[date, tuple[OHLCVCandle, ...]]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("The current time must be timezone-aware.")

    local_now = now.astimezone(_INDIA_TIMEZONE)
    candidate = local_now.date()
    if local_now < datetime.combine(candidate, _SESSION_END, tzinfo=_INDIA_TIMEZONE):
        candidate -= timedelta(days=1)

    weekdays_checked = 0
    while weekdays_checked < _MAX_WEEKDAYS:
        if candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
            continue

        weekdays_checked += 1
        candles = broker.get_historical_candles(
            instrument=_RELIANCE,
            start=datetime.combine(candidate, _SESSION_START, tzinfo=_INDIA_TIMEZONE),
            end=datetime.combine(candidate, _SESSION_END, tzinfo=_INDIA_TIMEZONE),
            interval=CandleInterval.ONE_MINUTE,
        )
        if candles:
            return candidate, candles

        candidate -= timedelta(days=1)

    raise SessionNotFoundError(
        "No completed NSE trading session was found in the last 10 weekdays."
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute features for a recent completed RELIANCE session."
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write every candle's features to PATH at full precision.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --export-csv to replace an existing file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Warm the feature engine on one session and report its latest snapshot."""
    args = _parse_args(argv)

    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        broker = GrowwBroker.authenticate(settings)
        trading_date, historical = _find_recent_completed_session(
            broker,
            now=datetime.now(tz=_INDIA_TIMEZONE),
        )
    except SessionNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except GrowwAuthenticationError:
        print("Groww authentication failed.", file=sys.stderr)
        return 1
    except GrowwBrokerError:
        print("Groww feature check failed.", file=sys.stderr)
        return 1

    # Sized to the session actually returned rather than to the default window,
    # so a longer-than-usual backfill is fed to the engine whole instead of
    # having its oldest candles evicted before they are ever seen.
    state = MarketState(max_candles=max(len(historical), 1))
    backfilled = state.backfill(_RELIANCE, historical)
    market = state.snapshot(_RELIANCE)
    if market is None:
        print("Market state produced no snapshot.", file=sys.stderr)
        return 1

    engine = FeatureEngine()
    snapshots = [
        snapshot
        for snapshot in (engine.update(candle) for candle in market.candles)
        if snapshot is not None
    ]
    latest = engine.snapshot(_RELIANCE)
    if latest is None:
        print("Feature engine produced no snapshot.", file=sys.stderr)
        return 1

    if args.export_csv is not None:
        try:
            _export_csv(args.export_csv, snapshots, overwrite=args.overwrite)
        except FileExistsError:
            print(
                f"{args.export_csv} already exists; pass --overwrite to replace it.",
                file=sys.stderr,
            )
            return 1
        except OSError:
            print(f"Could not write {args.export_csv}.", file=sys.stderr)
            return 1

    summary = {
        "trading_date": trading_date.isoformat(),
        "backfilled_candles": backfilled,
        "feature_candles": len(snapshots),
        "duplicate_candles": engine.duplicate_candle_count,
        "out_of_order_candles": engine.out_of_order_candle_count,
        "latest": _summarize(latest),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
