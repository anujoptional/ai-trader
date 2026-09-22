"""Check the feature engine against Groww historical and live data.

Backfills one full NSE session through the existing market-data path and folds
every completed candle through ``FeatureEngine``. The candles are historical,
but they are the same ``Candle`` objects the live path produces, so this
exercises the production calculation path rather than a parallel one.

``--live`` then keeps the same engine running against the live stream, so a
session's worth of history is followed by candles built from real ticks. That
is the only way to see the whole market-to-features path work on live data:
without it the engine is only ever fed candles the broker already assembled,
and the tick aggregation, volume stamping and minute-boundary handling in
between go unexercised.

Live candles reach the engine only when they close naturally, which needs a
tick from the following minute. The trailing partial minute is deliberately
left unflushed: a candle covering forty seconds looks exactly like one covering
sixty, and feeding it would report volume-derived features for a fraction of a
minute as though they described all of it.

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
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

from ai_trader.broker import MarketTick, OHLCVCandle, ReadOnlyBroker
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBroker,
    GrowwBrokerError,
    GrowwStreamConnectionError,
)
from ai_trader.cli._session import (
    INDIA_TIMEZONE,
    RELIANCE,
    SessionNotFoundError,
    find_recent_completed_session,
)
from ai_trader.config import ConfigurationError, load_groww_settings
from ai_trader.features import (
    DERIVED_FEATURE_NAMES,
    FEATURE_CONTEXT,
    FeatureEngine,
    FeatureReadiness,
    FeatureSnapshot,
)
from ai_trader.market import Candle, MarketState, VolumePoller

_INDIA_TIMEZONE = INDIA_TIMEZONE
_RELIANCE = RELIANCE

_DEFAULT_LIVE_SECONDS = 180.0
"""Long enough for a live candle to actually close.

The builder discards each instrument's first partial minute, and a candle only
closes when the next minute's first tick arrives, so a window has to span three
minute boundaries before the engine sees even one live candle. Three minutes is
the smallest default that is not routinely disappointing.
"""

_DEFAULT_LIVE_TICKS = 10_000
"""Effectively unbounded: the time window is the intended control.

``collect`` stops at whichever limit it reaches first and a liquid instrument
can tick several times a second, so a tick cap low enough to be interesting
would end the window early and silently.
"""

_LIVE_HEADROOM_CANDLES = 30
"""Retention added in live mode so live candles do not evict the backfill."""

_DISPLAY_EXPONENT = Decimal("0.000001")
"""Display precision for derived values; the engine itself never rounds."""

_CANDLE_FIELDS = ("open", "high", "low", "close")


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
    broker: ReadOnlyBroker,
    now: datetime,
) -> tuple[date, tuple[OHLCVCandle, ...]]:
    """Find a recent session, reading the full NSE window for backfill."""
    return find_recent_completed_session(broker, now, instrument=_RELIANCE)


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
    parser.add_argument(
        "--live",
        action="store_true",
        help="After backfilling, feed live ticks through the same engine.",
    )
    parser.add_argument(
        "--live-seconds",
        type=float,
        default=_DEFAULT_LIVE_SECONDS,
        metavar="SECONDS",
        help="How long to stream live ticks when --live is set.",
    )
    parser.add_argument(
        "--live-ticks",
        type=int,
        default=_DEFAULT_LIVE_TICKS,
        metavar="COUNT",
        help="Stop the live window early after COUNT ticks.",
    )
    args = parser.parse_args(argv)
    if args.live_seconds <= 0:
        parser.error("--live-seconds must be positive.")
    if args.live_ticks <= 0:
        parser.error("--live-ticks must be positive.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Warm the feature engine on one session and report its latest snapshot."""
    args = _parse_args(argv)

    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    engine = FeatureEngine()
    snapshots: list[FeatureSnapshot] = []
    live_candles = 0
    live_summary: dict[str, object] | None = None

    def on_live_candle(candle: Candle) -> None:
        # Fires on the broker's feed thread when a live minute closes. The
        # engine takes its own lock and list.append is atomic, so the snapshot
        # list needs no further synchronization.
        nonlocal live_candles
        live_candles += 1
        snapshot = engine.update(candle)
        if snapshot is not None:
            snapshots.append(snapshot)

    try:
        broker = GrowwBroker.authenticate(settings)
        trading_date, historical = _find_recent_completed_session(
            broker,
            now=datetime.now(tz=_INDIA_TIMEZONE),
        )

        if args.live:
            poller = VolumePoller(broker, (_RELIANCE,))
            stamped_count = 0

            def stamp(tick: MarketTick) -> MarketTick:
                # Wrapping the poller's own stamper rather than stamping at the
                # call site keeps the counting here while the state object owns
                # the seam, so no live path can forget to stamp.
                nonlocal stamped_count
                stamped = poller.stamp(tick)
                if stamped.cumulative_volume is not None:
                    stamped_count += 1
                return stamped

            # Room for the backfill plus the live candles that follow it, so
            # streaming does not evict the history the warm-up depended on out
            # from under the market snapshot.
            state = MarketState(
                max_candles=max(len(historical), 1) + _LIVE_HEADROOM_CANDLES,
                on_candle=on_live_candle,
                tick_stamper=stamp,
            )
        else:
            # Sized to the session actually returned rather than to the default
            # window, so a longer-than-usual backfill is fed to the engine whole
            # instead of having its oldest candles evicted before they are seen.
            state = MarketState(max_candles=max(len(historical), 1))

        backfilled = state.backfill(_RELIANCE, historical)
        market = state.snapshot(_RELIANCE)
        if market is None:
            print("Market state produced no snapshot.", file=sys.stderr)
            return 1

        # Backfill bypasses the candle builder, so ``on_live_candle`` does not
        # fire for these and the warm-up stays an ordinary loop.
        for candle in market.candles:
            snapshot = engine.update(candle)
            if snapshot is not None:
                snapshots.append(snapshot)
        backfill_snapshots = len(snapshots)

        if args.live:
            tick_count = 0

            def record(tick: MarketTick) -> None:
                nonlocal tick_count
                tick_count += 1
                state.record_tick(tick)

            with poller:
                stream = broker.create_ltp_stream((_RELIANCE,))
                stream.collect(
                    max_ticks=args.live_ticks,
                    timeout_seconds=args.live_seconds,
                    on_tick=record,
                )
            # No flush: the trailing partial minute is discarded rather than
            # closed, because a candle covering part of a minute is
            # indistinguishable downstream from one covering all of it.
            live_summary = {
                "live_ticks": tick_count,
                "live_candles": live_candles,
                "late_ticks": state.late_tick_count,
                "stamped_ticks": stamped_count,
                "stale_stamps": poller.stale_stamp_count,
                "session_volume": poller.latest(_RELIANCE),
                "polls": poller.poll_count,
                "poll_failures": poller.failure_count,
                "poll_regressions": poller.regression_count,
                "poll_error": poller.last_error,
            }
    except SessionNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    except GrowwAuthenticationError:
        print("Groww authentication failed.", file=sys.stderr)
        return 1
    except GrowwStreamConnectionError:
        print(
            "Groww live feed unreachable; the stream connection failed.",
            file=sys.stderr,
        )
        return 1
    except GrowwBrokerError:
        print("Groww feature check failed.", file=sys.stderr)
        return 1

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

    summary: dict[str, object] = {
        "trading_date": trading_date.isoformat(),
        "backfilled_candles": backfilled,
        "feature_candles": backfill_snapshots,
        "duplicate_candles": engine.duplicate_candle_count,
        "out_of_order_candles": engine.out_of_order_candle_count,
        "latest": _summarize(latest),
    }
    if live_summary is not None:
        summary["live"] = live_summary
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
