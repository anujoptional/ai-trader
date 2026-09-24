"""Check the deterministic scanner against Groww historical and live data.

Walks a session minute by minute through the production path — ``Candle`` to
``FeatureEngine`` to ``Scanner`` — and scans at every step, so the answer
printed for 11:47 is the one the scanner would have given at 11:47 rather than
one computed with the rest of the day in hand. ``--live`` then keeps the same
scanner running against real ticks, which is the only way to watch the whole
broker-to-candidate path work on a live feed.

This is the first check that reaches the top of the deterministic half of the
safety sandwich. Everything below it — broker, market state, feature engine —
already had one; the scanner did not, so until now the only evidence it ran on
live data at all was that the layers underneath it did.

**One instrument is not a universe.** Every check in this package watches
RELIANCE, so each cycle here scans exactly one name and the tally adds to one.
That is enough to prove the path carries a real candidate end to end, and enough
to show what the rules do minute by minute on live data; it is not a measurement
of how often the scanner fires across a market, and the replay engine in section
4.5 is what will answer that.

**Nothing here is evidence that a rule is worth trading.** A scanner is
validated by its candidates being measured against what happened next, and
nothing in this file measures an outcome. What it does validate is narrower and
still worth having: that the rules read what they claim to read, that they
decline when they should, and that the same minute scans the same way whether it
arrived as a broker candle or was built from ticks.

The cost screen stays off unless ``--max-atr-multiple`` is given. That is
deliberate rather than an oversight. ``FeasibilityPolicy`` refuses to default
the multiple because section 7 forbids inventing a threshold, and a CLI that
quietly picked one on the caller's behalf would put back exactly what the
library declines to assume.

Usage::

    python -m ai_trader.cli.check_scanner
    python -m ai_trader.cli.check_scanner --max-atr-multiple 30
    python -m ai_trader.cli.check_scanner --live --live-seconds 600
"""

import argparse
import csv
import json
import signal
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
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
from ai_trader.costs import (
    FIXED_CLIP_NOTIONAL,
    GROWW_INTRADAY_EQUITY,
    ZERODHA_INTRADAY_EQUITY,
    CostModel,
)
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine, FeatureSnapshot
from ai_trader.market import Candle, MarketState, StreamSupervisor, VolumePoller
from ai_trader.scanner import (
    Candidate,
    FeasibilityCheck,
    FeasibilityPolicy,
    MarketContext,
    PortfolioState,
    Scanner,
    ScannerConfig,
    ScanResult,
)

_INDIA_TIMEZONE = INDIA_TIMEZONE
_RELIANCE = RELIANCE

_DEFAULT_LIVE_SECONDS = 180.0
"""Long enough for a live candle to actually close.

The builder discards each instrument's first partial minute and a candle only
closes when the next minute's first tick arrives, so a window has to span three
minute boundaries before the scanner sees even one live cycle.
"""

_DEFAULT_LIVE_TICKS = 10_000
"""Effectively unbounded: the time window is the intended control."""

_LIVE_HEADROOM_CANDLES = 30
"""Retention added in live mode so live candles do not evict the backfill."""

_DISPLAY_EXPONENT = Decimal("0.000001")
"""Display precision for scores and fractions; nothing upstream ever rounds."""

_DEFAULT_GROSS_TARGET = Decimal("0.002")
"""The stated aim read as a move from the entry: 0.2% above the buy price.

A *published objective* rather than an invented threshold — it is the target the
strategy was written around, and section 7's rule is about inventing numbers
with nothing behind them, not about restating the one the caller gave. The
multiple it is screened against has no such provenance, which is why that one
has to be typed on the command line.
"""

_SCHEDULES: dict[str, CostModel] = {
    "groww": GROWW_INTRADAY_EQUITY,
    "zerodha": ZERODHA_INTRADAY_EQUITY,
    "kite": ZERODHA_INTRADAY_EQUITY,
}
"""The published schedules, under the names someone would actually type."""


def _price(value: Decimal) -> str:
    return format(value, "f")


def _exact(value: Decimal | None) -> str:
    """Full precision, for the CSV. Nothing here is ever compared rounded."""
    return "" if value is None else format(value, "f")


def _derived(value: Decimal | None) -> str | None:
    if value is None:
        return None
    # Pinned so the printed number depends only on the value, never on whatever
    # precision the surrounding process happens to be running at.
    with localcontext(FEATURE_CONTEXT):
        rounded = value.quantize(_DISPLAY_EXPONENT, rounding=ROUND_HALF_EVEN)
    return format(rounded, "f")


def _fraction(value: Decimal | None) -> str | None:
    """Render a fraction with its percentage beside it.

    Both, always. A fraction and a percentage of the same quantity differ by a
    hundred, and printing one alone is how that error survives a review.
    """
    if value is None:
        return None
    with localcontext(FEATURE_CONTEXT):
        percent = (value * 100).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
        exact = value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)
    return f"{exact} ({percent}%)"


@dataclass(slots=True)
class _Tally:
    """What a run of cycles produced, and what each one declined to produce.

    Aggregated rather than printed per cycle because a session is several
    hundred cycles and the interesting facts are distributional: how often a
    rule fired, which way, and which of the four ways of producing nothing was
    responsible. The per-cycle detail is available through ``--export-csv``.
    """

    cycles: int = 0
    cycles_with_candidates: int = 0
    candidates: int = 0
    considered: int = 0
    not_ready: int = 0
    unreachable: int = 0
    truncated: int = 0
    suppressed: dict[str, int] = field(default_factory=dict)
    by_rule: dict[str, int] = field(default_factory=dict)
    by_direction: dict[str, int] = field(default_factory=dict)
    feasibility: dict[str, int] = field(default_factory=dict)
    first_candidate_at: str | None = None
    last_candidate_at: str | None = None

    def record(self, result: ScanResult) -> None:
        self.cycles += 1
        self.considered += result.considered
        self.not_ready += result.not_ready
        self.unreachable += result.unreachable
        self.truncated += result.truncated
        for reason, count in result.suppressed.items():
            self.suppressed[reason.value] = self.suppressed.get(reason.value, 0) + count
        if not result.candidates:
            return
        self.cycles_with_candidates += 1
        self.candidates += len(result.candidates)
        stamp = result.as_of.isoformat()
        if self.first_candidate_at is None:
            self.first_candidate_at = stamp
        self.last_candidate_at = stamp
        for candidate in result.candidates:
            direction = candidate.direction.value
            self.by_direction[direction] = self.by_direction.get(direction, 0) + 1
            for rule in candidate.rules:
                self.by_rule[rule] = self.by_rule.get(rule, 0) + 1

    def record_feasibility(self, reason: str) -> None:
        self.feasibility[reason] = self.feasibility.get(reason, 0) + 1

    def summary(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "cycles_with_candidates": self.cycles_with_candidates,
            "candidates": self.candidates,
            "considered": self.considered,
            "not_ready": self.not_ready,
            "unreachable": self.unreachable,
            "truncated": self.truncated,
            "suppressed": dict(sorted(self.suppressed.items())),
            "by_rule": dict(sorted(self.by_rule.items())),
            "by_direction": dict(sorted(self.by_direction.items())),
            "feasibility": dict(sorted(self.feasibility.items())),
            "first_candidate_at": self.first_candidate_at,
            "last_candidate_at": self.last_candidate_at,
        }


def _candidate_summary(candidate: Candidate) -> dict[str, object]:
    summary: dict[str, object] = {
        "instrument": candidate.instrument.trading_symbol,
        "direction": candidate.direction.value,
        "score": _derived(candidate.score),
        "rules": list(candidate.rules),
        "as_of": candidate.as_of.isoformat(),
        "reference_price": _price(candidate.reference_price),
        "evidence": {
            name: _derived(value) for name, value in sorted(candidate.evidence.items())
        },
    }
    check = candidate.feasibility
    if check is not None:
        summary["feasibility"] = {
            "required_gross_fraction": _fraction(check.required_gross_fraction),
            "atr_fraction": _fraction(check.atr_fraction),
            "minutes_remaining": _derived(check.minutes_remaining),
            "reason": None if check.reason is None else check.reason.value,
        }
    return summary


def _scan_summary(result: ScanResult) -> dict[str, object]:
    return {
        "as_of": result.as_of.isoformat(),
        "candidates": [_candidate_summary(item) for item in result.candidates],
        "considered": result.considered,
        "not_ready": result.not_ready,
        "unreachable": result.unreachable,
        "suppressed": {
            reason.value: count for reason, count in sorted(result.suppressed.items())
        },
        "truncated": result.truncated,
    }


_CSV_HEADER = (
    "end_time",
    "close",
    "considered",
    "not_ready",
    "unreachable",
    "suppressed",
    "truncated",
    "candidates",
    "top_direction",
    "top_score",
    "top_rules",
    "required_gross_fraction",
    "atr_fraction",
    "minutes_remaining",
    "feasibility_reason",
)


def _feasibility_name(check: FeasibilityCheck) -> str:
    """What to call a screened cycle, in one place.

    Both the tally and the export name the same outcomes, and a passing check
    carries no reason of its own — it is an absence, which has to be given a
    word before it can be counted. Naming it here rather than at each site is
    what stops the summary and the CSV from disagreeing about what to call it.
    """
    return "reachable" if check.reason is None else check.reason.value


def _csv_row(
    snapshot: FeatureSnapshot,
    result: ScanResult,
    check: FeasibilityCheck | None,
) -> tuple[object, ...]:
    """One cycle, flattened to its headline and the reason behind it.

    The top candidate rather than all of them: the budget is five and a row per
    candidate would make the file hard to line up against a chart, which is the
    one thing it exists for.

    The screen's working comes from the cycle's own check and not from the top
    candidate's, though on a cycle that produced one the two are the same
    object. They part company exactly where this file earns its keep: when the
    screen is what rejected the cycle there is no candidate to read the numbers
    off, and sourcing them from one would blank the hurdle and the ATR on every
    row that needed them — naming a rejection while dropping the arithmetic
    behind it.
    """
    top = result.candidates[0] if result.candidates else None
    return (
        snapshot.candle_end_time.isoformat(),
        format(snapshot.close, "f"),
        result.considered,
        result.not_ready,
        result.unreachable,
        ";".join(
            f"{name.value}={count}" for name, count in sorted(result.suppressed.items())
        ),
        result.truncated,
        len(result.candidates),
        "" if top is None else top.direction.value,
        "" if top is None else format(top.score, "f"),
        "" if top is None else ";".join(top.rules),
        _exact(None if check is None else check.required_gross_fraction),
        _exact(None if check is None else check.atr_fraction),
        _exact(None if check is None else check.minutes_remaining),
        "" if check is None else _feasibility_name(check),
    )


def _export_csv(
    path: Path,
    rows: Sequence[tuple[object, ...]],
    *,
    overwrite: bool,
) -> None:
    """Write every cycle at full precision for offline comparison.

    An existing file is refused rather than truncated, for the same reason as in
    ``check_features``: a run that fails after opening the file would otherwise
    replace reference data with an empty header and report success.
    """
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_HEADER)
        writer.writerows(rows)


def _find_recent_completed_session(
    broker: ReadOnlyBroker,
    now: datetime,
) -> tuple[date, tuple[OHLCVCandle, ...]]:
    """Find a recent session, reading the full NSE window for backfill."""
    return find_recent_completed_session(broker, now, instrument=_RELIANCE)


@contextmanager
def _stop_on_interrupt(supervisor: StreamSupervisor) -> Iterator[None]:
    """Turn Ctrl-C into an orderly stop, then restore the previous handler.

    Resources are already released on an interrupt -- the supervisor closes its
    stream in a ``finally`` and the poller's context manager stops its thread --
    so this is about the report rather than about cleanup. A window long enough
    to be worth supervising is one whose counters are the point of running it,
    and the default handler discards them for a traceback.

    Installing a handler is only legal on the main thread. Anywhere else the
    default behaviour is kept rather than raising, because refusing to stream at
    all is a worse trade than losing one summary.
    """
    try:
        previous = signal.signal(signal.SIGINT, lambda *_: supervisor.stop())
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a recent completed RELIANCE session minute by minute."
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        metavar="COUNT",
        help="Candidate budget per cycle (default: the scanner's own).",
    )
    parser.add_argument(
        "--max-atr-multiple",
        type=Decimal,
        default=None,
        metavar="MULTIPLE",
        help=(
            "Turn the cost screen on, allowing a required move of at most this "
            "many one-minute ATRs. No default: stating it is the caller's job."
        ),
    )
    parser.add_argument(
        "--clip",
        type=Decimal,
        default=FIXED_CLIP_NOTIONAL,
        metavar="RUPEES",
        help="Intended turnover per leg when the cost screen is on.",
    )
    parser.add_argument(
        "--gross-target",
        type=Decimal,
        default=_DEFAULT_GROSS_TARGET,
        metavar="FRACTION",
        help="Exit distance above entry as a fraction, not a percentage.",
    )
    parser.add_argument(
        "--broker",
        choices=sorted(_SCHEDULES),
        default="groww",
        help="Whose published cost schedule the screen prices with.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write one row per scanned cycle to PATH at full precision.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --export-csv to replace an existing file.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="After backfilling, scan candles built from live ticks.",
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
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be at least 1.")
    if args.max_atr_multiple is not None and args.max_atr_multiple <= 0:
        parser.error("--max-atr-multiple must be positive.")
    if args.clip <= 0:
        parser.error("--clip must be positive.")
    if args.gross_target <= 0:
        parser.error("--gross-target must be positive.")
    return args


def _build_scanner(
    args: argparse.Namespace,
) -> tuple[Scanner, FeasibilityPolicy | None]:
    """Assemble the scanner the run will use, cost screen included or not."""
    policy: FeasibilityPolicy | None = None
    if args.max_atr_multiple is not None:
        policy = FeasibilityPolicy.from_gross_target(
            target_notional=args.clip,
            gross_target_fraction=args.gross_target,
            max_atr_multiple=args.max_atr_multiple,
            costs=_SCHEDULES[args.broker],
        )
    budget = (
        ScannerConfig().max_candidates
        if args.max_candidates is None
        else args.max_candidates
    )
    config = ScannerConfig(max_candidates=budget, feasibility=policy)
    return Scanner(config), policy


def main(argv: Sequence[str] | None = None) -> int:
    """Walk a session through the scanner and report what it produced."""
    args = _parse_args(argv)

    try:
        settings = load_groww_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        scanner, policy = _build_scanner(args)
    except (ArithmeticError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    engine = FeatureEngine()
    backfill_tally = _Tally()
    live_tally = _Tally()
    rows: list[tuple[object, ...]] = []
    latest: ScanResult | None = None
    live_candles = 0
    live_summary: dict[str, object] | None = None
    live_failed = False

    def scan(snapshot: FeatureSnapshot, tally: _Tally) -> None:
        """One decision cycle, scanned as of the candle that closed it.

        ``as_of`` is the candle's end time rather than the wall clock, which is
        what makes a backfilled minute and the same minute live produce the same
        answer. The scanner is stateless between cycles and everything that
        persists arrives as an argument, so calling it from the feed thread in
        live mode needs no synchronization of its own.
        """
        nonlocal latest
        as_of = snapshot.candle_end_time
        check: FeasibilityCheck | None = None
        if policy is not None:
            # A second call to the same pure function on the same snapshot as
            # the one the scan makes internally, so the two cannot disagree. It
            # is here because ``ScanResult`` counts rejections without naming
            # them, and which of the four reasons dominated is the diagnostic
            # actually worth having.
            check = policy.evaluate(snapshot)
            tally.record_feasibility(_feasibility_name(check))
        result = scanner.scan(
            (snapshot,),
            PortfolioState.empty(as_of),
            context=MarketContext(as_of=as_of),
        )
        tally.record(result)
        rows.append(_csv_row(snapshot, result, check))
        latest = result

    def on_live_candle(candle: Candle) -> None:
        # Fires on the broker's feed thread when a live minute closes.
        nonlocal live_candles
        live_candles += 1
        snapshot = engine.update(candle)
        if snapshot is not None:
            scan(snapshot, live_tally)

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
                nonlocal stamped_count
                stamped = poller.stamp(tick)
                if stamped.cumulative_volume is not None:
                    stamped_count += 1
                return stamped

            state = MarketState(
                max_candles=max(len(historical), 1) + _LIVE_HEADROOM_CANDLES,
                on_candle=on_live_candle,
                tick_stamper=stamp,
            )
        else:
            state = MarketState(max_candles=max(len(historical), 1))

        backfilled = state.backfill(_RELIANCE, historical)
        market = state.snapshot(_RELIANCE)
        if market is None:
            print("Market state produced no snapshot.", file=sys.stderr)
            return 1

        # Backfill bypasses the candle builder, so ``on_live_candle`` does not
        # fire for these and the walk-forward stays an ordinary loop.
        for candle in market.candles:
            snapshot = engine.update(candle)
            if snapshot is not None:
                scan(snapshot, backfill_tally)

        if args.live:
            supervisor = StreamSupervisor(
                open_stream=lambda: broker.create_ltp_stream((_RELIANCE,)),
                on_tick=state.record_tick,
                # Narrower than GrowwBrokerError deliberately: only the
                # connection error means "try again".
                retry_on=(GrowwStreamConnectionError,),
            )
            with poller, _stop_on_interrupt(supervisor):
                report = supervisor.run(
                    duration_seconds=args.live_seconds,
                    max_ticks=args.live_ticks,
                )
            # No flush: the trailing partial minute is discarded rather than
            # closed, because a candle covering part of a minute is
            # indistinguishable downstream from one covering all of it — and a
            # scanner fed one would score a fraction of a minute's volume as
            # though it described the whole.
            live_failed = report.sessions == 0 and report.failures > 0
            live_summary = {
                "live_ticks": report.ticks,
                "live_candles": live_candles,
                "late_ticks": state.late_tick_count,
                "stamped_ticks": stamped_count,
                "stale_stamps": poller.stale_stamp_count,
                "session_volume": poller.latest(_RELIANCE),
                "polls": poller.poll_count,
                "poll_failures": poller.failure_count,
                "poll_regressions": poller.regression_count,
                "poll_error": poller.last_error,
                "stream_sessions": report.sessions,
                "silent_sessions": report.silent_sessions,
                "reconnects": report.reconnects,
                "stream_failures": report.failures,
                "stopped_because": report.stopped_because,
                "stream_error": report.last_error,
                "scan": live_tally.summary(),
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
        print("Groww scanner check failed.", file=sys.stderr)
        return 1

    if latest is None:
        print("Scanner produced no cycle; the engine never warmed up.", file=sys.stderr)
        return 1

    if args.export_csv is not None:
        try:
            _export_csv(args.export_csv, rows, overwrite=args.overwrite)
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
        "duplicate_candles": engine.duplicate_candle_count,
        "out_of_order_candles": engine.out_of_order_candle_count,
        "rules": [rule.name for rule in scanner.rules],
        "max_candidates": scanner.config.max_candidates,
        "cost_screen": (
            None
            if policy is None
            else {
                "broker": args.broker,
                "target_notional": _price(policy.target_notional),
                "gross_target_fraction": _fraction(args.gross_target),
                "net_margin_fraction": _fraction(policy.net_margin_fraction),
                "required_gross_fraction_at_the_clip": _fraction(
                    policy.required_gross_fraction
                ),
                "max_atr_multiple": _price(policy.max_atr_multiple),
            }
        ),
        "backfill": backfill_tally.summary(),
        "latest_scan": _scan_summary(latest),
    }
    if live_summary is not None:
        summary["live"] = live_summary
    print(json.dumps(summary, indent=2, sort_keys=True))
    if live_failed:
        print(
            "Live window never opened a stream session; see live.stream_error.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
