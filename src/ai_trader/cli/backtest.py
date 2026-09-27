"""Run the strategy over cached history and write down what happened.

    .venv/Scripts/python.exe -m ai_trader.cli.backtest \\
        --symbols RELIANCE,HDFCBANK,INFY \\
        --start 2026-07-01 --end 2026-09-25 \\
        --latency-seconds 1 --half-spread 0.0002 --slippage 0.0001

Four steps and no fifth: read a universe and a window, pull the bars through
``CandleStore``, hand them to ``ReplayEngine``, print what ``replay.report``
makes of the result. Everything interesting happens in those layers.

**This file states no strategy default.** Every strategy option parses to
``None`` and is forwarded only when the caller actually passed it, so
``StrategyConfig`` remains the one place any of it is written down. A default
repeated here would be a second definition of the strategy wearing the name of
a convenience, and the failure mode is the one Section 7.1 exists to prevent:
replay measuring a configuration live trading would never run. The rule is
worth more than the ergonomics it costs -- ``--help`` cannot show you the
defaults, so it points at the module that holds them.

**The fill model has no default at all.** The caller must pass either
``--frictionless`` or all three of ``--latency-seconds``, ``--half-spread`` and
``--slippage``. ``replay.models`` refuses to default those three fields because
a silent zero is the most flattering thing a backtest can assume and the one a
reader is least likely to notice; a CLI that invented them would hand that
assumption back through the front door.

**``--offline`` never loads credentials.** Re-running a published sweep is the
mode that has to be reproducible, so it reads only what is cached and fails
loudly on a gap rather than fetching bars the original run never saw.

Exit codes: 0 success, 1 the run could not be completed, 2 the configuration or
the environment was wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ai_trader.broker import Instrument
from ai_trader.broker.groww import GrowwBroker, GrowwBrokerError
from ai_trader.clock import INDIA_TIMEZONE, SESSION_CLOSE_TIME, SESSION_OPEN_TIME
from ai_trader.config import ConfigurationError, load_groww_settings
from ai_trader.history import CandleStore, CandleStoreError
from ai_trader.market import Candle
from ai_trader.replay import (
    FRICTIONLESS,
    ExitReason,
    FillModel,
    ReplayConfig,
    ReplayEngine,
)
from ai_trader.replay.report import (
    ReportError,
    append_history,
    history_row,
    report_lines,
)
from ai_trader.strategy import ChandelierStop, ExitPolicy, FixedAtrStop, StrategyConfig

_DEFAULT_CACHE = Path("data/candles")
_DEFAULT_REPORT = Path("backtest_report.txt")
_DEFAULT_HISTORY = Path("backtest_history.tsv")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    broker = None
    if not args.offline:
        try:
            settings = load_groww_settings()
        except ConfigurationError as error:
            print(str(error), file=sys.stderr)
            return 2
        try:
            broker = GrowwBroker.authenticate(settings)
        except GrowwBrokerError:
            print("Groww authentication failed.", file=sys.stderr)
            return 1

    try:
        strategy = _build_strategy(args)
        fill = _build_fill(args)
    except (ArithmeticError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    universe = _universe(args)
    start = datetime.combine(args.start, SESSION_OPEN_TIME, tzinfo=INDIA_TIMEZONE)
    end = datetime.combine(args.end, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE)

    store = CandleStore(args.cache, broker)
    loaded: dict[Instrument, tuple[Candle, ...]] = {}
    try:
        for instrument in universe:
            loaded[instrument] = store.load(instrument, start, end)
    except CandleStoreError as error:
        print(str(error), file=sys.stderr)
        return 1
    except GrowwBrokerError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (ArithmeticError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    candles = [candle for bars in loaded.values() for candle in bars]
    if not candles:
        print(
            "No candles for that window. An offline run needs the range already "
            "cached, and the broker publishes only about three months of "
            "one-minute data.",
            file=sys.stderr,
        )
        return 1

    # Concatenated per instrument and deliberately not sorted here: the engine
    # folds every instrument's bar for a minute together before it scans, and
    # doing it twice would only be a chance to do it differently.
    engine = ReplayEngine(ReplayConfig(universe=universe, fill=fill, strategy=strategy))
    try:
        result = engine.run(candles)
    except (ArithmeticError, ValueError) as error:
        print(f"Replay failed: {error}", file=sys.stderr)
        return 1

    generated_at = datetime.now(INDIA_TIMEZONE)
    lines = report_lines(
        result,
        strategy,
        requested=(start, end),
        loaded=loaded,
        generated_at=generated_at,
        label=args.label,
    )
    row = history_row(result, strategy, generated_at=generated_at, label=args.label)

    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        print("\n".join(lines))

    if args.no_write:
        return 0
    try:
        _write_report(args.report, lines)
        append_history(args.history, row)
    except ReportError as error:
        print(str(error), file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Could not write the report: {error}", file=sys.stderr)
        return 1
    return 0


def _write_report(path: Path, lines: Sequence[str]) -> None:
    """Overwrite the report, in LF, whatever platform wrote it.

    ``newline=""`` rather than the default, because the default translates every
    LF to CRLF on Windows and the same run would then produce a file that
    differs from its Linux twin on every line -- a diff of two reports should
    show what the strategy did, not which machine ran it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")


def _universe(args: argparse.Namespace) -> tuple[Instrument, ...]:
    """The instruments to replay, in the order given, each one exactly once.

    Duplicates are dropped rather than replayed. The same symbol twice would put
    two identically-priced names in front of the scanner every minute, and a
    book that took both would be carrying double the intended exposure to one
    stock while its own position count said otherwise.
    """
    if args.symbols_file is not None:
        text = args.symbols_file.read_text(encoding="utf-8")
        names = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    else:
        names = [name.strip() for name in args.symbols.split(",")]

    seen: dict[str, None] = {}
    for name in names:
        if name:
            seen[name.upper()] = None
    return tuple(
        Instrument(exchange=args.exchange, trading_symbol=name) for name in seen
    )


def _build_fill(args: argparse.Namespace) -> FillModel:
    if args.frictionless:
        return FRICTIONLESS
    ambiguous = (
        {}
        if args.ambiguous_as is None
        else {"resolve_ambiguous_bar_as": ExitReason(args.ambiguous_as)}
    )
    return FillModel(
        latency_seconds=args.latency_seconds,
        half_spread_fraction=args.half_spread,
        slippage_fraction=args.slippage,
        **ambiguous,
    )


def _build_strategy(args: argparse.Namespace) -> StrategyConfig:
    """Forward only what was asked for, so no default is restated here."""
    given: dict[str, object] = {}
    for option, field in (
        ("clip", "target_notional"),
        ("gross_target", "gross_target_fraction"),
        ("max_candidates", "max_candidates"),
        ("max_open_positions", "max_open_positions"),
        ("max_atr_multiple", "max_atr_multiple"),
        ("cooldown_minutes", "cooldown_minutes"),
        ("square_off_minutes", "square_off_minutes_since_open"),
    ):
        value = getattr(args, option)
        if value is not None:
            given[field] = value

    if args.no_screen:
        given["screen_feasibility"] = False
    exit_policy = _build_exit_policy(args)
    if exit_policy is not None:
        given["exit_policy"] = exit_policy
    return StrategyConfig(**given)  # type: ignore[arg-type]


def _build_exit_policy(args: argparse.Namespace) -> ExitPolicy | None:
    """``None`` when neither stop option was given, so the shared default holds.

    Constructing the chosen class without a multiple rather than passing one
    keeps the number itself in ``strategy.exits``: ``--trailing`` alone trails
    at whatever the fixed stop would have sat at, which is what makes the two
    comparable in a sweep.
    """
    if not args.trailing and args.stop_atr is None:
        return None
    policy = ChandelierStop if args.trailing else FixedAtrStop
    return policy() if args.stop_atr is None else policy(args.stop_atr)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ai_trader.cli.backtest",
        description="Replay the strategy over cached one-minute history.",
        epilog=(
            "Strategy options left unset take the shared defaults in "
            "ai_trader.strategy.config, which is where they are written down. "
            "The fill model has no default and must be stated."
        ),
    )

    universe = parser.add_argument_group("universe")
    universe.add_argument(
        "--symbols", help="Comma-separated trading symbols, e.g. RELIANCE,INFY."
    )
    universe.add_argument(
        "--symbols-file",
        type=Path,
        help="One symbol per line; blank lines and #comments ignored.",
    )
    universe.add_argument(
        "--exchange", default="NSE", help="Exchange for every symbol (default NSE)."
    )

    window = parser.add_argument_group("window")
    window.add_argument(
        "--start", required=True, type=_date, help="First session, YYYY-MM-DD."
    )
    window.add_argument(
        "--end", required=True, type=_date, help="Last session, YYYY-MM-DD, inclusive."
    )
    window.add_argument(
        "--cache",
        type=Path,
        default=_DEFAULT_CACHE,
        help=f"Candle cache directory (default {_DEFAULT_CACHE}).",
    )
    window.add_argument(
        "--offline",
        action="store_true",
        help="Use only cached bars; never authenticate or fetch.",
    )

    fills = parser.add_argument_group(
        "fill model (required: --frictionless, or all three of the rest)"
    )
    fills.add_argument(
        "--frictionless",
        action="store_true",
        help="Assume no latency, spread or slippage. Not a trading assumption.",
    )
    fills.add_argument("--latency-seconds", type=_decimal, help="Delay before a fill.")
    fills.add_argument(
        "--half-spread", type=_decimal, help="Half the bid-ask spread, as a fraction."
    )
    fills.add_argument(
        "--slippage", type=_decimal, help="Slippage beyond the spread, as a fraction."
    )
    fills.add_argument(
        "--ambiguous-as",
        choices=[str(reason) for reason in (ExitReason.STOP, ExitReason.TARGET)],
        help="How to resolve a bar that touched both target and stop.",
    )

    knobs = parser.add_argument_group("strategy (unset means the shared default)")
    knobs.add_argument("--clip", type=_decimal, help="Rupee notional per position.")
    knobs.add_argument(
        "--gross-target", type=_decimal, help="Gross move targeted, as a fraction."
    )
    knobs.add_argument("--max-candidates", type=int, help="Candidates per scan.")
    knobs.add_argument("--max-open-positions", type=int, help="Book size.")
    knobs.add_argument(
        "--stop-atr", type=_decimal, help="Stop distance in one-minute ATRs."
    )
    knobs.add_argument(
        "--trailing", action="store_true", help="Trail the stop instead of fixing it."
    )
    knobs.add_argument(
        "--no-screen", action="store_true", help="Disable the cost feasibility screen."
    )
    knobs.add_argument(
        "--max-atr-multiple", type=_decimal, help="Cost screen's ATR multiple."
    )
    knobs.add_argument(
        "--cooldown-minutes", type=_decimal, help="Wait after exiting a name."
    )
    knobs.add_argument(
        "--square-off-minutes", type=_decimal, help="Square-off, minutes after open."
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--label", default="", help="Names this run in the sweep table."
    )
    output.add_argument(
        "--report",
        type=Path,
        default=_DEFAULT_REPORT,
        help=f"Overwritten each run (default {_DEFAULT_REPORT}).",
    )
    output.add_argument(
        "--history",
        type=Path,
        default=_DEFAULT_HISTORY,
        help=f"Appended each run (default {_DEFAULT_HISTORY}).",
    )
    output.add_argument(
        "--no-write", action="store_true", help="Print only; touch no files."
    )
    output.add_argument(
        "--json", action="store_true", help="Print the sweep row instead of the report."
    )

    args = parser.parse_args(argv)
    _validate(parser, args)
    return args


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (args.symbols is None) == (args.symbols_file is None):
        parser.error("Pass exactly one of --symbols or --symbols-file.")
    if args.symbols_file is not None and not args.symbols_file.is_file():
        parser.error(f"{args.symbols_file} is not a file.")
    if not args.exchange.strip():
        parser.error("--exchange cannot be blank.")
    if args.start > args.end:
        parser.error("--start must not be after --end.")

    stated = (args.latency_seconds, args.half_spread, args.slippage)
    if args.frictionless:
        if any(value is not None for value in stated):
            parser.error("--frictionless already fixes the fill model at zero.")
        if args.ambiguous_as is not None:
            # Without friction the two legs price identically, so the choice
            # would change the label on the exit and nothing else about it.
            parser.error("--ambiguous-as has no effect under --frictionless.")
    elif any(value is None for value in stated):
        parser.error(
            "State the fill model: --frictionless, or all of --latency-seconds, "
            "--half-spread and --slippage. There is no default, because an "
            "unstated assumption here silently flatters the result."
        )

    if "\t" in args.label or "\n" in args.label:
        parser.error("--label cannot contain a tab or a newline.")
    if args.no_screen and args.max_atr_multiple is not None:
        parser.error("--max-atr-multiple has no effect with --no-screen.")


def _date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from None


def _decimal(text: str) -> Decimal:
    """Parse exactly, and refuse the values that would poison the arithmetic.

    ``Decimal`` accepts ``NaN`` and ``Infinity``, and either would propagate
    silently through every cost calculation downstream -- comparisons against a
    NaN are simply false, so a screen would stop rejecting anything rather than
    raising.
    """
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not value.is_finite():
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
