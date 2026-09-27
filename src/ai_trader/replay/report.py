"""Turn a replay result into a page someone could be proved wrong by.

Two outputs, for two different readers. ``report_lines`` is the run in prose
and numbers, overwritten each time, for the person deciding whether the last
change helped. ``history_row`` is one line appended to a growing table, for the
sweep -- thirty runs at thirty stop multiples are only comparable if each one
recorded the same columns in the same order.

**Conditions before findings, always.** The header states the configuration,
the universe and the span *actually replayed* before a single result appears,
because Section 7.2's rule is that a result is meaningless unless the conditions
that produced it were recorded. The coverage block matters more than it looks:
the broker publishes roughly three months of one-minute data, so a run asked for
a year measured a quarter of one, and a report citing the requested dates would
overstate its own sample by a factor of four. It is derived from the bars
themselves rather than from the request or from the cache's own bookkeeping --
the same reason ``history.store`` derives coverage from the data instead of a
sidecar, which is that a second record of the same fact can disagree with it.

**Honesty before edge.** The ambiguous-exit rate, the entries the tape
swallowed and the positions still open at the end come *above* the expectancy,
not in a footnote under it. Every number in the edge section is conditional on
them: at a two per cent ambiguous rate the fill assumption barely mattered, and
at forty per cent the expectancy is mostly a report of what the caller chose to
assume. A reader who stops at the first number should have hit the caveat
first.

**Fractions are printed with their percentages.** A fraction and a percentage
of the same quantity differ by a hundred, and printing one alone is how that
error survives a review -- the same reasoning ``cli/check_scanner.py`` gives,
applied here because this is the file someone will quote from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

from ai_trader.broker import Instrument
from ai_trader.features import FEATURE_CONTEXT
from ai_trader.market import INDIA_TIMEZONE, Candle
from ai_trader.replay.models import ReplayResult
from ai_trader.strategy import StrategyConfig

_MONEY = Decimal("0.01")
_PERCENT = Decimal("0.0001")
_FRACTION = Decimal("0.00000001")
_TIMESTAMP = "%Y-%m-%d %H:%M:%S %Z"

HISTORY_FIELDS: tuple[str, ...] = (
    "generated_at",
    "label",
    "symbols",
    "sessions",
    "candles",
    "exit_kind",
    "stop_multiple",
    "cost_screen",
    "clip",
    "gross_target",
    "max_open_positions",
    "latency_seconds",
    "half_spread_fraction",
    "slippage_fraction",
    "ambiguous_as",
    "round_trips",
    "round_trips_per_session",
    "net_expectancy_rupees",
    "net_rupees",
    "gross_rupees",
    "total_costs",
    "turnover",
    "net_hit_rate",
    "gross_hit_rate",
    "max_drawdown_rupees",
    "mean_favourable_fraction",
    "mean_adverse_fraction",
    "ambiguous_exit_rate",
    "unfilled_entries",
    "unwound_entries",
    "open_at_end",
)
"""The sweep table's columns, in the order they are written.

Fixed rather than derived from whatever the run happened to produce. A sweep is
read by sorting one column against another, and a table whose columns moved
between runs cannot be sorted at all -- which is why ``append_history`` refuses
a file whose header disagrees with this tuple instead of appending under it.
"""


class ReportError(RuntimeError):
    """Raised when a report cannot be written without losing information."""


def report_lines(
    result: ReplayResult,
    strategy: StrategyConfig,
    *,
    requested: tuple[datetime, datetime],
    loaded: Mapping[Instrument, Sequence[Candle]],
    generated_at: datetime,
    label: str = "",
) -> tuple[str, ...]:
    """The whole run as text: conditions, honesty, edge, attribution, risk.

    ``loaded`` is the bars each instrument actually contributed, which is what
    the coverage block reports. Passing the candles rather than a precomputed
    span is deliberate: a span handed in alongside them could be wrong about
    them, and this is the section of the report whose only job is to be right
    about the sample.
    """
    lines: list[str] = []
    lines.extend(_conditions(result, strategy, requested, loaded, generated_at, label))
    lines.append("")
    lines.extend(_honesty(result))
    lines.append("")
    lines.extend(_edge(result))
    lines.append("")
    lines.extend(_attribution(result))
    lines.append("")
    lines.extend(_risk(result))
    return tuple(lines)


def history_row(
    result: ReplayResult,
    strategy: StrategyConfig,
    *,
    generated_at: datetime,
    label: str = "",
) -> dict[str, str]:
    """One sweep line: the knobs that were turned and what came out.

    Values are exact rather than rounded. The text report is for reading and
    rounds accordingly; this is for sorting and differencing, and a rounded
    expectancy makes two runs that differed by a paisa look identical.
    """
    # ``ExitPolicy`` does not require a multiple -- a future policy may be
    # parameterised some other way -- so it is read defensively and left blank
    # rather than invented.
    multiple = getattr(strategy.exit_policy, "multiple", None)
    fill = result.fill
    return {
        "generated_at": generated_at.astimezone(INDIA_TIMEZONE).isoformat(),
        "label": label,
        "symbols": str(len(result.universe)),
        "sessions": str(len(result.sessions)),
        "candles": str(result.candles_replayed),
        "exit_kind": type(strategy.exit_policy).__name__,
        "stop_multiple": "" if multiple is None else _exact(multiple),
        "cost_screen": _exact(strategy.max_atr_multiple)
        if strategy.screen_feasibility
        else "off",
        "clip": _exact(strategy.target_notional),
        "gross_target": _exact(strategy.gross_target_fraction),
        "max_open_positions": str(strategy.max_open_positions),
        "latency_seconds": _exact(fill.latency_seconds),
        "half_spread_fraction": _exact(fill.half_spread_fraction),
        "slippage_fraction": _exact(fill.slippage_fraction),
        "ambiguous_as": str(fill.resolve_ambiguous_bar_as),
        "round_trips": str(result.round_trips),
        "round_trips_per_session": _exact(result.round_trips_per_session),
        "net_expectancy_rupees": _exact(result.net_expectancy_rupees),
        "net_rupees": _exact(result.net_rupees),
        "gross_rupees": _exact(result.gross_rupees),
        "total_costs": _exact(result.total_costs),
        "turnover": _exact(result.turnover),
        "net_hit_rate": _exact(result.net_hit_rate),
        "gross_hit_rate": _exact(result.gross_hit_rate),
        "max_drawdown_rupees": _exact(result.max_drawdown_rupees),
        "mean_favourable_fraction": _exact(result.mean_favourable_fraction),
        "mean_adverse_fraction": _exact(result.mean_adverse_fraction),
        "ambiguous_exit_rate": _exact(result.ambiguous_exit_rate),
        "unfilled_entries": str(result.unfilled_entries),
        "unwound_entries": str(result.unwound_entries),
        "open_at_end": str(result.open_at_end),
    }


def append_history(path: Path, row: Mapping[str, str]) -> None:
    """Append one run to the sweep table, writing the header if it is new.

    An existing file whose header is not ``HISTORY_FIELDS`` is refused rather
    than appended to. Appending under a stale header would produce a table whose
    columns mean different things on different lines, and nothing downstream --
    a spreadsheet, a sort, a reader -- could detect that. Writing the file by
    hand is a recoverable mistake; a silently misaligned sweep is not.
    """
    missing = set(HISTORY_FIELDS) - set(row)
    unexpected = set(row) - set(HISTORY_FIELDS)
    if missing or unexpected:
        raise ReportError(
            f"history row does not match the table: missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )
    for field, value in row.items():
        if "\t" in value or "\n" in value:
            raise ReportError(f"history value for {field!r} contains a separator")

    header = "\t".join(HISTORY_FIELDS)
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = handle.readline().rstrip("\r\n")
        if existing != header:
            raise ReportError(
                f"{path} was written with different columns; move it aside rather "
                "than appending a row that would not line up."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(header + "\n")

    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(row[field] for field in HISTORY_FIELDS) + "\n")


def _conditions(
    result: ReplayResult,
    strategy: StrategyConfig,
    requested: tuple[datetime, datetime],
    loaded: Mapping[Instrument, Sequence[Candle]],
    generated_at: datetime,
    label: str,
) -> list[str]:
    start, end = requested
    lines = ["BACKTEST", "=" * 72]
    if label:
        lines.append(f"label                {label}")
    lines.append(
        f"generated            "
        f"{generated_at.astimezone(INDIA_TIMEZONE).strftime(_TIMESTAMP)}"
    )
    lines.append("")
    lines.append("Strategy")
    lines.extend(f"  {line}" for line in strategy.describe())
    lines.append("")
    lines.append("Fill assumptions (stated, not measured)")
    fill = result.fill
    lines.append(f"  latency              {fill.latency_seconds}s")
    lines.append(f"  half spread          {_fraction(fill.half_spread_fraction)}")
    lines.append(f"  slippage             {_fraction(fill.slippage_fraction)}")
    lines.append(f"  ambiguous bar        resolved as {fill.resolve_ambiguous_bar_as}")
    lines.append("")
    lines.append("Sample")
    lines.append(f"  requested            {_moment(start)} .. {_moment(end)}")
    lines.append(f"  sessions             {len(result.sessions)}")
    lines.append(f"  candles replayed     {result.candles_replayed:,}")
    lines.append(f"  decision cycles      {result.cycles:,}")
    lines.append(f"  universe             {len(result.universe)} instruments")
    lines.append("")
    lines.append("Bars actually replayed, per instrument (not what was asked for)")
    for instrument in result.universe:
        bars = loaded.get(instrument, ())
        name = f"{instrument.exchange}:{instrument.trading_symbol}"
        if not bars:
            lines.append(f"  {name:<20} no bars")
            continue
        first = min(candle.start_time for candle in bars)
        last = max(candle.end_time for candle in bars)
        lines.append(
            f"  {name:<20} {_moment(first)} .. {_moment(last)}  {len(bars):>7,} bars"
        )
    return lines


def _honesty(result: ReplayResult) -> list[str]:
    """What conditions every figure below. Deliberately above them."""
    lines = ["Read these first", "-" * 72]
    lines.append(
        f"  ambiguous exits      {result.ambiguous_exits} of {result.round_trips} "
        f"({_fraction(result.ambiguous_exit_rate)})"
    )
    lines.append(
        "                       exits the tape could not order, assigned by the "
        "fill model"
    )
    lines.append(
        f"  unfilled entries     {result.unfilled_entries} (never reached a bar)"
    )
    lines.append(
        f"  unwound entries      {result.unwound_entries} (filled on the last bar, "
        "dropped unjudged)"
    )
    lines.append(f"  open at end          {result.open_at_end}")
    lines.append("")
    lines.append("  The tape is gapless and a live one is not; fills are interpolated")
    lines.append("  open-to-close and ignore the bar's own high and low. Both flatter")
    lines.append("  the result, so every figure below is an upper bound.")
    return lines


def _edge(result: ReplayResult) -> list[str]:
    lines = ["Edge", "-" * 72]
    lines.append(f"  round trips          {result.round_trips}")
    if not result.trades:
        lines.append("")
        lines.append("  No completed round trips. Every average below would be zero")
        lines.append("  by construction, so none is reported.")
        return lines
    lines.append(
        f"  per session          {_quantize(result.round_trips_per_session, _MONEY)}"
    )
    lines.append(f"  net expectancy       Rs {_money(result.net_expectancy_rupees)}")
    lines.append(f"  net total            Rs {_money(result.net_rupees)}")
    lines.append(f"  gross total          Rs {_money(result.gross_rupees)}")
    lines.append(f"  costs paid           Rs {_money(result.total_costs)}")
    lines.append(f"  turnover             Rs {_money(result.turnover)}")
    lines.append(f"  net hit rate         {_fraction(result.net_hit_rate)}")
    lines.append(f"  gross hit rate       {_fraction(result.gross_hit_rate)}")
    lines.append(
        "                       the gap is trades that were right and still lost"
    )
    return lines


def _attribution(result: ReplayResult) -> list[str]:
    lines = ["Attribution", "-" * 72]
    if not result.trades:
        lines.append("  nothing to attribute")
        return lines

    lines.append("  exits")
    for reason, count in sorted(result.exit_reasons.items()):
        lines.append(f"    {str(reason):<18} {count}")

    lines.append("  by rule (overlapping: a trade counts under every rule that fired)")
    by_rule = result.round_trips_by_rule
    net_by_rule = result.net_rupees_by_rule
    for rule in sorted(by_rule):
        lines.append(
            f"    {rule:<18} {by_rule[rule]:>4} trades  Rs {_money(net_by_rule[rule])}"
        )

    lines.append("  by IST hour of entry")
    by_hour = result.round_trips_by_hour
    net_by_hour = result.net_rupees_by_hour
    for hour in sorted(by_hour):
        lines.append(
            f"    {hour:02d}:00{'':<13} {by_hour[hour]:>4} trades  "
            f"Rs {_money(net_by_hour[hour])}"
        )
    return lines


def _risk(result: ReplayResult) -> list[str]:
    lines = ["Risk and what was declined", "-" * 72]
    lines.append(f"  max drawdown         Rs {_money(result.max_drawdown_rupees)}")
    lines.append("                       realised curve; open positions not marked")
    lines.append(f"  mean favourable      {_fraction(result.mean_favourable_fraction)}")
    lines.append(f"  mean adverse         {_fraction(result.mean_adverse_fraction)}")
    lines.append(
        "                       read against the stop: well inside it means the "
        "stop is not binding"
    )
    lines.append(f"  candidates seen      {result.candidates_seen}")
    lines.append(f"  declined, book full  {result.declined_book_full}")
    lines.append(f"  declined, no ATR     {result.declined_no_volatility}")
    if result.suppressed:
        lines.append("  suppressed by the scanner")
        for reason, count in sorted(
            result.suppressed.items(), key=lambda item: str(item[0])
        ):
            lines.append(f"    {str(reason):<18} {count}")
    else:
        lines.append("  suppressed           none")
    return lines


def _moment(value: datetime) -> str:
    return value.astimezone(INDIA_TIMEZONE).strftime("%Y-%m-%d %H:%M")


def _money(value: Decimal) -> str:
    return f"{_quantize(value, _MONEY):,}"


def _fraction(value: Decimal) -> str:
    """A fraction and its percentage together, never one alone."""
    with localcontext(FEATURE_CONTEXT):
        percent = (value * 100).quantize(_PERCENT, rounding=ROUND_HALF_EVEN)
        exact = value.quantize(_FRACTION, rounding=ROUND_HALF_EVEN)
    return f"{exact} ({percent}%)"


def _quantize(value: Decimal, exponent: Decimal) -> Decimal:
    with localcontext(FEATURE_CONTEXT):
        return value.quantize(exponent, rounding=ROUND_HALF_EVEN)


def _exact(value: Decimal) -> str:
    return format(value, "f")


__all__ = [
    "HISTORY_FIELDS",
    "ReportError",
    "append_history",
    "history_row",
    "report_lines",
]
