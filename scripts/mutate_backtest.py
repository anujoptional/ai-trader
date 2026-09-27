"""Double validation for the report and the CLI: break one promise, expect red.

A passing suite proves the tests agree with the code, not that they would
notice if the code changed. Each entry below is a thing this layer promises,
broken the way it would plausibly break by accident -- a default added for
convenience, a newline left at its platform value, a coverage span taken from
the request instead of the bars. The run is green only when every one of them
makes some test fail.

The two files are mutated together because they only fail together: a reporter
that overstates its sample and a CLI that invents a strategy default are the
same defect seen from two ends, which is a run that quietly measured something
other than what it says it measured.

Throwaway; not part of the package. Run from the repository root:

    .venv/Scripts/python.exe scripts/mutate_backtest.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "src" / "ai_trader" / "replay" / "report.py"
CLI = ROOT / "src" / "ai_trader" / "cli" / "backtest.py"
TESTS = "tests/test_backtest_cli.py"

MUTATIONS: tuple[tuple[Path, str, str, str], ...] = (
    # --- the report states the sample it measured --------------------------
    (
        REPORT,
        "coverage starts at the requested date rather than the first bar",
        "        first = min(candle.start_time for candle in bars)",
        "        first = start",
    ),
    (
        REPORT,
        "coverage ends at the last bar's start, losing the final minute",
        "        last = max(candle.end_time for candle in bars)",
        "        last = max(candle.start_time for candle in bars)",
    ),
    (
        REPORT,
        "an instrument with no bars is passed over in silence",
        '            lines.append(f"  {name:<20} no bars")',
        "            pass",
    ),
    (
        REPORT,
        "the caveats move below the numbers they condition",
        '    lines.extend(_honesty(result))\n    lines.append("")\n    '
        "lines.extend(_edge(result))",
        '    lines.extend(_edge(result))\n    lines.append("")\n    '
        "lines.extend(_honesty(result))",
    ),
    (
        REPORT,
        "a fraction is printed without its percentage",
        'return f"{exact} ({percent}%)"',
        'return f"{exact}"',
    ),
    (
        REPORT,
        "the percentage is the fraction, unmultiplied",
        "percent = (value * 100).quantize(_PERCENT, rounding=ROUND_HALF_EVEN)",
        "percent = value.quantize(_PERCENT, rounding=ROUND_HALF_EVEN)",
    ),
    (
        REPORT,
        "averages over no trades are reported as zeroes",
        '    if not result.trades:\n        lines.append("")',
        '    if False:\n        lines.append("")',
    ),
    (
        REPORT,
        "the sweep row is rounded, so two runs a paisa apart read alike",
        'return format(value, "f")',
        'return str(value.quantize(Decimal("0.01")))',
    ),
    (
        REPORT,
        "a disabled cost screen is recorded as the multiple it did not apply",
        "        if strategy.screen_feasibility",
        "        if True",
    ),
    (
        REPORT,
        "a sweep file with other columns is appended to rather than refused",
        "        if existing != header:",
        "        if False:",
    ),
    (
        REPORT,
        "a row that is not the table is written anyway",
        "    if missing or unexpected:",
        "    if False:",
    ),
    (
        REPORT,
        "a tab inside a value shifts every column after it",
        '            raise ReportError(f"history value for {field!r} contains a '
        'separator")',
        "            pass",
    ),
    (
        REPORT,
        "the sweep table is written in the platform's newline",
        '    with path.open("a", encoding="utf-8", newline="") as handle:',
        '    with path.open("a", encoding="utf-8") as handle:',
    ),
    # --- the CLI states no strategy default --------------------------------
    (
        CLI,
        "the CLI restates a strategy default as a convenience",
        "    given: dict[str, object] = {}",
        "    given: dict[str, object] = {'max_open_positions': 5}",
    ),
    (
        CLI,
        "a stated knob is dropped on the way to the config",
        "        if value is not None:",
        "        if False:",
    ),
    (
        CLI,
        "--trailing is ignored and the fixed stop runs instead",
        "    policy = ChandelierStop if args.trailing else FixedAtrStop",
        "    policy = FixedAtrStop",
    ),
    (
        CLI,
        "--trailing alone invents a multiple instead of taking the shared one",
        "    return policy() if args.stop_atr is None else policy(args.stop_atr)",
        # Deliberately not Decimal(2): that *is* DEFAULT_STOP_ATR_MULTIPLE, so
        # the mutant would be equivalent and its survival would mean nothing.
        '    return policy(Decimal("3")) if args.stop_atr is None '
        "else policy(args.stop_atr)",
    ),
    (
        CLI,
        "--offline authenticates anyway",
        "    if not args.offline:",
        "    if True:",
    ),
    (
        CLI,
        "an unstated fill model is tolerated",
        "    elif any(value is None for value in stated):",
        "    elif False:",
    ),
    (
        CLI,
        "--frictionless silently overrides a stated fill number",
        "        if any(value is not None for value in stated):",
        "        if False:",
    ),
    (
        CLI,
        "a duplicate symbol is replayed twice",
        "        Instrument(exchange=args.exchange, trading_symbol=name) "
        "for name in seen",
        "        Instrument(exchange=args.exchange, trading_symbol=name)\n"
        "        for name in names\n"
        "        if name",
    ),
    (
        CLI,
        "symbols are not folded to upper case, so ALPHA and alpha are two names",
        "            seen[name.upper()] = None",
        "            seen[name] = None",
    ),
    (
        CLI,
        "NaN parses, and a screen comparing against it stops rejecting anything",
        "    if not value.is_finite():",
        "    if False:",
    ),
    (
        CLI,
        "a label carrying a tab is accepted and breaks the sweep table",
        '    if "\\t" in args.label or "\\n" in args.label:',
        "    if False:",
    ),
    (
        CLI,
        "an empty window is reported on rather than failed",
        "    if not candles:",
        "    if False:",
    ),
    (
        CLI,
        "missing configuration exits 1, indistinguishable from a failed run",
        "            return 2",
        "            return 1",
    ),
    (
        CLI,
        "the report is written in the platform's newline",
        '    with path.open("w", encoding="utf-8", newline="") as handle:',
        '    with path.open("w", encoding="utf-8") as handle:',
    ),
    (
        CLI,
        "--no-write writes anyway",
        "    if args.no_write:",
        "    if False:",
    ),
    (
        CLI,
        "--json prints the report instead of the row",
        "    if args.json:",
        "    if False:",
    ),
)


def _read(path: Path) -> str:
    """Read without translating line endings.

    ``Path.read_text``/``write_text`` pass through universal newlines, which on
    Windows turns every LF in the file into CRLF on the way back out. The repo
    stores LF and ``core.safecrlf`` refuses the mismatch, so a script that only
    meant to restore the original would leave it unrestorable.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def main() -> int:
    originals = {path: _read(path) for path in (REPORT, CLI)}
    survivors: list[str] = []
    try:
        for path, label, before, after in MUTATIONS:
            original = originals[path]
            if original.count(before) != 1:
                print(f"SKIP  {label}\n      anchor matched {original.count(before)}x")
                survivors.append(f"{label} (anchor did not match)")
                continue
            _write(path, original.replace(before, after))
            result = subprocess.run(
                [sys.executable, "-m", "pytest", TESTS, "-q", "-x"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            _write(path, original)
            if result.returncode == 0:
                print(f"SURVIVED  {label}")
                survivors.append(label)
            else:
                print(f"caught    {label}")
    finally:
        for path, original in originals.items():
            _write(path, original)

    print()
    caught = len(MUTATIONS) - len(survivors)
    print(f"{caught}/{len(MUTATIONS)} mutations caught")
    for survivor in survivors:
        print(f"  UNCAUGHT: {survivor}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
