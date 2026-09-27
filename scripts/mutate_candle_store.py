"""Double validation for the candle store: mutate an invariant, expect a failure.

A passing suite proves the tests agree with the code, not that they would
notice if the code changed. Each entry below breaks one thing the store
promises, in the way it would plausibly break by accident, and the run is only
green when every one of them makes some test fail.

Throwaway; not part of the package. Run from the repository root:

    .venv/Scripts/python.exe scripts/mutate_candle_store.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "ai_trader" / "history" / "store.py"

MUTATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "forward extend starts at the request, leaving a hole in the cache",
        "            missing.append((covered_to, end))",
        "            missing.append((start, end))",
    ),
    (
        "no clip to the last completed session; a partial day gets cached",
        "        end = min(end, horizon)",
        "        end = end",
    ),
    (
        "no paging; one oversized request the broker would refuse",
        "            stop = min(cursor + self._max_span, end)",
        "            stop = end",
    ),
    (
        "prices read through float",
        '        open=Decimal(row["open"]),',
        '        open=Decimal(str(float(row["open"]))),',
    ),
    (
        "slice admits a bar that starts inside the window but ends past it",
        "        if candle.start_time >= start and candle.end_time <= end",
        "        if candle.start_time >= start and candle.start_time <= end",
    ),
    (
        "coverage reports the last bar's start rather than the range it holds",
        "    return candles[0].start_time, candles[-1].end_time",
        "    return candles[0].start_time, candles[-1].start_time",
    ),
    (
        "rows trusted in file order rather than sorted",
        "        return tuple(sorted(candles, key=lambda candle: candle.start_time))",
        "        return tuple(candles)",
    ),
    (
        "an unusable symbol is mangled into a filename instead of refused",
        '                raise ValueError(f"Instrument cannot be a filename: '
        '{instrument!r}")',
        "                pass",
    ),
    (
        "a missing broker is tolerated, serving a short tape as if complete",
        "            raise CandleStoreError(",
        "            return () or CandleStoreError(",
    ),
    (
        "a page boundary skips the bar sitting on it",
        "            cursor = stop",
        "            cursor = stop + timedelta(minutes=1)",
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
    original = _read(TARGET)
    survivors: list[str] = []
    try:
        for label, before, after in MUTATIONS:
            if original.count(before) != 1:
                print(f"SKIP  {label}\n      anchor matched {original.count(before)}x")
                survivors.append(f"{label} (anchor did not match)")
                continue
            _write(TARGET, original.replace(before, after))
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_candle_store.py", "-q"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"SURVIVED  {label}")
                survivors.append(label)
            else:
                print(f"caught    {label}")
    finally:
        _write(TARGET, original)

    print()
    caught = len(MUTATIONS) - len(survivors)
    print(f"{caught}/{len(MUTATIONS)} mutations caught")
    for survivor in survivors:
        print(f"  UNCAUGHT: {survivor}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
