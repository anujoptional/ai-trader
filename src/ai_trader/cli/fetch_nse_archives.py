"""Download NSE's daily archives used for F&O-derived features (fno/nse_archive.py).

Resumable and idempotent: files already kept are not requested again, and every new
file is checked before it is written. Every weekday in the range is requested (a 404
is a holiday or a date before the dataset starts) plus the weekend special sessions
of the research calendar, so completeness never rests on an assumed calendar. The
manifest (``<root>/manifest/<dataset>.json``) lists every kept file with its row
count, size and SHA-256, re-checked when it is first listed.

  python -m ai_trader.cli.fetch_nse_archives --start 2020-01-01   # history
  python -m ai_trader.cli.fetch_nse_archives --update             # last 15 days

--update also keeps NSE's FII/DII cash flows of the latest session (NSE serves only
the latest day, so that series starts when the daily update started).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.fno.nse_archive import DATASETS, NseArchive, Outcome, record_fiidii
from ai_trader.news.nse import NseClient

CALENDAR = Path("data/candles_fno_basis_20261011/dataset.json")


def candidate_days(start: date, end: date, calendar: Path | None) -> list[date]:
    """Every weekday from ``start`` to ``end`` plus the calendar's weekend sessions."""
    days = set()
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.add(day)
        day += timedelta(days=1)
    if calendar is not None and calendar.exists():
        for text in json.loads(calendar.read_text("utf-8"))["session_dates"]:
            session = date.fromisoformat(text)
            if start <= session <= end:
                days.add(session)
    return sorted(days)


def update_manifest(archive: NseArchive, dataset: str, days: list[date]) -> dict:
    path = archive.root / "manifest" / f"{dataset}.json"
    manifest = json.loads(path.read_text("utf-8")) if path.exists() else {"files": {}}
    files = manifest["files"]
    fresh = [
        d
        for d in days
        if archive.path(dataset, d).exists()
        and (
            d.isoformat() not in files
            or files[d.isoformat()]["bytes"] != archive.path(dataset, d).stat().st_size
        )
    ]
    files.update(archive.verify(dataset, fresh))
    if path.exists() and not fresh:
        return manifest  # unchanged: keep the file (and its hash) as it is
    manifest["files"] = dict(sorted(files.items()))
    manifest["updated_at"] = datetime.now(INDIA_TIMEZONE).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), "utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nse"))
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument(
        "--update", action="store_true", help="the last 15 days (daily top-up)"
    )
    parser.add_argument(
        "--datasets", default=",".join(DATASETS), help="comma-separated names"
    )
    parser.add_argument("--calendar", type=Path, default=CALENDAR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=0.2)
    args = parser.parse_args(argv)
    today = datetime.now(INDIA_TIMEZONE).date()
    end = args.end or today
    start = args.start or (end - timedelta(days=15) if args.update else None)
    if start is None:
        parser.error("give --start or --update")
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")
    days = candidate_days(start, end, args.calendar)
    archive = NseArchive(args.root, spacing=args.spacing, workers=args.workers)
    counts: Counter[tuple[str, str]] = Counter()
    failed: list[Outcome] = []

    def progress(outcome: Outcome) -> None:
        counts[(outcome.dataset, outcome.status)] += 1
        if outcome.status == "failed":
            failed.append(outcome)
            print(
                f"FAILED {outcome.dataset} {outcome.day}: {outcome.detail}", flush=True
            )
        elif outcome.status == "downloaded" and outcome.detail:
            print(f"{outcome.dataset} {outcome.day}: {outcome.detail}", flush=True)
        done = sum(counts.values())
        if done % 500 == 0:
            print(f"{datetime.now():%H:%M:%S} {done} files checked", flush=True)

    print(
        f"{len(days)} candidate days {days[0]}..{days[-1]}; {', '.join(names)}",
        flush=True,
    )
    archive.sync(names, days, progress)
    for name in names:
        manifest = update_manifest(archive, name, days)
        line = ", ".join(
            f"{status} {counts[(name, status)]}"
            for status in ("downloaded", "kept", "not_published", "failed")
            if counts[(name, status)]
        )
        print(
            f"{name}: {line or 'no session in its range'}; "
            f"manifest {len(manifest['files'])} files",
            flush=True,
        )
    if args.update:
        try:
            kept = record_fiidii(args.root, NseClient(timeout_seconds=20, attempts=2))
            print(f"fiidii: {'kept ' + kept if kept else 'latest already kept'}")
        except Exception as error:
            print(f"fiidii: failed ({type(error).__name__}: {error})", flush=True)
            failed.append(Outcome("fiidii", end, "failed", str(error)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
