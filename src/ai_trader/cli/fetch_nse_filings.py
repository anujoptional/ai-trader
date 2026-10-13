"""Download NSE announcements, corporate actions and board meetings into SQLite.

Resumable: each (kind, day) is logged after it is stored, so reruns skip finished
days. Announcements are fetched one day per request (hundreds per day); actions
and board meetings in 31-day ranges. NSE's pace is about 1 request/second.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from ai_trader.news.nse import NseClient, NseError
from ai_trader.news.store import NewsStore

KINDS = ("announcements", "actions", "meetings")
CHUNK_DAYS = {"announcements": 1, "actions": 31, "meetings": 31}


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


def fetch(
    client: NseClient,
    store: NewsStore,
    start: date,
    end: date,
    kinds: tuple[str, ...] = KINDS,
    *,
    refetch_after: date | None = None,
    progress=print,
) -> dict[str, int]:
    """Fetch every missing day; days on/after ``refetch_after`` are always redone."""
    added = dict.fromkeys(kinds, 0)
    failures: list[str] = []
    for kind in kinds:
        done = store.fetched_days(kind)
        missing = [
            day
            for day in days(start, end)
            if day not in done or (refetch_after is not None and day >= refetch_after)
        ]
        size = CHUNK_DAYS[kind]
        chunks: list[list[date]] = []
        for day in missing:
            if (
                chunks
                and (day - chunks[-1][0]).days < size
                and (day - chunks[-1][-1]).days == 1
            ):
                chunks[-1].append(day)
            else:
                chunks.append([day])
        for number, chunk in enumerate(chunks, start=1):
            first, last = chunk[0], chunk[-1]
            try:
                if kind == "announcements":
                    items = client.announcements(first, last)
                    added[kind] += store.add_announcements(items)
                elif kind == "actions":
                    items = client.corporate_actions(first, last)
                    added[kind] += store.add_actions(items)
                else:
                    items = client.board_meetings(first, last)
                    added[kind] += store.add_meetings(items)
            except NseError as error:
                failures.append(f"{kind} {first}..{last}: {error}")
                progress(f"failed {kind} {first}..{last}: {error}")
                continue
            for day in chunk:
                store.mark_fetched(kind, day, len(items) if day == first else 0)
            if number % 50 == 0 or number == len(chunks):
                progress(f"{kind}: {number}/{len(chunks)} requests, through {last}")
    if failures:
        progress(f"failed {len(failures)}: {failures[:5]}")
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/news/nse_filings.sqlite"))
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--refetch-after", type=date.fromisoformat)
    args = parser.parse_args(argv)
    kinds = tuple(k for k in args.kinds.split(",") if k)
    if not set(kinds) <= set(KINDS) or args.start > args.end:
        parser.error("unknown kind or unordered dates")
    store = NewsStore(args.db)
    try:
        added = fetch(
            NseClient(),
            store,
            args.start,
            args.end,
            kinds,
            refetch_after=args.refetch_after,
        )
    finally:
        store.close()
    print(f"added {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
