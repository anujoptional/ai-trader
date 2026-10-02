"""A local cache of historical one-minute candles, one file per instrument.

Replay needs the same bars twice: once while a sweep is being written, and
again every time the sweep is re-run against a changed stop multiple or fill
model. Refetching them from the broker each time is slow, rate-limited and --
the part that matters -- **not reproducible**, because a vendor may revise a
bar and two runs a week apart would then be comparing strategies against
different tapes without saying so. Section 7.2 requires a result to be
reconstructible; a cache on disk is what makes that true in practice.

**How far back there is anything to ask for.** Groww's backtesting endpoint
serves one-minute candles from 2020; its deprecated predecessor served only the
last three months, and text in this project written against that figure was
wrong about the sample sizes a sweep can reach. No ceiling is enforced here,
because the boundary is a vendor's and moves: refusing a request that would
have partly succeeded is worse than serving what exists. What this module does
instead is tell the truth about what it got -- ``coverage`` reports the span
actually on disk, and a report that cites the *requested* span rather than that
one is misreporting its own sample size.

**Cached ranges are contiguous, never sparse.** Asking for a range that starts
after the cache ends fetches the gap as well, so the file always holds every
bar the broker had between its first and last timestamp. The alternative --
recording a set of disjoint islands -- would need its own index to stay
correct, and a read that silently spanned a hole would produce features
computed across a discontinuity that no downstream assertion could detect. The
cost of the choice is real: a request for last week after caching January
fetches the months between. It is paid once.

**What was asked for and what came back are two facts, kept apart.** Bars end
where the tape ends, which is not where the session ends: NSE's last one-minute
print lands at 15:28 for one symbol and 15:29 for another, and minutes inside
15:15--15:29 are routinely absent for every symbol. Deriving "what has been
fetched" from the bars therefore reads a request to 15:30 as short by a minute
or two, every single time, for the rest of the cache's life. That was once
written off here as a wasted call rather than a wrong answer. It is not: a
store built without a broker -- the mode a published sweep is re-run in -- turns
that phantom gap into a ``CandleStoreError`` and fails a session that a live run
would have traded straight through. A holiday at the start of a range does the
same thing.

So the fetched span is recorded beside the bars, in a one-row sidecar written
whenever a fetch completes, *including* when the fetch returned nothing -- the
holiday and the shut exchange are precisely the cases it exists for. It is not
a second record of the coverage the bars already state, and ``coverage`` still
reads the bars alone: the two cannot disagree because they do not describe the
same thing. The sidecar is consulted in one place, ``load``'s decision about
whether to call the broker, and it is only ever *widened* against a real bar
span, never used in place of one. A sidecar that is missing, stale or narrower
than the truth therefore costs a redundant fetch and can never shorten what is
served.

**Nothing incomplete is ever stored.** Requests are clipped to the close of the
last completed session, so today's half-finished tape cannot be written. Were
it written, the cache would serve that truncated day forever: coverage would
say the range was held, and no later call would go back for the rest of it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from ai_trader.broker import (
    MAX_HISTORICAL_SPAN,
    CandleInterval,
    Instrument,
    ReadOnlyBroker,
)
from ai_trader.clock import INDIA_TIMEZONE, SESSION_CLOSE_TIME
from ai_trader.market import Candle, to_candle

_FIELDS = ("start_time", "open", "high", "low", "close", "volume")
_FETCHED_FIELDS = ("fetched_from", "fetched_to")
_UNSAFE_IN_FILENAME = frozenset('<>:"/\\|?*')


class CandleStoreError(RuntimeError):
    """Raised when the store cannot serve or extend a range."""


def last_completed_session_close(now: datetime) -> datetime:
    """The close of the most recent session that has certainly finished.

    Walks back to the newest weekday whose 15:30 IST close is already past.
    Holidays are deliberately not consulted: this is an upper bound on what may
    be cached, and naming a holiday as complete costs nothing -- the broker
    returns no bars for it, and an empty day inside a contiguous range is
    indistinguishable from a weekend, which the range already tolerates.
    Shipping a holiday calendar to sharpen a bound that does not need
    sharpening would be a calendar to maintain and to be wrong about.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("The current time must be timezone-aware.")

    local = now.astimezone(INDIA_TIMEZONE)
    day = local.date()
    if local < datetime.combine(day, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE):
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return datetime.combine(day, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE)


class CandleStore:
    """Reads candles from disk, fetching and caching whatever is missing.

    The broker is optional, and that is a feature rather than a convenience: a
    store built without one serves whatever is already cached and raises if a
    range would require a fetch. That is the mode a re-run of a published sweep
    should use, because it cannot silently pull a revised bar -- if the cache
    does not hold the range, the run fails instead of quietly measuring
    something else.
    """

    def __init__(
        self,
        root: Path,
        broker: ReadOnlyBroker | None = None,
        *,
        interval: CandleInterval = CandleInterval.ONE_MINUTE,
    ) -> None:
        self._root = Path(root)
        self._broker = broker
        self._interval = interval
        self._max_span = MAX_HISTORICAL_SPAN[interval]

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, instrument: Instrument) -> Path:
        """Where this instrument's candles live.

        The name is the instrument, unescaped. Escaping would have to be
        reversible to stay collision-free -- ``M&M`` and ``M_M`` must not land
        in one file -- so a symbol carrying a character a path cannot hold is
        refused rather than mangled. No NSE equity symbol does.
        """
        for part in (instrument.exchange, instrument.trading_symbol):
            if not part or set(part) & _UNSAFE_IN_FILENAME:
                raise ValueError(f"Instrument cannot be a filename: {instrument!r}")
        return self._root / f"{instrument.exchange}_{instrument.trading_symbol}.csv"

    def fetched_path_for(self, instrument: Instrument) -> Path:
        """Where the record of what was *asked of the broker* lives.

        Derived from ``path_for`` so it inherits that method's refusal to mangle
        a symbol a path cannot hold, and named as a suffix of it so the two
        files sort together and a human clearing one cache directory clears
        both.
        """
        path = self.path_for(instrument)
        return path.with_name(f"{path.stem}.fetched.csv")

    def coverage(self, instrument: Instrument) -> tuple[datetime, datetime] | None:
        """The span actually held: first bar's start to last bar's end.

        ``None`` when nothing is cached. This is what a report should cite as
        the sample it measured, which is routinely shorter than what was asked
        for -- the tape stops printing before the session closes, and a range
        beginning on a holiday starts late -- and a run that quietly reports the
        requested dates instead is overstating itself.

        Deliberately blind to the fetched-span sidecar. That record answers a
        different question, "was the broker already asked about this", and
        letting it widen a *measured* sample would be the one way the two facts
        could come to contradict each other.
        """
        return _span(self._read(instrument))

    def load(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        *,
        now: datetime | None = None,
        refresh: bool = False,
    ) -> tuple[Candle, ...]:
        """Return every cached bar inside ``[start, end]``, fetching any gap.

        Both bounds are inclusive of whole bars and exclusive of partial ones: a
        bar counts when it both begins at or after ``start`` and ends at or
        before ``end``, so a window from one session's open to its close yields
        exactly that session and never half of the next bar.
        """
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("The start time must be timezone-aware.")
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("The end time must be timezone-aware.")
        if start >= end:
            raise ValueError("The start time must be before the end time.")

        horizon = last_completed_session_close(now or datetime.now(INDIA_TIMEZONE))
        end = min(end, horizon)
        if start >= end:
            return ()

        cached = self._read(instrument)
        covered = self._covered(instrument, cached)
        return self._extend(instrument, cached, covered, start, end, refresh=refresh)

    def _covered(
        self, instrument: Instrument, cached: tuple[Candle, ...]
    ) -> tuple[datetime, datetime] | None:
        """Everything known to have been fetched already, bars or not.

        The bar span widened by the recorded fetch span. The widening is one
        directional on purpose: the record may only extend a span real bars
        establish, never stand in for one. So a cache file deleted by hand while
        its sidecar survives refetches from scratch, where trusting the sidecar
        alone would have served nothing and called it complete.
        """
        held = _span(cached)
        if held is None:
            return None
        return _widen(held, self._read_fetched(instrument))

    def freeze(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        destination: Path,
        *,
        now: datetime | None = None,
    ) -> tuple[Candle, ...]:
        """Export a complete requested range without replacing an existing snapshot."""
        frozen = CandleStore(destination, interval=self._interval)
        paths = (frozen.path_for(instrument), frozen.fetched_path_for(instrument))
        if any(path.exists() for path in paths):
            raise CandleStoreError(
                "Frozen candle files already exist; use a new destination"
            )
        if end > last_completed_session_close(now or datetime.now(INDIA_TIMEZONE)):
            raise CandleStoreError("Cannot freeze an unfinished session")
        candles = self.load(instrument, start, end, now=now)
        if not candles:
            raise CandleStoreError("Cannot freeze an empty candle range")
        frozen._write(instrument, candles)
        frozen._write_fetched(instrument, (start, end))
        return candles

    def _extend(
        self,
        instrument: Instrument,
        cached: tuple[Candle, ...],
        covered: tuple[datetime, datetime] | None,
        start: datetime,
        end: datetime,
        *,
        refresh: bool = False,
    ) -> tuple[Candle, ...]:
        missing = _gaps(covered, start, end)
        if refresh:
            missing = (
                (
                    min((start, *(gap[0] for gap in missing))),
                    max((end, *(gap[1] for gap in missing))),
                ),
            )
        if not missing:
            return _slice(cached, start, end)

        fetched: list[Candle] = []
        for gap_start, gap_end in missing:
            fetched.extend(self._fetch(instrument, gap_start, gap_end))
        if fetched:
            cached = _merge(cached, fetched)
            self._write(instrument, cached)
        # Recorded whether or not bars came back, and outside the guard above
        # for that reason: a holiday, a suspended symbol and a session whose
        # last print lands before the close all return short, and they are
        # exactly the cases where deriving this from the bars asks the broker
        # the same question forever.
        self._write_fetched(instrument, _widen(covered, (start, end)))
        return _slice(cached, start, end)

    def _fetch(
        self, instrument: Instrument, start: datetime, end: datetime
    ) -> tuple[Candle, ...]:
        """Pull one range from the broker, in pages it is willing to serve.

        A page ends and the next begins at the same instant rather than a minute
        apart, so a bar sitting on a page boundary is returned whichever way the
        broker reads its bounds -- and if that means it is returned twice, the
        merge keeps one. Advancing a minute past the boundary instead would be
        correct only if the broker's end were inclusive, and wrong by one bar per
        page if it were not: a single-minute hole every thirty days, which is
        exactly the defect this cache exists to make impossible.
        """
        if self._broker is None:
            raise CandleStoreError(
                f"{instrument.exchange}:{instrument.trading_symbol} is not cached for "
                f"{start.isoformat()}..{end.isoformat()} and this store has no broker."
            )

        collected: list[Candle] = []
        cursor = start
        while cursor < end:
            stop = min(cursor + self._max_span, end)
            page = self._broker.get_historical_candles(
                instrument=instrument,
                start=cursor,
                end=stop,
                interval=self._interval,
            )
            collected.extend(to_candle(instrument, source) for source in page)
            cursor = stop
        return tuple(collected)

    def _read(self, instrument: Instrument) -> tuple[Candle, ...]:
        """Load the whole file, in time order.

        Sorting here rather than trusting the file is what lets ``coverage``
        read the ends of the list instead of scanning for a minimum and a
        maximum. The writer already emits sorted rows, so this is about the file
        having been edited or concatenated by hand between runs -- cheap
        insurance against a first row that is not the earliest bar, which would
        otherwise report a coverage span that is simply false.
        """
        path = self.path_for(instrument)
        if not path.exists():
            return ()
        with path.open(newline="", encoding="utf-8") as handle:
            rows = tuple(csv.DictReader(handle))
        candles = (_from_row(instrument, row) for row in rows)
        return tuple(sorted(candles, key=lambda candle: candle.start_time))

    def _write(self, instrument: Instrument, candles: Iterable[Candle]) -> None:
        """Replace the file atomically, so an interrupted write loses nothing.

        A crash midway through a direct rewrite would leave a truncated file
        that still parses -- a shorter range, indistinguishable from a real one,
        which the next run would treat as complete coverage. Writing beside the
        target and renaming means the file is either the old range or the new
        one.
        """
        _write_rows(
            self.path_for(instrument),
            _FIELDS,
            (_to_row(candle) for candle in candles),
        )

    def _read_fetched(self, instrument: Instrument) -> tuple[datetime, datetime] | None:
        """The span previously asked of the broker, or ``None`` if unrecorded.

        A missing file is the ordinary case for a cache written before this
        record existed, and the honest answer for it is "unknown", which costs
        one refetch and then writes the record.
        """
        path = self.fetched_path_for(instrument)
        if not path.exists():
            return None
        with path.open(newline="", encoding="utf-8") as handle:
            rows = tuple(csv.DictReader(handle))
        if not rows:
            return None
        row = rows[0]
        return (
            datetime.fromisoformat(row["fetched_from"]),
            datetime.fromisoformat(row["fetched_to"]),
        )

    def _write_fetched(
        self, instrument: Instrument, span: tuple[datetime, datetime]
    ) -> None:
        """Record the whole span now known to have been fetched.

        One row, overwritten, through the same atomic rename the bars use. The
        span is a single interval rather than a set of them because the cache is
        contiguous by construction: ``load`` fetches every gap between what it
        holds and what was asked for, so the union of the two is an interval and
        recording it as one cannot claim coverage of a hole.
        """
        start, end = span
        _write_rows(
            self.fetched_path_for(instrument),
            _FETCHED_FIELDS,
            (
                {
                    "fetched_from": start.astimezone(INDIA_TIMEZONE).isoformat(),
                    "fetched_to": end.astimezone(INDIA_TIMEZONE).isoformat(),
                },
            ),
        )


def _write_rows(
    path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, str]]
) -> None:
    """Write a CSV where a reader can only ever see a whole one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def _gaps(
    covered: tuple[datetime, datetime] | None, start: datetime, end: datetime
) -> tuple[tuple[datetime, datetime], ...]:
    """The parts of ``[start, end]`` nobody has asked the broker about yet.

    At most two, one either side, because ``covered`` is a single interval --
    which is the contiguity invariant restated as arithmetic.
    """
    if covered is None:
        return ((start, end),)
    covered_from, covered_to = covered
    missing: list[tuple[datetime, datetime]] = []
    if start < covered_from:
        missing.append((start, covered_from))
    if end > covered_to:
        missing.append((covered_to, end))
    return tuple(missing)


def _widen(
    span: tuple[datetime, datetime] | None, other: tuple[datetime, datetime] | None
) -> tuple[datetime, datetime] | None:
    """The smallest interval containing both, ignoring whichever is unknown."""
    if span is None:
        return other
    if other is None:
        return span
    return min(span[0], other[0]), max(span[1], other[1])


def _span(candles: tuple[Candle, ...]) -> tuple[datetime, datetime] | None:
    if not candles:
        return None
    return candles[0].start_time, candles[-1].end_time


def _slice(
    candles: tuple[Candle, ...], start: datetime, end: datetime
) -> tuple[Candle, ...]:
    return tuple(
        candle
        for candle in candles
        if candle.start_time >= start and candle.end_time <= end
    )


def _merge(cached: Iterable[Candle], fetched: Iterable[Candle]) -> tuple[Candle, ...]:
    """Combine two candle sets, keeping one bar per minute, in time order.

    The fetched bar wins a collision. The two should be identical -- the same
    minute from the same vendor -- and when they are not, the fresher read is
    the vendor's current answer, which is what a cache should hold.
    """
    by_minute = {candle.start_time: candle for candle in cached}
    by_minute.update({candle.start_time: candle for candle in fetched})
    return tuple(by_minute[key] for key in sorted(by_minute))


def _to_row(candle: Candle) -> dict[str, str]:
    """Render one candle as stored text, timestamped in IST.

    The offset is written, so the row names an instant either way and a file
    written before this zone was settled reads back as the same minute. IST is
    chosen for the reader: a cache of an Indian session is inspected by eye far
    more often than it is parsed, and 09:15 is the open on sight where 03:45 is
    the open only after arithmetic.
    """
    return {
        "start_time": candle.start_time.astimezone(INDIA_TIMEZONE).isoformat(),
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "volume": "" if candle.volume is None else str(candle.volume),
    }


def _from_row(instrument: Instrument, row: dict[str, str]) -> Candle:
    """Rebuild one candle from its stored row.

    Prices are read as ``Decimal`` from the text that was written, never
    through ``float``: a round trip through binary floating point would return
    a number that is not the price the exchange printed, and every cost
    calculation downstream is exact arithmetic that would then be exact about
    the wrong figure.

    ``end_time`` is recomputed rather than stored. It is always one minute past
    the start, so writing it would be storing a derived value that a hand-edited
    file could contradict.
    """
    start_time = datetime.fromisoformat(row["start_time"])
    volume = row["volume"]
    return Candle(
        instrument=instrument,
        start_time=start_time,
        end_time=start_time + timedelta(minutes=1),
        open=Decimal(row["open"]),
        high=Decimal(row["high"]),
        low=Decimal(row["low"]),
        close=Decimal(row["close"]),
        volume=None if volume == "" else int(volume),
    )


__all__ = ["CandleStore", "CandleStoreError", "last_completed_session_close"]
