"""Tests for the on-disk historical candle cache.

The store's whole job is to make a replay reproducible, so most of what is
asserted here is about *not* doing something: not refetching what is held, not
writing a partial session, not leaving a hole in the middle of a cached range,
and not passing prices through anything that could round them.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ai_trader.broker import (
    MAX_HISTORICAL_SPAN,
    CandleInterval,
    Instrument,
    OHLCVCandle,
)
from ai_trader.clock import (
    INDIA_TIMEZONE,
    SESSION_CLOSE_TIME,
    SESSION_MINUTES,
    SESSION_OPEN_TIME,
)
from ai_trader.history import (
    CandleStore,
    CandleStoreError,
    last_completed_session_close,
)
from ai_trader.history.store import _FIELDS

_INSTRUMENT = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_ONE_MINUTE = timedelta(minutes=1)
_MAX_SPAN = MAX_HISTORICAL_SPAN[CandleInterval.ONE_MINUTE]

# A range the chunker must split, expressed relative to the published limit so
# that raising the limit moves these tests with it rather than leaving them
# asserting something the chunker no longer does.
_PAGED_END = date(2026, 9, 24)
_PAGED_START = _PAGED_END - _MAX_SPAN - timedelta(days=3)


def _ist(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=INDIA_TIMEZONE)


def _weekdays(first: date, last: date) -> tuple[date, ...]:
    """Every weekday in ``[first, last]``, which is what a tape has bars for."""
    return tuple(
        first + timedelta(days=offset)
        for offset in range((last - first).days + 1)
        if (first + timedelta(days=offset)).weekday() < 5
    )


def _session_minutes(day: date) -> Iterator[datetime]:
    """Every one-minute bar start in one session, 09:15 through 15:29 IST."""
    cursor = datetime.combine(day, SESSION_OPEN_TIME, tzinfo=INDIA_TIMEZONE)
    close = datetime.combine(day, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE)
    while cursor < close:
        yield cursor
        cursor += _ONE_MINUTE


class FakeBroker:
    """Serves a synthetic session-shaped tape and records what was asked.

    Prices carry more significant digits than a float can hold, so any path
    that rounded one would be visible in a round-trip assertion rather than
    hiding behind a value that happens to survive.
    """

    def __init__(self, days: tuple[date, ...]) -> None:
        self.calls: list[tuple[datetime, datetime]] = []
        self._tape: dict[datetime, OHLCVCandle] = {}
        for index, day in enumerate(days):
            for offset, start in enumerate(_session_minutes(day)):
                base = Decimal("1234.567890123456789") + Decimal(index * 1000 + offset)
                self._tape[start.astimezone(UTC)] = OHLCVCandle(
                    timestamp=start,
                    open=base,
                    high=base + Decimal("0.25"),
                    low=base - Decimal("0.25"),
                    close=base + Decimal("0.125"),
                    volume=1000 + offset,
                )

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]:
        assert instrument == _INSTRUMENT
        assert interval is CandleInterval.ONE_MINUTE
        self.calls.append((start, end))
        return tuple(
            self._tape[key]
            for key in sorted(self._tape)
            if start <= key < end  # end-exclusive, the stricter of the two readings
        )

    @property
    def fetched_span(self) -> timedelta:
        """The longest single call made, exactly.

        A timedelta rather than ``.days``, which truncates: a page of thirty
        days and six hours reports thirty, so an assertion written against the
        truncated figure would pass a page the broker would refuse.
        """
        return max(end - start for start, end in self.calls)


# --------------------------------------------------------------------------
# last_completed_session_close
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "expected_day"),
    [
        (_ist(date(2026, 9, 25), 16, 0), date(2026, 9, 25)),  # Friday, after close
        (_ist(date(2026, 9, 25), 10, 0), date(2026, 9, 24)),  # Friday, mid-session
        (_ist(date(2026, 9, 26), 10, 0), date(2026, 9, 25)),  # Saturday morning
        (_ist(date(2026, 9, 26), 20, 0), date(2026, 9, 25)),  # Saturday evening
        (_ist(date(2026, 9, 27), 16, 0), date(2026, 9, 25)),  # Sunday evening
        (_ist(date(2026, 9, 28), 10, 0), date(2026, 9, 25)),  # Monday, mid-session
        (_ist(date(2026, 9, 28), 16, 0), date(2026, 9, 28)),  # Monday, after close
    ],
)
def test_last_completed_session_close_walks_back_to_a_finished_weekday(
    now: datetime, expected_day: date
) -> None:
    assert last_completed_session_close(now) == datetime.combine(
        expected_day, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE
    )


def test_last_completed_session_close_is_exactly_at_the_close() -> None:
    """The close itself counts as finished; one minute before it does not."""
    friday = date(2026, 9, 25)
    at_close = datetime.combine(friday, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE)

    assert last_completed_session_close(at_close).date() == friday
    assert last_completed_session_close(at_close - _ONE_MINUTE).date() == date(
        2026, 9, 24
    )


def test_last_completed_session_close_refuses_a_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        last_completed_session_close(datetime(2026, 9, 25, 16, 0))  # noqa: DTZ001


# --------------------------------------------------------------------------
# Fetching, caching, and not refetching
# --------------------------------------------------------------------------


def test_cold_load_fetches_and_warm_load_does_not(tmp_path: Path) -> None:
    day = date(2026, 9, 24)
    broker = FakeBroker((day,))
    store = CandleStore(tmp_path, broker)
    window = (_ist(day, 9, 15), _ist(day, 15, 30))
    now = _ist(date(2026, 9, 25), 16, 0)

    cold = store.load(_INSTRUMENT, *window, now=now)
    calls_after_cold = len(broker.calls)
    warm = CandleStore(tmp_path, broker).load(_INSTRUMENT, *window, now=now)

    assert calls_after_cold > 0
    assert len(broker.calls) == calls_after_cold, "a cached range refetched"
    assert cold == warm


def test_a_full_session_is_exactly_the_session(tmp_path: Path) -> None:
    """The window from open to close holds every bar and no part of the next."""
    day = date(2026, 9, 24)
    store = CandleStore(tmp_path, FakeBroker((day,)))

    candles = store.load(
        _INSTRUMENT,
        _ist(day, 9, 15),
        _ist(day, 15, 30),
        now=_ist(date(2026, 9, 25), 16, 0),
    )

    assert len(candles) == SESSION_MINUTES == 375
    assert candles[0].start_time == _ist(day, 9, 15).astimezone(UTC)
    assert candles[-1].end_time == _ist(day, 15, 30).astimezone(UTC)


def test_prices_survive_the_round_trip_exactly(tmp_path: Path) -> None:
    """A float anywhere in the path would change these digits."""
    day = date(2026, 9, 24)
    broker = FakeBroker((day,))
    store = CandleStore(tmp_path, broker)
    window = (_ist(day, 9, 15), _ist(day, 15, 30))
    now = _ist(date(2026, 9, 25), 16, 0)

    store.load(_INSTRUMENT, *window, now=now)
    from_disk = CandleStore(tmp_path).load(_INSTRUMENT, *window, now=now)

    first = from_disk[0]
    assert first.open == Decimal("1234.567890123456789")
    assert first.close == Decimal("1234.692890123456789")
    assert float(first.open) != first.open, "the test price is float-representable"


def test_coverage_reports_the_span_on_disk(tmp_path: Path) -> None:
    day = date(2026, 9, 24)
    store = CandleStore(tmp_path, FakeBroker((day,)))

    assert store.coverage(_INSTRUMENT) is None

    store.load(
        _INSTRUMENT,
        _ist(day, 10, 0),
        _ist(day, 11, 0),
        now=_ist(date(2026, 9, 25), 16, 0),
    )

    assert store.coverage(_INSTRUMENT) == (
        _ist(day, 10, 0).astimezone(UTC),
        _ist(day, 11, 0).astimezone(UTC),
    )


def test_coverage_may_be_shorter_than_the_request(tmp_path: Path) -> None:
    """The broker having less than was asked for must not be reported as more.

    A holiday at either end of a range does this every time it happens: a
    report that cited the requested dates here would overstate its own sample
    by two days.
    """
    held = date(2026, 9, 24)
    store = CandleStore(tmp_path, FakeBroker((held,)))

    store.load(
        _INSTRUMENT,
        _ist(date(2026, 9, 22), 9, 15),
        _ist(date(2026, 9, 25), 15, 30),
        now=_ist(date(2026, 9, 25), 16, 0),
    )

    coverage = store.coverage(_INSTRUMENT)
    assert coverage is not None
    assert coverage == (
        _ist(held, 9, 15).astimezone(UTC),
        _ist(held, 15, 30).astimezone(UTC),
    )


# --------------------------------------------------------------------------
# The contiguity invariant
# --------------------------------------------------------------------------


def test_extending_forward_refills_the_gap_it_skipped(tmp_path: Path) -> None:
    """A later, disjoint request must not leave a hole in the middle.

    Caching Tuesday and then asking for Friday has to pull Wednesday and
    Thursday too. Were it to store two islands, ``coverage`` would claim the
    whole span while the file held a two-day hole, and features computed across
    it would be wrong with nothing to detect them.
    """
    days = (date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 25))
    broker = FakeBroker(days)
    store = CandleStore(tmp_path, broker)
    now = _ist(date(2026, 9, 25), 16, 0)

    store.load(_INSTRUMENT, _ist(days[0], 9, 15), _ist(days[0], 15, 30), now=now)
    store.load(_INSTRUMENT, _ist(days[3], 9, 15), _ist(days[3], 15, 30), now=now)

    everything = CandleStore(tmp_path).load(
        _INSTRUMENT, _ist(days[0], 9, 15), _ist(days[3], 15, 30), now=now
    )

    assert len(everything) == SESSION_MINUTES * len(days)
    for day in days:
        assert any(
            candle.start_time == _ist(day, 9, 15).astimezone(UTC)
            for candle in everything
        ), f"{day} is missing from a range that claims to cover it"


def test_extending_backward_reaches_the_earlier_request(tmp_path: Path) -> None:
    days = (date(2026, 9, 22), date(2026, 9, 23))
    broker = FakeBroker(days)
    store = CandleStore(tmp_path, broker)
    now = _ist(date(2026, 9, 25), 16, 0)

    store.load(_INSTRUMENT, _ist(days[1], 9, 15), _ist(days[1], 15, 30), now=now)
    store.load(_INSTRUMENT, _ist(days[0], 9, 15), _ist(days[0], 15, 30), now=now)

    assert store.coverage(_INSTRUMENT) == (
        _ist(days[0], 9, 15).astimezone(UTC),
        _ist(days[1], 15, 30).astimezone(UTC),
    )


def test_a_narrower_request_inside_the_cache_fetches_nothing(tmp_path: Path) -> None:
    day = date(2026, 9, 24)
    broker = FakeBroker((day,))
    store = CandleStore(tmp_path, broker)
    now = _ist(date(2026, 9, 25), 16, 0)

    store.load(_INSTRUMENT, _ist(day, 9, 15), _ist(day, 15, 30), now=now)
    before = len(broker.calls)
    narrow = store.load(_INSTRUMENT, _ist(day, 10, 0), _ist(day, 10, 30), now=now)

    assert len(broker.calls) == before
    assert len(narrow) == 30


def test_a_stored_range_holds_every_minute_between_its_ends(tmp_path: Path) -> None:
    """Whatever is on disk is gapless as a sequence of the broker's own bars."""
    days = (date(2026, 9, 22), date(2026, 9, 23))
    store = CandleStore(tmp_path, FakeBroker(days))
    now = _ist(date(2026, 9, 25), 16, 0)

    store.load(_INSTRUMENT, _ist(days[0], 9, 15), _ist(days[1], 15, 30), now=now)

    with store.path_for(_INSTRUMENT).open(newline="", encoding="utf-8") as handle:
        stamps = [row["start_time"] for row in csv.DictReader(handle)]

    assert stamps == sorted(stamps), "rows are not in time order"
    assert len(stamps) == len(set(stamps)), "a minute is stored twice"


# --------------------------------------------------------------------------
# Paging against the broker's published limit
# --------------------------------------------------------------------------


def test_a_long_range_is_split_into_pages_the_broker_accepts(tmp_path: Path) -> None:
    broker = FakeBroker(_weekdays(_PAGED_START, _PAGED_END))
    store = CandleStore(tmp_path, broker)

    store.load(
        _INSTRUMENT,
        _ist(_PAGED_START, 9, 15),
        _ist(_PAGED_END, 15, 30),
        now=_ist(date(2026, 9, 25), 16, 0),
    )

    assert len(broker.calls) > 1, "a range longer than the limit was not split"
    assert broker.fetched_span <= _MAX_SPAN, "a page exceeded the published limit"
    for (_, earlier_end), (later_start, _) in zip(
        broker.calls[:-1], broker.calls[1:], strict=True
    ):
        assert later_start == earlier_end, "pages are not butted against each other"


def test_every_page_would_pass_the_brokers_own_validation(tmp_path: Path) -> None:
    """The chunker and the enforcer must agree, not merely both exist."""
    from ai_trader.broker.groww import _validate_period

    broker = FakeBroker(_weekdays(_PAGED_START, _PAGED_END))
    CandleStore(tmp_path, broker).load(
        _INSTRUMENT,
        _ist(_PAGED_START, 9, 15),
        _ist(_PAGED_END, 15, 30),
        now=_ist(date(2026, 9, 25), 16, 0),
    )

    for start, end in broker.calls:
        _validate_period(start, end, CandleInterval.ONE_MINUTE)

    assert len(broker.calls) > 1, "one page would not have exercised the chunker"


def test_an_over_long_span_is_refused_with_the_span_that_was_asked_for() -> None:
    """The limit and six hours is not the limit, and the refusal must say so.

    The guard compares the span exactly and used to print it as ``.days``,
    which truncates. Six hours over a thirty-day limit came back as "at most 30
    days ...; 30 days were requested" -- a sentence that refutes its own premise
    and sends the reader hunting for a bug in the comparison when the fix is to
    shorten the range by six hours.

    The accepted call above the refused one is what makes this about the
    message rather than about the boundary: exactly the limit is a span the
    broker serves, so the sentence under test is describing a real overshoot.
    """
    from ai_trader.broker import MAX_HISTORICAL_SPAN
    from ai_trader.broker.groww import _validate_period

    limit = MAX_HISTORICAL_SPAN[CandleInterval.ONE_MINUTE]
    start = _ist(date(2026, 9, 1), 9, 15)

    _validate_period(start, start + limit, CandleInterval.ONE_MINUTE)

    with pytest.raises(ValueError) as refused:
        _validate_period(
            start, start + limit + timedelta(hours=6), CandleInterval.ONE_MINUTE
        )

    message = str(refused.value)
    assert "6:00:00" in message, f"the overshoot is missing from {message!r}"
    assert str(limit.days) in message, "the limit is no longer stated"


# --------------------------------------------------------------------------
# Never storing an unfinished session
# --------------------------------------------------------------------------


def test_a_request_past_the_last_close_is_clipped(tmp_path: Path) -> None:
    """Today's half-written tape must not become tomorrow's cached truth."""
    days = (date(2026, 9, 24), date(2026, 9, 25))
    broker = FakeBroker(days)
    store = CandleStore(tmp_path, broker)

    store.load(
        _INSTRUMENT,
        _ist(days[0], 9, 15),
        _ist(days[1], 15, 30),
        now=_ist(days[1], 11, 0),  # Friday, mid-session
    )

    coverage = store.coverage(_INSTRUMENT)
    assert coverage is not None
    assert coverage[1] == _ist(days[0], 15, 30).astimezone(UTC)
    for _, end in broker.calls:
        assert end <= _ist(days[0], 15, 30), "an unfinished session was requested"


def test_a_window_entirely_in_the_future_is_empty_and_silent(tmp_path: Path) -> None:
    broker = FakeBroker((date(2026, 9, 25),))
    store = CandleStore(tmp_path, broker)

    candles = store.load(
        _INSTRUMENT,
        _ist(date(2026, 9, 25), 9, 15),
        _ist(date(2026, 9, 25), 15, 30),
        now=_ist(date(2026, 9, 25), 9, 0),  # before Friday's open
    )

    assert candles == ()
    assert broker.calls == []
    assert not store.path_for(_INSTRUMENT).exists()


# --------------------------------------------------------------------------
# Running without a broker
# --------------------------------------------------------------------------


def test_without_a_broker_a_cached_range_still_serves(tmp_path: Path) -> None:
    day = date(2026, 9, 24)
    window = (_ist(day, 9, 15), _ist(day, 15, 30))
    now = _ist(date(2026, 9, 25), 16, 0)
    CandleStore(tmp_path, FakeBroker((day,))).load(_INSTRUMENT, *window, now=now)

    offline = CandleStore(tmp_path).load(_INSTRUMENT, *window, now=now)

    assert len(offline) == SESSION_MINUTES


def test_without_a_broker_an_uncached_range_raises(tmp_path: Path) -> None:
    with pytest.raises(CandleStoreError, match="no broker"):
        CandleStore(tmp_path).load(
            _INSTRUMENT,
            _ist(date(2026, 9, 24), 9, 15),
            _ist(date(2026, 9, 24), 15, 30),
            now=_ist(date(2026, 9, 25), 16, 0),
        )


def test_without_a_broker_a_partly_cached_range_raises(tmp_path: Path) -> None:
    """Serving the held part silently would be a shorter tape wearing the name."""
    days = (date(2026, 9, 24), date(2026, 9, 25))
    now = _ist(date(2026, 9, 25), 16, 0)
    CandleStore(tmp_path, FakeBroker(days)).load(
        _INSTRUMENT, _ist(days[0], 9, 15), _ist(days[0], 15, 30), now=now
    )

    with pytest.raises(CandleStoreError):
        CandleStore(tmp_path).load(
            _INSTRUMENT, _ist(days[0], 9, 15), _ist(days[1], 15, 30), now=now
        )


# --------------------------------------------------------------------------
# What was asked for, versus what came back
# --------------------------------------------------------------------------


class ShortTapeBroker(FakeBroker):
    """A broker whose sessions stop printing before the close, as NSE's do.

    The real one-minute tape has no bars through most of 15:15--15:29 and lands
    its final print at 15:28 for one symbol and 15:29 for another. A fake that
    fills every minute to the close cannot exhibit the defect this section is
    about, so these tests would pass against a store that still derived
    "already fetched" from the bars.
    """

    def __init__(self, days: tuple[date, ...], *, silent_from: time) -> None:
        super().__init__(days)
        self._tape = {
            key: candle
            for key, candle in self._tape.items()
            if key.astimezone(INDIA_TIMEZONE).time() < silent_from
        }


def test_a_tape_that_stops_before_the_close_is_not_refetched(tmp_path: Path) -> None:
    """The window asks to 15:30; the tape ends at 15:28; the gap is a phantom."""
    day = date(2026, 9, 24)
    window = (_ist(day, 9, 15), _ist(day, 15, 30))
    now = _ist(date(2026, 9, 25), 16, 0)
    broker = ShortTapeBroker((day,), silent_from=time(15, 28))
    store = CandleStore(tmp_path, broker)

    store.load(_INSTRUMENT, *window, now=now)
    after_first = len(broker.calls)
    store.load(_INSTRUMENT, *window, now=now)

    assert len(broker.calls) == after_first, (
        "the store asked again for minutes the exchange never printed, which it "
        "will go on doing for the life of the cache"
    )
    assert after_first == 1, "the first load did not fetch, so nothing was proved"


def test_without_a_broker_a_tape_that_stops_early_still_serves(
    tmp_path: Path,
) -> None:
    """The failure the fetched-span record exists to prevent.

    An offline re-run of a published sweep has no broker by design. Reading the
    minutes after the last print as an unfetched gap kills that re-run on a
    session a live run would have traded straight through -- which is the whole
    asymmetry between historical and live this store is not allowed to have.
    """
    day = date(2026, 9, 24)
    window = (_ist(day, 9, 15), _ist(day, 15, 30))
    now = _ist(date(2026, 9, 25), 16, 0)
    broker = ShortTapeBroker((day,), silent_from=time(15, 28))
    CandleStore(tmp_path, broker).load(_INSTRUMENT, *window, now=now)

    served = CandleStore(tmp_path).load(_INSTRUMENT, *window, now=now)

    assert served[-1].end_time == _ist(day, 15, 28), "the fixture printed too late"
    assert len(served) == SESSION_MINUTES - 2, "the held session was not served whole"


def test_a_holiday_at_the_start_of_a_range_is_not_asked_for_twice(
    tmp_path: Path,
) -> None:
    """The exchange was shut on the first day, so no bar can ever prove it was."""
    shut, traded = date(2026, 9, 24), date(2026, 9, 25)
    window = (_ist(shut, 9, 15), _ist(traded, 15, 30))
    now = _ist(date(2026, 9, 26), 16, 0)
    broker = FakeBroker((traded,))
    store = CandleStore(tmp_path, broker)

    store.load(_INSTRUMENT, *window, now=now)
    after_first = len(broker.calls)
    store.load(_INSTRUMENT, *window, now=now)

    assert len(broker.calls) == after_first, "the shut day was fetched a second time"
    assert after_first == 1, "the first load did not fetch, so nothing was proved"


def test_the_fetched_record_does_not_widen_reported_coverage(tmp_path: Path) -> None:
    """Two facts, two files, and only one of them is the measured sample.

    ``coverage`` is what a report cites, so it must stay the span of bars that
    exist. If the record of what was requested could widen it, every report
    would claim a sample two minutes longer than the tape it measured.
    """
    day = date(2026, 9, 24)
    now = _ist(date(2026, 9, 25), 16, 0)
    store = CandleStore(tmp_path, ShortTapeBroker((day,), silent_from=time(15, 28)))
    store.load(_INSTRUMENT, _ist(day, 9, 15), _ist(day, 15, 30), now=now)

    assert store.coverage(_INSTRUMENT) == (_ist(day, 9, 15), _ist(day, 15, 28))
    assert store.fetched_path_for(_INSTRUMENT).exists(), "nothing was recorded"


def test_the_fetched_record_cannot_stand_in_for_missing_bars(tmp_path: Path) -> None:
    """A deleted cache file must refetch, not serve nothing and call it complete."""
    day = date(2026, 9, 24)
    window = (_ist(day, 9, 15), _ist(day, 15, 30))
    now = _ist(date(2026, 9, 25), 16, 0)
    broker = FakeBroker((day,))
    store = CandleStore(tmp_path, broker)
    store.load(_INSTRUMENT, *window, now=now)
    store.path_for(_INSTRUMENT).unlink()

    served = store.load(_INSTRUMENT, *window, now=now)

    assert len(broker.calls) == 2, "the surviving record suppressed a needed fetch"
    assert len(served) == SESSION_MINUTES, "the session was not rebuilt"


# --------------------------------------------------------------------------
# The file itself
# --------------------------------------------------------------------------


def test_the_file_stores_no_derived_or_redundant_column(tmp_path: Path) -> None:
    """``end_time`` is always start plus a minute; the symbol is the filename."""
    day = date(2026, 9, 24)
    store = CandleStore(tmp_path, FakeBroker((day,)))
    store.load(
        _INSTRUMENT,
        _ist(day, 9, 15),
        _ist(day, 15, 30),
        now=_ist(date(2026, 9, 25), 16, 0),
    )

    with store.path_for(_INSTRUMENT).open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    assert tuple(header) == _FIELDS
    assert "end_time" not in header
    assert "instrument" not in header


def test_rows_out_of_order_on_disk_do_not_corrupt_coverage(tmp_path: Path) -> None:
    """A hand-edited file must not be able to report a span it does not hold."""
    day = date(2026, 9, 24)
    store = CandleStore(tmp_path, FakeBroker((day,)))
    store.load(
        _INSTRUMENT,
        _ist(day, 9, 15),
        _ist(day, 15, 30),
        now=_ist(date(2026, 9, 25), 16, 0),
    )
    path = store.path_for(_INSTRUMENT)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(reversed(rows))

    assert CandleStore(tmp_path).coverage(_INSTRUMENT) == (
        _ist(day, 9, 15).astimezone(UTC),
        _ist(day, 15, 30).astimezone(UTC),
    )


def test_the_temporary_file_does_not_survive_a_write(tmp_path: Path) -> None:
    day = date(2026, 9, 24)
    store = CandleStore(tmp_path, FakeBroker((day,)))
    store.load(
        _INSTRUMENT,
        _ist(day, 9, 15),
        _ist(day, 15, 30),
        now=_ist(date(2026, 9, 25), 16, 0),
    )

    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("symbol", ["M&M", "BAJAJ-AUTO", "RELIANCE"])
def test_real_nse_symbols_are_valid_filenames(tmp_path: Path, symbol: str) -> None:
    path = CandleStore(tmp_path).path_for(Instrument("NSE", symbol))

    assert path.name == f"NSE_{symbol}.csv"
    assert path.parent == tmp_path


@pytest.mark.parametrize("symbol", ["../ESCAPE", "A/B", "A\\B", ""])
def test_a_symbol_that_is_not_a_filename_is_refused_not_mangled(
    tmp_path: Path, symbol: str
) -> None:
    """Sanitizing would map two symbols onto one file; refusing cannot."""
    with pytest.raises(ValueError, match="filename"):
        CandleStore(tmp_path).path_for(Instrument("NSE", symbol))


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (
            datetime(2026, 9, 24, 9, 15),  # noqa: DTZ001
            _ist(date(2026, 9, 24), 15, 30),
            "start time must be timezone-aware",
        ),
        (
            _ist(date(2026, 9, 24), 9, 15),
            datetime(2026, 9, 24, 15, 30),  # noqa: DTZ001
            "end time must be timezone-aware",
        ),
        (
            _ist(date(2026, 9, 24), 15, 30),
            _ist(date(2026, 9, 24), 9, 15),
            "start time must be before",
        ),
    ],
)
def test_load_refuses_an_unusable_window(
    tmp_path: Path, start: datetime, end: datetime, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CandleStore(tmp_path).load(_INSTRUMENT, start, end)
