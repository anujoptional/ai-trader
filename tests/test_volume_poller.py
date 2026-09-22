"""Tests for polled cumulative volume and the ticks it stamps."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import monotonic, sleep

import pytest

from ai_trader.broker import Instrument, MarketQuote, MarketTick
from ai_trader.market import CandleBuilder
from ai_trader.market.volume_poller import (
    DEFAULT_MAX_READING_AGE_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    VolumePoller,
    VolumePollerError,
)

RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
TCS = Instrument(exchange="NSE", trading_symbol="TCS")


def _quote(instrument: Instrument, volume: int) -> MarketQuote:
    return MarketQuote(
        instrument=instrument,
        last_price=Decimal("100"),
        last_trade_at=datetime(2026, 9, 21, 4, 0, tzinfo=UTC),
        open=Decimal("99"),
        high=Decimal("101"),
        low=Decimal("98"),
        previous_close=Decimal("97"),
        volume=volume,
        day_change=Decimal("3"),
        day_change_percent=Decimal("3.09"),
    )


def _tick(
    instrument: Instrument = RELIANCE,
    price: str = "100",
    second: int = 0,
    minute: int = 45,
    cumulative_volume: int | None = None,
) -> MarketTick:
    return MarketTick(
        instrument=instrument,
        price=Decimal(price),
        timestamp=datetime(2026, 9, 21, 3, minute, second, tzinfo=UTC),
        cumulative_volume=cumulative_volume,
    )


class _StubBroker:
    """A broker that serves a scripted volume per instrument."""

    def __init__(self, volumes: dict[Instrument, list[int | Exception]]) -> None:
        self._volumes = {key: list(value) for key, value in volumes.items()}
        self.call_count = 0

    def get_quote(self, instrument: Instrument) -> MarketQuote:
        self.call_count += 1
        queue = self._volumes[instrument]
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(value, Exception):
            raise value
        return _quote(instrument, value)


class _UnreadableQuote:
    """A quote whose volume raises when the poller reads it.

    Stands in for any failure landing after the broker call returns, which is
    the half of a poll that is easiest to leave outside a failure guard and the
    most costly to: the loop above has no guard of its own.
    """

    @property
    def volume(self) -> int:
        raise RuntimeError("volume is unreadable")


class _LateFailingBroker:
    """A broker whose quotes fail only once the poller reads them."""

    def __init__(self, failing: Instrument) -> None:
        self._failing = failing
        self.call_count = 0

    def get_quote(self, instrument: Instrument) -> MarketQuote | _UnreadableQuote:
        self.call_count += 1
        if instrument == self._failing:
            return _UnreadableQuote()
        return _quote(instrument, 7)


class _ManualClock:
    """A stand-in for ``datetime`` whose moment the test sets explicitly.

    Advancing by assignment rather than by consuming a scripted queue keeps a
    test readable when the number of ``now`` calls depends on how many polls
    succeed, which is exactly what the staleness tests vary.
    """

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def now(self, tz: object = None) -> datetime:
        return self.moment


def _wait_until(condition: Callable[[], bool], timeout_seconds: float = 5.0) -> None:
    """Sleep in short slices until ``condition`` holds or time runs out.

    Sleeping rather than spinning matters: a tight loop holds the GIL and can
    starve the very poll thread the caller is waiting on.
    """
    deadline = monotonic() + timeout_seconds
    while not condition() and monotonic() < deadline:
        sleep(0.01)


def test_poller_rejects_an_empty_instrument_list() -> None:
    with pytest.raises(VolumePollerError, match="at least one instrument"):
        VolumePoller(_StubBroker({}), ())


def test_poller_rejects_a_non_positive_interval() -> None:
    broker = _StubBroker({RELIANCE: [1]})
    with pytest.raises(VolumePollerError, match="positive number of seconds"):
        VolumePoller(broker, (RELIANCE,), interval_seconds=0)


def test_poller_deduplicates_instruments_and_keeps_order() -> None:
    broker = _StubBroker({RELIANCE: [1], TCS: [2]})
    poller = VolumePoller(broker, (RELIANCE, TCS, RELIANCE))

    assert poller.instruments == (RELIANCE, TCS)


def test_default_interval_is_used_when_none_is_given() -> None:
    poller = VolumePoller(_StubBroker({RELIANCE: [1]}), (RELIANCE,))

    assert poller.interval_seconds == DEFAULT_POLL_INTERVAL_SECONDS


def test_latest_is_none_before_any_poll() -> None:
    poller = VolumePoller(_StubBroker({RELIANCE: [1]}), (RELIANCE,))

    assert poller.latest(RELIANCE) is None
    assert poller.latest_observed_at(RELIANCE) is None
    assert poller.poll_count == 0


def test_poll_once_records_the_latest_total() -> None:
    broker = _StubBroker({RELIANCE: [1000, 1500]})
    poller = VolumePoller(broker, (RELIANCE,))

    assert poller.poll_once() == 1
    assert poller.latest(RELIANCE) == 1000
    assert poller.poll_once() == 1
    assert poller.latest(RELIANCE) == 1500
    assert poller.poll_count == 2
    assert poller.failure_count == 0
    assert poller.latest_observed_at(RELIANCE) is not None


def test_poll_once_tracks_instruments_independently() -> None:
    broker = _StubBroker({RELIANCE: [1000], TCS: [7]})
    poller = VolumePoller(broker, (RELIANCE, TCS))

    assert poller.poll_once() == 2
    assert poller.latest(RELIANCE) == 1000
    assert poller.latest(TCS) == 7


def test_a_failed_poll_is_counted_and_dropped() -> None:
    broker = _StubBroker({RELIANCE: [RuntimeError("quote is malformed")]})
    poller = VolumePoller(broker, (RELIANCE,))

    assert poller.poll_once() == 0
    assert poller.latest(RELIANCE) is None
    assert poller.poll_count == 1
    assert poller.failure_count == 1
    assert poller.last_error == "RuntimeError: quote is malformed"


def test_a_failed_poll_keeps_the_previous_reading() -> None:
    broker = _StubBroker({RELIANCE: [1000, RuntimeError("boom")]})
    poller = VolumePoller(broker, (RELIANCE,))
    poller.poll_once()

    assert poller.poll_once() == 0
    assert poller.latest(RELIANCE) == 1000
    assert poller.failure_count == 1


def test_a_failure_after_the_quote_lands_is_counted_and_dropped() -> None:
    broker = _LateFailingBroker(RELIANCE)
    poller = VolumePoller(broker, (RELIANCE, TCS))

    # TCS is still polled: the failure is contained within the one instrument
    # rather than abandoning the rest of the round.
    assert poller.poll_once() == 1
    assert poller.latest(RELIANCE) is None
    assert poller.latest(TCS) == 7
    assert poller.poll_count == 2
    assert poller.failure_count == 1
    assert poller.last_error == "RuntimeError: volume is unreadable"


def test_a_total_that_moves_backwards_within_a_session_is_rejected() -> None:
    broker = _StubBroker({RELIANCE: [1000, 400]})
    poller = VolumePoller(broker, (RELIANCE,))
    poller.poll_once()

    assert poller.poll_once() == 0
    assert poller.latest(RELIANCE) == 1000
    assert poller.regression_count == 1
    assert poller.failure_count == 0


def test_a_lower_total_on_a_new_session_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _StubBroker({RELIANCE: [9_000_000, 400]})
    poller = VolumePoller(broker, (RELIANCE,))
    clock = _ManualClock(datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    monkeypatch.setattr("ai_trader.market.volume_poller.datetime", clock)

    poller.poll_once()
    assert poller.latest(RELIANCE) == 9_000_000

    clock.moment = datetime(2026, 9, 22, 4, 0, tzinfo=UTC)
    assert poller.poll_once() == 1
    assert poller.latest(RELIANCE) == 400
    assert poller.regression_count == 0


def test_a_pre_open_reading_does_not_poison_the_session_that_follows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exchange's total resets at 09:15, not at midnight.

    The architecture has the poller warming up pre-open, where the quote
    endpoint still serves the previous session's closing total. Filing that
    under the calendar date would make every reading of the new session a
    backwards move against a stale high-water mark, rejecting all of them and
    leaving volume unavailable for the entire day.
    """
    broker = _StubBroker({RELIANCE: [9_000_000, 50_000, 60_000]})
    poller = VolumePoller(broker, (RELIANCE,))
    # 09:05 IST: pre-open, so this total belongs to the previous session.
    clock = _ManualClock(datetime(2026, 9, 22, 3, 35, tzinfo=UTC))
    monkeypatch.setattr("ai_trader.market.volume_poller.datetime", clock)

    poller.poll_once()
    assert poller.latest(RELIANCE) == 9_000_000

    # 09:20 IST: the session has opened and the counter has reset.
    clock.moment = datetime(2026, 9, 22, 3, 50, tzinfo=UTC)
    assert poller.poll_once() == 1
    assert poller.latest(RELIANCE) == 50_000

    clock.moment = datetime(2026, 9, 22, 3, 51, tzinfo=UTC)
    assert poller.poll_once() == 1
    assert poller.latest(RELIANCE) == 60_000
    assert poller.regression_count == 0


def test_an_unchanged_total_is_accepted() -> None:
    broker = _StubBroker({RELIANCE: [1000, 1000]})
    poller = VolumePoller(broker, (RELIANCE,))
    poller.poll_once()

    assert poller.poll_once() == 1
    assert poller.latest(RELIANCE) == 1000
    assert poller.regression_count == 0


def test_stamp_adds_the_polled_total_to_a_tick() -> None:
    poller = VolumePoller(_StubBroker({RELIANCE: [1000]}), (RELIANCE,))
    poller.poll_once()

    stamped = poller.stamp(_tick())

    assert stamped.cumulative_volume == 1000
    assert stamped.price == Decimal("100")
    assert stamped.timestamp == _tick().timestamp


def test_stamp_leaves_a_tick_that_already_carries_volume_untouched() -> None:
    poller = VolumePoller(_StubBroker({RELIANCE: [1000]}), (RELIANCE,))
    poller.poll_once()
    original = _tick(cumulative_volume=42)

    assert poller.stamp(original) is original


def test_stamp_leaves_a_tick_untouched_before_any_reading_lands() -> None:
    poller = VolumePoller(_StubBroker({RELIANCE: [1000]}), (RELIANCE,))
    original = _tick()

    assert poller.stamp(original) is original


def test_stamp_leaves_an_untracked_instrument_untouched() -> None:
    poller = VolumePoller(_StubBroker({RELIANCE: [1000]}), (RELIANCE,))
    poller.poll_once()
    original = _tick(instrument=TCS)

    assert poller.stamp(original) is original


def test_poller_rejects_a_non_positive_reading_age() -> None:
    broker = _StubBroker({RELIANCE: [1]})
    with pytest.raises(VolumePollerError, match="positive number of seconds"):
        VolumePoller(broker, (RELIANCE,), max_reading_age_seconds=0)


def test_the_reading_age_bound_scales_with_a_slower_cadence() -> None:
    broker = _StubBroker({RELIANCE: [1]})
    default = VolumePoller(broker, (RELIANCE,))
    slow = VolumePoller(broker, (RELIANCE,), interval_seconds=30)
    explicit = VolumePoller(broker, (RELIANCE,), max_reading_age_seconds=3)

    # A fixed bound would expire every reading of a slow poller before its
    # replacement was even due, so the default is the larger of the derived
    # figure and five rounds. An explicit bound is honoured as given.
    assert default.max_reading_age_seconds == DEFAULT_MAX_READING_AGE_SECONDS
    assert slow.max_reading_age_seconds == 150
    assert explicit.max_reading_age_seconds == 3


def test_a_reading_that_stops_being_replaced_stops_being_stamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _StubBroker({RELIANCE: [1000, RuntimeError("quote endpoint is down")]})
    poller = VolumePoller(broker, (RELIANCE,))
    clock = _ManualClock(datetime(2026, 9, 21, 4, 0, tzinfo=UTC))
    monkeypatch.setattr("ai_trader.market.volume_poller.datetime", clock)
    poller.poll_once()

    clock.moment += timedelta(seconds=poller.max_reading_age_seconds)
    # A failed poll cannot refresh the reading's age; that is what lets the
    # bound below distinguish an outage from an instrument that is merely flat.
    assert poller.poll_once() == 0
    assert poller.stamp(_tick()).cumulative_volume == 1000
    assert poller.stale_stamp_count == 0

    clock.moment += timedelta(seconds=1)
    assert poller.stamp(_tick()).cumulative_volume is None
    assert poller.stale_stamp_count == 1
    # The reading is withheld from ticks, not discarded: a caller reporting on
    # the poller can still show the last total it managed to read.
    assert poller.latest(RELIANCE) == 1000


def test_start_polls_inline_before_returning() -> None:
    broker = _StubBroker({RELIANCE: [1000]})
    poller = VolumePoller(broker, (RELIANCE,), interval_seconds=30)
    try:
        poller.start()
        assert poller.latest(RELIANCE) == 1000
        assert poller.is_running
    finally:
        poller.stop()
    assert not poller.is_running


def test_starting_a_running_poller_is_refused() -> None:
    poller = VolumePoller(
        _StubBroker({RELIANCE: [1000]}),
        (RELIANCE,),
        interval_seconds=30,
    )
    try:
        poller.start()
        with pytest.raises(VolumePollerError, match="already running"):
            poller.start()
    finally:
        poller.stop()


def test_a_stopped_poller_can_be_started_again() -> None:
    poller = VolumePoller(
        _StubBroker({RELIANCE: [1000]}),
        (RELIANCE,),
        interval_seconds=30,
    )
    poller.start()
    poller.stop()
    poller.start()
    try:
        assert poller.is_running
    finally:
        poller.stop()


def test_stopping_a_poller_that_never_started_is_safe() -> None:
    poller = VolumePoller(_StubBroker({RELIANCE: [1000]}), (RELIANCE,))

    poller.stop()

    assert not poller.is_running


def test_the_context_manager_starts_and_stops_the_thread() -> None:
    broker = _StubBroker({RELIANCE: [1000]})
    with VolumePoller(broker, (RELIANCE,), interval_seconds=30) as poller:
        assert poller.is_running
        assert poller.latest(RELIANCE) == 1000
    assert not poller.is_running


def test_the_thread_keeps_polling_on_its_cadence() -> None:
    broker = _StubBroker({RELIANCE: [1000, 1100, 1200, 1300]})
    poller = VolumePoller(broker, (RELIANCE,), interval_seconds=0.01)
    try:
        poller.start()
        _wait_until(lambda: poller.poll_count >= 4)
    finally:
        poller.stop()

    assert poller.poll_count >= 4
    assert poller.latest(RELIANCE) == 1300


def test_a_thread_poll_failure_never_escapes() -> None:
    broker = _StubBroker({RELIANCE: [RuntimeError("quote is malformed")]})
    poller = VolumePoller(broker, (RELIANCE,), interval_seconds=0.01)
    try:
        poller.start()
        _wait_until(lambda: poller.failure_count >= 3)
    finally:
        poller.stop()

    assert poller.failure_count >= 3
    assert poller.latest(RELIANCE) is None
    assert poller.is_running is False


def test_a_thread_failure_after_the_quote_lands_never_escapes() -> None:
    # The costly variant of the test above. A failure past the broker call that
    # escapes ends the poll thread outright, stranding volume for the rest of
    # the session, and a dead thread records no error to notice it by -- so the
    # assertion that matters is that the poller is still running.
    broker = _LateFailingBroker(RELIANCE)
    poller = VolumePoller(broker, (RELIANCE,), interval_seconds=0.01)
    try:
        poller.start()
        _wait_until(lambda: poller.failure_count >= 3)
        assert poller.is_running is True
    finally:
        poller.stop()

    assert poller.failure_count >= 3
    assert poller.latest(RELIANCE) is None


def test_stamped_ticks_give_the_first_emitted_candle_a_real_volume() -> None:
    broker = _StubBroker({RELIANCE: [1000, 1500, 2200]})
    poller = VolumePoller(broker, (RELIANCE,))
    builder = CandleBuilder()
    emitted = []

    for minute, second in ((45, 30), (46, 10), (47, 5)):
        poller.poll_once()
        candle = builder.add_tick(poller.stamp(_tick(minute=minute, second=second)))
        if candle is not None:
            emitted.append(candle)

    assert len(emitted) == 1
    assert emitted[0].start_time == datetime(2026, 9, 21, 3, 46, tzinfo=UTC)
    assert emitted[0].volume == 500


def test_unstamped_ticks_leave_the_candle_volume_unset() -> None:
    builder = CandleBuilder()
    emitted = []

    for minute, second in ((45, 30), (46, 10), (47, 5)):
        candle = builder.add_tick(_tick(minute=minute, second=second))
        if candle is not None:
            emitted.append(candle)

    assert len(emitted) == 1
    assert emitted[0].volume is None


def test_a_failed_poll_shifts_volume_forward_without_losing_any() -> None:
    broker = _StubBroker({RELIANCE: [1000, 1500, RuntimeError("boom"), 3000, 3400]})
    poller = VolumePoller(broker, (RELIANCE,))
    builder = CandleBuilder()
    volumes = []

    for minute, second in ((45, 30), (46, 10), (47, 5), (48, 5), (49, 5)):
        poller.poll_once()
        candle = builder.add_tick(poller.stamp(_tick(minute=minute, second=second)))
        if candle is not None:
            volumes.append(candle.volume)

    # The failed third poll left the 03:47 tick carrying the stale 1500, so
    # that minute reports nothing and 03:48 absorbs what it missed. The three
    # emitted minutes still sum to the 1000 -> 3000 span they cover, so a
    # dropped poll costs attribution between minutes, never accuracy overall.
    assert volumes == [500, 0, 1500]
    assert sum(volumes) == 3000 - 1000
    assert poller.failure_count == 1


def test_a_sustained_outage_reports_no_volume_rather_than_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outage = RuntimeError("quote endpoint is down")
    broker = _StubBroker({RELIANCE: [1000, 1500, 2200, outage]})
    poller = VolumePoller(broker, (RELIANCE,))
    clock = _ManualClock(datetime(2026, 9, 21, 3, 45, tzinfo=UTC))
    monkeypatch.setattr("ai_trader.market.volume_poller.datetime", clock)
    builder = CandleBuilder()
    volumes = []

    for minute in (45, 46, 47, 48, 49):
        poller.poll_once()
        candle = builder.add_tick(poller.stamp(_tick(minute=minute, second=5)))
        if candle is not None:
            volumes.append(candle.volume)
        clock.moment += timedelta(minutes=1)

    # A brief failure shifts volume between minutes, but an outage outlasting
    # the age bound must not keep stamping the total frozen at 2200: every
    # minute after it would then difference to zero and claim, with a readiness
    # flag vouching for it, that the instrument had stopped trading. Withheld
    # stamps make those minutes honestly unavailable instead.
    assert volumes == [500, None, None]
    assert poller.failure_count == 2
    assert poller.stale_stamp_count == 2
