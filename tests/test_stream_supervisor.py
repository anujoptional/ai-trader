from collections import deque
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from time import monotonic, sleep

import pytest

from ai_trader.broker import Instrument, MarketTick
from ai_trader.market.stream import (
    ClosableTickStream,
    StreamReport,
    StreamSupervisor,
    StreamSupervisorError,
)

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_MINUTE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
_FAST = {"session_seconds": 0.01, "base_delay_seconds": 0.001}


class _Dropped(RuntimeError):
    """Stands in for a transport failure the caller declared retryable."""


class _Broken(RuntimeError):
    """Stands in for a failure the caller did not declare retryable."""


class _Feed:
    """A scripted stand-in for a broker stream and the factory that opens it.

    Each entry in ``script`` is one ``collect`` call: an integer delivers that
    many ticks, an exception is raised instead. ``open_errors`` raises from the
    factory itself. Once the script is spent the feed asks the supervisor to
    stop, so a test ends on its script rather than on the wall clock.
    """

    def __init__(
        self,
        script: Iterable[int | BaseException],
        open_errors: Iterable[BaseException] = (),
    ) -> None:
        self._script: deque[int | BaseException] = deque(script)
        self._open_errors: deque[BaseException] = deque(open_errors)
        self.opens = 0
        self.tick_budgets: list[int] = []
        self.on_exhausted: Callable[[], None] | None = None

    def open(self) -> "_Feed":
        if self._open_errors:
            raise self._open_errors.popleft()
        self.opens += 1
        return self

    def collect(
        self,
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None] | None = None,
    ) -> tuple[MarketTick, ...]:
        self.tick_budgets.append(max_ticks)
        if not self._script:
            # A quiet market: the call runs its full length and returns nothing.
            sleep(timeout_seconds)
            return ()

        entry = self._script.popleft()
        if not self._script and self.on_exhausted is not None:
            self.on_exhausted()
        if isinstance(entry, BaseException):
            raise entry

        ticks = tuple(
            MarketTick(
                instrument=_RELIANCE,
                timestamp=_MINUTE + timedelta(seconds=index),
                price=Decimal("100"),
                cumulative_volume=None,
            )
            for index in range(min(entry, max_ticks))
        )
        for tick in ticks:
            if on_tick is not None:
                on_tick(tick)
        return ticks


def _supervise(
    feed: _Feed,
    on_tick: Callable[[MarketTick], None] | None = None,
    **options: object,
) -> StreamSupervisor:
    supervisor = StreamSupervisor(
        feed.open,
        on_tick if on_tick is not None else lambda tick: None,
        retry_on=(_Dropped,),
        **(_FAST | options),  # type: ignore[arg-type]
    )
    feed.on_exhausted = supervisor.stop
    return supervisor


def _run(feed: _Feed, **options: object) -> tuple[StreamReport, list[MarketTick]]:
    received: list[MarketTick] = []
    supervisor = _supervise(feed, received.append, **options)
    return supervisor.run(duration_seconds=5.0), received


def test_a_clean_run_delivers_every_tick_the_stream_produced() -> None:
    report, received = _run(_Feed([2, 3, 1]))

    assert [tick.price for tick in received] == [Decimal("100")] * 6
    assert report.ticks == 6
    assert report.sessions == 3
    assert report.failures == 0
    assert report.last_error is None
    assert report.stopped_because == "stopped"


def test_a_productive_session_reuses_the_stream_it_already_has() -> None:
    """Reconnecting per session would resolve instruments 375 times a day."""
    feed = _Feed([1, 1, 1])

    report, _ = _run(feed)

    assert feed.opens == 1
    assert report.reconnects == 0
    assert report.silent_sessions == 0


def test_a_silent_session_rebuilds_the_stream() -> None:
    """The likeliest way a WebSocket dies is by going quiet, not by raising.

    Nothing distinguishes a dead socket from a dormant one at this layer, so a
    session that delivered nothing is treated as suspect and the stream is
    rebuilt. The cost of being wrong is one reconnect against a quiet market;
    the cost of being right is the difference between a working feed and a
    process that sits silent until the close.
    """
    feed = _Feed([0, 0, 1])

    report, _ = _run(feed)

    assert report.silent_sessions == 2
    assert report.sessions == 3
    assert feed.opens == 3
    assert report.reconnects == 2
    # A rebuild is not a failure: nothing raised and nothing is owed a retry.
    assert report.failures == 0
    assert report.last_error is None


def test_a_retryable_failure_reconnects_and_the_run_continues() -> None:
    feed = _Feed([2, _Dropped("socket closed"), 3])

    report, received = _run(feed)

    assert len(received) == 5
    assert report.ticks == 5
    assert report.failures == 1
    assert report.last_error == "_Dropped: socket closed"
    assert feed.opens == 2
    assert report.reconnects == 1
    assert report.stopped_because == "stopped"


def test_ticks_delivered_before_a_failure_are_still_counted() -> None:
    """The consumer already has them; a report that omits them is simply wrong."""

    def collect_then_fail(
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None] | None = None,
    ) -> tuple[MarketTick, ...]:
        assert on_tick is not None
        on_tick(
            MarketTick(
                instrument=_RELIANCE,
                timestamp=_MINUTE,
                price=Decimal("100"),
                cumulative_volume=None,
            )
        )
        raise _Dropped("mid-session")

    feed = _Feed([1])
    feed.collect = collect_then_fail  # type: ignore[method-assign]
    received: list[MarketTick] = []
    supervisor = _supervise(feed, received.append, max_consecutive_failures=1)

    report = supervisor.run(duration_seconds=5.0)

    assert len(received) == 1
    assert report.ticks == 1
    assert report.stopped_because == "gave_up"


def test_an_error_outside_retry_on_propagates() -> None:
    """A consumer bug and a dropped socket arrive by the same route.

    Only the caller knows which is which, so anything it did not declare
    retryable is left to escape rather than being retried until the close.
    """
    supervisor = _supervise(_Feed([1, _Broken("contract changed"), 1]))

    with pytest.raises(_Broken, match="contract changed"):
        supervisor.run(duration_seconds=5.0)


def test_a_consumer_that_raises_is_not_retried() -> None:
    def explode(tick: MarketTick) -> None:
        raise _Broken("consumer bug")

    supervisor = _supervise(_Feed([1]), explode)

    with pytest.raises(_Broken, match="consumer bug"):
        supervisor.run(duration_seconds=5.0)


def test_repeated_failures_give_up_rather_than_spinning_all_day() -> None:
    feed = _Feed([_Dropped("one"), _Dropped("two"), _Dropped("three"), 5])

    report, received = _run(feed, max_consecutive_failures=3)

    assert report.stopped_because == "gave_up"
    assert report.failures == 3
    assert report.last_error == "_Dropped: three"
    # The fourth entry is never reached, so the budget really did stop the loop.
    assert received == []
    assert report.ticks == 0


def test_a_completed_session_resets_the_failure_budget() -> None:
    """Otherwise a feed dropping twice an hour exhausts a daylong run by noon."""
    feed = _Feed([_Dropped("a"), _Dropped("b"), 1, _Dropped("c"), _Dropped("d"), 1])

    report, received = _run(feed, max_consecutive_failures=3)

    assert report.stopped_because == "stopped"
    assert report.failures == 4
    assert len(received) == 2


def test_a_silent_session_also_resets_the_failure_budget() -> None:
    """It delivered nothing, but the transport worked, which is what was in doubt."""
    feed = _Feed([_Dropped("a"), _Dropped("b"), 0, _Dropped("c"), _Dropped("d"), 1])

    report, _ = _run(feed, max_consecutive_failures=3)

    assert report.stopped_because == "stopped"
    assert report.failures == 4


class _ClosableFeed(_Feed):
    """A feed whose stream holds a transport that has to be handed back.

    ``_Feed`` deliberately has no ``close``, so every other test in this file
    also covers the case the supervisor must leave alone: a replay or a file
    stream with nothing to release.
    """

    def __init__(
        self,
        script: Iterable[int | BaseException],
        open_errors: Iterable[BaseException] = (),
        close_error: BaseException | None = None,
    ) -> None:
        super().__init__(script, open_errors)
        self.closes = 0
        self._close_error = close_error

    def close(self) -> None:
        self.closes += 1
        if self._close_error is not None:
            raise self._close_error


def test_a_stream_discarded_after_a_failure_is_handed_back() -> None:
    """Groww's SDK opens a websocket, an event loop and a thread per stream and
    closes none of them, so dropping the reference leaks all three. A feed that
    fails every half hour leaks a dozen over a trading day.
    """
    feed = _ClosableFeed([2, _Dropped("socket closed"), 3])

    report, _ = _run(feed)

    # Once for the failed session, once for the stream still open at the end.
    assert feed.closes == 2
    assert report.failures == 1
    assert feed.opens == 2


def test_a_stream_discarded_after_a_silent_session_is_handed_back() -> None:
    """The commoner leak of the two: a quiet market rebuilds on a timer."""
    feed = _ClosableFeed([0, 0, 1])

    report, _ = _run(feed)

    assert report.silent_sessions == 2
    assert feed.closes == 3
    assert feed.opens == 3


def test_the_last_stream_is_handed_back_when_the_run_ends() -> None:
    """Nothing goes wrong here, which is the point.

    A driver calling ``run`` once per trading day would otherwise leak one live
    websocket a day on the path that always executes.
    """
    feed = _ClosableFeed([1, 1, 1])

    report, _ = _run(feed)

    assert report.stopped_because == "stopped"
    assert report.failures == 0
    assert feed.opens == 1
    assert feed.closes == 1


def test_a_transport_that_cannot_be_released_does_not_disturb_the_run() -> None:
    """Release happens where the loop can least afford a new exception: mid-retry,
    or on the way out with a report to return. A transport that will not shut
    down cleanly has already said what it is going to.
    """
    feed = _ClosableFeed([0, 2], close_error=RuntimeError("socket already gone"))

    report, received = _run(feed)

    assert len(received) == 2
    assert report.stopped_because == "stopped"
    assert report.failures == 0
    assert feed.closes == 2


def test_a_stream_with_nothing_to_release_is_left_alone() -> None:
    """Closing is a property of a transport, not of being a bounded stream."""
    feed = _Feed([1])

    report, received = _run(feed)

    assert not isinstance(feed, ClosableTickStream)
    assert len(received) == 1
    assert report.stopped_because == "stopped"


def test_a_factory_failure_is_retried_like_a_collection_failure() -> None:
    """Reconnecting is calling the factory, so its failures are the same failure."""
    feed = _Feed([2], open_errors=[_Dropped("no route to host")])

    report, received = _run(feed)

    assert len(received) == 2
    assert report.failures == 1
    assert report.last_error == "_Dropped: no route to host"
    assert feed.opens == 1
    assert report.reconnects == 0


def test_the_tick_budget_ends_the_run_without_being_exceeded() -> None:
    feed = _Feed([4, 4, 4])
    received: list[MarketTick] = []
    supervisor = _supervise(feed, received.append)

    report = supervisor.run(duration_seconds=5.0, max_ticks=5)

    assert len(received) == 5
    assert report.ticks == 5
    assert report.stopped_because == "tick_budget"
    # The remaining budget is passed down, so a well-behaved stream stops itself
    # rather than overshooting and having the surplus discarded.
    assert feed.tick_budgets == [5, 1]


def test_the_deadline_ends_a_run_the_script_never_finishes() -> None:
    feed = _Feed([1])
    feed.on_exhausted = None
    received: list[MarketTick] = []
    supervisor = StreamSupervisor(
        feed.open,
        received.append,
        retry_on=(_Dropped,),
        session_seconds=0.01,
    )

    report = supervisor.run(duration_seconds=0.2)

    assert report.stopped_because == "deadline"
    assert report.ticks == 1
    assert report.sessions > 1


def test_backoff_never_sleeps_past_the_deadline() -> None:
    """A thirty-second delay must not hold a run open thirty seconds past its end."""
    feed = _Feed([_Dropped("outage")] * 20)
    feed.on_exhausted = None
    supervisor = StreamSupervisor(
        feed.open,
        lambda tick: None,
        retry_on=(_Dropped,),
        session_seconds=0.01,
        base_delay_seconds=30.0,
        max_delay_seconds=30.0,
    )

    started = monotonic()
    report = supervisor.run(duration_seconds=0.2)
    elapsed = monotonic() - started

    assert report.stopped_because == "deadline"
    assert report.failures == 1
    assert elapsed < 5.0


def test_stop_ends_the_run_and_says_so() -> None:
    received: list[MarketTick] = []

    def stop_after_first(tick: MarketTick) -> None:
        received.append(tick)
        supervisor.stop()

    supervisor = _supervise(_Feed([1, 1, 1]), stop_after_first)

    report = supervisor.run(duration_seconds=5.0)

    assert report.stopped_because == "stopped"
    assert len(received) == 1


def test_a_second_run_starts_its_tick_count_from_zero() -> None:
    supervisor = _supervise(_Feed([2]))
    assert supervisor.run(duration_seconds=5.0).ticks == 2

    with pytest.raises(StreamSupervisorError, match="duration_seconds"):
        supervisor.run(duration_seconds=0)


def test_running_a_supervisor_from_inside_its_own_consumer_is_refused() -> None:
    """Two runs sharing one tick counter would report each other's ticks."""
    feed = _Feed([1])
    supervisor: StreamSupervisor

    def reenter(tick: MarketTick) -> None:
        supervisor.run(duration_seconds=5.0)

    supervisor = _supervise(feed, reenter)

    with pytest.raises(StreamSupervisorError, match="already running"):
        supervisor.run(duration_seconds=5.0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"retry_on": ()}, "retryable error type"),
        ({"session_seconds": 0}, "session_seconds"),
        ({"session_ticks": 0}, "session_ticks"),
        ({"max_consecutive_failures": 0}, "max_consecutive_failures"),
        ({"base_delay_seconds": 0}, "base_delay_seconds"),
        ({"max_delay_seconds": 0.0001}, "max_delay_seconds"),
    ],
)
def test_misconfiguration_is_refused_at_construction(
    options: dict[str, object],
    message: str,
) -> None:
    settings: dict[str, object] = {
        "retry_on": (_Dropped,),
        "base_delay_seconds": 0.5,
    }

    with pytest.raises(StreamSupervisorError, match=message):
        StreamSupervisor(
            _Feed(()).open,
            lambda tick: None,
            **(settings | options),  # type: ignore[arg-type]
        )


def test_a_non_positive_tick_budget_is_refused() -> None:
    supervisor = _supervise(_Feed([1]))

    with pytest.raises(StreamSupervisorError, match="max_ticks"):
        supervisor.run(duration_seconds=5.0, max_ticks=0)
