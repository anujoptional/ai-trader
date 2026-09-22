"""Supervised, reconnecting consumption of a bounded tick stream.

A broker stream call is bounded by design: it subscribes, collects until a tick
count or a timeout is reached, unsubscribes, and returns. That shape is right
for a probe and wrong for a trading session, which needs ticks to keep arriving
from 09:15 to 15:30 whatever the network does in between. This module is the
loop around it.

Reconnection is not only about errors. A socket that dies quietly raises
nothing: the subscription stays nominally alive, no frames arrive, and the
bounded call simply returns having collected nothing. That silent stall is the
failure a daylong run is most likely to meet, so a session ending with no ticks
delivered rebuilds the stream rather than waiting for an error that is never
coming. The session length is therefore also the stall-detection window, and
picking it trades two things against each other: every session boundary
unsubscribes and resubscribes, so ticks arriving in that gap are lost, and
shorter sessions mean more gaps. A minute keeps the loss to a handful of seams
per hour while still noticing a dead feed inside one candle.

Retryability is the caller's to declare, through ``retry_on``. This module
cannot infer it: a dropped connection and a bug in the consumer's callback both
surface as an exception from the same call, and retrying the second one replays
a deterministic failure until the day ends. Anything outside ``retry_on``
propagates, which is what makes a consumer bug loud rather than a statistic.

The supervisor holds no broker import and talks only to the ``TickStream``
protocol and a factory that returns one, so a reconnect is nothing more than
calling the factory again.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from time import monotonic
from typing import Literal, Protocol, runtime_checkable

from ai_trader.broker import MarketTick

_LOGGER = logging.getLogger(__name__)

DEFAULT_SESSION_SECONDS = 60.0
"""How long one bounded collection runs before the loop takes stock.

Long enough that a day's worth of subscribe/unsubscribe seams stays in the
hundreds, short enough that a silently dead feed is noticed within one candle.
"""

DEFAULT_SESSION_TICKS = 100_000
"""A per-session tick ceiling that a real session is not expected to reach.

The bounded call needs some count limit, but the limit that should end a session
is the clock. This is set far above any plausible minute of NSE cash ticks so it
acts as a backstop against a runaway feed rather than as a routine stop.
"""

DEFAULT_MAX_CONSECUTIVE_FAILURES = 10
"""Retryable failures in a row before the run stops trying.

Without a bound a permanently broken feed -- a revoked entitlement, a symbol the
broker no longer serves -- spins until the market closes and reports a day of
work that never happened. With the default backoff, ten in a row spans at least
two and a half minutes of waiting plus however long each failed attempt itself
takes, which is comfortably longer than the outages Groww's REST path was
measured to produce.
"""

DEFAULT_BASE_DELAY_SECONDS = 1.0
"""The first reconnect delay, doubling with each consecutive failure."""

DEFAULT_MAX_DELAY_SECONDS = 30.0
"""The ceiling on reconnect delay, so a long outage still retries twice a minute."""

_MAX_BACKOFF_EXPONENT = 30
"""Caps the doubling so a large failure budget cannot overflow the delay."""

StreamStopReason = Literal["deadline", "tick_budget", "stopped", "gave_up"]
"""Why ``run`` returned: time ran out, the tick budget was met, ``stop`` was
called, or too many retryable failures arrived in a row."""


class StreamSupervisorError(RuntimeError):
    """Raised when a supervisor is misconfigured or misused."""


class TickStream(Protocol):
    """A bounded tick collection, as the broker layer exposes it."""

    def collect(
        self,
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None] | None = None,
    ) -> tuple[MarketTick, ...]:
        """Collect ticks until the count or timeout limit is reached."""
        ...


@runtime_checkable
class ClosableTickStream(TickStream, Protocol):
    """A tick stream holding a transport that has to be handed back.

    Optional rather than folded into ``TickStream`` because closing is a
    property of a particular transport, not of being a bounded stream. A stream
    reading from a file or a replay has nothing to release, and requiring a
    no-op ``close`` from it would be ceremony. The supervisor checks at runtime
    and closes whatever offers it.
    """

    def close(self) -> None:
        """Release the underlying transport, without raising."""
        ...


@dataclass(frozen=True, slots=True)
class StreamReport:
    """What a supervised run did, including the parts that went wrong.

    A run that loses its connection twice and recovers is a success, and a run
    that gave up after ten failures is not, but both return rather than raise.
    The difference is only legible if the counts come back with the result, so
    they do.
    """

    ticks: int
    sessions: int
    silent_sessions: int
    reconnects: int
    failures: int
    stopped_because: StreamStopReason
    last_error: str | None


class StreamSupervisor:
    """Keep a bounded tick stream running for the length of a trading session."""

    def __init__(
        self,
        open_stream: Callable[[], TickStream],
        on_tick: Callable[[MarketTick], None],
        *,
        retry_on: tuple[type[BaseException], ...],
        session_seconds: float = DEFAULT_SESSION_SECONDS,
        session_ticks: int = DEFAULT_SESSION_TICKS,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
        base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
        max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    ) -> None:
        if not retry_on:
            raise StreamSupervisorError(
                "At least one retryable error type is required."
            )
        if session_seconds <= 0:
            raise StreamSupervisorError("session_seconds must be positive.")
        if session_ticks <= 0:
            raise StreamSupervisorError("session_ticks must be positive.")
        if max_consecutive_failures <= 0:
            raise StreamSupervisorError("max_consecutive_failures must be positive.")
        if base_delay_seconds <= 0:
            raise StreamSupervisorError("base_delay_seconds must be positive.")
        if max_delay_seconds < base_delay_seconds:
            raise StreamSupervisorError(
                "max_delay_seconds cannot be below base_delay_seconds."
            )

        self._open_stream = open_stream
        self._on_tick = on_tick
        self._retry_on = retry_on
        self._session_seconds = session_seconds
        self._session_ticks = session_ticks
        self._max_consecutive_failures = max_consecutive_failures
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._stopping = Event()
        self._running = False
        self._ticks = 0
        self._lock = Lock()

    @property
    def tick_count(self) -> int:
        """Ticks delivered to the consumer so far in the current or last run."""
        with self._lock:
            return self._ticks

    def stop(self) -> None:
        """Ask a running supervisor to return after the current session.

        Safe to call from another thread or a signal handler. A reconnect delay
        is interrupted immediately rather than slept out, so a stop during an
        outage does not wait the full backoff.
        """
        self._stopping.set()

    def run(
        self,
        duration_seconds: float,
        max_ticks: int | None = None,
    ) -> StreamReport:
        """Consume ticks until the duration elapses, stopping early on request.

        Returns rather than raises when the feed fails repeatedly: the caller
        decides what an interrupted session is worth. Errors outside ``retry_on``
        do propagate, because they are bugs rather than weather.
        """
        if duration_seconds <= 0:
            raise StreamSupervisorError("duration_seconds must be positive.")
        if max_ticks is not None and max_ticks <= 0:
            raise StreamSupervisorError("max_ticks must be positive.")

        with self._lock:
            if self._running:
                raise StreamSupervisorError("This supervisor is already running.")
            self._running = True
            self._ticks = 0
        self._stopping.clear()

        try:
            return self._run(deadline=monotonic() + duration_seconds, budget=max_ticks)
        finally:
            with self._lock:
                self._running = False

    def _run(self, *, deadline: float, budget: int | None) -> StreamReport:
        sessions = 0
        silent_sessions = 0
        opens = 0
        failures = 0
        consecutive_failures = 0
        last_error: str | None = None
        stream: TickStream | None = None
        stopped_because: StreamStopReason = "deadline"

        try:
            while True:
                if self._stopping.is_set():
                    stopped_because = "stopped"
                    break
                remaining_seconds = deadline - monotonic()
                if remaining_seconds <= 0:
                    stopped_because = "deadline"
                    break
                session_ticks = self._session_ticks
                if budget is not None:
                    session_ticks = min(session_ticks, budget - self.tick_count)
                    if session_ticks <= 0:
                        stopped_because = "tick_budget"
                        break

                delivered_before = self.tick_count
                try:
                    if stream is None:
                        stream = self._open_stream()
                        opens += 1
                    stream.collect(
                        max_ticks=session_ticks,
                        timeout_seconds=min(self._session_seconds, remaining_seconds),
                        on_tick=self._deliver,
                    )
                except self._retry_on as error:
                    # The stream is discarded whichever call failed. A factory
                    # that raised produced nothing to keep, and a collection
                    # that raised may leave a subscription the next call cannot
                    # clear.
                    self._release(stream)
                    stream = None
                    failures += 1
                    consecutive_failures += 1
                    last_error = _describe(error)
                    if consecutive_failures >= self._max_consecutive_failures:
                        stopped_because = "gave_up"
                        break
                    self._wait_before_retry(consecutive_failures, deadline)
                    continue

                sessions += 1
                # Reset on any session that completed, including a silent one:
                # the transport worked, so the earlier failures were an outage
                # rather than the feed being permanently gone. Without this a
                # stream that drops every half hour ends a six-hour day at
                # maximum backoff.
                consecutive_failures = 0
                if self.tick_count == delivered_before:
                    silent_sessions += 1
                    self._release(stream)
                    stream = None
        finally:
            # Also on the way out, including the normal end of a run. Every
            # exit above leaves the last stream open otherwise, and a daily
            # driver that calls run once per session would leak one transport
            # a day even if nothing ever went wrong.
            self._release(stream)

        return StreamReport(
            ticks=self.tick_count,
            sessions=sessions,
            silent_sessions=silent_sessions,
            reconnects=max(opens - 1, 0),
            failures=failures,
            stopped_because=stopped_because,
            last_error=last_error,
        )

    def _release(self, stream: TickStream | None) -> None:
        """Hand a stream's transport back, if it holds one.

        Swallows whatever close raises. A stream is released at the moments the
        loop is least able to cope with a new exception -- mid-retry, or on the
        way out with a report to return -- and a transport that cannot be shut
        down cleanly has already told us what it is going to.
        """
        if not isinstance(stream, ClosableTickStream):
            return
        try:
            stream.close()
        except Exception as error:
            _LOGGER.debug("Releasing a tick stream failed: %s", _describe(error))

    def _deliver(self, tick: MarketTick) -> None:
        """Count a tick, then hand it to the consumer.

        Counting first means a session that fails partway through still reports
        the ticks the consumer already saw, rather than discarding them along
        with the collection that raised.
        """
        with self._lock:
            self._ticks += 1
        self._on_tick(tick)

    def _wait_before_retry(self, consecutive_failures: int, deadline: float) -> None:
        """Back off before reconnecting, without sleeping past the deadline."""
        delay_seconds = min(
            self._base_delay_seconds
            * 2 ** min(consecutive_failures - 1, _MAX_BACKOFF_EXPONENT),
            self._max_delay_seconds,
        )
        remaining_seconds = deadline - monotonic()
        if remaining_seconds <= 0:
            return
        self._stopping.wait(min(delay_seconds, remaining_seconds))


def _describe(error: BaseException) -> str:
    """Summarize an error for a report, without assuming it has a message."""
    text = str(error).strip()
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


__all__ = [
    "DEFAULT_BASE_DELAY_SECONDS",
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "DEFAULT_MAX_DELAY_SECONDS",
    "DEFAULT_SESSION_SECONDS",
    "DEFAULT_SESSION_TICKS",
    "ClosableTickStream",
    "StreamReport",
    "StreamStopReason",
    "StreamSupervisor",
    "StreamSupervisorError",
    "TickStream",
]
