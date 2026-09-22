"""Live cumulative-volume polling for instruments whose ticks carry none.

Groww's price stream advertises an optional volume field but never populates
it, so every live candle closes with ``volume=None`` and every volume-derived
feature stays unavailable for the whole session. The exchange's running session
total is served on the REST quote endpoint instead. Differencing that total
across minute boundaries reconstructs exactly the per-minute volume the
historical feed supplies, which is what ``CumulativeVolumeTracker`` already
does once a tick carries the total.

This module supplies the total. It polls quotes on a fixed cadence, keeps only
the newest reading per instrument, and stamps that reading onto ticks on their
way to the candle builder. Nothing downstream changes: the builder already
reads ``MarketTick.cumulative_volume``, feeds the tracker, and closes each
candle with the minute volume the tracker returns.

A poll is disposable. Only the newest reading matters, so this module adds no
retry of its own: a call that raises is counted and dropped, because the next
poll supersedes it anyway and a failure must degrade volume rather than kill the
price feed. That is not the same as polling without retries. ``GrowwBroker``
wraps every raw call in its own eight-attempt budget, so a quote reaching this
module has already survived Groww's frequent transient 404s, and outright
failures here are rare -- none at all in 269 polls measured live across
2026-09-21 and 2026-09-22.

The broker's retries are absorbed into the poll's duration, which makes the
interval a floor rather than a period: a round takes the interval plus however
long the call itself took. Measured rounds ran 2.1 to 2.7 seconds against a 2.0
second setting, the spread tracking Groww's transient-404 rate on the day.
Staleness is bounded by the round, not the interval, so a caller comparing
configured cadence against boundary accuracy should assume the longer figure.

The cost of polling rather than streaming is a bounded smear at minute
boundaries: a tick is stamped with a total read up to one round earlier, so
trades executed either side of a boundary can be attributed to the adjacent
minute. Nothing is double counted or lost, because every reading is used
exactly once as the close of one minute and the baseline of the next.

That holds only while polls keep landing. A reading that stops being replaced
stops describing the present, and stamping it anyway is worse than not stamping
at all: differencing a frozen total reports zero volume for every minute the
freeze lasts, which downstream is a confident claim that the instrument stopped
trading rather than an admission that volume is unknown. Readings therefore
expire -- see ``DEFAULT_MAX_READING_AGE_SECONDS``.

This module reaches the broker and so must never be imported by the pure
aggregation modules beside it; the dependency runs one way only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, datetime
from threading import Event, Lock, Thread

from ai_trader.broker import Instrument, MarketTick, ReadOnlyBroker
from ai_trader.market._time import INDIA_TIMEZONE, trading_session_date

DEFAULT_POLL_INTERVAL_SECONDS = 2.0
"""Cadence balancing boundary accuracy against the cost of a round.

Two seconds bounds the boundary smear at a few percent of a minute. It is a
floor, not a period: Groww answers a large share of raw quote calls with a
transient 404, and the broker's retry of those is absorbed into the round, so
measured rounds ran 2.1 to 2.7 seconds. Pushing the interval much lower buys
little, because the call duration rather than the wait then dominates.
"""

_STOP_JOIN_TIMEOUT_SECONDS = 10.0

DEFAULT_MAX_READING_AGE_SECONDS = 15.0
"""How old a reading may be and still be stamped onto a tick.

Without a bound, a poller that stops receiving quotes keeps stamping the last
total it saw. Differencing a frozen total yields zero for every minute after
it froze, and zero is not missing data: it is a maximally strong "this
instrument stopped trading" reading, carried by a snapshot whose readiness flag
says the value is good. Bounding the age converts that into the ``None`` the
rest of the stack already knows how to withhold.

Fifteen seconds is derived from live runs on 2026-09-21 and 2026-09-22. A round
took 2.1 to 2.7 seconds, and a reading's age only stops advancing when a poll
fails or is rejected as a regression -- 13 of 269 polls across the two days.
Five consecutive such rounds is roughly 13.5 seconds at the slower round and
vanishingly unlikely, so ordinary flakiness never trips the bound. A quarter of
a minute also means a genuine stall smears at most the one candle it starts in
before volume goes honestly unavailable, rather than fabricating a second
figure.

This is the bound for the default cadence. A caller that polls more slowly gets
``_STALE_READING_ROUNDS`` rounds instead, because staleness is really measured
in rounds missed and a fixed bound below the interval would expire every
reading before its replacement was ever due.
"""

_STALE_READING_ROUNDS = 5
"""How many consecutive missed rounds a slow-polling caller is allowed.

Applied only when it yields a longer bound than the default, so the default
cadence keeps the figure derived above while a caller polling every thirty
seconds is not left expiring every reading before its replacement was due.
"""


class VolumePollerError(RuntimeError):
    """Raised when the poller is asked to do something it cannot."""


@dataclass(frozen=True, slots=True)
class _Reading:
    """The newest accepted cumulative total for one instrument.

    ``session`` is the trading session the reading belongs to. It exists to
    tell a genuine session rollover, where the exchange's running total resets
    to near zero, apart from a quote endpoint briefly serving a stale or corrupt
    smaller number. The first is expected once a day and must be accepted; the
    second must be rejected.

    It is the trading session and not the IST calendar date because the two
    differ for the nine hours before each open, and the poller is meant to be
    running in them: the architecture has it warming up pre-open. A pre-open
    quote serves the *previous* session's closing total, so filing it under the
    calendar date would put it in a session that has not started, and the reset
    at 09:15 would then read as a backwards move inside one session. Every
    reading for the rest of the day would be below that stale high-water mark
    and so rejected in turn, freezing the reading until it aged out and leaving
    volume unavailable for the whole session.

    Rejecting a genuine regression is not a nicety. ``CumulativeVolumeTracker``
    treats any total below the one it holds as a restarted counter and drops its
    baseline, so a single stale reading makes that minute's volume ``None`` --
    and one ``None`` volume disables strict session VWAP for the remainder of
    the day. Groww serves stale totals often enough for this to matter, and
    consistently so: eight of 147 polls on 2026-09-21 moved backwards, then two
    of 52 and three of 70 on 2026-09-22 -- roughly one poll in twenty, every
    time it has been measured.
    """

    cumulative_volume: int
    observed_at: datetime
    session: date


class VolumePoller:
    """Keep each instrument's cumulative session volume fresh by polling.

    The poller owns a daemon thread and is safe to share: ``stamp`` is called
    on the broker's feed thread while the poll loop writes from its own. Use it
    as a context manager so the thread is always stopped, or call ``start`` and
    ``stop`` explicitly.

    It never raises out of the poll loop. Every failure is counted and the last
    message retained, so a caller can surface degraded volume without the price
    path ever being interrupted.
    """

    def __init__(
        self,
        broker: ReadOnlyBroker,
        instruments: Iterable[Instrument],
        interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        max_reading_age_seconds: float | None = None,
    ) -> None:
        tracked = tuple(dict.fromkeys(instruments))
        if not tracked:
            raise VolumePollerError(
                "A volume poller requires at least one instrument to poll."
            )
        if interval_seconds <= 0:
            raise VolumePollerError(
                "The poll interval must be a positive number of seconds."
            )
        if max_reading_age_seconds is None:
            # Derived rather than fixed: staleness is really measured in rounds
            # missed, so a caller polling more slowly than the default must be
            # given proportionally longer or every reading expires before the
            # next is even due. The default cadence keeps the figure above.
            max_reading_age_seconds = max(
                interval_seconds * _STALE_READING_ROUNDS,
                DEFAULT_MAX_READING_AGE_SECONDS,
            )
        elif max_reading_age_seconds <= 0:
            raise VolumePollerError(
                "The maximum reading age must be a positive number of seconds."
            )
        self._broker = broker
        self._instruments = tracked
        self._interval_seconds = float(interval_seconds)
        self._max_reading_age_seconds = float(max_reading_age_seconds)
        self._readings: dict[Instrument, _Reading] = {}
        self._lock = Lock()
        self._stopping = Event()
        self._thread: Thread | None = None
        self._poll_count = 0
        self._failure_count = 0
        self._regression_count = 0
        self._stale_stamp_count = 0
        self._last_error: str | None = None

    @property
    def instruments(self) -> tuple[Instrument, ...]:
        """Return the instruments this poller tracks, in registration order."""
        return self._instruments

    @property
    def interval_seconds(self) -> float:
        """Return the cadence between poll rounds, in seconds."""
        return self._interval_seconds

    @property
    def max_reading_age_seconds(self) -> float:
        """Return how old a reading may be and still be stamped."""
        return self._max_reading_age_seconds

    @property
    def is_running(self) -> bool:
        """Return whether the poll thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def poll_count(self) -> int:
        """Return how many individual quote calls have been attempted."""
        with self._lock:
            return self._poll_count

    @property
    def failure_count(self) -> int:
        """Return how many quote calls failed and were dropped."""
        with self._lock:
            return self._failure_count

    @property
    def regression_count(self) -> int:
        """Return how many readings moved backwards within one session."""
        with self._lock:
            return self._regression_count

    @property
    def stale_stamp_count(self) -> int:
        """Return how many ticks went unstamped because the reading was old."""
        with self._lock:
            return self._stale_stamp_count

    @property
    def last_error(self) -> str | None:
        """Return the most recent poll failure message, if any."""
        with self._lock:
            return self._last_error

    def latest(self, instrument: Instrument) -> int | None:
        """Return the newest accepted cumulative total, or ``None``."""
        with self._lock:
            reading = self._readings.get(instrument)
            return None if reading is None else reading.cumulative_volume

    def latest_observed_at(self, instrument: Instrument) -> datetime | None:
        """Return when the newest accepted total was read, or ``None``."""
        with self._lock:
            reading = self._readings.get(instrument)
            return None if reading is None else reading.observed_at

    def stamp(self, tick: MarketTick) -> MarketTick:
        """Return ``tick`` carrying the newest cumulative total.

        A tick that already carries a total is returned untouched: a value the
        broker itself reported is always better than a polled one, and
        overwriting it would hide a stream that started serving volume. A tick
        for an untracked instrument, or one polled before any reading has
        landed, is also returned untouched, which leaves the candle's volume
        ``None`` exactly as it is today.

        So is a tick whose newest reading is older than
        ``max_reading_age_seconds``. A reading only ages when polling stops
        succeeding, and stamping a frozen total does not preserve the last known
        volume -- it reports zero volume for every minute the freeze lasts,
        because differencing a constant gives zero. Withholding the stamp is
        what makes a stalled poller show up downstream as unavailable rather
        than as an instrument that suddenly stopped trading.
        """
        if tick.cumulative_volume is not None:
            return tick
        # One acquisition: value and timestamp must describe the same reading,
        # since the poll thread can replace it between two separate reads.
        with self._lock:
            reading = self._readings.get(tick.instrument)
        if reading is None:
            return tick
        age_seconds = (
            datetime.now(tz=INDIA_TIMEZONE) - reading.observed_at
        ).total_seconds()
        if age_seconds > self._max_reading_age_seconds:
            with self._lock:
                self._stale_stamp_count += 1
            return tick
        return replace(tick, cumulative_volume=reading.cumulative_volume)

    def poll_once(self) -> int:
        """Poll every tracked instrument once and return how many succeeded.

        Exposed separately from the thread so a caller can drive the poller
        synchronously, which is what the tests do.
        """
        succeeded = 0
        for instrument in self._instruments:
            if self._poll_instrument(instrument):
                succeeded += 1
        return succeeded

    def start(self) -> None:
        """Start the poll thread, polling once inline before returning.

        The inline poll means ``latest`` is usually populated by the time the
        first tick arrives, so the first live minute is not wasted on a warm-up
        that the tracker would then have to baseline against.
        """
        if self.is_running:
            raise VolumePollerError("The volume poller is already running.")
        self._stopping.clear()
        self.poll_once()
        thread = Thread(
            target=self._run,
            name="volume-poller",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop(self, timeout_seconds: float = _STOP_JOIN_TIMEOUT_SECONDS) -> None:
        """Signal the poll thread to finish and wait for it.

        The handle is kept if the join times out. A round can outlast the
        timeout, because the broker absorbs its own retry budget into each call
        and a round covers every instrument. Clearing the handle regardless
        would report the poller stopped while the thread was still writing
        readings, and the next ``start`` would then run two poll threads against
        the same state. Keeping it leaves ``is_running`` true so ``start``
        refuses, and a later ``stop`` can join the same thread again.
        """
        self._stopping.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            return
        self._thread = None

    def __enter__(self) -> VolumePoller:
        """Start polling and return the poller."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop polling."""
        self.stop()

    def _run(self) -> None:
        while not self._stopping.wait(self._interval_seconds):
            for instrument in self._instruments:
                # Checked per instrument rather than per round: a round polls
                # every instrument, and each call carries the broker's retry
                # budget, so a whole round can take far longer than a caller
                # waiting on ``stop`` should have to block for.
                if self._stopping.is_set():
                    break
                self._poll_instrument(instrument)

    def _poll_instrument(self, instrument: Instrument) -> bool:
        """Poll one instrument and record the outcome, without ever raising.

        The whole body is inside the guard, not just the broker call. ``_run``
        has no guard of its own, so anything escaping here would end the poll
        thread silently and strand volume for the rest of the session -- the
        one failure mode this class exists to prevent, and the hardest to
        notice, because a dead thread reports no error at all.

        The attempt is counted up front so that the count stays exact wherever
        the body fails.
        """
        with self._lock:
            self._poll_count += 1
        try:
            quote = self._broker.get_quote(instrument)
            observed_at = datetime.now(tz=INDIA_TIMEZONE)
            return self._accept(instrument, quote.volume, observed_at)
        except Exception as error:
            # Deliberately broad: a poll failure must degrade volume, never
            # escape onto the caller's thread or stop the loop.
            with self._lock:
                self._failure_count += 1
                self._last_error = f"{type(error).__name__}: {error}"
            return False

    def _accept(
        self,
        instrument: Instrument,
        cumulative_volume: int,
        observed_at: datetime,
    ) -> bool:
        session = trading_session_date(observed_at)
        with self._lock:
            previous = self._readings.get(instrument)
            same_session = previous is not None and previous.session == session
            if (
                same_session
                and previous is not None
                and cumulative_volume < previous.cumulative_volume
            ):
                self._regression_count += 1
                return False
            self._readings[instrument] = _Reading(
                cumulative_volume=cumulative_volume,
                observed_at=observed_at,
                session=session,
            )
        return True


__all__ = [
    "DEFAULT_MAX_READING_AGE_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "VolumePoller",
    "VolumePollerError",
]
