"""Bounded, read-only Groww LTP streaming."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from threading import Event, Lock
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.broker import MarketTick
from ai_trader.broker.groww import (
    GrowwInstrument,
    GrowwStreamConnectionError,
    GrowwStreamError,
    _groww_epoch_datetime,
)

_LOGGER = logging.getLogger(__name__)

_CLOSE_TIMEOUT_SECONDS = 5.0
"""How long ``close`` waits for the vendor's websocket to shut down.

Bounded because closing runs on paths that are already handling a failure --
a supervisor discarding a stalled stream, a session ending. A transport wedged
badly enough not to answer in five seconds is exactly the one a caller must not
be blocked behind, and the thread holding it is a daemon that dies with the
process regardless.
"""


class _FeedClient(Protocol):
    def subscribe_ltp(
        self,
        instrument_list: list[dict[str, str]],
        on_data_received: Callable[[dict[str, Any]], None] | None = None,
    ) -> object:
        """Subscribe to LTP updates."""
        ...

    def unsubscribe_ltp(
        self,
        instrument_list: list[dict[str, str]],
    ) -> dict[str, bool]:
        """Unsubscribe from LTP updates."""
        ...

    def get_ltp(self) -> dict[str, Any]:
        """Return the latest raw LTP feed state."""
        ...


class _GrowwTickPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: int = Field(alias="tsInMillis")
    price: Decimal = Field(alias="ltp")
    cumulative_volume: Decimal | None = Field(default=None, alias="volume")


def _cumulative_volume(value: Decimal | None) -> int | None:
    """Return session volume as a positive integer, or None if unusable.

    Groww transports volume as a protobuf double, where an unset field is
    indistinguishable from a genuine zero. Accepting that zero would make it
    a baseline for volume differencing and attribute the whole session's
    volume to a single minute, so zero is reported as unknown instead.
    """
    if value is None or not value.is_finite() or value <= 0:
        return None
    return int(value)


def _shutdown_feed(feed: object) -> None:
    """Tear down the NATS transport a ``GrowwFeed`` opened, best effort.

    The SDK offers no way to do this. ``GrowwFeed`` opens a websocket, an
    asyncio loop and a daemon thread in its constructor and exposes no close,
    drain or disconnect of any kind, and it files the client in a class-level
    dict keyed by a JWT built from ``os.urandom(32)`` -- so every feed gets a
    fresh key, the dict only ever grows, and nothing is reclaimable by garbage
    collection even after the last reference to the feed is gone.

    That matters because streams are discarded routinely rather than
    exceptionally. A supervisor rebuilds one after every silent session and
    after every retryable failure, so a quiet afternoon alone can leak dozens
    of live websockets and their threads into a process meant to run all day.

    So this reaches into the vendor's internals deliberately: close the socket
    on the loop that owns it, stop the loop so its thread can exit, and drop
    the registry entry so the graph becomes collectable. Every step is guarded
    and optional. A vendor upgrade that renames any of these attributes turns
    this back into today's leak rather than into a crash, and callers close on
    paths that are already handling failure, where raising would replace a real
    error with a teardown one.
    """
    client = getattr(feed, "_nats_client", None)
    loop = getattr(client, "_loop", None)
    socket = getattr(client, "_socket", None)

    if loop is not None and socket is not None:
        try:
            closing = asyncio.run_coroutine_threadsafe(socket.close(), loop)
            closing.result(timeout=_CLOSE_TIMEOUT_SECONDS)
        except Exception as error:
            # Includes the loop having already stopped, in which case the
            # coroutine is scheduled and never run and this times out. The
            # socket dies with the process either way.
            _LOGGER.debug("Groww feed socket close failed: %s", error)

    if loop is not None:
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception as error:
            _LOGGER.debug("Groww feed loop stop failed: %s", error)

    registry = getattr(type(feed), "_nats_clients", None)
    key = getattr(feed, "_client_key", None)
    if isinstance(registry, dict) and key is not None:
        # Only if it is still ours. The vendor means this dict to be shared,
        # and only a quirk of key generation stops it ever being so; if that
        # is fixed upstream, evicting someone else's live client here would be
        # far worse than leaking ours.
        if registry.get(key) is client:
            registry.pop(key, None)


class GrowwLtpStream:
    """Collect normalized LTP ticks with deterministic stop conditions."""

    def __init__(
        self,
        feed: _FeedClient,
        instruments: Sequence[GrowwInstrument],
    ) -> None:
        if not instruments:
            raise ValueError("At least one streaming instrument is required.")
        self._feed = feed
        self._instruments = tuple(instruments)
        self._by_token = {
            instrument.exchange_token: instrument for instrument in self._instruments
        }
        self._closed = False

    def close(self) -> None:
        """Release the transport this stream's feed opened.

        Idempotent, and never raises: a caller closing a stream is usually
        already dealing with something that went wrong, and a teardown failure
        must not displace it. Collecting after this is a programming error but
        is not policed -- the vendor would simply deliver nothing.
        """
        if self._closed:
            return
        self._closed = True
        _shutdown_feed(self._feed)

    def __enter__(self) -> GrowwLtpStream:
        """Return the stream, so it can be used as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the transport."""
        self.close()

    def collect(
        self,
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None] | None = None,
    ) -> tuple[MarketTick, ...]:
        """Collect ticks until the count or timeout limit is reached."""
        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive.")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative.")

        sdk_instruments = [
            {
                "exchange": resolved.instrument.exchange,
                "segment": "CASH",
                "exchange_token": resolved.exchange_token,
            }
            for resolved in self._instruments
        ]
        ticks: list[MarketTick] = []
        consumer_error: BaseException | None = None
        feed_error: Exception | None = None
        finished = Event()
        closed = Event()
        lock = Lock()

        def on_data_received(meta: dict[str, Any]) -> None:
            nonlocal consumer_error, feed_error
            if closed.is_set():
                return
            reached_limit = False
            try:
                with lock:
                    if len(ticks) >= max_ticks:
                        return
                tick = self._normalize_tick(meta)
                if tick is None:
                    return
                with lock:
                    if len(ticks) >= max_ticks:
                        return
                    ticks.append(tick)
                    reached_limit = len(ticks) >= max_ticks
            except Exception as error:
                # Kept, not merely flagged. This runs on the vendor's thread, so
                # the exception cannot propagate; discarding it would leave the
                # caller with the fixed message below and no way to tell a
                # schema change, which needs code, from a network fault, which
                # needs patience. It is chained onto that message instead.
                with lock:
                    if feed_error is None:
                        feed_error = error
                finished.set()
                return
            if on_tick is not None:
                try:
                    on_tick(tick)
                except BaseException as error:
                    # Kept separate from feed_error, and raised whole rather
                    # than chained onto GrowwStreamError. This is the
                    # consumer's failure, not the feed's, and the two need
                    # different answers: a transport fault is worth retrying
                    # and a bug in a callback replays identically until the day
                    # ends. ai_trader.market.stream decides that by exception
                    # type, so presenting this as a GrowwStreamError would hand
                    # it a deterministic failure dressed as a transport one.
                    #
                    # BaseException, because this captures rather than swallows:
                    # anything escaping here dies on the vendor's thread, which
                    # is precisely the silence being fixed. Only the first is
                    # kept; later callbacks failing the same way are symptoms.
                    with lock:
                        if consumer_error is None:
                            consumer_error = error
                    finished.set()
                    return
            if reached_limit:
                finished.set()

        subscribed = False
        operation_failed = False
        try:
            self._feed.subscribe_ltp(
                sdk_instruments,
                on_data_received=on_data_received,
            )
            subscribed = True
            finished.wait(timeout_seconds)
        except Exception:
            operation_failed = True
        finally:
            closed.set()
            if subscribed:
                try:
                    self._feed.unsubscribe_ltp(sdk_instruments)
                except Exception:
                    operation_failed = True

        with lock:
            # One acquisition, and only after ``closed`` is set. Both failures
            # are written on the vendor's thread, and a callback already past
            # its ``closed`` check can still be landing as we arrive here.
            consumer_failure = consumer_error
            feed_failure = feed_error
            collected = tuple(ticks)

        if consumer_failure is not None:
            # Raised before either failure below. A callback that stopped the
            # collection tends to leave the feed in a state the unsubscribe
            # then trips over, so those can be set as well -- but they are the
            # consequence and this is the cause, and it is the cause a reader
            # needs. Re-raised whole rather than wrapped, so the consumer sees
            # its own exception exactly as if collect had called it inline.
            raise consumer_failure
        if feed_failure is not None:
            # Ranked above the transport flag, because both can be set at once:
            # a frame we could not normalize leaves the feed in a state the
            # unsubscribe may then trip over. Of the two, a feed that has
            # stopped speaking the shape we parse is the one certain to recur,
            # so it decides the class. Calling that pair retryable would hand a
            # supervisor a deterministic failure to replay.
            raise GrowwStreamError("Groww LTP stream failed.") from feed_failure
        if operation_failed:
            raise GrowwStreamConnectionError("Groww LTP stream failed.")
        return collected

    def _normalize_tick(self, meta: Mapping[str, Any]) -> MarketTick | None:
        """Normalize one feed update, or return None if it is not ours.

        Raising and returning ``None`` mean different things. ``None`` is "this
        frame is not ours", which a shared connection produces routinely and
        which must not disturb a collection in flight. Raising is "the feed is
        no longer speaking the shape we parse", which is not survivable by
        skipping, because every subsequent frame will be wrong the same way.

        The messages matter more here than they look. These are raised on the
        vendor's thread, where nothing can catch them, so ``collect`` keeps the
        exception and chains it onto the fixed ``GrowwStreamError`` it raises
        from the caller's thread. Whatever is said here is therefore the only
        detail a reader ever sees, and it has to be enough to tell a schema
        change apart from a network fault, because the first needs code and the
        second needs patience.
        """
        exchange = meta.get("exchange")
        segment = meta.get("segment")
        exchange_token = meta.get("feed_key")
        if not isinstance(exchange_token, str):
            # Not a foreign topic -- those carry a well-formed key we simply do
            # not recognise, and are handled below. A key of the wrong type
            # means the frame's shape itself has changed.
            raise TypeError(
                "Groww feed frame has a non-string feed_key: "
                f"{type(exchange_token).__name__}."
            )

        resolved = self._by_token.get(exchange_token)
        if resolved is None:
            # A shared connection may deliver topics we never subscribed to.
            # Skipping them is safer than discarding an in-flight collection.
            return None
        if exchange != resolved.instrument.exchange or segment != "CASH":
            raise ValueError(
                f"Groww feed frame for {resolved.instrument.trading_symbol} "
                f"arrived as {exchange!r}/{segment!r}, expected "
                f"{resolved.instrument.exchange!r}/'CASH'."
            )

        # Read defensively: get_ltp returns the feed's own live dictionary,
        # which the vendor's thread rewrites in place as quotes arrive. Our
        # entry can be absent here even though the frame announcing it just
        # landed, and indexing would turn that ordinary race into a KeyError --
        # which this module classes as unretryable and would end the day on.
        # Treating it as "not ours yet" drops one tick instead; the next frame
        # for the instrument carries the same total.
        raw_feed = self._feed.get_ltp()
        raw_tick = raw_feed.get(exchange, {}).get(segment, {}).get(exchange_token)
        if raw_tick is None:
            return None
        payload = _GrowwTickPayload.model_validate(raw_tick)
        return MarketTick(
            instrument=resolved.instrument,
            price=payload.price,
            timestamp=_groww_epoch_datetime(payload.timestamp),
            cumulative_volume=_cumulative_volume(payload.cumulative_volume),
        )


__all__ = ["GrowwLtpStream"]
