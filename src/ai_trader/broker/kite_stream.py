"""Bounded Kite full-mode ticks with exchange time and cumulative volume."""

from __future__ import annotations

import ssl as system_ssl
from collections.abc import Callable, Sequence
from datetime import datetime
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from ai_trader.broker import MarketTick
from ai_trader.broker.kite import (
    KiteInstrument,
    KiteStreamConnectionError,
    KiteStreamError,
    _price,
)
from ai_trader.clock import INDIA_TIMEZONE


def _connect_ticker(ticker: Any) -> None:
    """Reuse Kite's protocol with verified, hostname-aware platform TLS."""
    from autobahn.twisted.websocket import connectWS
    from twisted.internet import reactor, ssl

    endpoint = urlsplit(ticker.socket_url)
    if (
        endpoint.scheme != "wss"
        or endpoint.hostname != "ws.kite.trade"
        or endpoint.port not in (None, 443)
    ):
        raise KiteStreamError("Unexpected Kite stream endpoint.")
    authorities = [
        ssl.Certificate.loadPEM(system_ssl.DER_cert_to_PEM_cert(value).encode("ascii"))
        for value in system_ssl.create_default_context().get_ca_certs(binary_form=True)
    ]
    if not authorities:
        raise KiteStreamError("No system certificate authorities are available.")
    context = ssl.optionsForClientTLS(
        endpoint.hostname, trustRoot=ssl.trustRootFromCertificates(authorities)
    )
    ticker._create_connection(
        ticker.socket_url,
        useragent=ticker._user_agent(),
        headers={"X-Kite-Version": "3"},
    )
    ticker.factory.maxRetries = 0
    connectWS(ticker.factory, contextFactory=context, timeout=ticker.connect_timeout)
    if not reactor.running:
        ticker.websocket_thread = Thread(
            target=reactor.run, kwargs={"installSignalHandlers": False}, daemon=True
        )
        ticker.websocket_thread.start()


class KiteLtpStream:
    def __init__(
        self,
        ticker: Any,
        instruments: Sequence[KiteInstrument],
        *,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        strict: bool = True,
        queue_size: int = 10000,
    ) -> None:
        if not instruments or len(instruments) > 3000:
            raise ValueError("A Kite stream requires 1 to 3000 instruments.")
        self._ticker = ticker
        self._instruments = {item.instrument_token: item for item in instruments}
        if len(self._instruments) != len(instruments):
            raise ValueError("Streaming instruments must be unique.")
        self._dispatch = dispatch or (lambda operation: operation())
        # Tolerant streams skip (and count) a bad row or a full buffer instead of
        # failing the whole feed.
        self._strict = strict
        self.rejected = 0
        self.dropped = 0
        self._queue: Queue[MarketTick | None] = Queue(maxsize=queue_size)
        self._ready = Event()
        self._closed = Event()
        self._lock = Lock()
        self._collecting = Lock()
        self._failure: KiteStreamError | None = None
        ticker.on_connect = self._on_connect
        ticker.on_ticks = self._on_ticks
        ticker.on_close = self._on_error
        ticker.on_error = self._on_error

    def connect(self, timeout_seconds: float) -> KiteLtpStream:
        if timeout_seconds <= 0:
            raise ValueError("Stream connection timeout must be positive.")
        try:
            _connect_ticker(self._ticker)
            if not self._ready.wait(timeout_seconds):
                raise KiteStreamConnectionError("Kite stream connection timed out.")
            self._raise_failure()
            return self
        except KiteStreamError:
            self.close()
            raise
        except Exception:
            self.close()
            raise KiteStreamConnectionError("Kite stream connection failed.") from None

    def _fail(self, error: KiteStreamError) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error
        self._ready.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            pass

    def _raise_failure(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure from None

    def _on_connect(self, ticker: Any, response: Any) -> None:
        if self._closed.is_set():
            ticker.close()
            return
        try:
            tokens = list(self._instruments)
            ticker.subscribe(tokens)
            ticker.set_mode(ticker.MODE_FULL, tokens)
            self._ready.set()
        except Exception:
            self._fail(KiteStreamConnectionError("Kite stream subscription failed."))

    def _on_error(self, ticker: Any, code: Any, reason: Any) -> None:
        if not self._closed.is_set():
            self._fail(KiteStreamConnectionError("Kite stream disconnected."))

    def _on_ticks(self, ticker: Any, rows: list[dict]) -> None:
        if self._closed.is_set():
            return
        try:
            for row in rows:
                try:
                    tick = self._tick(row)
                except Exception:
                    if self._strict:
                        raise
                    self.rejected += 1
                    continue
                if tick is None:
                    continue
                try:
                    self._queue.put_nowait(tick)
                except Full:
                    if self._strict:
                        raise
                    self.dropped += 1
        except Exception:
            self._fail(
                KiteStreamError("Kite stream schema or buffer validation failed.")
            )

    def _tick(self, row: dict) -> MarketTick | None:
        instrument = self._instruments.get(row["instrument_token"])
        if instrument is None:
            return None
        stamp = row["exchange_timestamp"]
        if not isinstance(stamp, datetime) or not 2000 <= stamp.year <= 2100:
            raise ValueError
        stamp = stamp.astimezone(INDIA_TIMEZONE)
        volume = row.get("volume_traded")
        if volume is None and row.get("tradable") is not False:
            raise ValueError
        if volume is not None and (type(volume) is not int or volume < 0):
            raise ValueError
        return MarketTick(
            instrument.instrument,
            _price(row["last_price"]),
            stamp,
            volume,
        )

    def collect(
        self,
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None] | None = None,
    ) -> tuple[MarketTick, ...]:
        if max_ticks <= 0 or timeout_seconds < 0:
            raise ValueError("Tick budget must be positive and timeout nonnegative.")
        if self._closed.is_set() or not self._collecting.acquire(blocking=False):
            raise KiteStreamError("Kite stream is closed or already collecting.")
        deadline = monotonic() + timeout_seconds
        ticks = []
        try:
            while len(ticks) < max_ticks:
                self._raise_failure()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    tick = self._queue.get(timeout=remaining)
                except Empty:
                    break
                if tick is None:
                    self._raise_failure()
                    break
                ticks.append(tick)
                if on_tick is not None:
                    on_tick(tick)
            self._raise_failure()
            return tuple(ticks)
        finally:
            self._collecting.release()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            pass
        try:
            self._dispatch(self._ticker.close)
        except Exception:
            pass

    def __enter__(self) -> KiteLtpStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
