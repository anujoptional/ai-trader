"""Bounded, read-only Groww LTP streaming."""

from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from threading import Event, Lock
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.broker import MarketTick
from ai_trader.broker.groww import (
    GrowwInstrument,
    GrowwStreamError,
    _groww_epoch_datetime,
)


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
        callback_failed = Event()
        finished = Event()
        closed = Event()
        lock = Lock()

        def on_data_received(meta: dict[str, Any]) -> None:
            if closed.is_set():
                return
            try:
                with lock:
                    if len(ticks) >= max_ticks:
                        return
                tick = self._normalize_tick(meta)
                with lock:
                    if len(ticks) >= max_ticks:
                        return
                    ticks.append(tick)
                    reached_limit = len(ticks) >= max_ticks
                if on_tick is not None:
                    on_tick(tick)
                if reached_limit:
                    finished.set()
            except Exception:
                callback_failed.set()
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

        if operation_failed or callback_failed.is_set():
            raise GrowwStreamError("Groww LTP stream failed.")
        return tuple(ticks)

    def _normalize_tick(self, meta: Mapping[str, Any]) -> MarketTick:
        exchange = meta.get("exchange")
        segment = meta.get("segment")
        exchange_token = meta.get("feed_key")
        if not isinstance(exchange_token, str):
            raise TypeError

        resolved = self._by_token[exchange_token]
        if exchange != resolved.instrument.exchange or segment != "CASH":
            raise ValueError

        raw_feed = self._feed.get_ltp()
        raw_tick = raw_feed[exchange][segment][exchange_token]
        payload = _GrowwTickPayload.model_validate(raw_tick)
        return MarketTick(
            instrument=resolved.instrument,
            price=payload.price,
            timestamp=_groww_epoch_datetime(payload.timestamp),
        )


__all__ = ["GrowwLtpStream"]
