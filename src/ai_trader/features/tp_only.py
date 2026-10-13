"""Causal compact inputs and price-touch labels for the TP-only research protocol."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, localcontext

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE, ONE_MINUTE, session_minute_offset
from ai_trader.features.indicators import FEATURE_CONTEXT, SessionVwap
from ai_trader.market import Candle

TP_FRACTION = Decimal("0.002")
HORIZON_MINUTES = 60
CUTOFF_TIME = time(15, 15)
CUTOFF_MINUTES = 360
FEATURE_COLUMNS = (
    "log_return_5m",
    "log_return_previous_10m",
    "day_return",
    "gap_from_vendor_daily_close",
    "realized_volatility_20m",
    "prior_5_daily_close_return",
    "prior_5_cutoff_volatility",
    "vwap_z",
    "same_minute_rvol_5",
    "minutes_from_open",
    "benchmark_log_return_15m",
    "gold_log_return_15m",
    "vix_level",
)
MODEL_COLUMNS = (*FEATURE_COLUMNS, "side")


@dataclass(frozen=True, slots=True)
class CompletedSession:
    session_date: date
    daily_close: Decimal | None = None
    cutoff_variance: Decimal | None = None
    volumes: tuple[int | None, ...] = ()


@dataclass(frozen=True, slots=True)
class PriceContext:
    end_time: datetime
    close: Decimal
    log_return_15m: Decimal | None


@dataclass(frozen=True, slots=True)
class TouchLabel:
    hit: bool
    first_hit_minute: int | None
    favorable_excursion: Decimal
    adverse_excursion: Decimal
    terminal_return: Decimal


class TpFeatureState:
    """One instrument/session, using at most six preceding calendar records."""

    def __init__(
        self,
        instrument: Instrument,
        session_date: date,
        prior_sessions: Sequence[CompletedSession] = (),
    ) -> None:
        prior = tuple(prior_sessions[-6:])
        if any(record.session_date >= session_date for record in prior) or any(
            left.session_date >= right.session_date
            for left, right in zip(prior, prior[1:])
        ):
            raise ValueError("Prior sessions must be ordered and strictly in the past")
        self.instrument = instrument
        self.session_date = session_date
        self._prior = prior
        self._closes: deque[Decimal] = deque(maxlen=21)
        self._squared_returns: deque[Decimal] = deque(maxlen=20)
        self._volumes: list[int | None] = []
        self._open: Decimal | None = None
        self._last: Candle | None = None
        self._complete = True
        self._cutoff_variance = Decimal(0)
        self._vwap = SessionVwap()

    def update(self, candle: Candle) -> None:
        """Consume a completed regular one-minute bar no later than 15:15 IST."""
        minute = session_minute_offset(candle.start_time)
        if (
            candle.instrument != self.instrument
            or candle.start_time.astimezone(INDIA_TIMEZONE).date() != self.session_date
            or candle.end_time - candle.start_time != ONE_MINUTE
            or candle.start_time.second != 0
            or candle.start_time.microsecond != 0
            or not 0 <= minute < CUTOFF_MINUTES
            or candle.close <= 0
            or candle.open <= 0
            or (self._last is not None and candle.start_time < self._last.end_time)
        ):
            raise ValueError("Invalid or unordered TP feature candle")
        with localcontext(FEATURE_CONTEXT):
            if self._last is None and minute == 0:
                self._open = candle.open
            consecutive = self._last is not None and (
                candle.start_time == self._last.end_time
            )
            if not consecutive:
                self._closes.clear()
                self._squared_returns.clear()
            if self._closes:
                change = (candle.close / self._closes[-1]).ln()
                self._squared_returns.append(change * change)
            self._closes.append(candle.close)
            if len(self._volumes) != minute:
                self._complete = False
                self._volumes.extend([None] * (minute - len(self._volumes)))
            self._volumes.append(candle.volume)
            if self._complete:
                previous_close = candle.open if self._last is None else self._last.close
                change = (candle.close / previous_close).ln()
                self._cutoff_variance += change * change
            self._vwap.update(
                candle.start_time,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )
            self._last = candle

    def price_context(self) -> PriceContext:
        if self._last is None:
            raise ValueError("No completed candle")
        with localcontext(FEATURE_CONTEXT):
            change = (
                (self._closes[-1] / self._closes[-16]).ln()
                if len(self._closes) >= 16
                else None
            )
        return PriceContext(self._last.end_time, self._last.close, change)

    def values(
        self, context: Mapping[str, PriceContext], *, benchmark: str
    ) -> dict[str, Decimal | None]:
        if self._last is None:
            raise ValueError("No completed candle")
        as_of = self._last.end_time
        if any(item.end_time > as_of for item in context.values()):
            raise ValueError("Future context cannot enter model inputs")
        aligned = {
            name: item for name, item in context.items() if item.end_time == as_of
        }
        with localcontext(FEATURE_CONTEXT):
            last = self._last.close
            minute = session_minute_offset(self._last.start_time)
            prior_close = self._prior[-1].daily_close if self._prior else None
            week_close = self._prior[0].daily_close if len(self._prior) == 6 else None
            prior_variances = [record.cutoff_variance for record in self._prior[-5:]]
            prior_volumes = [
                record.volumes[minute] if minute < len(record.volumes) else None
                for record in self._prior[-5:]
            ]
            mean_volume = (
                Decimal(sum(prior_volumes)) / 5
                if len(prior_volumes) == 5
                and all(value is not None for value in prior_volumes)
                else None
            )
            vwap = self._vwap.value if self._complete else None
            deviation = self._vwap.deviation if self._complete else None
            values = {
                "log_return_5m": (last / self._closes[-6]).ln()
                if len(self._closes) >= 6
                else None,
                "log_return_previous_10m": (self._closes[-6] / self._closes[-16]).ln()
                if len(self._closes) >= 16
                else None,
                "day_return": last / self._open - 1 if self._open else None,
                "gap_from_vendor_daily_close": self._open / prior_close - 1
                if self._open and prior_close
                else None,
                "realized_volatility_20m": sum(self._squared_returns).sqrt()
                if len(self._squared_returns) == 20
                else None,
                "prior_5_daily_close_return": prior_close / week_close - 1
                if prior_close
                and week_close
                and all(record.daily_close is not None for record in self._prior)
                else None,
                "prior_5_cutoff_volatility": (sum(prior_variances) / 5).sqrt()
                if len(prior_variances) == 5
                and all(value is not None for value in prior_variances)
                else None,
                "vwap_z": (last - vwap) / deviation
                if vwap is not None and deviation
                else None,
                "same_minute_rvol_5": Decimal(self._last.volume) / mean_volume
                if self._last.volume is not None and mean_volume
                else None,
                "minutes_from_open": Decimal(session_minute_offset(as_of)),
                "benchmark_log_return_15m": aligned[benchmark].log_return_15m
                if benchmark in aligned
                else None,
                "gold_log_return_15m": aligned["GOLDBEES"].log_return_15m
                if "GOLDBEES" in aligned
                else None,
                "vix_level": aligned["INDIAVIX"].close
                if "INDIAVIX" in aligned
                else None,
            }
        return values

    def finish(self, daily_close: Decimal | None) -> CompletedSession:
        """Finalize for a subsequent session; never consumed as current-day input."""
        if daily_close is not None and (
            not daily_close.is_finite() or daily_close <= 0
        ):
            raise ValueError("Invalid vendor daily close")
        variance = (
            self._cutoff_variance
            if self._complete and len(self._volumes) == CUTOFF_MINUTES
            else None
        )
        return CompletedSession(
            self.session_date, daily_close, variance, tuple(self._volumes)
        )


def touch_label(signal: Candle, subsequent: Sequence[Candle], side: int) -> TouchLabel:
    """Target touch within all 60 subsequent minutes, regardless of adverse moves."""
    if side not in (-1, 1) or signal.close <= 0:
        raise ValueError("Invalid direction or reference price")
    decision = signal.end_time
    deadline = decision + HORIZON_MINUTES * ONE_MINUTE
    cutoff = datetime.combine(decision.date(), CUTOFF_TIME, INDIA_TIMEZONE)
    future = tuple(subsequent[:HORIZON_MINUTES])
    if deadline > cutoff or len(future) != HORIZON_MINUTES:
        raise ValueError("Incomplete future horizon before cutoff")
    if any(
        candle.instrument != signal.instrument
        or candle.start_time != decision + offset * ONE_MINUTE
        or candle.end_time != candle.start_time + ONE_MINUTE
        for offset, candle in enumerate(future)
    ):
        raise ValueError("Future horizon contains a missing minute or wrong instrument")
    with localcontext(FEATURE_CONTEXT):
        target = signal.close * (1 + side * TP_FRACTION)
        first = next(
            (
                offset + 1
                for offset, candle in enumerate(future)
                if (candle.high >= target if side == 1 else candle.low <= target)
            ),
            None,
        )
        high = max(candle.high for candle in future) / signal.close - 1
        low = min(candle.low for candle in future) / signal.close - 1
        return TouchLabel(
            first is not None,
            first,
            max(Decimal(0), high if side == 1 else -low),
            max(Decimal(0), -low if side == 1 else high),
            side * (future[-1].close / signal.close - 1),
        )
