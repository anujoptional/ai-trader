"""Incremental quantitative features derived from completed one-minute candles.

``MarketState`` answers "what happened"; this answers "what does it mean
numerically". The engine consumes the same ``Candle`` objects whether they came
from a historical Groww backfill, a replay, or a live ``CandleBuilder``, and it
has exactly one calculation path for all three. A warm-up is not a special mode,
it is simply the earlier part of the same candle sequence — which is what makes
a feature computed at 09:45 during warm-up identical to the same feature
computed at 09:45 live.

State is per-instrument and strictly isolated: instruments share no indicator,
no window, and no session. Each instrument's memory is bounded and constant —
roughly fifteen closes, twenty highs, twenty lows, twenty volumes, two short EMA
histories and a dozen scalars — so hundreds of instruments cost a predictable
amount however long the engine runs.

Candles must arrive in increasing start-time order per instrument. A repeated or
older candle is counted and discarded rather than folded in, because Wilder and
EMA smoothing have no inverse: a single stale candle would silently bias every
subsequent value with no way to detect or undo it. The engine is thread-safe,
since ``MarketState`` invokes its ``on_candle`` hook on a broker feed thread.

What a new session resets is a deliberate choice, and it is deliberately not
uniform. VWAP and the relative-volume baseline reset at every IST date change,
because both describe turnover *within* a session and yesterday's turnover says
nothing about today's. Price history does not reset: returns, the EMAs, RSI,
MACD and ATR all continue across the boundary, so the first candle of a session
reports an overnight gap rather than nothing at all. That is usually what a
scanner wants, but it does mean ``return_1`` on the 09:15 candle is a gap and
not a one-minute move, and that ATR's first true range of the day includes the
gap against yesterday's close. A consumer that wants intraday-only momentum
must use a fresh engine per session.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, localcontext
from threading import Lock

from ai_trader.broker import Instrument
from ai_trader.features.indicators import (
    FEATURE_CONTEXT,
    AverageTrueRange,
    ExponentialMovingAverage,
    MovingAverageConvergenceDivergence,
    RelativeStrengthIndex,
    SessionVwap,
    ratio_change,
    safe_divide,
    session_date,
)
from ai_trader.features.models import FeatureReadiness, FeatureSnapshot
from ai_trader.market.candles import Candle

_ROLLING_WINDOW = 20
"""Length of the rolling extreme and relative-volume windows, in candles."""

_SLOPE_LAG = 5
"""How many completed candles back an EMA slope reaches."""

_SLOPE_HISTORY = _SLOPE_LAG + 1
"""Retained EMA values: the lagged one, the current one, and those between."""

_CLOSE_HISTORY = 15
"""Retained previous closes, set by the longest return period."""


class _InstrumentFeatures:
    """Every indicator and bounded window backing a single instrument."""

    __slots__ = (
        "atr",
        "closes",
        "ema21",
        "ema21_history",
        "ema50",
        "ema9",
        "ema9_history",
        "highs",
        "last_candle_start",
        "lows",
        "macd",
        "rsi",
        "session",
        "volumes",
        "vwap",
    )

    def __init__(self) -> None:
        self.closes: deque[Decimal] = deque(maxlen=_CLOSE_HISTORY)
        self.highs: deque[Decimal] = deque(maxlen=_ROLLING_WINDOW)
        self.lows: deque[Decimal] = deque(maxlen=_ROLLING_WINDOW)
        self.volumes: deque[int | None] = deque(maxlen=_ROLLING_WINDOW)
        self.ema9 = ExponentialMovingAverage(9)
        self.ema21 = ExponentialMovingAverage(21)
        self.ema50 = ExponentialMovingAverage(50)
        self.ema9_history: deque[Decimal | None] = deque(maxlen=_SLOPE_HISTORY)
        self.ema21_history: deque[Decimal | None] = deque(maxlen=_SLOPE_HISTORY)
        self.rsi = RelativeStrengthIndex()
        self.macd = MovingAverageConvergenceDivergence()
        self.atr = AverageTrueRange()
        self.vwap = SessionVwap()
        self.session: date | None = None
        self.last_candle_start: datetime | None = None


def _lagged_return(
    closes: deque[Decimal],
    close: Decimal,
    period: int,
) -> Decimal | None:
    """Return the close-to-close change over ``period`` completed candles."""
    if len(closes) < period:
        return None
    return ratio_change(close, closes[-period])


def _slope(history: deque[Decimal | None]) -> Decimal | None:
    """Return an EMA's normalized change over the slope lag."""
    if len(history) < _SLOPE_HISTORY:
        return None
    return ratio_change(history[-1], history[0])


def _volume_ratio(history: deque[int | None], volume: int | None) -> Decimal | None:
    """Compare this candle's volume against the previous window's average.

    ``history`` must not yet contain the current candle, so the current volume
    never contributes to its own baseline. Any unknown volume in the window
    makes the average a fiction, so the whole ratio is withheld.

    This is a plain rolling ratio, not a time-of-day adjusted relative volume:
    it compares 10:17 against the preceding twenty minutes, not against 10:17 on
    previous days. A seasonality-adjusted version is deliberately left for later.
    """
    if volume is None or len(history) < _ROLLING_WINDOW:
        return None
    total = 0
    for observed in history:
        if observed is None:
            return None
        total += observed
    return safe_divide(Decimal(volume), Decimal(total) / _ROLLING_WINDOW)


def _compute(state: _InstrumentFeatures, candle: Candle) -> FeatureSnapshot:
    """Advance one instrument's state by one candle and describe the result."""
    high = candle.high
    low = candle.low
    close = candle.close
    volume = candle.volume

    session = session_date(candle.start_time)
    if session != state.session:
        state.session = session
        state.volumes.clear()

    # Read every backward-looking window before the current candle joins it, so
    # returns look back from today and the volume baseline excludes today.
    return_1 = _lagged_return(state.closes, close, 1)
    return_5 = _lagged_return(state.closes, close, 5)
    return_15 = _lagged_return(state.closes, close, 15)
    volume_ratio_20 = _volume_ratio(state.volumes, volume)

    state.closes.append(close)
    state.highs.append(high)
    state.lows.append(low)
    state.volumes.append(volume)

    # Rolling extremes do include the current candle, so a fresh high reads as a
    # distance of exactly zero rather than as a breakout above a stale window.
    complete_window = len(state.highs) >= _ROLLING_WINDOW
    rolling_high_20 = max(state.highs) if complete_window else None
    rolling_low_20 = min(state.lows) if complete_window else None

    ema9 = state.ema9.update(close)
    ema21 = state.ema21.update(close)
    ema50 = state.ema50.update(close)
    state.ema9_history.append(ema9)
    state.ema21_history.append(ema21)

    rsi14 = state.rsi.update(close)
    state.macd.update(close)
    state.atr.update(high, low, close)
    state.vwap.update(session, high, low, close, volume)

    atr14 = state.atr.value
    vwap = state.vwap.value

    # One mapping feeds both the snapshot and its readiness, so a flag cannot
    # disagree with the field it describes and a new feature cannot be declared
    # without one. A name that matches neither dataclass raises here rather than
    # producing a snapshot that quietly omits it.
    derived: dict[str, Decimal | None] = {
        "return_1": return_1,
        "return_5": return_5,
        "return_15": return_15,
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "ema9_slope_5": _slope(state.ema9_history),
        "ema21_slope_5": _slope(state.ema21_history),
        "rsi14": rsi14,
        "macd": state.macd.macd,
        "macd_signal": state.macd.signal,
        "macd_histogram": state.macd.histogram,
        "macd_histogram_change": state.macd.histogram_change,
        "true_range": state.atr.true_range,
        "atr14": atr14,
        "atr_pct": safe_divide(atr14, close),
        "candle_range_pct": safe_divide(high - low, close),
        "rolling_high_20": rolling_high_20,
        "rolling_low_20": rolling_low_20,
        "distance_from_high_20": ratio_change(close, rolling_high_20),
        "distance_from_low_20": ratio_change(close, rolling_low_20),
        "vwap": vwap,
        "price_vs_vwap": ratio_change(close, vwap),
        "volume_ratio_20": volume_ratio_20,
    }

    return FeatureSnapshot(
        instrument=candle.instrument,
        candle_start_time=candle.start_time,
        candle_end_time=candle.end_time,
        open=candle.open,
        high=high,
        low=low,
        close=close,
        volume=volume,
        readiness=FeatureReadiness(
            **{name: value is not None for name, value in derived.items()}
        ),
        **derived,
    )


class FeatureEngine:
    """Thread-safe incremental features for an arbitrary set of instruments."""

    def __init__(self) -> None:
        self._states: dict[Instrument, _InstrumentFeatures] = {}
        self._snapshots: dict[Instrument, FeatureSnapshot] = {}
        self._duplicate_candle_count = 0
        self._out_of_order_candle_count = 0
        self._lock = Lock()

    @property
    def duplicate_candle_count(self) -> int:
        """Candles rejected for repeating an instrument's newest candle."""
        with self._lock:
            return self._duplicate_candle_count

    @property
    def out_of_order_candle_count(self) -> int:
        """Candles rejected for predating an instrument's newest candle."""
        with self._lock:
            return self._out_of_order_candle_count

    def _update_locked(self, candle: Candle) -> FeatureSnapshot | None:
        """Fold in one candle. The caller holds the lock and the context."""
        state = self._states.get(candle.instrument)
        if state is None:
            state = _InstrumentFeatures()
            self._states[candle.instrument] = state

        newest = state.last_candle_start
        if newest is not None:
            if candle.start_time == newest:
                self._duplicate_candle_count += 1
                return None
            if candle.start_time < newest:
                self._out_of_order_candle_count += 1
                return None

        # ``_compute`` advances this instrument's indicators in place, so a
        # raise part-way through leaves them holding a candle that was never
        # fully folded in. No ordering of the assignment below repairs that:
        # recording the candle first swallows its retry as a duplicate, and
        # recording it last lets the retry fold the same candle in a second
        # time, permanently skewing state that cannot be unwound. The instrument
        # is therefore discarded and left to rebuild from scratch, reporting
        # nothing until it has the history to report honestly, because a feature
        # derived from half-absorbed state is worse than an absent one.
        #
        # ``BaseException`` rather than ``Exception`` because the likeliest way
        # to interrupt a fold is a Ctrl-C during a long warm-up, and
        # ``KeyboardInterrupt`` does not derive from ``Exception``. Catching the
        # narrower class would leave exactly this case half-advanced and let the
        # retry double-count it. Nothing is swallowed: the cleanup re-raises.
        try:
            snapshot = _compute(state, candle)
        except BaseException:
            del self._states[candle.instrument]
            self._snapshots.pop(candle.instrument, None)
            raise
        state.last_candle_start = candle.start_time
        self._snapshots[candle.instrument] = snapshot
        return snapshot

    def update(self, candle: Candle) -> FeatureSnapshot | None:
        """Fold in one completed candle, returning its snapshot if accepted."""
        with self._lock, localcontext(FEATURE_CONTEXT):
            return self._update_locked(candle)

    def warm_up(self, candles: Iterable[Candle]) -> int:
        """Replay historical candles, returning how many were accepted.

        The whole replay holds the lock, so a live candle arriving mid-warm-up
        waits rather than interleaving. Releasing between candles would let that
        live candle advance the ordering guard past the replay position, after
        which every remaining historical candle is rejected as out-of-order and
        the instrument silently reports features built from a partial history.

        Beyond that this is a convenience over ``update`` and nothing more;
        there is no separate historical code path to diverge from the live one.
        """
        accepted = 0
        with self._lock, localcontext(FEATURE_CONTEXT):
            for candle in candles:
                if self._update_locked(candle) is not None:
                    accepted += 1
        return accepted

    def snapshot(self, instrument: Instrument) -> FeatureSnapshot | None:
        """Return the latest snapshot for one instrument, if it has one."""
        with self._lock:
            return self._snapshots.get(instrument)

    def snapshots(self) -> tuple[FeatureSnapshot, ...]:
        """Return the latest snapshot per instrument, in a stable order."""
        with self._lock:
            known = list(self._snapshots.values())
        return tuple(
            sorted(
                known,
                key=lambda item: (
                    item.instrument.exchange,
                    item.instrument.trading_symbol,
                ),
            )
        )

    def is_ready(self, instrument: Instrument) -> bool:
        """Whether every price and momentum feature is available."""
        snapshot = self.snapshot(instrument)
        return snapshot is not None and snapshot.readiness.core_ready

    def instruments(self) -> tuple[Instrument, ...]:
        """Return every instrument with feature state, in a stable order."""
        with self._lock:
            known = list(self._states)
        return tuple(
            sorted(known, key=lambda item: (item.exchange, item.trading_symbol))
        )


__all__ = ["FeatureEngine"]
