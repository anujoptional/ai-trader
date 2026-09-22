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
fifteen closes, twenty highs, twenty lows, twenty volumes, a twenty-close
dispersion window, two short EMA histories and a few dozen scalars — so hundreds
of instruments cost a predictable amount however long the engine runs.

Candles must arrive in increasing start-time order per instrument. A repeated or
older candle is counted and discarded rather than folded in, because Wilder and
EMA smoothing have no inverse: a single stale candle would silently bias every
subsequent value with no way to detect or undo it. The engine is thread-safe,
since ``MarketState`` invokes its ``on_candle`` hook on a broker feed thread.

What a new session resets is a deliberate choice, and it is deliberately not
uniform. VWAP, on-balance volume, the session context and the relative-volume
baseline all reset at every IST date change, because each describes what
happened *within* a session and yesterday's says nothing about today's. Price
history does not reset: returns, the EMAs, RSI, MACD, ATR, the directional
index, the rolling twenty-candle extremes and the Bollinger window all continue
across the boundary, so the first candle of a session reports an overnight gap
rather than nothing at all. That is usually what a scanner wants, but it does
mean ``return_1`` on the 09:15 candle is a gap and not a one-minute move, and
that ATR's first true range of the day includes the gap against yesterday's
close. A consumer that wants intraday-only momentum must use a fresh engine per
session.

The session-scoped features go further and require an *anchor*: the engine must
have seen a candle from before the opening range closed, or the session high,
low, open, volume and opening range are all withheld for that whole session
rather than reported from a partial view. Starting an engine at 13:00 would
otherwise report a "session high" that is only the afternoon's high, which is a
worse answer than none. ``minutes_since_session_open`` is the single exemption,
since a clock reading is honest however late the engine started.
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
    DirectionalMovementIndex,
    ExponentialMovingAverage,
    MovingAverageConvergenceDivergence,
    OnBalanceVolume,
    RelativeStrengthIndex,
    RollingDispersion,
    SessionContext,
    SessionVwap,
    ratio_change,
    safe_divide,
    session_date,
    session_minute_offset,
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
"""Retained previous closes, set by the longest return period.

Deliberately not widened to twenty for the moving average and the Bollinger
bands: ``RollingDispersion`` keeps its own window, so widening this one would
retain five closes nothing reads.
"""

_BOLLINGER_MULTIPLIER = Decimal(2)
"""Standard deviations to either side of the mean for the Bollinger bands."""


class _InstrumentFeatures:
    """Every indicator and bounded window backing a single instrument."""

    __slots__ = (
        "atr",
        "closes",
        "dispersion",
        "dmi",
        "ema21",
        "ema21_history",
        "ema50",
        "ema9",
        "ema9_history",
        "highs",
        "last_candle_start",
        "lows",
        "macd",
        "obv",
        "rsi",
        "session",
        "session_context",
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
        self.dmi = DirectionalMovementIndex()
        self.dispersion = RollingDispersion(_ROLLING_WINDOW)
        self.obv = OnBalanceVolume()
        self.session_context = SessionContext()
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


def _bollinger(
    mean: Decimal | None,
    deviation: Decimal | None,
    close: Decimal,
) -> tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None]:
    """Return the upper band, lower band, bandwidth and %B for one candle.

    Bandwidth is the band width relative to its own mean, which makes it
    comparable across instruments at different price levels; %B places the close
    within the band, reading 0 at the lower band and 1 at the upper one. Both
    are withheld on a perfectly flat window, where the bands collapse onto the
    mean and the denominators go to zero, and %B is additionally free to fall
    outside 0..1, which is exactly the breakout a scanner wants to see.
    """
    if mean is None or deviation is None:
        return None, None, None, None
    offset = _BOLLINGER_MULTIPLIER * deviation
    upper = mean + offset
    lower = mean - offset
    width = upper - lower
    return upper, lower, safe_divide(width, mean), safe_divide(close - lower, width)


def _session_range_position(
    high: Decimal | None,
    low: Decimal | None,
    close: Decimal,
) -> tuple[Decimal | None, Decimal | None]:
    """Return the session range as a fraction of its low, and the close in it.

    Both are withheld until the session is anchored, and the position is
    additionally withheld on a session that has not moved at all, where the
    range is zero and "where in the range" has no answer.
    """
    if high is None or low is None:
        return None, None
    span = high - low
    return safe_divide(span, low), safe_divide(close - low, span)


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
    state.dmi.update(high, low, close)
    state.dispersion.update(close)
    state.obv.update(session, close, volume)
    state.session_context.update(candle.start_time, candle.open, high, low, volume)
    state.vwap.update(session, high, low, close, volume)

    atr14 = state.atr.value
    vwap = state.vwap.value
    vwap_deviation = state.vwap.deviation
    sma20 = state.dispersion.mean
    context = state.session_context
    session_high = context.high
    session_low = context.low

    # A z-score of the close against the session's own turnover-weighted spread,
    # so "two rupees above VWAP" reads differently in a quiet name than in a
    # volatile one. Withheld while the spread is still zero, which is every
    # session's first candle.
    price_vs_vwap_sigma = (
        None
        if vwap is None or vwap_deviation is None
        else safe_divide(close - vwap, vwap_deviation)
    )

    bollinger_upper_20, bollinger_lower_20, bollinger_bandwidth_20, percent_b_20 = (
        _bollinger(sma20, state.dispersion.standard_deviation, close)
    )
    session_range_pct, position_in_session_range = _session_range_position(
        session_high, session_low, close
    )

    session_open = context.open
    opening_range_high = context.opening_range_high
    opening_range_low = context.opening_range_low

    # The one session feature exempt from the anchor. Every other session field
    # describes something this engine must have watched accumulate, so a mid
    # session start leaves them unavailable; a clock reading is honest either
    # way, and a scanner needs it precisely to know how young the session is.
    minutes_since_session_open = Decimal(session_minute_offset(candle.start_time))

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
        "plus_di14": state.dmi.plus_di,
        "minus_di14": state.dmi.minus_di,
        "adx14": state.dmi.adx,
        "rolling_high_20": rolling_high_20,
        "rolling_low_20": rolling_low_20,
        "distance_from_high_20": ratio_change(close, rolling_high_20),
        "distance_from_low_20": ratio_change(close, rolling_low_20),
        "sma20": sma20,
        "bollinger_upper_20": bollinger_upper_20,
        "bollinger_lower_20": bollinger_lower_20,
        "bollinger_bandwidth_20": bollinger_bandwidth_20,
        "bollinger_percent_b_20": percent_b_20,
        "vwap": vwap,
        "price_vs_vwap": ratio_change(close, vwap),
        "vwap_deviation": vwap_deviation,
        "price_vs_vwap_sigma": price_vs_vwap_sigma,
        "volume_ratio_20": volume_ratio_20,
        "obv": state.obv.value,
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "session_range_pct": session_range_pct,
        "position_in_session_range": position_in_session_range,
        "distance_from_session_open": ratio_change(close, session_open),
        "minutes_since_session_open": minutes_since_session_open,
        "session_volume": context.volume,
        "opening_range_high": opening_range_high,
        "opening_range_low": opening_range_low,
        "distance_from_opening_range_high": ratio_change(close, opening_range_high),
        "distance_from_opening_range_low": ratio_change(close, opening_range_low),
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

    def forget(self, instrument: Instrument) -> bool:
        """Drop all feature state for an instrument, reporting if it was known.

        The mirror of ``MarketState.forget``, and it exists for the same reason:
        history is bounded per instrument but not across them, so a process that
        watches a different set each session accumulates both memory and stale
        instruments that ``snapshots`` keeps reporting. Without this the engine
        would outlive the evictions its candle source already performs.

        Everything goes, indicator windows included, so an instrument that
        returns is rebuilt from scratch and re-earns every readiness flag rather
        than resuming against a window with a hole in it.
        """
        with self._lock:
            known = instrument in self._states or instrument in self._snapshots
            self._states.pop(instrument, None)
            self._snapshots.pop(instrument, None)
        return known

    def retain(self, instruments: Iterable[Instrument]) -> tuple[Instrument, ...]:
        """Forget every instrument outside ``instruments``, returning those dropped.

        Declaring the live set rather than the dead one keeps eviction correct by
        construction, and lets a caller pass the same set to this and to
        ``MarketState.retain`` so the two cannot drift apart.
        """
        keep = set(instruments)
        dropped = tuple(
            instrument for instrument in self.instruments() if instrument not in keep
        )
        for instrument in dropped:
            self.forget(instrument)
        return dropped


__all__ = ["FeatureEngine"]
