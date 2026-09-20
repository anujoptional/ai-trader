"""Stateful, incremental indicator primitives with hand-verifiable formulas.

Each primitive keeps only the state its formula requires — a smoothed value, a
short seed buffer, a previous close — and exposes ``update`` plus a ``value``
property. None of them retain candle history, so an engine composing them stays
bounded in memory however long a session runs.

Every number here is a ``Decimal``, and ``FeatureEngine`` installs
``FEATURE_CONTEXT`` around each update so that the whole fold runs at one
precision. Several constants in these formulas have no finite decimal
expansion: the smoothing factor ``2 / 22`` of a 21-period EMA and the typical
price's division by three are both infinite. Their results therefore depend on
the active precision, and leaving that to the ambient ``decimal.getcontext()``
would let an unrelated part of the process silently change an indicator.
Pinning one context makes a candle sequence produce one answer, always.

Two primitives pin the context themselves rather than relying on the engine,
because each owns a number that outlives a single update. ``EMA`` pins the
derivation of its smoothing factor, computed once in the constructor and reused
for every later observation, so an unlucky ambient precision at construction
time cannot follow the instrument for its whole life. ``SessionVwap`` pins both
its accumulation and its division, because its sums are cumulative and its
value is the one reading a caller may take outside an engine update. The rest
of the primitives inherit the caller's ambient context when driven directly,
which is deliberate for the stateless-per-update ones and a known sharp edge
for ``WilderAverage``, whose running value would carry that precision forward.

That context traps ``DivisionByZero``, ``InvalidOperation`` and ``Overflow``
instead of yielding NaN or infinity, which is what a feature layer wants: a
nonsensical input raises rather than quietly propagating through a scanner. The
price is that every division must first prove its denominator usable, so the
guards in ``ratio_change`` and ``safe_divide`` are load-bearing rather than
defensive habit.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from zoneinfo import ZoneInfo

FEATURE_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
"""The single numeric context every feature calculation runs under."""

_INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
"""NSE trading days are delimited by the calendar date in this timezone."""

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_THREE = Decimal(3)
_FIFTY = Decimal(50)
_HUNDRED = Decimal(100)


def session_date(timestamp: datetime) -> date:
    """Return the NSE trading date containing ``timestamp``."""
    return timestamp.astimezone(_INDIA_TIMEZONE).date()


def safe_divide(
    numerator: Decimal | None,
    denominator: Decimal | None,
) -> Decimal | None:
    """Divide, returning ``None`` rather than raising on a missing or zero base."""
    if numerator is None or denominator is None or denominator == _ZERO:
        return None
    return numerator / denominator


def ratio_change(current: Decimal | None, base: Decimal | None) -> Decimal | None:
    """Return ``current / base - 1``, or ``None`` if that is not well defined.

    This is the project's one definition of a normalized change. Returns,
    distances from rolling extremes, EMA slopes and price-versus-VWAP all use
    it, so they are all fractions rather than percentages: ``0.001`` is 0.1%.
    """
    ratio = safe_divide(current, base)
    return None if ratio is None else ratio - _ONE


class ExponentialMovingAverage:
    """An EMA seeded by the simple average of its first ``period`` inputs.

    Seeding from an SMA rather than from the first observation avoids handing
    the earliest price the weight of an entire average, which would bias the
    series for roughly ``period`` candles. The value stays ``None`` until the
    seed is complete, and the seed buffer is discarded once it is used.
    """

    __slots__ = ("_alpha", "_one_minus_alpha", "_period", "_seed", "_value")

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("EMA period must be positive.")
        self._period = period
        with localcontext(FEATURE_CONTEXT):
            self._alpha = _TWO / (Decimal(period) + _ONE)
            self._one_minus_alpha = _ONE - self._alpha
        self._seed: list[Decimal] | None = []
        self._value = _ZERO

    @property
    def value(self) -> Decimal | None:
        """The current average, or ``None`` while it is still seeding."""
        return None if self._seed is not None else self._value

    def update(self, observation: Decimal) -> Decimal | None:
        """Fold in one observation and return the resulting average."""
        seed = self._seed
        if seed is not None:
            seed.append(observation)
            if len(seed) < self._period:
                return None
            self._value = sum(seed, _ZERO) / self._period
            self._seed = None
            return self._value
        self._value = self._alpha * observation + self._one_minus_alpha * self._value
        return self._value


class WilderAverage:
    """Wilder's smoothed average, seeded by the mean of its first ``period``.

    Both RSI and ATR are defined on exactly this smoothing, so they share one
    implementation instead of two that could drift apart.
    """

    __slots__ = ("_divisor", "_period", "_previous_weight", "_seed", "_value")

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("Wilder average period must be positive.")
        self._period = period
        self._divisor = Decimal(period)
        self._previous_weight = Decimal(period - 1)
        self._seed: list[Decimal] | None = []
        self._value = _ZERO

    @property
    def value(self) -> Decimal | None:
        """The current average, or ``None`` while it is still seeding."""
        return None if self._seed is not None else self._value

    def update(self, observation: Decimal) -> Decimal | None:
        """Fold in one observation and return the resulting average."""
        seed = self._seed
        if seed is not None:
            seed.append(observation)
            if len(seed) < self._period:
                return None
            self._value = sum(seed, _ZERO) / self._divisor
            self._seed = None
            return self._value
        weighted = self._value * self._previous_weight + observation
        self._value = weighted / self._divisor
        return self._value


def _relative_strength(average_gain: Decimal, average_loss: Decimal) -> Decimal:
    """Convert Wilder gain/loss averages into an RSI between 0 and 100.

    A flat stretch leaves both averages at zero, where the ratio is undefined.
    That resolves to 50 — the neutral reading — because an illiquid name that
    simply did not move must not appear to a scanner as maximally overbought.
    """
    if average_loss == _ZERO:
        return _HUNDRED if average_gain > _ZERO else _FIFTY
    if average_gain == _ZERO:
        return _ZERO
    strength = average_gain / average_loss
    return _HUNDRED - _HUNDRED / (_ONE + strength)


class RelativeStrengthIndex:
    """Wilder's RSI over close-to-close changes."""

    __slots__ = ("_average_gain", "_average_loss", "_previous_close", "_value")

    def __init__(self, period: int = 14) -> None:
        self._average_gain = WilderAverage(period)
        self._average_loss = WilderAverage(period)
        self._previous_close: Decimal | None = None
        self._value: Decimal | None = None

    @property
    def value(self) -> Decimal | None:
        """The current RSI, or ``None`` while it is still seeding."""
        return self._value

    def update(self, close: Decimal) -> Decimal | None:
        """Fold in one close and return the resulting RSI."""
        previous_close = self._previous_close
        self._previous_close = close
        if previous_close is None:
            return None
        change = close - previous_close
        average_gain = self._average_gain.update(max(change, _ZERO))
        average_loss = self._average_loss.update(max(-change, _ZERO))
        if average_gain is None or average_loss is None:
            return None
        self._value = _relative_strength(average_gain, average_loss)
        return self._value


class MovingAverageConvergenceDivergence:
    """MACD, its signal line, histogram, and the histogram's change.

    The signal line is an EMA of the MACD series, so it only begins seeding
    once MACD itself exists. Nothing downstream is emitted early: the histogram
    waits for the signal, and its change waits for a previous histogram.
    """

    __slots__ = (
        "_fast",
        "_histogram",
        "_histogram_change",
        "_macd",
        "_signal",
        "_slow",
    )

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("MACD fast period must be shorter than its slow period.")
        self._fast = ExponentialMovingAverage(fast)
        self._slow = ExponentialMovingAverage(slow)
        self._signal = ExponentialMovingAverage(signal)
        self._macd: Decimal | None = None
        self._histogram: Decimal | None = None
        self._histogram_change: Decimal | None = None

    @property
    def macd(self) -> Decimal | None:
        """The fast EMA minus the slow EMA."""
        return self._macd

    @property
    def signal(self) -> Decimal | None:
        """The EMA of the MACD series."""
        return self._signal.value

    @property
    def histogram(self) -> Decimal | None:
        """The MACD minus its signal line."""
        return self._histogram

    @property
    def histogram_change(self) -> Decimal | None:
        """How far the histogram moved since the previous candle."""
        return self._histogram_change

    def update(self, close: Decimal) -> None:
        """Fold in one close, advancing every MACD component."""
        fast = self._fast.update(close)
        slow = self._slow.update(close)
        if fast is None or slow is None:
            return
        macd = fast - slow
        self._macd = macd
        signal = self._signal.update(macd)
        if signal is None:
            return
        previous_histogram = self._histogram
        histogram = macd - signal
        self._histogram = histogram
        if previous_histogram is not None:
            self._histogram_change = histogram - previous_histogram


class AverageTrueRange:
    """True range per candle and its Wilder-smoothed average.

    The first candle has no previous close to gap against, so its true range is
    simply its own high-low span.
    """

    __slots__ = ("_average", "_previous_close", "_true_range")

    def __init__(self, period: int = 14) -> None:
        self._average = WilderAverage(period)
        self._previous_close: Decimal | None = None
        self._true_range: Decimal | None = None

    @property
    def true_range(self) -> Decimal | None:
        """The most recent candle's true range."""
        return self._true_range

    @property
    def value(self) -> Decimal | None:
        """The smoothed average true range, or ``None`` while seeding."""
        return self._average.value

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> None:
        """Fold in one candle, advancing the true range and its average."""
        previous_close = self._previous_close
        if previous_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        self._previous_close = close
        self._true_range = true_range
        self._average.update(true_range)


class SessionVwap:
    """Volume-weighted average price, reset at every NSE session boundary.

    Volume is used only where the broker actually reported it. A candle with
    unknown volume disables VWAP for the remainder of that session rather than
    being treated as zero volume, because a VWAP computed over an unknown
    fraction of the session's turnover is not a VWAP — it is a number that looks
    like one. The next session starts clean.
    """

    __slots__ = ("_disabled", "_price_volume", "_session", "_volume")

    def __init__(self) -> None:
        self._session: date | None = None
        self._price_volume = _ZERO
        self._volume = _ZERO
        self._disabled = False

    @property
    def value(self) -> Decimal | None:
        """The session VWAP, or ``None`` if it cannot be trusted.

        The division is pinned rather than left to the ambient context, because
        this is the one number a caller can read outside an engine update and a
        caller's own precision must not change what VWAP is.
        """
        if self._disabled or self._volume <= _ZERO:
            return None
        with localcontext(FEATURE_CONTEXT):
            return self._price_volume / self._volume

    def update(
        self,
        session: date,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: int | None,
    ) -> None:
        """Fold in one candle, resetting first if a new session has started."""
        if session != self._session:
            self._session = session
            self._price_volume = _ZERO
            self._volume = _ZERO
            self._disabled = False
        if volume is None:
            self._disabled = True
            return
        if self._disabled:
            return
        # Pinned for the same reason the division in ``value`` is, and more
        # urgently: these sums are cumulative, so a caller's precision would not
        # merely colour one reading but stay baked into every later one.
        with localcontext(FEATURE_CONTEXT):
            typical_price = (high + low + close) / _THREE
            self._price_volume += typical_price * volume
            self._volume += volume


__all__ = [
    "FEATURE_CONTEXT",
    "AverageTrueRange",
    "ExponentialMovingAverage",
    "MovingAverageConvergenceDivergence",
    "RelativeStrengthIndex",
    "SessionVwap",
    "WilderAverage",
    "ratio_change",
    "safe_divide",
    "session_date",
]
