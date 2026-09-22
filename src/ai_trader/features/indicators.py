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
value is the one reading a caller may take outside an engine update.

The rest of the primitives inherit the caller's ambient context when driven
directly. For the stateless-per-update ones that is harmless, but two carry a
running value, so whatever precision was in force when it was first computed
follows them for the life of the instrument: ``WilderAverage``, and ``EMA``
again by a different route -- its smoothing factor is pinned but its SMA seed
is not, and that seed becomes the value every later observation folds into.
Both are a known sharp edge for direct callers only. Under ``FeatureEngine``
neither can bite, because the engine installs ``FEATURE_CONTEXT`` around the
whole update, seeds included.

That context traps ``DivisionByZero``, ``InvalidOperation`` and ``Overflow``
instead of yielding NaN or infinity, which is what a feature layer wants: a
nonsensical input raises rather than quietly propagating through a scanner. The
price is that every division must first prove its denominator usable, so the
guards in ``ratio_change`` and ``safe_divide`` are load-bearing rather than
defensive habit.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

from ai_trader.market import INDIA_TIMEZONE, SESSION_OPEN_TIME

FEATURE_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
"""The single numeric context every feature calculation runs under."""

OPENING_RANGE_MINUTES = 15
"""Length of the opening range, in minutes from the session open.

Fifteen minutes is the conventional intraday opening range, and it doubles as
the window inside which a session is considered observed from its start.
"""

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_THREE = Decimal(3)
_FIFTY = Decimal(50)
_HUNDRED = Decimal(100)


def session_date(timestamp: datetime) -> date:
    """Return the NSE trading date containing ``timestamp``."""
    return timestamp.astimezone(INDIA_TIMEZONE).date()


def session_minute_offset(timestamp: datetime) -> int:
    """Return whole minutes from the NSE session open to ``timestamp``.

    Negative before 09:15 IST, which a pre-open candle would be. The count is
    read from the clock rather than from how many candles have been seen, so it
    stays correct across the gaps a thinly traded instrument leaves in its
    session — a name that prints nothing between 11:00 and 11:20 still reports
    the true elapsed minutes on its next candle.

    Candle start times are minute-aligned, so the difference is always a whole
    number of minutes and the floor division is exact even when negative. IST
    observes no daylight saving, so replacing the time of day cannot land on a
    nonexistent or ambiguous local moment.
    """
    local = timestamp.astimezone(INDIA_TIMEZONE)
    opening = local.replace(
        hour=SESSION_OPEN_TIME.hour,
        minute=SESSION_OPEN_TIME.minute,
        second=0,
        microsecond=0,
    )
    return int((local - opening).total_seconds()) // 60


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

    __slots__ = (
        "_disabled",
        "_price_volume",
        "_price_volume_squared",
        "_session",
        "_volume",
    )

    def __init__(self) -> None:
        self._session: date | None = None
        self._price_volume = _ZERO
        self._price_volume_squared = _ZERO
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

    @property
    def deviation(self) -> Decimal | None:
        """Volume-weighted standard deviation of typical price about VWAP.

        This is the natural scale for "far from VWAP". A rupee away is a lot in
        a quiet name and nothing in a volatile one, and a percentage does not
        fix that, because it ignores how much of the session's turnover actually
        happened at those prices. Unavailable exactly when VWAP itself is.

        Computed from the one-pass identity ``E[x²] - E[x]²`` rather than in two
        passes, because this class deliberately retains no candles and so cannot
        revisit them. That identity is exact in real arithmetic but can land a
        hair below zero at finite precision — a session that trades at a single
        price is the obvious case — and ``FEATURE_CONTEXT`` traps rather than
        returning NaN, so the clamp below is load-bearing rather than tidiness.
        """
        if self._disabled or self._volume <= _ZERO:
            return None
        with localcontext(FEATURE_CONTEXT):
            mean = self._price_volume / self._volume
            variance = self._price_volume_squared / self._volume - mean * mean
            if variance <= _ZERO:
                return _ZERO
            return variance.sqrt()

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
            self._price_volume_squared = _ZERO
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
            self._price_volume_squared += typical_price * typical_price * volume
            self._volume += volume


class RollingDispersion:
    """Mean and population standard deviation over a fixed window of closes.

    Together these are the Bollinger Band inputs, and separately they answer a
    question no smoothed average can: how tightly is price currently packed
    around its own mean. A squeeze — dispersion collapsing — is one of the few
    genuinely leading intraday signals, and it is invisible to EMA and RSI.

    Both readings are recomputed from the retained window on every update rather
    than carried as running sums. A running variance drifts: subtracting an old
    squared term from an accumulated total loses the precision that term was
    added with, and over a full session that error compounds silently. Twenty
    additions cost nothing, so the window is simply re-read.

    Population rather than sample standard deviation, because Bollinger Bands
    are defined that way, and a consumer cross-checking against any charting
    package would otherwise see a small systematic difference and have to work
    out which of the two was wrong.
    """

    __slots__ = ("_divisor", "_period", "_values")

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("Rolling dispersion period must be positive.")
        self._period = period
        self._divisor = Decimal(period)
        self._values: deque[Decimal] = deque(maxlen=period)

    @property
    def is_full(self) -> bool:
        """Whether the window holds a full period of observations."""
        return len(self._values) == self._period

    @property
    def mean(self) -> Decimal | None:
        """The window's arithmetic mean, or ``None`` while it is still filling.

        A partial window is refused rather than averaged over what is there: a
        "20-period mean" of three closes is a different statistic wearing the
        same name, and a band drawn from it would be far too tight.
        """
        if not self.is_full:
            return None
        with localcontext(FEATURE_CONTEXT):
            return sum(self._values, _ZERO) / self._divisor

    @property
    def standard_deviation(self) -> Decimal | None:
        """The window's population standard deviation, or ``None``.

        Two passes, from the mean, so the variance is a sum of squares and
        cannot come out negative the way the one-pass identity can once rounding
        bites. ``SessionVwap.deviation`` has no such luxury and clamps instead.
        """
        mean = self.mean
        if mean is None:
            return None
        with localcontext(FEATURE_CONTEXT):
            total = _ZERO
            for value in self._values:
                deviation = value - mean
                total += deviation * deviation
            return (total / self._divisor).sqrt()

    def update(self, value: Decimal) -> None:
        """Append one observation, evicting the oldest once the window is full."""
        self._values.append(value)


class DirectionalMovementIndex:
    """Wilder's directional movement system: +DI, -DI and ADX.

    The moving averages say which way price is going; this says whether it is
    going anywhere at all. A scanner needs both, because a stacked EMA ribbon
    inside a range is a very different setup from the same ribbon inside a
    trend, and no moving average distinguishes them — ADX is the standard way
    to tell a real move from chop that happens to be sloping.

    Directional movement is smoothed with the same ``WilderAverage`` as RSI and
    ATR. Wilder's own formulation smooths running *sums* rather than averages,
    which is worth stating because a reviewer will check it: +DI and -DI are
    ratios of two quantities smoothed identically, so the period divisor cancels
    and the readings match a sum-based implementation exactly.
    """

    __slots__ = (
        "_adx",
        "_minus_dm",
        "_plus_dm",
        "_previous_close",
        "_previous_high",
        "_previous_low",
        "_true_range",
    )

    def __init__(self, period: int = 14) -> None:
        self._true_range = WilderAverage(period)
        self._plus_dm = WilderAverage(period)
        self._minus_dm = WilderAverage(period)
        self._adx = WilderAverage(period)
        self._previous_high: Decimal | None = None
        self._previous_low: Decimal | None = None
        self._previous_close: Decimal | None = None

    @property
    def plus_di(self) -> Decimal | None:
        """Upward directional movement as a percentage of true range."""
        return self._directional_index(self._plus_dm.value)

    @property
    def minus_di(self) -> Decimal | None:
        """Downward directional movement as a percentage of true range."""
        return self._directional_index(self._minus_dm.value)

    @property
    def adx(self) -> Decimal | None:
        """Trend strength, ignoring direction, or ``None`` while seeding."""
        return self._adx.value

    def _directional_index(self, movement: Decimal | None) -> Decimal | None:
        average_range = self._true_range.value
        if movement is None or average_range is None or average_range <= _ZERO:
            # A zero average true range means fourteen candles that did not
            # move at all. Direction is then genuinely undefined rather than
            # zero, so both DIs and the ADX behind them stall honestly.
            return None
        with localcontext(FEATURE_CONTEXT):
            return _HUNDRED * movement / average_range

    def update(self, high: Decimal, low: Decimal, close: Decimal) -> None:
        """Fold in one candle, advancing the smoothed movement and the ADX."""
        previous_high = self._previous_high
        previous_low = self._previous_low
        previous_close = self._previous_close
        self._previous_high = high
        self._previous_low = low
        self._previous_close = close
        if previous_high is None or previous_low is None or previous_close is None:
            # Directional movement is defined against the preceding candle, so
            # the first candle of an instrument's life contributes nothing.
            return

        with localcontext(FEATURE_CONTEXT):
            up_move = high - previous_high
            down_move = previous_low - low
            # Only the larger of the two counts, and only if it is positive: a
            # candle that extends both ways is attributed to the side it
            # extended further, and an inside candle to neither.
            plus = up_move if up_move > down_move and up_move > _ZERO else _ZERO
            minus = down_move if down_move > up_move and down_move > _ZERO else _ZERO
            true_range = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
            self._true_range.update(true_range)
            self._plus_dm.update(plus)
            self._minus_dm.update(minus)

            plus_di = self.plus_di
            minus_di = self.minus_di
            if plus_di is None or minus_di is None:
                return
            total = plus_di + minus_di
            # Equal readings mean the candles pushed as hard one way as the
            # other, which is a real absence of direction rather than a missing
            # number, so the index is zero rather than unavailable.
            index = (
                _ZERO if total <= _ZERO else _HUNDRED * abs(plus_di - minus_di) / total
            )
            self._adx.update(index)


class OnBalanceVolume:
    """Session-scoped on-balance volume.

    Volume alone says how much traded; this says which side it traded on, by
    adding a candle's volume when the close rose and subtracting it when the
    close fell. Price making a new high while this does not is the classic tell
    that a move is not being funded, and it is not derivable from any of the
    price-only indicators here.

    The running total resets at every session boundary and starts from zero, so
    a reading is always "net flow so far today" rather than an arbitrary offset
    inherited from previous days. The session's first candle contributes
    nothing: it has no same-session predecessor to have risen or fallen against.
    A close exactly equal to its predecessor is still attribution — it
    contributes zero by definition — so the total becomes available on the
    second candle either way.

    Volume is treated as strictly as ``SessionVwap`` treats it. One candle with
    unknown volume disables the total for the remainder of the session, because
    a net flow that silently skips an unknown quantity is not a net flow.
    """

    __slots__ = ("_attributed", "_disabled", "_previous_close", "_session", "_value")

    def __init__(self) -> None:
        self._session: date | None = None
        self._value = _ZERO
        self._previous_close: Decimal | None = None
        self._attributed = False
        self._disabled = False

    @property
    def value(self) -> Decimal | None:
        """Net volume flow so far this session, or ``None``."""
        return None if self._disabled or not self._attributed else self._value

    def update(self, session: date, close: Decimal, volume: int | None) -> None:
        """Fold in one candle, resetting first if a new session has started."""
        if session != self._session:
            self._session = session
            self._value = _ZERO
            self._previous_close = None
            self._attributed = False
            self._disabled = False
        if self._disabled:
            return
        if volume is None:
            self._disabled = True
            return
        previous_close = self._previous_close
        self._previous_close = close
        if previous_close is None:
            return
        with localcontext(FEATURE_CONTEXT):
            if close > previous_close:
                self._value += volume
            elif close < previous_close:
                self._value -= volume
        self._attributed = True


class SessionContext:
    """Where the current candle sits inside its own trading session.

    Every other primitive here describes a rolling window that knows nothing
    about the clock. This one describes the day: what it opened at, how far it
    has travelled since, and what the first fifteen minutes established.
    Intraday setups are overwhelmingly stated in those terms — a break of the
    opening range, a reclaim of the day's open, a push into the high of the day
    — and not one of them can be expressed from rolling windows alone.

    Aggregates are only reported for a session the caller actually observed from
    its start. A session first seen at 11:00 has a "high so far" that is not the
    day's high, and publishing it as one would be a plain lie of exactly the
    kind strict VWAP exists to avoid, so everything but the clock reading stays
    ``None`` for the rest of that day. Observing from the start is read
    generously: any first candle inside the opening range counts, because a
    thinly traded name may simply not print at 09:15.
    """

    __slots__ = (
        "_anchored",
        "_high",
        "_low",
        "_open",
        "_opening_high",
        "_opening_low",
        "_opening_range_closed",
        "_session",
        "_volume",
        "_volume_disabled",
    )

    def __init__(self) -> None:
        self._session: date | None = None
        self._anchored = False
        self._open: Decimal | None = None
        self._high: Decimal | None = None
        self._low: Decimal | None = None
        self._opening_high: Decimal | None = None
        self._opening_low: Decimal | None = None
        self._opening_range_closed = False
        self._volume = _ZERO
        self._volume_disabled = False

    @property
    def is_anchored(self) -> bool:
        """Whether this session was observed from inside its opening range."""
        return self._anchored

    @property
    def open(self) -> Decimal | None:
        """The session's opening price."""
        return self._open

    @property
    def high(self) -> Decimal | None:
        """The highest price seen so far this session."""
        return self._high

    @property
    def low(self) -> Decimal | None:
        """The lowest price seen so far this session."""
        return self._low

    @property
    def volume(self) -> Decimal | None:
        """Total volume so far this session, or ``None`` if any is unknown."""
        if not self._anchored or self._volume_disabled:
            return None
        return self._volume

    @property
    def opening_range_high(self) -> Decimal | None:
        """The opening range's high, once that range has closed.

        Withheld until the range is complete rather than reported as it forms,
        because a breakout of a range that is still widening is not a breakout.
        """
        return self._opening_high if self._opening_range_closed else None

    @property
    def opening_range_low(self) -> Decimal | None:
        """The opening range's low, once that range has closed."""
        return self._opening_low if self._opening_range_closed else None

    def update(
        self,
        timestamp: datetime,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        volume: int | None,
    ) -> None:
        """Fold in one candle, resetting first if a new session has started.

        A candle from before 09:15 is ignored outright. The architecture has the
        stack warming up pre-open, so these do arrive, and letting one through
        would anchor the session to the pre-open equilibrium price and fold
        pre-open prints into the opening range. It would also be unrecoverable:
        the reset that establishes the anchor fires only when the session label
        changes, so a pre-open candle claiming today's label means 09:15 never
        re-anchors. Every session aggregate would stay wrong for the rest of the
        day while its readiness flag reported it good.
        """
        offset = session_minute_offset(timestamp)
        if offset < 0:
            return
        session = session_date(timestamp)
        if session != self._session:
            self._reset(session, offset, open_price)
        if not self._anchored:
            return
        self._high = high if self._high is None else max(self._high, high)
        self._low = low if self._low is None else min(self._low, low)
        if volume is None:
            self._volume_disabled = True
        elif not self._volume_disabled:
            with localcontext(FEATURE_CONTEXT):
                self._volume += volume
        if offset < OPENING_RANGE_MINUTES:
            self._opening_high = (
                high if self._opening_high is None else max(self._opening_high, high)
            )
            self._opening_low = (
                low if self._opening_low is None else min(self._opening_low, low)
            )
        else:
            # The first candle at or past the range's end closes it. A session
            # that stops printing before 09:30 therefore never publishes an
            # opening range, which is correct: it does not have one yet.
            self._opening_range_closed = True

    def _reset(self, session: date, offset: int, open_price: Decimal) -> None:
        self._session = session
        self._anchored = offset < OPENING_RANGE_MINUTES
        self._open = open_price if self._anchored else None
        self._high = None
        self._low = None
        self._opening_high = None
        self._opening_low = None
        self._opening_range_closed = False
        self._volume = _ZERO
        self._volume_disabled = False


__all__ = [
    "FEATURE_CONTEXT",
    "OPENING_RANGE_MINUTES",
    "SESSION_OPEN_TIME",
    "AverageTrueRange",
    "DirectionalMovementIndex",
    "ExponentialMovingAverage",
    "MovingAverageConvergenceDivergence",
    "OnBalanceVolume",
    "RelativeStrengthIndex",
    "RollingDispersion",
    "SessionContext",
    "SessionVwap",
    "WilderAverage",
    "ratio_change",
    "safe_divide",
    "session_date",
    "session_minute_offset",
]
