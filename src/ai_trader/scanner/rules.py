"""Deterministic rules, each encoding one measurable trading hypothesis.

A rule reads a ``FeatureSnapshot`` and either declines or returns a direction,
a score and the exact values it read. It never sizes, never proposes a stop and
never decides anything: a rule firing means "this is worth an AI call", nothing
more.

**Every feature is read through its own readiness flag, never through
``is not None``.** ``core_ready`` covers price and momentum only, so a rule that
gated on it and then dereferenced ``vwap`` would evaluate against nothing for a
whole live session without failing once. ``read_all`` makes that structural: a
rule declares what it needs, the values arrive only if every flag is set, and
the same dictionary becomes the candidate's evidence — so what a rule is
recorded as having seen is literally what it read.

Volume is the sharp edge of that rule. ``volume_ratio_20`` is unavailable for
the first twenty minutes of every session, which is prime scanning time, and
session VWAP is disabled for the rest of the day by a single unknown volume. A
rule that *needs* volume declares it and is skipped when it is missing. A rule
that merely *prefers* it scores without it rather than scoring it as zero —
substituting zero for an unavailable number is the one thing the feature layer
refuses to do, and a scanner that reintroduced it would systematically mark down
every breakout at the open.

**The numeric cut-offs below are placeholders, not findings.** They are
plausible conventional levels chosen so the layer can be built and replayed;
none has been measured on this market. Section 4.5's replay engine is what turns
any of them into evidence, and until it does, a threshold cited from here is
being cited from a guess.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from ai_trader.features import FeatureSnapshot, safe_divide
from ai_trader.scanner.models import Direction, MarketContext, PortfolioState

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)

ADX_TREND_FLOOR = Decimal("25")
ADX_TREND_CEILING = Decimal("50")
ADX_RANGE_CEILING = Decimal("20")
RSI_OVERBOUGHT = Decimal("70")
RSI_OVERSOLD = Decimal("30")
RSI_EXHAUSTION_HIGH = Decimal("80")
RSI_EXHAUSTION_LOW = Decimal("20")
RSI_REVERSION_SPAN = Decimal("15")
MACD_ATR_CEILING = Decimal("0.5")
BREAKOUT_ATR_TOLERANCE = Decimal("0.1")
VOLUME_CONFIRMATION_FLOOR = Decimal("1.5")
VOLUME_CONFIRMATION_CEILING = Decimal("3")
PERCENT_B_STRETCH = Decimal("0.5")
VWAP_SIGMA_TRIGGER = Decimal("2")
VWAP_SIGMA_CEILING = Decimal("4")
OPENING_RANGE_ATR_CEILING = Decimal("1")


@dataclass(frozen=True, slots=True)
class RuleSignal:
    """One rule's verdict on one snapshot.

    ``score`` is normalized to 0..1 on that rule's own terms, where 1 means "as
    convincing as this rule can be". Whether a 0.8 from one rule is comparable
    to a 0.8 from another is an **open question**, not a property being claimed:
    the scanner ranks across rules as though it were, because it must rank
    somehow, and replay is what will show whether the assumption holds.
    """

    direction: Direction
    score: Decimal
    evidence: Mapping[str, Decimal]


class Rule(Protocol):
    """One hypothesis, evaluated against one instrument at one candle.

    Rules are called under ``FEATURE_CONTEXT`` by the scanner, so arithmetic
    here matches the engine that produced the inputs without each rule
    installing the context itself. A rule invoked directly outside the scanner
    should install it too, or its rounding is whatever the ambient process
    happens to be using.
    """

    name: str
    required_features: tuple[str, ...]

    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
    ) -> RuleSignal | None:
        """Return a signal, or ``None`` when this rule has nothing to say."""
        ...


def available(snapshot: FeatureSnapshot, name: str) -> Decimal | None:
    """Return a derived feature only when its own readiness flag is set.

    The flag is the authority rather than the value. The engine guarantees the
    two agree, but a consumer that read the value directly would be relying on
    that guarantee silently; going through the flag means a rule states which
    features it is entitled to, and a future disagreement fails closed.
    """
    if not getattr(snapshot.readiness, name):
        return None
    value: Decimal | None = getattr(snapshot, name)
    return value


def read_all(
    snapshot: FeatureSnapshot,
    names: Sequence[str],
) -> dict[str, Decimal] | None:
    """Read every named feature, or ``None`` if any one is unavailable.

    All-or-nothing on purpose. A rule that received a partial reading would have
    to decide what a missing input means, and the answer is always the same —
    it cannot evaluate — so deciding it once here keeps every rule from
    inventing its own version of that answer.
    """
    values: dict[str, Decimal] = {}
    for name in names:
        value = available(snapshot, name)
        if value is None:
            return None
        values[name] = value
    return values


def missing_features(snapshot: FeatureSnapshot, rule: Rule) -> tuple[str, ...]:
    """Which of a rule's required features this snapshot cannot supply."""
    return tuple(
        name for name in rule.required_features if available(snapshot, name) is None
    )


def _clamp_unit(value: Decimal) -> Decimal:
    if value <= _ZERO:
        return _ZERO
    if value >= _ONE:
        return _ONE
    return value


def _ramp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    """Map ``low``..``high`` onto 0..1, clamping outside it."""
    scaled = safe_divide(value - low, high - low)
    return _ZERO if scaled is None else _clamp_unit(scaled)


def _mean(first: Decimal, second: Decimal) -> Decimal:
    return (first + second) / _TWO


class TrendContinuationRule:
    """An instrument already trending, with momentum pushing the same way.

    The ADX floor is what separates this from the mean-reversion rule below:
    the two are deliberately gated on opposite regimes, so a name cannot
    plausibly satisfy both and any instrument that somehow does is reporting
    contradictory evidence rather than double confirmation.

    An already-exhausted RSI vetoes the signal. Joining a trend at the point
    where it has gone parabolic is the failure mode this rule would otherwise
    have, and the veto is cheaper than discovering it in replay.
    """

    name = "trend_continuation"
    required_features = (
        "ema9",
        "ema21",
        "ema50",
        "adx14",
        "macd_histogram",
        "rsi14",
        "atr14",
    )

    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
    ) -> RuleSignal | None:
        read = read_all(snapshot, self.required_features)
        if read is None:
            return None
        if read["adx14"] < ADX_TREND_FLOOR:
            return None

        stacked_up = read["ema9"] > read["ema21"] > read["ema50"]
        stacked_down = read["ema9"] < read["ema21"] < read["ema50"]
        if stacked_up and read["macd_histogram"] > _ZERO:
            if read["rsi14"] >= RSI_EXHAUSTION_HIGH:
                return None
            direction = Direction.LONG
        elif stacked_down and read["macd_histogram"] < _ZERO:
            if read["rsi14"] <= RSI_EXHAUSTION_LOW:
                return None
            direction = Direction.SHORT
        else:
            return None

        strength = _ramp(read["adx14"], ADX_TREND_FLOOR, ADX_TREND_CEILING)
        # Normalized by ATR so the same score means the same thing on a 200-rupee
        # name and a 3000-rupee one; a raw histogram is denominated in price.
        pushed = safe_divide(abs(read["macd_histogram"]), read["atr14"])
        push = _ZERO if pushed is None else _ramp(pushed, _ZERO, MACD_ATR_CEILING)
        return RuleSignal(direction, _mean(strength, push), read)


class BreakoutRule:
    """A close at the edge of the twenty-candle range.

    The test is a tolerance rather than a strict break, because the rolling
    extremes *include* the current candle: ``rolling_high_20`` is already at
    least this candle's high, so ``close > rolling_high_20`` is unsatisfiable
    and a rule written that way would emit nothing, forever, without erroring.
    Closing within a fraction of an ATR of the window's high is the same
    hypothesis, expressed in terms the feature actually supports.

    Volume confirms but is not required. When ``volume_ratio_20`` is unavailable
    — which it is for the first twenty minutes of every session — the score is
    the tightness alone rather than the tightness averaged against a zero, since
    an unknown ratio is not a weak one.
    """

    name = "range_breakout"
    required_features = ("rolling_high_20", "rolling_low_20", "atr14")

    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
    ) -> RuleSignal | None:
        read = read_all(snapshot, self.required_features)
        if read is None:
            return None
        if read["rolling_high_20"] <= read["rolling_low_20"]:
            return None
        tolerance = BREAKOUT_ATR_TOLERANCE * read["atr14"]
        if tolerance <= _ZERO:
            return None

        close = snapshot.close
        below_high = read["rolling_high_20"] - close
        above_low = close - read["rolling_low_20"]
        if below_high <= tolerance and below_high <= above_low:
            direction, gap = Direction.LONG, below_high
        elif above_low <= tolerance:
            direction, gap = Direction.SHORT, above_low
        else:
            return None

        score = _ONE - _ramp(gap, _ZERO, tolerance)
        volume_ratio = available(snapshot, "volume_ratio_20")
        if volume_ratio is not None:
            read["volume_ratio_20"] = volume_ratio
            confirmation = _ramp(
                volume_ratio,
                VOLUME_CONFIRMATION_FLOOR,
                VOLUME_CONFIRMATION_CEILING,
            )
            score = _mean(score, confirmation)
        return RuleSignal(direction, score, read)


class MeanReversionRule:
    """A stretch outside the Bollinger band while the name is *not* trending.

    The ADX ceiling is the whole point. Price outside the band during a strong
    trend is the trend working, not an excess to fade, and fading it is how a
    mean-reversion rule loses money in exactly the conditions the trend rule was
    designed for.
    """

    name = "band_mean_reversion"
    required_features = ("bollinger_percent_b_20", "rsi14", "adx14")

    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
    ) -> RuleSignal | None:
        read = read_all(snapshot, self.required_features)
        if read is None:
            return None
        if read["adx14"] >= ADX_RANGE_CEILING:
            return None

        percent_b = read["bollinger_percent_b_20"]
        rsi = read["rsi14"]
        if percent_b <= _ZERO and rsi <= RSI_OVERSOLD:
            direction = Direction.LONG
            stretch = _ramp(-percent_b, _ZERO, PERCENT_B_STRETCH)
            extremity = _ramp(RSI_OVERSOLD - rsi, _ZERO, RSI_REVERSION_SPAN)
        elif percent_b >= _ONE and rsi >= RSI_OVERBOUGHT:
            direction = Direction.SHORT
            stretch = _ramp(percent_b - _ONE, _ZERO, PERCENT_B_STRETCH)
            extremity = _ramp(rsi - RSI_OVERBOUGHT, _ZERO, RSI_REVERSION_SPAN)
        else:
            return None
        return RuleSignal(direction, _mean(stretch, extremity), read)


class VwapReversionRule:
    """A price stretched far from session VWAP in standard-deviation terms.

    This is the rule that proves the layer degrades rather than guesses. Both of
    its inputs are volume-derived, so it is skipped outright whenever VWAP is
    disabled for the session — which one candle with unknown volume is enough to
    cause — while every price-and-momentum rule beside it keeps running.
    """

    name = "vwap_reversion"
    required_features = ("vwap", "price_vs_vwap_sigma")

    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
    ) -> RuleSignal | None:
        read = read_all(snapshot, self.required_features)
        if read is None:
            return None

        sigma = read["price_vs_vwap_sigma"]
        if sigma <= -VWAP_SIGMA_TRIGGER:
            direction, magnitude = Direction.LONG, -sigma
        elif sigma >= VWAP_SIGMA_TRIGGER:
            direction, magnitude = Direction.SHORT, sigma
        else:
            return None

        score = _ramp(magnitude, VWAP_SIGMA_TRIGGER, VWAP_SIGMA_CEILING)
        return RuleSignal(direction, score, read)


class OpeningRangeBreakoutRule:
    """A close beyond the range the first fifteen minutes established.

    No clock check is needed. ``opening_range_high`` is withheld until the range
    has closed and frozen once it has, so the feature being available *is* the
    time-of-day gate — and it is the accurate one, since it also withholds
    itself on a session the engine did not observe from the start, where an
    "opening range" computed from an 11:00 start would be fiction.
    """

    name = "opening_range_breakout"
    required_features = ("opening_range_high", "opening_range_low", "atr14")

    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
    ) -> RuleSignal | None:
        read = read_all(snapshot, self.required_features)
        if read is None:
            return None

        close = snapshot.close
        if close > read["opening_range_high"]:
            direction = Direction.LONG
            extension = close - read["opening_range_high"]
        elif close < read["opening_range_low"]:
            direction = Direction.SHORT
            extension = read["opening_range_low"] - close
        else:
            return None

        scaled = safe_divide(extension, read["atr14"])
        if scaled is None:
            return None
        score = _ramp(scaled, _ZERO, OPENING_RANGE_ATR_CEILING)
        return RuleSignal(direction, score, read)


DEFAULT_RULES: tuple[Rule, ...] = (
    TrendContinuationRule(),
    BreakoutRule(),
    MeanReversionRule(),
    VwapReversionRule(),
    OpeningRangeBreakoutRule(),
)
"""The rule set a scanner uses when the caller names none.

Deliberately contradictory: trend continuation and mean reversion encode
opposing views and are gated on opposing regimes. Which of them earns its place
is a question for replay, and shipping only the one that sounds most convincing
would be answering it by assertion.
"""


__all__ = [
    "DEFAULT_RULES",
    "BreakoutRule",
    "MeanReversionRule",
    "OpeningRangeBreakoutRule",
    "Rule",
    "RuleSignal",
    "TrendContinuationRule",
    "VwapReversionRule",
    "available",
    "missing_features",
    "read_all",
]
