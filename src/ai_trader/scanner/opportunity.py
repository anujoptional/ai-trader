"""A bounded signed score for speed-discounted intraday target opportunities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType

from ai_trader.features import FEATURE_CONTEXT, FeatureSnapshot
from ai_trader.scanner.rules import available

COMPONENT_NAMES = (
    "trend_alignment",
    "momentum",
    "directional_pressure",
    "range_position",
    "mean_reversion",
    "trend_pullback",
    "session_pressure",
    "volume_confirmation",
)


def centered_sigmoid(raw: Decimal) -> Decimal:
    """2 * logistic(raw) - 1, evaluated without exponentiating a positive number."""
    if not raw.is_finite():
        raise ValueError("The raw opportunity score must be finite")
    with localcontext(FEATURE_CONTEXT):
        magnitude = 2 / (1 + (-abs(raw)).exp()) - 1
        return magnitude if raw >= 0 else -magnitude


def _bounded(value: Decimal) -> Decimal:
    return centered_sigmoid(2 * value)


def _mean(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / len(values) if values else Decimal(0)


@dataclass(frozen=True, slots=True)
class OpportunityInputs:
    components: tuple[Decimal, ...]
    reachability: Decimal
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SignedOpportunity:
    score: Decimal
    raw_score: Decimal
    reachability: Decimal
    components: Mapping[str, Decimal]
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


@dataclass(frozen=True, slots=True)
class TargetScoreConfig:
    """Small, versioned model; coefficients are fitted offline, never during a scan."""

    coefficients: tuple[Decimal, ...] = (Decimal(0),) * len(COMPONENT_NAMES)
    target_fraction: Decimal = Decimal("0.002")
    decay_minutes: Decimal = Decimal(5)
    square_off_minutes: Decimal = Decimal(360)
    transform_version: int = 1
    score_threshold: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        if self.transform_version != 1 or type(self.transform_version) is not int:
            raise ValueError("Unsupported opportunity transform version")
        if len(self.coefficients) != len(COMPONENT_NAMES):
            raise ValueError(
                "One coefficient is required for each opportunity component"
            )
        values = (
            *self.coefficients,
            self.target_fraction,
            self.decay_minutes,
            self.square_off_minutes,
            self.score_threshold,
        )
        if any(
            not isinstance(value, Decimal) or not value.is_finite() for value in values
        ):
            raise ValueError("Opportunity parameters must be finite Decimals")
        if not 0 < self.target_fraction < 1 or self.decay_minutes <= 0:
            raise ValueError("Target and decay must be positive, with target below one")
        if not 0 < self.square_off_minutes <= 375:
            raise ValueError("Opportunity cutoff must be inside the regular session")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("The absolute score threshold must be in [0, 1]")
        if any(abs(value) > 20 for value in self.coefficients):
            raise ValueError("Opportunity coefficients must be bounded by twenty")

    def inputs(self, snapshot: FeatureSnapshot) -> OpportunityInputs:
        with localcontext(FEATURE_CONTEXT):
            return self._inputs(snapshot)

    def _inputs(self, snapshot: FeatureSnapshot) -> OpportunityInputs:
        empty = (Decimal(0),) * len(COMPONENT_NAMES)
        minutes = available(snapshot, "minutes_since_session_open")
        if minutes is None or not snapshot.readiness.core_ready:
            return OpportunityInputs(empty, Decimal(0), "not_ready")
        remaining = self.square_off_minutes - minutes
        if minutes <= 0 or remaining <= 0:
            return OpportunityInputs(empty, Decimal(0), "outside_window")
        atr = available(snapshot, "atr14")
        atr_fraction = available(snapshot, "atr_pct")
        names = (
            "ema9",
            "ema21",
            "ema50",
            "ema9_slope_5",
            "ema21_slope_5",
            "return_1",
            "return_5",
            "return_15",
            "macd_histogram",
            "macd_histogram_change",
            "plus_di14",
            "minus_di14",
            "adx14",
            "rolling_high_20",
            "rolling_low_20",
            "rsi14",
            "bollinger_percent_b_20",
        )
        readings = {name: available(snapshot, name) for name in names}
        if (
            atr is None
            or atr_fraction is None
            or any(value is None for value in readings.values())
        ):
            return OpportunityInputs(empty, Decimal(0), "not_ready")
        if atr <= 0 or atr_fraction <= 0:
            return OpportunityInputs(empty, Decimal(0), "no_volatility")
        trend_regime = min(Decimal(1), max(Decimal(0), readings["adx14"] / 50))
        alignment = _mean(
            [
                _bounded((readings["ema9"] - readings["ema21"]) / atr),
                _bounded((readings["ema21"] - readings["ema50"]) / atr),
                _bounded(readings["ema9_slope_5"] / atr_fraction),
                _bounded(readings["ema21_slope_5"] / atr_fraction),
            ]
        )
        momentum = _mean(
            [
                _bounded(
                    readings[f"return_{period}"]
                    / (atr_fraction * Decimal(period).sqrt())
                )
                for period in (1, 5, 15)
            ]
            + [
                _bounded(readings["macd_histogram"] / atr),
                _bounded(readings["macd_histogram_change"] / atr),
            ]
        )
        directional_sum = readings["plus_di14"] + readings["minus_di14"]
        pressure = (
            (readings["plus_di14"] - readings["minus_di14"]) / directional_sum
            if directional_sum > 0
            else Decimal(0)
        )
        upper, lower = readings["rolling_high_20"], readings["rolling_low_20"]
        range_position = _bounded(
            (2 * snapshot.close - upper - lower) / (upper - lower + 2 * atr)
        )
        stretches = [
            _bounded((readings["rsi14"] - 50) / 20),
            _bounded(2 * (readings["bollinger_percent_b_20"] - Decimal("0.5"))),
        ]
        vwap_sigma = available(snapshot, "price_vs_vwap_sigma")
        if vwap_sigma is not None:
            stretches.append(_bounded(vwap_sigma / 2))
        reversion = -_mean(stretches) * (1 - trend_regime)
        pullback = -_bounded((snapshot.close - readings["ema9"]) / atr) * trend_regime
        session_terms: list[Decimal] = []
        distance = available(snapshot, "distance_from_session_open")
        if distance is not None:
            session_terms.append(_bounded(distance / (atr_fraction * minutes.sqrt())))
        position = available(snapshot, "position_in_session_range")
        if position is not None:
            session_terms.append(2 * position - 1)
        opening_high = available(snapshot, "opening_range_high")
        opening_low = available(snapshot, "opening_range_low")
        if opening_high is not None and opening_low is not None:
            session_terms.append(
                _bounded(
                    (2 * snapshot.close - opening_high - opening_low)
                    / (opening_high - opening_low + 2 * atr)
                )
            )
        session_pressure = _mean(session_terms) * trend_regime
        volume_ratio = available(snapshot, "volume_ratio_20")
        confirmation = (
            _mean([alignment, momentum, pressure]) * volume_ratio / (1 + volume_ratio)
            if volume_ratio is not None and volume_ratio >= 0
            else Decimal(0)
        )
        reachability = (
            1 - (-self.decay_minutes * (atr_fraction / self.target_fraction) ** 2).exp()
        ) * (1 - (-remaining / self.decay_minutes).exp())
        return OpportunityInputs(
            (
                alignment,
                momentum,
                pressure,
                range_position * trend_regime,
                reversion,
                pullback,
                session_pressure,
                confirmation,
            ),
            reachability,
        )

    def evaluate(self, snapshot: FeatureSnapshot) -> SignedOpportunity:
        with localcontext(FEATURE_CONTEXT):
            inputs = self._inputs(snapshot)
            raw = inputs.reachability * sum(
                (
                    coefficient * value
                    for coefficient, value in zip(
                        self.coefficients, inputs.components, strict=True
                    )
                ),
                Decimal(0),
            )
            return SignedOpportunity(
                centered_sigmoid(raw),
                raw,
                inputs.reachability,
                dict(zip(COMPONENT_NAMES, inputs.components, strict=True)),
                inputs.reason,
            )
