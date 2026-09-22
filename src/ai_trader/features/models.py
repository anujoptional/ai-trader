"""Immutable models describing the quantitative state of one instrument.

These models are broker-independent and describe derived numbers only. They
carry no display formatting: every value is the exact ``Decimal`` the engine
computed, so a consumer can round it however it likes without the engine
having silently rounded first.

Readiness is explicit rather than inferred. ``FeatureReadiness`` carries one
flag per derived field, named identically to it, and that flag is true exactly
when the field holds a value. A consumer therefore never has to guess whether
``None`` means "not enough history yet" or "this instrument is broken", and the
correspondence is mechanical rather than a claim a docstring makes: the engine
builds both from one mapping, and ``DERIVED_FEATURE_NAMES`` lets a test walk
every pair without naming any of them.

Several derived values can be unavailable while their obvious parent is
available, which is why each gets its own flag rather than sharing one.
``atr_pct`` divides by the close, ``price_vs_vwap`` by the VWAP,
``price_vs_vwap_sigma`` by the VWAP's dispersion, ``bollinger_percent_b_20`` by
the band width and ``position_in_session_range`` by the session's range so far.
A zero denominator leaves the child ``None`` while the parent is a perfectly
good number: a session that has traded at one price all day has a real high and
a real low, and no meaningful position between them.

Session aggregates carry a further condition that is not about history length
at all. An engine that first sees an instrument at 11:00 knows the highest
price since 11:00, which is not the session high, so it withholds every session
aggregate for the rest of that day rather than publishing a number that looks
like one. ``minutes_since_session_open`` is the single exception, because it is
clock arithmetic against 09:15 IST rather than an observation, and is correct
however late the engine started.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal

from ai_trader.broker import Instrument


@dataclass(frozen=True, slots=True)
class FeatureReadiness:
    """Availability of every derived feature on a snapshot."""

    return_1: bool = False
    return_5: bool = False
    return_15: bool = False
    ema9: bool = False
    ema21: bool = False
    ema50: bool = False
    ema9_slope_5: bool = False
    ema21_slope_5: bool = False
    rsi14: bool = False
    macd: bool = False
    macd_signal: bool = False
    macd_histogram: bool = False
    macd_histogram_change: bool = False
    true_range: bool = False
    atr14: bool = False
    atr_pct: bool = False
    candle_range_pct: bool = False
    plus_di14: bool = False
    minus_di14: bool = False
    adx14: bool = False
    rolling_high_20: bool = False
    rolling_low_20: bool = False
    distance_from_high_20: bool = False
    distance_from_low_20: bool = False
    sma20: bool = False
    bollinger_upper_20: bool = False
    bollinger_lower_20: bool = False
    bollinger_bandwidth_20: bool = False
    bollinger_percent_b_20: bool = False
    vwap: bool = False
    price_vs_vwap: bool = False
    vwap_deviation: bool = False
    price_vs_vwap_sigma: bool = False
    volume_ratio_20: bool = False
    obv: bool = False
    session_open: bool = False
    session_high: bool = False
    session_low: bool = False
    session_range_pct: bool = False
    position_in_session_range: bool = False
    distance_from_session_open: bool = False
    minutes_since_session_open: bool = False
    session_volume: bool = False
    opening_range_high: bool = False
    opening_range_low: bool = False
    distance_from_opening_range_high: bool = False
    distance_from_opening_range_low: bool = False

    @property
    def core_ready(self) -> bool:
        """Whether every price and momentum feature is available.

        The set names one feature per family — trend, momentum, volatility,
        trend strength, position in range, dispersion — rather than every flag,
        because within a family the slowest member implies the rest. It is the
        single question a scanner asks before trusting an instrument at all.

        Two whole classes of feature are deliberately excluded. Volume-dependent
        readiness is, because VWAP, relative volume and on-balance volume can be
        unavailable for a whole session while trend, momentum and volatility
        remain perfectly valid. Session-dependent readiness is, for a sharper
        reason: an engine started mid-session withholds its session aggregates
        for the rest of that day by design, so including them would wedge
        ``core_ready`` false until the next open no matter how much price
        history accumulated.
        """
        return (
            self.return_15
            and self.ema50
            and self.rsi14
            and self.macd_histogram_change
            and self.atr14
            and self.adx14
            and self.rolling_high_20
            and self.bollinger_upper_20
        )


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """The quantitative state of one instrument after one completed candle."""

    instrument: Instrument
    candle_start_time: datetime
    candle_end_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None
    readiness: FeatureReadiness
    return_1: Decimal | None = None
    return_5: Decimal | None = None
    return_15: Decimal | None = None
    ema9: Decimal | None = None
    ema21: Decimal | None = None
    ema50: Decimal | None = None
    ema9_slope_5: Decimal | None = None
    ema21_slope_5: Decimal | None = None
    rsi14: Decimal | None = None
    macd: Decimal | None = None
    macd_signal: Decimal | None = None
    macd_histogram: Decimal | None = None
    macd_histogram_change: Decimal | None = None
    true_range: Decimal | None = None
    atr14: Decimal | None = None
    atr_pct: Decimal | None = None
    candle_range_pct: Decimal | None = None
    plus_di14: Decimal | None = None
    minus_di14: Decimal | None = None
    adx14: Decimal | None = None
    rolling_high_20: Decimal | None = None
    rolling_low_20: Decimal | None = None
    distance_from_high_20: Decimal | None = None
    distance_from_low_20: Decimal | None = None
    sma20: Decimal | None = None
    bollinger_upper_20: Decimal | None = None
    bollinger_lower_20: Decimal | None = None
    bollinger_bandwidth_20: Decimal | None = None
    bollinger_percent_b_20: Decimal | None = None
    vwap: Decimal | None = None
    price_vs_vwap: Decimal | None = None
    vwap_deviation: Decimal | None = None
    price_vs_vwap_sigma: Decimal | None = None
    volume_ratio_20: Decimal | None = None
    obv: Decimal | None = None
    session_open: Decimal | None = None
    session_high: Decimal | None = None
    session_low: Decimal | None = None
    session_range_pct: Decimal | None = None
    position_in_session_range: Decimal | None = None
    distance_from_session_open: Decimal | None = None
    minutes_since_session_open: Decimal | None = None
    session_volume: Decimal | None = None
    opening_range_high: Decimal | None = None
    opening_range_low: Decimal | None = None
    distance_from_opening_range_high: Decimal | None = None
    distance_from_opening_range_low: Decimal | None = None


DERIVED_FEATURE_NAMES: tuple[str, ...] = tuple(
    field.name for field in fields(FeatureReadiness)
)
"""Every derived feature name, in snapshot order.

Derived from ``FeatureReadiness`` rather than written out, so a new feature
reaches the engine's readiness mapping and the CLI's output by being declared
once. A consumer that spells the list itself would drift from the dataclass
without anything failing.
"""


__all__ = ["DERIVED_FEATURE_NAMES", "FeatureReadiness", "FeatureSnapshot"]
