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

Two derived values can be unavailable while their obvious parent is available,
which is why each gets its own flag rather than sharing one. ``atr_pct``
divides by the close, and ``price_vs_vwap`` divides by the VWAP; a zero
denominator leaves the child ``None`` while the parent is a perfectly good
number.
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
    rolling_high_20: bool = False
    rolling_low_20: bool = False
    distance_from_high_20: bool = False
    distance_from_low_20: bool = False
    vwap: bool = False
    price_vs_vwap: bool = False
    volume_ratio_20: bool = False

    @property
    def core_ready(self) -> bool:
        """Whether every price and momentum feature is available.

        Volume-dependent readiness is deliberately excluded: VWAP and relative
        volume can be unavailable for a whole session while trend, momentum and
        volatility remain perfectly valid.
        """
        return (
            self.return_15
            and self.ema50
            and self.rsi14
            and self.macd_histogram_change
            and self.atr14
            and self.rolling_high_20
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
    rolling_high_20: Decimal | None = None
    rolling_low_20: Decimal | None = None
    distance_from_high_20: Decimal | None = None
    distance_from_low_20: Decimal | None = None
    vwap: Decimal | None = None
    price_vs_vwap: Decimal | None = None
    volume_ratio_20: Decimal | None = None


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
