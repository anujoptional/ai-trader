"""Reference tests for the indicator primitives.

Every expected value here is arithmetic worked out by hand and written as a
literal, never a second implementation of the same formula. Periods are chosen
so that each smoothing factor has a finite decimal expansion — ``2 / (3 + 1)``
and ``2 / (9 + 1)`` are exact, while the production ``2 / (21 + 1)`` is not —
which is what lets these assertions be exact equalities rather than tolerances.
"""

from datetime import UTC, date, datetime
from decimal import Context, Decimal, Inexact, localcontext

import pytest

from ai_trader.features.indicators import (
    AverageTrueRange,
    ExponentialMovingAverage,
    MovingAverageConvergenceDivergence,
    RelativeStrengthIndex,
    SessionVwap,
    WilderAverage,
    ratio_change,
    safe_divide,
    session_date,
)

_SESSION = date(2026, 9, 14)
_NEXT_SESSION = date(2026, 9, 15)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (None, Decimal("2")),
        (Decimal("1"), None),
        (Decimal("1"), Decimal("0")),
    ],
)
def test_safe_divide_withholds_a_quotient_it_cannot_define(
    numerator: Decimal | None,
    denominator: Decimal | None,
) -> None:
    assert safe_divide(numerator, denominator) is None


def test_safe_divide_returns_the_quotient() -> None:
    assert safe_divide(Decimal("7"), Decimal("2")) == Decimal("3.5")


def test_ratio_change_is_a_fraction_not_a_percentage() -> None:
    assert ratio_change(Decimal("110"), Decimal("100")) == Decimal("0.1")
    assert ratio_change(Decimal("90"), Decimal("100")) == Decimal("-0.1")


@pytest.mark.parametrize(
    ("current", "base"),
    [
        (Decimal("110"), Decimal("0")),
        (Decimal("110"), None),
        (None, Decimal("100")),
    ],
)
def test_ratio_change_withholds_a_change_it_cannot_define(
    current: Decimal | None,
    base: Decimal | None,
) -> None:
    assert ratio_change(current, base) is None


def test_session_date_follows_the_indian_trading_day() -> None:
    # 03:45 UTC is 09:15 IST, the NSE open on the same calendar date.
    assert session_date(datetime(2026, 9, 14, 3, 45, tzinfo=UTC)) == date(2026, 9, 14)
    # 18:30 UTC is midnight IST, which already belongs to the next trading day.
    assert session_date(datetime(2026, 9, 14, 18, 30, tzinfo=UTC)) == date(2026, 9, 15)


def test_ema_stays_unavailable_until_its_seed_window_is_full() -> None:
    ema = ExponentialMovingAverage(9)

    results = [ema.update(Decimal(close)) for close in range(10, 18)]

    assert results == [None] * 8
    assert ema.value is None


def test_ema_seeds_on_the_simple_average_then_applies_its_smoothing() -> None:
    ema = ExponentialMovingAverage(9)

    seeded = None
    for close in range(10, 19):
        seeded = ema.update(Decimal(close))

    # (10 + 11 + ... + 18) / 9 == 126 / 9 == 14.
    assert seeded == Decimal("14")
    # alpha == 2 / (9 + 1) == 0.2, so 0.2 * 20 + 0.8 * 14 == 15.2.
    assert ema.update(Decimal("20")) == Decimal("15.2")
    assert ema.value == Decimal("15.2")


def test_ema_period_must_be_positive() -> None:
    with pytest.raises(ValueError, match="EMA period must be positive"):
        ExponentialMovingAverage(0)


def test_wilder_average_seeds_on_the_mean_then_smooths_by_its_period() -> None:
    average = WilderAverage(4)

    assert [average.update(Decimal(value)) for value in (1, 2, 3)] == [None] * 3
    # (1 + 2 + 3 + 4) / 4 == 2.5.
    assert average.update(Decimal("4")) == Decimal("2.5")
    # (2.5 * 3 + 6) / 4 == 13.5 / 4 == 3.375.
    assert average.update(Decimal("6")) == Decimal("3.375")
    assert average.value == Decimal("3.375")


def test_wilder_average_period_must_be_positive() -> None:
    with pytest.raises(ValueError, match="Wilder average period must be positive"):
        WilderAverage(0)


def test_rsi_needs_one_more_close_than_its_period() -> None:
    rsi = RelativeStrengthIndex(period=14)

    # Fourteen closes yield only thirteen changes, one short of the seed.
    results = [rsi.update(Decimal(100 + step)) for step in range(14)]

    assert results == [None] * 14
    # The fourteenth change completes the seed on a pure uptrend.
    assert rsi.update(Decimal("114")) == Decimal("100")


def test_rsi_reads_a_pure_downtrend_as_zero() -> None:
    rsi = RelativeStrengthIndex(period=14)

    value = None
    for step in range(15):
        value = rsi.update(Decimal(100 - step))

    assert value == Decimal("0")


def test_rsi_reads_a_flat_stretch_as_neutral() -> None:
    rsi = RelativeStrengthIndex(period=14)

    value = None
    for _ in range(15):
        value = rsi.update(Decimal("100"))

    # Both averages are zero and the ratio is undefined; a name that simply did
    # not move must not read to a scanner as maximally overbought.
    assert value == Decimal("50")


def test_rsi_smooths_gains_and_losses_the_wilder_way() -> None:
    rsi = RelativeStrengthIndex(period=2)

    assert rsi.update(Decimal("100")) is None
    assert rsi.update(Decimal("101")) is None
    # Seeded on changes +1 and -1: both averages are (1 + 0) / 2 == 0.5, so the
    # relative strength is 1 and the RSI is 100 - 100 / 2 == 50.
    assert rsi.update(Decimal("100")) == Decimal("50")
    # Change +1.5: gain (0.5 * 1 + 1.5) / 2 == 1, loss (0.5 * 1 + 0) / 2 == 0.25,
    # so the relative strength is 4 and the RSI is 100 - 100 / 5 == 80.
    assert rsi.update(Decimal("101.5")) == Decimal("80")


_MACD_CLOSES = ("10", "12", "14", "16", "20", "17.8", "22")
"""Closes chosen so a 3/4/3 MACD has exact smoothing factors throughout."""


def test_macd_components_become_available_in_dependency_order() -> None:
    macd = MovingAverageConvergenceDivergence(fast=3, slow=4, signal=3)

    availability = []
    for close in _MACD_CLOSES:
        macd.update(Decimal(close))
        availability.append(
            (
                macd.macd is not None,
                macd.signal is not None,
                macd.histogram is not None,
                macd.histogram_change is not None,
            )
        )

    # The slow EMA gates the MACD line, the MACD line gates the signal and
    # histogram, and the histogram gates its own change.
    assert availability == [
        (False, False, False, False),
        (False, False, False, False),
        (False, False, False, False),
        (True, False, False, False),
        (True, False, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ]


def test_macd_matches_hand_computed_moving_averages() -> None:
    macd = MovingAverageConvergenceDivergence(fast=3, slow=4, signal=3)

    for close in _MACD_CLOSES:
        macd.update(Decimal(close))

    # The 3-period EMA runs 12, 14, 17, 17.4, 19.7 and the 4-period EMA runs
    # 13, 15.8, 16.6, 18.76, so the MACD line runs 1, 1.2, 0.8, 0.94.
    assert macd.macd == Decimal("0.94")
    # The signal seeds on (1 + 1.2 + 0.8) / 3 == 1, then 0.5 * 0.94 + 0.5 * 1.
    assert macd.signal == Decimal("0.97")
    assert macd.histogram == Decimal("-0.03")
    # The previous histogram was 0.8 - 1 == -0.2, so the histogram rose 0.17.
    assert macd.histogram_change == Decimal("0.17")


def test_macd_fast_period_must_be_shorter_than_its_slow_period() -> None:
    with pytest.raises(ValueError, match="fast period must be shorter"):
        MovingAverageConvergenceDivergence(fast=26, slow=26)


_ATR_CANDLES = (
    ("12", "10", "11"),
    ("13", "11", "12"),
    ("15", "12", "14"),
    ("16", "13", "15"),
    ("20", "15", "19"),
)
"""High, low and close triples whose true ranges are 2, 2, 3, 3 and 5."""


def test_the_first_candle_has_no_gap_to_measure() -> None:
    atr = AverageTrueRange(period=4)

    atr.update(Decimal("12"), Decimal("10"), Decimal("11"))

    assert atr.true_range == Decimal("2")
    assert atr.value is None


def test_true_range_spans_a_gap_from_the_previous_close() -> None:
    atr = AverageTrueRange(period=4)
    atr.update(Decimal("11"), Decimal("9"), Decimal("10"))

    atr.update(Decimal("20"), Decimal("18"), Decimal("19"))

    # The candle's own span is only 2, but it opened 10 above the previous close.
    assert atr.true_range == Decimal("10")


def test_atr_seeds_on_the_mean_true_range_then_smooths() -> None:
    atr = AverageTrueRange(period=4)

    for high, low, close in _ATR_CANDLES[:4]:
        atr.update(Decimal(high), Decimal(low), Decimal(close))

    # True ranges 2, 2, 3, 3 average to 10 / 4 == 2.5.
    assert atr.value == Decimal("2.5")

    high, low, close = _ATR_CANDLES[4]
    atr.update(Decimal(high), Decimal(low), Decimal(close))

    # (2.5 * 3 + 5) / 4 == 12.5 / 4 == 3.125.
    assert atr.true_range == Decimal("5")
    assert atr.value == Decimal("3.125")


def test_vwap_weights_prices_by_volume() -> None:
    vwap = SessionVwap()

    vwap.update(_SESSION, Decimal("100"), Decimal("100"), Decimal("100"), 10)
    vwap.update(_SESSION, Decimal("102"), Decimal("102"), Decimal("102"), 30)

    # (100 * 10 + 102 * 30) / 40 == 4060 / 40 == 101.5.
    assert vwap.value == Decimal("101.5")


def test_vwap_weights_the_typical_price_not_the_close() -> None:
    vwap = SessionVwap()

    vwap.update(_SESSION, Decimal("102"), Decimal("99"), Decimal("99"), 10)

    # (102 + 99 + 99) / 3 == 100, which the close alone would have read as 99.
    assert vwap.value == Decimal("100")


def test_vwap_restarts_at_a_session_boundary() -> None:
    vwap = SessionVwap()
    vwap.update(_SESSION, Decimal("100"), Decimal("100"), Decimal("100"), 1_000)

    vwap.update(_NEXT_SESSION, Decimal("200"), Decimal("200"), Decimal("200"), 10)

    # Yesterday's thousand shares carry no weight into today.
    assert vwap.value == Decimal("200")


def test_vwap_is_unavailable_before_any_shares_trade() -> None:
    vwap = SessionVwap()

    vwap.update(_SESSION, Decimal("100"), Decimal("100"), Decimal("100"), 0)

    assert vwap.value is None


def test_one_missing_volume_disables_vwap_for_the_rest_of_the_session() -> None:
    vwap = SessionVwap()
    vwap.update(_SESSION, Decimal("100"), Decimal("100"), Decimal("100"), 10)

    vwap.update(_SESSION, Decimal("101"), Decimal("101"), Decimal("101"), None)
    vwap.update(_SESSION, Decimal("102"), Decimal("102"), Decimal("102"), 30)

    # An average over an unknown fraction of the session's turnover is not a
    # VWAP, so nothing is reported rather than something plausible-looking.
    assert vwap.value is None


def test_a_new_session_re_enables_vwap_after_a_missing_volume() -> None:
    vwap = SessionVwap()
    vwap.update(_SESSION, Decimal("100"), Decimal("100"), Decimal("100"), None)

    vwap.update(_NEXT_SESSION, Decimal("200"), Decimal("200"), Decimal("200"), 10)

    assert vwap.value == Decimal("200")


def test_an_ema_pins_its_own_smoothing_factor() -> None:
    reference = ExponentialMovingAverage(21)
    hostile = Context(prec=3)
    hostile.traps[Inexact] = True
    # ``2 / 22`` has no finite expansion, so a constructor that computed the
    # smoothing factor under the ambient context would raise here — and, absent
    # the trap, would quietly bake three digits into every value that follows.
    with localcontext(hostile):
        observed = ExponentialMovingAverage(21)

    for index in range(40):
        close = Decimal(100 + index % 7)
        assert observed.update(close) == reference.update(close)


def test_a_session_vwap_pins_its_own_context() -> None:
    # Both typical prices and the final division are non-terminating, so this
    # covers the accumulation and the reading of it, which pin separately.
    candles = (
        (Decimal("101"), Decimal("99"), Decimal("101"), 7),
        (Decimal("102"), Decimal("100"), Decimal("102"), 11),
    )
    reference = SessionVwap()
    for high, low, close, volume in candles:
        reference.update(_SESSION, high, low, close, volume)

    hostile = Context(prec=3)
    hostile.traps[Inexact] = True
    observed = SessionVwap()
    with localcontext(hostile):
        for high, low, close, volume in candles:
            observed.update(_SESSION, high, low, close, volume)
        value = observed.value

    assert value == reference.value


def test_a_true_range_gaps_from_the_low_not_from_the_close() -> None:
    """A down-gap is measured to the candle's low, which is Wilder's definition.

    The low sits below the close by construction, so a formula gapping from the
    close understates every downward gap. Here the correct third term, ``|94 -
    100| = 6``, is the largest of the three; gapping from the close would give
    ``|95 - 100| = 5`` and the high-low span of ``2`` would still lose, so the
    error is invisible in the maximum unless a fixture is built to expose it.
    """
    atr = AverageTrueRange(period=2)

    atr.update(Decimal("101"), Decimal("99"), Decimal("100"))
    assert atr.true_range == Decimal("2")
    assert atr.value is None

    atr.update(Decimal("96"), Decimal("94"), Decimal("95"))

    # max(96 - 94, |96 - 100|, |94 - 100|) = max(2, 4, 6).
    assert atr.true_range == Decimal("6")
    # Seeded by the mean of its first two true ranges: (2 + 6) / 2.
    assert atr.value == Decimal("4")
