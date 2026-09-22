"""Reference tests for the indicator primitives.

Every expected value here is arithmetic worked out by hand and written as a
literal, never a second implementation of the same formula. Periods are chosen
so that each smoothing factor has a finite decimal expansion — ``2 / (3 + 1)``
and ``2 / (9 + 1)`` are exact, while the production ``2 / (21 + 1)`` is not —
which is what lets these assertions be exact equalities rather than tolerances.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Context, Decimal, Inexact, localcontext

import pytest

from ai_trader.features.indicators import (
    AverageTrueRange,
    DirectionalMovementIndex,
    ExponentialMovingAverage,
    MovingAverageConvergenceDivergence,
    OnBalanceVolume,
    RelativeStrengthIndex,
    RollingDispersion,
    SessionContext,
    SessionVwap,
    WilderAverage,
    ratio_change,
    safe_divide,
    session_date,
)

_SESSION = date(2026, 9, 14)
_NEXT_SESSION = date(2026, 9, 15)
_SESSION_OPEN = datetime(2026, 9, 14, 3, 45, tzinfo=UTC)
_NEXT_SESSION_OPEN = datetime(2026, 9, 15, 3, 45, tzinfo=UTC)


def _at(minute: int, *, session_open: datetime = _SESSION_OPEN) -> datetime:
    """The start of the candle that many minutes into the session."""
    return session_open + timedelta(minutes=minute)


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


def test_dispersion_stays_unavailable_until_its_window_is_full() -> None:
    dispersion = RollingDispersion(4)

    for close in ("98", "98", "102"):
        dispersion.update(Decimal(close))
        assert dispersion.is_full is False
        # A "four-period mean" of three closes is a different statistic wearing
        # the same name, and a band drawn from it would be far too tight.
        assert dispersion.mean is None
        assert dispersion.standard_deviation is None

    dispersion.update(Decimal("102"))

    assert dispersion.is_full is True
    assert dispersion.mean == Decimal("100")


def test_dispersion_measures_the_spread_about_its_own_mean() -> None:
    dispersion = RollingDispersion(4)
    for close in ("98", "98", "102", "102"):
        dispersion.update(Decimal(close))

    # Mean 100, so the deviations are -2, -2, 2, 2 and the squares 4, 4, 4, 4.
    # Population variance is 16 / 4 = 4, whose root is exactly 2.
    assert dispersion.standard_deviation == Decimal("2")


def test_a_flat_window_has_no_spread_rather_than_a_negative_one() -> None:
    dispersion = RollingDispersion(4)
    for _ in range(4):
        dispersion.update(Decimal("100"))

    assert dispersion.mean == Decimal("100")
    # Summing squared deviations cannot go negative, which is the whole reason
    # this class pays for a second pass instead of the one-pass identity.
    assert dispersion.standard_deviation == Decimal("0")


def test_dispersion_forgets_closes_that_leave_its_window() -> None:
    dispersion = RollingDispersion(4)
    for close in ("98", "98", "102", "102"):
        dispersion.update(Decimal(close))

    dispersion.update(Decimal("106"))

    # The oldest 98 has been evicted, leaving 98, 102, 102, 106: a mean of 102.
    # A running-sum implementation that forgot to subtract would still say 100.
    assert dispersion.mean == Decimal("102")


_DMI_UPTREND = (
    ("10", "8", "8"),
    ("12", "10", "10"),
    ("14", "12", "12"),
    ("16", "14", "14"),
)
"""A two-rupee-per-candle ramp that closes on its low, so every true range is 4."""


def test_the_directional_index_reads_a_clean_ramp_as_one_sided() -> None:
    index = DirectionalMovementIndex(period=2)

    for high, low, close in _DMI_UPTREND:
        index.update(Decimal(high), Decimal(low), Decimal(close))

    # Each candle after the first moves its high up 2 and its low up 2, so the
    # downward movement is negative and only the upward side is credited. The
    # smoothed true range is 4 and the smoothed +DM is 2, giving 100 * 2 / 4.
    assert index.plus_di == Decimal("50")
    assert index.minus_di == Decimal("0")
    # DX is then 100 * |50 - 0| / 50 on every candle, and the ADX is its mean.
    assert index.adx == Decimal("100")


def test_the_directional_index_turns_when_the_trend_does() -> None:
    index = DirectionalMovementIndex(period=2)
    for high, low, close in _DMI_UPTREND:
        index.update(Decimal(high), Decimal(low), Decimal(close))

    # A reversal: the high falls 1 while the low falls 3, so the downward side
    # is the larger move and takes the whole attribution.
    index.update(Decimal("15"), Decimal("11"), Decimal("11"))

    # True range stays at 4, +DM smooths to (2 + 0) / 2 = 1 and -DM to
    # (0 + 3) / 2 = 1.5, giving 100 * 1 / 4 and 100 * 1.5 / 4.
    assert index.plus_di == Decimal("25")
    assert index.minus_di == Decimal("37.5")
    # DX drops to 100 * 12.5 / 62.5 = 20, and the ADX smooths (100 + 20) / 2.
    # Trend strength decays rather than flipping instantly, which is the point.
    assert index.adx == Decimal("60")


def test_an_outside_candle_is_credited_to_neither_side() -> None:
    index = DirectionalMovementIndex(period=2)

    # Each candle extends two rupees past the previous high and two past its
    # low. Neither move is larger, so Wilder credits neither.
    for high, low in (("10", "10"), ("12", "8"), ("14", "6"), ("16", "4")):
        index.update(Decimal(high), Decimal(low), Decimal("10"))

    assert index.plus_di == Decimal("0")
    assert index.minus_di == Decimal("0")
    # Equal readings are a real absence of direction rather than a missing
    # number, so the index is zero and the ADX behind it keeps advancing.
    assert index.adx == Decimal("0")


def test_a_motionless_instrument_has_no_direction_to_report() -> None:
    index = DirectionalMovementIndex(period=2)

    for _ in range(3):
        index.update(Decimal("100"), Decimal("100"), Decimal("100"))

    # Zero average true range is the one case that is genuinely undefined
    # rather than zero: there is no range to express the movement as a share of.
    assert index.plus_di is None
    assert index.minus_di is None
    assert index.adx is None


def test_on_balance_volume_needs_a_close_to_have_moved_against() -> None:
    obv = OnBalanceVolume()

    obv.update(_SESSION, Decimal("100"), 1_000)

    # The session's first candle has no same-session predecessor, so there is
    # nothing to call this volume buying or selling.
    assert obv.value is None


def test_on_balance_volume_signs_each_candle_by_its_direction() -> None:
    obv = OnBalanceVolume()
    obv.update(_SESSION, Decimal("100"), 1_000)

    obv.update(_SESSION, Decimal("102"), 500)
    assert obv.value == Decimal("500")

    obv.update(_SESSION, Decimal("101"), 300)
    assert obv.value == Decimal("200")

    # An unchanged close is attribution too: it contributes exactly zero, which
    # is different from contributing nothing.
    obv.update(_SESSION, Decimal("101"), 700)
    assert obv.value == Decimal("200")


def test_one_missing_volume_disables_on_balance_volume_for_the_session() -> None:
    obv = OnBalanceVolume()
    obv.update(_SESSION, Decimal("100"), 1_000)
    obv.update(_SESSION, Decimal("102"), 500)

    obv.update(_SESSION, Decimal("105"), None)
    assert obv.value is None

    # A running total with a hole in it is not a smaller total, it is a wrong
    # one, so later candles cannot repair it either.
    obv.update(_SESSION, Decimal("110"), 400)
    assert obv.value is None


def test_on_balance_volume_restarts_at_a_session_boundary() -> None:
    obv = OnBalanceVolume()
    obv.update(_SESSION, Decimal("100"), 1_000)
    obv.update(_SESSION, Decimal("105"), None)

    obv.update(_NEXT_SESSION, Decimal("100"), 900)
    # A new day clears both the total and the disabled flag, and starts again
    # with no predecessor to attribute against.
    assert obv.value is None

    obv.update(_NEXT_SESSION, Decimal("101"), 900)
    assert obv.value == Decimal("900")


def test_a_session_context_tracks_the_days_extremes_and_turnover() -> None:
    context = SessionContext()

    context.update(_at(0), Decimal("100"), Decimal("102"), Decimal("99"), 1_000)
    context.update(_at(1), Decimal("101"), Decimal("104"), Decimal("101"), 500)

    assert context.is_anchored is True
    # The open is the first candle's, not the latest one's.
    assert context.open == Decimal("100")
    assert context.high == Decimal("104")
    assert context.low == Decimal("99")
    assert context.volume == Decimal("1500")


def test_a_session_joined_after_its_opening_range_reports_nothing() -> None:
    context = SessionContext()

    context.update(_at(20), Decimal("100"), Decimal("102"), Decimal("99"), 1_000)

    # Naming the highest price seen since 09:35 the "session high" would put a
    # whole-day label on the back half of the day.
    assert context.is_anchored is False
    assert context.open is None
    assert context.high is None
    assert context.low is None
    assert context.volume is None


def test_a_pre_open_candle_does_not_anchor_the_session() -> None:
    """The architecture warms the stack up pre-open, so these do arrive.

    Anchoring on one would put the day's open at a pre-open equilibrium price
    and fold pre-open prints into the opening range. It would also be
    unrecoverable: the reset fires only when the session label changes, so a
    pre-open candle claiming today's label means 09:15 never re-anchors.
    """
    context = SessionContext()

    context.update(_at(-10), Decimal("500"), Decimal("505"), Decimal("495"), 9_000)

    assert context.is_anchored is False
    assert context.open is None

    context.update(_at(0), Decimal("100"), Decimal("102"), Decimal("99"), 1_000)

    # 09:15 still anchors, and on its own open rather than the pre-open one.
    assert context.is_anchored is True
    assert context.open == Decimal("100")
    assert context.high == Decimal("102")
    assert context.low == Decimal("99")
    assert context.volume == Decimal("1000")


def test_the_opening_range_closes_at_the_first_candle_past_it() -> None:
    context = SessionContext()

    for minute in range(15):
        high = Decimal(102 + minute)
        context.update(_at(minute), Decimal("100"), high, Decimal(99 - minute), 1_000)
        assert context.opening_range_high is None
        assert context.opening_range_low is None

    context.update(_at(15), Decimal("200"), Decimal("202"), Decimal("198"), 1_000)

    # The 09:30 candle closes the range rather than joining it, so its own high
    # of 202 is not the opening range's.
    assert context.opening_range_high == Decimal("116")
    assert context.opening_range_low == Decimal("85")
    # But it does count towards the session, which spans the whole day.
    assert context.high == Decimal("202")


def test_one_missing_volume_disables_the_session_total() -> None:
    context = SessionContext()
    context.update(_at(0), Decimal("100"), Decimal("102"), Decimal("99"), 1_000)

    context.update(_at(1), Decimal("100"), Decimal("102"), Decimal("99"), None)
    assert context.volume is None

    context.update(_at(2), Decimal("100"), Decimal("102"), Decimal("99"), 500)
    assert context.volume is None
    # The prices either side of the hole are still perfectly good.
    assert context.high == Decimal("102")


def test_a_session_context_restarts_at_a_session_boundary() -> None:
    context = SessionContext()
    for minute in range(20):
        context.update(
            _at(minute), Decimal("100"), Decimal("120"), Decimal("80"), 1_000
        )

    context.update(
        _at(0, session_open=_NEXT_SESSION_OPEN),
        Decimal("50"),
        Decimal("52"),
        Decimal("48"),
        7,
    )

    assert context.open == Decimal("50")
    assert context.high == Decimal("52")
    assert context.low == Decimal("48")
    assert context.volume == Decimal("7")
    # Today's opening range has not closed yet, so yesterday's must not stand in.
    assert context.opening_range_high is None


def test_the_vwap_deviation_is_the_spread_of_turnover_about_the_average() -> None:
    vwap = SessionVwap()

    # Symmetric candles, so each typical price is its own close: 100 then 106.
    vwap.update(_SESSION, Decimal("101"), Decimal("99"), Decimal("100"), 1)
    vwap.update(_SESSION, Decimal("107"), Decimal("105"), Decimal("106"), 1)

    assert vwap.value == Decimal("103")
    # E[x^2] - E[x]^2 = (10000 + 11236) / 2 - 103^2 = 10618 - 10609 = 9.
    assert vwap.deviation == Decimal("3")


def test_the_vwap_deviation_weights_prices_by_the_shares_behind_them() -> None:
    vwap = SessionVwap()

    vwap.update(_SESSION, Decimal("101"), Decimal("99"), Decimal("100"), 1)
    vwap.update(_SESSION, Decimal("106"), Decimal("104"), Decimal("105"), 4)

    # Four fifths of the day traded at 105, so the average sits there, not
    # halfway. (100 + 4 * 105) / 5 = 104.
    assert vwap.value == Decimal("104")
    # (10000 + 4 * 11025) / 5 - 104^2 = 10820 - 10816 = 4.
    assert vwap.deviation == Decimal("2")


def test_a_session_traded_at_one_price_has_no_spread_rather_than_an_error() -> None:
    vwap = SessionVwap()

    vwap.update(_SESSION, Decimal("100"), Decimal("100"), Decimal("100"), 1_000)

    # The one-pass identity can land a hair below zero here, and the pinned
    # context raises on a negative square root rather than returning NaN, so
    # this assertion is load-bearing rather than tidiness.
    assert vwap.value == Decimal("100")
    assert vwap.deviation == Decimal("0")


def test_the_vwap_deviation_is_withheld_exactly_when_vwap_is() -> None:
    vwap = SessionVwap()
    assert vwap.deviation is None

    vwap.update(_SESSION, Decimal("101"), Decimal("99"), Decimal("100"), 1_000)
    vwap.update(_SESSION, Decimal("101"), Decimal("99"), Decimal("100"), None)

    assert vwap.value is None
    assert vwap.deviation is None
