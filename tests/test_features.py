"""Tests for the feature engine over sequences of completed candles.

The fixtures here are deliberately small and exactly divisible: closes like 100
and 110 make an expected return an exact ``Decimal("0.1")`` rather than a
tolerance, so a wrong answer fails loudly instead of nearly passing.
"""

import sys
from collections import deque
from collections.abc import Sized
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal, Inexact, localcontext
from threading import Barrier, Thread

import pytest

from ai_trader.broker import Instrument
from ai_trader.features import DERIVED_FEATURE_NAMES, FeatureEngine, FeatureSnapshot
from ai_trader.features.indicators import (
    AverageTrueRange,
    ExponentialMovingAverage,
    MovingAverageConvergenceDivergence,
    RelativeStrengthIndex,
)
from ai_trader.market.candles import Candle

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_TCS = Instrument(exchange="NSE", trading_symbol="TCS")

_RACING_THREADS = 4
"""Threads replaying one instrument at once in the contention test."""

_TIGHT_SWITCH_INTERVAL = 1e-6
"""Seconds between forced thread switches, short enough to expose a race."""

_SLOPE_HISTORY = 6
"""EMA values spanned by a slope: the lagged one, the current one, and between."""

_SLOPE_TREND_CANDLES = 40
"""Candles in the slope fixtures, past the 21-period EMA's seed and lag."""

_SESSION_OPEN = datetime(2026, 9, 14, 3, 45, tzinfo=UTC)
"""09:15 IST on Monday 14 September 2026."""

_NEXT_SESSION_OPEN = datetime(2026, 9, 15, 3, 45, tzinfo=UTC)
"""The same opening minute one trading day later."""

_ONE = Decimal(1)
_TWO = Decimal(2)
"""How far a default candle's high and low straddle its close."""

_MINIMUM_CANDLES = (
    ("true_range", 1),
    ("candle_range_pct", 1),
    ("vwap", 1),
    ("price_vs_vwap", 1),
    ("return_1", 2),
    ("return_5", 6),
    ("ema9", 9),
    ("atr14", 14),
    ("atr_pct", 14),
    ("ema9_slope_5", 14),
    ("rsi14", 15),
    ("return_15", 16),
    ("rolling_high_20", 20),
    ("rolling_low_20", 20),
    ("distance_from_high_20", 20),
    ("distance_from_low_20", 20),
    ("ema21", 21),
    ("volume_ratio_20", 21),
    ("ema21_slope_5", 26),
    ("macd", 26),
    ("macd_signal", 34),
    ("macd_histogram", 34),
    ("macd_histogram_change", 35),
    ("ema50", 50),
)
"""How many candles each feature needs before it can first be computed.

Every derived feature appears exactly once; a test below asserts that against
``DERIVED_FEATURE_NAMES`` so a new feature cannot be added without stating when
it becomes available.
"""


def _candle(
    minute: int,
    close: Decimal,
    *,
    instrument: Instrument = _RELIANCE,
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: int | None = 1_000,
    session_open: datetime = _SESSION_OPEN,
) -> Candle:
    """One candle ``minute`` minutes into a session, with four distinct prices.

    The default candle straddles its close symmetrically, so its typical price
    ``(high + low + close) / 3`` is exactly the close and every VWAP expectation
    below stays hand-checkable by eye. Open, high, low and close are nonetheless
    four different numbers, which is what stops a formula that reads the wrong
    one from passing anyway: a helper leaving them all equal makes the four
    interchangeable, and an EMA fed the high or an ``atr_pct`` divided by the
    open then produces the same answer as the correct code.
    """
    start = session_open + timedelta(minutes=minute)
    high_price = close + _TWO if high is None else high
    low_price = close - _TWO if low is None else low
    return Candle(
        instrument=instrument,
        start_time=start,
        end_time=start + timedelta(minutes=1),
        # Offset from the low rather than the close so that an explicit high or
        # low still yields an open inside the candle's own range.
        open=min(low_price + _ONE, high_price),
        high=high_price,
        low=low_price,
        close=close,
        volume=volume,
    )


def _ramp(count: int, *, instrument: Instrument = _RELIANCE) -> tuple[Candle, ...]:
    """A run of same-session candles that both rise and fall."""
    return tuple(
        _candle(minute, Decimal(100 + minute % 7), instrument=instrument)
        for minute in range(count)
    )


def _varied(count: int) -> tuple[Candle, ...]:
    """Candles that swing in both directions with uneven ranges and volumes."""
    candles = []
    for minute in range(count):
        close = Decimal(95 + (minute * 7) % 11)
        candles.append(
            _candle(
                minute,
                close,
                high=close + Decimal(minute % 3),
                low=close - Decimal(minute % 4),
                volume=500 + (minute * 13) % 700,
            )
        )
    return tuple(candles)


def _retained_containers(obj: object, prefix: str = "") -> dict[str, Sized]:
    """Every sized object reachable through ``__slots__``, named by its path."""
    found: dict[str, Sized] = {}
    for name in getattr(type(obj), "__slots__", ()):
        value = getattr(obj, name, None)
        if isinstance(value, Sized):
            found[f"{prefix}{name}"] = value
        else:
            found.update(_retained_containers(value, f"{prefix}{name}."))
    return found


def test_a_split_warm_up_matches_one_continuous_run() -> None:
    candles = _varied(60)

    continuous = FeatureEngine()
    continuous.warm_up(candles)

    split = FeatureEngine()
    split.warm_up(candles[:30])
    for candle in candles[30:]:
        split.update(candle)

    # Warm-up is not a mode, it is the earlier part of the same sequence, so a
    # feature computed during a backfill must equal the live one exactly.
    assert split.snapshot(_RELIANCE) == continuous.snapshot(_RELIANCE)


def test_interleaved_instruments_do_not_share_state() -> None:
    reliance = _varied(40)
    tcs = tuple(
        _candle(minute, Decimal(2_000 + minute % 5), instrument=_TCS)
        for minute in range(40)
    )

    interleaved = FeatureEngine()
    for pair in zip(reliance, tcs, strict=True):
        for candle in pair:
            interleaved.update(candle)

    alone = FeatureEngine()
    alone.warm_up(reliance)

    assert interleaved.snapshot(_RELIANCE) == alone.snapshot(_RELIANCE)
    tcs_snapshot = interleaved.snapshot(_TCS)
    assert tcs_snapshot is not None
    assert tcs_snapshot.close == Decimal("2004")


def test_warm_up_reports_how_many_candles_it_accepted() -> None:
    engine = FeatureEngine()
    candles = _ramp(5)

    assert engine.warm_up(candles) == 5
    assert engine.warm_up(candles) == 0

    # Replaying a backfill is harmless: four candles predate the newest one and
    # the fifth repeats it.
    assert engine.out_of_order_candle_count == 4
    assert engine.duplicate_candle_count == 1


def test_a_repeated_candle_leaves_no_trace_in_the_computed_state() -> None:
    candles = _varied(30)
    clean = FeatureEngine()
    clean.warm_up(candles)

    polluted = FeatureEngine()
    for minute, candle in enumerate(candles):
        polluted.update(candle)
        assert polluted.update(_candle(minute, Decimal("9999"))) is None

    assert polluted.duplicate_candle_count == 30
    assert polluted.out_of_order_candle_count == 0
    # A rejected candle must not reach an EMA or a Wilder average: neither has
    # an inverse, so one fold-in would bias every later value undetectably.
    assert polluted.snapshot(_RELIANCE) == clean.snapshot(_RELIANCE)


def test_an_older_candle_leaves_no_trace_in_the_computed_state() -> None:
    candles = _varied(30)
    clean = FeatureEngine()
    clean.warm_up(candles)

    polluted = FeatureEngine()
    polluted.warm_up(candles)

    assert polluted.update(_candle(5, Decimal("9999"))) is None
    assert polluted.out_of_order_candle_count == 1
    assert polluted.duplicate_candle_count == 0
    assert polluted.snapshot(_RELIANCE) == clean.snapshot(_RELIANCE)


def test_a_candle_from_a_finished_session_cannot_reopen_it() -> None:
    engine = FeatureEngine()
    engine.warm_up(_ramp(30))
    engine.update(
        _candle(0, Decimal("200"), volume=10, session_open=_NEXT_SESSION_OPEN)
    )
    clean = engine.snapshot(_RELIANCE)

    stale = engine.update(_candle(29, Decimal("105")))

    # A late backfill, or a broker replaying a gap it noticed after the close.
    # Accepting it would roll the VWAP session back to yesterday and fold a
    # stale close into every average, so the guard has to reject a whole
    # finished session and not merely an earlier candle within the current one.
    assert stale is None
    assert engine.out_of_order_candle_count == 1
    assert engine.snapshot(_RELIANCE) == clean


def test_a_gap_in_the_session_does_not_reject_the_next_candle() -> None:
    engine = FeatureEngine()
    engine.update(_candle(0, Decimal("100")))

    snapshot = engine.update(_candle(20, Decimal("110")))

    assert snapshot is not None
    assert engine.out_of_order_candle_count == 0
    # A one-candle return looks back one candle, not one minute: a real Groww
    # session has genuine holes in it.
    assert snapshot.return_1 == Decimal("0.1")


def test_returns_look_back_over_completed_candles() -> None:
    engine = FeatureEngine()

    snapshot = None
    for minute, close in enumerate(("80", "50", "95", "98", "99", "100", "125")):
        snapshot = engine.update(_candle(minute, Decimal(close)))

    assert snapshot is not None
    # 125 / 100 - 1.
    assert snapshot.return_1 == Decimal("0.25")
    # 125 / 50 - 1, five completed candles back.
    assert snapshot.return_5 == Decimal("1.5")
    assert snapshot.return_15 is None


def test_the_candle_range_is_expressed_against_its_own_close() -> None:
    engine = FeatureEngine()

    snapshot = engine.update(
        _candle(0, Decimal("100"), high=Decimal("103"), low=Decimal("100"))
    )

    assert snapshot is not None
    assert snapshot.true_range == Decimal("3")
    assert snapshot.candle_range_pct == Decimal("0.03")


def test_the_true_range_gaps_against_the_previous_candles_close() -> None:
    engine = FeatureEngine()
    engine.update(_candle(0, Decimal("100"), high=Decimal("101"), low=Decimal("99")))

    snapshot = engine.update(
        _candle(1, Decimal("120"), high=Decimal("121"), low=Decimal("119"))
    )

    assert snapshot is not None
    # The primitives are verified in isolation elsewhere; what this pins is that
    # the engine hands them the right numbers. The candle's own span is only 2,
    # so max(121 - 119, |121 - 100|, |119 - 100|) = 21 can only come out right
    # if high, low and the carried previous close all arrive in the right place.
    assert snapshot.true_range == Decimal("21")


def test_rolling_extremes_include_the_candle_being_measured() -> None:
    engine = FeatureEngine()
    engine.warm_up(
        tuple(
            _candle(minute, Decimal("105"), low=Decimal("100")) for minute in range(19)
        )
    )

    # Pinned to its own close so that this candle really does set a fresh high,
    # which is the thing being measured.
    snapshot = engine.update(
        _candle(19, Decimal("110"), high=Decimal("110"), low=Decimal("108"))
    )

    assert snapshot is not None
    assert snapshot.rolling_high_20 == Decimal("110")
    assert snapshot.rolling_low_20 == Decimal("100")
    # A fresh high reads as a distance of exactly zero rather than as a
    # breakout above a window that has not caught up yet.
    assert snapshot.distance_from_high_20 == Decimal("0")
    assert snapshot.distance_from_low_20 == Decimal("0.1")


def test_rolling_extremes_track_highs_and_lows_rather_than_closes() -> None:
    engine = FeatureEngine()
    # Every candle's high sits above, and its low below, every close in the
    # window, so an implementation reading closes cannot produce these numbers.
    engine.warm_up(
        tuple(
            _candle(minute, Decimal("100"), high=Decimal("130"), low=Decimal("70"))
            for minute in range(20)
        )
    )

    snapshot = engine.snapshot(_RELIANCE)

    assert snapshot is not None
    assert snapshot.rolling_high_20 == Decimal("130")
    assert snapshot.rolling_low_20 == Decimal("70")


def test_rolling_extremes_stay_unavailable_until_the_window_fills() -> None:
    engine = FeatureEngine()

    snapshot = None
    for candle in _ramp(19):
        snapshot = engine.update(candle)

    assert snapshot is not None
    # A "20-period high" drawn from nineteen candles would simply be untrue.
    assert snapshot.rolling_high_20 is None
    assert snapshot.rolling_low_20 is None
    assert snapshot.distance_from_high_20 is None
    assert snapshot.readiness.rolling_high_20 is False


def test_price_against_vwap_is_a_signed_fraction() -> None:
    engine = FeatureEngine()
    engine.update(_candle(0, Decimal("100"), volume=100))

    snapshot = engine.update(_candle(1, Decimal("110"), volume=0))

    assert snapshot is not None
    # The second candle adds no turnover, so the VWAP is still the first price.
    assert snapshot.vwap == Decimal("100")
    assert snapshot.price_vs_vwap == Decimal("0.1")


def test_vwap_weights_each_candles_typical_price_by_its_own_volume() -> None:
    engine = FeatureEngine()
    engine.update(
        _candle(0, Decimal("100"), high=Decimal("112"), low=Decimal("94"), volume=10)
    )

    snapshot = engine.update(_candle(1, Decimal("110"), volume=30))

    assert snapshot is not None
    # Typical prices of (112 + 94 + 100) / 3 = 102 and a flat 110, weighted
    # 10 against 30: (1020 + 3300) / 40 = 108. Weighting the close instead
    # would read 107.5 and ignoring volume would read 106, so this one number
    # pins both halves of how the engine feeds its VWAP.
    assert snapshot.vwap == Decimal("108")


def test_vwap_and_the_volume_baseline_restart_each_session() -> None:
    engine = FeatureEngine()
    engine.warm_up(
        tuple(_candle(minute, Decimal("100"), volume=1_000) for minute in range(25))
    )

    snapshot = engine.update(
        _candle(0, Decimal("200"), volume=10, session_open=_NEXT_SESSION_OPEN)
    )

    assert snapshot is not None
    # Yesterday's turnover carries no weight into today's VWAP...
    assert snapshot.vwap == Decimal("200")
    # ...and today has no same-session volume baseline yet.
    assert snapshot.volume_ratio_20 is None
    assert snapshot.readiness.volume_ratio_20 is False
    # Price history does carry over: the session resets, the engine does not.
    assert snapshot.return_1 == Decimal("1")


def test_one_unknown_volume_disables_vwap_without_disturbing_prices() -> None:
    engine = FeatureEngine()
    engine.warm_up(_ramp(20))

    snapshot = engine.update(_candle(20, Decimal("106"), volume=None))

    assert snapshot is not None
    assert snapshot.vwap is None
    assert snapshot.price_vs_vwap is None
    assert snapshot.volume_ratio_20 is None
    assert snapshot.readiness.vwap is False
    # Everything that does not depend on volume is completely unaffected.
    assert snapshot.readiness.ema9 is True
    assert snapshot.readiness.rsi14 is True
    assert snapshot.readiness.rolling_high_20 is True
    # The candle's own high, two above its close, so this also shows the window
    # kept reading prices rather than falling back on the close it still had.
    assert snapshot.rolling_high_20 == Decimal("108")


def test_the_volume_baseline_excludes_the_candle_being_measured() -> None:
    engine = FeatureEngine()
    engine.warm_up(
        tuple(_candle(minute, Decimal("100"), volume=1_000) for minute in range(20))
    )

    snapshot = engine.update(_candle(20, Decimal("100"), volume=2_000))

    assert snapshot is not None
    # Folding the current 2,000 into its own baseline would drag the average up
    # and understate the surge; the previous twenty minutes averaged 1,000.
    assert snapshot.volume_ratio_20 == Decimal("2")


def test_the_volume_ratio_waits_for_a_full_baseline_window() -> None:
    engine = FeatureEngine()

    snapshots = [
        engine.update(_candle(minute, Decimal("100"), volume=1_000))
        for minute in range(20)
    ]

    assert all(snapshot is not None for snapshot in snapshots)
    assert all(snapshot.volume_ratio_20 is None for snapshot in snapshots)


def test_an_unknown_volume_suppresses_the_ratio_until_it_leaves_the_window() -> None:
    engine = FeatureEngine()
    engine.update(_candle(0, Decimal("100"), volume=None))
    engine.warm_up(
        tuple(_candle(minute, Decimal("100"), volume=1_000) for minute in range(1, 20))
    )

    suppressed = engine.update(_candle(20, Decimal("100"), volume=1_000))
    recovered = engine.update(_candle(21, Decimal("100"), volume=1_000))

    assert suppressed is not None
    # An average over a window containing an unknown quantity is a fiction.
    assert suppressed.volume_ratio_20 is None
    assert recovered is not None
    # The unknown candle has now been evicted from the twenty-candle window.
    assert recovered.volume_ratio_20 == Decimal("1")


def test_a_silent_window_gives_no_volume_ratio() -> None:
    engine = FeatureEngine()
    engine.warm_up(
        tuple(_candle(minute, Decimal("100"), volume=0) for minute in range(20))
    )

    snapshot = engine.update(_candle(20, Decimal("100"), volume=500))

    assert snapshot is not None
    # Comparing against a baseline of zero has no meaningful answer, and the
    # pinned numeric context would raise rather than invent one.
    assert snapshot.volume_ratio_20 is None


@pytest.mark.parametrize(("flag", "minimum"), _MINIMUM_CANDLES)
def test_a_feature_becomes_ready_at_its_documented_candle_count(
    flag: str,
    minimum: int,
) -> None:
    engine = FeatureEngine()
    candles = _ramp(minimum)

    engine.warm_up(candles[:-1])
    before = engine.snapshot(_RELIANCE)
    assert before is None or getattr(before.readiness, flag) is False

    after = engine.update(candles[-1])
    assert after is not None
    assert getattr(after.readiness, flag) is True


def test_every_derived_field_is_covered_by_a_readiness_flag() -> None:
    # The two dataclasses are built independently, so this is what proves the
    # engine's single mapping really does span every optional snapshot field.
    # Listing the identity and candle fields explicitly is deliberate: adding a
    # derived field without a flag must fail here, and the only way to make that
    # happen is to name what is *not* derived.
    identity_and_candle = {
        "instrument",
        "candle_start_time",
        "candle_end_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "readiness",
    }
    snapshot_fields = {field.name for field in fields(FeatureSnapshot)}

    assert identity_and_candle < snapshot_fields
    assert snapshot_fields - identity_and_candle == set(DERIVED_FEATURE_NAMES)


def test_every_derived_field_states_when_it_becomes_available() -> None:
    # Keeps the minimum-candle table honest: a feature added without a row here
    # would otherwise never have its first-available candle asserted at all.
    assert {name for name, _ in _MINIMUM_CANDLES} == set(DERIVED_FEATURE_NAMES)
    assert len(_MINIMUM_CANDLES) == len(DERIVED_FEATURE_NAMES)


def test_a_readiness_flag_is_true_exactly_when_its_field_has_a_value() -> None:
    engine = FeatureEngine()

    for candle in _varied(60):
        snapshot = engine.update(candle)
        assert snapshot is not None
        for name in DERIVED_FEATURE_NAMES:
            available = getattr(snapshot, name) is not None
            assert getattr(snapshot.readiness, name) is available


def test_core_readiness_arrives_with_the_slowest_price_feature() -> None:
    engine = FeatureEngine()
    engine.warm_up(_ramp(49))

    assert engine.is_ready(_RELIANCE) is False

    engine.update(_ramp(50)[-1])

    assert engine.is_ready(_RELIANCE) is True


def test_core_readiness_does_not_depend_on_volume() -> None:
    engine = FeatureEngine()
    engine.warm_up(
        tuple(
            _candle(minute, Decimal(100 + minute % 7), volume=None)
            for minute in range(50)
        )
    )

    snapshot = engine.snapshot(_RELIANCE)

    assert snapshot is not None
    # An instrument whose volume the broker never reported still has a valid
    # trend, momentum and volatility picture.
    assert snapshot.readiness.core_ready is True
    assert snapshot.readiness.vwap is False
    assert snapshot.readiness.volume_ratio_20 is False


def test_the_relative_strength_index_reads_the_closes_it_is_given() -> None:
    engine = FeatureEngine()

    closes = [Decimal(100)]
    for step in (3, -1) * 7:
        closes.append(closes[-1] + step)

    snapshot = None
    for minute, close in enumerate(closes):
        snapshot = engine.update(_candle(minute, close))

    assert snapshot is not None
    # Fourteen changes seed both Wilder averages: gains (7 * 3) / 14 = 1.5 and
    # losses (7 * 1) / 14 = 0.5, so RS = 3 and RSI = 100 - 100 / 4 = 75. An
    # inverted or high/low-fed RSI would still land inside the 0-100 band that
    # the bounds check below is able to see, which is why this is an exact value.
    assert snapshot.rsi14 == Decimal("75")


@pytest.mark.parametrize(
    ("field_name", "lower", "upper"),
    [
        ("rsi14", Decimal("0"), Decimal("100")),
        ("atr14", Decimal("0"), None),
        ("true_range", Decimal("0"), None),
        ("candle_range_pct", Decimal("0"), None),
        ("volume_ratio_20", Decimal("0"), None),
        ("vwap", Decimal("0"), None),
    ],
)
def test_bounded_features_never_leave_their_range(
    field_name: str,
    lower: Decimal,
    upper: Decimal | None,
) -> None:
    engine = FeatureEngine()

    for candle in _varied(60):
        snapshot = engine.update(candle)
        assert snapshot is not None
        value = getattr(snapshot, field_name)
        if value is None:
            continue
        assert value >= lower
        if upper is not None:
            assert value <= upper


def test_the_rolling_low_never_exceeds_the_rolling_high() -> None:
    engine = FeatureEngine()

    for candle in _varied(60):
        snapshot = engine.update(candle)
        assert snapshot is not None
        if snapshot.rolling_high_20 is None:
            continue
        assert snapshot.rolling_low_20 is not None
        assert snapshot.rolling_low_20 <= snapshot.rolling_high_20


def test_per_instrument_memory_stays_bounded_over_a_long_session() -> None:
    engine = FeatureEngine()

    # Four hundred candles is longer than a real NSE session, and every window
    # below must already have stopped growing well before the end of it.
    engine.warm_up(_varied(400))

    retained = _retained_containers(engine._states[_RELIANCE])

    # Walked rather than listed: a container added anywhere in the instrument
    # state, or inside any indicator it holds, surfaces here, which a hand-
    # written list of six attribute names could never do. The EMA and Wilder
    # seed buffers are absent because each is dropped once its average seeds.
    assert {name: len(window) for name, window in retained.items()} == {
        "closes": 15,
        "highs": 20,
        "lows": 20,
        "volumes": 20,
        "ema9_history": 6,
        "ema21_history": 6,
    }
    for name, window in retained.items():
        assert isinstance(window, deque), name
        assert window.maxlen == len(window), name


def test_features_do_not_depend_on_the_ambient_decimal_precision() -> None:
    candles = _varied(40)

    default = FeatureEngine()
    default.warm_up(candles)

    coarse = FeatureEngine()
    with localcontext() as context:
        context.prec = 6
        coarse.warm_up(candles)

    # A 21-period EMA smooths by 2 / 22 and a typical price divides by three;
    # neither terminates, so an unpinned engine would answer differently here.
    assert coarse.snapshot(_RELIANCE) == default.snapshot(_RELIANCE)


def test_a_snapshot_cannot_be_modified_after_it_is_returned() -> None:
    engine = FeatureEngine()
    snapshot = engine.update(_candle(0, Decimal("100")))

    assert snapshot is not None
    with pytest.raises(FrozenInstanceError):
        snapshot.close = Decimal("1")  # type: ignore[misc]


def test_an_unseen_instrument_has_nothing_to_report() -> None:
    engine = FeatureEngine()

    assert engine.snapshot(_TCS) is None
    assert engine.is_ready(_TCS) is False
    assert engine.snapshots() == ()
    assert engine.instruments() == ()


def test_instruments_and_snapshots_come_back_in_a_stable_order() -> None:
    engine = FeatureEngine()
    engine.update(_candle(0, Decimal("2000"), instrument=_TCS))
    engine.update(_candle(0, Decimal("100")))

    assert engine.instruments() == (_RELIANCE, _TCS)
    ordering = tuple(snapshot.instrument for snapshot in engine.snapshots())
    assert ordering == (_RELIANCE, _TCS)


def test_concurrent_updates_agree_with_a_serial_run() -> None:
    instruments = tuple(
        Instrument(exchange="NSE", trading_symbol=f"SYM{index}") for index in range(8)
    )
    # Each instrument trades at its own price level, so state leaking between
    # two of them would change the numbers rather than coincide with them.
    feeds = {
        item: tuple(
            _candle(minute, Decimal(100 + index * 10 + minute % 7), instrument=item)
            for minute in range(80)
        )
        for index, item in enumerate(instruments)
    }

    serial = FeatureEngine()
    for item in instruments:
        serial.warm_up(feeds[item])

    concurrent = FeatureEngine()
    threads = [
        Thread(target=concurrent.warm_up, args=(feeds[item],)) for item in instruments
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # ``MarketState`` invokes its ``on_candle`` hook outside its own lock, on a
    # broker feed thread, so concurrent updates are the arrangement this engine
    # actually runs under rather than a hypothetical one.
    assert concurrent.instruments() == serial.instruments()
    for item in instruments:
        assert concurrent.snapshot(item) == serial.snapshot(item)


def test_a_hostile_ambient_context_changes_no_feature() -> None:
    candles = _ramp(60)
    reference = FeatureEngine()
    reference.warm_up(candles)

    # Three significant digits, and trapping ``Inexact`` so that any arithmetic
    # escaping ``FEATURE_CONTEXT`` raises here instead of silently rounding.
    hostile = Context(prec=3)
    hostile.traps[Inexact] = True
    with localcontext(hostile):
        engine = FeatureEngine()
        # Both entry points, since each installs the context for itself.
        engine.warm_up(candles[:30])
        for candle in candles[30:]:
            engine.update(candle)
        observed = engine.snapshot(_RELIANCE)

    assert observed == reference.snapshot(_RELIANCE)


def test_a_failing_computation_discards_the_instrument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FeatureEngine()
    engine.warm_up(_ramp(30))
    assert engine.snapshot(_RELIANCE) is not None

    def _explode(*_args: object, **_kwargs: object) -> FeatureSnapshot:
        raise ArithmeticError("trapped")

    monkeypatch.setattr("ai_trader.features.engine._compute", _explode)
    with pytest.raises(ArithmeticError):
        engine.update(_candle(30, Decimal("101")))

    # Indicators advance in place, so a raise part-way through leaves them
    # holding a candle that was never fully folded in and cannot be unwound.
    # The instrument is dropped rather than left reporting values built on it,
    # and it disappears from both listings at once.
    assert engine.snapshot(_RELIANCE) is None
    assert engine.instruments() == ()
    assert engine.snapshots() == ()

    # The discarded candle is then free to be retried rather than swallowed as
    # a duplicate, and the instrument rebuilds from scratch.
    monkeypatch.undo()
    retried = engine.update(_candle(30, Decimal("101")))
    assert retried is not None
    assert retried.readiness.rsi14 is False


def test_each_indicator_is_fed_the_field_its_formula_names() -> None:
    """Every indicator reads the candle field its own definition specifies.

    The formulas themselves are verified against hand-computed literals in
    ``tests/test_indicators.py``; what is checked here is the wiring between a
    candle and a primitive, which those tests cannot see. The fixture's highs
    and lows straddle each close by a distance that changes candle to candle,
    because a fixed distance leaves close-to-close differencing unchanged — an
    RSI or a MACD fed the high would then agree with the correct answer no
    matter how wrong the wiring was.
    """
    candles = _varied(60)
    engine = FeatureEngine()
    engine.warm_up(candles)

    ema9 = ExponentialMovingAverage(9)
    ema21 = ExponentialMovingAverage(21)
    ema50 = ExponentialMovingAverage(50)
    rsi = RelativeStrengthIndex()
    macd = MovingAverageConvergenceDivergence()
    atr = AverageTrueRange()
    for candle in candles:
        ema9.update(candle.close)
        ema21.update(candle.close)
        ema50.update(candle.close)
        rsi.update(candle.close)
        macd.update(candle.close)
        atr.update(candle.high, candle.low, candle.close)

    snapshot = engine.snapshot(_RELIANCE)

    assert snapshot is not None
    assert snapshot.ema9 == ema9.value
    assert snapshot.ema21 == ema21.value
    assert snapshot.ema50 == ema50.value
    assert snapshot.rsi14 == rsi.value
    assert snapshot.macd == macd.macd
    assert snapshot.macd_signal == macd.signal
    assert snapshot.macd_histogram == macd.histogram
    assert snapshot.macd_histogram_change == macd.histogram_change
    assert snapshot.true_range == atr.true_range
    assert snapshot.atr14 == atr.value
    # Every return reads back the close that many candles earlier, so a period
    # is checked against the fixture's own history rather than against another
    # return. ``return_15`` reaches furthest and is the one most easily off.
    assert snapshot.return_1 == snapshot.close / candles[-2].close - 1
    assert snapshot.return_5 == snapshot.close / candles[-6].close - 1
    assert snapshot.return_15 == snapshot.close / candles[-16].close - 1
    # Normalized by the close, the price the feature is compared against, not by
    # the open, which is merely another number on the same candle.
    assert snapshot.atr_pct == atr.value / snapshot.close
    assert snapshot.candle_range_pct == (snapshot.high - snapshot.low) / snapshot.close


def test_threads_racing_on_one_instrument_neither_drop_nor_repeat_a_candle() -> None:
    """Four threads replaying one instrument still fold each candle in once.

    Exactly one thread can win each candle and the other three must be turned
    away. Without the engine's lock two of them clear the ordering guard
    together and fold one candle in twice, which no later state can unwind. The
    switch interval is shortened for the duration so that the interleaving this
    guards against is actually attempted rather than left to chance on a fast
    machine, and restored afterwards because it is process-wide.
    """
    candles = _ramp(200)
    engine = FeatureEngine()
    accepted = [0] * _RACING_THREADS
    start = Barrier(_RACING_THREADS)

    def replay(index: int) -> None:
        start.wait()
        accepted[index] = engine.warm_up(candles)

    threads = [Thread(target=replay, args=(index,)) for index in range(_RACING_THREADS)]
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(_TIGHT_SWITCH_INTERVAL)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(original_interval)

    rejected = engine.duplicate_candle_count + engine.out_of_order_candle_count
    assert sum(accepted) == len(candles)
    assert rejected == len(candles) * (_RACING_THREADS - 1)

    serial = FeatureEngine()
    serial.warm_up(candles)
    assert engine.snapshot(_RELIANCE) == serial.snapshot(_RELIANCE)


def test_an_ema_slope_rises_on_an_uptrend_and_falls_on_a_downtrend() -> None:
    """A slope reports where the EMA went, not where it came from.

    The two ends of the lag window are interchangeable to every other test:
    swapping them leaves availability, bounds and readiness untouched and only
    flips the sign, so a scanner would read every rally as a decline. Both the
    orientation and the exact value are pinned here, the value against an
    independently driven EMA so that a constant added anywhere in the chain is
    caught as well.
    """
    # Monotonic on purpose. The oscillating fixture the rest of this module
    # uses climbs and falls within every seven candles, so the sign of its slope
    # depends on where the window happens to land and says nothing about the
    # orientation of the formula.
    rising = tuple(
        _candle(minute, Decimal(100 + minute)) for minute in range(_SLOPE_TREND_CANDLES)
    )
    falling = tuple(
        _candle(minute, Decimal(100 + _SLOPE_TREND_CANDLES - minute))
        for minute in range(_SLOPE_TREND_CANDLES)
    )

    for candles, positive in ((rising, True), (falling, False)):
        engine = FeatureEngine()
        engine.warm_up(candles)
        snapshot = engine.snapshot(_RELIANCE)

        reference9 = ExponentialMovingAverage(9)
        reference21 = ExponentialMovingAverage(21)
        history9 = [reference9.update(candle.close) for candle in candles]
        history21 = [reference21.update(candle.close) for candle in candles]

        assert snapshot is not None
        assert snapshot.ema9_slope_5 == history9[-1] / history9[-_SLOPE_HISTORY] - 1
        assert snapshot.ema21_slope_5 == history21[-1] / history21[-_SLOPE_HISTORY] - 1
        assert (snapshot.ema9_slope_5 > 0) is positive
        assert (snapshot.ema21_slope_5 > 0) is positive


def test_threads_racing_through_update_are_serialized_candle_by_candle() -> None:
    """The same race driven through ``update``, which is the live entry point.

    The test above replays through ``warm_up``, which holds the lock for its
    whole loop and so fully serializes its threads; it would still pass if
    ``update`` were left unguarded. But ``update`` is the method
    ``MarketState`` calls from a broker feed thread, and it takes and releases
    the lock once per candle, so it is the path where threads genuinely
    interleave. Unguarded, two of them clear the ordering guard on one candle
    and fold it in twice, and a reader walking a volume window raises as
    another thread appends to it.
    """
    candles = _ramp(400)
    engine = FeatureEngine()
    accepted = [0] * _RACING_THREADS
    start = Barrier(_RACING_THREADS)

    def replay(index: int) -> None:
        start.wait()
        accepted[index] = sum(engine.update(candle) is not None for candle in candles)

    threads = [Thread(target=replay, args=(index,)) for index in range(_RACING_THREADS)]
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(_TIGHT_SWITCH_INTERVAL)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(original_interval)

    rejected = engine.duplicate_candle_count + engine.out_of_order_candle_count
    assert sum(accepted) == len(candles)
    assert rejected == len(candles) * (_RACING_THREADS - 1)

    # Interleaved acceptance is only correct if it lands on the same numbers a
    # single thread would have produced, so the snapshot is compared too.
    serial = FeatureEngine()
    serial.warm_up(candles)
    assert engine.snapshot(_RELIANCE) == serial.snapshot(_RELIANCE)
