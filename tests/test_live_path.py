"""End-to-end checks spanning the market and feature layers.

Every other test in this suite stops at one layer's edge: the market tests end
at a candle, and the feature tests begin from candles built by hand. The live
path runs through both -- ticks arrive from a broker feed, become candles,
become features -- and until this file nothing crossed that seam.

The fixtures model Kite cumulative session volume and a missing-volume control.
An overlapping subscription can preserve continuity at handoff; a late join
must leave the discarded partial minute visibly unavailable to the features.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument, MarketTick, OHLCVCandle
from ai_trader.features import FeatureEngine
from ai_trader.market import Candle, MarketState

_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_SESSION_OPEN = datetime(2026, 9, 14, 3, 45, tzinfo=UTC)

_BACKFILL_MINUTES = range(30)
_LIVE_MINUTES = range(30, 80)
"""Enough live candles after the handoff to fill every 20-candle window."""

_VOLUME_FEATURES = (
    "vwap",
    "price_vs_vwap",
    "vwap_deviation",
    "price_vs_vwap_sigma",
    "volume_ratio_20",
    "obv",
    "session_volume",
)
"""Every derived feature that cannot be computed without a candle's volume."""


def test_live_history_audit_keeps_missing_and_different_candles_visible() -> None:
    from ai_trader.market import to_candle
    from scripts.validate_kite_live import compare_candles

    candles = tuple(to_candle(_RELIANCE, _ohlcv(minute)) for minute in range(3))
    reference = (candles[0], replace(candles[1], close=Decimal("100.9")))
    result = compare_candles(candles, reference)
    assert result["live_candles"] == 3
    assert result["matched_timestamps"] == 2
    assert result["exact_ohlcv_matches"] == 1
    assert result["rows"][1]["differences"]["close"]["historical"] == "100.9"
    assert result["rows"][2]["historical_present"] is False
    with pytest.raises(ValueError, match="duplicate"):
        compare_candles(candles, (candles[0], candles[0]))


@pytest.mark.parametrize("invalidate_stream", [False, True])
def test_live_research_path_requires_aligned_context_and_complete_inputs(
    invalidate_stream: bool,
) -> None:
    from ai_trader.clock import INDIA_TIMEZONE
    from ai_trader.features.tp_only import CompletedSession
    from ai_trader.market import to_candle
    from scripts.validate_kite_live import ResearchPath

    today = _SESSION_OPEN.astimezone(INDIA_TIMEZONE).date()
    instruments = tuple(
        Instrument("NSE", symbol)
        for symbol in ("RELIANCE", "NIFTY", "GOLDBEES", "INDIAVIX")
    )
    prior = {
        item: tuple(
            CompletedSession(
                today - timedelta(days=offset),
                Decimal(100),
                Decimal("0.0001"),
                (1000,) * 360,
            )
            for offset in range(6, 0, -1)
        )
        for item in instruments
    }
    matrices = []

    def predict(matrix):
        matrices.append(matrix)
        return {"raw": [0.5, 0.5], "calibrated": [0.5, 0.5]}

    path = ResearchPath(instruments, instruments[:1], prior, today, predict=predict)
    path.live_after = _SESSION_OPEN + timedelta(minutes=24)
    for minute in range(25):
        for instrument in instruments:
            base = Decimal(100) + Decimal(minute) / 10
            candle = replace(
                to_candle(instrument, _ohlcv(minute)),
                open=base,
                high=base + 2,
                low=base - 1,
                close=base + 1,
            )
            path.add(candle)
    assert path.completed_batches == 1
    assert len(path.features) == 4
    assert len(path.predictions) == len(matrices) == 1
    assert path.predictions[0]["missing_inputs"] == []
    assert len(matrices[0][0]) == 14
    assert matrices[0][0][-1] == 1 and matrices[0][1][-1] == -1
    assert path.pending == {}
    assert path.predictions[0]["within_training_decision_window"] is True
    if invalidate_stream:
        path.invalidate("synthetic_reconnect")
    for minute in range(25, 305):
        for instrument in instruments:
            base = Decimal(100) + Decimal(minute) / 10
            path.add(
                replace(
                    to_candle(instrument, _ohlcv(minute)),
                    open=base,
                    high=base + 2,
                    low=base - 1,
                    close=base + 1,
                )
            )
    assert len(matrices) == (1 if invalidate_stream else 56)
    assert path.predictions[-2]["within_training_decision_window"] is True
    if invalidate_stream:
        assert (
            path.predictions[-2]["declined_reason"]
            == "unreconciled_stream_discontinuity"
        )
        assert "probabilities" not in path.predictions[-2]
    else:
        assert "probabilities" in path.predictions[-2]
    assert path.predictions[-1]["within_training_decision_window"] is False
    assert path.predictions[-1]["declined_reason"] == "outside_training_decision_window"
    assert "probabilities" not in path.predictions[-1]


@pytest.mark.parametrize(
    ("collections", "injected", "predictions", "expected"),
    [
        (3, False, [], False),
        (3, False, [{"declined_reason": "unavailable_inputs"}], False),
        (2, False, [{"probabilities": {"raw": [0.5, 0.5]}}], False),
        (3, False, [{"probabilities": {"raw": [0.5, 0.5]}}], True),
        (5, True, [{"probabilities": {"raw": [0.5, 0.5]}}], False),
    ],
)
def test_reconnect_probe_waits_for_an_eligible_prediction(
    collections: int, injected: bool, predictions: list[dict], expected: bool
) -> None:
    from scripts.validate_kite_live import reconnect_probe_ready

    assert reconnect_probe_ready(collections, injected, predictions) is expected


@pytest.mark.parametrize(
    ("missing", "unavailable"),
    [
        (None, 0),
        ("minute", 5),
        ("recent_minute", 6),
        ("prior_day", 4),
        ("volume", 1),
        ("context", 3),
        ("flat", 1),
    ],
)
def test_independent_tp_reference_checks_values_missingness_and_future_exclusion(
    missing: str | None,
    unavailable: int,
) -> None:
    from ai_trader.clock import INDIA_TIMEZONE
    from ai_trader.features.tp_only import TpFeatureState
    from ai_trader.market import to_candle
    from scripts.validate_kite_live import audit_tp_inputs, previous_sessions

    instruments = tuple(
        Instrument("NSE", symbol)
        for symbol in ("RELIANCE", "NIFTY", "GOLDBEES", "INDIAVIX")
    )
    today = _SESSION_OPEN.astimezone(INDIA_TIMEZONE).date()
    calendar = [today - timedelta(days=offset) for offset in range(6, 0, -1)]
    histories, daily, states = {}, {}, {}
    as_of = _SESSION_OPEN + timedelta(minutes=25)
    for instrument in instruments:
        bars = []
        for days_before in range(6, -1, -1):
            for minute in range(360 if days_before else 40):
                original = to_candle(instrument, _ohlcv(minute))
                price = Decimal(100 - days_before) + Decimal(minute) / 100
                if missing == "flat":
                    price = Decimal(100)
                if not days_before and minute >= 25:
                    price = Decimal(100000)
                bars.append(
                    replace(
                        original,
                        start_time=original.start_time - timedelta(days=days_before),
                        end_time=original.end_time - timedelta(days=days_before),
                        open=price,
                        high=price + 1,
                        low=price - 1,
                        close=price + Decimal("0.1"),
                    )
                )
        closes = {day: Decimal(95 + offset) for offset, day in enumerate(calendar)}
        if missing == "prior_day":
            bars = [bar for bar in bars if bar.start_time.date() != calendar[-1]]
            del closes[calendar[-1]]
        if missing in ("minute", "recent_minute"):
            missing_minute = 10 if missing == "minute" else 23
            bars = [
                bar
                for bar in bars
                if bar.start_time != _SESSION_OPEN + timedelta(minutes=missing_minute)
            ]
        if missing == "volume":
            bars = [
                replace(bar, volume=None)
                if bar.start_time == _SESSION_OPEN + timedelta(minutes=10)
                else bar
                for bar in bars
            ]
        if missing == "context" and instrument.trading_symbol != "RELIANCE":
            bars = [bar for bar in bars if bar.end_time != as_of]
        histories[instrument] = bars
        daily[instrument] = closes
        state = TpFeatureState(
            instrument,
            today,
            previous_sessions(instrument, bars, closes, calendar, today),
        )
        for bar in bars:
            if bar.start_time.date() == today and bar.end_time <= as_of:
                state.update(bar)
        states[instrument] = state
    context = {
        instrument.trading_symbol: state.price_context()
        for instrument, state in states.items()
    }
    values = states[instruments[0]].values(context, benchmark="NIFTY")
    row = {
        "symbol": "RELIANCE",
        "end_time": as_of.isoformat(),
        "inputs": {
            name: None if value is None else str(value)
            for name, value in values.items()
        },
    }
    result = audit_tp_inputs(histories, daily, calendar, [row])
    assert result["status"] == "passed"
    assert result["numeric_comparisons"] + result["unavailable_agreements"] == 13
    assert result["unavailable_agreements"] == unavailable
    row["inputs"]["minutes_from_open"] = "26"
    with pytest.raises(ValueError, match="arithmetic differs: minutes_from_open"):
        audit_tp_inputs(histories, daily, calendar, [row])


def test_prediction_comparison_does_not_call_withheld_output_a_numeric_mismatch() -> (
    None
):
    from scripts.validate_kite_live import compare_predictions

    row = {
        "symbol": "RELIANCE",
        "end_time": "2026-10-07T12:45:00+05:30",
        "inputs": {"day_return": "0.01"},
    }
    predicted = {**row, "probabilities": {"raw": [0.5, 0.5]}}
    assert (
        compare_predictions([predicted], [predicted])[0]["probabilities_exact"] is True
    )
    withheld = compare_predictions([row], [predicted])[0]
    assert withheld["probabilities_exact"] is None
    assert withheld["live_prediction_present"] is False
    assert withheld["historical_prediction_present"] is True
    assert compare_predictions([row], [row])[0]["probabilities_exact"] is None
    changed = {**predicted, "probabilities": {"raw": [0.6, 0.4]}}
    assert (
        compare_predictions([predicted], [changed])[0]["probabilities_exact"] is False
    )


def _ohlcv(minute: int) -> OHLCVCandle:
    return OHLCVCandle(
        timestamp=_SESSION_OPEN + timedelta(minutes=minute),
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=5_000 + minute,
    )


def _bare_ticks(minutes: range) -> tuple[MarketTick, ...]:
    """Two ticks a minute, before optional native cumulative volume is attached.

    Two rather than one so each candle has a high and a low that differ, and so
    the minute's volume is a difference between two readings inside it rather
    than a single point.
    """
    return tuple(
        MarketTick(
            instrument=_RELIANCE,
            timestamp=_SESSION_OPEN + timedelta(minutes=minute, seconds=30 * half),
            price=Decimal(100 + minute % 7) + (Decimal("0.5") if half else Decimal(0)),
        )
        for minute in minutes
        for half in (0, 1)
    )


def _stamper() -> Callable[[MarketTick], MarketTick]:
    """Supply the running session total carried by Kite full-mode packets.

    The total only ever grows, the way an exchange's session counter does, and
    grows by a different amount each minute so that ``volume_ratio_20`` is
    measured against a moving baseline rather than a constant.
    """
    running = 1_000_000

    def stamp(tick: MarketTick) -> MarketTick:
        nonlocal running
        minute = (tick.timestamp - _SESSION_OPEN) // timedelta(minutes=1)
        running += 400 + (minute % 5) * 50
        return replace(tick, cumulative_volume=running)

    return stamp


def _run_live_session(
    tick_stamper: Callable[[MarketTick], MarketTick] | None,
    *,
    continuous: bool = False,
) -> tuple[FeatureEngine, MarketState]:
    """Wire a state object to an engine the way the production path does.

    ``backfill`` deliberately does not fire ``on_candle``, so history is handed
    to the engine explicitly and only live candles arrive through the hook.
    That is the sequence ``check_features`` follows, and getting it wrong in
    either direction -- replaying history twice, or never seeding it -- is the
    mistake this helper exists to make impossible in these tests.
    """
    engine = FeatureEngine()

    def fold(candle: Candle) -> None:
        engine.update(candle)

    state = MarketState(on_candle=fold)
    state.backfill(_RELIANCE, tuple(_ohlcv(minute) for minute in _BACKFILL_MINUTES))
    seeded = state.snapshot(_RELIANCE)
    assert seeded is not None
    engine.warm_up(seeded.candles)

    minutes = range(29, 80) if continuous else _LIVE_MINUTES
    for tick in _bare_ticks(minutes):
        state.record_tick(tick if tick_stamper is None else tick_stamper(tick))
    state.flush()
    return engine, state


@pytest.mark.parametrize("continuous", [False, True])
def test_streaming_ticks_and_replaying_candles_land_on_identical_features(
    continuous,
) -> None:
    """The live path must compute what a replay of the same candles computes.

    ``warm_up`` is a loop over ``update``, so the engine cannot diverge from
    itself -- but that only covers the engine's own boundary. It says nothing
    about whether the candles ``MarketState`` builds from a tick stream are the
    candles a replay would hand over. Driving both and comparing the result is
    what makes a historical replay admissible as evidence about live behaviour,
    which the entire validation plan rests on.
    """
    live, state = _run_live_session(_stamper(), continuous=continuous)
    market = state.snapshot(_RELIANCE)
    assert market is not None

    replayed = FeatureEngine()
    replayed.warm_up(market.candles)

    streamed = live.snapshot(_RELIANCE)
    assert streamed is not None
    assert streamed == replayed.snapshot(_RELIANCE)
    assert streamed.candle_start_time == _SESSION_OPEN + timedelta(minutes=79)
    assert streamed.readiness.core_ready is continuous
    if not continuous:
        assert streamed.session_volume is None
        assert streamed.vwap is None
    assert live.duplicate_candle_count == 0
    assert live.out_of_order_candle_count == 0


def test_the_handoff_from_history_to_live_ticks_costs_exactly_one_candle() -> None:
    """A stream joined mid-minute yields a fragment, so that minute is dropped.

    The cost is worth naming precisely: one candle, at the join, once per
    instrument per run. Everything either side of it is continuous, and the
    gap is visible rather than papered over with a partial minute that would
    understate volume and misplace the open.
    """
    _, state = _run_live_session(_stamper())
    market = state.snapshot(_RELIANCE)
    assert market is not None

    starts = [candle.start_time for candle in market.candles]
    assert starts == [
        _SESSION_OPEN + timedelta(minutes=minute)
        for minute in (*_BACKFILL_MINUTES, *_LIVE_MINUTES[1:])
    ]
    assert state.duplicate_candle_count == 0
    assert state.late_tick_count == 0


def test_a_stamped_stream_carries_volume_all_the_way_into_the_features() -> None:
    """Native totals survive differencing, aggregation and derivation."""
    engine, _ = _run_live_session(_stamper(), continuous=True)
    snapshot = engine.snapshot(_RELIANCE)
    assert snapshot is not None

    # Minute 79 takes two increments of 400 + (79 % 5) * 50 = 600.
    assert snapshot.volume == 1_200
    # Against a 20-candle baseline averaging 1_000, that is exactly 1.2 -- an
    # exact Decimal rather than a tolerance, so a wrong answer fails loudly.
    assert snapshot.volume_ratio_20 == Decimal("1.2")
    assert snapshot.session_volume == 200_435
    for name in _VOLUME_FEATURES:
        assert getattr(snapshot, name) is not None, name
        assert getattr(snapshot.readiness, name), name


def test_an_unstamped_stream_withholds_volume_features_and_nothing_else() -> None:
    """Losing volume must cost volume features only, and must cost them fully.

    This is the system-level statement of the rule that a feature which cannot
    be computed honestly reports nothing. A broker outage, or simply forgetting
    to receive native volume, leaves every volume feature unavailable -- not zero,
    which claims nothing traded, and not a stale figure carried forward, which
    claims a VWAP that no longer describes the session. Price and momentum are
    untouched, so an instrument stays tradeable on the features that survive.
    """
    engine, _ = _run_live_session(None, continuous=True)
    snapshot = engine.snapshot(_RELIANCE)
    assert snapshot is not None

    assert snapshot.volume is None
    for name in _VOLUME_FEATURES:
        assert getattr(snapshot, name) is None, name
        assert not getattr(snapshot.readiness, name), name

    # The surviving features are not merely present, they are the ones a
    # scanner gates on, and they are identical to the stamped run's.
    assert snapshot.readiness.core_ready
    stamped, _ = _run_live_session(_stamper(), continuous=True)
    reference = stamped.snapshot(_RELIANCE)
    assert reference is not None
    for name in ("ema9", "ema21", "ema50", "rsi14", "atr14", "adx14", "return_15"):
        assert getattr(snapshot, name) == getattr(reference, name), name
