"""Tests for the deterministic scanner.

Two properties matter more than the rest and are tested hardest. The first is
that a rule never reads a feature whose readiness flag is unset — the failure
mode there is not an exception but a scanner that runs all session and finds
nothing, so it is checked mechanically across every rule rather than once by
hand. The second is that the rolling window includes the candle being measured,
which makes a strict breakout test unsatisfiable; that one is driven through the
real ``FeatureEngine`` rather than a hand-built snapshot, because a fixture
could assert the convenient thing while the engine did the other.
"""

from datetime import datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from ai_trader.broker import Instrument
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine, FeatureSnapshot
from ai_trader.features.models import FeatureReadiness
from ai_trader.market import INDIA_TIMEZONE, Candle
from ai_trader.scanner import (
    DEFAULT_RULES,
    BreakoutRule,
    Direction,
    MeanReversionRule,
    PortfolioState,
    Position,
    Rule,
    Scanner,
    ScannerConfig,
    SuppressionReason,
    TrendContinuationRule,
    VwapReversionRule,
)

_SESSION_OPEN = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)
_AS_OF = _SESSION_OPEN + timedelta(minutes=31)
_RELIANCE = Instrument(exchange="NSE", trading_symbol="RELIANCE")
_INFY = Instrument(exchange="NSE", trading_symbol="INFY")
_CLOSE = Decimal("100")

_TREND_LONG = {
    "ema9": Decimal("103"),
    "ema21": Decimal("102"),
    "ema50": Decimal("101"),
    "adx14": Decimal("30"),
    "macd_histogram": Decimal("0.5"),
    "rsi14": Decimal("60"),
    "atr14": Decimal("2"),
}
_BREAKOUT_LONG = {
    "rolling_high_20": Decimal("100.05"),
    "rolling_low_20": Decimal("95"),
    "atr14": Decimal("2"),
}
_REVERSION_LONG = {
    "bollinger_percent_b_20": Decimal("-0.25"),
    "rsi14": Decimal("22"),
    "adx14": Decimal("15"),
}
_VWAP_LONG = {
    "vwap": Decimal("103"),
    "price_vs_vwap_sigma": Decimal("-3"),
}
_OPENING_RANGE_LONG = {
    "opening_range_high": Decimal("99.5"),
    "opening_range_low": Decimal("98"),
    "atr14": Decimal("2"),
}

_FIRING: dict[str, dict[str, Decimal]] = {
    "trend_continuation": _TREND_LONG,
    "range_breakout": _BREAKOUT_LONG,
    "band_mean_reversion": _REVERSION_LONG,
    "vwap_reversion": _VWAP_LONG,
    "opening_range_breakout": _OPENING_RANGE_LONG,
}
"""A feature set that makes each rule fire, keyed by rule name.

A test below asserts this covers every rule in ``DEFAULT_RULES``, so a new rule
cannot be added without stating what makes it fire.
"""


def _snapshot(
    instrument: Instrument = _RELIANCE,
    *,
    close: Decimal = _CLOSE,
    minute: int = 30,
    volume: int | None = 1_000,
    **features: Decimal,
) -> FeatureSnapshot:
    """A snapshot carrying exactly the named features and no others.

    Readiness is derived from the arguments rather than passed separately, which
    reproduces the engine's invariant — a flag is set exactly when its field has
    a value — instead of letting a fixture set a flag the value does not back.
    """
    start = _SESSION_OPEN + timedelta(minutes=minute)
    return FeatureSnapshot(
        instrument=instrument,
        candle_start_time=start,
        candle_end_time=start + timedelta(minutes=1),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        readiness=FeatureReadiness(**dict.fromkeys(features, True)),
        **features,
    )


def _divergent_snapshot(
    instrument: Instrument = _RELIANCE,
    *,
    close: Decimal = _CLOSE,
    **features: Decimal,
) -> FeatureSnapshot:
    """A snapshot carrying values the readiness flags do not vouch for.

    The engine never produces this, and ``_snapshot`` cannot express it. It is
    built here deliberately, because a state that cannot occur today is exactly
    what a defence is for: the point of reading through the flag is that a
    future engine change, a deserialized journal entry or a hand-built replay
    fixture that lets the two drift apart is refused rather than believed.
    """
    start = _SESSION_OPEN + timedelta(minutes=30)
    return FeatureSnapshot(
        instrument=instrument,
        candle_start_time=start,
        candle_end_time=start + timedelta(minutes=1),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000,
        readiness=FeatureReadiness(),
        **features,
    )


def _empty() -> PortfolioState:
    return PortfolioState.empty(_AS_OF)


def _rising_candles(count: int) -> tuple[Candle, ...]:
    """A clean uptrend in which every candle closes at its own high.

    Closing at the high is what makes the last candle set the rolling high, so
    the breakout hypothesis is genuinely on the table; the default fixtures
    elsewhere straddle the close, which leaves a gap the breakout tolerance was
    never meant to span.
    """
    candles = []
    for minute in range(count):
        close = Decimal(100) + Decimal(minute) / 2
        start = _SESSION_OPEN + timedelta(minutes=minute)
        candles.append(
            Candle(
                instrument=_RELIANCE,
                start_time=start,
                end_time=start + timedelta(minutes=1),
                open=close - Decimal("0.5"),
                high=close,
                low=close - Decimal(1),
                close=close,
                volume=1_000,
            )
        )
    return tuple(candles)


# --- the readiness trap -------------------------------------------------------


def test_every_rule_has_a_fixture_that_makes_it_fire() -> None:
    assert {rule.name for rule in DEFAULT_RULES} == set(_FIRING)


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda rule: rule.name)
def test_a_rule_fires_on_its_own_fixture(rule: Rule) -> None:
    with localcontext(FEATURE_CONTEXT):
        signal = rule.evaluate(_snapshot(**_FIRING[rule.name]), _empty(), None)
    assert signal is not None
    assert Decimal(0) <= signal.score <= Decimal(1)


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda rule: rule.name)
def test_withholding_any_required_feature_silences_a_rule(rule: Rule) -> None:
    """A rule with an incomplete reading declines rather than improvising.

    This is the ordinary case — the engine has not warmed up, or volume never
    arrived — and it says nothing about *how* the rule checks. The flag itself
    is tested below.
    """
    firing = _FIRING[rule.name]
    for withheld in rule.required_features:
        reduced = {name: firing[name] for name in firing if name != withheld}
        with localcontext(FEATURE_CONTEXT):
            signal = rule.evaluate(_snapshot(**reduced), _empty(), None)
        assert signal is None, f"{rule.name} fired without {withheld}"


@pytest.mark.parametrize("rule", DEFAULT_RULES, ids=lambda rule: rule.name)
def test_a_rule_trusts_the_readiness_flag_over_the_value(rule: Rule) -> None:
    """The section 4.3 trap, checked against every rule rather than one.

    A rule that tested ``is not None`` instead of consulting the flag would
    fire here, scoring against numbers the engine has not declared usable.
    Withholding the feature outright cannot catch that mistake, because it
    removes the value too and both spellings then decline for the same reason.
    """
    with localcontext(FEATURE_CONTEXT):
        signal = rule.evaluate(
            _divergent_snapshot(**_FIRING[rule.name]), _empty(), None
        )
    assert signal is None, f"{rule.name} read past its readiness flag"


def test_the_scanner_treats_unvouched_values_as_no_data_at_all() -> None:
    result = Scanner().scan([_divergent_snapshot(**_TREND_LONG)], _empty())
    assert result.candidates == ()
    assert result.not_ready == 1
    assert result.considered == 0


def test_a_core_ready_snapshot_still_withholds_vwap_from_the_scanner() -> None:
    """``core_ready`` covers price and momentum, never volume.

    This is the exact shape of the trap: the snapshot looks fully warmed up, one
    volume-derived input is missing, and the scanner must drop the rule that
    needs it while the price rules carry on.
    """
    snapshot = _snapshot(
        **_TREND_LONG,
        return_15=Decimal("0.01"),
        macd_histogram_change=Decimal("0.1"),
        rolling_high_20=Decimal("104"),
        bollinger_upper_20=Decimal("105"),
        price_vs_vwap_sigma=Decimal("-3"),
    )
    assert snapshot.readiness.core_ready is True
    assert snapshot.readiness.vwap is False

    result = Scanner().scan([snapshot], _empty())

    assert result.considered == 1
    fired = {name for candidate in result.candidates for name in candidate.rules}
    assert "vwap_reversion" not in fired
    assert "trend_continuation" in fired


def test_an_unknown_volume_is_not_scored_as_a_weak_one() -> None:
    """Breakout scores higher without volume than with flat volume.

    Averaging an unknown ratio against zero would mark down every breakout in
    the first twenty minutes of the session, which is when the feature is
    unavailable and when breakouts are most common.
    """
    scanner = Scanner()
    shared = dict(_BREAKOUT_LONG)

    unknown = scanner.scan([_snapshot(**shared)], _empty()).candidates[0]
    flat = scanner.scan(
        [_snapshot(**shared, volume_ratio_20=Decimal(1))], _empty()
    ).candidates[0]

    assert unknown.score > flat.score
    assert "volume_ratio_20" not in unknown.evidence
    assert flat.evidence["volume_ratio_20"] == Decimal(1)


# --- the inclusive rolling window ---------------------------------------------


def test_a_fresh_high_never_closes_above_its_own_rolling_window() -> None:
    """The property that makes a strict breakout test unsatisfiable.

    Asserted against the real engine, because this is a fact about the feature
    rather than about the scanner, and a scanner test that assumed it would keep
    passing if the engine's window stopped including the current candle.
    """
    engine = FeatureEngine()
    snapshot = None
    for candle in _rising_candles(25):
        snapshot = engine.update(candle)

    assert snapshot is not None
    assert snapshot.rolling_high_20 is not None
    assert snapshot.close == snapshot.rolling_high_20
    assert not snapshot.close > snapshot.rolling_high_20


def test_the_breakout_rule_fires_on_a_fresh_high_from_the_real_engine() -> None:
    engine = FeatureEngine()
    snapshot = None
    for candle in _rising_candles(25):
        snapshot = engine.update(candle)

    assert snapshot is not None
    result = Scanner().scan([snapshot], _empty())

    breakout = [c for c in result.candidates if "range_breakout" in c.rules]
    assert len(breakout) == 1
    assert breakout[0].direction is Direction.LONG
    assert breakout[0].reference_price == snapshot.close
    assert breakout[0].as_of == snapshot.candle_end_time


# --- suppression --------------------------------------------------------------


def test_each_suppression_reason_withholds_a_name_from_scanning() -> None:
    snapshot = _snapshot(**_TREND_LONG)
    cases = {
        SuppressionReason.ENTRIES_BLOCKED: PortfolioState(
            as_of=_AS_OF, new_entries_blocked=True
        ),
        SuppressionReason.STALE_FEED: PortfolioState(
            as_of=_AS_OF, stale_instruments=frozenset({_RELIANCE})
        ),
        SuppressionReason.POSITION_LIMIT: PortfolioState(
            as_of=_AS_OF, at_position_limit=frozenset({_RELIANCE})
        ),
        SuppressionReason.COOLDOWN: PortfolioState(
            as_of=_AS_OF,
            cooldown_until={_RELIANCE: _AS_OF + timedelta(minutes=5)},
        ),
    }
    for reason, portfolio in cases.items():
        result = Scanner().scan([snapshot], portfolio)
        assert result.candidates == ()
        assert result.suppressed == {reason: 1}
        # A suppressed name is never evaluated, so it is neither considered nor
        # counted as unready; conflating them would hide the kill switch.
        assert result.considered == 0
        assert result.not_ready == 0


def test_a_session_wide_block_is_reported_ahead_of_a_per_name_one() -> None:
    portfolio = PortfolioState(
        as_of=_AS_OF,
        new_entries_blocked=True,
        stale_instruments=frozenset({_RELIANCE}),
    )
    assert portfolio.suppression(_RELIANCE) is SuppressionReason.ENTRIES_BLOCKED


def test_an_expired_cooldown_lets_a_name_be_scanned_again() -> None:
    portfolio = PortfolioState(
        as_of=_AS_OF,
        cooldown_until={_RELIANCE: _AS_OF - timedelta(seconds=1)},
    )
    result = Scanner().scan([_snapshot(**_TREND_LONG)], portfolio)
    assert result.suppressed == {}
    assert len(result.candidates) == 1


def test_a_naive_clock_fails_loudly_rather_than_skipping_a_cooldown() -> None:
    portfolio = PortfolioState(
        as_of=datetime(2026, 9, 22, 9, 45),
        cooldown_until={_RELIANCE: _AS_OF},
    )
    with pytest.raises(TypeError):
        portfolio.suppression(_RELIANCE)


def test_a_configured_window_suppresses_outside_its_bounds() -> None:
    scanner = Scanner(
        ScannerConfig(
            earliest_minutes_since_open=Decimal(20),
            latest_minutes_since_open=Decimal(300),
        )
    )
    early = _snapshot(**_TREND_LONG, minutes_since_session_open=Decimal(5), minute=5)
    inside = _snapshot(**_TREND_LONG, minutes_since_session_open=Decimal(30))

    assert scanner.scan([early], _empty()).suppressed == {
        SuppressionReason.OUTSIDE_WINDOW: 1
    }
    assert len(scanner.scan([inside], _empty()).candidates) == 1


def test_a_configured_window_fails_closed_when_the_clock_is_unavailable() -> None:
    """No readable session clock plus a configured window means no scanning.

    The alternative is a "no entries near the close" setting that silently
    stops applying the moment the feature it depends on goes missing.
    """
    scanner = Scanner(ScannerConfig(latest_minutes_since_open=Decimal(300)))
    result = scanner.scan([_snapshot(**_TREND_LONG)], _empty())
    assert result.suppressed == {SuppressionReason.OUTSIDE_WINDOW: 1}


def test_no_window_is_configured_by_default() -> None:
    config = ScannerConfig()
    assert config.earliest_minutes_since_open is None
    assert config.latest_minutes_since_open is None


# --- ranking, budget and counters ---------------------------------------------


def test_the_budget_truncates_and_says_how_much_it_dropped() -> None:
    snapshots = [
        _snapshot(
            Instrument(exchange="NSE", trading_symbol=f"SYM{index:02d}"),
            **_TREND_LONG,
        )
        for index in range(7)
    ]
    result = Scanner(ScannerConfig(max_candidates=3)).scan(snapshots, _empty())

    assert len(result.candidates) == 3
    assert result.truncated == 4
    assert result.considered == 7


def test_ranking_does_not_depend_on_the_order_snapshots_arrive_in() -> None:
    """Identical scores must still produce one fixed order.

    A tie broken by insertion order would let the same market produce a
    different shortlist depending on how the subscription happened to be
    iterated, and a replay could not then reproduce the live run.
    """
    snapshots = [
        _snapshot(
            Instrument(exchange="NSE", trading_symbol=f"SYM{index:02d}"),
            **_TREND_LONG,
        )
        for index in range(6)
    ]
    scanner = Scanner(ScannerConfig(max_candidates=3))

    forward = scanner.scan(snapshots, _empty()).candidates
    backward = scanner.scan(list(reversed(snapshots)), _empty()).candidates

    assert forward == backward
    assert len({candidate.score for candidate in forward}) == 1


def test_a_higher_score_outranks_a_lower_one() -> None:
    strong = _snapshot(_RELIANCE, **_BREAKOUT_LONG)
    weak = _snapshot(_INFY, **_OPENING_RANGE_LONG)
    result = Scanner().scan([weak, strong], _empty())

    assert [candidate.instrument for candidate in result.candidates] == [
        _RELIANCE,
        _INFY,
    ]
    assert result.candidates[0].score > result.candidates[1].score


def test_a_snapshot_no_rule_can_read_counts_as_unready() -> None:
    result = Scanner().scan([_snapshot(), _snapshot(_INFY, **_TREND_LONG)], _empty())
    assert result.not_ready == 1
    assert result.considered == 1
    assert result.candidates[0].instrument == _INFY


def test_a_quiet_market_is_considered_rather_than_unready() -> None:
    """Every rule ran and none was convinced — a different fact from no data."""
    calm = _snapshot(**{**_TREND_LONG, "adx14": Decimal(5)})
    result = Scanner().scan([calm], _empty())

    assert result.candidates == ()
    assert result.considered == 1
    assert result.not_ready == 0


def test_two_snapshots_for_one_instrument_in_a_cycle_are_rejected() -> None:
    snapshot = _snapshot(**_TREND_LONG)
    with pytest.raises(ValueError, match="RELIANCE"):
        Scanner().scan([snapshot, snapshot], _empty())


def test_a_budget_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ScannerConfig(max_candidates=0)


# --- what the scanner deliberately does not decide ----------------------------


def test_opposing_rules_on_one_name_both_survive() -> None:
    """Conflicting signals are the AI layer's job, not the scanner's.

    A close at the top of its range that is also stretched above VWAP is a real
    disagreement. Dropping one side here would hide it rather than settle it.
    """
    snapshot = _snapshot(
        rolling_high_20=_CLOSE,
        rolling_low_20=Decimal(95),
        atr14=Decimal(2),
        vwap=Decimal(97),
        price_vs_vwap_sigma=Decimal(3),
    )
    result = Scanner().scan([snapshot], _empty())

    directions = {candidate.direction for candidate in result.candidates}
    assert directions == {Direction.LONG, Direction.SHORT}
    assert all(candidate.instrument == _RELIANCE for candidate in result.candidates)


def test_trend_and_mean_reversion_are_gated_on_opposite_regimes() -> None:
    trending = _snapshot(**_TREND_LONG, **{"bollinger_percent_b_20": Decimal("-0.3")})
    with localcontext(FEATURE_CONTEXT):
        assert MeanReversionRule().evaluate(trending, _empty(), None) is None

        ranging = _snapshot(**{**_TREND_LONG, "adx14": Decimal(10)})
        assert TrendContinuationRule().evaluate(ranging, _empty(), None) is None


def test_an_exhausted_trend_is_not_joined() -> None:
    exhausted = _snapshot(**{**_TREND_LONG, "rsi14": Decimal(85)})
    with localcontext(FEATURE_CONTEXT):
        assert TrendContinuationRule().evaluate(exhausted, _empty(), None) is None


def test_a_candidate_carries_no_size_stop_or_target() -> None:
    """The scanner's output is a hypothesis; sizing belongs to risk alone."""
    candidate = Scanner().scan([_snapshot(**_TREND_LONG)], _empty()).candidates[0]
    forbidden = {"quantity", "size", "stop", "stop_loss", "target", "take_profit"}
    assert forbidden.isdisjoint(dir(candidate))


def test_evidence_records_exactly_what_the_rules_read() -> None:
    snapshot = _snapshot(**_TREND_LONG)
    candidate = Scanner().scan([snapshot], _empty()).candidates[0]

    assert dict(candidate.evidence) == _TREND_LONG
    assert candidate.rules == ("trend_continuation",)


def test_agreeing_rules_are_both_named_and_do_not_inflate_the_score() -> None:
    """Agreement is recorded for replay to weigh, not rewarded with a bonus."""
    # Both rules read ``atr14``, and they must read the same one: a candidate
    # whose evidence held two values for one feature would be describing a
    # snapshot that never existed.
    snapshot = _snapshot(**{**_BREAKOUT_LONG, **_OPENING_RANGE_LONG})
    scanner = Scanner()

    combined = scanner.scan([snapshot], _empty()).candidates[0]
    alone = scanner.scan([_snapshot(**_BREAKOUT_LONG)], _empty()).candidates[0]

    assert combined.rules == ("opening_range_breakout", "range_breakout")
    assert combined.score == alone.score
    assert combined.score <= Decimal(1)


# --- the portfolio snapshot ---------------------------------------------------


def test_an_empty_book_suppresses_nothing() -> None:
    portfolio = PortfolioState.empty(_AS_OF)
    assert portfolio.suppression(_RELIANCE) is None
    assert portfolio.open_positions == {}
    assert portfolio.new_entries_blocked is False


def test_the_book_is_copied_and_sealed_on_construction() -> None:
    """A live handle to the caller's dict would let the book change mid-cycle.

    The scanner, the AI and risk all read one ``PortfolioState`` within a single
    decision cycle. If the position manager kept a mutable reference, the three
    could each see a different book and the resulting bug would be
    timing-dependent and effectively unreproducible.
    """
    position = Position(instrument=_RELIANCE, quantity=10, average_price=_CLOSE)
    book = {_RELIANCE: position}
    portfolio = PortfolioState(as_of=_AS_OF, open_positions=book)

    book[_INFY] = Position(instrument=_INFY, quantity=-5, average_price=_CLOSE)
    assert _INFY not in portfolio.open_positions

    with pytest.raises(TypeError):
        portfolio.open_positions[_INFY] = position  # type: ignore[index]


def test_a_scan_result_seals_its_own_tallies() -> None:
    result = Scanner().scan([_snapshot(**_TREND_LONG)], _empty())
    assert isinstance(result.candidates, tuple)
    with pytest.raises(TypeError):
        result.suppressed[SuppressionReason.COOLDOWN] = 1  # type: ignore[index]


def test_the_vwap_rule_is_the_one_that_needs_volume() -> None:
    """Every other rule keeps working on a session with no volume at all.

    This is the degradation policy in miniature: a missing input removes the
    rules that depend on it and nothing else.
    """
    volume_backed = {
        rule.name
        for rule in DEFAULT_RULES
        if {"vwap", "price_vs_vwap_sigma", "obv", "session_volume"}
        & set(rule.required_features)
    }
    assert volume_backed == {VwapReversionRule.name}
    assert "volume_ratio_20" not in BreakoutRule.required_features
