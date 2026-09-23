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
from ai_trader.costs import CostModel
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine, FeatureSnapshot
from ai_trader.features.models import FeatureReadiness
from ai_trader.market import INDIA_TIMEZONE, Candle
from ai_trader.scanner import (
    DEFAULT_RULES,
    BreakoutRule,
    Direction,
    FeasibilityPolicy,
    FeasibilityReason,
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

_REACHABLE_ATR = Decimal("0.002")
"""An ``atr_pct`` far above the hurdle the default policy below computes."""

_UNREACHABLE_ATR = Decimal("0.0001")
"""An ``atr_pct`` far below it.

Both are an order of magnitude clear of the boundary on purpose. The rates in
``ai_trader.costs`` are unverified estimates and will move when a real contract
note is parsed; these tests are about the screen's behaviour, not about where
the line currently falls, and should not start failing when it shifts. The one
test that does pin the boundary exactly uses ``_FREE_MODEL`` so the arithmetic
is legible without the rate schedule in it.
"""

_FREE_MODEL = CostModel(
    brokerage_fraction=Decimal(0),
    brokerage_cap=Decimal(0),
    securities_transaction_tax_fraction=Decimal(0),
    exchange_transaction_fraction=Decimal(0),
    regulator_fee_fraction=Decimal(0),
    stamp_duty_fraction=Decimal(0),
    goods_and_services_tax_fraction=Decimal(0),
)
"""A schedule charging nothing, so the required move is exactly the margin."""


def _policy(**overrides: object) -> FeasibilityPolicy:
    """A cost screen at a one-lakh clip asking for a tenth of a percent net."""
    settings: dict[str, object] = {
        "target_notional": Decimal(100_000),
        "net_margin_fraction": Decimal("0.001"),
        "max_atr_multiple": Decimal(3),
    }
    settings.update(overrides)
    return FeasibilityPolicy(**settings)  # type: ignore[arg-type]


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


# --- the cost screen ----------------------------------------------------------


def test_no_cost_screen_is_configured_by_default() -> None:
    """Off unless asked for, and for the same reason the window is unbounded.

    The screen needs a position size and a target margin. Defaulting either
    would put a hard-coded target back into the system one layer down, where it
    would be harder to see than the one section 7 forbids.
    """
    assert ScannerConfig().feasibility is None

    # No ``atr_pct`` anywhere in the firing fixtures, and it does not matter.
    result = Scanner().scan([_snapshot(**_TREND_LONG)], _empty())
    assert len(result.candidates) == 1
    assert result.unreachable == 0
    assert result.candidates[0].feasibility is None


def test_a_name_too_quiet_to_pay_for_its_round_trip_is_dropped() -> None:
    scanner = Scanner(ScannerConfig(feasibility=_policy()))
    result = scanner.scan(
        [_snapshot(**_TREND_LONG, atr_pct=_UNREACHABLE_ATR)], _empty()
    )

    assert result.candidates == ()
    assert result.unreachable == 1
    assert result.not_ready == 0
    assert result.considered == 0


def test_a_reachable_name_carries_the_screen_working_on_its_candidate() -> None:
    """Recorded at the moment of the decision, not recomputed later.

    A journal that recomputed the hurdle from a snapshot that had since moved
    on would report a number the screen never actually used.
    """
    policy = _policy()
    scanner = Scanner(ScannerConfig(feasibility=policy))
    snapshot = _snapshot(
        **_TREND_LONG,
        atr_pct=_REACHABLE_ATR,
        minutes_since_session_open=Decimal(30),
    )

    check = scanner.scan([snapshot], _empty()).candidates[0].feasibility
    assert check is not None
    assert check.reachable
    assert check.reason is None
    assert check.atr_fraction == _REACHABLE_ATR
    assert check.required_gross_fraction == policy.required_gross_fraction_at(_CLOSE)
    assert check.minutes_remaining == Decimal(345)


def test_the_hurdle_is_priced_on_what_the_clip_fills_not_on_the_clip() -> None:
    """A fixed clip buys whole shares, so a dear name overshoots it.

    At a one-lakh clip a stock quoted at Rs 1,50,000 buys one share and turns
    over half again what the clip asked for, and a round trip on a larger
    notional costs a smaller fraction of it. Screening that name against the
    full-clip figure would overstate its hurdle — failing closed on exactly the
    names the fixed buying model pushes furthest past the clip, and refusing
    ones the decision could in fact afford.
    """
    policy = _policy()
    at_the_clip = policy.required_gross_fraction_at(_CLOSE)
    overshooting = policy.required_gross_fraction_at(Decimal(150_000))

    assert overshooting < at_the_clip
    # The whole of the difference is in the cost term, since the margin is
    # common to both; measured there it is a fifth rather than a rounding.
    cost_at_the_clip = at_the_clip - policy.net_margin_fraction
    cost_overshooting = overshooting - policy.net_margin_fraction
    assert (cost_at_the_clip - cost_overshooting) / cost_at_the_clip > Decimal("0.15")


def test_the_full_clip_hurdle_is_a_ceiling_no_quote_exceeds() -> None:
    """The policy's own number is the worst case, not the typical one.

    It is reached only where the price divides the clip exactly. Every other
    quote rounds up to the next whole share, turns over more than the clip and
    pays a smaller fraction for it, so the figure that describes the policy can
    be reported without ever understating what a particular name must do.
    """
    policy = _policy()
    ceiling = policy.required_gross_fraction

    for price in (
        _CLOSE,
        Decimal("2450"),
        Decimal("512.30"),
        Decimal(99_999),
        Decimal(140_000),
    ):
        assert policy.required_gross_fraction_at(price) <= ceiling, price


def test_no_quote_is_too_dear_to_size() -> None:
    """One share clears a floor, so there is no price the screen refuses.

    The clip is a minimum on turnover rather than a maximum, which removes the
    whole category of "too expensive to buy". The screen carries no reason for
    it, and a reason that existed would describe a buying model this system
    does not have.
    """
    policy = _policy()

    assert policy.required_gross_fraction_at(Decimal("100000.01")) > 0
    assert policy.required_gross_fraction_at(Decimal(5_000_000)) > 0
    assert [reason.value for reason in FeasibilityReason] == [
        "volatility_unknown",
        "volatility_too_low",
        "clock_unknown",
        "session_too_short",
    ]


def test_a_name_dearer_than_the_clip_is_an_ordinary_candidate() -> None:
    """Price above the clip is not a rejection reason under a floor.

    The configured size decides how much a trade deploys, not whether one is
    possible. A name that would fill at forty per cent above the clip is
    screened on whether it can move far enough, exactly like every other name,
    and is neither refused nor blamed on a cold feature engine.
    """
    scanner = Scanner(ScannerConfig(feasibility=_policy()))
    snapshot = _snapshot(close=Decimal(140_000), **_TREND_LONG, atr_pct=_REACHABLE_ATR)

    result = scanner.scan([snapshot], _empty())

    assert result.unreachable == 0
    assert result.not_ready == 0
    assert len(result.candidates) == 1


def test_a_dear_name_is_judged_on_its_volatility_like_any_other() -> None:
    """Nothing short-circuits ahead of the volatility test any more."""
    snapshot = _snapshot(close=Decimal(140_000), atr_pct=_UNREACHABLE_ATR)
    with localcontext(FEATURE_CONTEXT):
        check = _policy().evaluate(snapshot)

    assert check.reason is FeasibilityReason.VOLATILITY_TOO_LOW
    assert check.atr_fraction == _UNREACHABLE_ATR


def test_a_dear_name_reports_the_hurdle_its_own_fill_faces() -> None:
    """Not the policy's figure — its fill is larger, so its hurdle is lower.

    Recording the policy's number here would overstate by a tenth what this
    name was actually asked to do, which is precisely the kind of quiet
    disagreement between the screen and the trade that the shared sizer exists
    to rule out.
    """
    policy = _policy()
    snapshot = _snapshot(close=Decimal(140_000), atr_pct=_REACHABLE_ATR)
    with localcontext(FEATURE_CONTEXT):
        check = policy.evaluate(snapshot)

    assert check.required_gross_fraction == policy.required_gross_fraction_at(
        Decimal(140_000)
    )
    assert check.required_gross_fraction < policy.required_gross_fraction


def test_the_screen_sizes_through_the_policy_the_trade_will_use() -> None:
    """A screen that sized differently from the trade would measure nothing.

    The hurdle the screen applies and the hurdle the decision faces are the
    same arithmetic on the same quantity, because both come from one
    ``SizingPolicy`` rather than from two implementations free to disagree.
    """
    policy = _policy()
    estimate = policy.sizing.estimate(Decimal("2450"))

    assert estimate.quantity == 41
    assert estimate.notional == Decimal("100450")
    assert (
        policy.required_gross_fraction_at(Decimal("2450"))
        == estimate.required_gross_fraction
    )


def test_the_sizer_is_derived_from_the_screen_not_configured_beside_it() -> None:
    """Composing it makes disagreement impossible rather than merely unlikely."""
    policy = _policy(costs=_FREE_MODEL)
    sizing = policy.sizing

    assert sizing.target_notional == policy.target_notional
    assert sizing.net_margin_fraction == policy.net_margin_fraction
    assert sizing.costs is policy.costs
    # A schedule charging nothing leaves the margin standing alone, which is
    # only visible if the model really did travel through.
    assert policy.required_gross_fraction_at(_CLOSE) == policy.net_margin_fraction


def test_a_missing_atr_fails_closed_and_counts_as_unready_not_unreachable() -> None:
    """Could-not-measure and measured-and-no are different facts.

    A session reporting four hundred unreachable names is a market too quiet to
    trade; a session reporting four hundred not-ready ones is an engine that has
    not warmed up. Merging them would make the second read as the first, and the
    operator would go looking at the market instead of at the feed.
    """
    scanner = Scanner(ScannerConfig(feasibility=_policy()))
    result = scanner.scan([_snapshot(**_TREND_LONG)], _empty())

    assert result.candidates == ()
    assert result.not_ready == 1
    assert result.unreachable == 0


def test_the_screen_reads_atr_through_the_readiness_flag() -> None:
    """A value the engine never vouched for is not a measurement."""
    divergent = _divergent_snapshot(**_TREND_LONG, atr_pct=_REACHABLE_ATR)
    with localcontext(FEATURE_CONTEXT):
        check = _policy().evaluate(divergent)

    assert check.reason is FeasibilityReason.VOLATILITY_UNKNOWN
    assert check.atr_fraction is None


def test_a_motionless_name_is_rejected_rather_than_divided_by() -> None:
    """A zero ATR is a real reading, not an error.

    Expressing the test as "how many ATRs away is the hurdle" would divide by
    it, and would raise on exactly the names the screen exists to reject.
    """
    with localcontext(FEATURE_CONTEXT):
        check = _policy().evaluate(_snapshot(**_TREND_LONG, atr_pct=Decimal(0)))

    assert check.reason is FeasibilityReason.VOLATILITY_TOO_LOW
    assert check.atr_fraction == 0


def test_the_multiple_is_inclusive_at_its_own_boundary() -> None:
    """Exactly enough headroom passes; a hair less does not.

    Pinned against a schedule that charges nothing, so the required move is the
    margin itself and the boundary can be read off without the rate table.
    """
    policy = _policy(
        costs=_FREE_MODEL,
        net_margin_fraction=Decimal("0.001"),
        max_atr_multiple=Decimal(2),
    )
    with localcontext(FEATURE_CONTEXT):
        assert policy.required_gross_fraction == Decimal("0.001")
        exact = policy.evaluate(_snapshot(**_TREND_LONG, atr_pct=Decimal("0.0005")))
        short = policy.evaluate(_snapshot(**_TREND_LONG, atr_pct=Decimal("0.00049")))

    assert exact.reachable
    assert short.reason is FeasibilityReason.VOLATILITY_TOO_LOW


def test_the_time_gate_is_off_unless_a_minimum_is_configured() -> None:
    """Same reasoning as the session window: no invented cut-off by default."""
    policy = _policy()
    assert policy.min_minutes_remaining is None

    with localcontext(FEATURE_CONTEXT):
        # No clock at all, and the screen still passes the name.
        check = policy.evaluate(_snapshot(**_TREND_LONG, atr_pct=_REACHABLE_ATR))

    assert check.reachable
    assert check.minutes_remaining is None


def test_too_little_session_left_is_rejected_when_the_gate_is_on() -> None:
    scanner = Scanner(
        ScannerConfig(feasibility=_policy(min_minutes_remaining=Decimal(60)))
    )
    late = _snapshot(
        **_TREND_LONG,
        atr_pct=_REACHABLE_ATR,
        minutes_since_session_open=Decimal(350),
        minute=350,
    )
    result = scanner.scan([late], _empty())

    assert result.candidates == ()
    assert result.unreachable == 1
    assert result.not_ready == 0


def test_a_configured_time_gate_fails_closed_when_the_clock_is_unavailable() -> None:
    """An unreadable clock counts as unready, not as unreachable.

    The market did not answer the question; the feed failed to ask it.
    """
    scanner = Scanner(
        ScannerConfig(feasibility=_policy(min_minutes_remaining=Decimal(60)))
    )
    result = scanner.scan([_snapshot(**_TREND_LONG, atr_pct=_REACHABLE_ATR)], _empty())

    assert result.candidates == ()
    assert result.not_ready == 1
    assert result.unreachable == 0


def test_the_clock_is_recorded_even_when_it_is_not_gating() -> None:
    """Evidence is collected whether or not it is being acted on."""
    with localcontext(FEATURE_CONTEXT):
        check = _policy().evaluate(
            _snapshot(
                **_TREND_LONG,
                atr_pct=_REACHABLE_ATR,
                minutes_since_session_open=Decimal(100),
            )
        )

    assert check.reachable
    assert check.minutes_remaining == Decimal(275)


def test_the_screen_runs_before_any_rule_is_evaluated() -> None:
    """A name that could never have been traded takes none of the budget.

    ``considered`` counts snapshots on which at least one rule ran, so a zero
    here is the observable form of "the rules never saw it" — and the snapshot
    used would otherwise fire ``trend_continuation`` outright.
    """
    scanner = Scanner(ScannerConfig(feasibility=_policy()))
    result = scanner.scan(
        [_snapshot(**_TREND_LONG, atr_pct=_UNREACHABLE_ATR)], _empty()
    )

    assert result.considered == 0
    assert Scanner().scan([_snapshot(**_TREND_LONG)], _empty()).considered == 1


def test_suppression_is_reported_ahead_of_the_cost_screen() -> None:
    """A kill-switched session must not read as an untradeable market."""
    scanner = Scanner(ScannerConfig(feasibility=_policy()))
    portfolio = PortfolioState(as_of=_AS_OF, new_entries_blocked=True)
    result = scanner.scan(
        [_snapshot(**_TREND_LONG, atr_pct=_UNREACHABLE_ATR)], portfolio
    )

    assert result.suppressed == {SuppressionReason.ENTRIES_BLOCKED: 1}
    assert result.unreachable == 0


def test_the_screen_filters_but_does_not_reorder() -> None:
    """Headroom is not a term in the score.

    Whether spare volatility *should* influence rank is a real question, but it
    is one for replay to measure. Folding it in here would mean deciding how
    many points a basis point of ATR is worth against a point of trend strength,
    and there is nothing behind such a number.
    """
    strong = _snapshot(_RELIANCE, **_BREAKOUT_LONG, atr_pct=_REACHABLE_ATR)
    weak = _snapshot(_INFY, **_OPENING_RANGE_LONG, atr_pct=_REACHABLE_ATR * 100)
    scanner = Scanner(ScannerConfig(feasibility=_policy()))

    screened = scanner.scan([weak, strong], _empty()).candidates
    unscreened = Scanner().scan([weak, strong], _empty()).candidates

    assert [candidate.instrument for candidate in screened] == [_RELIANCE, _INFY]
    assert [candidate.score for candidate in screened] == [
        candidate.score for candidate in unscreened
    ]


def test_a_short_clears_the_same_hurdle_as_a_long() -> None:
    """One buy and one sell either way, so the round trip costs the same.

    Shorts are in scope, and the screen would be wrong in one direction if the
    cost model had a side to it.
    """
    short = _snapshot(
        vwap=Decimal(97),
        price_vs_vwap_sigma=Decimal(3),
        atr_pct=_REACHABLE_ATR,
    )
    long = _snapshot(_INFY, **_VWAP_LONG, atr_pct=_REACHABLE_ATR)
    scanner = Scanner(ScannerConfig(feasibility=_policy()))

    result = scanner.scan([short, long], _empty())
    hurdles = {
        (candidate.direction, candidate.feasibility.required_gross_fraction)
        for candidate in result.candidates
        if candidate.feasibility is not None
    }

    assert {direction for direction, _ in hurdles} == {Direction.LONG, Direction.SHORT}
    assert len({hurdle for _, hurdle in hurdles}) == 1


def test_a_feasibility_check_names_no_price_and_no_size() -> None:
    """A required gross move is not a target and not an exit level.

    It is a fraction and it is attached to no price. The screen does size
    internally — it has to, because a fixed clip fills a whole number of shares
    and the cost fraction depends on what actually filled — but the share count
    is working rather than a proposal, and carrying it here would read as a
    recommendation to the layer above. The risk engine remains the only thing
    that decides both the size and where a position is closed.
    """
    scanner = Scanner(ScannerConfig(feasibility=_policy()))
    snapshot = _snapshot(**_TREND_LONG, atr_pct=_REACHABLE_ATR)
    check = scanner.scan([snapshot], _empty()).candidates[0].feasibility

    assert check is not None
    forbidden = {
        "quantity",
        "size",
        "notional",
        "target_notional",
        "price",
        "stop",
        "stop_loss",
        "target",
        "take_profit",
    }
    assert forbidden.isdisjoint(dir(check))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_notional": Decimal(0)}, "target_notional must be positive"),
        ({"net_margin_fraction": Decimal(0)}, "net_margin_fraction must be positive"),
        ({"max_atr_multiple": Decimal(0)}, "max_atr_multiple must be positive"),
        (
            {"min_minutes_remaining": Decimal(-1)},
            "min_minutes_remaining cannot be negative",
        ),
        (
            {"square_off_minutes_since_open": Decimal(0)},
            "square_off_minutes_since_open must be positive",
        ),
    ],
)
def test_an_incoherent_policy_is_rejected_at_construction(
    overrides: dict[str, Decimal], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _policy(**overrides)


def test_the_atr_multiple_has_no_default() -> None:
    """The assumption with the least evidence behind it must be stated.

    Section 7 says an invented threshold is not evidence. A default here would
    be one, silently applied by every caller that did not think about it.
    """
    with pytest.raises(TypeError):
        FeasibilityPolicy(  # type: ignore[call-arg]
            target_notional=Decimal(100_000), net_margin_fraction=Decimal("0.001")
        )


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
