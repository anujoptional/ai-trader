"""One config, one scanner, no way for two callers to disagree.

``StrategyConfig`` exists to make a specific bug unwriteable. Before it, the clip
size could be set in three places, the cost schedule in four and the rule list in
two, and nothing checked that they matched. The concrete failure is arithmetic,
not hypothetical: a feasibility screen configured at a one-lakh clip prices the
round-trip hurdle at 0.083% of notional, while a sizer configured at forty
thousand really pays 0.153%. A scanner wired that way shows the AI names it has
screened as affordable and then takes them at a size that cannot pay for them.
The report would blame the strategy; the cause would be two config sites.

**What this file checks, in order of how much it matters.**

The first group is the guarantee itself, and it is checked *behaviourally*
rather than structurally. Comparing two ``ScannerConfig`` objects with ``==``
would show that two callers built equal descriptions of a scanner. Running both
scanners over the same tape and comparing every scan shows that they answer the
same question the same way, which is the claim Section 7.1 actually makes:
replay sees exactly the candidate stream the AI would have seen. Structural
equality is checked too, because when the behavioural test fails it is the
faster thing to read.

Every equality test here is paired with a test that the equality is not vacuous.
Two scanners that both find nothing agree perfectly, and a ``scanner_config``
that ignored its inputs would make every comparison in this file pass while
proving nothing at all. So the tape is required to produce candidates, and a
deliberately different config is required to produce a different answer.

The second group is the derivation: the screen's hurdle and the trade's target
must be two readings of one number rather than two numbers that happen to agree.
The third is that the refusals fire when the config is *written*, not several
steps into a run that has already fetched a year of candles.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from ai_trader.broker import Instrument
from ai_trader.costs import GROWW_INTRADAY_EQUITY, ZERODHA_INTRADAY_EQUITY
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine
from ai_trader.market import INDIA_TIMEZONE, Candle
from ai_trader.replay import FillModel, ReplayConfig
from ai_trader.scanner import (
    DEFAULT_RULES,
    MarketContext,
    PortfolioState,
    Scanner,
    ScanResult,
    VwapReversionRule,
)
from ai_trader.strategy import ChandelierStop, FixedAtrStop, StrategyConfig

_ONE_MINUTE = timedelta(minutes=1)
_ORIGIN = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)
_MINUTES = 240

_NAMES = (
    (Instrument(exchange="NSE", trading_symbol="ALPHA"), 0, Decimal("2450")),
    (Instrument(exchange="NSE", trading_symbol="BRAVO"), 17, Decimal("1310")),
    (Instrument(exchange="NSE", trading_symbol="DELTA"), 31, Decimal("880")),
)

_FILL = FillModel(
    latency_seconds=Decimal(60),
    half_spread_fraction=Decimal("0.0002"),
    slippage_fraction=Decimal("0.0001"),
)
"""Only needed because ``ReplayConfig`` requires one. Nothing here fills."""

_SMALLER_CLIP = Decimal(40_000)
"""A clip that is valid but not the default, for the derivation tests.

Forty thousand rather than the twenty thousand the module docstring tells the
story with, because at twenty thousand the default 0.2% gross target does not
clear its own costs and the config refuses to exist -- which is itself worth a
test, and gets one below. Forty thousand is the smaller of two *constructible*
clips, and it pays 0.153% against the default clip's 0.083%, so a screen sized
at the wrong one is off by a factor of nearly two and cannot hide in rounding.
"""


# --- a tape, so the comparison is behavioural --------------------------------


def _triangle(step: int, half: int) -> int:
    """A triangular wave, up to ``half`` then back down. Deliberately not random.

    The same generator the replay tests use, for the same reason: a failure has
    to be reproducible from this file alone, and two triangles of coprime
    periods give trends and reversals without repetition over a session.
    """
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


_QUIET = Instrument(exchange="NSE", trading_symbol="QUIET")
"""A name that trades all day and never goes anywhere.

A sixty-paise range on a two-thousand-rupee share is a one-minute ATR of about
0.046%. Against a 0.2% gross target that is hopeless -- three ATRs of room is
0.137%, so the target sits beyond any move this stock plausibly makes -- and it
is exactly the kind of name the cost screen exists to keep away from the AI.
Deliberately not a *broken* name: it has volume, it has a price, every indicator
warms up on it, and every rule can score it. Nothing but the cost arithmetic
disqualifies it, which is what makes it a test of the cost arithmetic.
"""

_QUIET_BASE = Decimal(2000)
_QUIET_RANGE = Decimal("0.60")


def _quiet_candle(minute_index: int) -> Candle:
    start = _ORIGIN + minute_index * _ONE_MINUTE
    low = _QUIET_BASE if minute_index % 2 else _QUIET_BASE + _QUIET_RANGE
    high = low + _QUIET_RANGE
    return Candle(
        instrument=_QUIET,
        start_time=start,
        end_time=start + _ONE_MINUTE,
        open=low,
        high=high,
        low=low,
        close=high,
        volume=5_000,
    )


def _minute(minute_index: int, *, quiet: bool = False) -> tuple[Candle, ...]:
    out: list[Candle] = []
    for instrument, phase, base in _NAMES:
        start = _ORIGIN + minute_index * _ONE_MINUTE
        open_price = _close_price(base, minute_index, phase)
        close_price = _close_price(base, minute_index + 1, phase)
        out.append(
            Candle(
                instrument=instrument,
                start_time=start,
                end_time=start + _ONE_MINUTE,
                open=open_price,
                high=max(open_price, close_price) + Decimal("0.85"),
                low=min(open_price, close_price) - Decimal("0.75"),
                close=close_price,
                volume=4_000 + (minute_index * 137 + phase * 29) % 900,
            )
        )
    if quiet:
        out.append(_quiet_candle(minute_index))
    return tuple(out)


def _scan_every_minute(
    *scanners: Scanner, quiet: bool = False
) -> tuple[tuple[ScanResult, ...], ...]:
    """Drive every scanner over one tape, and return each one's scan stream.

    One ``FeatureEngine``, folded once, shared by all the scanners. Building an
    engine per scanner would leave open the possibility that two streams differ
    because their *inputs* differed; sharing the fold means the scanners are the
    only thing that can vary, which is what is being compared.

    The book is empty at every minute. This file is about configuration, and an
    empty book is the state in which the scanner's own settings -- the candidate
    budget, the cost screen, the session window -- are the only things deciding
    the answer. Suppression by the book is the replay tests' subject.
    """
    streams: tuple[list[ScanResult], ...] = tuple([] for _ in scanners)
    with localcontext(FEATURE_CONTEXT):
        features = FeatureEngine()
        for minute_index in range(_MINUTES):
            for candle in _minute(minute_index, quiet=quiet):
                features.update(candle)
            as_of = _ORIGIN + (minute_index + 1) * _ONE_MINUTE
            snapshots = features.snapshots()
            portfolio = PortfolioState(as_of=as_of)
            context = MarketContext(as_of=as_of)
            for stream, scanner in zip(streams, scanners, strict=True):
                stream.append(scanner.scan(snapshots, portfolio, context=context))
    return tuple(tuple(stream) for stream in streams)


def _candidate_count(stream: tuple[ScanResult, ...]) -> int:
    return sum(len(result.candidates) for result in stream)


# --- the tape has to do something before agreement means anything ------------


def test_the_tape_produces_candidates() -> None:
    """Non-vacuity for every comparison below.

    Two scanners that both find nothing agree perfectly. If a future edit to the
    fixture, the rules or the cost screen stopped this tape producing
    candidates, every equality test in this file would still pass while checking
    nothing, so the tape's output is asserted here rather than assumed.
    """
    (stream,) = _scan_every_minute(StrategyConfig().scanner())

    assert len(stream) == _MINUTES
    assert _candidate_count(stream) > 50
    assert sum(1 for result in stream if result.candidates) > 20


# --- the guarantee this package exists to give -------------------------------


def test_replay_and_live_build_the_same_scanner() -> None:
    """One config, two callers, identical scans at every minute of a session.

    This is the whole point of the package, stated as a test. The two calls
    below stand in for the replay engine and a live session: neither configures
    a scanner, both ask ``StrategyConfig`` for one. There is no assertion here
    that they *should* be equal -- there is no mechanism by which they could
    differ, and that is the property being locked in. A future edit that gives
    either path its own construction site is what this test is waiting for.

    Compared at the level of the whole ``ScanResult``, not just the candidates:
    the counts, the suppression reasons and the truncation flag are part of what
    the AI would be shown, and a difference in any of them is a difference in
    what the two paths saw.
    """
    strategy = StrategyConfig()
    replay_stream, live_stream = _scan_every_minute(
        strategy.scanner(), strategy.scanner()
    )

    assert _candidate_count(replay_stream) > 0
    assert replay_stream == live_stream


def test_the_scanner_is_the_same_scanner_structurally_too() -> None:
    """The faster thing to read when the behavioural test above fails.

    ``Scanner`` is not a dataclass, so ``==`` on two of them is identity and
    would be false however identically they were built. The comparable parts are
    the config and the rule list, and those are what a drift would show up in.
    """
    strategy = StrategyConfig()
    replay, live = strategy.scanner(), strategy.scanner()

    assert replay is not live
    assert replay.config == live.config
    assert replay.rules == live.rules
    assert replay.config == strategy.scanner_config()


def test_two_different_strategies_build_different_scanners() -> None:
    """The anti-vacuity check for the two tests above.

    If ``scanner_config`` ignored the fields it reads -- returning a default, or
    dropping the screen -- the equality tests would pass and mean nothing. So a
    config that differs in a field the scanner actually consults has to produce
    a visibly different answer over the same tape.

    ``max_candidates`` is the field chosen because its effect is unambiguous:
    a one-candidate budget over a tape that regularly offers more must return
    strictly fewer candidates in total, and must say it truncated.
    """
    default_stream, narrow_stream = _scan_every_minute(
        StrategyConfig().scanner(),
        StrategyConfig(max_candidates=1).scanner(),
    )

    assert _candidate_count(narrow_stream) < _candidate_count(default_stream)
    assert any(result.truncated for result in narrow_stream)
    for result in narrow_stream:
        assert len(result.candidates) <= 1


def test_the_scanner_scores_with_the_rules_it_was_given() -> None:
    """``scanner()`` passes ``self.rules`` through, not the module default.

    Added because mutation testing found the gap: replacing ``self.rules`` with
    ``DEFAULT_RULES`` inside ``scanner()`` left every other test in this file
    passing, for the boring reason that ``rules`` *defaults* to ``DEFAULT_RULES``
    and nothing here had ever set it to anything else. A field nobody varies is a
    field nobody is testing.

    One rule out of the five, chosen because it still fires: on this tape the
    VWAP rule alone accounts for 37 of the 219 candidates the full set produces.
    A subset that scored nothing would make the behavioural assertion true for
    the wrong reason -- a scanner that had broken entirely would pass it too --
    so the count is bounded on both sides.
    """
    rules = (VwapReversionRule(),)
    strategy = StrategyConfig(rules=rules)

    assert strategy.scanner().rules == rules
    assert strategy.scanner().rules != DEFAULT_RULES

    default_stream, one_rule_stream = _scan_every_minute(
        StrategyConfig().scanner(), strategy.scanner()
    )
    scored = _candidate_count(one_rule_stream)

    assert 0 < scored < _candidate_count(default_stream)


def test_the_screen_refuses_a_name_that_cannot_pay_for_its_round_trip() -> None:
    """The screen's actual job, on a name where it has a real opinion.

    ``_QUIET`` is liquid, warm and scoreable; the only thing wrong with it is
    that its target sits further away than it ever moves. With the screen on it
    is refused as *unreachable* -- measured and declined -- rather than as
    not-ready, and it never reaches the AI. The distinction matters because the
    two counters are read very differently in a report: a session of unreachable
    names is a market too quiet to trade at this clip, a session of not-ready
    ones is an engine that never warmed up.
    """
    (stream,) = _scan_every_minute(StrategyConfig().scanner(), quiet=True)

    assert sum(result.unreachable for result in stream) > 100
    assert not any(
        candidate.instrument == _QUIET
        for result in stream
        for candidate in result.candidates
    )


def test_without_the_screen_the_unaffordable_name_reaches_the_ai() -> None:
    """Why the screen is on by default, stated as the damage of turning it off.

    The same stock, the same tape, the same rules. With the cost screen removed
    it is not merely considered -- it is *ranked*, repeatedly, and takes a third
    of the candidate budget across the session. Every one of those is a trade
    idea whose target the stock has no realistic way of reaching, competing for
    a place against names that could.

    This is the test that makes ``screen_feasibility`` worth having as a field
    at all: it measures what the screen is buying, which is the number the sweep
    in Step 7 will want to put against what it costs.
    """
    screened, unscreened = _scan_every_minute(
        StrategyConfig().scanner(),
        StrategyConfig(screen_feasibility=False).scanner(),
        quiet=True,
    )

    def quiet_candidates(stream: tuple[ScanResult, ...]) -> int:
        return sum(
            1
            for result in stream
            for candidate in result.candidates
            if candidate.instrument == _QUIET
        )

    assert quiet_candidates(screened) == 0
    assert quiet_candidates(unscreened) > 50
    assert _candidate_count(unscreened) > _candidate_count(screened)
    assert sum(result.unreachable for result in unscreened) == 0


def test_turning_the_screen_off_removes_it_rather_than_loosening_it() -> None:
    """``None``, not a policy with permissive settings.

    A screen that was merely relaxed would still reject something, and the
    sweep's question -- what does the screen cost us? -- would be answered
    against a screen rather than against no screen.
    """
    off = StrategyConfig(screen_feasibility=False)

    assert off.feasibility_policy() is None
    assert off.scanner_config().feasibility is None
    assert off.scanner().config.feasibility is None


# --- derived, not restated ---------------------------------------------------


def test_the_screen_is_sized_by_the_sizer_that_will_take_the_trade() -> None:
    """The named bug, checked at the clip where it would bite.

    The screen's ``net_margin_fraction`` is not recomputed from the gross target;
    it is taken from the sizer. The distinction is invisible today, because both
    conversions would produce the same number, and it is the entire point: if
    either conversion is ever changed, they still cannot disagree, because there
    is only one of them.
    """
    for notional in (_SMALLER_CLIP, Decimal(100_000), Decimal(500_000)):
        strategy = StrategyConfig(target_notional=notional)
        sizing = strategy.sizing_policy()
        screen = strategy.feasibility_policy()

        assert screen is not None
        assert screen.net_margin_fraction == sizing.net_margin_fraction
        assert screen.target_notional == sizing.target_notional == notional
        assert strategy.net_margin_fraction == sizing.net_margin_fraction


def test_a_smaller_clip_really_does_move_the_hurdle() -> None:
    """Anti-vacuity for the test above: the numbers must actually differ.

    If the cost arithmetic were insensitive to clip size, the agreement asserted
    above would be free. Fixed charges amortise over notional, so a smaller clip
    must keep strictly less of the same gross target -- and the gap between the
    two clips is what the old two-site bug would have silently swallowed.
    """
    small = StrategyConfig(target_notional=_SMALLER_CLIP)
    large = StrategyConfig(target_notional=Decimal(100_000))

    assert small.net_margin_fraction < large.net_margin_fraction
    # Nearly a factor of two in the hurdle -- 0.153% against 0.083% -- which is
    # the size of the error a screen sized at the wrong clip would introduce.
    small_hurdle = small.gross_target_fraction - small.net_margin_fraction
    large_hurdle = large.gross_target_fraction - large.net_margin_fraction
    assert small_hurdle > large_hurdle * Decimal("1.5")


def test_every_assumption_the_screen_makes_comes_from_the_config() -> None:
    """No field reaches the screen except through ``StrategyConfig``.

    Each of these is a value the screen would otherwise have to be told
    separately, and each one told separately is a place replay and live could
    come apart. ``square_off_minutes_since_open`` matters most: the screen uses
    it to decide how much of the session is left, so a screen that disagreed
    with the engine about when the session ends would reject names the engine
    was still willing to trade.
    """
    strategy = StrategyConfig(
        target_notional=_SMALLER_CLIP,
        costs=ZERODHA_INTRADAY_EQUITY,
        max_atr_multiple=Decimal(7),
        min_minutes_remaining=Decimal(45),
        square_off_minutes_since_open=Decimal(300),
    )
    screen = strategy.feasibility_policy()

    assert screen is not None
    assert screen.target_notional == _SMALLER_CLIP
    assert screen.costs == ZERODHA_INTRADAY_EQUITY
    assert screen.max_atr_multiple == Decimal(7)
    assert screen.min_minutes_remaining == Decimal(45)
    assert screen.square_off_minutes_since_open == Decimal(300)


def test_the_scanner_window_comes_from_the_config() -> None:
    """The session window reaches the scanner rather than being defaulted."""
    strategy = StrategyConfig(
        max_candidates=2,
        earliest_minutes_since_open=Decimal(15),
        latest_minutes_since_open=Decimal(330),
    )
    config = strategy.scanner_config()

    assert config.max_candidates == 2
    assert config.earliest_minutes_since_open == Decimal(15)
    assert config.latest_minutes_since_open == Decimal(330)


# --- refusals happen when the config is written ------------------------------


def test_a_target_its_own_costs_would_eat_is_refused_at_write_time() -> None:
    """The sizer is built in ``__post_init__`` and thrown away, for this.

    A twenty-thousand clip against the default 0.2% gross target does not clear
    its own charges -- it leaves about -0.07% -- and the strategy that describes
    it cannot be constructed at all. That is the strongest available form of the
    module's promise: the mismatch its docstring tells the story with is not
    merely detected, it is unrepresentable.

    The refusal has to happen here, at the moment the configuration is written,
    rather than several steps into a run that has already fetched a year of
    candles.
    """
    with pytest.raises(ValueError, match="does not clear costs at notional 20000"):
        StrategyConfig(target_notional=Decimal(20_000))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_candidates", 0, "max_candidates must be positive"),
        ("max_candidates", -1, "max_candidates must be positive"),
        ("max_open_positions", 0, "max_open_positions must be positive"),
        ("cooldown_minutes", Decimal(-1), "cooldown_minutes cannot be negative"),
        (
            "square_off_minutes_since_open",
            Decimal(0),
            "square_off_minutes_since_open must be positive",
        ),
        ("tick_size", Decimal(0), "tick_size must be positive"),
        ("tick_size", Decimal("-0.05"), "tick_size must be positive"),
    ],
)
def test_a_field_outside_its_range_is_refused(
    field: str, value: object, message: str
) -> None:
    """Each validation, by the value that should trip it.

    ``cooldown_minutes`` is the one that admits zero -- no cooldown is a real
    setting -- while the rest require strictly positive values, so the boundary
    is part of what is being checked rather than an accident of the operator
    somebody typed.
    """
    with pytest.raises(ValueError, match=message):
        StrategyConfig(**{field: value})


def test_zero_cooldown_is_allowed_because_it_means_something() -> None:
    """The boundary the test above deliberately does not reject."""
    assert StrategyConfig(cooldown_minutes=Decimal(0)).cooldown_minutes == 0


# --- the header --------------------------------------------------------------


def test_describe_reports_the_derived_figures_not_the_declared_ones() -> None:
    """A report header has to show what the run actually used.

    This also exercises the whole of ``describe``, which is worth doing for a
    reason that is not obvious from reading it: an earlier draft interpolated
    ``self.costs.name``, and ``CostModel`` has no ``name`` field. It would have
    raised at the top of every report, after the run had finished.
    """
    strategy = StrategyConfig()
    lines = strategy.describe()
    text = "\n".join(lines)
    hurdle = strategy.gross_target_fraction - strategy.net_margin_fraction

    assert len(lines) == 10
    assert f"{hurdle:.3%}" in text
    assert f"{strategy.net_margin_fraction:.3%}" in text
    assert "Groww intraday equity" in text
    assert strategy.exit_policy.description in text
    # Rates are fractions of notional everywhere in this system; a header that
    # printed 0.200000% for a 0.2% target would be the first place that slips.
    assert "0.200%" in text
    assert "0.200000%" not in text


def test_describe_follows_the_configuration_it_is_given() -> None:
    """Anti-vacuity: the header is not a fixed block of text."""
    default = "\n".join(StrategyConfig().describe())
    changed = "\n".join(
        StrategyConfig(
            costs=ZERODHA_INTRADAY_EQUITY,
            exit_policy=ChandelierStop(Decimal(3)),
            max_open_positions=1,
        ).describe()
    )

    assert changed != default
    assert "Zerodha intraday equity" in changed
    assert "1 positions" in changed


def test_an_unpublished_schedule_is_described_by_its_terms() -> None:
    """``CostModel`` carries no name, so the label is recovered by comparison.

    Two schedules that agree on every rate are the same schedule whatever either
    is called, which is why the label is derived rather than stored. Anything
    unrecognised is reported by the terms that actually vary between brokers.
    """
    custom = dataclasses.replace(
        GROWW_INTRADAY_EQUITY, brokerage_fraction=Decimal("0.00001")
    )
    text = "\n".join(StrategyConfig(costs=custom).describe())

    assert "custom, brokerage" in text
    assert "Groww" not in text


# --- nothing downstream may restate the strategy -----------------------------


def test_replay_config_adds_only_what_replay_alone_needs() -> None:
    """``ReplayConfig`` composes the strategy rather than copying fields out.

    The three extra fields earn their place: a universe because a live session
    gets its names from a watchlist, a fill model because replay has to model
    the latency and spread a live session observes, and the date range, which is
    implicit in the candles. A fourth field naming anything the strategy already
    states would be a second site for it, which is the bug this package exists
    to prevent -- so the field list is asserted exactly rather than loosely.
    """
    names = {field.name for field in dataclasses.fields(ReplayConfig)}

    assert names == {"universe", "fill", "strategy"}


def test_the_engine_scans_with_the_strategys_own_scanner() -> None:
    """The composition is real: the strategy on a ``ReplayConfig`` is the one.

    ``ReplayEngine`` builds its scanner inside ``run`` and never exposes it, so
    the reachable claim here is that the strategy it would build one from is the
    strategy it was handed, unmodified. That the engine's *scans* match a
    scanner built from that strategy is checked over a real tape in
    ``tests/test_replay_equivalence.py``; this is the cheap structural half.
    """
    strategy = StrategyConfig(exit_policy=FixedAtrStop(Decimal("1.5")))
    config = ReplayConfig(
        universe=(_NAMES[0][0],),
        fill=_FILL,
        strategy=strategy,
    )

    assert config.strategy is strategy
    assert config.strategy.scanner().config == strategy.scanner_config()
    assert config.strategy.exit_policy == FixedAtrStop(Decimal("1.5"))
