"""A declined candidate must land in a counter, not in a gap between counters.

``ReplayResult`` states the invariant in its own docstring: "every candidate
either became a trade, was declined for a stated reason, or is still open at the
end. A run whose numbers do not add up has a bug, and the arithmetic to notice
that is available to anyone holding this object." A review found the engine
breaking it. ``_queue`` declines three different ways and counted two of them;
the third was a bare ``continue``. On the fifteen-name sweep that silence
swallowed 696 of 15,479 candidates, so the arithmetic a reader was invited to do
came out 696 short with nothing in the report to say where they had gone.

**What the third decline actually is.** The scanner suppresses anything the book
holds or has queued -- ``ReplayPortfolio.state`` folds ``committed`` into
``at_position_limit`` unconditionally -- so nothing reaching ``_queue`` was
committed before the call. The only commit it can collide with is one made by an
earlier turn of its own loop, which makes every hit provably the same name
arriving twice in one candidate list. The scanner emits that on purpose: its
accumulator is keyed on ``(instrument, direction)``, and Section 4.6 makes
handing a name that one rule reads as long and another as short on to the AI
deliberate, because resolving the disagreement is what the AI is for. Replay has
no AI, so it resolves by rank -- and the counter is the record that a choice was
made. Without it a conflicted name reads in the report exactly like a name the
rules agreed about.

**Why stub rules rather than the default five.** Whether the real rules happen to
disagree about a synthetic name is a property of the price fixture, not of the
engine, and a test whose subject can be edited away by a change of wave phase is
not testing the engine. Two rules that always disagree make the conflict a
condition of the fixture rather than a coincidence inside it.

**Why one of the three terms needs the screen switched off.** With the cost
screen on, a snapshot whose ``atr_pct`` is not ready is suppressed by the
scanner as ``VOLATILITY_UNKNOWN`` before the engine ever sees it, so the
engine's own no-ATR guard cannot fire -- which is why the fifteen-name sweep
measured it at exactly zero rather than at something small. That guard is the
backstop for a run configured with ``screen_feasibility=False``, and the last
test below is the configuration under which it is reachable at all.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.features import FeatureSnapshot
from ai_trader.market import Candle
from ai_trader.replay import (
    FillModel,
    ReplayConfig,
    ReplayEngine,
    ReplayResult,
)
from ai_trader.scanner import (
    Direction,
    MarketContext,
    PortfolioState,
    RuleSignal,
    available,
)
from ai_trader.strategy import FixedAtrStop, StrategyConfig

_ONE_MINUTE = timedelta(minutes=1)

_OPEN = datetime(2026, 9, 22, 9, 15, tzinfo=INDIA_TIMEZONE)
"""09:15 IST on Tuesday 22 September 2026."""

_MINUTES = 240
"""Four hours, as elsewhere: long enough to warm the indicators and to trade."""

_NAMES = (
    (Instrument(exchange="NSE", trading_symbol="ALPHA"), 0, Decimal("2450")),
    (Instrument(exchange="NSE", trading_symbol="BRAVO"), 17, Decimal("1310")),
    (Instrument(exchange="NSE", trading_symbol="DELTA"), 31, Decimal("880")),
)

_UNIVERSE = tuple(instrument for instrument, _, _ in _NAMES)

_SLOTS = 2
"""One fewer slot than there are names, so the budget decline fires too.

The reconciliation is only worth asserting if more than one of its terms is
doing something. Three names into two slots means a flat book takes two and
declines the third for want of room, in the same cycle in which the conflicted
sides are declined for a different reason.
"""

_STRONG = Decimal("0.9")
_WEAK = Decimal("0.4")
"""Two scores, far apart and never equal.

``_rank`` orders on ``(-score, trading_symbol, exchange, direction.value)``, so
the gap decides which side of a conflicted name is taken without the tie-break
being involved. Equal scores would leave that to ``direction.value``, which is
alphabetical and would make the test a claim about the word "long".
"""


@dataclass(frozen=True, slots=True)
class _AlwaysSays:
    """A rule with one opinion, held regardless of what the tape did.

    Not a strategy. It exists so that every snapshot produces both a long and a
    short candidate for the same name, which is the situation Section 4.6 hands
    to the AI and this engine has to resolve by itself.
    """

    name: str
    direction: Direction
    score: Decimal
    required_features: tuple[str, ...] = ()

    def evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
    ) -> RuleSignal:
        atr = available(snapshot, "atr_pct")
        return RuleSignal(
            direction=self.direction,
            score=self.score,
            evidence={} if atr is None else {"atr_pct": atr},
        )


_DISAGREEING = (
    _AlwaysSays(name="always_long", direction=Direction.LONG, score=_STRONG),
    _AlwaysSays(name="always_short", direction=Direction.SHORT, score=_WEAK),
)


def _triangle(step: int, half: int) -> int:
    """A triangular wave, up to ``half`` then back down. Deliberately not random."""
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _tape() -> tuple[Candle, ...]:
    """One continuous session, every name printing every minute."""
    out: list[Candle] = []
    for instrument, phase, base in _NAMES:
        for minute_index in range(_MINUTES):
            start = _OPEN + minute_index * _ONE_MINUTE
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
                    volume=4_000 + (minute_index * 137 + phase * 29) % 2_600,
                )
            )
    return tuple(out)


def _result(*, screen: bool = True) -> ReplayResult:
    config = ReplayConfig(
        universe=_UNIVERSE,
        fill=FillModel(
            latency_seconds=Decimal(60),
            half_spread_fraction=Decimal("0.0002"),
            slippage_fraction=Decimal("0.0001"),
        ),
        strategy=StrategyConfig(
            exit_policy=FixedAtrStop(Decimal("1.5")),
            max_open_positions=_SLOTS,
            rules=_DISAGREEING,
            screen_feasibility=screen,
        ),
    )
    return ReplayEngine(config).run(_tape())


def _accounted_for(result: ReplayResult) -> int:
    """Every candidate the engine was handed, by what became of it.

    ``open_at_end`` is deliberately absent: the final flush squares those
    positions off and they arrive in ``trades`` like any other round trip, so
    adding it would count them twice.
    """
    return (
        result.declined_book_full
        + result.declined_no_volatility
        + result.declined_other_side_taken
        + result.round_trips
        + result.unwound_entries
        + result.unfilled_entries
    )


# --- the arithmetic closes -------------------------------------------------


def test_every_candidate_the_engine_saw_is_accounted_for() -> None:
    """The invariant ``ReplayResult`` claims, done as the arithmetic it invites.

    The non-vacuity guards come second on purpose. Under the defect the sum is
    short by exactly the number of conflicts, and that difference is the thing
    worth reading off a failure; a run in which no name was ever conflicted
    would balance for an uninteresting reason, which is what the guards rule
    out once the sum has had its say.
    """
    result = _result()

    assert _accounted_for(result) == result.candidates_seen
    assert result.declined_other_side_taken > 0, (
        "no name reached the book twice, so the sum above balanced without "
        "the conflict this file is about ever happening"
    )
    assert result.declined_book_full > 0, "the budget never bound"


def test_the_conflicted_sides_are_not_hiding_in_the_other_counters() -> None:
    """The new counter is a new fact, not a relabelling of an existing one.

    A fix that filed conflicted names under ``declined_book_full`` would close
    the sum above exactly as well and would still misreport every one of them,
    so the count is pinned from the other side. Every conflict is the second
    arrival of a name this same loop just committed; every commitment either
    becomes a round trip, is abandoned unfilled, or is unwound. In this fixture
    the two sides of a name always arrive together -- both rules fire on every
    snapshot -- so the relation is exact rather than a bound: one conflict per
    commitment, no commitment without one.

    Either direction of a mislabelling breaks it. Filing conflicts as budget
    refusals sends the left side to zero; filing budget refusals as conflicts
    sends it to the sum of both.
    """
    result = _result()

    committed = result.round_trips + result.unfilled_entries + result.unwound_entries
    assert result.declined_other_side_taken == committed
    assert result.declined_book_full != result.declined_other_side_taken, (
        "the two counters agree to the digit, which is what a relabelling "
        "would look like if the fixture happened to balance"
    )


# --- the side that was taken is the side that was ranked -------------------


def test_the_higher_ranked_side_is_the_one_that_trades() -> None:
    """What the report tells a reader about a conflict has to be true.

    ``declined, both ways`` is printed with "the higher-ranked side was taken
    and this one dropped", and a reader has no way to check that. It is a
    theorem here rather than a coincidence: the weak side can only be taken if
    the strong side of the same name was declined, and both sides read one
    snapshot through one book, so anything that declines the strong side
    declines the weak one too.
    """
    result = _result()

    assert result.round_trips > 0, "a run with no trades asserts nothing below"
    wrong = [trade for trade in result.trades if trade.direction is not Direction.LONG]
    assert not wrong, (
        f"{len(wrong)} trades took the weaker side, first "
        f"{wrong[0].instrument.trading_symbol} at {wrong[0].entry_time}"
    )


# --- the no-ATR backstop, which only a screenless run can reach -------------


def test_the_no_atr_decline_is_reachable_and_counted_when_the_screen_is_off() -> None:
    """With no screen ahead of it, the engine's own volatility guard fires.

    Both runs are asserted, because the claim is a pair: the guard is dead
    under the default configuration -- not because it is wrong but because the
    screen refuses the same snapshot earlier and for a better-stated reason --
    and live under the configuration the screen is not there. The sum closes
    either way, which is the point of asserting it twice.
    """
    screened = _result()
    unscreened = _result(screen=False)

    assert screened.declined_no_volatility == 0
    assert unscreened.declined_no_volatility > 0
    assert _accounted_for(unscreened) == unscreened.candidates_seen
