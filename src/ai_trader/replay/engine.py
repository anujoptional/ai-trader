"""The loop: candles in, trades out, with nothing read out of order.

One minute at a time, in five steps whose order is the whole correctness
argument:

1. **Fold** every candle for this minute into the ``FeatureEngine``.
2. **Advance** open positions against those candles. Exits happen first because
   an exit at 10:31 frees a slot a 10:31 decision can use, which is the order
   reality works in.
3. **Fill** any queued entry whose moment has arrived.
4. **Scan** — build a ``PortfolioState`` from the book as it now stands, ask the
   scanner, and queue what it returns.
5. **Fill again**, which only does anything at zero latency, where the fill is
   the signal bar's own close.

Steps 3 and 5 are the same call in two places rather than two mechanisms, and
that is what makes latency a continuous parameter instead of a special case: at
L=0 a signal fills at its bar's close, at L=60 at the next bar's close, and
everything between interpolates.

**The loop cannot look ahead, structurally.** It never holds a bar it has not
reached. A queued entry records the moment it may fill and then waits to be
passed a bar that covers that moment; it is never given the chance to shop
among later prices. The prefix test in ``tests/test_replay_no_lookahead.py``
proves the consequence — truncate the input at any bar and every trade that had
already closed comes back byte for byte — but the reason it passes is here, in
the shape of the loop, not in the test.

**Two things this layer models rather than observes, and both flatter it.**

The fill price inside a bar is interpolated between that bar's open and close.
It is a straight line drawn through a minute that did not travel in a straight
line, and it is drawn using the bar's close, which had not printed when a live
order would have filled. It does not move any *decision* — the decision was
made at the previous close and nothing can change it — but it does mean a fill
price is a model, not a price that traded.

And the tape replay runs on is cleaner than the tape a live session sees. The
historical endpoint returns every minute; a live session produces a candle only
when a tick arrives, so quiet minutes simply have no bar and indicator periods
are counted in candles rather than in minutes. Replaying history therefore runs
a gapless series that live trading will not get. The same asymmetry sits in the
volume feed: withheld cumulative volume makes a live scan see less than replay
does, which flatters ``VwapReversionRule`` specifically. Neither is a bug to be
fixed here. Both are reasons a replay number is an upper bound.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, localcontext

from ai_trader.broker import Instrument
from ai_trader.features import FEATURE_CONTEXT, FeatureEngine, FeatureSnapshot
from ai_trader.market import (
    INDIA_TIMEZONE,
    SESSION_OPEN_TIME,
    Candle,
    trading_session_date,
)
from ai_trader.replay.models import FillModel, ReplayResult, SimulatedTrade
from ai_trader.replay.portfolio import ReplayPortfolio
from ai_trader.scanner import (
    Candidate,
    MarketContext,
    PortfolioState,
    ScanResult,
    SuppressionReason,
    available,
)
from ai_trader.strategy import StrategyConfig


def minutes_since_open(moment: datetime) -> Decimal:
    """How far into the session a moment is, in IST minutes.

    Negative before the open, which is deliberate: a pre-open bar should fail
    an "at least this far in" test rather than wrap around into passing one.
    """
    local = moment.astimezone(INDIA_TIMEZONE)
    opened = local.replace(
        hour=SESSION_OPEN_TIME.hour,
        minute=SESSION_OPEN_TIME.minute,
        second=0,
        microsecond=0,
    )
    return Decimal((local - opened).total_seconds()) / 60


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """A run: the strategy, plus the three things only a replay needs.

    The split is the point. ``strategy`` is everything a live session would
    assume too -- clip size, costs, rules, cost screen, stop, book limits,
    square-off -- and it is *composed* rather than copied, so there is no field
    here that could be set to disagree with it. The engine below never builds a
    ``Scanner``, a ``SizingPolicy`` or a ``FeasibilityPolicy`` of its own; it
    asks ``StrategyConfig`` for them, which is the same call the live path will
    make. That is the mechanical content of the claim in Section 7.1 that the
    fork between replay and live happens after the scanner.

    The three fields that are genuinely replay-only earn their place:
    ``universe`` because a live session gets its names from a watchlist and a
    replay gets them from whatever candles it was handed; ``fill`` because
    replay has to *model* the latency and spread a live session simply observes;
    and the date range, which is implicit in the candles passed to ``run``.

    Section 7.2 requires a result to be reconstructible, so this whole object is
    cited by the ``ReplayResult`` it produced. A net expectancy reported without
    the fill model, the stop and the book size that produced it is not a
    finding, it is a number.
    """

    universe: tuple[Instrument, ...]
    fill: FillModel
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def __post_init__(self) -> None:
        if not self.universe:
            raise ValueError("universe cannot be empty")


@dataclass(frozen=True, slots=True)
class ReplayCycle:
    """One minute's worth of input and output, exactly as the scanner saw it.

    Handed to ``ReplayEngine.run``'s observer so that a caller can check the
    engine's work without re-deriving it. ``snapshots`` and ``portfolio`` are
    the two arguments the scan was given and ``result`` is what came back, so
    an outsider holding the same candles can rebuild ``snapshots`` from
    scratch, re-run the scan against the recorded ``portfolio``, and compare.
    That is the shape of the equivalence test in
    ``tests/test_replay_equivalence.py``: the book is an input to both paths,
    the features and the scan are what is being checked.
    """

    as_of: datetime
    candles: tuple[Candle, ...]
    snapshots: tuple[FeatureSnapshot, ...]
    portfolio: PortfolioState
    result: ScanResult


class ReplayEngine:
    """Runs a configuration over a stream of candles and reports what happened.

    Stateless between runs. Every call to ``run`` builds a fresh
    ``FeatureEngine`` and a fresh book, because the point of this object is to
    be called repeatedly across a sweep of fill models and stop multiples, and a
    sweep whose second run inherited the first run's indicator state would be
    measuring the order the sweep happened to iterate in.
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config

    @property
    def config(self) -> ReplayConfig:
        return self._config

    def run(
        self,
        candles: Iterable[Candle],
        *,
        on_cycle: Callable[[ReplayCycle], None] | None = None,
    ) -> ReplayResult:
        """Replay a candle stream end to end.

        Candles may arrive in any order and for any mix of instruments; they
        are sorted into minutes here, because "what did the scanner see at
        10:31" is only answerable if every instrument's 10:31 bar is folded in
        before the question is asked. Sorting is on the bar's end time, which
        is the moment its information became available.

        ``on_cycle`` receives a ``ReplayCycle`` for every minute, carrying the
        scan's inputs as well as its output. It exists so that the equivalence
        test can audit the scan stream without this loop growing a debug mode,
        and it follows the ``on_candle`` callback the market layer already uses.

        The whole run happens inside ``FEATURE_CONTEXT``. The scanner installs
        it per cycle anyway; installing it here as well means the cost and fill
        arithmetic between cycles is evaluated at the same precision as the
        features it is being compared against, rather than at whatever context
        the caller happened to have.
        """
        config = self._config
        with localcontext(FEATURE_CONTEXT):
            return self._run(config, candles, on_cycle)

    def _run(
        self,
        config: ReplayConfig,
        candles: Iterable[Candle],
        on_cycle: Callable[[ReplayCycle], None] | None,
    ) -> ReplayResult:
        features = FeatureEngine()
        strategy = config.strategy
        scanner = strategy.scanner()
        book = ReplayPortfolio(
            universe=config.universe,
            max_open_positions=strategy.max_open_positions,
            sizing=strategy.sizing_policy(),
            costs=strategy.costs,
            fill=config.fill,
            exit_policy=strategy.exit_policy,
            cooldown_minutes=strategy.cooldown_minutes,
        )

        trades: list[SimulatedTrade] = []
        sessions: list[date] = []
        suppressed: dict[SuppressionReason, int] = {}
        last_close: dict[Instrument, Decimal] = {}
        last_seen: dict[Instrument, datetime] = {}
        session: date | None = None
        counts = _Counters()

        for cycle in _by_minute(candles):
            counts.candles += len(cycle)
            counts.cycles += 1
            moment = cycle[0].end_time
            bar_session = trading_session_date(moment)

            if session is not None and bar_session != session:
                trades.extend(self._square_off_all(book, last_close, last_seen, counts))
                book.reset_session()
            if bar_session != session:
                sessions.append(bar_session)
                session = bar_session

            bars = {candle.instrument: candle for candle in cycle}
            for candle in cycle:
                last_close[candle.instrument] = candle.close
                last_seen[candle.instrument] = candle.end_time
                features.update(candle)

            for candle in cycle:
                closed = book.advance(candle)
                if closed is not None:
                    trades.append(closed)

            self._fill_pending(book, bars, moment)

            elapsed = minutes_since_open(moment)
            past_cutoff = elapsed >= strategy.square_off_minutes_since_open
            if past_cutoff:
                trades.extend(self._square_off_all(book, last_close, last_seen, counts))

            snapshots = features.snapshots()
            state = book.state(moment, entries_blocked=past_cutoff)
            result = scanner.scan(snapshots, state, context=MarketContext(as_of=moment))
            if on_cycle is not None:
                on_cycle(
                    ReplayCycle(
                        as_of=moment,
                        candles=cycle,
                        snapshots=snapshots,
                        portfolio=state,
                        result=result,
                    )
                )
            for reason, count in result.suppressed.items():
                suppressed[reason] = suppressed.get(reason, 0) + count
            counts.candidates += len(result.candidates)

            if not past_cutoff:
                self._queue(book, result.candidates, snapshots, counts)
                self._fill_pending(book, bars, moment)

        # Only the *final* flush counts as "open at end". Session rollovers and
        # the intraday cutoff use the same helper and are ordinary square-offs,
        # so the running total is read before and after rather than reused.
        before_final_flush = counts.squared_off_at_end
        trades.extend(self._square_off_all(book, last_close, last_seen, counts))
        counts.open_at_end = counts.squared_off_at_end - before_final_flush
        for pending in book.pending_entries():
            book.abandon(pending)
            counts.unfilled += 1

        return ReplayResult(
            trades=tuple(trades),
            universe=config.universe,
            sessions=tuple(sessions),
            fill=config.fill,
            candles_replayed=counts.candles,
            cycles=counts.cycles,
            candidates_seen=counts.candidates,
            declined_book_full=counts.book_full,
            declined_no_volatility=counts.no_volatility,
            unfilled_entries=counts.unfilled,
            open_at_end=counts.open_at_end,
            unwound_entries=counts.unwound,
            suppressed=suppressed,
        )

    def _queue(
        self,
        book: ReplayPortfolio,
        candidates: Sequence[Candidate],
        snapshots: Sequence[FeatureSnapshot],
        counts: _Counters,
    ) -> None:
        """Take candidates in rank order until the book is full.

        The scanner has already suppressed names that are held or queued, so
        anything arriving here is genuinely new. What it cannot know is how
        many of its own five candidates the book has room for, because that
        depends on how many of them get taken — so the budget is spent here,
        in order, and whatever falls off the end is counted rather than
        silently dropped.
        """
        by_instrument = {snapshot.instrument: snapshot for snapshot in snapshots}
        for candidate in candidates:
            if book.is_committed(candidate.instrument):
                continue
            if book.committed() >= book.max_open_positions:
                counts.book_full += 1
                continue
            snapshot = by_instrument.get(candidate.instrument)
            atr = None if snapshot is None else available(snapshot, "atr_pct")
            if atr is None:
                # Nothing to place a stop against. A position that cannot be
                # stopped is not a position this layer is willing to model,
                # and inventing a rupee stop would make the result a function
                # of the share price rather than of the strategy.
                counts.no_volatility += 1
                continue
            book.queue(candidate, atr)

    def _fill_pending(
        self,
        book: ReplayPortfolio,
        bars: dict[Instrument, Candle],
        moment: datetime,
    ) -> None:
        """Fill every queued entry whose moment this minute has reached."""
        for pending in book.pending_entries():
            if pending.fill_at > moment:
                continue
            candle = bars.get(pending.instrument)
            if candle is None:
                # The name went quiet. A live order would still be resting, so
                # the entry stays queued and fills on whatever bar comes next.
                continue
            price = _price_at(candle, pending.fill_at)
            book.enter(pending, price, pending.fill_at, candle.end_time)

    def _square_off_all(
        self,
        book: ReplayPortfolio,
        last_close: dict[Instrument, Decimal],
        last_seen: dict[Instrument, datetime],
        counts: _Counters,
    ) -> list[SimulatedTrade]:
        """Close everything still open, at each name's own last printed price.

        Each name squares off against its *own* last bar rather than against a
        common timestamp, because a name that stopped ticking at 14:40 has no
        15:30 price and pretending otherwise would invent one.
        """
        closed: list[SimulatedTrade] = []
        for instrument in book.open_instruments():
            price = last_close.get(instrument)
            at = last_seen.get(instrument)
            if price is None or at is None:
                continue
            trade = book.square_off(instrument, price, at)
            if trade is not None:
                closed.append(trade)
                counts.squared_off_at_end += 1
            else:
                # The book refused and dropped it: this position filled on the
                # very bar the session ran out on, so there is no bar it was
                # ever fully on and no round trip to record. Counted rather
                # than ignored, because a run where this number is large is a
                # run whose cutoff is misconfigured.
                counts.unwound += 1
        return closed


@dataclass(slots=True)
class _Counters:
    candles: int = 0
    cycles: int = 0
    candidates: int = 0
    book_full: int = 0
    no_volatility: int = 0
    unfilled: int = 0
    squared_off_at_end: int = 0
    open_at_end: int = 0
    unwound: int = 0


def _by_minute(candles: Iterable[Candle]) -> Iterable[tuple[Candle, ...]]:
    """Group candles into the minutes they closed in, earliest first.

    Sorted by end time and then by symbol, so that a run over the same data is
    byte-identical whatever order the caller assembled it in. Ordering within a
    minute is not supposed to matter — every bar in a group is folded in before
    anything is asked of the features — and sorting anyway means that if it ever
    does start to matter, it will do so reproducibly.
    """
    ordered = sorted(
        candles,
        key=lambda c: (
            c.end_time,
            c.instrument.exchange,
            c.instrument.trading_symbol,
        ),
    )
    group: list[Candle] = []
    for candle in ordered:
        if group and candle.end_time != group[0].end_time:
            yield tuple(group)
            group = []
        group.append(candle)
    if group:
        yield tuple(group)


def _price_at(candle: Candle, moment: datetime) -> Decimal:
    """The modelled price inside a bar at a given instant.

    A straight line from open to close. The bar's high and low are deliberately
    ignored: they say the price visited those levels but not when, and picking
    one would be choosing a fill from a path nobody recorded. Interpolating is
    a smaller lie than that, and it makes latency continuous — at the bar's end
    this returns exactly the close, which is what a zero-latency fill should be.

    A moment at or before the bar's start means the fill was due during a gap
    with no bar at all, so it gets this bar's open: the first price actually
    available after the order was live.
    """
    if moment <= candle.start_time:
        return candle.open
    span = Decimal((candle.end_time - candle.start_time).total_seconds())
    if span <= 0:
        return candle.close
    elapsed = Decimal((moment - candle.start_time).total_seconds())
    if elapsed >= span:
        return candle.close
    return candle.open + (candle.close - candle.open) * elapsed / span


__all__ = ["ReplayConfig", "ReplayCycle", "ReplayEngine", "minutes_since_open"]
