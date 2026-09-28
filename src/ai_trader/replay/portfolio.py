"""The simulated book: what is held, what is queued, and when it closes.

This is the object that makes replay a measurement of *trading* rather than a
measurement of signalling. Without it the scanner would be asked what it liked
every minute of every session and the answer would be a count of opinions. With
it, an instrument already held cannot be entered again, a full book turns the
sixth idea into a suppression rather than a trade, and "round trips per session"
means what it says.

**It builds the same ``PortfolioState`` the live path will build.** That type's
own docstring sets the terms: "Built from real fills in live trading, simulated
fills in shadow mode, and simulated fills over historical candles in replay. One
object, one set of consumers, three sources — which is exactly what makes the
three modes comparable." Replay gets no private channel into the scanner. It
fills in the same fields, and the scanner cannot tell it apart from the real
thing, which is the property the whole comparison rests on.

**Pending entries are held here, not in the engine, and that is a bug fix.** A
signal at 10:30 that fills at 10:30:02 is not in ``open_positions`` when the
10:31 scan runs, so a book that tracked only open positions would signal the
same name a second time and enter it twice. Anything committed — held *or*
queued — is reported to the scanner as being at its position limit, so the
scanner declines it for a stated reason that shows up in ``ScanResult``.

**Exits are checked before entries are filled, within a bar.** An exit at 10:31
frees a slot that a 10:31 decision can use, which is the order reality works in.
The engine owns that sequencing; what this module guarantees is the narrower
rule that a position cannot exit on the bar it entered on. Its fill happened
somewhere inside that bar, and the bar's high and low include the part of the
minute that had already passed — so testing them against a target would let a
trade exit on a move that happened before it was in. The cost is that exits are
late by up to one bar; that direction is the safe one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from ai_trader.broker import Instrument
from ai_trader.clock import (
    ONE_MINUTE,
    ONE_SECOND,
    exact_timedelta,
    trading_session_date,
)
from ai_trader.costs import CostModel, RoundTripCost, SizingPolicy
from ai_trader.market import Candle
from ai_trader.replay.models import ExitReason, FillModel, SimulatedTrade
from ai_trader.scanner import Candidate, Direction, PortfolioState, Position
from ai_trader.strategy import DEFAULT_EXIT_POLICY, ExitPolicy, stop_price_for


@dataclass(frozen=True, slots=True)
class PendingEntry:
    """A decision taken, waiting for the clock to reach its fill.

    Everything here was known at ``signal_time``. That is the whole point of
    the type: the gap between deciding and filling is where a backtest cheats,
    and a record that cannot see past its own signal cannot cheat across it.
    ``atr_fraction`` in particular is captured from the snapshot that produced
    the candidate, not re-read later, so the stop is placed using the
    volatility the decision saw rather than the volatility that followed it.
    """

    instrument: Instrument
    direction: Direction
    rules: tuple[str, ...]
    score: Decimal
    signal_time: datetime
    atr_fraction: Decimal
    fill_at: datetime


@dataclass(slots=True)
class _OpenTrade:
    """A filled position, accumulating what it needs to describe itself later."""

    instrument: Instrument
    direction: Direction
    session: date
    rules: tuple[str, ...]
    score: Decimal
    signal_time: datetime
    entry_time: datetime
    entry_price: Decimal
    quantity: int
    notional: Decimal
    target_price: Decimal
    stop_price: Decimal
    cost: RoundTripCost
    entry_bar_end: datetime
    atr_fraction: Decimal
    """The volatility the *signal* saw, carried forward from ``PendingEntry``.

    Retained rather than discarded at entry because a trailing stop is re-asked
    every bar and has to be re-asked in the same units it was first asked in.
    Re-reading the ATR from each new bar would let the stop widen or tighten on
    volatility that printed after the decision, which is lookahead wearing the
    costume of a risk rule.
    """
    max_favourable_fraction: Decimal = Decimal(0)
    max_adverse_fraction: Decimal = Decimal(0)
    bars_held: int = 0
    ambiguous_exit: bool = False

    @property
    def position(self) -> Position:
        signed = self.quantity if self.direction is Direction.LONG else -self.quantity
        return Position(
            instrument=self.instrument,
            quantity=signed,
            average_price=self.entry_price,
        )

    def observe(self, candle: Candle) -> None:
        """Record how far the bar went each way, as fractions of entry."""
        if self.direction is Direction.LONG:
            favourable = (candle.high - self.entry_price) / self.entry_price
            adverse = (self.entry_price - candle.low) / self.entry_price
        else:
            favourable = (self.entry_price - candle.low) / self.entry_price
            adverse = (candle.high - self.entry_price) / self.entry_price
        self.max_favourable_fraction = max(self.max_favourable_fraction, favourable)
        self.max_adverse_fraction = max(self.max_adverse_fraction, adverse)

    def touches_target(self, candle: Candle) -> bool:
        if self.direction is Direction.LONG:
            return candle.high >= self.target_price
        return candle.low <= self.target_price

    def touches_stop(self, candle: Candle) -> bool:
        if self.direction is Direction.LONG:
            return candle.low <= self.stop_price
        return candle.high >= self.stop_price

    def stop_reference(self, candle: Candle) -> Decimal:
        """The price a stop actually gets, before spread and slippage.

        The stop level unless the bar opened through it, in which case the open
        — because a stop that was already breached when the minute began does
        not fill at the level, it fills at the first price available. Minute
        bars inside a session rarely gap, but the ones that do are exactly the
        bars a stop exists for, and modelling them at the level would credit the
        strategy with a fill nobody could have got.
        """
        if self.direction is Direction.LONG:
            return min(self.stop_price, candle.open)
        return max(self.stop_price, candle.open)


@dataclass(slots=True)
class ReplayPortfolio:
    """The book, its capacity, and the rules for getting in and out of it.

    ``universe`` is held because a full book has to be expressed to the scanner
    somehow, and the only vocabulary ``PortfolioState`` offers is
    ``at_position_limit`` — a set of names. So when capacity is gone, every name
    is reported as being at its limit, the scanner suppresses the lot, and
    ``ScanResult.suppressed`` counts it. The alternative, filtering the
    scanner's output afterwards, would produce the same trades and lose the
    reason, and the reason is the more useful half.

    ``cooldown_minutes`` defaults to zero, meaning the gate is off, following
    ``FeasibilityPolicy.min_minutes_remaining`` and the unbounded session window
    in ``ScannerConfig``: a gate with no measurement behind it starts open and
    is a parameter to sweep, not a constant to invent. Zero is the permissive
    reading and it will show up as a higher trade count, so it is worth saying
    that a run which leaves it at zero is reporting an upper bound on frequency.
    """

    universe: tuple[Instrument, ...]
    max_open_positions: int
    sizing: SizingPolicy
    costs: CostModel
    fill: FillModel
    exit_policy: ExitPolicy = DEFAULT_EXIT_POLICY
    cooldown_minutes: Decimal = Decimal(0)
    _open: dict[Instrument, _OpenTrade] = field(default_factory=dict, init=False)
    _pending: dict[Instrument, PendingEntry] = field(default_factory=dict, init=False)
    _trades_today: dict[Instrument, int] = field(default_factory=dict, init=False)
    _cooldown: dict[Instrument, datetime] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.max_open_positions <= 0:
            raise ValueError(
                f"max_open_positions must be positive, got {self.max_open_positions}"
            )
        if self.cooldown_minutes < 0:
            raise ValueError(
                f"cooldown_minutes cannot be negative, got {self.cooldown_minutes}"
            )
        # Rejected here rather than at the first exit. A cooldown of a
        # ten-billionth of a minute passes the ``> 0`` gate below and converts to
        # a zero-length timedelta, so the gate reports itself on while being off
        # -- a run would show the cooldown in its conditions and never once
        # apply it.
        exact_timedelta(self.cooldown_minutes, ONE_MINUTE, name="cooldown_minutes")

    @property
    def tick(self) -> Decimal:
        """One price grid for targets and stops alike, taken from the sizer."""
        return self.sizing.tick_size

    def committed(self) -> int:
        """Slots consumed: held plus queued. Queued still costs a slot."""
        return len(self._open) + len(self._pending)

    def is_committed(self, instrument: Instrument) -> bool:
        return instrument in self._open or instrument in self._pending

    def open_instruments(self) -> tuple[Instrument, ...]:
        return tuple(self._open)

    def pending_entries(self) -> tuple[PendingEntry, ...]:
        return tuple(self._pending.values())

    def state(
        self,
        as_of: datetime,
        *,
        stale: frozenset[Instrument] = frozenset(),
        entries_blocked: bool = False,
    ) -> PortfolioState:
        """What the scanner is told, in the type the live path will fill in."""
        committed = frozenset(self._open) | frozenset(self._pending)
        if self.committed() >= self.max_open_positions:
            at_limit = committed | frozenset(self.universe)
        else:
            at_limit = committed
        return PortfolioState(
            as_of=as_of,
            open_positions={
                instrument: trade.position for instrument, trade in self._open.items()
            },
            at_position_limit=at_limit,
            trades_today=dict(self._trades_today),
            cooldown_until=dict(self._cooldown),
            stale_instruments=stale,
            new_entries_blocked=entries_blocked,
        )

    def queue(self, candidate: Candidate, atr_fraction: Decimal) -> PendingEntry:
        """Accept a candidate and fix the moment its order can fill.

        The fill moment is the signal time plus the configured latency, and it
        is computed once, here, from a timestamp the candidate carried. Nothing
        downstream may move it, because moving it forward on the basis of what
        prices did next is the exact shape of the error this layer exists to
        avoid.

        The latency is converted exactly rather than through a float. It cannot
        fail here — ``FillModel`` refuses an unrepresentable one at construction
        — but it is converted by the same call that does the refusing, so there
        is one rule for what a stated duration means rather than two.
        """
        latency = exact_timedelta(
            self.fill.latency_seconds, ONE_SECOND, name="latency_seconds"
        )
        pending = PendingEntry(
            instrument=candidate.instrument,
            direction=candidate.direction,
            rules=candidate.rules,
            score=candidate.score,
            signal_time=candidate.as_of,
            atr_fraction=atr_fraction,
            fill_at=candidate.as_of + latency,
        )
        self._pending[pending.instrument] = pending
        return pending

    def enter(
        self,
        pending: PendingEntry,
        reference_price: Decimal,
        at: datetime,
        bar_end: datetime,
    ) -> None:
        """Turn a queued decision into a position at the price it really got.

        ``reference_price`` is where the tape was. The fill model turns it into
        where the order filled: half a spread plus slippage against the trade,
        rounded onto the tick grid in the same direction. Both halves matter.
        Skipping the friction would charge an exit for crossing the book while
        letting the entry in free, and the trade would keep the difference.
        Skipping the rounding would record an entry at whatever the friction
        arithmetic produced, a price off the exchange's grid that could not
        have traded.

        The target then comes from ``SizingPolicy.estimate`` on that *filled*
        price, not on the price the signal saw. That matters: a long that paid
        up to get in must clear its costs from where it actually is, so friction
        at entry pushes the target further away rather than being absorbed.

        The stop comes from ``self.exit_policy`` with a favourable excursion of
        zero, because at entry there has not been one. That is the same call
        ``advance`` makes at the end of every bar the position survives, which is
        what lets a trailing policy start exactly where a fixed one does and
        diverge only as the trade goes the right way. The multiple behind it is
        still a convention rather than a measurement — Section 7.2 forbids
        calling an invented threshold evidence — but it is now a convention
        stated once on ``StrategyConfig`` instead of a number two callers could
        pass differently.
        """
        direction = pending.direction
        estimate = self.sizing.estimate(
            self.fill.entry_price(reference_price, direction, self.tick)
        )
        if direction is Direction.LONG:
            target = estimate.long_exit_price
        else:
            target = estimate.short_exit_price
        stop = self.exit_policy.stop_price(
            entry_price=estimate.entry_price,
            direction=direction,
            atr_fraction=pending.atr_fraction,
            favourable_fraction=Decimal(0),
            tick=self.tick,
        )
        self._pending.pop(pending.instrument, None)
        self._open[pending.instrument] = _OpenTrade(
            instrument=pending.instrument,
            direction=direction,
            session=trading_session_date(at),
            rules=pending.rules,
            score=pending.score,
            signal_time=pending.signal_time,
            entry_time=at,
            entry_price=estimate.entry_price,
            quantity=estimate.quantity,
            notional=estimate.notional,
            target_price=target,
            stop_price=stop,
            cost=self.costs.round_trip(estimate.notional),
            entry_bar_end=bar_end,
            atr_fraction=pending.atr_fraction,
        )

    def abandon(self, pending: PendingEntry) -> None:
        """Drop a queued entry that never found a bar to fill on."""
        self._pending.pop(pending.instrument, None)

    def advance(self, candle: Candle) -> SimulatedTrade | None:
        """Walk one open position forward by one bar. Returns a trade if it ended.

        The first test is the one that keeps this honest: a bar that ended at or
        before the entry bar's end is a bar the position was not fully in, and
        it is skipped entirely — not even its excursions are counted.

        A position that survives leaves with a stop re-asked of the exit policy.
        The ordering there is the whole of the no-lookahead argument and is worth
        stating plainly: this bar is tested against the level that was standing
        when it opened, and only then does the level move. So the stop a bar
        could have been hit by was computed from bars strictly before it. Trail
        first and test second and a stop could fire at a level that came into
        existence partway through the very bar that hit it, which is reading an
        intrabar path no one recorded. The price of doing it this way is that a
        trail lags by one bar; that is the direction that costs the strategy
        money rather than flattering it.

        A fixed policy returns the same number here every time, so this call is
        free in the sense that matters — it changes no existing behaviour and
        adds no branch distinguishing one kind of stop from another.
        """
        trade = self._open.get(candle.instrument)
        if trade is None or candle.end_time <= trade.entry_bar_end:
            return None

        trade.bars_held += 1
        trade.observe(candle)

        hit_target = trade.touches_target(candle)
        hit_stop = trade.touches_stop(candle)
        if hit_target and hit_stop:
            trade.ambiguous_exit = True
            reason = self.fill.resolve_ambiguous_bar_as
        elif hit_target:
            reason = ExitReason.TARGET
        elif hit_stop:
            reason = ExitReason.STOP
        else:
            trade.stop_price = self.exit_policy.stop_price(
                entry_price=trade.entry_price,
                direction=trade.direction,
                atr_fraction=trade.atr_fraction,
                favourable_fraction=trade.max_favourable_fraction,
                tick=self.tick,
            )
            return None

        if reason is ExitReason.TARGET:
            # A resting limit order fills at its own price and supplies
            # liquidity rather than taking it, so no spread and no re-rounding:
            # SizingPolicy already put this price on a tick.
            price = trade.target_price
        else:
            price = self.fill.market_exit_price(
                trade.stop_reference(candle), trade.direction, self.tick
            )
        return self._close(trade, price, candle.end_time, reason)

    def square_off(
        self, instrument: Instrument, price: Decimal, at: datetime
    ) -> SimulatedTrade | None:
        """Close a position at the end of its session, whatever it is showing.

        Intraday means intraday. A position still open at the cutoff is closed
        at the market, paying the full adverse fill, because that is what the
        broker's own square-off does and carrying it overnight would turn an
        intraday result into a different strategy's result.

        Returns ``None`` — having dropped the position — for one that has not
        lived through a bar yet, which is the same rule ``advance`` applies and
        the one this module's docstring states. It fires when an entry fills on
        the very bar a session ends on: closing it at that instant would record
        a round trip whose entry and exit share a timestamp, which pays two legs
        of costs for a price move that is zero by construction. That is not a
        trade the strategy made, it is the tape running out mid-order, and
        counting it would put a guaranteed loss into the expectancy for every
        session. The caller sees ``None`` and counts it as unwound.

        What reaches this guard is only a session boundary or the end of the
        tape. The intraday cutoff does not, because the engine abandons queued
        entries unfilled at the cutoff rather than filling them into a
        square-off on the same bar — which it used to do at any non-zero
        latency, filing a decision no bar ever priced under "the tape ran out
        mid-order" and letting the latency flag decide which it was.
        """
        trade = self._open.get(instrument)
        if trade is None:
            return None
        if at <= trade.entry_bar_end:
            del self._open[instrument]
            return None
        exit_price = self.fill.market_exit_price(price, trade.direction, self.tick)
        return self._close(trade, exit_price, at, ExitReason.SESSION_END)

    def _close(
        self,
        trade: _OpenTrade,
        exit_price: Decimal,
        at: datetime,
        reason: ExitReason,
    ) -> SimulatedTrade:
        if trade.direction is Direction.LONG:
            move = exit_price - trade.entry_price
        else:
            move = trade.entry_price - exit_price
        gross_fraction = move / trade.entry_price
        # The method the costs package names as the one replay will use: given
        # what the price did, say what the account kept. Charged on the entry
        # notional for both legs, which is the simplification the cost model
        # already committed to and documents: on a Rs 1,00,000 leg it misprices
        # a 0.2% round trip by about 6 paise out of Rs 82.68 — some seven
        # hundredths of a per cent of a charge that is itself under a tenth of
        # a per cent of notional.
        net_fraction = self.costs.net_fraction(trade.notional, gross_fraction)

        del self._open[trade.instrument]
        self._trades_today[trade.instrument] = (
            self._trades_today.get(trade.instrument, 0) + 1
        )
        if self.cooldown_minutes > 0:
            self._cooldown[trade.instrument] = at + exact_timedelta(
                self.cooldown_minutes, ONE_MINUTE, name="cooldown_minutes"
            )

        return SimulatedTrade(
            instrument=trade.instrument,
            direction=trade.direction,
            session=trade.session,
            rules=trade.rules,
            score=trade.score,
            signal_time=trade.signal_time,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            notional=trade.notional,
            target_price=trade.target_price,
            stop_price=trade.stop_price,
            exit_time=at,
            exit_price=exit_price,
            exit_reason=reason,
            cost=trade.cost,
            gross_fraction=gross_fraction,
            net_fraction=net_fraction,
            net_rupees=trade.notional * net_fraction,
            max_favourable_fraction=trade.max_favourable_fraction,
            max_adverse_fraction=trade.max_adverse_fraction,
            bars_held=trade.bars_held,
            ambiguous_exit=trade.ambiguous_exit,
        )

    def reset_session(self) -> tuple[PendingEntry, ...]:
        """Forget what belonged to the day that just ended.

        ``trades_today`` means today. Cooldowns are cleared with it: an
        overnight gap is longer than any cooldown this system would configure,
        and carrying one across a boundary would suppress a name on the next
        morning's open for a trade that closed the previous afternoon.

        **Queued entries go too, and they are why this returns anything.** A
        pending entry is a decision taken on one session's information, waiting
        for a price; the price it was waiting for stops existing when the
        session does. Left in place it fills against the next morning's first
        bar instead, and the record that comes out of that is worse than a
        late fill. ``enter`` is passed ``fill_at``, which is yesterday's clock,
        while ``_price_at`` hands a fill due before a bar started that bar's
        open — so the trade is booked to yesterday's session, stamped with
        yesterday's entry time, at a price that had not printed then, and exits
        on a day it is not recorded as belonging to. The same stale entry keeps
        consuming a book slot and keeps ``is_committed`` true for its name, so
        it suppresses that name on the new day as well.

        Handed back rather than discarded because dropping them silently would
        trade one wrong number for another: the run would report fewer unfilled
        entries than it had. The caller counts these exactly as it counts the
        ones still queued when the tape runs out — they are the same event, a
        decision the tape never gave a price to.
        """
        dropped = tuple(self._pending.values())
        self._pending.clear()
        self._trades_today.clear()
        self._cooldown.clear()
        return dropped


__all__ = ["PendingEntry", "ReplayPortfolio", "stop_price_for"]
