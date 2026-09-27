"""Turnover is a statement of what passed through the account, not an estimate.

``ReplayResult.turnover`` says it is "the figure a broker's own statement can be
checked against", and that sentence is the whole specification. A statement
prices the opening leg at the price it opened and the closing leg at the price
it closed. Two legs of one round trip are the same number of shares at two
different prices, and the gap between them is the trade.

**Why it is easy to get wrong here.** The cost model charges both legs on the
entry notional, deliberately, and records why: ``round_trip`` is called when a
position opens, before the exit price exists, and ``required_gross_fraction`` is
*solving for* the move, so the exit notional it would need depends on the answer
it is computing. Neither reason survives into this object. A closed trade knows
both prices exactly. Carrying the cost model's simplification across into a
reported total is how the run comes to claim a turnover no statement would show
-- and turnover is a denominator, so everything expressed against it inherits
the error.

**Two claims.** The first is arithmetic: the total equals what the legs
transacted, reconstructed from prices and share counts. The second is what makes
the first mean anything -- that the closing leg moves with the trade, up for a
long that gained and down for a short that did, which is the direction a doubled
entry has no way to express.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.broker import Instrument
from ai_trader.market import INDIA_TIMEZONE, Candle
from ai_trader.replay import (
    FillModel,
    ReplayConfig,
    ReplayEngine,
    ReplayResult,
)
from ai_trader.scanner import Direction
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

_CONFIG = ReplayConfig(
    universe=_UNIVERSE,
    fill=FillModel(
        latency_seconds=Decimal(60),
        half_spread_fraction=Decimal("0.0002"),
        slippage_fraction=Decimal("0.0001"),
    ),
    strategy=StrategyConfig(
        exit_policy=FixedAtrStop(Decimal("1.5")),
        max_open_positions=len(_UNIVERSE),
    ),
)
"""A slot per name, which is as much book as this universe can use.

Not a claim that the position limit never binds -- with three names and three
slots it binds often, whenever all three are already in positions. It is simply
the largest book the tape can fill, and so the setting that produces the most
round trips to sum over. The trade count is asserted rather than assumed, since
a run with no trades satisfies every assertion below by summing nothing.
"""


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


def _result() -> ReplayResult:
    return ReplayEngine(_CONFIG).run(_tape())


def test_turnover_prices_each_leg_at_the_price_that_leg_traded() -> None:
    """The total equals both legs, reconstructed from prices and share counts.

    Deliberately not restated as ``notional + exit_notional``, which is the
    property's own expression and would assert nothing. Going back to
    ``entry_price`` and ``quantity`` also pins the thing that makes the two
    readings comparable at all: ``notional`` is the *entry* leg, priced at the
    price the entry filled.
    """
    result = _result()
    trades = result.trades

    assert len(trades) > 10, "a run with no round trips proves nothing here"

    statement = sum(
        ((trade.entry_price + trade.exit_price) * trade.quantity for trade in trades),
        Decimal(0),
    )
    assert result.turnover == statement


def test_the_closing_leg_is_not_the_opening_leg_counted_twice() -> None:
    """What the legs differ by is what the price did, and it is not nothing.

    The non-vacuity is load-bearing rather than tidy. Every trade here closes
    somewhere, and if every one happened to close where it opened then twice the
    entry would be the right answer and the inequality below would hold for a
    reason unrelated to the fix. So the drift is measured first and required to
    be non-zero; the inequality then follows from it rather than from luck.
    """
    result = _result()
    trades = result.trades

    doubled_entry = sum((trade.notional * 2 for trade in trades), Decimal(0))
    drift = sum(
        ((trade.exit_price - trade.entry_price) * trade.quantity for trade in trades),
        Decimal(0),
    )

    assert drift != 0, "every trade closed where it opened; nothing to distinguish"
    assert result.turnover == doubled_entry + drift
    assert result.turnover != doubled_entry


def test_the_closing_leg_moves_with_the_trade() -> None:
    """The direction claim: which way a leg moved says which way the trade went.

    This is what a doubled entry cannot express at all, and it is the sense in
    which turnover is a record rather than an approximation. A long that gained
    sold for more than it paid; a short that gained bought back for less than it
    sold. Read as inequalities on prices, so nothing here depends on rounding.
    """
    result = _result()

    winners = 0
    for trade in result.trades:
        if trade.gross_fraction == 0:
            continue
        gained = trade.gross_fraction > 0
        went_up = trade.exit_notional > trade.notional
        expected_up = gained if trade.direction is Direction.LONG else not gained
        assert went_up is expected_up, (
            f"{trade.instrument.trading_symbol} {trade.direction.value} closed at "
            f"{trade.exit_price} against an entry of {trade.entry_price}, but its "
            f"gross fraction is {trade.gross_fraction}."
        )
        winners += 1 if gained else 0

    # Both sides of the branch above have to occur, or half of it is untested.
    assert winners > 0
    assert winners < sum(1 for trade in result.trades if trade.gross_fraction != 0)
