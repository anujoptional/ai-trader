"""What the strategy is, stated once, so replay and live cannot disagree.

This package holds the decisions that are the *same* in a backtest and in a live
session: how large a clip is, what a round trip costs, which rules run, what the
cost screen demands, where the stop sits, how many positions the book carries,
when the session is squared off. Everything downstream derives from
``StrategyConfig`` and nothing downstream may state any of it again.

**The dependency direction is the point.** ``strategy`` depends on ``scanner``
and ``costs`` and on nothing else; ``replay`` depends on ``strategy``. So a
replay engine cannot configure a scanner differently from the way a live session
would, because neither of them configures one at all -- both ask this package
for it. That is the mechanical form of the claim in Section 7.1 that the fork
between replay and live happens *after* the scanner: replay sees exactly the
candidate stream the AI would have seen, because the same constructor built it.

What is genuinely not here: a ``FillModel``, a universe, a date range, a broker
session. A fill model exists because replay has to guess what live trading
simply observes, so it belongs to replay and to nothing else.
"""

from ai_trader.strategy.config import (
    DEFAULT_MAX_ATR_MULTIPLE,
    DEFAULT_MAX_OPEN_POSITIONS,
    DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN,
    StrategyConfig,
)
from ai_trader.strategy.exits import (
    DEFAULT_EXIT_POLICY,
    DEFAULT_STOP_ATR_MULTIPLE,
    ChandelierStop,
    ExitPolicy,
    FixedAtrStop,
    stop_price_for,
)

__all__ = [
    "DEFAULT_EXIT_POLICY",
    "DEFAULT_MAX_ATR_MULTIPLE",
    "DEFAULT_MAX_OPEN_POSITIONS",
    "DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN",
    "DEFAULT_STOP_ATR_MULTIPLE",
    "ChandelierStop",
    "ExitPolicy",
    "FixedAtrStop",
    "StrategyConfig",
    "stop_price_for",
]
