"""Replay: run the built pipeline over historical candles and score the result.

``Candle -> FeatureEngine -> Scanner -> Candidate -> simulated outcome``, which
is the pipeline Section 4.5 specifies, with a book in the middle so that the
answer is about trading rather than about signalling.

**What this layer is for.** Every threshold in ``scanner/rules.py`` is a
convention chosen so the layer could be built, and Section 7.2 is explicit that
an invented threshold is not evidence. Replay is the machine that turns those
conventions into measurements: it reports **net expectancy per round trip** and
**round trips per session**, net of the full cost model, over a universe that is
recorded rather than assumed.

**What this layer is not for.** Section 7.1 draws the line: replay establishes
whether the *candidates* are worth anything; shadow trading establishes whether
the *AI* picks the right ones. A good replay number is not evidence about the
AI, because the AI is not in the loop here.

**Read the three modules in the order a trade meets them.** ``models`` holds the
fill assumptions and the trade record. ``portfolio`` holds the book. ``engine``
holds the loop, and its docstring carries the no-lookahead argument and the two
places this layer models rather than observes.
"""

from ai_trader.replay.engine import (
    ReplayConfig,
    ReplayCycle,
    ReplayEngine,
    minutes_since_open,
)
from ai_trader.replay.models import (
    FRICTIONLESS,
    ExitReason,
    FillModel,
    ReplayResult,
    SimulatedTrade,
)
from ai_trader.replay.portfolio import PendingEntry, ReplayPortfolio, stop_price_for

__all__ = [
    "FRICTIONLESS",
    "ExitReason",
    "FillModel",
    "PendingEntry",
    "ReplayConfig",
    "ReplayCycle",
    "ReplayEngine",
    "ReplayPortfolio",
    "ReplayResult",
    "SimulatedTrade",
    "minutes_since_open",
    "stop_price_for",
]
