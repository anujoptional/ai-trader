"""Deterministic candidate generation: what the AI is allowed to see.

This package is the first deterministic half of the safety sandwich. It turns
feature snapshots into a short, ranked list of research hypotheses, and by
omission decides everything the AI layer will never be asked about.

Nothing here reaches the broker or the tick stream. The dependency runs one way
— ``scanner`` on ``features`` on ``market`` on ``broker`` — so a scanner rule
cannot accidentally read a price the feature engine has not finished
processing.
"""

from ai_trader.scanner.models import (
    Candidate,
    Direction,
    MarketContext,
    PortfolioState,
    Position,
    ScanResult,
    SuppressionReason,
)
from ai_trader.scanner.rules import (
    DEFAULT_RULES,
    BreakoutRule,
    MeanReversionRule,
    OpeningRangeBreakoutRule,
    Rule,
    RuleSignal,
    TrendContinuationRule,
    VwapReversionRule,
    available,
)
from ai_trader.scanner.scanner import (
    DEFAULT_MAX_CANDIDATES,
    Scanner,
    ScannerConfig,
)

__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_RULES",
    "BreakoutRule",
    "Candidate",
    "Direction",
    "MarketContext",
    "MeanReversionRule",
    "OpeningRangeBreakoutRule",
    "PortfolioState",
    "Position",
    "Rule",
    "RuleSignal",
    "ScanResult",
    "Scanner",
    "ScannerConfig",
    "SuppressionReason",
    "TrendContinuationRule",
    "VwapReversionRule",
    "available",
]
