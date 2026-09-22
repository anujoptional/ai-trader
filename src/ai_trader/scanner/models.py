"""Immutable artifacts the scanner consumes and produces.

Three things cross this boundary. ``PortfolioState`` comes down from the
position manager and says which names must not be scanned at all.
``MarketContext`` is reserved for microstructure and regime state and carries
nothing yet. ``Candidate`` goes up to the AI layer.

A ``Candidate`` is a **research hypothesis, not permission to trade**. Nothing
here proposes a size, a stop or a target: those belong to the AI layer as
proposals and to the risk engine as decisions. The scanner's whole authority is
to say "this name, this direction, this evidence, this rank" — and, by omission,
to decide what the AI is never shown.

Every mapping and set is copied and sealed on construction rather than merely
documented as read-only. A decision cycle that mutated the book halfway through
would let the scanner, the AI and risk each see a different portfolio within one
cycle, and the resulting bugs are timing-dependent and effectively
unreproducible. Sealing makes that impossible rather than discouraged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType

from ai_trader.broker import Instrument


class Direction(StrEnum):
    """Which way a hypothesis leans."""

    LONG = "LONG"
    SHORT = "SHORT"


class SuppressionReason(StrEnum):
    """Why a name was withheld from scanning entirely.

    Suppression is not rejection. A suppressed name is never evaluated, so it
    consumes no part of the candidate budget and no part of an LLM call. A name
    that *is* evaluated and produces nothing simply had no rule fire on it, and
    is counted separately.
    """

    ENTRIES_BLOCKED = "entries_blocked"
    STALE_FEED = "stale_feed"
    POSITION_LIMIT = "position_limit"
    COOLDOWN = "cooldown"
    OUTSIDE_WINDOW = "outside_window"


@dataclass(frozen=True, slots=True)
class Position:
    """One open position as the position manager sees it.

    ``quantity`` is signed — positive long, negative short — so a consumer never
    has to pair it with a separate side field to know which way the book leans.
    """

    instrument: Instrument
    quantity: int
    average_price: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """The book at one instant, published once per decision cycle.

    Built from real fills in live trading, simulated fills in shadow mode, and
    simulated fills over historical candles in replay. One object, one set of
    consumers, three sources — which is exactly what makes the three modes
    comparable.

    ``at_position_limit`` is carried as a precomputed set rather than derived
    here from ``open_positions`` against a configured maximum. The limit is a
    risk-configuration fact the position manager already holds, and re-deriving
    it in the scanner would be a second implementation of a risk rule, free to
    disagree with the first. The scanner is told which names are full; it does
    not work it out.

    ``as_of`` must be timezone-aware, because cooldowns are compared against it.
    A naive value raises when a cooldown is present, which is the right failure:
    the alternative to a loud comparison error is a cooldown that silently never
    fires and lets a name re-enter inside its own blackout.
    """

    as_of: datetime
    open_positions: Mapping[Instrument, Position] = field(default_factory=dict)
    at_position_limit: frozenset[Instrument] = frozenset()
    trades_today: Mapping[Instrument, int] = field(default_factory=dict)
    cooldown_until: Mapping[Instrument, datetime] = field(default_factory=dict)
    stale_instruments: frozenset[Instrument] = frozenset()
    new_entries_blocked: bool = False

    def __post_init__(self) -> None:
        # Copy first, then seal. Wrapping the caller's own dict would leave them
        # holding a live handle to something this object promises is a snapshot.
        object.__setattr__(
            self, "open_positions", MappingProxyType(dict(self.open_positions))
        )
        object.__setattr__(
            self, "trades_today", MappingProxyType(dict(self.trades_today))
        )
        object.__setattr__(
            self, "cooldown_until", MappingProxyType(dict(self.cooldown_until))
        )
        object.__setattr__(self, "at_position_limit", frozenset(self.at_position_limit))
        object.__setattr__(self, "stale_instruments", frozenset(self.stale_instruments))

    @classmethod
    def empty(cls, as_of: datetime) -> PortfolioState:
        """A book holding nothing, with nothing suppressed.

        The position manager does not exist yet. This exists so the scanner can
        be written, tested and replayed against its real signature now, rather
        than as a pure function of ``FeatureSnapshot`` that would have to be
        rewritten the moment suppression became possible.
        """
        return cls(as_of=as_of)

    def suppression(self, instrument: Instrument) -> SuppressionReason | None:
        """Why this name must not be scanned, or ``None`` if it may be.

        The first matching reason wins. The order is diagnostic precedence
        rather than semantics: a name can easily be both stale and in cooldown,
        and reporting the session-wide block before the per-name ones keeps a
        kill-switched cycle from reading as four hundred individually
        uninteresting suppressions.
        """
        if self.new_entries_blocked:
            return SuppressionReason.ENTRIES_BLOCKED
        if instrument in self.stale_instruments:
            return SuppressionReason.STALE_FEED
        if instrument in self.at_position_limit:
            return SuppressionReason.POSITION_LIMIT
        deadline = self.cooldown_until.get(instrument)
        if deadline is not None and deadline > self.as_of:
            return SuppressionReason.COOLDOWN
        return None


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Reserved per-cycle context. Deliberately carries nothing yet.

    The seam exists so that adding microstructure is not a rewrite of every rule
    signature; the fields do not, because they should be built when a rule
    actually needs them rather than speculatively. Three groups are reserved:

    - **Microstructure per symbol** — bid, ask, spread, top-of-book depth, order
      imbalance. A candidate whose theoretical edge is smaller than its spread
      is not a candidate.
    - **Index and regime state** — index trend and volatility.
    - **Staleness per symbol** — age of the last tick.

    When they arrive, every field is optional and explicitly unavailable when
    missing, exactly as ``FeatureSnapshot`` treats an unready feature. A rule
    that needs a spread is suppressed when the spread is unknown; it is never
    run against a guessed one.
    """

    as_of: datetime


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked hypothesis about one instrument in one direction.

    ``rules`` names every rule that fired, not just the strongest. Two
    independent rules agreeing on a direction is different evidence from one
    rule firing alone, and collapsing that to a single name would destroy the
    distinction before replay could measure whether it matters.

    ``evidence`` carries the exact ``Decimal`` values the rules read, so the
    journal and the AI prompt describe what the rule actually saw rather than
    re-reading the snapshot later and risking a different answer. Keys are
    feature names, so two rules reading ``adx14`` from one snapshot contribute
    the same value by construction and the merge cannot conflict.
    """

    instrument: Instrument
    direction: Direction
    score: Decimal
    rules: tuple[str, ...]
    as_of: datetime
    reference_price: Decimal
    evidence: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(self, "rules", tuple(self.rules))


@dataclass(frozen=True, slots=True)
class ScanResult:
    """What one decision cycle produced, and what it declined to produce.

    The counters are not decoration. A scanner can emit nothing for a whole
    session for three very different reasons — everything suppressed, nothing
    ready, or every rule evaluated and none convinced — and without the tally
    those are indistinguishable from the outside. ``truncated`` is the tuning
    signal for the candidate budget: a cycle that truncates is one where the
    budget, not the market, chose what the AI saw.
    """

    as_of: datetime
    candidates: tuple[Candidate, ...]
    considered: int = 0
    not_ready: int = 0
    suppressed: Mapping[SuppressionReason, int] = field(default_factory=dict)
    truncated: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "suppressed", MappingProxyType(dict(self.suppressed)))


__all__ = [
    "Candidate",
    "Direction",
    "MarketContext",
    "PortfolioState",
    "Position",
    "ScanResult",
    "SuppressionReason",
]
