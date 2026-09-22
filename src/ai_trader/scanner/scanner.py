"""The deterministic scanner: the first half of the safety sandwich.

One decision cycle in, one ``ScanResult`` out. The scanner reads feature
snapshots and the portfolio, runs every rule that has the inputs it needs, and
ranks what fired down to a fixed budget.

Its real authority is **omission**. Whatever this layer does not emit, the AI
never sees and therefore can never propose, so a silent failure here is not a
missed trade that shows up in the logs — it is a scanner that appears to be
working while showing the model an empty market. That is why every path that
declines is counted: an empty result with ``considered`` at four hundred and
``not_ready`` at zero is a quiet market, an empty result with ``not_ready`` at
four hundred is a broken feature engine, and an empty result with
``suppressed`` full of ``entries_blocked`` is the kill switch doing its job.
Without the tally all three print as ``candidates: 0``.

What this layer deliberately does not do:

- **Size, stop or target.** Those are the AI's proposals and the risk engine's
  decisions.
- **Resolve a name that is long by one rule and short by another.** Section 4.6
  names conflicting signals as exactly what the AI is for. Dropping the conflict
  here would hide the disagreement rather than resolve it.
- **Decide staleness.** The position manager publishes ``stale_instruments``;
  a second staleness judgement here would be free to disagree with the first.
- **Re-derive position limits.** Same reason, and it is a risk rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext

from ai_trader.broker import Instrument
from ai_trader.features import FEATURE_CONTEXT, FeatureSnapshot
from ai_trader.scanner.models import (
    Candidate,
    Direction,
    MarketContext,
    PortfolioState,
    ScanResult,
    SuppressionReason,
)
from ai_trader.scanner.rules import DEFAULT_RULES, Rule, available

DEFAULT_MAX_CANDIDATES = 5
"""How many candidates one cycle may hand to the AI.

Section 4.4 calls this the most consequential tuning knob in the system, and
five is the top of its suggested 3–5 starting range rather than a measured
optimum. Too low and the best setup of the day is ranked sixth and never seen;
too high and the model's attention is spread across noise. ``ScanResult.
truncated`` is the measurement that will eventually replace this guess.
"""


@dataclass(frozen=True, slots=True)
class ScannerConfig:
    """Tuning for one scanner.

    Both session-window bounds default to ``None``, meaning no restriction.
    Picking a cut-off — no entries in the first ten minutes, none in the last
    thirty — would be inventing a threshold, and section 7 is explicit that an
    invented threshold is not evidence. The knobs exist so replay can establish
    the edges; until it has, the honest default is to scan the whole session.

    Both bounds are compared against ``minutes_since_session_open``, which is
    the one session feature that survives a mid-session start: it is clock
    arithmetic against 09:15 rather than an aggregate over candles the engine
    never saw.
    """

    max_candidates: int = DEFAULT_MAX_CANDIDATES
    earliest_minutes_since_open: Decimal | None = None
    latest_minutes_since_open: Decimal | None = None

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError(
                f"max_candidates must be at least 1, got {self.max_candidates}"
            )


@dataclass(slots=True)
class _Accumulator:
    """One ``(instrument, direction)`` pair as rules pile onto it."""

    score: Decimal
    rules: list[str] = field(default_factory=list)
    evidence: dict[str, Decimal] = field(default_factory=dict)


class Scanner:
    """Runs a rule set over one cycle's snapshots.

    Stateless between cycles on purpose. Everything that persists — positions,
    cooldowns, trade counts — lives in ``PortfolioState`` and arrives as an
    argument, so scanning the same inputs twice gives the same answer and a
    replay is a genuine rerun rather than a different object's second opinion.
    """

    def __init__(
        self,
        config: ScannerConfig | None = None,
        rules: Sequence[Rule] = DEFAULT_RULES,
    ) -> None:
        self._config = config or ScannerConfig()
        self._rules = tuple(rules)

    @property
    def config(self) -> ScannerConfig:
        return self._config

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def scan(
        self,
        snapshots: Iterable[FeatureSnapshot],
        portfolio: PortfolioState,
        *,
        context: MarketContext | None = None,
    ) -> ScanResult:
        """Produce this cycle's ranked candidates and the tally behind them.

        ``context`` is the reserved ``MarketContext`` seam. It is accepted and
        passed to every rule today so that adding microstructure later is a
        change to the rules that want it rather than to every signature between
        here and them.
        """
        considered = 0
        not_ready = 0
        suppressed: dict[SuppressionReason, int] = {}
        accumulated: dict[tuple[Instrument, Direction], _Accumulator] = {}
        seen: dict[Instrument, FeatureSnapshot] = {}

        # The context covers rule evaluation and scoring both, so no rule has to
        # install it and none can forget to. Comparisons are exact regardless;
        # it is the divisions inside the scoring ramps that would otherwise
        # round differently here than in the engine that produced the inputs.
        with localcontext(FEATURE_CONTEXT):
            for snapshot in snapshots:
                instrument = snapshot.instrument
                if instrument in seen:
                    raise ValueError(
                        "one cycle carried two snapshots for "
                        f"{instrument.exchange}:{instrument.trading_symbol}"
                    )
                seen[instrument] = snapshot

                reason = self._suppression(snapshot, portfolio)
                if reason is not None:
                    suppressed[reason] = suppressed.get(reason, 0) + 1
                    continue

                evaluated = self._evaluate(snapshot, portfolio, context, accumulated)
                if evaluated:
                    considered += 1
                else:
                    not_ready += 1

            ranked = self._rank(accumulated, seen)

        budget = self._config.max_candidates
        return ScanResult(
            as_of=portfolio.as_of,
            candidates=ranked[:budget],
            considered=considered,
            not_ready=not_ready,
            suppressed=suppressed,
            truncated=max(0, len(ranked) - budget),
        )

    def _suppression(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
    ) -> SuppressionReason | None:
        """Portfolio suppression first, then the session window.

        The portfolio's reasons come first because they are the ones an operator
        needs to see: a kill-switched session reporting ``outside_window`` for
        every name would bury the fact that entries are blocked at all.
        """
        reason = portfolio.suppression(snapshot.instrument)
        if reason is not None:
            return reason
        if self._outside_window(snapshot):
            return SuppressionReason.OUTSIDE_WINDOW
        return None

    def _outside_window(self, snapshot: FeatureSnapshot) -> bool:
        earliest = self._config.earliest_minutes_since_open
        latest = self._config.latest_minutes_since_open
        if earliest is None and latest is None:
            return False
        minutes = available(snapshot, "minutes_since_session_open")
        if minutes is None:
            # A window was configured and the clock cannot be read. Failing
            # closed keeps a configured "no entries near the close" from
            # quietly becoming "entries at any time".
            return True
        if earliest is not None and minutes < earliest:
            return True
        return latest is not None and minutes > latest

    def _evaluate(
        self,
        snapshot: FeatureSnapshot,
        portfolio: PortfolioState,
        context: MarketContext | None,
        accumulated: dict[tuple[Instrument, Direction], _Accumulator],
    ) -> bool:
        """Run every runnable rule. Returns whether any rule could run at all.

        A snapshot no rule could read is counted as not-ready rather than as
        considered-and-unconvincing, because those are different failures: the
        first says the feature engine has not warmed up or volume is missing,
        the second says the market is quiet.
        """
        ran_any = False
        for rule in self._rules:
            if any(
                available(snapshot, name) is None for name in rule.required_features
            ):
                continue
            ran_any = True
            signal = rule.evaluate(snapshot, portfolio, context)
            if signal is None:
                continue

            key = (snapshot.instrument, signal.direction)
            entry = accumulated.get(key)
            if entry is None:
                entry = _Accumulator(score=signal.score)
                accumulated[key] = entry
            elif signal.score > entry.score:
                # Max rather than a sum: two rules agreeing is recorded in
                # ``rules`` for replay to weigh, and inventing a confirmation
                # bonus here would be a tuning constant with nothing behind it.
                entry.score = signal.score
            entry.rules.append(rule.name)
            # Safe to merge flat. Both rules read the same snapshot through the
            # same flags, so a key they share carries the same value and the
            # later write cannot change it.
            entry.evidence.update(signal.evidence)
        return ran_any

    def _rank(
        self,
        accumulated: dict[tuple[Instrument, Direction], _Accumulator],
        snapshots: dict[Instrument, FeatureSnapshot],
    ) -> tuple[Candidate, ...]:
        """Order candidates by a key that is total, so ties cannot drift.

        Score alone leaves ties, and a tie broken by dictionary order would make
        the budget cut depend on the order instruments happened to arrive in —
        the same inputs producing a different shortlist on a different day, and
        a replay that cannot reproduce the live run it is meant to explain.
        Symbol, exchange and direction settle every remaining tie because
        ``(instrument, direction)`` is already unique per cycle.
        """
        candidates = []
        for (instrument, direction), entry in accumulated.items():
            snapshot = snapshots[instrument]
            candidates.append(
                Candidate(
                    instrument=instrument,
                    direction=direction,
                    score=entry.score,
                    rules=tuple(sorted(entry.rules)),
                    as_of=snapshot.candle_end_time,
                    reference_price=snapshot.close,
                    evidence=entry.evidence,
                )
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.instrument.trading_symbol,
                candidate.instrument.exchange,
                candidate.direction.value,
            )
        )
        return tuple(candidates)


__all__ = ["DEFAULT_MAX_CANDIDATES", "Scanner", "ScannerConfig"]
