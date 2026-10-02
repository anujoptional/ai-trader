"""One object that says what the strategy is, so two runs cannot disagree.

**The problem this exists to solve.** ``target_notional`` could be set in three
places, ``costs`` in four, and ``rules`` in two, and nothing checked that they
matched. The concrete failure is not hypothetical: a feasibility screen
configured at a one-lakh clip prices the round-trip hurdle at about 0.083%,
while a sizer configured at twenty thousand really pays about 0.272%. A scanner
wired that way shows the AI names it has screened as affordable and then takes
them at a size that cannot pay for them, three times over. The report would
blame the strategy. The cause would be two config sites disagreeing.

**So the fork happens after the strategy, not before it.** Replay, shadow and
live all build their scanner from one ``StrategyConfig`` by calling one
constructor -- not by building equal configs, which is a property somebody has
to keep true, but by calling the same code, which is a property that cannot
come apart. ``ReplayConfig`` adds a ``FillModel``, a universe and a date range,
and those really are replay-only: a fill model exists because replay has to
guess what live trading observes.

**Every field has a named default, and that is the safety argument.** A required
argument looks stricter and is in fact weaker for this purpose, because a
required argument can be handed two different values at two call sites while a
shared constant cannot drift. The defaults are conventions, not measurements,
and each one says so at its definition. The point of naming them is that the
number appears once, in a place that explains itself, instead of appearing
twice in two callers' keyword arguments.

**Derived, not restated.** ``gross_target_fraction`` is the term the strategy
was described in -- "sell 0.2% above the buy" -- and the net margin the sizer
and the screen both need is computed from it exactly once, here, and handed to
both. Neither can be given a different one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_trader.clock import ONE_MINUTE, exact_timedelta
from ai_trader.costs import (
    FIXED_CLIP_NOTIONAL,
    GROWW_INTRADAY_EQUITY,
    NSE_EQUITY_TICK,
    STATED_GROSS_TARGET,
    ZERODHA_INTRADAY_EQUITY,
    CostModel,
    SizingPolicy,
)
from ai_trader.scanner import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RULES,
    DEFAULT_SCORING,
    FeasibilityPolicy,
    Rule,
    Scanner,
    ScannerConfig,
    ScoringConfig,
)
from ai_trader.scanner.opportunity import TargetScoreConfig
from ai_trader.strategy.exits import (
    DEFAULT_EXIT_POLICY,
    ChandelierStop,
    ExitPolicy,
    FixedAtrStop,
)

DEFAULT_MAX_ATR_MULTIPLE = Decimal(3)
"""How many one-minute ATRs the required move may be before a name is dropped.

The screen compares the required gross move (fees plus margin, approximately
0.2% at the stated clip), not just the approximately 0.083% fee component.
Three one-minute ATRs is a distance, not a prediction of a three-minute holding
period. The multiple remains an uncalibrated research assumption.

``FeasibilityPolicy`` deliberately gives this field no default of its own, on
the grounds that a caller must state the assumption. That is still right for a
caller reaching for the screen directly. Stating it *once here* is the stronger
version of the same rule: the assumption is named, explained, and impossible to
set differently in replay than in live.
"""

DEFAULT_MAX_OPEN_POSITIONS = 3
"""Held positions the book will carry at once.

Smaller than the five-candidate budget on purpose. If the book could hold every
candidate the scanner returns, the ranking would never be consulted and the
score would be decoration; making capacity the binding constraint is what turns
rank into a decision the replay can be asked about. Three at a one-lakh clip is
about three lakh of gross intraday exposure.
"""

DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN = Decimal(360)
"""15:15 -- six hours after the 09:15 open, fifteen minutes before the close.

Intraday positions are closed by the broker if the account does not close them
first, and a broker's own auto square-off runs somewhere in the last part of the
session at a price the account does not choose. Being early is therefore the
safe side of the only question this number really asks. It is a parameter to
confirm against the broker's published cutoff, not a researched quantity, and it
is set here rather than left unset because ``None`` -- running to the last bar --
is the one setting that is definitely wrong for an intraday strategy.

When it is set, entries stop at the same moment as well, so that the last
minutes of a session are not spent opening positions that will be closed
minutes later at market.
"""


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """What the strategy is: size, costs, rules, screen, exits, book limits.

    Everything downstream is derived from this and nothing downstream may state
    any of it again. A ``ReplayResult`` cites the instance that produced it, so
    the run is reconstructible in the sense Section 7.2 requires.

    ``screen_feasibility`` exists so that the sweep can measure what the cost
    screen costs. Turning it off shows the scanner names it believes cannot pay
    for themselves, which is not a way to trade but is the only way to find out
    whether the screen is rejecting names that would have worked.
    """

    target_notional: Decimal = FIXED_CLIP_NOTIONAL
    gross_target_fraction: Decimal = STATED_GROSS_TARGET
    costs: CostModel = GROWW_INTRADAY_EQUITY
    tick_size: Decimal = NSE_EQUITY_TICK

    rules: Sequence[Rule] = DEFAULT_RULES
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    earliest_minutes_since_open: Decimal | None = None
    latest_minutes_since_open: Decimal | None = None

    screen_feasibility: bool = True
    max_atr_multiple: Decimal = DEFAULT_MAX_ATR_MULTIPLE
    min_minutes_remaining: Decimal | None = None

    exit_policy: ExitPolicy = DEFAULT_EXIT_POLICY
    max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS
    cooldown_minutes: Decimal = Decimal(0)
    square_off_minutes_since_open: Decimal = DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN
    scoring: ScoringConfig = DEFAULT_SCORING
    target_score: TargetScoreConfig | None = None

    def __post_init__(self) -> None:
        if self.target_score is not None and (
            self.target_score.target_fraction != self.gross_target_fraction
            or self.target_score.square_off_minutes
            != self.square_off_minutes_since_open
        ):
            raise ValueError(
                "Target-time scoring must share the strategy target and cutoff"
            )
        builtins = {type(rule) for rule in DEFAULT_RULES}
        configured: list[Rule] = []
        for rule in self.rules:
            if type(rule) in builtins:
                rule = type(rule)(self.scoring)
            configured.append(rule)
        object.__setattr__(self, "rules", tuple(configured))
        if self.max_candidates <= 0:
            raise ValueError(
                f"max_candidates must be positive, got {self.max_candidates}"
            )
        if self.max_open_positions <= 0:
            raise ValueError(
                f"max_open_positions must be positive, got {self.max_open_positions}"
            )
        if self.cooldown_minutes < 0:
            raise ValueError(
                f"cooldown_minutes cannot be negative, got {self.cooldown_minutes}"
            )
        if self.square_off_minutes_since_open <= 0:
            raise ValueError(
                "square_off_minutes_since_open must be positive, got "
                f"{self.square_off_minutes_since_open}"
            )
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        # Same discard-the-result move as the sizer below, for the same reason.
        # A cooldown that is not a whole number of microseconds cannot be the
        # duration the run reports, and this object is the one both paths read,
        # so refusing it here refuses it for the live path too rather than only
        # for whichever replay happens to convert it first.
        exact_timedelta(self.cooldown_minutes, ONE_MINUTE, name="cooldown_minutes")
        # Build the sizer now and discard it. ``from_gross_target`` refuses a
        # target its own costs would consume, and that refusal belongs at the
        # moment the configuration is written rather than several steps into a
        # run that has already fetched a year of candles.
        self.sizing_policy()

    @property
    def net_margin_fraction(self) -> Decimal:
        """What the strategy keeps, after charges, on a clip-sized round trip.

        Derived from ``gross_target_fraction`` rather than stated beside it.
        Stating both would allow a pair that does not satisfy its own arithmetic,
        and would leave no answer to the question of which one the strategy
        actually meant.
        """
        return self.sizing_policy().net_margin_fraction

    def sizing_policy(self) -> SizingPolicy:
        """The sizer the book uses to buy shares and place targets."""
        return SizingPolicy.from_gross_target(
            target_notional=self.target_notional,
            gross_target_fraction=self.gross_target_fraction,
            costs=self.costs,
            tick_size=self.tick_size,
        )

    def feasibility_policy(self) -> FeasibilityPolicy | None:
        """The cost screen, sized by the same sizer the trade will use.

        ``net_margin_fraction`` is taken from ``sizing_policy`` rather than
        recomputed, which is the specific guarantee this module exists to give:
        the screen's hurdle and the trade's target are two readings of one
        number. ``FeasibilityPolicy.from_gross_target`` would derive the same
        value today; going through the sizer means it still would if either
        conversion were ever changed.

        Returns ``None`` when the screen is off, which is what ``ScannerConfig``
        already understands to mean "do not screen".
        """
        if not self.screen_feasibility:
            return None
        return FeasibilityPolicy(
            target_notional=self.target_notional,
            net_margin_fraction=self.sizing_policy().net_margin_fraction,
            max_atr_multiple=self.max_atr_multiple,
            costs=self.costs,
            min_minutes_remaining=self.min_minutes_remaining,
            square_off_minutes_since_open=self.square_off_minutes_since_open,
        )

    def scanner_config(self) -> ScannerConfig:
        return ScannerConfig(
            max_candidates=self.max_candidates,
            earliest_minutes_since_open=self.earliest_minutes_since_open,
            latest_minutes_since_open=self.latest_minutes_since_open,
            feasibility=self.feasibility_policy(),
        )

    def scanner(self) -> Scanner:
        """The scanner itself, so that every caller builds it the same way.

        This method is the load-bearing one. Replay and live could each build a
        ``ScannerConfig`` from the fields above and would almost certainly build
        equal ones -- but "almost certainly equal" is a property somebody has to
        keep true as the code changes. Calling one constructor is a property that
        cannot come apart, and it is what makes the claim in Section 7.1 --
        replay sees exactly what the AI sees -- checkable rather than aspirational.
        """
        return Scanner(
            self.scanner_config(), self.rules, target_score=self.target_score
        )

    def to_dict(self) -> dict[str, Any]:
        """A complete versioned configuration, with decimal values stored exactly."""
        builtins = {type(rule) for rule in DEFAULT_RULES}
        if any(type(rule) not in builtins for rule in self.rules):
            raise ValueError("Only registered built-in rules can be serialized")
        if type(self.exit_policy) not in (FixedAtrStop, ChandelierStop):
            raise ValueError("Only registered exit policies can be serialized")
        values: dict[str, Any] = {}
        version = 1
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "target_score":
                if value is not None:
                    values[item.name] = {
                        "transform_version": value.transform_version,
                        "coefficients": [
                            _decimal_text(coefficient)
                            for coefficient in value.coefficients
                        ],
                        "target_fraction": _decimal_text(value.target_fraction),
                        "decay_minutes": _decimal_text(value.decay_minutes),
                        "square_off_minutes": _decimal_text(value.square_off_minutes),
                    }
                    version = 2
                    if value.score_threshold:
                        values[item.name]["score_threshold"] = _decimal_text(
                            value.score_threshold
                        )
                        version = 3
            elif item.name == "rules":
                values[item.name] = [rule.name for rule in self.rules]
            elif item.name in ("scoring", "costs"):
                values[item.name] = {
                    member.name: _decimal_text(getattr(value, member.name))
                    for member in fields(value)
                }
            elif item.name == "exit_policy":
                values[item.name] = {
                    "kind": type(value).__name__,
                    "multiple": _decimal_text(value.multiple),
                }
            else:
                values[item.name] = (
                    _decimal_text(value) if isinstance(value, Decimal) else value
                )
        return {
            "schema_version": version,
            "strategy": values,
        }

    @classmethod
    def from_dict(cls, document: object) -> StrategyConfig:
        """Reject missing fields rather than silently inheriting newer defaults."""
        payload = _object_fields(document, {"schema_version", "strategy"})
        version = payload["schema_version"]
        if type(version) is not int or version not in (1, 2, 3):
            raise ValueError("Unsupported strategy schema_version")
        expected = {item.name for item in fields(cls)}
        if version == 1:
            expected.remove("target_score")
        values = _object_fields(payload["strategy"], expected)
        if version >= 2:
            target_fields = {
                "transform_version",
                "coefficients",
                "target_fraction",
                "decay_minutes",
                "square_off_minutes",
            }
            if version == 3:
                target_fields.add("score_threshold")
            target = _object_fields(
                values.pop("target_score"),
                target_fields,
            )
            if not isinstance(target["coefficients"], list):
                raise ValueError("Target-time coefficients must be an ordered list")
            values["target_score"] = TargetScoreConfig(
                coefficients=tuple(
                    _read_decimal(value) for value in target["coefficients"]
                ),
                target_fraction=_read_decimal(target["target_fraction"]),
                decay_minutes=_read_decimal(target["decay_minutes"]),
                square_off_minutes=_read_decimal(target["square_off_minutes"]),
                transform_version=target["transform_version"],
                score_threshold=_read_decimal(target["score_threshold"])
                if version == 3
                else Decimal(0),
            )
        scoring = _object_fields(
            values.pop("scoring"), {item.name for item in fields(ScoringConfig)}
        )
        values["scoring"] = ScoringConfig(
            **{name: _read_decimal(value) for name, value in scoring.items()}
        )
        costs = _object_fields(
            values.pop("costs"), {item.name for item in fields(CostModel)}
        )
        amounts = {name: _read_decimal(value) for name, value in costs.items()}
        if any(value < 0 for value in amounts.values()):
            raise ValueError("Cost amounts cannot be negative")
        values["costs"] = CostModel(**amounts)
        registry = {rule.name: type(rule) for rule in DEFAULT_RULES}
        names = values.pop("rules")
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) or name not in registry for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("rules must contain unique registered rule names")
        values["rules"] = tuple(registry[name]() for name in names)
        policy = _object_fields(values.pop("exit_policy"), {"kind", "multiple"})
        policies = {
            policy_type.__name__: policy_type
            for policy_type in (FixedAtrStop, ChandelierStop)
        }
        if not isinstance(policy["kind"], str) or policy["kind"] not in policies:
            raise ValueError("Unknown exit policy")
        values["exit_policy"] = policies[policy["kind"]](
            _read_decimal(policy["multiple"])
        )
        optional = {
            "earliest_minutes_since_open",
            "latest_minutes_since_open",
            "min_minutes_remaining",
        }
        for name in values:
            if name in ("rules", "scoring", "costs", "exit_policy", "target_score"):
                continue
            if name in ("max_candidates", "max_open_positions"):
                if type(values[name]) is not int:
                    raise ValueError(f"{name} must be an integer")
            elif name == "screen_feasibility":
                if type(values[name]) is not bool:
                    raise ValueError("screen_feasibility must be a boolean")
            elif values[name] is None and name in optional:
                continue
            else:
                values[name] = _read_decimal(values[name])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> StrategyConfig:
        """Load only the supplied strategy document; never environment settings."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        """Create an experiment configuration without overwriting an existing one."""
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)

    @property
    def fingerprint(self) -> str:
        """Stable identity of resolved settings, separate from the code version."""
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _costs_label(self) -> str:
        """Which schedule, by name where the name is one this system publishes.

        ``CostModel`` carries no name field, deliberately -- it is a bundle of
        rates, and two schedules that agree on every rate are the same schedule
        whatever either is called. So the label is recovered by comparison, and
        anything unrecognised is described by the only terms that actually vary
        between brokers: brokerage is commercial, the rest is statute or tariff.
        """
        if self.costs == GROWW_INTRADAY_EQUITY:
            return "Groww intraday equity"
        if self.costs == ZERODHA_INTRADAY_EQUITY:
            return "Zerodha intraday equity"
        return (
            f"custom, brokerage {self.costs.brokerage_fraction:.4%} "
            f"capped at Rs {self.costs.brokerage_cap}"
        )

    def describe(self) -> tuple[str, ...]:
        """The resolved configuration, one line each, for a report header.

        Resolved rather than declared. The net margin and the hurdle are the
        derived figures the run truly used, so a reader comparing two reports
        does not have to redo either conversion to see what differed -- which is
        the whole use of a header on a sweep.
        """
        sizing = self.sizing_policy()
        hurdle = self.gross_target_fraction - sizing.net_margin_fraction
        screen = (
            f"{self.max_atr_multiple}x one-minute ATR"
            if self.screen_feasibility
            else "off"
        )
        return (
            f"clip                 Rs {self.target_notional:,}",
            f"gross target         {self.gross_target_fraction:.3%}",
            f"cost hurdle          {hurdle:.3%} (derived)",
            f"net margin           {sizing.net_margin_fraction:.3%} (derived)",
            f"costs                {self._costs_label()}",
            f"rules                {len(self.rules)}, top {self.max_candidates}"
            if self.target_score is None
            else (
                f"score                signed target-time, "
                f"top {self.max_candidates} by magnitude, "
                f"min |score| {self.target_score.score_threshold}"
            ),
            f"cost screen          {screen}",
            f"exit                 {self.exit_policy.description}",
            f"book                 {self.max_open_positions} positions",
            f"square-off           {self.square_off_minutes_since_open} min after open",
        )


def _object_fields(value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Configuration must contain exactly: {sorted(expected)}")
    return dict(value)


def _read_decimal(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("Configuration decimals must be strings")
    parsed = Decimal(value)
    if not parsed.is_finite():
        raise ValueError("Configuration decimals must be finite")
    return parsed


def _decimal_text(value: Decimal) -> str:
    if not value:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


__all__ = [
    "DEFAULT_MAX_ATR_MULTIPLE",
    "DEFAULT_MAX_OPEN_POSITIONS",
    "DEFAULT_SQUARE_OFF_MINUTES_SINCE_OPEN",
    "StrategyConfig",
]
