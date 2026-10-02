from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.features import FeatureEngine
from ai_trader.market import Candle
from ai_trader.replay import ReplayResult
from ai_trader.replay.models import FRICTIONLESS, ExitReason
from ai_trader.scanner import Direction, PortfolioState, Scanner
from ai_trader.scanner.opportunity import (
    COMPONENT_NAMES,
    TargetScoreConfig,
    centered_sigmoid,
)
from ai_trader.strategy import StrategyConfig
from scripts.research_signed_score import (
    Observation,
    TargetOutcome,
    _trade_summary,
    fit_coefficients,
    predict,
    rank_correlation,
    select_threshold,
    target_outcome,
    threshold_metrics,
)
from scripts.research_tp_policy import (
    PriorFrame,
    choose_policy,
    fit_tp,
    isolated_trade,
    precision_summary,
    previous_sessions,
    signed_tp_confidence,
    thresholds,
    tp_inputs,
)

_OPEN = datetime(2026, 9, 1, 10, 0, tzinfo=INDIA_TIMEZONE)
_INSTRUMENT = Instrument("NSE", "RELIANCE")


def _bar(minute: int, *, high: str = "100", low: str = "100") -> Candle:
    start = _OPEN + timedelta(minutes=minute)
    return Candle(
        instrument=_INSTRUMENT,
        start_time=start,
        end_time=start + timedelta(minutes=1),
        open=Decimal(100),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(100),
        volume=1000,
    )


@pytest.mark.parametrize("entry", ["100", "1300.1", "1327.9"])
@pytest.mark.parametrize("fraction", ["0.0015", "0.002", "0.003"])
def test_symmetric_research_barriers_preserve_ticks_size_and_costs(
    entry: str, fraction: str
) -> None:
    from ai_trader.costs import SizingPolicy
    from scripts.research_horizon_probability import SymmetricSizing, SymmetricStop

    price, distance = Decimal(entry), Decimal(fraction)
    base = SizingPolicy.from_gross_target(
        target_notional=Decimal(100_000),
        gross_target_fraction=distance,
        tick_size=Decimal("0.1"),
    )
    sizing = SymmetricSizing(
        target_notional=base.target_notional,
        net_margin_fraction=base.net_margin_fraction,
        costs=base.costs,
        tick_size=base.tick_size,
        barrier_fraction=distance,
    )
    estimate = sizing.estimate(price)
    assert estimate.quantity == base.estimate(price).quantity
    assert estimate.cost == base.estimate(price).cost
    assert estimate.long_exit_price - price == price - estimate.short_exit_price
    assert distance <= (estimate.long_exit_price - price) / price
    assert (
        estimate.long_exit_price - price
    ) / price < distance + base.tick_size / price
    for direction in Direction:
        stop = SymmetricStop(distance).stop_price(
            entry_price=price,
            direction=direction,
            atr_fraction=Decimal("0.9"),
            favourable_fraction=Decimal("0.8"),
            tick=base.tick_size,
        )
        target = (
            estimate.long_exit_price
            if direction is Direction.LONG
            else estimate.short_exit_price
        )
        assert target % base.tick_size == stop % base.tick_size == 0
        assert abs(stop - price) == abs(target - price)


@pytest.mark.parametrize("fraction", ["0", "-0.1", "1", "NaN", "Infinity"])
def test_symmetric_research_barriers_reject_invalid_fractions(fraction: str) -> None:
    from scripts.research_horizon_probability import SymmetricStop, symmetric_prices

    with pytest.raises(ValueError, match="fraction"):
        symmetric_prices(Decimal(100), Decimal(fraction), Decimal("0.1"))
    with pytest.raises(ValueError, match="fraction"):
        SymmetricStop(Decimal(fraction))


@pytest.mark.parametrize("direction", list(Direction))
def test_horizon_labels_use_symmetric_executed_entry_and_expire(
    direction: Direction,
) -> None:
    from ai_trader.replay.models import FillModel
    from scripts.research_horizon_probability import horizon_trade

    fill = FillModel(Decimal(1), Decimal(0), Decimal(0))
    kwargs = {"high": "100.2"} if direction is Direction.LONG else {"low": "99.8"}
    strategy = replace(StrategyConfig(), tick_size=Decimal("0.1"))
    later = [_bar(index, **(kwargs if index == 7 else {})) for index in range(1, 9)]
    expired = horizon_trade(
        _bar(0), later, direction, strategy, Decimal("0.002"), Decimal("0.1"), fill=fill
    )
    assert expired.exit_reason is ExitReason.HORIZON
    assert expired.holding_minutes == 6
    assert expired.entry_price == Decimal(100)
    assert abs(expired.target_price - expired.entry_price) == abs(
        expired.stop_price - expired.entry_price
    )
    later[5] = _bar(6, **kwargs)
    boundary_hit = horizon_trade(
        _bar(0), later, direction, strategy, Decimal("0.002"), Decimal("0.1"), fill=fill
    )
    assert boundary_hit.exit_reason is ExitReason.TARGET
    assert boundary_hit.exit_time <= boundary_hit.entry_time + timedelta(minutes=6)
    longer = horizon_trade(
        _bar(0),
        later,
        direction,
        strategy,
        Decimal("0.002"),
        Decimal("0.12"),
        fill=fill,
    )
    assert longer == boundary_hit


def test_horizon_labels_fail_closed_and_do_not_credit_stop_first_or_entry_bar() -> None:
    from ai_trader.replay.models import FillModel
    from scripts.research_horizon_probability import horizon_trade

    strategy = replace(StrategyConfig(), tick_size=Decimal("0.1"))
    fill = FillModel(Decimal(1), Decimal(0), Decimal(0))
    future = [_bar(index) for index in range(1, 8)]
    future[0] = _bar(1, high="110", low="90")
    arguments = (Direction.LONG, strategy, Decimal("0.002"), Decimal("0.1"))
    expired = horizon_trade(_bar(0), future, *arguments, fill=fill)
    assert expired.exit_reason is ExitReason.HORIZON
    future[1] = _bar(2, low="99.8")
    future[2] = _bar(3, high="101")
    stopped = horizon_trade(_bar(0), future, *arguments, fill=fill)
    assert stopped.exit_reason is ExitReason.STOP
    future[1] = _bar(2, low="99.8", high="100.2")
    ambiguous = horizon_trade(_bar(0), future, *arguments, fill=fill)
    assert ambiguous.exit_reason is ExitReason.STOP and ambiguous.ambiguous_exit
    with pytest.raises(ValueError, match="missing minutes"):
        horizon_trade(_bar(0), future[1:], *arguments, fill=fill)
    with pytest.raises(ValueError, match="completed minutes"):
        horizon_trade(_bar(0), future[:2], *arguments, fill=fill)
    assert horizon_trade(_bar(310), (), *arguments, fill=fill) is None
    with pytest.raises(ValueError, match="TARGET or STOP"):
        FillModel(Decimal(1), Decimal(0), Decimal(0), ExitReason.HORIZON)


def test_weekly_context_uses_only_five_completed_earlier_sessions() -> None:
    from scripts.research_horizon_probability import horizon_inputs, weekly_context

    days = [date(2026, 7, number) for number in (1, 2, 3, 6, 7, 8, 9)]
    tape = []
    for number, day in enumerate(days):
        opening = datetime.combine(day, datetime.min.time(), INDIA_TIMEZONE).replace(
            hour=9, minute=15
        )
        price = Decimal(100 + number)
        for minute in range(360):
            tape.append(
                replace(
                    _bar(0),
                    start_time=opening + timedelta(minutes=minute),
                    end_time=opening + timedelta(minutes=minute + 1),
                    open=price,
                    high=price + 1,
                    low=price - 1,
                    close=price,
                )
            )
    contexts = weekly_context(tape)
    assert all(not contexts[day] for day in days[:5])
    assert [frame.session for frame in contexts[days[5]]] == days[:5]
    changed = [
        replace(bar, high=bar.high + 100) if bar.start_time.date() >= days[5] else bar
        for bar in tape
    ]
    assert weekly_context(changed)[days[5]] == contexts[days[5]]
    assert weekly_context(tape[: 5 * 360 + 1])[days[5]] == contexts[days[5]]
    broken = [bar for index, bar in enumerate(tape) if index != 360]
    assert not weekly_context(broken)[days[5]]
    snapshot = _features()
    signal = _bar(80)
    long = horizon_inputs(
        snapshot, signal, Direction.LONG, contexts[days[5]], Decimal("0.002")
    )
    short = horizon_inputs(
        snapshot, signal, Direction.SHORT, contexts[days[5]], Decimal("0.002")
    )
    assert len(long["intraday_week"]) == 18
    assert long["intraday_week"][-6] == -short["intraday_week"][-6]
    assert long["intraday_week"][-4:-2] == short["intraday_week"][-4:-2]
    with pytest.raises(ValueError, match="repeat"):
        horizon_inputs(
            snapshot,
            signal,
            Direction.LONG,
            [contexts[days[5]][0]] * 5,
            Decimal("0.002"),
        )


def test_joint_event_probabilities_remain_monotone_after_calibration() -> None:
    from scripts.research_horizon_probability import MassCalibration, cumulative_tp

    mass = (0.05, 0.1, 0.2, 0.15, 0.1, 0.4, 0.0)
    assert cumulative_tp(mass) == pytest.approx((0.05, 0.15, 0.35))
    identity = MassCalibration(1, (0.0,) * len(mass))
    assert identity.apply(mass) == pytest.approx(mass, abs=1e-11)
    mapped = MassCalibration(0.7, (0.2, -0.3, 0.4, 0.1, -0.4, 0.2, 0.0)).apply(mass)
    probabilities = cumulative_tp(mapped)
    assert probabilities == tuple(sorted(probabilities))
    assert 0 <= probabilities[0] <= probabilities[-1] <= 1
    with pytest.raises(ValueError, match="sum to one"):
        cumulative_tp((0.1, 0.1, 0.1))


def test_mass_calibration_cannot_fit_future_labels() -> None:
    pytest.importorskip("scipy")
    from scripts.research_horizon_probability import fit_mass_calibration

    today = date(2026, 8, 10)
    mass = (0.6, 0.1, 0.3)
    earlier = [(today, mass, 0)] * 2 + [(today, mass, 1)] * 3 + [(today, mass, 2)] * 5
    fitted = fit_mass_calibration(earlier, through=today)
    future = [(date(2026, 9, 1), mass, 0)] * 100
    assert fitted == fit_mass_calibration([*earlier, *future], through=today)
    assert fitted.apply(mass)[0] < mass[0]


@pytest.mark.parametrize(
    "specification",
    [
        {"kind": "logistic", "C": 1.0},
        {"kind": "trees", "max_depth": 2},
        {"kind": "mlp", "width": 8, "alpha": 10.0},
    ],
)
def test_time_models_learn_a_signal_and_ignore_future_training_rows(
    specification,
) -> None:
    pytest.importorskip("sklearn")
    from scripts.research_horizon_probability import fit_time_model, predict_mass

    inputs = {
        (_OPEN + timedelta(minutes=index), Direction.LONG): (
            float(index % 3 - 1),
            float(index % 2),
        )
        for index in range(240)
    }
    outcomes = {key: index % 3 for index, key in enumerate(inputs)}
    fitted = fit_time_model(inputs, outcomes, specification, [_OPEN.date()], stride=1)
    predictions = predict_mass(fitted, inputs, 3)
    assert fitted["training_actions"] == 240
    assert all(
        max(range(3), key=lambda label: mass[label]) == outcomes[key]
        for key, mass in predictions.items()
    )
    future = (_OPEN + timedelta(days=1), Direction.LONG)
    refitted = fit_time_model(
        {**inputs, future: (1000.0, -1000.0)},
        {**outcomes, future: 2},
        specification,
        [_OPEN.date()],
        stride=1,
    )
    assert predict_mass(refitted, inputs, 3) == predictions


def test_event_selection_uses_validation_skill_not_test_hits_or_event_rarity() -> None:
    from scripts.research_horizon_probability import select_event

    policies = {
        "rare": {
            "validation_skill": {"brier": -0.1, "log_loss": 0.0},
            "horizon_hours": "0.1",
            "barrier_fraction": "0.004",
            "test_tp_rate": 1,
        },
        "useful": {
            "validation_skill": {"brier": 0.03, "log_loss": 0.02},
            "horizon_hours": "0.5",
            "barrier_fraction": "0.002",
            "test_tp_rate": 0,
        },
    }
    assert select_event(policies) == "useful"
    policies["useful"]["test_tp_rate"] = 0.9
    assert select_event(policies) == "useful"


def test_time_model_artifact_round_trip_checks_hash_before_loading(tmp_path) -> None:
    pytest.importorskip("sklearn")
    from scripts.research_horizon_probability import (
        fit_time_model,
        load_model,
        predict_mass,
        save_model,
    )

    inputs = {
        (_OPEN + timedelta(minutes=index), Direction.LONG): (float(index % 3),)
        for index in range(90)
    }
    outcomes = {key: index % 3 for index, key in enumerate(inputs)}
    fitted = fit_time_model(
        inputs, outcomes, {"kind": "logistic", "C": 1}, [_OPEN.date()], stride=1
    )
    path = tmp_path / "own_model.pkl"
    digest = save_model(path, fitted, overwrite=False)
    assert predict_mass(load_model(path, digest), inputs, 3) == predict_mass(
        fitted, inputs, 3
    )
    with pytest.raises(ValueError, match="refusing to deserialize"):
        load_model(path, "incorrect")


def test_time_classes_and_short_horizons_do_not_credit_a_later_tp() -> None:
    from ai_trader.replay.models import FillModel
    from scripts.research_horizon_probability import (
        cases_for_horizon,
        horizon_trade,
        outcome_class,
    )

    fill = FillModel(Decimal(1), Decimal(0), Decimal(0))
    bars = [_bar(index, high="101" if index == 20 else "100") for index in range(62)]
    strategy = replace(StrategyConfig(), tick_size=Decimal("0.1"))
    trade = horizon_trade(
        bars[0],
        bars[1:],
        Direction.LONG,
        strategy,
        Decimal("0.002"),
        Decimal(1),
        fill=fill,
    )
    horizons = tuple(Decimal(value) for value in ("0.1", "0.25", "0.5", "1"))
    assert outcome_class(trade, horizons) == 2
    action = (bars[0].end_time, Direction.LONG)
    cases = cases_for_horizon(
        {action: trade},
        {action[0]: Decimal("0.6")},
        {_OPEN.date(): bars},
        strategy,
        Decimal("0.002"),
        Decimal("0.1"),
    )
    assert cases[action].exit_reason is ExitReason.HORIZON
    with pytest.raises(ValueError, match="full largest horizon"):
        outcome_class(cases[action], horizons)


def test_horizon_threshold_reports_are_based_on_executed_trades() -> None:
    from scripts.research_horizon_probability import horizon_trade, threshold_study

    strategy = replace(
        StrategyConfig(), tick_size=Decimal("0.1"), screen_feasibility=False
    )
    snapshot = _features()
    bars = [_bar(index, high="101" if index % 7 == 2 else "100") for index in range(25)]
    cases, scores, snapshots = {}, {}, {}
    for index, score in ((0, "0.6"), (7, "0.6"), (14, "0.8")):
        moment = bars[index].end_time
        cases[(moment, Direction.LONG)] = horizon_trade(
            bars[index],
            bars[index + 1 :],
            Direction.LONG,
            strategy,
            Decimal("0.002"),
            Decimal("0.1"),
        )
        scores[moment] = Decimal(score)
        snapshots[moment] = snapshot
    threshold, curve, result = threshold_study(
        cases, scores, snapshots, strategy, [_OPEN.date()], ["0", "0.7"], seed=1
    )
    assert [row["trades"] for row in curve] == [3, 1]
    assert [row["tp_hits"] for row in curve] == [3, 1]
    assert threshold == "0"
    assert len(result.trades) == 3


def test_fast_long_and_short_are_symmetric_and_use_only_subsequent_candles() -> None:
    signal = _bar(0, high="110", low="90")
    cutoff = _OPEN + timedelta(minutes=3)
    long = target_outcome(signal, (_bar(1, high="100.2"), _bar(2)), cutoff)
    short = target_outcome(signal, (_bar(1, low="99.8"), _bar(2)), cutoff)
    assert long.utility == 1
    assert short.utility == -1
    assert long.first_minutes == short.first_minutes == 1


def test_slow_target_is_discounted_and_no_intraday_target_is_zero() -> None:
    cutoff = _OPEN + timedelta(minutes=7)
    fast = target_outcome(
        _bar(0), tuple(_bar(index, high="100.2") for index in range(1, 7)), cutoff
    )
    slow = target_outcome(
        _bar(0),
        (*(_bar(index) for index in range(1, 6)), _bar(6, high="100.2")),
        cutoff,
    )
    absent = target_outcome(
        _bar(0), tuple(_bar(index) for index in range(1, 7)), cutoff
    )
    assert 0 < slow.utility < fast.utility == 1
    assert slow.first_minutes == 6
    assert absent.utility == 0


def test_opposite_target_first_prevents_credit_for_a_later_reversal() -> None:
    outcome = target_outcome(
        _bar(0),
        (_bar(1, low="99.8"), _bar(2, high="100.2")),
        _OPEN + timedelta(minutes=3),
    )
    assert outcome.utility == -1
    assert outcome.short_minutes == 1
    assert outcome.long_minutes == 2


def test_both_targets_in_one_bar_are_unknown_not_a_guessed_winner() -> None:
    outcome = target_outcome(
        _bar(0), (_bar(1, high="100.2", low="99.8"),), _OPEN + timedelta(minutes=2)
    )
    assert outcome.utility is None
    assert outcome.excluded_reason == "ambiguous_bar"


def test_targets_after_cutoff_or_on_the_next_day_do_not_count() -> None:
    later = replace(
        _bar(2, high="110"),
        start_time=_OPEN + timedelta(days=1),
        end_time=_OPEN + timedelta(days=1, minutes=1),
    )
    outcome = target_outcome(_bar(0), (_bar(1), later), _OPEN + timedelta(minutes=2))
    assert outcome.utility == 0


def test_missing_minutes_cannot_be_labelled_as_no_opportunity() -> None:
    outcome = target_outcome(
        _bar(0), (_bar(2, high="100.2"),), _OPEN + timedelta(minutes=3)
    )
    assert outcome.utility is None
    assert outcome.excluded_reason == "missing_minute"


def test_signal_bar_cannot_be_reused_as_a_future_outcome() -> None:
    with pytest.raises(ValueError, match="after the signal"):
        target_outcome(_bar(0), (_bar(0),), _OPEN + timedelta(minutes=2))


@pytest.mark.parametrize("raw", ["0", "0.1", "1", "10", "1000"])
def test_centered_sigmoid_is_bounded_odd_and_neutral_at_zero(raw: str) -> None:
    value = Decimal(raw)
    assert -1 <= centered_sigmoid(-value) <= 0
    assert centered_sigmoid(-value) == -centered_sigmoid(value)
    assert 0 <= centered_sigmoid(value) <= 1
    assert centered_sigmoid(Decimal(0)) == 0


def _features(direction: int = 1):
    engine = FeatureEngine()
    snapshot = None
    for index in range(80):
        price = Decimal(100) + direction * Decimal(index) / 10
        candle = replace(
            _bar(index),
            open=price,
            high=price + Decimal("0.2"),
            low=price - Decimal("0.2"),
            close=price,
        )
        snapshot = engine.update(candle)
    return snapshot


def test_opportunity_runtime_is_directional_and_uses_saved_coefficients() -> None:
    config = TargetScoreConfig((Decimal(3), *(Decimal(0) for _ in COMPONENT_NAMES[1:])))
    rising, falling = _features(), _features(-1)
    assert 0 < config.evaluate(rising).score < 1
    assert -1 < config.evaluate(falling).score < 0
    assert TargetScoreConfig().evaluate(rising).score == 0
    assert config.evaluate(rising) == config.evaluate(rising)


def test_opportunity_fails_closed_on_missing_core_and_at_cutoff() -> None:
    config = TargetScoreConfig((Decimal(3), *(Decimal(0) for _ in COMPONENT_NAMES[1:])))
    snapshot = _features()
    unavailable = replace(snapshot, readiness=replace(snapshot.readiness, ema50=False))
    assert config.evaluate(unavailable).score == 0
    assert config.evaluate(unavailable).reason == "not_ready"
    closed = replace(snapshot, minutes_since_session_open=Decimal(360))
    assert config.evaluate(closed).score == 0
    assert config.evaluate(closed).reason == "outside_window"


def test_opportunity_precision_is_independent_of_the_callers_context() -> None:
    from decimal import localcontext

    snapshot = _features()
    config = TargetScoreConfig((Decimal(3), *(Decimal(0) for _ in COMPONENT_NAMES[1:])))
    expected = config.evaluate(snapshot)
    with localcontext() as context:
        context.prec = 8
        assert config.evaluate(snapshot) == expected


def test_rank_correlation_accounts_for_ties_and_constant_scores() -> None:
    assert rank_correlation([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1)
    assert rank_correlation([1, 1, 2, 3], [-1, -1, -2, -3]) == pytest.approx(-1)
    assert rank_correlation([0, 0, 0], [1, 2, 3]) is None


def test_signed_scanner_ranks_a_strong_short_ahead_of_a_weak_long() -> None:
    config = TargetScoreConfig((Decimal(3), *(Decimal(0) for _ in COMPONENT_NAMES[1:])))
    strong_short = replace(_features(-1), instrument=Instrument("NSE", "SHORT_NAME"))
    weak_long = replace(
        _features(),
        instrument=Instrument("NSE", "LONG_NAME"),
        atr_pct=Decimal("0.00001"),
    )
    scanner = Scanner(target_score=config)
    result = scanner.scan(
        (weak_long, strong_short), PortfolioState.empty(strong_short.candle_end_time)
    )
    assert len(result.candidates) == 2
    assert result.candidates[0].direction is Direction.SHORT
    assert result.candidates[0].score < 0 < result.candidates[1].score
    assert abs(result.candidates[0].score) > abs(result.candidates[1].score)
    blocked = scanner.scan(
        (strong_short,),
        PortfolioState(as_of=strong_short.candle_end_time, new_entries_blocked=True),
    )
    assert blocked.candidates == ()


def test_regularized_fit_recovers_a_known_directional_signal_without_an_intercept() -> (
    None
):
    pytest.importorskip("scipy")
    snapshot = replace(_features(), minutes_since_session_open=Decimal(1))
    rows = [
        Observation(
            snapshot,
            TargetOutcome(
                1 if direction > 0 else None,
                1 if direction < 0 else None,
                direction,
                1,
                Decimal(direction),
            ),
            (float(direction), *(0.0 for _ in COMPONENT_NAMES[1:])),
            1.0,
            0.0,
        )
        for direction in (-1, 1)
        for _ in range(20)
    ]
    coefficients = fit_coefficients(rows, (0,), 0.01)
    assert coefficients[0] > 1
    assert coefficients[1:] == (0.0,) * (len(COMPONENT_NAMES) - 1)
    assert predict(rows[0], coefficients) < -0.5
    assert predict(rows[-1], coefficients) > 0.5


def test_signed_predictions_do_not_change_when_future_candles_are_appended() -> None:
    model = TargetScoreConfig((Decimal(2), *(Decimal(0) for _ in COMPONENT_NAMES[1:])))
    engine = FeatureEngine()
    recorded = []
    bars = []
    for index in range(95):
        price = Decimal(100) + Decimal(index % 11) / 10
        bar = replace(
            _bar(index), open=price, high=price + 1, low=price - 1, close=price
        )
        bars.append(bar)
        recorded.append(model.evaluate(engine.update(bar)))
    prefix = FeatureEngine()
    for index, bar in enumerate(bars[:80]):
        assert model.evaluate(prefix.update(bar)) == recorded[index]


def test_target_tick_rounding_cannot_credit_a_move_smaller_than_requested() -> None:
    signal = replace(
        _bar(0),
        open=Decimal("100.1"),
        high=Decimal("100.1"),
        low=Decimal("100.1"),
        close=Decimal("100.1"),
    )
    short_of_target = replace(
        _bar(1, high="100.3"),
        open=Decimal("100.1"),
        low=Decimal("100.1"),
        close=Decimal("100.1"),
    )
    outcome = target_outcome(signal, (short_of_target,), _OPEN + timedelta(minutes=2))
    assert outcome.utility == 0


def test_threshold_is_inclusive_symmetric_and_does_not_trade_zero() -> None:
    snapshot = _features()
    rows = [
        Observation(snapshot, outcome, (0.0,) * len(COMPONENT_NAMES), 1.0, 0.0)
        for outcome in (
            TargetOutcome(1, None, 1, 1, Decimal(1)),
            TargetOutcome(None, 1, -1, 1, Decimal(-1)),
            TargetOutcome(None, 2, -1, 2, Decimal("-0.8")),
            TargetOutcome(None, None, 0, None, Decimal(0)),
            TargetOutcome(1, None, 1, 1, Decimal(1)),
            TargetOutcome(1, 1, 0, 1, None, "ambiguous_bar"),
        )
    ]
    scores = [0.02, -0.02, 0.03, -0.04, 0.0, 0.5]
    measured = threshold_metrics(rows, scores, 0.02)
    assert measured["signals"] == 4
    assert measured["coverage"] == pytest.approx(4 / 5)
    assert measured["excluded_observations"] == 1
    assert measured["correct_first_target"] == 2
    assert measured["opposite_first_target"] == 1
    assert measured["neither_target"] == 1
    assert measured["accuracy"] == measured["fast_precision"] == 0.5
    assert measured["fast_recall"] == 0.5
    assert measured["equal_day_utility_per_decision"] == pytest.approx(1.2 / 5)
    assert threshold_metrics(rows, scores, 0)["signals"] == 4
    abstain = threshold_metrics(rows, scores, 1)
    assert abstain["signals"] == 0
    assert abstain["accuracy"] is None
    assert abstain["equal_day_utility_per_decision"] == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.01, float("nan"), float("inf")])
def test_threshold_metrics_reject_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        threshold_metrics([], [], threshold)


@pytest.mark.parametrize("direction", [-1, 1])
def test_scanner_threshold_filters_candidates_without_changing_raw_scores(
    direction: int,
) -> None:
    model = TargetScoreConfig((Decimal(3), *(Decimal(0) for _ in COMPONENT_NAMES[1:])))
    snapshot = _features(direction)
    opportunity = model.evaluate(snapshot)
    equal = replace(model, score_threshold=abs(opportunity.score))
    above = replace(equal, score_threshold=equal.score_threshold + Decimal("0.0001"))
    portfolio = PortfolioState.empty(snapshot.candle_end_time)
    result = Scanner(target_score=equal).scan((snapshot,), portfolio)
    assert len(result.candidates) == 1
    assert result.candidates[0].score == opportunity.score
    assert above.evaluate(snapshot) == opportunity
    assert Scanner(target_score=above).scan((snapshot,), portfolio).candidates == ()
    assert (
        Scanner(target_score=TargetScoreConfig())
        .scan((snapshot,), portfolio)
        .candidates
        == ()
    )


@pytest.mark.parametrize("threshold", ["-0.1", "1.1", "NaN", "Infinity"])
def test_runtime_threshold_rejects_invalid_values(threshold: str) -> None:
    with pytest.raises(ValueError):
        TargetScoreConfig(score_threshold=Decimal(threshold))


def test_threshold_selection_cannot_see_forward_day_labels() -> None:
    snapshot = _features()
    predicted = []
    for day_number in (2, 3, 4):
        day_snapshot = replace(
            snapshot,
            candle_start_time=snapshot.candle_start_time.replace(day=day_number),
            candle_end_time=snapshot.candle_end_time.replace(day=day_number),
        )
        for index in range(20):
            correct = index < 10
            outcome = TargetOutcome(
                1 if correct else None,
                None if correct else 1,
                1 if correct else -1,
                1,
                Decimal(1 if correct else -1),
            )
            predicted.append(
                (
                    Observation(
                        day_snapshot, outcome, (0.0,) * len(COMPONENT_NAMES), 1.0, 0.0
                    ),
                    0.03 if correct else 0.005,
                )
            )
    name = "compact_8_ridge_0.1"
    options = {
        "selection_days": (date(2026, 9, 2), date(2026, 9, 3)),
        "minimum_signals": 20,
    }
    winner, sweep = select_threshold({name: predicted}, (0, 0.02, 1), **options)
    assert winner["threshold"] == 0.02
    assert winner["signals"] == 20
    assert winner["accuracy"] == 1
    changed = [
        (replace(row, outcome=TargetOutcome(None, 1, -1, 1, Decimal(-1))), -score)
        if row.session.day == 4
        else (row, score)
        for row, score in predicted
    ]
    assert select_threshold({name: changed}, (0, 0.02, 1), **options) == (winner, sweep)
    no_support, _ = select_threshold(
        {name: predicted},
        (0.02,),
        **{
            **options,
            "minimum_signals": 21,
        },
    )
    assert no_support is None
    bad = [
        (replace(row, outcome=TargetOutcome(None, 1, -1, 1, Decimal(-1))), score)
        for row, score in predicted
    ]
    assert select_threshold({name: bad}, (0, 0.02, 1), **options)[0] is None


def test_no_trades_do_not_report_a_success_rate_or_profit_factor() -> None:
    measured = _trade_summary(
        ReplayResult(
            trades=(),
            universe=(_INSTRUMENT,),
            sessions=(date(2026, 9, 4),),
            fill=FRICTIONLESS,
        )
    )
    assert measured["round_trips"] == 0
    assert measured["net_rupees"] == 0
    assert measured["net_hit_rate"] is None
    assert measured["profit_factor"] is None
    assert measured["average_net_win_rupees"] is None
    assert measured["average_net_loss_rupees"] is None
    assert measured["tp_hits"] == 0
    assert measured["tp_hit_rate"] is None


def test_tp_labels_use_execution_and_exclude_entry_bar_and_profitable_stop() -> None:
    strategy = StrategyConfig(tick_size=Decimal("0.1"), screen_feasibility=False)
    origin = _OPEN.replace(hour=15, minute=10)
    bars = tuple(
        replace(
            _bar(index, high="102" if index == 1 else "100", low="100"),
            start_time=origin + timedelta(minutes=index),
            end_time=origin + timedelta(minutes=index + 1),
        )
        for index in range(5)
    )
    trade = isolated_trade(bars[0], bars[1:], Decimal("0.01"), Direction.LONG, strategy)
    assert trade.exit_reason is ExitReason.SESSION_END
    touched = (bars[1], replace(bars[2], high=Decimal(102)), *bars[3:])
    target = isolated_trade(bars[0], touched, Decimal("0.01"), Direction.LONG, strategy)
    assert target.exit_reason is ExitReason.TARGET
    ambiguous = isolated_trade(
        bars[0],
        (bars[1], replace(bars[2], high=Decimal(102), low=Decimal(97)), *bars[3:]),
        Decimal("0.01"),
        Direction.LONG,
        strategy,
    )
    assert ambiguous.exit_reason is ExitReason.STOP
    assert ambiguous.ambiguous_exit
    profitable_stop = replace(target, exit_reason=ExitReason.STOP)
    result = ReplayResult(
        trades=(target, profitable_stop),
        universe=(_INSTRUMENT,),
        sessions=(origin.date(),),
        fill=FRICTIONLESS,
    )
    summary = precision_summary(result)
    assert summary["net_winners"] == 2
    assert summary["tp_hits"] == 1
    assert summary["tp_hit_rate"] == 0.5
    assert (
        isolated_trade(bars[-2], bars[-1:], Decimal("0.01"), Direction.LONG, strategy)
        is None
    )
    assert thresholds()[0] == 0 and thresholds()[-1] == 1


def test_prior_context_never_reads_current_or_later_session_extremes() -> None:
    origin = _OPEN.replace(day=31, month=8, hour=9, minute=15)
    previous = tuple(
        replace(
            _bar(index),
            start_time=origin + timedelta(minutes=index),
            end_time=origin + timedelta(minutes=index + 1),
        )
        for index in range(360)
    )
    today = tuple(
        replace(
            _bar(index),
            start_time=_OPEN.replace(hour=9, minute=15) + timedelta(minutes=index),
            end_time=_OPEN.replace(hour=9, minute=15) + timedelta(minutes=index + 1),
        )
        for index in range(360)
    )
    before = previous_sessions((*previous, *today[:10]))
    later = (*today[:10], *(replace(bar, high=Decimal(200)) for bar in today[10:]))
    assert before[_OPEN.date()] == previous_sessions((*previous, *later))[_OPEN.date()]
    assert before[_OPEN.date()].high == 100
    snapshot = _features()
    prior = before[_OPEN.date()]
    current = replace(
        _bar(79),
        start_time=snapshot.candle_start_time,
        end_time=snapshot.candle_end_time,
    )
    with pytest.raises(ValueError, match="session open"):
        tp_inputs(snapshot, current, Direction.LONG, prior, include_context=True)
    snapshot = replace(
        snapshot,
        session_open=Decimal(100),
        readiness=replace(snapshot.readiness, session_open=True),
    )
    values = tp_inputs(snapshot, current, Direction.LONG, prior, include_context=True)
    assert len(values) == 17
    larger_target = tp_inputs(
        snapshot,
        current,
        Direction.LONG,
        prior,
        include_context=True,
        target_fraction=Decimal("0.003"),
    )
    assert larger_target[8] < values[8]
    assert larger_target[:8] == values[:8]
    with pytest.raises(ValueError, match="earlier session"):
        tp_inputs(
            snapshot,
            current,
            Direction.LONG,
            PriorFrame(snapshot.candle_start_time.date(), *([Decimal(100)] * 4)),
            include_context=True,
        )


def test_tp_confidence_keeps_probability_and_direction_distinct() -> None:
    assert signed_tp_confidence(0.8, 0.3) == Decimal("0.6")
    assert signed_tp_confidence(0.3, 0.8) == Decimal("-0.6")
    assert signed_tp_confidence(0.4, 0.3) == 0
    assert signed_tp_confidence(0.8, 0.8) == 0
    with pytest.raises(ValueError):
        signed_tp_confidence(float("nan"), 0.5)


def test_signed_tp_probability_preserves_probability_without_an_implicit_floor() -> (
    None
):
    from scripts.research_tp_policy import signed_tp_probability

    assert signed_tp_probability(0.8, 0.3) == Decimal("0.8")
    assert signed_tp_probability(0.3, 0.8) == Decimal("-0.8")
    assert signed_tp_probability(0.4, 0.3) == Decimal("0.4")
    assert signed_tp_probability(0.3, 0.4) == Decimal("-0.4")
    assert signed_tp_probability(0.8, 0.8) == 0
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            signed_tp_probability(invalid, 0.5)


def test_binary_tp_fit_never_uses_check_day_labels() -> None:
    pytest.importorskip("scipy")
    strategy = StrategyConfig(screen_feasibility=False)
    origin = _OPEN.replace(hour=15, minute=10)
    bars = tuple(
        replace(
            _bar(index, high="102" if index == 2 else "100"),
            start_time=origin + timedelta(minutes=index),
            end_time=origin + timedelta(minutes=index + 1),
        )
        for index in range(5)
    )
    trade = isolated_trade(bars[0], bars[1:], Decimal("0.01"), Direction.LONG, strategy)
    inputs = {}
    cases = {}
    for day in (1, 2, 3, 4):
        for direction in Direction:
            key = (_OPEN.replace(day=day, hour=10, minute=1), direction)
            inputs[key] = (1.0 if direction is Direction.LONG else -1.0,)
            cases[key] = replace(
                trade,
                exit_reason=(
                    ExitReason.TARGET
                    if direction is Direction.LONG
                    else ExitReason.STOP
                ),
            )
    before = date(2026, 9, 4)
    fitted = fit_tp(inputs, cases, before=before)
    changed = {
        key: replace(trade, exit_reason=ExitReason.STOP)
        if key[0].date() == before
        else trade
        for key, trade in cases.items()
    }
    assert fit_tp(inputs, changed, before=before) == fitted
    assert fitted[1] > 0
    stronger = fit_tp(inputs, cases, before=before, penalty=1.0)
    assert 0 < stronger[1] < fitted[1]


def test_tp_rate_selection_accepts_sparse_results_but_never_reads_check_metrics() -> (
    None
):
    sparse = {
        "threshold": "0.8",
        "tp_fraction": "0.002",
        "sl_atr": "2",
        "selection": {"round_trips": 1, "tp_hit_rate": 1.0, "represented_sessions": 1},
        "check": {"tp_hit_rate": 0.0},
    }
    broad = {
        "threshold": "0.02",
        "tp_fraction": "0.002",
        "sl_atr": "2",
        "selection": {"round_trips": 20, "tp_hit_rate": 0.8, "represented_sessions": 2},
        "check": {"tp_hit_rate": 1.0},
    }
    assert choose_policy([sparse, broad]) is sparse
    sparse["check"]["tp_hit_rate"] = 1.0
    broad["check"]["tp_hit_rate"] = 0.0
    assert choose_policy([sparse, broad]) is sparse
    assert (
        choose_policy(
            [
                {
                    **sparse,
                    "selection": {"round_trips": 0, "tp_hit_rate": None},
                }
            ]
        )
        is None
    )


def test_probability_calibration_matches_event_rate_and_ignores_later_labels() -> None:
    pytest.importorskip("scipy")
    from scripts.research_tp_probability import fit_calibration, probability_metrics

    before = date(2026, 9, 11)
    samples = [(before, 0.8, int(index < 2)) for index in range(10)]
    samples += [(date(2026, 9, 21), 0.8, 1)] * 100
    fitted = fit_calibration(samples, through=before)
    corrected = fitted.apply(0.8)
    assert abs(corrected - 0.2) < abs(0.8 - 0.2)
    assert fitted.apply(0.3) <= fitted.apply(0.8)
    assert fit_calibration(samples[:10], through=before) == fitted
    actual = probability_metrics([(corrected, int(index < 2)) for index in range(10)])
    assert actual["tp_rate"] == 0.2
    assert actual["gap"] == pytest.approx(corrected - 0.2)
    assert (
        actual["brier"]
        < probability_metrics([(0.8, int(index < 2)) for index in range(10)])["brier"]
    )


def test_probability_metrics_keep_empty_and_extreme_probabilities_honest() -> None:
    from scripts.research_tp_probability import PlattCalibration, probability_metrics

    assert probability_metrics([])["tp_rate"] is None
    exact = probability_metrics([(0.0, 0), (1.0, 1)])
    assert exact["brier"] == exact["gap"] == exact["ece"] == 0
    assert sum(bucket["observations"] for bucket in exact["bins"]) == 2
    with pytest.raises(ValueError):
        PlattCalibration(-1, 0)
    with pytest.raises(ValueError):
        probability_metrics([(float("nan"), 1)])


def test_probability_ranking_can_be_distinguished_from_calibration() -> None:
    from scripts.research_tp_probability import probability_metrics

    perfect_order = probability_metrics([(0.2, 0), (0.3, 0), (0.4, 1), (0.5, 1)])
    assert perfect_order["roc_auc"] == 1
    assert perfect_order["gap"] != 0
    assert probability_metrics([(0.9, 0), (0.8, 0), (0.2, 1)])["roc_auc"] == 0
    assert probability_metrics([(0.5, 0), (0.5, 1)])["roc_auc"] == 0.5
    assert probability_metrics([(0.5, 1)])["roc_auc"] is None


def test_probability_study_phases_do_not_reuse_selection_days_as_test() -> None:
    import json
    from pathlib import Path

    from scripts.research_tp_probability import phase_for

    root = Path(__file__).parents[1]
    protocol = json.loads(
        (root / "backtests/tp_probability_2026q3/protocol.json").read_text()
    )
    assert phase_for(date(2026, 8, 21), protocol) == "fit"
    assert phase_for(date(2026, 8, 31), protocol) == "model_selection"
    assert phase_for(date(2026, 9, 11), protocol) == "calibration"
    assert phase_for(date(2026, 9, 18), protocol) == "threshold_selection"
    assert phase_for(date(2026, 9, 21), protocol) == "test"
    assert phase_for(date(2026, 9, 29), protocol) is None
    protocol["phases"]["test"][0] = "2026-09-18"
    with pytest.raises(ValueError, match="overlap"):
        phase_for(date(2026, 9, 18), protocol)


def test_trade_probability_breakdown_keeps_long_and_short_outcomes_separate() -> None:
    from scripts.research_tp_policy import RESEARCH_FILL
    from scripts.research_tp_probability import trade_diagnostics

    origin = _OPEN.replace(hour=15, minute=10)
    bars = tuple(
        replace(
            _bar(index, high="102" if index == 2 else "100"),
            start_time=origin + timedelta(minutes=index),
            end_time=origin + timedelta(minutes=index + 1),
        )
        for index in range(5)
    )
    strategy = StrategyConfig(screen_feasibility=False)
    long = isolated_trade(bars[0], bars[1:], Decimal("0.01"), Direction.LONG, strategy)
    short = isolated_trade(
        bars[0], bars[1:], Decimal("0.01"), Direction.SHORT, strategy
    )
    assert long.exit_reason is ExitReason.TARGET
    assert short.exit_reason is ExitReason.STOP
    result = ReplayResult(
        trades=(
            replace(long, score=Decimal("0.8")),
            replace(short, score=Decimal("-0.7")),
        ),
        universe=(_INSTRUMENT,),
        sessions=(origin.date(),),
        fill=RESEARCH_FILL,
    )
    measured = trade_diagnostics(result, {Direction.LONG: 0.5, Direction.SHORT: 0.5})
    long_result = measured["by_direction"]["LONG"]
    short_result = measured["by_direction"]["SHORT"]
    assert long_result["tp_hits"] == 1
    assert short_result["tp_hits"] == 0
    assert long_result["probability"]["mean_probability"] == 0.8
    assert short_result["probability"]["mean_probability"] == 0.7
    assert short_result["probability"]["gap"] == 0.7
    assert measured["probability"]["mean_probability"] == 0.75
    assert measured["probability"]["tp_rate"] == 0.5
    outcomes = measured["outcomes"]
    assert outcomes["sl_percent"]["mean"] == pytest.approx(
        50
        * (
            float(abs(long.stop_price / long.entry_price - 1))
            + float(abs(short.stop_price / short.entry_price - 1))
        )
    )
    assert outcomes["tp_within_minutes"]["5"]["fraction_of_all_actions"] == 0.5
    assert outcomes["successful_tp_minutes"]["median"] == float(long.holding_minutes)


def test_feature_contribution_profile_uses_training_variation_not_just_weights() -> (
    None
):
    from scripts.research_tp_probability import coefficient_profile

    sampled = {
        (_OPEN.replace(minute=1), Direction.LONG): (-1.0, 0.0),
        (_OPEN.replace(minute=6), Direction.LONG): (1.0, 0.0),
    }
    result = coefficient_profile(
        ("intercept", "varying", "constant"), (0.1, 0.2, 10), sampled
    )
    assert result["features"][0]["feature"] == "varying"
    assert result["features"][0][
        "mean_absolute_centered_logit_effect"
    ] == pytest.approx(0.2)
    assert result["features"][1]["mean_absolute_centered_logit_effect"] == 0
    assert result["sampled_actions"] == 2


def test_month_split_keeps_august_out_of_base_coefficient_training() -> None:
    import json
    from pathlib import Path

    from scripts.research_tp_probability import model_training_sessions, phase_for

    root = Path(__file__).parents[1]
    protocol = json.loads(
        (root / "backtests/tp_probability_month_split/protocol.json").read_text()
    )
    phases = {
        "fit": (date(2026, 7, 2), date(2026, 7, 31)),
        "model_selection": (date(2026, 8, 3), date(2026, 8, 7)),
    }
    assert model_training_sessions(phases, protocol) == phases["fit"]
    assert model_training_sessions(phases, {}) == (
        *phases["fit"],
        *phases["model_selection"],
    )
    assert phase_for(date(2026, 8, 3), protocol) == "model_selection"
    assert phase_for(date(2026, 8, 10), protocol) == "calibration"
    assert phase_for(date(2026, 8, 31), protocol) == "threshold_selection"
    assert phase_for(date(2026, 9, 1), protocol) == "test"
    assert phase_for(date(2026, 9, 28), protocol) == "test"
    with pytest.raises(ValueError, match="boolean"):
        model_training_sessions(phases, {"refit_model_selection": "false"})
