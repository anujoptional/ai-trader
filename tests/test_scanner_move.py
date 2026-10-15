from collections import Counter
from datetime import date

import numpy as np
import pytest

pytest.importorskip("torch")  # the research scripts import the training stack

from scripts import research_delivery_pipeline as pipeline  # noqa: E402
from scripts import research_scanner_move as move  # noqa: E402
from scripts.research_fno_scanner import epoch_day  # noqa: E402

FIRST_DAY = 20_000  # 2024-10-04, before the closing auction


def random_path(
    sessions: int, seed: int, first_day: int = FIRST_DAY, length: int = 375
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete sessions of paise prices; some minutes open away from the previous
    close and some are wide enough to reach both levels."""
    rng = np.random.default_rng(seed)
    days, minutes, rows = [], [], []
    price = 100_000
    for session in range(sessions):
        price = max(1_000, int(price * (1 + rng.normal(0, 0.01))))
        for minute in range(length):
            opened = price
            if rng.random() < 0.01:
                opened = int(price * (1 + rng.choice((-1, 1)) * 0.012))
            close = max(1_000, int(opened * (1 + rng.normal(0, 0.0012))))
            spread = 0.02 if rng.random() < 0.01 else 0.001
            high = max(opened, close) + int(opened * spread * rng.random())
            low = min(opened, close) - int(opened * spread * rng.random())
            days.append(first_day + session)
            minutes.append(minute)
            rows.append((opened, high, low, close))
            price = close
    return np.array(days), np.array(minutes), np.array(rows, dtype=np.int64)


def plain_loop(session: np.ndarray, buy: int, bps: int = 80) -> tuple:
    """Touched, up, down, first move (1 up, 0 down, None unknown), minutes, and
    whether a minute's open decided the first move."""
    entry = int(session[buy, 0])
    up, down = entry * (10_000 + bps), entry * (10_000 - bps)
    reached_up = reached_down = by_open = False
    first = minutes = None
    for i, (opened, high, low, _) in enumerate(session[buy:].tolist()):
        hit_up, hit_down = high * 10_000 >= up, low * 10_000 <= down
        if minutes is None and (hit_up or hit_down):
            minutes = i + 1
            by_open = opened * 10_000 >= up or opened * 10_000 <= down
            if opened * 10_000 >= up:
                first = 1
            elif opened * 10_000 <= down:
                first = 0
            elif hit_up and hit_down:
                first = None
            else:
                first = 1 if hit_up else 0
        reached_up |= hit_up
        reached_down |= hit_down
    return reached_up or reached_down, reached_up, reached_down, first, minutes, by_open


def check_day(days, minutes, ohlc, day, counts, bps: int = 80) -> None:
    out = move.move_labels(ohlc, days, minutes, day, counts, bps)
    start = int(np.searchsorted(days, day))
    end = int(np.searchsorted(days, day, side="right"))
    for slot, decision in enumerate(move.DECISIONS):
        touched, up, down, first, wait, by_open = plain_loop(
            ohlc[start:end], int(decision) + 1, bps
        )
        counts["test_first_move_by_open"] += by_open
        assert out[slot, move.TOUCHED] == touched
        assert out[slot, move.UP] == up and out[slot, move.DOWN] == down
        if first is None:
            assert np.isnan(out[slot, move.UP_FIRST])
        else:
            assert out[slot, move.UP_FIRST] == first
        if wait is None:
            assert np.isnan(out[slot, move.MINUTES])
        else:
            assert out[slot, move.MINUTES] == wait


@pytest.mark.parametrize("bps", [80, 100])
def test_labels_match_a_plain_loop_on_random_paths(bps: int) -> None:
    counts: Counter = Counter()
    outcomes = []
    for seed in (1, 2, 3):
        days, minutes, ohlc = random_path(5, seed)
        for day in np.unique(days):
            check_day(days, minutes, ohlc, int(day), counts, bps)
            outcomes.append(
                move.move_labels(ohlc, days, minutes, int(day), Counter(), bps)
            )
    stacked = np.concatenate(outcomes)
    touched = stacked[:, move.TOUCHED]
    assert 0 < touched.mean() < 1  # both outcomes occur
    assert counts["same_minute_order_unknown"] > 0
    assert counts["test_first_move_by_open"] > 0
    assert counts["sessions_labelled"] == 15


def test_a_bigger_move_is_touched_less_often() -> None:
    days, minutes, ohlc = random_path(5, 1)
    small = np.concatenate(
        [
            move.move_labels(ohlc, days, minutes, int(day), Counter(), 80)
            for day in np.unique(days)
        ]
    )[:, move.TOUCHED]
    large = np.concatenate(
        [
            move.move_labels(ohlc, days, minutes, int(day), Counter(), 100)
            for day in np.unique(days)
        ]
    )[:, move.TOUCHED]
    assert (large <= small).all() and large.sum() < small.sum()


def test_from_the_closing_auction_the_session_ends_at_15_14() -> None:
    first = epoch_day("2026-08-03")
    days, minutes, ohlc = random_path(3, 17, first, 360)
    counts: Counter = Counter()
    for day in np.unique(days):
        check_day(days, minutes, ohlc, int(day), counts)
    assert counts["sessions_labelled"] == 3
    # a 360-minute session before the auction began is incomplete
    days, minutes, ohlc = random_path(1, 19, first - 1, 360)
    out = move.move_labels(ohlc, days, minutes, first - 1, counts)
    assert np.isnan(out).all() and counts["sessions_incomplete"] == 1


def test_a_session_missing_a_minute_has_no_label() -> None:
    days, minutes, ohlc = random_path(2, 13)
    keep = ~((days == FIRST_DAY) & (minutes == 370))
    counts: Counter = Counter()
    out = move.move_labels(ohlc[keep], days[keep], minutes[keep], FIRST_DAY, counts)
    assert np.isnan(out).all() and counts["sessions_incomplete"] == 1
    later = move.move_labels(
        ohlc[keep], days[keep], minutes[keep], FIRST_DAY + 1, counts
    )
    assert np.isfinite(later[:, move.TOUCHED]).all()


def test_new_inputs_use_only_minutes_up_to_the_decision() -> None:
    rng = np.random.default_rng(5)
    close = 500 * np.exp(np.cumsum(rng.normal(0, 0.001, 345)))
    prices = np.column_stack(
        (close, close * 1.001, close * 0.999, close, rng.integers(1, 9, 345))
    )
    values = move.extra_columns(prices)
    slot = 10
    d = int(move.DECISIONS[slot])
    returns = np.diff(np.log(close[: d + 1]))
    assert values[slot, 0] == pytest.approx(close[d] / prices[: d + 1, 2].min() - 1)
    assert values[slot, 1] == pytest.approx(np.sqrt(20 * np.sum(returns**2) / d))
    changed = prices.copy()
    changed[d + 1 :, :4] *= 1.5
    assert np.array_equal(move.extra_columns(changed)[: slot + 1], values[: slot + 1])


@pytest.mark.parametrize("step", [0.008, 0.01])
def test_movement_scales_divide_the_move_by_volatility_over_the_time_left(
    step: float,
) -> None:
    features = np.full((1, 2, 1, len(move.NAMES)), np.nan, np.float32)
    col = {name: i for i, name in enumerate(move.NAMES)}
    features[0, :, 0, col["slot"]] = (350, 35)
    features[0, :, 0, col["hist_rv"]] = 0.004
    features[0, :, 0, col["rv_open"]] = 0.002
    features[0, :, 0, col["atr20"]] = 0.02
    move.movement_scales(features, step)
    for i, left in enumerate((350, 35)):
        row = features[0, i, 0]
        assert row[col["z_hist"]] == pytest.approx(step / (0.004 * np.sqrt(left / 20)))
        assert row[col["z_open"]] == pytest.approx(step / (0.002 * np.sqrt(left / 20)))
        assert row[col["z_atr"]] == pytest.approx(step / (0.02 * np.sqrt(left / 375)))


def test_pruning_over_the_base_inputs_matches_the_research_pruning() -> None:
    rng = np.random.default_rng(3)
    shape = (40, 8, 5)
    days = np.arange(40) + epoch_day("2021-03-01")
    base = rng.normal(size=(*shape, len(pipeline.FEATURES))).astype(np.float32)
    base[..., pipeline.FEATURES.index("hist_rv")] = base[
        ..., pipeline.FEATURES.index("atr20")
    ] * 2 + rng.normal(0, 0.01, shape).astype(np.float32)
    base[3, :, 1, pipeline.FEATURES.index("ret60d")] = np.nan  # optional, missing
    full = np.concatenate(
        (base, rng.normal(size=(*shape, len(move.EXTRA))).astype(np.float32)), axis=-1
    )
    full[..., move.NAMES.index("z_atr")] = full[..., move.NAMES.index("z_hist")] * 3
    available = np.ones(shape, bool)
    available[5, :, 2] = False
    research = {"days": days, "available": available, "features": base}
    ours = {"days": days, "available": available, "features": full}
    assert move.prune_inputs(ours, pipeline.FEATURES) == pipeline.prune(research)
    extended, dropped = move.prune_inputs(ours, move.NAMES)
    assert [n for n in extended if n in pipeline.FEATURES] == pipeline.prune(research)[
        0
    ]
    assert {"feature": "z_atr", "duplicates": "z_hist"} in [
        {k: d[k] for k in ("feature", "duplicates")} for d in dropped
    ]


def toy_data() -> tuple[dict, np.ndarray]:
    """Four sessions x 64 checks x 3 stocks. Stock 0 is at 0.85 for the first ten
    checks every day (8 of 10 move); stock 1 has one 0.95 estimate with an unknown
    outcome on days 0 and 1; everything else is 0.15, and stock 2 moves at every
    check of day 0."""
    days = np.array([epoch_day(f"2025-01-0{i}") for i in range(1, 5)])
    shape = (4, move.SLOTS, 3)
    p = np.full(shape, 0.15, np.float32)
    outcomes = np.zeros((*shape, len(move.OUTCOMES)), np.float32)
    outcomes[..., move.UP_FIRST] = np.nan
    outcomes[..., move.MINUTES] = np.nan
    p[:, :10, 0] = 0.85
    moved = np.zeros(shape, bool)
    moved[:, :8, 0] = True
    moved[0, :, 2] = True
    outcomes[..., move.TOUCHED] = moved
    outcomes[..., move.UP] = moved
    outcomes[..., move.UP_FIRST] = np.where(moved, 1, np.nan)
    outcomes[..., move.MINUTES] = np.where(moved, 5, np.nan)
    p[0, 20, 1] = p[1, 20, 1] = 0.95
    outcomes[0, 20, 1] = outcomes[1, 20, 1] = np.nan
    data = {"days": days, "available": np.ones(shape, bool), "outcomes": outcomes}
    return data, p


def test_evaluate_reports_bands_and_candidates() -> None:
    data, p = toy_data()
    result = move.evaluate(
        p, data, (("test_all", "2025-01-01", "2025-01-31"),), np.full(64, 0.1)
    )["test_all"]
    assert result["sessions"] == 4 and result["estimates"] == 768
    bands = result["bands"]
    assert bands[8]["estimates"] == 40 and bands[8]["hit_rate"] == 0.8
    assert bands[8]["mean_probability"] == 0.85
    assert bands[8]["hit_rate_95"] == [0.8, 0.8]  # every day 8 of 10
    assert bands[9]["estimates"] == 2 and bands[9]["known_outcomes"] == 0
    assert bands[9]["hit_rate"] is None
    assert bands[1]["known_outcomes"] == 726
    assert bands[1]["hit_rate"] == pytest.approx(64 / 726, abs=1e-4)
    mass = 726 * float(np.float32(0.15)) + 40 * float(np.float32(0.85))
    assert result["mean_probability"] == pytest.approx(mass / 766, abs=1e-4)
    expected_ece = (abs(726 * 0.15 - 64) + abs(40 * 0.85 - 32)) / 766
    assert result["calibration_error"] == pytest.approx(expected_ece, abs=1e-4)
    top = result["candidates"]["0.80"]
    assert top["estimates"] == 42 and top["per_session_mean"] == 10.5
    assert top["stocks_per_session_mean"] == 1.5
    assert top["known_outcomes"] == 40 and top["hit_rate"] == 0.8
    assert top["first_surfacing_hit_rate"] == 1.0  # stock 0 moves at its first check
    assert top["up_touched"] == 0.8 and top["down_touched"] == 0.0
    assert top["up_first_share_of_moves"] == 1.0
    assert top["minutes_to_move_median_p75"] == [5.0, 5.0]
    assert top["by_time"][0] == {
        "checks": "09:40-10:55",
        "share_of_estimates": round(40 / 42, 4),
        "hit_rate": 0.8,
    }
    assert top["by_time"][1]["hit_rate"] is None  # only unknown outcomes there
    assert result["candidates"]["0.90"]["known_outcomes"] == 0


def test_day_bootstrap_resamples_whole_sessions() -> None:
    counts = np.full((30, 2), 10.0)
    counts[:, 1] = 0
    low, high = move.day_bootstrap(counts, np.full((30, 2), 7.0))
    assert low[0] == high[0] == pytest.approx(0.7)
    assert np.isnan(low[1]) and np.isnan(high[1])
    hits = np.where(np.arange(30)[:, None] < 15, 10.0, 0.0) * np.ones((1, 2))
    low, high = move.day_bootstrap(counts, hits)
    assert low[0] < 0.5 < high[0]


def test_test_period_follows_every_choice() -> None:
    last_choice = max(date.fromisoformat(move.RESEARCH[k][1]) for k in move.RESEARCH)
    assert last_choice < date.fromisoformat(move.TEST[0])
    production = move.PRODUCTION
    assert date.fromisoformat(production["calibration"][1]) < date.fromisoformat(
        production["holdout"][0]
    )
