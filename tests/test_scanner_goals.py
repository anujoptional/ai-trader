from collections import Counter

import numpy as np
import pytest

pytest.importorskip("torch")  # the research script imports the training stack

from scripts import research_scanner_goals as goals  # noqa: E402

FIRST_DAY = 20_000


def random_path(
    sessions: int,
    seed: int,
    first_day: int = FIRST_DAY,
    session_minutes: int = goals.SESSION_MINUTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete sessions of paise prices with overnight gaps and some minutes wide
    enough to reach both levels."""
    rng = np.random.default_rng(seed)
    days, minutes, rows = [], [], []
    price = 100_000
    for session in range(sessions):
        price = max(1_000, int(price * (1 + rng.normal(0, 0.02))))
        for minute in range(session_minutes):
            close = max(1_000, int(price * (1 + rng.normal(0, 0.0015))))
            spread = 0.03 if rng.random() < 0.01 else 0.001
            high = max(price, close) + int(price * spread * rng.random())
            low = min(price, close) - int(price * spread * rng.random())
            days.append(first_day + session)
            minutes.append(minute)
            rows.append((price, high, low, close))
            price = close
    return np.array(days), np.array(minutes), np.array(rows, dtype=np.int64)


def first_touch(ohlc: np.ndarray, start: int, entry: int, bps: int) -> int | None:
    """Plain loop: 1 when +bps comes first, 0 when -bps does, None if neither."""
    up, down = entry * (10_000 + bps), entry * (10_000 - bps)
    for opened, high, low, _ in ohlc[start:].tolist():
        if opened * 10_000 <= down:
            return 0
        if opened * 10_000 >= up:
            return 1
        if low * 10_000 <= down:
            return 0
        if high * 10_000 >= up:
            return 1
    return None


def labels_for(days, minutes, ohlc, calendar, day, counts=None) -> np.ndarray:
    return goals.day_labels(
        ohlc,
        days,
        minutes,
        np.unique(days),
        calendar,
        [],
        day,
        Counter() if counts is None else counts,
    )


def test_labels_match_a_plain_loop_on_random_paths() -> None:
    counts: Counter = Counter()
    for seed in (1, 2, 3):
        days, minutes, ohlc = random_path(6, seed)
        calendar = np.unique(days)
        for day in calendar:
            out = labels_for(days, minutes, ohlc, calendar, int(day), counts)
            first = int(np.searchsorted(days, day))
            close = int(np.searchsorted(days, day, side="right"))
            for slot, decision in enumerate(goals.DECISIONS):
                start = first + int(decision) + 1
                entry = int(ohlc[start, 0])
                touched = (ohlc[start:close, 1] * 10_000 >= entry * 10_100).any()
                assert out[slot, 0] == touched
                for goal, bps in enumerate(goals.BRACKETS, start=1):
                    expected = first_touch(ohlc, start, entry, bps)
                    if expected is None:
                        assert np.isnan(out[slot, goal])
                    else:
                        assert out[slot, goal] == expected
    # the paths exercise every branch: later sessions, data end, same minute
    assert counts["b_resolved_after_the_session"] > 0
    assert counts["d_unknown_open_at_data_end"] > 0
    assert counts["b_same_minute_counted_down"] > 0


def test_outcomes_needing_a_missing_session_are_unknown() -> None:
    days, minutes, ohlc = random_path(4, 7)
    calendar = np.unique(days)
    keep = days != FIRST_DAY + 1  # the stock has no minutes on the second day
    out = labels_for(days[keep], minutes[keep], ohlc[keep], calendar, FIRST_DAY)
    full = labels_for(days, minutes, ohlc, calendar, FIRST_DAY)
    same_day = np.isfinite(out)
    assert same_day.any() and (~same_day[:, 1:]).any()
    assert np.array_equal(out[same_day], full[same_day])
    first = 0
    for slot, decision in enumerate(goals.DECISIONS):
        start = first + int(decision) + 1
        entry = int(ohlc[start, 0])
        for goal, bps in enumerate(goals.BRACKETS, start=1):
            within = first_touch(ohlc[: goals.SESSION_MINUTES], start, entry, bps)
            assert np.isfinite(out[slot, goal]) == (within is not None)


def test_the_close_goal_needs_every_minute_to_15_29() -> None:
    days, minutes, ohlc = random_path(2, 11)
    calendar = np.unique(days)
    keep = ~((days == FIRST_DAY) & (minutes == goals.SESSION_MINUTES - 1))
    out = labels_for(days[keep], minutes[keep], ohlc[keep], calendar, FIRST_DAY)
    assert np.isnan(out[:, 0]).all()
    assert np.isfinite(labels_for(days, minutes, ohlc, calendar, FIRST_DAY)[:, 0]).all()


def test_from_the_closing_auction_the_session_ends_at_15_14() -> None:
    first = goals.AUCTION_FROM
    days, minutes, ohlc = random_path(3, 17, first, goals.AUCTION_MINUTES)
    calendar = np.unique(days)
    for day in calendar:
        out = labels_for(days, minutes, ohlc, calendar, int(day))
        start = int(np.searchsorted(days, day))
        close = start + goals.AUCTION_MINUTES
        for slot, decision in enumerate(goals.DECISIONS):
            buy = start + int(decision) + 1
            touched = (ohlc[buy:close, 1] * 10_000 >= ohlc[buy, 0] * 10_100).any()
            assert out[slot, 0] == touched
    # the last pre-auction session still needs every minute to 15:29
    before, minutes_before, ohlc_before = random_path(
        1, 19, first - 1, goals.AUCTION_MINUTES
    )
    out = labels_for(before, minutes_before, ohlc_before, np.unique(before), first - 1)
    assert np.isnan(out[:, 0]).all()
    assert goals.session_minutes(first - 1) == goals.SESSION_MINUTES


def test_a_session_missing_minutes_before_15_00_has_no_labels() -> None:
    days, minutes, ohlc = random_path(2, 13)
    calendar = np.unique(days)
    keep = ~((days == FIRST_DAY) & (minutes == 100))
    counts: Counter = Counter()
    out = labels_for(days[keep], minutes[keep], ohlc[keep], calendar, FIRST_DAY, counts)
    assert np.isnan(out).all() and counts["sessions_incomplete_before_1500"] == 1
