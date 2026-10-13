import numpy as np
import pytest

pytest.importorskip("catboost")
pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("sklearn")

from scripts import research_delivery_catboost as study

DAYS = ("2024-01-01", "2024-01-02", "2024-01-03")
CALENDAR = {day: index for index, day in enumerate(DAYS)}


def flat(price: float, count: int = 375) -> list[tuple[float, ...]]:
    return [(price, price, price, price)] * count


def minutes(sessions: dict[str, list[tuple[float, ...]]]) -> tuple:
    days, minute, prices = [], [], []
    for day, rows in sessions.items():
        for index, row in enumerate(rows):
            days.append(day)
            minute.append(index)
            prices.append(row)
    return (
        days,
        np.asarray(minute, dtype=np.int32),
        np.asarray(prices, dtype=np.float64),
        np.full(len(days), 100, dtype=np.int64),
    )


def daily(**overrides) -> dict[str, tuple[float, ...]]:
    bars = {day: (100.0, 100.0, 100.0, 100.0, 1.0) for day in DAYS}
    bars.update(overrides)
    return bars


def outcome(sessions, bars=None, events=(), last_day=2) -> dict:
    stock = study.build_stock(
        "TEST", minutes(sessions), bars or daily(), CALENDAR, list(events)
    )
    entry = study.decision_row(stock, 0)
    return study.label_entry(stock, 0, entry, last_day)


def test_take_profit_on_a_later_day_and_gap_through_the_stop() -> None:
    later = flat(100.0)
    later[10] = (100.0, 101.3, 100.0, 101.0)
    result = outcome({DAYS[0]: flat(100.0), DAYS[1]: later})
    assert (result["label"], result["exit_day"], result["sessions"]) == (1, 1, 1)
    assert result["gross"] == pytest.approx(0.0125)
    gapped = flat(99.0)
    result = outcome({DAYS[0]: flat(100.0), DAYS[1]: gapped})
    assert result["label"] == 0 and result["gross"] == pytest.approx(-0.01)


def test_same_minute_touches_count_as_stop_and_same_day_exits_are_flagged() -> None:
    session = flat(100.0)
    session[350] = (100.0, 101.3, 99.2, 100.0)
    result = outcome({DAYS[0]: session})
    assert result["label"] == 0 and result["ambiguous"] and result["same_day"]
    assert result["gross"] == pytest.approx(-0.0075)


def test_missing_minute_day_uses_daily_bar_or_is_excluded() -> None:
    sessions = {DAYS[0]: flat(100.0), DAYS[2]: flat(100.0)}
    result = outcome(sessions, daily(**{DAYS[1]: (100.0, 101.5, 99.5, 101.0, 1.0)}))
    assert result["label"] == 1 and result["daily_fallback"]
    result = outcome(sessions, daily(**{DAYS[1]: (100.0, 101.5, 99.0, 100.0, 1.0)}))
    assert result == {"status": "daily_ambiguous"}


def test_unresolved_windows_are_censored_and_demerger_windows_excluded() -> None:
    sessions = {day: flat(100.0) for day in DAYS}
    assert outcome(sessions)["status"] == "censored"
    later = flat(100.0)
    later[5] = (100.0, 101.3, 100.0, 101.0)
    sessions[DAYS[2]] = later
    result = outcome(sessions, events=[(DAYS[1], 1.0, "demerger")])
    assert result == {"status": "demerger_window"}


def test_unadjusted_dividend_is_total_return_adjusted() -> None:
    after = flat(99.0)
    after[20] = (99.0, 99.0 * 1.0126, 99.0, 99.0)
    sessions = {DAYS[0]: flat(100.0), DAYS[1]: after}
    assert outcome(sessions)["label"] == 0
    result = outcome(sessions, events=[(DAYS[1], 0.99, "dividend")])
    assert result["label"] == 1 and result["exit_day"] == 1


def test_dividend_classification_uses_bhavcopy_ratio_evidence() -> None:
    bars = {"2024-01-01": (100.0,) * 4 + (1.0,), "2024-01-02": (97.0,) * 4 + (1.0,)}
    adjusted = study.classify_dividends(
        [("2024-01-02", 3.0)], bars, {"2024-01-01": 1.0309, "2024-01-02": 1.0}, []
    )[0]
    assert adjusted["kite_adjusted"] and adjusted["evidence"] == "bhavcopy_ratio"
    assert adjusted["yield"] == pytest.approx(3 / 103.09)
    raw = study.classify_dividends(
        [("2024-01-02", 3.0)], bars, {"2024-01-01": 1.0, "2024-01-02": 1.0}, []
    )[0]
    assert not raw["kite_adjusted"] and not raw["rule_agrees"]


def test_expiry_rules_switch_to_tuesday_and_skip_holidays() -> None:
    sessions = {"2025-08-27", "2025-08-28", "2025-09-29", "2025-09-30"}
    assert study.monthly_expiry(2025, 8, sessions) == "2025-08-28"
    assert study.monthly_expiry(2025, 9, sessions) == "2025-09-30"
    assert study.monthly_expiry(2025, 8, sessions - {"2025-08-28"}) == "2025-08-27"


def test_phase_purge_and_percentile_ranks() -> None:
    phases = study.assign_phases(
        np.asarray(["2025-06-27", "2025-06-30", "2025-07-01"]),
        np.asarray(["2025-06-30", "2025-07-01", "2025-07-02"]),
    )
    assert phases.tolist() == ["train", "", "validation"]
    ranks = study.percentile_ranks(np.asarray([3.0, np.nan, 1.0, 2.0]))
    assert ranks[0] == 1 and np.isnan(ranks[1]) and ranks[2] == 0 and ranks[3] == 0.5
    assert study.rsi(np.arange(1, 70, dtype=float)) == 100
