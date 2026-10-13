from datetime import date

from scripts.validate_candle_quality import (
    action_kind,
    expected_minutes,
    factor_segments,
    paise,
)


def test_sessions_follow_normal_cas_and_special_calendars() -> None:
    assert len(expected_minutes(date(2025, 6, 2), cas=True)) == 375
    assert len(expected_minutes(date(2026, 8, 3), cas=True)) == 360
    assert len(expected_minutes(date(2026, 8, 3), cas=False)) == 375
    assert expected_minutes(date(2021, 11, 4), cas=True) == set(range(1095, 1155))
    outage = expected_minutes(date(2021, 2, 24), cas=False)
    assert 11 * 60 + 44 in outage and 12 * 60 not in outage and 16 * 60 in outage


def test_paise_prices_and_factor_segments() -> None:
    assert paise("1301.25") and paise("650") and not paise("650.625")
    assert (
        action_kind("Face Value Split (Sub-Division) - From Rs 10 To Rs 2") == "split"
    )
    assert action_kind("Demerger") == "demerger"
    assert action_kind("Interim Dividend - Rs 5 Per Share") == "dividend"
    segments = factor_segments(
        [("2024-01-01", 2.0), ("2024-02-01", 2.001), ("2024-06-03", 1.0)]
    )
    assert [(s["first"], s["last"], s["samples"]) for s in segments] == [
        ("2024-01-01", "2024-02-01", 2),
        ("2024-06-03", "2024-06-03", 1),
    ]
