import json

import numpy as np
import pandas as pd
import pytest

from scripts import research_fno_direction as study
from scripts import research_fno_entry_timing as timing


def test_minute_prices_takes_entry_opens_and_the_exit_close(tmp_path) -> None:
    path = tmp_path / "NSE_ABC.csv"
    rows = ["start_time,open,high,low,close,volume"]
    for day in ("2024-01-01", "2024-01-02"):
        for minute in ("09:15", "09:16", "09:18", "09:20", "14:59"):
            if day == "2024-01-02" and minute == "09:16":
                continue  # a minute without trades
            price = 100 + int(minute[3:])
            rows.append(f"{day}T{minute}:00+05:30,{price},0,0,{price + 0.5},1")
    path.write_text("\n".join(rows) + "\n", "utf-8")
    out = timing.minute_prices((str(path), "ABC")).set_index("date")
    assert out.loc["2024-01-01", "09:15"] == 115
    assert out.loc["2024-01-01", "09:18"] == 118
    assert out.loc["2024-01-01", "exit"] == 159.5
    assert np.isnan(out.loc["2024-01-02", "09:16"])
    assert np.isnan(out.loc["2024-01-02", "09:25"])  # bar absent from the file
    assert set(out["kite_symbol"]) == {"ABC"}


def test_excluded_days_read_every_flag(tmp_path) -> None:
    flags = {
        "ABC": {
            "dropped": ["2024-01-01"],
            "coarse_vendor_rounding": ["2024-01-02"],
            "imprecise_fit": [],
            "minutes_outside_nse_range": ["2024-01-03"],
            "structural_break": ["2024-01-04"],
        }
    }
    (tmp_path / "exclusions.json").write_text(json.dumps(flags), "utf-8")
    assert timing.excluded_days(tmp_path) == {
        ("ABC", "2024-01-01"),
        ("ABC", "2024-01-02"),
        ("ABC", "2024-01-03"),
        ("ABC", "2024-01-04"),
    }


def book_frame() -> pd.DataFrame:
    score = np.arange(50, dtype=float)
    return pd.DataFrame(
        {
            "date": "2024-01-01",
            "symbol": [f"S{i}" for i in range(50)],
            "score": score,
            "target": score / 1000,
        }
    )


def test_book_legs_and_members_take_the_ten_extremes() -> None:
    data = book_frame()
    legs = timing.book_legs(data, "score", "target")
    assert legs.loc["2024-01-01", "long"] == pytest.approx(
        np.mean(range(40, 50)) / 1000
    )
    assert legs.loc["2024-01-01", "short"] == pytest.approx(np.mean(range(10)) / 1000)
    assert legs.loc["2024-01-01", "all"] == pytest.approx(24.5 / 1000)
    members = timing.leg_members(data, "score", "target")
    assert sorted(members.loc[members["side"] == "long", "symbol"]) == sorted(
        f"S{i}" for i in range(40, 50)
    )
    assert len(members) == 20
    # Fewer than 40 stocks: no book that session.
    assert timing.book_legs(data.head(39), "score", "target").empty
    assert timing.leg_members(data.head(39), "score", "target").empty


def test_profile_measures_the_drawdown_of_the_cumulative_sum() -> None:
    days = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01", "2025-01-01"]
    net = pd.Series([0.001, -0.003, 0.0005, 0.002, 0.001], index=days)
    out = timing.profile(net)
    assert out["discovery"]["max_drawdown_bp"] == pytest.approx(30.0)
    assert out["discovery"]["worst_bp"] == pytest.approx(-30.0)
    assert out["discovery"]["positive_months"] == "1/2"
    assert out["holdout"]["median_bp"] == pytest.approx(10.0)


def test_within_day_auc_averages_each_session() -> None:
    rows = []
    for day in range(12):
        for i in range(20):
            up = int(i >= 10)
            if day % 2 and i in (9, 10):  # odd sessions: one pair swapped
                up = 1 - up
            rows.append((f"2025-01-{day + 1:02d}", i, up))
    data = pd.DataFrame(rows, columns=["day", "score", "up"])
    out = study.within_day_auc(data["day"], data["up"], data["score"])
    assert out["sessions"] == 12
    assert out["mean"] == pytest.approx((1.0 + 0.99) / 2)
    few = study.within_day_auc(
        data["day"].head(100), data["up"].head(100), data["score"]
    )
    assert few == {"sessions": 5}
