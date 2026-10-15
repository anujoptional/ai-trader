from datetime import date

import numpy as np
import pandas as pd
import pytest

from ai_trader.fno.iv import black76
from scripts import research_fno_direction as study


def frame(rows):
    data = pd.DataFrame(rows, columns=["symbol", "session", "value"])
    return data.sort_values(["symbol", "session"], ignore_index=True)


def test_neighbour_takes_only_the_very_next_or_previous_session() -> None:
    data = frame([("A", 0, 1.0), ("A", 1, 2.0), ("A", 3, 4.0), ("B", 1, 9.0)])
    nxt = study.neighbour(data, ["value"], 1)["value"].tolist()
    assert nxt[0] == 2.0 and np.isnan(nxt[1]) and np.isnan(nxt[2]) and np.isnan(nxt[3])
    prev = study.neighbour(data, ["value"], -1)["value"].tolist()
    assert np.isnan(prev[0]) and prev[1] == 1.0 and np.isnan(prev[2])


def test_targets_use_the_next_session_and_drop_corporate_action_jumps() -> None:
    data = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "session": [0, 1, 2] * 2,
            "cm_open": [100.0, 101.0, 50.0, 10.0, 10.0, 11.0],
            "cm_close": [100.0, 102.0, 51.0, 10.0, 10.5, 11.0],
            # A splits 1:2 on day 3: NSE's previous close is not adjusted.
            "cm_prev_close": [99.0, 100.0, 102.0, 10.0, 10.0, 10.5],
        }
    )
    indices = pd.DataFrame(
        {"nifty_open": [1.0, 1.0, 2.0], "nifty_close": [1.0, 1.1, 2.0]},
        index=["2024-01-01", "2024-01-02", "2024-01-03"],
    )
    study.add_targets(data, indices)
    a = data[data["symbol"] == "A"]
    assert a["oc_next"].iloc[0] == pytest.approx(102 / 101 - 1)
    assert a["cc_next"].iloc[0] == pytest.approx(0.02)
    assert np.isnan(a["cc_next"].iloc[1]) and np.isnan(a["gap_next"].iloc[1])
    assert a["oc_next"].iloc[1] == pytest.approx(51 / 50 - 1)  # same-day: still fine
    assert np.isnan(a["oc_next"].iloc[2])  # no next session
    assert data["nifty_oc_next"].iloc[0] == pytest.approx(0.1)


def test_daily_ic_and_quintile_spread_find_a_planted_signal() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for day in range(60):
        signal = rng.normal(size=100)
        target = 0.3 * signal + rng.normal(size=100)
        rows += [
            (f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}", s, t)
            for s, t in zip(signal, target, strict=True)
        ]
    data = pd.DataFrame(rows, columns=["date", "x", "y"])
    ic = study.daily_ic(data, "x", "y")
    assert len(ic) == 60 and 0.2 < ic.mean() < 0.4
    assert study.newey_west_t(ic.to_numpy()) > 10
    assert study.quintile_spread(data, "x", "y").mean() > 0.5
    noise = data.assign(y=rng.normal(size=len(data)))
    assert abs(study.daily_ic(noise, "x", "y").mean()) < 0.05


def test_the_keep_rule_needs_strength_and_yearly_consistency() -> None:
    years = {y: 0.02 for y in study.DISCOVERY_YEARS}
    assert study.kept({"days": 1000, "mean": 0.02, "t_nw": 4.0, "by_year": years})
    flipped = years | {"2023": -0.01, "2024": -0.01}
    assert not study.kept({"days": 1000, "mean": 0.02, "t_nw": 4.0, "by_year": flipped})
    assert not study.kept({"days": 1000, "mean": 0.02, "t_nw": 2.0, "by_year": years})


def test_book_returns_long_minus_short_after_charges() -> None:
    data = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 40,
            "score": np.arange(40.0),
            "ret": np.r_[np.full(10, -0.01), np.zeros(20), np.full(10, 0.02)],
        }
    )
    net = study.book(data, "score", "ret", size=10)
    assert net.iloc[0] == pytest.approx(0.03 - 2 * study.COST)


def option_table(day: date, forward: float, call_vol: float, put_vol: float) -> dict:
    expiry = np.datetime64("2024-07-25", "D")
    years = (expiry - np.datetime64(day, "D")).astype(int) / 365
    strikes = [97.5, 100.0, 102.5, 130.0]
    rows = [("ABC", "F", "", np.nan, forward)]
    for k in strikes:
        rows.append(
            ("ABC", "O", "CE", k, float(black76(forward, k, years, call_vol, True)))
        )
        rows.append(
            ("ABC", "O", "PE", k, float(black76(forward, k, years, put_vol, False)))
        )
    n = len(rows)
    return {
        "symbol": np.array([r[0] for r in rows]),
        "kind": np.array([r[1] for r in rows]),
        "underlying": np.array(["S"] * n),
        "option": np.array([r[2] for r in rows]),
        "strike": np.array([r[3] for r in rows]),
        "expiry": np.array([expiry] * n),
        "close": np.array([r[4] for r in rows]),
        "settle": np.array([r[4] for r in rows]),
        "contracts": np.ones(n, dtype=np.int64),
        "oi": np.full(n, 100, dtype=np.int64),
    }


def test_call_put_spread_is_zero_at_parity_and_positive_for_dear_calls() -> None:
    day = date(2024, 7, 5)
    fair = study.call_put_spread(option_table(day, 100.0, 0.3, 0.3), day)
    assert fair["pairs"].iloc[0] == 3  # the 130 strike is outside the 5% band
    assert abs(fair["cpiv"].iloc[0]) < 1e-9 and abs(fair["parity_bp"].iloc[0]) < 1e-6
    dear = study.call_put_spread(option_table(day, 100.0, 0.35, 0.3), day)
    assert dear["cpiv"].iloc[0] == pytest.approx(0.05, abs=1e-9)
    assert dear["parity_bp"].iloc[0] > 0
    near_expiry = date(2024, 7, 23)  # under 3 days: no expiry qualifies
    assert study.call_put_spread(
        option_table(near_expiry, 100.0, 0.3, 0.3), near_expiry
    ).empty


def test_composite_signs_and_centres_daily_ranks() -> None:
    data = pd.DataFrame(
        {"date": ["d"] * 4, "a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}
    )
    both = study.composite(data, {"a": 1.0, "b": -1.0})
    assert both.tolist() == pytest.approx([-0.375, -0.125, 0.125, 0.375])
