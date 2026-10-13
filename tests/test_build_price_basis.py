from datetime import date

import numpy as np
import pytest

from scripts.build_price_basis import (
    event_ratio,
    fit_day,
    nse_symbol,
    quantum,
    snap,
    symbol_history,
    tolerance,
)


def test_day_fit_recovers_multiplicative_and_additive_adjustments() -> None:
    nse = np.array([300.0, 315.0, 290.0, 310.0])
    a, c, residual = fit_day(nse * 0.9218 - 4.0, nse, None)
    assert a == pytest.approx(0.9218) and c == pytest.approx(-4.0)
    assert residual < 1e-9
    # Rounded to a one-rupee vendor quantum, the fit stays within tolerance.
    rounded = np.round(nse / 3)
    a, c, residual = fit_day(rounded, nse, None)
    assert residual <= tolerance(1.0, rounded[3])
    assert (rounded - c) / a == pytest.approx(nse, abs=1.5)
    # A flat day borrows the previous slope instead of an ill-posed fit.
    flat = np.array([100.0, 100.05, 99.95, 100.0])
    a, c, residual = fit_day(flat * 0.5, flat, 0.5)
    assert a == 0.5 and residual < 1e-9


def test_quantum_and_snap() -> None:
    assert quantum(["333", "336.0", "320"]) == 1.0
    assert quantum(["101.05", "101.10", "100.95"]) == 0.05
    assert quantum(["101.123456"]) == 0.0
    assert snap(0.3341) == (pytest.approx(1 / 3), True)
    assert snap(0.62)[1] is False


def test_event_ratio_measures_split_and_flags_unadjusted_breaks() -> None:
    rng = np.random.default_rng(0)
    days = [f"2024-01-{d:02d}" for d in range(1, 31)]
    nse, kite, steps = {}, {}, {}
    for i, day in enumerate(days):
        level = 1000 * (1 + 0.01 * np.sin(i)) * (0.2 if day >= "2024-01-16" else 1)
        bar = level * np.array([1.0, 1.02, 0.98, 1.0 + rng.normal(0, 0.005)])
        nse[day] = bar
        # Kite: back-adjusted by the 1:5 split and rounded to a 0.05 quantum.
        kite[day] = np.round(bar * (0.2 if day < "2024-01-16" else 1) / 0.05) * 0.05
        steps[day] = 0.05
    event = event_ratio("2024-01-16", days, kite, nse, steps)
    assert event["status"] == "adjusted" and event["snapped"]
    assert event["ratio"] == pytest.approx(0.2)
    unadjusted = {d: np.round(nse[d] / 0.05) * 0.05 for d in days}
    event = event_ratio("2024-01-16", days, unadjusted, nse, steps)
    assert event["status"] == "kite_unadjusted"


def test_symbol_history_follows_rename_chains(tmp_path) -> None:
    path = tmp_path / "changes.csv"
    path.write_text(
        "Mindtree merger,LTI,LTIM,14-NOV-2022\nLTM Limited,LTIM,LTM,27-FEB-2026\n",
        encoding="utf-8",
    )
    history = symbol_history(path)
    assert nse_symbol("LTM", date(2026, 3, 2), history) == "LTM"
    assert nse_symbol("LTM", date(2024, 1, 2), history) == "LTIM"
    assert nse_symbol("LTM", date(2021, 1, 4), history) == "LTI"
