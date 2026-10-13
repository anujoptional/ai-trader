"""Scanner-filter study: rules that concentrate quick LONG opportunities (research).

Slots: 50 stocks, 5-minute decisions 09:40-13:30 on verified Kite minutes (2024-2026).
Opportunity: +0.3% touched before -0.3% by 15:00 (same-minute touches = stop). Each
rule is scored as a filter: candidates per day, precision, lift over all slots,
recall of all opportunities, and a same-day exit proxy (TP +0.3% else 15:00 close).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from ai_trader.costs import ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore
from scripts.research_exit_policy import (
    EPOCH,
    FIRST_DECISION,
    LEVELS,
    NONE,
    SLOTS,
    STEP,
    build_universe,
    epoch_day,
)
from scripts.research_intraday_strategy import read_sessions
from scripts.train_tp_only import EXPECTED_COLUMNS, file_hash, write_json

LEVEL = 30
FIT_END = "2025-06-30"
PERIODS = (
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2025H2", "2025-07-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-09-30"),
)
CLIPS = (100_000, 500_000)


def nifty_moves(candles: Path) -> dict[int, np.ndarray]:
    store = CandleStore(candles)
    quality = json.loads(store.verified_path("report.json").read_text("utf-8"))
    quarantined = set(
        quality["instruments"]["NSE:NIFTY"]["quality"]["quarantined_dates"]
    )
    sessions = read_sessions(store.verified_path("NSE_NIFTY.csv"), quarantined)
    decisions = FIRST_DECISION + STEP * np.arange(SLOTS)
    return {
        epoch_day(day): prices[decisions, 3] / prices[0, 0] - 1
        for day, (prices, _) in sessions.items()
    }


def rule_masks(feature: dict[str, np.ndarray], low: float, high: float) -> dict:
    up_market = feature["nifty_day"] > 0
    strong = (feature["day_return"] > feature["nifty_day"]) & (feature["vwap_z"] > 0)
    aligned = up_market & strong
    moving = feature["realized_volatility_20m"] >= high
    return {
        "all_slots": np.ones_like(up_market),
        "low_movement": feature["realized_volatility_20m"] <= low,
        "high_movement": moving,
        "nifty_up": up_market,
        "nifty_down": feature["nifty_day"] < 0,
        "strong_vs_nifty_above_vwap": strong,
        "aligned": aligned,
        "aligned_high_movement": aligned & moving,
        "aligned_rvol_1_5": aligned & (feature["same_minute_rvol_5"] >= 1.5),
        "aligned_high_movement_rvol_1_5": aligned
        & moving
        & (feature["same_minute_rvol_5"] >= 1.5),
        "dip_in_up_market": up_market & (feature["vwap_z"] <= -1.5),
        "weak_in_down_market": (feature["nifty_day"] < 0)
        & (feature["day_return"] < feature["nifty_day"])
        & (feature["vwap_z"] < 0),
    }


def score(mask, label, quick, moved, outcome, days, sessions) -> dict:
    chosen = mask & np.isfinite(outcome)
    if not chosen.any():
        return {"candidates": 0}
    base = label[np.isfinite(outcome)].mean()
    result = {
        "candidates": int(chosen.sum()),
        "candidates_per_day": float(chosen.sum() / sessions),
        "precision_percent": 100 * float(label[chosen].mean()),
        "lift": float(label[chosen].mean() / base),
        "recall_percent": 100
        * float(label[chosen].sum() / label[np.isfinite(outcome)].sum()),
        "quick_tp_60m_percent": 100 * float(quick[chosen].mean()),
        "moved_0_3_percent": 100 * float(moved[chosen].mean()),
        "proxy_gross_percent": 100 * float(outcome[chosen].mean()),
    }
    for clip in CLIPS:
        cost = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(Decimal(clip)))
        result[f"proxy_net_percent_{clip // 1000}k"] = 100 * float(
            outcome[chosen].mean() - cost
        )
    daily = [label[chosen & (days == d)].mean() for d in np.unique(days[chosen])]
    result["days_with_candidates"] = len(daily)
    result["day_precision_p25_p75"] = np.percentile(daily, (25, 75)).tolist()
    return result


def run(data: Path, candles: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    output.mkdir(parents=True)
    universe, coverage = build_universe(data, candles, None)
    level = LEVELS.index(LEVEL)
    up = universe.up_minutes[..., level].astype(np.int64)
    down = universe.down_minutes[..., level].astype(np.int64)
    available = universe.available
    label = (up < down) & (up < NONE)
    quick = up <= 60
    moved = np.minimum(up, down) < NONE
    outcome = np.where(up < NONE, LEVEL / 10_000, universe.time_return[..., -1])
    outcome = np.where(available, outcome, np.nan)
    nifty = nifty_moves(candles)
    nifty_day = np.stack(
        [nifty.get(int(day), np.full(SLOTS, np.nan)) for day in universe.days]
    )
    long_rows = universe.features[:, :, 0, :]
    feature = {name: long_rows[..., i] for i, name in enumerate(EXPECTED_COLUMNS)}
    feature["nifty_day"] = nifty_day
    days = np.repeat(universe.days[:, None], SLOTS, axis=1)
    fit = available & (days <= epoch_day(FIT_END))
    low, high = np.nanquantile(feature["realized_volatility_20m"][fit], (0.3, 0.7))
    masks = rule_masks(feature, float(low), float(high))
    usable = available & np.isfinite(nifty_day)
    report_rules = {}
    for name, mask in masks.items():
        report_rules[name] = {}
        for period, first, last in PERIODS:
            inside = usable & (days >= epoch_day(first)) & (days <= epoch_day(last))
            sessions = len(np.unique(days[inside]))
            report_rules[name][period] = score(
                mask & inside,
                np.where(inside, label, False),
                quick,
                moved,
                np.where(inside, outcome, np.nan),
                days,
                sessions,
            )
    side_check = float(
        np.nanmax(
            np.abs(universe.features[..., 0, :13] - universe.features[..., 1, :13])
        )
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "research_complete_not_promoted",
        "opportunity": f"LONG +{LEVEL / 100}% before -{LEVEL / 100}% by 15:00",
        "proxy": "TP +0.3% else exit at the 14:59-15:00 close; intraday charges",
        "movement_thresholds_fit_2024_01_2025_06": [float(low), float(high)],
        "features_identical_across_sides": side_check == 0.0,
        "coverage": {k: v for k, v in coverage.items() if k != "symbols"},
        "first_day": date.fromordinal(EPOCH + int(universe.days.min())).isoformat(),
        "rules": report_rules,
        "source_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data, args.candles, args.output)
    print(json.dumps({"rules": len(report["rules"]), "first_day": report["first_day"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
