"""Forward record of the opening-auction book found on Oct 16 (handover section 9:
research_fno_direction + research_fno_entry_timing). Paper only, nothing is traded.

  python -m scripts.auction_book pick    # evening, after the NSE task: the book for
                                         # the session after the latest features date
  python -m scripts.auction_book score   # every recorded book whose session's NSE
                                         # cash file has arrived

pick: the F&O stocks of the latest end-of-day date in data/fno/features, ranked by
the composite of the study's 26 kept figures with their frozen discovery signs
(KEPT_SIGNS); the 10 highest long, the 10 lowest short, Rs 2 lakh each at that
close (whole shares). Written once to data/paper/auction_book/picks_<date>.json
with the time it was made; a book made after 09:00 IST of its session is scored but
marked as not a forward record.
score: in at NSE's official open of the next session (the pre-open auction price),
out at its official close (the closing auction); Zerodha intraday charges per trade
(paper/charges.py), no slippage (both are single auction prices). The research's
equal-weight book from the official open to the official close made +40.2 bp a
session gross in Jan 2025 - Sep 2026. Results: results_<session>.json and a row of
data/paper/auction_book/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.fno.features import read_cm
from ai_trader.paper.charges import trade_charges

FOLDER = Path("data/paper/auction_book")
DAILY = Path("data/nse/daily")
STUDY_REPORT = Path("data/research/fno_direction_20261016/report.json")
CAPITAL = Decimal("200000")
SIZE = 10
MIN_SCORED = 150  # F&O stocks with a composite; NSE lists about 210
# research_fno_direction's discovery signs (2020-2024), frozen on 2026-10-16.
KEPT_SIGNS = {
    "ret_1d": -1.0,
    "vol_ratio": -1.0,
    "rv20": -1.0,
    "deliv_pct": 1.0,
    "deliv_z": 1.0,
    "fut_basis": 1.0,
    "fut_basis_ann": 1.0,
    "fut_oi_chg": -1.0,
    "buildup": -1.0,
    "fut_turnover_oi": -1.0,
    "rollover": -1.0,
    "pcr_oi": 1.0,
    "pcr_vol": 1.0,
    "pcr_oi_near": 1.0,
    "d_pcr_oi": 1.0,
    "opt_oi_flow": 1.0,
    "os_ratio": -1.0,
    "iv30": -1.0,
    "d_iv30": -1.0,
    "skew": 1.0,
    "d_skew": 1.0,
    "term": 1.0,
    "cpiv": -1.0,
    "parity_bp": -1.0,
    "mwpl": -1.0,
    "d_mwpl": -1.0,
}
RESULT_COLUMNS = [
    "session",
    "features_date",
    "forward",
    "long_bp",
    "short_bp",
    "fno_average_bp",
    "book_gross_bp",
    "gross_rs",
    "charges_rs",
    "net_rs",
    "net_bp",
    "trades",
    "missing",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(day: pd.DataFrame, size: int = SIZE) -> dict[str, list[dict]]:
    """The ``size`` highest and lowest composites of one date (ties by symbol), each
    sized to whole shares of ``CAPITAL`` at the date's close."""
    ranked = day.dropna(subset=["composite", "cm_close"])
    ranked = ranked[ranked["cm_close"] > 0].sort_values(["composite", "symbol"])

    def entries(rows: pd.DataFrame) -> list[dict]:
        return [
            {
                "symbol": row.symbol,
                "composite": round(float(row.composite), 6),
                "close": float(row.cm_close),
                "qty": int(CAPITAL // Decimal(str(row.cm_close))),
            }
            for row in rows.itertuples()
        ]

    return {
        "long": entries(ranked.tail(size).iloc[::-1]),
        "short": entries(ranked.head(size)),
    }


def next_session(features_date: str, daily: Path) -> str | None:
    """The first NSE cash file after ``features_date`` (sessions skip holidays)."""
    later = sorted(p.stem for p in daily.glob("*.csv") if p.stem > features_date)
    return later[0] if later else None


def is_forward(created_at: str, session: str) -> bool:
    made = datetime.fromisoformat(created_at)
    opening = datetime.combine(date.fromisoformat(session), time(9, 0), INDIA_TIMEZONE)
    return made < opening


def score(picks: dict, cm: dict[str, dict], session: str) -> dict:
    """Each pick in at the official open and out at the official close of
    ``session``, with Zerodha intraday charges; the equal-weight legs as research."""
    trades, missing = [], []
    for side in ("long", "short"):
        for pick in picks[side]:
            row = cm.get(pick["symbol"], {})
            entry, exit_ = row.get("cm_open", math.nan), row.get("cm_close", math.nan)
            if not (entry > 0 and exit_ > 0):
                missing.append(pick["symbol"])
                continue
            qty = Decimal(pick["qty"])
            opened, closed = Decimal(str(entry)) * qty, Decimal(str(exit_)) * qty
            buy, sell = (opened, closed) if side == "long" else (closed, opened)
            charges = trade_charges(buy, sell, same_session=True).total
            change = exit_ / entry - 1
            trades.append(
                {
                    "symbol": pick["symbol"],
                    "side": side,
                    "qty": pick["qty"],
                    "open": entry,
                    "close": exit_,
                    "return_bp": round(
                        (change if side == "long" else -change) * 1e4, 2
                    ),
                    "gross_rs": float(sell - buy),
                    "charges_rs": float(charges),
                }
            )
    universe = [cm[s] for s in picks["universe"] if s in cm]
    moves = [r["cm_close"] / r["cm_open"] - 1 for r in universe if r["cm_open"] > 0]

    def leg(side: str) -> float:
        values = [t["return_bp"] for t in trades if t["side"] == side]
        return sum(values) / len(values) if values else math.nan

    long_bp, short_bp = leg("long"), leg("short")
    gross = sum(t["gross_rs"] for t in trades)
    charges = sum(t["charges_rs"] for t in trades)
    side_capital = float(CAPITAL) * SIZE
    return {
        "session": session,
        "features_date": picks["features_date"],
        "forward": is_forward(picks["created_at_utc"], session),
        "long_bp": round(long_bp, 2),
        "short_bp": round(short_bp, 2),
        "fno_average_bp": round(sum(moves) / len(moves) * 1e4, 2) if moves else None,
        "book_gross_bp": round(long_bp + short_bp, 2),
        "gross_rs": round(gross, 2),
        "charges_rs": round(charges, 2),
        "net_rs": round(gross - charges, 2),
        "net_bp": round((gross - charges) / side_capital * 1e4, 2),
        "trades": trades,
        "missing": missing,
    }


def composite_day(features_date: str | None) -> tuple[str, pd.DataFrame, dict]:
    """The composite of every F&O stock on ``features_date`` (default: the latest
    date in data/fno/features), with the input files' hashes."""
    from scripts import research_fno_direction as study

    data = study.load_stocks(study.FEATURES_DIR)
    day = features_date or max(data["date"])
    npz = study.NORMALIZED / day[:4] / f"{day}.npz"
    if not npz.exists():
        raise SystemExit(f"no F&O session file {npz}")
    study.add_features(data)
    spreads = study.spread_session(str(npz))
    rows = data[(data["date"] == day) & (data["fno"] == 1)].merge(
        spreads[["symbol", "cpiv", "parity_bp"]], on="symbol", how="left"
    )
    rows["composite"] = study.composite(rows, KEPT_SIGNS)
    inputs = {
        str(p): sha256(p)
        for p in (study.FEATURES_DIR / f"stock_daily_{day[:4]}.csv.gz", npz)
    }
    return day, rows, inputs


def pick(folder: Path, features_date: str | None) -> int:
    if STUDY_REPORT.exists():
        report = json.loads(STUDY_REPORT.read_text("utf-8"))
        if report["kept_signs"] != KEPT_SIGNS:
            raise SystemExit(f"{STUDY_REPORT} kept_signs differ from KEPT_SIGNS")
    day, rows, inputs = composite_day(features_date)
    path = folder / f"picks_{day}.json"
    if path.exists():
        print(f"book for the session after {day} already recorded: {path}")
        return 0
    scored = int(rows["composite"].notna().sum())
    if scored < MIN_SCORED:
        print(f"{day}: only {scored} F&O stocks have a composite; no book recorded")
        return 1
    created = datetime.now(UTC)
    record = {
        "created_at_utc": created.isoformat(timespec="seconds"),
        "created_at_ist": created.astimezone(INDIA_TIMEZONE).isoformat(
            timespec="seconds"
        ),
        "features_date": day,
        "rule": "10 highest composite long, 10 lowest short, Rs 2 lakh each at the "
        "features date's close; in at the next official open, out at its close",
        "signs": KEPT_SIGNS,
        "study_report_sha256": sha256(STUDY_REPORT) if STUDY_REPORT.exists() else None,
        "inputs_sha256": inputs,
        "fno_stocks": int(len(rows)),
        "with_composite": scored,
        "universe": sorted(rows["symbol"]),
        **choose(rows),
    }
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), "utf-8")
    print(f"{created.astimezone(INDIA_TIMEZONE):%Y-%m-%d %H:%M} IST book after {day}:")
    for side in ("long", "short"):
        names = ", ".join(f"{p['symbol']} x{p['qty']}" for p in record[side])
        print(f"  {side}: {names}")
    return 0


def score_all(folder: Path, daily: Path) -> int:
    for path in sorted(folder.glob("picks_*.json")):
        picks = json.loads(path.read_text("utf-8"))
        session = next_session(picks["features_date"], daily)
        if session is None:
            print(f"{path.name}: its session's NSE file has not arrived")
            continue
        out = folder / f"results_{session}.json"
        if out.exists():
            continue
        cm = read_cm((daily / f"{session}.csv").read_text("utf-8"))
        result = score(picks, cm, session)
        out.write_text(json.dumps(result, indent=1), "utf-8")
        table = folder / "results.csv"
        new = not table.exists()
        with table.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(RESULT_COLUMNS)
            writer.writerow(
                [
                    len(result[c]) if c in ("trades", "missing") else result[c]
                    for c in RESULT_COLUMNS
                ]
            )
        print(
            f"{session}: long {result['long_bp']} bp, short {result['short_bp']} bp, "
            f"book {result['book_gross_bp']} bp gross; Rs {result['net_rs']:+,.0f} "
            f"net ({result['net_bp']} bp of one side)"
            + ("" if result["forward"] else " [made after the open: not forward]")
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pick", "score"))
    parser.add_argument("--features-date", help="pick: this end-of-day date")
    parser.add_argument("--folder", type=Path, default=FOLDER)
    parser.add_argument("--daily", type=Path, default=DAILY)
    args = parser.parse_args(argv)
    if args.command == "pick":
        return pick(args.folder, args.features_date)
    return score_all(args.folder, args.daily)


if __name__ == "__main__":
    raise SystemExit(main())
