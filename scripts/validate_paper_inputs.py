"""Validate the live paper session's AI inputs against independent sources.

Reads the raw inputs each decision cycle captured (``data/paper/capture/<day>/``),
the history cache the session used, the AI journals, and checks them against:

- NSE's official end-of-day files (closes; futures OI of the previous session and,
  when their F&O files are in the same folder, of earlier sessions),
- NSE's index API (previous closes of NIFTY 50 and INDIA VIX),
- NSE's announcements API, fetched afresh for the last week (filings shown),
- the research feature code (``scripts/research_delivery_pipeline.py`` and the
  movement inputs of ``scripts/research_scanner_move.py``),
- Kite's own complete bars fetched after the close (bars used live were final?),
- internal consistency of every order-book snapshot,
- the v3 requests and replies in the journals (scanner output absent, values at
  the current price recomputed here, web searches),
- a dry run of a long and a short on every captured order book: fills, targets
  and exits (target, else 15:00) recomputed by hand.

Writes ``report.json`` and prints a summary. Read-only: no orders, no Kite login.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import re
import sqlite3
import time as clock
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ai_trader.ai.packet import V3_DAILY, V3_MARKET, V3_OI, V3_TODAY
from ai_trader.broker import DepthLevel, DepthSnapshot, Instrument
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.news.features import WEEK_DAYS, routine
from ai_trader.news.nse import NseClient
from ai_trader.news.store import disseminated_at
from ai_trader.scanner.upside import AUCTION_FROM, LAST, LOOKBACK, fill_quiet_minutes
from scripts.research_delivery_pipeline import (
    STOCK_FEATURES,
    daily_columns,
    session_features,
    wilder_rsi,
)

FIELDS = ("open", "high", "low", "close", "volume")
PROMPTS = (
    "(prompt_version like 'delivery-v3%' or prompt_version like 'delivery-v4%'"
    " or prompt_version like 'direction-%')"
)
EXIT_MINUTE = 345  # 15:00, minutes from 09:15
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
URL_DATES = (
    re.compile(r"(?<!\d)(20\d\d)[/_-](\d\d)[/_-](\d\d)(?!\d)"),
    re.compile(r"(?<!\d)(20\d\d)(\d\d)(\d\d)(?!\d)"),
)
URL_WORD_DATE = re.compile(
    r"(?<![a-z\d])(\d{1,2})-(" + "|".join(MONTHS) + r")[a-z]*-(20\d\d)"
    r"|(?<![a-z])(" + "|".join(MONTHS) + r")[a-z]*-(\d{1,2})-(20\d\d)"
)


def url_date(url: str, today: date) -> date | None:
    """A publication date written in a URL (e.g. /2026/10/14/, oct-14-2026)."""
    text = url.lower()
    found: list[date] = []
    for pattern in URL_DATES:
        for y, m, d in pattern.findall(text):
            try:
                found.append(date(int(y), int(m), int(d)))
            except ValueError:
                pass
    for d1, m1, y1, m2, d2, y2 in URL_WORD_DATE.findall(text):
        day_text, month, year = (d1, m1, y1) if d1 else (d2, m2, y2)
        try:
            found.append(date(int(year), MONTHS.index(month[:3]) + 1, int(day_text)))
        except ValueError:
            pass
    found = [d for d in found if date(2000, 1, 1) <= d <= today]
    return max(found) if found else None


def source_ages(urls: list[str], today: date) -> dict:
    ages = [(today - d).days for u in urls if (d := url_date(u, today)) is not None]
    return {
        "sources": len(urls),
        "dated_in_url": len(ages),
        "today": sum(a == 0 for a in ages),
        "within_7_days": sum(a <= 7 for a in ages),
        "older_than_30_days": sum(a > 30 for a in ages),
    }


def bhavcopy(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fetch_nse_files(folder: Path, sessions: list[date]) -> list[str]:
    """NSE's official cash bhavcopy of the last session and F&O bhavcopies of the
    last six (the 5-session OI change), from NSE's public archive into ``folder``;
    files already there are kept. Returns what could not be fetched."""
    folder.mkdir(parents=True, exist_ok=True)
    wanted = [("cm", sessions[-1])] + [("fo", d) for d in sessions[-6:]]
    problems = []
    for segment, day in wanted:
        name = f"BhavCopy_NSE_{segment.upper()}_0_0_0_{day:%Y%m%d}_F_0000.csv"
        if (folder / name).exists():
            continue
        url = f"https://nsearchives.nseindia.com/content/{segment}/{name}.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            blob = urllib.request.urlopen(request, timeout=60).read()
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                (folder / name).write_bytes(archive.read(archive.namelist()[0]))
        except Exception as error:
            problems.append(f"{name}: {type(error).__name__}")
    return problems


def fo_bhavcopies(folder: Path) -> dict[date, dict[str, dict[str, str]]]:
    """Stock-futures rows of every NSE F&O bhavcopy in ``folder``, by trade date."""
    days = {}
    for path in folder.glob("BhavCopy_NSE_FO_0_0_0_*_F_0000.csv"):
        stamp = path.name.split("_")[6]
        days[date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))] = {
            r["FinInstrmNm"]: r for r in bhavcopy(path) if r["FinInstrmTp"] == "STF"
        }
    return days


def quote_mid(raw: dict) -> float:
    if raw["bids"] and raw["asks"]:
        return (float(raw["bids"][0][0]) + float(raw["asks"][0][0])) / 2
    return float(raw["last_price"])


def nse_oi_change(
    fo_days: dict[date, dict[str, dict[str, str]]],
    sessions: list[date],
    symbol: str,
    contracts: list[str],
    back: int,
) -> float | str | None:
    """Total OI change of ``contracts`` over the last ``back`` sessions by NSE's
    files; None when the window starts before the last expiry (a month then
    trading has expired since), as the session withholds it."""
    start, end = sessions[-1 - back], sessions[-1]
    if start not in fo_days or end not in fo_days:
        return f"no NSE F&O file for {start if start not in fo_days else end}"
    then = [r for r in fo_days[start].values() if r["TckrSymb"] == symbol]
    near = min(then, key=lambda r: r["XpryDt"])
    if near["FinInstrmNm"] != contracts[0] and near["XpryDt"] != start.isoformat():
        return None
    totals = [
        sum(int(float(fo_days[d][c]["OpnIntrst"])) for c in contracts)
        for d in (start, end)
    ]
    return totals[1] / totals[0] - 1


def expected_now(
    daily: np.ndarray,
    bars: np.ndarray,
    minute: int,
    nifty_closes: list[float] | None,
    nifty_now: float,
) -> dict[str, float | None]:
    """The request's values at the current price, recomputed here in percent from
    the cached daily history and the captured minute bars."""
    filled = fill_quiet_minutes(bars, minute)
    if filled is None:
        return {}
    price = float(filled[minute][3])
    high = max(float(row[1]) for row in filled[: minute + 1])
    closes = [float(c) for c in daily[:, 3]]
    highs = [float(h) for h in daily[:, 1]]
    out: dict[str, float | None] = {}
    for label, back in (("today", 1), ("5d", 5), ("20d", 20), ("60d", 60)):
        out[f"change_{label}_pct"] = (
            (price / closes[-back] - 1) * 100 if len(closes) >= back else None
        )
    for days in (50, 200):
        out[f"vs_{days}d_avg_pct"] = (
            (price / (sum(closes[-days:]) / days) - 1) * 100
            if len(closes) >= days
            else None
        )
    out["vs_250d_high_pct"] = (
        (price / max(*highs[-250:], high) - 1) * 100 if len(highs) >= 250 else None
    )
    if nifty_closes:
        nifty_20d = (nifty_now / nifty_closes[-20] - 1) * 100
        out["nifty_change_20d_pct"] = nifty_20d
        out["nifty_vs_50d_avg_pct"] = (
            nifty_now / (sum(nifty_closes[-50:]) / 50) - 1
        ) * 100
        change = out["change_20d_pct"]
        out["vs_nifty_20d_pct"] = None if change is None else change - nifty_20d
    return out


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def snapshot_from(raw: dict, symbol: str) -> DepthSnapshot:
    def levels(rows):
        return tuple(DepthLevel(Decimal(p), int(q), int(o)) for p, q, o in rows)

    return DepthSnapshot(
        instrument=Instrument("NSE", symbol),
        timestamp=datetime.fromisoformat(raw["timestamp"]),
        last_price=Decimal(raw["last_price"]),
        last_trade_at=datetime.fromisoformat(raw["last_trade_at"])
        if raw["last_trade_at"]
        else None,
        volume=raw["volume"],
        total_buy_quantity=raw["total_buy_quantity"],
        total_sell_quantity=raw["total_sell_quantity"],
        bids=levels(raw["bids"]),
        asks=levels(raw["asks"]),
        previous_close=Decimal(raw["previous_close"]),
        lower_circuit=Decimal(raw["lower_circuit"]) if raw["lower_circuit"] else None,
        upper_circuit=Decimal(raw["upper_circuit"]) if raw["upper_circuit"] else None,
        open_interest=raw["open_interest"],
    )


@lru_cache(maxsize=8)
def recorded_basis(folder: Path) -> dict:
    path = folder / "basis.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def session_history(folder: Path, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """The minute sessions and daily rows the session used: Kite's cache, on the
    NSE price basis it recorded (``basis.npz``/``basis.json``) when there is one."""
    with np.load(folder / f"{symbol}.npz") as saved:
        sessions, daily = saved["sessions"].copy(), saved["daily"].copy()
        dates = [date.fromordinal(int(d)) for d in saved.get("session_dates", [])]
    if (folder / "basis.npz").exists():
        with np.load(folder / "basis.npz") as saved:
            daily = saved[symbol].copy()
        stock = recorded_basis(folder).get("stocks", {}).get(symbol, {})
        scales = stock.get("volume_scale", {})
        for session, day in zip(sessions, dates, strict=False):
            session[:, 4] *= scales.get(day.isoformat(), 1.0)
    return sessions, daily


def research_stock_features(
    bars: np.ndarray,
    minute: int,
    folder: Path,
    symbol: str,
    day: date,
    fno: bool,
    move: float | None,
) -> dict[str, float]:
    """The stock features at ``minute`` by the research code, from raw inputs."""
    from scripts import research_scanner_move as research_move

    sessions, daily = session_history(folder, symbol)
    filled = fill_quiet_minutes(bars, minute, max_share=1.0)
    bars = bars if filled is None else filled
    padded = np.full((max(LAST + 1, minute + 1), 5), np.nan)
    padded[: minute + 1] = bars[: minute + 1]
    decision = np.array([minute])
    part = session_features(padded, decision)
    history = [session_features(s, decision) for s in sessions[-LOOKBACK:]]
    columns = daily_columns(
        daily,
        len(daily),
        wilder_rsi(daily[:, 3]),
        [h["day_rv"] for h in history],
        [np.atleast_1d(h["cumvol"]) for h in history],
        part,
        slots=1,
    )
    values = dict(zip(STOCK_FEATURES, columns[0], strict=True))
    values["slot"] = (360 if fno and day >= AUCTION_FROM else 375) - 1 - minute
    if move is not None:
        extra = research_move.extra_columns(padded, decision)[0]
        values |= dict(zip(research_move.RAW_EXTRA, extra, strict=True))
        names = research_move.NAMES
        grid = np.array(
            [[np.float32(values.get(name, np.nan)) for name in names]], np.float32
        )
        research_move.movement_scales(grid, move)
        for name in ("z_hist", "z_open", "z_atr"):
            values[name] = grid[0, names.index(name)]
    return {k: float(v) for k, v in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default=date.today().isoformat())
    parser.add_argument(
        "--paper", type=Path, default=Path("data/paper"), help="captures and cache"
    )
    parser.add_argument(
        "--journals",
        type=Path,
        default=Path("data/paper/intraday"),
        help="folder of decisions.sqlite and shadows/*/decisions.sqlite",
    )
    parser.add_argument(
        "--scanner", type=Path, default=Path("data/models/scanner_move10_by_close_v1")
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--independent",
        action="store_true",
        help="also compare prices sent to the AI with Yahoo Finance minute bars",
    )
    parser.add_argument(
        "--nse",
        type=Path,
        default=None,
        help="bhavcopy folder (default <output>/nse; missing files are downloaded)",
    )
    parser.add_argument(
        "--previous",
        default=None,
        help="previous session date (default: the last NSE daily file before --day)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report folder (default data/validation_<YYYYMMDD>/auto)",
    )
    args = parser.parse_args()
    bundle = json.loads((args.scanner / "bundle.json").read_text("utf-8"))
    move = bundle["move_bps"] / 10_000 if "move_bps" in bundle else None
    day = date.fromisoformat(args.day)
    args.output = args.output or Path(f"data/validation_{day:%Y%m%d}") / "auto"
    folder = args.paper / "capture" / args.day
    captures = {
        p.stem: json.loads(p.read_text("utf-8"))
        for p in sorted(folder.glob("[0-9]*.json"))
    }
    if not captures:
        print(f"no captured decisions for {args.day}: nothing to validate")
        return 0
    sessions = sorted(
        d
        for p in Path("data/nse/daily").glob("*.csv")
        if (d := date.fromisoformat(p.stem)) < day
    )
    args.previous = args.previous or sessions[-1].isoformat()
    args.nse = args.nse or args.output / "nse"
    for note in fetch_nse_files(
        args.nse, [s for s in sessions if s <= date.fromisoformat(args.previous)]
    ):
        print("NSE file not fetched:", note)
    previous = args.previous.replace("-", "")
    previous_day = date.fromisoformat(args.previous)
    final_path = folder / "final_bars.json"
    final = (
        json.loads(final_path.read_text("utf-8"))["bars"] if final_path.exists() else {}
    )
    cache = args.paper / "cache" / args.day
    report: dict = {"day": args.day, "cycles": sorted(captures)}

    # NSE references -----------------------------------------------------------------
    cm = {
        r["TckrSymb"]: r
        for r in bhavcopy(args.nse / f"BhavCopy_NSE_CM_0_0_0_{previous}_F_0000.csv")
        if r["SctySrs"] in ("EQ", "BE", "BZ", "SM", "ST")
    }
    fo_days = fo_bhavcopies(args.nse)
    fo = fo_days[previous_day]
    nse = NseClient(timeout_seconds=15, attempts=2)
    indices = {
        r["index"]: r for r in nse.get_json("/api/allIndices", {}).get("data", [])
    }
    announcements = [
        a
        for back in range(WEEK_DAYS, -1, -1)
        for a in nse.announcements(
            day - timedelta(days=back), day - timedelta(days=back)
        )
    ]

    # 1. bars: completeness, sanity, finality -------------------------------------
    bars_report = Counter()
    revisions = []
    for name, cap in captures.items():
        minute = cap["minute"]
        for symbol, info in cap["stocks"].items():
            rows = np.array(info["bars"], dtype=float)
            bars_report["stock_cycles"] += 1
            bars_report["missing_rows"] += int(np.isnan(rows[:, 3]).sum())
            o, h, low, c, v = rows.T
            bad = (
                (low > np.minimum(o, c) + 1e-9)
                | (h < np.maximum(o, c) - 1e-9)
                | (v < 0)
            )
            bars_report["ohlc_violations"] += int(np.nansum(bad))
            reference = final.get(symbol)
            if reference is None:
                later = [
                    c["stocks"][symbol]["bars"]
                    for k, c in sorted(captures.items())
                    if k > name and symbol in c["stocks"]
                ]
                reference = later[-1] if later else None
                if reference is not None:
                    bars_report["checked_against_later_capture"] += 1
            if reference is None:
                continue
            ref = np.array(reference, dtype=float)[: minute + 1]
            differ = ~np.all(
                np.isclose(rows, ref, rtol=0, atol=1e-9, equal_nan=True), axis=1
            )
            bars_report["rows_compared"] += len(rows)
            bars_report["rows_revised"] += int(differ.sum())
            if differ[-1]:
                bars_report["decision_bar_revised"] += 1
            if differ.any():
                worst = np.nanmax(np.abs(rows - ref), axis=0)
                revisions.append(
                    {
                        "cycle": name,
                        "symbol": symbol,
                        "revised_minutes": np.flatnonzero(differ).tolist()[:10],
                        "max_abs_diff": dict(zip(FIELDS, worst.round(4).tolist())),
                    }
                )
    report["bars"] = dict(bars_report) | {
        "final_bars_available": bool(final),
        "revisions": revisions[:30],
        "fetch_seconds": [
            round(
                (
                    datetime.fromisoformat(c["fetch_finished"])
                    - datetime.fromisoformat(c["decision_at"])
                ).total_seconds()
            )
            for c in captures.values()
        ],
        "skipped": {k: c["skipped"] for k, c in captures.items()},
        "stocks_missing_minutes": {
            k: dict(list(c["missing_minutes"].items())[:8]) for k, c in captures.items()
        },
    }

    # 2. features: live vs research code on the same raw inputs; prev close vs NSE ----
    worst_feature, feature_rows, close_checks = {}, 0, Counter()
    close_mismatches = []
    for name, cap in captures.items():
        minute = cap["minute"]
        for symbol, info in cap["stocks"].items():
            path = cache / f"{symbol}.npz"
            if not path.exists():
                continue
            rows = np.array(info["bars"], dtype=float)
            research = research_stock_features(
                rows, minute, cache, symbol, day, info["fno"], move
            )
            live = info["features"]
            feature_rows += 1
            for key, value in research.items():
                if key not in live or live[key] is None or not np.isfinite(value):
                    continue
                diff = relative(live[key], value)
                worst_feature[key] = max(worst_feature.get(key, 0.0), diff)
            with np.load(path) as saved:
                kite_close = float(saved["daily"][-1, 3])
                kite_date = date.fromordinal(int(saved["dates"][-1]))
            official = cm.get(symbol)
            close_checks["checked"] += 1
            if official is None:
                close_checks["not_in_bhavcopy"] += 1
            elif (
                kite_date.strftime("%Y%m%d") != previous
                or abs(kite_close - float(official["ClsPric"])) > 0.001
            ):
                close_checks["mismatch"] += 1
                close_mismatches.append(
                    (symbol, kite_date.isoformat(), kite_close, official["ClsPric"])
                )
            else:
                close_checks["match"] += 1
    report["features"] = {
        "stock_cycles_recomputed": feature_rows,
        "max_relative_difference": {
            k: float(f"{v:.3g}") for k, v in sorted(worst_feature.items())
        },
        "previous_close_vs_nse": dict(close_checks),
        "previous_close_mismatches": close_mismatches[:20],
        "live_prescreen_vs_exact": prescreen_gaps(captures),
    }

    # 3. market inputs: from captured index bars and NSE previous closes --------------
    market = []
    nifty_prev = float(indices["NIFTY 50"]["previousClose"])
    vix_prev = float(indices["INDIA VIX"]["previousClose"])
    for name, cap in captures.items():
        minute = cap["minute"]
        nifty = np.array(cap["index"]["NIFTY"], dtype=float)
        vix = np.array(cap["index"]["INDIAVIX"], dtype=float)
        sample = next(iter(cap["stocks"].values()), None)
        if sample is None:
            continue
        live = sample["features"]
        expected = {
            "nifty_day": nifty[minute, 3] / nifty_prev - 1,
            "nifty_ret15": nifty[minute, 3] / nifty[minute - 15, 3] - 1,
            "vix": vix[minute, 3],
            "vix_chg": vix[minute, 3] / vix_prev - 1,
        }
        market.append(
            {
                "cycle": name,
                **{
                    k: {"live": round(live[k], 6), "independent": round(float(v), 6)}
                    for k, v in expected.items()
                },
            }
        )
    report["market"] = {
        "nse_previous_close": {"NIFTY 50": nifty_prev, "INDIA VIX": vix_prev},
        "cycles": market,
    }

    # 4. order books: internal consistency, LTP vs bars, previous close vs NSE --------
    books = Counter()
    book_issues = []
    moves: list[float] = []
    quote_ages: dict[str, list[float]] = {"NSE": [], "NFO": []}
    for name, cap in captures.items():
        minute = cap["minute"]
        decided = datetime.fromisoformat(cap["decision_at"])
        for key, raw in (cap.get("depth") or {}).items():
            exchange, symbol = key.split(":")
            books["snapshots"] += 1
            quote_ages.setdefault(exchange, []).append(
                (datetime.fromisoformat(raw["timestamp"]) - decided).total_seconds()
            )
            bids, asks = raw["bids"], raw["asks"]
            last = float(raw["last_price"])
            issues = []
            if not bids or not asks:
                issues.append("one-sided")
            else:
                best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
                if best_bid >= best_ask:
                    issues.append("crossed")
                if [float(b[0]) for b in bids] != sorted(
                    (float(b[0]) for b in bids), reverse=True
                ):
                    issues.append("bids not descending")
                if [float(a[0]) for a in asks] != sorted(float(a[0]) for a in asks):
                    issues.append("asks not ascending")
                mid = (best_bid + best_ask) / 2
                if abs(last / mid - 1) > 0.005:
                    issues.append(f"ltp {last} far from mid {mid:.2f}")
                if raw["total_buy_quantity"] < sum(b[1] for b in bids):
                    issues.append("total buy < top-5 bids")
                if raw["total_sell_quantity"] < sum(a[1] for a in asks):
                    issues.append("total sell < top-5 asks")
            low, high = raw["lower_circuit"], raw["upper_circuit"]
            if low and high and not float(low) <= last <= float(high):
                issues.append("ltp outside circuit band")
            stale = (
                (
                    datetime.fromisoformat(raw["timestamp"])
                    - datetime.fromisoformat(raw["last_trade_at"])
                ).total_seconds()
                if raw["last_trade_at"]
                else None
            )
            if stale is not None and stale > 120:
                issues.append(f"last trade {stale:.0f} s before quote")
            if exchange == "NSE":
                official = cm.get(symbol)
                if (
                    official
                    and abs(float(raw["previous_close"]) - float(official["ClsPric"]))
                    > 0.001
                ):
                    issues.append(
                        f"previous close {raw['previous_close']} "
                        f"vs NSE {official['ClsPric']}"
                    )
                bars = cap["stocks"].get(symbol, {}).get("bars")
                if bars and bars[minute][3]:
                    moves.append(abs(last / bars[minute][3] - 1) * 100)
            if issues:
                books["with_issues"] += 1
                book_issues.append({"cycle": name, "instrument": key, "issues": issues})
    report["order_books"] = dict(books) | {
        "max_quote_vs_decision_close_pct": round(max(moves), 3) if moves else None,
        "quote_seconds_after_bar_close": {
            exchange: {
                "min": round(min(ages), 1),
                "mean": round(float(np.mean(ages)), 1),
                "max": round(max(ages), 1),
                "before_bar_close": sum(a < 0 for a in ages),
            }
            for exchange, ages in quote_ages.items()
            if ages
        },
        "issues": book_issues[:30],
    }

    # 5. futures: both contracts' previous OI vs NSE; previous-session and live values
    oi_checks = []
    for name, cap in captures.items():
        depth = cap.get("depth") or {}
        for symbol, contracts in (cap.get("futures") or {}).items():
            values = (cap.get("oi") or {}).get(symbol)
            if not contracts or not values or "previous_oi" not in values:
                continue  # none, or captured before October 14 (near month only)
            check = {"cycle": name, "symbol": symbol, "contracts": contracts}
            check["previous_oi_kite_nse"] = {
                c: [
                    values["previous_oi"].get(c),
                    int(float(fo[c]["OpnIntrst"])) if c in fo else None,
                ]
                for c in contracts
            }
            if contracts[0] in fo:
                expiry = date.fromisoformat(fo[contracts[0]]["XpryDt"])
                check["days_to_expiry_sent_nse"] = [
                    values.get("days_to_expiry"),
                    (expiry - day).days,
                ]
            path = cache / f"{symbol}.npz"
            if path.exists():
                with np.load(path) as saved:
                    sessions = [date.fromordinal(int(d)) for d in saved["dates"]]
                for key, back in (
                    ("oi_change_prev_session", 1),
                    ("oi_change_5_sessions", 5),
                ):
                    check[f"{key}_sent_nse"] = [
                        values.get(key),
                        nse_oi_change(fo_days, sessions, symbol, contracts, back),
                    ]
                earlier = fo_days.get(sessions[-2], {}).get(contracts[0])
                if earlier and contracts[0] in fo:
                    check["futures_change_prev_session_sent_nse"] = [
                        values.get("futures_change_prev_session"),
                        float(fo[contracts[0]]["ClsPric"]) / float(earlier["ClsPric"])
                        - 1,
                    ]
            quotes = [depth.get(f"NFO:{c}") for c in contracts]
            if quotes[0]:
                held = [
                    (quote["open_interest"], int(float(fo[c]["OpnIntrst"])))
                    for c, quote in zip(contracts, quotes, strict=True)
                    if quote and quote["open_interest"] and c in fo
                ]
                before = sum(b for _, b in held)
                check["oi_change_today_sent_nse"] = [
                    values.get("oi_change_today"),
                    sum(n for n, _ in held) / before - 1 if before else None,
                ]
                stock = depth.get(f"NSE:{symbol}")
                if stock:
                    check["premium_now_sent_recomputed"] = [
                        values.get("premium_now"),
                        quote_mid(quotes[0]) / quote_mid(stock) - 1,
                    ]
            oi_checks.append(check)
    report["futures_oi"] = oi_checks

    # 6. filings: what the AI was shown vs a fresh NSE fetch of the whole week ---------
    filing_checks = []
    since = datetime.combine(previous_day, datetime.min.time(), INDIA_TIMEZONE).replace(
        hour=15, minute=30
    )
    journal = sqlite3.connect(args.journals / "decisions.sqlite")
    requests = journal.execute(
        f"select cycle_id, request_json from cycles where {PROMPTS}"
        " and substr(cycle_id, 1, 8) = ?",
        (day.strftime("%Y%m%d"),),
    ).fetchall()
    for cycle_id, text in requests:
        request = json.loads(text)
        as_of = datetime.fromisoformat(request["as_of"])
        for stock in request["stocks"]:
            if "today" not in stock:  # identity-only requests show no filings
                continue
            public = [
                a
                for a in announcements
                if a.symbol == stock["symbol"] and disseminated_at(a) <= as_of
            ]
            recent = [a for a in public if since < disseminated_at(a)]
            week_start = as_of - timedelta(days=WEEK_DAYS)
            shown = stock.get("nse_filings") or {}
            filing_checks.append(
                {
                    "cycle": cycle_id,
                    "symbol": stock["symbol"],
                    "since_prev_close_shown_nse": [
                        shown.get("filings_since_prev_close"),
                        len(recent),
                    ],
                    "material_shown_nse": [
                        shown.get("material_since_prev_close"),
                        sum(not routine(a.subject) for a in recent),
                    ],
                    "last_7_days_shown_nse": [
                        shown.get("filings_last_7_days"),
                        sum(week_start < disseminated_at(a) for a in public),
                    ],
                    "subjects_nse": [a.subject[:80] for a in recent][:5],
                }
            )
    report["filings"] = filing_checks

    # 7. requests and replies: no scanner output, values = captured features, searches
    calls = []
    for journal_path in [
        args.journals / "decisions.sqlite",
        *sorted((args.journals / "shadows").glob("*/decisions.sqlite")),
    ]:
        book = (
            "main" if journal_path.parent == args.journals else journal_path.parent.name
        )
        db = sqlite3.connect(journal_path)
        for cycle_id, model, status, cost, latency, req, reply in db.execute(
            "select cycle_id, model, status, cost_usd, latency_seconds, request_json,"
            f" response_json from cycles where {PROMPTS}"
            " and substr(cycle_id, 1, 8) = ?",
            (day.strftime("%Y%m%d"),),
        ):
            request, answer = json.loads(req), json.loads(reply or "{}")
            hhmm = cycle_id.split("-")[1]
            bar_close = datetime.combine(
                day, time(int(hhmm[:2]), int(hhmm[2:])), INDIA_TIMEZONE
            )
            as_of_age = (
                datetime.fromisoformat(request["as_of"]) - bar_close
            ).total_seconds()
            cap = captures.get(cycle_id.split("-")[1])
            mapping_errors = now_errors = 0
            for stock in request["stocks"]:
                symbol = stock["symbol"]
                live = (cap or {}).get("stocks", {}).get(symbol, {}).get("features")
                if not live or "today" not in stock:  # identity-only requests
                    continue
                for label, source, scale in V3_TODAY + V3_DAILY:
                    sent, value = stock[stock_block(label)].get(label), live.get(source)
                    if (
                        value is not None
                        and sent is not None
                        and abs(sent - round(scale * value, 3)) > 1e-9
                    ):
                        mapping_errors += 1
                for label, source, scale in V3_MARKET:
                    sent, value = request["market"].get(label), live.get(source)
                    if (
                        value is not None
                        and sent is not None
                        and abs(sent - round(scale * value, 3)) > 1e-9
                    ):
                        mapping_errors += 1
                captured_oi = (cap.get("oi") or {}).get(symbol) or {}
                for label, source, scale in (
                    V3_OI if "previous_oi" in captured_oi else ()
                ):
                    sent = (stock.get("futures_oi") or {}).get(label)
                    value = captured_oi.get(source)
                    expected = (
                        value
                        if scale is None or value is None
                        else round(scale * value, 3)
                    )
                    mapping_errors += sent != expected
                path = cache / f"{symbol}.npz"
                if path.exists() and cap.get("nifty_closes"):
                    _sessions, daily = session_history(cache, symbol)
                    minute = cap["minute"]
                    expected_values = expected_now(
                        daily,
                        np.array(cap["stocks"][symbol]["bars"], dtype=float),
                        minute,
                        cap["nifty_closes"],
                        cap["index"]["NIFTY"][minute][3],
                    )
                    for label, value in expected_values.items():
                        block = (
                            request["market"]
                            if label.startswith("nifty_")
                            else stock[stock_block(label)]
                        )
                        sent = block.get(label)
                        if (sent is None) != (value is None) or (
                            value is not None and abs(sent - value) > 0.0011
                        ):
                            now_errors += 1
            names = {s["symbol"] for s in request["stocks"]}
            companies = {
                s["symbol"]: (s.get("company") or "").lower() for s in request["stocks"]
            }
            searches = answer.get("web_searches", [])
            queries = " ".join(str(s.get("query") or "") for s in searches).lower()
            sources = [u for s in searches for u in s.get("sources", [])]
            calls.append(
                {
                    "journal": book,
                    "cycle": cycle_id,
                    "model": model,
                    "status": status,
                    "cost_usd": round(cost, 4),
                    "seconds": round(latency, 1),
                    "as_of_seconds_after_bar_close": round(as_of_age, 1),
                    "stocks": sorted(names),
                    "alphabetical": [s["symbol"] for s in request["stocks"]]
                    == sorted(names),
                    "scanner_output_sent": any(
                        word in req for word in ("p_up", "probability", "scanner")
                    ),
                    "mapping_errors": mapping_errors,
                    "values_now_errors": now_errors,
                    "searches": len(searches),
                    "queries": [s.get("query") for s in searches][:6],
                    "stocks_searched": sorted(
                        s
                        for s in names
                        if s.lower() in queries
                        or (companies[s] and companies[s].split()[0] in queries)
                    ),
                    "sources": len(sources),
                    "source_dates": source_ages(sources, day),
                    "domains": Counter(urlparse(u).netloc for u in sources).most_common(
                        5
                    ),
                    "verdicts": [
                        (
                            d["symbol"],
                            d["verdict"],
                            d.get("p_green", d.get("p_up_first", d.get("confidence"))),
                        )
                        for d in answer.get("decisions", [])
                    ],
                }
            )
    report["ai_calls"] = calls

    # 8. executions: fills recomputed from the captured order book --------------------
    report["executions"] = [
        e for c in captures.values() for e in c.get("executions", [])
    ]
    report["execution_check"] = execution_check(captures, args.journals)
    report["execution_dry_run"] = dry_run(captures, final, day, args.output)

    # 9. the chain: scanner arithmetic, candidates to the AI, live exits --------------
    report["scanner_recompute"] = scanner_recompute(captures, args.scanner)
    report["candidates_to_ai"] = candidates_to_ai(
        captures, args.journals, day, args.threshold
    )
    report["live_exits"] = live_exits(final, args.journals, day) if final else None

    # 10. optional: prices sent to the AI vs another vendor's minute bars -------------
    if args.independent:
        report["independent_prices"] = independent_prices(
            args.journals, final, day, args.output
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=1, default=str), "utf-8"
    )
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        summarize(report)
    print(printed.getvalue(), end="")
    (args.output / "summary.txt").write_text(printed.getvalue(), "utf-8")
    return 0


def book_stores(journals: Path) -> dict[str, Path]:
    """Each AI's paper store: the main AI's, then every shadow's, by folder name."""
    stores = {"main": journals / "paper.sqlite"}
    for folder in sorted((journals / "shadows").glob("*")):
        if (folder / "paper.sqlite").exists():
            stores[folder.name] = folder / "paper.sqlite"
    return stores


def book_fill(
    raw: dict, side: str, tick: Decimal, target: Decimal = Decimal("0.01")
) -> tuple[int, Decimal, Decimal]:
    """Quantity, average fill and target (a fraction of the fill) of a Rs 2 lakh
    entry, by hand from the book (asks for a long, bids for a short; the rest at the
    deepest level)."""
    levels = [
        (Decimal(p), int(q)) for p, q, _ in raw["asks" if side == "LONG" else "bids"]
    ]
    quantity = int(Decimal(200_000) // levels[0][0])
    left, value = quantity, Decimal(0)
    for price, size in levels:
        take = min(left, size)
        value += take * price
        left -= take
    value += left * levels[-1][0]
    fill = value / quantity
    if side == "LONG":
        level = (fill * (1 + target) / tick).to_integral_value("ROUND_CEILING")
    else:
        level = (fill * (1 - target) / tick).to_integral_value("ROUND_FLOOR")
    return quantity, fill, level * tick


def store_target(store) -> Decimal:
    """The take-profit a paper store was created with (its saved PaperConfig)."""
    found = re.search(r"target=Decimal\('([0-9.]+)'\)", store.setting("config") or "")
    return Decimal(found.group(1)) if found else Decimal("0.01")


def execution_check(captures: dict, journals: Path) -> dict:
    """Every real paper entry against the order book captured when it was made."""
    from ai_trader.paper.store import PaperStore

    stores = {name: PaperStore(path) for name, path in book_stores(journals).items()}
    checked, problems = 0, []
    for name, cap in captures.items():
        for e in cap.get("executions", []):
            if e["position_id"] is None or not e["depth"]:
                continue
            _, model, effort = e["book"].split(":")
            store = stores.get(f"{model}_{effort}", stores["main"])
            position = store.position(e["position_id"])
            quantity, fill, target = book_fill(
                e["depth"], position.side, position.tick, store_target(store)
            )
            checked += 1
            if (
                position.symbol != e["symbol"]
                or position.side != e["side"]
                or (position.quantity, position.entry_price, position.target_price)
                != (quantity, fill, target)
            ):
                problems.append(
                    {
                        "cycle": name,
                        "book": e["book"],
                        "symbol": e["symbol"],
                        "paper": [position.quantity, str(position.entry_price)],
                        "by_hand": [quantity, str(fill), str(target)],
                    }
                )
    return {"entries_checked": checked, "mismatches": problems[:20]}


def scanner_recompute(captures: dict, scanner: Path) -> dict:
    """Each captured probability recomputed by the frozen model from the captured
    inputs (the scanner's arithmetic, independently of the scan)."""
    from ai_trader.scanner.upside import ScannerBundle

    bundle = ScannerBundle.load(scanner)
    worst, checked, unscoreable = 0.0, 0, []
    for name, cap in captures.items():
        symbols = list(cap["stocks"])
        rows = [bundle.vector(cap["stocks"][s]["features"]) for s in symbols]
        scored = [(s, r) for s, r in zip(symbols, rows, strict=True) if r is not None]
        unscoreable += [
            f"{name} {s}" for s, r in zip(symbols, rows, strict=True) if r is None
        ]
        if not scored:
            continue
        probabilities = bundle.predict(np.vstack([r for _, r in scored]))
        for (symbol, _), p in zip(scored, probabilities, strict=True):
            worst = max(worst, abs(float(p) - cap["stocks"][symbol]["probability"]))
            checked += 1
    return {
        "model": bundle.version,
        "stock_scans": checked,
        "max_abs_difference": worst,
        "unscoreable": unscoreable[:10],
    }


def candidates_to_ai(
    captures: dict, journals: Path, day: date, threshold: float
) -> dict:
    """Per scan, the stocks at or above the threshold that the main AI did not hold
    or trade earlier today (all of them without the one-per-stock rule) must be
    exactly the stocks it was asked about."""
    from ai_trader.paper.store import PaperStore

    store = PaperStore(journals / "paper.sqlite")
    one_per_stock = "one_per_stock=False" not in (store.setting("config") or "")
    positions = [
        p for p in store.positions() if p.entry_at.date() == day and one_per_stock
    ]
    skipped = {e["detail"]["decision_at"] for e in store.events("skip")}
    asked: dict[str, set[str]] = {}
    db = sqlite3.connect(journals / "decisions.sqlite")
    for cycle_id, text in db.execute(
        "select cycle_id, request_json from cycles where substr(cycle_id, 1, 8) = ?",
        (day.strftime("%Y%m%d"),),
    ):
        stocks = json.loads(text).get("stocks") or []
        asked.setdefault(cycle_id.split("-")[1], set()).update(
            s["symbol"] for s in stocks
        )
    checked, problems = 0, []
    for name, cap in captures.items():
        if cap["decision_at"] in skipped:
            continue
        decided = datetime.fromisoformat(cap["decision_at"])
        excluded = {p.symbol for p in positions if p.entry_at < decided}
        stocks = cap["stocks"]
        if any(abs(i["probability"] - threshold) < 1e-6 for i in stocks.values()):
            continue  # rounded onto the threshold: cannot tell from the capture
        expected = {s for s, i in stocks.items() if i["probability"] >= threshold}
        expected -= excluded
        got = asked.get(name, set())
        checked += 1
        if expected != got:
            problems.append(
                {
                    "cycle": name,
                    "not_asked": sorted(expected - got),
                    "asked_but_not_a_candidate": sorted(got - expected),
                }
            )
    return {"scans_checked": checked, "mismatches": problems[:20]}


def live_exits(final: dict, journals: Path, day: date) -> dict:
    """Every paper position entered today, every AI: its recorded exit against a
    plain loop over Kite's final minute bars (target, else 15:00)."""
    from ai_trader.paper.store import PaperStore

    opening = datetime.combine(day, time(9, 15), INDIA_TIMEZONE)
    out = {}
    for label, path in book_stores(journals).items():
        checked, problems, still_open = 0, [], []
        for p in PaperStore(path).positions():
            if p.entry_at.date() != day:
                continue
            rows = final.get(p.symbol)
            entry_minute = int((p.entry_at - opening).total_seconds() // 60)
            expected = (
                plain_exit(rows, entry_minute, p.side, p.target_price) if rows else None
            )
            if p.status == "open":
                still_open.append(
                    {"symbol": p.symbol, "side": p.side, "by_rules": str(expected)}
                )
                continue
            got = (
                int((p.exit_at - opening).total_seconds() // 60),
                p.exit_price,
                p.exit_reason,
            )
            checked += 1
            if got != expected:
                problems.append(
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "paper": str(got),
                        "final_bars": str(expected),
                    }
                )
        out[label] = {
            "closed_checked": checked,
            "mismatches": problems,
            "open": still_open,
        }
    return out


def yahoo_minutes(symbol: str, day: date, folder: Path) -> dict[int, float]:
    """Yahoo Finance one-minute closes of ``day`` by minute index from 09:15
    (cached under ``folder``)."""
    path = folder / f"{symbol}.json"
    if not path.exists():
        start = datetime.combine(day, time(9, 0), INDIA_TIMEZONE)
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            + urllib.parse.quote(symbol)
            + f"?interval=1m&period1={int(start.timestamp())}"
            + f"&period2={int((start + timedelta(hours=7)).timestamp())}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        clock.sleep(0.4)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(request, timeout=40).read())
    result = json.loads(path.read_text("utf-8"))["chart"]["result"][0]
    opening = datetime.combine(day, time(9, 15), INDIA_TIMEZONE).timestamp()
    closes = result["indicators"]["quote"][0]["close"]
    return {
        int((stamp - opening) // 60): close
        for stamp, close in zip(result.get("timestamp") or [], closes, strict=True)
        if close is not None
    }


def independent_prices(journals: Path, final: dict, day: date, output: Path) -> dict:
    """The price each AI request sent as "last_price" vs Yahoo's close of the same
    minute (another vendor), and Kite's final minute closes vs Yahoo's."""
    sent: dict[str, list[tuple[int, float]]] = {}
    for path in [
        journals / "decisions.sqlite",
        *sorted((journals / "shadows").glob("*/decisions.sqlite")),
    ]:
        db = sqlite3.connect(path)
        for cycle_id, text in db.execute(
            f"select cycle_id, request_json from cycles where {PROMPTS}"
            " and substr(cycle_id, 1, 8) = ?",
            (day.strftime("%Y%m%d"),),
        ):
            hhmm = cycle_id.split("-")[1]
            minute = int(hhmm[:2]) * 60 + int(hhmm[2:]) - 9 * 60 - 15 - 1
            for stock in json.loads(text).get("stocks") or []:
                price = (stock.get("today") or {}).get("last_price")
                if price is not None:
                    sent.setdefault(stock["symbol"], []).append((minute, float(price)))
    gaps, kite_gaps, missing, worst = [], [], [], []
    for symbol in sorted(sent):
        try:
            yahoo = yahoo_minutes(f"{symbol}.NS", day, output / "yahoo")
        except Exception as error:
            missing.append(f"{symbol}: {type(error).__name__}")
            continue
        for minute, price in set(sent[symbol]):
            if minute in yahoo:
                gap = abs(price / yahoo[minute] - 1) * 10_000
                gaps.append(gap)
                worst.append((round(gap, 1), symbol, minute, price, yahoo[minute]))
        for minute, row in enumerate(final.get(symbol) or []):
            if row[3] is not None and minute in yahoo:
                kite_gaps.append(abs(row[3] / yahoo[minute] - 1) * 10_000)

    def spread(values: list[float]) -> dict:
        if not values:
            return {}
        return {
            "compared": len(values),
            "median_bp": round(float(np.median(values)), 2),
            "p95_bp": round(float(np.percentile(values, 95)), 2),
            "max_bp": round(max(values), 2),
            "over_10bp": sum(v > 10 for v in values),
        }

    return {
        "sent_last_price_vs_yahoo": spread(gaps),
        "kite_final_closes_vs_yahoo": spread(kite_gaps),
        "largest": sorted(worst, reverse=True)[:10],
        "not_on_yahoo": missing[:20],
    }


def prescreen_gaps(captures: dict) -> dict:
    """How far the stream-bar probabilities were from the exact re-score."""
    gaps = [
        abs(cap["prescreen"][s] - cap["stocks"][s]["probability"])
        for cap in captures.values()
        for s in cap.get("prescreen") or {}
        if s in cap["stocks"]
    ]
    modes = Counter(cap.get("mode", "full") for cap in captures.values())
    return {
        "cycles_by_mode": dict(modes),
        "stocks_compared": len(gaps),
        "max_abs_difference": round(max(gaps), 4) if gaps else None,
        "mean_abs_difference": round(float(np.mean(gaps)), 4) if gaps else None,
    }


def stock_block(label: str) -> str:
    return "today" if label in {name for name, _, _ in V3_TODAY} else "daily"


def plain_exit(rows, after: int, side: str, target: Decimal):
    """The first minute after ``after`` and before 15:00 that reaches the target
    (at its open if it opens beyond it); else 15:00 at the last close from
    ``after`` to 14:59. By a plain loop."""
    last_close = None
    for i in range(after, min(len(rows), EXIT_MINUTE)):
        if rows[i][0] is None:
            continue
        o, h, low, close = (Decimal(str(round(rows[i][k], 2))) for k in range(4))
        if i > after:
            if side == "LONG" and o >= target or side == "SHORT" and o <= target:
                return i, o, "tp_gap"
            if side == "LONG" and h >= target or side == "SHORT" and low <= target:
                return i, target, "tp"
        last_close = close
    return None if last_close is None else (EXIT_MINUTE, last_close, "time_1500")


def dry_run(captures: dict, final: dict, day: date, output: Path) -> list[dict]:
    """A long and a short on every captured live order book, in scratch stores;
    fills, targets and exits checked by hand."""
    results = []
    scratch = output / "dry_run"
    scratch.mkdir(parents=True, exist_ok=True)
    opening = datetime.combine(day, datetime.min.time(), INDIA_TIMEZONE).replace(
        hour=9, minute=15
    )
    for name, cap in captures.items():
        for key, raw in (cap.get("depth") or {}).items():
            exchange, symbol = key.split(":")
            if exchange != "NSE":
                continue
            for side, book in (("LONG", "asks"), ("SHORT", "bids")):
                if raw[book]:
                    results.append(
                        dry_run_one(
                            name, symbol, side, raw, captures, final, opening, scratch
                        )
                    )
    return results


def dry_run_one(name, symbol, side, raw, captures, final, opening, scratch) -> dict:
    from ai_trader.broker import OHLCVCandle
    from ai_trader.paper.broker import PaperBroker, PaperConfig
    from ai_trader.paper.replay import nse_tick
    from ai_trader.paper.store import PaperStore

    path = scratch / f"{name}_{symbol}_{side}.sqlite"
    path.unlink(missing_ok=True)
    broker = PaperBroker(PaperStore(path), PaperConfig(capital=Decimal(10**9)))
    snapshot = snapshot_from(raw, symbol)
    tick = nse_tick(snapshot.last_price)
    now = snapshot.timestamp
    result = broker.enter(
        symbol,
        side,
        snapshot,
        tick,
        now,
        cycle_id=name,
        probability=0,
        confidence=0,
        reason="dry run",
    )
    position = result.position
    check = {
        "cycle": name,
        "symbol": symbol,
        "side": side,
        "filled": position is not None,
    }
    if position is None:
        check["refused"] = result.reason
        return check
    levels = [
        (Decimal(p), int(q)) for p, q, _ in raw["asks" if side == "LONG" else "bids"]
    ]
    quantity = int(Decimal(200_000) // levels[0][0])
    left, value = quantity, Decimal(0)
    for price, size in levels:
        take = min(left, size)
        value += take * price
        left -= take
    value += left * levels[-1][0]
    fill = value / quantity
    if side == "LONG":
        target = (fill * Decimal("1.01") / tick).to_integral_value("ROUND_CEILING")
    else:
        target = (fill * Decimal("0.99") / tick).to_integral_value("ROUND_FLOOR")
    target *= tick
    check |= {
        "quantity_ok": position.quantity == quantity,
        "fill_ok": position.entry_price == fill,
        "target_ok": position.target_price == target and position.stop_price is None,
        "fill": str(round(fill, 2)),
        "best_price": str(levels[0][0]),
        "thin_book": left > 0,
    }
    rows = final.get(symbol) or next(
        (
            c["stocks"][symbol]["bars"]
            for _, c in sorted(captures.items(), reverse=True)
            if symbol in c["stocks"]
        ),
        None,
    )
    check["exit_bars"] = "final" if symbol in final else "latest capture"
    if rows:
        entry_minute = int((now - opening).total_seconds() // 60)
        bars = [
            OHLCVCandle(
                timestamp=opening + timedelta(minutes=i),
                open=Decimal(str(round(row[0], 2))),
                high=Decimal(str(round(row[1], 2))),
                low=Decimal(str(round(row[2], 2))),
                close=Decimal(str(round(row[3], 2))),
                volume=int(row[4]) if row[4] is not None else None,
            )
            for i, row in enumerate(rows)
            if row[0] is not None and i >= entry_minute
        ]
        # judged after 15:02, so a position without its target is closed at 15:00
        end = opening + timedelta(minutes=max(len(rows), EXIT_MINUTE) + 3)
        closed = broker.apply_bars(position, bars, end)
        expected = plain_exit(rows, entry_minute, side, position.target_price)
        got = (
            (
                int((closed.exit_at - opening).total_seconds() // 60),
                closed.exit_price,
                closed.exit_reason,
            )
            if closed
            else None
        )
        check |= {"exit": str(got), "exit_ok": got == expected}
    return check


def summarize(report: dict) -> None:
    b, f = report["bars"], report["features"]
    print(f"cycles: {', '.join(report['cycles'])}")
    print(
        f"bars: {b.get('stock_cycles', 0)} stock-cycles, missing rows "
        f"{b.get('missing_rows', 0)}, OHLC violations {b.get('ohlc_violations', 0)}; "
        f"final bars {'yes' if b['final_bars_available'] else 'not yet'}: "
        f"{b.get('rows_revised', 0)} of {b.get('rows_compared', 0)} rows revised, "
        f"decision bar revised {b.get('decision_bar_revised', 0)}; "
        f"fetch s {b['fetch_seconds']}"
    )
    print(f"skips: {list(b['skipped'].values())[-1] if b['skipped'] else None}")
    print(f"live pre-screen vs exact: {f['live_prescreen_vs_exact']}")
    worst = max(f["max_relative_difference"].values(), default=0)
    print(
        f"features: {f['stock_cycles_recomputed']} recomputed by research code, worst "
        f"relative diff {worst:.2g}; previous close vs NSE {f['previous_close_vs_nse']}"
    )
    for m in report["market"]["cycles"]:
        print("market", m["cycle"], {k: v for k, v in m.items() if k != "cycle"})
    o = report["order_books"]
    print(
        f"order books: {o.get('snapshots', 0)} snapshots, "
        f"with issues {o.get('with_issues', 0)}, largest quote move from the "
        f"decision close {o.get('max_quote_vs_decision_close_pct')}%"
    )
    for issue in o["issues"][:8]:
        print("  ", issue)
    for check in report["futures_oi"][:8]:
        print("oi", check)
    for check in report["filings"][:12]:
        print("filings", check)
    for call in report["ai_calls"]:
        print(
            "ai",
            call["journal"],
            call["cycle"],
            call["status"],
            f"${call['cost_usd']}",
            f"{call['seconds']}s",
            "alpha" if call["alphabetical"] else "NOT-ALPHA",
            "SCANNER-SENT" if call["scanner_output_sent"] else "facts-only",
            f"map_err={call['mapping_errors']}",
            f"now_err={call['values_now_errors']}",
            f"searches={call['searches']}",
            f"searched={call['stocks_searched']}",
            f"sources={call['sources']}",
            call["domains"][:3],
            call["verdicts"],
        )
    print(f"executions: {len(report['executions'])}")
    for check in report["execution_dry_run"]:
        print("dry-run entry", check)
    calls = report["ai_calls"]
    if calls:
        ages = [c["as_of_seconds_after_bar_close"] for c in calls]
        dates = Counter()
        for c in calls:
            dates.update(c["source_dates"])
        print(
            f"AI requests: as_of {min(ages)}-{max(ages)} s after the bar close; "
            f"web sources {dict(dates)}"
        )
    print(
        "order-book quote time after the bar close (s):",
        report["order_books"]["quote_seconds_after_bar_close"],
    )
    print("real entries vs captured book:", report["execution_check"])
    print("scanner arithmetic:", report["scanner_recompute"])
    print("scanner -> AI:", report["candidates_to_ai"])
    for book, result in (report["live_exits"] or {}).items():
        print(
            f"live exits {book}: {result['closed_checked']} checked, "
            f"{len(result['mismatches'])} mismatches, {len(result['open'])} open",
            result["mismatches"][:5],
        )
    if "independent_prices" in report:
        print("independent prices:", report["independent_prices"])


if __name__ == "__main__":
    raise SystemExit(main())
