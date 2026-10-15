"""End-to-end validation of the F&O-derived dataset: NSE archive -> normalized
sessions -> features, and the Kite minute capture.

  python -m scripts.validate_fno_data --output data/fno/validation/<date>

Checks (each reports counts and examples; nothing is fixed silently):
 1. coverage: every dataset's sessions against the research calendar (Kite, 2020 to
    Sep 2026) and, after it, against NSE's cash bhavcopies;
 2. formats: the old and the UDiFF F&O bhavcopy of the same sessions, parsed
    separately, give the same contracts and values;
 3. cache: normalized session files equal a fresh parse of the raw file;
 4. Kite: NSE's near-month futures settlement and OI equal Kite's continuous futures
    day bars (data/oi_kite_20261011), volume = contracts x lot where NSE gives it;
 5. identity: each contract's OI change equals today's OI minus the previous
    session's (a new contract: its whole OI);
 6. cash: delivery % = deliverable / traded quantity; OHLC ordering;
 7. participants: TOTAL = the sum of the four participants; each instrument's total
    long = total short; total long = the sum of the long columns;
 8. ban: the ban list of a session = the stocks NSE's previous combined-OI file said
    "No Fresh Positions" for;
 9. indices: NSE's all-index closes = Kite's daily bars of 12 indices (incl. VIX);
10. underlying: the UDiFF file's underlying price = NSE's official close of the stock
    (cash bhavcopy) or index (all-index file) that session;
11. implied volatility: an independent root finder (scipy brentq) on a sample;
12. features: a sample of stock rows recomputed from the raw files by separate code;
13. NIFTY 30-day ATM IV against India VIX (NSE's close; correlation, level);
14. put-call parity forward against the futures settlement (stock near expiry);
15. the Kite minute capture against NSE's end-of-day file (when captured).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from ai_trader.fno import bhavcopy
from ai_trader.fno.features import (
    INDEX_FILE_NAMES,
    read_ban,
    read_cm,
    read_indices,
    read_mwpl,
    read_participants,
)
from ai_trader.fno.iv import RATE, implied_vol
from ai_trader.fno.nse_archive import DATASETS, nse_date
from ai_trader.paper.basis import nse_symbol, symbol_history

NSE = Path("data/nse")
FNO = Path("data/fno")
CALENDAR = Path("data/candles_fno_basis_20261011/dataset.json")
KITE_FUTURES = Path("data/oi_kite_20261011")
VIX = Path("data/candles_fno_basis_20261011/daily/NSE_INDIAVIX.csv")
# Kite's daily index bars in the research dataset -> the index's name in NSE's file.
KITE_INDICES = {
    "NSE_NIFTY.csv": "Nifty 50",
    "NSE_BANKNIFTY.csv": "Nifty Bank",
    "NSE_FINNIFTY.csv": "Nifty Financial Services",
    "NSE_INDIAVIX.csv": "India VIX",
    "NSE_NIFTY500.csv": "Nifty 500",
    "NSE_NIFTYAUTO.csv": "Nifty Auto",
    "NSE_NIFTYFMCG.csv": "Nifty FMCG",
    "NSE_NIFTYIT.csv": "Nifty IT",
    "NSE_NIFTYMETAL.csv": "Nifty Metal",
    "NSE_NIFTYMIDCAP150.csv": "Nifty Midcap 150",
    "NSE_NIFTYPHARMA.csv": "Nifty Pharma",
    "NSE_NIFTYREALTY.csv": "Nifty Realty",
}


def manifest_days(name: str) -> list[date]:
    path = NSE / "manifest" / f"{name}.json"
    if not path.exists():
        return []
    return sorted(date.fromisoformat(d) for d in json.loads(path.read_text())["files"])


def coverage() -> dict:
    calendar = sorted(
        date.fromisoformat(d) for d in json.loads(CALENDAR.read_text())["session_dates"]
    )
    end = calendar[-1]
    cm = manifest_days("cm")
    later = [d for d in cm if d > end]
    sessions = set(calendar) | set(later)
    result = {"calendar_sessions": len(calendar), "after_calendar_sessions": len(later)}
    fo = set(manifest_days("fo")) | set(manifest_days("fo_old"))
    for name, days in (
        ("fo_any_format", fo),
        ("cm", set(cm)),
        ("participant_oi", set(manifest_days("participant_oi"))),
        ("participant_vol", set(manifest_days("participant_vol"))),
        ("ban", set(manifest_days("ban"))),
        ("mwpl", set(manifest_days("mwpl"))),
        ("indices", set(manifest_days("indices"))),
    ):
        missing = sorted(d for d in sessions if d not in days)
        extra = sorted(d for d in days if d not in sessions and d <= max(sessions))
        result[name] = {
            "files": len(days),
            "missing_sessions": [d.isoformat() for d in missing],
            "files_on_non_sessions": [d.isoformat() for d in extra],
            # e.g. the ban list of the next trade date, published the evening before
            "files_after_last_session": [
                d.isoformat() for d in sorted(days) if d > max(sessions)
            ],
        }
    udiff = manifest_days("fo")
    result["udiff_first_session"] = udiff[0].isoformat() if udiff else None
    return result


def _key_table(table: dict[str, np.ndarray]) -> dict[tuple, int]:
    keys = {}
    for i in range(len(table["symbol"])):
        strike = table["strike"][i]
        keys[
            (
                str(table["symbol"][i]),
                str(table["kind"][i]),
                str(table["underlying"][i]),
                str(table["expiry"][i]),
                None if math.isnan(strike) else float(strike),
                str(table["option"][i]),
            )
        ] = i
    return keys


def formats() -> dict:
    both = sorted(set(manifest_days("fo")) & set(manifest_days("fo_old")))
    fields = ("open", "high", "low", "close", "settle", "contracts", "oi", "chg_oi")
    mismatches: Counter[str] = Counter()
    examples = []
    rows = only_old = only_new = 0
    for day in both:
        new = bhavcopy.parse_udiff(
            DATASETS["fo"].text(DATASETS["fo"].path(NSE, day).read_bytes(), day), day
        )
        old = bhavcopy.parse_old(
            DATASETS["fo_old"].text(
                DATASETS["fo_old"].path(NSE, day).read_bytes(), day
            ),
            day,
        )
        new_keys, old_keys = _key_table(new), _key_table(old)
        only_new += len(new_keys.keys() - old_keys.keys())
        only_old += len(old_keys.keys() - new_keys.keys())
        for key in new_keys.keys() & old_keys.keys():
            i, j = new_keys[key], old_keys[key]
            rows += 1
            for name in fields:
                if new[name][i] != old[name][j]:
                    mismatches[name] += 1
                    if len(examples) < 10:
                        examples.append(
                            [
                                day.isoformat(),
                                *map(str, key),
                                name,
                                str(new[name][i]),
                                str(old[name][j]),
                            ]
                        )
            if abs(new["value"][i] - old["value"][j]) > 1000.0:
                mismatches["value_over_1000_rupees"] += 1
    return {
        "sessions_in_both_formats": len(both),
        "first": both[0].isoformat() if both else None,
        "last": both[-1].isoformat() if both else None,
        "contracts_compared": rows,
        "only_in_udiff": only_new,
        "only_in_old": only_old,
        "mismatches": dict(mismatches),
        "examples": examples,
    }


def sessions_pass(sample_days: set[date]) -> tuple[dict, dict]:
    """One pass over every normalized session: the cache check (sample), the OI
    identity, and the near-month futures per (symbol, date) for the Kite check."""
    days = sorted(
        date.fromisoformat(p.stem) for p in (FNO / "normalized").rglob("*.npz")
    )
    days_set = set(days)
    mismatch_days: Counter[str] = Counter()
    identity: Counter[str] = Counter()
    identity_examples = []
    cache = {"sessions_checked": 0, "differences": []}
    near: dict[tuple[str, date], tuple[float, int, int, int]] = {}
    previous: dict[tuple, int] | None = None
    previous_table = None
    calendar = sorted(
        {
            date.fromisoformat(d)
            for d in json.loads(CALENDAR.read_text())["session_dates"]
        }
        | set(manifest_days("cm"))
    )
    for day in days:
        table, _ = bhavcopy.load_table(
            FNO / "normalized" / f"{day.year}" / f"{day}.npz"
        )
        earlier = [d for d in calendar if d < day]
        # The session before this one has no F&O file (2021-03-30): its changes in
        # OI are against a session we lack, so they are counted apart.
        after_gap = previous is not None and earlier and earlier[-1] not in days_set
        if day in sample_days:
            fresh = bhavcopy.load_session(NSE, day)
            cache["sessions_checked"] += 1
            for name in bhavcopy.COLUMNS:
                if (
                    not np.array_equal(fresh[0][name], table[name], equal_nan=True)
                    if (fresh[0][name].dtype.kind == "f")
                    else not np.array_equal(fresh[0][name], table[name])
                ):
                    cache["differences"].append([day.isoformat(), name])
        keys = _key_table(table)
        for key, i in keys.items():
            oi, change = int(table["oi"][i]), int(table["chg_oi"][i])
            if previous is None:  # the first session: no earlier file to compare
                identity["first_session_skipped"] += 1
                continue
            if key in previous:
                expected = oi - int(previous_table["oi"][previous[key]])
                status = "continuing"
            else:
                expected, status = oi, "new"
            if after_gap:
                status = f"after_missing_session_{status}"
            identity[f"{status}_checked"] += 1
            if change != expected:
                identity[f"{status}_mismatch"] += 1
                mismatch_days[day.isoformat()] += 1
                if len(identity_examples) < 10:
                    identity_examples.append(
                        [day.isoformat(), *map(str, key), change, expected]
                    )
        futures = np.flatnonzero((table["kind"] == "F") & (table["underlying"] == "S"))
        by_symbol: dict[str, list[int]] = defaultdict(list)
        for i in futures:
            by_symbol[str(table["symbol"][i])].append(int(i))
        for symbol, rows in by_symbol.items():
            i = min(rows, key=lambda r: table["expiry"][r])
            near[(symbol, day)] = (
                float(table["settle"][i]),
                float(table["close"][i]),
                int(table["oi"][i]),
                int(table["contracts"][i]),
                int(table["lot"][i]),
            )
        previous, previous_table = keys, table
    identity_report = dict(identity)
    # Known causes: 2021-03-31 follows 2021-03-30, which has no F&O file; NSE moved
    # the June 2023 expiry from 06-29 to 06-28 (holiday), changing the contracts'
    # expiry dates; on 2026-01-12 NSE's change in OI differs from the difference of
    # its own 01-09 and 01-12 OI for 1,292 contracts.
    identity_report["mismatch_days"] = len(mismatch_days)
    identity_report["top_mismatch_days"] = mismatch_days.most_common(12)
    identity_report["examples"] = identity_examples
    return {"cache": cache, "identity": identity_report}, near


def kite_check(near: dict) -> dict:
    """NSE's near-month stock futures against Kite's continuous futures day bars.
    Kite's close is the contract's close (NSE's settlement price except on expiry
    day, when NSE settles at the underlying's close) and Kite ADJUSTS its history
    for corporate actions (RELIANCE x0.5 before the Oct 2024 bonus): a mismatch
    whose Kite/NSE ratio recurs (3+ sessions of the symbol) is counted as Kite's
    adjustment; volume is compared on unadjusted sessions only."""
    history = symbol_history((NSE / "symbolchange.csv").read_text("utf-8"))
    counts: Counter[str] = Counter()
    examples = []
    for path in sorted(KITE_FUTURES.glob("*.csv")):
        symbol = path.stem
        mismatched: list[tuple[str, float, float, float]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                day = date.fromisoformat(row["session_date"])
                name = nse_symbol(symbol, day, history.get(symbol, []))
                found = near.get((name, day))
                if found is None:
                    counts["no_nse_row"] += 1
                    continue
                settle, close, oi, contracts, lot = found
                kite_close = float(row["close"])
                counts["compared"] += 1
                counts["oi_equal"] += oi == int(row["open_interest"])
                if abs(close - kite_close) < 0.005:
                    counts["close_equal"] += 1
                    counts["close_equal_settle_too"] += abs(settle - close) < 0.005
                    if lot:
                        counts["volume_compared"] += 1
                        counts["volume_equal"] += contracts * lot == int(row["volume"])
                else:
                    mismatched.append((day.isoformat(), close, kite_close, settle))
        ratios = Counter(round(kite / nse, 4) for _, nse, kite, _ in mismatched)
        for day, nse, kite, _settle in mismatched:
            if ratios[round(kite / nse, 4)] >= 3:
                counts["close_kite_adjusted"] += 1
            else:
                counts["close_unexplained"] += 1
                if len(examples) < 15:
                    examples.append([symbol, day, nse, kite])
    return {**counts, "unexplained_examples": examples}


def cash() -> dict:
    counts: Counter[str] = Counter()
    files: Counter[str] = Counter()
    for day in manifest_days("cm"):
        text = DATASETS["cm"].path(NSE, day).read_text("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = [c.strip() for c in next(reader)]
        at = {n: i for i, n in enumerate(header)}
        for raw in reader:
            if len(raw) < len(header) or raw[at["SERIES"]].strip() != "EQ":
                continue
            try:
                o, h, low, c = (
                    float(raw[at[n]])
                    for n in ("OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE")
                )
                volume = float(raw[at["TTL_TRD_QNTY"]])
            except ValueError:
                counts["unparsable_prices"] += 1
                continue
            counts["eq_rows"] += 1
            if not low <= min(o, c) <= max(o, c) <= h:
                counts["ohlc_out_of_order"] += 1
            try:
                quantity = float(raw[at["DELIV_QTY"]])
                percent = float(raw[at["DELIV_PER"]])
            except (ValueError, KeyError):
                counts["no_delivery_figures"] += 1
                continue
            counts["delivery_checked"] += 1
            if volume > 0 and abs(quantity / volume * 100 - percent) > 0.006:
                counts["delivery_pct_mismatch"] += 1
                files[day.isoformat()] += 1
    # NSE's own files disagree (delivery % vs its quantities) on a few whole days.
    return {**counts, "delivery_pct_mismatch_files": files.most_common(10)}


def participants() -> dict:
    counts: Counter[str] = Counter()
    examples = []
    worst = 0.0
    pairs = (
        ("fut_idx_long", "fut_idx_short"),
        ("fut_stk_long", "fut_stk_short"),
        ("opt_idx_call_long", "opt_idx_call_short"),
        ("opt_idx_put_long", "opt_idx_put_short"),
        ("opt_stk_call_long", "opt_stk_call_short"),
        ("opt_stk_put_long", "opt_stk_put_short"),
    )
    longs = [a for a, _ in pairs]
    for name in ("participant_oi", "participant_vol"):
        for day in manifest_days(name):
            text = DATASETS[name].text(DATASETS[name].path(NSE, day).read_bytes(), day)
            values = read_participants(text)
            rows = [r for r in csv.reader(io.StringIO(text)) if r and r[0].strip()]
            total = next(r for r in rows if r[0].strip() == "TOTAL")
            totals = [float(x) for x in total[1:15]]
            columns = [
                n
                for n in (
                    "fut_idx_long",
                    "fut_idx_short",
                    "fut_stk_long",
                    "fut_stk_short",
                    "opt_idx_call_long",
                    "opt_idx_put_long",
                    "opt_idx_call_short",
                    "opt_idx_put_short",
                    "opt_stk_call_long",
                    "opt_stk_put_long",
                    "opt_stk_call_short",
                    "opt_stk_put_short",
                    "total_long",
                    "total_short",
                )
            ]
            counts[f"{name}_files"] += 1
            for column, value in zip(columns, totals, strict=True):
                summed = sum(
                    values[f"{w}_{column}"] for w in ("client", "dii", "fii", "pro")
                )
                if summed != value:
                    counts[f"{name}_total_not_sum"] += 1
                    worst = max(worst, abs(summed - value) / max(abs(value), 1.0))
                    if len(examples) < 10:
                        examples.append([name, day.isoformat(), column, summed, value])
            by_column = dict(zip(columns, totals, strict=True))
            for a, b in pairs:
                if by_column[a] != by_column[b]:
                    counts[f"{name}_long_ne_short"] += 1
            for who in ("client", "dii", "fii", "pro"):
                if (
                    sum(values[f"{who}_{c}"] for c in longs)
                    != values[f"{who}_total_long"]
                ):
                    counts[f"{name}_total_long_not_sum"] += 1
    # NSE's TOTAL row is off the sum of the four rows by a contract or two on
    # some days (rounding inside NSE's file); the largest gap, relative:
    return {**counts, "largest_relative_gap": worst, "examples": examples}


def ban_check() -> dict:
    ban_days = manifest_days("ban")
    mwpl_days = manifest_days("mwpl")
    counts: Counter[str] = Counter()
    examples = []
    for day in ban_days:
        before = [d for d in mwpl_days if d < day]
        if not before:
            continue
        prior = before[-1]
        mwpl = read_mwpl(
            DATASETS["mwpl"].text(DATASETS["mwpl"].path(NSE, prior).read_bytes(), prior)
        )
        expected = {
            s for s, r in mwpl.items() if "no fresh" in str(r["mwpl_limit"]).lower()
        }
        banned = read_ban(DATASETS["ban"].path(NSE, day).read_text("utf-8-sig"))
        counts["sessions"] += 1
        counts["equal"] += banned == expected
        if banned != expected and len(examples) < 10:
            examples.append(
                [day.isoformat(), sorted(banned - expected), sorted(expected - banned)]
            )
    return {**counts, "examples": examples}


def independent_black76(forward, strike, years, vol, call) -> float:
    """Black-76 with math.erf, written apart from fno/iv.py."""

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    root = vol * math.sqrt(years)
    d1 = (math.log(forward / strike) + root * root / 2) / root
    d2 = d1 - root
    discount = math.exp(-RATE * years)
    if call:
        return discount * (forward * cdf(d1) - strike * cdf(d2))
    return discount * (strike * cdf(-d2) - forward * cdf(-d1))


def iv_check(rng: random.Random) -> dict:
    from scipy.optimize import brentq

    days = rng.sample(sorted((FNO / "normalized").rglob("*.npz")), 20)
    worst = 0.0
    checked = 0
    for path in days:
        table, _ = bhavcopy.load_table(path)
        day = date.fromisoformat(path.stem)
        futures = {
            (str(table["symbol"][i]), table["expiry"][i]): float(table["settle"][i])
            for i in np.flatnonzero(table["kind"] == "F")
        }
        options = [
            i
            for i in np.flatnonzero((table["kind"] == "O") & (table["contracts"] > 0))
            if (str(table["symbol"][i]), table["expiry"][i]) in futures
        ]
        for i in rng.sample(options, min(100, len(options))):
            forward = futures[(str(table["symbol"][i]), table["expiry"][i])]
            years = float((table["expiry"][i] - np.datetime64(day)).astype(int)) / 365
            call = table["option"][i] == "CE"
            price, strike = float(table["close"][i]), float(table["strike"][i])
            mine = float(implied_vol(price, forward, strike, years, call))
            if not math.isfinite(mine):
                continue

            def gap(s, f=forward, k=strike, t=years, c=call, p=price):
                return independent_black76(f, k, t, s, c) - p

            theirs = brentq(gap, 1e-4, 5.0, xtol=1e-14, rtol=1e-14, maxiter=500)
            worst = max(worst, abs(mine - theirs))
            checked += 1
    return {"options_checked": checked, "max_abs_iv_difference": worst, "rate": RATE}


def _raw_rows(day: date) -> tuple[list[dict], str]:
    for name in ("fo", "fo_old"):
        spec = DATASETS[name]
        path = spec.path(NSE, day)
        if path.exists():
            text = spec.text(path.read_bytes(), day)
            return list(csv.DictReader(io.StringIO(text))), name
    return [], ""


def features_check(rng: random.Random, samples: int = 300) -> dict:
    """Stock rows recomputed from the raw files with csv.DictReader (separate code)."""
    years = sorted((FNO / "features").glob("stock_daily_*.csv.gz"))
    candidates = []
    for path in years:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["fno"] == "1" and rng.random() < 0.002:
                    candidates.append(row)
    picks = rng.sample(candidates, min(samples, len(candidates)))
    counts: Counter[str] = Counter()
    examples = []
    cache: dict[date, tuple[list[dict], str]] = {}
    for row in picks:
        day, symbol = date.fromisoformat(row["date"]), row["symbol"]
        if day not in cache:
            cache[day] = _raw_rows(day)
        raw, source = cache[day]
        if source == "fo":
            mine = [
                r
                for r in raw
                if r["TckrSymb"].strip() == symbol
                and r["FinInstrmTp"].strip() in ("STF", "STO")
            ]
            futures = [r for r in mine if r["FinInstrmTp"].strip() == "STF"]
            options = [r for r in mine if r["FinInstrmTp"].strip() == "STO"]
            oi, chg, settle, expiry = (
                "OpnIntrst",
                "ChngInOpnIntrst",
                "SttlmPric",
                "XpryDt",
            )
            strike_col, type_col = "StrkPric", "OptnTp"
        else:
            mine = [
                r
                for r in raw
                if r["SYMBOL"].strip() == symbol
                and r["INSTRUMENT"].strip() in ("FUTSTK", "OPTSTK")
            ]
            futures = [r for r in mine if r["INSTRUMENT"].strip() == "FUTSTK"]
            options = [r for r in mine if r["INSTRUMENT"].strip() == "OPTSTK"]
            oi, chg, settle, expiry = "OPEN_INT", "CHG_IN_OI", "SETTLE_PR", "EXPIRY_DT"
            strike_col, type_col = "STRIKE_PR", "OPTION_TYP"
        expected = {}
        if futures:
            near = min(futures, key=lambda r: nse_date(r[expiry]))
            expected["fut_oi_total"] = sum(float(r[oi]) for r in futures)
            expected["fut_chg_oi_total"] = sum(float(r[chg]) for r in futures)
            expected["fut_settle_near"] = float(near[settle])
            close = row["cm_close"]
            if close:
                expected["fut_basis_pct"] = (
                    float(near[settle]) / float(close) - 1
                ) * 100
        if options:
            ce = [r for r in options if r[type_col].strip() == "CE"]
            pe = [r for r in options if r[type_col].strip() == "PE"]
            expected["opt_oi_ce"] = sum(float(r[oi]) for r in ce)
            expected["opt_oi_pe"] = sum(float(r[oi]) for r in pe)
            if expected["opt_oi_ce"]:
                expected["opt_pcr_oi"] = expected["opt_oi_pe"] / expected["opt_oi_ce"]
            first = min(nse_date(r[expiry]) for r in options)
            strikes = sorted(
                {float(r[strike_col]) for r in options if nse_date(r[expiry]) == first}
            )
            ce_at = defaultdict(float)
            pe_at = defaultdict(float)
            for r in options:
                if nse_date(r[expiry]) == first:
                    side = ce_at if r[type_col].strip() == "CE" else pe_at
                    side[float(r[strike_col])] += float(r[oi])
            if sum(ce_at.values()) + sum(pe_at.values()) > 0:
                pain = {
                    s: sum(o * max(s - k, 0) for k, o in ce_at.items())
                    + sum(o * max(k - s, 0) for k, o in pe_at.items())
                    for s in strikes
                }
                expected["opt_max_pain_near"] = min(strikes, key=lambda s: (pain[s], s))
            if ce_at and max(ce_at.values()) > 0:
                expected["opt_call_wall_near"] = min(
                    strikes, key=lambda s: (-ce_at.get(s, 0.0), s)
                )
            if pe_at and max(pe_at.values()) > 0:
                expected["opt_put_wall_near"] = min(
                    strikes, key=lambda s: (-pe_at.get(s, 0.0), s)
                )
        counts["rows"] += 1
        for name, value in expected.items():
            got = row[name]
            counts["values"] += 1
            if got == "" or abs(float(got) - value) > 1e-9 * max(1.0, abs(value)):
                counts["mismatches"] += 1
                if len(examples) < 15:
                    examples.append([row["date"], symbol, name, got, value])
    return {**counts, "examples": examples}


def vix_check() -> dict:
    """NIFTY's 30-day ATM IV against India VIX: NSE's own close (all-index file),
    else Kite's daily bars."""
    vix, source = {}, "nse_ind_close_all"
    for day in manifest_days("indices"):
        text = DATASETS["indices"].path(NSE, day).read_text("utf-8-sig")
        close = read_indices(text).get("India VIX", {}).get("close", math.nan)
        if math.isfinite(close):
            vix[day.isoformat()] = close
    if not vix:
        source = "kite"
        with VIX.open(newline="") as handle:
            for row in csv.DictReader(handle):
                vix[row["timestamp"][:10]] = float(row["close"])
    pairs = []
    with gzip.open(FNO / "features" / "index_daily.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["symbol"] == "NIFTY" and row["iv_atm_30d"] and row["date"] in vix:
                pairs.append((float(row["iv_atm_30d"]) * 100, vix[row["date"]]))
    if len(pairs) < 10:
        return {"pairs": len(pairs), "vix_source": source}
    ours, theirs = np.array(pairs).T
    return {
        "vix_source": source,
        "pairs": len(pairs),
        "correlation": float(np.corrcoef(ours, theirs)[0, 1]),
        "median_vix_minus_atm_iv_points": float(np.median(theirs - ours)),
        "p05_p95_difference": [
            float(np.percentile(theirs - ours, 5)),
            float(np.percentile(theirs - ours, 95)),
        ],
    }


def indices_check() -> dict:
    """NSE's all-index closes against Kite's daily index bars (12 indices)."""
    nse: dict[str, dict[str, dict[str, float]]] = {}
    for day in manifest_days("indices"):
        text = DATASETS["indices"].path(NSE, day).read_text("utf-8-sig")
        nse[day.isoformat()] = read_indices(text)
    counts: Counter[str] = Counter()
    examples = []
    folder = VIX.parent
    for filename, name in KITE_INDICES.items():
        path = folder / filename
        if not path.exists():
            counts["kite_file_missing"] += 1
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                day = row["timestamp"][:10]
                if day not in nse:
                    counts["kite_day_without_nse_file"] += 1
                    continue
                official = nse[day].get(name)
                if official is None:
                    counts["index_not_in_nse_file"] += 1
                    continue
                counts["compared"] += 1
                gap = abs(official["close"] - float(row["close"]))
                counts["close_equal"] += gap < 0.005
                # Kite carries some index values 0.01 below NSE's (2020).
                counts["close_within_0_01"] += gap < 0.0101
                if gap >= 0.0101 and len(examples) < 15:
                    examples.append([name, day, official["close"], row["close"]])
    return {**counts, "nse_sessions": len(nse), "examples": examples}


def underlying_check() -> dict:
    """UDiFF's underlying price against NSE's official closes of the same session:
    the stock's cash bhavcopy close (EQ, else BE) and the index's all-index close."""
    counts: Counter[str] = Counter()
    examples = []
    for day in manifest_days("fo"):
        cm_path = DATASETS["cm"].path(NSE, day)
        index_path = DATASETS["indices"].path(NSE, day)
        cm = read_cm(cm_path.read_text("utf-8-sig")) if cm_path.exists() else {}
        indices = (
            read_indices(index_path.read_text("utf-8-sig"))
            if index_path.exists()
            else {}
        )
        table, _ = bhavcopy.load_table(
            FNO / "normalized" / f"{day.year}" / f"{day}.npz"
        )
        seen = set()
        for i in range(len(table["symbol"])):
            symbol, kind = str(table["symbol"][i]), str(table["underlying"][i])
            price = float(table["underlying_price"][i])
            if (symbol, price) in seen or not math.isfinite(price):
                continue
            seen.add((symbol, price))
            if kind == "I":
                official = indices.get(INDEX_FILE_NAMES.get(symbol, ""), {}).get(
                    "close", math.nan
                )
                group = "index"
            else:
                official = cm.get(symbol, {}).get("cm_close", math.nan)
                group = "stock"
            if not math.isfinite(official):
                counts[f"{group}_no_official_close"] += 1
                continue
            counts[f"{group}_compared"] += 1
            if abs(price - official) < 0.005:
                counts[f"{group}_equal"] += 1
            elif len(examples) < 15:
                examples.append([day.isoformat(), symbol, price, official])
    return {**counts, "examples": examples}


def parity_check(rng: random.Random) -> dict:
    gaps = []
    paths = sorted((FNO / "normalized").rglob("*.npz"))
    for path in rng.sample(paths, min(60, len(paths))):
        table, _ = bhavcopy.load_table(path)
        day = np.datetime64(path.stem)
        stock = table["underlying"] == "S"
        for i in np.flatnonzero((table["kind"] == "F") & stock):
            symbol, expiry = table["symbol"][i], table["expiry"][i]
            years = float((expiry - day).astype(int)) / 365
            if years <= 0:
                continue
            group = np.flatnonzero(
                (table["symbol"] == symbol)
                & (table["expiry"] == expiry)
                & (table["kind"] == "O")
                & (table["contracts"] >= 10)
            )
            calls = {
                table["strike"][j]: table["close"][j]
                for j in group
                if table["option"][j] == "CE"
            }
            puts = {
                table["strike"][j]: table["close"][j]
                for j in group
                if table["option"][j] == "PE"
            }
            common = [k for k in calls if k in puts]
            if not common:
                continue
            k = min(common, key=lambda s: abs(s - table["settle"][i]))
            forward = k + (calls[k] - puts[k]) * math.exp(RATE * years)
            gaps.append((forward / table["settle"][i] - 1) * 1e4)
    gaps = np.abs(np.array(gaps))
    return {
        "pairs": int(len(gaps)),
        "median_abs_gap_bp": float(np.median(gaps)) if len(gaps) else None,
        "p90_abs_gap_bp": float(np.percentile(gaps, 90)) if len(gaps) else None,
    }


def capture_check() -> dict:
    root = FNO / "kite_minute"
    if not (root / "contracts.json").exists():
        return {"status": "no capture yet"}
    contracts = json.loads((root / "contracts.json").read_text())
    counts: Counter[str] = Counter()
    examples = []
    for path in sorted(root.glob("*.npz")):
        day = date.fromisoformat(path.stem)
        loaded = bhavcopy.load_session(NSE, day)
        if loaded is None:
            counts["sessions_without_nse_file_yet"] += 1
            continue
        table, _ = loaded
        keys = _key_table(table)
        with np.load(path) as bars:
            ids, minute = bars["contract"], bars["minute"]
            volume, oi = bars["volume"], bars["oi"]
        order = np.lexsort((minute, ids))
        ids, volume, oi = ids[order], volume[order], oi[order]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ends = np.r_[starts[1:], len(ids)] - 1
        sums = np.add.reduceat(volume, starts) if len(ids) else []
        for start, end, total in zip(starts, ends, sums, strict=True):
            info = contracts[str(int(ids[start]))]
            kind = "F" if info["kind"] == "FUT" else "O"
            underlying = (
                "I"
                if info["underlying"]
                in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")
                else "S"
            )
            key = (
                info["underlying"],
                kind,
                underlying,
                info["expiry"],
                None if kind == "F" else float(info["strike"]),
                "" if kind == "F" else info["kind"],
            )
            if key not in keys:
                counts["not_in_nse_file"] += 1
                continue
            i = keys[key]
            last_oi = int(oi[end])
            lot = int(table["lot"][i]) or int(info["lot_size"])
            expected = int(table["contracts"][i]) * lot
            counts["contracts"] += 1
            counts["volume_equal"] += int(total) == expected
            counts["last_oi_equal"] += last_oi == int(table["oi"][i])
            if (int(total) != expected or last_oi != int(table["oi"][i])) and len(
                examples
            ) < 15:
                examples.append(
                    [
                        day.isoformat(),
                        info["tradingsymbol"],
                        int(total),
                        expected,
                        last_oi,
                        int(table["oi"][i]),
                    ]
                )
    return {**counts, "examples": examples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--skip", default="", help="comma-separated check names")
    parser.add_argument("--only", default="", help="run just these checks")
    args = parser.parse_args(argv)
    names = ("coverage", "formats", "sessions", "cash", "participants", "ban")
    names += ("indices", "underlying", "implied_vol", "features", "vix", "parity")
    names += ("capture",)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if only:
        skip |= set(names) - only
    rng = random.Random(args.seed)
    report: dict = {}

    def run(name, function, *a):
        if name in skip:
            return
        print(f"check {name} ...", flush=True)
        report[name] = function(*a)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=1, default=str)
        )

    run("coverage", coverage)
    run("formats", formats)
    if "sessions" not in skip:
        all_days = sorted(p.stem for p in (FNO / "normalized").rglob("*.npz"))
        sample = {
            date.fromisoformat(d) for d in rng.sample(all_days, min(40, len(all_days)))
        }
        print("check sessions (cache, identity) ...", flush=True)
        checks, near = sessions_pass(sample)
        report.update(checks)
        run("kite_futures", kite_check, near)
    run("cash", cash)
    run("participants", participants)
    run("ban", ban_check)
    run("indices", indices_check)
    run("underlying", underlying_check)
    run("implied_vol", iv_check, rng)
    run("features", features_check, rng)
    run("vix", vix_check)
    run("parity", parity_check, rng)
    run("capture", capture_check)
    (args.output / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
