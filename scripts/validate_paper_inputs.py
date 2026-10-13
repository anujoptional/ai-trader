"""Validate the live paper session's AI inputs against independent sources.

Reads the raw inputs each decision cycle captured (``data/paper/capture/<day>/``),
the history cache the session used, the AI journals, and checks them against:

- NSE's official end-of-day files of the previous session (closes, futures OI),
- NSE's index API (previous closes of NIFTY 50 and INDIA VIX),
- NSE's announcements API, fetched afresh (filings the AI was shown),
- the research feature code (``scripts/research_delivery_pipeline.py``),
- Kite's own complete bars fetched after the close (bars used live were final?),
- internal consistency of every order-book snapshot,
- the v3 requests and replies in the journals (scanner output absent, web searches).

Writes ``report.json`` and prints a summary. Read-only: no orders, no Kite login.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ai_trader.ai.packet import V3_DAILY, V3_MARKET, V3_TODAY
from ai_trader.broker import DepthLevel, DepthSnapshot, Instrument
from ai_trader.clock import INDIA_TIMEZONE
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


def bhavcopy(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def research_stock_features(
    bars: np.ndarray, minute: int, cache: Path, day: date, fno: bool
) -> dict[str, float]:
    """The stock features at ``minute`` by the research code, from raw inputs."""
    with np.load(cache) as saved:
        sessions, daily = saved["sessions"], saved["daily"]
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
    return {k: float(v) for k, v in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", default=date.today().isoformat())
    parser.add_argument("--paper", type=Path, default=Path("data/paper"))
    parser.add_argument("--nse", type=Path, required=True, help="bhavcopy folder")
    parser.add_argument("--previous", required=True, help="previous session date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    day = date.fromisoformat(args.day)
    previous = args.previous.replace("-", "")
    folder = args.paper / "capture" / args.day
    captures = {
        p.stem: json.loads(p.read_text("utf-8"))
        for p in sorted(folder.glob("[0-9]*.json"))
    }
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
    fo = {
        r["FinInstrmNm"]: r
        for r in bhavcopy(args.nse / f"BhavCopy_NSE_FO_0_0_0_{previous}_F_0000.csv")
        if r["FinInstrmTp"] == "STF"
    }
    nse = NseClient(timeout_seconds=15, attempts=2)
    indices = {
        r["index"]: r for r in nse.get_json("/api/allIndices", {}).get("data", [])
    }
    announcements = nse.announcements(day, day)

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
            research = research_stock_features(rows, minute, path, day, info["fno"])
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
    for name, cap in captures.items():
        minute = cap["minute"]
        for key, raw in (cap.get("depth") or {}).items():
            exchange, symbol = key.split(":")
            books["snapshots"] += 1
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
            if low and high and not float(low) < last < float(high):
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
        "issues": book_issues[:30],
    }

    # 5. futures OI: previous-day baseline vs NSE F&O bhavcopy; live change -----------
    oi_checks = []
    for name, cap in captures.items():
        for symbol, contracts in (cap.get("futures") or {}).items():
            values = (cap.get("oi") or {}).get(symbol) or {}
            if not contracts:
                continue
            near = contracts[0]
            official = fo.get(near)
            live_raw = (cap.get("depth") or {}).get(f"NFO:{near}")
            check = {"cycle": name, "symbol": symbol, "near": near}
            if official:
                check["prev_oi_kite"] = values.get("open_interest")
                check["prev_oi_nse"] = float(official["OpnIntrst"])
                check["prev_fut_close_kite"] = values.get("fut_close")
                check["prev_fut_close_nse"] = float(official["ClsPric"])
                check["expiry_nse"] = official["XpryDt"]
                if len(contracts) > 1 and contracts[1] in fo:
                    check["next_expiry_nse"] = fo[contracts[1]]["XpryDt"]
            if live_raw and values.get("open_interest"):
                live_change = live_raw["open_interest"] / values["open_interest"] - 1
                check["oi_change_today_recomputed"] = round(live_change, 6)
                check["oi_change_today_sent"] = values.get("oi_change_since_prev_close")
            oi_checks.append(check)
    report["futures_oi"] = oi_checks

    # 6. filings: what the AI was shown vs a fresh NSE fetch --------------------------
    filing_checks = []
    since = datetime.combine(
        date.fromisoformat(f"{previous[:4]}-{previous[4:6]}-{previous[6:]}"),
        datetime.min.time(),
        INDIA_TIMEZONE,
    ).replace(hour=15, minute=30)
    journal = sqlite3.connect(args.paper / "decisions.sqlite")
    requests = journal.execute(
        "select cycle_id, request_json from cycles"
        " where prompt_version like 'delivery-v3%'"
        " and substr(cycle_id, 1, 8) = ?",
        (day.strftime("%Y%m%d"),),
    ).fetchall()
    for cycle_id, text in requests:
        request = json.loads(text)
        as_of = datetime.fromisoformat(request["as_of"])
        for stock in request["stocks"]:
            fresh = [
                a
                for a in announcements
                if a.symbol == stock["symbol"] and since < disseminated_at(a) <= as_of
            ]
            shown = (stock.get("nse_filings") or {}).get("filings_since_prev_close")
            filing_checks.append(
                {
                    "cycle": cycle_id,
                    "symbol": stock["symbol"],
                    "shown_to_ai": shown,
                    "nse_now": len(fresh),
                    "subjects_nse": [a.subject[:80] for a in fresh][:5],
                }
            )
    report["filings"] = filing_checks

    # 7. requests and replies: no scanner output, values = captured features, searches
    calls = []
    for path in [
        args.paper / "decisions.sqlite",
        *sorted((args.paper / "shadows").glob("*/decisions.sqlite")),
    ]:
        db = sqlite3.connect(path)
        for cycle_id, model, status, cost, latency, req, reply in db.execute(
            "select cycle_id, model, status, cost_usd, latency_seconds, request_json,"
            " response_json from cycles where prompt_version like 'delivery-v3%'"
            " and substr(cycle_id, 1, 8) = ?",
            (day.strftime("%Y%m%d"),),
        ):
            request, answer = json.loads(req), json.loads(reply or "{}")
            cap = captures.get(cycle_id.split("-")[1])
            mapping_errors = 0
            for stock in request["stocks"]:
                live = (
                    (cap or {})
                    .get("stocks", {})
                    .get(stock["symbol"], {})
                    .get("features")
                )
                if not live:
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
            names = {s["symbol"] for s in request["stocks"]}
            companies = {
                s["symbol"]: (s.get("company") or "").lower() for s in request["stocks"]
            }
            searches = answer.get("web_searches", [])
            queries = " ".join(str(s.get("query") or "") for s in searches).lower()
            sources = [u for s in searches for u in s.get("sources", [])]
            calls.append(
                {
                    "journal": path.parent.name,
                    "cycle": cycle_id,
                    "model": model,
                    "status": status,
                    "cost_usd": round(cost, 4),
                    "seconds": round(latency, 1),
                    "stocks": sorted(names),
                    "alphabetical": [s["symbol"] for s in request["stocks"]]
                    == sorted(names),
                    "scanner_output_sent": any(
                        word in req for word in ("p_up", "probability", "scanner")
                    ),
                    "mapping_errors": mapping_errors,
                    "searches": len(searches),
                    "queries": [s.get("query") for s in searches][:6],
                    "stocks_searched": sorted(
                        s
                        for s in names
                        if s.lower() in queries
                        or (companies[s] and companies[s].split()[0] in queries)
                    ),
                    "sources": len(sources),
                    "domains": Counter(urlparse(u).netloc for u in sources).most_common(
                        5
                    ),
                    "verdicts": [
                        (d["symbol"], d["verdict"], d["confidence"])
                        for d in answer.get("decisions", [])
                    ],
                }
            )
    report["ai_calls"] = calls

    # 8. executions: fills recomputed from the captured order book --------------------
    report["executions"] = [
        e for c in captures.values() for e in c.get("executions", [])
    ]
    report["execution_dry_run"] = dry_run(captures, final, day, args.output)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=1, default=str), "utf-8"
    )
    summarize(report)
    return 0


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


def plain_exit(rows, after: int, target: Decimal, stop: Decimal):
    """First minute after ``after`` that reaches a level, by a plain loop."""
    for i in range(after + 1, len(rows)):
        if rows[i][0] is None:
            continue
        o, h, low = (Decimal(str(round(rows[i][k], 2))) for k in (0, 1, 2))
        if o <= stop:
            return i, o, "sl_gap"
        if o >= target:
            return i, o, "tp_gap"
        if low <= stop:
            return i, stop, "sl_same_minute" if h >= target else "sl"
        if h >= target:
            return i, target, "tp"
    return None


def dry_run(captures: dict, final: dict, day: date, output: Path) -> list[dict]:
    """Buy every captured live order book in a scratch store; check by hand."""
    from ai_trader.broker import OHLCVCandle
    from ai_trader.paper.broker import PaperBroker, PaperConfig
    from ai_trader.paper.replay import nse_tick
    from ai_trader.paper.store import PaperStore

    results = []
    scratch = output / "dry_run"
    scratch.mkdir(parents=True, exist_ok=True)
    opening = datetime.combine(day, datetime.min.time(), INDIA_TIMEZONE).replace(
        hour=9, minute=15
    )
    for name, cap in captures.items():
        for key, raw in (cap.get("depth") or {}).items():
            exchange, symbol = key.split(":")
            if exchange != "NSE" or not raw["asks"]:
                continue
            path = scratch / f"{name}_{symbol}.sqlite"
            path.unlink(missing_ok=True)
            broker = PaperBroker(PaperStore(path), PaperConfig(capital=Decimal(10**9)))
            snapshot = snapshot_from(raw, symbol)
            tick = nse_tick(snapshot.last_price)
            now = snapshot.timestamp
            result = broker.buy(
                symbol,
                snapshot,
                tick,
                now,
                cycle_id=name,
                probability=0,
                confidence=0,
                reason="dry run",
            )
            position = result.position
            check = {"cycle": name, "symbol": symbol, "filled": position is not None}
            if position is None:
                check["refused"] = result.reason
                results.append(check)
                continue
            asks = [(Decimal(p), int(q)) for p, q, _ in raw["asks"]]
            quantity = int(Decimal(200_000) // asks[0][0])
            left, cost = quantity, Decimal(0)
            for price, size in asks:
                take = min(left, size)
                cost += take * price
                left -= take
            cost += left * asks[-1][0]
            fill = cost / quantity
            up = (fill * Decimal("1.01") / tick).to_integral_value(
                "ROUND_CEILING"
            ) * tick
            down = (fill * Decimal("0.99") / tick).to_integral_value(
                "ROUND_FLOOR"
            ) * tick
            check |= {
                "quantity_ok": position.quantity == quantity,
                "fill_ok": position.entry_price == fill,
                "levels_ok": position.target_price == up
                and position.stop_price == down,
                "fill": str(round(fill, 2)),
                "best_ask": str(asks[0][0]),
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
                buy_minute = int((now - opening).total_seconds() // 60)
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
                    if row[0] is not None
                ]
                end = opening + timedelta(minutes=len(rows) + 2)
                closed = broker.apply_bars(position, bars, end)
                expected = plain_exit(
                    rows, buy_minute, position.target_price, position.stop_price
                )
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
            results.append(check)
    return results


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
            f"searches={call['searches']}",
            f"searched={call['stocks_searched']}",
            f"sources={call['sources']}",
            call["domains"][:3],
            call["verdicts"],
        )
    print(f"executions: {len(report['executions'])}")
    for check in report["execution_dry_run"]:
        print("dry-run buy", check)


if __name__ == "__main__":
    raise SystemExit(main())
