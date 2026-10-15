"""Compare the AIs of one paper-trading day: cost, verdicts, hindsight, paper P&L.

Reads the main AI (``data/paper/intraday``) and every shadow (``.../shadows/*``):
each journal's calls and verdicts, each paper store's trades, and the day's
captured final bars (``data/paper/capture``). Hindsight compares the price at the
decision (the decision minute's close, the AI's "last_price") with the session's
last minute close: green if higher, red if lower. A LONG is right when green, a
SHORT when red. Run after the session (15:31) for hindsight.

    python -m scripts.compare_ai_decisions --day 2026-10-14
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.ai.decision import RESEARCH
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.paper.store import PaperStore

OPEN_MINUTE = 9 * 60 + 15


def close_direction(rows: list[list[float | None]], minute: int) -> str:
    """ "green", "red", "flat" or "no_data": the session's last close vs the close
    of decision bar ``minute``."""
    bars = np.array(rows, dtype=float)
    if minute >= len(bars) or not np.isfinite(bars[minute, 3]):
        return "no_data"
    closes = bars[minute + 1 :, 3]
    closes = closes[np.isfinite(closes)]
    if not len(closes):
        return "no_data"
    change = closes[-1] - bars[minute, 3]
    return "green" if change > 0 else "red" if change < 0 else "flat"


def decision_minute(cycle_id: str) -> int:
    """Bar index of the decision minute from a ``YYYYMMDD-HHMM-n`` cycle id."""
    hhmm = cycle_id.split("-")[1]
    return int(hhmm[:2]) * 60 + int(hhmm[2:]) - OPEN_MINUTE - 1


def compare_book(
    journal: Path, store: Path, day: date, bars: dict[str, Any] | None
) -> dict[str, Any]:
    key = day.strftime("%Y%m%d")
    db = sqlite3.connect(journal)
    cycles = db.execute(
        "SELECT decider, status, cost_usd, web_searches, latency_seconds "
        "FROM cycles WHERE substr(cycle_id, 1, 8) = ?",
        (key,),
    ).fetchall()
    verdicts = db.execute(
        "SELECT cycle_id, symbol, verdict, confidence FROM verdicts "
        "WHERE substr(cycle_id, 1, 8) = ?",
        (key,),
    ).fetchall()
    one_word = {
        cycle_id
        for (cycle_id,) in db.execute(
            "SELECT cycle_id FROM cycles WHERE substr(cycle_id, 1, 8) = ? "
            "AND prompt_version = ?",
            (key, RESEARCH.version),
        )
    }
    db.close()
    # One-word verdicts carry no probability (stored as 0.5): no Brier, no check.
    scored = [v for v in verdicts if v[0] not in one_word]
    calls = [c for c in cycles if c[1] == "ok"]
    result: dict[str, Any] = {
        "ai": Counter(c[0] for c in cycles).most_common(1)[0][0]
        if cycles
        else journal.parent.name,
        "calls": len(calls),
        "failed_or_skipped": len(cycles) - len(calls),
        "cost_usd": round(sum(c[2] for c in cycles), 4),
        "web_searches": sum(c[3] or 0 for c in cycles),
        "avg_seconds": round(sum(c[4] for c in calls) / len(calls), 1)
        if calls
        else None,
        "verdicts": dict(Counter(v[2] for v in verdicts)),
        # The verdict contradicts the AI's own P(green) by a clear margin.
        "disagree": sum(
            (verdict == "LONG" and p < 0.4) or (verdict == "SHORT" and p > 0.6)
            for _, _, verdict, p in scored
        ),
    }
    if bars is not None:
        outcomes: dict[str, Counter] = {}
        errors = []
        for cycle_id, symbol, verdict, confidence in verdicts:
            rows = bars.get(symbol)
            seen = (
                close_direction(rows, decision_minute(cycle_id)) if rows else "no_data"
            )
            outcomes.setdefault(verdict, Counter())[seen] += 1
            if seen in ("green", "red") and cycle_id not in one_word:
                errors.append((confidence - (seen == "green")) ** 2)
        result["hindsight"] = {k: dict(v) for k, v in outcomes.items() if v}
        result["brier"] = round(float(np.mean(errors)), 4) if errors else None
        result["resolved"] = len(errors)
    if store.exists():
        paper = PaperStore(store)
        held = paper.positions("open")
        closed = [p for p in paper.positions("closed") if p.exit_at.date() == day]
        entered = [p for p in paper.positions() if p.entry_at.date() == day]
        result["paper"] = {
            "entered": len(entered),
            "long": sum(p.side == "LONG" for p in entered),
            "short": sum(p.side == "SHORT" for p in entered),
            "closed": len(closed),
            "won": sum(p.net_pnl > 0 for p in closed),
            "target_exits": sum(p.exit_reason.startswith("tp") for p in closed),
            "net_pnl": float(sum(p.net_pnl for p in closed)),
            "open": len(held),
        }
    return result


def lines(result: dict[str, Any]) -> list[str]:
    verdicts = result["verdicts"]
    text = [
        result["ai"],
        f"  AI: {result['calls']} calls (+{result['failed_or_skipped']} failed or "
        f"skipped), ${result['cost_usd']:.2f}, {result['web_searches']} web tool "
        f"calls, {result['avg_seconds']} s per call",
        f"  verdicts: {sum(verdicts.values())} ({verdicts.get('LONG', 0)} LONG, "
        f"{verdicts.get('SHORT', 0)} SHORT, {verdicts.get('NO_TRADE', 0)} NO_TRADE); "
        f"against its own P(green) (LONG below 0.4, SHORT above 0.6): "
        f"{result['disagree']}",
    ]
    for verdict, seen in result.get("hindsight", {}).items():
        right = {"LONG": "green", "SHORT": "red"}.get(verdict)
        calls = seen.get("green", 0) + seen.get("red", 0)
        text.append(
            f"  hindsight {verdict}: closed green {seen.get('green', 0)}, red "
            f"{seen.get('red', 0)}, flat {seen.get('flat', 0)}"
            + (f", no data {seen['no_data']}" if seen.get("no_data") else "")
            + (
                f" -> right {seen.get(right, 0)} of {calls}"
                if right is not None and calls
                else ""
            )
        )
    if result.get("brier") is not None:
        text.append(
            f"  P(green at close) vs outcome (Brier, lower is better): "
            f"{result['brier']} on {result['resolved']} resolved verdicts"
        )
    if "paper" in result:
        p = result["paper"]
        text.append(
            f"  paper: {p['entered']} entered ({p['long']} long, {p['short']} short), "
            f"{p['closed']} closed today ({p['won']} won, {p['target_exits']} at the "
            f"target) net Rs {p['net_pnl']:+,.2f}, {p['open']} open"
        )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--day",
        type=date.fromisoformat,
        default=datetime.now(INDIA_TIMEZONE).date(),
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("data/paper/intraday"))
    parser.add_argument("--capture-dir", type=Path, default=Path("data/paper/capture"))
    args = parser.parse_args(argv)
    final = args.capture_dir / args.day.isoformat() / "final_bars.json"
    bars = json.loads(final.read_text("utf-8"))["bars"] if final.exists() else None
    if bars is None:
        print("final bars not saved yet (the run saves them at 15:31): no hindsight")
    folders = [args.paper_dir] + sorted((args.paper_dir / "shadows").glob("*"))
    for folder in folders:
        journal = folder / "decisions.sqlite"
        if journal.exists():
            result = compare_book(journal, folder / "paper.sqlite", args.day, bars)
            if result["calls"] + result["failed_or_skipped"]:
                print("\n".join(lines(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
