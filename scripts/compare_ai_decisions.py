"""Compare the AIs of one paper-trading day: cost, verdicts, hindsight, paper P&L.

Reads the main AI (``data/paper``) and every shadow (``data/paper/shadows/*``):
each journal's calls and verdicts, each paper store's trades, and the day's
captured final bars. Hindsight buys at the open of the minute after the decision
and asks which comes first, +1% or -1% (both in one minute counts as -1%);
"neither" means neither by 15:30. Run after the session (15:31) for hindsight.

    python -m scripts.compare_ai_decisions --day 2026-10-13
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

from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.paper.store import PaperStore

OPEN_MINUTE = 9 * 60 + 15


def first_touch(rows: list[list[float | None]], minute: int) -> str:
    """ "up", "down", "neither" or "no_entry" for a buy after bar ``minute``."""
    bars = np.array(rows, dtype=float)
    if minute + 1 >= len(bars) or not np.isfinite(bars[minute + 1, 0]):
        return "no_entry"
    entry = bars[minute + 1, 0]
    for open_, high, low in bars[minute + 1 :, :3]:
        if not np.isfinite(open_):
            continue
        if low <= entry * 0.99:
            return "down"
        if high >= entry * 1.01:
            return "up"
    return "neither"


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
    db.close()
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
    }
    if bars is not None:
        outcomes: dict[str, Counter] = {"BUY": Counter(), "NO_BUY": Counter()}
        errors = []
        for cycle_id, symbol, verdict, confidence in verdicts:
            rows = bars.get(symbol)
            touch = first_touch(rows, decision_minute(cycle_id)) if rows else "no_bars"
            outcomes.setdefault(verdict, Counter())[touch] += 1
            if touch in ("up", "down"):
                errors.append((confidence - (touch == "up")) ** 2)
        result["hindsight"] = {k: dict(v) for k, v in outcomes.items() if v}
        result["brier"] = round(float(np.mean(errors)), 4) if errors else None
        result["resolved"] = len(errors)
    if store.exists():
        paper = PaperStore(store)
        held = paper.positions("open")
        closed = [p for p in paper.positions("closed") if p.exit_at.date() == day]
        bought = [p for p in paper.positions() if p.entry_at.date() == day]
        result["paper"] = {
            "bought": len(bought),
            "closed": len(closed),
            "won": sum(p.net_pnl > 0 for p in closed),
            "net_pnl": float(sum(p.net_pnl for p in closed)),
            "open": len(held),
        }
    return result


def lines(result: dict[str, Any]) -> list[str]:
    verdicts = result["verdicts"]
    text = [
        result["ai"],
        f"  AI: {result['calls']} calls (+{result['failed_or_skipped']} failed or "
        f"skipped), ${result['cost_usd']:.2f}, {result['web_searches']} web "
        f"searches, {result['avg_seconds']} s per call",
        f"  verdicts: {sum(verdicts.values())} ({verdicts.get('BUY', 0)} BUY, "
        f"{verdicts.get('NO_BUY', 0)} NO_BUY)",
    ]
    for verdict, touches in result.get("hindsight", {}).items():
        text.append(
            f"  hindsight {verdict}: +1% first {touches.get('up', 0)}, -1% first "
            f"{touches.get('down', 0)}, neither {touches.get('neither', 0)}"
            + (
                f", no data {touches.get('no_entry', 0) + touches.get('no_bars', 0)}"
                if touches.get("no_entry") or touches.get("no_bars")
                else ""
            )
        )
    if "brier" in result:
        text.append(
            f"  confidence vs outcome (Brier, lower is better): {result['brier']} "
            f"on {result['resolved']} resolved verdicts"
        )
    if "paper" in result:
        p = result["paper"]
        text.append(
            f"  paper: {p['bought']} bought, {p['closed']} closed today "
            f"({p['won']} won) net Rs {p['net_pnl']:+,.2f}, {p['open']} open"
        )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--day",
        type=date.fromisoformat,
        default=datetime.now(INDIA_TIMEZONE).date(),
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("data/paper"))
    args = parser.parse_args(argv)
    final = args.paper_dir / "capture" / args.day.isoformat() / "final_bars.json"
    bars = json.loads(final.read_text("utf-8"))["bars"] if final.exists() else None
    if bars is None:
        print("final bars not saved yet (the run saves them at 15:31): no hindsight")
    folders = [args.paper_dir] + sorted((args.paper_dir / "shadows").glob("*"))
    for folder in folders:
        journal = folder / "decisions.sqlite"
        if journal.exists():
            result = compare_book(journal, folder / "paper.sqlite", args.day, bars)
            print("\n".join(lines(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
