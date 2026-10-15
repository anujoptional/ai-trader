"""Paper trade report: every trade with its P&L, cumulative totals, and the open
portfolio marked at the latest prices.

Written as plain files so it can be opened without the program: ``trades.csv``,
``open_positions.csv``, ``daily.csv``, ``summary.json`` and a readable
``report.txt``. Open positions show what closing now would leave after charges
(intraday if opened today, delivery otherwise). Longs gain when the price rises,
shorts when it falls.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

from ai_trader.paper.charges import trade_charges
from ai_trader.paper.store import PaperStore, Position

PAISA = Decimal("0.01")


def _r(value: Decimal) -> Decimal:
    return value.quantize(PAISA, rounding=ROUND_HALF_EVEN)


def _pct(numerator: Decimal, denominator: Decimal) -> float:
    return round(float(numerator / denominator * 100), 3) if denominator else 0.0


def _legs(position: Position, exit_value: Decimal) -> tuple[Decimal, Decimal]:
    """(bought, sold) rupee values of a position closed at ``exit_value``."""
    if position.side == "LONG":
        return position.entry_value, exit_value
    return exit_value, position.entry_value


def build_report(
    store: PaperStore,
    marks: Mapping[str, Decimal],
    now: datetime,
    capital: Decimal | None,
    target: Decimal | None = None,
) -> dict[str, Any]:
    """``capital`` None: a portfolio without a capital limit (no equity or cash)."""
    closed = sorted(store.positions("closed"), key=lambda p: (p.exit_at, p.id))
    held = store.positions("open")
    trades, cumulative = [], Decimal(0)
    for p in closed:
        cumulative += p.net_pnl
        trades.append(
            {
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side,
                "entry_at": p.entry_at.isoformat(timespec="seconds"),
                "entry_price": str(_r(p.entry_price)),
                "quantity": p.quantity,
                "entry_value": str(_r(p.entry_value)),
                "exit_at": p.exit_at.isoformat(timespec="minutes"),
                "exit_price": str(p.exit_price),
                "exit_reason": p.exit_reason,
                "charges_kind": (p.charges_detail or {}).get("kind"),
                "gross_pnl": str(p.gross_pnl),
                "charges": str(p.charges),
                "net_pnl": str(p.net_pnl),
                "net_return_pct": _pct(p.net_pnl, p.entry_value),
                "cumulative_net_pnl": str(_r(cumulative)),
                "scanner_probability": round(p.scanner_probability, 4),
                "ai_confidence": round(p.ai_confidence, 3),
                "ai_reason": p.ai_reason,
            }
        )
    daily: dict[str, dict[str, Any]] = {}
    running = Decimal(0)
    for p in closed:
        day = p.exit_at.date().isoformat()
        row = daily.setdefault(
            day, {"date": day, "trades": 0, "wins": 0, "net_pnl": Decimal(0)}
        )
        row["trades"] += 1
        row["wins"] += p.net_pnl > 0
        row["net_pnl"] += p.net_pnl
    for row in daily.values():
        running += row["net_pnl"]
        row["cumulative_net_pnl"] = str(_r(running))
        row["net_pnl"] = str(_r(row["net_pnl"]))
    open_rows, unrealised = [], Decimal(0)
    for p in held:
        mark = marks.get(p.symbol)
        row = {
            "id": p.id,
            "symbol": p.symbol,
            "side": p.side,
            "entry_at": p.entry_at.isoformat(timespec="seconds"),
            "entry_price": str(_r(p.entry_price)),
            "quantity": p.quantity,
            "entry_value": str(_r(p.entry_value)),
            "target_price": str(p.target_price),
            "last_price": str(mark) if mark is not None else None,
            "gross_pnl": None,
            "net_pnl_if_closed_now": None,
            "return_pct": None,
            "days_held": (now.date() - p.entry_at.date()).days,
        }
        if mark is not None:
            bought, sold = _legs(p, mark * p.quantity)
            gross = sold - bought
            charges = trade_charges(bought, sold, now.date() == p.entry_at.date()).total
            row["gross_pnl"] = str(_r(gross))
            row["net_pnl_if_closed_now"] = str(_r(gross - charges))
            row["return_pct"] = _pct(gross, p.entry_value)
            unrealised += gross - charges
        open_rows.append(row)
    wins = [p for p in closed if p.net_pnl > 0]
    realised = sum((p.net_pnl for p in closed), Decimal(0))
    deployed = sum((p.entry_value for p in held), Decimal(0))
    summary = {
        "as_of": now.isoformat(timespec="seconds"),
        "starting_capital": None if capital is None else str(capital),
        "target_pct": None if target is None else f"{float(target) * 100:g}",
        "closed_trades": len(closed),
        "long_trades": sum(p.side == "LONG" for p in closed),
        "short_trades": sum(p.side == "SHORT" for p in closed),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "win_rate_pct": round(100 * len(wins) / len(closed), 2) if closed else None,
        "target_exits": sum(p.exit_reason.startswith("tp") for p in closed),
        "time_exits": sum(p.exit_reason.startswith("time") for p in closed),
        "stop_exits": sum(p.exit_reason.startswith("sl") for p in closed),
        "gross_pnl": str(_r(sum((p.gross_pnl for p in closed), Decimal(0)))),
        "charges": str(_r(sum((p.charges for p in closed), Decimal(0)))),
        "realised_net_pnl": str(_r(realised)),
        "average_net_pnl_per_trade": str(_r(realised / len(closed)))
        if closed
        else None,
        "best_trade": str(max(p.net_pnl for p in closed)) if closed else None,
        "worst_trade": str(min(p.net_pnl for p in closed)) if closed else None,
        "open_positions": len(held),
        "deployed": str(_r(deployed)),
        "unrealised_net_pnl_if_closed_now": str(_r(unrealised)),
        "positions_without_price": sum(r["last_price"] is None for r in open_rows),
        "equity": None if capital is None else str(_r(capital + realised + unrealised)),
        "cash_available": None
        if capital is None
        else str(_r(capital + realised - deployed)),
    }
    return {
        "summary": summary,
        "trades": trades,
        "daily": list(daily.values()),
        "open_positions": open_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    trade_fields = [
        "id",
        "symbol",
        "side",
        "entry_at",
        "entry_price",
        "quantity",
        "entry_value",
        "exit_at",
        "exit_price",
        "exit_reason",
        "charges_kind",
        "gross_pnl",
        "charges",
        "net_pnl",
        "net_return_pct",
        "cumulative_net_pnl",
        "scanner_probability",
        "ai_confidence",
        "ai_reason",
    ]
    open_fields = [
        "id",
        "symbol",
        "side",
        "entry_at",
        "entry_price",
        "quantity",
        "entry_value",
        "target_price",
        "last_price",
        "gross_pnl",
        "net_pnl_if_closed_now",
        "return_pct",
        "days_held",
    ]
    _write_csv(directory / "trades.csv", report["trades"], trade_fields)
    _write_csv(directory / "open_positions.csv", report["open_positions"], open_fields)
    _write_csv(
        directory / "daily.csv",
        report["daily"],
        ["date", "trades", "wins", "net_pnl", "cumulative_net_pnl"],
    )
    (directory / "summary.json").write_text(
        json.dumps(report, indent=1, default=str), "utf-8"
    )
    (directory / "report.txt").write_text(render(report), "utf-8")


def render(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"PAPER TRADING REPORT  {s['as_of']}",
        "",
        f"Closed trades {s['closed_trades']} ({s['long_trades']} long, "
        f"{s['short_trades']} short): {s['wins']} wins, {s['losses']} losses"
        + (f" ({s['win_rate_pct']}%)" if s["win_rate_pct"] is not None else ""),
        f"Exits: {s['target_exits']} at the "
        + (f"{s['target_pct']}% " if s.get("target_pct") else "")
        + f"target, {s['time_exits']} at 15:00"
        + (f", {s['stop_exits']} at a stop" if s["stop_exits"] else ""),
        f"Gross P&L Rs {s['gross_pnl']}  charges Rs {s['charges']}  "
        f"net Rs {s['realised_net_pnl']}",
        f"Open positions {s['open_positions']}: value Rs {s['deployed']}, "
        f"net if closed now Rs {s['unrealised_net_pnl_if_closed_now']}",
        "Capital: no limit"
        if s["starting_capital"] is None
        else f"Capital Rs {s['starting_capital']}  equity Rs {s['equity']}  "
        f"cash Rs {s['cash_available']}",
        "",
        "Trades:",
    ]
    for t in report["trades"]:
        lines.append(
            f"  {t['exit_at'][:16]} {t['symbol']:<12} {t['side']:<5} qty "
            f"{t['quantity']:>5} in {t['entry_price']:>10} out {t['exit_price']:>10} "
            f"{t['exit_reason']:<10} net {t['net_pnl']:>10} cum "
            f"{t['cumulative_net_pnl']:>11}"
        )
    lines.append("")
    lines.append("Open portfolio:")
    for p in report["open_positions"]:
        lines.append(
            f"  {p['symbol']:<12} {p['side']:<5} qty {p['quantity']:>5} in "
            f"{p['entry_price']:>10} last {p['last_price'] or '-':>10} return "
            f"{p['return_pct'] if p['return_pct'] is not None else '-':>7}% "
            f"net if closed {p['net_pnl_if_closed_now'] or '-':>10} "
            f"(target {p['target_price']}, else out at 15:00)"
        )
    return "\n".join(lines) + "\n"
