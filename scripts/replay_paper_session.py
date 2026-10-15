"""End-to-end offline validation of the paper-trading system on past sessions.

Runs the real ``PaperSession`` (scanner bundle, components, decider, paper broker,
report) over consecutive historical sessions from the local Kite files, with a
simulated clock. Then checks independently:

1. every exit against a plain loop over the raw minute files (the +-1% target
   after the entry minute, else the last close before 15:00), and that nothing
   is left open;
2. live scanner features against the research code on the same real data;
3. report totals against the closed trades.

``--decider trade-all`` is a mechanical stand-in that answers LONG or SHORT (by
symbol) for every candidate, to exercise execution and limits; ``--decider
openai`` uses the real AI without web search (costs a few cents per session).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

from ai_trader.ai.decision import (
    DIRECTION,
    CycleResult,
    DecisionRequest,
    OpenAIDecider,
    Verdict,
    request_json,
)
from ai_trader.ai.journal import DecisionJournal
from ai_trader.broker import CandleInterval, Instrument
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.paper.broker import PaperBroker, PaperConfig
from ai_trader.paper.replay import ReplayData, ReplayLiveBars
from ai_trader.paper.session import (
    PaperSession,
    ReplayClock,
    SessionSettings,
    _array,
)
from ai_trader.paper.store import PaperStore
from ai_trader.scanner import upside
from ai_trader.scanner.upside import ScannerBundle


class TradeAllDecider:
    """Mechanical test stand-in: LONG or SHORT every candidate (by its symbol)."""

    def decide(self, request: DecisionRequest) -> CycleResult:
        verdicts = []
        for c in request.candidates:
            side = "LONG" if sum(map(ord, c.symbol)) % 2 == 0 else "SHORT"
            p_green = 0.6 if side == "LONG" else 0.4
            verdicts.append(
                Verdict(c.symbol, side, p_green, "replay: trade every one", "trade-all")
            )
        return CycleResult(
            tuple(verdicts),
            "ok",
            "trade-all",
            request_json=request_json(request),
            prompt_version="trade-all",
        )


def plain_exit(path: Path, entry_at: datetime, side: str, target: Decimal):
    """The first minute after the entry minute and before 15:00 that reaches the
    target (at its open if it opens beyond it); else the 15:00 exit at the last
    close from the entry minute to 14:59. By a plain loop over the raw file."""
    entry_minute = entry_at.replace(second=0, microsecond=0)
    deadline = entry_at.replace(hour=15, minute=0, second=0, microsecond=0)
    last_close = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for stamp, open_, high, low, close, _volume in reader:
            moment = datetime.fromisoformat(stamp)
            if moment < entry_minute:
                continue
            if moment >= deadline:
                break
            o, h, low_ = Decimal(open_), Decimal(high), Decimal(low)
            if moment > entry_minute:
                if side == "LONG":
                    if o >= target:
                        return moment, o, "tp_gap"
                    if h >= target:
                        return moment, target, "tp"
                else:
                    if o <= target:
                        return moment, o, "tp_gap"
                    if low_ <= target:
                        return moment, target, "tp"
            last_close = Decimal(close)
    return None if last_close is None else (deadline, last_close, "time_1500")


def research_features(
    candles: Path, symbol: str, day: date, minute: int, move: float
) -> dict:
    """Research inputs for one stock-minute from the same files, as
    scripts/research_scanner_move stored them (movement inputs included)."""
    from scripts import research_scanner_move as research
    from scripts.research_delivery_pipeline import MARKET, STOCK_FEATURES, epoch_day
    from scripts.research_delivery_pipeline import market_context as research_market
    from scripts.research_scanner_goals import DECISIONS, session_minutes

    calendar, days, market = research_market(candles, DECISIONS)
    rows = research.stock_rows((symbol, candles, calendar, round(move * 10_000)))
    epoch = epoch_day(day.isoformat())
    stock_row = int(np.flatnonzero(rows["days"] == epoch)[0])
    market_row = int(np.flatnonzero(days == epoch)[0])
    slot = int(np.flatnonzero(DECISIONS == minute)[0])
    names = (*STOCK_FEATURES, *research.RAW_EXTRA)
    values = dict(zip(names, rows["features"][stock_row, slot], strict=True))
    values["slot"] = np.float32(session_minutes(epoch) - 1 - minute)
    values |= dict(zip(MARKET, market[market_row, slot], strict=True))
    one = np.float32(1)
    values["rel_day"] = (one + values["gap"]) * (one + values["ret_open"]) - one
    values["rel_day"] -= values["nifty_day"]
    values["rel_20d"] = values["ret20d"] - values["nifty_ret20d"]
    grid = np.full((1, len(research.NAMES)), np.nan, np.float32)
    for column, name in enumerate(research.NAMES):
        if name in values:
            grid[0, column] = values[name]
    research.movement_scales(grid, move)
    return {name: float(grid[0, i]) for i, name in enumerate(research.NAMES)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candles", type=Path, default=Path("data/candles_fno_20261011")
    )
    parser.add_argument("--oi", type=Path, default=Path("data/oi_kite_20261011"))
    parser.add_argument(
        "--scanner", type=Path, default=Path("data/models/scanner_move10_by_close_v1")
    )
    parser.add_argument("--days", required=True, help="comma-separated ISO dates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decider", choices=("trade-all", "openai"), default="trade-all"
    )
    parser.add_argument("--stocks", type=int, default=0, help="limit (0 = all)")
    parser.add_argument("--max-positions", type=int, default=25)
    parser.add_argument(
        "--live",
        action="store_true",
        help="pre-screen on stream-style bars, re-score the shortlist on history",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; keep earlier validations")
    universe = json.loads((args.candles / "universe.json").read_text("utf-8"))
    symbols = sorted(
        e["symbol"] for e in universe["instruments"] if e["role"] == "equity_target"
    )
    if args.stocks:
        symbols = symbols[: args.stocks]
    bundle = ScannerBundle.load(args.scanner)
    store = PaperStore(args.output / "paper.sqlite")
    broker = PaperBroker(store, PaperConfig(max_positions=args.max_positions))
    journal = DecisionJournal(args.output / "decisions.sqlite")
    if args.decider == "openai":
        from ai_trader.ai.openai_client import OpenAIResponses
        from ai_trader.config import load_openai_settings

        settings = load_openai_settings()
        decider = OpenAIDecider(
            OpenAIResponses(
                settings.api_key.get_secret_value(),
                timeout_seconds=settings.timeout_seconds,
            ),
            settings.model,
            reasoning_effort=settings.reasoning_effort,
            policy=DIRECTION,  # no web search: it would see news after the replay
        )
        name = f"openai:{settings.model}:{settings.reasoning_effort}"
    else:
        decider, name = TradeAllDecider(), "trade-all"
    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)
        print(message, flush=True)

    sessions = []
    days = [date.fromisoformat(text) for text in args.days.split(",")]
    clock = ReplayClock(datetime.combine(days[0], datetime.min.time(), INDIA_TIMEZONE))
    data = ReplayData(
        args.candles, clock, oi=args.oi, first_day=days[0] - timedelta(days=70)
    )
    for day in days:
        clock.current = datetime(
            day.year, day.month, day.day, 8, 50, tzinfo=INDIA_TIMEZONE
        )
        session = PaperSession(
            data,
            clock,
            bundle,
            symbols,
            broker,
            decider,
            name,
            journal,
            report_dir=args.output / "report",
            settings=SessionSettings(workers=1),
            log=lambda m, d=day: log(f"{d} {m}"),
            live=ReplayLiveBars(data, day) if args.live else None,
        )
        session.prepare()
        session.exit_cycle()
        session.run(
            datetime(day.year, day.month, day.day, 15, 31, tzinfo=INDIA_TIMEZONE)
        )
        sessions.append(session)

    # 1. every exit, independently; nothing may be left open
    closed = store.positions("closed")
    mismatches = []
    for p in closed:
        expected = plain_exit(
            args.candles / f"NSE_{p.symbol}.csv", p.entry_at, p.side, p.target_price
        )
        got = (p.exit_at, p.exit_price, p.exit_reason)
        if expected is None or tuple(expected) != got:
            mismatches.append(
                {
                    "id": p.id,
                    "symbol": p.symbol,
                    "side": p.side,
                    "paper": str(got),
                    "plain": str(expected),
                }
            )
    still_open = store.positions("open")

    # 2. live features vs research on the same files (first candidates of each day)
    feature_checks = []
    for session in sessions:
        day_text = session.today.isoformat()
        scans = [e for e in store.events("scan") if e["at"][:10] == day_text]
        picks = [
            (s["detail"]["decision_at"], c["symbol"])
            for s in scans
            for c in s["detail"]["candidates"]
        ][:3]
        for decision_at, symbol in picks:
            moment = datetime.fromisoformat(decision_at)
            minute = upside.minute_index(moment) - 1
            clock.current = moment
            opening = moment.replace(hour=9, minute=15)
            arrays = {
                name: _array(
                    data.get_historical_candles(
                        Instrument("NSE", name),
                        opening,
                        moment,
                        CandleInterval.ONE_MINUTE,
                    ),
                    moment.date(),
                    minute + 1,
                )
                for name in (symbol, "NIFTY", "INDIAVIX")
            }
            live = upside.all_features(
                upside.stock_features(
                    arrays[symbol],
                    minute,
                    session.histories[symbol],
                    session.previous_session,
                    upside.session_minutes(moment.date(), symbol in session.fno),
                ),
                upside.market_features(
                    arrays["NIFTY"], arrays["INDIAVIX"], minute, session.market
                ),
                bundle.move,
            )
            research = research_features(
                args.candles, symbol, moment.date(), minute, bundle.move
            )
            worst = max(
                abs(live[k] - research[k]) / max(abs(research[k]), 1e-12)
                for k in bundle.features
                if np.isfinite(research[k])
            )
            feature_checks.append(
                {
                    "symbol": symbol,
                    "decision_at": decision_at,
                    "max_relative_difference": worst,
                }
            )

    # 3. report totals
    report = json.loads((args.output / "report" / "summary.json").read_text("utf-8"))
    total = sum((p.net_pnl for p in closed), Decimal(0))
    summary = {
        "days": args.days,
        "decider": name,
        "stocks": len(symbols),
        "closed_trades": len(closed),
        "sides": dict(Counter(p.side for p in closed)),
        "exit_reasons": dict(Counter(p.exit_reason for p in closed)),
        "left_open": [{"id": p.id, "symbol": p.symbol} for p in still_open],
        "exit_mismatches": mismatches,
        "feature_checks": feature_checks,
        "report_net_matches": report["summary"]["realised_net_pnl"]
        == str(total.quantize(Decimal("0.01"))),
        "entry_refusals": dict(
            Counter(e["detail"]["reason"] for e in store.events("entry_refused"))
        ),
        "errors": len(store.events("error")),
    }
    (args.output / "validation.json").write_text(json.dumps(summary, indent=1), "utf-8")
    (args.output / "session.log").write_text("\n".join(logs), "utf-8")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "feature_checks"}, indent=1
        )
    )
    print("feature checks:", feature_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
