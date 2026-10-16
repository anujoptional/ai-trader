"""Autonomous Claude intraday session (experiment, branch experiment/autonomous-agent).

Paper only: no order reaches Kite. Claude runs the whole session by itself: it picks
its own stocks (movers over the ~650 largest NSE companies, any NSE symbol, the web),
its strategy, the size of every trade (whole shares, up to 5x the Rs 20 lakh wallet)
and its exits. There is no scanner, model estimate or rule from us. Code answers its
tool calls with facts (live quotes and order books, bars, NSE daily data with F&O
positioning, NSE filings), fills its orders against the live order book, watches its
stops, targets and alerts every 20 seconds, and closes everything at 15:10.

Timeline: Kite sign-in and data load; 09:00 first turn (pre-open; orders before
09:15 fill at the official opening price); turns whenever Claude asks (1-30 minutes)
or an alert, stop or target wakes it; last turn starts by 15:07; 15:10 everything is
closed at market; a short review turn; the report; optionally the F&O capture.

  python -m ai_trader.cli.run_agent --login-wait 360 --fno-capture data/fno/kite_minute

State per day under data/paper/autonomous/YYYYMMDD: paper.sqlite (positions),
turns.jsonl (one line per turn: tools, searches, trades, notes, cost),
transcripts/turn_NNN.json (everything Claude saw and said), report.json. The log is
data/paper/logs/YYYYMMDD.log. Requires KITE_API_KEY / KITE_API_SECRET and
ANTHROPIC_API_KEY in .env. A second copy, weekends and NSE holidays exit at once.
"""

from __future__ import annotations

import argparse
import json
import threading
import traceback
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from ai_trader.agent.account import PaperAccount
from ai_trader.agent.brain import CLOSE_AT, AgentTools, AutonomousAgent, spent_today
from ai_trader.agent.market import Filings, MarketView, NseDaily
from ai_trader.agent.session import (
    REVIEW,
    SessionRunner,
    close_everything,
    report,
)
from ai_trader.ai.anthropic_client import CLAUDE_EFFORTS, AnthropicMessages
from ai_trader.ai.openai_client import PRICES
from ai_trader.broker.kite import KiteBrokerError
from ai_trader.broker.kite_auth import login_browser
from ai_trader.cli.run_paper import (
    RunLog,
    closed_today,
    single_instance,
    universe_profiles,
    universe_symbols,
)
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import (
    ConfigurationError,
    load_anthropic_settings,
    load_kite_login_settings,
)
from ai_trader.fno.kite_capture import capture as capture_fno
from ai_trader.news.nse import NseClient
from ai_trader.paper.session import SystemClock

WATCH_SECONDS = 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("data/universe_mcap_20261012/universe.json"),
        help="the scan list for movers and breadth (any NSE symbol can be looked up)",
    )
    parser.add_argument("--features", type=Path, default=Path("data/fno/features"))
    parser.add_argument("--dir", type=Path, default=Path("data/paper/autonomous"))
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("data/paper/run.lock"),
        help="shared with the regular paper run: only one Kite session a day",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("data/paper/logs"))
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", choices=CLAUDE_EFFORTS, default="medium")
    parser.add_argument("--budget", type=float, default=30.0, help="USD a day")
    parser.add_argument("--wallet", type=Decimal, default=Decimal(2_000_000))
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--ai-timeout", type=float, default=300.0)
    parser.add_argument("--login-wait", type=int, default=360)
    parser.add_argument("--fno-capture", type=Path, default=None)
    parser.add_argument("--fno-capture-until", default="18:15")
    args = parser.parse_args(argv)
    if args.model not in PRICES:
        parser.error(f"--model must be one of {', '.join(PRICES)}")

    now = datetime.now(INDIA_TIMEZONE)
    log = RunLog(args.log_dir / f"{now:%Y%m%d}.log")
    lock = single_instance(args.lock)
    if lock is None:
        log("another paper run is already running; not starting the agent")
        return 3
    close_at = datetime.combine(now.date(), CLOSE_AT, INDIA_TIMEZONE)
    reason = closed_today(now, close_at, NseClient(timeout_seconds=10, attempts=2))
    if reason:
        log(f"no agent session today: {reason}")
        return 0
    try:
        anthropic = load_anthropic_settings()
    except ConfigurationError as error:
        log(f"Claude is not configured ({error})")
        return 2
    day_dir = args.dir / f"{now:%Y%m%d}"
    daily = NseDaily(args.features, now.date())
    scan = universe_symbols(args.universe)
    profiles = universe_profiles(args.universe)
    log(
        f"autonomous agent {args.model} ({args.effort}); wallet Rs {args.wallet:,} "
        f"up to {args.leverage}x; AI budget ${args.budget:g}; NSE daily data to "
        f"{daily.latest}; scan list {len(scan)} stocks; state in {day_dir}"
    )
    deadline = datetime.now(INDIA_TIMEZONE) + timedelta(minutes=args.login_wait)
    log(
        "Kite login: sign in on the Kite tab that just opened in the browser "
        f"(waiting until {deadline:%H:%M})"
    )
    try:
        kite = login_browser(
            load_kite_login_settings(), timeout_seconds=60 * args.login_wait
        )
    except (ConfigurationError, KiteBrokerError, ValueError) as error:
        log(f"Kite login failed: {error}")
        return 2
    log("Kite login done")

    clock = SystemClock()
    nse = NseClient(timeout_seconds=15, attempts=2)
    market = MarketView(
        kite,
        scan,
        profiles,
        daily,
        clock.now,
        filings=Filings(lambda start, end: nse.announcements(start, end), clock.now),
    )
    account = PaperAccount(day_dir / "paper.sqlite", args.wallet, args.leverage)
    tools = AgentTools(market, account)
    agent = AutonomousAgent(
        AnthropicMessages(
            anthropic.api_key.get_secret_value(), timeout_seconds=args.ai_timeout
        ),
        args.model,
        args.effort,
        tools,
        budget_usd=args.budget,
        spent_usd=spent_today(day_dir),
        journal_dir=day_dir,
        log=log,
        thinking=not args.no_thinking,
    )

    runner = SessionRunner(tools, agent, log, clock.now())
    stop = threading.Event()

    def watcher() -> None:
        while not stop.wait(WATCH_SECONDS):
            runner.watch(clock.now())

    threading.Thread(target=watcher, name="agent-watch", daemon=True).start()
    today = now.date()
    try:
        while (pause := runner.step(clock.now())) is not None:
            if pause > 0:
                runner.wake.wait(pause)
        closed = close_everything(tools, log)
    except BaseException:
        log("agent run stopped:\n" + traceback.format_exc())
        raise
    finally:
        stop.set()
    if agent.budget_left() > 0:
        try:
            agent.turn(REVIEW, [{"event": "closed_at_1510", "closed": closed}])
        except Exception as error:
            log(f"review turn failed: {type(error).__name__}: {error}")
    summary = report(account, agent)
    (day_dir / "report.json").write_text(json.dumps(summary, indent=1), "utf-8")
    log(
        f"agent day: {summary['trades']} trades, {summary['won']} won, gross Rs "
        f"{summary['gross_rs']:,.0f}, charges Rs {summary['charges_rs']:,.0f}, net Rs "
        f"{summary['net_rs']:,.0f}; AI ${summary['ai_spend_usd']:.2f} over "
        f"{summary['turns']} turns"
    )
    if args.fno_capture is not None:
        try:
            until = datetime.combine(
                today, time.fromisoformat(args.fno_capture_until), INDIA_TIMEZONE
            )
            capture_fno(kite, today, args.fno_capture, log, deadline=until)
        except Exception:
            log("F&O capture failed:\n" + traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
