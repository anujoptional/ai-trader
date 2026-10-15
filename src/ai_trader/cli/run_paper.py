"""Live paper trading: scanner -> components -> AI LONG / SHORT / NO_TRADE -> paper.

Read-only Kite access; no order is ever placed. One run covers one session: it
signs in to Kite (browser), loads history (about 10 minutes for 500 stocks at the
Kite rate limit; a same-day restart reuses the cache, so start by 09:15), then
every 5 minutes from 09:40 to 14:55 scans the universe and sends every stock whose
P(1% move either way before the close) is at or above the threshold (no limit on
how many) to the AIs, which call the direction to the close: LONG becomes a paper
buy, SHORT a paper short sale, each with a 1% target (--take-profit) and no stop;
anything still open is closed at 15:00. Positions are checked every minute and the
trade report is rewritten after every change. --prompt research gives the AIs only
each stock's symbol and company: they research it themselves (web search) and answer
one word, LONG, SHORT or NO_TRADE. --no-limits drops the capital, open-position and
one-per-stock limits: every pick is asked about again each scan, held or not.
--fno-capture DIR records, after the session, Kite minute bars with open interest of
every live future and the near options (fno/kite_capture.py): Kite keeps no intraday
history of expired contracts, so this is the only way to have it.

Requires KITE_API_KEY / KITE_API_SECRET and OPENAI_API_KEY in .env, and
ANTHROPIC_API_KEY for Claude. The main AI is OPENAI_MODEL / OPENAI_REASONING_EFFORT
from .env unless --main is given. Each AI spends at most AI_DAILY_BUDGET_USD a day
(--ai-budget overrides it; inf for no cap). Weekends, NSE holidays, starts after the
close and a second copy exit at once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import traceback
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import IO, Any

from ai_trader.ai.anthropic_client import CLAUDE_EFFORTS, AnthropicMessages
from ai_trader.ai.decision import (
    DIRECTION,
    RESEARCH,
    OpenAIDecider,
    Policy,
    provider,
)
from ai_trader.ai.journal import DecisionJournal
from ai_trader.ai.openai_client import PRICES, OpenAIResponses
from ai_trader.broker.kite import KiteBrokerError
from ai_trader.broker.kite_auth import login_browser
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import (
    REASONING_EFFORTS,
    AnthropicSettings,
    ConfigurationError,
    OpenAISettings,
    load_anthropic_settings,
    load_kite_login_settings,
    load_openai_settings,
)
from ai_trader.fno.kite_capture import capture as capture_fno
from ai_trader.news.nse import NseClient
from ai_trader.news.store import NewsStore
from ai_trader.paper.basis import PriceBasis
from ai_trader.paper.broker import PaperBroker, PaperConfig
from ai_trader.paper.session import Book, PaperSession, SessionSettings, SystemClock
from ai_trader.paper.store import PaperStore
from ai_trader.scanner.upside import ScannerBundle


def universe_symbols(path: Path) -> list[str]:
    """Stocks in rank order (market cap) when ranked, else alphabetical."""
    universe = json.loads(path.read_text("utf-8"))
    stocks = [e for e in universe["instruments"] if e["role"] == "equity_target"]
    return [
        e["symbol"]
        for e in sorted(stocks, key=lambda e: (e.get("rank", 0), e["symbol"]))
    ]


def universe_profiles(path: Path) -> dict[str, dict[str, str]]:
    universe = json.loads(path.read_text("utf-8"))
    return {
        entry["symbol"]: {
            "company": entry.get("company", ""),
            "industry": entry.get("industry", ""),
        }
        for entry in universe["instruments"]
        if entry["role"] == "equity_target"
    }


class RunLog:
    """Prints each line with the IST time and appends it to a log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")  # e.g. Rs symbols in AI reasons

    def __call__(self, message: str) -> None:
        line = f"{datetime.now(INDIA_TIMEZONE):%H:%M:%S} {message}"
        with self._lock:
            print(line, flush=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def single_instance(path: Path) -> IO[str] | None:
    """An exclusive lock for the whole run; the OS frees it when the process ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def closed_today(now: datetime, until: datetime, nse: Any) -> str | None:
    """Why there is no session to trade, or None; NSE's holiday list is best effort."""
    if now.weekday() >= 5:
        return f"{now:%A}, the market is closed"
    if now >= until:
        return f"it is past {until:%H:%M}, today's session is over"
    try:
        rows = nse.get_json("/api/holiday-master", {"type": "trading"}).get("CM")
    except Exception:
        return None
    for row in rows or []:
        try:
            day = datetime.strptime(str(row.get("tradingDate")), "%d-%b-%Y").date()
        except ValueError:
            continue
        if day == now.date():
            return f"NSE trading holiday ({row.get('description')})"
    return None


def shadow_spec(text: str) -> tuple[str, str]:
    model, _, effort = text.strip().partition(":")
    efforts = CLAUDE_EFFORTS if provider(model) == "anthropic" else REASONING_EFFORTS
    if model not in PRICES or effort not in efforts:
        raise argparse.ArgumentTypeError(
            f"expected MODEL:EFFORT with MODEL in {', '.join(PRICES)} and EFFORT in "
            + ", ".join(efforts)
        )
    return model, effort


def ai_decider(
    openai: OpenAISettings,
    model: str,
    effort: str,
    journal: DecisionJournal,
    anthropic: AnthropicSettings | None = None,
    *,
    policy: Policy = DIRECTION,
    timeout: float | None = None,
    budget: float | None = None,
) -> OpenAIDecider:
    """Each AI's spend is capped at AI_DAILY_BUDGET_USD (or ``budget``) a day in its
    own journal."""
    client: OpenAIResponses | AnthropicMessages
    if provider(model) == "anthropic":
        if anthropic is None:
            raise ValueError(f"{model} needs ANTHROPIC_API_KEY in .env")
        client = AnthropicMessages(
            anthropic.api_key.get_secret_value(),
            timeout_seconds=timeout or anthropic.timeout_seconds,
        )
    else:
        client = OpenAIResponses(
            openai.api_key.get_secret_value(),
            timeout_seconds=timeout or openai.timeout_seconds,
        )
    daily = openai.daily_budget_usd if budget is None else budget
    return OpenAIDecider(
        client,
        model,
        reasoning_effort=effort,
        web_search=True,
        require_search=True,
        budget_left=lambda: (
            daily - journal.spent_usd(datetime.now(INDIA_TIMEZONE).date())
        ),
        policy=policy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("data/universe_mcap_20261012/universe.json"),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="scan this many highest-ranked stocks that can be scored; lower-ranked "
        "stocks in the universe file replace any that cannot",
    )
    parser.add_argument(
        "--scanner", type=Path, default=Path("data/models/scanner_move10_by_close_v1")
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("data/paper"),
        help="run lock, history cache and raw input capture",
    )
    # The intraday LONG/SHORT design (from 2026-10-14) keeps its own portfolios and
    # journals; the delivery-era stores in data/paper are kept as they are.
    parser.add_argument(
        "--store", type=Path, default=Path("data/paper/intraday/paper.sqlite")
    )
    parser.add_argument(
        "--journal", type=Path, default=Path("data/paper/intraday/decisions.sqlite")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("data/paper/intraday/report")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("data/paper/logs"))
    parser.add_argument(
        "--news-db", type=Path, default=Path("data/news/nse_filings.sqlite")
    )
    parser.add_argument(
        "--nse-dir",
        type=Path,
        default=Path("data/nse"),
        help="NSE's daily bhavcopies and symbol changes, kept for the price basis",
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--capital", type=Decimal, default=Decimal(5_000_000))
    parser.add_argument("--per-trade", type=Decimal, default=Decimal(200_000))
    parser.add_argument("--max-positions", type=int, default=25)
    parser.add_argument(
        "--no-limits",
        action="store_true",
        help="no capital or open-position limit and no one-per-stock rule (a stock "
        "may be entered again while held); needs a new portfolio",
    )
    parser.add_argument("--until", default="15:31", help="IST time to stop (HH:MM)")
    parser.add_argument(
        "--login-wait",
        type=int,
        default=10,
        help="minutes to wait for the Kite sign-in in the browser (at most 480)",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--shadow",
        type=shadow_spec,
        action="append",
        default=[],
        metavar="MODEL:EFFORT",
        help="another AI asked the same questions, with its own paper portfolio "
        "under <store dir>/shadows/MODEL_EFFORT (repeatable)",
    )
    parser.add_argument(
        "--main",
        type=shadow_spec,
        default=None,
        metavar="MODEL:EFFORT",
        help="the main AI (default: OPENAI_MODEL / OPENAI_REASONING_EFFORT)",
    )
    parser.add_argument(
        "--prompt",
        choices=("direction", "research"),
        default="direction",
        help="direction: market facts + web search, LONG/SHORT/NO_TRADE with p_green "
        "and a reason; research: symbol and company only, the AI researches each "
        "stock itself and answers one word",
    )
    parser.add_argument(
        "--take-profit",
        type=Decimal,
        default=Decimal(1),
        help="target in percent of the entry price (else out at 15:00)",
    )
    parser.add_argument(
        "--ai-timeout",
        type=float,
        default=None,
        help="seconds to wait for one AI answer (default: from .env)",
    )
    parser.add_argument(
        "--ai-budget",
        type=float,
        default=None,
        help="USD each AI may spend a day, inf for no cap (default: "
        "AI_DAILY_BUDGET_USD)",
    )
    parser.add_argument(
        "--fno-capture",
        type=Path,
        default=None,
        help="after the session, record Kite minute bars with OI of F&O contracts "
        "here (fno/kite_capture.py)",
    )
    parser.add_argument(
        "--fno-capture-until",
        default="18:15",
        help="stop the F&O capture at this IST time (it resumes the next day)",
    )
    args = parser.parse_args(argv)
    policy = RESEARCH if args.prompt == "research" else DIRECTION

    log = RunLog(args.log_dir / f"{datetime.now(INDIA_TIMEZONE):%Y%m%d}.log")
    lock = single_instance(args.paper_dir / "run.lock")
    if lock is None:
        log("another paper run is already running; not starting a second one")
        return 3
    now = datetime.now(INDIA_TIMEZONE)
    stop = datetime.combine(now.date(), time.fromisoformat(args.until), INDIA_TIMEZONE)
    reason = closed_today(now, stop, NseClient(timeout_seconds=10, attempts=2))
    if reason:
        log(f"no paper run today: {reason}")
        return 0
    bundle = ScannerBundle.load(args.scanner)
    symbols = universe_symbols(args.universe)
    try:
        openai = load_openai_settings()
    except ConfigurationError as error:
        log(f"OpenAI is not configured ({error}); the AI layer is required.")
        return 2
    main_model, main_effort = args.main or (openai.model, openai.reasoning_effort)
    models = [main_model] + [model for model, _ in args.shadow]
    anthropic = None
    if any(provider(model) == "anthropic" for model in models):
        try:
            anthropic = load_anthropic_settings()
        except ConfigurationError as error:
            log(f"Claude is not configured ({error}).")
            return 2
    store = PaperStore(args.store)
    try:
        config = PaperConfig(
            capital=None if args.no_limits else args.capital,
            per_trade=args.per_trade,
            max_positions=None if args.no_limits else args.max_positions,
            target=args.take_profit / 100,
            one_per_stock=not args.no_limits,
        )
    except ValueError as error:
        log(str(error))
        return 2
    options = {"policy": policy, "timeout": args.ai_timeout, "budget": args.ai_budget}
    shadows = []
    try:
        broker = PaperBroker(store, config)
        for model, effort in dict.fromkeys(args.shadow):
            if (model, effort) == (main_model, main_effort):
                continue
            folder = args.store.parent / "shadows" / f"{model}_{effort}"
            journal = DecisionJournal(folder / "decisions.sqlite")
            shadows.append(
                Book(
                    f"{provider(model)}:{model}:{effort}",
                    ai_decider(openai, model, effort, journal, anthropic, **options),
                    journal,
                    PaperBroker(PaperStore(folder / "paper.sqlite"), config),
                    folder / "report",
                    f"[{model} {effort}] ",
                )
            )
    except ValueError as error:
        log(str(error))
        return 2
    budget = openai.daily_budget_usd if args.ai_budget is None else args.ai_budget
    log(
        f"scanner {bundle.version}; {len(symbols)} stocks; threshold "
        f"{args.threshold:.0%}; AI {main_model} ({main_effort}), "
        f"prompt {policy.version}; target {args.take_profit}% else out at 15:00; "
        f"{len(broker.open_positions())} positions open; portfolio {args.store}"
    )
    log(
        "limits: "
        + (
            "none (capital, open positions, per stock)"
            if args.no_limits
            else f"capital Rs {args.capital:,}, at most {args.max_positions} open, "
            "one position per stock a session"
        )
        + "; AI spend cap "
        + ("none" if math.isinf(budget) else f"${budget:g} a day per AI")
    )
    for shadow in shadows:
        log(
            f"shadow AI {shadow.name}: {len(shadow.broker.open_positions())} "
            f"positions open; report in {shadow.report_dir}"
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
    journal = DecisionJournal(args.journal)
    decider = ai_decider(openai, main_model, main_effort, journal, anthropic, **options)
    clock = SystemClock()
    news = NewsStore(args.news_db)
    session = PaperSession(
        kite,
        clock,
        bundle,
        symbols,
        broker,
        decider,
        f"{provider(main_model)}:{main_model}:{main_effort}",
        journal,
        report_dir=args.report_dir,
        settings=SessionSettings(
            threshold=args.threshold,
            workers=args.workers,
            universe_target=args.target,
            ai_input="identity" if policy is RESEARCH else "facts",
        ),
        news=news,
        nse=NseClient(timeout_seconds=10, attempts=2),
        cache_dir=args.paper_dir / "cache",
        log=log,
        shadows=shadows,
        profiles=universe_profiles(args.universe),
        capture_dir=args.paper_dir / "capture",
        stream=lambda instruments: kite.fork_history().create_ltp_stream(
            instruments, strict=False, queue_size=200_000
        ),
        label=f"[{main_model} {main_effort}] " if shadows else "",
        basis=PriceBasis(args.nse_dir, NseClient(timeout_seconds=30, attempts=3), news),
    )
    try:
        session.prepare()
        session.exit_cycle()
        session.run(stop)
    except BaseException:
        log("run stopped:\n" + traceback.format_exc())
        raise
    if args.fno_capture is not None:
        try:
            until = datetime.combine(
                now.date(), time.fromisoformat(args.fno_capture_until), INDIA_TIMEZONE
            )
            capture_fno(kite, now.date(), args.fno_capture, log, deadline=until)
        except Exception:
            log("F&O capture failed:\n" + traceback.format_exc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
