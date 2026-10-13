"""Live paper trading: scanner -> components -> AI BUY / NO_BUY -> paper execution.

Read-only Kite access; no order is ever placed. One run covers one session: it
signs in to Kite (browser), loads history (about 10 minutes for 500 stocks at the
Kite rate limit; a same-day restart reuses the cache, so start by 09:15), then
every 5 minutes from 09:40 to 14:55
scans the universe and sends every stock with P(+1% before the close) at or above
the threshold to the AI; BUY verdicts become paper buys with a +1% sell limit and a
-1% stop. Held positions are checked every minute and carried across sessions in
the paper store. The trade report is rewritten after every change.

Requires KITE_API_KEY / KITE_API_SECRET and OPENAI_API_KEY in .env, and
ANTHROPIC_API_KEY for Claude shadows. Each AI spends at most AI_DAILY_BUDGET_USD a
day. Weekends, NSE holidays, starts after the close and a second copy exit at once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import IO, Any

from ai_trader.ai.anthropic_client import CLAUDE_EFFORTS, AnthropicMessages
from ai_trader.ai.decision import DELIVERY_V4, OpenAIDecider, provider
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
) -> OpenAIDecider:
    """Each AI's spend is capped at AI_DAILY_BUDGET_USD a day in its own journal."""
    client: OpenAIResponses | AnthropicMessages
    if provider(model) == "anthropic":
        if anthropic is None:
            raise ValueError(f"{model} needs ANTHROPIC_API_KEY in .env")
        client = AnthropicMessages(
            anthropic.api_key.get_secret_value(),
            timeout_seconds=anthropic.timeout_seconds,
        )
    else:
        client = OpenAIResponses(
            openai.api_key.get_secret_value(), timeout_seconds=openai.timeout_seconds
        )
    return OpenAIDecider(
        client,
        model,
        reasoning_effort=effort,
        web_search=True,
        require_search=True,
        budget_left=lambda: (
            openai.daily_budget_usd
            - journal.spent_usd(datetime.now(INDIA_TIMEZONE).date())
        ),
        policy=DELIVERY_V4,
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
        "--scanner", type=Path, default=Path("data/models/scanner_up1_by_close_v1")
    )
    parser.add_argument("--store", type=Path, default=Path("data/paper/paper.sqlite"))
    parser.add_argument(
        "--journal", type=Path, default=Path("data/paper/decisions.sqlite")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("data/paper/report"))
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
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--capital", type=Decimal, default=Decimal(5_000_000))
    parser.add_argument("--per-trade", type=Decimal, default=Decimal(200_000))
    parser.add_argument("--max-positions", type=int, default=25)
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
    args = parser.parse_args(argv)

    log = RunLog(args.log_dir / f"{datetime.now(INDIA_TIMEZONE):%Y%m%d}.log")
    lock = single_instance(args.store.parent / "run.lock")
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
    anthropic = None
    if any(provider(model) == "anthropic" for model, _ in args.shadow):
        try:
            anthropic = load_anthropic_settings()
        except ConfigurationError as error:
            log(f"Claude is not configured ({error}).")
            return 2
    store = PaperStore(args.store)
    config = PaperConfig(
        capital=args.capital,
        per_trade=args.per_trade,
        max_positions=args.max_positions,
    )
    shadows = []
    try:
        broker = PaperBroker(store, config)
        for model, effort in dict.fromkeys(args.shadow):
            if (model, effort) == (openai.model, openai.reasoning_effort):
                continue
            folder = args.store.parent / "shadows" / f"{model}_{effort}"
            journal = DecisionJournal(folder / "decisions.sqlite")
            shadows.append(
                Book(
                    f"{provider(model)}:{model}:{effort}",
                    ai_decider(openai, model, effort, journal, anthropic),
                    journal,
                    PaperBroker(PaperStore(folder / "paper.sqlite"), config),
                    folder / "report",
                    f"[{model} {effort}] ",
                )
            )
    except ValueError as error:
        log(str(error))
        return 2
    log(
        f"scanner {bundle.version}; {len(symbols)} stocks; threshold "
        f"{args.threshold:.0%}; AI {openai.model} ({openai.reasoning_effort}); "
        f"{len(broker.open_positions())} positions held"
    )
    for shadow in shadows:
        log(
            f"shadow AI {shadow.name}: {len(shadow.broker.open_positions())} "
            f"positions held; report in {shadow.report_dir}"
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
    decider = ai_decider(openai, openai.model, openai.reasoning_effort, journal)
    clock = SystemClock()
    news = NewsStore(args.news_db)
    session = PaperSession(
        kite,
        clock,
        bundle,
        symbols,
        broker,
        decider,
        f"openai:{openai.model}:{openai.reasoning_effort}",
        journal,
        report_dir=args.report_dir,
        settings=SessionSettings(
            threshold=args.threshold,
            workers=args.workers,
            universe_target=args.target,
        ),
        news=news,
        nse=NseClient(timeout_seconds=10, attempts=2),
        cache_dir=args.store.parent / "cache",
        log=log,
        shadows=shadows,
        profiles=universe_profiles(args.universe),
        capture_dir=args.store.parent / "capture",
        stream=lambda instruments: kite.fork_history().create_ltp_stream(
            instruments, strict=False, queue_size=200_000
        ),
        label=f"[{openai.model} {openai.reasoning_effort}] " if shadows else "",
        basis=PriceBasis(args.nse_dir, NseClient(timeout_seconds=30, attempts=3), news),
    )
    try:
        session.prepare()
        session.exit_cycle()
        session.run(stop)
    except BaseException:
        log("run stopped:\n" + traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
