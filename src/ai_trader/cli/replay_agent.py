"""Replay past sessions with the autonomous agent (experiment, branch
experiment/autonomous-agent): the same prompt, tools, account and timeline as a live
day, on Kite minute bars, NSE daily data and NSE filings as of each simulated moment
(agent/replay.py explains what is and is not reproduced). Paper only.

  download  Kite minute bars of the scan list and the indices for a date range
            (Kite sign-in in the browser; about 5 minutes for 660 symbols)
  run       replay whole sessions, all days at once; real Claude calls, so each day
            can spend up to --budget dollars

  python -m ai_trader.cli.replay_agent download --start 2026-10-06 --end 2026-10-16
  python -m ai_trader.cli.replay_agent run --days 2026-10-14 2026-10-15 --budget 30

A run writes data/paper/autonomous/replay/<name>/: replay.log, summary.json and per
day paper.sqlite, turns.jsonl, transcripts/ and report.json.
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_trader.agent.replay import INDEX_FILES, download_minutes, replay_day
from ai_trader.ai.anthropic_client import CLAUDE_EFFORTS, AnthropicMessages
from ai_trader.ai.openai_client import PRICES
from ai_trader.broker.kite import KiteBrokerError
from ai_trader.broker.kite_auth import login_browser
from ai_trader.cli.run_paper import RunLog, universe_profiles, universe_symbols
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import (
    ConfigurationError,
    load_anthropic_settings,
    load_kite_login_settings,
)

ROOT = Path("data/paper/autonomous/replay")
UNIVERSE = Path("data/universe_mcap_20261012/universe.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--universe", type=Path, default=UNIVERSE)
    parser.add_argument("--minutes", type=Path, default=ROOT / "minutes")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--start", type=date.fromisoformat, required=True)
    fetch.add_argument("--end", type=date.fromisoformat, required=True)
    fetch.add_argument("--login-wait", type=int, default=10, help="minutes")
    run = commands.add_parser("run")
    run.add_argument("--days", type=date.fromisoformat, nargs="+", required=True)
    run.add_argument("--name", default=None, help="folder under the replay root")
    run.add_argument("--features", type=Path, default=Path("data/fno/features"))
    run.add_argument(
        "--news-db", type=Path, default=Path("data/news/nse_filings.sqlite")
    )
    run.add_argument("--model", default="claude-opus-5-5")
    run.add_argument("--effort", choices=CLAUDE_EFFORTS, default="medium")
    run.add_argument("--budget", type=float, default=30.0, help="USD per day")
    run.add_argument("--wallet", type=Decimal, default=Decimal(2_000_000))
    run.add_argument("--leverage", type=int, default=5)
    run.add_argument("--no-thinking", action="store_true")
    run.add_argument(
        "--until", type=time.fromisoformat, default=None, help="end early (HH:MM)"
    )
    args = parser.parse_args(argv)

    if args.command == "download":
        log = RunLog(args.minutes / "download.log")
        try:
            kite = login_browser(
                load_kite_login_settings(), timeout_seconds=60 * args.login_wait
            )
        except (ConfigurationError, KiteBrokerError, ValueError) as error:
            log(f"Kite login failed: {error}")
            return 2
        symbols = [*universe_symbols(args.universe), *INDEX_FILES.values()]
        log(f"downloading minute bars {args.start}..{args.end} for {len(symbols)}")
        failed = download_minutes(
            kite, symbols, args.start, args.end, args.minutes, log
        )
        manifest = {
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "symbols": len(symbols),
            "failed": failed,
            "downloaded_at": datetime.now(INDIA_TIMEZONE).isoformat(timespec="seconds"),
        }
        (args.minutes / "manifest.json").write_text(
            json.dumps(manifest, indent=1), "utf-8"
        )
        log(f"download done: {len(failed)} failed {sorted(failed)[:20]}")
        return 0

    if args.model not in PRICES:
        parser.error(f"--model must be one of {', '.join(PRICES)}")
    try:
        key = load_anthropic_settings().api_key.get_secret_value()
    except ConfigurationError as error:
        print(f"Claude is not configured ({error})")
        return 2
    name = args.name or datetime.now(INDIA_TIMEZONE).strftime("%Y%m%d_%H%M")
    root = ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    log = RunLog(root / "replay.log")
    scan = universe_symbols(args.universe)
    profiles = universe_profiles(args.universe)
    log(
        f"replay {name}: days {[d.isoformat() for d in args.days]}, {args.model} "
        f"({args.effort}), ${args.budget:g} a day, wallet Rs {args.wallet:,} up to "
        f"{args.leverage}x" + (f", until {args.until:%H:%M}" if args.until else "")
    )

    def one(day: date) -> dict[str, Any]:
        try:
            return replay_day(
                day,
                AnthropicMessages(key, timeout_seconds=300),
                root / day.isoformat(),
                minutes=args.minutes,
                features=args.features,
                scan=scan,
                profiles=profiles,
                news_db=args.news_db,
                model=args.model,
                effort=args.effort,
                budget=args.budget,
                wallet=args.wallet,
                leverage=args.leverage,
                log=log,
                thinking=not args.no_thinking,
                stop_at=args.until,
            )
        except Exception:
            log(f"replay {day} failed:\n{traceback.format_exc()}")
            return {"day": day.isoformat(), "error": traceback.format_exc()[-2000:]}

    with ThreadPoolExecutor(max_workers=len(args.days)) as pool:
        days = list(pool.map(one, args.days))
    done = [d for d in days if "error" not in d]
    summary = {
        "name": name,
        "days": days,
        "net_rs": round(sum(d["net_rs"] for d in done), 2),
        "ai_spend_usd": round(sum(d["ai_spend_usd"] for d in done), 2),
        "trades": sum(d["trades"] for d in done),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1), "utf-8")
    for d in done:
        log(
            f"{d['day']}: net Rs {d['net_rs']:,.0f} from {d['trades']} trades "
            f"({d['won']} won), charges Rs {d['charges_rs']:,.0f}, AI "
            f"${d['ai_spend_usd']:.2f}, NIFTY {d['nifty_change_pct']}%"
        )
    log(
        f"replay total: net Rs {summary['net_rs']:,.0f} over {len(done)} days, "
        f"{summary['trades']} trades, AI ${summary['ai_spend_usd']:.2f}"
    )
    return 0 if len(done) == len(days) else 1


if __name__ == "__main__":
    raise SystemExit(main())
