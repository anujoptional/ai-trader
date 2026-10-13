"""Shadow-mode decision loop: movement scanner -> order book, OI, filings -> verdicts.

Read-only and research-only: it never places orders, sizes positions or exits. At
each 5-minute decision time (09:40-13:30 IST) it fetches today's minute bars,
scores the universe with the frozen CatBoost movement model, takes the top
candidates, adds live depth (stock and near/next futures), daily and live OI and
as-of NSE filings, asks the decider for LONG / SHORT / NO_TRADE and journals it.

Inputs: a Kite login (browser) each run; optional OPENAI_API_KEY in .env (without
it the deterministic rule baseline decides); the scanner model file; optionally a
futures-OI directory from ``download_futures_oi``.
"""

from __future__ import annotations

import argparse
import json
import time as clock
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

from ai_trader.ai.cycle import (
    StockInputs,
    build_request,
    complete,
    daily_array,
    score_universe,
    session_arrays,
)
from ai_trader.ai.decision import RuleDecider
from ai_trader.ai.journal import DecisionJournal
from ai_trader.broker import CandleInterval, Instrument
from ai_trader.broker.kite import KiteBroker, KiteBrokerError
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import (
    ConfigurationError,
    load_kite_login_settings,
    load_openai_settings,
)
from ai_trader.news.features import event_context
from ai_trader.news.nse import NseClient, NseError
from ai_trader.news.store import NewsStore
from ai_trader.oi.features import daily_oi_features, live_oi_context, monthly_expiries
from ai_trader.orderbook.features import depth_features
from ai_trader.scanner.movement import DECISIONS, OPEN_MINUTE

HISTORY_DAYS = 45


def ist(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, time(hour, minute), INDIA_TIMEZONE)


def decision_times(day: date) -> list[datetime]:
    """Close times of the decision bars (bar d closes one minute after it opens)."""
    return [
        ist(day, 0, 0) + timedelta(minutes=OPEN_MINUTE + int(d) + 1) for d in DECISIONS
    ]


def load_history(
    broker: KiteBroker, symbol: str, today: date
) -> tuple[list, np.ndarray, list[date]]:
    instrument = Instrument("NSE", symbol)
    start = ist(today - timedelta(days=HISTORY_DAYS), 9, 0)
    end = ist(today - timedelta(days=1), 23, 59)
    minutes = broker.get_historical_candles(
        instrument, start, end, CandleInterval.ONE_MINUTE
    )
    daily = broker.get_historical_candles(
        instrument, ist(today - timedelta(days=90), 0, 0), end, CandleInterval.ONE_DAY
    )
    dates = sorted(c.timestamp.date() for c in daily)
    return complete(session_arrays(minutes)), daily_array(daily), dates


def today_bars(broker: KiteBroker, symbol: str, now: datetime) -> np.ndarray:
    candles = broker.get_historical_candles(
        Instrument("NSE", symbol),
        ist(now.date(), 9, 15),
        now,
        CandleInterval.ONE_MINUTE,
    )
    sessions = session_arrays(candles)
    return sessions.get(now.date(), np.full((345, 5), np.nan))


def read_oi(
    broker: KiteBroker,
    symbol: str,
    sessions: list[date],
    spot: dict[date, float],
    today: date,
) -> dict:
    """Recent continuous near-month futures bars, fetched now so they are current."""
    try:
        bars = broker.get_continuous_futures_daily(
            symbol, today - timedelta(days=40), today - timedelta(days=1)
        )
    except (KiteBrokerError, ValueError):
        return {}
    return daily_oi_features(bars, spot, monthly_expiries(sessions))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--news-db", type=Path, default=Path("data/news/nse_filings.sqlite")
    )
    parser.add_argument(
        "--journal", type=Path, default=Path("data/journal/decisions.sqlite")
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--once", action="store_true", help="one cycle at the latest grid time"
    )
    args = parser.parse_args(argv)

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(args.model))
    universe = json.loads(args.universe.read_text("utf-8"))
    symbols = sorted(
        e["symbol"] for e in universe["instruments"] if e["role"] == "equity_target"
    )
    try:
        broker = KiteBroker.authenticate(load_kite_login_settings())
    except (ConfigurationError, KiteBrokerError) as error:
        print(f"Kite login failed: {error}")
        return 2
    try:
        settings = load_openai_settings()
    except ConfigurationError:
        settings = None
    journal = DecisionJournal(args.journal)
    if settings:
        from ai_trader.ai.decision import OpenAIDecider
        from ai_trader.ai.openai_client import OpenAIResponses

        decider = OpenAIDecider(
            OpenAIResponses(
                settings.api_key.get_secret_value(),
                timeout_seconds=settings.timeout_seconds,
            ),
            settings.model,
            reasoning_effort=settings.reasoning_effort,
            web_search=settings.web_search,
            budget_left=lambda: (
                settings.daily_budget_usd
                - journal.spent_usd(datetime.now(INDIA_TIMEZONE).date())
            ),
        )
        decider_name = f"openai:{settings.model}"
    else:
        decider, decider_name = RuleDecider(), "rules-v1"
    print(f"decider: {decider_name}")

    today = datetime.now(INDIA_TIMEZONE).date()
    print(f"loading history for {len(symbols)} stocks (about 3 minutes)...")
    history = {}
    for symbol in symbols + ["NIFTY", "INDIAVIX"]:
        try:
            history[symbol] = load_history(broker, symbol, today)
        except (KiteBrokerError, ValueError) as error:
            print(f"history unavailable for {symbol}: {error}")
    sessions = (
        sorted(set(history["NIFTY"][2]) | {today}) if "NIFTY" in history else [today]
    )
    futures = {}
    for row in broker.nfo_futures():
        if row["name"] in history:
            futures.setdefault(row["name"], []).append(row)
    near_next = {
        name: [
            Instrument("NFO", r["tradingsymbol"])
            for r in sorted(rows, key=lambda r: r["expiry"])[:2]
        ]
        for name, rows in futures.items()
    }
    news = NewsStore(args.news_db)
    nse = NseClient()
    try:
        news.add_meetings(nse.board_meetings(today, today + timedelta(days=3)))
        news.add_actions(nse.corporate_actions(today, today + timedelta(days=1)))
    except NseError as error:
        print(f"NSE refresh failed: {error}")
    traded: set[str] = set()
    print("loading futures OI (about 90 seconds)...")
    oi_daily = {
        s: read_oi(
            broker,
            s,
            sessions,
            dict(zip(history[s][2], history[s][1][:, 3].tolist(), strict=False)),
            today,
        )
        for s in symbols
        if s in history
    }

    times = decision_times(today)
    now = datetime.now(INDIA_TIMEZONE)
    pending = (
        [t for t in times if t <= now][-1:]
        if args.once
        else [t for t in times if t > now]
    )
    for decision_at in pending:
        wait = (decision_at - datetime.now(INDIA_TIMEZONE)).total_seconds() + 5
        if wait > 0:
            clock.sleep(wait)
        minute = int(DECISIONS[times.index(decision_at)])
        try:
            news.add_announcements(nse.announcements(today, today))
        except NseError as error:
            print(f"NSE announcements failed: {error}")
        stocks, nifty, vix = {}, None, None
        for symbol, (past, daily, _dates) in history.items():
            try:
                bars = today_bars(broker, symbol, decision_at)
            except KiteBrokerError:
                continue
            if symbol == "NIFTY":
                nifty = (bars, float(daily[-1, 3]))
            elif symbol == "INDIAVIX":
                vix = bars
            else:
                stocks[symbol] = StockInputs(bars, past, daily)
        if nifty is None or vix is None:
            print(f"{decision_at:%H:%M} skipped: index data unavailable")
            continue
        ranked = score_universe(
            lambda x: model.predict_proba(x)[:, 1],
            stocks,
            nifty[0],
            nifty[1],
            vix,
            minute,
        )
        top = [r[0] for r in ranked if r[0] not in traded][: args.top_k]
        instruments = [Instrument("NSE", s) for s in top]
        instruments += [f for s in top for f in near_next.get(s, [])]
        snapshots, _missing = broker.get_depth(instruments)
        by_symbol = {s.instrument.trading_symbol: s for s in snapshots}
        depth = {s: by_symbol[s] for s in top if s in by_symbol}
        since = ist(today - timedelta(days=1), 15, 30)
        oi_context, news_context = {}, {}
        for symbol in top:
            daily_oi = oi_daily.get(symbol) or {}
            previous = [d for d in sorted(daily_oi) if d < today]
            # Only the previous session's bar counts; older bars would be stale.
            fresh = previous and len(sessions) > 1 and previous[-1] == sessions[-2]
            values = dict(daily_oi[previous[-1]]) if fresh else {}
            contracts = near_next.get(symbol, [])
            near = by_symbol.get(contracts[0].trading_symbol) if contracts else None
            rolled = values.get("calendar_days_to_expiry") == 0  # new contract today
            if near is not None and not rolled:
                book = depth_features(near)
                values |= live_oi_context(
                    near.open_interest,
                    None,
                    values.get("open_interest"),
                    book["day_change_pct"] / 100,
                )
            oi_context[symbol] = values
            news_context[symbol] = event_context(news, symbol, decision_at, since)
        request = build_request(
            decision_at,
            ranked,
            top_k=args.top_k,
            depth=depth,
            oi=oi_context,
            news=news_context,
            exclude=traded,
        )
        result = decider.decide(request)
        journal.record(f"{decision_at:%Y%m%dT%H%M}", decision_at, decider_name, result)
        for verdict in result.verdicts:
            if verdict.verdict != "NO_TRADE":
                traded.add(verdict.symbol)
            print(
                f"{decision_at:%H:%M} {verdict.symbol:12s} {verdict.verdict:8s} "
                f"{verdict.confidence:.2f} {verdict.reason}"
            )
        print(f"{decision_at:%H:%M} status={result.status} cost=${result.cost_usd:.4f}")
    journal.close()
    news.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
