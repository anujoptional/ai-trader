"""End-to-end research replay: CatBoost movement scanner -> components -> verdicts.

Research only, on the NSE-basis corrected Kite dataset (F&O universe). Quality
flags from the basis build (coarse vendor rounding, imprecise fit, structural break)
and illiquid sessions from the independent audit are excluded.

1. Scanner: CatBoost P(two-sided 1% move within 60 minutes) at every 5-minute
   decision close 09:40-13:30; train 2020-2023, early stop 2024, test 2025 onward.
2. Components per candidate, strictly as-of the decision: scanner features, daily
   futures OI features from the previous session, NSE filings context.
   Historical order books do not exist, so the book block is empty in replay.
3. Deciders on the top-5 candidates at 8 decision times a day: the rule baseline,
   a CatBoost direction policy (thresholds chosen on 2024 only), naive baselines,
   and the OpenAI decider on a cost-capped post-cutoff sample when a key exists.
4. Trades: entry at the next minute's open, exit at the 14:59 close, one trade per
   stock per day, Zerodha intraday charges at Rs 2 lakh on both sides.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time as clock
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

import numpy as np

from ai_trader.ai.decision import (
    INSTRUCTIONS,
    SCHEMA,
    VERDICTS,
    DecisionRequest,
    RuleDecider,
    request_json,
)
from ai_trader.ai.packet import candidate, scanner_block
from ai_trader.broker import FuturesDay
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.costs import ZERODHA_INTRADAY_EQUITY
from ai_trader.news.features import event_context
from ai_trader.news.store import NewsStore
from ai_trader.oi.features import daily_oi_features, monthly_expiries
from scripts.research_fno_scanner import (
    DECISIONS,
    EPOCH,
    FEATURES,
    OPEN_MINUTE,
    QUICK,
    SEED,
    assemble,
    epoch_day,
    top_k,
)
from scripts.train_tp_only import file_hash, write_json

NOTIONAL_INR = 200_000
COST = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(Decimal(NOTIONAL_INR)))
REPLAY_SLOTS = (1, 7, 13, 19, 25, 31, 37, 43)
TOP_K = 5
DIRECTION_POOL = 10
MAX_TRADES_PER_DAY = 10
# After gpt-6-luna's (2026-05-18) and gpt-6.1-sol's (2026-04-30) knowledge cutoffs.
OPENAI_FROM = "2026-06-01"
OI_FEATURES = (
    "oi_change_1d",
    "oi_change_5d",
    "fut_return_1d",
    "basis_pct",
    "volume_to_oi",
    "calendar_days_to_expiry",
    "buildup_score",
)
BUILDUP_SCORE = {
    "long_buildup": 1.0,
    "short_covering": 0.5,
    "neutral": 0.0,
    "long_unwinding": -0.5,
    "short_buildup": -1.0,
}
PERIODS = (
    ("validation_2024", "2024-01-01", "2024-12-31"),
    ("test_2025H1", "2025-01-01", "2025-06-30"),
    ("test_2025H2", "2025-07-01", "2025-12-31"),
    ("test_2026", "2026-01-01", "2026-12-31"),
    ("test_all", "2025-01-01", "2026-12-31"),
)


def as_date(epoch: int) -> date:
    return date.fromordinal(int(epoch) + EPOCH)


def decision_time(epoch: int, slot: int) -> datetime:
    minute = OPEN_MINUTE + int(DECISIONS[slot]) + 1
    return datetime.combine(
        as_date(epoch), time(minute // 60, minute % 60), INDIA_TIMEZONE
    )


def apply_exclusions(data: dict, candles: Path, audit: Path | None) -> dict:
    """Mark flagged stock-days unavailable; returns counts by reason."""
    column = {symbol: i for i, symbol in enumerate(data["stocks"])}
    row = {int(day): i for i, day in enumerate(data["days"])}
    counts: dict[str, int] = {}

    def drop(symbol: str, day_text: str, reason: str) -> None:
        r, c = row.get(epoch_day(day_text)), column.get(symbol)
        if r is not None and c is not None and data["available"][r, :, c].any():
            data["available"][r, :, c] = False
            counts[reason] = counts.get(reason, 0) + 1

    flags = json.loads((candles / "exclusions.json").read_text("utf-8"))
    for symbol, reasons in flags.items():
        for reason in ("coarse_vendor_rounding", "imprecise_fit", "structural_break"):
            for day_text in reasons.get(reason, []):
                drop(symbol, day_text, reason)
    if audit is not None:
        audited = json.loads(audit.read_text("utf-8"))["instrument_days"]
        for symbol, by_day in audited.items():
            for day_text, reasons in by_day.items():
                if "illiquid_session" in reasons:
                    drop(symbol, day_text, "illiquid_session")
    return counts


def movement_target(data: dict) -> np.ndarray:
    return (data["big"] <= QUICK) | (data["big_down"] <= QUICK)


def period_mask(days: np.ndarray, first: str, last: str) -> np.ndarray:
    return (days >= epoch_day(first)) & (days <= epoch_day(last))


def fit_movement(data: dict, target: np.ndarray, threads: int):
    from catboost import CatBoostClassifier

    flat = data["features"].reshape(-1, len(FEATURES))
    labels = target.reshape(-1)
    day = np.broadcast_to(data["days"][:, None, None], target.shape).reshape(-1)
    available = data["available"].reshape(-1)
    rng = np.random.default_rng(SEED)

    def sample(first: str, last: str, size: int) -> np.ndarray:
        rows = np.flatnonzero(available & period_mask(day, first, last))
        return np.sort(rng.choice(rows, size=min(size, len(rows)), replace=False))

    fit_rows = sample("2020-01-01", "2023-12-31", 3_000_000)
    stop_rows = sample("2024-01-01", "2024-12-31", 600_000)
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.08,
        iterations=1500,
        l2_leaf_reg=10,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=80,
        use_best_model=True,
        random_seed=SEED,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    started = clock.monotonic()
    model.fit(
        flat[fit_rows], labels[fit_rows], eval_set=(flat[stop_rows], labels[stop_rows])
    )
    score = np.full(available.shape, np.nan, np.float32)
    rows = np.flatnonzero(available)
    for start in range(0, len(rows), 2_000_000):
        chunk = rows[start : start + 2_000_000]
        score[chunk] = model.predict_proba(flat[chunk])[:, 1]
    info = {
        "target": "|move| >= 1% (first touch of +1% or -1% from the decision close) "
        "within 60 minutes",
        "trees": int(model.tree_count_),
        "fit_rows": len(fit_rows),
        "stop_rows": len(stop_rows),
        "fit_seconds": round(clock.monotonic() - started, 1),
        "importance": dict(
            zip(FEATURES, model.get_feature_importance().round(3).tolist(), strict=True)
        ),
    }
    return score.reshape(target.shape), model, info


def scanner_metrics(score: np.ndarray, target: np.ndarray, data: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    available = data["available"]
    days = np.broadcast_to(data["days"][:, None, None], available.shape)
    rng = np.random.default_rng(SEED)
    result = {}
    for name, first, last in PERIODS:
        inside = available & period_mask(days, first, last)
        if not inside.any():
            continue
        rows = np.flatnonzero(inside.reshape(-1))
        rows = rng.choice(rows, size=min(len(rows), 2_000_000), replace=False)
        flat_score, flat_target = score.reshape(-1), target.reshape(-1)
        base = float(target[inside].mean())
        entry = {
            "rows": int(inside.sum()),
            "base_rate_percent": 100 * base,
            "auc": float(roc_auc_score(flat_target[rows], flat_score[rows])),
        }
        for k in (5, 10):
            chosen = top_k(score, inside, k)
            hit = float(target[chosen].mean())
            entry[f"top{k}"] = {
                "precision_percent": 100 * hit,
                "lift": hit / base,
                "recall_percent": 100
                * float(target[chosen].sum() / target[inside].sum()),
            }
        result[name] = entry
    return result


def read_spot(candles: Path, symbol: str) -> dict[date, float]:
    with (candles / "daily" / f"NSE_{symbol}.csv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        return {date.fromisoformat(row[0][:10]): float(row[4]) for row in reader}


def load_oi(directory: Path, candles: Path, data: dict) -> tuple[np.ndarray, dict]:
    """OI features per (day, stock) as of the PREVIOUS session's close."""
    days = [as_date(d) for d in data["days"]]
    expiries = monthly_expiries(days)
    result = np.full(
        (len(days), len(data["stocks"]), len(OI_FEATURES)), np.nan, np.float32
    )
    raw: dict[tuple[str, date], dict] = {}
    covered = 0
    for column, symbol in enumerate(data["stocks"]):
        path = directory / f"{symbol}.csv"
        if not path.exists():
            continue
        covered += 1
        with path.open(encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader)
            bars = [
                FuturesDay(
                    date.fromisoformat(r[0]),
                    *(Decimal(x) for x in r[1:5]),
                    int(r[5]),
                    int(r[6]),
                )
                for r in reader
            ]
        features = daily_oi_features(bars, read_spot(candles, symbol), expiries)
        ordered = sorted(features)
        if not ordered:
            continue
        positions = np.searchsorted(
            np.asarray(ordered, dtype="datetime64[D]"),
            np.asarray(days, dtype="datetime64[D]"),
        )
        for row, day in enumerate(days):
            position = int(positions[row])
            if position == 0:
                continue
            previous = ordered[position - 1]
            if row > 0 and previous != days[row - 1]:
                continue  # stale: the previous session has no futures bar
            values = dict(features[previous])
            basis = values.get("basis_pct")
            if basis is not None and abs(basis) > 5:
                values["basis_pct"] = None  # price bases differ (pre-split history)
            values["buildup_score"] = BUILDUP_SCORE.get(values.get("buildup_1d"))
            raw[(symbol, day)] = values
            result[row, column] = [
                np.nan if values.get(name) is None else float(values[name])
                for name in OI_FEATURES
            ]
    return result, {"symbols_with_oi": covered, "raw": raw, "expiries": len(expiries)}


def candidates_mask(score: np.ndarray, available: np.ndarray, k: int) -> np.ndarray:
    return top_k(score, available, k)


def fit_direction(data: dict, score: np.ndarray, oi: np.ndarray | None, threads: int):
    """CatBoost P(ret_entry > 0) on movement candidates; thresholds chosen on 2024."""
    from catboost import CatBoostClassifier

    pool = candidates_mask(score, data["available"], DIRECTION_POOL)
    day_idx, slot_idx, stock_idx = np.nonzero(pool)
    x = data["features"][day_idx, slot_idx, stock_idx]
    names = list(FEATURES)
    if oi is not None:
        x = np.concatenate([x, oi[day_idx, stock_idx]], axis=1)
        names += list(OI_FEATURES)
    x = np.concatenate([x, score[day_idx, slot_idx, stock_idx][:, None]], axis=1)
    names.append("movement_probability")
    ret = data["ret_entry"][day_idx, slot_idx, stock_idx]
    days = data["days"][day_idx]
    train = period_mask(days, "2020-01-01", "2023-12-31")
    stop = period_mask(days, "2024-01-01", "2024-12-31")
    model = CatBoostClassifier(
        depth=5,
        learning_rate=0.05,
        iterations=1500,
        l2_leaf_reg=20,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        random_seed=SEED,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x[train], ret[train] > 0, eval_set=(x[stop], ret[stop] > 0))
    p_up = np.full(score.shape, np.nan, np.float32)
    p_up[day_idx, slot_idx, stock_idx] = model.predict_proba(x)[:, 1]
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for name, first, last in PERIODS:
        inside = period_mask(days, first, last)
        if inside.sum() > 100:
            aucs[name] = float(
                roc_auc_score(
                    ret[inside] > 0, p_up[day_idx, slot_idx, stock_idx][inside]
                )
            )
    return (
        p_up,
        model,
        {
            "rows": int(len(ret)),
            "train_rows": int(train.sum()),
            "stop_rows": int(stop.sum()),
            "trees": int(model.tree_count_),
            "auc": aucs,
            "importance": dict(
                zip(
                    names, model.get_feature_importance().round(3).tolist(), strict=True
                )
            ),
        },
    )


def simulate(trades: list[dict]) -> dict:
    """Per-trade and per-day statistics of a list of executed trades."""
    if not trades:
        return {"trades": 0}
    net = np.array([t["net"] for t in trades])
    by_day: dict[int, float] = {}
    count: dict[int, int] = {}
    for t in trades:
        by_day[t["day"]] = by_day.get(t["day"], 0.0) + t["net"] * NOTIONAL_INR
        count[t["day"]] = count.get(t["day"], 0) + 1
    daily = np.array([by_day[d] for d in sorted(by_day)])
    counts = np.array([count[d] for d in sorted(by_day)])
    equity = np.cumsum(daily)
    drawdown = (
        float(np.max(np.maximum.accumulate(equity) - equity)) if len(equity) else 0.0
    )
    # Trades on one day share the market, so resample whole days.
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(daily), size=(2000, len(daily)))
    means = daily[picks].sum(axis=1) / counts[picks].sum(axis=1) / NOTIONAL_INR * 1e4
    longs = [t for t in trades if t["side"] == "LONG"]
    return {
        "trades": len(trades),
        "long_trades": len(longs),
        "trading_days": len(by_day),
        "hit_rate_percent": 100 * float((net > 0).mean()),
        "mean_net_bps": 1e4 * float(net.mean()),
        "mean_net_bps_ci95_day_bootstrap": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "share_of_bootstrap_means_above_zero": float((means > 0).mean()),
        "median_net_bps": 1e4 * float(np.median(net)),
        "total_net_inr": float(net.sum() * NOTIONAL_INR),
        "mean_daily_inr": float(daily.mean()),
        "daily_sharpe_annualized": float(daily.mean() / daily.std() * np.sqrt(250))
        if daily.std() > 0
        else None,
        "max_drawdown_inr": drawdown,
    }


def execute(verdicts: list[tuple[int, int, int, str]], data: dict) -> list[dict]:
    """First verdict per stock-day, at most MAX_TRADES_PER_DAY, in time order."""
    trades, seen, per_day = [], set(), {}
    for day, slot, stock, side in sorted(verdicts):
        if side == "NO_TRADE" or (day, stock) in seen:
            continue
        if per_day.get(day, 0) >= MAX_TRADES_PER_DAY:
            continue
        seen.add((day, stock))
        per_day[day] = per_day.get(day, 0) + 1
        gross = float(data["ret_entry"][day, slot, stock])
        trades.append(
            {
                "day": int(data["days"][day]),
                "slot": slot,
                "stock": data["stocks"][stock],
                "side": side,
                "gross": gross if side == "LONG" else -gross,
                "net": (gross if side == "LONG" else -gross) - COST,
            }
        )
    return trades


def by_period(trades: list[dict]) -> dict:
    out = {}
    for name, first, last in PERIODS:
        lo, hi = epoch_day(first), epoch_day(last)
        out[name] = simulate([t for t in trades if lo <= t["day"] <= hi])
    return out


def token_stats(requests: list[str]) -> dict | None:
    """Exact o200k_base token counts of the real per-cycle model input."""
    try:
        import tiktoken
    except ImportError:
        return None
    encoding = tiktoken.get_encoding("o200k_base")
    fixed = len(encoding.encode(INSTRUCTIONS)) + len(
        encoding.encode(json.dumps(SCHEMA, separators=(",", ":")))
    )
    counts = np.asarray([len(encoding.encode(r)) for r in requests])
    return {
        "encoding": "o200k_base (GPT-6 tokenizer unpublished; treat as estimate)",
        "cycles": len(counts),
        "instructions_and_schema_tokens": fixed,
        "request_tokens_mean": float(counts.mean()) if len(counts) else None,
        "request_tokens_p90": float(np.percentile(counts, 90)) if len(counts) else None,
        "request_tokens_max": int(counts.max()) if len(counts) else None,
    }


def evaluate_openai(
    args: argparse.Namespace,
    model: str,
    effort: str | None,
    cycles: list[tuple[DecisionRequest, list[tuple[int, int, int]]]],
    verdicts: dict[str, list[tuple[int, int, int, str]]],
    data: dict,
    output: Path,
) -> dict:
    """OpenAI decider on evenly spaced post-cutoff cycles, cost-capped, journaled."""
    from ai_trader.ai.decision import OpenAIDecider
    from ai_trader.ai.journal import DecisionJournal
    from ai_trader.ai.openai_client import OpenAIResponses
    from ai_trader.config import ConfigurationError, load_openai_settings

    try:
        settings = load_openai_settings()
    except ConfigurationError as error:
        return {"status": f"skipped: {error}"}
    effort = effort or settings.reasoning_effort
    chosen = [
        cycles[int(i)]
        for i in np.unique(np.linspace(0, len(cycles) - 1, args.openai_cycles).round())
    ]
    # Journaled answers are reused by request hash, which does not include the effort.
    journal = DecisionJournal(output / f"openai_journal_{model}_{effort}.sqlite")
    spent = [0.0]
    client = OpenAIResponses(
        settings.api_key.get_secret_value(),
        timeout_seconds=settings.timeout_seconds,
        attempts=6,
    )
    decider = OpenAIDecider(
        client,
        model,
        reasoning_effort=effort,
        web_search=False,  # never in replay: search would leak the future
        budget_left=lambda: args.openai_budget_usd - spent[0],
    )
    ai_verdicts, keys, statuses, usage = [], set(), {}, []
    lock = threading.Lock()

    def decide(request: DecisionRequest):
        result = decider.decide(request)
        with lock:
            spent[0] += result.cost_usd
        return result

    results = [
        journal.completed(f"replay-{number:05d}", request_json(request))
        for number, (request, _) in enumerate(chosen)
    ]
    missing = [i for i, result in enumerate(results) if result is None]
    with ThreadPoolExecutor(max_workers=args.openai_workers) as pool:
        fresh = list(pool.map(decide, [chosen[i][0] for i in missing]))
    for i, result in zip(missing, fresh, strict=True):
        results[i] = result
    latencies = []
    for number, ((request, positions), result) in enumerate(
        zip(chosen, results, strict=True)
    ):
        if number in set(missing):
            journal.record(f"replay-{number:05d}", request.as_of, "openai", result)
        statuses[result.status[:40]] = statuses.get(result.status[:40], 0) + 1
        if result.usage:
            usage.append(result.usage)
            latencies.append(result.latency_seconds)
        for position, verdict in zip(positions, result.verdicts, strict=True):
            ai_verdicts.append((*position, verdict.verdict))
            keys.add(position)
    journal.close()
    directional = [
        (side, float(data["ret_entry"][day, slot, stock]))
        for day, slot, stock, side in ai_verdicts
        if side != "NO_TRADE"
    ]
    same = {name: [v for v in rows if v[:3] in keys] for name, rows in verdicts.items()}
    return {
        "status": "completed",
        "model": model,
        "reasoning_effort": effort,
        "cycles": len(chosen),
        "from": OPENAI_FROM,
        "statuses": statuses,
        "reused_cycles": len(chosen) - len(missing),
        "cost_usd_this_run": spent[0],
        "cost_usd": float(sum(r.cost_usd for r in results)),
        "latency_seconds": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "p90": float(np.percentile(latencies, 90)) if latencies else None,
        },
        "directional_verdicts": len(directional),
        "directional_hit_percent": 100
        * float(np.mean([(r > 0) == (s == "LONG") for s, r in directional]))
        if directional
        else None,
        "tokens": {
            "input": int(sum(u.input_tokens for u in usage)),
            "cached": int(sum(u.cached_tokens for u in usage)),
            "output": int(sum(u.output_tokens for u in usage)),
            "reasoning": int(sum(u.reasoning_tokens for u in usage)),
        },
        "verdict_counts": {
            side: sum(1 for v in ai_verdicts if v[3] == side) for side in VERDICTS
        },
        "openai": simulate(execute(ai_verdicts, data)),
        "openai_long_only": simulate(execute(long_only(ai_verdicts), data)),
        "other_policies_same_cycles": {
            name: simulate(execute(rows, data)) for name, rows in same.items()
        },
        "other_policies_same_cycles_long_only": {
            name: simulate(execute(long_only(rows), data))
            for name, rows in same.items()
        },
    }


def long_only(rows: list[tuple[int, int, int, str]]) -> list[tuple[int, int, int, str]]:
    """The same verdicts with SHORT replaced by NO_TRADE (delivery cannot short)."""
    return [(*row[:3], row[3] if row[3] == "LONG" else "NO_TRADE") for row in rows]


def choose_thresholds(
    p_up: np.ndarray, candidates: list[tuple[int, int, int]], data: dict
) -> dict:
    """Grid-search LONG/SHORT probability thresholds on 2024 validation only."""
    best = {"long": 1.1, "short": -0.1, "mean_net_bps": -1e9}
    validation = [
        c
        for c in candidates
        if period_mask(data["days"][c[0]], "2024-01-01", "2024-12-31")
    ]
    for long_t in (0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 1.1):
        for short_t in (-0.1, 0.35, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48):
            verdicts = []
            for day, slot, stock in validation:
                p = float(p_up[day, slot, stock])
                side = (
                    "LONG" if p >= long_t else "SHORT" if p <= short_t else "NO_TRADE"
                )
                verdicts.append((day, slot, stock, side))
            stats = simulate(execute(verdicts, data))
            if stats["trades"] >= 200 and stats["mean_net_bps"] > best["mean_net_bps"]:
                best = {"long": long_t, "short": short_t, **stats}
    return best


def run(args: argparse.Namespace) -> dict:
    output: Path = args.output
    if output.exists() and not args.resume:
        raise ValueError("Output exists; keep earlier studies (or --resume)")
    output.mkdir(parents=True, exist_ok=True)
    started = clock.monotonic()
    data = assemble(args.candles, args.workers)
    excluded = apply_exclusions(data, args.candles, args.audit)
    target = movement_target(data)
    score, model, movement_info = fit_movement(data, target, args.workers)
    model.save_model(str(output / "scanner_movement_v1.cbm"))
    scanner = scanner_metrics(score, target, data)
    oi, oi_info = (None, {"symbols_with_oi": 0, "raw": {}})
    if args.oi and args.oi.exists():
        oi, oi_info = load_oi(args.oi, args.candles, data)
    p_up, direction_model, direction_info = fit_direction(data, score, oi, args.workers)
    direction_model.save_model(str(output / "direction_v1.cbm"))
    news = NewsStore(args.news) if args.news and args.news.exists() else None
    rule = RuleDecider()
    available = data["available"]
    replay_days = [
        i for i, d in enumerate(data["days"]) if epoch_day("2024-01-01") <= d
    ]
    candidates: list[tuple[int, int, int]] = []
    verdicts: dict[str, list[tuple[int, int, int, str]]] = {
        "rule_oi_book": [],
        "always_long": [],
        "momentum": [],
        "contrarian": [],
        "random": [],
    }
    rng = np.random.default_rng(SEED)
    column = {name: i for i, name in enumerate(FEATURES)}
    samples = []
    news_material: dict[tuple[int, int, int], int] = {}
    momentum_side: dict[tuple[int, int, int], str] = {}
    test_requests: list[str] = []
    late_cycles: list[tuple[DecisionRequest, list[tuple[int, int, int]]]] = []
    for row in replay_days:
        epoch = int(data["days"][row])
        since = decision_time(int(data["days"][row - 1]), 0).replace(hour=15, minute=30)
        for slot in REPLAY_SLOTS:
            ranked = np.where(available[row, slot], score[row, slot], -np.inf)
            order = [
                int(s) for s in np.argsort(-ranked)[:TOP_K] if np.isfinite(ranked[s])
            ]
            if not order:
                continue
            as_of = decision_time(epoch, slot)
            packets = []
            for rank, stock in enumerate(order, start=1):
                symbol = data["stocks"][stock]
                values = dict(
                    zip(
                        FEATURES,
                        data["features"][row, slot, stock].tolist(),
                        strict=True,
                    )
                )
                block = scanner_block(float(score[row, slot, stock]), rank, values)
                oi_values = oi_info["raw"].get((symbol, as_date(epoch)))
                news_context = (
                    event_context(news, symbol, as_of, since)
                    if news and epoch >= epoch_day("2024-12-08")
                    else None
                )
                packets.append(
                    candidate(symbol, block, oi=oi_values, news=news_context)
                )
                candidates.append((row, slot, stock))
                if news_context is not None:
                    news_material[(row, slot, stock)] = news_context[
                        "material_since_prev_close"
                    ]
                f = data["features"][row, slot, stock]
                trend = f[column["day_ret"]] > 0 and f[column["vwap_dev"]] > 0
                down = f[column["day_ret"]] < 0 and f[column["vwap_dev"]] < 0
                momentum_side[(row, slot, stock)] = (
                    "LONG" if trend else "SHORT" if down else "NO_TRADE"
                )
                verdicts["always_long"].append((row, slot, stock, "LONG"))
                verdicts["momentum"].append(
                    (
                        row,
                        slot,
                        stock,
                        "LONG" if trend else "SHORT" if down else "NO_TRADE",
                    )
                )
                verdicts["contrarian"].append(
                    (
                        row,
                        slot,
                        stock,
                        "SHORT" if trend else "LONG" if down else "NO_TRADE",
                    )
                )
                verdicts["random"].append(
                    (row, slot, stock, ("LONG", "SHORT")[int(rng.integers(2))])
                )
            request = DecisionRequest(
                as_of,
                {
                    "nifty_day_pct": round(
                        100
                        * float(
                            data["features"][row, slot, order[0], column["nifty_day"]]
                        ),
                        3,
                    ),
                    "nifty_ret15_pct": round(
                        100
                        * float(
                            data["features"][row, slot, order[0], column["nifty_ret15"]]
                        ),
                        3,
                    ),
                    "india_vix": round(
                        float(data["features"][row, slot, order[0], column["vix"]]), 2
                    ),
                },
                tuple(packets),
            )
            result = rule.decide(request)
            for stock, verdict in zip(order, result.verdicts, strict=True):
                verdicts["rule_oi_book"].append((row, slot, stock, verdict.verdict))
            if len(samples) < 2 and news and epoch >= epoch_day("2025-06-02"):
                samples.append(json.loads(result.request_json))
            if epoch >= epoch_day("2025-01-01"):
                test_requests.append(result.request_json)
            if epoch >= epoch_day(OPENAI_FROM):
                late_cycles.append((request, [(row, slot, s) for s in order]))
    for name, wanted in (
        ("momentum_with_news", True),
        ("momentum_without_news", False),
    ):
        verdicts[name] = [
            (*key, momentum_side[key] if (count > 0) == wanted else "NO_TRADE")
            for key, count in news_material.items()
        ]
    thresholds = choose_thresholds(p_up, candidates, data)
    verdicts["direction_model"] = [
        (
            day,
            slot,
            stock,
            "LONG"
            if p_up[day, slot, stock] >= thresholds["long"]
            else "SHORT"
            if p_up[day, slot, stock] <= thresholds["short"]
            else "NO_TRADE",
        )
        for day, slot, stock in candidates
    ]
    verdicts["direction_model_long_only"] = [
        (d, s, k, v if v == "LONG" else "NO_TRADE")
        for d, s, k, v in verdicts["direction_model"]
    ]
    policies = {name: by_period(execute(v, data)) for name, v in verdicts.items()}
    openai_result = None
    if args.openai_cycles:
        openai_result = {}
        for item in (args.openai_models or "gpt-6-luna").split(","):
            if not item:
                continue
            model, _, effort = item.partition(":")
            result = evaluate_openai(
                args,
                model,
                effort or args.openai_effort,
                late_cycles,
                verdicts,
                data,
                output,
            )
            openai_result[f"{model}:{result.get('reasoning_effort', 'skipped')}"] = (
                result
            )
    if news:
        news.close()
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "research_only_not_promoted",
        "dataset": str(args.candles),
        "dataset_sha256": file_hash(args.candles / "dataset.json"),
        "excluded_stock_days": excluded,
        "universe": {"stocks": len(data["stocks"]), "days": len(data["days"])},
        "cost_round_trip_fraction": COST,
        "notional_inr": NOTIONAL_INR,
        "replay": {
            "decision_times": [
                decision_time(0, s).strftime("%H:%M") for s in REPLAY_SLOTS
            ],
            "top_k": TOP_K,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "entry": "open of the minute after the decision close",
            "exit": "14:59 close",
        },
        "scanner": {"model": movement_info, "metrics": scanner},
        "open_interest": {k: v for k, v in oi_info.items() if k != "raw"},
        "direction_model": {**direction_info, "thresholds_from_2024": thresholds},
        "policies": policies,
        "tokens": token_stats(test_requests),
        "openai_evaluation": openai_result,
        "sample_requests": samples,
        "runtime_minutes": round((clock.monotonic() - started) / 60, 1),
        "source_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--oi", type=Path)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--openai-cycles", type=int, default=0)
    parser.add_argument(
        "--openai-models", default=None, help="comma-separated model[:effort]"
    )
    parser.add_argument("--openai-effort", default=None)
    parser.add_argument("--openai-budget-usd", type=float, default=5.0)
    # Six workers hit the account's rate limit (50 of 688 calls); three did not.
    parser.add_argument("--openai-workers", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse journaled successful OpenAI cycles in an existing output",
    )
    report = run(parser.parse_args())
    summary = {
        "scanner_test_auc": report["scanner"]["metrics"].get("test_all", {}).get("auc"),
        "policies_test_all": {
            name: {
                k: result["test_all"].get(k)
                for k in ("trades", "hit_rate_percent", "mean_net_bps", "total_net_inr")
            }
            for name, result in report["policies"].items()
        },
        "runtime_minutes": report["runtime_minutes"],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
