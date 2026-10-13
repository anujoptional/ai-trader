"""Delivery (CNC, long-only) strategy search on Kite daily candles (research only).

Signals at a close, trades at the next session's close. Select on Aug 2024-Jun 2025
(highest CAGR with drawdown no worse than holding NIFTYBEES), then evaluate the
frozen choice once on Jul 2025-Sep 2026.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import UTC, date, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ai_trader.costs import (
    EXCHANGE_TRANSACTION_FRACTION,
    GOODS_AND_SERVICES_TAX_FRACTION,
    INVESTOR_PROTECTION_FUND_FRACTION,
    REGULATOR_FEE_FRACTION,
)
from ai_trader.history import CandleStore
from scripts.research_orb_strategy import read_symbols
from scripts.train_tp_only import file_hash, write_json

DAILY_HEADER = ["timestamp", "open", "high", "low", "close", "volume"]
EXCLUDED = ("TMPV",)
EQUITY_ETFS = ("NIFTYBEES", "BANKBEES")
METAL_ETFS = ("GOLDBEES", "SILVERBEES")
INDEX_PROXIES = (
    "NIFTYIT",
    "NIFTYPHARMA",
    "NIFTYFMCG",
    "NIFTYAUTO",
    "NIFTYMETAL",
    "NIFTYREALTY",
    "NIFTYMIDCAP150",
)
ROTATION = EQUITY_ETFS + METAL_ETFS + INDEX_PROXIES
REGIME = "NIFTY"
CAPITAL_INR = 1_000_000
DP_INR = 15.34
SLIPPAGE = 0.0002
CASH_RATE = 0.055
PROXY_EXPENSE = 0.0025
DAYS_PER_YEAR = 250
START = "2024-08-01"
TRAIN = ("2024-08-01", "2025-06-30")
TEST = ("2025-07-01", "2026-09-30")
SOURCES = (
    "research_delivery_strategy.py",
    "research_orb_strategy.py",
    "train_tp_only.py",
)
_LEVIES = float(
    EXCHANGE_TRANSACTION_FRACTION
    + INVESTOR_PROTECTION_FUND_FRACTION
    + REGULATOR_FEE_FRACTION
) * (1 + float(GOODS_AND_SERVICES_TAX_FRACTION))


def side_costs(kind: str) -> tuple[float, float]:
    stt = {"stock": 0.001, "equity_etf": 0.0, "metal_etf": 0.0}[kind]
    sell_stt = {"stock": 0.001, "equity_etf": 0.00001, "metal_etf": 0.0}[kind]
    return stt + 0.00015 + _LEVIES, sell_stt + _LEVIES


def read_closes(path: Path) -> dict[str, float]:
    closes = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != DAILY_HEADER:
            raise ValueError(f"Unexpected daily header in {path.name}")
        for timestamp, *_, close, _ in reader:
            closes[timestamp[:10]] = float(close)
    return closes


def load_prices(candles: Path, stocks: list[str]) -> tuple[list[str], np.ndarray, list]:
    store = CandleStore(candles)
    names = [*stocks, *ROTATION, REGIME]
    series = {
        name: read_closes(store.verified_path(f"daily/NSE_{name}.csv"))
        for name in names
    }
    dates = sorted(set().union(*(series[name] for name in [*stocks, *EQUITY_ETFS])))
    prices = np.full((len(dates), len(names)), np.nan)
    for column, name in enumerate(names):
        for row, day in enumerate(dates):
            prices[row, column] = series[name].get(day, np.nan)
    return dates, prices, names


def forward_fill(prices: np.ndarray) -> np.ndarray:
    filled = prices.copy()
    for row in range(1, len(filled)):
        missing = np.isnan(filled[row])
        filled[row, missing] = filled[row - 1, missing]
    return filled


def top(scores: np.ndarray, count: int) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(scores))
    return valid[np.argsort(-scores[valid], kind="stable")[:count]]


def simulate(
    prices: np.ndarray,
    tradable: np.ndarray,
    buy_cost: np.ndarray,
    sell_cost: np.ndarray,
    drag: np.ndarray,
    goals: dict[int, tuple[np.ndarray, bool]],
    start: int,
) -> dict[str, np.ndarray]:
    days, assets = prices.shape
    cash_daily = (1 + CASH_RATE) ** (1 / DAYS_PER_YEAR) - 1
    weights = np.zeros(assets)
    cash = 1.0
    value = float(CAPITAL_INR)
    returns = np.zeros(days)
    costs = np.zeros(days)
    invested = np.zeros(days)
    pending: tuple[np.ndarray, bool] | None = None
    for t in range(start, days):
        if t > start:
            change = np.nan_to_num(prices[t] / prices[t - 1] - 1) - drag
            grown = weights * (1 + change)
            total = grown.sum() + cash * (1 + cash_daily)
            weights, cash = grown / total, cash * (1 + cash_daily) / total
            returns[t] = total - 1
            value *= total
        if pending is not None:
            goal, full = pending
            target = goal.copy() if full else np.where(goal > 0, weights, 0.0)
            if not full:
                entries = (goal > 0) & (weights <= 1e-12)
                room = 1 - target.sum()
                wanted = goal[entries].sum()
                if wanted > 0:
                    target[entries] = goal[entries] * min(1.0, max(room, 0.0) / wanted)
            target = np.where(tradable[t], target, weights)
            for _ in range(2):
                trade = target - weights
                buys, sells = np.clip(trade, 0, None), np.clip(-trade, 0, None)
                cost = (
                    buys @ buy_cost
                    + sells @ sell_cost
                    + SLIPPAGE * (buys.sum() + sells.sum())
                    + DP_INR * np.count_nonzero(sells > 1e-9) / value
                )
                left = cash + weights.sum() - target.sum() - cost
                if left >= 0 or buys.sum() <= 0:
                    break
                target = (
                    weights
                    + trade.clip(None, 0)
                    + buys * max(0.0, 1 + left / buys.sum())
                )
            cash = cash + weights.sum() - target.sum() - cost
            weights, cash = target / (1 - cost), cash / (1 - cost)
            returns[t] = (1 + returns[t]) * (1 - cost) - 1
            value *= 1 - cost
            costs[t] = cost
            pending = None
        if t in goals:
            pending = goals[t]
        invested[t] = weights.sum()
    return {"returns": returns, "costs": costs, "invested": invested}


def statistics(result: dict[str, np.ndarray], rows: np.ndarray) -> dict:
    returns = result["returns"][rows]
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    years = len(returns) / DAYS_PER_YEAR
    deviation = returns.std(ddof=1)
    return {
        "sessions": int(len(returns)),
        "total_return_percent": 100 * float(equity[-1] - 1),
        "cagr_percent": 100 * float(equity[-1] ** (1 / years) - 1),
        "volatility_percent": 100 * float(deviation * np.sqrt(DAYS_PER_YEAR)),
        "sharpe": float(returns.mean() / deviation * np.sqrt(DAYS_PER_YEAR))
        if deviation > 0
        else None,
        "max_drawdown_percent": 100 * float((1 - equity / peak).max()),
        "cost_drag_percent_per_year": 100 * float(result["costs"][rows].sum() / years),
        "average_invested_percent": 100 * float(result["invested"][rows].mean()),
    }


def build_goals(
    family: str,
    params: dict,
    prices: np.ndarray,
    names: list[str],
    stocks: list[str],
    start: int,
) -> dict[int, tuple[np.ndarray, bool]]:
    index = {name: column for column, name in enumerate(names)}
    stock_columns = np.asarray([index[name] for name in stocks])
    rotation_columns = np.asarray([index[name] for name in ROTATION])
    regime = prices[:, index[REGIME]]
    days, assets = prices.shape
    goals = {}
    every = params.get("every", 1)
    for t in range(start, days - 1, every):
        goal = np.zeros(assets)
        full = False
        if family == "momentum":
            lag, look = params["skip"], params["lookback"]
            score = (
                prices[t - lag, stock_columns] / prices[t - lag - look, stock_columns]
                - 1
            )
            risk_on = not params["regime"] or regime[t] > regime[t - 99 : t + 1].mean()
            if risk_on:
                goal[stock_columns[top(score, params["count"])]] = 1 / params["count"]
        elif family == "reversal":
            look = params["lookback"]
            score = -(prices[t, stock_columns] / prices[t - look, stock_columns] - 1)
            if params["trend"]:
                average = prices[t - 49 : t + 1, stock_columns].mean(axis=0)
                score = np.where(prices[t, stock_columns] > average, score, np.nan)
            goal[stock_columns[top(score, params["count"])]] = 1 / params["count"]
        elif family == "lowvol":
            window = np.log(prices[t - 63 : t + 1, stock_columns])
            score = -np.diff(window, axis=0).std(axis=0)
            risk_on = not params["regime"] or regime[t] > regime[t - 99 : t + 1].mean()
            if risk_on:
                goal[stock_columns[top(score, params["count"])]] = 1 / params["count"]
        elif family == "rotation":
            look = params["lookback"]
            score = prices[t, rotation_columns] / prices[t - look, rotation_columns] - 1
            if params["absolute"]:
                score = np.where(score > 0, score, np.nan)
            goal[rotation_columns[top(score, params["count"])]] = 1 / params["count"]
        elif family == "trend":
            if regime[t] > regime[t - params["average"] + 1 : t + 1].mean():
                goal[index["NIFTYBEES"]] = 1
            elif params["fallback"] == "gold":
                goal[index["GOLDBEES"]] = 1
        else:
            full = True
            if params["mix"] == "nifty":
                goal[index["NIFTYBEES"]] = 1
            elif params["mix"] == "gold":
                goal[index["GOLDBEES"]] = 1
            elif params["mix"] == "balanced":
                goal[index["NIFTYBEES"]], goal[index["GOLDBEES"]] = 0.6, 0.4
            else:
                goal[stock_columns] = 1 / len(stock_columns)
        goals[t] = (goal, full)
    return goals


def variant_grid() -> list[tuple[str, str, dict]]:
    grid = []
    for look, skip, count, every, regime in product(
        (63, 126), (0, 10), (5, 10), (5, 21), (False, True)
    ):
        grid.append(
            (
                f"momentum_l{look}_s{skip}_top{count}_r{every}"
                f"{'_regime' if regime else ''}",
                "momentum",
                {
                    "lookback": look,
                    "skip": skip,
                    "count": count,
                    "every": every,
                    "regime": regime,
                },
            )
        )
    for look, trend, count, every in product((5, 10), (False, True), (5, 10), (5, 10)):
        grid.append(
            (
                f"reversal_l{look}_top{count}_r{every}{'_trend' if trend else ''}",
                "reversal",
                {"lookback": look, "trend": trend, "count": count, "every": every},
            )
        )
    for count, regime in product((5, 10), (False, True)):
        grid.append(
            (
                f"lowvol_top{count}_r21{'_regime' if regime else ''}",
                "lowvol",
                {"count": count, "every": 21, "regime": regime},
            )
        )
    for look, count, absolute, every in product(
        (21, 63, 126), (1, 2, 3), (False, True), (5, 21)
    ):
        grid.append(
            (
                f"rotation_l{look}_top{count}_r{every}{'_abs' if absolute else ''}",
                "rotation",
                {
                    "lookback": look,
                    "count": count,
                    "absolute": absolute,
                    "every": every,
                },
            )
        )
    for average, fallback in product((50, 100), ("cash", "gold")):
        grid.append(
            (
                f"trend_sma{average}_{fallback}",
                "trend",
                {"average": average, "fallback": fallback, "every": 1},
            )
        )
    for mix, every in (("nifty", 10_000), ("gold", 10_000), ("balanced", 63)):
        grid.append((f"hold_{mix}", "hold", {"mix": mix, "every": every}))
    grid.append(("hold_equal_stocks_r63", "hold", {"mix": "stocks", "every": 63}))
    return grid


def run(candles: Path, constituents: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier studies")
    here = Path(__file__).parent
    sources = {name: file_hash(here / name) for name in SOURCES}
    stocks = [name for name in read_symbols(constituents) if name not in EXCLUDED]
    output.mkdir(parents=True)
    write_json(
        output / "protocol.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "data": "verified Kite daily closes (split/bonus adjusted, dividends "
            "excluded); sector indices stand in for sector ETFs with a "
            f"{100 * PROXY_EXPENSE}%/year drag",
            "universe": {
                "stocks": f"{constituents.name} minus {EXCLUDED} (demerger jump)",
                "rotation": ROTATION,
            },
            "execution": "signal at close t, trade at close t+1; entries fill "
            "free slots (1/k each), continuing holdings are not trimmed except "
            "for hold_* rebalanced mixes",
            "costs": {
                "stock_buy_sell": side_costs("stock"),
                "equity_etf_buy_sell": side_costs("equity_etf"),
                "metal_etf_buy_sell": side_costs("metal_etf"),
                "dp_inr_per_scrip_sale": DP_INR,
                "slippage_per_side": SLIPPAGE,
                "capital_inr": CAPITAL_INR,
                "cash_rate": CASH_RATE,
            },
            "selection_rule": "highest train CAGR among variants whose train max "
            "drawdown is no worse than hold_nifty's; evaluate frozen choice on test",
            "periods": {"start": START, "train": TRAIN, "test": TEST},
            "taxes": "not deducted: STCG 20% (+4% cess) under 12 months, LTCG 12.5% "
            "(+4% cess) above Rs 1.25 lakh per year after 12 months",
            "versions": {name: version(name) for name in ("numpy", "scipy")},
            "source_sha256": sources,
        },
    )
    (output / "code").mkdir()
    for name in SOURCES:
        shutil.copyfile(here / name, output / "code" / name)

    dates, raw, names = load_prices(candles, stocks)
    prices = forward_fill(raw)
    tradable = np.isfinite(raw)
    kinds = [
        "stock"
        if name in stocks
        else "metal_etf"
        if name in METAL_ETFS
        else "equity_etf"
        for name in names
    ]
    costs = np.asarray([side_costs(kind) for kind in kinds])
    drag = np.asarray(
        [
            PROXY_EXPENSE / DAYS_PER_YEAR if name in INDEX_PROXIES else 0.0
            for name in names
        ]
    )
    tradable[:, names.index(REGIME)] = False
    start = dates.index(min(day for day in dates if day >= START))
    labels = np.asarray(dates)
    train = np.flatnonzero((labels > dates[start]) & (labels <= TRAIN[1]))
    test = np.flatnonzero((labels >= TEST[0]) & (labels <= TEST[1]))
    records = []
    for name, family, params in variant_grid():
        goals = build_goals(family, params, prices, names, stocks, start)
        result = simulate(
            prices, tradable, costs[:, 0], costs[:, 1], drag, goals, start
        )
        records.append(
            {
                "name": name,
                "family": family,
                "params": params,
                "train": statistics(result, train),
                "test": statistics(result, test),
            }
        )
    benchmark = next(r for r in records if r["name"] == "hold_nifty")
    limit = benchmark["train"]["max_drawdown_percent"]
    eligible = [r for r in records if r["train"]["max_drawdown_percent"] <= limit]
    chosen = max(eligible, key=lambda record: record["train"]["cagr_percent"])
    by_train = sorted(records, key=lambda record: -record["train"]["cagr_percent"])
    train_values = [r["train"]["cagr_percent"] for r in records]
    test_values = [r["test"]["cagr_percent"] for r in records]
    report = {
        "status": "research_complete_not_promoted",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "coverage": {
            "stocks": len(stocks),
            "rotation_assets": len(ROTATION),
            "first_signal": dates[start],
            "train_sessions": len(train),
            "test_sessions": len(test),
            "last_date": dates[-1],
        },
        "drawdown_limit_percent": limit,
        "eligible_variants": len(eligible),
        "chosen": chosen,
        "benchmarks": {
            r["name"]: {"train": r["train"], "test": r["test"]}
            for r in records
            if r["family"] == "hold"
        },
        "train_test_rank_correlation": float(
            spearmanr(train_values, test_values).statistic
        ),
        "top_train": by_train[:15],
        "variants": records,
        "generated": date.today().isoformat(),
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--constituents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.candles, args.constituents, args.output)
    print(json.dumps({"chosen": report["chosen"]["name"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
