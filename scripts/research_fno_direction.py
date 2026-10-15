"""Direction study: do NSE's end-of-day F&O, cash and participant figures tell which
way a stock (against the market) or NIFTY goes in the next session?

  python -m scripts.research_fno_direction --output data/research/fno_direction_20261016

Inputs, all local: data/fno/features (fetch_nse_archives + build_fno_dataset),
data/fno/normalized (call-put implied-volatility spread), the Kite minute candles of
the 213 F&O stocks and NIFTY (09:40->15:00 window) and the movement scanner's test
predictions (data/research/scanner_move10_20261014).

Protocol, fixed before the holdout is looked at:
- A session's figures are published after its close, so they are matched only with
  the NEXT session's returns: open->close (primary: an intraday trade), close->close,
  the overnight gap, and 09:40->15:00 (the paper run's window; 213 stocks).
- Discovery 2020-01-01..2024-12-31 chooses everything; the holdout (2025-01-01 on) is
  evaluated once.
- Stocks: each session the F&O stocks are ranked by a metric; the daily Spearman
  correlation (IC) of the rank with the next-session return measures direction
  against the market (a market move shifts every stock alike and cancels). A metric
  is kept if its discovery IC on open->close has |t| >= 3 (Newey-West, 5 lags) and
  the same sign in at least 4 of the 5 discovery years.
- NIFTY: Spearman correlation of each market figure with NIFTY's next open->close
  return, kept on the same rule.
- Kept metrics are combined with their discovery signs (equal weight of ranks); a
  CatBoost and a logistic model on all metrics are fitted on 2020-2023 (early
  stopping 2024). All are tested once on the holdout: IC, a long-short top/bottom-10
  book after charges, and the side of the movement scanner's own picks (p >= 0.90).
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.fno import bhavcopy
from ai_trader.fno.iv import RATE, implied_vol
from ai_trader.paper.basis import nse_symbol, symbol_history

FEATURES_DIR = Path("data/fno/features")
NORMALIZED = Path("data/fno/normalized")
MINUTES = Path("data/candles_fno_20261011")
SCANNER = Path("data/research/scanner_move10_20261014/test_predictions.npz")
SYMBOLCHANGE = Path("data/nse/symbolchange.csv")
DISCOVERY_END = "2024-12-31"
FIT_END = "2023-12-31"
HOLDOUT_START = "2025-01-01"
DISCOVERY_YEARS = ("2020", "2021", "2022", "2023", "2024")
MIN_STOCKS = 30
NW_LAGS = 5
KEEP_T, KEEP_YEARS = 3.0, 4
GAP_LIMIT = 0.15  # an overnight jump beyond this is taken as a corporate action
COST = 0.00059 + 0.0003  # intraday round trip at Rs 2 lakh (Zerodha) + slippage
BOOK = 10
SEED = 20261016
TARGETS = ("oc_next", "cc_next", "gap_next", "window_next")
PRIMARY = "oc_next"

STOCK_COLUMNS = [
    "date",
    "symbol",
    "cm_prev_close",
    "cm_open",
    "cm_high",
    "cm_low",
    "cm_close",
    "cm_vwap",
    "cm_volume",
    "cm_turnover_lakh",
    "cm_deliv_pct",
    "fno",
    "in_ban",
    "mwpl_oi_pct",
    "fut_settle_near",
    "fut_oi_total",
    "fut_oi_change_pct",
    "fut_value_total",
    "fut_basis_pct",
    "fut_basis_ann_pct",
    "fut_rollover_pct",
    "opt_oi_ce",
    "opt_oi_pe",
    "opt_chg_oi_ce",
    "opt_chg_oi_pe",
    "opt_pcr_oi",
    "opt_pcr_vol",
    "opt_pcr_oi_near",
    "opt_value_total",
    "opt_max_pain_dist_pct",
    "opt_call_wall_dist_pct",
    "opt_put_wall_dist_pct",
    "iv_atm_near",
    "iv_atm_30d",
    "iv_skew_near",
    "iv_term",
]
# name -> what it is (all known at the session's close)
STOCK_FEATURES = {
    "ret_1d": "the session's return (close / previous close)",
    "ret_5d": "5-session return",
    "ret_20d": "20-session return",
    "close_vwap": "close against the session's average price",
    "range_pos": "close within the day's range (0 low, 1 high)",
    "vol_ratio": "volume / its previous 20-session average",
    "rv20": "20-session volatility of daily returns",
    "deliv_pct": "delivery % of traded quantity",
    "deliv_z": "delivery % against its previous 20 sessions (z)",
    "fut_basis": "near future's premium over the cash close %",
    "fut_basis_ann": "premium, annualised",
    "fut_oi_chg": "change in total futures OI %",
    "buildup": "long build-up +1, short covering +0.5, long unwinding -0.5, "
    "short build-up -1 (price and futures OI)",
    "fut_turnover_oi": "futures value traded / OI value",
    "rollover": "next/far months' share of futures OI %",
    "pcr_oi": "put / call open interest",
    "pcr_vol": "put / call contracts traded",
    "pcr_oi_near": "put / call OI, nearest expiry",
    "d_pcr_oi": "change in put / call OI",
    "opt_oi_flow": "(put OI added - call OI added) / option OI",
    "os_ratio": "option premium traded / stock turnover",
    "max_pain_dist": "max pain against the close %",
    "call_wall_dist": "largest call OI strike against the close %",
    "put_wall_dist": "largest put OI strike against the close %",
    "iv30": "30-day ATM implied volatility",
    "d_iv30": "change in 30-day ATM implied volatility",
    "iv_rv": "30-day implied minus 20-session realised volatility",
    "skew": "IV(0.95 F) - IV(1.05 F), nearest expiry",
    "d_skew": "change in skew",
    "term": "next minus nearest expiry ATM IV",
    "cpiv": "call minus put implied volatility at the same strikes (near ATM)",
    "parity_bp": "call - put - discounted (F - K), bp of F (put-call parity gap)",
    "mwpl": "combined OI / market-wide position limit %",
    "d_mwpl": "change in position-limit use",
    "in_ban": "in the F&O ban (1/0)",
}
MARKET_FEATURES = {
    "fii_fut_long": "FII long share of index futures OI %",
    "d_fii_fut_long": "change in FII long share",
    "client_fut_long": "client (retail) long share of index futures %",
    "pro_fut_long": "proprietary long share of index futures %",
    "fii_call_net": "FII index calls long - short (contracts, change)",
    "fii_put_net": "FII index puts long - short (contracts, change)",
    "fii_stk_fut_long": "FII long share of stock futures %",
    "nifty_pcr_oi": "NIFTY put / call OI",
    "nifty_pcr_vol": "NIFTY put / call contracts",
    "nifty_iv30": "NIFTY 30-day ATM IV",
    "d_nifty_iv30": "change in NIFTY 30-day ATM IV",
    "nifty_skew": "NIFTY skew IV(0.95F) - IV(1.05F)",
    "nifty_basis": "NIFTY near future premium %",
    "nifty_oi_chg": "change in NIFTY futures OI %",
    "vix": "India VIX",
    "d_vix": "change in India VIX %",
    "nifty_ret_1d": "NIFTY's session return",
    "nifty_ret_5d": "NIFTY's 5-session return",
    "breadth": "share of F&O stocks that rose",
    "avg_deliv": "average delivery % of F&O stocks",
    "nifty_pe": "NIFTY P/E",
    "stocks_in_ban": "stocks in the F&O ban",
}


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def neighbour(frame: pd.DataFrame, columns: list[str], step: int) -> pd.DataFrame:
    """Each row's ``columns`` at the same symbol's session ``step`` away (+1 next,
    -1 previous); NaN when the symbol has no row on exactly that session. ``frame``
    must be sorted by symbol and session."""
    shifted = frame.groupby("symbol", sort=False)[columns + ["session"]].shift(-step)
    found = shifted["session"] == frame["session"] + step
    return shifted[columns].where(found, np.nan)


def load_stocks(folder: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(path, usecols=STOCK_COLUMNS, dtype={"date": str, "symbol": str})
        for path in sorted(folder.glob("stock_daily_*.csv.gz"))
    ]
    data = pd.concat(frames, ignore_index=True)
    ever = set(data.loc[data["fno"] == 1, "symbol"])
    data = data[data["symbol"].isin(ever)].copy()
    sessions = sorted(data["date"].unique())
    data["session"] = data["date"].map({d: i for i, d in enumerate(sessions)})
    return data.sort_values(["symbol", "session"], ignore_index=True)


def load_indices(folder: Path) -> pd.DataFrame:
    frame = pd.read_csv(folder / "indices_daily.csv.gz", dtype={"date": str})
    nifty = frame[frame["index"] == "Nifty 50"].set_index("date")
    vix = frame[frame["index"] == "India VIX"].set_index("date")
    out = pd.DataFrame(
        {
            "nifty_open": nifty["open"],
            "nifty_close": nifty["close"],
            "nifty_pe": nifty["pe"],
            "vix": vix["close"],
        }
    )
    return out.sort_index()


def add_targets(data: pd.DataFrame, indices: pd.DataFrame) -> None:
    """Next-session returns of each row (NaN where the next session is missing or
    the overnight jump looks like a corporate action)."""
    nxt = neighbour(data, ["cm_open", "cm_close", "cm_prev_close"], 1)
    same_base = (nxt["cm_prev_close"] - data["cm_close"]).abs() <= 1e-6 * data[
        "cm_close"
    ]
    gap = nxt["cm_open"] / data["cm_close"] - 1
    clean = same_base & (gap.abs() <= GAP_LIMIT)
    data["oc_next"] = nxt["cm_close"] / nxt["cm_open"] - 1
    data["cc_next"] = (nxt["cm_close"] / data["cm_close"] - 1).where(clean)
    data["gap_next"] = gap.where(clean)
    sessions = sorted(data["date"].unique())
    following = dict(zip(sessions[:-1], sessions[1:], strict=True))
    nifty = indices.reindex(sessions)
    next_day = data["date"].map(following)
    n_open = next_day.map(nifty["nifty_open"])
    n_close = next_day.map(nifty["nifty_close"])
    data["nifty_oc_next"] = n_close / n_open - 1


def add_features(data: pd.DataFrame) -> None:
    g = data.groupby("symbol", sort=False)
    gap_today = data["cm_open"] / data["cm_prev_close"] - 1
    data["ret_1d"] = (data["cm_close"] / data["cm_prev_close"] - 1).where(
        gap_today.abs() <= GAP_LIMIT
    )
    log_ret = np.log1p(data["ret_1d"])
    for days in (5, 20):
        data[f"ret_{days}d"] = np.expm1(
            log_ret.groupby(data["symbol"], sort=False).transform(
                lambda s, d=days: s.rolling(d, min_periods=d).sum()
            )
        )
    data["rv20"] = log_ret.groupby(data["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=15).std()
    )
    data["close_vwap"] = data["cm_close"] / data["cm_vwap"] - 1
    span = data["cm_high"] - data["cm_low"]
    data["range_pos"] = ((data["cm_close"] - data["cm_low"]) / span).where(span > 0)
    prior_volume = g["cm_volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean()
    )
    data["vol_ratio"] = data["cm_volume"] / prior_volume
    data["deliv_pct"] = data["cm_deliv_pct"]
    prior = g["cm_deliv_pct"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean()
    )
    spread = g["cm_deliv_pct"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).std()
    )
    data["deliv_z"] = ((data["cm_deliv_pct"] - prior) / spread).where(spread > 0)
    data["fut_basis"] = data["fut_basis_pct"]
    data["fut_basis_ann"] = data["fut_basis_ann_pct"]
    data["fut_oi_chg"] = data["fut_oi_change_pct"]
    up, oi_up = np.sign(data["ret_1d"]), data["fut_oi_change_pct"] > 0
    data["buildup"] = (up * np.where(oi_up, 1.0, 0.5)).where(
        data["ret_1d"].notna() & data["fut_oi_change_pct"].notna()
    )
    oi_value = data["fut_oi_total"] * data["fut_settle_near"]
    data["fut_turnover_oi"] = (data["fut_value_total"] / oi_value).where(oi_value > 0)
    data["rollover"] = data["fut_rollover_pct"]
    data["pcr_oi"] = data["opt_pcr_oi"]
    data["pcr_vol"] = data["opt_pcr_vol"]
    data["pcr_oi_near"] = data["opt_pcr_oi_near"]
    option_oi = data["opt_oi_ce"] + data["opt_oi_pe"]
    data["opt_oi_flow"] = (
        (data["opt_chg_oi_pe"] - data["opt_chg_oi_ce"]) / option_oi
    ).where(option_oi > 0)
    turnover = data["cm_turnover_lakh"] * 1e5
    data["os_ratio"] = (data["opt_value_total"] / turnover).where(turnover > 0)
    data["max_pain_dist"] = data["opt_max_pain_dist_pct"]
    data["call_wall_dist"] = data["opt_call_wall_dist_pct"]
    data["put_wall_dist"] = data["opt_put_wall_dist_pct"]
    data["iv30"] = data["iv_atm_30d"]
    data["iv_rv"] = data["iv_atm_30d"] - data["rv20"] * math.sqrt(252)
    data["skew"] = data["iv_skew_near"]
    data["term"] = data["iv_term"]
    data["mwpl"] = data["mwpl_oi_pct"]
    previous = neighbour(data, ["opt_pcr_oi", "iv_atm_30d", "iv_skew_near"], -1)
    previous_mwpl = neighbour(data, ["mwpl_oi_pct"], -1)
    data["d_pcr_oi"] = data["opt_pcr_oi"] - previous["opt_pcr_oi"]
    data["d_iv30"] = data["iv_atm_30d"] - previous["iv_atm_30d"]
    data["d_skew"] = data["iv_skew_near"] - previous["iv_skew_near"]
    data["d_mwpl"] = data["mwpl_oi_pct"] - previous_mwpl["mwpl_oi_pct"]


def call_put_spread(
    table: dict[str, np.ndarray], day: date, band: float = 0.05, min_days: int = 3
) -> pd.DataFrame:
    """Per stock: OI-weighted mean of call IV - put IV, and of the put-call parity
    gap (C - P - DF (F - K)) / F in bp, over strikes within ``band`` of the forward
    traded on both sides, for the nearest expiry at least ``min_days`` away that
    has a future (its settlement is the forward). Cremers & Weinbaum (2010)."""
    session = np.datetime64(day, "D")
    stock = table["underlying"] == "S"
    futures = np.flatnonzero(stock & (table["kind"] == "F"))
    forward = {
        (str(table["symbol"][i]), table["expiry"][i]): float(table["settle"][i])
        for i in futures
    }
    traded = np.flatnonzero(
        stock & (table["kind"] == "O") & (table["contracts"] > 0) & (table["close"] > 0)
    )
    best: dict[str, np.datetime64] = {}
    for i in traded:
        symbol, expiry = str(table["symbol"][i]), table["expiry"][i]
        days = int((expiry - session).astype(int))
        if days >= min_days and (symbol, expiry) in forward:
            if symbol not in best or expiry < best[symbol]:
                best[symbol] = expiry
    sides: dict[tuple[str, float], dict[str, tuple[float, float]]] = {}
    for i in traded:
        symbol, expiry = str(table["symbol"][i]), table["expiry"][i]
        if best.get(symbol) != expiry:
            continue
        key = (symbol, float(table["strike"][i]))
        sides.setdefault(key, {})[str(table["option"][i])] = (
            float(table["close"][i]),
            float(table["oi"][i]),
        )
    rows = []
    for (symbol, strike), quotes in sides.items():
        if "CE" not in quotes or "PE" not in quotes:
            continue
        f = forward[(symbol, best[symbol])]
        if abs(strike / f - 1) > band:
            continue
        years = float((best[symbol] - session).astype(int)) / 365
        rows.append((symbol, strike, f, years, *quotes["CE"], *quotes["PE"]))
    columns = [
        "symbol",
        "strike",
        "forward",
        "years",
        "call",
        "call_oi",
        "put",
        "put_oi",
    ]
    pairs = pd.DataFrame(rows, columns=columns)
    if pairs.empty:
        return pd.DataFrame(columns=["symbol", "cpiv", "parity_bp", "pairs"])
    pairs["iv_call"] = implied_vol(
        pairs["call"], pairs["forward"], pairs["strike"], pairs["years"], True
    )
    pairs["iv_put"] = implied_vol(
        pairs["put"], pairs["forward"], pairs["strike"], pairs["years"], False
    )
    pairs = pairs[np.isfinite(pairs["iv_call"]) & np.isfinite(pairs["iv_put"])].copy()
    discount = np.exp(-RATE * pairs["years"])
    pairs["gap_bp"] = (
        (pairs["call"] - pairs["put"] - discount * (pairs["forward"] - pairs["strike"]))
        / pairs["forward"]
        * 1e4
    )
    pairs["weight"] = pairs["call_oi"] + pairs["put_oi"] + 1.0
    pairs["w_iv"] = pairs["weight"] * (pairs["iv_call"] - pairs["iv_put"])
    pairs["w_gap"] = pairs["weight"] * pairs["gap_bp"]
    sums = pairs.groupby("symbol")[["weight", "w_iv", "w_gap"]].sum()
    out = pd.DataFrame(
        {
            "cpiv": sums["w_iv"] / sums["weight"],
            "parity_bp": sums["w_gap"] / sums["weight"],
            "pairs": pairs.groupby("symbol").size(),
        }
    )
    return out.reset_index()


def spread_session(path: str) -> pd.DataFrame:
    table, _ = bhavcopy.load_table(Path(path))
    day = date.fromisoformat(Path(path).stem)
    out = call_put_spread(table, day)
    out["date"] = day.isoformat()
    return out


def window_prices(task: tuple[str, str]) -> pd.DataFrame:
    """09:40 open and 14:59 close per session of one Kite minute file."""
    path, symbol = task
    frame = pd.read_csv(path, usecols=["start_time", "open", "close"])
    hhmm = frame["start_time"].str.slice(11, 16)
    day = frame["start_time"].str.slice(0, 10)
    first = frame.loc[hhmm == "09:40", "open"].set_axis(day[hhmm == "09:40"])
    last = frame.loc[hhmm == "14:59", "close"].set_axis(day[hhmm == "14:59"])
    both = pd.DataFrame({"w_open": first, "w_close": last}).dropna()
    both["kite_symbol"] = symbol
    return both.rename_axis("date").reset_index()


def newey_west_t(values: np.ndarray, lags: int = NW_LAGS) -> float:
    n = len(values)
    if n < 10:
        return float("nan")
    x = values - values.mean()
    variance = x @ x / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (x[lag:] @ x[:-lag]) / n
    if variance <= 0:
        return float("nan")
    return float(values.mean() / math.sqrt(variance / n))


def daily_ic(frame: pd.DataFrame, feature: str, target: str) -> pd.Series:
    """Per session: Spearman correlation of ``feature`` with ``target`` across the
    stocks having both (sessions with fewer than MIN_STOCKS left out)."""
    sub = frame[["date", feature, target]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    groups = sub.groupby("date", sort=True)
    x = groups[feature].rank()
    y = groups[target].rank()
    parts = pd.DataFrame({"date": sub["date"], "x": x, "y": y})
    parts["xy"], parts["xx"], parts["yy"] = x * y, x * x, y * y
    s = parts.groupby("date").agg(
        n=("x", "size"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxy=("xy", "sum"),
        sxx=("xx", "sum"),
        syy=("yy", "sum"),
    )
    cov = s["sxy"] - s["sx"] * s["sy"] / s["n"]
    vx = s["sxx"] - s["sx"] ** 2 / s["n"]
    vy = s["syy"] - s["sy"] ** 2 / s["n"]
    ok = (s["n"] >= MIN_STOCKS) & (vx > 0) & (vy > 0)
    return (cov / np.sqrt(vx * vy))[ok]


def quintile_spread(frame: pd.DataFrame, feature: str, target: str) -> pd.Series:
    """Per session: mean ``target`` of the top fifth by ``feature`` minus the bottom."""
    sub = frame[["date", feature, target]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    pct = sub.groupby("date")[feature].rank(pct=True)
    parts = pd.DataFrame(
        {
            "date": sub["date"],
            "top": sub[target].where(pct > 0.8),
            "bottom": sub[target].where(pct <= 0.2),
        }
    )
    means = parts.groupby("date")[["top", "bottom"]].mean()
    counts = sub.groupby("date").size()
    return (means["top"] - means["bottom"])[counts >= MIN_STOCKS]


def summarize(series: pd.Series, scale: float = 1.0) -> dict:
    series = series.dropna()
    if series.empty:
        return {"days": 0}
    by_year = series.groupby(series.index.str.slice(0, 4)).mean() * scale
    return {
        "days": int(len(series)),
        "mean": rounded(series.mean() * scale, 5),
        "t_nw": rounded(newey_west_t(series.to_numpy()), 2),
        "positive_share": rounded(float((series > 0).mean()), 3),
        "by_year": {year: rounded(v, 5) for year, v in by_year.items()},
    }


def split(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    return (
        series[series.index <= DISCOVERY_END],
        series[series.index >= HOLDOUT_START],
    )


def kept(summary: dict) -> bool:
    """The discovery rule: |t| >= KEEP_T and the mean's sign in >= KEEP_YEARS years."""
    if summary.get("days", 0) == 0 or summary["t_nw"] is None:
        return False
    sign = np.sign(summary["mean"])
    agree = sum(
        1
        for year in DISCOVERY_YEARS
        if summary["by_year"].get(year) is not None
        and np.sign(summary["by_year"][year]) == sign
    )
    return abs(summary["t_nw"]) >= KEEP_T and agree >= KEEP_YEARS


def centred_rank(frame: pd.DataFrame, name: str) -> pd.Series:
    """The session's rank of ``name`` scaled to (-0.5, 0.5), symmetric around 0."""
    grouped = frame.groupby("date")[name]
    count = grouped.transform("count")
    return (grouped.rank() - (count + 1) / 2) / count


def composite(frame: pd.DataFrame, signs: dict[str, float]) -> pd.Series:
    """Equal-weight mean of the signed, centred daily ranks of the kept metrics."""
    parts = [sign * centred_rank(frame, name) for name, sign in signs.items()]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def book(frame: pd.DataFrame, score: str, target: str, size: int = BOOK) -> pd.Series:
    """Per session: long the ``size`` highest scores, short the lowest, equal
    weights; return on one side's capital after both sides' round trips."""
    sub = frame[["date", score, target]].dropna()
    order = sub.groupby("date")[score].rank(method="first")
    count = sub.groupby("date")[score].transform("size")
    longs = sub[target].where(order > count - size)
    shorts = sub[target].where(order <= size)
    parts = pd.DataFrame({"date": sub["date"], "long": longs, "short": shorts})
    means = parts.groupby("date")[["long", "short"]].mean()
    enough = sub.groupby("date").size() >= 4 * size
    return (means["long"] - means["short"] - 2 * COST)[enough]


def spearman(x: pd.Series, y: pd.Series) -> dict:
    both = pd.concat([x, y], axis=1).dropna()
    n = len(both)
    if n < 30:
        return {"n": n}
    r = float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()))
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return {"n": n, "r": rounded(r), "t": rounded(t, 2)}


def market_frame(
    data: pd.DataFrame, indices: pd.DataFrame, folder: Path
) -> pd.DataFrame:
    market = pd.read_csv(folder / "market_daily.csv.gz", dtype={"date": str})
    market = market.set_index("date").sort_index()
    index_rows = pd.read_csv(folder / "index_daily.csv.gz", dtype={"date": str})
    nifty_fno = index_rows[index_rows["symbol"] == "NIFTY"].set_index("date")
    sessions = sorted(data["date"].unique())
    out = pd.DataFrame(index=pd.Index(sessions, name="date"))
    out["fii_fut_long"] = market["oi_fii_idx_fut_long_pct"]
    out["d_fii_fut_long"] = out["fii_fut_long"].diff()
    out["client_fut_long"] = market["oi_client_idx_fut_long_pct"]
    out["pro_fut_long"] = market["oi_pro_idx_fut_long_pct"]
    out["fii_call_net"] = market["oi_fii_idx_call_net"].diff()
    out["fii_put_net"] = market["oi_fii_idx_put_net"].diff()
    out["fii_stk_fut_long"] = market["oi_fii_stk_fut_long_pct"]
    out["nifty_pcr_oi"] = nifty_fno["opt_pcr_oi"]
    out["nifty_pcr_vol"] = nifty_fno["opt_pcr_vol"]
    out["nifty_iv30"] = nifty_fno["iv_atm_30d"]
    out["d_nifty_iv30"] = out["nifty_iv30"].diff()
    out["nifty_skew"] = nifty_fno["iv_skew_near"]
    out["nifty_basis"] = nifty_fno["fut_basis_pct"]
    out["nifty_oi_chg"] = nifty_fno["fut_oi_change_pct"]
    nifty = indices.reindex(sessions)
    out["vix"] = nifty["vix"]
    out["d_vix"] = nifty["vix"].pct_change() * 100
    close = nifty["nifty_close"]
    out["nifty_ret_1d"] = close.pct_change()
    out["nifty_ret_5d"] = close.pct_change(5)
    out["nifty_pe"] = nifty["nifty_pe"]
    fno = data[data["fno"] == 1]
    out["breadth"] = fno.groupby("date")["ret_1d"].apply(lambda r: (r > 0).mean())
    out["avg_deliv"] = fno.groupby("date")["deliv_pct"].mean()
    out["stocks_in_ban"] = market["stocks_in_ban"]
    out["nifty_oc_next"] = (nifty["nifty_close"] / nifty["nifty_open"] - 1).shift(-1)
    out["nifty_cc_next"] = close.pct_change().shift(-1)
    out["nifty_gap_next"] = (nifty["nifty_open"] / close.shift(1) - 1).shift(-1)
    return out


def market_study(frame: pd.DataFrame) -> dict:
    results = {}
    discovery = frame[frame.index <= DISCOVERY_END]
    holdout = frame[frame.index >= HOLDOUT_START]
    for name in MARKET_FEATURES:
        entry = {}
        for target in ("nifty_oc_next", "nifty_cc_next", "nifty_gap_next"):
            entry[target] = {
                "discovery": spearman(discovery[name], discovery[target]),
                "holdout": spearman(holdout[name], holdout[target]),
            }
        years = {}
        for year in DISCOVERY_YEARS:
            part = discovery[discovery.index.str.startswith(year)]
            years[year] = spearman(part[name], part["nifty_oc_next"]).get("r")
        entry["discovery_by_year_oc"] = years
        main = entry["nifty_oc_next"]["discovery"]
        sign = np.sign(main.get("r") or 0)
        agree = sum(1 for r in years.values() if r is not None and np.sign(r) == sign)
        entry["kept"] = bool(abs(main.get("t") or 0) >= KEEP_T and agree >= KEEP_YEARS)
        results[name] = entry
    return results


def market_rule(frame: pd.DataFrame, name: str, sign: float) -> dict:
    """Long NIFTY for the next session's open->close when the figure is on the
    discovery-signed side of its discovery median, else short; holdout only."""
    discovery = frame[frame.index <= DISCOVERY_END]
    median = discovery[name].median()
    holdout = frame[frame.index >= HOLDOUT_START][[name, "nifty_oc_next"]].dropna()
    side = np.where(sign * (holdout[name] - median) > 0, 1.0, -1.0)
    gross = side * holdout["nifty_oc_next"]
    series = pd.Series(gross.to_numpy(), index=holdout.index)
    return {
        "days": int(len(series)),
        "hit_rate": rounded(float((series > 0).mean()), 3),
        "mean_gross_bp": rounded(series.mean() * 1e4, 2),
        "t_nw": rounded(newey_west_t(series.to_numpy()), 2),
    }


def fit_models(
    data: pd.DataFrame, features: list[str], market: list[str]
) -> tuple[pd.Series, pd.Series, dict]:
    """CatBoost and logistic regression on centred daily ranks of every stock metric
    (plus market figures for CatBoost); label: the next open->close beats NIFTY's."""
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rows = data[
        (data["fno"] == 1) & data[PRIMARY].notna() & data["nifty_oc_next"].notna()
    ]
    ranks = pd.DataFrame(index=rows.index)
    for name in features:
        ranks[name] = centred_rank(rows, name)
    for name in market:
        ranks[name] = rows[name]
    label = (rows[PRIMARY] > rows["nifty_oc_next"]).astype(int)
    fit = rows["date"] <= FIT_END
    select = (rows["date"] > FIT_END) & (rows["date"] <= DISCOVERY_END)
    test = rows["date"] >= HOLDOUT_START
    model = CatBoostClassifier(
        iterations=2000,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=10,
        loss_function="Logloss",
        od_type="Iter",
        od_wait=150,
        random_seed=SEED,
        thread_count=8,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(ranks[fit], label[fit], eval_set=(ranks[select], label[select]))
    trees = pd.Series(model.predict_proba(ranks)[:, 1], index=ranks.index)
    linear_inputs = ranks[features].fillna(0.0)
    logistic = LogisticRegression(C=0.1, max_iter=2000)
    logistic.fit(linear_inputs[fit], label[fit])
    linear = pd.Series(logistic.predict_proba(linear_inputs)[:, 1], index=ranks.index)
    importance = sorted(
        zip(ranks.columns, model.get_feature_importance(), strict=True),
        key=lambda item: -item[1],
    )
    report = {
        "rows": {
            "fit": int(fit.sum()),
            "select": int(select.sum()),
            "test": int(test.sum()),
        },
        "trees": int(model.tree_count_),
        "auc": {
            "catboost": {
                "select": rounded(roc_auc_score(label[select], trees[select])),
                "test": rounded(roc_auc_score(label[test], trees[test])),
            },
            "logistic": {
                "select": rounded(roc_auc_score(label[select], linear[select])),
                "test": rounded(roc_auc_score(label[test], linear[test])),
            },
        },
        "test_base_rate": rounded(float(label[test].mean())),
        "importance_top12": [[n, rounded(v, 2)] for n, v in importance[:12]],
        "logistic_coefficients": {
            n: rounded(c, 4)
            for n, c in sorted(
                zip(features, logistic.coef_[0], strict=True), key=lambda i: -abs(i[1])
            )[:12]
        },
    }
    return trees, linear, report


def within_day_auc(day: pd.Series, label: pd.Series, score: pd.Series) -> dict:
    """Mean over sessions (>= 20 picks, both outcomes) of the session's own AUC:
    the market moves a session's picks together, a pooled AUC mixes that in."""
    from sklearn.metrics import roc_auc_score

    values = []
    for index in day.groupby(day).groups.values():
        outcome = label.loc[index]
        if len(index) >= 20 and 0 < outcome.mean() < 1:
            values.append(roc_auc_score(outcome, score.loc[index]))
    if len(values) < 10:
        return {"sessions": len(values)}
    v = np.array(values)
    return {
        "sessions": len(v),
        "mean": rounded(float(v.mean())),
        "t_vs_half": rounded((v.mean() - 0.5) / (v.std(ddof=1) / math.sqrt(len(v))), 2),
    }


def scanner_sides(data: pd.DataFrame, scores: dict[str, float], path: Path) -> dict:
    """The movement scanner's own picks (p >= 0.90, 2025-2026 test): does the
    previous session's score, oriented by its discovery sign (``scores``), pick
    the side that moved 1% first?"""
    from sklearn.metrics import roc_auc_score

    saved = np.load(path)
    days = np.array(
        [date.fromordinal(int(d) + date(1970, 1, 1).toordinal()) for d in saved["days"]]
    )
    pick = (saved["probability"] >= 0.90) & (saved["up_first"] >= 0)
    rows = pd.DataFrame(
        {
            "day": days[saved["day"][pick]],
            "slot": saved["slot"][pick],
            "kite": saved["stocks"][saved["stock"][pick]],
            "up_first": saved["up_first"][pick].astype(int),
        }
    )
    history = symbol_history(SYMBOLCHANGE.read_text("utf-8"))
    rows["symbol"] = [
        nse_symbol(k, d, history.get(k, []))
        for k, d in zip(rows["kite"], rows["day"], strict=True)
    ]
    rows["day"] = rows["day"].map(date.isoformat)
    sessions = sorted(data["date"].unique())
    before = dict(zip(sessions[1:], sessions[:-1], strict=True))
    rows["date"] = rows["day"].map(before)
    keyed = data.set_index(["symbol", "date"])[list(scores)]
    joined = rows.join(keyed, on=["symbol", "date"])
    first = joined.sort_values("slot").drop_duplicates(["day", "kite"])
    out = {"rows_all": int(len(joined)), "rows_first_pick": int(len(first))}
    for label, frame in (("all", joined), ("first_pick", first)):
        result = {"up_first_share": rounded(float(frame["up_first"].mean()))}
        for score, sign in scores.items():
            sub = frame[["day", score, "up_first"]].dropna()
            if len(sub) < 100 or sub["up_first"].nunique() < 2:
                continue
            oriented = sign * sub[score]
            side_right = np.where(
                oriented > oriented.median(), sub["up_first"], 1 - sub["up_first"]
            )
            result[score] = {
                "sign": sign,
                "rows": int(len(sub)),
                "auc": rounded(roc_auc_score(sub["up_first"], oriented)),
                "side_right": rounded(float(side_right.mean())),
                "auc_within_session": within_day_auc(
                    sub["day"], sub["up_first"], oriented
                ),
            }
        out[label] = result
    return out


def stock_study(data: pd.DataFrame) -> dict:
    fno = data[data["fno"] == 1]
    results = {}
    for name in STOCK_FEATURES:
        entry = {}
        for target in TARGETS:
            ic = daily_ic(fno, name, target)
            spread = quintile_spread(fno, name, target)
            ic_d, ic_h = split(ic)
            sp_d, sp_h = split(spread)
            entry[target] = {
                "ic_discovery": summarize(ic_d),
                "ic_holdout": summarize(ic_h),
                "q5_q1_bp_discovery": summarize(sp_d, 1e4),
                "q5_q1_bp_holdout": summarize(sp_h, 1e4),
            }
        entry["kept"] = kept(entry[PRIMARY]["ic_discovery"])
        results[name] = entry
        log(
            f"{name:16s} IC disc {entry[PRIMARY]['ic_discovery'].get('mean')} "
            f"t {entry[PRIMARY]['ic_discovery'].get('t_nw')} | hold "
            f"{entry[PRIMARY]['ic_holdout'].get('mean')} "
            f"t {entry[PRIMARY]['ic_holdout'].get('t_nw')}"
            + ("  KEPT" if entry["kept"] else "")
        )
    return results


def score_report(data: pd.DataFrame, score: str) -> dict:
    fno = data[data["fno"] == 1]
    out = {}
    for target in TARGETS:
        ic_d, ic_h = split(daily_ic(fno, score, target))
        out[target] = {"ic_discovery": summarize(ic_d), "ic_holdout": summarize(ic_h)}
    for target in (PRIMARY, "window_next"):
        disc, hold = split(book(fno, score, target))
        out[f"book_{target}_net_bp"] = {
            "discovery": summarize(disc, 1e4),
            "holdout": summarize(hold, 1e4),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    log("loading stock-days")
    data = load_stocks(FEATURES_DIR)
    indices = load_indices(FEATURES_DIR)
    add_targets(data, indices)
    add_features(data)
    log(f"{len(data)} rows, {int((data['fno'] == 1).sum())} F&O stock-days")

    log("call-put implied-volatility spread from the normalized sessions")
    paths = [str(p) for p in sorted(NORMALIZED.rglob("*.npz"))]
    with ProcessPoolExecutor(args.workers) as pool:
        spreads = pd.concat(
            pool.map(spread_session, paths, chunksize=8), ignore_index=True
        )
    data = data.merge(
        spreads[["date", "symbol", "cpiv", "parity_bp"]],
        on=["date", "symbol"],
        how="left",
    )
    data = data.sort_values(["symbol", "session"], ignore_index=True)

    log("09:40 -> 15:00 window from the minute candles")
    history = symbol_history(SYMBOLCHANGE.read_text("utf-8"))
    tasks = [
        (str(p), p.stem.removeprefix("NSE_"))
        for p in sorted(MINUTES.glob("NSE_*.csv"))
        if not p.stem.endswith(".fetched")
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        windows = pd.concat(pool.map(window_prices, tasks), ignore_index=True)
    windows["symbol"] = [
        nse_symbol(k, date.fromisoformat(d), history.get(k, []))
        for k, d in zip(windows["kite_symbol"], windows["date"], strict=True)
    ]
    nifty_window = windows[windows["kite_symbol"] == "NIFTY"].set_index("date")
    windows["w_ret"] = windows["w_close"] / windows["w_open"] - 1
    keyed = windows.set_index(["symbol", "date"])["w_ret"]
    sessions = sorted(data["date"].unique())
    following = dict(zip(sessions[:-1], sessions[1:], strict=True))
    next_day = data["date"].map(following)
    data["window_next"] = pd.Series(
        keyed.reindex(list(zip(data["symbol"], next_day, strict=True))).to_numpy(),
        index=data.index,
    )
    nifty_w = nifty_window["w_close"] / nifty_window["w_open"] - 1
    data["nifty_window_next"] = next_day.map(nifty_w)
    log(f"window returns for {int(data['window_next'].notna().sum())} stock-days")

    log("single metrics (stocks)")
    stocks = stock_study(data)
    signs = {
        name: float(np.sign(entry[PRIMARY]["ic_discovery"]["mean"]))
        for name, entry in stocks.items()
        if entry["kept"]
    }
    log(f"kept: {signs}")

    log("market figures (NIFTY)")
    market = market_frame(data, indices, FEATURES_DIR)
    market_results = market_study(market)
    market_rules = {
        name: market_rule(
            market, name, float(np.sign(entry["nifty_oc_next"]["discovery"]["r"]))
        )
        for name, entry in market_results.items()
        if entry["kept"]
    }

    log("combinations")
    combined: dict = {}
    scores = []
    if signs:
        data["composite"] = composite(data[data["fno"] == 1], signs)
        combined["composite"] = score_report(data, "composite")
        scores.append("composite")
    data = data.join(
        market.drop(columns=[c for c in market.columns if "next" in c]), on="date"
    )
    trees, linear, model_report = fit_models(
        data, list(STOCK_FEATURES), list(MARKET_FEATURES)
    )
    data["catboost"], data["logistic"] = trees, linear
    combined["catboost"] = score_report(data, "catboost")
    combined["logistic"] = score_report(data, "logistic")
    scores += ["catboost", "logistic"]
    oriented = {name: 1.0 for name in scores} | signs

    log("the movement scanner's picks")
    sides = scanner_sides(data, oriented, SCANNER)

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "research",
        "protocol": __doc__,
        "periods": {
            "discovery": ["2020-01-01", DISCOVERY_END],
            "fit": ["2020-01-01", FIT_END],
            "holdout": [HOLDOUT_START, max(sessions)],
        },
        "rules": {
            "keep": f"|t_NW| >= {KEEP_T} and same sign in >= {KEEP_YEARS} of "
            f"{len(DISCOVERY_YEARS)} discovery years (primary target {PRIMARY})",
            "cost_per_round_trip": COST,
            "gap_limit": GAP_LIMIT,
            "book_size": BOOK,
        },
        "rows": {
            "stock_days": int(len(data)),
            "fno_stock_days": int((data["fno"] == 1).sum()),
            "with_window": int(data["window_next"].notna().sum()),
            "with_cpiv": int(data["cpiv"].notna().sum()),
        },
        "stock_features": STOCK_FEATURES,
        "market_features": MARKET_FEATURES,
        "stocks": stocks,
        "kept_signs": signs,
        "market": market_results,
        "market_rules_holdout": market_rules,
        "combined": combined,
        "models": model_report,
        "scanner_picks": sides,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=1), "utf-8")
    log(f"report in {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
