"""Follow-up to research_fno_direction: WHEN in the next session does the edge of the
end-of-day metrics happen, and could an intraday trade capture it?

  python -m scripts.research_fno_entry_timing \
      --direction data/research/fno_direction_20261016/report.json \
      --output data/research/fno_entry_timing_20261016

The direction study's composite of the kept metrics ranks the next open->close
return (holdout IC 0.066; a top/bottom-10 book +22 bp a session after charges) but
ranks the overnight gap with the opposite sign and keeps little from 09:40 on.

Protocol (metrics, signs and the composite fixed by that study's discovery run;
every other score oriented by its discovery IC; discovery 2020-2024 and holdout
2025 on reported side by side, nothing chosen on the holdout):
- Minute prices on NSE's price basis (data/candles_fno_basis_20261011, its exclusion
  lists applied): for each next session the 09:15 bar's open (the pre-open auction
  price), the open of each later entry minute and the 14:59 bar's close (the paper
  run's exit); NIFTY at the same minutes. The files carry later corporate actions
  back (one factor per stock and period), so only ratios within a session are used.
- For each entry minute -> 14:59 and each score: daily IC; the top/bottom-10 book
  gross and net of 0.089% (Rs 2 lakh at Zerodha + 0.03% slippage), 0.15% and 0.25% a
  round trip; the book's NIFTY beta and alpha over the same minutes; the legs
  against the stocks' average; and the opening leg 09:15 -> entry.
- Scores: the composite; the realised overnight gap (public from the 09:08 auction,
  so only for entries after 09:15); gap and composite mixed; and the "tug of war"
  pair (a stock's mean overnight and mean open->close return over its last 20
  sessions; Lou, Polk & Skouras 2019).
- The overnight leg (NSE's close -> next official open, the auction price) of the
  composite, gross, for reference only: holding overnight needs delivery or futures.
- The auction book (composite, 09:15 -> 14:59): worst session, drawdown, positive
  months, the names it trades most and the book without the 10 names most frequent
  in discovery.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.paper.basis import nse_symbol, symbol_history
from scripts import research_fno_direction as study

BASIS = Path("data/candles_fno_basis_20261011")
ENTRIES = ("09:15", "09:16", "09:18", "09:20", "09:25", "09:30", "09:40", "10:15")
EXIT = "14:59"
COSTS = (0.00089, 0.0015, 0.0025)
EXCLUDED = (
    "dropped",
    "coarse_vendor_rounding",
    "imprecise_fit",
    "minutes_outside_nse_range",
    "structural_break",
)
GAP_SCORES = ("gap_fade", "gap_mix")


def minute_prices(task: tuple[str, str]) -> pd.DataFrame:
    """Per session of one minute file: the open of each entry minute's bar and the
    close of the exit bar (NaN where a bar is missing)."""
    path, symbol = task
    frame = pd.read_csv(path, usecols=["start_time", "open", "close"])
    frame = frame.drop_duplicates("start_time")
    hhmm = frame["start_time"].str.slice(11, 16).to_numpy()
    day = frame["start_time"].str.slice(0, 10).to_numpy()
    columns = {}
    for minute in ENTRIES:
        pick = hhmm == minute
        columns[minute] = pd.Series(frame["open"].to_numpy()[pick], index=day[pick])
    pick = hhmm == EXIT
    columns["exit"] = pd.Series(frame["close"].to_numpy()[pick], index=day[pick])
    out = pd.DataFrame(columns)
    out["kite_symbol"] = symbol
    return out.rename_axis("date").reset_index()


def excluded_days(folder: Path) -> set[tuple[str, str]]:
    flags = json.loads((folder / "exclusions.json").read_text("utf-8"))
    return {
        (symbol, day)
        for symbol, reasons in flags.items()
        for reason in EXCLUDED
        for day in reasons.get(reason, [])
    }


def add_tug_of_war(data: pd.DataFrame) -> None:
    """Mean overnight gap and mean open->close return over the last 20 sessions."""
    gap = data["cm_open"] / data["cm_prev_close"] - 1
    data["gap_today"] = gap.where(gap.abs() <= study.GAP_LIMIT)
    data["oc_today"] = data["cm_close"] / data["cm_open"] - 1
    for name, column in (("ovn20", "gap_today"), ("intra20", "oc_today")):
        data[name] = data.groupby("symbol", sort=False)[column].transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )


def book_legs(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    """Per session: mean ``target`` of the 10 highest and 10 lowest scores and of
    all stocks (sessions with fewer than 40 stocks left out)."""
    sub = frame[["date", score, target]].dropna()
    order = sub.groupby("date")[score].rank(method="first")
    count = sub.groupby("date")[score].transform("size")
    parts = pd.DataFrame(
        {
            "date": sub["date"],
            "long": sub[target].where(order > count - study.BOOK),
            "short": sub[target].where(order <= study.BOOK),
            "all": sub[target],
        }
    )
    means = parts.groupby("date")[["long", "short", "all"]].mean()
    return means[sub.groupby("date").size() >= 4 * study.BOOK]


def book_gross(frame: pd.DataFrame, score: str, target: str) -> pd.Series:
    legs = book_legs(frame, score, target)
    return legs["long"] - legs["short"]


def leg_members(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    """The (date, symbol, side) rows of each session's book."""
    sub = frame[["date", "symbol", score, target]].dropna()
    order = sub.groupby("date")[score].rank(method="first")
    count = sub.groupby("date")[score].transform("size")
    side = np.where(
        order > count - study.BOOK, "long", np.where(order <= study.BOOK, "short", "")
    )
    picked = sub.assign(side=side)
    picked = picked[(count >= 4 * study.BOOK) & (picked["side"] != "")]
    return picked[["date", "symbol", "side"]]


def profile(net: pd.Series) -> dict:
    """Worst and best session, deepest drawdown of the cumulative sum and positive
    months, per period, in bp of one side's capital."""
    out = {}
    parts = study.split(net.dropna())
    for name, part in zip(("discovery", "holdout"), parts, strict=True):
        bp = part * 1e4
        path = bp.cumsum()
        months = bp.groupby(bp.index.str.slice(0, 7)).sum()
        out[name] = {
            "median_bp": study.rounded(bp.median(), 2),
            "worst_bp": study.rounded(bp.min(), 1),
            "best_bp": study.rounded(bp.max(), 1),
            "max_drawdown_bp": study.rounded((path.cummax() - path).max(), 1),
            "positive_months": f"{int((months > 0).sum())}/{len(months)}",
        }
    return out


def auction_book(data: pd.DataFrame, target: str) -> dict:
    members = leg_members(data, "composite", target)
    sessions = members.groupby("date").ngroups
    frequent = (
        members.loc[members["date"] <= study.DISCOVERY_END, "symbol"]
        .value_counts()
        .head(10)
    )
    shares = (members.groupby(["side", "symbol"]).size() / sessions).round(3)
    net = book_gross(data, "composite", target) - 2 * COSTS[0]
    rest = data[~data["symbol"].isin(frequent.index)]
    return {
        "profile_net_0.089pct": profile(net),
        "top_names_share_of_sessions": {
            side: shares[side].sort_values(ascending=False).head(8).to_dict()
            for side in ("long", "short")
        },
        "discovery_most_frequent": frequent.index.tolist(),
        "without_them_net_0.089pct_bp": periods(
            book_gross(rest, "composite", target) - 2 * COSTS[0], 1e4
        ),
    }


def periods(series: pd.Series, scale: float = 1.0) -> dict:
    discovery, holdout = study.split(series)
    return {
        "discovery": study.summarize(discovery, scale),
        "holdout": study.summarize(holdout, scale),
    }


def beta_alpha(gross: pd.Series, market: pd.Series) -> dict:
    """OLS of the book's gross return on NIFTY's over the same minutes, per period;
    alpha in bp a session with its Newey-West t."""
    out = {}
    both = pd.concat([gross, market], axis=1, keys=["book", "nifty"]).dropna()
    for name, part in zip(("discovery", "holdout"), study.split(both), strict=True):
        if len(part) < 30:
            out[name] = {"days": len(part)}
            continue
        x, y = part["nifty"].to_numpy(), part["book"].to_numpy()
        beta = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
        alpha = y - beta * x
        out[name] = {
            "days": len(part),
            "beta": study.rounded(beta, 3),
            "alpha_bp": study.rounded(alpha.mean() * 1e4, 2),
            "alpha_t_nw": study.rounded(study.newey_west_t(alpha), 2),
        }
    return out


def evaluate(frame: pd.DataFrame, score: str, target: str, market: pd.Series) -> dict:
    ic = study.daily_ic(frame, score, target)
    legs = book_legs(frame, score, target)
    gross = legs["long"] - legs["short"]
    out = {
        "ic": periods(ic),
        "book_gross_bp": periods(gross, 1e4),
        "long_vs_all_bp": periods(legs["long"] - legs["all"], 1e4),
        "all_vs_short_bp": periods(legs["all"] - legs["short"], 1e4),
        "beta_alpha": beta_alpha(gross, market.reindex(gross.index)),
    }
    for cost in COSTS:
        out[f"book_net_{cost * 100:.3f}pct_bp"] = periods(gross - 2 * cost, 1e4)
    return out


def oriented_signs(frame: pd.DataFrame, scores: list[str], target: str) -> dict:
    """Each score's sign: that of its discovery IC on ``target``."""
    signs = {}
    for score in scores:
        discovery, _ = study.split(study.daily_ic(frame, score, target))
        signs[score] = float(np.sign(discovery.mean()))
    return signs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    direction = json.loads(args.direction.read_text("utf-8"))
    kept = direction["kept_signs"]

    study.log("loading stock-days")
    data = study.load_stocks(study.FEATURES_DIR)
    indices = study.load_indices(study.FEATURES_DIR)
    study.add_targets(data, indices)
    study.add_features(data)
    add_tug_of_war(data)
    paths = [str(p) for p in sorted(study.NORMALIZED.rglob("*.npz"))]
    with ProcessPoolExecutor(args.workers) as pool:
        spreads = pd.concat(
            pool.map(study.spread_session, paths, chunksize=8), ignore_index=True
        )
    data = data.merge(
        spreads[["date", "symbol", "cpiv", "parity_bp"]],
        on=["date", "symbol"],
        how="left",
    )
    data = data[data["fno"] == 1].copy()
    data["composite"] = study.composite(data, kept)

    study.log("minute prices at the entry minutes and the exit")
    tasks = [
        (str(p), p.stem.removeprefix("NSE_"))
        for p in sorted(BASIS.glob("NSE_*.csv"))
        if not p.stem.endswith(".fetched")
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        minutes = pd.concat(pool.map(minute_prices, tasks), ignore_index=True)
    bad = excluded_days(BASIS)
    flagged = [
        (k, d) in bad
        for k, d in zip(minutes["kite_symbol"], minutes["date"], strict=True)
    ]
    minutes = minutes[~np.array(flagged, dtype=bool)]
    nifty = minutes[minutes["kite_symbol"] == "NIFTY"].set_index("date")
    history = symbol_history(study.SYMBOLCHANGE.read_text("utf-8"))
    minutes = minutes.assign(
        symbol=[
            nse_symbol(k, date.fromisoformat(d), history.get(k, []))
            for k, d in zip(minutes["kite_symbol"], minutes["date"], strict=True)
        ]
    ).drop_duplicates(["symbol", "date"])
    keyed = minutes.set_index(["symbol", "date"])
    sessions = sorted(data["date"].unique())
    following = dict(zip(sessions[:-1], sessions[1:], strict=True))
    data["next_date"] = data["date"].map(following)
    at = keyed.reindex(list(zip(data["symbol"], data["next_date"], strict=True)))
    at.index = data.index
    next_open = study.neighbour(data, ["cm_open"], 1)["cm_open"]
    recent = data["next_date"] >= "2026-01-01"
    agree = (at["09:15"] / next_open - 1).abs()[recent].dropna()
    feature_dates = pd.Index(sessions[:-1], name="date")
    traded_dates = [following[d] for d in feature_dates]
    targets = {}
    market = {}
    for minute in ENTRIES:
        name = f"e{minute.replace(':', '')}"
        data[name] = at["exit"] / at[minute] - 1
        data[f"open_leg_{name}"] = at[minute] / at["09:15"] - 1
        targets[name] = minute
        nifty_return = nifty["exit"] / nifty[minute] - 1
        market[name] = pd.Series(
            nifty_return.reindex(traded_dates).to_numpy(), index=feature_dates
        )
    data["overnight"] = data["gap_next"]
    study.log(
        f"{int(at['exit'].notna().sum())} stock-days with minutes; in 2026 the 09:15 "
        f"open is within 0.05% of NSE's open on {float((agree <= 0.0005).mean()):.4f}"
    )

    primary = "e0916"
    tug = oriented_signs(data, ["ovn20", "intra20"], primary)
    data["gap_fade"] = -data["gap_next"]
    gap_sign = oriented_signs(data, ["gap_fade"], primary)["gap_fade"]
    data["gap_fade"] *= gap_sign
    for name, sign in tug.items():
        data[name] *= sign
    data["gap_mix"] = study.composite(data, {"composite": 1.0, "gap_fade": 1.0})
    data["tug_mix"] = study.composite(data, {"ovn20": 1.0, "intra20": 1.0})
    scores = ["composite", "gap_fade", "gap_mix", "ovn20", "intra20", "tug_mix"]

    report: dict = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "research",
        "protocol": __doc__,
        "direction_report": str(args.direction),
        "orientation": {"gap_fade": gap_sign, **tug},
        "rows": {
            "stock_days": int(len(data)),
            "with_minutes": int(at["exit"].notna().sum()),
            "open_agrees_0.05pct_2026": study.rounded(float((agree <= 0.0005).mean())),
        },
        "entries": {},
    }
    for name, minute in targets.items():
        study.log(f"entry {minute} -> {EXIT}")
        entry = {}
        for score in scores:
            if score in GAP_SCORES and minute == "09:15":
                continue
            entry[score] = evaluate(data, score, name, market[name])
            if minute != "09:15":
                leg = f"open_leg_{name}"
                entry[score]["open_leg"] = {
                    "ic": periods(study.daily_ic(data, score, leg)),
                    "book_gross_bp": periods(book_gross(data, score, leg), 1e4),
                }
            hold = entry[score]["book_net_0.089pct_bp"]["holdout"]
            study.log(
                f"  {score:9s} IC hold {entry[score]['ic']['holdout'].get('mean')} "
                f"net bp disc "
                f"{entry[score]['book_net_0.089pct_bp']['discovery'].get('mean')} "
                f"hold {hold.get('mean')} (t {hold.get('t_nw')})"
            )
        report["entries"][minute] = entry
    report["auction_book"] = auction_book(data, "e0915")
    report["overnight_composite"] = {
        "ic": periods(study.daily_ic(data, "composite", "overnight")),
        "book_gross_bp_long_low": periods(
            -book_gross(data, "composite", "overnight"), 1e4
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=1), "utf-8")
    study.log(f"report in {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
