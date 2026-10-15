"""Per-session features from NSE's end-of-day files.

Everything here is published after the session (F&O bhavcopy and cash bhavcopy in the
evening, participant OI and the next day's ban list later), so a session's values are
usable from the next session on. ``spot`` is the stock's official cash close (EQ, else
BE) or the index's official close (NSE's all-index file), else the F&O file's
underlying price (UDiFF only).

Futures (stock and index): nearest expiry on or after the session (``near``), the next
two; settlement prices; open interest per expiry and in total with its change; basis
over spot; rollover share; contracts and value traded; build-up from the spot's move
and the change in total futures OI.

Options: open interest, contracts and value by calls and puts (all expiries and the
nearest), put-call ratios, max pain and the call/put OI walls of the nearest expiry
(distances from spot), and implied volatility (fno/iv.py) of traded options (contracts
> 0, close > 0) at their close: ATM IV per expiry from the out-of-the-money smile (puts
below the forward, calls above, both averaged at a strike equal to it) interpolated
linearly in strike at the forward from the nearest points on both sides within 10%;
the 25-delta-like skew IV(0.95 F) - IV(1.05 F) from points bracketing each; and a
30-day ATM IV interpolated in total variance between the expiries around 30 days.
The forward of an expiry is its futures settlement price, else the put-call-parity
forward at the strike where call and put (both traded) are closest.
"""

from __future__ import annotations

import csv
import io
import math
from datetime import date

import numpy as np

from ai_trader.fno.iv import RATE, implied_vol

PARTICIPANT_COLUMNS = (
    ("Future Index Long", "fut_idx_long"),
    ("Future Index Short", "fut_idx_short"),
    ("Future Stock Long", "fut_stk_long"),
    ("Future Stock Short", "fut_stk_short"),
    ("Option Index Call Long", "opt_idx_call_long"),
    ("Option Index Put Long", "opt_idx_put_long"),
    ("Option Index Call Short", "opt_idx_call_short"),
    ("Option Index Put Short", "opt_idx_put_short"),
    ("Option Stock Call Long", "opt_stk_call_long"),
    ("Option Stock Put Long", "opt_stk_put_long"),
    ("Option Stock Call Short", "opt_stk_call_short"),
    ("Option Stock Put Short", "opt_stk_put_short"),
    ("Total Long Contracts", "total_long"),
    ("Total Short Contracts", "total_short"),
)
PARTICIPANT_ROWS = {"Client": "client", "DII": "dii", "FII": "fii", "Pro": "pro"}
# F&O index symbol -> its name in NSE's all-index close file.
INDEX_FILE_NAMES = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "Nifty Midcap Select",
    "NIFTYNXT50": "Nifty Next 50",
}
INDEX_FIELDS = (
    ("Open Index Value", "open"),
    ("High Index Value", "high"),
    ("Low Index Value", "low"),
    ("Closing Index Value", "close"),
    ("Points Change", "change"),
    ("Change(%)", "change_pct"),
    ("Volume", "volume"),
    ("Turnover (Rs. Cr.)", "turnover_cr"),
    ("P/E", "pe"),
    ("P/B", "pb"),
    ("Div Yield", "div_yield"),
)
CM_FIELDS = (
    ("PREV_CLOSE", "cm_prev_close"),
    ("OPEN_PRICE", "cm_open"),
    ("HIGH_PRICE", "cm_high"),
    ("LOW_PRICE", "cm_low"),
    ("CLOSE_PRICE", "cm_close"),
    ("AVG_PRICE", "cm_vwap"),
    ("TTL_TRD_QNTY", "cm_volume"),
    ("TURNOVER_LACS", "cm_turnover_lakh"),
    ("NO_OF_TRADES", "cm_trades"),
    ("DELIV_QTY", "cm_deliv_qty"),
    ("DELIV_PER", "cm_deliv_pct"),
)
FUTURES_FIELDS = (
    "fut_expiry_near",
    "fut_dte_near",
    "fut_settle_near",
    "fut_settle_next",
    "fut_oi_near",
    "fut_oi_next",
    "fut_oi_far",
    "fut_oi_total",
    "fut_chg_oi_total",
    "fut_oi_change_pct",
    "fut_contracts_total",
    "fut_value_total",
    "fut_basis_pct",
    "fut_basis_ann_pct",
    "fut_rollover_pct",
    "fut_buildup",
)
OPTION_FIELDS = (
    "opt_oi_ce",
    "opt_oi_pe",
    "opt_chg_oi_ce",
    "opt_chg_oi_pe",
    "opt_pcr_oi",
    "opt_contracts_ce",
    "opt_contracts_pe",
    "opt_pcr_vol",
    "opt_value_total",
    "opt_expiry_near",
    "opt_pcr_oi_near",
    "opt_max_pain_near",
    "opt_max_pain_dist_pct",
    "opt_call_wall_near",
    "opt_call_wall_dist_pct",
    "opt_put_wall_near",
    "opt_put_wall_dist_pct",
    "iv_expiry_near",
    "iv_atm_near",
    "iv_expiry_next",
    "iv_atm_next",
    "iv_atm_30d",
    "iv_skew_near",
    "iv_term",
    "iv_points_near",
)
NAN = float("nan")


def _float(text: str) -> float:
    text = text.strip()
    try:
        return float(text) if text and text != "-" else NAN
    except ValueError:
        return NAN


def read_cm(text: str) -> dict[str, dict[str, object]]:
    """Every EQ (else BE) row of a ``sec_bhavdata_full`` file, by symbol."""
    reader = csv.reader(io.StringIO(text))
    header = [c.strip() for c in next(reader)]
    at = {name: i for i, name in enumerate(header)}
    rows: dict[str, dict[str, object]] = {}
    for raw in reader:
        if len(raw) < len(header):
            continue
        series, symbol = raw[at["SERIES"]].strip(), raw[at["SYMBOL"]].strip()
        if series not in ("EQ", "BE") or (symbol in rows and series != "EQ"):
            continue
        row: dict[str, object] = {"series": series}
        for column, name in CM_FIELDS:
            row[name] = _float(raw[at[column]]) if column in at else NAN
        rows[symbol] = row
    return rows


def read_participants(text: str) -> dict[str, float]:
    """``<participant>_<column>`` contracts from one participant OI/volume file."""
    lines = text.splitlines()[1:]
    rows = [r for r in csv.reader(io.StringIO("\n".join(lines))) if any(r)]
    header = [c.strip() for c in rows[0]]
    expected = ["Client Type"] + [name for name, _ in PARTICIPANT_COLUMNS]
    if header[: len(expected)] != expected:
        raise ValueError(f"unexpected participant columns {header}")
    values: dict[str, float] = {}
    for row in rows[1:]:
        who = PARTICIPANT_ROWS.get(row[0].strip())
        if who is None:
            continue
        for (_, name), cell in zip(PARTICIPANT_COLUMNS, row[1:], strict=False):
            values[f"{who}_{name}"] = float(cell.strip())
    return values


def read_mwpl(text: str) -> dict[str, dict[str, object]]:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    header = [c.strip() for c in rows[0]]
    at = {name: i for i, name in enumerate(header)}
    found: dict[str, dict[str, object]] = {}
    for row in rows[1:]:
        mwpl = _float(row[at["MWPL"]])
        oi = _float(row[at["Open Interest"]])
        equivalent = (
            _float(row[at["Future Equivalent Open Interest"]])
            if "Future Equivalent Open Interest" in at
            else NAN
        )
        found[row[at["NSE Symbol"]].strip()] = {
            "mwpl_oi_pct": oi / mwpl * 100 if mwpl > 0 else NAN,
            "mwpl_feoi_pct": equivalent / mwpl * 100 if mwpl > 0 else NAN,
            "mwpl_limit": row[at["Limit for Next Day"]].strip()
            if "Limit for Next Day" in at
            else "",
        }
    return found


def read_ban(text: str) -> set[str]:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    return {row[1].strip() for row in rows[1:]}


def read_indices(text: str) -> dict[str, dict[str, float]]:
    """Every index of an ``ind_close_all`` file by name ("-" or blank = NaN)."""
    reader = csv.reader(io.StringIO(text))
    header = [c.strip() for c in next(reader)]
    at = {name: i for i, name in enumerate(header)}
    found: dict[str, dict[str, float]] = {}
    for raw in reader:
        if not raw or not raw[0].strip():
            continue
        found[raw[0].strip()] = {
            name: _float(raw[at[column]]) if column in at else NAN
            for column, name in INDEX_FIELDS
        }
    return found


def _interpolate(points: list[tuple[float, float]], at: float) -> float:
    """Linear in strike between the nearest points on both sides; NaN otherwise."""
    below = [p for p in points if p[0] <= at]
    above = [p for p in points if p[0] >= at]
    if not below or not above:
        return NAN
    (k1, v1), (k2, v2) = max(below), min(above)
    if k1 == k2:
        return v1
    return v1 + (v2 - v1) * (at - k1) / (k2 - k1)


def _smile(strikes, calls, vols, forward) -> list[tuple[float, float]]:
    """Out-of-the-money IV by strike: puts below the forward, calls above, the mean
    of both at a strike equal to it."""
    by_strike: dict[float, list[float]] = {}
    for k, is_call, v in zip(strikes, calls, vols, strict=True):
        if not math.isfinite(v):
            continue
        if (k < forward and not is_call) or (k > forward and is_call) or k == forward:
            by_strike.setdefault(float(k), []).append(float(v))
    return sorted((k, sum(v) / len(v)) for k, v in by_strike.items())


def _parity_forward(strikes, calls, prices, years) -> float:
    calls_at = {k: p for k, c, p in zip(strikes, calls, prices, strict=True) if c}
    puts_at = {k: p for k, c, p in zip(strikes, calls, prices, strict=True) if not c}
    common = [k for k in calls_at if k in puts_at]
    if not common:
        return NAN
    k = min(common, key=lambda s: (abs(calls_at[s] - puts_at[s]), s))
    return float(k + (calls_at[k] - puts_at[k]) * math.exp(RATE * years))


def _max_pain(strikes, ce_oi, pe_oi) -> float:
    if ce_oi.sum() + pe_oi.sum() <= 0:
        return NAN
    payout = (
        np.maximum(strikes[:, None] - strikes[None, :], 0) @ ce_oi
        + np.maximum(strikes[None, :] - strikes[:, None], 0) @ pe_oi
    )
    return float(strikes[int(np.argmin(payout))])


def _pct(value: float, base: float) -> float:
    return (value / base - 1) * 100 if base and math.isfinite(base) else NAN


def underlying_features(
    table: dict[str, np.ndarray], rows: np.ndarray, day: date, spot: float
) -> dict[str, object]:
    """Futures and options features of one underlying (``rows`` index ``table``)."""
    out: dict[str, object] = {name: NAN for name in FUTURES_FIELDS + OPTION_FIELDS}
    out["fut_buildup"] = ""
    session = np.datetime64(day, "D")
    kind = table["kind"][rows]

    futures = rows[kind == "F"]
    settles: dict[np.datetime64, float] = {}
    if len(futures):
        order = futures[np.argsort(table["expiry"][futures], kind="stable")]
        expiries = table["expiry"][order]
        if len(np.unique(expiries)) != len(expiries):
            raise ValueError("two futures of one expiry")
        oi = table["oi"][order].astype(float)
        settle = table["settle"][order]
        settles = {e: float(s) for e, s in zip(expiries, settle, strict=True)}
        total, change = float(oi.sum()), float(table["chg_oi"][order].sum())
        dte = int((expiries[0] - session).astype(int))
        out.update(
            fut_expiry_near=str(expiries[0]),
            fut_dte_near=dte,
            fut_settle_near=float(settle[0]),
            fut_settle_next=float(settle[1]) if len(order) > 1 else NAN,
            fut_oi_near=oi[0],
            fut_oi_next=oi[1] if len(order) > 1 else NAN,
            fut_oi_far=oi[2] if len(order) > 2 else NAN,
            fut_oi_total=total,
            fut_chg_oi_total=change,
            fut_oi_change_pct=change / (total - change) * 100
            if total - change > 0
            else NAN,
            fut_contracts_total=float(table["contracts"][order].sum()),
            fut_value_total=float(table["value"][order].sum()),
            fut_basis_pct=_pct(float(settle[0]), spot),
            fut_rollover_pct=float(oi[1:].sum()) / total * 100 if total > 0 else NAN,
        )
        if math.isfinite(out["fut_basis_pct"]):
            out["fut_basis_ann_pct"] = out["fut_basis_pct"] * 365 / max(dte, 1)

    options = rows[kind == "O"]
    if not len(options):
        return out
    call = table["option"][options] == "CE"
    oi = table["oi"][options].astype(float)
    contracts = table["contracts"][options].astype(float)
    out.update(
        opt_oi_ce=float(oi[call].sum()),
        opt_oi_pe=float(oi[~call].sum()),
        opt_chg_oi_ce=float(table["chg_oi"][options][call].sum()),
        opt_chg_oi_pe=float(table["chg_oi"][options][~call].sum()),
        opt_contracts_ce=float(contracts[call].sum()),
        opt_contracts_pe=float(contracts[~call].sum()),
        opt_value_total=float(table["value"][options].sum()),
    )
    out["opt_pcr_oi"] = out["opt_oi_pe"] / out["opt_oi_ce"] if out["opt_oi_ce"] else NAN
    out["opt_pcr_vol"] = (
        out["opt_contracts_pe"] / out["opt_contracts_ce"]
        if out["opt_contracts_ce"]
        else NAN
    )
    option_expiry = table["expiry"][options]
    near = option_expiry.min()
    at_near = option_expiry == near
    strikes = table["strike"][options][at_near]
    grid = np.unique(strikes)
    ce_oi = np.zeros(len(grid))
    pe_oi = np.zeros(len(grid))
    where = np.searchsorted(grid, strikes)
    np.add.at(ce_oi, where[call[at_near]], oi[at_near][call[at_near]])
    np.add.at(pe_oi, where[~call[at_near]], oi[at_near][~call[at_near]])
    pain = _max_pain(grid, ce_oi, pe_oi)
    out.update(
        opt_expiry_near=str(near),
        opt_pcr_oi_near=pe_oi.sum() / ce_oi.sum() if ce_oi.sum() else NAN,
        opt_max_pain_near=pain,
        opt_max_pain_dist_pct=_pct(pain, spot),
    )
    if ce_oi.max() > 0:
        wall = float(grid[int(np.argmax(ce_oi))])
        out.update(opt_call_wall_near=wall, opt_call_wall_dist_pct=_pct(wall, spot))
    if pe_oi.max() > 0:
        wall = float(grid[int(np.argmax(pe_oi))])
        out.update(opt_put_wall_near=wall, opt_put_wall_dist_pct=_pct(wall, spot))

    traded = options[(contracts > 0) & (table["close"][options] > 0)]
    if not len(traded):
        return out
    expiries = sorted(e for e in np.unique(table["expiry"][traded]) if e > session)
    atm: list[tuple[float, float]] = []
    for position, e in enumerate(expiries):
        group = traded[table["expiry"][traded] == e]
        years = float((e - session).astype(int)) / 365
        k = table["strike"][group]
        is_call = table["option"][group] == "CE"
        price = table["close"][group]
        forward = settles.get(e, NAN)
        if not (math.isfinite(forward) and forward > 0):
            forward = _parity_forward(k, is_call, price, years)
        if not math.isfinite(forward):
            continue
        vols = implied_vol(price, forward, k, years, is_call)
        smile = _smile(k, is_call, vols, forward)
        near_points = [p for p in smile if abs(p[0] / forward - 1) <= 0.10]
        level = _interpolate(near_points, forward)
        if position == 0:
            skew = _interpolate(smile, 0.95 * forward) - _interpolate(
                smile, 1.05 * forward
            )
            out.update(
                iv_expiry_near=str(e),
                iv_atm_near=level,
                iv_skew_near=skew,
                iv_points_near=len(smile),
            )
        elif position == 1:
            out.update(iv_expiry_next=str(e), iv_atm_next=level)
        if math.isfinite(level):
            atm.append((years, level))
    if math.isfinite(out["iv_atm_near"]) and math.isfinite(out["iv_atm_next"]):
        out["iv_term"] = out["iv_atm_next"] - out["iv_atm_near"]
    target = 30 / 365
    shorter = [p for p in atm if p[0] <= target]
    longer = [p for p in atm if p[0] >= target]
    if shorter and longer:
        (t1, v1), (t2, v2) = max(shorter), min(longer)
        if t1 == t2:
            out["iv_atm_30d"] = v1
        else:
            w = v1 * v1 * t1 + (v2 * v2 * t2 - v1 * v1 * t1) * (target - t1) / (t2 - t1)
            out["iv_atm_30d"] = math.sqrt(w / target) if w > 0 else NAN
    return out


def buildup(spot_change: float, oi_change: float) -> str:
    if not (math.isfinite(spot_change) and math.isfinite(oi_change)):
        return ""
    if spot_change == 0 or oi_change == 0:
        return "neutral"
    if spot_change > 0:
        return "long_buildup" if oi_change > 0 else "short_covering"
    return "short_buildup" if oi_change > 0 else "long_unwinding"


def session_features(
    day: date,
    table: dict[str, np.ndarray] | None,
    cm: dict[str, dict[str, object]],
    mwpl: dict[str, dict[str, object]],
    ban: set[str] | None,
    indices: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """(stock rows, index rows) of one session."""
    stock_rows: dict[str, dict[str, object]] = {}
    for symbol, row in cm.items():
        stock_rows[symbol] = {"date": day.isoformat(), "symbol": symbol, **row}
    index_rows = []
    if table is not None:
        order = np.lexsort((table["symbol"],))
        symbols = table["symbol"][order]
        starts = np.flatnonzero(np.r_[True, symbols[1:] != symbols[:-1]])
        for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True):
            rows = order[start:stop]
            symbol = str(symbols[start])
            is_index = table["underlying"][rows[0]] == "I"
            if is_index:
                official = (indices or {}).get(INDEX_FILE_NAMES.get(symbol, ""), {})
                spot = official.get("close", NAN)
                if not (math.isfinite(spot) and spot > 0):
                    reported = table["underlying_price"][rows]
                    finite = reported[np.isfinite(reported) & (reported > 0)]
                    spot = float(finite[0]) if len(finite) else NAN
                features = underlying_features(table, rows, day, spot)
                index_rows.append(
                    {
                        "date": day.isoformat(),
                        "symbol": symbol,
                        "spot": spot,
                        **features,
                    }
                )
                continue
            row = stock_rows.setdefault(
                symbol, {"date": day.isoformat(), "symbol": symbol, "series": ""}
            )
            spot = row.get("cm_close", NAN)
            if not (isinstance(spot, float) and math.isfinite(spot)):
                reported = table["underlying_price"][rows]
                finite = reported[np.isfinite(reported) & (reported > 0)]
                spot = float(finite[0]) if len(finite) else NAN
            features = underlying_features(table, rows, day, float(spot))
            previous = row.get("cm_prev_close", NAN)
            close = row.get("cm_close", NAN)
            features["fut_buildup"] = buildup(
                float(close) - float(previous)
                if isinstance(close, float) and isinstance(previous, float)
                else NAN,
                float(features["fut_chg_oi_total"]),
            )
            row.update(fno=1, **features)
    for symbol, row in stock_rows.items():
        row.setdefault("fno", 0)
        row["in_ban"] = "" if ban is None else int(symbol in ban)
        limits = mwpl.get(symbol, {})
        row["mwpl_oi_pct"] = limits.get("mwpl_oi_pct", NAN)
        row["mwpl_feoi_pct"] = limits.get("mwpl_feoi_pct", NAN)
        row["mwpl_limit"] = limits.get("mwpl_limit", "")
    return list(stock_rows.values()), index_rows
