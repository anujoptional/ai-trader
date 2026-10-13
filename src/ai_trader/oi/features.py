"""Open-interest features of stock futures.

Research (``daily_oi_features``) uses Kite's continuous near-month series, which
switches to the next contract after each monthly expiry, so OI and price changes
across a switch compare different contracts. Those changes are withheld (None)
rather than misreported. All features for session ``t`` use only data at or before
``t``'s close; callers use them for decisions on the following session.

The paper session uses the near and next contracts' own day bars and live quotes
(``previous_session_futures``, ``live_futures``): their OI is summed, so positions
rolled from one month to the next are not read as positions opened or closed.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from ai_trader.broker import DepthSnapshot, FuturesDay

# NSE moved monthly equity-derivative expiry from the last Thursday to the last
# Tuesday for contracts expiring from 1 September 2025.
TUESDAY_EXPIRY_FROM = date(2025, 9, 1)
PRICE_DEADBAND = 0.002
OI_DEADBAND = 0.01


def monthly_expiries(sessions: Sequence[date]) -> list[date]:
    """Monthly expiry sessions: the last Thursday (Tuesday from Sep 2025) of each
    month, moved to the previous trading session when that day is a holiday.
    An expiry after the last known session keeps its calendar date."""
    trading = sorted(set(sessions))
    if not trading:
        return []
    known = set(trading)
    result = []
    year, month = trading[0].year, trading[0].month
    while (year, month) <= (trading[-1].year, trading[-1].month):
        last = date(year, month, calendar.monthrange(year, month)[1])
        weekday = calendar.TUESDAY if last >= TUESDAY_EXPIRY_FROM else calendar.THURSDAY
        day = last - timedelta(days=(last.weekday() - weekday) % 7)
        if day > trading[-1]:
            result.append(day)  # holidays after the data are unknown
        else:
            while day not in known and day >= trading[0]:
                day -= timedelta(days=1)
            if day in known:
                result.append(day)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def buildup(price_change: float | None, oi_change: float | None) -> str | None:
    if price_change is None or oi_change is None:
        return None
    if abs(price_change) < PRICE_DEADBAND or abs(oi_change) < OI_DEADBAND:
        return "neutral"
    if oi_change > 0:
        return "long_buildup" if price_change > 0 else "short_buildup"
    return "short_covering" if price_change > 0 else "long_unwinding"


def daily_oi_features(
    bars: Sequence[FuturesDay],
    spot_close: Mapping[date, float],
    expiries: Sequence[date],
) -> dict[date, dict[str, float | str | None]]:
    """Per-session OI features; changes spanning a contract switch are None."""
    ordered = sorted(bars, key=lambda bar: bar.session_date)
    expiry_list = sorted(expiries)
    result: dict[date, dict[str, float | str | None]] = {}
    for i, bar in enumerate(ordered):
        day = bar.session_date
        upcoming = [e for e in expiry_list if e >= day]
        expiry = upcoming[0] if upcoming else None

        def same_contract(j: int) -> bool:
            # No expiry strictly between bar j and today: one contract throughout.
            first = ordered[j].session_date
            return not any(first <= e < day for e in expiry_list)

        def change(j: int, field: str) -> float | None:
            if j < 0 or not same_contract(j):
                return None
            then = float(getattr(ordered[j], field))
            return float(getattr(bar, field)) / then - 1 if then > 0 else None

        price_1d, oi_1d = change(i - 1, "close"), change(i - 1, "open_interest")
        spot = spot_close.get(day)
        days_left = (expiry - day).days if expiry else None
        basis = float(bar.close) / spot - 1 if spot else None
        result[day] = {
            "fut_close": float(bar.close),
            "open_interest": float(bar.open_interest),
            "oi_change_1d": oi_1d,
            "oi_change_5d": change(i - 5, "open_interest"),
            "fut_return_1d": price_1d,
            "buildup_1d": buildup(price_1d, oi_1d),
            "basis_pct": basis * 100 if basis is not None else None,
            "annualized_basis_pct": basis * 100 * 365 / days_left
            if basis is not None and days_left and days_left >= 3
            else None,
            "calendar_days_to_expiry": float(days_left)
            if days_left is not None
            else None,
            "volume_to_oi": bar.volume / bar.open_interest
            if bar.open_interest
            else None,
        }
    return result


def live_oi_context(
    near_oi: int | None,
    next_oi: int | None,
    previous_total_oi: float | None,
    price_change: float | None,
) -> dict[str, float | str | None]:
    """Intraday OI context from live quotes of the near and next contracts."""
    total = (near_oi or 0) + (next_oi or 0) if near_oi or next_oi else None
    change = total / previous_total_oi - 1 if total and previous_total_oi else None
    return {
        "total_oi": float(total) if total else None,
        "oi_change_since_prev_close": change,
        "intraday_buildup": buildup(price_change, change),
    }


def _open_interest(
    bars: Sequence[FuturesDay], sessions: Sequence[date]
) -> dict[date, int | None]:
    """OI at each session's close. A contract with no trade in a session has no bar
    and an unchanged OI, so the last known value carries over."""
    known = {bar.session_date: bar.open_interest for bar in bars}
    result: dict[date, int | None] = {}
    value = None
    for day in sessions:
        value = known.get(day, value)
        result[day] = value
    return result


def previous_session_futures(
    bars: Mapping[str, Sequence[FuturesDay]],
    near: str,
    sessions: Sequence[date],
    since: date,
) -> dict[str, Any]:
    """Futures facts at the last session's close, from the day bars of the contracts
    trading now (``near`` first, then the next month), keyed by trading symbol.

    ``sessions`` are at least two trading days, oldest first, ending with the last
    session. OI is the contracts' total. Changes start no earlier than ``since``, the
    last expiry day: before its close the expiring month held positions these lack.
    """
    held = {name: _open_interest(series, sessions) for name, series in bars.items()}

    def total(day: date) -> int | None:
        values = [by_day[day] for by_day in held.values()]
        known = [value for value in values if value is not None]
        return sum(known) if len(known) == len(values) else None

    def change(back: int) -> float | None:
        if len(sessions) <= back or sessions[-1 - back] < since:
            return None
        before, now = total(sessions[-1 - back]), total(sessions[-1])
        return now / before - 1 if before and now is not None else None

    closes = {bar.session_date: float(bar.close) for bar in bars[near]}
    last, before = sessions[-1], sessions[-2]
    price = (
        closes[last] / closes[before] - 1
        if last in closes and before in closes
        else None
    )
    oi_change = change(1)
    return {
        "previous_oi": {name: by_day[last] for name, by_day in held.items()},
        "oi_change_prev_session": oi_change,
        "futures_change_prev_session": price,
        "buildup_prev_session": buildup(price, oi_change),
        "oi_change_5_sessions": change(5),
    }


def _mid(quote: DepthSnapshot) -> Decimal:
    if quote.bids and quote.asks:
        return (quote.bids[0].price + quote.asks[0].price) / 2
    return quote.last_price


def live_futures(
    previous_oi: Mapping[str, int | None],
    quotes: Mapping[str, DepthSnapshot],
    near: str,
    stock: DepthSnapshot | None,
) -> dict[str, float | str | None]:
    """Futures facts now, from one set of live quotes: the OI of the contracts with a
    known last close against that close, the near month's move since its previous
    close, and its premium over the stock (quote mids, which are simultaneous)."""
    contract = quotes.get(near)
    if contract is None:
        return {}
    now = before = 0
    for name, previous in previous_oi.items():
        quote = quotes.get(name)
        if previous and quote is not None and quote.open_interest is not None:
            now += quote.open_interest
            before += previous
    change = now / before - 1 if before else None
    price = (
        float(contract.last_price / contract.previous_close) - 1
        if contract.previous_close
        else None
    )
    return {
        "premium_now": float(_mid(contract) / _mid(stock)) - 1 if stock else None,
        "oi_change_today": change,
        "buildup_today": buildup(price, change),
    }
