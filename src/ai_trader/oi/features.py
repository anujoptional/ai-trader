"""Daily open-interest features from Kite's continuous near-month futures.

Kite's continuous series follows the near-month contract and switches to the next
contract after each monthly expiry, so OI and price changes across a switch compare
different contracts. Those changes are withheld (None) rather than misreported.
All features for session ``t`` use only data at or before ``t``'s close; callers use
them for decisions on the following session.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from datetime import date, timedelta

from ai_trader.broker import FuturesDay

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
