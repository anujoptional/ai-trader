"""Black-76 option prices and implied volatility.

NSE's stock and index options are European. Priced off the forward F of the same
expiry (the futures settlement price, or the put-call-parity forward when there is no
future of that expiry), Black-76 needs no dividend assumption:

    call = DF * (F N(d1) - K N(d2)),   put = DF * (K N(-d2) - F N(-d1)),
    d1 = (ln(F / K) + s^2 T / 2) / (s sqrt(T)),   d2 = d1 - s sqrt(T),
    DF = exp(-r T),   T = calendar days to expiry / 365.

The rate is a flat 6.5%: for a one-month option a 2-point error in it moves the
implied volatility by about 0.2% of itself. Implied volatility is found by bisection
to machine precision and is NaN where the price is not above the discounted intrinsic
value or not below the discounted forward (call) / strike (put).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

RATE = 0.065
LOW, HIGH = 1e-4, 5.0  # volatility search interval (0.01% to 500%)


def black76(forward, strike, years, vol, call, rate: float = RATE) -> np.ndarray:
    forward, strike, years, vol, call = np.broadcast_arrays(
        np.asarray(forward, float),
        np.asarray(strike, float),
        np.asarray(years, float),
        np.asarray(vol, float),
        np.asarray(call, bool),
    )
    root = vol * np.sqrt(years)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(forward / strike) + 0.5 * root * root) / root
    d2 = d1 - root
    discount = np.exp(-rate * years)
    calls = discount * (forward * ndtr(d1) - strike * ndtr(d2))
    puts = discount * (strike * ndtr(-d2) - forward * ndtr(-d1))
    return np.where(call, calls, puts)


def implied_vol(
    price, forward, strike, years, call, rate: float = RATE, iterations: int = 100
) -> np.ndarray:
    price, forward, strike, years, call = np.broadcast_arrays(
        np.asarray(price, float),
        np.asarray(forward, float),
        np.asarray(strike, float),
        np.asarray(years, float),
        np.asarray(call, bool),
    )
    discount = np.exp(-rate * np.where(years > 0, years, 0.0))
    intrinsic = discount * np.where(
        call, np.maximum(forward - strike, 0.0), np.maximum(strike - forward, 0.0)
    )
    ceiling = discount * np.where(call, forward, strike)
    valid = (
        np.isfinite(price)
        & np.isfinite(forward)
        & np.isfinite(strike)
        & (years > 0)
        & (forward > 0)
        & (strike > 0)
        & (price > intrinsic * (1 + 1e-12) + 1e-12)
        & (price < ceiling)
    )
    low = np.full(price.shape, LOW)
    high = np.full(price.shape, HIGH)
    safe_years = np.where(valid, years, 1.0)
    safe_forward = np.where(valid, forward, 1.0)
    safe_strike = np.where(valid, strike, 1.0)
    valid &= black76(safe_forward, safe_strike, safe_years, HIGH, call, rate) >= price
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        above = (
            black76(safe_forward, safe_strike, safe_years, middle, call, rate) > price
        )
        high = np.where(above, middle, high)
        low = np.where(above, low, middle)
    return np.where(valid, 0.5 * (low + high), np.nan)
