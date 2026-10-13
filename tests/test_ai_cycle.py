from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np

from ai_trader.ai.cycle import (
    StockInputs,
    build_request,
    complete,
    daily_array,
    score_universe,
    session_arrays,
)
from ai_trader.ai.decision import request_json
from ai_trader.broker import DepthLevel, DepthSnapshot, Instrument, OHLCVCandle
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.scanner.movement import DECISIONS

DAY = datetime(2026, 10, 12, 9, 15, tzinfo=INDIA_TIMEZONE)


def synthetic_sessions(days: int, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    sessions, price = [], 500.0
    for _ in range(days):
        close = price * np.exp(np.cumsum(rng.normal(0, 0.0012, 345)))
        open_ = np.concatenate(([price], close[:-1]))
        high = np.maximum(open_, close) * 1.0004
        low = np.minimum(open_, close) * 0.9996
        volume = rng.integers(1_000, 20_000, 345).astype(float)
        sessions.append(np.column_stack((open_, high, low, close, volume)).round(2))
        price = float(close[-1])
    return sessions


def candles(session: np.ndarray, start: datetime, upto: int) -> list[OHLCVCandle]:
    return [
        OHLCVCandle(
            start + timedelta(minutes=i),
            *(Decimal(str(round(x, 2))) for x in row[:4]),
            int(row[4]),
        )
        for i, row in enumerate(session[: upto + 1])
    ]


def test_session_arrays_pad_the_future_and_complete_filters_partials() -> None:
    session = synthetic_sessions(1)[0]
    sessions = session_arrays(candles(session, DAY, 100))
    today = sessions[DAY.date()]
    assert np.isfinite(today[:101, :4]).all() and np.isnan(today[101:]).all()
    assert complete(sessions) == []
    full = session_arrays(candles(session, DAY - timedelta(days=1), 344))
    assert len(complete(full)) == 1
    assert daily_array([]).shape == (0, 5)


def test_scoring_and_request_assembly() -> None:
    sessions = synthetic_sessions(26)
    daily = np.asarray(
        [
            [s[0, 0], s[:, 1].max(), s[:, 2].min(), s[-1, 3], s[:, 4].sum()]
            for s in sessions
        ]
    )
    minute = int(DECISIONS[5])
    today = sessions[-1].copy()
    today[minute + 1 :] = np.nan
    stocks = {
        "AAA": StockInputs(today, sessions[:-1], daily[:-1]),
        "BBB": StockInputs(
            today * np.array([1, 1, 1, 1, 2]), sessions[:-1], daily[:-1]
        ),
        "CCC": StockInputs(today, sessions[:5], daily[:5]),  # too little history
    }
    nifty = sessions[0].copy()
    vix = np.full((345, 5), 13.0)
    ranked = score_universe(lambda x: x[:, 2] / 100, stocks, nifty, 500.0, vix, minute)
    assert [r[0] for r in ranked] == ["BBB", "AAA"]  # higher relative volume first
    as_of = DAY + timedelta(minutes=minute + 1)
    snapshot = DepthSnapshot(
        Instrument("NSE", "BBB"),
        as_of,
        Decimal("100"),
        as_of,
        10,
        300,
        100,
        (DepthLevel(Decimal("99.95"), 500, 2),),
        (DepthLevel(Decimal("100.05"), 400, 2),),
        Decimal("99"),
    )
    request = build_request(
        as_of,
        ranked,
        top_k=5,
        depth={"BBB": snapshot},
        oi={"AAA": {"buildup_1d": "short_buildup", "oi_change_1d": 0.05}},
        news={"AAA": {"since_prev_close": 1, "items": [{"subject": "Results"}]}},
        exclude={"ZZZ"},
    )
    by_symbol = {c.symbol: c for c in request.candidates}
    assert by_symbol["BBB"].orderbook["imbalance_total"] == 0.5
    assert by_symbol["AAA"].open_interest["oi_change_1d_pct"] == 5.0
    assert by_symbol["AAA"].announcements == ({"subject": "Results"},)
    assert request.market["india_vix"] == 13.0
    assert "NaN" not in request_json(request)
