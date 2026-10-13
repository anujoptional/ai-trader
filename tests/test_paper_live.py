from datetime import date, datetime
from decimal import Decimal

import numpy as np

from ai_trader.broker import Instrument, MarketTick
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.paper.live import LiveBars
from ai_trader.scanner.upside import fill_quiet_minutes

DAY = date(2026, 10, 13)


def tick(symbol: str, hh: int, mm: int, ss: int, price: str, volume: int | None):
    stamp = datetime(2026, 10, 13, hh, mm, ss, tzinfo=INDIA_TIMEZONE)
    return MarketTick(Instrument("NSE", symbol), Decimal(price), stamp, volume)


def test_live_bars_from_ticks_and_seeded_history() -> None:
    live = LiveBars(DAY, ["ABC"])
    for t in (
        tick("ABC", 9, 15, 1, "100", 1000),  # pre-open volume lands in 09:15
        tick("ABC", 9, 15, 30, "101", 1200),
        tick("ABC", 9, 15, 59, "99.5", 1300),
        tick("ABC", 9, 17, 2, "100.5", 1350),  # 09:16 had no trade
        tick("ABC", 9, 14, 59, "98", 900),  # before the open: ignored
        tick("XYZ", 9, 15, 1, "5", 10),  # not subscribed: ignored
    ):
        live.on_tick(t)
    rows = live.bars("ABC", 2)
    assert rows[0].tolist() == [100, 101, 99.5, 99.5, 1300]
    assert np.isnan(rows[1]).all()
    assert rows[2].tolist() == [100.5, 100.5, 100.5, 100.5, 50]
    assert not live.complete("ABC", 2) and live.complete("ABC", 0)
    assert live.ticks == 6 and live.silent_seconds() < 5

    seeded = np.array([[100, 101, 99, 100, 700], [100, 100, 100, 100, 0]], float)
    live.seed({"ABC": seeded}, 1)
    live.on_tick(tick("ABC", 9, 16, 10, "120", 1400))  # seeded minute: price kept
    live.on_tick(tick("ABC", 9, 17, 40, "101", 1500))
    rows = live.bars("ABC", 2)
    assert rows[:2, :4].tolist() == seeded[:, :4].tolist()
    assert rows[2].tolist() == [100.5, 101, 100.5, 101, 150]


def test_after_the_open_a_first_tick_only_sets_the_volume_baseline() -> None:
    live = LiveBars(DAY, ["ABC"], from_open=False)
    live.on_tick(tick("ABC", 11, 0, 1, "100", 50_000))
    live.on_tick(tick("ABC", 11, 0, 30, "100.2", 50_400))
    assert live.bars("ABC", 105)[105, 4] == 400


def test_no_trade_minutes_are_filled_up_to_a_liquidity_limit() -> None:
    rows = np.full((21, 5), np.nan)
    rows[2:] = [10, 11, 9, 10.5, 100]
    rows[5] = np.nan
    filled = fill_quiet_minutes(rows, 20, max_share=0.2)
    assert filled[0].tolist() == [10, 10, 10, 10, 0]  # before the first trade
    assert filled[5].tolist() == [10.5, 10.5, 10.5, 10.5, 0]
    assert fill_quiet_minutes(rows, 20, max_share=0.1) is None  # 3 of 21 missing
    assert fill_quiet_minutes(np.full((5, 5), np.nan), 4) is None
