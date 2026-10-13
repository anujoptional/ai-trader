from decimal import Decimal

import numpy as np
import pytest

pytest.importorskip("torch")  # the research script imports the training stack

from scripts import research_delivery_pipeline as pipe  # noqa: E402


def flat_session(price: float = 100.0) -> np.ndarray:
    """A complete 09:15-14:59 session at one price, volume 1,000 a minute."""
    prices = np.full((pipe.LAST + 1, 5), price)
    prices[:, 4] = 1000
    return prices


def test_target_counts_only_the_120_minutes_after_the_buy() -> None:
    decision = int(pipe.DECISIONS[0])
    prices = flat_session()
    prices[decision + 120, 1] = 101.0  # last minute of the window: +1% exactly
    entry, first = pipe.target_outcomes(prices)
    assert entry[0] == 10_000 and first[0] == 120
    late = flat_session()
    late[decision + 121, 1] = 101.0  # one minute too late
    assert pipe.target_outcomes(late)[1][0] == pipe.NONE
    short = flat_session()
    short[decision + 5, 1] = 100.99  # a paisa short of +1%
    assert pipe.target_outcomes(short)[1][0] == pipe.NONE


def test_target_uses_the_next_minute_open_as_the_buy_price() -> None:
    decision = int(pipe.DECISIONS[0])
    prices = flat_session(99.0)
    prices[decision, 3] = 98.0  # the decision close is not the buy price
    prices[decision + 3, 1] = 99.99  # +1% of the 99.00 buy
    entry, first = pipe.target_outcomes(prices)
    assert entry[0] == 9_900 and first[0] == 3


def ohlc(rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int64)


def test_first_exit_at_the_target_or_the_stop() -> None:
    quiet = [(10_000, 10_050, 9_950, 10_000)] * 600
    up = ohlc([*quiet, (10_000, 10_100, 9_990, 10_090)])
    assert pipe.first_exit(up, 0, 10_000) == (600, 101.0, "tp")
    down = ohlc([*quiet, (10_000, 10_010, 9_900, 9_950)])
    assert pipe.first_exit(down, 0, 10_000) == (600, 99.0, "sl")


def test_a_minute_reaching_both_levels_counts_as_the_stop() -> None:
    both = ohlc([(10_000, 10_000, 10_000, 10_000), (10_000, 10_150, 9_850, 10_000)])
    assert pipe.first_exit(both, 0, 10_000) == (1, 99.0, "sl_same_minute")


def test_opening_beyond_a_level_fills_at_the_open() -> None:
    gap_down = ohlc([(10_000, 10_020, 9_980, 10_000), (9_800, 9_850, 9_700, 9_800)])
    assert pipe.first_exit(gap_down, 0, 10_000) == (1, 98.0, "sl_gap")
    gap_up = ohlc([(10_000, 10_020, 9_980, 10_000), (10_250, 10_300, 10_200, 10_260)])
    assert pipe.first_exit(gap_up, 0, 10_000) == (1, 102.5, "tp_gap")


def test_the_buy_minute_itself_can_exit_and_an_unfinished_trade_stays_open() -> None:
    same = ohlc([(10_000, 10_100, 9_990, 10_050)])
    assert pipe.first_exit(same, 0, 10_000) == (0, 101.0, "tp")
    never = ohlc([(10_000, 10_090, 9_910, 10_020)] * 3)
    assert pipe.first_exit(never, 0, 10_000) == (2, 100.2, "open")


def test_same_session_sales_pay_intraday_charges_and_overnight_ones_delivery() -> None:
    intraday = pipe.charges(200_000, 202_000, same_session=True)
    expected = float(pipe.ZERODHA_INTRADAY_EQUITY.round_trip(Decimal("200000")).total)
    assert intraday == pytest.approx(expected) and 115 < intraday < 121
    delivery = pipe.charges(200_000, 202_000, same_session=False)
    stt = 0.001 * 200_000 + 0.001 * 202_000
    assert delivery == pytest.approx(
        pipe.BUY_SIDE * 200_000 + pipe.SELL_SIDE * 202_000 + pipe.DP_INR
    )
    assert stt < delivery < stt + 0.00015 * 200_000 + 50


def trade(day: int, slot: int, stock: int, entry: tuple, exit_: tuple) -> pipe.Trade:
    return pipe.Trade(
        day, slot, stock, "S", entry[0], entry[1], exit_[0], exit_[1],
        100.0, 101.0, "tp", exit_[0] - entry[0], 2000, 100.0, 1900.0, 0.0095, True,
    )  # fmt: skip


def test_execute_buys_once_per_stock_per_day_and_never_while_holding() -> None:
    outcomes = {
        (0, 0, 7): trade(0, 0, 7, (100, 25), (101, 30)),  # held into the next day
        (0, 5, 7): trade(0, 5, 7, (100, 50), (100, 60)),  # same stock, same day
        (1, 0, 7): trade(1, 0, 7, (101, 25), (101, 40)),  # still held
        (1, 1, 7): trade(1, 1, 7, (101, 30), (101, 90)),  # exit minute: still held
        (1, 4, 7): trade(1, 4, 7, (101, 45), (101, 95)),  # free again
        (0, 1, 9): trade(0, 1, 9, (100, 30), (100, 31)),  # another stock
    }
    taken = pipe.execute(list(outcomes), outcomes)
    assert [(t.day, t.slot, t.stock) for t in taken] == [
        (0, 0, 7),
        (0, 1, 9),
        (1, 4, 7),
    ]


def test_holding_is_unreliable_across_missing_or_flagged_sessions() -> None:
    calendar = np.array([10, 11, 12, 13])
    assert pipe.complete_holding(10, 12, 2, calendar, [])
    assert not pipe.complete_holding(10, 12, 1, calendar, [])  # a session missing
    assert not pipe.complete_holding(10, 12, 2, calendar, [12])
    assert pipe.complete_holding(10, 12, 2, calendar, [10, 13])  # outside the hold


def test_rsi_uses_only_the_row_and_earlier_closes() -> None:
    closes = np.array([10, 11, 10.5, 11.5, 12, 11, 12.5, 13, 12, 13.5, 14, 13.5, 14.5])
    rsi = pipe.wilder_rsi(closes, period=4)
    assert np.isnan(rsi[:4]).all() and np.isfinite(rsi[4:]).all()
    change = np.diff(closes[:5])
    gain, loss = change.clip(min=0).mean(), (-change).clip(min=0).mean()
    assert rsi[4] == pytest.approx(100 - 100 / (1 + gain / loss))
    changed = closes.copy()
    changed[8:] = 1.0
    assert np.allclose(pipe.wilder_rsi(changed, period=4)[:8], rsi[:8], equal_nan=True)
    assert pipe.wilder_rsi(np.arange(1.0, 30.0), period=14)[-1] == 100.0


def test_daily_features_never_read_the_decision_day_row() -> None:
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 300))
    daily = np.column_stack(
        (closes, closes + 1, closes - 1, closes, rng.integers(1e5, 1e6, 300))
    ).astype(float)
    position = 260  # row 260 is the decision day; rows before it are history
    part = pipe.session_features(flat_session(float(closes[259])))
    day_rv = list(rng.uniform(0.01, 0.02, 25))
    volumes = [np.full(pipe.SLOTS, 5e4)] * 25
    rsi = pipe.wilder_rsi(daily[:, 3])
    before = pipe.daily_columns(daily, position, rsi, day_rv, volumes, part)
    later = daily.copy()
    later[position:] *= 3  # the decision day and everything after
    later_rsi = pipe.wilder_rsi(later[:, 3])
    after = pipe.daily_columns(later, position, later_rsi, day_rv, volumes, part)
    assert np.allclose(before, after, equal_nan=True)
    column = pipe.STOCK_FEATURES.index
    assert before[0, column("ret5d")] == pytest.approx(closes[259] / closes[254] - 1)
    assert before[0, column("dist_sma50")] == pytest.approx(
        closes[259] / closes[210:260].mean() - 1
    )


def test_session_features_use_past_minutes_only() -> None:
    prices = flat_session()
    decision = int(pipe.DECISIONS[10])
    prices[decision + 1 :, :4] = 200.0  # everything after the decision close
    part = pipe.session_features(prices)
    assert part["ret15"][10] == 0.0 and part["vwap_dev"][10] == pytest.approx(0.0)
    assert np.isnan(part["ret60"][0]) and part["ret60"][12] == pytest.approx(1.0)
