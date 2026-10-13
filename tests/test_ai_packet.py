import numpy as np
import pytest

from ai_trader.ai.packet import daily_now, market_v3, stock_v3

CLOSES = np.arange(1.0, 301.0)  # 300 completed sessions, oldest first
HIGHS = CLOSES + 1


def test_daily_values_are_measured_at_the_current_price() -> None:
    now = daily_now(CLOSES, 330.0, HIGHS, 331.0, {"change_20d": 0.05})
    assert now["change_today"] == pytest.approx(330 / 300 - 1)
    assert now["change_5d"] == pytest.approx(330 / 296 - 1)  # since 5 sessions ago
    assert now["change_60d"] == pytest.approx(330 / 241 - 1)
    assert now["vs_50d_avg"] == pytest.approx(330 / np.mean(CLOSES[-50:]) - 1)
    assert now["vs_200d_avg"] == pytest.approx(330 / np.mean(CLOSES[-200:]) - 1)
    assert now["vs_250d_high"] == pytest.approx(330 / 331 - 1)  # today's high
    assert now["vs_nifty_20d"] == pytest.approx(330 / 281 - 1 - 0.05)
    short = daily_now(CLOSES[-30:], 330.0, HIGHS[-30:], 331.0)
    assert short["change_20d"] is not None and short["change_60d"] is None
    assert short["vs_50d_avg"] is None and short["vs_250d_high"] is None
    assert "vs_nifty_20d" not in short


def test_v3_stock_takes_daily_values_now_not_the_scanners_last_close_ones() -> None:
    features = {"ret5d": 0.5, "dist_sma50": 0.5, "slot": 100.0, "atr20": 0.02}
    now = daily_now(CLOSES, 330.0, HIGHS, 331.0, {"change_20d": 0.05})
    stock = stock_v3(
        "ABC",
        features,
        330.0,
        now=now,
        oi={"premium_now": 0.0013, "buildup_prev_session": "short_covering"},
    )
    assert stock["today"]["change_today_pct"] == 10.0
    assert stock["daily"]["change_5d_pct"] == round((330 / 296 - 1) * 100, 3)
    assert stock["daily"]["vs_50d_avg_pct"] == round(now["vs_50d_avg"] * 100, 3)
    assert stock["daily"]["avg_daily_range_20d_pct"] == 2.0
    assert stock["futures_oi"]["premium_now_pct"] == 0.13
    assert stock["futures_oi"]["buildup_prev_session"] == "short_covering"
    market = market_v3({"nifty_day": 0.01}, daily_now(CLOSES, 315.0))
    assert market["nifty_change_today_pct"] == 1.0
    assert market["nifty_change_20d_pct"] == round((315 / 281 - 1) * 100, 3)
    assert market_v3({"nifty_day": 0.01})["nifty_change_20d_pct"] is None
