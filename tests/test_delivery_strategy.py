import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")
pytest.importorskip("scipy")

from scripts import research_delivery_strategy as delivery


def test_delivery_costs_match_published_rupee_example() -> None:
    buy, sell = delivery.side_costs("stock")
    rupees = 100_000 * (buy + sell) + delivery.DP_INR
    assert rupees == pytest.approx(237.82, abs=0.01)
    assert delivery.side_costs("metal_etf")[1] == pytest.approx(
        delivery.side_costs("equity_etf")[1] - 0.00001
    )


def flat_costs(assets: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    return np.full(assets, fraction), np.full(assets, fraction)


def test_signal_executes_next_close_and_switches_assets(monkeypatch) -> None:
    monkeypatch.setattr(delivery, "SLIPPAGE", 0.0)
    monkeypatch.setattr(delivery, "DP_INR", 0.0)
    monkeypatch.setattr(delivery, "CASH_RATE", 0.0)
    prices = np.array([[100.0, 50], [100, 50], [110, 50], [110, 55], [121, 55]])
    goals = {0: (np.array([1.0, 0]), False), 2: (np.array([0, 1.0]), False)}
    result = delivery.simulate(
        prices, np.ones(prices.shape, bool), *flat_costs(2, 0.0), np.zeros(2), goals, 0
    )
    assert result["returns"].tolist() == pytest.approx([0, 0, 0.1, 0, 0])


def test_costs_shrink_purchases_and_are_not_double_counted(monkeypatch) -> None:
    monkeypatch.setattr(delivery, "SLIPPAGE", 0.0)
    monkeypatch.setattr(delivery, "DP_INR", 0.0)
    monkeypatch.setattr(delivery, "CASH_RATE", 0.0)
    prices = np.array([[100.0], [100], [110]])
    goals = {0: (np.array([1.0]), False)}
    result = delivery.simulate(
        prices, np.ones(prices.shape, bool), *flat_costs(1, 0.01), np.zeros(1), goals, 0
    )
    assert result["returns"][1] == pytest.approx(-0.0099)
    assert result["returns"][2] == pytest.approx(0.1 * 0.99 / 0.9901, rel=1e-6)
    assert result["invested"][1] <= 1


def test_momentum_picks_strongest_stock_only_in_risk_on_regime() -> None:
    days = 220
    names = ["A", "B", *delivery.ROTATION, delivery.REGIME]
    prices = np.full((days, len(names)), 100.0)
    prices[:, 0] = np.linspace(100, 200, days)
    prices[:, 1] = np.linspace(200, 100, days)
    prices[:, -1] = np.linspace(100, 150, days)
    params = {"lookback": 63, "skip": 10, "count": 1, "every": 21, "regime": True}
    goals = delivery.build_goals("momentum", params, prices, names, ["A", "B"], 150)
    assert sorted(goals)[:2] == [150, 171]
    assert goals[150][0][:2].tolist() == [1.0, 0.0]
    prices[:, -1] = np.linspace(150, 100, days)
    goals = delivery.build_goals("momentum", params, prices, names, ["A", "B"], 150)
    assert goals[150][0].sum() == 0
