from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from ai_trader.scanner import upside

GRID = upside.DECISIONS


def research_pipeline():
    pytest.importorskip("torch")  # the research script imports the training stack
    from scripts import research_delivery_pipeline

    return research_delivery_pipeline


def weekdays(start: date, count: int) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def session(rng: np.random.Generator, start: float) -> np.ndarray:
    """A complete 09:15-14:59 minute session: open/high/low/close/volume."""
    close = start * np.exp(np.cumsum(rng.normal(0, 0.001, upside.LAST + 1)))
    open_ = np.concatenate(([start], close[:-1]))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, len(close)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, len(close)))
    volume = rng.integers(100, 5000, len(close)).astype(float)
    return np.column_stack((open_, high, low, close, volume))


def daily_bars(rng: np.random.Generator, count: int, start: float) -> np.ndarray:
    close = start * np.exp(np.cumsum(rng.normal(0, 0.012, count)))
    open_ = close * (1 + rng.normal(0, 0.004, count))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, count))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, count))
    volume = rng.integers(10**5, 10**6, count).astype(float)
    return np.column_stack((open_, high, low, close, volume))


@pytest.mark.parametrize("auction", [False, True])
def test_live_features_equal_the_research_features(monkeypatch, auction) -> None:
    pipe = research_pipeline()
    from scripts import research_scanner_move as move

    rng = np.random.default_rng(7)
    days = weekdays(date(2025, 6, 2) if not auction else date(2025, 9, 1), 300)
    if auction:
        days = weekdays(date(2025, 7, 1), 300)
        assert days[-1] >= upside.AUCTION_FROM
    today, prior = days[-1], days[:-1]
    stock_daily = daily_bars(rng, len(prior), 500)
    nifty_daily = daily_bars(rng, len(prior), 24000)
    vix_daily = daily_bars(rng, len(prior), 13)
    history = [session(rng, 500) for _ in range(25)]
    today_prices = session(rng, float(stock_daily[-1, 3]) * 1.01)
    nifty_today = session(rng, float(nifty_daily[-1, 3]))
    vix_today = session(rng, float(vix_daily[-1, 3]))
    minutes = upside.session_minutes(today, closing_auction=True)
    assert minutes == (upside.AUCTION_MINUTES if auction else upside.SESSION_MINUTES)

    # research: stock rows exactly as scripts/research_delivery_pipeline builds them
    part = pipe.session_features(today_prices, GRID)
    parts = [pipe.session_features(s, GRID) for s in history]
    research_stock = pipe.daily_columns(
        stock_daily,
        len(stock_daily),
        pipe.wilder_rsi(stock_daily[:, 3]),
        [p["day_rv"] for p in parts],
        [p["cumvol"] for p in parts],
        part,
        len(GRID),
    )
    epoch = [pipe.epoch_day(d.isoformat()) for d in days]

    class Store:
        def __init__(self, _path) -> None:
            pass

        def verified_path(self, name: str) -> str:
            return name

    def read_daily(name: str):
        values = nifty_daily if "NIFTY" in name else vix_daily
        return np.asarray(epoch[:-1], dtype=np.int64), values

    def read_sessions(name: str):
        return {epoch[-1]: nifty_today if "NIFTY" in name else vix_today}

    monkeypatch.setattr(pipe, "CandleStore", Store)
    monkeypatch.setattr(pipe, "read_daily", read_daily)
    monkeypatch.setattr(pipe, "read_sessions", read_sessions)
    _calendar, _days, market = pipe.market_context(Path("."), GRID)
    research_market = market[0]

    stock_history = upside.StockHistory(tuple(history), stock_daily, tuple(prior))
    market_history = upside.MarketHistory(
        nifty_daily[:, 3], tuple(prior), float(vix_daily[-1, 3])
    )
    for slot in (0, 1, 17, 40, 63):
        minute = int(GRID[slot])
        live_stock = upside.stock_features(
            today_prices, minute, stock_history, prior[-1], minutes
        )
        live_market = upside.market_features(
            nifty_today, vix_today, minute, market_history
        )
        live = upside.all_features(live_stock, live_market, 0.01)
        # research stored every feature as float32 and combined them in float32
        expected = {
            name: np.float32(value)
            for name, value in zip(
                pipe.STOCK_FEATURES, research_stock[slot], strict=True
            )
        }
        expected["slot"] = np.float32(minutes - 1 - minute)
        expected |= dict(zip(pipe.MARKET, research_market[slot], strict=True))
        one = np.float32(1)
        day_return = (one + expected["gap"]) * (one + expected["ret_open"])
        expected["rel_day"] = day_return - one - expected["nifty_day"]
        expected["rel_20d"] = expected["ret20d"] - expected["nifty_ret20d"]
        # movement inputs exactly as scripts/research_scanner_move stores them
        grid = np.full((1, 1, 1, len(move.NAMES)), np.nan, np.float32)
        for name in ("slot", "hist_rv", "atr20"):
            grid[0, 0, 0, move.NAMES.index(name)] = expected[name]
        raw = move.extra_columns(today_prices[: upside.LAST + 1], GRID)[slot]
        for name, value in zip(move.RAW_EXTRA, raw, strict=True):
            grid[0, 0, 0, move.NAMES.index(name)] = value
            expected[name] = np.float32(value)
        move.movement_scales(grid, 0.01)
        for name in ("z_hist", "z_open", "z_atr"):
            expected[name] = grid[0, 0, 0, move.NAMES.index(name)]
        assert set(live) == set(move.NAMES)
        for name in move.NAMES:
            np.testing.assert_allclose(
                live[name], expected[name], rtol=2e-7, atol=0, err_msg=name
            )


def test_live_features_ignore_bars_after_the_decision() -> None:
    rng = np.random.default_rng(3)
    prior = weekdays(date(2025, 1, 1), 280)
    history = upside.StockHistory(
        tuple(session(rng, 100) for _ in range(20)),
        daily_bars(rng, 280, 100),
        tuple(prior),
    )
    today = session(rng, 100)
    later = today.copy()
    later[60:] *= 1.5
    first = upside.stock_features(today, 59, history, prior[-1], 375)
    assert first == upside.stock_features(later, 59, history, prior[-1], 375)


def test_no_features_without_the_previous_session_or_history() -> None:
    rng = np.random.default_rng(4)
    prior = weekdays(date(2025, 1, 1), 280)
    daily = daily_bars(rng, 280, 100)
    today = session(rng, 100)
    short = upside.StockHistory(
        tuple(session(rng, 100) for _ in range(19)), daily, tuple(prior)
    )
    assert upside.stock_features(today, 24, short, prior[-1], 375) is None
    full = upside.StockHistory(
        tuple(session(rng, 100) for _ in range(20)), daily, tuple(prior)
    )
    missed = prior[-1] + timedelta(days=1)  # the stock did not trade that session
    assert upside.stock_features(today, 24, full, missed, 375) is None
    assert upside.stock_features(today, 24, full, prior[-1], 375) is not None


class _Model:
    def __init__(self, raw: float) -> None:
        self.raw = raw

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (np.full(len(rows), 1 - self.raw), np.full(len(rows), self.raw))
        )


def test_bundle_applies_its_calibration_and_requires_inputs() -> None:
    bundle = upside.ScannerBundle(
        "test",
        ("a", "b"),
        frozenset({"b"}),
        frozenset(),
        {"chosen": "platt", "platt_slope": 2.0, "platt_intercept": -0.5},
        _Model(0.6),
    )
    assert bundle.vector({"a": 1.0}) is not None  # optional input may be missing
    assert bundle.vector({"b": 1.0}) is None
    logit = np.log(0.6 / 0.4)
    expected = 1 / (1 + np.exp(-(2.0 * logit - 0.5)))
    assert bundle.predict(np.ones((1, 2)))[0] == pytest.approx(expected)
    isotonic = upside.ScannerBundle(
        "test",
        ("a",),
        frozenset(),
        frozenset(),
        {"chosen": "isotonic", "isotonic_x": [0.0, 1.0], "isotonic_y": [0.2, 0.8]},
        _Model(0.5),
    )
    assert isotonic.predict(np.ones((1, 1)))[0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("name", "inputs", "move"),
    [("scanner_up1_by_close_v1", 31, None), ("scanner_move10_by_close_v1", 35, 0.01)],
)
def test_production_bundle_loads_and_matches_the_live_features(
    name: str, inputs: int, move: float | None
) -> None:
    directory = Path("data/models") / name
    if not (directory / "bundle.json").exists():
        pytest.skip("production scanner bundle not present")
    pytest.importorskip("catboost")
    bundle = upside.ScannerBundle.load(directory)
    assert len(bundle.features) == inputs and "slot" in bundle.features
    assert bundle.move == move
    rng = np.random.default_rng(5)
    prior = weekdays(date(2025, 1, 1), 280)
    history = upside.StockHistory(
        tuple(session(rng, 100) for _ in range(20)),
        daily_bars(rng, 280, 100),
        tuple(prior),
    )
    market = upside.MarketHistory(daily_bars(rng, 280, 25000)[:, 3], tuple(prior), 13)
    live = upside.all_features(
        upside.stock_features(session(rng, 100), 59, history, prior[-1], 375),
        upside.market_features(session(rng, 25000), session(rng, 13), 59, market),
        bundle.move,
    )
    assert set(bundle.features) <= set(live)
    assert 0 < bundle.predict(bundle.vector(live)[None, :])[0] < 1
