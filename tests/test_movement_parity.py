import csv
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rtdl_revisiting_models")

from ai_trader.scanner.movement import (  # noqa: E402
    DECISIONS,
    FEATURES,
    STOCK_FEATURES,
    context_features,
    feature_vector,
    slot_of,
    stock_features,
)
from scripts.research_fno_scanner import (  # noqa: E402
    EPOCH,
    stock_rows,  # noqa: E402
)
from scripts.research_fno_scanner import FEATURES as RESEARCH_FEATURES


def synthetic_sessions(days: int, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    sessions, price = [], 500.0
    for _ in range(days):
        close = price * np.exp(np.cumsum(rng.normal(0, 0.0012, 345)))
        open_ = np.concatenate(([price * (1 + rng.normal(0, 0.004))], close[:-1]))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0008, 345))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0008, 345))
        volume = rng.integers(1_000, 20_000, 345).astype(float)
        session = np.column_stack((open_, high, low, close, volume))
        session[:, :4] = np.round(session[:, :4], 2)
        session[:, 1] = session[:, :4].max(axis=1)
        session[:, 2] = session[:, :4].min(axis=1)
        sessions.append(session)
        price = float(close[-1])
    return sessions


def write_store(root, symbol: str, dates: list[date], sessions: list[np.ndarray]):
    (root / "daily").mkdir(parents=True)
    minute_path = root / f"NSE_{symbol}.csv"
    daily_path = root / "daily" / f"NSE_{symbol}.csv"
    with minute_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_time", "open", "high", "low", "close", "volume"])
        for day, session in zip(dates, sessions, strict=True):
            for minute, (o, h, lo, c, v) in enumerate(session):
                total = 9 * 60 + 15 + minute
                stamp = f"{day.isoformat()}T{total // 60:02d}:{total % 60:02d}:00+05:30"
                writer.writerow([stamp, o, h, lo, c, int(v)])
    with daily_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for day, s in zip(dates, sessions, strict=True):
            writer.writerow(
                [
                    f"{day.isoformat()}T00:00:00+05:30",
                    s[0, 0],
                    s[:, 1].max(),
                    s[:, 2].min(),
                    s[-1, 3],
                    int(s[:, 4].sum()),
                ]
            )
    files = {}
    for path in (minute_path, daily_path):
        files[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (root / "dataset.json").write_text(
        json.dumps({"provider": "kite", "schema_version": 1, "files": files}), "utf-8"
    )


def test_live_features_match_research_assembly(tmp_path) -> None:
    assert FEATURES == RESEARCH_FEATURES
    dates, day = [], date(2026, 8, 3)
    while len(dates) < 26:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    sessions = synthetic_sessions(len(dates))
    write_store(tmp_path, "XYZ", dates, sessions)
    calendar = np.asarray([d.toordinal() - EPOCH for d in dates], dtype=np.int64)
    research = stock_rows(("XYZ", tmp_path, calendar))
    daily = np.asarray(
        [
            [s[0, 0], s[:, 1].max(), s[:, 2].min(), s[-1, 3], s[:, 4].sum()]
            for s in sessions
        ]
    )
    for slot in (0, 9, 30, 46):
        minute = int(DECISIONS[slot])
        today = sessions[-1].copy()
        today[minute + 1 :] = np.nan  # the future is not available live
        live = stock_features(today, minute, sessions[:-1], daily[:-1])
        expected = research["features"][-1, slot]
        actual = np.asarray([live[name] for name in STOCK_FEATURES])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)
    assert slot_of(int(DECISIONS[3])) == 3 and slot_of(25) is None


def test_context_and_vector_order() -> None:
    nifty = np.full((345, 5), 100.0)
    nifty[40:, 3] = 101.0
    vix = np.full((345, 5), 12.5)
    context = context_features(nifty, 100.0, vix, 49, 0.02)
    assert context["nifty_day"] == pytest.approx(0.01)
    assert context["nifty_ret15"] == pytest.approx(0.01)
    assert context["rel_strength"] == pytest.approx(0.01)
    assert context["slot"] == 5
    stock = dict.fromkeys(STOCK_FEATURES, 0.0)
    assert feature_vector(stock, context).shape == (20,)
