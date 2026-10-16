import json

import pandas as pd
import pytest

from scripts import auction_book as book

HEADER = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
    "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, "
    "DELIV_QTY, DELIV_PER"
)


def cm_line(symbol: str, open_: float, close: float) -> str:
    return f"{symbol}, EQ, 16-Oct-2026, 0, {open_}, 0, 0, 0, {close}, 0, 0, 0, 0, 0, 0"


def test_choose_takes_the_extremes_and_sizes_whole_shares() -> None:
    day = pd.DataFrame(
        {
            "symbol": [f"S{i:02d}" for i in range(30)],
            "composite": [float(i) for i in range(30)],
            "cm_close": [300.0] * 30,
        }
    )
    day.loc[5, "composite"] = None  # unscored stocks are never picked
    picks = book.choose(day, size=3)
    assert [p["symbol"] for p in picks["long"]] == ["S29", "S28", "S27"]
    assert [p["symbol"] for p in picks["short"]] == ["S00", "S01", "S02"]
    assert picks["long"][0]["qty"] == 666  # Rs 2 lakh / Rs 300, whole shares


def test_next_session_is_the_first_later_cash_file(tmp_path) -> None:
    for day in ("2026-10-15", "2026-10-16", "2026-10-19"):
        (tmp_path / f"{day}.csv").write_text(HEADER, "utf-8")
    assert book.next_session("2026-10-15", tmp_path) == "2026-10-16"
    assert book.next_session("2026-10-16", tmp_path) == "2026-10-19"
    assert book.next_session("2026-10-19", tmp_path) is None


def test_a_book_made_before_nine_ist_is_forward() -> None:
    assert book.is_forward("2026-10-15T23:00:00+00:00", "2026-10-16")  # 04:30 IST
    assert not book.is_forward("2026-10-16T03:31:00+00:00", "2026-10-16")  # 09:01


def test_score_buys_longs_and_sells_shorts_at_the_auctions() -> None:
    picks = {
        "features_date": "2026-10-15",
        "created_at_utc": "2026-10-15T23:00:00+00:00",
        "universe": ["UP", "DOWN", "GONE", "FLAT"],
        "long": [{"symbol": "UP", "qty": 100}, {"symbol": "GONE", "qty": 10}],
        "short": [{"symbol": "DOWN", "qty": 50}],
    }
    cm = {
        "UP": {"cm_open": 100.0, "cm_close": 101.0},
        "DOWN": {"cm_open": 200.0, "cm_close": 198.0},
        "FLAT": {"cm_open": 50.0, "cm_close": 50.0},
    }
    result = book.score(picks, cm, "2026-10-16")
    assert result["missing"] == ["GONE"]
    assert result["long_bp"] == pytest.approx(100.0)
    assert result["short_bp"] == pytest.approx(100.0)  # a 1% fall earns a short 1%
    assert result["book_gross_bp"] == pytest.approx(200.0)
    assert result["gross_rs"] == pytest.approx(100 * 1.0 + 50 * 2.0)
    assert 0 < result["charges_rs"] < 50
    assert result["net_rs"] == pytest.approx(result["gross_rs"] - result["charges_rs"])
    assert result["fno_average_bp"] == pytest.approx((100 - 100 + 0) / 3)
    assert result["forward"]


def test_score_all_records_each_session_once(tmp_path) -> None:
    daily = tmp_path / "daily"
    daily.mkdir()
    lines = [HEADER, cm_line("UP", 100, 101), cm_line("DOWN", 200, 198)]
    (daily / "2026-10-16.csv").write_text("\n".join(lines) + "\n", "utf-8")
    picks = {
        "features_date": "2026-10-15",
        "created_at_utc": "2026-10-15T23:00:00+00:00",
        "universe": ["UP", "DOWN"],
        "long": [{"symbol": "UP", "qty": 100}],
        "short": [{"symbol": "DOWN", "qty": 50}],
    }
    (tmp_path / "picks_2026-10-15.json").write_text(json.dumps(picks), "utf-8")
    assert book.score_all(tmp_path, daily) == 0
    assert book.score_all(tmp_path, daily) == 0
    rows = (tmp_path / "results.csv").read_text("utf-8").splitlines()
    assert len(rows) == 2 and rows[1].startswith("2026-10-16,2026-10-15,True,100.0")
    saved = json.loads((tmp_path / "results_2026-10-16.json").read_text("utf-8"))
    assert saved["book_gross_bp"] == pytest.approx(200.0)
