"""The paper universe carries a complete static profile for every stock."""

import json
from pathlib import Path

from scripts.fill_universe_profiles import SECTORS

UNIVERSE = (
    Path(__file__).resolve().parents[1] / "data/universe_mcap_20261012/universe.json"
)


def test_every_universe_stock_has_a_company_and_an_nse_sector() -> None:
    universe = json.loads(UNIVERSE.read_text("utf-8"))
    stocks = [e for e in universe["instruments"] if e["role"] == "equity_target"]
    assert len(stocks) == 650
    assert [e["symbol"] for e in stocks if not e["company"].strip()] == []
    assert [e["symbol"] for e in stocks if e["industry"] not in SECTORS] == []
    assert [e["symbol"] for e in stocks if not e["industry_source"]] == []
