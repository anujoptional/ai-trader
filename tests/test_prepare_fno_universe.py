import csv
import json

import pytest

from ai_trader.broker import Instrument
from ai_trader.broker.kite import kite_symbol
from ai_trader.cli.download_history import CONTEXT_SYMBOLS
from scripts.prepare_fno_universe import prepare


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fno_universe_keeps_stock_futures_and_context(tmp_path) -> None:
    write_csv(
        tmp_path / "nfo.csv",
        [
            {"name": "RELIANCE", "segment": "NFO-FUT"},
            {"name": "RELIANCE", "segment": "NFO-FUT"},
            {"name": "M&M", "segment": "NFO-FUT"},
            {"name": "NIFTY", "segment": "NFO-FUT"},
            {"name": "OPTONLY", "segment": "NFO-OPT"},
        ],
    )
    cash = [
        {
            "exchange": "NSE",
            "segment": "NSE",
            "instrument_type": "EQ",
            "tradingsymbol": symbol,
            "name": symbol,
            "instrument_token": str(index + 1),
            "tick_size": "0.05",
        }
        for index, symbol in enumerate(("RELIANCE", "M&M", "OPTONLY"))
    ]
    for offset, symbol in enumerate(CONTEXT_SYMBOLS):
        name = kite_symbol(Instrument("NSE", symbol))
        cash.append(
            {
                "exchange": "NSE",
                "segment": "NSE" if name == symbol else "INDICES",
                "instrument_type": "EQ",
                "tradingsymbol": name,
                "name": name,
                "instrument_token": str(1000 + offset),
                "tick_size": "0.01",
            }
        )
    write_csv(tmp_path / "nse.csv", cash)
    write_csv(
        tmp_path / "sectors.csv",
        [
            {
                "Company Name": "Reliance Industries Ltd.",
                "Industry": "Oil Gas & Consumable Fuels",
                "Symbol": "RELIANCE",
                "ISIN Code": "INE002A01018",
            }
        ],
    )
    output = tmp_path / "universe" / "universe.json"
    document = prepare(
        tmp_path / "nfo.csv", tmp_path / "nse.csv", tmp_path / "sectors.csv", output
    )
    targets = [e for e in document["instruments"] if e["role"] == "equity_target"]
    assert [entry["symbol"] for entry in targets] == ["M&M", "RELIANCE"]
    assert targets[1]["industry"] == "Oil Gas & Consumable Fuels"
    assert targets[0]["industry"] is None
    assert document["excluded_underlyings"] == ["NIFTY"]
    context = [
        e["symbol"] for e in document["instruments"] if e["role"] != "equity_target"
    ]
    assert context == list(CONTEXT_SYMBOLS)
    assert json.loads(output.read_text())["instruments"] == document["instruments"]
    with pytest.raises(ValueError, match="already exists"):
        prepare(
            tmp_path / "nfo.csv", tmp_path / "nse.csv", tmp_path / "sectors.csv", output
        )
