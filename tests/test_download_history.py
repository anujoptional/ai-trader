import csv
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument, OHLCVCandle
from ai_trader.broker.kite import KiteBrokerError, kite_symbol
from ai_trader.cli.download_history import (
    CONTEXT_SYMBOLS,
    download,
    download_windows,
    prepare_universe,
    read_universe,
)
from ai_trader.clock import INDIA_TIMEZONE


def test_download_extends_cached_ranges_without_large_uncheckpointed_gaps():
    start = datetime(2024, 1, 1, 9, 15, tzinfo=INDIA_TIMEZONE)
    held = (
        datetime(2026, 7, 1, 9, 15, tzinfo=INDIA_TIMEZONE),
        datetime(2026, 9, 28, 15, 30, tzinfo=INDIA_TIMEZONE),
    )
    end = datetime(2026, 9, 30, 15, 30, tzinfo=INDIA_TIMEZONE)
    windows = download_windows(start, end, held)
    assert windows[0][1] == held[0]
    assert windows[-1] == (held[1], end)
    assert min(first for first, _ in windows) == start
    assert all(
        timedelta(0) < last - first <= timedelta(days=28) for first, last in windows
    )
    assert download_windows(held[0], held[1], held) == ()
    with pytest.raises(ValueError, match="ordered"):
        download_windows(end, start, None)


def test_download_preserves_successful_pages_and_does_not_print_broker_errors(tmp_path):
    class Broker:
        calls = 0

        def get_historical_candles(self, instrument, start, end, interval):
            self.calls += 1
            if self.calls > 1:
                raise KiteBrokerError("untrusted response text")
            return (
                OHLCVCandle(
                    start, Decimal(100), Decimal(101), Decimal(99), Decimal(100), 10
                ),
            )

    start = datetime(2024, 1, 1, 9, 15, tzinfo=INDIA_TIMEZONE)
    messages = []
    report = download(
        Broker(),
        [Instrument("NSE", "TEST")],
        start,
        start + timedelta(days=60),
        tmp_path / "candles",
        tmp_path / "download.json",
        now=start + timedelta(days=100),
        universe_sha256="fixture",
        emit=messages.append,
    )
    assert (tmp_path / "candles/NSE_TEST.csv").exists()
    assert len(report["instruments"]["NSE:TEST"]["completed_windows"]) == 1
    assert report["instrument_status_counts"] == {"failed": 1}
    assert "untrusted response" not in " ".join(messages)


def test_universe_snapshot_uses_public_csv_symbols_and_separates_context(tmp_path):
    nifty = tmp_path / "nifty.csv"
    master = tmp_path / "master.csv"
    with nifty.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("Symbol", "Company Name", "Industry")
        )
        writer.writeheader()
        writer.writerows(
            {
                "Symbol": f"STOCK{index}",
                "Company Name": "Example, Limited",
                "Industry": "Testing",
            }
            for index in range(50)
        )
    with master.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "exchange",
                "segment",
                "tradingsymbol",
                "name",
                "instrument_type",
                "instrument_token",
                "tick_size",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "exchange": "NSE",
                "segment": "INDICES" if symbol in CONTEXT_SYMBOLS[:12] else "NSE",
                "tradingsymbol": kite_symbol(Instrument("NSE", symbol)),
                "name": symbol,
                "instrument_type": "EQ",
                "instrument_token": index + 1,
                "tick_size": "0.05",
            }
            for index, symbol in enumerate(
                [*(f"STOCK{index}" for index in range(50)), *CONTEXT_SYMBOLS]
            )
        )
    path = tmp_path / "universe.json"
    document = prepare_universe(nifty, master, path)
    assert document["provider"] == "kite"
    assert document["instruments"][50]["kite_symbol"] == "NIFTY 50"
    assert len(read_universe(path)[1]) == 66
    assert (
        sum(item["role"] == "equity_target" for item in document["instruments"]) == 50
    )
    assert all(item["role"] == "context_only" for item in document["instruments"][50:])
    assert document["instruments"][0]["name"] == "Example, Limited"
    with pytest.raises(ValueError, match="already exists"):
        prepare_universe(nifty, master, path)


def test_refresh_can_revisit_a_previously_empty_tail_and_reports_actual_coverage(
    tmp_path,
):
    class Broker:
        calls = 0

        def get_historical_candles(self, instrument, start, end, interval):
            self.calls += 1
            return tuple(
                OHLCVCandle(
                    start + timedelta(minutes=minute),
                    Decimal(100),
                    Decimal(101),
                    Decimal(99),
                    Decimal(100),
                    10,
                )
                for minute in range(self.calls)
            )

    start = datetime(2024, 1, 1, 9, 15, tzinfo=INDIA_TIMEZONE)
    broker = Broker()
    arguments = (
        broker,
        [Instrument("NSE", "TEST")],
        start,
        start + timedelta(minutes=2),
        tmp_path / "candles",
        tmp_path / "download.json",
    )
    options = {
        "now": start + timedelta(days=10),
        "universe_sha256": "fixture",
        "emit": lambda message: None,
    }
    first = download(*arguments, **options)
    assert first["instruments"]["NSE:TEST"]["coverage_boundary_warnings"] == [
        "requested_end_not_observed"
    ]
    download(*arguments, **options)
    assert broker.calls == 1
    refreshed = download(*arguments, **options, refresh=True)
    assert broker.calls == 2
    assert refreshed["instruments"]["NSE:TEST"]["candles"] == 2
    assert refreshed["instruments"]["NSE:TEST"]["coverage_boundary_warnings"] == []
