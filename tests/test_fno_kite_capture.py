import zlib
from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np

from ai_trader.broker.kite import KiteBroker
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.fno import kite_capture as kc

TODAY = date(2026, 10, 16)


def instrument(symbol, name, segment, kind, expiry, strike=0.0, token=None):
    return {
        "exchange": "NFO",
        "segment": segment,
        "tradingsymbol": symbol,
        "name": name,
        "instrument_type": kind,
        "expiry": expiry,
        "strike": strike,
        "instrument_token": token or zlib.crc32(symbol.encode()),
        "lot_size": 100,
        "tick_size": 0.05,
    }


def dump():
    rows = [
        instrument("ABC26OCTFUT", "ABC", "NFO-FUT", "FUT", date(2026, 10, 27)),
        instrument("ABC26NOVFUT", "ABC", "NFO-FUT", "FUT", date(2026, 11, 24)),
        instrument("NIFTY26OCTFUT", "NIFTY", "NFO-FUT", "FUT", date(2026, 10, 27)),
        instrument("OLD26SEPFUT", "OLD", "NFO-FUT", "FUT", date(2026, 9, 29)),
    ]
    for expiry in (date(2026, 10, 27), date(2026, 11, 24), date(2026, 12, 29)):
        for strike in (80, 90, 100, 110, 120):
            for kind in ("CE", "PE"):
                rows.append(
                    instrument(
                        f"ABC{expiry:%y%b}{strike}{kind}".upper(),
                        "ABC",
                        "NFO-OPT",
                        kind,
                        expiry,
                        float(strike),
                    )
                )
    for strike in (24000, 25000, 26000):
        rows.append(
            instrument(
                f"NIFTY26O20{strike}CE",
                "NIFTY",
                "NFO-OPT",
                "CE",
                date(2026, 10, 20),
                float(strike),
            )
        )
    rows.append(
        instrument(
            "FINNIFTY26OCT25000CE",
            "FINNIFTY",
            "NFO-OPT",
            "CE",
            date(2026, 10, 27),
            25000.0,
        )
    )
    return rows


def test_selection_takes_all_futures_then_near_options_inside_the_band() -> None:
    prices = {"NSE:ABC": Decimal(100), "NSE:NIFTY 50": Decimal(25000)}
    chosen = kc.select_contracts(dump(), prices, TODAY)
    futures = [r for r in chosen if r["segment"] == "NFO-FUT"]
    assert {r["tradingsymbol"] for r in futures} == {
        "ABC26OCTFUT",
        "ABC26NOVFUT",
        "NIFTY26OCTFUT",
    }  # the expired OLD contract is not live
    assert chosen[: len(futures)] == futures
    options = chosen[len(futures) :]
    assert [r["tradingsymbol"] for r in options[:3]] == [
        "NIFTY26O2024000CE",  # 4% band, both edges included
        "NIFTY26O2025000CE",
        "NIFTY26O2026000CE",
    ]
    stock = [r for r in options if r["name"] == "ABC"]
    assert {r["strike"] for r in stock} == {90.0, 100.0, 110.0}  # within 10% of 100
    assert {r["expiry"] for r in stock} == {date(2026, 10, 27), date(2026, 11, 24)}
    assert all(
        r["name"] != "FINNIFTY" for r in options
    )  # index options: NIFTY, BANKNIFTY
    expiries = [r["expiry"] for r in stock]
    assert expiries == sorted(expiries)


class FakeBroker:
    def __init__(self, bars_by_token, fail=()):
        self.bars_by_token, self.fail, self.requests = bars_by_token, set(fail), []

    def nfo_instruments(self):
        return dump()

    def ltp_map(self, keys):
        assert "NSE:NIFTY 50" in keys and "NSE:ABC" in keys and "NSE:NIFTY" not in keys
        return {"NSE:ABC": Decimal(100), "NSE:NIFTY 50": Decimal(25000)}

    def get_minute_bars_with_oi(self, token, start, end):
        self.requests.append((token, start, end))
        if token in self.fail:
            raise RuntimeError("Kite said no")
        return [b for b in self.bars_by_token.get(token, []) if start <= b[0] <= end]


def bars(token_day_minutes, price="10.05"):
    out = {}
    for token, day, minutes in token_day_minutes:
        opening = datetime(day.year, day.month, day.day, 9, 15, tzinfo=INDIA_TIMEZONE)
        out.setdefault(token, []).extend(
            (
                opening + timedelta(minutes=m),
                Decimal(price),
                Decimal(price),
                Decimal(price),
                Decimal(price),
                100 * (m + 1),
                5000 + m,
            )
            for m in minutes
        )
    return out


def test_capture_backfills_once_then_continues_from_the_last_session(tmp_path) -> None:
    rows = {r["tradingsymbol"]: r for r in dump()}
    fut = rows["ABC26OCTFUT"]["instrument_token"]
    yesterday = TODAY - timedelta(days=1)
    data = bars([(fut, yesterday, [0, 1]), (fut, TODAY, [0, 1, 374])])
    broker = FakeBroker(data)
    logs = []
    summary = kc.capture(broker, TODAY, tmp_path, logs.append)
    assert summary["failed"] == 0 and summary["bars"] == 5
    assert summary["sessions"] == [yesterday.isoformat(), TODAY.isoformat()]
    first_start = next(s for t, s, _ in broker.requests if t == fut)
    assert first_start.date() == TODAY - timedelta(days=kc.LOOKBACK_DAYS)
    with np.load(tmp_path / f"{TODAY}.npz") as day:
        assert list(day["minute"]) == [0, 1, 374]
        assert list(day["close"]) == [1005, 1005, 1005] and list(day["oi"]) == [
            5000,
            5001,
            5374,
        ]
    register = kc.MinuteStore(tmp_path)
    assert register.contracts["0"]["tradingsymbol"] == "ABC26OCTFUT"
    assert register.state["ABC26OCTFUT"] == TODAY.isoformat()
    # The next session starts after the last captured one.
    tomorrow = TODAY + timedelta(days=1)
    broker2 = FakeBroker(bars([(fut, tomorrow, [5])]))
    kc.capture(broker2, tomorrow, tmp_path, logs.append)
    second_start = next(s for t, s, _ in broker2.requests if t == fut)
    assert second_start.date() == tomorrow


def test_merge_keeps_the_latest_bar_and_failures_are_retried_later(tmp_path) -> None:
    rows = {r["tradingsymbol"]: r for r in dump()}
    fut = rows["ABC26OCTFUT"]["instrument_token"]
    broken = rows["ABC26NOVFUT"]["instrument_token"]
    broker = FakeBroker(bars([(fut, TODAY, [3])], price="10.00"), fail=[broken])
    summary = kc.capture(broker, TODAY, tmp_path, lambda m: None)
    assert summary["failed"] == 1
    store = kc.MinuteStore(tmp_path)
    assert "ABC26NOVFUT" not in store.state  # tried again by the next run
    # A later capture of the same minute (e.g. a rerun) replaces it.
    store.flush(
        {
            TODAY.isoformat(): [
                np.array([[store._ids["ABC26OCTFUT"], 3, 1010, 1010, 1010, 1010, 7, 8]])
            ]
        },
        {},
    )
    store.merge()
    with np.load(tmp_path / f"{TODAY}.npz") as day:
        assert list(day["close"]) == [1010] and list(day["volume"]) == [7]


def test_a_contract_with_unusable_bars_is_skipped_and_the_rest_kept(tmp_path) -> None:
    rows = {r["tradingsymbol"]: r for r in dump()}
    fut = rows["ABC26OCTFUT"]["instrument_token"]
    odd = rows["ABC26NOVFUT"]["instrument_token"]
    data = bars([(fut, TODAY, [0])])
    data |= bars([(odd, TODAY, [0])], price="10.005")  # not whole paise
    summary = kc.capture(FakeBroker(data), TODAY, tmp_path, lambda m: None)
    assert summary["failed"] == 1 and summary["fetched"] >= 1
    store = kc.MinuteStore(tmp_path)
    assert "ABC26NOVFUT" not in store.state and "ABC26NOVFUT" not in store._ids
    assert store.state["ABC26OCTFUT"] == TODAY.isoformat()
    with np.load(tmp_path / f"{TODAY}.npz") as day:
        assert list(day["contract"]) == [store._ids["ABC26OCTFUT"]]


def test_capture_stops_at_its_deadline_and_resumes(tmp_path) -> None:
    broker = FakeBroker({})
    late = datetime(2026, 10, 16, 18, 30, tzinfo=INDIA_TIMEZONE)
    summary = kc.capture(
        broker,
        TODAY,
        tmp_path,
        lambda m: None,
        deadline=late - timedelta(minutes=1),
        now=lambda: late,
    )
    assert summary["stopped_at_deadline"] and summary["fetched"] == 0
    assert broker.requests == []


def test_prices_must_be_whole_paise() -> None:
    assert kc.paise(Decimal("123.45")) == 12345
    try:
        kc.paise(Decimal("1.005"))
    except ValueError:
        return
    raise AssertionError("a fraction of a paisa was accepted")


class FakeClient:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def historical_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows


def test_kite_minute_bars_with_oi_are_parsed_and_checked() -> None:
    client = FakeClient(
        [
            {
                "date": "2026-10-16T09:15:00+05:30",
                "open": 10.05,
                "high": 10.5,
                "low": 10.0,
                "close": 10.2,
                "volume": 300,
                "oi": 4500,
            },
            {
                "date": "2026-10-16T09:16:00+05:30",
                "open": 10.2,
                "high": 10.2,
                "low": 10.2,
                "close": 10.2,
                "volume": 0,
                "oi": 4500,
            },
        ]
    )
    broker = KiteBroker(client)
    start = datetime(2026, 10, 16, 9, 15, tzinfo=INDIA_TIMEZONE)
    bars = broker.get_minute_bars_with_oi(42, start, start + timedelta(hours=7))
    assert client.calls[0]["oi"] is True and client.calls[0]["interval"] == "minute"
    assert bars[0][1:] == (
        Decimal("10.05"),
        Decimal("10.5"),
        Decimal("10.0"),
        Decimal("10.2"),
        300,
        4500,
    )
    client.rows = client.rows[::-1]
    try:
        broker.get_minute_bars_with_oi(42, start, start + timedelta(hours=7))
    except Exception as error:
        assert "time order" in str(error)
    else:
        raise AssertionError("out-of-order bars were accepted")
