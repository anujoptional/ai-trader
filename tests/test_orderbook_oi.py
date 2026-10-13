from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import DepthLevel, DepthSnapshot, FuturesDay, Instrument
from ai_trader.broker.kite import HistoryRateLimiter, KiteBroker, KiteBrokerError
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.oi.features import (
    buildup,
    daily_oi_features,
    live_futures,
    monthly_expiries,
    previous_session_futures,
)
from ai_trader.orderbook.features import depth_features
from ai_trader.orderbook.recorder import DepthRecorder, record

T0 = datetime(2026, 10, 12, 10, 0, tzinfo=INDIA_TIMEZONE)


def quote_row(**overrides) -> dict:
    row = {
        "timestamp": "2026-10-12 10:00:01",
        "last_trade_time": "2026-10-12 10:00:00",
        "last_price": 100.5,
        "volume": 12000,
        "buy_quantity": 30000,
        "sell_quantity": 10000,
        "ohlc": {"open": 99, "high": 101, "low": 98.5, "close": 99.0},
        "lower_circuit_limit": 89.1,
        "upper_circuit_limit": 108.9,
        "oi": 0,
        "depth": {
            "buy": [{"price": 100.45, "quantity": 500, "orders": 3}]
            + [
                {"price": 100.4 - i * 0.05, "quantity": 1000, "orders": 5}
                for i in range(4)
            ],
            "sell": [{"price": 100.5, "quantity": 100, "orders": 1}]
            + [
                {"price": 100.55 + i * 0.05, "quantity": 1000, "orders": 5}
                for i in range(3)
            ]
            + [{"price": 0, "quantity": 0, "orders": 0}],
        },
    }
    row.update(overrides)
    return row


class FakeClient:
    def __init__(self) -> None:
        self.quotes: list[list[str]] = []
        self.history: list[dict] = []

    def quote(self, keys):
        self.quotes.append(keys)
        return {"NSE:TCS": quote_row(), "NFO:TCS26OCTFUT": quote_row(oi=1500)}

    def instruments(self, exchange):
        return [
            {
                "segment": "NFO-FUT",
                "name": "TCS",
                "expiry": date(2026, 11, 24),
                "instrument_token": 2,
                "tradingsymbol": "TCS26NOVFUT",
            },
            {
                "segment": "NFO-FUT",
                "name": "TCS",
                "expiry": date(2026, 10, 27),
                "instrument_token": 1,
                "tradingsymbol": "TCS26OCTFUT",
            },
            {
                "segment": "NFO-OPT",
                "name": "TCS",
                "expiry": date(2026, 10, 27),
                "instrument_token": 3,
                "tradingsymbol": "TCS26OCT3000CE",
            },
        ]

    def historical_data(self, **kwargs):
        self.history.append(kwargs)
        return [
            {
                "date": "2026-10-09T00:00:00+0530",
                "open": 10,
                "high": 11,
                "low": 9.5,
                "close": 10.5,
                "volume": 100,
                "oi": 5000,
            }
        ]


def test_kite_depth_parses_levels_and_reports_missing() -> None:
    broker = KiteBroker(FakeClient(), limiter=HistoryRateLimiter())
    snapshots, missing = broker.get_depth(
        [
            Instrument("NSE", "TCS"),
            Instrument("NFO", "TCS26OCTFUT"),
            Instrument("NSE", "X"),
        ]
    )
    assert [s.instrument.trading_symbol for s in snapshots] == ["TCS", "TCS26OCTFUT"]
    assert missing == (Instrument("NSE", "X"),)
    assert len(snapshots[0].asks) == 4 and snapshots[0].open_interest is None
    assert snapshots[1].open_interest == 1500
    assert snapshots[0].timestamp.tzinfo is not None


def test_forked_kite_clients_share_both_rate_limits(monkeypatch) -> None:
    import ai_trader.broker.kite as kite

    def keyed_client(*_args) -> FakeClient:
        client = FakeClient()
        client.api_key, client.access_token = "key", "token"
        return client

    monkeypatch.setattr(kite, "_history_client", keyed_client)
    broker = KiteBroker(keyed_client(), limiter=HistoryRateLimiter())
    worker = broker.fork_history()
    assert worker is not broker
    assert worker._history_limiter is broker._history_limiter
    assert worker._quote_limiter is broker._quote_limiter
    assert worker.fork_history()._quote_limiter is broker._quote_limiter


def test_continuous_futures_use_near_contract_with_oi() -> None:
    client = FakeClient()
    broker = KiteBroker(client, limiter=HistoryRateLimiter())
    bars = broker.get_continuous_futures_daily(
        "TCS", date(2020, 1, 1), date(2026, 10, 9)
    )
    assert {call["instrument_token"] for call in client.history} == {1}
    assert all(call["continuous"] and call["oi"] for call in client.history)
    assert len(client.history) == 3  # 1000-day windows over ~6.8 years
    assert bars == (
        FuturesDay(
            date(2026, 10, 9),
            Decimal("10"),
            Decimal("11"),
            Decimal("9.5"),
            Decimal("10.5"),
            100,
            5000,
        ),
    )


def snapshot(bids, asks, **kw) -> DepthSnapshot:
    return DepthSnapshot(
        instrument=Instrument("NSE", "TCS"),
        timestamp=T0,
        last_price=Decimal("100.5"),
        last_trade_at=T0 - timedelta(seconds=2),
        volume=1000,
        total_buy_quantity=kw.get("buy", 3000),
        total_sell_quantity=kw.get("sell", 1000),
        bids=tuple(DepthLevel(Decimal(str(p)), q, 1) for p, q in bids),
        asks=tuple(DepthLevel(Decimal(str(p)), q, 1) for p, q in asks),
        previous_close=Decimal("100"),
        upper_circuit=Decimal("110"),
        lower_circuit=Decimal("90"),
    )


def test_depth_features_measure_spread_imbalance_and_slippage() -> None:
    features = depth_features(
        snapshot([(100.4, 300), (100.3, 5000)], [(100.6, 100), (100.7, 5000)]), as_of=T0
    )
    assert features["spread_bps"] == pytest.approx(0.2 / 100.5 * 1e4)
    assert features["imbalance_total"] == pytest.approx(0.5)
    assert features["imbalance_l1"] == pytest.approx(0.5)
    assert features["microprice_offset_bps"] > 0
    assert features["buy_slippage_bps"] > features["spread_bps"] / 2
    assert features["seconds_since_trade"] == 2
    thin = depth_features(snapshot([(100.4, 10)], [(100.6, 10)]))
    assert thin["buy_slippage_bps"] is None  # five levels cannot fill Rs 2 lakh
    locked = depth_features(snapshot([(110, 9000)], []))
    assert locked["two_sided"] == 0 and locked["spread_bps"] is None


def test_recorder_round_trips_and_reads_as_of(tmp_path) -> None:
    recorder = DepthRecorder(tmp_path / "book.sqlite")
    snap = snapshot([(100.4, 300)], [(100.6, 100)])
    recorder.write(T0, [snap])
    assert recorder.latest(snap.instrument, T0 - timedelta(seconds=1)) is None
    assert recorder.latest(snap.instrument, T0) == snap
    recorder.close()


def test_record_loop_survives_failed_polls(tmp_path) -> None:
    clock = [T0]
    calls = []

    class Source:
        def get_depth(self, instruments):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("network")
            return (snapshot([(100.4, 300)], [(100.6, 100)]),), ()

    def sleep(seconds):
        clock[0] += timedelta(seconds=seconds)

    recorder = DepthRecorder(tmp_path / "book.sqlite")
    stats = record(
        Source(),
        [Instrument("NSE", "TCS")],
        recorder,
        interval_seconds=30,
        until=T0 + timedelta(seconds=90),
        now=lambda: clock[0],
        sleep=sleep,
    )
    assert stats == {"cycles": 3, "snapshots": 2, "missing": 0, "errors": 1}


def test_expiries_switch_to_tuesday_and_skip_holidays() -> None:
    sessions = [date(2025, 8, 1) + timedelta(days=i) for i in range(80)]
    sessions = [d for d in sessions if d.weekday() < 5 and d != date(2025, 8, 28)]
    expiries = monthly_expiries(sessions)
    assert expiries[:2] == [date(2025, 8, 27), date(2025, 9, 30)]
    partial = [date(2026, 10, 1), date(2026, 10, 9)]
    assert monthly_expiries(partial) == [date(2026, 10, 27)]


def test_oi_features_withhold_changes_across_contract_switch() -> None:
    days = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1)]
    bars = [
        FuturesDay(d, Decimal(p), Decimal(p), Decimal(p), Decimal(p), 10, oi)
        for d, p, oi in zip(
            days, ("100", "102", "101", "103"), (1000, 1100, 400, 5000), strict=True
        )
    ]
    features = daily_oi_features(bars, {days[1]: 101.0}, [date(2026, 9, 29)])
    assert features[days[1]]["oi_change_1d"] == pytest.approx(0.1)
    assert features[days[1]]["buildup_1d"] == "long_buildup"
    assert features[days[1]]["basis_pct"] == pytest.approx((102 / 101 - 1) * 100)
    assert features[days[2]]["oi_change_1d"] is None  # first day of next contract
    assert features[days[3]]["oi_change_1d"] == pytest.approx(11.5)
    assert buildup(-0.01, 0.05) == "short_buildup"
    assert buildup(0.001, 0.05) == "neutral"


def test_futures_daily_reads_one_contract_as_traded() -> None:
    client = FakeClient()
    broker = KiteBroker(client, limiter=HistoryRateLimiter())
    bars = broker.get_futures_daily("TCS26NOVFUT", date(2026, 10, 1), date(2026, 10, 9))
    assert [(c["instrument_token"], c["continuous"]) for c in client.history] == [
        (2, False)
    ]
    assert client.history[0]["oi"] and bars[0].open_interest == 5000
    with pytest.raises(KiteBrokerError):
        broker.get_futures_daily("TCS26DECFUT", date(2026, 10, 1), date(2026, 10, 9))


def test_futures_close_may_lie_outside_the_traded_range() -> None:
    # SAIL26OCTFUT on 2026-09-24, as Kite returned it and NSE's F&O bhavcopy has it:
    # the settlement close 184.53 is below the day's lowest trade, 185.06.
    row = {
        "date": "2026-09-24T00:00:00+0530",
        "open": 188.95,
        "high": 188.95,
        "low": 185.06,
        "close": 184.53,
        "volume": 3327600,
        "oi": 80948100,
    }
    client = FakeClient()
    client.historical_data = lambda **kwargs: [row]
    broker = KiteBroker(client, limiter=HistoryRateLimiter())
    (bar,) = broker.get_futures_daily(
        "TCS26OCTFUT", date(2026, 9, 24), date(2026, 9, 24)
    )
    assert bar.close == Decimal("184.53") and bar.open_interest == 80948100
    client.historical_data = lambda **kwargs: [row | {"open": 190.0}]
    with pytest.raises(KiteBrokerError):  # a traded price outside the range is bad data
        broker.get_futures_daily("TCS26OCTFUT", date(2026, 9, 24), date(2026, 9, 24))


SESSIONS = [date(2026, 9, 30)] + [date(2026, 10, d) for d in (1, 5, 6, 7, 8, 9)]


def contract(rows) -> tuple[FuturesDay, ...]:
    return tuple(
        FuturesDay(d, Decimal(p), Decimal(p), Decimal(p), Decimal(p), 10, oi)
        for d, p, oi in rows
    )


NEAR = contract(
    (d, "99" if d == SESSIONS[-1] else "100", oi)
    for d, oi in zip(SESSIONS, (1000, 1000, 1050, 1100, 1100, 1000, 900), strict=True)
)
# No trade on Oct 8: no bar, and the OI of Oct 7 still stands.
NEXT = contract(
    (d, "101", oi)
    for d, oi in zip(SESSIONS, (100, 100, 100, 150, 200, None, 300), strict=True)
    if oi is not None
)


def test_previous_session_oi_sums_both_months_so_a_roll_is_not_unwinding() -> None:
    facts = previous_session_futures(
        {"N": NEAR, "X": NEXT}, "N", SESSIONS, date(2026, 9, 29)
    )
    # 100 contracts moved from the near month to the next: the near month alone
    # fell 10% (long unwinding); the total did not change.
    assert facts["oi_change_prev_session"] == 0.0
    assert facts["futures_change_prev_session"] == pytest.approx(-0.01)
    assert facts["buildup_prev_session"] == "neutral"
    assert facts["oi_change_5_sessions"] == pytest.approx(1200 / 1100 - 1)
    assert facts["previous_oi"] == {"N": 900, "X": 300}


def test_previous_session_oi_changes_start_at_the_last_expiry() -> None:
    bars = {"N": NEAR, "X": NEXT}
    week = previous_session_futures(bars, "N", SESSIONS, date(2026, 10, 6))
    assert week["oi_change_5_sessions"] is None and week["oi_change_prev_session"] == 0
    # The day after an expiry: yesterday the expiring month still held positions.
    after = previous_session_futures(bars, "N", SESSIONS, SESSIONS[-1])
    assert after["oi_change_prev_session"] is None
    assert after["buildup_prev_session"] is None
    assert after["previous_oi"] == {"N": 900, "X": 300}


def quote(last: str, previous: str, bid: str, ask: str, oi=None) -> DepthSnapshot:
    return DepthSnapshot(
        instrument=Instrument("NFO", "N"),
        timestamp=T0,
        last_price=Decimal(last),
        last_trade_at=T0,
        volume=1,
        total_buy_quantity=1,
        total_sell_quantity=1,
        bids=(DepthLevel(Decimal(bid), 1, 1),),
        asks=(DepthLevel(Decimal(ask), 1, 1),),
        previous_close=Decimal(previous),
        open_interest=oi,
    )


def test_live_futures_compare_the_same_contracts_and_simultaneous_mids() -> None:
    near = quote("98", "99", "97.9", "98.1", oi=1000)
    stock = quote("97.5", "98", "97.4", "97.6")
    quotes = {"N": near, "X": quote("99", "99", "98.9", "99.1", oi=350)}
    facts = live_futures({"N": 900, "X": 300}, quotes, "N", stock)
    assert facts["oi_change_today"] == pytest.approx(1350 / 1200 - 1)
    assert facts["buildup_today"] == "short_buildup"  # price down, OI up
    assert facts["premium_now"] == pytest.approx(98.0 / 97.5 - 1)
    alone = live_futures({"N": 900, "X": 300}, {"N": near}, "N", stock)
    assert alone["oi_change_today"] == pytest.approx(1000 / 900 - 1)
    assert live_futures({"N": 900, "X": 300}, {"X": quotes["X"]}, "N", stock) == {}
