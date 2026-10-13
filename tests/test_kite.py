from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from ai_trader.broker import CandleInterval, Instrument
from ai_trader.broker.kite import (
    KiteAuthenticationError,
    KiteBroker,
    KiteBrokerError,
    KiteHistoricalBroker,
)
from ai_trader.clock import INDIA_TIMEZONE

RELIANCE = Instrument("NSE", "RELIANCE")
BEGINNING = datetime(2026, 9, 1, 9, 15, tzinfo=INDIA_TIMEZONE)


def test_kite_dataset_refuses_tampering_and_provider_mixing(tmp_path) -> None:
    import json

    from ai_trader.cli.compare_broker_history import _digest
    from ai_trader.history import CandleStore, CandleStoreError
    from ai_trader.market import Candle

    store = CandleStore(tmp_path)
    candle = Candle(
        RELIANCE,
        BEGINNING,
        BEGINNING + timedelta(minutes=1),
        Decimal(100),
        Decimal(101),
        Decimal(99),
        Decimal(100),
        10,
    )
    store._write(RELIANCE, (candle,))
    path = store.path_for(RELIANCE)
    manifest = {
        "schema_version": 1,
        "provider": "kite",
        "files": {path.name: _digest(path)},
    }
    (tmp_path / "dataset.json").write_text(json.dumps(manifest))
    validated = CandleStore(tmp_path, Mock())
    assert validated.load(RELIANCE, BEGINNING, candle.end_time) == (candle,)
    with pytest.raises(CandleStoreError, match="immutable"):
        validated.load(RELIANCE, BEGINNING, candle.end_time + timedelta(minutes=1))
    path.write_text(path.read_text().replace("100", "200"))
    with pytest.raises(CandleStoreError, match="fingerprint"):
        CandleStore(tmp_path).load(RELIANCE, BEGINNING, candle.end_time)
    manifest["provider"] = "retired-provider"
    (tmp_path / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(CandleStoreError, match="provenance"):
        CandleStore(tmp_path)


def test_native_volume_never_fills_unknown_ticks() -> None:
    from ai_trader.broker import MarketTick
    from ai_trader.market import StreamVolume

    observer = StreamVolume((RELIANCE,))
    with observer:
        known = MarketTick(RELIANCE, Decimal(100), BEGINNING, 50)
        unknown = MarketTick(RELIANCE, Decimal(100), BEGINNING + timedelta(seconds=1))
        assert observer.stamp(known) is known
        assert observer.latest(RELIANCE) == 50
        assert observer.stamp(unknown) is unknown
        assert observer.latest(RELIANCE) is None
    assert observer.poll_count == observer.failure_count == 0


def test_kite_capture_preserves_rejection_but_runtime_still_refuses(
    kite_client,
) -> None:
    original = kite_client.historical_data.return_value[0]
    kite_client.historical_data.return_value = [
        original,
        {**original, "date": BEGINNING + timedelta(minutes=1), "open": 0},
    ]
    broker = KiteBroker(kite_client)
    capture = broker.capture_historical_candles(
        RELIANCE, BEGINNING, BEGINNING + timedelta(minutes=2), CandleInterval.ONE_MINUTE
    )
    assert len(capture.candles) == len(capture.rejected) == 1
    assert capture.rejected[0]["values"]["open"] == "0"
    with pytest.raises(KiteBrokerError, match="validation"):
        broker.get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(minutes=2),
            CandleInterval.ONE_MINUTE,
        )


def test_staging_quarantines_entire_bad_session(tmp_path, kite_client) -> None:
    from datetime import date

    from ai_trader.cli.compare_broker_history import _read_snapshot
    from ai_trader.cli.migrate_kite_data import stage_instrument
    from ai_trader.history import CandleStore

    original = kite_client.historical_data.return_value[0]
    later = BEGINNING + timedelta(days=1)
    kite_client.historical_data.side_effect = [
        [
            original,
            {**original, "date": BEGINNING + timedelta(minutes=1), "open": 0},
            {**original, "date": later},
        ],
        [
            {**original, "date": stamp.replace(hour=0, minute=0)}
            for stamp in (BEGINNING, later)
        ],
    ]
    report = stage_instrument(
        KiteBroker(kite_client),
        {"exchange": "NSE", "symbol": "RELIANCE", "role": "equity_target"},
        date(2026, 9, 1),
        date(2026, 9, 2),
        tmp_path,
    )
    assert report["quality"]["quarantined_dates"] == ["2026-09-01"]
    assert len(report["quality"]["rejected_rows"]) == 1
    assert report["quality"]["usable"]["minute_rows"] == 1
    stored = CandleStore(tmp_path / "candles")._read(RELIANCE)
    assert [bar.start_time for bar in stored] == [later]
    assert (
        _read_snapshot(tmp_path / "daily" / "NSE_RELIANCE.csv")[0].timestamp.date()
        == later.date()
    )


def test_staged_migration_flags_gaps_and_blocks_severe_prices(
    tmp_path, kite_client
) -> None:
    from dataclasses import replace
    from datetime import date

    from ai_trader.broker import OHLCVCandle
    from ai_trader.cli.migrate_kite_data import stage_instrument, validate_history

    candle = OHLCVCandle(
        BEGINNING, Decimal(100), Decimal(101), Decimal(99), Decimal(100), 5
    )
    result = validate_history(
        (candle,),
        (replace(candle, timestamp=BEGINNING.replace(hour=0, minute=0)),),
        cas=True,
    )
    assert result["missing_minutes"] == 359
    assert result["status"] == "validated_with_gaps"
    bad = replace(candle, high=Decimal(10000), close=Decimal(10000))
    assert validate_history((bad,), (candle,), cas=True)["status"] == "blocked"
    entry = {"exchange": "NSE", "symbol": "RELIANCE", "role": "equity_target"}
    broker = KiteBroker(kite_client)
    kite_client.historical_data.side_effect = [
        kite_client.historical_data.return_value,
        [
            {
                **kite_client.historical_data.return_value[0],
                "date": BEGINNING.replace(hour=0, minute=0),
            }
        ],
    ]
    staged = stage_instrument(
        broker, entry, date(2026, 9, 1), date(2026, 9, 1), tmp_path
    )
    assert staged["provider"] == "kite"
    assert staged["quality"]["minute_rows"] == 1
    assert (
        stage_instrument(broker, entry, date(2026, 9, 1), date(2026, 9, 1), tmp_path)
        == staged
    )
    assert kite_client.historical_data.call_count == 2


def test_staging_retries_transient_failures_but_not_expired_sessions(
    monkeypatch,
) -> None:
    from ai_trader.cli import migrate_kite_data as migrate

    delays: list[float] = []
    monkeypatch.setattr(migrate, "sleep", delays.append)
    broker = Mock()
    broker.capture_historical_candles.side_effect = [
        KiteBrokerError("Kite rate limit reached; retry later."),
        "capture",
    ]
    assert (
        migrate._capture(broker, RELIANCE, BEGINNING, BEGINNING, CandleInterval.ONE_DAY)
        == "capture"
    )
    assert delays == [2]
    broker.capture_historical_candles.side_effect = KiteAuthenticationError("expired")
    with pytest.raises(KiteAuthenticationError):
        migrate._capture(broker, RELIANCE, BEGINNING, BEGINNING, CandleInterval.ONE_DAY)
    assert delays == [2]
    broker.capture_historical_candles.side_effect = KiteBrokerError("failed")
    with pytest.raises(KiteBrokerError):
        migrate._capture(broker, RELIANCE, BEGINNING, BEGINNING, CandleInterval.ONE_DAY)
    assert delays == [2, *migrate.RETRY_DELAYS_SECONDS]


def test_staging_quarantines_unknown_sessions_and_honours_chunk_days(
    tmp_path, kite_client
) -> None:
    from datetime import date, time

    from ai_trader.cli.migrate_kite_data import stage_instrument, windows_for

    assert windows_for(date(2021, 11, 4), cas=True) == ((time(18, 15), time(19, 15)),)
    assert windows_for(date(2021, 2, 24), cas=False)[-1] == (time(15, 45), time(17))
    original = kite_client.historical_data.return_value[0]
    later = BEGINNING + timedelta(days=1)
    kite_client.historical_data.side_effect = [
        [original],
        [{**original, "date": later.replace(hour=15, minute=20)}],
        [
            {**original, "date": stamp.replace(hour=0, minute=0)}
            for stamp in (BEGINNING, later)
        ],
    ]
    report = stage_instrument(
        KiteBroker(kite_client),
        {"exchange": "NSE", "symbol": "RELIANCE", "role": "equity_target"},
        date(2026, 9, 1),
        date(2026, 9, 2),
        tmp_path,
        minute_days=1,
    )
    assert kite_client.historical_data.call_count == 3
    assert report["quality"]["outside_session_dates"] == ["2026-09-02"]
    assert report["quality"]["quarantined_dates"] == ["2026-09-02"]
    assert report["quality"]["usable"]["outside_session_rows"] == 0
    assert report["quality"]["usable"]["minute_rows"] == 1


def test_retirement_is_explicit_and_preserves_kite_and_source_code(tmp_path) -> None:
    from ai_trader.cli.migrate_kite_data import retirement_plan

    deleted = (
        "data/.candles.retired-provider/NSE_TEST.csv",
        "data/research/tp_only_20261003/train.npz",
        "data/broker_comparisons/example/groww.csv",
        "backtests/report_baseline_62s.txt",
        "data/regrid/sweep_new.tsv",
    )
    retained = (
        "data/candles/NSE_TEST.csv",
        "data/research/tp_only_kite_20261005/train.npz",
        "data/research/unrelated_user_work.txt",
        "data/regrid/audit_tape.py",
        "data/broker_comparisons/example/kite.csv",
        "backtests/README.md",
    )
    for name in (*deleted, *retained):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    planned = retirement_plan(tmp_path)
    for name in deleted:
        assert any(
            tmp_path / name == path or (tmp_path / name).is_relative_to(path)
            for path in planned
        )
    for name in retained:
        assert not any(
            tmp_path / name == path or (tmp_path / name).is_relative_to(path)
            for path in planned
        )
    assert all((tmp_path / name).exists() for name in (*deleted, *retained))


def test_independent_label_check_catches_boundary_and_timing_errors() -> None:
    import numpy as np

    from ai_trader.cli.migrate_kite_data import check_labels
    from ai_trader.market import Candle

    bars = tuple(
        Candle(
            RELIANCE,
            BEGINNING + timedelta(minutes=minute),
            BEGINNING + timedelta(minutes=minute + 1),
            Decimal(100),
            Decimal("100.2") if minute == 24 else Decimal("100.1"),
            Decimal("99.9"),
            Decimal(100),
            100,
        )
        for minute in range(85)
    )
    moments = np.asarray([int(bars[24].end_time.timestamp())] * 2)
    matrix = np.zeros((2, 14), dtype=np.float32)
    matrix[:, -1] = [1, -1]
    arrays = {
        "timestamps": moments,
        "features": matrix,
        "labels": np.asarray([0, 0]),
        "first_hit_minutes": np.asarray([0, 0]),
        "adverse_excursion": np.asarray([0.001, 0.001]),
        "terminal_return": np.asarray([0.0, 0.0]),
    }
    assert check_labels(bars, arrays) == 2
    arrays["labels"][0] = 1
    with pytest.raises(ValueError, match="TP label"):
        check_labels(bars, arrays)
    arrays["labels"][0] = 0
    with pytest.raises(ValueError, match="missing future"):
        check_labels(bars[:50] + bars[51:], arrays)


def test_promotion_checks_fingerprints_and_keeps_retired_cache(
    tmp_path, kite_client
) -> None:
    import json
    from dataclasses import asdict
    from datetime import date

    from ai_trader.cli.download_history import write_report
    from ai_trader.cli.migrate_kite_data import promote_dataset, stage_instrument
    from ai_trader.history import CandleStore

    stage = tmp_path / "stage"
    destination = tmp_path / "candles"
    destination.mkdir()
    (destination / "old.csv").write_text("old data")
    kite_client.historical_data.side_effect = [
        kite_client.historical_data.return_value,
        [
            {
                **kite_client.historical_data.return_value[0],
                "date": BEGINNING.replace(hour=0, minute=0),
            }
        ],
    ]
    entry = stage_instrument(
        KiteBroker(kite_client),
        {"exchange": "NSE", "symbol": "RELIANCE", "role": "equity_target"},
        date(2026, 9, 1),
        date(2026, 9, 1),
        stage,
    )
    write_report(
        stage / "plan.json", {"provider": "kite", "instruments": [asdict(RELIANCE)]}
    )
    write_report(stage / "universe.json", {"provider": "kite", "instruments": [entry]})
    report = {
        "provider": "kite",
        "status": "staged_and_validated",
        "failures": {},
        "requested_start": "2026-09-01",
        "requested_end": "2026-09-01",
        "instruments": {"NSE:RELIANCE": entry},
    }
    write_report(stage / "report.json", report)
    with pytest.raises(ValueError, match="occupied"):
        promote_dataset(stage, destination)
    original = entry["minute_sha256"]
    entry["minute_sha256"] = "wrong"
    write_report(stage / "report.json", report)
    with pytest.raises(ValueError, match="fingerprint"):
        promote_dataset(stage, destination, replace=True)
    assert (destination / "old.csv").read_text() == "old data"
    entry["minute_sha256"] = original
    write_report(stage / "report.json", report)
    result = promote_dataset(stage, destination, replace=True)
    assert result["instruments"] == 1
    assert (
        tmp_path / ".candles.retired-provider" / "old.csv"
    ).read_text() == "old data"
    assert not (destination / "old.csv").exists()
    assert (
        len(
            CandleStore(destination).load(
                RELIANCE, BEGINNING, BEGINNING + timedelta(minutes=1)
            )
        )
        == 1
    )
    from ai_trader.history import CandleStoreError

    with pytest.raises(CandleStoreError, match="missing minutes"):
        CandleStore(destination).load(
            RELIANCE, BEGINNING, BEGINNING + timedelta(minutes=2)
        )
    assert (
        len(
            CandleStore(destination).load(
                RELIANCE,
                BEGINNING,
                BEGINNING + timedelta(minutes=2),
                allow_incomplete=True,
            )
        )
        == 1
    )
    manifest = json.loads((destination / "dataset.json").read_text())
    assert manifest["session_dates"] == ["2026-09-01"]
    assert manifest["provider"] == "kite"


def test_kite_real_sdk_stream_setup_without_network(monkeypatch) -> None:
    import kiteconnect.ticker as sdk
    from OpenSSL import SSL
    from twisted.internet import ssl

    from ai_trader.broker import kite_stream
    from ai_trader.broker.kite import KiteInstrument
    from ai_trader.broker.kite_stream import KiteLtpStream

    ticker = sdk.KiteTicker("synthetic-key", "synthetic-token", connect_timeout=20)
    resolved = KiteInstrument(RELIANCE, 738561, "2885", Decimal("0.1"))

    def connect_transport(host, port, factory, timeout, bind_address):
        assert host == "ws.kite.trade"
        assert port == 443
        assert factory.wrappedFactory is ticker.factory
        assert timeout == 20
        ticker.ws = Mock()
        ticker.on_connect(ticker, {})

    monkeypatch.setattr(sdk.reactor, "connectTCP", connect_transport)
    monkeypatch.setattr(sdk.reactor, "running", True)
    context_factory = Mock(wraps=ssl.optionsForClientTLS)
    monkeypatch.setattr(ssl, "optionsForClientTLS", context_factory)
    stream = KiteLtpStream(ticker, (resolved,))
    with stream.connect(0.1):
        assert ticker.ws.sendMessage.call_count == 2
        assert ticker.factory.maxRetries == 0
    context_factory.assert_called_once()
    assert context_factory.call_args.args == ("ws.kite.trade",)
    assert context_factory.call_args.kwargs["trustRoot"] is not None
    context = context_factory.call_args.kwargs["trustRoot"]
    options = ssl.optionsForClientTLS("ws.kite.trade", trustRoot=context)
    connection = options.clientConnectionForTLS(Mock())
    assert connection.get_context().get_verify_mode() & SSL.VERIFY_PEER

    empty_trust = Mock()
    empty_trust.get_ca_certs.return_value = []
    monkeypatch.setattr(
        kite_stream.system_ssl, "create_default_context", lambda: empty_trust
    )
    with pytest.raises(kite_stream.KiteStreamError, match="certificate authorities"):
        kite_stream._connect_ticker(ticker)


@pytest.mark.parametrize(
    "endpoint", ["ws://ws.kite.trade", "wss://example.test", "wss://ws.kite.trade:80"]
)
def test_kite_stream_refuses_untrusted_endpoints(endpoint) -> None:
    from ai_trader.broker.kite_stream import KiteStreamError, _connect_ticker

    ticker = Mock(socket_url=endpoint)
    with pytest.raises(KiteStreamError, match="Unexpected"):
        _connect_ticker(ticker)
    ticker._create_connection.assert_not_called()


def test_kite_full_stream_has_volume_and_bounded_lifecycle(monkeypatch) -> None:
    from ai_trader.broker.kite import (
        KiteInstrument,
        KiteStreamConnectionError,
        KiteStreamError,
    )
    from ai_trader.broker.kite_stream import KiteLtpStream

    monkeypatch.setattr(
        "ai_trader.broker.kite_stream._connect_ticker",
        lambda ticker: ticker.connect(threaded=True),
    )
    ticker = Mock(MODE_FULL="full")
    ticker.connect.side_effect = lambda **kwargs: ticker.on_connect(ticker, {})
    resolved = KiteInstrument(RELIANCE, 738561, "2885", Decimal("0.1"))
    stream = KiteLtpStream(ticker, (resolved,)).connect(0.1)
    ticker.subscribe.assert_called_once_with([738561])
    ticker.set_mode.assert_called_once_with("full", [738561])
    local_sdk_stamp = datetime.fromtimestamp(BEGINNING.timestamp())
    row = {
        "instrument_token": 738561,
        "last_price": 100,
        "exchange_timestamp": local_sdk_stamp,
        "volume_traded": 50,
    }
    ticker.on_ticks(ticker, [row])
    ticks = stream.collect(max_ticks=1, timeout_seconds=0.1)
    assert ticks[0].timestamp == BEGINNING
    assert ticks[0].cumulative_volume == 50
    ticker.on_ticks(ticker, [row])
    with pytest.raises(RuntimeError, match="consumer"):
        stream.collect(
            max_ticks=1,
            timeout_seconds=0.1,
            on_tick=Mock(side_effect=RuntimeError("consumer")),
        )
    ticker.on_ticks(ticker, [{**row, "exchange_timestamp": None}])
    with pytest.raises(KiteStreamError, match="schema") as error:
        stream.collect(max_ticks=1, timeout_seconds=0.1)
    assert not isinstance(error.value, KiteStreamConnectionError)
    stream.close()
    stream.close()
    ticker.close.assert_called_once()
    disconnected = KiteLtpStream(Mock(), (resolved,))
    disconnected._on_error(None, 403, "secret-vendor-response")
    with pytest.raises(KiteStreamConnectionError, match="disconnected"):
        disconnected.collect(max_ticks=1, timeout_seconds=0.1)


def test_tolerant_kite_stream_skips_bad_rows_and_counts_overflow(monkeypatch) -> None:
    from ai_trader.broker.kite import KiteInstrument
    from ai_trader.broker.kite_stream import KiteLtpStream

    monkeypatch.setattr(
        "ai_trader.broker.kite_stream._connect_ticker",
        lambda ticker: ticker.connect(threaded=True),
    )
    ticker = Mock(MODE_FULL="full")
    ticker.connect.side_effect = lambda **kwargs: ticker.on_connect(ticker, {})
    resolved = KiteInstrument(RELIANCE, 738561, "2885", Decimal("0.1"))
    stream = KiteLtpStream(ticker, (resolved,), strict=False, queue_size=2).connect(0.1)
    row = {
        "instrument_token": 738561,
        "last_price": 100,
        "exchange_timestamp": datetime.fromtimestamp(BEGINNING.timestamp()),
        "volume_traded": 50,
    }
    ticker.on_ticks(ticker, [{**row, "exchange_timestamp": None}, row, row, row])
    assert stream.rejected == 1 and stream.dropped == 1
    assert len(stream.collect(max_ticks=5, timeout_seconds=0.1)) == 2
    stream.close()


def test_kite_read_only_market_contract(kite_client: Mock) -> None:
    broker = KiteBroker(kite_client)
    kite_client.profile.return_value = {
        "exchanges": ["NSE", "BSE", "NFO"],
        "email": "private@example.test",
        "user_id": "private-user",
    }
    profile = broker.get_user_profile()
    assert profile.active_segments == ("CASH", "FNO")
    assert profile.ddpi_enabled is None
    from ai_trader.cli.check_kite import _profile_summary

    assert _profile_summary(profile)["ddpi_status"] == "unknown"
    assert "private" not in repr(profile)
    kite_client.ltp.return_value = {"NSE:RELIANCE": {"last_price": 102}}
    assert broker.get_ltp((RELIANCE,))[0].price == 102
    kite_client.quote.return_value = {
        "NSE:RELIANCE": {
            "last_trade_time": "2026-09-01 09:15:00",
            "last_price": 102,
            "volume": 50,
            "ohlc": {"open": 100, "high": 103, "low": 99, "close": 100},
        }
    }
    quote = broker.get_quote(RELIANCE)
    assert quote.last_trade_at == BEGINNING
    assert quote.day_change == 2
    assert quote.day_change_percent == 2
    assert quote.previous_close == 100
    kite_client.quote.return_value["NSE:RELIANCE"]["last_trade_time"] = None
    with pytest.raises(KiteBrokerError, match="missing or invalid"):
        broker.get_quote(RELIANCE)
    kite_client.ltp.return_value = {}
    with pytest.raises(KiteBrokerError, match="missing or invalid"):
        broker.get_ltp((RELIANCE,))
    assert not hasattr(broker, "place_order")


@pytest.fixture
def kite_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr("ai_trader.broker.kite.sleep", lambda delay: None)
    client = Mock()
    client.instruments.return_value = [
        {
            "exchange": "NSE",
            "segment": "NSE",
            "instrument_type": "EQ",
            "tradingsymbol": "RELIANCE",
            "instrument_token": 738561,
            "exchange_token": "2885",
            "tick_size": 0.1,
        }
    ]
    client.historical_data.return_value = [
        {
            "date": BEGINNING.isoformat(),
            "open": 1300.1,
            "high": 1302.0,
            "low": 1299.9,
            "close": 1301.2,
            "volume": 120,
        }
    ]
    return client


def test_kite_history_resolves_exact_equity_and_normalizes_ist(
    kite_client: Mock,
) -> None:
    broker = KiteHistoricalBroker(kite_client)
    ending = BEGINNING + timedelta(minutes=5)
    candles = broker.get_historical_candles(
        RELIANCE, BEGINNING.astimezone(UTC), ending, CandleInterval.ONE_MINUTE
    )
    assert len(candles) == 1
    assert candles[0].timestamp == BEGINNING
    assert candles[0].timestamp.tzinfo is INDIA_TIMEZONE
    assert candles[0].close == Decimal("1301.2")
    assert candles[0].volume == 120
    assert broker.resolve_instrument(RELIANCE).tick_size == Decimal("0.1")
    kite_client.instruments.assert_called_once_with("NSE")
    kite_client.historical_data.assert_called_once_with(
        instrument_token=738561,
        from_date="2026-09-01 09:15:00",
        to_date="2026-09-01 09:20:00",
        interval="minute",
        continuous=False,
        oi=False,
    )
    assert not hasattr(broker, "place_order")


def test_kite_history_paces_consecutive_requests(
    kite_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [10.0]
    waits = []

    def wait(delay: float) -> None:
        waits.append(delay)
        now[0] += delay

    monkeypatch.setattr("ai_trader.broker.kite.monotonic", lambda: now[0])
    monkeypatch.setattr("ai_trader.broker.kite.sleep", wait)
    broker = KiteHistoricalBroker(kite_client)
    for elapsed in (0, 0.1, 1):
        now[0] += elapsed
        broker.get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(minutes=1),
            CandleInterval.ONE_MINUTE,
        )
    assert waits == [pytest.approx(0.3)]
    assert kite_client.historical_data.call_count == 3


def test_kite_workers_share_rate_budget_and_resolve_indices(
    kite_client: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trader.broker.kite import HistoryRateLimiter

    clock = [10.0]
    waits = []

    def wait(delay):
        waits.append(delay)
        clock[0] += delay

    monkeypatch.setattr("ai_trader.broker.kite.monotonic", lambda: clock[0])
    monkeypatch.setattr("ai_trader.broker.kite.sleep", wait)
    limiter = HistoryRateLimiter()
    for broker in (
        KiteBroker(kite_client, limiter=limiter),
        KiteBroker(kite_client, limiter=limiter),
    ):
        broker.get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(minutes=1),
            CandleInterval.ONE_MINUTE,
        )
    assert waits == [pytest.approx(0.4)]
    kite_client.instruments.return_value.append(
        {
            "exchange": "NSE",
            "segment": "INDICES",
            "instrument_type": "EQ",
            "tradingsymbol": "NIFTY 50",
            "instrument_token": 256265,
            "exchange_token": "1001",
            "tick_size": 0,
        }
    )
    result = KiteBroker(kite_client).resolve_instrument(Instrument("NSE", "NIFTY"))
    assert result.instrument_type == "INDEX"
    assert result.trading_symbol == "NIFTY 50"
    assert result.tick_size == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("high", 1200),
        ("close", float("nan")),
        ("volume", -1),
        ("volume", 1.5),
        ("date", "2026-09-01T09:15:00"),
        ("date", "2026-09-01T09:15:01+05:30"),
        ("date", "2026-08-31T09:15:00+05:30"),
    ],
)
def test_kite_rejects_bad_candles(kite_client: Mock, field: str, value: object) -> None:
    kite_client.historical_data.return_value[0][field] = value
    with pytest.raises(KiteBrokerError, match="validation"):
        KiteHistoricalBroker(kite_client).get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(minutes=5),
            CandleInterval.ONE_MINUTE,
        )


def test_kite_rejects_duplicate_bars_and_ambiguous_symbols(kite_client: Mock) -> None:
    kite_client.historical_data.return_value *= 2
    with pytest.raises(KiteBrokerError, match="validation"):
        KiteHistoricalBroker(kite_client).get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(minutes=5),
            CandleInterval.ONE_MINUTE,
        )
    kite_client.instruments.return_value *= 2
    with pytest.raises(KiteBrokerError, match="ambiguous"):
        KiteHistoricalBroker(kite_client).resolve_instrument(RELIANCE)


def test_kite_failures_do_not_expose_vendor_messages(kite_client: Mock) -> None:
    kite_client.historical_data.side_effect = RuntimeError("secret-access-token")
    with pytest.raises(KiteBrokerError) as error:
        KiteHistoricalBroker(kite_client).get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(minutes=5),
            CandleInterval.ONE_MINUTE,
        )
    assert "secret-access-token" not in str(error.value)
    assert error.value.__suppress_context__ is True
    token_error = type("TokenException", (Exception,), {})
    kite_client.historical_data.side_effect = token_error("secret-access-token")
    with pytest.raises(KiteAuthenticationError, match="expired"):
        KiteHistoricalBroker(kite_client).get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(minutes=5),
            CandleInterval.ONE_MINUTE,
        )


def test_kite_rejects_naive_and_oversized_requests(kite_client: Mock) -> None:
    broker = KiteHistoricalBroker(kite_client)
    with pytest.raises(ValueError, match="timezone"):
        broker.get_historical_candles(
            RELIANCE,
            BEGINNING.replace(tzinfo=None),
            BEGINNING,
            CandleInterval.ONE_MINUTE,
        )
    with pytest.raises(ValueError, match="window"):
        broker.get_historical_candles(
            RELIANCE,
            BEGINNING,
            BEGINNING + timedelta(days=61),
            CandleInterval.ONE_MINUTE,
        )
    kite_client.historical_data.assert_not_called()


def test_kite_daily_closes_are_separate_from_minute_candles(kite_client: Mock) -> None:
    kite_client.historical_data.return_value[0]["date"] = "2026-09-01T00:00:00+05:30"
    closes = KiteHistoricalBroker(kite_client).get_daily_closes(
        RELIANCE, BEGINNING, BEGINNING + timedelta(hours=1)
    )
    assert closes[0].session_date == BEGINNING.date()
    assert closes[0].close == Decimal("1301.2")
    assert kite_client.historical_data.call_args.kwargs["interval"] == "day"


def test_kite_browser_flow_rejects_wrong_state_and_never_logs_tokens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import threading
    from http.client import HTTPConnection
    from urllib.parse import parse_qs, urlencode, urlsplit

    pytest.importorskip("kiteconnect")
    from http.server import HTTPServer

    from ai_trader.broker import kite_auth
    from ai_trader.config import KiteLoginSettings

    client = Mock()
    client.login_url.return_value = (
        "https://kite.zerodha.com/connect/login?api_key=fake-key&v=3"
    )
    client.generate_session.return_value = {"access_token": "fake-access-token"}
    monkeypatch.setattr("kiteconnect.KiteConnect", Mock(return_value=client))
    address = {}

    class EphemeralServer(HTTPServer):
        def __init__(self, location, handler):
            super().__init__(location, handler)
            address["port"] = self.server_port

    monkeypatch.setattr(kite_auth, "HTTPServer", EphemeralServer)
    responses = []
    workers = []

    def open_browser(url):
        outer = parse_qs(urlsplit(url).query)
        state = parse_qs(outer["redirect_params"][0])["state"][0]

        def callback():
            for supplied_state in ("incorrect-state", state):
                connection = HTTPConnection("127.0.0.1", address["port"], timeout=5)
                query = urlencode(
                    {
                        "state": supplied_state,
                        "request_token": "fake-request-token",
                        "status": "success",
                        "action": "login",
                    }
                )
                connection.request("GET", "/kite/callback?" + query)
                response = connection.getresponse()
                responses.append((response.status, response.read().decode()))
                connection.close()

        worker = threading.Thread(target=callback)
        workers.append(worker)
        worker.start()
        return True

    broker = kite_auth.login_browser(
        KiteLoginSettings(api_key="fake-key", api_secret="fake-secret"),
        timeout_seconds=5,
        callback_port=0,
        open_browser=open_browser,
    )
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    assert [status for status, _ in responses] == [400, 200]
    assert isinstance(broker, KiteHistoricalBroker)
    client.generate_session.assert_called_once_with(
        "fake-request-token", api_secret="fake-secret"
    )
    client.set_access_token.assert_called_once_with("fake-access-token")
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert all("fake-" not in text for _, text in responses)


def test_kite_browser_failure_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("kiteconnect")
    from ai_trader.broker.kite_auth import login_browser
    from ai_trader.config import KiteLoginSettings

    client = Mock()
    client.login_url.return_value = (
        "https://kite.zerodha.com/connect/login?api_key=fake-key&v=3"
    )
    monkeypatch.setattr("kiteconnect.KiteConnect", Mock(return_value=client))
    with pytest.raises(KiteAuthenticationError, match="system browser"):
        login_browser(
            KiteLoginSettings(api_key="fake-key", api_secret="fake-secret"),
            callback_port=0,
            open_browser=lambda url: False,
        )
    client.generate_session.assert_not_called()


def test_broker_comparison_separates_missing_bars_and_value_differences() -> None:
    from dataclasses import replace

    from ai_trader.broker import OHLCVCandle
    from ai_trader.cli.compare_broker_history import compare_candles

    first = OHLCVCandle(
        BEGINNING, Decimal(100), Decimal(101), Decimal(99), Decimal(100), 10
    )
    second = replace(first, timestamp=BEGINNING + timedelta(minutes=1), volume=None)
    groww_only = replace(first, timestamp=BEGINNING + timedelta(minutes=2))
    kite_only = replace(first, timestamp=BEGINNING + timedelta(minutes=3))
    report, differences = compare_candles(
        (first, second, groww_only),
        (first, replace(second, close=Decimal("100.1"), volume=20), kite_only),
    )
    assert report["common_timestamps"] == 2
    assert report["all_ohlcv_exact_bars"] == 1
    assert report["groww_only_timestamps"] == report["kite_only_timestamps"] == 1
    assert report["fields"]["close"]["maximum_absolute_difference"] == "0.1"
    assert report["fields"]["volume"]["missing_value_pairs"] == 1
    assert {row["status"] for row in differences} == {
        "different",
        "groww_only",
        "kite_only",
    }
    assert report["daily"]["2026-09-01"]["groww"]["regular_bars"] == 3
    assert report["daily"]["2026-09-01"]["groww"]["missing_regular_minutes"] == 372
    with pytest.raises(ValueError, match="duplicate"):
        compare_candles((first, first), (first,))
    unknown_volume, _ = compare_candles((second,), (second,))
    assert unknown_volume["all_ohlc_exact_bars"] == 1
    assert unknown_volume["all_ohlcv_exact_bars"] == 0
    assert unknown_volume["fields"]["volume"]["compared"] == 0


def test_broker_comparison_detects_a_timestamp_offset() -> None:
    from dataclasses import replace

    from ai_trader.broker import OHLCVCandle
    from ai_trader.cli.compare_broker_history import compare_candles

    candles = tuple(
        OHLCVCandle(
            BEGINNING + timedelta(minutes=index),
            Decimal(100 + index),
            Decimal(101 + index),
            Decimal(99 + index),
            Decimal(100 + index),
            10,
        )
        for index in range(3)
    )
    shifted = tuple(
        replace(candle, timestamp=candle.timestamp + timedelta(minutes=1))
        for candle in candles
    )
    report, _ = compare_candles(candles, shifted)
    offsets = report["offset_check_groww_t_vs_kite_t_plus_minutes"]
    assert offsets["1"]["close_exact"] == 3
    assert offsets["0"]["close_exact"] == 0


def test_broker_comparison_distinguishes_continuous_session_from_cas() -> None:
    from datetime import time

    from ai_trader.broker import OHLCVCandle
    from ai_trader.cli.compare_broker_history import compare_candles

    continuous = tuple(
        OHLCVCandle(
            BEGINNING + timedelta(minutes=offset),
            Decimal(100),
            Decimal(101),
            Decimal(99),
            Decimal(100),
            10,
        )
        for offset in range(360)
    )
    auction = OHLCVCandle(
        BEGINNING.replace(hour=15, minute=29),
        Decimal(100),
        Decimal(101),
        Decimal(99),
        Decimal(100),
        200,
    )
    report, differences = compare_candles(
        (*continuous, auction), continuous, session_close=time(15, 15)
    )
    assert report["session"]["expected_minutes_per_observed_date"] == 360
    assert report["kite_regular_bars"] == report["groww_regular_bars"] == 360
    assert report["groww_outside_regular_session"] == 1
    assert report["daily"]["2026-09-01"]["kite"]["missing_regular_minutes"] == 0
    assert differences == []
    legacy, _ = compare_candles(continuous, continuous, session_close=time(15, 30))
    assert legacy["daily"]["2026-09-01"]["kite"]["missing_regular_minutes"] == 15
    with pytest.raises(ValueError, match="verified"):
        compare_candles(continuous, continuous, session_close=time(15))


def test_legacy_comparison_refuses_to_relabel_the_kite_cache(tmp_path) -> None:
    import json
    from datetime import time

    from ai_trader.cli.compare_broker_history import (
        run_comparison,
        run_monthly_comparisons,
    )

    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "provider": "kite",
                "schema_version": 1,
                "files": {},
            }
        )
    )
    broker = Mock()
    with pytest.raises(ValueError, match="cannot be labeled"):
        run_comparison(broker, RELIANCE, "2026-09", tmp_path, tmp_path / "output")
    with pytest.raises(ValueError, match="cannot be labeled"):
        run_monthly_comparisons(
            broker,
            RELIANCE,
            "2026-09",
            "2026-09",
            tmp_path,
            tmp_path / "output",
            session_closes={"2026-09": time(15, 15)},
        )
    assert not broker.mock_calls
    assert not (tmp_path / "output").exists()


def test_broker_comparison_keeps_the_source_cache_unchanged(
    tmp_path, kite_client: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ai_trader.cli import compare_broker_history
    from ai_trader.cli.compare_broker_history import _digest, run_comparison
    from ai_trader.history import CandleStore
    from ai_trader.market import Candle

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare_broker_history, "version", lambda name: "test-sdk")
    cache = tmp_path / "groww"
    store = CandleStore(cache)
    candle = Candle(
        RELIANCE,
        BEGINNING,
        BEGINNING + timedelta(minutes=1),
        Decimal("1300.1"),
        Decimal(1302),
        Decimal("1299.9"),
        Decimal("1301.2"),
        120,
    )
    store._write(RELIANCE, (candle,))
    source = store.path_for(RELIANCE)
    digest = _digest(source)
    kite_client.historical_data.side_effect = [
        kite_client.historical_data.return_value,
        [],
    ]
    output = tmp_path / "comparison"
    report = run_comparison(
        KiteHistoricalBroker(kite_client), RELIANCE, "2026-09", cache, output
    )
    assert report["comparison"]["common_timestamps"] == 1
    assert report["comparison"]["all_ohlcv_exact_bars"] == 1
    assert _digest(source) == digest
    assert not store.fetched_path_for(RELIANCE).exists()
    assert all(
        (output / name).is_file()
        for name in ("groww.csv", "kite.csv", "differences.csv", "report.json")
    )
    assert kite_client.historical_data.call_count == 2
    requests = kite_client.historical_data.call_args_list
    assert requests[0].kwargs["from_date"] == "2026-09-01 09:15:00"
    assert requests[0].kwargs["to_date"] == "2026-09-29 09:14:00"
    assert requests[1].kwargs["from_date"] == "2026-09-29 09:15:00"
    assert requests[1].kwargs["to_date"] == "2026-09-30 15:29:00"
    with pytest.raises(ValueError, match="already exists"):
        run_comparison(
            KiteHistoricalBroker(kite_client), RELIANCE, "2026-09", cache, output
        )
    original_report_hash = _digest(output / "report.json")
    login = Mock(side_effect=AssertionError("Offline comparison attempted login."))
    monkeypatch.setattr(compare_broker_history, "login_browser", login)
    monkeypatch.setattr(compare_broker_history, "load_kite_login_settings", login)
    monkeypatch.setattr(compare_broker_history, "load_kite_settings", login)
    revised = tmp_path / "revised"
    assert (
        compare_broker_history.main(
            [
                "--saved-comparison",
                str(output),
                "--output",
                str(revised),
                "--month",
                "2026-09",
                "--session-close",
                "15:15",
            ]
        )
        == 0
    )
    login.assert_not_called()
    assert _digest(source) == digest
    assert _digest(output / "report.json") == original_report_hash
    assert _digest(revised / "kite.csv") == _digest(output / "kite.csv")
    import json

    revised_report = json.loads((revised / "report.json").read_text(encoding="utf-8"))
    assert revised_report["parent_report_sha256"] == original_report_hash
    assert revised_report["comparison"]["session"]["close_exclusive"] == "15:15"
    assert (
        revised_report["comparison"]["daily"]["2026-09-01"]["kite"][
            "missing_regular_minutes"
        ]
        == 359
    )
    (output / "kite.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        compare_broker_history.recompare_saved(
            output,
            tmp_path / "rejected",
            RELIANCE,
            "2026-09",
            session_close=BEGINNING.replace(hour=15, minute=15).time(),
        )


def test_history_client_keeps_iso_dates_for_fast_adapter_parsing() -> None:
    pytest.importorskip("kiteconnect")
    from ai_trader.broker.kite import _history_client

    client = _history_client("fake-key", "fake-access-token")
    rows = client._format_historical(
        {"candles": [["2026-09-01T09:15:00+0530", 1300.1, 1302.0, 1299.9, 1301.2, 120]]}
    )
    assert rows == [
        {
            "date": "2026-09-01T09:15:00+0530",
            "open": 1300.1,
            "high": 1302.0,
            "low": 1299.9,
            "close": 1301.2,
            "volume": 120,
        }
    ]
    sdk = Mock()
    sdk.instruments.return_value = [
        {
            "exchange": "NSE",
            "segment": "NSE",
            "instrument_type": "EQ",
            "tradingsymbol": "RELIANCE",
            "instrument_token": 738561,
            "exchange_token": "2885",
            "tick_size": 0.1,
        }
    ]
    sdk.historical_data.return_value = rows
    candles = KiteHistoricalBroker(sdk).get_historical_candles(
        RELIANCE, BEGINNING, BEGINNING + timedelta(minutes=1), CandleInterval.ONE_MINUTE
    )
    assert candles[0].timestamp == BEGINNING
    assert candles[0].timestamp.isoformat() == "2026-09-01T09:15:00+05:30"


def test_kite_optional_sdk_missing_is_actionable_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from ai_trader.broker.kite_auth import login_browser
    from ai_trader.config import KiteLoginSettings, KiteSettings

    monkeypatch.setitem(sys.modules, "kiteconnect", None)
    with pytest.raises(KiteBrokerError, match="declared Kite Connect dependency"):
        KiteHistoricalBroker.from_session(
            KiteSettings(api_key="fake-key", access_token="fake-access-token")
        )
    with pytest.raises(
        KiteAuthenticationError, match="declared Kite Connect dependency"
    ):
        login_browser(KiteLoginSettings(api_key="fake-key", api_secret="fake-secret"))


def test_kite_cli_missing_configuration_does_not_open_browser(
    monkeypatch, capsys, tmp_path
) -> None:
    from ai_trader.cli import compare_broker_history as cli
    from ai_trader.config import ConfigurationError

    loader = Mock(
        side_effect=ConfigurationError(
            "Missing required environment variables: KITE_API_SECRET"
        )
    )
    login = Mock()
    monkeypatch.setattr(cli, "load_kite_login_settings", loader)
    monkeypatch.setattr(cli, "login_browser", login)
    assert (
        cli.main(
            [
                "--month",
                "2026-09",
                "--session-close",
                "15:15",
                "--output",
                str(tmp_path / "result"),
                "--browser-login",
            ]
        )
        == 1
    )
    login.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "KITE_API_SECRET" in captured.err


def test_kite_monthly_range_is_resumable_and_preserves_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch, kite_client: Mock
) -> None:
    from datetime import time

    from ai_trader.cli import compare_broker_history as cli
    from ai_trader.history import CandleStore
    from ai_trader.market import Candle

    assert cli.months_inclusive("2023-12", "2024-02") == (
        "2023-12",
        "2024-01",
        "2024-02",
    )
    with pytest.raises(ValueError, match="first month"):
        cli.months_inclusive("2024-02", "2024-01")
    monkeypatch.setattr(cli, "version", lambda name: "test-sdk")
    cache = tmp_path / "groww"
    store = CandleStore(cache)
    starts = tuple(BEGINNING.replace(year=2024, month=month) for month in (1, 2))
    store._write(
        RELIANCE,
        tuple(
            Candle(
                RELIANCE,
                stamp,
                stamp + timedelta(minutes=1),
                Decimal(100),
                Decimal(101),
                Decimal(99),
                Decimal(100),
                10,
            )
            for stamp in starts
        ),
    )
    source_hash = cli._digest(store.path_for(RELIANCE))

    def history(**kwargs):
        return [
            {
                "date": datetime.fromisoformat(kwargs["from_date"]).replace(
                    tzinfo=INDIA_TIMEZONE
                ),
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "volume": 10,
            }
        ]

    kite_client.historical_data.side_effect = history
    output = tmp_path / "range"
    closes = {"2024-01": time(15, 30), "2024-02": time(15, 15)}
    broker = KiteHistoricalBroker(kite_client)
    reports = cli.run_monthly_comparisons(
        broker,
        RELIANCE,
        "2024-01",
        "2024-02",
        cache,
        output,
        session_closes=closes,
    )
    assert list(reports) == ["2024-01", "2024-02"]
    assert kite_client.historical_data.call_count == 4
    assert (
        cli.run_monthly_comparisons(
            broker,
            RELIANCE,
            "2024-01",
            "2024-02",
            cache,
            output,
            session_closes=closes,
        )
        == reports
    )
    assert kite_client.historical_data.call_count == 4
    assert cli._digest(store.path_for(RELIANCE)) == source_hash
    assert not store.fetched_path_for(RELIANCE).exists()
    daily = cli.download_daily_reference(broker, RELIANCE, "2024-01", "2024-02", output)
    assert daily["interval"] == "1d"
    assert kite_client.historical_data.call_count == 5
    assert (
        cli.download_daily_reference(broker, RELIANCE, "2024-01", "2024-02", output)
        == daily
    )
    assert kite_client.historical_data.call_count == 5
    combined = cli.combine_monthly_comparisons(output, reports)
    assert combined["comparison"]["common_timestamps"] == 2
    assert combined["comparison"]["kite_regular_bars"] == 4
    assert combined["comparison"]["fields"]["close"]["exact"] == 2
    assert len(cli._read_snapshot(output / "combined" / "kite.csv")) == 4
    assert cli.combine_monthly_comparisons(output, reports) == combined
    monkeypatch.setattr(cli, "load_kite_settings", Mock(return_value=object()))
    monkeypatch.setattr(
        cli.KiteHistoricalBroker, "from_session", Mock(return_value=broker)
    )
    assert (
        cli.main(
            [
                "--existing-session",
                "--start-month",
                "2024-01",
                "--end-month",
                "2024-02",
                "--session-close",
                "15:30",
                "--session-change-month",
                "2024-02",
                "--changed-session-close",
                "15:15",
                "--groww-cache",
                str(cache),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert kite_client.historical_data.call_count == 5
    with pytest.raises(ValueError, match="plan"):
        cli.run_monthly_comparisons(
            broker,
            RELIANCE,
            "2024-01",
            "2024-02",
            cache,
            output,
            session_closes={month: time(15, 30) for month in closes},
        )
    (output / "months" / "2024-01" / "kite.csv").write_text("tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        cli.run_monthly_comparisons(
            broker,
            RELIANCE,
            "2024-01",
            "2024-02",
            cache,
            output,
            session_closes=closes,
        )


def test_nse_benchmark_checks_instrument_date_and_numeric_schema() -> None:
    from datetime import date

    from scripts.audit_broker_history import native_factor, parse_nse_day

    header = (
        "SYMBOL, SERIES, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
        "CLOSE_PRICE, LAST_PRICE, TTL_TRD_QNTY\n"
    )
    payload = (
        header + "RELIANCE, EQ, 01-Jan-2024, 100, 102, 99, 101, 100, 20\n"
    ).encode()
    result = parse_nse_day(payload, date(2024, 1, 1))
    assert result["close"] == "101"
    assert result["volume"] == "20"
    with pytest.raises(ValueError, match="date"):
        parse_nse_day(payload, date(2024, 1, 2))
    with pytest.raises(ValueError, match="exactly one"):
        parse_nse_day(payload.replace(b"RELIANCE", b"OTHER"), date(2024, 1, 1))
    with pytest.raises(ValueError, match="invalid"):
        parse_nse_day(payload.replace(b"102", b"NaN"), date(2024, 1, 1))
    assert native_factor("kite", date(2024, 10, 25)) == 2
    assert native_factor("kite", date(2024, 10, 28)) == 1
    assert native_factor("groww", date(2024, 10, 25)) == 1


def test_history_quality_normalizes_bonus_and_checks_exchange_bounds() -> None:
    from dataclasses import replace
    from datetime import date

    from ai_trader.broker import OHLCVCandle
    from scripts.audit_broker_history import (
        audit_coverage,
        benchmark_nse,
        is_continuous,
        normalized_agreement,
        session_windows,
    )

    stamp = BEGINNING.replace(year=2024, month=1)
    raw = OHLCVCandle(stamp, Decimal(100), Decimal(102), Decimal(99), Decimal(101), 20)
    adjusted = OHLCVCandle(
        stamp, Decimal(50), Decimal(51), Decimal("49.5"), Decimal("50.5"), 40
    )
    agreement = normalized_agreement((raw,), (adjusted,))
    assert agreement["metrics"]["all"]["fields"]["close"]["exact"] == 1
    assert agreement["metrics"]["all"]["fields"]["volume"]["exact"] == 1
    reference = {
        "date": "2024-01-01",
        "open": "100",
        "high": "102",
        "low": "99",
        "close": "101",
        "last": "100.9",
        "volume": "20",
    }
    bad = replace(raw, high=Decimal(10200), close=Decimal(10100))
    report, rows = benchmark_nse((bad,), (adjusted,), (adjusted,), (reference,))
    assert report["kite_daily_vs_nse"]["close"]["exact"] == 1
    assert report["kite_daily_vs_nse"]["volume"]["exact"] == 1
    assert report["minute_outside_official_daily_range"][0]["provider"] == "groww"
    assert rows[0]["kite_last_close_native"] == "101.0"
    assert len(session_windows(date(2024, 3, 2))) == 2
    evening = replace(raw, timestamp=stamp.replace(month=11, hour=18))
    assert is_continuous(evening)
    assert not is_continuous(
        replace(evening, timestamp=evening.timestamp.replace(hour=9))
    )
    coverage = audit_coverage((), (evening,), (evening,), ())
    assert coverage["expected_minutes"] == 60
    assert coverage["providers"]["kite"]["missing_minutes"] == 59
    assert coverage["providers"]["groww"]["empty_sessions"] == 1


def test_saved_quality_audit_cli_never_authenticates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys

    from ai_trader.broker import OHLCVCandle
    from ai_trader.cli.compare_broker_history import (
        _digest,
        _save_comparison,
        _write_snapshot,
    )
    from scripts import audit_broker_history as audit

    source = tmp_path / "source"
    evidence = tmp_path / "evidence"
    output = tmp_path / "quality"
    candle = OHLCVCandle(
        BEGINNING, Decimal(100), Decimal(102), Decimal(99), Decimal(101), 20
    )
    _save_comparison(
        source / "combined",
        (candle,),
        (candle,),
        [],
        {
            "instrument": {"exchange": "NSE", "trading_symbol": "RELIANCE"},
            "groww_source_sha256": "fixture",
        },
    )
    _write_snapshot(source / "kite_daily.csv", (candle,))
    (source / "kite_daily.json").write_text(
        json.dumps(
            {
                "sha256": _digest(source / "kite_daily.csv"),
            }
        )
    )
    (evidence / "nse").mkdir(parents=True)
    (evidence / "nse" / "2026-09-01.csv").write_text(
        "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,LAST_PRICE,TTL_TRD_QNTY\n"
        "RELIANCE,EQ,01-Sep-2026,100,102,99,101,101,20\n"
    )
    parent = _digest(source / "combined" / "report.json")
    (evidence / "evidence.json").write_text(
        json.dumps(
            {
                "source_report_sha256": parent,
                "selected_dates": ["2026-09-01"],
            }
        )
    )
    forbidden = Mock(
        side_effect=AssertionError("Offline audit accessed credentials/network")
    )
    monkeypatch.setattr(audit, "capture_groww_probes", forbidden)
    monkeypatch.setattr(audit, "urlopen", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_broker_history",
            "--source",
            str(source),
            "--output",
            str(output),
            "--evidence",
            str(evidence),
        ],
    )
    assert audit.main() == 0
    forbidden.assert_not_called()
    report = json.loads((output / "report.json").read_text())
    assert report["exchange_benchmark"]["sample_days"] == 1
    assert report["exchange_benchmark"]["minute_outside_official_daily_range"] == []
    assert report["source_report_sha256"] == parent
    assert _digest(source / "combined" / "report.json") == parent
    with pytest.raises(ValueError, match="new audit"):
        audit.main()
