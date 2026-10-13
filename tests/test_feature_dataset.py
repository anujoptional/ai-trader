from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.features import FeatureEngine
from ai_trader.features.tp_only import (
    MODEL_COLUMNS,
    CompletedSession,
    PriceContext,
    TpFeatureState,
    touch_label,
)
from ai_trader.market import Candle
from scripts.analyze_feature_dataset import (
    FEATURE_COLUMNS,
    dataset_row,
    diagnostic_event,
    feature_vector,
    summarize_outcomes,
)


def candles():
    start = datetime(2026, 7, 1, 10, 29, tzinfo=INDIA_TIMEZONE)
    return tuple(
        Candle(
            Instrument("NSE", "RELIANCE"),
            start + timedelta(minutes=minute),
            start + timedelta(minutes=minute + 1),
            Decimal(100),
            Decimal("100.1"),
            Decimal("99.9"),
            Decimal(100),
            100,
        )
        for minute in range(62)
    )


def test_dataset_label_has_symmetric_barriers_and_a_full_horizon():
    bars = candles()
    future = list(bars[1:])
    future[1] = replace(future[1], high=Decimal("100.3"))
    long = diagnostic_event(bars[0], future, 1)
    short = diagnostic_event(bars[0], future, -1)
    assert long["label_event"] == "TP"
    assert short["label_event"] == "SL"
    assert long["label_class"] == 0
    assert short["label_class"] == 4
    assert long["label_event_minutes"] == 2
    assert Decimal(long["label_tp_price"]) - 100 == 100 - Decimal(
        long["label_sl_price"]
    )
    assert [long[f"label_tp_{horizon}m"] for horizon in (6, 15, 30, 60)] == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="missing minute"):
        diagnostic_event(bars[0], future[:10] + future[11:], 1)


def test_dataset_labels_exclude_entry_bar_and_resolve_double_touch_as_stop():
    bars = candles()
    future = list(bars[1:])
    future[0] = replace(future[0], high=Decimal(101), low=Decimal(99))
    assert diagnostic_event(bars[0], future, 1)["label_event"] == "TIMEOUT"
    future[1] = replace(future[1], high=Decimal(101), low=Decimal(99))
    for side in (1, -1):
        result = diagnostic_event(bars[0], future, side)
        assert result["label_event"] == "SL"
        assert result["label_double_touch"]


def test_dataset_inputs_reject_future_context_and_mark_stale_or_missing_inputs():
    snapshot = FeatureEngine().update(candles()[0])
    assert snapshot is not None
    context = {
        "NIFTY": {
            "as_of": snapshot.candle_end_time + timedelta(minutes=1),
            "return_15": Decimal("0.01"),
        }
    }
    with pytest.raises(ValueError, match="Future context"):
        feature_vector(snapshot, context)
    context["NIFTY"]["as_of"] = snapshot.candle_end_time - timedelta(minutes=1)
    values = feature_vector(snapshot, context)
    assert values["market_return_15"] is None
    assert values["market_excess_return_15"] is None
    first = dataset_row(snapshot, context, {"label_event": "TP"}, 1)
    second = dataset_row(snapshot, context, {"label_event": "SL"}, 1)
    assert all(first[name] == second[name] for name in FEATURE_COLUMNS)
    assert first["missing_market_return_15"] == 1


def test_empty_outcome_group_has_undefined_rates_not_zero_probability():
    result = summarize_outcomes([])
    assert result["actions"] == 0
    assert all(value is None for value in result["tp_rate"].values())


def test_shortened_future_cannot_be_labeled_as_a_sixty_minute_timeout():
    bars = candles()
    with pytest.raises(ValueError, match="Incomplete future"):
        diagnostic_event(bars[0], bars[1:30], 1)


def test_tp_only_counts_first_future_bar_and_both_sides_despite_adverse_moves():
    bars = candles()
    future = list(bars[1:])
    future[0] = replace(future[0], high=Decimal(101), low=Decimal(99))
    assert touch_label(bars[0], future, 1).first_hit_minute == 1
    assert touch_label(bars[0], future, -1).first_hit_minute == 1
    future[0] = replace(future[0], high=Decimal("100.1"))
    future[59] = replace(future[59], high=Decimal("100.2"))
    result = touch_label(bars[0], future, 1)
    assert result.hit and result.first_hit_minute == 60
    assert result.adverse_excursion == Decimal("0.01")
    assert not touch_label(bars[0], bars[1:], 1).hit
    with pytest.raises(ValueError, match="missing minute"):
        touch_label(bars[0], future[:10] + future[11:], 1)
    with pytest.raises(ValueError, match="Incomplete future"):
        touch_label(bars[0], future[:59], 1)


def tp_session(day: date, *, missing: int | None = None):
    start = datetime.combine(day, datetime.min.time(), INDIA_TIMEZONE).replace(
        hour=9, minute=15
    )
    return tuple(
        Candle(
            Instrument("NSE", "RELIANCE"),
            start + timedelta(minutes=minute),
            start + timedelta(minutes=minute + 1),
            Decimal(100),
            Decimal(105),
            Decimal(99),
            Decimal(100) + Decimal(minute + 1) / 100,
            100 + minute,
        )
        for minute in range(360)
        if minute != missing
    )


def test_tp_history_uses_six_dated_closes_but_only_360_cutoff_minutes():
    records = []
    for day_number in range(1, 7):
        day = date(2026, 6, day_number)
        state = TpFeatureState(Instrument("NSE", "RELIANCE"), day)
        for bar in tp_session(day):
            state.update(bar)
        record = state.finish(Decimal(99 + day_number))
        assert record.cutoff_variance is not None
        records.append(record)
    today = date(2026, 6, 7)
    state = TpFeatureState(Instrument("NSE", "RELIANCE"), today, records)
    bars = tp_session(today)
    for bar in bars[:25]:
        state.update(bar)
    context = {
        name: PriceContext(bars[24].end_time, Decimal(15), Decimal("0.01"))
        for name in ("NIFTY", "GOLDBEES", "INDIAVIX")
    }
    values = state.values(context, benchmark="NIFTY")
    assert tuple(values) == MODEL_COLUMNS[:-1]
    assert values["prior_5_daily_close_return"] == Decimal("0.05")
    assert values["gap_from_vendor_daily_close"] == Decimal(100) / 105 - 1
    assert values["same_minute_rvol_5"] == 1
    assert values["minutes_from_open"] == 25
    assert values["log_return_5m"] == (Decimal("100.25") / Decimal("100.20")).ln()
    assert (
        values["log_return_previous_10m"]
        == (Decimal("100.20") / Decimal("100.10")).ln()
    )
    assert all(value is not None for value in values.values())
    assert not {"symbol", "instrument", "tp_fraction", "horizon"} & set(MODEL_COLUMNS)
    context["NIFTY"] = replace(context["NIFTY"], end_time=bars[25].end_time)
    with pytest.raises(ValueError, match="Future context"):
        state.values(context, benchmark="NIFTY")


def test_tp_minute_gap_invalidates_variance_not_independent_daily_close():
    day = date(2026, 6, 1)
    state = TpFeatureState(Instrument("NSE", "RELIANCE"), day)
    for bar in tp_session(day, missing=100):
        state.update(bar)
    record = state.finish(Decimal(102))
    assert record.daily_close == 102
    assert record.cutoff_variance is None
    assert record.volumes[100] is None
    values = state.values({}, benchmark="NIFTY")
    assert values["realized_volatility_20m"] is not None
    assert values["vwap_z"] is None
    assert values["day_return"] is not None
    with pytest.raises(ValueError, match="strictly in the past"):
        TpFeatureState(Instrument("NSE", "RELIANCE"), day, [record])


def test_kite_daily_snapshots_are_derived_and_checked_against_validated_source(
    tmp_path,
):
    import json

    from ai_trader.history import CandleStore
    from scripts.prepare_tp_only import daily_closes, sha256

    cache = tmp_path / "cache"
    (cache / "daily").mkdir(parents=True)
    daily = cache / "daily" / "NSE_RELIANCE.csv"
    daily.write_text(
        "timestamp,close\n2026-06-01T00:00:00+05:30,102\n2026-06-03T00:00:00+05:30,104\n"
    )
    (cache / "dataset.json").write_text(
        json.dumps(
            {
                "provider": "kite",
                "schema_version": 1,
                "session_dates": ["2026-06-01", "2026-06-02", "2026-06-03"],
                "files": {"daily/NSE_RELIANCE.csv": sha256(daily)},
            }
        )
    )
    store = CandleStore(cache)
    instrument = Instrument("NSE", "RELIANCE")
    start, end = date(2026, 6, 1), date(2026, 6, 3)
    root = tmp_path / "daily_snapshots"
    values = daily_closes(instrument, start, end, root, None, store=store)
    assert values == {start: Decimal(102), end: Decimal(104)}
    assert "2026-06-02" in store.dataset["session_dates"]
    snapshot = root / "NSE_RELIANCE.json"
    document = json.loads(snapshot.read_text())
    document["closes"][0]["close"] = "999"
    snapshot.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="validated Kite"):
        daily_closes(instrument, start, end, root, None, store=store)


def test_tp_missing_prior_session_is_not_replaced_with_an_older_complete_one():
    prior = [CompletedSession(date(2026, 6, day), Decimal(100)) for day in range(1, 7)]
    prior[-1] = CompletedSession(date(2026, 6, 6))
    state = TpFeatureState(Instrument("NSE", "RELIANCE"), date(2026, 6, 7), prior)
    for bar in tp_session(date(2026, 6, 7))[:25]:
        state.update(bar)
    values = state.values({}, benchmark="NIFTY")
    assert values["gap_from_vendor_daily_close"] is None
    assert values["prior_5_daily_close_return"] is None


def test_tp_feature_prefix_is_unchanged_by_future_candle_values():
    day = date(2026, 6, 1)
    bars = tp_session(day)
    state = TpFeatureState(bars[0].instrument, day)
    for bar in bars[:25]:
        state.update(bar)
    first = state.values({}, benchmark="NIFTY")
    changed_future = [replace(bar, close=Decimal(99)) for bar in bars[25:]]
    assert changed_future[0].close != bars[25].close
    other = TpFeatureState(bars[0].instrument, day)
    for bar in (*bars[:25], *changed_future)[:25]:
        other.update(bar)
    assert other.values({}, benchmark="NIFTY") == first


def test_tp_dataset_pairs_sides_and_excludes_incomplete_future():
    from scripts.prepare_tp_only import instrument_rows, phase_for

    calendar = [date(2026, 6, day) for day in range(1, 9)]
    grouped = {day: tp_session(day) for day in calendar}
    closes = {day: Decimal(102) for day in calendar}
    context = {
        name: {
            bar.end_time: PriceContext(bar.end_time, Decimal(15), Decimal("0.01"))
            for bars in grouped.values()
            for bar in bars
        }
        for name in ("NIFTY", "GOLDBEES", "INDIAVIX")
    }
    instrument = Instrument("NSE", "RELIANCE")
    rows, report = instrument_rows(instrument, calendar, grouped, closes, context)
    evaluation = rows["evaluation"]
    assert len(evaluation) == 2 * 2 * 56
    assert report["counts"]["included_decisions"] == 112
    for long_row, short_row in zip(evaluation[::2], evaluation[1::2]):
        assert long_row[2] == short_row[2]
        assert long_row[0][:-1] == short_row[0][:-1]
        assert long_row[0][-1] == 1 and short_row[0][-1] == -1
    grouped[calendar[-1]] = grouped[calendar[-1]][:310]
    shorter, exclusions = instrument_rows(
        instrument, calendar, grouped, closes, context
    )
    assert len(shorter["evaluation"]) < len(evaluation)
    assert exclusions["counts"]["excluded_incomplete_future"] > 0
    assert phase_for(date(2025, 6, 30)) == "train"
    assert phase_for(date(2025, 7, 1)) == "validation"
