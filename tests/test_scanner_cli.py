"""Tests for the scanner command-line check.

Every Groww call is mocked, for the same reason as in the feature tests: the
scanner is pure arithmetic over snapshots, so a test asserting the broker is
never touched after the session fetch is testing the design rather than the
mock.

Two fixture sessions, because the interesting property of this layer is what it
*declines* to produce. A quiet session that oscillates in a narrow band must
produce nothing while reporting a high ``considered`` — the market had nothing
to say, and the engine was working. A session that opens quietly and then trends
away must produce candidates. A check that only ever saw one of the two could
not tell a working scanner from a silent one.
"""

import csv
import json
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytest import CaptureFixture

from ai_trader.broker import MarketQuote, MarketTick, OHLCVCandle
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBrokerError,
    GrowwStreamConnectionError,
)
from ai_trader.cli.check_scanner import _CSV_HEADER, _RELIANCE, main
from ai_trader.config import ConfigurationError

_TRADING_DATE = date(2026, 9, 11)

_SESSION_OPEN = datetime(2026, 9, 11, 3, 45, tzinfo=UTC)
"""09:15 IST on Friday 11 September 2026."""

_FULL_SESSION = 60
"""Candles enough for every feature, the fifty-period EMA included."""

_OPENING_RANGE = 15
"""Minutes the opening-range feature spans before it freezes."""

_BASE = Decimal(100)
_ONE = Decimal("1")


def _candle(minutes: int, close: Decimal) -> OHLCVCandle:
    """One broker candle ``minutes`` into the session, timestamped at its open."""
    return OHLCVCandle(
        timestamp=_SESSION_OPEN + timedelta(minutes=minutes),
        open=close,
        high=close + _ONE,
        low=close - _ONE,
        close=close,
        volume=1_000 + minutes,
    )


def _quiet(count: int) -> tuple[OHLCVCandle, ...]:
    """A session cycling through a narrow band: nothing worth a trade.

    The cycle keeps every close inside the twenty-candle range and inside the
    opening range, so no rule can fire and the scanner's silence is the market's
    rather than the engine's.
    """
    return tuple(_candle(minute, _BASE + minute % 7) for minute in range(count))


def _trending(count: int) -> tuple[OHLCVCandle, ...]:
    """A session that opens flat and then walks away from its opening range.

    Flat for the opening range so the frozen high is exactly ``_BASE + 1``, then
    one rupee a minute, which puts the close above that high from the second
    minute of the walk onwards.
    """
    return tuple(
        _candle(
            minute,
            _BASE
            if minute < _OPENING_RANGE
            else _BASE + Decimal(minute - _OPENING_RANGE + 1),
        )
        for minute in range(count)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run(
    broker: Mock,
    candles: tuple[OHLCVCandle, ...],
    argv: Sequence[str] = (),
) -> int:
    with (
        patch("ai_trader.cli.check_scanner.load_groww_settings"),
        patch(
            "ai_trader.cli.check_scanner.GrowwBroker.authenticate",
            return_value=broker,
        ),
        patch(
            "ai_trader.cli.check_scanner._find_recent_completed_session",
            return_value=(_TRADING_DATE, candles),
        ),
    ):
        return main(argv)


def _summary(
    capsys: CaptureFixture[str],
    candles: tuple[OHLCVCandle, ...] = (),
    argv: Sequence[str] = (),
) -> dict:
    """Run against ``candles`` and return the parsed summary, asserting success."""
    exit_code = _run(Mock(), candles or _quiet(_FULL_SESSION), argv)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    return json.loads(captured.out)


_LIVE_OPEN = datetime(2026, 9, 14, 3, 45, tzinfo=UTC)
"""09:15 IST on Monday 14 September 2026: the session after the backfill.

Live candles have to start after every backfilled one, or the engine rejects
them as out of order and the test would be measuring the guard instead of the
live path. Using the next trading day is how live mode actually runs.
"""

_LIVE_MINUTES = 3
"""The shortest run that closes exactly one live candle.

The builder discards each instrument's first partial minute and only closes a
candle when the following minute's first tick arrives, so three ticks on three
consecutive minutes yield one candle: minute 0 discarded, minute 1 closed by the
minute-2 tick, minute 2 left open and never flushed.
"""


def _quote(volume: int) -> MarketQuote:
    return MarketQuote(
        instrument=_RELIANCE,
        last_price=Decimal("101"),
        last_trade_at=_LIVE_OPEN,
        open=Decimal("100"),
        high=Decimal("102"),
        low=Decimal("99"),
        previous_close=Decimal("100"),
        volume=volume,
        day_change=Decimal("1"),
        day_change_percent=Decimal("1"),
    )


def _ticks(open_time: datetime) -> tuple[MarketTick, ...]:
    """One tick a minute, volume unset the way Groww's stream really sends it."""
    return tuple(
        MarketTick(
            instrument=_RELIANCE,
            timestamp=open_time + timedelta(minutes=minute),
            price=Decimal(150 + minute),
            cumulative_volume=None,
        )
        for minute in range(_LIVE_MINUTES)
    )


def _collect(ticks: tuple[MarketTick, ...]) -> Callable[..., tuple[MarketTick, ...]]:
    def collect(
        *,
        max_ticks: int,
        timeout_seconds: float,
        on_tick: Callable[[MarketTick], None],
    ) -> tuple[MarketTick, ...]:
        assert max_ticks > 0
        assert timeout_seconds > 0
        for tick in ticks:
            on_tick(tick)
        return ticks

    return collect


def _streaming_broker(
    ticks: tuple[MarketTick, ...],
    *,
    collect: Callable[..., tuple[MarketTick, ...]] | BaseException | None = None,
) -> Mock:
    """A broker whose stream replays ``ticks``, unless ``collect`` says otherwise."""
    stream = Mock()
    # Spelled out rather than left to Mock's auto-created attributes. The
    # supervisor decides whether a stream is worth closing with a
    # runtime-checkable protocol check, and since 3.12 those read attributes
    # statically, which never fires ``Mock.__getattr__``.
    stream.collect = Mock(side_effect=_collect(ticks) if collect is None else collect)
    stream.close = Mock()
    broker = Mock()
    broker.create_ltp_stream.return_value = stream
    # Live mode polls the quote endpoint for the session total, so the mock has
    # to answer with a real quote rather than another Mock.
    broker.get_quote.return_value = _quote(900_000)
    return broker


def _run_live(
    broker: Mock,
    candles: tuple[OHLCVCandle, ...],
    argv: Sequence[str] = (),
    *,
    live_ticks: int = _LIVE_MINUTES,
    live_seconds: float = 1.0,
) -> int:
    """Run one supervised live window, bounded by the tick budget.

    The budget rather than the clock, because the fake collection returns
    instantly: the supervisor would reopen the stream and replay the same ticks
    until the second elapsed, and every count under test would then depend on
    how fast the machine ran.
    """
    return _run(
        broker,
        candles,
        (
            "--live",
            "--live-seconds",
            str(live_seconds),
            "--live-ticks",
            str(live_ticks),
            *argv,
        ),
    )


def test_every_candle_becomes_one_decision_cycle(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(capsys)

    assert summary["trading_date"] == "2026-09-11"
    assert summary["backfilled_candles"] == _FULL_SESSION
    assert summary["duplicate_candles"] == 0
    assert summary["out_of_order_candles"] == 0

    backfill = summary["backfill"]
    # One cycle per candle, no candle scanned twice and none skipped: the walk
    # forward is what makes a historical answer the answer that minute had.
    assert backfill["cycles"] == _FULL_SESSION
    assert backfill["considered"] + backfill["not_ready"] == _FULL_SESSION


def test_the_run_states_which_rules_and_what_budget_produced_it(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(capsys)

    # A candidate count means nothing without the rule set behind it, so the
    # configuration is reported beside the result rather than assumed.
    assert summary["rules"] == [
        "trend_continuation",
        "range_breakout",
        "band_mean_reversion",
        "vwap_reversion",
        "opening_range_breakout",
    ]
    assert summary["max_candidates"] == 5


def test_a_quiet_session_produces_nothing_while_the_engine_works(
    capsys: CaptureFixture[str],
) -> None:
    backfill = _summary(capsys)["backfill"]

    # The distinction the whole tally exists to draw. Zero candidates with a
    # high ``considered`` is a quiet market; zero with a high ``not_ready``
    # would be a broken engine, and the two must never read the same.
    assert backfill["candidates"] == 0
    assert backfill["cycles_with_candidates"] == 0
    assert backfill["considered"] > 0
    assert backfill["unreachable"] == 0
    assert backfill["suppressed"] == {}
    assert backfill["first_candidate_at"] is None


def test_the_first_candle_is_the_only_one_the_engine_cannot_score(
    capsys: CaptureFixture[str],
) -> None:
    backfill = _summary(capsys)["backfill"]

    # Every rule needs at least a one-candle return, so the opening candle is
    # structurally unscoreable and every later one is not. More than one would
    # mean a feature was silently unavailable for the session.
    assert backfill["not_ready"] == 1


def test_a_trending_session_breaks_its_opening_range(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(capsys, _trending(_FULL_SESSION))
    backfill = summary["backfill"]

    assert backfill["candidates"] > 0
    assert "opening_range_breakout" in backfill["by_rule"]
    # 04:01 UTC is 09:31 IST, the close of the first minute of the walk — one
    # minute before the breakout rule can fire, because the close has to clear
    # the frozen high rather than merely reach it. What fires here is the VWAP
    # rule, and a flat open is why: fifteen identical candles leave the price's
    # deviation from VWAP with almost no standard deviation, so the first real
    # move reads as several sigma. A deliberately contradictory rule set will do
    # this, and the tally is where it becomes visible.
    assert backfill["first_candidate_at"] == "2026-09-11T04:01:00+00:00"


def test_a_parabolic_trend_is_vetoed_by_the_exhaustion_guard(
    capsys: CaptureFixture[str],
) -> None:
    backfill = _summary(capsys, _trending(_FULL_SESSION))["backfill"]

    # A monotone ramp pins RSI at 100, and the trend rule refuses to join a
    # trend that has already gone parabolic. The rule is runnable here — every
    # feature it needs is available by candle fifty — so its absence is the
    # veto firing rather than a missing input.
    assert "trend_continuation" not in backfill["by_rule"]


def test_the_scan_is_stamped_with_the_candle_not_the_clock(
    capsys: CaptureFixture[str],
) -> None:
    latest = _summary(capsys)["latest_scan"]

    # The property the whole layer rests on: a minute scanned from history is
    # scanned as of that minute. Stamping the wall clock would make a replay of
    # September read as though it happened today.
    assert latest["as_of"] == "2026-09-11T04:45:00+00:00"


def test_one_instrument_cannot_fill_a_five_candidate_budget(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(capsys, _trending(_FULL_SESSION))

    # A universe of one can produce at most one candidate per direction, so the
    # budget never binds here. The count this check reports is a property of a
    # one-name universe, not a measurement of the market.
    assert summary["backfill"]["truncated"] == 0
    assert len(summary["latest_scan"]["candidates"]) <= 2


def test_a_budget_of_one_truncates_a_two_sided_cycle(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(capsys, _trending(_FULL_SESSION), ("--max-candidates", "1"))

    assert summary["max_candidates"] == 1
    assert len(summary["latest_scan"]["candidates"]) == 1


def test_the_cost_screen_is_off_until_a_multiple_is_stated(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(capsys, _trending(_FULL_SESSION))

    # ``max_atr_multiple`` is the assumption with the least evidence behind it,
    # so there is no default to inherit: no flag, no screen, and the tally says
    # so rather than reporting a hurdle nobody chose.
    assert summary["cost_screen"] is None
    assert summary["backfill"]["feasibility"] == {}
    assert summary["backfill"]["unreachable"] == 0


def test_the_cost_screen_reports_the_hurdle_it_priced(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(capsys, _trending(_FULL_SESSION), ("--max-atr-multiple", "30"))

    screen = summary["cost_screen"]
    assert screen["broker"] == "groww"
    assert screen["target_notional"] == "100000"
    assert screen["max_atr_multiple"] == "30"
    # Stated as a fraction with its percentage beside it, always: the two differ
    # by a hundred and printing one alone is how that error survives review.
    assert screen["gross_target_fraction"] == "0.00200000 (0.2000%)"
    # Every screened cycle is accounted for under a named reason.
    assert sum(summary["backfill"]["feasibility"].values()) == _FULL_SESSION


def test_the_screen_holds_a_name_back_until_its_volatility_is_known(
    capsys: CaptureFixture[str],
) -> None:
    without = _summary(capsys)["backfill"]
    with_screen = _summary(capsys, argv=("--max-atr-multiple", "30"))["backfill"]

    # The quiet session rather than the trending one: the trending fixture
    # opens dead flat, and a zero-variance series leaves ADX, RSI and percent-b
    # undefined, so it reports the same ``not_ready`` with the screen on or off
    # for a reason that has nothing to do with the screen.
    #
    # Turning the screen on raises ``not_ready`` from one candle to thirteen,
    # because a cost hurdle priced against an unknown ATR is not a hurdle. The
    # fourteen-period range needs fourteen candles, so the screen fails closed
    # over the warm-up and the rules never run. Worth pinning: it would be easy
    # to read the jump as the screen rejecting names on their merits.
    assert without["not_ready"] == 1
    assert with_screen["not_ready"] == 13
    assert with_screen["feasibility"]["volatility_unknown"] == 13
    assert with_screen["unreachable"] == 0


def test_an_unreachable_hurdle_rejects_every_name_by_name(
    capsys: CaptureFixture[str],
) -> None:
    summary = _summary(
        capsys, _trending(_FULL_SESSION), ("--max-atr-multiple", "0.0001")
    )

    backfill = summary["backfill"]
    feasibility = backfill["feasibility"]
    # A hurdle no one-minute move could clear. The run must still succeed and
    # say *why* it is empty: "the market is too quiet to pay for a round trip"
    # is a different answer from "nothing was set up", and an operator seeing
    # zero candidates needs to be able to tell them apart.
    assert backfill["candidates"] == 0
    assert backfill["unreachable"] > 0
    assert feasibility["volatility_too_low"] > 0
    assert "reachable" not in feasibility


def test_the_broker_choice_moves_the_hurdle_below_the_cap(
    capsys: CaptureFixture[str],
) -> None:
    # A target wide enough that both schedules clear it at this clip. At the
    # stated 0.2% they do not, which is its own test below.
    common = ("--max-atr-multiple", "30", "--clip", "20000", "--gross-target", "0.005")
    groww = _summary(capsys, _trending(_FULL_SESSION), common)["cost_screen"]
    zerodha = _summary(
        capsys, _trending(_FULL_SESSION), (*common, "--broker", "zerodha")
    )["cost_screen"]

    # Below Rs 66,666.67 a leg the caps have not bound and the two schedules
    # genuinely differ, so the screen must price with the one it was told to.
    assert groww["net_margin_fraction"] != zerodha["net_margin_fraction"]
    assert zerodha["broker"] == "zerodha"


def test_a_clip_too_small_for_its_target_is_refused(
    capsys: CaptureFixture[str],
) -> None:
    exit_code = _run(
        Mock(),
        _trending(_FULL_SESSION),
        ("--max-atr-multiple", "30", "--clip", "20000"),
    )

    captured = capsys.readouterr()
    # The most consequential arithmetic in the layer. A round trip costs about
    # 0.271% of a twenty-thousand-rupee clip at Groww's rates, so the stated
    # 0.2% target is not merely thin there — it is negative, and no rule tuning
    # could rescue it. The run refuses rather than screening against a hurdle
    # that cannot be met, and says what it was short by.
    assert exit_code == 1
    assert "does not clear costs" in captured.err
    assert captured.out == ""


def test_kite_and_zerodha_are_one_rate_card(capsys: CaptureFixture[str]) -> None:
    common = ("--max-atr-multiple", "30", "--clip", "20000")
    kite = _summary(capsys, _trending(_FULL_SESSION), (*common, "--broker", "kite"))[
        "cost_screen"
    ]
    zerodha = _summary(
        capsys, _trending(_FULL_SESSION), (*common, "--broker", "zerodha")
    )["cost_screen"]

    assert kite["net_margin_fraction"] == zerodha["net_margin_fraction"]


def test_the_scanner_never_calls_the_broker() -> None:
    broker = Mock()

    exit_code = _run(broker, _trending(_FULL_SESSION))

    assert exit_code == 0
    # The session fetch is mocked out, so any call recorded here would mean the
    # scanner or the cost screen reached for the network mid-cycle.
    assert broker.method_calls == []


def test_the_stream_is_untouched_unless_live_is_asked_for() -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    _run(broker, _quiet(_FULL_SESSION))

    broker.create_ltp_stream.assert_not_called()
    broker.get_quote.assert_not_called()


def test_the_export_carries_one_row_per_cycle(tmp_path: Path) -> None:
    export = tmp_path / "scan.csv"

    exit_code = _run(Mock(), _trending(_FULL_SESSION), ("--export-csv", str(export)))

    assert exit_code == 0
    rows = _read_csv(export)
    assert len(rows) == _FULL_SESSION
    assert tuple(rows[0]) == _CSV_HEADER
    # The file exists to be lined up against a chart minute by minute, so the
    # timestamps have to be the candle ends in order.
    assert rows[0]["end_time"] == "2026-09-11T03:46:00+00:00"
    assert rows[-1]["end_time"] == "2026-09-11T04:45:00+00:00"


def test_the_export_records_the_screen_that_rejected_a_cycle(
    tmp_path: Path,
) -> None:
    export = tmp_path / "scan.csv"

    _run(
        Mock(),
        _trending(_FULL_SESSION),
        ("--max-atr-multiple", "0.0001", "--export-csv", str(export)),
    )

    rows = _read_csv(export)
    reasons = {row["feasibility_reason"] for row in rows}
    # Two reasons, and the difference between them matters: the early candles
    # have no ATR to screen against yet, the later ones have one and fail it.
    # Collapsing those into a single "rejected" would hide a warming engine
    # behind a quiet market.
    assert reasons == {"volatility_unknown", "volatility_too_low"}
    assert not any(row["top_direction"] for row in rows)
    # Full precision, not the display rounding: the export is for comparison
    # against a trusted implementation, and six places would hide a difference.
    #
    # The first candle closes at a price that divides the clip exactly, so it
    # buys the clip and faces the stated target to the last place. The last
    # closes dearer, so whole shares overshoot the clip, the capped brokerage
    # spreads over a larger turnover, and the hurdle comes in fractionally
    # under. That ordering is the whole reason the screen prices per name
    # rather than once per cycle -- screening everything at the clip figure
    # would be too strict, not too loose.
    assert Decimal(rows[0]["required_gross_fraction"]) == Decimal("0.002")
    assert Decimal(rows[-1]["required_gross_fraction"]) < Decimal("0.002")
    # And the ATR that lost against it. Naming a rejection without the two
    # numbers that produced it makes the row unfalsifiable -- the reader cannot
    # tell a hurdle that was marginally missed from one that was never close.
    # These columns come from the cycle's own screen rather than a surviving
    # candidate's, because on exactly these rows no candidate survived.
    assert rows[-1]["atr_fraction"]
    assert rows[-1]["minutes_remaining"]
    assert rows[0]["atr_fraction"] == ""  # unknown, which is why it was held


def test_an_existing_export_is_refused_without_overwrite(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    export = tmp_path / "scan.csv"
    export.write_text("reference data", encoding="utf-8")

    exit_code = _run(Mock(), _quiet(_FULL_SESSION), ("--export-csv", str(export)))

    assert exit_code == 1
    assert "--overwrite" in capsys.readouterr().err
    # A run that failed after opening the file would otherwise have replaced
    # reference data with an empty header and reported success.
    assert export.read_text(encoding="utf-8") == "reference data"


def test_overwrite_replaces_an_existing_export(tmp_path: Path) -> None:
    export = tmp_path / "scan.csv"
    export.write_text("stale", encoding="utf-8")

    exit_code = _run(
        Mock(),
        _quiet(_FULL_SESSION),
        ("--export-csv", str(export), "--overwrite"),
    )

    assert exit_code == 0
    assert len(_read_csv(export)) == _FULL_SESSION


def test_live_candles_are_scanned_on_the_same_engine(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    exit_code = _run_live(broker, _trending(_FULL_SESSION))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""

    summary = json.loads(captured.out)
    live = summary["live"]
    assert live["live_ticks"] == _LIVE_MINUTES
    # Three ticks, one candle: the first minute is discarded and the last is
    # left open rather than flushed into a partial candle a scanner would score
    # as though it covered the whole minute.
    assert live["live_candles"] == 1
    assert live["late_ticks"] == 0
    # And that one candle went through the same scan the backfill did.
    assert live["scan"]["cycles"] == 1
    assert summary["backfill"]["cycles"] == _FULL_SESSION


def test_the_live_scan_is_tallied_apart_from_the_backfill(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    _run_live(broker, _trending(_FULL_SESSION))

    summary = json.loads(capsys.readouterr().out)
    # Two tallies, never summed. A live count folded into a six-hour backfill
    # would be invisible, and the live window is the part under test.
    assert summary["live"]["scan"]["cycles"] == 1
    assert summary["backfill"]["cycles"] == _FULL_SESSION


def test_live_ticks_are_stamped_with_the_polled_session_volume(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(_ticks(_LIVE_OPEN))

    _run_live(broker, _trending(_FULL_SESSION))

    live = json.loads(capsys.readouterr().out)["live"]
    # Groww's stream carries no volume, so every tick arrives bare and the
    # poller is the only thing standing between the scanner and a session whose
    # volume features are all unavailable.
    assert live["stamped_ticks"] == _LIVE_MINUTES
    assert live["session_volume"] == 900_000
    assert live["poll_failures"] == 0
    assert live["stale_stamps"] == 0


def test_a_live_window_that_never_connects_fails_loudly(
    capsys: CaptureFixture[str],
) -> None:
    broker = _streaming_broker(
        _ticks(_LIVE_OPEN), collect=GrowwStreamConnectionError("socket closed")
    )
    broker.create_ltp_stream.side_effect = GrowwStreamConnectionError("refused")

    exit_code = _run_live(broker, _quiet(_FULL_SESSION), live_seconds=0.5)

    captured = capsys.readouterr()
    assert exit_code == 1
    # The summary is printed first and the failure reported after it, so the
    # backfill result is not lost to a live window that never opened.
    assert json.loads(captured.out)["live"]["stream_sessions"] == 0
    assert "never opened a stream session" in captured.err


def test_a_missing_configuration_is_its_own_exit_code(
    capsys: CaptureFixture[str],
) -> None:
    with patch(
        "ai_trader.cli.check_scanner.load_groww_settings",
        side_effect=ConfigurationError("GROWW_API_KEY is not set"),
    ):
        exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    # Two rather than one, so a shell can tell "you have not configured this"
    # from "this ran and failed".
    assert "GROWW_API_KEY is not set" in captured.err
    assert captured.out == ""


def test_a_rejected_login_says_so_without_leaking_the_reason(
    capsys: CaptureFixture[str],
) -> None:
    with (
        patch("ai_trader.cli.check_scanner.load_groww_settings"),
        patch(
            "ai_trader.cli.check_scanner.GrowwBroker.authenticate",
            side_effect=GrowwAuthenticationError("token abc123 rejected"),
        ),
    ):
        exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    # The broker's message may quote the credential it refused, so the check
    # prints its own sentence rather than relaying one.
    assert "abc123" not in captured.err


def test_a_broker_failure_is_reported_without_a_traceback(
    capsys: CaptureFixture[str],
) -> None:
    with (
        patch("ai_trader.cli.check_scanner.load_groww_settings"),
        patch(
            "ai_trader.cli.check_scanner.GrowwBroker.authenticate",
            side_effect=GrowwBrokerError("upstream 502"),
        ),
    ):
        exit_code = main([])

    assert exit_code == 1
    assert "Groww scanner check failed." in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--live-seconds", "0"),
        ("--live-ticks", "0"),
        ("--live-seconds", "-1"),
        ("--max-atr-multiple", "0"),
        ("--max-atr-multiple", "-3"),
        ("--clip", "0"),
        ("--gross-target", "0"),
    ],
)
def test_non_positive_options_are_refused(
    option: str,
    value: str,
    capsys: CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([option, value])

    assert exit_info.value.code == 2
    assert "must be positive" in capsys.readouterr().err


def test_a_budget_below_one_is_refused(capsys: CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--max-candidates", "0"])

    assert exit_info.value.code == 2
    assert "at least 1" in capsys.readouterr().err


def test_an_unknown_broker_is_refused(capsys: CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--broker", "hdfc"])

    assert exit_info.value.code == 2
    assert "hdfc" in capsys.readouterr().err
