"""The reporter must not overstate the run, and the CLI must not restate it.

Two claims, and they are the two ways this layer can lie without crashing.

**The report cannot overstate its own sample.** A run asks for a year and the
broker holds a quarter of one; a header citing the request would quadruple the
sample it claims to have measured. So the coverage block is derived from the
bars that actually reached the engine, and the tests below hand the reporter a
request deliberately wider than the bars to prove the two are not the same
number wearing one label.

**The CLI cannot become a second definition of the strategy.** Section 7.1's
guarantee is that replay and live trading read one configuration, and the way a
command line breaks that is by helpfully defaulting a flag -- after which the
sweep measures something live trading would never run, and nothing fails. The
tests here pin ``_build_strategy`` to ``StrategyConfig()`` field by field, so a
default added to the parser shows up as a failure rather than as a quietly
different backtest.

The fixture is local rather than shared with ``test_replay_equivalence``. That
file's session has to be expressible as ticks *and* as candles, which is most of
its length and none of this file's business; what is needed here is only a tape
that trades.
"""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ai_trader.broker import CandleInterval, Instrument, OHLCVCandle
from ai_trader.cli import backtest
from ai_trader.clock import INDIA_TIMEZONE, SESSION_CLOSE_TIME, SESSION_OPEN_TIME
from ai_trader.config import ConfigurationError
from ai_trader.history import CandleStore, last_completed_session_close
from ai_trader.market import Candle
from ai_trader.replay import (
    FRICTIONLESS,
    ExitReason,
    FillModel,
    ReplayConfig,
    ReplayEngine,
)
from ai_trader.replay.report import (
    HISTORY_FIELDS,
    ReportError,
    append_history,
    history_row,
    report_lines,
)
from ai_trader.strategy import (
    DEFAULT_STOP_ATR_MULTIPLE,
    ChandelierStop,
    FixedAtrStop,
    StrategyConfig,
)

_ONE_MINUTE = timedelta(minutes=1)
_SESSION_MINUTES = 375

_NAMES = (
    (Instrument(exchange="NSE", trading_symbol="ALPHA"), 0, Decimal("2450")),
    (Instrument(exchange="NSE", trading_symbol="BRAVO"), 17, Decimal("1310")),
    (Instrument(exchange="NSE", trading_symbol="DELTA"), 31, Decimal("880")),
)
_UNIVERSE = tuple(instrument for instrument, _, _ in _NAMES)

_FILL = FillModel(
    latency_seconds=Decimal(60),
    half_spread_fraction=Decimal("0.0002"),
    slippage_fraction=Decimal("0.0001"),
)
"""Friction on, for the same reason the equivalence tests switch it on: a
frictionless run reports costs of zero, and a reporter that dropped the cost
line entirely would still look right."""

_KNOBS = ("--stop-atr", "1.5", "--max-open-positions", "2")
"""Enough of a stop to be hit and a book small enough that the ranking decides
something. Stated explicitly because the end-to-end run has to produce trades;
the claim that *unset* knobs stay unset is tested separately and directly."""


# --- a session that trades ---------------------------------------------------


def _triangle(step: int, half: int) -> int:
    return step if step <= half else 2 * half - step


def _close_price(base: Decimal, minute_index: int, phase: int) -> Decimal:
    """Two triangular waves of coprime period, superposed.

    Nothing is drawn from a generator, seeded or otherwise: a failure here has
    to be reproducible from this file alone. Every increment is a whole number
    of ticks, so the prices sit on the exchange's grid and the fill model's
    rounding is exercised rather than being a no-op.
    """
    fast = _triangle((minute_index * 5 + phase) % 46, 23)
    slow = _triangle((minute_index * 2 + phase) % 97, 48)
    return base + Decimal(fast) * Decimal("0.35") + Decimal(slow) * Decimal("0.25")


def _session_date() -> date:
    """The most recent session the store is willing to consider complete.

    Derived rather than written down. A fixed date would drift out of the
    broker's rolling window and, worse, would sooner or later sit in the future
    of whoever runs the suite -- at which point ``CandleStore`` clips the window
    to nothing and every assertion below becomes vacuous rather than red.
    """
    return last_completed_session_close(datetime.now(UTC)).date()


class _FakeBroker:
    """Serves the synthetic session and nothing else."""

    def __init__(self, day: date) -> None:
        self._origin = datetime.combine(day, SESSION_OPEN_TIME, tzinfo=INDIA_TIMEZONE)

    def get_historical_candles(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> tuple[OHLCVCandle, ...]:
        assert interval is CandleInterval.ONE_MINUTE
        phase, base = next(
            (phase, base)
            for candidate, phase, base in _NAMES
            if candidate == instrument
        )
        out: list[OHLCVCandle] = []
        for minute_index in range(_SESSION_MINUTES):
            begins = self._origin + minute_index * _ONE_MINUTE
            if not start <= begins < end:
                continue
            open_price = _close_price(base, minute_index, phase)
            close_price = _close_price(base, minute_index + 1, phase)
            out.append(
                OHLCVCandle(
                    timestamp=begins,
                    open=open_price,
                    high=max(open_price, close_price) + Decimal("0.85"),
                    low=min(open_price, close_price) - Decimal("0.75"),
                    close=close_price,
                    volume=1_000 + (minute_index * 137 + phase * 29) % 900,
                )
            )
        return tuple(out)


def _seed(root: Path, day: date) -> dict[Instrument, tuple[Candle, ...]]:
    """Fill the cache through the real store, so the CLI reads a real file."""
    store = CandleStore(root, _FakeBroker(day))
    start = datetime.combine(day, SESSION_OPEN_TIME, tzinfo=INDIA_TIMEZONE)
    end = datetime.combine(day, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE)
    return {instrument: store.load(instrument, start, end) for instrument in _UNIVERSE}


def _run(loaded: dict[Instrument, tuple[Candle, ...]]):
    strategy = StrategyConfig(
        exit_policy=FixedAtrStop(Decimal("1.5")), max_open_positions=2
    )
    engine = ReplayEngine(
        ReplayConfig(universe=_UNIVERSE, fill=_FILL, strategy=strategy)
    )
    candles = [candle for bars in loaded.values() for candle in bars]
    return strategy, engine.run(candles)


@pytest.fixture(scope="module")
def replayed() -> tuple[dict[Instrument, tuple[Candle, ...]], StrategyConfig, object]:
    """One replay, shared by the report tests. Built once because it is slow."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        loaded = _seed(Path(directory), _session_date())
    strategy, result = _run(loaded)
    return loaded, strategy, result


def test_the_fixture_actually_trades(replayed) -> None:
    """Everything below reports on a run; a dead tape would report on nothing.

    Without this, a fixture that stopped producing candidates would leave every
    other assertion here passing over an empty result -- the report would still
    have its sections, the sweep row would still have its columns, and none of
    it would be evidence.
    """
    _, _, result = replayed
    assert result.round_trips > 0
    assert result.candidates_seen > 0
    assert result.candles_replayed == len(_UNIVERSE) * _SESSION_MINUTES


# --- the report states the sample it measured, not the one it asked for ------


def test_coverage_is_derived_from_the_bars_not_the_request(replayed) -> None:
    """A request far wider than the tape must not widen the reported sample."""
    loaded, strategy, result = replayed
    day = _session_date()
    requested = (
        datetime(day.year - 1, 1, 1, tzinfo=INDIA_TIMEZONE),
        datetime(day.year + 1, 1, 1, tzinfo=INDIA_TIMEZONE),
    )
    text = "\n".join(
        report_lines(
            result,
            strategy,
            requested=requested,
            loaded=loaded,
            generated_at=datetime.now(UTC),
        )
    )
    assert f"{day.year - 1}-01-01" in text, "the request is still disclosed"
    assert f"{day.isoformat()} 09:15 .. {day.isoformat()} 15:30" in text
    assert f"{_SESSION_MINUTES:,} bars" in text
    assert f"{day.year + 1}-01-01 .." not in text.split("Bars actually replayed")[1]


def test_an_instrument_with_no_bars_says_so(replayed) -> None:
    """Silence about an empty name would read as a name that simply did nothing."""
    loaded, strategy, result = replayed
    thinned = {
        instrument: bars
        for instrument, bars in loaded.items()
        if instrument != _UNIVERSE[0]
    }
    text = "\n".join(
        report_lines(
            result,
            strategy,
            requested=(datetime.now(UTC), datetime.now(UTC)),
            loaded=thinned,
            generated_at=datetime.now(UTC),
        )
    )
    assert f"NSE:{_UNIVERSE[0].trading_symbol}" in text
    assert "no bars" in text


def test_the_caveats_come_before_the_numbers(replayed) -> None:
    """Section order is the claim: a reader who stops early stops on the caveat."""
    loaded, strategy, result = replayed
    text = "\n".join(
        report_lines(
            result,
            strategy,
            requested=(datetime.now(UTC), datetime.now(UTC)),
            loaded=loaded,
            generated_at=datetime.now(UTC),
        )
    )
    assert text.index("Read these first") < text.index("Edge")
    assert text.index("ambiguous exits") < text.index("net expectancy")
    assert text.index("upper bound") < text.index("net expectancy")


def test_fractions_are_printed_with_their_percentage(replayed) -> None:
    """A fraction and a percentage of it differ by a hundred."""
    loaded, strategy, result = replayed
    text = "\n".join(
        report_lines(
            result,
            strategy,
            requested=(datetime.now(UTC), datetime.now(UTC)),
            loaded=loaded,
            generated_at=datetime.now(UTC),
        )
    )
    for line in text.splitlines():
        if "hit rate" in line or "mean favourable" in line:
            assert "%" in line and "(" in line

    # A known fraction, checked digit for digit. The assertions above prove a
    # percentage was printed; only a value the test already knows can prove it
    # was a hundred times the fraction beside it rather than a copy of it.
    assert "half spread          0.00020000 (0.0200%)" in text


def test_no_round_trips_reports_nothing_rather_than_zero() -> None:
    """Zero averages over zero trades are arithmetic, not a finding."""
    day = _session_date()
    empty_strategy = StrategyConfig(max_candidates=1, max_open_positions=1)
    engine = ReplayEngine(
        ReplayConfig(universe=_UNIVERSE, fill=_FILL, strategy=empty_strategy)
    )
    result = engine.run(())
    text = "\n".join(
        report_lines(
            result,
            empty_strategy,
            requested=(
                datetime.combine(day, SESSION_OPEN_TIME, tzinfo=INDIA_TIMEZONE),
                datetime.combine(day, SESSION_CLOSE_TIME, tzinfo=INDIA_TIMEZONE),
            ),
            loaded={},
            generated_at=datetime.now(UTC),
        )
    )
    assert "No completed round trips" in text
    assert "net expectancy" not in text
    assert "nothing to attribute" in text


# --- the sweep table ---------------------------------------------------------


def test_the_sweep_row_is_exactly_the_table(replayed) -> None:
    _, strategy, result = replayed
    row = history_row(result, strategy, generated_at=datetime.now(UTC), label="x")
    assert set(row) == set(HISTORY_FIELDS)


def test_the_sweep_row_is_unrounded(replayed) -> None:
    """Two runs a paisa apart must not read as identical."""
    _, strategy, result = replayed
    row = history_row(result, strategy, generated_at=datetime.now(UTC))
    assert Decimal(row["net_expectancy_rupees"]) == result.net_expectancy_rupees
    assert Decimal(row["net_rupees"]) == result.net_rupees
    assert Decimal(row["stop_multiple"]) == Decimal("1.5")
    assert row["exit_kind"] == "FixedAtrStop"


def test_a_disabled_screen_is_recorded_as_off(replayed) -> None:
    """Blank would read as unknown; a multiple would read as a screen that ran."""
    _, _, result = replayed
    strategy = StrategyConfig(screen_feasibility=False)
    row = history_row(result, strategy, generated_at=datetime.now(UTC))
    assert row["cost_screen"] == "off"


def test_history_writes_one_header_and_then_rows(tmp_path, replayed) -> None:
    _, strategy, result = replayed
    path = tmp_path / "nested" / "sweep.tsv"
    for label in ("first", "second"):
        append_history(
            path,
            history_row(result, strategy, generated_at=datetime.now(UTC), label=label),
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0]) == HISTORY_FIELDS
    assert [row["label"] for row in rows] == ["first", "second"]


def test_history_refuses_a_file_with_other_columns(tmp_path, replayed) -> None:
    """A misaligned sweep is undetectable downstream; a refusal is not."""
    _, strategy, result = replayed
    path = tmp_path / "sweep.tsv"
    path.write_text("label\tnet_rupees\n", encoding="utf-8")
    with pytest.raises(ReportError, match="different columns"):
        append_history(
            path, history_row(result, strategy, generated_at=datetime.now(UTC))
        )


def test_history_refuses_a_row_that_is_not_the_table(tmp_path, replayed) -> None:
    _, strategy, result = replayed
    row = history_row(result, strategy, generated_at=datetime.now(UTC))
    del row["net_rupees"]
    with pytest.raises(ReportError, match="missing"):
        append_history(tmp_path / "sweep.tsv", row)

    row = history_row(result, strategy, generated_at=datetime.now(UTC))
    row["extra"] = "1"
    with pytest.raises(ReportError, match="unexpected"):
        append_history(tmp_path / "sweep.tsv", row)


def test_history_refuses_a_value_carrying_a_separator(tmp_path, replayed) -> None:
    """A tab inside a cell shifts every column after it by one."""
    _, strategy, result = replayed
    row = history_row(result, strategy, generated_at=datetime.now(UTC))
    row["label"] = "a\tb"
    with pytest.raises(ReportError, match="separator"):
        append_history(tmp_path / "sweep.tsv", row)


def test_history_is_written_with_unix_newlines(tmp_path, replayed) -> None:
    """A diff of two sweeps should show the strategy, not the operating system."""
    _, strategy, result = replayed
    path = tmp_path / "sweep.tsv"
    append_history(path, history_row(result, strategy, generated_at=datetime.now(UTC)))
    assert b"\r\n" not in path.read_bytes()


# --- the CLI states no strategy default --------------------------------------


def _args(*extra: str) -> object:
    return backtest._parse_args(
        ["--symbols", "ALPHA", "--start", "2026-09-01", "--end", "2026-09-02", *extra]
    )


def test_unset_knobs_leave_the_shared_defaults_untouched() -> None:
    """Field by field, because one helpful default here forks the strategy.

    Compared against a bare ``StrategyConfig`` rather than against written-down
    numbers: a test that restated the defaults would be the second definition it
    exists to forbid.
    """
    built = backtest._build_strategy(_args("--frictionless"))
    assert built == StrategyConfig()


def test_a_stated_knob_reaches_the_config() -> None:
    built = backtest._build_strategy(
        _args("--frictionless", "--clip", "45000", "--max-open-positions", "4")
    )
    assert built.target_notional == Decimal(45000)
    assert built.max_open_positions == 4
    assert built.gross_target_fraction == StrategyConfig().gross_target_fraction


def test_trailing_alone_trails_at_the_shared_multiple() -> None:
    """``--trailing`` must change the trail and nothing else, or a sweep of the
    two policies is comparing two differences at once."""
    built = backtest._build_strategy(_args("--frictionless", "--trailing"))
    assert isinstance(built.exit_policy, ChandelierStop)
    assert built.exit_policy.multiple == DEFAULT_STOP_ATR_MULTIPLE

    fixed = backtest._build_strategy(_args("--frictionless", "--stop-atr", "2.5"))
    assert isinstance(fixed.exit_policy, FixedAtrStop)
    assert fixed.exit_policy.multiple == Decimal("2.5")

    both = backtest._build_strategy(
        _args("--frictionless", "--trailing", "--stop-atr", "2.5")
    )
    assert isinstance(both.exit_policy, ChandelierStop)
    assert both.exit_policy.multiple == Decimal("2.5")


def test_the_fill_model_must_be_stated() -> None:
    """No default, because a silent zero is the most flattering assumption here."""
    with pytest.raises(SystemExit) as raised:
        _args()
    assert raised.value.code == 2

    with pytest.raises(SystemExit):
        _args("--latency-seconds", "1", "--half-spread", "0.0002")


def test_frictionless_refuses_to_share_the_sentence() -> None:
    """Both stated is a caller who believes one of the two and is wrong."""
    with pytest.raises(SystemExit):
        _args("--frictionless", "--slippage", "0.0001")
    with pytest.raises(SystemExit):
        _args("--frictionless", "--ambiguous-as", "target")


def test_frictionless_is_the_shared_constant() -> None:
    assert backtest._build_fill(_args("--frictionless")) is FRICTIONLESS


def test_the_ambiguous_rule_is_forwarded_only_when_chosen() -> None:
    """Unstated means the model's own default, not one this file re-decides."""
    stated = ("--latency-seconds", "1", "--half-spread", "0", "--slippage", "0")
    assert (
        backtest._build_fill(_args(*stated)).resolve_ambiguous_bar_as is ExitReason.STOP
    )
    assert (
        backtest._build_fill(
            _args(*stated, "--ambiguous-as", "target")
        ).resolve_ambiguous_bar_as
        is ExitReason.TARGET
    )


def test_a_symbol_named_twice_is_replayed_once() -> None:
    """Twice would double the book's exposure while its position count denied it."""
    universe = backtest._universe(
        backtest._parse_args(
            [
                "--symbols",
                "alpha, BRAVO ,ALPHA",
                "--start",
                "2026-09-01",
                "--end",
                "2026-09-02",
                "--frictionless",
            ]
        )
    )
    assert universe == (
        Instrument(exchange="NSE", trading_symbol="ALPHA"),
        Instrument(exchange="NSE", trading_symbol="BRAVO"),
    )


def test_a_symbols_file_drops_comments_and_blanks(tmp_path) -> None:
    path = tmp_path / "names.txt"
    path.write_text(
        "# universe\nALPHA\n\n  BRAVO  # the second one\n", encoding="utf-8"
    )
    universe = backtest._universe(
        backtest._parse_args(
            [
                "--symbols-file",
                str(path),
                "--start",
                "2026-09-01",
                "--end",
                "2026-09-02",
                "--frictionless",
            ]
        )
    )
    assert [instrument.trading_symbol for instrument in universe] == ["ALPHA", "BRAVO"]


def test_exactly_one_source_of_symbols(tmp_path) -> None:
    for extra in ((), ("--symbols", "A", "--symbols-file", str(tmp_path / "x.txt"))):
        with pytest.raises(SystemExit):
            backtest._parse_args(
                [
                    "--start",
                    "2026-09-01",
                    "--end",
                    "2026-09-02",
                    "--frictionless",
                    *extra,
                ]
            )


def test_a_backwards_window_is_refused() -> None:
    with pytest.raises(SystemExit):
        backtest._parse_args(
            [
                "--symbols",
                "ALPHA",
                "--start",
                "2026-09-02",
                "--end",
                "2026-09-01",
                "--frictionless",
            ]
        )


@pytest.mark.parametrize("text", ["abc", "NaN", "Infinity", "-Infinity", ""])
def test_unusable_numbers_are_refused_not_raised(text: str) -> None:
    """``InvalidOperation`` is an ``ArithmeticError``, which argparse does not
    catch, and a NaN compares false against everything it is screened by."""
    with pytest.raises(SystemExit):
        _args("--frictionless", "--clip", text)


def test_a_label_cannot_break_the_table() -> None:
    with pytest.raises(SystemExit):
        _args("--frictionless", "--label", "a\tb")


def test_a_screen_multiple_without_a_screen_is_refused() -> None:
    """Accepting it would print a screen setting the run did not apply."""
    with pytest.raises(SystemExit):
        _args("--frictionless", "--no-screen", "--max-atr-multiple", "3")


# --- end to end --------------------------------------------------------------


def _cli(tmp_path: Path, *extra: str) -> list[str]:
    day = _session_date()
    return [
        "--symbols",
        ",".join(instrument.trading_symbol for instrument in _UNIVERSE),
        "--start",
        day.isoformat(),
        "--end",
        day.isoformat(),
        "--cache",
        str(tmp_path / "cache"),
        "--offline",
        "--latency-seconds",
        "60",
        "--half-spread",
        "0.0002",
        "--slippage",
        "0.0001",
        "--report",
        str(tmp_path / "report.txt"),
        "--history",
        str(tmp_path / "sweep.tsv"),
        *_KNOBS,
        *extra,
    ]


def test_a_cached_run_writes_a_report_and_a_sweep_row(tmp_path, capsys) -> None:
    _seed(tmp_path / "cache", _session_date())
    assert backtest.main(_cli(tmp_path)) == 0

    report = (tmp_path / "report.txt").read_bytes()
    assert b"\r\n" not in report, "reports diff across machines otherwise"
    text = report.decode("utf-8")
    assert "BACKTEST" in text
    assert "Read these first" in text
    assert capsys.readouterr().out.strip() == text.strip()

    with (tmp_path / "sweep.tsv").open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert int(rows[0]["candles"]) == len(_UNIVERSE) * _SESSION_MINUTES
    assert int(rows[0]["round_trips"]) > 0
    assert rows[0]["sessions"] == "1"


def test_offline_never_reaches_for_credentials(tmp_path, monkeypatch) -> None:
    """The reproducible mode must not be able to pull a revised bar."""

    def explode() -> None:
        raise AssertionError("--offline loaded credentials")

    monkeypatch.setattr(backtest, "load_groww_settings", explode)
    monkeypatch.setattr(
        backtest, "GrowwBroker", None
    )  # attribute access would fail loudly
    _seed(tmp_path / "cache", _session_date())
    assert backtest.main(_cli(tmp_path)) == 0


def test_no_write_touches_nothing(tmp_path, capsys) -> None:
    _seed(tmp_path / "cache", _session_date())
    assert backtest.main(_cli(tmp_path, "--no-write")) == 0
    assert not (tmp_path / "report.txt").exists()
    assert not (tmp_path / "sweep.tsv").exists()
    assert "BACKTEST" in capsys.readouterr().out


def test_json_prints_the_sweep_row(tmp_path, capsys) -> None:
    import json

    _seed(tmp_path / "cache", _session_date())
    assert backtest.main(_cli(tmp_path, "--json", "--no-write")) == 0
    row = json.loads(capsys.readouterr().out)
    assert set(row) == set(HISTORY_FIELDS)


def test_an_uncached_offline_window_fails_rather_than_reporting(
    tmp_path, capsys
) -> None:
    """Empty is not a result; it is a run that did not happen."""
    assert backtest.main(_cli(tmp_path)) == 1
    assert "not cached" in capsys.readouterr().err


def test_a_window_in_the_future_fails_rather_than_reporting(tmp_path, capsys) -> None:
    """The one way to get zero bars without an error, and it must still fail.

    ``CandleStore`` clips every request to the last completed session, so a
    window beyond it comes back empty with nothing raised -- no broker call, no
    missing file, no exception. A run that reported on that would print a whole
    page of zeroes under a header naming dates it never saw a bar from.
    """
    _seed(tmp_path / "cache", _session_date())
    ahead = (datetime.now(UTC) + timedelta(days=30)).date().isoformat()
    assert backtest.main(_cli(tmp_path, "--start", ahead, "--end", ahead)) == 1
    assert "No candles" in capsys.readouterr().err
    assert not (tmp_path / "report.txt").exists()


def test_missing_credentials_are_a_configuration_failure(tmp_path, monkeypatch) -> None:
    """Exit 2, not 1: nothing was attempted, the environment was wrong."""

    def explode() -> None:
        raise ConfigurationError("GROWW_API_KEY is not set")

    monkeypatch.setattr(backtest, "load_groww_settings", explode)
    online = [flag for flag in _cli(tmp_path) if flag != "--offline"]
    assert backtest.main(online) == 2


def test_a_latency_the_clock_cannot_keep_is_a_configuration_failure(
    tmp_path, capsys
) -> None:
    """Exit 2 for the same reason, from the other side of the same ``try``.

    ``FillModel`` refuses a latency that is not a whole number of microseconds
    rather than rounding one, and a refused flag is the configuration being
    wrong -- not the run failing. It used to come back as 1, which is also what
    an uncached window and a dead broker return, so a sweep script retrying the
    failures that are worth retrying would have retried this one forever.

    The cache is seeded so that the only thing wrong with the run is the
    latency. On an empty cache exit 1 is available for an honest reason, and
    this would then be passing on whichever code it happened to get.
    """
    _seed(tmp_path / "cache", _session_date())
    # argparse keeps the last occurrence, so this replaces the latency ``_cli``
    # states rather than joining it.
    refused = _cli(tmp_path, "--latency-seconds", "0.0000001")

    assert backtest.main(refused) == 2
    error = capsys.readouterr().err
    assert "latency_seconds" in error, "the refusal does not name the flag"
    assert "microseconds" in error
    assert not (tmp_path / "report.txt").exists()
    assert not (tmp_path / "sweep.tsv").exists(), "a refused run wrote a sweep row"


def test_a_target_its_own_costs_would_eat_is_a_configuration_failure(
    tmp_path, capsys
) -> None:
    """The other constructor inside that ``try``, refused the same way.

    A twenty-thousand clip pays more in charges than a 0.2% move earns, so
    ``StrategyConfig`` will not build a sizer for it. Both arms are tested
    because the comment on that ``except`` claims both, and a guard that caught
    only the fill model would read as correct against either one alone.
    """
    _seed(tmp_path / "cache", _session_date())

    assert backtest.main(_cli(tmp_path, "--clip", "20000")) == 2
    assert "does not clear costs" in capsys.readouterr().err


def test_the_run_is_reproducible(tmp_path) -> None:
    """Same tape, same knobs, same numbers -- twice, into one sweep file."""
    _seed(tmp_path / "cache", _session_date())
    assert backtest.main(_cli(tmp_path, "--label", "one")) == 0
    assert backtest.main(_cli(tmp_path, "--label", "two")) == 0

    with (tmp_path / "sweep.tsv").open(encoding="utf-8", newline="") as handle:
        first, second = tuple(csv.DictReader(handle, delimiter="\t"))
    volatile = {"generated_at", "label"}
    assert {k: v for k, v in first.items() if k not in volatile} == {
        k: v for k, v in second.items() if k not in volatile
    }
