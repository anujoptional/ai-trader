"""Offline feature/dataset diagnostics, not a trainer or executable trade policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from math import sqrt
from pathlib import Path
from statistics import StatisticsError, correlation, fmean, quantiles

from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.features import DERIVED_FEATURE_NAMES, FeatureEngine, FeatureSnapshot
from ai_trader.history import CandleStore
from ai_trader.market import Candle

HORIZONS = (6, 15, 30, 60)
FEATURE_COLUMNS = (
    "return_5",
    "market_excess_return_15",
    "intraday_return_60m",
    "ema_spread_atr",
    "adx_fraction",
    "directional_pressure",
    "bollinger_position",
    "vwap_z",
    "atr_fraction",
    "volatility_expansion",
    "same_minute_rvol",
    "body_atr",
    "bar_close_location",
    "session_range_position",
    "opening_gap",
    "prior_5_session_return",
    "prior_20_session_return",
    "prior_5_session_range",
    "prior_5_session_position",
    "session_fraction",
    "market_return_15",
    "sector_excess_return_15",
    "vix_level",
    "gold_return_15",
)
CONTEXT_SYMBOLS = ("NIFTY", "BANKNIFTY", "NIFTYIT", "INDIAVIX", "GOLDBEES")
TARGET_SYMBOLS = ("RELIANCE", "HDFCBANK", "INFY", "SBIN")
SECTOR_SYMBOLS = {"HDFCBANK": "BANKNIFTY", "SBIN": "BANKNIFTY", "INFY": "NIFTYIT"}


def _ratio(numerator, denominator) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _number(value) -> float | None:
    return None if value is None else float(value)


def _difference(left, right):
    return None if left is None or right is None else left - right


def feature_vector(
    snapshot: FeatureSnapshot, context: Mapping[str, Mapping]
) -> dict[str, float | None]:
    """Normalized candidates using only snapshots closing at the decision time."""
    for item in context.values():
        if item["as_of"] > snapshot.candle_end_time:
            raise ValueError("Future context cannot enter a decision row")
    aligned = {
        name: item
        for name, item in context.items()
        if item["as_of"] == snapshot.candle_end_time
    }
    market = aligned.get("NIFTY", {}).get("return_15")
    sector = aligned.get(
        SECTOR_SYMBOLS.get(snapshot.instrument.trading_symbol), {}
    ).get("return_15")
    directional_total = (
        None
        if snapshot.plus_di14 is None or snapshot.minus_di14 is None
        else snapshot.plus_di14 + snapshot.minus_di14
    )
    rv60 = _number(snapshot.realized_volatility_60m)
    values = {
        "return_5": _number(snapshot.return_5),
        "market_excess_return_15": _number(_difference(snapshot.return_15, market)),
        "intraday_return_60m": _number(snapshot.intraday_return_60m),
        "ema_spread_atr": _ratio(
            _difference(snapshot.ema9, snapshot.ema21), snapshot.atr14
        ),
        "adx_fraction": _ratio(snapshot.adx14, Decimal(100)),
        "directional_pressure": _ratio(
            _difference(snapshot.plus_di14, snapshot.minus_di14), directional_total
        ),
        "bollinger_position": _number(snapshot.bollinger_percent_b_20),
        "vwap_z": _number(snapshot.price_vs_vwap_sigma),
        "atr_fraction": _number(snapshot.atr_pct),
        "volatility_expansion": _ratio(
            _number(snapshot.realized_volatility_20m),
            None if rv60 is None else rv60 / sqrt(3),
        ),
        "same_minute_rvol": _number(snapshot.volume_ratio_same_minute_5),
        "body_atr": _ratio(snapshot.close - snapshot.open, snapshot.atr14),
        "bar_close_location": _ratio(
            snapshot.close - snapshot.low, snapshot.high - snapshot.low
        ),
        "session_range_position": _number(snapshot.position_in_session_range),
        "opening_gap": _number(snapshot.gap_from_previous_close),
        "prior_5_session_return": _number(snapshot.prior_5_session_return),
        "prior_20_session_return": _number(snapshot.prior_20_session_return),
        "prior_5_session_range": _number(snapshot.prior_5_session_range_pct),
        "prior_5_session_position": _number(snapshot.position_in_prior_5_session_range),
        "session_fraction": _ratio(snapshot.minutes_since_session_open, Decimal(375)),
        "market_return_15": _number(market),
        "sector_excess_return_15": _number(_difference(sector, market)),
        "vix_level": _number(aligned.get("INDIAVIX", {}).get("close")),
        "gold_return_15": _number(aligned.get("GOLDBEES", {}).get("return_15")),
    }
    assert tuple(values) == FEATURE_COLUMNS
    return values


def diagnostic_event(
    signal: Candle,
    subsequent: Sequence[Candle],
    side: int,
    fraction: Decimal = Decimal("0.002"),
) -> dict:
    """Next-open, unrounded barriers; skip entry bar and make double touches SL."""
    if side not in (-1, 1) or not fraction.is_finite() or not 0 < fraction < 1:
        raise ValueError("Invalid side or symmetric barrier fraction")
    entered_at = signal.end_time
    deadline = entered_at + timedelta(minutes=HORIZONS[-1])
    cutoff = datetime.combine(
        entered_at.astimezone(INDIA_TIMEZONE).date(), time(15, 15), INDIA_TIMEZONE
    )
    if deadline > cutoff:
        raise ValueError("The full diagnostic horizon must precede square-off")
    future = tuple(subsequent[: HORIZONS[-1]])
    if len(future) != HORIZONS[-1]:
        raise ValueError("Incomplete future horizon")
    for offset, bar in enumerate(future):
        if (
            bar.instrument != signal.instrument
            or bar.start_time != entered_at + timedelta(minutes=offset)
            or bar.end_time != bar.start_time + timedelta(minutes=1)
        ):
            raise ValueError("Future horizon crosses an instrument or missing minute")
    entry = future[0].open
    upper, lower = entry * (1 + fraction), entry * (1 - fraction)
    result = "TIMEOUT"
    event_minutes = HORIZONS[-1]
    ambiguous = False
    for bar in future[1:]:
        target = bar.high >= upper if side == 1 else bar.low <= lower
        stop = bar.low <= lower if side == 1 else bar.high >= upper
        if target or stop:
            result = "SL" if stop else "TP"
            ambiguous = target and stop
            event_minutes = int((bar.end_time - entered_at).total_seconds() / 60)
            break
    bucket = next(
        index for index, horizon in enumerate(HORIZONS) if event_minutes <= horizon
    )
    outcome_class = 8 if result == "TIMEOUT" else bucket + (4 if result == "SL" else 0)
    return {
        "label_event": result,
        "label_event_minutes": event_minutes,
        "label_class": outcome_class,
        "label_double_touch": ambiguous,
        "label_entry_price": str(entry),
        "label_tp_price": str(upper if side == 1 else lower),
        "label_sl_price": str(lower if side == 1 else upper),
        **{
            f"label_tp_{horizon}m": int(result == "TP" and event_minutes <= horizon)
            for horizon in HORIZONS
        },
    }


def dataset_row(
    snapshot: FeatureSnapshot, context: Mapping[str, Mapping], label: dict, side: int
) -> dict:
    values = feature_vector(snapshot, context)
    return {
        "symbol": snapshot.instrument.trading_symbol,
        "decision_time_ist": snapshot.candle_end_time.astimezone(
            INDIA_TIMEZONE
        ).isoformat(),
        "side": side,
        "tp_fraction": 0.002,
        **values,
        **{f"missing_{name}": int(value is None) for name, value in values.items()},
        **label,
    }


def summarize_outcomes(rows: Sequence[dict]) -> dict:
    return {
        "actions": len(rows),
        "sessions": len({row["decision_time_ist"][:10] for row in rows}),
        "symbols": len({row["symbol"] for row in rows}),
        "tp_rate": {
            str(horizon): fmean(row[f"label_tp_{horizon}m"] for row in rows)
            if rows
            else None
            for horizon in HORIZONS
        },
        "event_counts": dict(Counter(row["label_event"] for row in rows)),
        "double_touches": sum(row["label_double_touch"] for row in rows),
    }


def pattern_report(rows: Sequence[dict]) -> dict:
    july = [row for row in rows if row["decision_time_ist"].startswith("2026-07")]
    cuts = quantiles(
        [row["atr_fraction"] for row in july if row["atr_fraction"] is not None],
        n=4,
        method="inclusive",
    )
    result = {"july_atr_quartiles": cuts, "periods": {}}
    for month in ("2026-07", "2026-08"):
        month_rows = [row for row in rows if row["decision_time_ist"].startswith(month)]
        groups = {
            "all": month_rows,
            "low_volatility": [
                row
                for row in month_rows
                if row["atr_fraction"] is not None and row["atr_fraction"] <= cuts[0]
            ],
            "high_volatility": [
                row
                for row in month_rows
                if row["atr_fraction"] is not None and row["atr_fraction"] > cuts[2]
            ],
            "momentum_aligned": [
                row
                for row in month_rows
                if row["market_excess_return_15"] is not None
                and row["side"] * row["market_excess_return_15"] > 0
            ],
            "momentum_opposed": [
                row
                for row in month_rows
                if row["market_excess_return_15"] is not None
                and row["side"] * row["market_excess_return_15"] < 0
            ],
            "volume_surge": [
                row
                for row in month_rows
                if row["same_minute_rvol"] is not None
                and row["same_minute_rvol"] >= 1.5
            ],
            "normal_volume": [
                row
                for row in month_rows
                if row["same_minute_rvol"] is not None and row["same_minute_rvol"] < 1.5
            ],
            "first_hour": [
                row for row in month_rows if row["session_fraction"] <= 60 / 375
            ],
            "later": [row for row in month_rows if row["session_fraction"] > 60 / 375],
            "long": [row for row in month_rows if row["side"] == 1],
            "short": [row for row in month_rows if row["side"] == -1],
        }
        result["periods"][month] = {
            name: summarize_outcomes(group) for name, group in groups.items()
        }
        result["periods"][month]["by_symbol"] = {
            symbol: summarize_outcomes(
                [row for row in month_rows if row["symbol"] == symbol]
            )
            for symbol in TARGET_SYMBOLS
        }
    return result


def redundant_pairs(rows: Sequence[dict], names: Sequence[str]) -> list[dict]:
    by_symbol = defaultdict(list)
    for row in rows:
        if row["decision_time_ist"].startswith("2026-07"):
            by_symbol[row["symbol"]].append(row)
    pairs = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            correlations = {}
            counts = {}
            for symbol, observations in by_symbol.items():
                known = [
                    (row[left], row[right])
                    for row in observations
                    if row[left] is not None and row[right] is not None
                ]
                if len(known) < 100:
                    continue
                try:
                    correlations[symbol] = correlation(
                        [pair[0] for pair in known], [pair[1] for pair in known]
                    )
                    counts[symbol] = len(known)
                except StatisticsError:
                    continue
            if len(correlations) == len(TARGET_SYMBOLS):
                floor = min(abs(value) for value in correlations.values())
                if floor >= 0.90:
                    pairs.append(
                        {
                            "left": left,
                            "right": right,
                            "minimum_absolute_correlation": floor,
                            "within_symbol_correlations": correlations,
                            "paired_rows": counts,
                        }
                    )
    return sorted(
        pairs, key=lambda item: item["minimum_absolute_correlation"], reverse=True
    )[:20]


def missingness(rows: Sequence[dict], names: Sequence[str]) -> dict:
    return {
        name: {
            "missing": sum(row[name] is None for row in rows),
            "rows": len(rows),
            "fraction": fmean(row[name] is None for row in rows) if rows else None,
        }
        for name in names
    }


def run_study(cache: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(
            "Choose a new output directory to preserve earlier diagnostics"
        )
    start = datetime(2026, 6, 1, 9, 15, tzinfo=INDIA_TIMEZONE)
    end = datetime(2026, 8, 31, 15, 30, tzinfo=INDIA_TIMEZONE)
    store = CandleStore(cache)
    source_hashes = {}
    context = {}
    rows = []
    raw = []
    exclusions = Counter()

    def load(symbol):
        from ai_trader.broker import Instrument

        instrument = Instrument("NSE", symbol)
        source_hashes[symbol] = hashlib.sha256(
            store.path_for(instrument).read_bytes()
        ).hexdigest()
        return tuple(
            bar
            for bar in store.load(instrument, start, end)
            if time(9, 15) <= bar.start_time.time() < time(15, 30)
        )

    for symbol in CONTEXT_SYMBOLS:
        engine = FeatureEngine()
        observations = {}
        for bar in load(symbol):
            snapshot = engine.update(bar)
            if snapshot is not None and snapshot.candle_end_time.date() >= date(
                2026, 7, 1
            ):
                observations[snapshot.candle_end_time] = {
                    "as_of": snapshot.candle_end_time,
                    "return_15": snapshot.return_15,
                    "close": snapshot.close,
                }
        context[symbol] = observations
        print(f"Context {symbol}: {len(observations)} snapshots", flush=True)

    for symbol in TARGET_SYMBOLS:
        bars = load(symbol)
        engine = FeatureEngine()
        count_before = len(rows)
        for index, bar in enumerate(bars):
            snapshot = engine.update(bar)
            if snapshot is None or snapshot.candle_end_time.date() < date(2026, 7, 1):
                continue
            minute = int(snapshot.minutes_since_session_open)
            if minute % 5 or not 5 <= minute <= 300:
                continue
            raw.append(
                {
                    "symbol": symbol,
                    "decision_time_ist": snapshot.candle_end_time.isoformat(),
                    **{
                        name: _number(getattr(snapshot, name))
                        for name in DERIVED_FEATURE_NAMES
                    },
                }
            )
            contexts = {
                name: observations[snapshot.candle_end_time]
                for name, observations in context.items()
                if snapshot.candle_end_time in observations
            }
            for side in (1, -1):
                try:
                    label = diagnostic_event(bar, bars[index + 1 : index + 62], side)
                except ValueError as error:
                    exclusions[str(error)] += 1
                    continue
                rows.append(dataset_row(snapshot, contexts, label, side))
        print(
            f"Target {symbol}: {len(rows) - count_before} labeled side-actions",
            flush=True,
        )
    ordered = sorted(
        rows, key=lambda row: (row["decision_time_ist"], row["symbol"], -row["side"])
    )
    sample = [row for row in ordered if "T10:30:00" in row["decision_time_ist"]][:6]
    unique_inputs = [row for row in rows if row["side"] == 1]
    report = {
        "purpose": (
            "Descriptive dataset/feature diagnostics; "
            "no neural network or trading policy fitted"
        ),
        "warmup": "2026-06-01 through 2026-06-30",
        "analysis_periods": ["2026-07", "2026-08"],
        "targets": TARGET_SYMBOLS,
        "contexts": CONTEXT_SYMBOLS,
        "source_sha256": source_hashes,
        "code_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                *(Path(__file__).resolve().parents[1] / "src/ai_trader/features").glob(
                    "*.py"
                ),
            )
        },
        "feature_columns": FEATURE_COLUMNS,
        "other_inputs": [
            "side",
            "tp_fraction",
            *[f"missing_{name}" for name in FEATURE_COLUMNS],
        ],
        "not_inputs": ["symbol", "decision_time_ist", "all label_* columns"],
        "label_contract": {
            "fraction": "0.002",
            "entry": "next candle open; no fees, friction or tick rounding",
            "entry_bar": "excluded",
            "double_touch": "SL for either side",
            "deadline": "entry plus 60 minutes, at or before 15:15 IST",
            "resolution": (
                "event recorded at candle end, not an observed intrabar hit time"
            ),
            "classes": [
                "TP_0_6",
                "TP_6_15",
                "TP_15_30",
                "TP_30_60",
                "SL_0_6",
                "SL_6_15",
                "SL_15_30",
                "SL_30_60",
                "TIMEOUT_60",
            ],
        },
        "rows": len(rows),
        "decision_candidates": len(raw),
        "excluded_side_actions": dict(exclusions),
        "patterns": pattern_report(rows),
        "original_feature_missingness": missingness(raw, DERIVED_FEATURE_NAMES),
        "candidate_feature_missingness": missingness(unique_inputs, FEATURE_COLUMNS),
        "raw_feature_redundancy": redundant_pairs(raw, DERIVED_FEATURE_NAMES),
        "candidate_feature_redundancy": redundant_pairs(unique_inputs, FEATURE_COLUMNS),
        "sample_selection": (
            "First six chronological eligible rows at 10:30 IST, "
            "alphabetic symbol and LONG before SHORT; not chosen by outcome"
        ),
        "limitations": [
            "Four stocks and two months, not the whole universe/history "
            "or an untouched final test.",
            "Overlapping future horizons, paired directions and correlated "
            "stocks are not independent observations.",
            "Fixed diagnostic TP/SL/horizons are not tuned or claimed optimal.",
            "Barriers omit historical tick schedules, fills, spread and costs; "
            "labels are not production-training approval.",
            "Correlations are within-stock July-only and do not establish "
            "redundant predictive information.",
            "Raw 5/15-period returns are observed-candle returns and can bridge "
            "gaps/overnight; strict new 60m returns do not.",
            "No energy-sector context was fetched; RELIANCE sector input "
            "is unavailable, not a guessed index.",
            "No fabricated volume, price imputation, data upload, "
            "NN training or live order.",
        ],
    }
    output.mkdir(parents=True)
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output / "sample.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "sample_rows": len(sample),
                "report": str(output / "report.json"),
            }
        )
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/candles"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_study(args.cache, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
