"""Timestamped external-feed comparison and independent scanner-input arithmetic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import datetime, time, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ai_trader.broker import Instrument
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.features import FeatureEngine, FeatureSnapshot
from ai_trader.history import CandleStore
from ai_trader.market import Candle

SCANNER_INPUTS = (
    "ema9",
    "ema21",
    "ema50",
    "rsi14",
    "macd_histogram",
    "atr14",
    "adx14",
    "atr_pct",
    "rolling_high_20",
    "rolling_low_20",
    "bollinger_percent_b_20",
    "vwap",
    "price_vs_vwap_sigma",
    "volume_ratio_20",
    "opening_range_high",
    "opening_range_low",
    "minutes_since_session_open",
)


def _smooth(
    values: Sequence[Decimal | None],
    period: int,
    *,
    wilder: bool = False,
) -> list[Decimal | None]:
    output: list[Decimal | None] = []
    seed: list[Decimal] = []
    previous: Decimal | None = None
    alpha = Decimal(1) / period if wilder else Decimal(2) / (period + 1)
    for value in values:
        if value is None:
            output.append(None)
            continue
        if previous is None:
            seed.append(value)
            if len(seed) == period:
                previous = sum(seed) / period
        else:
            previous += alpha * (value - previous)
        output.append(previous)
    return output


def reference_inputs(candles: Sequence[Candle]) -> list[dict[str, Decimal | None]]:
    """Batch formulas; no production indicator or scoring helpers are called."""
    if not candles:
        return []
    with localcontext() as context:
        context.prec = 60
        return _reference_inputs(candles)


def _reference_inputs(bars: Sequence[Candle]) -> list[dict[str, Decimal | None]]:
    closes = [bar.close for bar in bars]
    averages = {period: _smooth(closes, period) for period in (9, 12, 21, 26, 50)}
    macd = [
        None if slow is None else fast - slow
        for fast, slow in zip(averages[12], averages[26], strict=True)
    ]
    macd_signal = _smooth(macd, 9)
    histogram = [
        None if signal is None else value - signal
        for value, signal in zip(macd, macd_signal, strict=True)
    ]
    changes = [None] + [
        current - previous for previous, current in zip(closes, closes[1:])
    ]
    gains = _smooth(
        [None if change is None else max(change, Decimal(0)) for change in changes],
        14,
        wilder=True,
    )
    losses = _smooth(
        [None if change is None else max(-change, Decimal(0)) for change in changes],
        14,
        wilder=True,
    )
    rsi_values = [
        None
        if gain is None
        else (
            Decimal(50)
            if gain == loss == 0
            else Decimal(100)
            if loss == 0
            else 100 * gain / (gain + loss)
        )
        for gain, loss in zip(gains, losses, strict=True)
    ]
    true_ranges = [bars[0].high - bars[0].low]
    positive_dm: list[Decimal | None] = [None]
    negative_dm: list[Decimal | None] = [None]
    for previous, current in zip(bars, bars[1:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
        upward = current.high - previous.high
        downward = previous.low - current.low
        positive_dm.append(max(upward, Decimal(0)) if upward > downward else Decimal(0))
        negative_dm.append(
            max(downward, Decimal(0)) if downward > upward else Decimal(0)
        )
    atr_values = _smooth(true_ranges, 14, wilder=True)
    directional_ranges = _smooth([None, *true_ranges[1:]], 14, wilder=True)
    smooth_positive = _smooth(positive_dm, 14, wilder=True)
    smooth_negative = _smooth(negative_dm, 14, wilder=True)
    dx_values = [
        None
        if positive is None or not movement_range
        else (
            Decimal(0)
            if positive + negative == 0
            else 100 * abs(positive - negative) / (positive + negative)
        )
        for positive, negative, movement_range in zip(
            smooth_positive, smooth_negative, directional_ranges, strict=True
        )
    ]
    adx_values = _smooth(dx_values, 14, wilder=True)
    references: list[dict[str, Decimal | None]] = []
    session_bars: list[Candle] = []
    session_date = None
    for index, bar in enumerate(bars):
        if bar.start_time.date() != session_date:
            session_date = bar.start_time.date()
            session_bars = []
        session_bars.append(bar)
        current_window = bars[max(0, index - 19) : index + 1]
        ready_window = len(current_window) == 20
        mean = sum(item.close for item in current_window) / 20 if ready_window else None
        deviation = (
            (sum((item.close - mean) ** 2 for item in current_window) / 20).sqrt()
            if ready_window
            else None
        )
        band_position = (
            (bar.close - (mean - 2 * deviation)) / (4 * deviation)
            if deviation
            else None
        )
        known_volume = all(item.volume is not None for item in session_bars)
        session_volume = (
            sum(item.volume for item in session_bars) if known_volume else 0
        )
        typical_prices = [
            (item.high + item.low + item.close) / 3 for item in session_bars
        ]
        vwap = (
            sum(
                price * item.volume
                for price, item in zip(typical_prices, session_bars, strict=True)
            )
            / session_volume
            if session_volume
            else None
        )
        vwap_deviation = (
            (
                sum(
                    item.volume * (price - vwap) ** 2
                    for price, item in zip(typical_prices, session_bars, strict=True)
                )
                / session_volume
            ).sqrt()
            if session_volume
            else None
        )
        sigma = (bar.close - vwap) / vwap_deviation if vwap_deviation else None
        earlier_volumes = [item.volume for item in session_bars[-21:-1]]
        ratio = None
        if (
            bar.volume is not None
            and len(earlier_volumes) == 20
            and all(value is not None for value in earlier_volumes)
            and sum(earlier_volumes)
        ):
            ratio = Decimal(bar.volume) * 20 / sum(earlier_volumes)
        opening_bars = [
            item for item in session_bars if item.start_time.time() < time(9, 30)
        ]
        opening_ready = bar.start_time.time() >= time(9, 30) and bool(opening_bars)
        references.append(
            {
                "ema9": averages[9][index],
                "ema21": averages[21][index],
                "ema50": averages[50][index],
                "rsi14": rsi_values[index],
                "macd_histogram": histogram[index],
                "atr14": atr_values[index],
                "adx14": adx_values[index],
                "atr_pct": None
                if atr_values[index] is None
                else atr_values[index] / bar.close,
                "rolling_high_20": max(item.high for item in current_window)
                if ready_window
                else None,
                "rolling_low_20": min(item.low for item in current_window)
                if ready_window
                else None,
                "bollinger_percent_b_20": band_position,
                "vwap": vwap,
                "price_vs_vwap_sigma": sigma,
                "volume_ratio_20": ratio,
                "opening_range_high": max(item.high for item in opening_bars)
                if opening_ready
                else None,
                "opening_range_low": min(item.low for item in opening_bars)
                if opening_ready
                else None,
                "minutes_since_session_open": Decimal(
                    bar.end_time.hour * 60 + bar.end_time.minute - 555
                ),
            }
        )
    return references


def snapshots(bars: Sequence[Candle]) -> tuple[FeatureSnapshot, ...]:
    engine = FeatureEngine()
    return tuple(
        snapshot for bar in bars if (snapshot := engine.update(bar)) is not None
    )


def validate_arithmetic(bars: Sequence[Candle]) -> dict:
    worst = dict.fromkeys(SCANNER_INPUTS, Decimal(0))
    numeric = 0
    unavailable = 0
    for bar, actual, reference in zip(
        bars, snapshots(bars), reference_inputs(bars), strict=True
    ):
        for name, expected in reference.items():
            observed = getattr(actual, name)
            if (observed is None) != (expected is None):
                raise ValueError(f"Readiness mismatch at {bar.start_time} for {name}")
            if expected is None:
                unavailable += 1
                continue
            error = abs(observed - expected)
            if error > max(Decimal("1e-16"), abs(expected) * Decimal("1e-20")):
                raise ValueError(f"Reference mismatch at {bar.start_time} for {name}")
            worst[name] = max(worst[name], error)
            numeric += 1
    return {
        "bars": len(bars),
        "numeric_comparisons": numeric,
        "unavailable_agreements": unavailable,
        "max_absolute_error": {name: str(value) for name, value in worst.items()},
        "status": "passed",
    }


def parse_yahoo(payload: dict, instrument: Instrument) -> tuple[Candle, ...]:
    results = payload.get("chart", {}).get("result")
    if not results or len(results) != 1:
        raise ValueError("Yahoo returned no single chart result")
    chart = results[0]
    if (
        chart["meta"]["symbol"] != instrument.trading_symbol + ".NS"
        or chart["meta"].get("dataGranularity") != "1m"
    ):
        raise ValueError("Yahoo symbol or candle interval does not match")
    quotes = chart["indicators"]["quote"][0]
    fields = ("open", "high", "low", "close", "volume")
    if any(len(quotes[name]) != len(chart["timestamp"]) for name in fields):
        raise ValueError("Yahoo quote arrays are not timestamp-aligned")
    bars: list[Candle] = []
    for index, epoch in enumerate(chart["timestamp"]):
        begins = datetime.fromtimestamp(epoch, INDIA_TIMEZONE)
        if not time(9, 15) <= begins.time() < time(15, 30):
            continue
        if any(quotes[name][index] is None for name in fields[:4]):
            continue
        prices = {name: Decimal(str(quotes[name][index])) for name in fields[:4]}
        for name, value in prices.items():
            rounded = value.quantize(Decimal("0.01"))
            if abs(rounded - value) > Decimal("0.0001"):
                raise ValueError("Yahoo price has more than float representation noise")
            prices[name] = rounded
        bars.append(
            Candle(
                instrument=instrument,
                start_time=begins,
                end_time=begins + timedelta(minutes=1),
                **prices,
                volume=quotes["volume"][index],
            )
        )
    if not bars or len({bar.start_time for bar in bars}) != len(bars):
        raise ValueError("Yahoo bars are empty or contain duplicate timestamps")
    return tuple(sorted(bars, key=lambda bar: bar.start_time))


def compare_feeds(
    groww: Sequence[Candle], yahoo: Sequence[Candle]
) -> tuple[dict, list[dict]]:
    left = {bar.start_time: bar for bar in groww}
    right = {bar.start_time: bar for bar in yahoo}
    common = sorted(left.keys() & right.keys())
    if not common:
        raise ValueError("No overlapping timestamps across feeds")
    market_fields = ("open", "high", "low", "close", "volume")
    exact = {
        name: sum(
            getattr(left[stamp], name) == getattr(right[stamp], name)
            for stamp in common
        )
        for name in market_fields
    }
    market_differences = {
        name: str(
            max(
                (
                    abs(getattr(left[stamp], name) - getattr(right[stamp], name))
                    for stamp in common
                    if getattr(left[stamp], name) is not None
                    and getattr(right[stamp], name) is not None
                ),
                default=Decimal(0),
            )
        )
        for name in market_fields
    }
    left_features = {item.candle_start_time: item for item in snapshots(groww)}
    right_features = {
        bar.start_time: values
        for bar, values in zip(yahoo, reference_inputs(yahoo), strict=True)
    }
    rows: list[dict] = []
    matched = dict.fromkeys(SCANNER_INPUTS, 0)
    comparable = dict.fromkeys(SCANNER_INPUTS, 0)
    unavailable_mismatches = dict.fromkeys(SCANNER_INPUTS, 0)
    feature_differences = dict.fromkeys(SCANNER_INPUTS, Decimal(0))
    for stamp in common:
        for name in SCANNER_INPUTS:
            actual = getattr(left_features[stamp], name)
            external = right_features[stamp][name]
            difference = (
                None if actual is None or external is None else abs(actual - external)
            )
            unavailable_mismatches[name] += (actual is None) != (external is None)
            if difference is not None:
                feature_differences[name] = max(feature_differences[name], difference)
            agrees = difference is not None and difference <= max(
                Decimal("1e-8"), abs(actual) * Decimal("1e-10")
            )
            comparable[name] += difference is not None
            matched[name] += agrees
            rows.append(
                {
                    "bar_start_ist": stamp.isoformat(),
                    "decision_time_ist": left_features[
                        stamp
                    ].candle_end_time.isoformat(),
                    "feature": name,
                    "groww": "" if actual is None else str(actual),
                    "yahoo": "" if external is None else str(external),
                    "absolute_difference": ""
                    if difference is None
                    else str(difference),
                    "within_tolerance": agrees,
                }
            )
    return {
        "common_minutes": len(common),
        "groww_only_minutes": len(left.keys() - right.keys()),
        "yahoo_only_minutes": len(right.keys() - left.keys()),
        "exact_ohlcv_matches": exact,
        "comparable_features": comparable,
        "max_absolute_ohlcv_difference": market_differences,
        "max_absolute_feature_difference": {
            name: str(value) for name, value in feature_differences.items()
        },
        "readiness_mismatches": unavailable_mismatches,
        "features_within_tolerance": matched,
        "all_comparable_features_match": all(
            matched[name] == comparable[name] for name in SCANNER_INPUTS
        ),
        "close_matches_by_minute_offset": {
            str(offset): sum(
                left[stamp].close == right[stamp + timedelta(minutes=offset)].close
                for stamp in left
                if stamp + timedelta(minutes=offset) in right
            )
            for offset in (-1, 0, 1)
        },
        "method": (
            "Production features on Groww versus 60-digit reference formulas on "
            "Yahoo, each using its own complete feed from the same requested start. "
            "No missing candle or volume is filled and no provider is presumed "
            "authoritative."
        ),
    }, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/candles"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only comparison outputs, not the cached reference.",
    )
    args = parser.parse_args(argv)
    try:
        start = datetime.fromisoformat(args.start + "T09:15:00+05:30")
        end = datetime.fromisoformat(args.end + "T15:30:00+05:30")
        instrument = Instrument("NSE", "RELIANCE")
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS?"
            + urlencode(
                {
                    "period1": int(start.timestamp()),
                    "period2": int(end.timestamp()),
                    "interval": "1m",
                }
            )
        )
        if args.fetch:
            if args.reference.exists():
                raise ValueError(
                    "Reference file already exists; use it offline or choose a new file"
                )
            with urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30
            ) as response:
                raw = response.read()
            parse_yahoo(json.loads(raw, parse_float=Decimal), instrument)
            args.reference.parent.mkdir(parents=True, exist_ok=True)
            args.reference.write_bytes(raw)
        yahoo = tuple(
            bar
            for bar in parse_yahoo(
                json.loads(args.reference.read_bytes(), parse_float=Decimal), instrument
            )
            if start <= bar.start_time and bar.end_time <= end
        )
        groww = tuple(
            bar
            for bar in CandleStore(args.cache).load(instrument, start, end)
            if time(9, 15) <= bar.start_time.time() < time(15, 30)
        )
        if not groww or not yahoo:
            raise ValueError("Both feeds need bars inside the requested window")
        comparison, rows = compare_feeds(groww, yahoo)
        report = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "external_source": url,
            "reference_file": args.reference.as_posix(),
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "comparison_file": args.reference.with_suffix(".comparison.csv").as_posix(),
            "reference_precision": 60,
            "arithmetic_groww": validate_arithmetic(groww),
            "arithmetic_yahoo": validate_arithmetic(yahoo),
            "feed_parity": comparison,
            "limitations": (
                "This compares independently sourced candles and independently "
                "calculated formulas, not TradingView's displayed indicator values. "
                "EMA initialization, missing bars and volume differences can "
                "prevent cross-feed equality."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "x"
        with args.output.open(mode, encoding="utf-8", newline="") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with args.reference.with_suffix(".comparison.csv").open(
            mode, encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except (ArithmeticError, OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "arithmetic_groww": report["arithmetic_groww"],
                "arithmetic_yahoo": report["arithmetic_yahoo"],
                "feed_parity": comparison,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
