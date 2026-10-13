"""Training-period move-size statistics for the 50 target stocks; no model or broker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai_trader.costs import FIXED_CLIP_NOTIONAL, ZERODHA_INTRADAY_EQUITY
from ai_trader.history import CandleStore

SESSION_MINUTES = 375
FIRST_CALL = 24
LAST_CALL = 254
DEADLINE = 344
LEVELS_PERMILLE = (2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20)
TRAINING_START = "2024-01-01"
TRAINING_END = "2025-12-31"
HORIZONS = (15, 30, 60, 120)
STOCKS = 50
LOW_SENTINEL = 10**12
BASIS_POINT_BINS = 2001
HEADER = ["start_time", "open", "high", "low", "close", "volume"]


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_sessions(path: Path, quarantined: set[str]) -> tuple[dict, dict]:
    rows: dict[str, list] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != HEADER:
            raise ValueError(f"Unexpected candle header in {path.name}")
        for start, _, high, low, close, _ in reader:
            day = start[:10]
            if not TRAINING_START <= day <= TRAINING_END:
                continue
            if start[16:] != ":00+05:30":
                raise ValueError(f"Unexpected timestamp {start} in {path.name}")
            minute = int(start[11:13]) * 60 + int(start[14:16]) - 555
            if 0 <= minute < SESSION_MINUTES:
                rows[day].append((minute, high, low, close))
    sessions, excluded = {}, {"quarantined": 0, "incomplete_or_special": 0}
    for day, values in rows.items():
        if day in quarantined:
            excluded["quarantined"] += 1
            continue
        values.sort()
        if [value[0] for value in values] != list(range(SESSION_MINUTES)):
            excluded["incomplete_or_special"] += 1
            continue
        scaled = np.asarray([value[1:] for value in values], dtype=np.float64) * 100
        prices = np.rint(scaled).astype(np.int64)
        if (
            np.abs(scaled - prices).max() > 1e-6
            or (prices <= 0).any()
            or (prices[:, 0] < prices[:, 1:].max(axis=1)).any()
            or (prices[:, 1] > prices[:, 2]).any()
        ):
            raise ValueError(f"Invalid or non-paise prices on {day} in {path.name}")
        sessions[day] = prices
    return sessions, excluded


def sequential_moves(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, permille: int
) -> list[tuple[int, int, int]]:
    """(reference candle, touch candle, +1 up / -1 down / 0 both) per X% move."""
    moves = []
    reference = 0
    upper, lower = high * 1000, low * 1000
    while reference < len(close) - 1:
        base = close[reference]
        up = upper[reference + 1 :] >= base * (1000 + permille)
        down = lower[reference + 1 :] <= base * (1000 - permille)
        touched = up | down
        if not touched.any():
            break
        offset = int(touched.argmax())
        touch = reference + 1 + offset
        moves.append((reference, touch, int(up[offset]) - int(down[offset])))
        reference = touch
    return moves


def call_outcomes(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, permille: tuple[int, ...]
) -> dict:
    """First touch by 15:00 from every 09:40-13:30 close, both directions."""
    references = np.arange(FIRST_CALL, LAST_CALL + 1)
    future = np.arange(DEADLINE + 1)[None, :] > references[:, None]
    running_high = np.maximum.accumulate(
        np.where(future, high[None, : DEADLINE + 1], 0), axis=1
    )
    running_low = np.minimum.accumulate(
        np.where(future, low[None, : DEADLINE + 1], LOW_SENTINEL), axis=1
    )
    scaled_high, scaled_low = running_high * 1000, running_low * 1000
    base = close[references]
    rows = np.arange(len(references))
    long_return = close[DEADLINE] / base - 1
    outcomes = {"long_return_at_1500": long_return}
    for level in permille:
        # Rows are monotone, so the count of untouched columns is the first touch.
        for side, first, adverse in (
            (
                1,
                (scaled_high < (base * (1000 + level))[:, None]).sum(axis=1),
                running_low,
            ),
            (
                -1,
                (scaled_low > (base * (1000 - level))[:, None]).sum(axis=1),
                running_high,
            ),
        ):
            hit = first <= DEADLINE
            before = adverse[rows, np.clip(first - 1, 0, DEADLINE)]
            excursion = np.maximum(side * (base - before), 0) / base
            outcomes[(level, side)] = {
                "minutes": np.where(hit, first - references, -1),
                "adverse": np.where(hit & (first - 1 > references), excursion, 0.0),
            }
    return outcomes


def quantiles(histogram: np.ndarray, scale: float = 1.0) -> dict:
    total = int(histogram.sum())
    if not total:
        return {"count": 0}
    cumulative = np.cumsum(histogram)
    values = np.arange(len(histogram))
    return {
        "count": total,
        "mean": float((histogram * values).sum() / total) * scale,
        **{
            name: float(np.searchsorted(cumulative, share * total)) * scale
            for name, share in (
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("p90", 0.9),
            )
        },
    }


def call_summary(
    minutes: np.ndarray,
    adverse: np.ndarray,
    miss_sum: float,
    miss_count: int,
    gross_sum: float,
    calls: int,
    level: int,
    charges: float,
) -> dict:
    cumulative = np.cumsum(minutes)
    miss_mean = float(miss_sum) / max(int(miss_count), 1)
    move = level / 1000
    return {
        "tp_by_1500_percent": 100 * int(minutes.sum()) / calls,
        **{
            f"tp_within_{horizon}m_percent": 100 * int(cumulative[horizon]) / calls
            for horizon in HORIZONS
        },
        "minutes_to_touch": quantiles(minutes),
        "adverse_before_touch_percent": quantiles(adverse, scale=0.01),
        "no_touch_mean_return_at_1500_percent": 100 * miss_mean,
        "tp_or_1500_gross_mean_percent": 100 * float(gross_sum) / calls,
        "break_even_tp_by_1500_percent_after_charges": 100
        * (charges - miss_mean)
        / (move - miss_mean),
    }


def analyze(candles: Path, manifest: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; preserve earlier analyses")
    store = CandleStore(candles)
    quality = json.loads(store.verified_path("report.json").read_text("utf-8"))
    document = json.loads(manifest.read_text(encoding="utf-8"))
    symbols = sorted({shard["symbol"] for shard in document["shards"]})
    if len(symbols) != STOCKS:
        raise ValueError("Expected the 50 target stocks in the TP manifest")
    levels = len(LEVELS_PERMILLE)
    move_counts = np.zeros((levels, 3), dtype=np.int64)
    move_minutes = np.zeros((levels, SESSION_MINUTES), dtype=np.int64)
    tradable_moves = np.zeros(levels, dtype=np.int64)
    daily_moves: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(levels))
    daily_stocks: dict[str, int] = defaultdict(int)
    touch_minutes = np.zeros((levels, 2, DEADLINE + 1), dtype=np.int64)
    adverse_bps = np.zeros((levels, 2, BASIS_POINT_BINS), dtype=np.int64)
    miss_sum = np.zeros((levels, 2))
    miss_count = np.zeros((levels, 2), dtype=np.int64)
    gross_sum = np.zeros((levels, 2))
    drift_sum, calls, stock_sessions = 0.0, 0, 0
    excluded = {"quarantined": 0, "incomplete_or_special": 0}
    for symbol in symbols:
        label = f"NSE:{symbol}"
        sessions, dropped = read_sessions(
            store.verified_path(f"NSE_{symbol}.csv"),
            set(quality["instruments"][label]["quality"]["quarantined_dates"]),
        )
        for reason, count in dropped.items():
            excluded[reason] += count
        for day, prices in sessions.items():
            high, low, close = prices[:, 0], prices[:, 1], prices[:, 2]
            stock_sessions += 1
            daily_stocks[day] += 1
            for index, level in enumerate(LEVELS_PERMILLE):
                moves = sequential_moves(high, low, close, level)
                daily_moves[day][index] += len(moves)
                for reference, touch, direction in moves:
                    move_counts[index, {1: 0, -1: 1, 0: 2}[direction]] += 1
                    move_minutes[index, touch - reference] += 1
                    tradable_moves[index] += (
                        reference >= FIRST_CALL and touch <= DEADLINE
                    )
            outcomes = call_outcomes(high, low, close, LEVELS_PERMILLE)
            long_return = outcomes["long_return_at_1500"]
            drift_sum += float(long_return.sum())
            calls += len(long_return)
            for index, level in enumerate(LEVELS_PERMILLE):
                for column, side in enumerate((1, -1)):
                    result = outcomes[(level, side)]
                    hit = result["minutes"] >= 0
                    touch_minutes[index, column] += np.bincount(
                        result["minutes"][hit], minlength=DEADLINE + 1
                    )
                    adverse_bps[index, column] += np.bincount(
                        np.minimum(
                            np.floor(result["adverse"][hit] * 10_000).astype(int),
                            BASIS_POINT_BINS - 1,
                        ),
                        minlength=BASIS_POINT_BINS,
                    )
                    missed = side * long_return[~hit]
                    miss_sum[index, column] += float(missed.sum())
                    miss_count[index, column] += len(missed)
                    gross_sum[index, column] += float(
                        missed.sum() + hit.sum() * level / 1000
                    )
    sessions_used = len(daily_stocks)
    distinct = []
    for index, level in enumerate(LEVELS_PERMILLE):
        total = int(move_counts[index].sum())
        per_stock = total / stock_sessions
        distinct.append(
            {
                "move_percent": level / 10,
                "moves_per_session_50_stocks": STOCKS * per_stock,
                "moves_per_stock_session": per_stock,
                "raw_moves_per_session_available_stocks": float(
                    np.mean([daily_moves[day][index] for day in daily_stocks])
                ),
                "up_share_percent": 100 * move_counts[index, 0] / total,
                "down_share_percent": 100 * move_counts[index, 1] / total,
                "same_candle_both_share_percent": 100 * move_counts[index, 2] / total,
                "tradable_moves_per_session_50_stocks": STOCKS
                * tradable_moves[index]
                / stock_sessions,
                "minutes_per_move": quantiles(move_minutes[index]),
            }
        )
    from_call = []
    charges = float(ZERODHA_INTRADAY_EQUITY.round_trip_fraction(FIXED_CLIP_NOTIONAL))
    for index, level in enumerate(LEVELS_PERMILLE):
        sides = {
            name: call_summary(
                touch_minutes[index, column],
                adverse_bps[index, column],
                miss_sum[index, column],
                miss_count[index, column],
                gross_sum[index, column],
                calls,
                level,
                charges,
            )
            for column, name in enumerate(("long", "short"))
        }
        sides["both"] = call_summary(
            touch_minutes[index].sum(axis=0),
            adverse_bps[index].sum(axis=0),
            miss_sum[index].sum(),
            miss_count[index].sum(),
            gross_sum[index].sum(),
            2 * calls,
            level,
            charges,
        )
        from_call.append({"move_percent": level / 10, **sides})
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__), output / "analysis_source.py")
    report = {
        "status": "descriptive_training_period_statistics",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "candles_dataset_sha256": file_hash(candles / "dataset.json"),
            "candles_quality_report_sha256": file_hash(candles / "report.json"),
            "universe_manifest_sha256": file_hash(manifest),
            "analysis_source_sha256": file_hash(output / "analysis_source.py"),
        },
        "period": {
            "first": TRAINING_START,
            "last": TRAINING_END,
            "basis": (
                "training span of the October 10 scanner model; January-September "
                "2026 validation/evaluation dates are not read"
            ),
        },
        "universe": symbols,
        "sessions": sessions_used,
        "stock_sessions": stock_sessions,
        "mean_stocks_per_session": stock_sessions / sessions_used,
        "excluded_stock_sessions": excluded,
        "definitions": {
            "session": "complete 375-minute 09:15-15:30 regular session only",
            "touch": (
                "minute high >= close*(1+X) up or low <= close*(1-X) down, "
                "exact paise integers"
            ),
            "distinct_moves": (
                "start at the 09:15-09:16 close; each first +/-X touch is one move; "
                "the touching candle's close becomes the next reference; whole "
                "session; minutes = touch candle end minus reference close"
            ),
            "tradable_moves": "distinct moves starting >= 09:40 and ending <= 15:00",
            "from_call": (
                "every one-minute close 09:40-13:30; first touch through the "
                "14:59-15:00 candle; adverse = worst opposite excursion in candles "
                "before the touch candle"
            ),
            "per_session_50_stocks": "mean per stock-session times 50",
            "gross_mean": "TP credited at exactly X; otherwise 15:00 close return",
            "break_even": (
                "hit rate by 15:00 at which TP X and the unconditional mean miss "
                "return cover the estimated Zerodha charges at INR 100,000"
            ),
        },
        "charges_round_trip_fraction": charges,
        "unconditional_long_return_at_1500_percent": 100 * drift_sum / calls,
        "calls": calls,
        "distinct_moves": distinct,
        "from_call": from_call,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.candles, args.manifest, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("sessions", "stock_sessions", "excluded_stock_sessions")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
