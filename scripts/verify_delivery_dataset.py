"""Independent re-computation of delivery labels and features for a random sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

TP, SL = 0.0125, 0.0075
HORIZON = 10


def load_minutes(path: Path) -> dict[str, list[tuple]]:
    sessions: dict[str, list[tuple]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for start, o, h, low, c, v in reader:
            sessions.setdefault(start[:10], []).append(
                (start[11:16], float(o), float(h), float(low), float(c), int(v or 0))
            )
    return sessions


def load_daily(path: Path) -> dict[str, tuple]:
    with path.open(encoding="utf-8") as handle:
        return {
            r["timestamp"][:10]: tuple(
                float(r[k]) for k in ("open", "high", "low", "close", "volume")
            )
            for r in csv.DictReader(handle)
        }


def factor(events: list, day: str) -> float:
    value = 1.0
    for ex, multiplier, _ in events:
        if ex > day:
            value *= multiplier
    return value


def label(symbol_minutes, symbol_daily, events, calendar, day) -> dict:
    session = symbol_minutes[day]
    entry = next(row for row in session if row[0] == "14:59")
    price = entry[4] * factor(events, day)
    take, stop = price * (1 + TP), price * (1 - SL)
    position = calendar.index(day)
    window = calendar[position : position + HORIZON + 1]
    for current in window:
        scale = factor(events, current)
        rows = symbol_minutes.get(current)
        if rows is None:
            bar = symbol_daily.get(current)
            if bar is None:
                return {"status": "missing_day"}
            o, h, low, c = (value * scale for value in bar[:4])
            if o >= take:
                return {"label": 1, "exit": current, "gross": o / price - 1}
            if o <= stop:
                return {"label": 0, "exit": current, "gross": o / price - 1}
            if h >= take and low <= stop:
                return {"status": "daily_ambiguous"}
            if h >= take:
                return {"label": 1, "exit": current, "gross": take / price - 1}
            if low <= stop:
                return {"label": 0, "exit": current, "gross": stop / price - 1}
            continue
        for clock, o, h, low, c, _ in rows:
            if current == day and clock <= "14:59":
                continue
            o, h, low = o * scale, h * scale, low * scale
            up, down = h >= take, low <= stop
            if up and down:
                if o >= take:
                    return {"label": 1, "exit": current, "gross": o / price - 1}
                fill = o if o <= stop else stop
                return {"label": 0, "exit": current, "gross": fill / price - 1}
            if up:
                return {"label": 1, "exit": current, "gross": max(take, o) / price - 1}
            if down:
                return {"label": 0, "exit": current, "gross": min(stop, o) / price - 1}
    if len(window) < HORIZON + 1:
        return {"status": "censored"}
    last = window[-1]
    rows = symbol_minutes.get(last)
    close = rows[-1][4] if rows else symbol_daily[last][3]
    return {"label": 0, "exit": last, "gross": close * factor(events, last) / price - 1}


def features(symbol_minutes, symbol_daily, events, day, nifty_daily) -> dict:
    session = symbol_minutes[day]
    today = [row for row in session if row[0] <= "14:59"]
    scale = factor(events, day)
    price = today[-1][4] * scale
    high = max(row[2] for row in today) * scale
    low = min(row[3] for row in today) * scale
    history = sorted(d for d in symbol_daily if d < day)
    adjusted = {
        d: [v * factor(events, d) for v in symbol_daily[d][:4]] for d in history[-130:]
    }
    closes = [adjusted[d][3] for d in history[-130:]]
    true_ranges = []
    tail = history[-15:]
    for previous, current in zip(tail[:-1], tail[1:], strict=True):
        h, low_, prev_close = (
            adjusted[current][1],
            adjusted[current][2],
            adjusted[previous][3],
        )
        true_ranges.append(max(h, prev_close) - min(low_, prev_close))
    series = closes[-60:] + [price]
    gains = [max(b - a, 0.0) for a, b in zip(series[:-1], series[1:], strict=True)]
    losses = [max(a - b, 0.0) for a, b in zip(series[:-1], series[1:], strict=True)]
    gain, loss = sum(gains[:14]) / 14, sum(losses[:14]) / 14
    for up, down in zip(gains[14:], losses[14:], strict=True):
        gain, loss = (gain * 13 + up) / 14, (loss * 13 + down) / 14
    return {
        "ret_5d": math.log(price / closes[-5]),
        "ret_21d": math.log(price / closes[-21]),
        "ma_dist_20": math.log(price / ((sum(closes[-19:]) + price) / 20)),
        "atr_14": sum(true_ranges) / 14 / price,
        "rsi_14": 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss),
        "clv_today": (price - low) / (high - low) if high > low else 0.5,
        "dist_high_20": math.log(
            price / max(max(adjusted[d][1] for d in history[-19:]), high)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()
    data = np.load(args.study / "dataset.npz", allow_pickle=False)
    audit = json.loads((args.study / "corporate_actions_audit.json").read_text())
    names = list(data["feature_names"])
    symbols = sorted({str(s) for s in data["symbols"]})
    dailies = {s: load_daily(args.candles / "daily" / f"NSE_{s}.csv") for s in symbols}
    calendar = sorted(set().union(*(set(d) for d in dailies.values())))
    rng = random.Random(20261011)
    picks = sorted(rng.sample(range(len(data["labels"])), args.samples))
    by_symbol: dict[str, list[int]] = {}
    for index in picks:
        by_symbol.setdefault(str(data["symbols"][index]), []).append(index)
    label_mismatch, feature_errors, checked = [], {}, 0
    for symbol, indices in sorted(by_symbol.items()):
        minutes = load_minutes(args.candles / f"NSE_{symbol}.csv")
        events = [tuple(e) for e in audit[symbol]["applied_factors"]]
        for index in indices:
            day = str(data["dates"][index])
            mine = label(minutes, dailies[symbol], events, calendar, day)
            theirs = (
                int(data["labels"][index]),
                str(data["exit_dates"][index]),
                float(data["gross"][index]),
            )
            if (
                "label" not in mine
                or (mine["label"], mine["exit"]) != theirs[:2]
                or abs(mine["gross"] - theirs[2]) > 1e-9
            ):
                label_mismatch.append(
                    {
                        "symbol": symbol,
                        "date": day,
                        "independent": mine,
                        "study": theirs,
                    }
                )
            for name, value in features(
                minutes, dailies[symbol], events, day, None
            ).items():
                error = abs(value - float(data["features"][index, names.index(name)]))
                scale = max(1.0, abs(value))
                feature_errors[name] = max(feature_errors.get(name, 0.0), error / scale)
            checked += 1
    result = {
        "checked": checked,
        "label_mismatches": label_mismatch,
        "max_relative_feature_error": feature_errors,
        "feature_tolerance": 1e-5,
        "passed": not label_mismatch
        and all(v <= 1e-5 for v in feature_errors.values()),
    }
    (args.study / "verification.json").write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "label_mismatches"}
            | {"label_mismatches": len(label_mismatch)}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
