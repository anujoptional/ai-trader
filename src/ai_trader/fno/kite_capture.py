"""Kite minute bars with open interest for F&O contracts, recorded day by day.

Kite serves minute history only for contracts still trading, at most 60 days per
request; once a contract expires its intraday history is gone, and NSE publishes no
intraday F&O history. So after each paper session this records, for every live NFO
future (stocks and indices) and the options of the nearest two expiries within 10% of
the underlying's last price (NIFTY and BANKNIFTY: 4%), every minute bar (open, high,
low, close, volume, open interest) from the session after the contract's last capture,
or 59 days back for a contract not seen before, up to today. Futures go first, then
index options, then stock options nearest expiry first, so a capture cut short by its
deadline loses the least; the next run continues where it stopped.

Layout under ``root`` (data/fno/kite_minute):
- <YYYY-MM-DD>.npz per session: ``contract`` (id in contracts.json), ``minute``
  (minutes after 09:15), ``open``/``high``/``low``/``close`` in paise, ``volume``,
  ``oi`` (all int64), sorted by contract and minute;
- contracts.json: id -> tradingsymbol, token, underlying, kind (FUT/CE/PE), expiry,
  strike, lot size, tick size;
- state.json: each contract's last captured session;
- parts/: batches not yet merged into their session files (merged at the end of a
  run, or by the next one after an interruption).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.clock import INDIA_TIMEZONE

INDEX_NAMES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")
INDEX_OPTIONS = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK"}
STOCK_BAND, INDEX_BAND, EXPIRIES, LOOKBACK_DAYS = 0.10, 0.04, 2, 59
BATCH = 400
FIELDS = ("contract", "minute", "open", "high", "low", "close", "volume", "oi")


def _expiry(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def paise(price: Decimal) -> int:
    value = price * 100
    if value != value.to_integral_value():
        raise ValueError(f"price {price} is not in whole paise")
    return int(value)


def select_contracts(
    instruments: list[dict[str, Any]],
    prices: dict[str, Decimal],
    today: date,
    *,
    stock_band: float = STOCK_BAND,
    index_band: float = INDEX_BAND,
    expiries: int = EXPIRIES,
) -> list[dict[str, Any]]:
    """Live futures of every underlying, then index and stock options of the nearest
    ``expiries`` expiries within the band of the last price, in capture order."""
    live = [
        r
        for r in instruments
        if r.get("exchange") == "NFO"
        and r.get("segment") in ("NFO-FUT", "NFO-OPT")
        and _expiry(r["expiry"]) >= today
    ]
    futures = sorted(
        (r for r in live if r["segment"] == "NFO-FUT"),
        key=lambda r: (r["name"], _expiry(r["expiry"])),
    )
    options_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in live:
        if r["segment"] == "NFO-OPT" and r["instrument_type"] in ("CE", "PE"):
            options_by_name[r["name"]].append(r)
    index_options, stock_options = [], []
    for name, rows in sorted(options_by_name.items()):
        if name in INDEX_NAMES:
            if name not in INDEX_OPTIONS:
                continue
            price, band, bucket = (
                prices.get(INDEX_OPTIONS[name]),
                index_band,
                index_options,
            )
        else:
            price, band, bucket = prices.get(f"NSE:{name}"), stock_band, stock_options
        if price is None or price <= 0:
            continue
        nearest = sorted({_expiry(r["expiry"]) for r in rows})[:expiries]
        low, high = float(price) * (1 - band), float(price) * (1 + band)
        bucket.extend(
            r
            for r in rows
            if _expiry(r["expiry"]) in nearest and low <= float(r["strike"]) <= high
        )
    stock_options.sort(
        key=lambda r: (_expiry(r["expiry"]), r["name"], float(r["strike"]))
    )
    index_options.sort(
        key=lambda r: (_expiry(r["expiry"]), r["name"], float(r["strike"]))
    )
    return futures + index_options + stock_options


class MinuteStore:
    """Session files, contract register and capture state under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "parts").mkdir(parents=True, exist_ok=True)
        self._contracts_path = root / "contracts.json"
        self._state_path = root / "state.json"
        self.contracts: dict[str, dict[str, Any]] = (
            json.loads(self._contracts_path.read_text("utf-8"))
            if self._contracts_path.exists()
            else {}
        )
        self._ids = {c["tradingsymbol"]: int(i) for i, c in self.contracts.items()}
        self.state: dict[str, str] = (
            json.loads(self._state_path.read_text("utf-8"))
            if self._state_path.exists()
            else {}
        )

    def contract_id(self, row: dict[str, Any]) -> int:
        symbol = row["tradingsymbol"]
        if symbol not in self._ids:
            new = len(self._ids)
            self._ids[symbol] = new
            self.contracts[str(new)] = {
                "tradingsymbol": symbol,
                "token": int(row["instrument_token"]),
                "underlying": row["name"],
                "kind": "FUT"
                if row["segment"] == "NFO-FUT"
                else row["instrument_type"],
                "expiry": _expiry(row["expiry"]).isoformat(),
                "strike": float(row["strike"]) if row["segment"] == "NFO-OPT" else None,
                "lot_size": int(row["lot_size"]),
                "tick_size": float(row["tick_size"]),
            }
        return self._ids[symbol]

    def _write_json(self, path: Path, value: object) -> None:
        part = path.with_name(path.name + ".part")
        part.write_text(json.dumps(value, indent=0, sort_keys=True), "utf-8")
        part.replace(path)

    def flush(self, batch: dict[str, list[np.ndarray]], done: dict[str, str]) -> None:
        """Write a batch of bars by session, then mark its contracts captured."""
        stamp = f"{time.time_ns()}"
        for day, chunks in batch.items():
            rows = np.concatenate(chunks)
            np.savez_compressed(
                self.root / "parts" / f"{day}_{stamp}.npz",
                **{name: rows[:, i] for i, name in enumerate(FIELDS)},
            )
        self._write_json(self._contracts_path, self.contracts)
        self.state.update(done)
        self._write_json(self._state_path, self.state)

    def merge(self) -> list[str]:
        """Merge every pending part into its session file (later rows win)."""
        parts: dict[str, list[Path]] = defaultdict(list)
        for path in sorted((self.root / "parts").glob("*.npz")):
            parts[path.name.split("_")[0]].append(path)
        for day, paths in sorted(parts.items()):
            target = self.root / f"{day}.npz"
            frames = [target] if target.exists() else []
            frames += paths
            columns = {name: [] for name in FIELDS}
            for frame in frames:
                with np.load(frame) as data:
                    for name in FIELDS:
                        columns[name].append(data[name])
            table = {name: np.concatenate(columns[name]) for name in FIELDS}
            key = table["contract"] * 1000 + table["minute"]
            # Keep the last occurrence of each (contract, minute): later parts win.
            order = np.argsort(key, kind="stable")
            last = np.r_[key[order][1:] != key[order][:-1], True]
            keep = order[last]
            temporary = target.with_name(target.name + ".part.npz")
            np.savez_compressed(temporary, **{n: table[n][keep] for n in FIELDS})
            temporary.replace(target)
            for path in paths:
                path.unlink()
        return sorted(parts)


def capture(
    broker: Any,
    today: date,
    root: Path,
    log: Callable[[str], None],
    *,
    deadline: datetime | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(INDIA_TIMEZONE),
) -> dict[str, Any]:
    """Record minute bars with OI of today's selection; returns a summary."""
    store = MinuteStore(root)
    leftover = store.merge()
    if leftover:
        log(f"F&O capture: merged unfinished parts of {', '.join(leftover)}")
    instruments = broker.nfo_instruments()
    names = sorted({r["name"] for r in instruments if r.get("segment") == "NFO-OPT"})
    keys = [INDEX_OPTIONS[n] for n in names if n in INDEX_OPTIONS]
    keys += [f"NSE:{n}" for n in names if n not in INDEX_NAMES]
    prices = broker.ltp_map(keys)
    chosen = select_contracts(instruments, prices, today)
    opening = datetime.combine(today, datetime.min.time(), INDIA_TIMEZONE)
    end = opening.replace(hour=15, minute=30)
    summary = {"contracts": len(chosen), "fetched": 0, "bars": 0, "skipped": 0}
    summary.update(failed=0, stopped_at_deadline=False)
    batch: dict[str, list[np.ndarray]] = defaultdict(list)
    done: dict[str, str] = {}
    log(
        f"F&O capture: {len(chosen)} contracts "
        f"({sum(r['segment'] == 'NFO-FUT' for r in chosen)} futures)"
    )
    for count, row in enumerate(chosen, 1):
        if deadline is not None and now() >= deadline:
            summary["stopped_at_deadline"] = True
            break
        symbol = row["tradingsymbol"]
        last = store.state.get(symbol)
        first = (
            date.fromisoformat(last) + timedelta(days=1)
            if last
            else today - timedelta(days=LOOKBACK_DAYS)
        )
        first = max(first, today - timedelta(days=LOOKBACK_DAYS))
        if first > today:
            summary["skipped"] += 1
            continue
        start = datetime.combine(first, datetime.min.time(), INDIA_TIMEZONE).replace(
            hour=9, minute=15
        )
        try:
            bars = broker.get_minute_bars_with_oi(
                int(row["instrument_token"]), start, end
            )
            by_day: dict[str, list[list[int]]] = defaultdict(list)
            for stamp, o, h, low, c, volume, oi in bars:
                opening_minute = stamp.replace(
                    hour=9, minute=15, second=0, microsecond=0
                )
                minute = int((stamp - opening_minute).total_seconds() // 60)
                if minute < 0:
                    continue
                by_day[stamp.date().isoformat()].append(
                    [minute, *(paise(p) for p in (o, h, low, c)), volume, oi]
                )
        except Exception as error:  # one contract's failure never stops the rest
            summary["failed"] += 1
            if summary["failed"] <= 5:
                log(f"F&O capture: {symbol} failed: {error}")
            continue
        ident = store.contract_id(row)
        for day, rows in by_day.items():
            batch[day].append(
                np.array([[ident, *values] for values in rows], dtype=np.int64)
            )
        summary["fetched"] += 1
        summary["bars"] += len(bars)
        done[symbol] = today.isoformat()
        if count % BATCH == 0:
            store.flush(batch, done)
            batch, done = defaultdict(list), {}
            log(f"F&O capture: {count}/{len(chosen)} contracts, {summary['bars']} bars")
    store.flush(batch, done)
    summary["sessions"] = store.merge()
    log(
        f"F&O capture done: {summary['fetched']} contracts, {summary['bars']} bars, "
        f"{summary['failed']} failed"
        + (", stopped at the deadline" if summary["stopped_at_deadline"] else "")
    )
    return summary
