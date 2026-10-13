"""Daily history on NSE's price basis: official prices, structural actions only.

Kite back-adjusts its history for corporate actions, including dividends of about
2% of the price or more, and it adjusted some demergers by a wrong ratio or not at
all (VEDL, TMPV, GUJENERGY in 2025-26). The scanner was trained on the research
basis of ``scripts/build_price_basis.py``, which this reproduces for the session:

- a session's prices are NSE's official bhavcopy row (``sec_bhavdata_full``, EQ
  series, else BE; renamed symbols followed through NSE's ``symbolchange.csv``)
  times S, the product of the ratios of later structural actions; its volume is
  divided by S;
- a structural action (split, consolidation, bonus, rights, demerger, scheme,
  capital reduction, from NSE's corporate actions) takes Kite's own adjustment at
  its ex-date when that is a real split/bonus ratio (within 0.5%), otherwise NSE's
  ex-date open over the stock's previous close: the whole opening move;
- dividends are not adjusted (nor are they in NSE's adjusted 52-week high).

A stock is left out for the day when its basis cannot be established: a structural
action today that is not a clean split/bonus (its opening move is not known before
the open), a session missing from NSE's files where a later action makes Kite's
row differ from NSE's, Kite and NSE disagreeing on a session nothing has adjusted
since, or Kite adjusting its history where NSE lists no action.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from ai_trader.news.nse import HEADERS, parse_nse_date

BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/products/content/"
    "sec_bhavdata_full_{day:%d%m%Y}.csv"
)
SYMBOL_CHANGES_URL = (
    "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"
)
SERIES = ("EQ", "BE")
FIELDS = ("OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY")
KINDS = (
    ("split", ("split", "sub-division", "sub division")),
    ("consolidation", ("consolidat",)),
    ("bonus", ("bonus",)),
    ("demerger", ("demerger", "de-merger", "spin off", "spin-off")),
    ("rights", ("rights",)),
    ("capital_reduction", ("reduction",)),
    ("scheme", ("amalgamation", "merger", "scheme of arrangement")),
    ("buyback", ("buy back", "buyback")),
    ("dividend", ("dividend",)),
)
STRUCTURAL = frozenset(
    ("split", "consolidation", "bonus", "demerger", "rights", "capital_reduction")
    + ("scheme",)
)
PRICE_KINDS = STRUCTURAL | {"dividend"}  # what Kite may adjust its history for
# Price ratios of real splits (face value F1 -> F2) and bonuses (a new for b held:
# b / (a + b)), as the research basis snaps them.
SPLIT_BONUS_RATIOS = frozenset(
    {Fraction(f2, f1) for f1 in (10, 5, 2) for f2 in (1, 2, 5) if f2 < f1}
    | {Fraction(b, a + b) for a in range(1, 11) for b in range(1, 6)}
    | {Fraction(10, 11)}
)
SNAP_TOLERANCE = 0.005
UNADJUSTED_TOLERANCE = 0.005  # also the largest unexplained step in Kite's history
ACTION_CHUNK = timedelta(days=90)

Row = tuple[float, float, float, float, float]


def action_kind(subject: str) -> str:
    text = subject.lower()
    for kind, words in KINDS:
        if any(word in text for word in words):
            return kind
    return "other"


def snap(ratio: float) -> float | None:
    """The real split/bonus ratio within 0.5% of ``ratio``, if there is one."""
    best = min(SPLIT_BONUS_RATIOS, key=lambda f: abs(float(f) / ratio - 1))
    return float(best) if abs(float(best) / ratio - 1) <= SNAP_TOLERANCE else None


def read_bhavcopy(
    text: str, day: date | None = None, symbols: set[str] | None = None
) -> dict[str, Row]:
    """EQ rows (else BE) of one NSE ``sec_bhavdata_full`` file, by symbol.

    Raises ValueError when it is not a bhavcopy, or not ``day``'s."""
    reader = csv.reader(io.StringIO(text))
    header = [name.strip() for name in next(reader, [])]
    index = {name: i for i, name in enumerate(header)}
    lacking = [n for n in ("SYMBOL", "SERIES", "DATE1", *FIELDS) if n not in index]
    if lacking:
        raise ValueError(f"not an NSE bhavcopy (no {', '.join(lacking)})")
    rows: dict[str, Row] = {}
    checked = day is None
    for raw in reader:
        if len(raw) < len(header):
            continue
        if not checked:
            stamp = parse_nse_date(raw[index["DATE1"]].strip())
            if stamp != day:
                raise ValueError(f"bhavcopy of {stamp}, expected {day}")
            checked = True
        series, symbol = raw[index["SERIES"]].strip(), raw[index["SYMBOL"]].strip()
        if series not in SERIES or (symbols is not None and symbol not in symbols):
            continue
        if symbol in rows and series != "EQ":
            continue
        try:
            rows[symbol] = tuple(float(raw[index[n]]) for n in FIELDS)
        except ValueError:
            continue
    return rows


def symbol_history(text: str) -> dict[str, list[tuple[date, str]]]:
    """For each current symbol, its (change date, previous symbol), newest first."""
    changes: dict[str, tuple[date, str]] = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 4:
            continue
        old, new = row[1].strip(), row[2].strip()
        changed = parse_nse_date(row[3].strip().title())
        if old and new and changed and old != new:
            changes[new] = (changed, old)
    history: dict[str, list[tuple[date, str]]] = {}
    for symbol in changes:
        chain, current, seen = [], symbol, set()
        while current in changes and current not in seen:
            seen.add(current)
            changed, old = changes[current]
            chain.append((changed, old))
            current = old
        history[symbol] = chain
    return history


def nse_symbol(symbol: str, day: date, history: Sequence[tuple[date, str]]) -> str:
    """The symbol NSE listed ``symbol`` under on ``day``."""
    current = symbol
    for changed, old in history:
        if day < changed:
            current = old
        else:
            break
    return current


@dataclass(frozen=True)
class Rebased:
    """One stock's daily rows on the basis, or why it is left out today."""

    daily: np.ndarray | None
    events: tuple[dict[str, Any], ...] = ()
    reason: str | None = None
    # Basis volume over Kite's, per session: maps Kite's minute history alike.
    volume_scale: Mapping[date, float] = field(default_factory=dict)


def rebase(
    dates: Sequence[date],
    kite: np.ndarray,
    nse: Mapping[date, Row],
    actions: Iterable[tuple[date, str]],
    today: date,
) -> Rebased:
    """``kite``: Kite's daily rows (open, high, low, close, volume) on ``dates``,
    completed sessions oldest first; ``nse``: NSE's rows on those dates (raw);
    ``actions``: NSE's corporate actions of the stock as (ex-date, subject)."""
    if not len(dates):
        return Rebased(None, reason="no daily history")
    kinds = [(x, action_kind(s)) for x, s in actions if dates[0] < x <= today]
    acted = sorted({x for x, kind in kinds if kind in PRICE_KINDS})
    structural: dict[date, list[str]] = {}
    for x, kind in kinds:
        if kind in STRUCTURAL:
            structural.setdefault(x, []).append(kind)
    ratio = {
        d: float(kite[i, 3]) / nse[d][3]
        for i, d in enumerate(dates)
        if d in nse and nse[d][3] > 0 and kite[i, 3] > 0
    }

    def left_out(reason: str) -> Rebased:
        return Rebased(None, reason=reason)

    # Kite and NSE agree wherever nothing can have adjusted a session since.
    last = acted[-1] if acted else None
    for i, d in enumerate(dates):
        if d in nse and (last is None or d >= last):
            mine, theirs = kite[i], nse[d]
            if (
                not np.allclose(mine[:4], theirs[:4], rtol=0, atol=1e-6)
                or abs(mine[4] - theirs[4]) > 0.5
            ):
                return left_out(f"Kite and NSE differ on {d}")
    # Every step in Kite's adjustment falls on an NSE action's ex-date.
    known = [d for d in dates if d in ratio]
    for before, after in zip(known, known[1:], strict=False):
        step = ratio[before] / ratio[after]
        if abs(step - 1) > UNADJUSTED_TOLERANCE and not any(
            before < x <= after for x in acted
        ):
            return left_out(f"Kite adjusted its history at {after} for no NSE action")

    events = []
    for x, kinds in sorted(structural.items()):
        before = [d for d in dates if d < x]
        after = [d for d in dates if d >= x]
        cum, ex = before[-1], after[0] if after else None
        if cum not in ratio or (ex is not None and ex not in ratio):
            return left_out(f"no NSE row around the {'/'.join(kinds)} of {x}")
        step = ratio[cum] / (ratio[ex] if ex is not None else 1.0)
        clean = snap(step) if abs(step - 1) > UNADJUSTED_TOLERANCE else None
        if clean is not None:
            value, source = clean, "split/bonus ratio (Kite's, snapped)"
        elif ex is not None:
            value, source = nse[ex][0] / nse[cum][3], "NSE ex-date open / close before"
        else:
            return left_out(
                f"{'/'.join(kinds)} on {x}: its ratio is known only after the open"
            )
        events.append(
            {
                "ex_date": x,
                "kinds": kinds,
                "ratio": value,
                "source": source,
                "kite_step": step,
            }
        )
    daily = np.empty((len(dates), 5))
    scale: dict[date, float] = {}
    for i, d in enumerate(dates):
        factor = 1.0
        for event in events:
            if d < event["ex_date"]:
                factor *= event["ratio"]
        if d in nse:
            o, h, lo, c, v = nse[d]
        elif not any(x > d for x in acted):
            o, h, lo, c, v = (float(x) for x in kite[i])  # as traded: equals NSE's
        else:
            return left_out(f"no NSE row for {d}, adjusted by Kite since")
        daily[i] = (o * factor, h * factor, lo * factor, c * factor, v / factor)
        if kite[i, 4] > 0 and abs(daily[i, 4] / kite[i, 4] - 1) > 1e-9:
            scale[d] = float(daily[i, 4] / kite[i, 4])
    return Rebased(daily, tuple(events), None, scale)


def _download(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": HEADERS["User-Agent"], "Accept": "*/*"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class NseDailyArchive:
    """NSE's daily bhavcopies, one file per session, kept in ``folder``."""

    def __init__(
        self,
        folder: Path,
        *,
        download: Callable[[str], bytes] = _download,
        pause: float = 0.3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.folder = folder
        self._download = download
        self._pause = pause
        self._sleep = sleep

    def path(self, day: date) -> Path:
        return self.folder / f"{day.isoformat()}.csv"

    def sync(self, days: Iterable[date]) -> tuple[list[date], list[date]]:
        """Download the sessions not yet kept: (downloaded, still missing)."""
        added, missing = [], []
        for day in sorted(set(days)):
            path = self.path(day)
            if path.exists():
                continue
            blob = None
            for attempt in range(2):
                try:
                    blob = self._download(BHAVCOPY_URL.format(day=day))
                    read_bhavcopy(blob.decode("utf-8"), day)
                    break
                except Exception:
                    blob = None
                    self._sleep(2.0 * (attempt + 1))
            if blob is None:
                missing.append(day)
                continue
            self.folder.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_bytes(blob)
            part.replace(path)
            added.append(day)
            self._sleep(self._pause)
        return added, missing

    def rows(self, day: date, symbols: set[str] | None = None) -> dict[str, Row]:
        path = self.path(day)
        if not path.exists():
            return {}
        return read_bhavcopy(path.read_text("utf-8"), day, symbols)


def fetch_actions(nse: Any, start: date, end: date) -> list[Any]:
    """NSE's corporate actions with ex-dates from ``start`` to ``end``."""
    actions: list[Any] = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + ACTION_CHUNK)
        actions.extend(nse.corporate_actions(cursor, stop))
        cursor = stop + timedelta(days=1)
    return actions


class PriceBasis:
    """The NSE price basis for one session: ``load`` once, then ``rebase``."""

    def __init__(
        self,
        folder: Path,
        nse: Any,
        news: Any = None,
        *,
        download: Callable[[str], bytes] = _download,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.folder = folder
        self.nse = nse
        self.news = news
        self._download = download
        self._sleep = sleep
        self.history: dict[str, list[tuple[date, str]]] = {}
        self.rows: dict[date, dict[str, Row]] = {}
        self.actions: dict[str, list[tuple[date, str]]] = {}

    def load(
        self, sessions: Sequence[date], symbols: Iterable[str], today: date
    ) -> list[str]:
        """Renames, NSE's daily files for ``sessions`` and corporate actions up to
        ``today``; returns notes for the run log. Raises when the actions are
        unavailable from NSE and from the stored copy."""
        notes = []
        changes = self.folder / "symbolchange.csv"
        try:
            text = self._download(SYMBOL_CHANGES_URL).decode("utf-8", "replace")
            if len(symbol_history(text)) < 100:
                raise ValueError("too few symbol changes")
            changes.parent.mkdir(parents=True, exist_ok=True)
            changes.write_text(text, "utf-8")
        except Exception as error:
            text = changes.read_text("utf-8") if changes.exists() else ""
            notes.append(
                f"NSE symbol changes not downloaded ({type(error).__name__}); "
                + ("using the stored copy" if text else "renames unknown")
            )
        self.history = symbol_history(text)
        wanted = list(symbols)
        names = {
            n for s in wanted for n in (s, *(o for _, o in self.history.get(s, [])))
        }
        archive = NseDailyArchive(
            self.folder / "daily", download=self._download, sleep=self._sleep
        )
        added, missing = archive.sync(sessions)
        self.rows = {day: archive.rows(day, names) for day in sessions}
        notes.append(
            f"NSE daily files: {len(sessions) - len(missing)} of {len(sessions)} "
            f"sessions ({len(added)} downloaded)"
            + (f"; missing {', '.join(map(str, missing))}" if missing else "")
        )
        try:
            fetched = fetch_actions(self.nse, sessions[0], today)
            if self.news is not None:
                self.news.add_actions(fetched)
            pairs = [(a.symbol, a.ex_date, a.subject) for a in fetched if a.ex_date]
            notes.append(f"NSE corporate actions: {len(fetched)} since {sessions[0]}")
        except Exception as error:
            if self.news is None:
                raise
            pairs = [
                (name, date.fromisoformat(r["ex_date"]), r["subject"])
                for name in names
                for r in self.news.actions(name, sessions[0], today)
                if r["ex_date"]
            ]
            notes.append(
                f"NSE corporate actions not fetched ({type(error).__name__}); "
                f"using the {len(pairs)} stored for these stocks"
            )
        self.actions = {}
        for name, ex_date, subject in pairs:
            self.actions.setdefault(name, []).append((ex_date, subject))
        return notes

    def rebase(
        self, symbol: str, dates: Sequence[date], kite: np.ndarray, today: date
    ) -> Rebased:
        history = self.history.get(symbol, [])
        nse = {}
        for day in dates:
            row = self.rows.get(day, {}).get(nse_symbol(symbol, day, history))
            if row is not None:
                nse[day] = row
        names = {symbol, *(old for _, old in history)}
        actions = [a for name in names for a in self.actions.get(name, [])]
        return rebase(dates, kite, nse, actions, today)
