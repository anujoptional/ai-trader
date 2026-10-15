"""NSE F&O bhavcopy rows in one schema, from either NSE format.

Columns (numpy arrays of one session): ``symbol``; ``kind`` "F" (future) or "O"
(option); ``underlying`` "S" (stock) or "I" (index); ``expiry`` (datetime64[D]);
``strike`` (NaN for futures); ``option`` "CE", "PE" or "" (futures); ``open``,
``high``, ``low``, ``close``, ``settle`` (rupees; NSE writes 0 for the open/high/low
of a contract that did not trade); ``contracts`` traded; ``value`` traded (rupees);
``oi`` open interest and ``chg_oi`` its change (units of the underlying, as NSE
publishes them); and from the UDiFF format only (NaN / 0 in the old one):
``last``, ``prev_close``, ``underlying_price``, ``lot``.

Old format: INSTRUMENT FUTIDX/FUTSTK/OPTIDX/OPTSTK (and FUTIVX, VIX futures),
OPTION_TYP "XX" for futures, VAL_INLAKH in lakh. UDiFF: FinInstrmTp IDF/IDO/STF/STO,
TtlTradgVol in contracts, TtlTrfVal in rupees, OpnIntrst in units.
"""

from __future__ import annotations

import csv
import functools
import io
from datetime import date
from pathlib import Path

import numpy as np

from ai_trader.fno.nse_archive import DATASETS, nse_date

TEXT = ("symbol", "kind", "underlying", "option")
FLOATS = ("strike", "open", "high", "low", "close", "settle", "value")
FLOATS += ("last", "prev_close", "underlying_price")
INTEGERS = ("contracts", "oi", "chg_oi", "lot")
COLUMNS = TEXT + ("expiry",) + FLOATS + INTEGERS
UDIFF_KINDS = {
    "STF": ("F", "S"),
    "STO": ("O", "S"),
    "IDF": ("F", "I"),
    "IDO": ("O", "I"),
}


def _number(text: str) -> float:
    text = text.strip()
    return float(text) if text else float("nan")


@functools.lru_cache(maxsize=4096)
def _date(text: str) -> date:
    return nse_date(text)


def _integer(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    value = float(text)
    if value != int(value):
        raise ValueError(f"not a whole number: {text!r}")
    return int(value)


def _table(records: dict[str, list]) -> dict[str, np.ndarray]:
    table = {}
    for name in TEXT:
        table[name] = np.array(records[name], dtype=str)
    table["expiry"] = np.array(records["expiry"], dtype="datetime64[D]")
    for name in FLOATS:
        table[name] = np.array(records[name], dtype=np.float64)
    for name in INTEGERS:
        table[name] = np.array(records[name], dtype=np.int64)
    return table


def parse_udiff(text: str, day: date) -> dict[str, np.ndarray]:
    reader = csv.reader(io.StringIO(text))
    header = [c.strip() for c in next(reader)]
    at = {name: i for i, name in enumerate(header)}
    records: dict[str, list] = {name: [] for name in COLUMNS}
    for row in reader:
        if not any(c.strip() for c in row):
            continue
        if _date(row[at["TradDt"]]) != day:
            raise ValueError(f"row of {row[at['TradDt']]} in {day}'s file")
        kind, underlying = UDIFF_KINDS[row[at["FinInstrmTp"]].strip()]
        option = row[at["OptnTp"]].strip()
        if (kind == "O") != (option in ("CE", "PE")):
            raise ValueError(f"option type {option!r} for {kind}")
        records["symbol"].append(row[at["TckrSymb"]].strip())
        records["kind"].append(kind)
        records["underlying"].append(underlying)
        records["option"].append(option)
        records["expiry"].append(_date(row[at["XpryDt"]]))
        records["strike"].append(
            _number(row[at["StrkPric"]]) if kind == "O" else float("nan")
        )
        for name, column in (
            ("open", "OpnPric"),
            ("high", "HghPric"),
            ("low", "LwPric"),
            ("close", "ClsPric"),
            ("settle", "SttlmPric"),
            ("value", "TtlTrfVal"),
            ("last", "LastPric"),
            ("prev_close", "PrvsClsgPric"),
            ("underlying_price", "UndrlygPric"),
        ):
            records[name].append(_number(row[at[column]]))
        for name, column in (
            ("contracts", "TtlTradgVol"),
            ("oi", "OpnIntrst"),
            ("chg_oi", "ChngInOpnIntrst"),
            ("lot", "NewBrdLotQty"),
        ):
            records[name].append(_integer(row[at[column]]))
    return _table(records)


def parse_old(text: str, day: date) -> dict[str, np.ndarray]:
    reader = csv.reader(io.StringIO(text))
    header = [c.strip() for c in next(reader)]
    at = {name: i for i, name in enumerate(header)}
    records: dict[str, list] = {name: [] for name in COLUMNS}
    for row in reader:
        if not any(c.strip() for c in row):
            continue
        if _date(row[at["TIMESTAMP"]]) != day:
            raise ValueError(f"row of {row[at['TIMESTAMP']]} in {day}'s file")
        instrument = row[at["INSTRUMENT"]].strip()
        if instrument[:3] not in ("FUT", "OPT"):
            raise ValueError(f"unknown instrument {instrument!r}")
        kind = "F" if instrument.startswith("FUT") else "O"
        option = row[at["OPTION_TYP"]].strip()
        if kind == "F":
            if option not in ("XX", ""):
                raise ValueError(f"option type {option!r} on a future")
            option = ""
        elif option not in ("CE", "PE"):
            raise ValueError(f"option type {option!r}")
        records["symbol"].append(row[at["SYMBOL"]].strip())
        records["kind"].append(kind)
        records["underlying"].append("S" if instrument.endswith("STK") else "I")
        records["option"].append(option)
        records["expiry"].append(_date(row[at["EXPIRY_DT"]]))
        records["strike"].append(
            _number(row[at["STRIKE_PR"]]) if kind == "O" else float("nan")
        )
        for name, column in (
            ("open", "OPEN"),
            ("high", "HIGH"),
            ("low", "LOW"),
            ("close", "CLOSE"),
            ("settle", "SETTLE_PR"),
        ):
            records[name].append(_number(row[at[column]]))
        records["value"].append(_number(row[at["VAL_INLAKH"]]) * 100_000)
        for name in ("last", "prev_close", "underlying_price"):
            records[name].append(float("nan"))
        for name, column in (
            ("contracts", "CONTRACTS"),
            ("oi", "OPEN_INT"),
            ("chg_oi", "CHG_IN_OI"),
        ):
            records[name].append(_integer(row[at[column]]))
        records["lot"].append(0)
    return _table(records)


def load_session(root: Path, day: date) -> tuple[dict[str, np.ndarray], str] | None:
    """The session's rows from the UDiFF file, else the old one: (table, source)."""
    for name, parse in (("fo", parse_udiff), ("fo_old", parse_old)):
        spec = DATASETS[name]
        path = spec.path(root, day)
        if path.exists():
            return parse(spec.text(path.read_bytes(), day), day), name
    return None


def save_table(path: Path, table: dict[str, np.ndarray], source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part.npz")
    np.savez_compressed(part, source=np.array(source), **table)
    part.replace(path)


def load_table(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in COLUMNS}, str(data["source"])
