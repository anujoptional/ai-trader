"""NSE's public daily archives behind the F&O-derived features, kept as downloaded.

One file per session and dataset under ``data/nse`` (official NSE files, no login):

- ``fo``: the F&O bhavcopy in NSE's UDiFF format (from about January 2024): every
  futures and options contract of the day, OHLC, settlement price, contracts
  traded, turnover, open interest and its change, underlying price, lot size.
- ``fo_old``: the same in NSE's previous format (``foDDMONYYYYbhav.csv``), the only
  one before 2024; NSE stopped it after 5 July 2024. Both formats exist in
  between, which lets the two parsers be checked against each other.
- ``cm``: the cash-market ``sec_bhavdata_full`` (official OHLC, average price,
  deliverable quantity and percentage), in ``data/nse/daily`` as paper/basis.py
  keeps it.
- ``participant_oi`` / ``participant_vol``: open interest and volume by client type
  (Client, DII, FII, Pro) in index and stock futures and options.
- ``ban``: stocks in the F&O ban period for the trade date.
- ``mwpl``: each F&O stock's open interest against its market-wide position limit.
- ``indices``: every NSE index's official OHLC, volume, turnover, P/E, P/B and
  dividend yield, and India VIX (``ind_close_all``; about 80 indices in 2020, 170
  in 2026).

Every file is checked on arrival (the zip is intact, the header is the dataset's,
the file is that session's) and written atomically; a 404 means NSE has no such
file (a holiday, a date before the dataset starts, or not published yet).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from xml.etree import ElementTree

from ai_trader.news.nse import HEADERS
from ai_trader.paper.basis import read_bhavcopy

ARCHIVE = "https://nsearchives.nseindia.com"
OLD_ARCHIVE = "https://archives.nseindia.com"  # tried when ARCHIVE answers 404
UDIFF_FROM = date(2023, 10, 1)  # tried from here; NSE's first UDiFF F&O file is later
OLD_FO_UNTIL = date(2024, 7, 5)  # the last session of the old F&O format
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN")
MONTHS += ("JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
UDIFF_FO_COLUMNS = (
    "TradDt",
    "FinInstrmTp",
    "TckrSymb",
    "XpryDt",
    "StrkPric",
    "OptnTp",
    "OpnPric",
    "HghPric",
    "LwPric",
    "ClsPric",
    "LastPric",
    "PrvsClsgPric",
    "UndrlygPric",
    "SttlmPric",
    "OpnIntrst",
    "ChngInOpnIntrst",
    "TtlTradgVol",
    "TtlTrfVal",
    "NewBrdLotQty",
)
OLD_FO_COLUMNS = (
    "INSTRUMENT",
    "SYMBOL",
    "EXPIRY_DT",
    "STRIKE_PR",
    "OPTION_TYP",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "SETTLE_PR",
    "CONTRACTS",
    "VAL_INLAKH",
    "OPEN_INT",
    "CHG_IN_OI",
    "TIMESTAMP",
)
PARTICIPANTS = ("Client", "DII", "FII", "Pro", "TOTAL")


class NotPublished(Exception):
    """HTTP 404: NSE has no such file."""


class WrongDay(ValueError):
    """The file is another session's (NSE serves the previous one on some holidays)."""


def nse_date(text: str) -> date:
    """NSE's dates: 03-JAN-2020, 03-Jan-2020, 26-Feb-20, 2020-01-03, 03-01-2020,
    Jan 03 2020, Oct 15, 2026, Jul 04,2024, Jan 28, 2020."""
    value = " ".join(text.replace(",", " ").split()).rstrip(".")
    for pattern in ("%d-%b-%Y", "%Y-%m-%d", "%b %d %Y", "%d-%b-%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(
                value.title() if "-" in value else value, pattern
            ).date()
        except ValueError:
            continue
    raise ValueError(f"not an NSE date: {text!r}")


def _unzip_csv(blob: bytes, member: str | None) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) > 1 and member is not None:
                names = [n for n in names if n.lower() == member.lower()]
            if len(names) != 1:
                raise ValueError(f"expected one CSV in the zip, found {names}")
            data = archive.read(names[0])  # checks the member's CRC-32
    except (zipfile.BadZipFile, EOFError, zlib.error) as error:
        raise ValueError(f"corrupt zip: {error}") from error
    return data.decode("utf-8-sig")


def unzip_csv(blob: bytes, member: str | None = None) -> str:
    """The CSV inside an NSE zip, its CRC checked. A zip may bundle several days'
    files (combineoi of 2025-07-02 holds 30-06, 01-07 and 02-07): then the one named
    ``member`` is taken. NSE served combineoi_30062025.zip with every LF byte turned
    into CRLF (as text); undoing that gives NSE's intact file (the same bytes it
    bundled in the 01-07 zip), accepted only if the member's CRC-32 then matches."""
    try:
        return _unzip_csv(blob, member)
    except ValueError as error:
        if "corrupt zip" not in str(error) or b"\r\n" not in blob:
            raise
        try:
            return _unzip_csv(blob.replace(b"\r\n", b"\n"), member)
        except ValueError:
            raise error from None


def xlsx_csv(blob: bytes) -> str:
    """An XLSX sheet as CSV text, cell texts as stored (NSE served the cash
    bhavcopy of 2022-08-08 this way; its cells are text, numbers as written)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        shared = [
            "".join(t.text or "" for t in item.iter(f"{{{ns['m']}}}t"))
            for item in ElementTree.fromstring(
                archive.read("xl/sharedStrings.xml")
            ).findall("m:si", ns)
        ]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for row in sheet.find("m:sheetData", ns).findall("m:row", ns):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", ns):
            letters = re.match(r"[A-Z]+", cell.get("r")).group(0)
            index = 0
            for char in letters:
                index = index * 26 + ord(char) - 64
            value, inline = cell.find("m:v", ns), cell.find("m:is", ns)
            if cell.get("t") == "s" and value is not None:
                text = shared[int(value.text)]
            elif inline is not None:
                text = "".join(t.text or "" for t in inline.iter(f"{{{ns['m']}}}t"))
            else:
                text = value.text if value is not None else ""
            cells[index - 1] = text.strip()
        if cells:
            writer.writerow([cells.get(i, "") for i in range(max(cells) + 1)])
    return out.getvalue()


def _rows(text: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]


def _same_day(found: date, day: date) -> None:
    if found != day:
        raise WrongDay(f"the file is {found}'s, not {day}'s")


def _one_session(rows: list[list[str]], at: int, day: date) -> None:
    """Every row carries the first row's date, and that is ``day``."""
    first = rows[1][at].strip()
    _same_day(nse_date(first), day)
    if any(row[at].strip() != first for row in rows[1:]):
        raise ValueError(f"the file mixes sessions with {day}")


def check_udiff_fo(text: str, day: date) -> int:
    rows = _rows(text)
    header = [c.strip() for c in rows[0]]
    lacking = [c for c in UDIFF_FO_COLUMNS if c not in header]
    if lacking:
        raise ValueError(f"not a UDiFF F&O bhavcopy (no {', '.join(lacking)})")
    _one_session(rows, header.index("TradDt"), day)
    if len(rows) < 1001:
        raise ValueError(f"only {len(rows) - 1} rows")
    return len(rows) - 1


def check_old_fo(text: str, day: date) -> int:
    rows = _rows(text)
    header = [c.strip() for c in rows[0]]
    if tuple(header[: len(OLD_FO_COLUMNS)]) != OLD_FO_COLUMNS:
        raise ValueError("not an old-format F&O bhavcopy")
    _one_session(rows, header.index("TIMESTAMP"), day)
    if len(rows) < 1001:
        raise ValueError(f"only {len(rows) - 1} rows")
    return len(rows) - 1


def check_cm(text: str, day: date) -> int:
    rows = _rows(text)
    header = [c.strip() for c in rows[0]]
    if "DATE1" not in header:
        raise ValueError("not an NSE sec_bhavdata_full file")
    _same_day(nse_date(rows[1][header.index("DATE1")]), day)
    kept = read_bhavcopy(text, day)
    if len(kept) < 500:
        raise ValueError(f"only {len(kept)} EQ/BE rows")
    return len(kept)


def participant_title_date(text: str, year_hint: int | None = None) -> date:
    """The date in the title line, which NSE writes many ways (all quotes dropped
    first): "as on Jan 03 2020", "as on Jan 28, 2020.", "as on Jul 04,2024",
    "as on Oct 15, 2026", "as on Sep 15,2021", ",2021Participant wise ... as on
    Jun 10", and with no year at all "as on Jun 08" (then ``year_hint``)."""
    title = text.splitlines()[0].replace('"', "")
    found = re.search(r"as on\s+([A-Za-z]{3})\s+(\d{1,2})\s*,?\s*(\d{4})?", title)
    if not found:
        raise ValueError(f"no date in the participant title {title!r}")
    year = found.group(3)
    if year is None:
        years = re.findall(r"(?<!\d)(20\d\d)(?!\d)", title)
        if len(years) == 1:
            year = years[0]
        elif not years and year_hint is not None:
            year = str(year_hint)
        else:
            raise ValueError(f"no year in the participant title {title!r}")
    return nse_date(f"{found.group(1)} {found.group(2)} {year}")


def check_participant(text: str, day: date) -> int:
    _same_day(participant_title_date(text, day.year), day)
    rows = _rows("\n".join(text.splitlines()[1:]))
    if rows[0][0].strip() != "Client Type":
        raise ValueError("no 'Client Type' header")
    kinds = [row[0].strip() for row in rows[1 : 1 + len(PARTICIPANTS)]]
    if kinds != list(PARTICIPANTS):
        raise ValueError(f"unexpected participant rows {kinds}")
    return len(kinds)


def check_ban(text: str, day: date) -> int:
    rows = _rows(text)
    found = re.search(r"Trade Date\s+(\d{1,2}-[A-Za-z]{3}-\d{4})", rows[0][0])
    if not found:
        raise ValueError(f"no trade date in the ban list title {rows[0]!r}")
    _same_day(nse_date(found.group(1)), day)
    if "NIL" in rows[0][0].upper().split(":")[-1] and len(rows) > 1:
        raise ValueError("a NIL ban list with rows")
    for row in rows[1:]:
        if len(row) < 2 or not row[0].strip().isdigit() or not row[1].strip():
            raise ValueError(f"unexpected ban row {row!r}")
    return len(rows) - 1


def check_mwpl(text: str, day: date) -> int:
    rows = _rows(text)
    header = [c.strip() for c in rows[0]]
    for column in ("Date", "NSE Symbol", "MWPL", "Open Interest"):
        if column not in header:
            raise ValueError(f"not a combined-OI file (no {column})")
    _one_session(rows, 0, day)
    return len(rows) - 1


def check_indices(text: str, day: date) -> int:
    """The all-index file. Its dates are DD-MM-YYYY, except that NSE wrote them
    month first (MM-DD-YYYY) on 2023-04-06, 04-10 and 04-11; either order is
    accepted when it gives the session (a previous session's file, served for a
    holiday, cannot match in either order)."""
    rows = _rows(text)
    header = [c.strip() for c in rows[0]]
    for column in ("Index Name", "Index Date", "Closing Index Value", "P/E"):
        if column not in header:
            raise ValueError(f"not an all-index close file (no {column})")
    at = header.index("Index Date")
    first = rows[1][at].strip()
    if any(row[at].strip() != first for row in rows[1:]):
        raise ValueError(f"the file mixes sessions with {day}")
    try:
        found: date | None = nse_date(first)
    except ValueError:  # e.g. 04-13-2023, month first with a day above 12
        found = None
    numeric = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", first)
    if found != day and numeric:
        month, dom, year = (int(part) for part in numeric.groups())
        if (year, month, dom) == (day.year, day.month, day.day):
            found = day
    if found is None:
        raise ValueError(f"not an NSE date: {first!r}")
    _same_day(found, day)
    names = [row[0].strip() for row in rows[1:]]
    if "Nifty 50" not in names or len(set(names)) != len(names):
        raise ValueError("no Nifty 50 row, or an index listed twice")
    if len(names) < 50:
        raise ValueError(f"only {len(names)} indices")
    return len(names)


@dataclass(frozen=True)
class Dataset:
    name: str
    folder: str  # under the archive root
    zipped: bool
    first: date | None  # no file is requested before this session
    last: date | None  # nor after this one
    filename: Callable[[date], str]
    url: Callable[[date], str]
    check: Callable[[str, date], int]
    flat: bool = False  # files directly in the folder, not in a year folder

    def path(self, root: Path, day: date) -> Path:
        if self.flat:
            return root / self.folder / self.filename(day)
        return root / self.folder / f"{day.year}" / self.filename(day)

    def wanted(self, day: date) -> bool:
        return (self.first is None or day >= self.first) and (
            self.last is None or day <= self.last
        )

    def member(self, day: date) -> str:
        """The CSV a zipped file holds for ``day``: its name without .zip."""
        name = self.filename(day)[: -len(".zip")]
        return name if name.endswith(".csv") else f"{name}.csv"

    def text(self, blob: bytes, day: date | None = None) -> str:
        if not self.zipped:
            return blob.decode("utf-8-sig")
        return unzip_csv(blob, None if day is None else self.member(day))


def _udiff_name(day: date) -> str:
    return f"BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


def _old_name(day: date) -> str:
    return f"fo{day:%d}{MONTHS[day.month - 1]}{day:%Y}bhav.csv.zip"


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (
        Dataset(
            "fo",
            "fo",
            True,
            UDIFF_FROM,
            None,
            _udiff_name,
            lambda d: f"{ARCHIVE}/content/fo/{_udiff_name(d)}",
            check_udiff_fo,
        ),
        Dataset(
            "fo_old",
            "fo_old",
            True,
            None,
            OLD_FO_UNTIL,
            _old_name,
            lambda d: (
                f"{ARCHIVE}/content/historical/DERIVATIVES/{d:%Y}/"
                f"{MONTHS[d.month - 1]}/{_old_name(d)}"
            ),
            check_old_fo,
        ),
        Dataset(
            "cm",
            "daily",
            False,
            None,
            None,
            lambda d: f"{d.isoformat()}.csv",
            lambda d: f"{ARCHIVE}/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv",
            check_cm,
            flat=True,
        ),
        Dataset(
            "participant_oi",
            "participant_oi",
            False,
            None,
            None,
            lambda d: f"fao_participant_oi_{d:%d%m%Y}.csv",
            lambda d: f"{ARCHIVE}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv",
            check_participant,
        ),
        Dataset(
            "participant_vol",
            "participant_vol",
            False,
            None,
            None,
            lambda d: f"fao_participant_vol_{d:%d%m%Y}.csv",
            lambda d: f"{ARCHIVE}/content/nsccl/fao_participant_vol_{d:%d%m%Y}.csv",
            check_participant,
        ),
        Dataset(
            "ban",
            "ban",
            False,
            None,
            None,
            lambda d: f"fo_secban_{d:%d%m%Y}.csv",
            lambda d: f"{ARCHIVE}/archives/fo/sec_ban/fo_secban_{d:%d%m%Y}.csv",
            check_ban,
        ),
        Dataset(
            "mwpl",
            "mwpl",
            True,
            None,
            None,
            lambda d: f"combineoi_{d:%d%m%Y}.zip",
            lambda d: f"{ARCHIVE}/archives/nsccl/mwpl/combineoi_{d:%d%m%Y}.zip",
            check_mwpl,
        ),
        Dataset(
            "indices",
            "indices",
            False,
            None,
            None,
            lambda d: f"ind_close_all_{d:%d%m%Y}.csv",
            lambda d: f"{ARCHIVE}/content/indices/ind_close_all_{d:%d%m%Y}.csv",
            check_indices,
        ),
    )
}


class Pacer:
    """At most one request start per ``spacing`` seconds, across threads."""

    def __init__(self, spacing: float) -> None:
        self._spacing = spacing
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._spacing
        if start > now:
            time.sleep(start - now)


def http_get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": HEADERS["User-Agent"], "Accept": "*/*"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise NotPublished(url) from None
        raise


@dataclass(frozen=True)
class Outcome:
    dataset: str
    day: date
    status: str  # "kept", "downloaded", "not_published", "failed"
    detail: str = ""


def record_fiidii(root: Path, nse: object) -> str | None:
    """Keep NSE's FII/DII cash-market flows (Rs crore) of the latest session in
    ``root/fiidii/<date>.json``. NSE serves only the latest day, so history exists
    only from when this started; returns the session date kept, if new."""
    rows = nse.get_json("/api/fiidiiTradeReact", {})
    if not isinstance(rows, list) or not rows:
        raise ValueError("no FII/DII rows")
    days = {nse_date(str(row["date"])) for row in rows}
    categories = sorted(str(row["category"]) for row in rows)
    if len(days) != 1 or categories != ["DII", "FII/FPI"]:
        raise ValueError(f"unexpected FII/DII rows {rows!r}")
    for row in rows:
        for field in ("buyValue", "sellValue", "netValue"):
            float(row[field])
    (day,) = days
    path = root / "fiidii" / f"{day.isoformat()}.json"
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, sort_keys=True), "utf-8")
    return day.isoformat()


class NseArchive:
    """Downloads, checks and keeps the datasets' files under ``root``."""

    def __init__(
        self,
        root: Path,
        *,
        get: Callable[[str], bytes] = http_get,
        spacing: float = 0.2,
        workers: int = 4,
        attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = root
        self._get = get
        self._pacer = Pacer(spacing)
        self._workers = workers
        self._attempts = attempts
        self._sleep = sleep

    def path(self, dataset: str, day: date) -> Path:
        return DATASETS[dataset].path(self.root, day)

    def _download(self, url: str) -> tuple[bytes, str]:
        """The file from NSE's archive host, else (404) from its older host, which
        still serves a few files the new one lacks (the ban list of 2025-05-06)."""
        urls = [url]
        if url.startswith(ARCHIVE):
            urls.append(OLD_ARCHIVE + url[len(ARCHIVE) :])
        for position, candidate in enumerate(urls):
            self._pacer.wait()
            try:
                return self._get(candidate), candidate
            except NotPublished:
                if position == len(urls) - 1:
                    raise
        raise AssertionError("unreachable")

    def fetch(self, dataset: str, day: date) -> Outcome:
        spec = DATASETS[dataset]
        path = spec.path(self.root, day)
        if path.exists():
            return Outcome(dataset, day, "kept")
        failure = ""
        for attempt in range(self._attempts):
            try:
                blob, source = self._download(spec.url(day))
                original = None
                if dataset == "cm" and blob[:2] == b"PK":
                    original, blob = blob, xlsx_csv(blob).encode("utf-8")
                spec.check(spec.text(blob, day), day)
            except NotPublished:
                return Outcome(dataset, day, "not_published")
            except WrongDay as error:
                return Outcome(dataset, day, "not_published", str(error))
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"[:200]
                self._sleep(2.0 + 4.0 * attempt)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if original is not None:
                path.with_suffix(".xlsx").write_bytes(original)
            part = path.with_name(path.name + ".part")
            part.write_bytes(blob)
            part.replace(path)
            notes = ["converted from XLSX"] if original else []
            if source != spec.url(day):
                notes.append(f"from {source}")
            return Outcome(dataset, day, "downloaded", "; ".join(notes))
        return Outcome(dataset, day, "failed", failure)

    def sync(
        self,
        datasets: Sequence[str],
        days: Iterable[date],
        progress: Callable[[Outcome], None] | None = None,
    ) -> list[Outcome]:
        jobs = [
            (name, day)
            for day in sorted(set(days))
            for name in datasets
            if DATASETS[name].wanted(day)
        ]
        outcomes: list[Outcome] = []
        with ThreadPoolExecutor(self._workers) as pool:
            for outcome in pool.map(lambda job: self.fetch(*job), jobs):
                outcomes.append(outcome)
                if progress is not None:
                    progress(outcome)
        return outcomes

    def verify(self, dataset: str, days: Iterable[date]) -> dict[str, dict]:
        """Every kept file re-read and re-checked: {iso date: rows, bytes, sha256}."""
        spec = DATASETS[dataset]
        found = {}
        for day in sorted(set(days)):
            path = spec.path(self.root, day)
            if not path.exists():
                continue
            blob = path.read_bytes()
            rows = spec.check(spec.text(blob, day), day)
            found[day.isoformat()] = {
                "file": path.relative_to(self.root).as_posix(),
                "rows": rows,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        return found
