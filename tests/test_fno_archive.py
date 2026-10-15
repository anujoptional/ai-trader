import hashlib
import io
import json
import zipfile
from datetime import date

import pytest

from ai_trader.cli.fetch_nse_archives import candidate_days, update_manifest
from ai_trader.fno import nse_archive as na

UDIFF = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,"
    "TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,"
    "Rsvd4\n"
)
OLD = (
    "INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,"
    "CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,\n"
)


def zipped(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("03-JAN-2020", date(2020, 1, 3)),
        ("30-Jan-2020", date(2020, 1, 30)),
        ("26-Feb-20", date(2020, 2, 26)),
        ("2026-10-15", date(2026, 10, 15)),
        ("Jan 03 2020", date(2020, 1, 3)),
        ("Oct 15, 2026", date(2026, 10, 15)),
        ("Jul 04,2024", date(2024, 7, 4)),
        ("Jan 28, 2020.", date(2020, 1, 28)),
        ("03-01-2020", date(2020, 1, 3)),
    ],
)
def test_nse_writes_dates_many_ways(text, expected) -> None:
    assert na.nse_date(text) == expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "Participant wise Open Interest (no. of contracts) in Equity Derivatives "
            "as on Jan 03 2020,,,",
            date(2020, 1, 3),
        ),
        (
            '""Participant wise Open Interest as on Oct 15, 2026"",,,',
            date(2026, 10, 15),
        ),
        (
            'Participant wise Trading Volume as on Sep 15,"2021""""",,',
            date(2021, 9, 15),
        ),
        ('",2021Participant wise Trading Volume as on Jun 10",,,', date(2021, 6, 10)),
        ("Participant wise Open Interest as on Jan 28, 2020.,,", date(2020, 1, 28)),
    ],
)
def test_participant_titles(title, expected) -> None:
    assert na.participant_title_date(title + "\nClient Type,x\n") == expected


def test_a_participant_title_without_a_year_takes_the_sessions() -> None:
    text = "Participant wise Open Interest as on Jun 08,,,,\nClient Type,x\n"
    assert na.participant_title_date(text, 2021) == date(2021, 6, 8)
    with pytest.raises(ValueError):
        na.participant_title_date(text)


def participant_file(title: str, extra: str = "") -> str:
    header = "Client Type," + ",".join(
        name
        for name in [
            "Future Index Long",
            "Future Index Short",
            "Future Stock Long",
            "Future Stock Short\t",
            "Option Index Call Long",
            "Option Index Put Long",
            "Option Index Call Short",
            "Option Index Put Short",
            "Option Stock Call Long",
            "Option Stock Put Long",
            "Option Stock Call Short",
            "Option Stock Put Short",
            "Total Long Contracts   ",
            "Total Short Contracts",
        ]
    )
    rows = [f"{who}," + ",".join(str(i) for i in range(14)) for who in na.PARTICIPANTS]
    return "\n".join([title, header, *rows]) + extra


def test_checks_accept_their_session_and_reject_other_days() -> None:
    day = date(2026, 10, 15)
    udiff = UDIFF + "".join(
        f"2026-10-15,2026-10-15,FO,NSE,STO,{i},,ABC,,2026-10-27,2026-10-27,"
        f"{100 + i}.00,CE,X,1,1,1,1,1,1,100,1,10,0,1,100,1,F1,50,,,,,\n"
        for i in range(1001)
    )
    assert na.check_udiff_fo(udiff, day) == 1001
    with pytest.raises(na.WrongDay):
        na.check_udiff_fo(udiff, date(2026, 10, 14))
    old = OLD + "".join(
        f"OPTSTK,ABC,27-Oct-2026,{100 + i},CE,1,1,1,1,1,1,0.5,10,0,15-OCT-2026,\n"
        for i in range(1001)
    )
    assert na.check_old_fo(old, day) == 1001
    with pytest.raises(na.WrongDay):
        na.check_old_fo(old, date(2026, 10, 16))
    title = 'Participant wise Trading Volume as on Sep 15,"2021""""",,'
    junk = "\n,,,805930,,61382827\n,,,206,,1\n"
    assert na.check_participant(participant_file(title, junk), date(2021, 9, 15)) == 5
    with pytest.raises(na.WrongDay):
        na.check_participant(participant_file(title), date(2021, 9, 16))
    nil = "Securities in Ban For Trade Date 01-JAN-2020: NIL\n"
    assert na.check_ban(nil, date(2020, 1, 1)) == 0
    banned = "Securities in Ban For Trade Date 15-OCT-2026:\n1,AMBUJACEM\n2,SAIL\n"
    assert na.check_ban(banned, day) == 2
    with pytest.raises(ValueError):
        na.check_ban("Securities in Ban For Trade Date 15-OCT-2026:\nAMBUJACEM\n", day)
    mwpl = (
        "Date, ISIN, Scrip Name, NSE Symbol, MWPL, Open Interest, Limit for Next Day\n"
        "26-Feb-20,INE1,YES BANK LIMITED,YESBANK,100,93,No Fresh Positions\n"
    )
    assert na.check_mwpl(mwpl, date(2020, 2, 26)) == 1
    with pytest.raises(na.WrongDay):
        na.check_mwpl(mwpl, date(2020, 2, 27))


INDICES_HEADER = (
    "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,"
    "Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,"
    "Div Yield\n"
)


def indices_file(day: str, count: int = 60) -> str:
    return (
        INDICES_HEADER
        + f"Nifty 50,{day},12261.1,12265.6,12191.35,12226.65,-55.55,-.45,428770054,"
        "16827.27,28.44,3.77,1.23\n"
        + "".join(f"Index {i},{day},1,1,1,1,0,0,-,-,-,-,-\n" for i in range(count))
    )


def test_index_file_check() -> None:
    day = date(2020, 1, 3)
    assert na.check_indices(indices_file("03-01-2020"), day) == 61
    with pytest.raises(na.WrongDay):
        na.check_indices(indices_file("02-01-2020"), day)
    with pytest.raises(ValueError, match="only 11 indices"):
        na.check_indices(indices_file("03-01-2020", 10), day)
    twice = indices_file("03-01-2020") + "Index 1,03-01-2020,1,1,1,1,0,0,-,-,-,-,-\n"
    with pytest.raises(ValueError, match="listed twice"):
        na.check_indices(twice, day)
    with pytest.raises(ValueError, match="mixes sessions"):
        na.check_indices(
            indices_file("03-01-2020") + "Late,06-01-2020,1,1,1,1,0,0,-,-,-,-,-\n", day
        )
    spec = na.DATASETS["indices"]
    assert spec.url(day).endswith("/content/indices/ind_close_all_03012020.csv")


def test_index_files_written_month_first_are_their_sessions() -> None:
    # NSE wrote 04-06-2023 in the file of 6 April 2023 (also 04-10, 04-11).
    assert na.check_indices(indices_file("04-06-2023"), date(2023, 4, 6)) == 61
    assert na.check_indices(indices_file("04-06-2023"), date(2023, 6, 4)) == 61
    assert na.check_indices(indices_file("04-13-2023"), date(2023, 4, 13)) == 61
    with pytest.raises(na.WrongDay):  # the previous session's file
        na.check_indices(indices_file("05-04-2023"), date(2023, 4, 6))
    with pytest.raises(ValueError, match="not an NSE date"):
        na.check_indices(indices_file("04-13-2023"), date(2023, 4, 14))


def test_cm_check_treats_a_previous_sessions_file_as_another_day() -> None:
    header = (
        "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
        "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, "
        "NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    )
    rows = "".join(
        f"S{i}, EQ, 20-Feb-2020, 10, 10, 11, 9, 10, 10.5, 10.2, 100, 0.01, 5, 40, "
        "40.00\n"
        for i in range(500)
    )
    assert na.check_cm(header + rows, date(2020, 2, 20)) == 500
    with pytest.raises(na.WrongDay):  # NSE serves this for the 21 Feb 2020 holiday
        na.check_cm(header + rows, date(2020, 2, 21))


class FakeNse:
    def __init__(self, files: dict[str, bytes], flaky: int = 0) -> None:
        self.files, self.flaky, self.calls = files, flaky, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if self.flaky:
            self.flaky -= 1
            raise OSError("connection reset")
        if url not in self.files:
            raise na.NotPublished(url)
        return self.files[url]


def test_archive_keeps_checked_files_once_and_reports_holidays(tmp_path) -> None:
    day, holiday = date(2020, 1, 3), date(2020, 2, 21)
    title = "Securities in Ban For Trade Date 03-JAN-2020:\n1,YESBANK\n"
    ban_url = na.DATASETS["ban"].url(day)
    fake = FakeNse({ban_url: title.encode()}, flaky=1)
    archive = na.NseArchive(tmp_path, get=fake, spacing=0, sleep=lambda s: None)
    outcomes = archive.sync(["ban"], [day, holiday])
    assert {(o.day, o.status) for o in outcomes} == {
        (day, "downloaded"),
        (holiday, "not_published"),
    }
    path = archive.path("ban", day)
    assert path.read_text() == title and not list(tmp_path.rglob("*.part"))
    calls = len(fake.calls)
    assert archive.fetch("ban", day).status == "kept" and len(fake.calls) == calls
    listed = archive.verify("ban", [day, holiday])
    assert listed == {
        "2020-01-03": {
            "file": "ban/2020/fo_secban_03012020.csv",
            "rows": 1,
            "bytes": len(title),
            "sha256": hashlib.sha256(title.encode()).hexdigest(),
        }
    }


def test_the_manifest_lists_new_files_and_is_left_alone_when_nothing_changed(
    tmp_path,
) -> None:
    first, second = date(2020, 1, 3), date(2020, 1, 6)
    files = {
        na.DATASETS["ban"].url(d): f"Securities in Ban For Trade Date {d:%d-%b-%Y}:\n"
        "1,YESBANK\n".encode()
        for d in (first, second)
    }
    archive = na.NseArchive(tmp_path, get=FakeNse(files), spacing=0)
    archive.sync(["ban"], [first])
    manifest = update_manifest(archive, "ban", [first, second])
    assert list(manifest["files"]) == ["2020-01-03"]
    path = tmp_path / "manifest" / "ban.json"
    written = path.read_bytes()
    assert update_manifest(archive, "ban", [first, second]) == manifest
    assert path.read_bytes() == written  # nothing new: not rewritten
    archive.sync(["ban"], [second])
    assert list(update_manifest(archive, "ban", [first, second])["files"]) == [
        "2020-01-03",
        "2020-01-06",
    ]
    assert json.loads(path.read_text())["files"]["2020-01-06"]["rows"] == 1


def test_a_file_missing_on_the_archive_host_comes_from_the_older_host(
    tmp_path,
) -> None:
    day = date(2025, 5, 6)
    url = na.DATASETS["ban"].url(day)
    old = na.OLD_ARCHIVE + url[len(na.ARCHIVE) :]
    title = "Securities in Ban For Trade Date 06-MAY-2025:\n1,MANAPPURAM\n2,RBLBANK\n"
    fake = FakeNse({old: title.encode()})
    archive = na.NseArchive(tmp_path, get=fake, spacing=0, sleep=lambda s: None)
    outcome = archive.fetch("ban", day)
    assert outcome.status == "downloaded" and outcome.detail == f"from {old}"
    assert fake.calls == [url, old]
    assert archive.path("ban", day).read_text() == title
    holiday = date(2025, 5, 1)
    outcome = archive.fetch("ban", holiday)
    assert outcome.status == "not_published" and len(fake.calls) == 4


def test_a_file_that_never_checks_out_is_not_kept(tmp_path) -> None:
    day = date(2020, 1, 3)
    fake = FakeNse({na.DATASETS["ban"].url(day): b"<html>blocked</html>"})
    archive = na.NseArchive(tmp_path, get=fake, spacing=0, sleep=lambda s: None)
    outcome = archive.fetch("ban", day)
    assert outcome.status == "failed" and len(fake.calls) == 3
    assert not archive.path("ban", day).exists()


def test_zipped_datasets_need_exactly_one_csv() -> None:
    text = "Date, ISIN, Scrip Name, NSE Symbol, MWPL, Open Interest\n"
    blob = zipped("combineoi_03012020.csv", text)
    assert na.DATASETS["mwpl"].text(blob) == text
    with pytest.raises(ValueError):
        na.unzip_csv(zipped("a.xml", "<x/>"))


def test_a_zip_bundling_several_days_gives_the_sessions_file() -> None:
    # NSE's combineoi_02072025.zip holds the files of 30-06, 01-07 and 02-07.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for stamp in ("01072025", "02072025", "30062025"):
            archive.writestr(f"combineoi_{stamp}.csv", f"file {stamp}\n")
    spec, day = na.DATASETS["mwpl"], date(2025, 7, 2)
    assert spec.member(day) == "combineoi_02072025.csv"
    assert spec.text(buffer.getvalue(), day) == "file 02072025\n"
    with pytest.raises(ValueError, match="expected one CSV"):
        spec.text(buffer.getvalue())
    with pytest.raises(ValueError, match="expected one CSV"):
        spec.text(buffer.getvalue(), date(2025, 7, 3))
    assert na.DATASETS["fo"].member(day) == "BhavCopy_NSE_FO_0_0_0_20250702_F_0000.csv"
    assert na.DATASETS["fo_old"].member(date(2020, 1, 3)) == "fo03JAN2020bhav.csv"


def test_a_zip_served_with_lf_turned_into_crlf_is_restored_exactly() -> None:
    # NSE served combineoi_30062025.zip this way; its CRC-32 proves the repair.
    text = "Date, ISIN, Scrip Name, NSE Symbol, MWPL, Open Interest\n30-JUN-2025,X\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("combineoi_30062025.csv", text)
    intact = buffer.getvalue()
    assert b"\r\n" not in intact and b"\n" in intact
    damaged = intact.replace(b"\n", b"\r\n")
    assert na.unzip_csv(damaged) == text
    with pytest.raises(ValueError, match="corrupt zip"):
        na.unzip_csv(intact[: len(intact) // 2] + b"\r\n" + intact[len(intact) // 2 :])
    with pytest.raises(ValueError, match="corrupt zip"):
        na.unzip_csv(b"PK\x03\x04 not a zip")


def test_format_switch_and_file_names() -> None:
    old, new = na.DATASETS["fo_old"], na.DATASETS["fo"]
    assert old.wanted(date(2024, 7, 5)) and not old.wanted(date(2024, 7, 8))
    assert new.wanted(date(2024, 7, 8)) and not new.wanted(date(2020, 1, 3))
    assert old.url(date(2020, 1, 3)).endswith(
        "/content/historical/DERIVATIVES/2020/JAN/fo03JAN2020bhav.csv.zip"
    )
    assert new.url(date(2026, 10, 15)).endswith(
        "/content/fo/BhavCopy_NSE_FO_0_0_0_20261015_F_0000.csv.zip"
    )
    assert na.DATASETS["cm"].path(na.Path("r"), date(2020, 1, 3)).as_posix() == (
        "r/daily/2020-01-03.csv"
    )


def test_candidate_days_are_weekdays_plus_calendar_weekend_sessions(tmp_path) -> None:
    calendar = tmp_path / "dataset.json"
    calendar.write_text('{"session_dates": ["2024-03-02", "2024-03-04"]}')
    days = candidate_days(date(2024, 3, 1), date(2024, 3, 4), calendar)
    assert days == [date(2024, 3, 1), date(2024, 3, 2), date(2024, 3, 4)]


def xlsx(rows: list[list[str]]) -> bytes:
    strings = sorted({cell for row in rows for cell in row})
    shared = "".join(f"<si><t>{s}</t></si>" for s in strings)
    sheet = "".join(
        f'<row r="{r + 1}">'
        + "".join(
            f'<c r="{chr(65 + c)}{r + 1}" t="s"><v>{strings.index(cell)}</v></c>'
            for c, cell in enumerate(row)
        )
        + "</row>"
        for r, row in enumerate(rows)
    )
    space = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{space}">{shared}</sst>')
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{space}"><sheetData>{sheet}</sheetData></worksheet>',
        )
    return buffer.getvalue()


def test_a_cash_bhavcopy_served_as_xlsx_is_kept_as_csv_with_the_original(
    tmp_path,
) -> None:
    day = date(2022, 8, 8)
    header = [
        "SYMBOL",
        " SERIES",
        " DATE1",
        " PREV_CLOSE",
        " OPEN_PRICE",
        " HIGH_PRICE",
        " LOW_PRICE",
        " LAST_PRICE",
        " CLOSE_PRICE",
        " AVG_PRICE",
        " TTL_TRD_QNTY",
        " TURNOVER_LACS",
        " NO_OF_TRADES",
        " DELIV_QTY",
        " DELIV_PER",
    ]
    rows = [header] + [
        [
            f"S{i}",
            " EQ",
            " 08-Aug-2022",
            "108.1",
            "109.45",
            "115.45",
            "108.7",
            "110.85",
            "111.15",
            "112.1",
            "1687369",
            "1891.62",
            "21649",
            "571768",
            "33.89",
        ]
        for i in range(500)
    ]
    blob = xlsx(rows)
    fake = FakeNse({na.DATASETS["cm"].url(day): blob})
    archive = na.NseArchive(tmp_path, get=fake, spacing=0, sleep=lambda s: None)
    outcome = archive.fetch("cm", day)
    assert outcome.status == "downloaded" and outcome.detail == "converted from XLSX"
    kept = archive.path("cm", day)
    assert kept.with_suffix(".xlsx").read_bytes() == blob
    text = kept.read_text()
    assert text.splitlines()[1].startswith("S0,EQ,08-Aug-2022,108.1,")
    assert na.check_cm(text, day) == 500


class FakeJson:
    def __init__(self, rows):
        self.rows = rows

    def get_json(self, path, params):
        assert path == "/api/fiidiiTradeReact"
        return self.rows


def test_fii_dii_flows_are_kept_once_per_session(tmp_path) -> None:
    rows = [
        {
            "buyValue": "19438.65",
            "category": "DII",
            "date": "15-Oct-2026",
            "netValue": "3319.39",
            "sellValue": "16119.26",
        },
        {
            "buyValue": "12735.59",
            "category": "FII/FPI",
            "date": "15-Oct-2026",
            "netValue": "-5285.37",
            "sellValue": "18020.96",
        },
    ]
    assert na.record_fiidii(tmp_path, FakeJson(rows)) == "2026-10-15"
    assert na.record_fiidii(tmp_path, FakeJson(rows)) is None
    assert (tmp_path / "fiidii" / "2026-10-15.json").exists()
    with pytest.raises(ValueError):
        na.record_fiidii(tmp_path, FakeJson(rows[:1]))
