from datetime import date, timedelta

import numpy as np
import pytest

from ai_trader.paper.basis import (
    BHAVCOPY_URL,
    SYMBOL_CHANGES_URL,
    PriceBasis,
    action_kind,
    nse_symbol,
    read_bhavcopy,
    rebase,
    snap,
    symbol_history,
)

DAYS = [date(2026, 9, 1) + timedelta(days=i) for i in range(6)]
TODAY = date(2026, 9, 7)


def market(*closes: float, volume: float = 1000.0) -> dict[date, tuple]:
    """NSE rows: open = previous close (first day: its own close)."""
    rows, previous = {}, closes[0]
    for day, close in zip(DAYS, closes, strict=False):
        rows[day] = (
            previous,
            max(previous, close),
            min(previous, close),
            close,
            volume,
        )
        previous = close
    return rows


def kite_of(nse: dict, factors: dict[date, float], volumes: dict | None = None):
    """Kite's rows: NSE's times the product of later factors (volume divided)."""
    out = []
    for day in DAYS:
        scale = np.prod([f for ex, f in factors.items() if day < ex] or [1.0])
        o, h, lo, c, v = nse[day]
        out.append(
            (
                o * scale,
                h * scale,
                lo * scale,
                c * scale,
                (volumes or {}).get(day, v / scale),
            )
        )
    return np.array(out)


def test_kinds_and_split_bonus_ratios() -> None:
    assert action_kind("Face Value Split (Sub-Division) - From Rs 10/-") == "split"
    assert action_kind("Bonus 1:1") == "bonus"
    assert action_kind("Demerger") == "demerger"
    assert action_kind("Rights 2:25 @ Premium Rs 17/-") == "rights"
    assert action_kind("Interim Dividend - Rs 5 Per Share") == "dividend"
    assert action_kind("Annual General Meeting") == "other"
    assert snap(0.50004) == 0.5 and snap(0.2) == 0.2 and snap(1 / 3) == 1 / 3
    assert snap(0.97) is None and snap(0.6054) is None


def test_unadjusted_history_is_nse_as_traded() -> None:
    nse = market(100, 101, 102, 101, 103, 104)
    result = rebase(DAYS, kite_of(nse, {}), nse, [], TODAY)
    assert result.reason is None and not result.events
    assert result.daily.tolist() == [list(nse[d]) for d in DAYS]
    assert not result.volume_scale


def test_split_takes_kites_snapped_ratio_and_scales_volume() -> None:
    nse = market(1000, 1010, 1020, 205, 206, 207)  # 10 -> 2 split from DAYS[3]
    kite = kite_of(nse, {DAYS[3]: 0.2})
    result = rebase(DAYS, kite, nse, [(DAYS[3], "Split From Rs 10 To Rs 2")], TODAY)
    (event,) = result.events
    assert event["ratio"] == 0.2 and event["source"].startswith("split/bonus")
    assert result.daily[2, 3] == pytest.approx(204) and result.daily[2, 4] == 5000
    assert result.daily[4].tolist() == list(nse[DAYS[4]])
    assert not result.volume_scale  # Kite scales the volume alike


def test_dividends_are_not_adjusted_even_when_kite_did() -> None:
    nse = market(100, 101, 102, 98, 99, 100)  # a 3% dividend from DAYS[3]
    kite = kite_of(nse, {DAYS[3]: 0.97}, {d: nse[d][4] for d in DAYS})
    result = rebase(DAYS, kite, nse, [(DAYS[3], "Dividend - Rs 3 Per Share")], TODAY)
    assert result.reason is None and not result.events
    assert result.daily.tolist() == [list(nse[d]) for d in DAYS]


@pytest.mark.parametrize("kite_factor", [0.5234, 1.0])
def test_demerger_takes_the_ex_date_opening_move(kite_factor: float) -> None:
    nse = market(770, 772, 774, 290, 291, 292)
    nse[DAYS[3]] = (289.5, 292, 288, 290, 1000)  # the price the market discovered
    kite = kite_of(nse, {DAYS[3]: kite_factor})  # Kite's wrong ratio, or none
    result = rebase(DAYS, kite, nse, [(DAYS[3], "Demerger")], TODAY)
    (event,) = result.events
    assert event["ratio"] == pytest.approx(289.5 / 774)
    assert event["source"].startswith("NSE ex-date open")
    assert result.daily[2, 3] == pytest.approx(289.5)  # no false overnight drop
    assert result.volume_scale[DAYS[2]] == pytest.approx(kite_factor / event["ratio"])


def test_a_bonus_that_is_not_an_equity_bonus_uses_the_market() -> None:
    nse = market(13748, 13700, 13748, 13743, 13800, 13810)
    nse[DAYS[3]] = (13743, 13760, 13700, 13743, 1000)
    result = rebase(
        DAYS,
        kite_of(nse, {}),  # Kite left it alone: no equity changed hands
        nse,
        [(DAYS[3], "Scheme Of Arrangement - Bonus Ncrps 46:1")],
        TODAY,
    )
    (event,) = result.events
    assert event["ratio"] == pytest.approx(13743 / 13748)  # not 1/47


def test_structural_action_today() -> None:
    nse = market(1000, 1010, 1020, 1030, 1040, 1050)
    split = kite_of(nse, {TODAY: 0.5})  # Kite adjusted today's 1:1 bonus already
    result = rebase(DAYS, split, nse, [(TODAY, "Bonus 1:1")], TODAY)
    assert result.events[0]["ratio"] == 0.5
    assert result.daily[-1, 3] == pytest.approx(525)
    rights = kite_of(nse, {TODAY: 0.982})
    result = rebase(
        DAYS, rights, nse, [(TODAY, "Rights 2:25 @ Premium Rs 17/-")], TODAY
    )
    assert result.daily is None and "after the open" in result.reason


def test_a_missing_nse_session_takes_kites_row_only_when_as_traded() -> None:
    nse = market(100, 101, 102, 101, 103, 104)
    kite = kite_of(nse, {})
    partial = {d: r for d, r in nse.items() if d != DAYS[-1]}
    result = rebase(DAYS, kite, partial, [], TODAY)
    assert result.daily[-1].tolist() == kite[-1].tolist()
    dividend = [(TODAY, "Dividend - Rs 3 Per Share")]
    result = rebase(DAYS, kite_of(nse, {TODAY: 0.97}), partial, dividend, TODAY)
    assert result.daily is None and "no NSE row" in result.reason


def test_disagreement_and_unexplained_adjustments_leave_a_stock_out() -> None:
    nse = market(100, 101, 102, 101, 103, 104)
    kite = kite_of(nse, {})
    kite[4, 3] += 0.05
    assert "differ" in rebase(DAYS, kite, nse, [], TODAY).reason
    unexplained = kite_of(nse, {DAYS[3]: 0.9})
    small = [(DAYS[5], "Dividend - Rs 0.50 Per Share")]  # Kite left it alone
    reason = rebase(DAYS, unexplained, nse, small, TODAY).reason
    assert reason == f"Kite adjusted its history at {DAYS[3]} for no NSE action"


def test_bhavcopy_reading() -> None:
    text = (
        "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
        "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY\n"
        "ABC, BE, 01-Sep-2026, 9, 10, 11, 9, 10, 10.5, 10, 50\n"
        "ABC, EQ, 01-Sep-2026, 9, 10, 12, 9, 11, 11.5, 11, 100\n"
        "XYZ, SM, 01-Sep-2026, 9, 10, 12, 9, 11, 11.5, 11, 100\n"
        "BEO, BE, 01-Sep-2026, 9, 1, 2, 1, 2, 2, 2, 7\n"
    )
    rows = read_bhavcopy(text, DAYS[0])
    assert rows == {"ABC": (10, 12, 9, 11.5, 100), "BEO": (1, 2, 1, 2, 7)}
    assert read_bhavcopy(text, DAYS[0], {"BEO"}) == {"BEO": (1, 2, 1, 2, 7)}
    with pytest.raises(ValueError, match="expected"):
        read_bhavcopy(text, DAYS[1])
    with pytest.raises(ValueError, match="not an NSE bhavcopy"):
        read_bhavcopy("<html>blocked</html>", DAYS[0])


def test_symbol_changes_are_followed_back() -> None:
    text = (
        "SM_NAME,OLD,NEW,DATE\n"
        "Tata Motors,TATAMOTORS,TMPV,24-OCT-2025\n"
        "Gujarat,GUJGASLTD,GUJENERGY,01-JUL-2026\n"
    )
    history = symbol_history(text)
    assert nse_symbol("TMPV", date(2025, 10, 23), history["TMPV"]) == "TATAMOTORS"
    assert nse_symbol("TMPV", date(2025, 10, 24), history["TMPV"]) == "TMPV"
    assert nse_symbol("ABC", date(2025, 1, 1), []) == "ABC"


class Action:
    def __init__(self, symbol: str, ex_date: date, subject: str) -> None:
        self.symbol, self.ex_date, self.subject = symbol, ex_date, subject


def bhavcopy(day: date, rows: dict[str, tuple]) -> bytes:
    lines = [
        "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY"
    ]
    for symbol, (o, h, lo, c, v) in rows.items():
        lines.append(f"{symbol},EQ,{day:%d-%b-%Y},{o},{h},{lo},{c},{v}")
    return ("\n".join(lines) + "\n").encode()


def test_price_basis_downloads_follows_renames_and_falls_back_to_stored_actions(
    tmp_path,
) -> None:
    nse = market(1000, 1010, 1020, 205, 206, 207)
    changes = "NAME,OLD,NEW,DATE\n" + "\n".join(
        [f"X{i},O{i},N{i},01-JAN-2020" for i in range(100)]
        + ["Abc,ABCOLD,ABC,03-SEP-2026"]
    )
    served = {SYMBOL_CHANGES_URL: changes.encode()}
    for day in DAYS:
        name = "ABCOLD" if day < date(2026, 9, 3) else "ABC"
        served[BHAVCOPY_URL.format(day=day)] = bhavcopy(day, {name: nse[day]})
    asked = []

    def download(url: str) -> bytes:
        asked.append(url)
        return served[url]

    class Nse:
        def corporate_actions(self, start, end):
            return [Action("ABC", DAYS[3], "Face Value Split From Rs 10 To Rs 2")]

    class Store:
        def __init__(self) -> None:
            self.kept = []

        def add_actions(self, items) -> None:
            self.kept.extend(items)

        def actions(self, symbol, start, end):
            return [
                {"ex_date": a.ex_date.isoformat(), "subject": a.subject}
                for a in self.kept
                if a.symbol == symbol
            ]

    store = Store()
    basis = PriceBasis(tmp_path, Nse(), store, download=download, sleep=lambda s: None)
    notes = basis.load(DAYS, ["ABC"], TODAY)
    assert "6 of 6 sessions (6 downloaded)" in notes[0]
    kite = kite_of(nse, {DAYS[3]: 0.2})
    result = basis.rebase("ABC", DAYS, kite, TODAY)
    assert result.events[0]["ratio"] == 0.2
    assert result.daily[0, 3] == pytest.approx(200)
    # A second load reuses the files kept, and stored actions stand in for NSE's.
    asked.clear()

    class DownNse:
        def corporate_actions(self, start, end):
            raise TimeoutError("NSE down")

    again = PriceBasis(
        tmp_path, DownNse(), store, download=download, sleep=lambda s: None
    )
    notes = again.load(DAYS, ["ABC"], TODAY)
    assert asked == [SYMBOL_CHANGES_URL] and "stored" in notes[-1]
    assert again.rebase("ABC", DAYS, kite, TODAY).events[0]["ratio"] == 0.2
