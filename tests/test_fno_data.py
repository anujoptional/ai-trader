from datetime import date

import numpy as np
import pytest

from ai_trader.fno import bhavcopy, features
from ai_trader.fno.iv import RATE, black76, implied_vol

DAY = date(2024, 7, 5)
UDIFF_HEAD = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,"
    "TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,"
    "Rsvd4\n"
)
OLD_HEAD = (
    "INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,"
    "CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,\n"
)
# The same three contracts in both formats (as NSE published both on 5 Jul 2024).
UDIFF = UDIFF_HEAD + (
    "2024-07-05,2024-07-05,FO,NSE,STF,1,,ABC,,2024-07-25,2024-07-25,,,ABC24JULFUT,"
    "101.00,103.50,100.00,102.00,102.05,100.50,101.20,102.10,450000,15000,120,"
    "12345678.50,99,F1,1500,,,,,\n"
    "2024-07-05,2024-07-05,FO,NSE,STO,2,,ABC,,2024-07-25,2024-07-25,100.00,CE,"
    "ABC24JUL100CE,3.00,3.50,2.80,3.20,3.25,2.90,101.20,3.20,30000,-1500,40,"
    "190000.00,12,F1,1500,,,,,\n"
    "2024-07-05,2024-07-05,FO,NSE,IDO,3,,NIFTY,,2024-07-11,2024-07-11,24000.00,PE,"
    "NIFTY24JUL24000PE,0.00,0.00,0.00,55.00,55.00,60.00,24300.00,54.00,7500,0,0,"
    "0.00,0,F1,25,,,,,\n"
)
OLD = OLD_HEAD + (
    "FUTSTK,ABC,25-Jul-2024,0,XX,101,103.5,100,102,102.1,120,123.4567850,450000,"
    "15000,05-JUL-2024,\n"
    "OPTSTK,ABC,25-Jul-2024,100,CE,3,3.5,2.8,3.2,3.2,40,1.9,30000,-1500,05-JUL-2024,\n"
    "OPTIDX,NIFTY,11-Jul-2024,24000,PE,0,0,0,55,54,0,0,7500,0,05-JUL-2024,\n"
)


def test_both_formats_normalize_to_the_same_rows() -> None:
    new, old = bhavcopy.parse_udiff(UDIFF, DAY), bhavcopy.parse_old(OLD, DAY)
    shared = ("symbol", "kind", "underlying", "option", "expiry", "contracts", "oi")
    for name in shared + ("chg_oi",):
        assert list(new[name]) == list(old[name]), name
    for name in ("strike", "open", "high", "low", "close", "settle"):
        np.testing.assert_array_equal(new[name], old[name])
    np.testing.assert_allclose(new["value"], old["value"], atol=0.01)
    assert list(new["kind"]) == ["F", "O", "O"]
    assert list(new["underlying"]) == ["S", "S", "I"]
    assert np.isnan(new["strike"][0]) and new["strike"][2] == 24000
    assert new["lot"][0] == 1500 and old["lot"][0] == 0
    assert new["underlying_price"][1] == 101.2 and np.isnan(old["underlying_price"][1])
    with pytest.raises(ValueError):
        bhavcopy.parse_udiff(UDIFF, date(2024, 7, 4))


def test_normalized_tables_round_trip(tmp_path) -> None:
    table = bhavcopy.parse_udiff(UDIFF, DAY)
    path = tmp_path / "2024" / "2024-07-05.npz"
    bhavcopy.save_table(path, table, "fo")
    loaded, source = bhavcopy.load_table(path)
    assert source == "fo"
    for name in bhavcopy.COLUMNS:
        np.testing.assert_array_equal(loaded[name], table[name])


def test_black76_known_value_and_parity() -> None:
    # F = K = 100, T = 1, r = 0, s = 20%: call = put = 100 (2 N(0.1) - 1).
    call = black76(100, 100, 1.0, 0.2, True, rate=0.0)
    assert float(call) == pytest.approx(7.965567455405804, abs=1e-12)
    forward, strike, years = 250.0, np.array([200.0, 250.0, 300.0]), 30 / 365
    calls = black76(forward, strike, years, 0.35, True)
    puts = black76(forward, strike, years, 0.35, False)
    np.testing.assert_allclose(
        calls - puts, np.exp(-RATE * years) * (forward - strike), atol=1e-11
    )


def test_implied_vol_recovers_the_volatility_to_machine_precision() -> None:
    grid = np.array(
        np.meshgrid(
            [0.05, 0.2, 0.6, 1.5],  # volatility
            [0.6, 0.9, 1.0, 1.1, 1.6],  # strike / forward
            [2 / 365, 30 / 365, 1.0],  # years
            [True, False],
        )
    ).reshape(4, -1)
    vol, moneyness, years, call = grid[0], grid[1], grid[2], grid[3].astype(bool)
    price = black76(100.0, 100.0 * moneyness, years, vol, call)
    found = implied_vol(price, 100.0, 100.0 * moneyness, years, call)
    intrinsic = np.exp(-RATE * years) * np.where(
        call, np.maximum(100 - 100 * moneyness, 0), np.maximum(100 * moneyness - 100, 0)
    )
    time_value = price - intrinsic
    # Only a price with time value carries the volatility; none is not invented.
    informative = time_value > 1e-6
    assert informative.sum() > 80 and np.isfinite(found[informative]).all()
    np.testing.assert_allclose(found[informative], vol[informative], rtol=1e-8)
    assert np.isnan(found[time_value < 1e-12]).all()
    priced = np.isfinite(found)
    repriced = black76(100.0, 100.0 * moneyness, years, found, call)
    np.testing.assert_allclose(repriced[priced], price[priced], rtol=1e-12, atol=1e-12)


def test_implied_vol_is_nan_outside_the_no_arbitrage_bounds() -> None:
    discount = np.exp(-RATE * 0.1)
    prices = np.array([20 * discount - 1e-6, 100 * discount, 1.0, 5.0])
    found = implied_vol(
        prices,
        [120, 120, 100, 100],
        [100, 100, 100, 100],
        [0.1, 0.1, 0.0, -1.0],
        [True, True, True, True],
    )
    assert np.isnan(found).all()


def session_table(day: date, spot: float, vol_of) -> dict[str, np.ndarray]:
    """One stock: futures of two expiries, options priced at vol_of(strike/forward)."""
    rows: dict[str, list] = {name: [] for name in bhavcopy.COLUMNS}

    def add(kind, expiry, strike, option, close, settle, contracts, oi, chg):
        rows["symbol"].append("ABC")
        rows["kind"].append(kind)
        rows["underlying"].append("S")
        rows["option"].append(option)
        rows["expiry"].append(np.datetime64(expiry, "D"))
        rows["strike"].append(strike)
        for name, value in (
            ("open", close),
            ("high", close),
            ("low", close),
            ("close", close),
            ("settle", settle),
            ("value", 1.0),
            ("last", close),
            ("prev_close", close),
            ("underlying_price", spot),
        ):
            rows[name].append(value)
        rows["contracts"].append(contracts)
        rows["oi"].append(oi)
        rows["chg_oi"].append(chg)
        rows["lot"].append(100)

    near, far = date(2024, 7, 25), date(2024, 8, 29)
    forwards = {near: spot * 1.004, far: spot * 1.011}
    add("F", near, np.nan, "", forwards[near], forwards[near], 500, 9000, 600)
    add("F", far, np.nan, "", forwards[far], forwards[far], 50, 1000, 200)
    for expiry, forward in forwards.items():
        years = (expiry - day).days / 365
        for strike in np.arange(80.0, 121.0, 2.5):
            for option in ("CE", "PE"):
                price = float(
                    black76(
                        forward, strike, years, vol_of(strike / forward), option == "CE"
                    )
                )
                oi = 1000 + (500 if (option == "CE") == (strike > 100) else 0)
                add(
                    "O",
                    expiry,
                    strike,
                    option,
                    price,
                    price,
                    10,
                    oi,
                    -50 if option == "CE" else 70,
                )
    return {
        k: np.array(v) if k not in ("expiry",) else np.array(v, "datetime64[D]")
        for k, v in rows.items()
    }


def test_session_features_recover_known_volatility_and_open_interest() -> None:
    day = date(2024, 7, 5)
    table = session_table(day, 100.0, lambda m: 0.30)
    cm = {"ABC": {"series": "EQ", "cm_prev_close": 99.0, "cm_close": 100.0}}
    stocks, indices = features.session_features(day, table, cm, {}, {"XYZ"})
    (row,) = stocks
    assert indices == [] and row["fno"] == 1 and row["in_ban"] == 0
    assert row["fut_expiry_near"] == "2024-07-25" and row["fut_dte_near"] == 20
    assert row["fut_oi_total"] == 10000 and row["fut_chg_oi_total"] == 800
    assert row["fut_oi_change_pct"] == pytest.approx(800 / 9200 * 100)
    assert row["fut_basis_pct"] == pytest.approx(0.4)
    assert row["fut_rollover_pct"] == pytest.approx(10.0)
    assert row["fut_buildup"] == "long_buildup"
    assert row["iv_atm_near"] == pytest.approx(0.30, abs=1e-9)
    assert row["iv_atm_next"] == pytest.approx(0.30, abs=1e-9)
    assert row["iv_atm_30d"] == pytest.approx(0.30, abs=1e-9)
    assert row["iv_skew_near"] == pytest.approx(0.0, abs=1e-9)
    strikes = np.arange(80.0, 121.0, 2.5)
    ce = np.array([1500.0 if k > 100 else 1000.0 for k in strikes])
    pe = np.array([1000.0 if k > 100 else 1500.0 for k in strikes])
    pain = [
        sum(c * max(s - k, 0) for k, c in zip(strikes, ce, strict=True))
        + sum(p * max(k - s, 0) for k, p in zip(strikes, pe, strict=True))
        for s in strikes
    ]
    assert row["opt_max_pain_near"] == strikes[int(np.argmin(pain))]
    assert row["opt_pcr_oi_near"] == pytest.approx(pe.sum() / ce.sum())
    assert row["opt_call_wall_near"] == 102.5 and row["opt_put_wall_near"] == 80.0
    assert row["opt_chg_oi_ce"] == -50 * 34 and row["opt_chg_oi_pe"] == 70 * 34


def test_skew_and_term_structure_follow_the_smile() -> None:
    day = date(2024, 7, 5)
    table = session_table(day, 100.0, lambda m: 0.30 - 0.5 * (m - 1))
    (row,), _ = features.session_features(day, table, {}, {}, None)
    assert row["iv_skew_near"] == pytest.approx(0.05, abs=2e-3)
    assert row["iv_atm_near"] == pytest.approx(0.30, abs=2e-3)
    # No cash row and no ban list: spot is the F&O file's underlying price.
    assert row["in_ban"] == "" and row["series"] == ""
    assert row["fut_basis_pct"] == pytest.approx(0.4) and row["fut_buildup"] == ""


def test_buildup_classes() -> None:
    assert features.buildup(1, 5) == "long_buildup"
    assert features.buildup(-1, 5) == "short_buildup"
    assert features.buildup(1, -5) == "short_covering"
    assert features.buildup(-1, -5) == "long_unwinding"
    assert features.buildup(0, 5) == "neutral"
    assert features.buildup(float("nan"), 5) == ""


def test_readers_of_the_side_files() -> None:
    cm = features.read_cm(
        "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
        "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, "
        "NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
        "ABC, BE, 05-Jul-2024, 9, 9, 10, 8, 9.5, 9.6, 9.4, 100, 0.01, 4, 30, 30.00\n"
        "ABC, EQ, 05-Jul-2024, 10, 10, 11, 9, 10, 10.5, 10.2, 200, 0.02, 5, -, -\n"
        "XYZ, BE, 05-Jul-2024, 5, 5, 5, 5, 5, 5, 5, 10, 0.0, 1, 10, 100.00\n"
    )
    assert cm["ABC"]["series"] == "EQ" and cm["ABC"]["cm_close"] == 10.5
    assert np.isnan(cm["ABC"]["cm_deliv_pct"]) and cm["XYZ"]["cm_deliv_pct"] == 100.0
    mwpl = features.read_mwpl(
        "Date, ISIN, Scrip Name, NSE Symbol, MWPL, Open Interest, Future Equivalent "
        "Open Interest, Limit for Next Day\n"
        "15-OCT-2026,INE1,A,AMBUJACEM,1000,1100,940.5,No Fresh Positions\n"
    )
    assert mwpl["AMBUJACEM"]["mwpl_oi_pct"] == pytest.approx(110.0)
    assert mwpl["AMBUJACEM"]["mwpl_feoi_pct"] == pytest.approx(94.05)
    assert features.read_ban(
        "Securities in Ban For Trade Date 15-OCT-2026:\n1,AMBUJACEM\n"
    ) == {"AMBUJACEM"}
    header = "Client Type," + ",".join(n for n, _ in features.PARTICIPANT_COLUMNS)
    body = "\n".join(
        f"{w}," + ",".join(str(i) for i in range(14))
        for w in ("Client", "DII", "FII", "Pro", "TOTAL")
    )
    values = features.read_participants(f"title as on Jan 03 2020\n{header}\n{body}\n")
    assert values["fii_fut_idx_short"] == 1 and "total_fut_idx_long" not in values


def test_indices_are_read_and_give_index_underlyings_their_official_close() -> None:
    found = features.read_indices(
        "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,"
        "Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,"
        "P/B,Div Yield\n"
        "Nifty 50,05-07-2024,24213.35,24363,24168.85,24323.85,21.7,.09,298381204,"
        "32052.48,23.15,4.19,1.23\n"
        "India VIX,05-07-2024,12.5,13,12.1,12.9,0.3,2.4,-,-,-,-,-\n"
    )
    assert found["Nifty 50"]["close"] == 24323.85 and found["Nifty 50"]["pe"] == 23.15
    assert found["Nifty 50"]["change_pct"] == 0.09
    assert found["India VIX"]["close"] == 12.9 and np.isnan(found["India VIX"]["pe"])
    day = date(2024, 7, 5)
    table = session_table(day, 100.0, lambda m: 0.20)
    table["symbol"] = np.full(len(table["symbol"]), "NIFTY")
    table["underlying"] = np.full(len(table["symbol"]), "I")
    # The F&O file's underlying price (here 100) gives way to NSE's index close.
    official = {"Nifty 50": {"close": 100.4}}
    _, (row,) = features.session_features(day, table, {}, {}, None, official)
    assert row["spot"] == 100.4 and row["fut_basis_pct"] == pytest.approx(0.0)
    _, (row,) = features.session_features(day, table, {}, {}, None)
    assert row["spot"] == 100.0 and row["fut_basis_pct"] == pytest.approx(0.4)
