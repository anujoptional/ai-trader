import io
import json
import zipfile

import numpy as np

from ai_trader.fno.kite_capture import FIELDS
from ai_trader.fno.nse_archive import DATASETS
from scripts import validate_fno_data as check
from tests.test_fno_data import DAY, UDIFF


def test_capture_check_compares_each_contracts_day_with_nses_file(
    tmp_path, monkeypatch
) -> None:
    nse, fno = tmp_path / "nse", tmp_path / "fno"
    path = DATASETS["fo"].path(nse, DAY)
    path.parent.mkdir(parents=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(DATASETS["fo"].member(DAY), UDIFF)
    path.write_bytes(buffer.getvalue())
    store = fno / "kite_minute"
    store.mkdir(parents=True)
    contracts = {
        "0": {"tradingsymbol": "ABC24JULFUT", "underlying": "ABC", "kind": "FUT"},
        "1": {"tradingsymbol": "ABC24JUL100CE", "underlying": "ABC", "kind": "CE"},
        "2": {"tradingsymbol": "XYZ24JULFUT", "underlying": "XYZ", "kind": "FUT"},
    }
    for ident, info in contracts.items():
        info.update(
            expiry="2024-07-25", lot_size=1500, token=int(ident), tick_size=0.05
        )
        info["strike"] = 100.0 if info["kind"] == "CE" else None
    (store / "contracts.json").write_text(json.dumps(contracts))
    # NSE: the future traded 120 contracts (x 1500) and ended with OI 450000; the
    # call 40 contracts and OI 30000. Rows arrive unsorted, as a merge could leave.
    rows = np.array(
        [
            # contract, minute, open, high, low, close, volume, oi
            [0, 374, 10200, 10200, 10200, 10200, 30000, 450000],
            [1, 5, 320, 320, 320, 320, 60000, 29000],  # last OI differs from NSE
            [0, 0, 10100, 10100, 10100, 10100, 150000, 445000],
            [2, 3, 500, 500, 500, 500, 1500, 1500],  # not in NSE's file
        ],
        dtype=np.int64,
    )
    np.savez_compressed(
        store / f"{DAY.isoformat()}.npz",
        **{name: rows[:, i] for i, name in enumerate(FIELDS)},
    )
    np.savez_compressed(
        store / "2024-07-08.npz", **{name: rows[:1, i] for i, name in enumerate(FIELDS)}
    )
    monkeypatch.setattr(check, "NSE", nse)
    monkeypatch.setattr(check, "FNO", fno)
    result = check.capture_check()
    assert result["contracts"] == 2
    assert result["volume_equal"] == 2
    assert result["last_oi_equal"] == 1
    assert result["not_in_nse_file"] == 1
    assert result["sessions_without_nse_file_yet"] == 1
    assert result["examples"] == [
        [DAY.isoformat(), "ABC24JUL100CE", 60000, 60000, 29000, 30000]
    ]


def test_capture_check_without_a_capture(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(check, "FNO", tmp_path)
    assert check.capture_check() == {"status": "no capture yet"}
