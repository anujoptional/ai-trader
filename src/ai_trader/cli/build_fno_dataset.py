"""Build the F&O-derived dataset from NSE's archive (``fetch_nse_archives``).

  python -m ai_trader.cli.build_fno_dataset            # every session in data/nse

Outputs (under ``--out``, default data/fno):
- normalized/<YYYY>/<date>.npz: the session's F&O bhavcopy in one schema
  (fno/bhavcopy.py; the UDiFF file where NSE has one, else the old format);
- features/stock_daily_<YYYY>.csv.gz: one row per stock and session, every EQ/BE
  stock in NSE's cash bhavcopy with its official prices and delivery, plus the
  futures/options/IV/ban/MWPL features for F&O stocks (fno/features.py);
- features/index_daily.csv.gz: the same futures/options/IV features per F&O index
  (spot = NSE's official index close);
- features/market_daily.csv.gz: participant-wise OI and volume (Client, DII, FII,
  Pro) with long shares, and the number of stocks in the F&O ban;
- features/indices_daily.csv.gz: every NSE index's official OHLC, volume, turnover,
  P/E, P/B and dividend yield, and India VIX, per session;
- features/build.json: sessions, row counts, and the SHA-256 of every source
  manifest and of the code that built it.

All values of a session come from files NSE publishes after it closes: use them
from the next session on.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.fno import bhavcopy, features
from ai_trader.fno.nse_archive import DATASETS

STOCK_COLUMNS = (
    ["date", "symbol", "series"]
    + [name for _, name in features.CM_FIELDS]
    + ["fno", "in_ban", "mwpl_oi_pct", "mwpl_feoi_pct", "mwpl_limit"]
    + list(features.FUTURES_FIELDS)
    + list(features.OPTION_FIELDS)
)
INDEX_COLUMNS = (
    ["date", "symbol", "spot"]
    + list(features.FUTURES_FIELDS)
    + list(features.OPTION_FIELDS)
)
WHO = ("client", "dii", "fii", "pro")
INDICES_COLUMNS = ["date", "index"] + [name for _, name in features.INDEX_FIELDS]


def market_columns() -> list[str]:
    raw = [
        f"{kind}_{who}_{name}"
        for kind in ("oi", "vol")
        for who in WHO
        for _, name in features.PARTICIPANT_COLUMNS
    ]
    derived = [
        f"oi_{who}_{name}"
        for who in WHO
        for name in (
            "idx_fut_long_pct",
            "stk_fut_long_pct",
            "idx_call_net",
            "idx_put_net",
        )
    ]
    return ["date", *raw, *derived, "stocks_in_ban", "fno_stocks"]


def _cell(value: object) -> str:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else format(value, ".12g")
    return str(value)


def _text(dataset: str, root: Path, day: date) -> str | None:
    spec = DATASETS[dataset]
    path = spec.path(root, day)
    return spec.text(path.read_bytes(), day) if path.exists() else None


def build_session(args: tuple[str, str, str]) -> dict[str, object]:
    """One session's normalized table (cached) and its feature rows as CSV lines."""
    iso, nse_root, out_root = args
    day = date.fromisoformat(iso)
    root, out = Path(nse_root), Path(out_root)
    errors: list[str] = []

    def attempt(component: str, read, fallback):
        try:
            return read()
        except Exception as error:
            errors.append(f"{component}: {type(error).__name__}: {error}"[:300])
            return fallback

    cached = out / "normalized" / f"{day.year}" / f"{iso}.npz"
    inputs = [DATASETS[name].path(root, day) for name in ("fo", "fo_old")]
    newest = max(
        [Path(bhavcopy.__file__).stat().st_mtime_ns]
        + [path.stat().st_mtime_ns for path in inputs if path.exists()]
    )
    table, source = None, ""
    if cached.exists() and cached.stat().st_mtime_ns > newest:
        table, source = bhavcopy.load_table(cached)
    else:
        loaded = attempt("fo", lambda: bhavcopy.load_session(root, day), None)
        if loaded is not None:
            table, source = loaded
            bhavcopy.save_table(cached, table, source)
    cm_text = _text("cm", root, day)
    cm = attempt("cm", lambda: features.read_cm(cm_text), {}) if cm_text else {}
    mwpl_text = _text("mwpl", root, day)
    mwpl = (
        attempt("mwpl", lambda: features.read_mwpl(mwpl_text), {}) if mwpl_text else {}
    )
    ban_text = _text("ban", root, day)
    ban = (
        attempt("ban", lambda: features.read_ban(ban_text), None) if ban_text else None
    )
    indices_text = _text("indices", root, day)
    indices = (
        attempt("indices", lambda: features.read_indices(indices_text), {})
        if indices_text
        else {}
    )
    stock_rows, index_rows = features.session_features(
        day, table, cm, mwpl, ban, indices
    )
    market: dict[str, object] = {"date": iso}
    for kind, dataset in (("oi", "participant_oi"), ("vol", "participant_vol")):
        text = _text(dataset, root, day)
        if text is not None:
            values = attempt(dataset, lambda t=text: features.read_participants(t), {})
            for name, value in values.items():
                market[f"{kind}_{name}"] = value
    for who in WHO:
        get = market.get
        long_, short = get(f"oi_{who}_fut_idx_long"), get(f"oi_{who}_fut_idx_short")
        if long_ is not None and long_ + short > 0:
            market[f"oi_{who}_idx_fut_long_pct"] = long_ / (long_ + short) * 100
        long_, short = get(f"oi_{who}_fut_stk_long"), get(f"oi_{who}_fut_stk_short")
        if long_ is not None and long_ + short > 0:
            market[f"oi_{who}_stk_fut_long_pct"] = long_ / (long_ + short) * 100
        if get(f"oi_{who}_opt_idx_call_long") is not None:
            market[f"oi_{who}_idx_call_net"] = (
                market[f"oi_{who}_opt_idx_call_long"]
                - market[f"oi_{who}_opt_idx_call_short"]
            )
            market[f"oi_{who}_idx_put_net"] = (
                market[f"oi_{who}_opt_idx_put_long"]
                - market[f"oi_{who}_opt_idx_put_short"]
            )
    market["stocks_in_ban"] = "" if ban is None else len(ban)
    market["fno_stocks"] = sum(1 for r in stock_rows if r.get("fno") == 1)
    return {
        "date": iso,
        "source": source,
        "errors": errors,
        "has_cm": cm_text is not None,
        "stock": [[_cell(r.get(c, "")) for c in STOCK_COLUMNS] for r in stock_rows],
        "index": [[_cell(r.get(c, "")) for c in INDEX_COLUMNS] for r in index_rows],
        "market": [_cell(market.get(c, "")) for c in market_columns()],
        "indices": [
            [iso, name] + [_cell(values[n]) for _, n in features.INDEX_FIELDS]
            for name, values in sorted(indices.items())
        ],
    }


def sessions(root: Path) -> list[date]:
    """Sessions with an F&O or cash bhavcopy in the archive."""
    found = set()
    for name in ("fo", "fo_old", "cm"):
        manifest = root / "manifest" / f"{name}.json"
        if manifest.exists():
            found |= {
                date.fromisoformat(d)
                for d in json.loads(manifest.read_text("utf-8"))["files"]
            }
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nse", type=Path, default=Path("data/nse"))
    parser.add_argument("--out", type=Path, default=Path("data/fno"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    days = [d for d in sessions(args.nse) if d >= args.start]
    folder = args.out / "features"
    folder.mkdir(parents=True, exist_ok=True)
    writers: dict[str, tuple] = {}

    def writer(name: str, columns: list[str]):
        if name not in writers:
            handle = gzip.open(folder / f"{name}.csv.gz.part", "wt", newline="")
            out = csv.writer(handle)
            out.writerow(columns)
            writers[name] = (handle, out)
        return writers[name][1]

    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    errors: dict[str, list[str]] = {}
    jobs = [(d.isoformat(), str(args.nse), str(args.out)) for d in days]
    with ProcessPoolExecutor(args.workers) as pool:
        for done, result in enumerate(pool.map(build_session, jobs, chunksize=4), 1):
            year = result["date"][:4]
            if result["errors"]:
                errors[result["date"]] = result["errors"]
                print(f"{result['date']}: {'; '.join(result['errors'])}", flush=True)
            stock = writer(f"stock_daily_{year}", list(STOCK_COLUMNS))
            stock.writerows(result["stock"])
            writer("index_daily", list(INDEX_COLUMNS)).writerows(result["index"])
            writer("market_daily", market_columns()).writerow(result["market"])
            writer("indices_daily", INDICES_COLUMNS).writerows(result["indices"])
            for name, n in (
                (f"stock_daily_{year}", len(result["stock"])),
                ("index_daily", len(result["index"])),
                ("market_daily", 1),
                ("indices_daily", len(result["indices"])),
            ):
                counts[name] = counts.get(name, 0) + n
            sources[result["source"] or "none"] = (
                sources.get(result["source"] or "none", 0) + 1
            )
            if done % 100 == 0:
                print(
                    f"{datetime.now():%H:%M:%S} {done}/{len(jobs)} sessions", flush=True
                )
    for name, (handle, _) in writers.items():
        handle.close()
        (folder / f"{name}.csv.gz.part").replace(folder / f"{name}.csv.gz")
    code = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(features.__file__).parent.glob("*.py"))
    }
    code[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((args.nse / "manifest").glob("*.json"))
    }
    build = {
        "built_at": datetime.now(INDIA_TIMEZONE).isoformat(timespec="seconds"),
        "sessions": len(days),
        "first": days[0].isoformat() if days else None,
        "last": days[-1].isoformat() if days else None,
        "fo_source_sessions": sources,
        "rows": counts,
        "unreadable_files": errors,
        "stock_columns": list(STOCK_COLUMNS),
        "index_columns": list(INDEX_COLUMNS),
        "market_columns": market_columns(),
        "indices_columns": INDICES_COLUMNS,
        "code_sha256": code,
        "source_manifests_sha256": manifests,
        "usable_from": "the session after each row's date (published after it)",
    }
    (folder / "build.json").write_text(json.dumps(build, indent=1), "utf-8")
    print(json.dumps({k: build[k] for k in ("sessions", "first", "last", "rows")}))
    print(f"sessions with an unreadable file: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
