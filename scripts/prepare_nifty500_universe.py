"""Paper-trading universe: NSE Nifty 500 constituents (EQ series) plus context.

Kite has no market-capitalisation data, so the top 500 by market cap is taken from
NSE's Nifty 500 index list (the largest eligible companies by full market
capitalisation; https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv).
REITs (series RR) and trade-for-trade stocks (series BE, which cannot be sold the
day they are bought) are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.train_tp_only import file_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    kept = [r for r in rows if r["Series"].strip() == "EQ"]
    excluded = [
        {"symbol": r["Symbol"].strip(), "series": r["Series"].strip()}
        for r in rows
        if r["Series"].strip() != "EQ"
    ]
    universe = {
        "source": "NSE Nifty 500 constituents (ind_nifty500list.csv)",
        "csv_sha256": file_hash(args.csv),
        "constituents": len(rows),
        "excluded": excluded,
        "instruments": [
            {
                "symbol": r["Symbol"].strip(),
                "role": "equity_target",
                "company": r["Company Name"].strip(),
                "industry": r["Industry"].strip(),
                "isin": r["ISIN Code"].strip(),
            }
            for r in sorted(kept, key=lambda r: r["Symbol"].strip())
        ]
        + [
            {"symbol": "NIFTY", "role": "context"},
            {"symbol": "INDIAVIX", "role": "context"},
        ],
    }
    args.output.write_text(json.dumps(universe, indent=1), "utf-8")
    print(f"{len(kept)} stocks; excluded {excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
