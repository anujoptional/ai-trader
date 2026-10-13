"""Paper-trading universe: the largest NSE stocks by market capitalisation.

Ranks every Listed, EQ-series security in NSE's daily market-cap file
(``mcapDDMMYYYY.csv`` from the daily PR bundle,
https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PRDDMMYY.zip) by
market cap and keeps the top ``--target`` plus ``--reserves`` more. The session
scans the ``target`` highest-ranked stocks it can score at each decision and fills
any gap (too little history, too few trades today) with the next ranked reserve.
Industry comes from NSE's Nifty 500 list where the stock is in it. Trade-for-trade
(BE), REIT/InvIT and other non-EQ series are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.train_tp_only import file_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--industries", type=Path, help="earlier universe.json")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--reserves", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.mcap.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            {k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(handle)
        ]
    listed = [r for r in rows if r["Series"] == "EQ" and r["Category"] == "Listed"]
    ranked = sorted(listed, key=lambda r: -float(r["Market Cap(Rs.)"]))
    industries = {}
    if args.industries:
        earlier = json.loads(args.industries.read_text("utf-8"))
        industries = {
            e["symbol"]: e.get("industry", "")
            for e in earlier["instruments"]
            if e["role"] == "equity_target"
        }
    kept = ranked[: args.target + args.reserves]
    universe = {
        "source": f"NSE market capitalisation file {args.mcap.name}",
        "csv_sha256": file_hash(args.mcap),
        "trade_date": kept[0]["Trade Date"],
        "target": args.target,
        "instruments": [
            {
                "symbol": r["Symbol"],
                "role": "equity_target",
                "rank": rank,
                "company": r["Security Name"],
                "industry": industries.get(r["Symbol"], ""),
                "market_cap_cr": round(float(r["Market Cap(Rs.)"]) / 1e7, 1),
            }
            for rank, r in enumerate(kept, start=1)
        ]
        + [
            {"symbol": "NIFTY", "role": "context"},
            {"symbol": "INDIAVIX", "role": "context"},
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(universe, indent=1), "utf-8")
    print(
        f"{len(listed)} listed EQ stocks; kept ranks 1-{len(kept)}; "
        f"rank {args.target} market cap Rs {kept[args.target - 1]['Market Cap(Rs.)']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
