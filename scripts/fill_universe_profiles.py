"""Complete the paper-trading universe's static profile of every stock.

Company: NSE's full registered name (``EQUITY_L.csv``, NSE's list of listed
equities). Industry: the NSE sector (one of 22) from NSE's index constituent lists
(Nifty 500, then Nifty Total Market, then Nifty Microcap 250); a stock in none of
them takes ``industry_overrides.csv`` (sector chosen from NSE's industry
classification structure, with the basis written next to it). Every stock must end
with both, and the industry must be one of NSE's sectors; otherwise nothing is
written. Usage:

    python -m scripts.fill_universe_profiles --universe data/universe_mcap_20261012
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

INDEX_LISTS = (
    "ind_nifty500list.csv",
    "ind_niftytotalmarket_list.csv",
    "ind_niftymicrocap250_list.csv",
)
SECTORS = frozenset(
    {
        "Automobile and Auto Components",
        "Capital Goods",
        "Chemicals",
        "Construction",
        "Construction Materials",
        "Consumer Durables",
        "Consumer Services",
        "Diversified",
        "Fast Moving Consumer Goods",
        "Financial Services",
        "Forest Materials",
        "Healthcare",
        "Information Technology",
        "Media Entertainment & Publication",
        "Metals & Mining",
        "Oil Gas & Consumable Fuels",
        "Power",
        "Realty",
        "Services",
        "Telecommunication",
        "Textiles",
        "Utilities",
    }
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(handle)
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True, help="universe folder")
    parser.add_argument(
        "--nifty500",
        type=Path,
        default=Path("data/universe_nifty500_20261012/ind_nifty500list.csv"),
    )
    args = parser.parse_args()
    sources = args.universe / "industry_sources"
    lists = [args.nifty500] + [sources / name for name in INDEX_LISTS[1:]]
    industry: dict[str, tuple[str, str]] = {}
    for path in lists:
        for r in rows(path):
            industry.setdefault(r["Symbol"], (r["Industry"], f"NSE {path.name}"))
    overrides = {
        r["symbol"]: (r["industry"], f"manual: {r['basis']}")
        for r in rows(args.universe / "industry_overrides.csv")
    }
    names = {r["SYMBOL"]: r["NAME OF COMPANY"] for r in rows(sources / "EQUITY_L.csv")}
    path = args.universe / "universe.json"
    universe = json.loads(path.read_text("utf-8"))
    problems, used = [], set()
    for entry in universe["instruments"]:
        if entry["role"] != "equity_target":
            continue
        symbol = entry["symbol"]
        if symbol in industry and symbol in overrides:
            problems.append(f"{symbol}: in an NSE list and in the overrides")
        value, source = industry.get(symbol) or overrides.get(symbol) or ("", "")
        used.add(symbol)
        entry["company"] = names.get(symbol) or entry.get("company", "")
        entry["industry"] = value
        entry["industry_source"] = source
        if not entry["company"]:
            problems.append(f"{symbol}: no company name")
        if value not in SECTORS:
            problems.append(f"{symbol}: industry {value!r} is not an NSE sector")
    problems += [
        f"{s}: override for a stock outside the universe"
        for s in overrides.keys() - used
    ]
    if problems:
        raise SystemExit("not written:\n" + "\n".join(problems))
    universe["profiles"] = {
        "company": "NSE EQUITY_L.csv (full registered name)",
        "industry": "NSE sector: index constituent lists, else industry_overrides.csv",
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *lists,
                sources / "EQUITY_L.csv",
                args.universe / "industry_overrides.csv",
            ]
        },
    }
    path.write_text(json.dumps(universe, indent=1), "utf-8")
    stocks = [e for e in universe["instruments"] if e["role"] == "equity_target"]
    by_source = {}
    for e in stocks:
        key = e["industry_source"].split(":")[0]
        by_source[key] = by_source.get(key, 0) + 1
    print(f"{len(stocks)} stocks complete; industry from {by_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
