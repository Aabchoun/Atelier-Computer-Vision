"""Download FIRMS hotspots for an area and period, and print a per-day summary.

Use it to pick the case-study fire (days with many high-confidence hotspots), then to fill
the `firms.csv` of a scene.

    python scripts/fetch_firms.py --bbox 8.4 36.6 9.2 37.1 --start 2023-07-17 --end 2023-07-31
    python scripts/fetch_firms.py --bbox W S E N --start ... --end ... --source VIIRS_NOAA20_SP --out data/scenes/my_fire/firms.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.config import resolve_path  # noqa: E402
from firespread.io.firms_api import SOURCES, fetch_hotspots, write_hotspots_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="degrees")
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--source", default="VIIRS_SNPP_SP", choices=SOURCES, help="*_SP for past fires, *_NRT for recent days")
    parser.add_argument("--out", default=None, help="CSV path (default: data/raw/firms/<source>_<start>_<end>.csv)")
    args = parser.parse_args()

    rows = fetch_hotspots(tuple(args.bbox), args.start, args.end, args.source)
    out = resolve_path(args.out or f"data/raw/firms/{args.source}_{args.start}_{args.end}.csv")
    write_hotspots_csv(rows, out)

    print(f"{len(rows)} hotspots ({args.source}) saved to {out}")
    if rows:
        per_day = Counter(r["acq_date"] for r in rows)
        high = Counter(r["acq_date"] for r in rows if r.get("confidence", "").lower() in ("h", "high") or _number(r.get("confidence")) >= 80)
        print("\n date        hotspots  high-confidence")
        for day in sorted(per_day):
            print(f" {day}  {per_day[day]:>8}  {high[day]:>15}")


def _number(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


if __name__ == "__main__":
    main()
