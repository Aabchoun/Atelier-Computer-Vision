"""Build one scene per day of a fire, sharing the downloads, for the day-by-day evaluation.

    python scripts/build_sequence.py --name evros_2023 --bbox 25.65 40.74 26.50 41.32 --fire-start 2023-08-19 --start 2023-08-19 --end 2023-09-02

Layout:
    data/sequences/<name>/
        sequence.json          days, Sentinel-2 dates, provenance
        shared/                downloaded once (+ .json info): s2_<date>, era5_<date>, dem, worldcover, firms_raw.csv
        days/<YYYY-MM-DD>/     scene.json (points to ../../shared/...), firms.csv, truth.npz

A day gets an `s2_post` scene only if Sentinel-2 passed over the AOI that day with enough visible
area. On the other days the coordinator falls back on its last known optical layers (revisit gap),
which is how the real system works. Downloads are cached in shared/: re-running resumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.config import resolve_path  # noqa: E402
from firespread.common.scene import DEM, ERA5, S2_POST, S2_PRE, TRUTH, WORLDCOVER, Raster  # noqa: E402
from firespread.io import gee  # noqa: E402
from firespread.io.firms import points_from_rows, write_firms_csv  # noqa: E402
from firespread.io.firms_api import SOURCES, fetch_hotspots, write_hotspots_csv  # noqa: E402
from firespread.io.scene_io import raster_to_npz, write_scene_json  # noqa: E402


def cached(shared: Path, key: str, download: Callable[[], Raster]) -> dict[str, Any]:
    """scene.json entry of shared/<key>.npz, downloading it only if it is not there yet."""
    npz, info_path = shared / f"{key}.npz", shared / f"{key}.json"
    if npz.exists() and info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    print(f"- downloading {key} ...", end=" ", flush=True)
    info = raster_to_npz(download(), npz)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"done ({time.perf_counter() - start:.0f} s)")
    return info


def reference(info: dict[str, Any], shared: Path, day_dir: Path) -> dict[str, Any]:
    """The same entry, with a file path relative to the day folder."""
    return {**info, "file": Path(os.path.relpath(shared / info["file"], day_dir)).as_posix()}


def firms_by_day(shared: Path, bbox, start: date, end: date, sources: list[str]) -> dict[str, list[dict[str, str]]]:
    path = shared / "firms_raw.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [row for source in sources for row in fetch_hotspots(bbox, start, end, source)]
        write_hotspots_csv(rows, path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["acq_date"]].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--fire-start", type=date.fromisoformat, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="first forecast day")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="last forecast day (truth = end + 1)")
    parser.add_argument("--min-usable", type=float, default=0.5, help="minimum visible share of the AOI for a Sentinel-2 day")
    parser.add_argument("--resolution", type=float, default=375.0)
    parser.add_argument("--fine-factor", type=int, default=16)
    parser.add_argument("--firms-sources", nargs="+", default=["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"], choices=SOURCES)
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    root = resolve_path(f"data/sequences/{args.name}")
    shared = root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    grid = gee.make_grid(bbox, args.resolution)
    print(f"Grid: {grid.width} x {grid.height} cells at {grid.resolution:.0f} m in {grid.crs}")
    gee.initialize()

    s2_days = gee.list_s2_days(grid, args.fire_start - timedelta(days=45), args.end)
    pre = gee.pick_pre_fire(s2_days, args.fire_start)
    if pre is None:
        sys.exit("No usable pre-fire Sentinel-2 scene.")
    post_days = [d for d in s2_days if args.start <= d.day <= args.end and d.usable >= args.min_usable]
    print(f"Pre-fire scene: {pre.day}. Sentinel-2 during the period: " + ", ".join(f"{d.day:%m-%d} ({d.usable:.0%})" for d in post_days))

    ff = args.fine_factor
    pre_info = cached(shared, f"s2_{pre.day}", lambda: gee.s2_raster(grid, pre.day, ff))
    post_infos = {d.day: cached(shared, f"s2_{d.day}", lambda day=d.day: gee.s2_raster(grid, day, ff)) for d in post_days}
    dem_info = cached(shared, "dem", lambda: gee.dem_raster(grid))
    worldcover_info = cached(shared, "worldcover", lambda: gee.worldcover_raster(grid, ff))
    firms = firms_by_day(shared, bbox, args.start, args.end + timedelta(days=1), args.firms_sources)

    days = []
    day = args.start
    while day <= args.end:
        era5_info = cached(shared, f"era5_{day}", lambda d=day: gee.era5_raster(grid, d))
        day_dir = root / "days" / day.isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)

        points = points_from_rows(firms.get(day.isoformat(), []), grid, "VIIRS")
        write_firms_csv(points, day_dir / "firms.csv")
        truth_points = points_from_rows(firms.get((day + timedelta(days=1)).isoformat(), []), grid, "VIIRS").filter(0.5)
        truth = Raster({"fire_next": grid.rasterize_points(truth_points.x, truth_points.y)}, grid.resolution,
                       datetime.combine(day + timedelta(days=1), datetime.min.time()), "FIRMS VIIRS (day t+1, nominal+high)")

        rasters = {
            S2_PRE: reference(pre_info, shared, day_dir),
            ERA5: reference(era5_info, shared, day_dir),
            DEM: reference(dem_info, shared, day_dir),
            WORLDCOVER: reference(worldcover_info, shared, day_dir),
            TRUTH: raster_to_npz(truth, day_dir / "truth.npz"),
        }
        if day in post_infos:
            rasters[S2_POST] = reference(post_infos[day], shared, day_dir)
        write_scene_json(day_dir, f"{args.name}_{day}", grid, rasters, "firms.csv" if len(points) else None, "VIIRS",
                         provenance=gee.provenance(sequence=args.name, forecast_day=day, s2_pre=pre.day, s2_post=day if day in post_infos else None))
        n_truth = int(truth["fire_next"].sum())
        print(f"  {day}: {len(points):5d} hotspots, truth {n_truth:4d} cells, Sentinel-2 {'yes' if day in post_infos else '-'}")
        days.append({"day": day.isoformat(), "hotspots": len(points), "truth_cells": n_truth, "s2_post": day in post_infos})
        day += timedelta(days=1)

    (root / "sequence.json").write_text(json.dumps({
        "name": args.name, "bbox_lonlat": list(bbox), "grid": grid.to_dict(), "fire_start": args.fire_start.isoformat(),
        "s2_pre": pre.day.isoformat(), "firms_sources": args.firms_sources, "days": days,
        "s2_days": [{"day": d.day.isoformat(), "coverage": round(d.coverage, 3), "cloud": round(d.cloud, 3)} for d in s2_days],
    }, indent=2), encoding="utf-8")
    print(f"\nSequence saved to {root}\nNext: python scripts/run_sequence.py --sequence {root.relative_to(resolve_path('.')).as_posix()}")


if __name__ == "__main__":
    main()
