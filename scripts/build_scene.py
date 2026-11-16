"""Build a real scene (all layers on one grid) from Earth Engine + FIRMS, ready for run_pipeline.py.

    # 1. look at the Sentinel-2 acquisitions first (coverage / clouds per day)
    python scripts/build_scene.py --name evros --bbox 25.65 40.74 26.50 41.32 --fire-start 2023-08-19 --date 2023-08-23 --list-s2

    # 2. build the scene for forecast day t = --date (FIRMS on day t, ground truth = FIRMS on day t+1)
    python scripts/build_scene.py --name evros_2023-08-23 --bbox 25.65 40.74 26.50 41.32 --fire-start 2023-08-19 --date 2023-08-23

The pre-fire and post (latest during-fire) Sentinel-2 scenes are chosen automatically unless
--pre-date / --post-date are given. Needs GEE_PROJECT and FIRMS_MAP_KEY in .env.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.config import resolve_path  # noqa: E402
from firespread.common.scene import DEM, ERA5, S2_POST, S2_PRE, TRUTH, WORLDCOVER, Raster, RawScene  # noqa: E402
from firespread.io import gee  # noqa: E402
from firespread.io.firms import points_from_rows  # noqa: E402
from firespread.io.firms_api import SOURCES, fetch_hotspots, write_hotspots_csv  # noqa: E402
from firespread.io.scene_io import save_raw_scene  # noqa: E402


def fetch_firms_day(bbox, day: date, sources: list[str]) -> list[dict[str, str]]:
    rows = []
    for source in sources:
        rows += fetch_hotspots(bbox, day, day, source)
    return rows


def step(message: str):
    print(f"- {message} ...", end=" ", flush=True)
    return time.perf_counter()


def done(start: float, detail: str = "") -> None:
    print(f"done ({time.perf_counter() - start:.0f} s){' ' + detail if detail else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="scene name, e.g. evros_2023-08-23")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--fire-start", type=date.fromisoformat, required=True, help="first day of the fire (from fetch_firms.py)")
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="forecast day t")
    parser.add_argument("--pre-date", type=date.fromisoformat, default=None, help="pre-fire Sentinel-2 day (default: auto)")
    parser.add_argument("--post-date", type=date.fromisoformat, default=None, help="during-fire Sentinel-2 day (default: auto)")
    parser.add_argument("--resolution", type=float, default=375.0, help="common grid resolution (m)")
    parser.add_argument("--fine-factor", type=int, default=16, help="optical pixels per grid cell per side (16 -> 23.4 m)")
    parser.add_argument("--firms-sources", nargs="+", default=["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"], choices=SOURCES)
    parser.add_argument("--out", default=None, help="default: data/scenes/<name>")
    parser.add_argument("--list-s2", action="store_true", help="only list the Sentinel-2 acquisitions and stop")
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    grid = gee.make_grid(bbox, args.resolution)
    print(f"Grid: {grid.width} x {grid.height} cells at {grid.resolution:.0f} m in {grid.crs} "
          f"({grid.width * grid.resolution / 1000:.0f} x {grid.height * grid.resolution / 1000:.0f} km)")
    gee.initialize()

    t = step("Listing Sentinel-2 acquisitions")
    days = gee.list_s2_days(grid, args.fire_start - timedelta(days=45), args.date)
    done(t)
    print(gee.describe(days))
    if args.list_s2:
        return

    by_day = {d.day: d for d in days}
    pre = by_day.get(args.pre_date) if args.pre_date else gee.pick_pre_fire(days, args.fire_start)
    post = by_day.get(args.post_date) if args.post_date else gee.pick_post(days, args.date)
    if pre is None:
        sys.exit("No usable pre-fire Sentinel-2 scene: give --pre-date explicitly.")
    print(f"Pre-fire scene: {pre.day} ({pre.usable:.0%} usable)")
    print(f"Post scene:     {post.day} ({post.usable:.0%} usable)" if post else "Post scene:     none (the fire layer will come from FIRMS only)")

    rasters: dict[str, Raster] = {}
    t = step(f"Sentinel-2 pre-fire {pre.day}")
    rasters[S2_PRE] = gee.s2_raster(grid, pre.day, args.fine_factor)
    done(t)
    if post:
        t = step(f"Sentinel-2 post {post.day}")
        rasters[S2_POST] = gee.s2_raster(grid, post.day, args.fine_factor)
        done(t, f"cloud mask {rasters[S2_POST].cloud_fraction:.0%}")
    t = step(f"ERA5-Land {args.date}")
    rasters[ERA5] = gee.era5_raster(grid, args.date)
    done(t)
    t = step("Copernicus DEM")
    rasters[DEM] = gee.dem_raster(grid)
    done(t)
    t = step("ESA WorldCover")
    rasters[WORLDCOVER] = gee.worldcover_raster(grid, args.fine_factor)
    done(t)

    out = resolve_path(args.out or f"data/scenes/{args.name}")
    t = step(f"FIRMS {args.date} and {args.date + timedelta(days=1)} (truth)")
    rows_t = fetch_firms_day(bbox, args.date, args.firms_sources)
    rows_next = fetch_firms_day(bbox, args.date + timedelta(days=1), args.firms_sources)
    write_hotspots_csv(rows_t, out / "firms_raw_t.csv")
    write_hotspots_csv(rows_next, out / "firms_raw_t+1.csv")
    done(t, f"{len(rows_t)} hotspots on day t, {len(rows_next)} on day t+1")
    if not rows_t:
        print("  WARNING: no FIRMS hotspot on the forecast day: is the fire still active?")

    firms = points_from_rows(rows_t, grid, "VIIRS")
    truth_points = points_from_rows(rows_next, grid, "VIIRS").filter(0.5)  # nominal + high confidence
    rasters[TRUTH] = Raster(
        {"fire_next": grid.rasterize_points(truth_points.x, truth_points.y)},
        grid.resolution, truth_points.acquisition_time, "FIRMS VIIRS (day t+1, nominal+high)",
    )

    scene = RawScene(name=args.name, grid=grid, rasters=rasters, firms=firms)
    save_raw_scene(scene, out, provenance=gee.provenance(
        bbox_lonlat=list(bbox), fire_start=args.fire_start, forecast_day=args.date,
        s2_pre=pre.day, s2_post=post.day if post else None, fine_factor=args.fine_factor,
        firms_sources=args.firms_sources,
        firms_per_satellite=dict(Counter(r.get("satellite", "?") for r in rows_t)),
        s2_days=[{"day": d.day.isoformat(), "coverage": round(d.coverage, 3), "cloud": round(d.cloud, 3)} for d in days],
    ))
    print(f"\nScene saved to {out}")
    print(f"Next: python scripts/run_pipeline.py --scene {out.relative_to(resolve_path('.'))}")


if __name__ == "__main__":
    main()
