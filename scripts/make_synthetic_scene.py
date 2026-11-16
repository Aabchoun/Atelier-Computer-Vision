"""Create a synthetic scene in data/scenes/ so the pipeline can run without real data.

    python scripts/make_synthetic_scene.py
    python scripts/make_synthetic_scene.py --cloud-cover 0.4 --out data/scenes/synthetic_cloudy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.config import resolve_path  # noqa: E402
from firespread.io import make_synthetic_scene, save_raw_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/scenes/synthetic", help="output scene directory")
    parser.add_argument("--size", type=int, default=96, help="grid cells per side (375 m each)")
    parser.add_argument("--cloud-cover", type=float, default=0.0, help="fraction of the post-fire scene under clouds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = resolve_path(args.out)
    scene = make_synthetic_scene(size=args.size, cloud_cover=args.cloud_cover, seed=args.seed, name=out.name)
    save_raw_scene(scene, out)
    print(f"Scene '{scene.name}' saved to {out}")
    print(f"  grid: {scene.grid.width} x {scene.grid.height} cells at {scene.grid.resolution:.0f} m, {scene.grid.crs}")
    for key, raster in scene.rasters.items():
        print(f"  {key:<10} {raster.shape[0]:>4} x {raster.shape[1]:<4} at {raster.resolution:>7.1f} m  cloud {raster.cloud_fraction:.0%}")
    print(f"  firms      {len(scene.firms)} hotspots")


if __name__ == "__main__":
    main()
