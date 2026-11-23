"""Satellite image in -> active fire now + next-day spread forecast (see firespread/image_pipeline.py).

    # Sentinel-2 L2A export from Copernicus Browser (zip of one GeoTIFF per band): weather + terrain from Open-Meteo
    python scripts/analyse_image.py data/images/my_fire.zip

    # several days ahead, explicit date (when the file names do not contain it)
    python scripts/analyse_image.py data/images/my_fire.zip --date 2023-08-23 --days 3

    # not georeferenced (e.g. a Sen2Fire patch) or offline: give the weather by hand
    python scripts/analyse_image.py patch.npz --wind-speed 6 --wind-from 45 --t2m 34 --rh 20

Writes outputs/images/<name>_forecast.png and <name>_forecast.npz (all maps).
Uses models/fire_unet.pt and models/spread_unet.pt when they exist, else the baselines.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread import image_pipeline as ip  # noqa: E402
from firespread.common.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="GeoTIFF(s), a .zip of GeoTIFFs, or a Sen2Fire .npz")
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="acquisition date (default: from the file names)")
    parser.add_argument("--resolution", type=float, default=None, help="pixel size in m (default: from the GeoTIFF)")
    parser.add_argument("--days", type=int, default=1, help="forecast horizon in days")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--wind-speed", type=float, default=None, help="m/s (manual weather: give all four)")
    parser.add_argument("--wind-from", type=float, default=None, help="deg, direction the wind comes from (0 = north)")
    parser.add_argument("--t2m", type=float, default=None, help="daily max temperature, degC")
    parser.add_argument("--rh", type=float, default=None, help="daily min relative humidity, %%")
    parser.add_argument("--offline", action="store_true", help="no Open-Meteo requests")
    parser.add_argument("--fire-model", default="models/fire_unet.pt")
    parser.add_argument("--spread-model", default="models/spread_unet.pt")
    parser.add_argument("--out", default="outputs/images")
    args = parser.parse_args()

    image = ip.read_image(args.files)
    if args.date:
        image.date = args.date
    if args.resolution:
        image.resolution = args.resolution
    manual = [args.wind_speed, args.wind_from, args.t2m, args.rh]
    weather = ip.Weather(*manual) if all(v is not None for v in manual) else None
    cfg = load_config()["agents"]["fire"]

    print(f"Image {image.name}: {image.shape[0]} x {image.shape[1]} px at {image.resolution:.1f} m, bands {sorted(image.bands)}, "
          f"date {image.date}, centre {image.center_lonlat()}")
    result = ip.analyse(image, resolve_path(args.fire_model), resolve_path(args.spread_model), weather=weather,
                        threshold=args.threshold, days=args.days, online=not args.offline,
                        swir2_min=cfg["swir2_min"], ratio=cfg["ratio_min"])
    for key, value in result.summary().items():
        print(f"  {key}: {value}")
    for note in result.notes:
        print(f"  note: {note}")

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{image.name}_forecast.png"
    ip.figure(image, result).savefig(png, dpi=110)
    np.savez_compressed(out / f"{image.name}_forecast.npz", fire_prob=result.fire_prob,
                        spread=np.stack(result.spread), **{f"cells_{k}": v for k, v in result.layers.items()})
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
