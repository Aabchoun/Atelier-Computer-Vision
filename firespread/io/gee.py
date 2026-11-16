"""Build a real scene from Google Earth Engine on the common grid (phase 2: shared scene).

Needs GEE_PROJECT in .env and `earthengine authenticate` (once per machine).
Pixels are fetched with ee.data.computePixels directly as numpy arrays on the scene grid
(tiled to stay below the 48 MB request limit): no Google Drive export, no GeoTIFF handling.

Layers (all clipped to the same UTM bbox):
    s2_pre, s2_post  Sentinel-2 L2A (COPERNICUS/S2_SR_HARMONIZED), daily mosaic, cloud mask from SCL
    era5             ERA5-Land hourly, mean over the daytime burning period (land only: NaN over sea)
    dem              Copernicus DEM GLO-30
    worldcover       ESA WorldCover v200 (2021)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

from firespread.common.credentials import get_credential
from firespread.common.grid import Grid
from firespread.common.scene import Raster

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = ("B2", "B3", "B4", "B8", "B8A", "B11", "B12")
S2_SCALE = 1e-4
SCL_CLOUD_CLASSES = (3, 8, 9, 10)  # cloud shadow, cloud medium / high probability, thin cirrus
ERA5_LAND = "ECMWF/ERA5_LAND/HOURLY"
ERA5_BANDS = {"u10": "u_component_of_wind_10m", "v10": "v_component_of_wind_10m", "t2m": "temperature_2m", "d2m": "dewpoint_temperature_2m"}
DEM_COLLECTION = "COPERNICUS/DEM/GLO30_2024_1"
WORLDCOVER = "ESA/WorldCover/v200"
NODATA = -9999.0
MAX_TILE = 1024  # pixels per side and request: 8 float32 bands = 32 MB < 48 MB limit


def _ee():
    try:
        import ee
    except ImportError as exc:
        raise ImportError('Earth Engine needs: pip install -e ".[gee]"') from exc
    return ee


def initialize() -> None:
    _ee().Initialize(project=get_credential("GEE_PROJECT"))


# ------------------------------------------------------------------ grid
def utm_crs(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def make_grid(bbox_lonlat: tuple[float, float, float, float], resolution: float = 375.0, crs: str | None = None) -> Grid:
    """Common grid covering a lon/lat bbox, in the local UTM zone, snapped to the resolution."""
    from pyproj import Transformer

    west, south, east, north = bbox_lonlat
    crs = crs or utm_crs((west + east) / 2, (south + north) / 2)
    lons, lats = np.linspace(west, east, 21), np.linspace(south, north, 21)
    edge_lon = np.r_[lons, lons, np.full(21, west), np.full(21, east)]
    edge_lat = np.r_[np.full(21, south), np.full(21, north), lats, lats]
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(edge_lon, edge_lat)
    snap_down = lambda v: float(np.floor(v / resolution) * resolution)  # noqa: E731
    snap_up = lambda v: float(np.ceil(v / resolution) * resolution)  # noqa: E731
    return Grid.from_bbox(crs, (snap_down(np.min(x)), snap_down(np.min(y)), snap_up(np.max(x)), snap_up(np.max(y))), resolution)


def region(grid: Grid):
    return _ee().Geometry.Rectangle(list(grid.bbox), proj=grid.crs, geodesic=False)


def fetch_pixels(image, grid: Grid, factor: int = 1, dtype: str = "float32") -> dict[str, np.ndarray]:
    """Bands of `image` on the grid refined `factor` times (pixel = resolution / factor).

    float32: masked pixels -> NaN.  uint16: masked pixels -> 0 (half the transfer, for Sentinel-2).
    """
    ee = _ee()
    image = ee.Image(image)
    image = image.toFloat().unmask(NODATA, False) if dtype == "float32" else image.unmask(0, False).toUint16()
    bands = image.bandNames().getInfo()
    res = grid.resolution / factor
    width, height = grid.width * factor, grid.height * factor
    out = {b: np.zeros((height, width), dtype=np.float32 if dtype == "float32" else np.uint16) for b in bands}
    for row0 in range(0, height, MAX_TILE):
        for col0 in range(0, width, MAX_TILE):
            h, w = min(MAX_TILE, height - row0), min(MAX_TILE, width - col0)
            tile = ee.data.computePixels({
                "expression": image,
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {"width": w, "height": h},
                    "affineTransform": {
                        "scaleX": res, "shearX": 0, "translateX": grid.left + col0 * res,
                        "shearY": 0, "scaleY": -res, "translateY": grid.top - row0 * res,
                    },
                    "crsCode": grid.crs,
                },
            })
            for b in bands:
                out[b][row0:row0 + h, col0:col0 + w] = np.asarray(tile[b])
    if dtype == "float32":
        for values in out.values():
            values[values == NODATA] = np.nan
    return out


# ------------------------------------------------------------- Sentinel-2
@dataclass
class S2Day:
    day: date
    time: datetime  # UTC time of the first granule
    coverage: float  # share of the AOI observed that day
    cloud: float  # share of the observed AOI flagged cloud / shadow / cirrus by SCL

    @property
    def usable(self) -> float:
        """Share of the AOI actually visible (observed and not cloudy)."""
        return self.coverage * (1 - self.cloud)


def _s2_collection(grid: Grid, start: date, end: date):
    ee = _ee()
    return ee.ImageCollection(S2_COLLECTION).filterBounds(region(grid)).filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())


def list_s2_days(grid: Grid, start: date, end: date, scale: float = 300.0) -> list[S2Day]:
    """Every day with a Sentinel-2 acquisition over the AOI, with coverage and cloud fraction."""
    ee = _ee()
    col = _s2_collection(grid, start, end)
    times: dict[date, list[datetime]] = {}
    for t in col.aggregate_array("system:time_start").getInfo():
        dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc).replace(tzinfo=None)
        times.setdefault(dt.date(), []).append(dt)
    geom = region(grid)
    days = []
    for day in sorted(times):
        scl = _s2_collection(grid, day, day).select("SCL").mosaic().unmask(0, False)
        valid = scl.gt(0)
        cloud = scl.remap(list(SCL_CLOUD_CLASSES), [1] * len(SCL_CLOUD_CLASSES), 0).And(valid)
        stats = (
            ee.Image.cat(valid.rename("valid"), cloud.rename("cloud"))
            .reduceRegion(ee.Reducer.mean(), geom, scale, crs=grid.crs, maxPixels=1e9)
            .getInfo()
        )
        coverage = float(stats.get("valid") or 0.0)
        cloud_fraction = float(stats.get("cloud") or 0.0) / coverage if coverage > 0 else 1.0
        days.append(S2Day(day, min(times[day]), coverage, cloud_fraction))
    return days


def s2_raster(grid: Grid, day: date, factor: int = 16) -> Raster:
    """Daily Sentinel-2 mosaic. cloud_mask = SCL cloud classes OR not observed (outside swath)."""
    col = _s2_collection(grid, day, day)
    times = col.aggregate_array("system:time_start").getInfo()
    if not times:
        raise ValueError(f"no Sentinel-2 acquisition over the AOI on {day}")
    pixels = fetch_pixels(col.mosaic().select([*S2_BANDS, "SCL"]), grid, factor, dtype="uint16")
    scl = pixels.pop("SCL")
    cloud_mask = (scl == 0) | np.isin(scl, SCL_CLOUD_CLASSES)
    return Raster(
        bands={b: pixels[b].astype(np.float32) * np.float32(S2_SCALE) for b in S2_BANDS},
        resolution=grid.resolution / factor,
        acquisition_time=datetime.fromtimestamp(min(times) / 1000, tz=timezone.utc).replace(tzinfo=None),
        sensor="Sentinel-2 L2A",
        cloud_mask=cloud_mask,
    )


# ------------------------------------------------------- other layers
def era5_raster(grid: Grid, day: date, hours: tuple[int, int] = (9, 16)) -> Raster:
    """ERA5-Land mean over the daytime burning period (UTC hours), bilinear on the grid."""
    ee = _ee()
    start = datetime(day.year, day.month, day.day, hours[0])
    end = datetime(day.year, day.month, day.day, hours[1])
    col = (
        ee.ImageCollection(ERA5_LAND)
        .filterDate(start.isoformat(), end.isoformat())
        .select(list(ERA5_BANDS.values()), list(ERA5_BANDS))
        .map(lambda image: image.resample("bilinear"))
    )
    pixels = fetch_pixels(col.mean(), grid, factor=1)
    middle = start + (end - start) / 2
    return Raster(pixels, grid.resolution, middle, f"ERA5-Land hourly ({hours[0]:02d}-{hours[1]:02d} UTC mean)")


def dem_raster(grid: Grid, factor: int = 4) -> Raster:
    ee = _ee()
    dem = ee.ImageCollection(DEM_COLLECTION).select("DEM").map(lambda image: image.resample("bilinear")).mosaic().rename("elevation")
    return Raster(fetch_pixels(dem, grid, factor), grid.resolution / factor, datetime(2021, 1, 1), "Copernicus DEM GLO-30")


def worldcover_raster(grid: Grid, factor: int = 16) -> Raster:
    ee = _ee()
    image = ee.ImageCollection(WORLDCOVER).first().select("Map").rename("landcover")
    landcover = fetch_pixels(image, grid, factor, dtype="uint16")["landcover"].astype(np.float32)  # 0 = no data
    return Raster({"landcover": landcover}, grid.resolution / factor, datetime(2021, 1, 1), "ESA WorldCover v200")


# --------------------------------------------------------- selection
def pick_pre_fire(days: list[S2Day], fire_start: date, max_days_before: int = 45) -> S2Day | None:
    """Latest almost cloud-free, almost full-coverage scene before the fire started."""
    window = [d for d in days if fire_start - timedelta(days=max_days_before) <= d.day < fire_start and d.coverage >= 0.9]
    clear = [d for d in window if d.cloud <= 0.05]
    if clear:
        return max(clear, key=lambda d: d.day)
    return max(window, key=lambda d: d.usable) if window else None


def pick_post(days: list[S2Day], forecast_day: date, max_days_before: int = 3, min_usable: float = 0.5) -> S2Day | None:
    """Most recent scene in the few days up to the forecast day with enough visible area."""
    window = [d for d in days if forecast_day - timedelta(days=max_days_before) <= d.day <= forecast_day and d.usable >= min_usable]
    return max(window, key=lambda d: d.day) if window else None


def describe(days: list[S2Day]) -> str:
    lines = [" date        time   coverage  cloud  usable"]
    lines += [f" {d.day}  {d.time:%H:%M}  {d.coverage:>8.0%}  {d.cloud:>5.0%}  {d.usable:>6.0%}" for d in days]
    return "\n".join(lines)


def provenance(**items: Any) -> dict[str, Any]:
    return {k: (v.isoformat() if isinstance(v, (date, datetime)) else v) for k, v in items.items()}
