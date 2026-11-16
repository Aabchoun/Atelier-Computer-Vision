"""Local on-disk format of a scene (see data/README.md):

    data/scenes/<name>/
        scene.json     grid + one entry per raster (file, bands, resolution, time, sensor, ...)
        <raster>.npz   or <raster>.tif (GeoTIFF exported from Earth Engine, needs rasterio)
        firms.csv      FIRMS hotspots (FIRMS download format or x/y format)

A raster "file" may be a relative path to another folder (e.g. "../../shared/s2_2023-08-03.npz"):
scene sequences share their downloads this way.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from firespread.common.grid import Grid
from firespread.common.scene import Raster, RawScene
from firespread.io.firms import read_firms_csv, write_firms_csv

SCENE_FILE = "scene.json"
CLOUD_MASK_BAND = "cloud_mask"
REFLECTANCE_SCALE = 1e-4  # optical bands are stored as uint16 = reflectance * 10000 (like Sentinel-2 L2A)


def _is_optical(raster: Raster) -> bool:
    return all(band.startswith("B") and band[1:].rstrip("A").isdigit() for band in raster.bands)


def raster_to_npz(raster: Raster, path: str | Path) -> dict[str, Any]:
    """Write one raster as .npz and return its scene.json entry ("file" = the file name)."""
    path = Path(path)
    optical = _is_optical(raster)
    if optical:  # 4x smaller files; no-data pixels (0) are covered by the cloud mask
        arrays = {b: np.clip(np.rint(np.nan_to_num(a) / REFLECTANCE_SCALE), 0, 65535).astype(np.uint16) for b, a in raster.bands.items()}
    else:
        arrays = {band: np.asarray(a, np.float32) for band, a in raster.bands.items()}
    if raster.cloud_mask is not None:
        arrays[CLOUD_MASK_BAND] = raster.cloud_mask.astype(np.uint8)
    np.savez_compressed(path, **arrays)
    return {
        "file": path.name,
        "bands": list(arrays),
        "cloud_mask": CLOUD_MASK_BAND if raster.cloud_mask is not None else None,
        "resolution": raster.resolution,
        "acquisition_time": raster.acquisition_time.isoformat(),
        "sensor": raster.sensor,
        "scale": REFLECTANCE_SCALE if optical else 1.0,
    }


def write_scene_json(directory: str | Path, name: str, grid: Grid, rasters: dict[str, dict[str, Any]], firms_file: str | None = None,
                     firms_sensor: str = "VIIRS", provenance: dict[str, Any] | None = None) -> Path:
    meta: dict[str, Any] = {
        "name": name,
        "grid": grid.to_dict(),
        "rasters": rasters,
        "firms": {"file": firms_file, "sensor": firms_sensor} if firms_file else None,
    }
    if provenance:
        meta["provenance"] = provenance
    path = Path(directory) / SCENE_FILE
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def save_raw_scene(scene: RawScene, directory: str | Path, provenance: dict[str, Any] | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rasters = {key: raster_to_npz(raster, directory / f"{key}.npz") for key, raster in scene.rasters.items()}
    firms_file, firms_sensor = None, "VIIRS"
    if scene.firms is not None:
        write_firms_csv(scene.firms, directory / "firms.csv")
        firms_file, firms_sensor = "firms.csv", scene.firms.sensor
    write_scene_json(directory, scene.name, scene.grid, rasters, firms_file, firms_sensor, provenance)
    return directory


def load_raw_scene(directory: str | Path) -> RawScene:
    directory = Path(directory)
    meta = json.loads((directory / SCENE_FILE).read_text(encoding="utf-8"))
    grid = Grid.from_dict(meta["grid"])
    rasters = {key: _load_raster(directory, info, grid) for key, info in meta["rasters"].items()}
    firms = None
    if meta.get("firms"):
        firms = read_firms_csv(directory / meta["firms"]["file"], grid, meta["firms"].get("sensor"))
    return RawScene(name=meta.get("name", directory.name), grid=grid, rasters=rasters, firms=firms)


def load_scene_raster(directory: str | Path, key: str) -> Raster | None:
    """One raster of a scene without loading the others (e.g. only the ground truth)."""
    directory = Path(directory)
    meta = json.loads((directory / SCENE_FILE).read_text(encoding="utf-8"))
    info = meta["rasters"].get(key)
    return None if info is None else _load_raster(directory, info, Grid.from_dict(meta["grid"]))


def _load_raster(directory: Path, info: dict[str, Any], grid: Grid) -> Raster:
    path = directory / info["file"]
    names = list(info["bands"])  # all bands in file order, including the cloud mask if any
    if path.suffix == ".npz":
        with np.load(path) as npz:
            arrays = {name: npz[name].astype(np.float32) for name in names}
    elif path.suffix.lower() in (".tif", ".tiff"):
        arrays = _read_geotiff(path, names, grid)
    else:
        raise ValueError(f"unsupported raster format: {path.name}")
    mask_name = info.get("cloud_mask")
    cloud_mask = arrays.pop(mask_name) > 0 if mask_name else None
    scale = np.float32(info.get("scale", 1.0))  # e.g. 0.0001 for Sentinel-2 L2A
    return Raster(
        bands={name: a * scale for name, a in arrays.items()},
        resolution=float(info["resolution"]),
        acquisition_time=datetime.fromisoformat(info["acquisition_time"]),
        sensor=info["sensor"],
        cloud_mask=cloud_mask,
    )


def _read_geotiff(path: Path, band_names: list[str], grid: Grid) -> dict[str, np.ndarray]:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError('Reading GeoTIFF scenes needs rasterio: pip install -e ".[geo]"') from exc
    with rasterio.open(path) as src:
        if src.crs is not None and src.crs.to_string() != grid.crs:
            raise ValueError(f"{path.name} is in {src.crs}, the scene grid is {grid.crs}: export it in the scene CRS")
        tolerance = max(abs(src.res[0]), abs(src.res[1]))
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        if any(abs(a - b) > tolerance for a, b in zip(bounds, grid.bbox)):
            raise ValueError(f"{path.name} bounds {bounds} do not match the scene bbox {grid.bbox}")
        data = src.read().astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    if data.shape[0] != len(band_names):
        raise ValueError(f"{path.name} has {data.shape[0]} bands but scene.json lists {len(band_names)}")
    return dict(zip(band_names, data))
