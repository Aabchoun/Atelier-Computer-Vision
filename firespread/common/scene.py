"""Scene inputs (raw satellite data over one AOI) and the shared datacube (blackboard).

RawScene      - what the satellites give for one area of interest and time window, each raster at
                its native resolution (all covering the same bbox).
SceneDatacube - the shared blackboard: every agent output resampled onto the common grid, stored
                with the message that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from firespread.common.grid import Grid
from firespread.common.message import AgentMessage, json_default

# Standard names of the raw inputs of a scene. Loaders map sensor-specific names onto these.
S2_PRE = "s2_pre"  # optical scene before the fire (Sentinel-2, or Landsat renamed to S2 bands)
S2_POST = "s2_post"  # latest cloud-free-ish optical scene during / after the fire
ERA5 = "era5"  # bands: u10, v10 (m/s), t2m, d2m (K)
DEM = "dem"  # band: elevation (m)
WORLDCOVER = "worldcover"  # band: landcover (ESA WorldCover class codes)
TRUTH = "truth"  # evaluation only, e.g. band fire_next (active fire at t+1)

STATIC_INPUTS = (DEM, WORLDCOVER, TRUTH)


@dataclass
class Raster:
    """One acquisition covering the whole AOI.

    Optical bands use Sentinel-2 names (B2, B3, B4, B8, B8A, B11, B12) and surface reflectance in
    [0, 1]. Landsat loaders must rename bands to their S2 equivalents (L8 B7 -> B12, B6 -> B11,
    B5 -> B8, B4 -> B4, B3 -> B3, B2 -> B2) so agents stay sensor-agnostic.
    """

    bands: dict[str, np.ndarray]
    resolution: float
    acquisition_time: datetime
    sensor: str
    cloud_mask: np.ndarray | None = None  # True = cloud / cloud shadow

    def __getitem__(self, band: str) -> np.ndarray:
        return self.bands[band]

    def __contains__(self, band: str) -> bool:
        return band in self.bands

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.bands.values())).shape

    @property
    def cloud_fraction(self) -> float:
        return float(self.cloud_mask.mean()) if self.cloud_mask is not None else 0.0

    def masked(self, band: str) -> np.ndarray:
        """Band as float32 with NaN under clouds."""
        out = np.array(self.bands[band], dtype=np.float32)
        if self.cloud_mask is not None:
            out[self.cloud_mask] = np.nan
        return out


@dataclass
class FirePoints:
    """Active-fire hotspots (FIRMS VIIRS/MODIS), projected in the scene CRS."""

    x: np.ndarray
    y: np.ndarray
    confidence: np.ndarray  # 0..1
    acquisition_time: datetime  # latest overpass in the set
    sensor: str = "VIIRS"
    times: np.ndarray | None = None  # per-point UTC time (datetime64[s]); None = all at acquisition_time

    def __len__(self) -> int:
        return int(np.asarray(self.x).size)

    def subset(self, keep: np.ndarray) -> FirePoints:
        times = None if self.times is None else self.times[keep]
        latest = times.max().astype("datetime64[s]").item() if times is not None and times.size else self.acquisition_time
        return FirePoints(self.x[keep], self.y[keep], self.confidence[keep], latest, self.sensor, times)

    def filter(self, min_confidence: float) -> FirePoints:
        return self.subset(np.asarray(self.confidence) >= min_confidence)


@dataclass
class RawScene:
    name: str
    grid: Grid
    rasters: dict[str, Raster] = field(default_factory=dict)
    firms: FirePoints | None = None

    def get(self, key: str) -> Raster | None:
        return self.rasters.get(key)

    def latest_time(self) -> datetime:
        """Time of the most recent observation: the reference 'now' of a forecast."""
        times = [r.acquisition_time for k, r in self.rasters.items() if k not in STATIC_INPUTS]
        if self.firms is not None and len(self.firms):
            times.append(self.firms.acquisition_time)
        if not times:
            raise ValueError(f"scene {self.name!r} has no dated observation")
        return max(times)


class SceneDatacube:
    """Shared blackboard: aligned layers (same CRS, extent and resolution) + their messages."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self._layers: dict[str, AgentMessage] = {}

    def put(self, message: AgentMessage) -> None:
        if message.data is None or message.data.shape != self.grid.shape:
            shape = None if message.data is None else message.data.shape
            raise ValueError(f"layer {message.layer!r} has shape {shape}, grid is {self.grid.shape}")
        self._layers[message.layer] = message

    def get(self, layer: str, default: np.ndarray | None = None) -> np.ndarray | None:
        message = self._layers.get(layer)
        return default if message is None else message.data

    def message(self, layer: str) -> AgentMessage | None:
        return self._layers.get(layer)

    def has(self, layer: str) -> bool:
        return layer in self._layers

    __contains__ = has

    @property
    def names(self) -> list[str]:
        return list(self._layers)

    def stack(self, layers: list[str], fill: float = np.nan) -> tuple[np.ndarray, list[str]]:
        """(C, H, W) array of `layers`; missing layers are filled with `fill` and listed."""
        missing = [name for name in layers if name not in self._layers]
        arrays = [
            self._layers[name].data if name in self._layers else np.full(self.grid.shape, fill, np.float32)
            for name in layers
        ]
        return np.stack(arrays).astype(np.float32), missing

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "layers.npz", **{name: m.data for name, m in self._layers.items()})
        for name, message in self._layers.items():
            message.data_ref = f"layers.npz:{name}"
        meta = {"grid": self.grid.to_dict(), "layers": {n: m.to_dict() for n, m in self._layers.items()}}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=json_default), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> SceneDatacube:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        cube = cls(Grid.from_dict(meta["grid"]))
        with np.load(directory / "layers.npz") as npz:
            for name, info in meta["layers"].items():
                cube.put(AgentMessage.from_dict(info, data=npz[name]))
        return cube
