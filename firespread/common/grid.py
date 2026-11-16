"""Common working grid and resampling onto it.

Every raster of a scene covers exactly the scene bounding box (the area of interest); only the
pixel size differs (Sentinel-2 10-20 m, DEM 30 m, FIRMS 375 m, ERA5 ~25 km). Resampling is
therefore done by array shape. When exporting real data (phase 2), clip every layer to the same
bbox and CRS, ideally at an integer multiple of the grid resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

RESAMPLING_METHODS = ("mean", "max", "nearest")


@dataclass(frozen=True)
class Grid:
    """North-up grid. `left`/`top` = upper-left corner in CRS units (metres for UTM)."""

    crs: str
    left: float
    top: float
    resolution: float
    width: int
    height: int

    @classmethod
    def from_bbox(cls, crs: str, bbox: tuple[float, float, float, float], resolution: float) -> Grid:
        min_x, min_y, max_x, max_y = bbox
        return cls(
            crs=crs,
            left=float(min_x),
            top=float(max_y),
            resolution=float(resolution),
            width=int(round((max_x - min_x) / resolution)),
            height=int(round((max_y - min_y) / resolution)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.left,
            self.top - self.height * self.resolution,
            self.left + self.width * self.resolution,
            self.top,
        )

    @property
    def cell_area_ha(self) -> float:
        return self.resolution**2 / 10_000.0

    def pixel_size(self, shape: tuple[int, ...]) -> float:
        """Pixel size (m) of an array of `shape` covering the scene bbox."""
        return self.height * self.resolution / shape[0]

    def resample(self, array: np.ndarray, method: str = "mean") -> np.ndarray:
        """Bring an array covering the scene bbox onto this grid (see `resize`)."""
        return resize(array, self.shape, method)

    def rasterize_points(self, x, y, values=None) -> np.ndarray:
        """Burn points into the grid, keeping the maximum value per cell (0 where there is no point)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        values = np.ones_like(x) if values is None else np.asarray(values, dtype=float)
        out = np.zeros(self.shape, dtype=np.float32)
        cols = np.floor((x - self.left) / self.resolution).astype(int)
        rows = np.floor((self.top - y) / self.resolution).astype(int)
        inside = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        np.maximum.at(out, (rows[inside], cols[inside]), values[inside].astype(np.float32))
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Grid:
        return cls(**d)


def resize(array: np.ndarray, shape: tuple[int, int], method: str = "mean") -> np.ndarray:
    """Resample a 2-D array covering the same extent to `shape`. NaN = no observation.

    mean    - continuous layers: area mean when downsampling (NaN ignored), bilinear when upsampling
    max     - masks / detections: any positive pixel marks the coarse cell (nearest when upsampling)
    nearest - categorical layers (land cover classes, aspect)
    """
    if method not in RESAMPLING_METHODS:
        raise ValueError(f"unknown resampling method {method!r}, expected one of {RESAMPLING_METHODS}")
    a = np.asarray(array, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    (h, w), (H, W) = a.shape, shape
    if (h, w) == (H, W):
        return a.copy()
    if method == "nearest":
        return _nearest(a, H, W)
    if h >= H and w >= W:
        return _block_reduce(a, H, W, method)
    if h <= H and w <= W:
        return _bilinear(a, H, W) if method == "mean" else _nearest(a, H, W)
    return _nearest(a, H, W)


def _nearest(a: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = a.shape
    rows = np.minimum(((np.arange(H) + 0.5) * h / H).astype(int), h - 1)
    cols = np.minimum(((np.arange(W) + 0.5) * w / W).astype(int), w - 1)
    return a[np.ix_(rows, cols)]


def _block_reduce(a: np.ndarray, H: int, W: int, method: str) -> np.ndarray:
    h, w = a.shape
    row_starts = (np.arange(H) * h) // H
    col_starts = (np.arange(W) * w) // W
    if method == "max":  # fmax ignores NaN unless the whole block is NaN
        return np.fmax.reduceat(np.fmax.reduceat(a, row_starts, axis=0), col_starts, axis=1)
    valid = ~np.isnan(a)
    total = np.add.reduceat(np.add.reduceat(np.where(valid, a, 0.0), row_starts, axis=0), col_starts, axis=1)
    count = np.add.reduceat(np.add.reduceat(valid.astype(np.float32), row_starts, axis=0), col_starts, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1.0), np.nan).astype(np.float32)


def _bilinear(a: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = a.shape
    r = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0, h - 1)
    c = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0, w - 1)
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
    fr, fc = (r - r0)[:, None], (c - c0)[None, :]
    top = a[np.ix_(r0, c0)] * (1 - fc) + a[np.ix_(r0, c1)] * fc
    bottom = a[np.ix_(r1, c0)] * (1 - fc) + a[np.ix_(r1, c1)] * fc
    return (top * (1 - fr) + bottom * fr).astype(np.float32)
