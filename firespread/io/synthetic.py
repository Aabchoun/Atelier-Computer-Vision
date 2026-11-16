"""Synthetic scene so the whole pipeline runs before any real data is downloaded.

It mimics one area of interest with every input of the real system: Sentinel-2 pre/post-fire,
FIRMS hotspots, ERA5 wind, a DEM and ESA WorldCover land cover, plus the next-day active-fire
mask as ground truth. The physics is crude: use it to test the plumbing, never to report results.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from firespread.common.grid import Grid, resize
from firespread.common.scene import DEM, ERA5, S2_POST, S2_PRE, TRUTH, WORLDCOVER, FirePoints, Raster, RawScene

BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
# Surface reflectance per WorldCover class for BANDS
SPECTRA = {
    10: (0.03, 0.05, 0.03, 0.30, 0.15, 0.07),  # tree cover
    20: (0.05, 0.07, 0.07, 0.25, 0.22, 0.13),  # shrubland
    30: (0.06, 0.09, 0.08, 0.28, 0.25, 0.15),  # grassland
    40: (0.07, 0.10, 0.09, 0.32, 0.26, 0.16),  # cropland
    50: (0.12, 0.13, 0.14, 0.20, 0.25, 0.22),  # built-up
    60: (0.15, 0.18, 0.22, 0.27, 0.35, 0.30),  # bare
    80: (0.06, 0.05, 0.03, 0.02, 0.01, 0.005),  # water
}
BURNED = (0.04, 0.05, 0.06, 0.10, 0.20, 0.20)
ACTIVE_FIRE = (0.05, 0.06, 0.08, 0.15, 0.35, 0.60)
SMOKE_DELTA = (0.25, 0.15, 0.08, 0.03, 0.005, 0.0)  # bright in blue, transparent in SWIR
CLOUD = (0.45, 0.45, 0.45, 0.50, 0.40, 0.30)  # bright everywhere, including SWIR
NON_BURNABLE = (50, 60, 80)


def make_synthetic_scene(size: int = 96, fine_factor: int = 4, cloud_cover: float = 0.0, seed: int = 0, name: str = "synthetic") -> RawScene:
    """size: grid cells per side (375 m each); fine_factor: optical pixels per grid cell per side;
    cloud_cover: fraction of the post-fire scene covered by clouds, centred on the fire front."""
    rng = np.random.default_rng(seed)
    grid = Grid(crs="EPSG:32632", left=500_000.0, top=4_100_000.0, resolution=375.0, width=size, height=size)
    n = size * fine_factor
    v, u = (np.mgrid[0:n, 0:n] + 0.5) / n  # fractional row / column coordinates

    def smooth_noise(cells: int) -> np.ndarray:
        return resize(rng.random((cells, cells)), (n, n), "mean")

    # --- terrain and land cover
    elevation = (150 + 450 * _blob(v, u, 0.25, 0.55, 0.12) + 300 * _blob(v, u, 0.75, 0.25, 0.15) + 40 * smooth_noise(8)).astype(np.float32)
    noise = smooth_noise(6)
    landcover = np.full((n, n), 30, dtype=np.float32)
    landcover[noise > 0.45] = 20
    landcover[(noise > 0.62) | (elevation > 330)] = 10
    landcover[(v > 0.8) & (u < 0.45)] = 40
    landcover[elevation > 560] = 60
    landcover[_ellipse(v, u, 0.18, 0.15, 0.06, 0.10)] = 80  # lake
    landcover[_ellipse(v, u, 0.53, 0.56, 0.03, 0.03)] = 50  # village downwind of the fire
    burnable = ~np.isin(landcover, NON_BURNABLE)

    # --- fire: elliptical scar elongated downwind, active front on the downwind edge
    wind_u, wind_v = 6.0, -2.0  # m/s, blowing towards the east-south-east
    wind_dir = np.array([-wind_v, wind_u]) / np.hypot(wind_u, wind_v)  # (row, col) direction
    ignition = np.array([0.45, 0.30])

    def fire_ellipse(length: float, width: float) -> tuple[np.ndarray, np.ndarray]:
        centre = ignition + wind_dir * length
        dv, du = v - centre[0], u - centre[1]
        along = dv * wind_dir[0] + du * wind_dir[1]
        across = -dv * wind_dir[1] + du * wind_dir[0]
        return (along / length) ** 2 + (across / width) ** 2, along

    ell, along = fire_ellipse(0.10, 0.05)
    scar = (ell <= 1) & burnable
    front = scar & (ell > 0.7) & (along > 0.03)
    ell_next, along_next = fire_ellipse(0.12, 0.058)  # front ~4 cells further the next day
    fire_next = (ell_next <= 1) & (ell > 0.85) & (along_next > 0) & burnable

    # --- smoke plume from the front tip, widening downwind
    tip = ignition + wind_dir * 0.20
    dv, du = v - tip[0], u - tip[1]
    along_p = dv * wind_dir[0] + du * wind_dir[1]
    across_p = -dv * wind_dir[1] + du * wind_dir[0]
    downwind = np.clip(along_p, 0, None)
    smoke = np.where(along_p > -0.02, np.exp(-((across_p / (0.02 + 0.25 * downwind)) ** 2)) * np.exp(-downwind / 0.3), 0.0)

    # --- Sentinel-2 surface reflectance
    pre, post = _render(landcover), _render(landcover)
    for i, band in enumerate(BANDS):
        post[band][scar] = BURNED[i]
        post[band][front] = ACTIVE_FIRE[i]
        post[band] += smoke * SMOKE_DELTA[i]
    for scene in (pre, post):
        for band in BANDS:
            scene[band] = np.clip(scene[band] * (1 + 0.04 * rng.standard_normal((n, n))) + 0.003 * rng.standard_normal((n, n)), 0, 1).astype(np.float32)
    cloud_mask = np.zeros((n, n), dtype=bool)
    if cloud_cover > 0:
        distance = np.hypot(v - tip[0], u - tip[1]) * (0.8 + 0.4 * smooth_noise(5))
        cloud_mask = distance <= np.quantile(distance, min(cloud_cover, 1.0))
        for i, band in enumerate(BANDS):
            post[band][cloud_mask] = CLOUD[i]

    # --- FIRMS hotspots (375 m): ~10 % omission + two false alarms
    front_cells = resize(front.astype(np.float32), grid.shape, "max") > 0
    rows, cols = np.nonzero(front_cells)
    keep = rng.random(rows.size) < 0.9
    rows, cols = rows[keep], cols[keep]
    x = grid.left + (cols + 0.5) * grid.resolution + rng.uniform(-120, 120, rows.size)
    y = grid.top - (rows + 0.5) * grid.resolution + rng.uniform(-120, 120, rows.size)
    confidence = rng.choice([0.6, 0.9], size=rows.size, p=[0.4, 0.6])
    extent = size * grid.resolution
    false_alarms = [(0.53, 0.56, 0.3), (0.15, 0.85, 0.6)]  # hot roof in the village, nominal hotspot in a clear area
    x = np.r_[x, [grid.left + c * extent for _, c, _ in false_alarms]]
    y = np.r_[y, [grid.top - r * extent for r, _, _ in false_alarms]]
    confidence = np.r_[confidence, [c for _, _, c in false_alarms]]

    t_post = datetime(2023, 7, 22, 10, 10)
    t_firms = datetime(2023, 7, 22, 12, 30)
    fine_res = grid.resolution / fine_factor
    era5_cells = 3
    era5 = {
        "u10": wind_u + 0.4 * rng.standard_normal((era5_cells, era5_cells)),
        "v10": wind_v + 0.4 * rng.standard_normal((era5_cells, era5_cells)),
        "t2m": 309.0 + rng.standard_normal((era5_cells, era5_cells)),
        "d2m": 281.0 + 0.5 * rng.standard_normal((era5_cells, era5_cells)),
    }
    rasters = {
        S2_PRE: Raster(pre, fine_res, t_post - timedelta(days=10), "Sentinel-2 L2A (synthetic)", np.zeros((n, n), bool)),
        S2_POST: Raster(post, fine_res, t_post, "Sentinel-2 L2A (synthetic)", cloud_mask),
        ERA5: Raster({k: a.astype(np.float32) for k, a in era5.items()}, extent / era5_cells, datetime(2023, 7, 22, 12), "ERA5 (synthetic)"),
        DEM: Raster({"elevation": elevation}, fine_res, datetime(2021, 1, 1), "Copernicus DEM (synthetic)"),
        WORLDCOVER: Raster({"landcover": landcover}, fine_res, datetime(2021, 1, 1), "ESA WorldCover (synthetic)"),
        TRUTH: Raster({"fire_next": resize(fire_next.astype(np.float32), grid.shape, "max")}, grid.resolution, t_firms + timedelta(days=1), "synthetic truth"),
    }
    firms = FirePoints(x, y, confidence, t_firms, "VIIRS")
    return RawScene(name=name, grid=grid, rasters=rasters, firms=firms)


def _render(landcover: np.ndarray) -> dict[str, np.ndarray]:
    bands = {band: np.zeros(landcover.shape, dtype=np.float32) for band in BANDS}
    for code, reflectance in SPECTRA.items():
        mask = landcover == code
        for band, value in zip(BANDS, reflectance):
            bands[band][mask] = value
    return bands


def _blob(v, u, cv, cu, sigma):
    return np.exp(-((v - cv) ** 2 + (u - cu) ** 2) / (2 * sigma**2))


def _ellipse(v, u, cv, cu, rv, ru):
    return ((v - cv) / rv) ** 2 + ((u - cu) / ru) ** 2 <= 1
