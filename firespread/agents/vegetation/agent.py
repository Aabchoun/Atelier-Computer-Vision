"""Vegetation / fuel agent: land cover, NDVI and the fuel parameters used by the simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from firespread.common.agent import Agent, AgentInputError
from firespread.common.indices import ndvi as compute_ndvi
from firespread.common.message import AgentMessage
from firespread.common.scene import S2_POST, S2_PRE, WORLDCOVER, RawScene, SceneDatacube

WORLDCOVER_CLASSES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare / sparse vegetation",
    70: "snow and ice",
    80: "permanent water",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}
BUILT_UP = 50

# class -> p_veg (Alexandridis et al. 2008 scale); None = non-burnable. Overridden by the config.
DEFAULT_P_VEG: dict[int, float | None] = {10: 0.4, 20: 0.0, 30: -0.1, 40: -0.3, 50: None, 60: None, 70: None, 80: None, 90: -0.3, 95: None, 100: -0.3}


class VegetationAgent(Agent):
    """Outputs: ndvi, landcover (WorldCover codes), burnable (0/1), p_veg, p_den.

    Fuel describes the state BEFORE the fire, so the pre-fire scene is used when available.
    Baseline (no training): land cover from ESA WorldCover if present, otherwise from NDVI
    thresholds; p_veg from a class lookup table, p_den from NDVI.
    Trained model: replace `landcover_from_ndvi` with a classifier trained on EuroSAT.
    """

    name = "vegetation"
    produces = ("ndvi", "landcover", "burnable", "p_veg", "p_den")

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        s2 = raw.get(S2_PRE)
        if s2 is None:
            s2 = raw.get(S2_POST)
        if s2 is None:
            raise AgentInputError("needs an optical scene")
        conf = self.config.get("confidence", 0.7)
        cloud = s2.cloud_fraction

        ndvi = compute_ndvi(s2.masked("B8"), s2.masked("B4"))
        p_den = self.density_from_ndvi(ndvi)

        worldcover = raw.get(WORLDCOVER)
        if worldcover is not None:
            landcover = np.asarray(worldcover["landcover"], dtype=np.float32)
            lc_time, lc_sensor, lc_conf, lc_source = worldcover.acquisition_time, worldcover.sensor, conf, "worldcover"
        else:
            landcover = landcover_from_ndvi(ndvi)
            lc_time, lc_sensor, lc_conf, lc_source = s2.acquisition_time, s2.sensor, 0.6 * conf * (1 - cloud), "ndvi-thresholds"

        p_veg, burnable = fuel_from_landcover(landcover, self.config.get("p_veg", DEFAULT_P_VEG))

        optical = dict(acquisition_time=s2.acquisition_time, sensor=s2.sensor, confidence=conf * (1 - cloud), cloud_fraction=cloud)
        cover = dict(acquisition_time=lc_time, sensor=lc_sensor, confidence=lc_conf, extras={"source": lc_source})
        grid = raw.grid
        return [
            self.message(grid, "ndvi", ndvi, **optical),
            self.message(grid, "p_den", p_den, **optical),
            self.message(grid, "landcover", landcover, **cover),
            self.message(grid, "burnable", burnable, **cover),
            self.message(grid, "p_veg", p_veg, **cover),
        ]

    def density_from_ndvi(self, ndvi: np.ndarray) -> np.ndarray:
        values: dict[str, Any] = self.config.get("p_den", {"sparse": -0.4, "normal": 0.0, "dense": 0.3})
        p_den = np.select(
            [ndvi < self.config.get("ndvi_sparse", 0.3), ndvi > self.config.get("ndvi_dense", 0.6)],
            [values["sparse"], values["dense"]],
            default=values["normal"],
        ).astype(np.float32)
        p_den[np.isnan(ndvi)] = np.nan
        return p_den


def landcover_from_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """Crude WorldCover-like classes from NDVI (0 = unknown, e.g. under clouds)."""
    classes = np.select(
        [ndvi < 0.0, ndvi < 0.15, ndvi < 0.35, ndvi < 0.55, ndvi >= 0.55],
        [80, 60, 30, 20, 10],
        default=0,
    )
    return classes.astype(np.float32)


def fuel_from_landcover(landcover: np.ndarray, table: dict[int, float | None]) -> tuple[np.ndarray, np.ndarray]:
    """(p_veg, burnable) from land-cover codes. Unknown classes: burnable with p_veg = 0."""
    classes = np.rint(np.nan_to_num(landcover)).astype(int)
    p_veg = np.zeros(classes.shape, dtype=np.float32)
    burnable = np.ones(classes.shape, dtype=np.float32)
    for code, value in table.items():
        mask = classes == int(code)
        if value is None:
            burnable[mask] = 0.0
        else:
            p_veg[mask] = value
    return p_veg, burnable
