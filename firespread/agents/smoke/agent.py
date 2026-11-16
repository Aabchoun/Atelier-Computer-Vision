"""Smoke agent: smoke plume mask on Sentinel-2 and plume direction (a check on the wind layer)."""

from __future__ import annotations

import numpy as np

from firespread.common.agent import Agent, AgentInputError
from firespread.common.grid import Grid
from firespread.common.message import AgentMessage
from firespread.common.scene import S2_POST, Raster, RawScene, SceneDatacube


class SmokeAgent(Agent):
    """Output: `smoke` (0..1 score at native resolution, NaN under clouds).
    extras["plume_direction_deg"]: direction the plume points to (0 = north, clockwise).

    Baseline (no training): haze index, smoke is bright in blue but transparent in SWIR,
    whereas clouds are bright in SWIR too (classic smoke/cloud confusion):
        score = clip((B2 - k * B4 - offset) / scale, 0, 1), set to 0 where B11 > swir1_max
    Trained model: encoder pretrained on USTC_SmokeRS, U-Net fine-tuned on your own labelled
    Sentinel-2 plumes, with clouds as an explicit negative class (set agents.smoke.checkpoint).
    """

    name = "smoke"
    produces = ("smoke",)

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        s2 = raw.get(S2_POST)
        if s2 is None:
            raise AgentInputError("needs a post-fire optical scene")
        score = self.score(s2)
        if s2.cloud_mask is not None:
            score[s2.cloud_mask] = np.nan
        extras = {"model": "unet" if self.config.get("checkpoint") else "haze-index"}
        direction = plume_direction(score, cube.get("fire"), raw.grid, self.config.get("fire_threshold", 0.5))
        if direction is not None:
            extras["plume_direction_deg"] = round(direction, 1)
        return [
            self.message(
                raw.grid,
                "smoke",
                score,
                acquisition_time=s2.acquisition_time,
                sensor=s2.sensor,
                confidence=self.config.get("confidence", 0.4) * (1.0 - s2.cloud_fraction),
                cloud_fraction=s2.cloud_fraction,
                extras=extras,
            )
        ]

    def score(self, s2: Raster) -> np.ndarray:
        loaded = self.load_model()
        if loaded is not None:
            from firespread.common.nn import predict_proba

            model, meta = loaded
            return predict_proba(model, np.stack([s2[b] for b in meta["inputs"]]), meta.get("mean"), meta.get("std"))

        haze = s2["B2"] - self.config.get("haze_red_coef", 0.5) * s2["B4"] - self.config.get("haze_offset", 0.08)
        score = np.clip(haze / self.config.get("haze_scale", 0.05), 0.0, 1.0).astype(np.float32)
        score[s2["B11"] > self.config.get("swir1_max", 0.30)] = 0.0  # clouds and hot pixels are bright in SWIR
        return score


def plume_direction(smoke: np.ndarray, fire: np.ndarray | None, grid: Grid, fire_threshold: float = 0.5, min_pixels: int = 10) -> float | None:
    """Direction (deg, 0 = north, clockwise) from the fire centroid to the smoke centroid."""
    if fire is None:
        return None
    fire_mask = fire >= fire_threshold
    weights = np.nan_to_num(smoke)
    if not fire_mask.any() or (weights > 0.5).sum() < min_pixels:
        return None
    fire_row, fire_col = _centroid(fire_mask.astype(np.float64))
    smoke_row, smoke_col = _centroid(weights.astype(np.float64))
    east = (smoke_col - fire_col) * grid.width * grid.resolution
    north = -(smoke_row - fire_row) * grid.height * grid.resolution
    if np.hypot(east, north) < grid.resolution:
        return None
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def _centroid(weights: np.ndarray) -> tuple[float, float]:
    """Weighted centroid in fractional (row, col) coordinates of the bbox, so any resolution works."""
    h, w = weights.shape
    total = weights.sum()
    rows = (np.arange(h) + 0.5) / h
    cols = (np.arange(w) + 0.5) / w
    return float((weights.sum(axis=1) * rows).sum() / total), float((weights.sum(axis=0) * cols).sum() / total)
