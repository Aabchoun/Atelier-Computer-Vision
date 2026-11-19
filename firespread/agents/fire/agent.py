"""Fire agent: active fire from optical SWIR bands + FIRMS hotspots."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from firespread.common.agent import Agent, AgentInputError
from firespread.common.message import AgentMessage
from firespread.common.scene import S2_POST, Raster, RawScene, SceneDatacube


BAND_SUBSTITUTES = {"B8A": "B8", "B8": "B8A"}


class FireAgent(Agent):
    """Outputs up to three layers, fused into the single `fire` layer by the coordinator:

    fire_s2        - active-fire probability at native optical resolution, NaN under clouds
    fire_firms     - FIRMS hotspots acquired before / around the optical scene (up to
                     `firms_concurrent_hours` after it): the optical scene can confirm or reject them
    fire_firms_new - FIRMS hotspots acquired later than that: newer than the optical snapshot,
                     so it cannot reject them (the front has moved since)
    FIRMS values = detection confidence rasterised on the common grid (0 = no detection).

    Baseline (no training): Murphy et al. (2016) Landsat-8 test adapted to Sentinel-2
    (B12 ~ L8 B7, B11 ~ L8 B6, B8A/B8 ~ L8 B5):
        B12 >= swir2_min  and  B12 / B11 >= 1.4  and  B12 / NIR >= 1.4
    swir2_min is 0.25 on surface reflectance (the paper's 0.15 flags fresh char, see config).
    Trained model: U-Net on the Landsat-8 Active Fire dataset, fine-tuned on Sen2Fire
    (set agents.fire.checkpoint).
    """

    name = "fire"
    produces = ("fire_s2", "fire_firms", "fire_firms_new")

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        messages = []
        s2 = raw.get(S2_POST)
        if s2 is not None:
            prob = self.detect(s2)
            if s2.cloud_mask is not None:
                prob[s2.cloud_mask] = np.nan  # no observation under clouds
            messages.append(
                self.message(
                    raw.grid,
                    "fire_s2",
                    prob,
                    acquisition_time=s2.acquisition_time,
                    sensor=s2.sensor,
                    confidence=self.config.get("s2_confidence", 0.8) * (1.0 - s2.cloud_fraction),
                    cloud_fraction=s2.cloud_fraction,
                    extras={"model": "unet" if self.config.get("checkpoint") else "swir-threshold"},
                )
            )
        if raw.firms is not None and len(raw.firms):
            points = raw.firms.filter(self.config.get("firms_min_confidence", 0.5))
            groups = {"fire_firms": points}
            if s2 is not None and points.times is not None:
                cutoff = np.datetime64(s2.acquisition_time + timedelta(hours=self.config.get("firms_concurrent_hours", 3)), "s")
                late = points.times > cutoff
                groups = {"fire_firms": points.subset(~late), "fire_firms_new": points.subset(late)}
            for layer, group in groups.items():
                if len(group) == 0:
                    continue
                messages.append(
                    self.message(
                        raw.grid,
                        layer,
                        raw.grid.rasterize_points(group.x, group.y, group.confidence),
                        acquisition_time=group.acquisition_time,
                        sensor=f"FIRMS {raw.firms.sensor}",
                        confidence=self.config.get("firms_confidence", 0.6),
                        extras={"n_points": len(group), "n_rejected_low_confidence": len(raw.firms) - len(points)},
                    )
                )
        if not messages:
            raise AgentInputError("no optical scene and no FIRMS hotspot")
        return messages

    def detect(self, s2: Raster) -> np.ndarray:
        loaded = self.load_model()
        if loaded is not None:
            from firespread.common.nn import predict_proba

            model, meta = loaded
            # B8A (narrow NIR) is close to B8: substitute it when a scene does not provide it
            x = np.stack([s2[band] if band in s2 else s2[BAND_SUBSTITUTES[band]] for band in meta["inputs"]])
            return predict_proba(model, x, meta.get("mean"), meta.get("std"))

        nir = s2["B8A"] if "B8A" in s2 else s2["B8"]
        return swir_fire_test(nir, s2["B11"], s2["B12"], self.config.get("swir2_min", 0.25), self.config.get("ratio_min", 1.4))


def swir_fire_test(nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray, swir2_min: float = 0.25, ratio: float = 1.4) -> np.ndarray:
    """Baseline active-fire mask (0/1 float32) from surface reflectance; also used as the benchmark
    the trained fire U-Net must beat (scripts/train_fire.py)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        fire = (swir2 >= swir2_min) & (swir2 / swir1 >= ratio) & (swir2 / nir >= ratio)
    return fire.astype(np.float32)
