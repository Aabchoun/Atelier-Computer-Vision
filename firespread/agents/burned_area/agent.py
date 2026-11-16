"""Burned-area agent: burn scar from pre/post-fire optical scenes (dNBR). Also the validation truth."""

from __future__ import annotations

import numpy as np

from firespread.common.agent import Agent, AgentInputError
from firespread.common.indices import nbr, ndvi
from firespread.common.message import AgentMessage
from firespread.common.scene import S2_POST, S2_PRE, RawScene, SceneDatacube


class BurnedAreaAgent(Agent):
    """Outputs: dnbr, burned (0/1 at native resolution -> burned fraction per grid cell).

    Baseline (no training): dNBR = NBR_pre - NBR_post with NBR = (B8 - B12) / (B8 + B12),
    burned where dNBR >= dnbr_threshold AND the pixel was vegetated before the fire
    (pre-fire NDVI >= min_pre_ndvi; dNBR is pure noise over water and bare soil).
    Trained model: U-Net on CaBuAr pre/post pairs, validated against EFFIS polygons
    (set agents.burned_area.checkpoint; inputs named "pre:B12", "post:B8", ...).
    """

    name = "burned_area"
    produces = ("dnbr", "burned")

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        pre, post = raw.get(S2_PRE), raw.get(S2_POST)
        if pre is None or post is None:
            raise AgentInputError("needs a pre-fire and a post-fire optical scene")
        if pre.shape != post.shape:
            raise AgentInputError(f"pre {pre.shape} and post {post.shape} scenes must have the same resolution")

        dnbr = nbr(pre.masked("B8"), pre.masked("B12")) - nbr(post.masked("B8"), post.masked("B12"))
        loaded = self.load_model()
        if loaded is not None:
            from firespread.common.nn import predict_proba

            model, meta = loaded
            scenes = {"pre": pre, "post": post}
            x = np.stack([scenes[name.split(":")[0]][name.split(":")[1]] for name in meta["inputs"]])
            burned = predict_proba(model, x, meta.get("mean"), meta.get("std"))
        else:
            vegetated = ndvi(pre.masked("B8"), pre.masked("B4")) >= self.config.get("min_pre_ndvi", 0.15)
            burned = ((dnbr >= self.config.get("dnbr_threshold", 0.1)) & vegetated).astype(np.float32)
        burned[np.isnan(dnbr)] = np.nan

        cloud = float(np.isnan(dnbr).mean())  # cloudy on either date
        common = dict(
            acquisition_time=post.acquisition_time,
            sensor=post.sensor,
            confidence=self.config.get("confidence", 0.8) * (1 - cloud),
            cloud_fraction=cloud,
            extras={"pre_time": pre.acquisition_time.isoformat()},
        )
        return [
            self.message(raw.grid, "dnbr", dnbr, **common),
            self.message(raw.grid, "burned", burned, **common),
        ]
