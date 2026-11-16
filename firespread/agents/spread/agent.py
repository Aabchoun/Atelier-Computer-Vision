"""Spread prediction agent: active-fire mask at t + horizon from the shared datacube."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from firespread.common.agent import Agent, AgentInputError
from firespread.common.message import AgentMessage
from firespread.common.scene import RawScene, SceneDatacube


class SpreadAgent(Agent):
    """Output: `spread_prob` = P(cell has active fire on day t + horizon).

    model: persistence - baseline "tomorrow = today". Always report it: many models barely beat it.
           unet        - checkpoint trained on Next Day Wildfire Spread / WildfireSpreadTS.
                         Multi-day horizons are predicted autoregressively.

    The checkpoint stores its ordered list of input layers. The model can only run if the datacube
    provides them, so train on features YOUR agents produce (fire, wind_u, wind_v, t2m, rh,
    elevation, slope, ndvi, p_veg...), not on every variable of the training dataset.
    Missing layers are filled with the training mean and reported.
    """

    name = "spread"
    produces = ("spread_prob",)

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        fire_msg = cube.message("fire")
        if fire_msg is None:
            raise AgentInputError("no fused fire layer in the datacube")
        threshold = self.config.get("fire_threshold", 0.5)
        horizon = int(self.config.get("horizon_days", 1))
        model_name = self.config.get("model", "persistence")

        if model_name == "persistence":
            prob = (fire_msg.data >= threshold).astype(np.float32)
            missing: list[str] = []
        else:
            loaded = self.load_model()
            if loaded is None:
                raise ValueError(f"spread model {model_name!r} needs agents.spread.checkpoint")
            from firespread.common.nn import predict_proba

            model, meta = loaded
            inputs = list(meta["inputs"])
            x, missing = cube.stack(inputs, fill=np.nan)
            fire_index = inputs.index("fire") if "fire" in inputs else None
            for _ in range(horizon):
                prob = predict_proba(model, x, meta.get("mean"), meta.get("std"))
                if fire_index is not None:
                    x[fire_index] = (prob >= threshold).astype(np.float32)
            model_name = Path(str(self.config["checkpoint"])).stem

        return [
            self.message(
                cube.grid,
                "spread_prob",
                prob,
                acquisition_time=fire_msg.acquisition_time,
                sensor="datacube",
                confidence=self.config.get("confidence", 0.5),
                extras={"model": model_name, "horizon_days": horizon, "missing_inputs": missing},
            )
        ]
