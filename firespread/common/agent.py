"""Base class shared by every system agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

import numpy as np

from firespread.common.config import resolve_path
from firespread.common.grid import Grid
from firespread.common.message import AgentMessage
from firespread.common.scene import RawScene, SceneDatacube


class AgentInputError(RuntimeError):
    """The agent cannot run because an input is missing (no Sentinel-2 pass, no fire layer...)."""


class Agent(ABC):
    """A specialised model with a fixed input/output contract: run(raw, cube) -> messages.

    * Perception agents read the raw satellite inputs (`raw`). They may return arrays at any
      resolution covering the scene bbox: the coordinator resamples them onto the common grid.
    * Forecasting agents read the shared datacube (`cube`).

    Every agent has a baseline that needs no training. Setting `checkpoint` in its config section
    switches it to the trained model (see firespread/common/nn.py for the checkpoint format).
    """

    name: ClassVar[str] = "agent"
    produces: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self._loaded: tuple[Any, dict[str, Any]] | None = None

    @abstractmethod
    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        """Return one message per output layer. Raise AgentInputError if inputs are missing."""

    def message(
        self,
        grid: Grid,
        layer: str,
        data: np.ndarray,
        *,
        acquisition_time: datetime,
        sensor: str,
        confidence: float,
        cloud_fraction: float = 0.0,
        extras: dict[str, Any] | None = None,
    ) -> AgentMessage:
        data = np.asarray(data, dtype=np.float32)
        return AgentMessage(
            agent=self.name,
            layer=layer,
            acquisition_time=acquisition_time,
            sensor=sensor,
            crs=grid.crs,
            bbox=grid.bbox,
            resolution=grid.pixel_size(data.shape),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            cloud_fraction=float(cloud_fraction),
            extras=dict(extras or {}),
            data=data,
        )

    def load_model(self) -> tuple[Any, dict[str, Any]] | None:
        """(model, meta) from `config['checkpoint']`, or None when the agent uses its baseline."""
        checkpoint = self.config.get("checkpoint")
        if not checkpoint:
            return None
        if self._loaded is None:
            path = resolve_path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"{self.name}: checkpoint not found: {path}")
            from firespread.common.nn import load_checkpoint

            self._loaded = load_checkpoint(path)
        return self._loaded
