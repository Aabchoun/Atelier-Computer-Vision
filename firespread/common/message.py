"""Common message format returned by every agent (project summary, section 3):

    { agent, layer, acquisition_time, sensor, CRS, bbox, resolution, data_ref, confidence, cloud_fraction }

Two fields are added for the coordinator:
  * ``stale``  - True when the coordinator fell back on an older layer (failed / cloudy agent);
  * ``extras`` - small scalar outputs, e.g. the smoke plume direction or the model name.

The array itself travels in memory in ``data`` and is referenced by ``data_ref`` once saved.
All times are UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y) in CRS units


@dataclass
class AgentMessage:
    agent: str
    layer: str
    acquisition_time: datetime
    sensor: str
    crs: str
    bbox: BBox
    resolution: float  # metres per pixel of `data`
    confidence: float  # 0..1, agent's global confidence in this layer
    cloud_fraction: float = 0.0  # 0..1, share of the source scene hidden by clouds
    data_ref: str | None = None
    stale: bool = False
    extras: dict[str, Any] = field(default_factory=dict)
    data: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly metadata (without the array)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "data":
                continue
            value = getattr(self, f.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any], data: np.ndarray | None = None) -> AgentMessage:
        d = dict(d)
        d["acquisition_time"] = datetime.fromisoformat(d["acquisition_time"])
        d["bbox"] = tuple(d["bbox"])
        return cls(**d, data=data)


def json_default(value: Any) -> Any:
    """`default=` hook for json.dumps (numpy scalars, arrays, datetimes, paths)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
