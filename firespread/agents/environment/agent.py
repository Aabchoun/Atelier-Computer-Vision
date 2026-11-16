"""Environment agent: weather (ERA5) and terrain (DEM). No learning, only conversion and resampling."""

from __future__ import annotations

import numpy as np

from firespread.common.agent import Agent, AgentInputError
from firespread.common.indices import relative_humidity
from firespread.common.message import AgentMessage
from firespread.common.scene import DEM, ERA5, RawScene, SceneDatacube


class EnvironmentAgent(Agent):
    """Outputs (units): wind_u, wind_v (m/s, towards east / north), t2m (degC), rh (%),
    elevation (m), slope (deg), aspect (deg, azimuth of the downslope direction, NaN if flat).
    """

    name = "environment"
    produces = ("wind_u", "wind_v", "t2m", "rh", "elevation", "slope", "aspect")

    def run(self, raw: RawScene, cube: SceneDatacube) -> list[AgentMessage]:
        conf = self.config.get("confidence", 0.9)
        messages = []
        era5 = raw.get(ERA5)
        if era5 is not None:
            t2m_c = np.asarray(era5["t2m"], np.float32) - 273.15
            d2m_c = np.asarray(era5["d2m"], np.float32) - 273.15
            common = dict(acquisition_time=era5.acquisition_time, sensor=era5.sensor, confidence=conf)
            messages += [
                self.message(raw.grid, "wind_u", era5["u10"], **common),
                self.message(raw.grid, "wind_v", era5["v10"], **common),
                self.message(raw.grid, "t2m", t2m_c, **common),
                self.message(raw.grid, "rh", relative_humidity(t2m_c, d2m_c), **common),
            ]
        dem = raw.get(DEM)
        if dem is not None:
            elevation = np.asarray(dem["elevation"], np.float32)
            slope, aspect = slope_aspect(elevation, raw.grid.pixel_size(elevation.shape))
            common = dict(acquisition_time=dem.acquisition_time, sensor=dem.sensor, confidence=conf)
            messages += [
                self.message(raw.grid, "elevation", elevation, **common),
                self.message(raw.grid, "slope", slope, **common),
                self.message(raw.grid, "aspect", aspect, **common),
            ]
        if not messages:
            raise AgentInputError("needs ERA5 and/or a DEM")
        return messages


def slope_aspect(elevation: np.ndarray, pixel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (deg) and aspect (deg clockwise from north, direction the slope faces)."""
    d_row, d_col = np.gradient(elevation.astype(np.float64), pixel_size)
    dz_east, dz_north = d_col, -d_row  # rows increase southwards
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_north)))
    aspect = np.degrees(np.arctan2(-dz_east, -dz_north)) % 360.0
    aspect[slope < 0.1] = np.nan
    return slope.astype(np.float32), aspect.astype(np.float32)
