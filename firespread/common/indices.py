"""Spectral indices and meteorological conversions shared by several agents."""

from __future__ import annotations

import numpy as np


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where undefined."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Sentinel-2: NDVI = (B8 - B4) / (B8 + B4)."""
    return normalized_difference(nir, red)


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio. Sentinel-2: (B8 - B12) / (B8 + B12)."""
    return normalized_difference(nir, swir2)


def relative_humidity(temperature_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from air and dew-point temperature (degC), Magnus formula."""

    def saturation_vapour_pressure(t):
        return 6.112 * np.exp(17.62 * t / (243.12 + t))

    rh = 100.0 * saturation_vapour_pressure(np.asarray(dewpoint_c)) / saturation_vapour_pressure(np.asarray(temperature_c))
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


def wind_direction_deg(u: float | np.ndarray, v: float | np.ndarray):
    """Direction the wind blows TOWARDS, degrees clockwise from north (u = east, v = north)."""
    return np.degrees(np.arctan2(u, v)) % 360.0


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two directions, in [0, 180] degrees."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))
