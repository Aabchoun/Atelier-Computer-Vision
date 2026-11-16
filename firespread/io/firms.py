"""FIRMS hotspots from CSV files.

Accepts both the FIRMS download format (latitude, longitude, confidence, acq_date, acq_time, ...)
and the project format written by `write_firms_csv` (x, y, confidence, acq_datetime, sensor).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from firespread.common.grid import Grid
from firespread.common.scene import FirePoints

# VIIRS confidence classes -> 0..1 (MODIS gives 0-100, divided by 100)
VIIRS_CONFIDENCE = {"l": 0.3, "low": 0.3, "n": 0.6, "nominal": 0.6, "h": 0.9, "high": 0.9}


def parse_confidence(value: str) -> float:
    text = str(value).strip().lower()
    if text in VIIRS_CONFIDENCE:
        return VIIRS_CONFIDENCE[text]
    number = float(text)
    return number / 100.0 if number > 1 else number


def _parse_time(row: dict[str, str]) -> datetime:
    if row.get("acq_datetime"):
        return datetime.fromisoformat(row["acq_datetime"])
    hhmm = str(row["acq_time"]).zfill(4)
    return datetime.strptime(f"{row['acq_date']} {hhmm}", "%Y-%m-%d %H%M")


def read_firms_csv(path: str | Path, grid: Grid, sensor: str | None = None) -> FirePoints:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return points_from_rows(rows, grid, sensor)


def points_from_rows(rows: list[dict[str, str]], grid: Grid, sensor: str | None = None) -> FirePoints:
    """FirePoints in the grid CRS from CSV rows (FIRMS download format or project x/y format)."""
    if not rows:
        empty = np.zeros(0)
        return FirePoints(empty, empty, empty, datetime.min, sensor or "VIIRS")
    if "x" in rows[0]:
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
    else:
        lon = np.array([float(r["longitude"]) for r in rows])
        lat = np.array([float(r["latitude"]) for r in rows])
        x, y = _project(lon, lat, grid.crs)
    confidence = np.array([parse_confidence(r["confidence"]) for r in rows])
    times = [_parse_time(r) for r in rows]
    sensor = sensor or rows[0].get("sensor") or rows[0].get("instrument") or "VIIRS"
    return FirePoints(x, y, confidence, max(times), sensor, np.array(times, dtype="datetime64[s]"))


def write_firms_csv(points: FirePoints, path: str | Path) -> None:
    if points.times is not None:
        times = [t.astype("datetime64[s]").item().isoformat() for t in points.times]
    else:
        times = [points.acquisition_time.isoformat()] * len(points)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "confidence", "acq_datetime", "sensor"])
        for x, y, c, t in zip(points.x, points.y, points.confidence, times):
            writer.writerow([f"{x:.2f}", f"{y:.2f}", f"{c:.3f}", t, points.sensor])


def _project(lon: np.ndarray, lat: np.ndarray, crs: str) -> tuple[np.ndarray, np.ndarray]:
    if crs.upper() in ("EPSG:4326", "WGS84"):
        return lon, lat
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError('Projecting FIRMS lat/lon needs pyproj: pip install -e ".[geo]"') from exc
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)
