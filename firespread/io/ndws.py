"""Next Day Wildfire Spread (Huot et al. 2022, Kaggle `fantineh/next-day-wildfire-spread`).

64 x 64 tiles at 1 km over the US (2012-2020), stored as TFRecords with one float feature per
variable (4096 values each). This module converts the raw variables into OUR datacube layers, so a
model trained here takes exactly what the environment / vegetation / fire agents produce:

    fire        <- PrevFireMask (1 = fire, 0 = no fire, -1 = unknown -> NaN)
    wind_u/v    <- vs (wind speed, m/s) and th (direction the wind comes FROM, deg)  -> towards east / north
    t2m         <- tmmx (daily max, K) -> degC   (our ERA5 layer is the 09-16 UTC mean, i.e. afternoon in Europe)
    rh          <- sph (specific humidity) + tmmx + pressure from elevation -> %
    elevation   <- elevation (m)
    slope       <- computed from elevation at 1 km, same formula as the environment agent
    ndvi        <- NDVI (scaled by 1e4 in the dataset) -> -1..1
    target      <- FireMask (1 / 0 / -1 = ignored by the loss and the metrics)

Raw TFRecords are read with the pure-Python `tfrecord` package (no TensorFlow), then cached as one
.npz per split so later runs start in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from firespread.agents.environment.agent import slope_aspect

TILE = 64
PIXEL_SIZE_M = 1000.0
LAYERS = ("fire", "wind_u", "wind_v", "t2m", "rh", "elevation", "slope", "ndvi")
SPLITS = ("train", "eval", "test")
CACHE_VERSION = 2  # bump when convert_sample changes, so old caches are rebuilt

# physically plausible range of each raw variable; anything outside becomes NaN (= "not observed")
VALID_RANGE = {
    "elevation": (-500.0, 9000.0),  # m
    "vs": (0.0, 60.0),  # m/s
    "th": (0.0, 360.0),  # deg
    "tmmx": (213.15, 333.15),  # K (-60 .. +60 degC)
    "sph": (0.0, 0.05),  # kg/kg
    "NDVI": (-10000.0, 10000.0),  # NDVI * 1e4
}


def convert_sample(raw: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One raw NDWS record -> ({layer: (64, 64) float32}, target (64, 64) float32 in {1, 0, -1})."""

    def get(name: str) -> np.ndarray:
        value = np.asarray(raw[name], dtype=np.float32).reshape(TILE, TILE)
        if name in VALID_RANGE:  # NDWS contains fill / corrupt values (e.g. tmmx ~ 0 K): -> NaN
            low, high = VALID_RANGE[name]
            value = np.where((value >= low) & (value <= high), value, np.nan)
        return value

    prev = get("PrevFireMask")
    fire = np.where(prev < 0, np.nan, (prev > 0).astype(np.float32))

    speed, direction = get("vs"), np.radians(get("th"))
    wind_u = -speed * np.sin(direction)  # meteorological "from" direction -> vector towards
    wind_v = -speed * np.cos(direction)

    elevation = get("elevation")
    t_c = get("tmmx") - 273.15
    rh = relative_humidity_from_q(get("sph"), t_c, elevation)
    slope, _ = slope_aspect(elevation, PIXEL_SIZE_M)

    ndvi = get("NDVI")
    ndvi = ndvi / 1e4  # stored as NDVI * 1e4

    layers = {
        "fire": fire,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "t2m": t_c,
        "rh": rh,
        "elevation": elevation,
        "slope": slope,
        "ndvi": np.clip(ndvi, -1, 1),
    }
    target = get("FireMask")
    target = np.where(target < 0, -1.0, (target > 0).astype(np.float32)).astype(np.float32)
    return {k: v.astype(np.float32) for k, v in layers.items()}, target


def relative_humidity_from_q(q: np.ndarray, t_c: np.ndarray, elevation_m: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from specific humidity (kg/kg), temperature (degC) and elevation (m)."""
    p_hpa = 1013.25 * (1 - 2.25577e-5 * np.clip(elevation_m, -500, 9000)) ** 5.25588
    e = q * p_hpa / (0.622 + 0.378 * q)
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    return np.clip(100 * e / es, 0, 100).astype(np.float32)


def split_files(raw_dir: str | Path, split: str) -> list[Path]:
    files = sorted(p for p in Path(raw_dir).rglob("*.tfrecord") if f"_{split}_" in p.name)
    if not files:
        raise FileNotFoundError(f"no '*_{split}_*.tfrecord' under {raw_dir}")
    return files


def read_split(raw_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """All samples of a split -> X (N, len(LAYERS), 64, 64), Y (N, 64, 64)."""
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as exc:
        raise ImportError("Reading NDWS needs the `tfrecord` package: pip install tfrecord") from exc

    xs, ys = [], []
    for path in split_files(raw_dir, split):
        for raw in tfrecord_loader(str(path), None):
            layers, target = convert_sample(raw)
            xs.append(np.stack([layers[name] for name in LAYERS]))
            ys.append(target)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def load_split(raw_dir: str | Path, split: str, cache_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Like read_split, cached as <cache_dir>/<split>.npz (default: <raw_dir>/cache)."""
    cache = Path(cache_dir or Path(raw_dir) / "cache") / f"{split}.npz"
    if cache.exists():
        with np.load(cache) as npz:
            if tuple(npz["layers"]) == LAYERS and "version" in npz.files and int(npz["version"]) == CACHE_VERSION:
                return npz["x"], npz["y"]
    x, y = read_split(raw_dir, split)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y, layers=np.array(LAYERS), version=CACHE_VERSION)
    return x, y


def select_layers(x: np.ndarray, inputs: list[str]) -> np.ndarray:
    unknown = [name for name in inputs if name not in LAYERS]
    if unknown:
        raise ValueError(f"layers not available from NDWS: {unknown} (available: {LAYERS})")
    return x[:, [LAYERS.index(name) for name in inputs]]
