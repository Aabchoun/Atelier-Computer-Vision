"""One satellite image in -> where the fire is now + where it will be tomorrow.

    image  = read_image(["scene.zip"])              # Sentinel-2 L2A GeoTIFF(s), a .zip of them, or a Sen2Fire .npz
    result = analyse(image, fire_model="models/fire_unet.pt", spread_model="models/spread_unet.pt")

Steps (each one is an agent of the full system, run on the uploaded image only):
  1. fire       fire probability at the image resolution: trained fire U-Net, else the SWIR baseline
  2. 1 km grid  the spread model was trained on 1 km cells (Next Day Wildfire Spread), so the image is
                aggregated: fire = at least `min_fire_pixels` fire pixels in the cell (isolated pixels are
                mostly false alarms: bright roofs, greenhouses), NDVI = mean of the cell
  3. weather    wind, daily max temperature and min humidity (the variables of the training data) for the
                image date, from Open-Meteo (free, no key, needs internet), or given by hand
     terrain    elevation of every 1 km cell from the Open-Meteo elevation API (Copernicus DEM), slope from it;
                without it the model gets the training mean (flat terrain)
  4. spread     next-day fire probability: trained spread U-Net (several days = applied again on its own
                forecast), else persistence ("tomorrow = today")
  5. simulation physics-based forecast from the same 1 km layers: Alexandridis cellular automaton
                (Monte Carlo), fuel density from NDVI, non-burnable where NDVI < 0.15 (water, town, burn scar)

Images: Copernicus Browser (browser.dataspace.copernicus.eu) -> Sentinel-2 L2A -> Download -> Analytical,
TIFF 32-bit float, bands B02 B03 B04 B08 B8A B11 B12. One file per band (band name in the file name)
or one multi-band file are both read. Reflectance in [0, 1] or digital numbers (x 1e4) are both accepted.
"""

from __future__ import annotations

import io
import json
import re
import urllib.parse
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from firespread.agents.environment.agent import slope_aspect
from firespread.agents.fire.agent import BAND_SUBSTITUTES, swir_fire_test
from firespread.io.sen2fire import PIPELINE_BANDS, S2_ORDER

CELL_M = 1000.0  # spread model training resolution (Next Day Wildfire Spread)
BAND_RE = re.compile(r"(?<![A-Za-z0-9])B(8A|0?[1-9]|1[0-2])(?![0-9A-Za-z])", re.IGNORECASE)
DATE_RE = re.compile(r"(20\d\d)-?(\d\d)-?(\d\d)")
TIMEOUT_S = 20


# ---------------------------------------------------------------------------------------------
# 1. reading the image
# ---------------------------------------------------------------------------------------------
@dataclass
class SatImage:
    bands: dict[str, np.ndarray]  # Sentinel-2 name -> reflectance (H, W) in [0, 1]
    resolution: float  # metres per pixel
    name: str = "image"
    date: date | None = None
    transform: Any = None  # rasterio Affine of the image CRS (None = not georeferenced)
    crs: Any = None
    label: np.ndarray | None = None  # reference fire mask, when the file has one (Sen2Fire)
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.bands.values())).shape

    @property
    def georeferenced(self) -> bool:
        return self.transform is not None and self.crs is not None

    def band(self, name: str) -> np.ndarray:
        if name in self.bands:
            return self.bands[name]
        if BAND_SUBSTITUTES.get(name) in self.bands:
            return self.bands[BAND_SUBSTITUTES[name]]
        raise KeyError(f"band {name} missing (image has {sorted(self.bands)})")

    def has(self, *names: str) -> bool:
        return all(n in self.bands or BAND_SUBSTITUTES.get(n) in self.bands for n in names)

    def lonlat(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Longitude / latitude of pixel positions (fractional row / col, 0.5 = pixel centre)."""
        from rasterio.warp import transform as warp

        t, rows, cols = self.transform, np.asarray(rows, float), np.asarray(cols, float)
        xs, ys = t.c + t.a * cols + t.b * rows, t.f + t.d * cols + t.e * rows
        lon, lat = warp(self.crs, "EPSG:4326", np.ravel(xs).tolist(), np.ravel(ys).tolist())
        return np.reshape(lon, np.shape(rows)), np.reshape(lat, np.shape(rows))

    def center_lonlat(self) -> tuple[float, float] | None:
        if not self.georeferenced:
            return None
        h, w = self.shape
        lon, lat = self.lonlat(np.array([h / 2]), np.array([w / 2]))
        return float(lon[0]), float(lat[0])


def band_name(text: str) -> str | None:
    """'..._Sentinel-2_L2A_B8A_(Raw).tiff' -> 'B8A', 'B04' -> 'B4'."""
    found = BAND_RE.findall(Path(text).name)
    if not found:
        return None
    value = found[-1].upper()
    return "B8A" if value == "8A" else f"B{int(value)}"


def date_in(text: str) -> date | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    try:
        return date(*map(int, match.groups()))
    except ValueError:
        return None


def read_image(sources: list) -> SatImage:
    """Sources = paths or (file name, bytes) pairs: GeoTIFF(s), a .zip of GeoTIFFs, or a Sen2Fire .npz."""
    items = [(Path(s).name, Path(s).read_bytes()) if isinstance(s, (str, Path)) else (s[0], s[1]) for s in sources]
    files: list[tuple[str, bytes]] = []
    for name, data in items:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                files += [(n, z.read(n)) for n in z.namelist() if n.lower().endswith((".tif", ".tiff", ".npz"))]
        else:
            files.append((name, data))
    if not files:
        raise ValueError("no .tif / .tiff / .npz file found")
    image = _read_npz(*files[0]) if len(files) == 1 and files[0][0].lower().endswith(".npz") else _read_tiffs(files)
    if len(items) == 1:  # name the result after the uploaded file (zip / npz / tif), without the band part
        image.name = re.sub(r"[^\w.-]+", "_", Path(items[0][0]).stem).strip("_")
    return image


def _read_npz(name: str, data: bytes) -> SatImage:
    with np.load(io.BytesIO(data)) as npz:
        arrays = {k: npz[k] for k in npz.files}
    image = next((a for a in arrays.values() if a.ndim == 3), None)  # Sen2Fire: `image`, 12 or 13 bands
    label = next((a for k, a in arrays.items() if a.ndim == 2 and k.lower() in ("label", "labels", "mask", "gt", "y")), None)
    if image is None:
        raise ValueError(f"no 3-D image array among keys {list(arrays)}")
    image = np.asarray(image, np.float32)
    if image.shape[-1] < image.shape[0]:  # channels last -> channels first
        image = np.moveaxis(image, -1, 0)
    if image.shape[0] not in (len(PIPELINE_BANDS), 12, 13):
        raise ValueError(f"{image.shape[0]} bands: expected 7 ({list(PIPELINE_BANDS)}) or 12 (B1..B12 without B10)")
    order = S2_ORDER if image.shape[0] >= 12 else PIPELINE_BANDS
    bands = {b: image[i] for i, b in enumerate(order) if i < image.shape[0]}
    img = SatImage(_to_reflectance(bands), resolution=10.0, name=Path(name).stem, date=date_in(name))
    img.notes.append("Sen2Fire-style patch: not georeferenced, 10 m pixels assumed; give the weather by hand.")
    if label is not None and np.max(label) > 0:
        img.label = (np.asarray(label) > 0).astype(np.float32)
    return img


def _read_tiffs(files: list[tuple[str, bytes]]) -> SatImage:
    try:
        from rasterio.io import MemoryFile
    except ImportError as exc:
        raise ImportError("Reading GeoTIFFs needs rasterio: pip install rasterio") from exc

    bands: dict[str, np.ndarray] = {}
    transform = crs = None
    ref_shape = None
    for name, data in files:
        with MemoryFile(data) as mem, mem.open() as src:
            arrays = src.read().astype(np.float32)
            if src.nodata is not None:
                arrays[arrays == src.nodata] = np.nan
            if src.count == 1:
                band = band_name(name) or band_name(src.descriptions[0] or "")
                if band is None:
                    raise ValueError(f"cannot tell which Sentinel-2 band {name!r} is: put e.g. 'B12' in its file name")
                names = [band]
            else:
                names = [band_name(d or "") for d in src.descriptions]
                if None in names:  # no band descriptions: guess from the number of bands
                    guesses = {12: S2_ORDER, 13: S2_ORDER[:10] + ("B10",) + S2_ORDER[10:], len(PIPELINE_BANDS): PIPELINE_BANDS}
                    if src.count not in guesses:
                        raise ValueError(f"{name!r} has {src.count} unnamed bands: export one file per band or name the bands")
                    names = list(guesses[src.count])
            if ref_shape is None or src.height * src.width > ref_shape[0] * ref_shape[1]:
                ref_shape, transform, crs = (src.height, src.width), src.transform, src.crs
            for band, array in zip(names, arrays):
                bands[band] = array
    bands = {b: _resize_nearest(a, ref_shape) for b, a in bands.items()}  # 20 m bands -> 10 m grid
    resolution = _pixel_size_m(transform, crs, ref_shape)
    img = SatImage(_to_reflectance(bands), resolution, name=Path(files[0][0]).stem, transform=transform, crs=crs,
                   date=next((d for d in (date_in(n) for n, _ in files) if d), None))
    missing = [b for b in PIPELINE_BANDS if not img.has(b)]
    if missing:
        img.notes.append(f"missing bands {missing}: the fire U-Net needs {list(PIPELINE_BANDS)}, the SWIR baseline is used.")
    return img


def _to_reflectance(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sample = np.concatenate([a[np.isfinite(a)].ravel()[::97] for a in bands.values()])
    scale = 1e-4 if sample.size and np.median(sample) > 2.0 else 1.0  # digital numbers (x 1e4) -> reflectance
    return {b: np.asarray(a, np.float32) * scale for b, a in bands.items()}


def _resize_nearest(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if a.shape == tuple(shape):
        return a
    rows = np.minimum((np.arange(shape[0]) + 0.5) * a.shape[0] / shape[0], a.shape[0] - 1).astype(int)
    cols = np.minimum((np.arange(shape[1]) + 0.5) * a.shape[1] / shape[1], a.shape[1] - 1).astype(int)
    return a[np.ix_(rows, cols)]


def _pixel_size_m(transform, crs, shape) -> float:
    size_x, size_y = abs(transform.a), abs(transform.e)
    if crs is not None and crs.is_geographic:  # degrees (Copernicus Browser default: WGS84)
        lat = transform.f + transform.e * shape[0] / 2
        size_x *= 111_320 * np.cos(np.radians(lat))
        size_y *= 110_540
    return float((size_x + size_y) / 2)


# ---------------------------------------------------------------------------------------------
# 2. fire detection
# ---------------------------------------------------------------------------------------------
def detect_fire(image: SatImage, fire_model: str | Path | None = None, swir2_min: float = 0.25, ratio: float = 1.4) -> tuple[np.ndarray, str]:
    """Fire probability (H, W) and the method used."""
    if fire_model and Path(fire_model).exists() and image.has(*PIPELINE_BANDS):
        from firespread.common.nn import load_checkpoint, predict_proba

        model, meta = load_checkpoint(fire_model)
        x = np.stack([image.band(b) for b in meta["inputs"]])
        prob = predict_proba(model, x, meta.get("mean"), meta.get("std"))
        method = "fire U-Net"
    else:
        prob = swir_fire_test(image.band("B8A"), image.band("B11"), image.band("B12"), swir2_min, ratio)
        method = "SWIR baseline"
    invalid = ~np.isfinite(image.band("B12"))
    prob[invalid] = np.nan
    return prob.astype(np.float32), method


# ---------------------------------------------------------------------------------------------
# 3. 1 km grid, weather and terrain
# ---------------------------------------------------------------------------------------------
def cell_factor(resolution: float) -> int:
    return max(1, int(round(CELL_M / resolution)))


def block_reduce(a: np.ndarray, factor: int, how: str = "mean") -> np.ndarray:
    """(H, W) -> (ceil(H / f), ceil(W / f)); partial edge cells use the pixels they have."""
    h, w = a.shape
    H, W = -(-h // factor), -(-w // factor)
    padded = np.full((H * factor, W * factor), np.nan, np.float32)
    padded[:h, :w] = a
    blocks = padded.reshape(H, factor, W, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells stay NaN
        return (np.nanmax if how == "max" else np.nanmean)(blocks, axis=(1, 3)).astype(np.float32)


def upsample(a: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    return np.kron(a, np.ones((factor, factor), np.float32))[: shape[0], : shape[1]]


@dataclass
class Weather:
    wind_speed: float  # m/s, daily mean at 10 m
    wind_from: float  # deg, direction the wind comes FROM (meteorological)
    t2m: float  # degC, daily maximum
    rh: float  # %, daily minimum
    source: str = "manual"

    @property
    def uv(self) -> tuple[float, float]:
        d = np.radians(self.wind_from)
        return -self.wind_speed * np.sin(d), -self.wind_speed * np.cos(d)  # vector towards (east, north)


def _get_json(url: str, params: dict[str, Any]) -> dict:
    query = urllib.parse.urlencode(params, safe=",")
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "firespread-demo"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_weather(lon: float, lat: float, day: date) -> Weather:
    """Daily weather of `day` at one point, in the units of the spread model's training data (GRIDMET):
    mean wind speed and direction, max temperature, min relative humidity. ERA5 archive for old dates,
    forecast API for the last weeks (the archive lags by a few days)."""
    recent = day >= date.today() - timedelta(days=60)
    url = "https://api.open-meteo.com/v1/forecast" if recent else "https://archive-api.open-meteo.com/v1/archive"
    data = _get_json(url, {
        "latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": day.isoformat(), "end_date": day.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms", "timezone": "auto",
    })
    hourly = {k: np.array([np.nan if v is None else v for v in vals], float) for k, vals in data["hourly"].items() if k != "time"}
    speed, direction = hourly["wind_speed_10m"], np.radians(hourly["wind_direction_10m"])
    if not np.isfinite(speed).any():
        raise ValueError(f"Open-Meteo has no weather for {day}")
    # mean direction = direction of the mean "from" vector
    mean_from = np.degrees(np.arctan2(np.nanmean(speed * np.sin(direction)), np.nanmean(speed * np.cos(direction)))) % 360
    return Weather(float(np.nanmean(speed)), float(mean_from), float(np.nanmax(hourly["temperature_2m"])),
                   float(np.nanmin(hourly["relative_humidity_2m"])), source="Open-Meteo " + ("forecast" if recent else "ERA5 archive"))


def fetch_elevation(image: SatImage, factor: int, max_points: int = 2500) -> np.ndarray:
    """Elevation (m) of every 1 km cell centre (Open-Meteo elevation API, Copernicus DEM 90 m)."""
    h, w = image.shape
    H, W = -(-h // factor), -(-w // factor)
    stride = max(1, int(np.ceil(np.sqrt(H * W / max_points))))  # big images: query fewer cells, then repeat
    rows = (np.arange(0, H, stride) + 0.5 * stride) * factor
    cols = (np.arange(0, W, stride) + 0.5 * stride) * factor
    rr, cc = np.meshgrid(np.minimum(rows, h - 0.5), np.minimum(cols, w - 0.5), indexing="ij")
    lon, lat = image.lonlat(rr, cc)
    values: list[float] = []
    flat_lon, flat_lat = lon.ravel(), lat.ravel()
    for start in range(0, flat_lon.size, 100):  # the API takes up to 100 points per request
        data = _get_json("https://api.open-meteo.com/v1/elevation", {
            "latitude": ",".join(f"{v:.4f}" for v in flat_lat[start : start + 100]),
            "longitude": ",".join(f"{v:.4f}" for v in flat_lon[start : start + 100]),
        })
        values += [np.nan if v is None else v for v in data["elevation"]]
    elevation = np.array(values, np.float32).reshape(lon.shape)
    return upsample(elevation, stride, (H, W))


# ---------------------------------------------------------------------------------------------
# 4. the whole chain
# ---------------------------------------------------------------------------------------------
@dataclass
class Result:
    fire_prob: np.ndarray  # (H, W) image resolution
    fire_method: str
    factor: int  # image pixels per 1 km cell (per side)
    layers: dict[str, np.ndarray]  # 1 km inputs of the spread model
    spread: list[np.ndarray]  # next-day fire probability, one (h, w) map per forecast day
    spread_method: str
    weather: Weather
    threshold: float
    notes: list[str]
    simulation: np.ndarray | None = None  # P(cell burning during the last forecast day), cellular automaton

    @property
    def fire_cells(self) -> np.ndarray:
        return self.layers["fire"]

    def summary(self) -> dict[str, Any]:
        km2 = (CELL_M / 1000) ** 2
        today = int(np.nansum(self.fire_cells >= 0.5))
        out = {
            "fire detection": self.fire_method,
            "spread forecast": self.spread_method,
            "burning cells today (1 km²)": today,
            "weather": f"{self.weather.source}: wind {self.weather.wind_speed:.1f} m/s from {self.weather.wind_from:.0f}°, "
                       f"Tmax {self.weather.t2m:.1f} °C, RHmin {self.weather.rh:.0f} %",
        }
        for d, prob in enumerate(self.spread, start=1):
            mask = prob >= self.threshold
            out[f"day +{d}: cells on fire (km²)"] = round(float(mask.sum() * km2), 1)
            out[f"day +{d}: new cells (km²)"] = round(float((mask & ~(self.fire_cells >= 0.5)).sum() * km2), 1)
        if self.simulation is not None:
            out[f"simulation day +{len(self.spread)}: cells burning with P >= 0.5 (km²)"] = round(float((self.simulation >= 0.5).sum() * km2), 1)
        return out


def simulate_spread(layers: dict[str, np.ndarray], days: int, threshold: float = 0.5, config: dict[str, Any] | None = None) -> np.ndarray:
    """Cellular-automaton forecast (simulation agent) on the 1 km layers: P(burning during the last day)."""
    from firespread.agents.simulation.cellular_automaton import CAParams, initial_state, run_monte_carlo, spread_probabilities

    config = config or {}
    ndvi = np.nan_to_num(layers["ndvi"], nan=0.0)
    burnable = (ndvi >= 0.15).astype(np.float32)
    p_den = np.select([ndvi < 0.3, ndvi > 0.6], [-0.4, 0.3], 0.0).astype(np.float32)  # Alexandridis density classes
    elevation = layers["elevation"]
    elevation = np.where(np.isfinite(elevation), elevation, np.nanmean(elevation) if np.isfinite(elevation).any() else 0.0)
    state0 = initial_state(layers["fire"], np.zeros_like(ndvi), burnable, threshold)
    probs = spread_probabilities(np.zeros_like(ndvi), p_den, elevation, layers["wind_u"], layers["wind_v"], CELL_M, CAParams(**config.get("params", {})))
    return run_monte_carlo(state0, probs, days, int(config.get("steps_per_day", 4)), int(config.get("n_runs", 30)), int(config.get("seed", 0))).active_prob


def analyse(image: SatImage, fire_model: str | Path | None = None, spread_model: str | Path | None = None,
            weather: Weather | None = None, elevation: np.ndarray | None = None, fire_prob: np.ndarray | None = None,
            fire_method: str = "given", threshold: float = 0.5, days: int = 1, online: bool = True,
            swir2_min: float = 0.25, ratio: float = 1.4, simulate: bool = True, sim_config: dict[str, Any] | None = None,
            min_fire_pixels: int = 3) -> Result:
    notes = list(image.notes)
    if fire_prob is None:
        fire_prob, fire_method = detect_fire(image, fire_model, swir2_min, ratio)

    factor = cell_factor(image.resolution)
    fire_px = np.where(np.isnan(fire_prob), np.nan, (fire_prob >= threshold).astype(np.float32))
    n_fire = block_reduce(fire_px, factor, "mean") * factor**2  # fire pixels per cell (valid pixels only)
    fire = np.where(np.isnan(n_fire), np.nan, (n_fire >= min(min_fire_pixels, factor**2) - 1e-3).astype(np.float32))
    with np.errstate(divide="ignore", invalid="ignore"):
        nir, red = image.band("B8"), image.band("B4")
        ndvi = np.clip((nir - red) / (nir + red), -1, 1)
    ndvi = block_reduce(ndvi, factor, "mean")
    shape = fire.shape
    if min(shape) < 8:
        notes.append(f"the image covers only {shape[0]} x {shape[1]} km: the spread model was trained on 64 x 64 km tiles.")

    if weather is None:
        center = image.center_lonlat()
        if not (online and center and image.date):
            raise ValueError("weather is needed: give it by hand, or use a georeferenced image with a date and internet access")
        weather = fetch_weather(*center, image.date)
    if elevation is None and online and image.georeferenced:
        try:
            elevation = fetch_elevation(image, factor)
        except Exception as exc:  # noqa: BLE001 - terrain is optional
            notes.append(f"elevation download failed ({exc}): flat terrain assumed.")
    if elevation is None:
        elevation = np.full(shape, np.nan, np.float32)
        notes.append("no elevation: the spread model gets the training mean (flat terrain).")
    known = np.isfinite(elevation)
    slope = np.full(shape, np.nan, np.float32)
    if known.any():
        slope, _ = slope_aspect(np.where(known, elevation, np.nanmean(elevation)), CELL_M)
        slope[~known] = np.nan

    u, v = weather.uv
    layers = {
        "fire": fire,
        "wind_u": np.full(shape, u, np.float32),
        "wind_v": np.full(shape, v, np.float32),
        "t2m": np.full(shape, weather.t2m, np.float32),
        "rh": np.full(shape, weather.rh, np.float32),
        "elevation": elevation.astype(np.float32),
        "slope": slope.astype(np.float32),
        "ndvi": ndvi,
    }

    spread: list[np.ndarray] = []
    if spread_model and Path(spread_model).exists():
        from firespread.common.nn import load_checkpoint, predict_proba

        model, meta = load_checkpoint(spread_model)
        current = dict(layers)
        current["fire"] = np.nan_to_num(fire)
        for _ in range(days):
            prob = predict_proba(model, np.stack([current[n] for n in meta["inputs"]]), meta.get("mean"), meta.get("std"))
            spread.append(prob)
            current["fire"] = (prob >= threshold).astype(np.float32)
        spread_method = "spread U-Net"
    else:
        spread = [np.nan_to_num(fire)] * days
        spread_method = "persistence (no spread model)"
    simulation = simulate_spread(layers, days, threshold, sim_config) if simulate else None
    return Result(fire_prob, fire_method, factor, layers, spread, spread_method, weather, threshold, notes, simulation)


# ---------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------
def stretch(rgb: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return np.nan_to_num(np.clip((np.moveaxis(rgb, 0, -1) - lo) / max(hi - lo, 1e-6), 0, 1))


def figure(image: SatImage, result: Result, day: int = 1):
    """Panels: true colour, infrared, fire probability, 1 km inputs, forecast, simulation (when run)."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    shape = image.shape
    rgb = stretch(np.stack([image.band("B4"), image.band("B3"), image.band("B2")]))
    ir = stretch(np.stack([image.band("B12"), image.band("B8A"), image.band("B4")]))
    today = upsample(np.nan_to_num(result.fire_cells), result.factor, shape)
    forecasts = [(f"Day +{day}: {result.spread_method.split(' (')[0]}", result.spread[day - 1])]
    if result.simulation is not None:
        forecasts.append((f"Day +{len(result.spread)}: simulation (CA)", result.simulation))

    fig = Figure(figsize=(3.4 * (4 + len(forecasts)), 4))
    axes = fig.subplots(1, 4 + len(forecasts))
    axes[0].imshow(rgb)
    axes[0].set_title("True colour (B4 B3 B2)")
    axes[1].imshow(ir)
    axes[1].set_title("Infrared (B12 B8A B4)")
    axes[2].imshow(result.fire_prob, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"Fire now ({result.fire_method})")
    if image.label is not None:
        axes[2].contour(image.label, levels=[0.5], colors="cyan", linewidths=0.8)

    axes[3].imshow(result.layers["ndvi"], cmap="YlGn", vmin=0, vmax=0.9, interpolation="nearest")
    axes[3].contour(np.nan_to_num(result.fire_cells), levels=[0.5], colors="red", linewidths=1.0)
    h, w = result.fire_cells.shape
    axes[3].quiver([w / 2], [h / 2], [result.weather.uv[0]], [result.weather.uv[1]], color="black", scale=40, width=0.02)
    axes[3].set_title("1 km cells: NDVI, fire, wind")

    for ax, (title, prob) in zip(axes[4:], forecasts):
        big = upsample(prob, result.factor, shape)
        new = (big >= result.threshold) & (today < 0.5)
        ax.imshow(rgb)
        ax.imshow(np.ma.masked_where(big < 0.05, big), cmap="autumn_r", vmin=0, vmax=1, alpha=0.55)
        if today.max() > 0:
            ax.contour(today, levels=[0.5], colors="red", linewidths=1.2)
        if new.any():
            ax.contour(new.astype(float), levels=[0.5], colors="yellow", linewidths=1.2)
        ax.set_title(title)
    axes[-1].legend(handles=[Line2D([], [], color="red", label="fire now"), Line2D([], [], color="yellow", label="new tomorrow")],
                    loc="lower right", fontsize=8)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    return fig
