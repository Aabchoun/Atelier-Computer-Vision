"""Download FIRMS active-fire hotspots with the Area API (needs FIRMS_MAP_KEY in .env).

    https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[west,south,east,north]/[DAY_RANGE]/[DATE]

DAY_RANGE is 1..5 days starting at DATE, so longer periods are split into chunks.
Limit: 5000 transactions per 10 minutes per key.
*_SP  = standard processing (archive): use it for past fires (available a few months after acquisition).
*_NRT = near real time (recent weeks): use it for the live FIRMS-triggered flow.
"""

from __future__ import annotations

import csv
import urllib.error
import urllib.request
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

from firespread.common.credentials import get_credential

API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAY_RANGE = 5
SOURCES = (
    "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT",
    "MODIS_SP", "MODIS_NRT", "LANDSAT_NRT",
)
DEFAULT_COLUMNS = ("latitude", "longitude", "confidence", "acq_date", "acq_time")


class FirmsError(RuntimeError):
    pass


def fetch_hotspots(bbox: tuple[float, float, float, float], start: date, end: date, source: str = "VIIRS_SNPP_SP", map_key: str | None = None, timeout: float = 60.0) -> list[dict[str, str]]:
    """Hotspots inside bbox = (west, south, east, north) in degrees, from `start` to `end` inclusive."""
    if source not in SOURCES:
        raise ValueError(f"unknown FIRMS source {source!r}; valid: {', '.join(SOURCES)}")
    if end < start:
        raise ValueError("end date is before start date")
    key = map_key or get_credential("FIRMS_MAP_KEY")
    area = ",".join(str(v) for v in bbox)
    rows: list[dict[str, str]] = []
    seen: set[tuple] = set()
    day = start
    while day <= end:
        n_days = min(MAX_DAY_RANGE, (end - day).days + 1)
        text = _get(f"{API_URL}/{key}/{source}/{area}/{n_days}/{day.isoformat()}", key, timeout)
        for row in parse_csv(text, key):
            ident = tuple(row.get(c) for c in ("latitude", "longitude", "acq_date", "acq_time", "satellite"))
            if ident not in seen:
                seen.add(ident)
                rows.append(row)
        day += timedelta(days=n_days)
    return rows


def check_map_key(map_key: str | None = None, timeout: float = 30.0) -> int:
    """Raise FirmsError if FIRMS rejects the key; otherwise return the hotspots of a small test query."""
    key = map_key or get_credential("FIRMS_MAP_KEY")
    return len(parse_csv(_get(f"{API_URL}/{key}/VIIRS_SNPP_NRT/8,36,10,38/1", key, timeout), key))


def parse_csv(text: str, key: str = "") -> list[dict[str, str]]:
    text = text.strip()
    if not text:
        return []
    header = text.splitlines()[0].lower()
    if "latitude" not in header or "longitude" not in header:
        raise FirmsError(f"Unexpected answer from FIRMS: {_redact(text[:300], key)}")
    return list(csv.DictReader(StringIO(text)))


def write_hotspots_csv(rows: list[dict[str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else list(DEFAULT_COLUMNS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _get(url: str, key: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "firespread/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise FirmsError(f"FIRMS returned HTTP {exc.code}: {_redact(body, key)}") from None
    except urllib.error.URLError as exc:
        raise FirmsError(f"Cannot reach FIRMS: {_redact(str(exc.reason), key)}") from None


def _redact(text: str, key: str) -> str:
    """Never let the key leak into logs or error messages."""
    return text.replace(key, "***") if key else text
