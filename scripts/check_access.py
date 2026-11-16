"""Check that your personal access works: FIRMS MAP_KEY and Google Earth Engine.

    python scripts/check_access.py

Needs in .env (project root): FIRMS_MAP_KEY=... and GEE_PROJECT=...
and, once per machine, Earth Engine authentication: earthengine authenticate
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.credentials import MissingCredentialError, get_credential  # noqa: E402
from firespread.io.firms_api import FirmsError, check_map_key  # noqa: E402


def check_firms() -> bool:
    try:
        n = check_map_key()
    except (MissingCredentialError, FirmsError) as exc:
        print(f"[FAIL] FIRMS: {exc}")
        return False
    print(f"[ OK ] FIRMS: key accepted ({n} VIIRS hotspots over northern Tunisia in the last 24 h)")
    return True


def check_gee() -> bool:
    try:
        project = get_credential("GEE_PROJECT")
    except MissingCredentialError as exc:
        print(f"[FAIL] Earth Engine: {exc}")
        return False
    try:
        import ee
    except ImportError:
        print('[FAIL] Earth Engine: package missing, run: pip install -e ".[gee]"')
        return False
    try:
        ee.Initialize(project=project)
        n = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ee.Geometry.Point(9.0, 36.9))
            .filterDate("2023-07-01", "2023-08-01")
            .size()
            .getInfo()
        )
    except Exception as exc:  # ee raises various exception types for auth / project problems
        print(f"[FAIL] Earth Engine: {exc}")
        print("       -> run 'earthengine authenticate' once, and check that GEE_PROJECT is a Cloud project")
        print("          registered for Earth Engine (non-commercial) in https://code.earthengine.google.com")
        return False
    print(f"[ OK ] Earth Engine: project '{project}' works ({n} Sentinel-2 scenes over Tabarka in July 2023)")
    return True


if __name__ == "__main__":
    results = [check_firms(), check_gee()]
    sys.exit(0 if all(results) else 1)
