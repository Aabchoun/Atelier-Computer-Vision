"""Earth Engine helpers that need no network (grid construction, scene selection)."""

from datetime import date, datetime

import pytest

pytest.importorskip("pyproj")

from firespread.io.gee import S2Day, make_grid, pick_post, pick_pre_fire, utm_crs  # noqa: E402


def day(d, coverage=1.0, cloud=0.0):
    return S2Day(d, datetime(d.year, d.month, d.day, 9, 19), coverage, cloud)


def test_utm_zone():
    assert utm_crs(26.0, 41.0) == "EPSG:32635"  # Evros
    assert utm_crs(9.0, 36.9) == "EPSG:32632"  # Tabarka


def test_grid_covers_bbox_and_is_snapped():
    grid = make_grid((25.65, 40.74, 26.50, 41.32), 375.0)
    assert grid.crs == "EPSG:32635"
    assert grid.left % 375 == 0 and grid.top % 375 == 0
    assert 180 < grid.width < 200 and 170 < grid.height < 185  # ~72 x 66 km


def test_scene_selection():
    days = [day(date(2023, 8, 3), cloud=0.03), day(date(2023, 8, 8), cloud=0.08), day(date(2023, 8, 18), cloud=0.29),
            day(date(2023, 8, 20), coverage=0.64, cloud=0.12), day(date(2023, 8, 23), cloud=0.08)]
    assert pick_pre_fire(days, date(2023, 8, 19)).day == date(2023, 8, 3)  # latest almost cloud-free
    assert pick_post(days, date(2023, 8, 23)).day == date(2023, 8, 23)
    assert pick_post(days, date(2023, 8, 22)).day == date(2023, 8, 20)
    assert pick_post(days, date(2023, 8, 27)) is None  # nothing in the 3 days before
