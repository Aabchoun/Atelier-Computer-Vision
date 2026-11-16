"""Credentials and FIRMS parsing, without network access."""

import pytest

from firespread.common.credentials import MissingCredentialError, get_credential, read_env_file
from firespread.common.grid import Grid
from firespread.io.firms import read_firms_csv
from firespread.io.firms_api import FirmsError, parse_csv, write_hotspots_csv

FIRMS_CSV = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
36.95,8.75,340.1,0.4,0.37,2023-07-24,130,N,VIIRS,n,2.0NRT,290.2,5.1,N
36.96,8.76,367.0,0.4,0.37,2023-07-24,1242,N,VIIRS,h,2.0NRT,300.5,25.3,D
"""


def test_env_file_parsing(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nFIRMS_MAP_KEY="abc123"\n\nGEE_PROJECT=ee-team\nBROKEN LINE\n', encoding="utf-8")
    assert read_env_file(env) == {"FIRMS_MAP_KEY": "abc123", "GEE_PROJECT": "ee-team"}


def test_environment_variable_wins_and_missing_value_raises(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FIRMS_MAP_KEY=from_file\nGEE_PROJECT=\n", encoding="utf-8")
    monkeypatch.setenv("FIRMS_MAP_KEY", "from_env")
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    assert get_credential("FIRMS_MAP_KEY", env) == "from_env"
    with pytest.raises(MissingCredentialError):
        get_credential("GEE_PROJECT", env)


def test_firms_error_answer_never_shows_the_key():
    with pytest.raises(FirmsError) as info:
        parse_csv("Invalid MAP_KEY secret123.", key="secret123")
    assert "secret123" not in str(info.value)
    assert parse_csv("") == []


def test_downloaded_csv_is_readable_by_the_scene_loader(tmp_path):
    rows = parse_csv(FIRMS_CSV)
    path = write_hotspots_csv(rows, tmp_path / "firms.csv")
    grid = Grid(crs="EPSG:4326", left=8.5, top=37.1, resolution=0.01, width=50, height=50)
    points = read_firms_csv(path, grid)
    assert len(points) == 2
    assert points.confidence.tolist() == [0.6, 0.9]
    assert points.acquisition_time.hour == 12 and points.acquisition_time.minute == 42
