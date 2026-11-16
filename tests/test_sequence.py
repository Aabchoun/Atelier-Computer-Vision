"""Multi-day behaviour: shared scene files, burned-area history, tolerance metrics, CA calibration."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from firespread.agents.simulation.calibration import calibrate, case_from_cube
from firespread.common.config import load_config
from firespread.common.metrics import tolerance_metrics
from firespread.coordinator import Coordinator
from firespread.io import load_raw_scene, make_synthetic_scene
from firespread.io.scene_io import raster_to_npz, write_scene_json


@pytest.fixture(scope="module")
def scene():
    return make_synthetic_scene(size=48, seed=0)


def config():
    cfg = load_config()
    cfg["coordinator"]["strict"] = True
    cfg["agents"]["simulation"]["n_runs"] = 5
    return cfg


def test_tolerance_metrics_accept_one_cell_offset():
    truth = np.zeros((10, 10))
    truth[5, 5] = 1
    pred = np.zeros((10, 10))
    pred[5, 6] = 1  # one cell off
    m = tolerance_metrics(pred, truth, radius=1)
    assert m["f1@1"] == 1.0
    assert tolerance_metrics(pred, truth, radius=0)["f1@0"] == 0.0


def test_scene_can_point_to_shared_files(tmp_path, scene):
    shared, day = tmp_path / "shared", tmp_path / "days" / "d1"
    shared.mkdir()
    day.mkdir(parents=True)
    info = raster_to_npz(scene.get("dem"), shared / "dem.npz")
    write_scene_json(day, "d1", scene.grid, {"dem": {**info, "file": "../../shared/dem.npz"}})
    loaded = load_raw_scene(day)
    np.testing.assert_allclose(loaded.get("dem")["elevation"], scene.get("dem")["elevation"])


def test_day_without_sentinel2_skips_agents_and_falls_back(scene):
    coordinator = Coordinator(config())
    coordinator.run(scene)
    next_day = replace(scene, rasters={k: v for k, v in scene.rasters.items() if k != "s2_post"},
                       firms=replace(scene.firms, acquisition_time=scene.firms.acquisition_time + timedelta(days=1)))
    result = coordinator.run(next_day)
    assert any("skipped" in e.message for e in result.report.by_level("info"))
    assert not result.report.by_level("warning") or all("skipped" not in e.message for e in result.report.by_level("warning"))
    assert result.cube.message("smoke").stale and result.cube.message("burned").stale
    assert not result.cube.has("fire_s2")  # an old optical fire snapshot is not reused for today's fire


def test_burned_area_is_completed_with_previous_fire(scene):
    coordinator = Coordinator(config())
    previous_fire = coordinator.run(scene).cube.get("fire") >= 0.5
    del coordinator.agents["burned_area"]
    later = replace(scene, firms=replace(scene.firms, acquisition_time=scene.firms.acquisition_time + timedelta(days=1)))
    cube = coordinator.run(later).cube
    assert previous_fire.any()
    assert np.all(cube.get("burned")[previous_fire] == 1)


def test_calibration_returns_the_best_tried_combination(scene):
    cube = Coordinator(config()).run(scene).cube
    truth = scene.grid.resample(scene.get("truth")["fire_next"], "max")
    case = case_from_cube(cube, truth)
    params, steps, best, tried = calibrate([case], {"p_h": [0.1, 0.58]}, [1, 4], n_runs=3)
    assert len(tried) == 4
    assert best["f1"] == max(t["f1"] for t in tried)
    assert params.p_h in (0.1, 0.58) and steps in (1, 4)
