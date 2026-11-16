"""End-to-end tests of the multi-agent pipeline on the synthetic scene."""

import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from firespread.common.config import load_config
from firespread.common.message import json_default
from firespread.common.scene import TRUTH, FirePoints, SceneDatacube
from firespread.coordinator import Coordinator
from firespread.evaluation import evaluate_forecasts
from firespread.io import load_raw_scene, make_synthetic_scene, save_raw_scene

EXPECTED_LAYERS = (
    "fire_s2", "fire_firms", "fire", "smoke", "ndvi", "landcover", "burnable", "p_veg", "p_den", "dnbr", "burned",
    "wind_u", "wind_v", "t2m", "rh", "elevation", "slope", "aspect", "spread_prob", "sim_prob", "sim_reach_prob",
    "risk", "disagreement", "uncertain",
)


@pytest.fixture(scope="module")
def scene():
    return make_synthetic_scene(size=48, seed=0)


def config(strict=True):
    cfg = load_config()
    cfg["coordinator"]["strict"] = strict
    cfg["agents"]["simulation"]["n_runs"] = 5
    return cfg


def test_full_pipeline_produces_aligned_layers(scene):
    result = Coordinator(config()).run(scene)
    cube = result.cube
    for layer in EXPECTED_LAYERS:
        assert cube.get(layer).shape == scene.grid.shape, layer
    risk = cube.get("risk")
    assert np.all((risk >= 0) & (risk <= 1))
    assert (cube.get("fire") >= 0.5).sum() > 0
    assert not result.report.by_level("error")


def test_every_message_follows_the_common_format(scene):
    cube = Coordinator(config()).run(scene).cube
    required = {"agent", "layer", "acquisition_time", "sensor", "crs", "bbox", "resolution", "data_ref", "confidence", "cloud_fraction"}
    for name in cube.names:
        meta = cube.message(name).to_dict()
        assert required <= set(meta)
        assert meta["resolution"] == scene.grid.resolution
        json.dumps(meta, default=json_default)


def test_low_confidence_firms_false_alarm_is_dropped(scene):
    cube = Coordinator(config()).run(scene).cube
    assert cube.message("fire_firms").extras["n_rejected_low_confidence"] >= 1


def test_firms_after_the_optical_scene_is_not_rejected_by_it(scene):
    """A clear Sentinel-2 pixel without fire rejects a concurrent hotspot, not a later one."""
    grid, firms = scene.grid, scene.firms
    s2_time = scene.get("s2_post").acquisition_time
    row, col = 2, grid.width - 3  # clear corner, far from the fire
    x, y = grid.left + (col + 0.5) * grid.resolution, grid.top - (row + 0.5) * grid.resolution

    def fused_fire_with_hotspot_at(t):
        times = np.r_[np.full(len(firms), np.datetime64(firms.acquisition_time, "s")), np.datetime64(t, "s")]
        points = FirePoints(np.r_[firms.x, x], np.r_[firms.y, y], np.r_[firms.confidence, 0.9], max(firms.acquisition_time, t), "VIIRS", times)
        return Coordinator(config()).run(replace(scene, firms=points)).cube.get("fire")[row, col]

    assert fused_fire_with_hotspot_at(s2_time + timedelta(hours=1)) < 0.5
    assert fused_fire_with_hotspot_at(s2_time + timedelta(hours=10)) >= 0.5


def test_ablation_runs_without_an_agent(scene):
    cfg = config()
    cfg["agents"]["enabled"].remove("environment")
    result = Coordinator(cfg).run(scene)
    assert not result.cube.has("wind_u")
    assert "wind_u" in result.cube.message("sim_prob").extras["missing_inputs"]


def test_clouds_hide_s2_and_firms_still_gives_the_fire():
    cloudy = make_synthetic_scene(size=48, cloud_cover=0.3, seed=0)
    cube = Coordinator(config()).run(cloudy).cube
    assert np.isnan(cube.get("fire_s2")).any()
    assert (cube.get("fire") >= 0.5).sum() > 0


def test_fallback_on_last_known_layer_when_scene_is_overcast(scene):
    coordinator = Coordinator(config())
    coordinator.run(scene)  # clear day: layers are remembered
    overcast = make_synthetic_scene(size=48, cloud_cover=1.0, seed=0)
    result = coordinator.run(overcast)
    assert result.cube.message("fire_s2").stale
    assert any("last known" in e.message for e in result.report.by_level("warning"))


def test_failing_agent_does_not_stop_the_pipeline(scene):
    coordinator = Coordinator(config(strict=False))

    def broken(raw, cube):
        raise RuntimeError("simulated crash")

    coordinator.agents["smoke"].run = broken
    result = coordinator.run(scene)
    assert result.cube.has("risk")
    assert any("simulated crash" in e.message for e in result.report.by_level("warning"))


def test_scene_and_results_roundtrip(tmp_path, scene):
    save_raw_scene(scene, tmp_path / "scene")
    loaded = load_raw_scene(tmp_path / "scene")
    assert loaded.grid == scene.grid
    assert len(loaded.firms) == len(scene.firms)
    np.testing.assert_allclose(loaded.get("s2_post")["B12"], scene.get("s2_post")["B12"], atol=1e-4)  # uint16 storage
    np.testing.assert_allclose(loaded.get("era5")["u10"], scene.get("era5")["u10"])  # float storage

    result = Coordinator(config()).run(loaded)
    result.report.evaluation = evaluate_forecasts(result.cube, loaded.grid.resample(loaded.get(TRUTH)["fire_next"], "max"))
    out = result.save(tmp_path / "run")
    reloaded = SceneDatacube.load(out / "datacube")
    np.testing.assert_array_equal(reloaded.get("risk"), result.cube.get("risk"))
    assert (out / "report.md").exists() and (out / "quicklook.png").exists()
