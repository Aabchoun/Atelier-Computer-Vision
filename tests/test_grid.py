import numpy as np
import pytest

from firespread.common.grid import Grid


def make_grid(n=4):
    return Grid(crs="EPSG:32632", left=0.0, top=n * 100.0, resolution=100.0, width=n, height=n)


def test_block_mean_and_max():
    grid = make_grid()
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert grid.resample(a, "mean")[0, 0] == np.mean([0, 1, 8, 9])
    assert grid.resample(a, "max")[0, 0] == 9


def test_nan_is_ignored_when_downsampling():
    grid = make_grid()
    a = np.ones((8, 8), np.float32)
    a[0, 0] = np.nan
    a[2:4, 2:4] = np.nan  # a whole block unobserved
    mean = grid.resample(a, "mean")
    assert mean[0, 0] == 1.0
    assert np.isnan(mean[1, 1])


def test_upsampling_keeps_corner_values():
    grid = make_grid()
    coarse = np.array([[0.0, 1.0], [2.0, 3.0]], np.float32)
    nearest = grid.resample(coarse, "nearest")
    bilinear = grid.resample(coarse, "mean")
    assert nearest.shape == bilinear.shape == (4, 4)
    assert nearest[0, 0] == 0 and nearest[3, 3] == 3
    assert bilinear[0, 0] == 0 and bilinear[3, 3] == 3


def test_rasterize_points_keeps_max_and_drops_outside():
    grid = make_grid()
    out = grid.rasterize_points([50, 60, 350, 999], [350, 340, 50, 50], [0.6, 0.9, 0.3, 1.0])
    assert out[0, 0] == np.float32(0.9)
    assert out[3, 3] == np.float32(0.3)
    assert out.sum() == pytest.approx(1.2)


def test_bbox_roundtrip():
    bbox = (500_000.0, 4_000_000.0, 536_000.0, 4_036_000.0)
    assert Grid.from_bbox("EPSG:32632", bbox, 375.0).bbox == bbox
