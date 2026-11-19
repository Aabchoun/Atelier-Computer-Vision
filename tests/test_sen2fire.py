import numpy as np
import pytest

from firespread.agents.fire.agent import swir_fire_test
from firespread.io.sen2fire import PIPELINE_BANDS, S2_ORDER, describe, load_patch, split_files


def write_patch(path, channels_last=False):
    image = np.zeros((13, 32, 32), np.int16)  # 12 S2 bands + aerosol, digital numbers
    for i in range(12):
        image[i] = (i + 1) * 100  # band k has value 100 * (its position + 1)
    label = np.zeros((32, 32), np.uint8)
    label[:4, :4] = 1
    if channels_last:
        image = np.moveaxis(image, 0, -1)
    np.savez(path, image=image, label=label)


@pytest.mark.parametrize("channels_last", [False, True])
def test_load_patch_bands_and_scale(tmp_path, channels_last):
    write_patch(tmp_path / "p.npz", channels_last)
    x, y = load_patch(tmp_path / "p.npz")
    assert x.shape == (len(PIPELINE_BANDS), 32, 32)
    for k, band in enumerate(PIPELINE_BANDS):
        assert np.allclose(x[k], (S2_ORDER.index(band) + 1) * 100 / 1e4)  # reflectance
    assert y.sum() == 16 and set(np.unique(y)) == {0.0, 1.0}
    assert "fire pixels" in describe(tmp_path / "p.npz")


def test_split_files(tmp_path):
    for scene in ("scene1", "scene2", "scene3", "scene4"):
        (tmp_path / "Sen2Fire" / scene).mkdir(parents=True)
        write_patch(tmp_path / "Sen2Fire" / scene / f"{scene}_patch_1_1.npz")
    assert len(split_files(tmp_path, "train")) == 2
    assert [p.parent.name for p in split_files(tmp_path, "test")] == ["scene4"]


def test_swir_fire_test():
    nir, swir1 = np.array([0.1, 0.1]), np.array([0.1, 0.1])
    swir2 = np.array([0.5, 0.12])
    assert swir_fire_test(nir, swir1, swir2).tolist() == [1.0, 0.0]
