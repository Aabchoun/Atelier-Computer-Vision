import numpy as np

from firespread.io.ndws import LAYERS, TILE, convert_sample, relative_humidity_from_q, select_layers


def raw_record(**overrides):
    n = TILE * TILE
    raw = {
        "elevation": np.full(n, 200.0),
        "th": np.full(n, 270.0),  # wind FROM the west
        "vs": np.full(n, 5.0),
        "tmmn": np.full(n, 290.0),
        "tmmx": np.full(n, 303.15),
        "sph": np.full(n, 0.008),
        "pr": np.zeros(n),
        "pdsi": np.zeros(n),
        "NDVI": np.full(n, 5000.0),
        "population": np.zeros(n),
        "erc": np.full(n, 50.0),
        "PrevFireMask": np.zeros(n),
        "FireMask": np.zeros(n),
    }
    raw.update(overrides)
    return raw


def test_convert_units_and_wind_direction():
    prev = np.zeros(TILE * TILE)
    prev[0], prev[1] = 1, -1
    target = np.zeros(TILE * TILE)
    target[2], target[3] = 1, -1
    layers, y = convert_sample(raw_record(PrevFireMask=prev, FireMask=target))

    assert set(layers) == set(LAYERS)
    assert np.allclose(layers["wind_u"], 5.0, atol=1e-4)  # blows towards the east
    assert np.allclose(layers["wind_v"], 0.0, atol=1e-4)
    assert np.allclose(layers["t2m"], 30.0, atol=1e-3)
    assert np.allclose(layers["ndvi"], 0.5)
    assert np.allclose(layers["slope"], 0.0)
    assert 0 < layers["rh"][0, 0] < 100
    assert layers["fire"].flat[0] == 1 and np.isnan(layers["fire"].flat[1])
    assert y.flat[2] == 1 and y.flat[3] == -1 and y.flat[0] == 0


def test_corrupt_values_become_nan():
    tmmx = np.full(TILE * TILE, 303.15)
    tmmx[5] = 0.0  # fill value seen in the dataset
    with np.errstate(all="raise"):  # no overflow / division warnings any more
        layers, _ = convert_sample(raw_record(tmmx=tmmx))
    assert np.isnan(layers["t2m"].flat[5]) and np.isnan(layers["rh"].flat[5])
    assert np.isfinite(layers["rh"].flat[0])


def test_relative_humidity_saturated():
    # q at saturation for 20 degC, sea level: about 0.0147 kg/kg
    rh = relative_humidity_from_q(np.array([0.0147]), np.array([20.0]), np.array([0.0]))
    assert 95 < rh[0] <= 100


def test_select_layers_order():
    x = np.arange(len(LAYERS), dtype=np.float32)[None, :, None, None] * np.ones((2, 1, 4, 4), np.float32)
    out = select_layers(x, ["ndvi", "fire"])
    assert out[0, 0, 0, 0] == LAYERS.index("ndvi") and out[0, 1, 0, 0] == 0
