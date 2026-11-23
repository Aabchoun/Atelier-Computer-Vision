import io
import zipfile

import numpy as np
import pytest

from firespread import image_pipeline as ip


def fake_bands(h=200, w=300):
    rng = np.random.default_rng(0)
    bands = {b: (0.05 + 0.02 * rng.random((h, w))).astype(np.float32) for b in ("B2", "B3", "B4", "B8", "B8A", "B11", "B12")}
    bands["B8"] += 0.25  # vegetation everywhere
    bands["B8A"] += 0.25
    bands["B12"][40:60, 100:130] = 0.6  # hot spot: passes the SWIR test
    return bands


def geotiff_zip(bands, resolution=20.0):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for band, array in bands.items():
            mem = rasterio.MemoryFile()
            h, w = array.shape
            with mem.open(driver="GTiff", height=h, width=w, count=1, dtype="float32", crs="EPSG:32635",
                          transform=from_origin(400000, 4560000, resolution, resolution)) as dst:
                dst.write(array, 1)
            z.writestr(f"2023-08-23-00_00_Sentinel-2_L2A_{band.replace('B', 'B0') if len(band) == 2 else band}_(Raw).tiff", mem.read())
    return buffer.getvalue()


def test_band_and_date_names():
    assert ip.band_name("2023-08-23_Sentinel-2_L2A_B8A_(Raw).tiff") == "B8A"
    assert ip.band_name("x_B04.tif") == "B4"
    assert ip.band_name("B12.tif") == "B12"
    assert ip.band_name("image.tif") is None
    assert str(ip.date_in("2023-08-23-00:00_x")) == "2023-08-23"


def test_block_reduce_and_upsample():
    a = np.zeros((5, 5), np.float32)
    a[4, 4] = 1
    cells = ip.block_reduce(a, 2, "max")
    assert cells.shape == (3, 3) and cells[2, 2] == 1 and cells.sum() == 1
    assert ip.upsample(cells, 2, (5, 5)).shape == (5, 5)


def test_read_geotiff_zip_and_analyse_offline():
    data = geotiff_zip(fake_bands())
    image = ip.read_image([("scene.zip", data)])
    assert image.georeferenced and abs(image.resolution - 20) < 1e-6
    assert str(image.date) == "2023-08-23" and set(image.bands) >= {"B2", "B8A", "B12"}
    lon, lat = image.center_lonlat()
    assert 20 < lon < 30 and 35 < lat < 45  # UTM 35N

    weather = ip.Weather(wind_speed=5, wind_from=270, t2m=35, rh=15)  # westerly wind -> blows towards east
    assert weather.uv[0] > 4.9 and abs(weather.uv[1]) < 1e-6
    result = ip.analyse(image, weather=weather, online=False, days=2)
    assert result.fire_method == "SWIR baseline" and result.factor == 50
    assert result.fire_cells.shape == (4, 6)
    assert np.nansum(result.fire_cells) >= 1  # the hot spot is found
    assert len(result.spread) == 2
    assert "burning cells today (1 km²)" in result.summary()
    ip.figure(image, result)


def test_dn_scaled_to_reflectance_and_manual_weather_required():
    bands = {b: a * 1e4 for b, a in fake_bands(80, 140).items()}
    image = ip.read_image([("p.zip", geotiff_zip(bands, resolution=10.0))])
    assert np.nanmax(image.band("B12")) == pytest.approx(0.6, rel=1e-3)
    image.transform = None  # not georeferenced, no internet -> weather must be given
    with pytest.raises(ValueError, match="weather"):
        ip.analyse(image, online=False)
