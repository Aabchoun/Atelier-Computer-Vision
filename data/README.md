# data/

This folder stays local: it is ignored by Git, except for this file. Each member downloads their own datasets here.

```
data/
├── raw/                          training datasets, one folder each, as downloaded
│   ├── next_day_wildfire_spread/ Spread agent      (Kaggle)
│   ├── wildfire_spread_ts/       Spread agent      (Zenodo, ~48 GB: start with one year)
│   ├── landsat8_active_fire/     Fire agent        (Pereira et al.)
│   ├── sen2fire/                 Fire + Smoke      (Zenodo)
│   ├── ustc_smokers/             Smoke agent       (Hugging Face)
│   ├── eurosat/                  Vegetation agent  (GitHub)
│   └── cabuar/                   Burned-area agent (Hugging Face)
├── labels/
│   └── smoke/                    your own Sentinel-2 smoke masks (CVAT / SAM exports)
└── scenes/
    └── <scene_name>/             case-study scenes in pipeline format (below)
```

## Scene format (what the pipeline reads)

Make one folder per area of interest and time window. Every raster covers **exactly the same bbox, in the same CRS**. Only the pixel size can differ between rasters, and it is best kept at an integer multiple of the grid resolution.

```
data/scenes/<scene_name>/
├── scene.json
├── s2_pre.tif   | .npz       optical scene before the fire
├── s2_post.tif  | .npz       latest optical scene during the fire
├── era5.tif     | .npz       bands u10, v10, t2m, d2m
├── dem.tif      | .npz       band elevation
├── worldcover.tif | .npz     band landcover (optional: the NDVI fallback is used if absent)
├── truth.npz                 optional, evaluation only: band fire_next
└── firms.csv                 FIRMS download (latitude/longitude) or x/y in the scene CRS
```

`scene.json`:

```json
{
  "name": "my_fire_2023",
  "grid": {"crs": "EPSG:32632", "left": 500000.0, "top": 4100000.0, "resolution": 375.0, "width": 96, "height": 96},
  "rasters": {
    "s2_post": {
      "file": "s2_post.tif",
      "bands": ["B2", "B3", "B4", "B8", "B11", "B12", "cloud_mask"],
      "cloud_mask": "cloud_mask",
      "resolution": 20.0,
      "acquisition_time": "2023-07-22T10:10:00",
      "sensor": "Sentinel-2 L2A",
      "scale": 0.0001
    }
  },
  "firms": {"file": "firms.csv", "sensor": "VIIRS"}
}
```

- `bands` lists every band in file order. `cloud_mask` names the band that holds the mask (e.g. from the SCL band or s2cloudless), or is `null`.
- `scale` converts stored values to reflectance in [0, 1]. Sentinel-2 L2A exported from Earth Engine needs `0.0001`.
- Landsat scenes must use the Sentinel-2 band names (L8 B7 → `B12`, B6 → `B11`, B5 → `B8`).
- Reading `.tif` files needs `pip install -e ".[geo]"` (rasterio). Reading FIRMS latitude/longitude columns needs pyproj, which is in the same extra.
- `python scripts/make_synthetic_scene.py` writes a complete example to `data/scenes/synthetic/`.
