"""Sen2Fire (Xu, Berg & Haglund, IGARSS 2024; Zenodo record 10881058).

2466 patches of 512 x 512 at 10 m over the 2019-2020 New South Wales bushfires, one .npz per patch
in scene1/ .. scene4/. Each patch holds 12 Sentinel-2 L2A bands (B1..B12 without the cirrus band
B10) plus the Sentinel-5P aerosol index, and a 0/1 fire label derived from MODIS MOD14A1.
Official split: scenes 1-2 train, scene 3 validation, scene 4 test.

Our fire agent only receives B2, B3, B4, B8, B8A, B11, B12 as surface reflectance in [0, 1], so the
loader returns those bands by Sentinel-2 name and converts digital numbers (x 1e4) to reflectance.
The .npz key names are detected (the image is the 3-D array, the label the 2-D one); run
`python scripts/train_fire.py --inspect` once to check them on the downloaded files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

S2_ORDER = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
PIPELINE_BANDS = ("B2", "B3", "B4", "B8", "B8A", "B11", "B12")
SPLIT_SCENES = {"train": ("scene1", "scene2"), "eval": ("scene3",), "test": ("scene4",)}


def split_files(root: str | Path, split: str) -> list[Path]:
    root = Path(root)
    files: list[Path] = []
    for scene in SPLIT_SCENES[split]:
        matches = [d for d in root.rglob(scene) if d.is_dir()]
        for directory in matches:
            files.extend(sorted(directory.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz patches for split {split!r} ({SPLIT_SCENES[split]}) under {root}")
    return files


def _find_arrays(npz) -> tuple[np.ndarray, np.ndarray]:
    image = label = None
    for key in npz.files:
        array = npz[key]
        if array.ndim == 3 and image is None and min(array.shape) >= len(S2_ORDER):
            image = array
        elif array.ndim == 2 and label is None and key.lower() in ("label", "labels", "mask", "gt", "y"):
            label = array
    if label is None:  # fall back on any 2-D integer-like array
        for key in npz.files:
            array = npz[key]
            if array.ndim == 2 and np.all(np.isin(np.unique(array[::16, ::16]), (0, 1))):
                label = array
                break
    if image is None or label is None:
        raise ValueError(f"cannot identify image / label arrays among keys {npz.files}")
    if image.shape[-1] < image.shape[0]:  # channels last -> channels first
        image = np.moveaxis(image, -1, 0)
    return image, label


def load_patch(path: str | Path, bands: tuple[str, ...] = PIPELINE_BANDS) -> tuple[np.ndarray, np.ndarray]:
    """-> (len(bands), H, W) float32 reflectance, (H, W) float32 label in {0, 1}."""
    with np.load(path) as npz:
        image, label = _find_arrays(npz)
    index = [S2_ORDER.index(b) for b in bands]
    x = image[index].astype(np.float32)
    if np.nanmax(x) > 2:  # digital numbers -> reflectance
        x = x / 1e4
    return x, (label > 0).astype(np.float32)


def describe(path: str | Path) -> str:
    """Human-readable summary of one patch file (keys, shapes, value ranges) to check the loader."""
    lines = [str(path)]
    with np.load(path) as npz:
        for key in npz.files:
            a = npz[key]
            lines.append(f"  {key:10s} shape {a.shape} dtype {a.dtype} min {np.nanmin(a):.4g} max {np.nanmax(a):.4g}")
        image, label = _find_arrays(npz)
    lines.append("  per-channel mean (assumed order " + ",".join(S2_ORDER) + ", then extra channels):")
    lines.append("   " + " ".join(f"{m:.0f}" if m > 2 else f"{m:.3f}" for m in image.reshape(image.shape[0], -1).mean(1)))
    lines.append(f"  fire pixels: {(label > 0).mean():.3%}")
    return "\n".join(lines)
