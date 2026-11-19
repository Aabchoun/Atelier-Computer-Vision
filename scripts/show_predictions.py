"""Side-by-side figures of trained-model predictions on test samples (for reports / presentations).

    python scripts/show_predictions.py fire     # Sen2Fire test scene  -> outputs/figures/fire_predictions.png
    python scripts/show_predictions.py spread   # NDWS test split      -> outputs/figures/spread_predictions.png

Uses the caches written by train_fire.py / train_spread.py and the checkpoints in models/.
Rows are test samples that contain fire, chosen evenly from the most to the least fire.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.agents.fire.agent import swir_fire_test  # noqa: E402
from firespread.common.config import load_config, resolve_path  # noqa: E402
from firespread.common.metrics import segmentation_metrics  # noqa: E402
from firespread.common.nn import load_checkpoint, predict_proba  # noqa: E402
from firespread.io.ndws import LAYERS, load_split  # noqa: E402
from firespread.io.sen2fire import PIPELINE_BANDS  # noqa: E402


def stretch(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) reflectance -> (H, W, 3) in [0, 1] with a 2-98 % stretch per image."""
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return np.clip((np.moveaxis(rgb, 0, -1) - lo) / max(hi - lo, 1e-6), 0, 1)


def pick(fire_fraction: np.ndarray, n: int) -> list[int]:
    """n samples with fire, spread from the most to the least burning one."""
    with_fire = np.flatnonzero(fire_fraction > 0)
    ordered = with_fire[np.argsort(-fire_fraction[with_fire])]
    if len(ordered) <= n:
        return ordered.tolist()
    return ordered[np.linspace(0, len(ordered) - 1, n).round().astype(int)].tolist()


def f1_text(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> str:
    f1 = segmentation_metrics(pred, truth, threshold)["f1"]
    return "F1 n/a" if np.isnan(f1) else f"F1 {f1:.2f}"


def fire_figure(args) -> Path:
    from matplotlib.figure import Figure

    cache = resolve_path(args.data or "data/raw/sen2fire") / "cache"
    x = np.load(cache / "test_x.npy", mmap_mode="r")
    y = np.load(cache / "test_y.npy", mmap_mode="r")
    model, meta = load_checkpoint(resolve_path(args.model or "models/fire_unet.pt"))
    if list(meta["inputs"]) != list(PIPELINE_BANDS):
        raise ValueError(f"checkpoint inputs {meta['inputs']} differ from {PIPELINE_BANDS}")
    cfg = load_config()["agents"]["fire"]
    band = {name: i for i, name in enumerate(PIPELINE_BANDS)}

    rows = pick(np.array([y[i].mean() for i in range(len(y))]), args.n)
    titles = ["True colour (B4 B3 B2)", "Infrared (B12 B8A B4)", "Label (MODIS)", "Baseline: SWIR test", "Fire U-Net probability"]
    fig = Figure(figsize=(3.2 * len(titles), 3.3 * len(rows)))
    axes = np.atleast_2d(fig.subplots(len(rows), len(titles)))
    for r, i in enumerate(rows):
        xi, yi = np.asarray(x[i], np.float32), np.asarray(y[i], np.float32)
        baseline = swir_fire_test(xi[band["B8A"]], xi[band["B11"]], xi[band["B12"]], cfg["swir2_min"], cfg["ratio_min"])
        prob = predict_proba(model, xi, meta["mean"], meta["std"])
        panels = [
            (stretch(xi[[band["B4"], band["B3"], band["B2"]]]), None),
            (stretch(xi[[band["B12"], band["B8A"], band["B4"]]]), None),
            (yi, "Reds"),
            (baseline, "Reds"),
            (prob, "magma"),
        ]
        for c, (img, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        axes[r, 2].set_ylabel(f"test patch {i}\nfire {yi.mean():.1%}", fontsize=9)
        axes[r, 3].set_xlabel(f1_text(baseline, yi), fontsize=9)
        axes[r, 4].set_xlabel(f1_text(prob, yi), fontsize=9)
        axes[r, 4].contour(yi, levels=[0.5], colors="cyan", linewidths=0.8)
    fig.suptitle("Fire agent on the Sen2Fire test scene (cyan = MODIS label outline)", fontsize=13)
    fig.tight_layout()
    return save(fig, args.out or "outputs/figures/fire_predictions.png")


def spread_figure(args) -> Path:
    from matplotlib.figure import Figure

    data_dir = resolve_path(args.data or "data/raw/next_day_wildfire_spread")
    x_all, y = load_split(data_dir, "test")
    model, meta = load_checkpoint(resolve_path(args.model or "models/spread_unet.pt"))
    inputs = list(meta["inputs"])
    x = x_all[:, [LAYERS.index(name) for name in inputs]]
    idx = {name: LAYERS.index(name) for name in LAYERS}

    growth = np.array([np.mean(y[i] == 1) for i in range(len(y))])
    rows = pick(growth, args.n)
    titles = ["Fire today", "Fuel (NDVI) + wind", "Fire tomorrow (truth)", "Persistence", "Spread U-Net probability"]
    fig = Figure(figsize=(3.2 * len(titles), 3.3 * len(rows)))
    axes = np.atleast_2d(fig.subplots(len(rows), len(titles)))
    step = 8
    gy, gx = np.mgrid[step // 2 : 64 : step, step // 2 : 64 : step]
    for r, i in enumerate(rows):
        today = np.nan_to_num(x_all[i, idx["fire"]])
        truth = np.where(y[i] < 0, np.nan, y[i])
        prob = predict_proba(model, x[i], meta["mean"], meta["std"])
        for c, (img, cmap, vmin, vmax) in enumerate(
            [(today, "Reds", 0, 1), (x_all[i, idx["ndvi"]], "YlGn", 0, 0.9), (truth, "Reds", 0, 1), (today, "Reds", 0, 1), (prob, "magma", 0, 1)]
        ):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        u, v = x_all[i, idx["wind_u"]], x_all[i, idx["wind_v"]]
        axes[r, 1].quiver(gx, gy, u[gy, gx], v[gy, gx], color="black", scale=60)  # quiver angles are in screen space: +v points up = north
        axes[r, 1].contour(today, levels=[0.5], colors="red", linewidths=0.8)
        axes[r, 0].set_ylabel(f"test tile {i}", fontsize=9)
        axes[r, 3].set_xlabel(f1_text(today, y[i]), fontsize=9)
        axes[r, 4].set_xlabel(f1_text(prob, y[i]), fontsize=9)
        axes[r, 4].contour(np.nan_to_num(truth), levels=[0.5], colors="cyan", linewidths=0.8)
    fig.suptitle("Spread agent on the Next Day Wildfire Spread test split (1 km cells, cyan = true fire tomorrow)", fontsize=13)
    fig.tight_layout()
    return save(fig, args.out or "outputs/figures/spread_predictions.png")


def save(fig, path: str) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", choices=("fire", "spread"))
    parser.add_argument("--n", type=int, default=5, help="number of test samples (rows)")
    parser.add_argument("--model", default=None, help="checkpoint (default models/<agent>_unet.pt)")
    parser.add_argument("--data", default=None, help="dataset folder (default data/raw/...)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    path = fire_figure(args) if args.agent == "fire" else spread_figure(args)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
