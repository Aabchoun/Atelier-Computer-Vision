"""Quick-look figure of a pipeline run (matplotlib, no display needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from firespread.common.scene import SceneDatacube

PANELS = (
    ("fire", "Fused active fire", "hot_r", (0, 1)),
    ("smoke", "Smoke", "Greys", (0, 1)),
    ("p_veg", "Fuel (p_veg)", "YlGn", (-0.4, 0.4)),
    ("burned", "Burned area (dNBR)", "Reds", (0, 1)),
    ("elevation", "Elevation + wind", "terrain", None),
    ("spread_prob", "Spread prediction", "magma", (0, 1)),
    ("sim_prob", "CA simulation", "magma", (0, 1)),
    ("risk", "Risk (cyan = uncertain)", "inferno", (0, 1)),
)


def save_quicklook(cube: SceneDatacube, path: str | Path) -> Path:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 8.5))
    axes = fig.subplots(2, 4)
    for ax, (layer, title, cmap, limits) in zip(axes.flat, PANELS):
        ax.set_axis_off()
        data = cube.get(layer)
        message = cube.message(layer)
        if data is None:
            ax.set_title(f"{title}\n(not available)", fontsize=10)
            continue
        vmin, vmax = limits if limits else (np.nanmin(data), np.nanmax(data))
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title + (" [stale]" if message.stale else ""), fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        if layer == "elevation" and cube.has("wind_u"):
            step = max(cube.grid.height // 8, 1)
            rows, cols = np.mgrid[step // 2 : cube.grid.height : step, step // 2 : cube.grid.width : step]
            ax.quiver(cols, rows, cube.get("wind_u")[rows, cols], cube.get("wind_v")[rows, cols], color="black")
        if layer == "risk" and cube.has("uncertain") and cube.get("uncertain").any():
            ax.contour(cube.get("uncertain"), levels=[0.5], colors="cyan", linewidths=0.8)
    fig.suptitle("Fire spread simulator & predictor: shared datacube", fontsize=13)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    return path


# Reference categorical slots 1-3 (validated all-pairs for 3 series), light chart chrome.
SERIES = {"spread_prob": ("#2a78d6", "Persistence (spread agent)"), "sim_prob": ("#eb6834", "Cellular automaton"), "risk": ("#1baf7a", "Risk (ensemble)")}
INK, INK_2, MUTED, GRID, AXIS, SURFACE, S2_BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"


def save_sequence_plot(rows: list[dict], path: str | Path, title: str) -> Path:
    """Day-by-day F1 (1-cell tolerance on top, exact below) of each forecast; Sentinel-2 days shaded."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    days = [r["day"][5:] for r in rows]
    x = np.arange(len(rows))
    fig = Figure(figsize=(11, 6.5), facecolor=SURFACE)
    axes = fig.subplots(2, 1, sharex=True)
    for ax, metric, label in zip(axes, ("f1@1", "f1"), ("F1, 1-cell tolerance", "F1, exact cell")):
        ax.set_facecolor(SURFACE)
        for i, row in enumerate(rows):
            if row["s2"]:
                ax.axvspan(i - 0.5, i + 0.5, color=S2_BAND, zorder=0, linewidth=0)
        for layer, (color, name) in SERIES.items():
            values = [r[layer][metric] if layer in r else np.nan for r in rows]
            ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=5, label=name, zorder=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel(label, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    axes[1].set_xticks(x, days, rotation=45)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=S2_BAND, edgecolor=AXIS, linewidth=0.5))
    labels.append("Sentinel-2 pass that day")
    axes[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    return path
