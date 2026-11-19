"""Fire Spread Simulator & Predictor: demo dashboard.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Tabs:
  Project         what the system does, the agents, which ones are trained
  Full system     run the whole multi-agent pipeline on a synthetic scene (baseline or trained models)
  Spread model    next-day fire forecast on Next Day Wildfire Spread test tiles, with "what if" wind / weather
  Fire detection  active-fire detection on Sen2Fire test patches (or an uploaded .npz patch)
  Training        test metrics and validation curves of the trained models

Tabs that need a trained model or a dataset explain what is missing instead of failing.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firespread.agents.fire.agent import swir_fire_test  # noqa: E402
from firespread.common.config import load_config  # noqa: E402
from firespread.common.metrics import evaluate  # noqa: E402
from firespread.common.scene import TRUTH  # noqa: E402
from firespread.coordinator import Coordinator  # noqa: E402
from firespread.evaluation import evaluate_forecasts  # noqa: E402
from firespread.io import make_synthetic_scene  # noqa: E402
from firespread.io.ndws import LAYERS  # noqa: E402
from firespread.io.sen2fire import PIPELINE_BANDS, load_patch  # noqa: E402

MODELS = ROOT / "models"
SPREAD_CKPT, FIRE_CKPT = MODELS / "spread_unet.pt", MODELS / "fire_unet.pt"
NDWS_DIR = ROOT / "data" / "raw" / "next_day_wildfire_spread"
SEN2FIRE_CACHE = ROOT / "data" / "raw" / "sen2fire" / "cache"
METRICS = ("precision", "recall", "f1", "iou", "auc_pr")

st.set_page_config(page_title="Fire Spread Simulator", page_icon="🔥", layout="wide")


# ---------------------------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    from firespread.common.nn import load_checkpoint

    return load_checkpoint(path)


@st.cache_resource(show_spinner="Loading Next Day Wildfire Spread test split…")
def ndws_test():
    from firespread.io.ndws import load_split

    x, y = load_split(NDWS_DIR, "test")
    growth = np.array([np.mean(t == 1) for t in y])
    order = np.flatnonzero(growth > 0)
    return x, y, order[np.argsort(-growth[order])]


@st.cache_resource(show_spinner="Loading Sen2Fire test scene…")
def sen2fire_test():
    x = np.load(SEN2FIRE_CACHE / "test_x.npy", mmap_mode="r")
    y = np.load(SEN2FIRE_CACHE / "test_y.npy", mmap_mode="r")
    frac = np.array([y[i].mean() for i in range(len(y))])
    order = np.flatnonzero(frac > 0)
    return x, y, order[np.argsort(-frac[order])]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import segmentation_models_pytorch  # noqa: F401
    except ImportError:
        return False
    return True


def model_report(name: str) -> dict | None:
    path = MODELS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def predict(path: Path, x: np.ndarray) -> np.ndarray:
    from firespread.common.nn import predict_proba

    model, meta = load_model(str(path))
    return predict_proba(model, x, meta.get("mean"), meta.get("std"))


# ---------------------------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------------------------
def stretch(rgb: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return np.clip((np.moveaxis(rgb, 0, -1) - lo) / max(hi - lo, 1e-6), 0, 1)


def panels(items: list[tuple[str, np.ndarray, str | None, tuple | None]], outline: np.ndarray | None = None, wind=None):
    """Row of image panels: (title, image, cmap, (vmin, vmax)). `outline` drawn in cyan on the last one."""
    fig = Figure(figsize=(3.3 * len(items), 3.5))
    axes = np.atleast_1d(fig.subplots(1, len(items)))
    for k, (ax, (title, img, cmap, lim)) in enumerate(zip(axes, items)):
        kwargs = {"cmap": cmap, "vmin": lim[0], "vmax": lim[1]} if cmap else {}
        ax.imshow(img, interpolation="nearest", **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        if wind is not None and wind[0] == k:
            _, u, v = wind
            step = max(img.shape[0] // 8, 1)
            gy, gx = np.mgrid[step // 2 : img.shape[0] : step, step // 2 : img.shape[1] : step]
            ax.quiver(gx, gy, u[gy, gx], v[gy, gx], color="black", scale=60)
    if outline is not None and np.nanmax(outline) > 0:
        axes[-1].contour(np.nan_to_num(outline), levels=[0.5], colors="cyan", linewidths=0.9)
    fig.tight_layout()
    return fig


def metrics_row(results: dict[str, dict[str, float]]) -> list[dict]:
    return [{"model": name, **{k: round(float(v[k]), 3) for k in METRICS if k in v}} for name, v in results.items()]


# ---------------------------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------------------------
def tab_project() -> None:
    st.markdown(
        """
A **multi-agent computer-vision system** for vegetated areas. Perception agents detect the fire, smoke,
fuel and burned area on satellite data. A deep network predicts where the fire will be tomorrow,
and a cellular automaton simulates the same spread physically. A coordinator puts everything on
**one common grid** and fuses it into a risk map, uncertain zones and alerts.
"""
    )
    st.graphviz_chart(
        """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f4f2", fontname="Helvetica", fontsize=11];
  edge [color="#888"];
  data [label="Satellite & weather data\\nSentinel-2 · FIRMS · ERA5 · DEM · WorldCover", fillcolor="#e8eef9"];
  subgraph cluster_p { label="Perception agents"; style=dashed; color="#bbb";
    fire [label="Fire\\n(U-Net, trained)", fillcolor="#fde2d4"]; smoke [label="Smoke"]; veg [label="Vegetation / fuel"];
    burn [label="Burned area"]; env [label="Environment\\nwind · humidity · slope"]; }
  cube [label="Shared datacube\\n(one grid, 375 m)", fillcolor="#fff4c2"];
  spread [label="Spread agent\\n(U-Net, trained)", fillcolor="#fde2d4"];
  sim [label="Simulation agent\\n(cellular automaton)"];
  coord [label="Coordinator\\nrisk · uncertainty · alerts", fillcolor="#dff3e8"];
  data -> {fire smoke veg burn env}; {fire smoke veg burn env} -> cube; cube -> {spread sim}; {spread sim} -> coord;
}
"""
    )
    rows = []
    for agent, ckpt, baseline, dataset in (
        ("Fire", "fire_unet", "SWIR threshold test", "Sen2Fire"),
        ("Spread", "spread_unet", "persistence (tomorrow = today)", "Next Day Wildfire Spread"),
        ("Burned area", None, "dNBR", "CaBuAr (planned)"),
        ("Smoke", None, "haze index", "USTC_SmokeRS + own labels (planned)"),
        ("Vegetation", None, "WorldCover + NDVI", "EuroSAT (optional)"),
        ("Environment", None, "ERA5 + DEM (no training needed)", "-"),
        ("Simulation", None, "Alexandridis cellular automaton (calibrated)", "-"),
    ):
        report = model_report(ckpt) if ckpt else None
        status, score = "baseline", ""
        if report:
            test = report["test"]
            base_name = next(n for n in test if not n.startswith("unet"))
            status = "trained ✅"
            score = f"F1 {test['unet@0.5']['f1']:.3f} vs baseline {test[base_name]['f1']:.3f}"
        rows.append({"agent": agent, "current method": "U-Net" if report else baseline, "status": status, "test score": score, "training data": dataset})
    st.subheader("Agents")
    st.dataframe(rows, hide_index=True, width="stretch")
    if not torch_available():
        st.info("PyTorch is not installed here: the trained models cannot run on this machine (baselines still work).")


def tab_full_system() -> None:
    st.markdown("Runs **every agent and the coordinator** on a synthetic scene that has a known ground truth for the next day.")
    c1, c2, c3, c4 = st.columns(4)
    seed = c1.number_input("Scene seed", 0, 999, 0)
    size = c2.select_slider("Grid size (cells of 375 m)", [64, 96, 128], 96)
    clouds = c3.slider("Cloud cover on the optical scene", 0.0, 0.8, 0.0, 0.1)
    disabled = c4.multiselect("Switch off agents (ablation)", ["smoke", "vegetation", "burned_area", "environment"])
    can_torch = torch_available()
    t1, t2 = st.columns(2)
    use_spread = t1.toggle("Use the trained spread U-Net", value=SPREAD_CKPT.exists() and can_torch, disabled=not (SPREAD_CKPT.exists() and can_torch))
    use_fire = t2.toggle("Use the trained fire U-Net", value=False, disabled=not (FIRE_CKPT.exists() and can_torch),
                         help="The synthetic scene is drawn at ~94 m, coarser than Sentinel-2; the baseline is the safer choice here.")

    if not st.button("Run the system", type="primary"):
        return
    config = load_config()
    config["agents"]["enabled"] = [a for a in config["agents"]["enabled"] if a not in disabled]
    if use_spread:
        config["agents"]["spread"].update(model="unet", checkpoint=str(SPREAD_CKPT))
    if use_fire:
        config["agents"]["fire"]["checkpoint"] = str(FIRE_CKPT)
    with st.spinner("Running the agents…"):
        raw = make_synthetic_scene(size=int(size), cloud_cover=float(clouds), seed=int(seed))
        result = Coordinator(config).run(raw)
        truth = raw.get(TRUTH)
        if truth is not None and "fire_next" in truth:
            result.report.evaluation = evaluate_forecasts(result.cube, raw.grid.resample(truth["fire_next"], "max"))
        from firespread.viz import save_quicklook

        with tempfile.TemporaryDirectory() as tmp:
            png = save_quicklook(result.cube, Path(tmp) / "quicklook.png")
            st.image(png.read_bytes(), width="stretch")

    report = result.report
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Alerts")
        for event in report.by_level("alert") or []:
            st.error(event.message)
        if not report.by_level("alert"):
            st.success("No alert")
        if report.summary:
            st.subheader("Summary")
            st.dataframe([{"item": k, "value": v} for k, v in report.summary.items()], hide_index=True)
    with right:
        if report.evaluation:
            st.subheader("Forecast vs ground truth (next day)")
            st.dataframe(metrics_row(report.evaluation), hide_index=True, width="stretch")
        st.subheader("Event log")
        for e in report.events:
            st.markdown(f"- **{e.level.upper()}** [{e.source}] {e.message}")


def tab_spread() -> None:
    st.markdown(
        "A **64 × 64 km tile (1 km cells)** from the Next Day Wildfire Spread **test split**, never seen in training. "
        "Change the wind or the weather to see how the forecast reacts."
    )
    if not (NDWS_DIR / "cache" / "test.npz").exists():
        st.warning("Next Day Wildfire Spread is not available here (run scripts/train_spread.py on this machine first).")
        return
    x_all, y, order = ndws_test()
    have_model = SPREAD_CKPT.exists() and torch_available()
    if not have_model:
        st.info("No trained spread model here: only persistence is shown.")

    c1, c2 = st.columns([1, 2])
    rank = c1.slider("Tile (sorted from largest to smallest fire)", 0, len(order) - 1, 0)
    i = int(order[rank])
    with c2:
        w1, w2, w3, w4 = st.columns(4)
        rotate = w1.slider("Rotate wind (°)", -180, 180, 0, 15)
        speed = w2.slider("Wind speed ×", 0.0, 3.0, 1.0, 0.25)
        d_temp = w3.slider("Temperature +°C", -10, 10, 0)
        d_rh = w4.slider("Humidity + %", -30, 30, 0, 5)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)

    sample = x_all[i].copy()
    idx = {name: LAYERS.index(name) for name in LAYERS}
    u, v = sample[idx["wind_u"]], sample[idx["wind_v"]]
    a = np.radians(-rotate)  # positive = clockwise on the map
    sample[idx["wind_u"]], sample[idx["wind_v"]] = speed * (u * np.cos(a) - v * np.sin(a)), speed * (u * np.sin(a) + v * np.cos(a))
    sample[idx["t2m"]] += d_temp
    sample[idx["rh"]] = np.clip(sample[idx["rh"]] + d_rh, 0, 100)

    today = np.nan_to_num(sample[idx["fire"]])
    truth = np.where(y[i] < 0, np.nan, y[i])
    items = [
        ("Fire today", today, "Reds", (0, 1)),
        ("Fuel (NDVI) + wind", sample[idx["ndvi"]], "YlGn", (0, 0.9)),
        ("Fire tomorrow (truth)", truth, "Reds", (0, 1)),
        ("Persistence", today, "Reds", (0, 1)),
    ]
    results = {"persistence": evaluate(today, y[i], threshold)}
    if have_model:
        _, meta = load_model(str(SPREAD_CKPT))
        prob = predict(SPREAD_CKPT, sample[[LAYERS.index(n) for n in meta["inputs"]]])
        items.append(("Spread U-Net probability", prob, "magma", (0, 1)))
        results["spread U-Net"] = evaluate(prob, y[i], threshold)
    st.pyplot(panels(items, outline=truth, wind=(1, sample[idx["wind_u"]], sample[idx["wind_v"]])))
    st.caption("Cyan outline on the last panel = true fire tomorrow. Arrows show where the wind blows to.")
    st.dataframe(metrics_row(results), hide_index=True)
    if rotate or speed != 1 or d_temp or d_rh:
        st.caption("Inputs modified: the truth panel still shows what really happened with the original weather.")


def tab_fire() -> None:
    st.markdown(
        "A **512 × 512 Sentinel-2 patch (10 m pixels)**. Active fire glows bright in the short-wave infrared bands. "
        "Labels come from MODIS (1 km), so they mark coarse fire zones."
    )
    have_model = FIRE_CKPT.exists() and torch_available()
    if not have_model:
        st.info("No trained fire model here: only the SWIR baseline is shown.")
    source = st.radio("Image", ["Sen2Fire test scene", "Upload a Sen2Fire-style .npz patch"], horizontal=True)
    label = None
    if source.startswith("Sen2Fire"):
        if not (SEN2FIRE_CACHE / "test_x.npy").exists():
            st.warning("Sen2Fire is not available here (run scripts/train_fire.py on this machine first).")
            return
        x, y, order = sen2fire_test()
        rank = st.slider("Patch (sorted from most to least fire)", 0, len(order) - 1, 0)
        i = int(order[rank])
        xi, label = np.asarray(x[i], np.float32), np.asarray(y[i], np.float32)
    else:
        upload = st.file_uploader("Patch with an `image` (12 bands B1..B12 without B10) and optionally a `label`", type="npz")
        if upload is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            f.write(upload.getvalue())
        xi, label = load_patch(f.name)
        Path(f.name).unlink(missing_ok=True)
        if label.max() == 0:
            label = None
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, key="fire_threshold")

    band = {name: k for k, name in enumerate(PIPELINE_BANDS)}
    cfg = load_config()["agents"]["fire"]
    baseline = swir_fire_test(xi[band["B8A"]], xi[band["B11"]], xi[band["B12"]], cfg["swir2_min"], cfg["ratio_min"])
    items = [
        ("True colour (B4 B3 B2)", stretch(xi[[band["B4"], band["B3"], band["B2"]]]), None, None),
        ("Infrared (B12 B8A B4)", stretch(xi[[band["B12"], band["B8A"], band["B4"]]]), None, None),
    ]
    if label is not None:
        items.append(("Label (MODIS)", label, "Reds", (0, 1)))
    items.append(("Baseline: SWIR test", baseline, "Reds", (0, 1)))
    results = {}
    if label is not None:
        results["SWIR baseline"] = evaluate(baseline, label, 0.5)
    if have_model:
        with st.spinner("Running the fire U-Net…"):
            prob = predict(FIRE_CKPT, xi)
        items.append(("Fire U-Net probability", prob, "magma", (0, 1)))
        if label is not None:
            results["fire U-Net"] = evaluate(prob, label, threshold)
        st.metric("Area flagged as fire by the U-Net", f"{np.mean(prob >= threshold):.1%} of the patch")
    st.pyplot(panels(items, outline=label))
    if results:
        st.dataframe(metrics_row(results), hide_index=True)


LOG_LINE = re.compile(r"epoch\s+(\d+)/\d+\s+loss ([\d.]+)\s+val P ([\d.]+) R ([\d.]+) F1 ([\d.]+) AUC-PR ([\d.]+)")


def tab_training() -> None:
    found = False
    for name, title in (("spread_unet", "Spread U-Net (Next Day Wildfire Spread)"), ("fire_unet", "Fire U-Net (Sen2Fire)")):
        report = model_report(name)
        if report is None:
            continue
        found = True
        st.subheader(title)
        a, b = st.columns([1, 1])
        with a:
            st.markdown(f"Inputs: `{', '.join(report['inputs'])}` · best epoch **{report['best_epoch']}** of {report['epochs']}")
            st.dataframe(metrics_row(report["test"]), hide_index=True, width="stretch")
        log = next((p for p in (ROOT / f"train_{name.split('_')[0]}.log", ROOT / "docs" / "results" / f"train_{name.split('_')[0]}.log") if p.exists()), None)
        with b:
            if log is None:
                st.caption("No training log found (train_<agent>.log in the project root).")
                continue
            rows = [m.groups() for m in LOG_LINE.finditer(log.read_text(encoding="utf-8", errors="ignore"))]
            if rows:
                epochs = [int(r[0]) for r in rows]
                fig = Figure(figsize=(6, 3.2))
                ax = fig.subplots()
                ax.plot(epochs, [float(r[1]) for r in rows], label="training loss", color="#8a8a8a")
                ax2 = ax.twinx()
                ax2.plot(epochs, [float(r[5]) for r in rows], label="val AUC-PR", color="#d9480f")
                ax2.plot(epochs, [float(r[4]) for r in rows], label="val F1", color="#1c7ed6")
                ax2.axvline(report["best_epoch"], color="#2b8a3e", linestyle="--", linewidth=1, label="kept model")
                ax.set_xlabel("epoch"), ax.set_ylabel("loss"), ax2.set_ylabel("validation score")
                lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
                ax2.legend(lines, [h.get_label() for h in lines], fontsize=8, loc="lower right")
                fig.tight_layout()
                st.pyplot(fig)
    if not found:
        st.info("No trained model found in models/. Train one with scripts/train_spread.py or scripts/train_fire.py.")


st.title("🔥 Fire Spread Simulator & Predictor")
tabs = st.tabs(["Project", "Full system", "Spread model", "Fire detection", "Training"])
with tabs[0]:
    tab_project()
with tabs[1]:
    tab_full_system()
with tabs[2]:
    tab_spread()
with tabs[3]:
    tab_fire()
with tabs[4]:
    tab_training()
