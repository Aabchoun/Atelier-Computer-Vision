"""Minimal dashboard: browse the layers and the report of a saved pipeline run.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

TODO(phase 5): Folium map in geographic coordinates, time slider, CA animation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firespread.common.scene import SceneDatacube  # noqa: E402

CMAPS = {
    "fire": "hot_r", "fire_s2": "hot_r", "fire_firms": "hot_r", "smoke": "Greys", "elevation": "terrain",
    "landcover": "tab20", "ndvi": "RdYlGn", "dnbr": "RdYlGn_r", "p_veg": "YlGn", "risk": "inferno",
    "spread_prob": "magma", "sim_prob": "magma", "sim_reach_prob": "magma", "disagreement": "PuOr",
}

st.set_page_config(page_title="Fire spread", layout="wide")
st.title("Fire Spread Simulator & Predictor")

runs = sorted(p.parent for p in (ROOT / "outputs").glob("*/datacube/meta.json"))
if not runs:
    st.info("No run found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.parent.name)
cube = SceneDatacube.load(run)
layer = st.sidebar.selectbox("Layer", cube.names, index=cube.names.index("risk") if "risk" in cube.names else 0)
message = cube.message(layer)

left, right = st.columns([2, 1])
with left:
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    image = ax.imshow(cube.get(layer), cmap=CMAPS.get(layer, "viridis"), interpolation="nearest")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.8)
    st.pyplot(fig)
with right:
    st.subheader(layer)
    st.json(message.to_dict())

report = run.parent / "report.md"
if report.exists():
    st.markdown(report.read_text(encoding="utf-8"))
