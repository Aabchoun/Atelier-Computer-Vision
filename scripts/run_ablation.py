"""Ablation study: switch off one perception agent at a time and measure the forecast drop.

This is the main argument for the multi-agent design (all agents share one scene, so it is clean).

    python scripts/run_ablation.py --scene data/scenes/synthetic --layer sim_prob
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.agents import PERCEPTION_AGENTS  # noqa: E402
from firespread.common.config import load_config, resolve_path  # noqa: E402
from firespread.common.metrics import evaluate  # noqa: E402
from firespread.common.scene import TRUTH  # noqa: E402
from firespread.coordinator import Coordinator  # noqa: E402
from firespread.evaluation import results_table  # noqa: E402
from firespread.io import load_raw_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="data/scenes/synthetic")
    parser.add_argument("--config", default=None)
    parser.add_argument("--layer", default="risk", help="forecast layer to evaluate (spread_prob, sim_prob, risk)")
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_raw_scene(resolve_path(args.scene))
    truth_raster = raw.get(TRUTH)
    if truth_raster is None:
        sys.exit("The scene has no ground truth (raster 'truth' with band 'fire_next').")
    truth = raw.grid.resample(truth_raster["fire_next"], "max")

    results = {}
    # The fire agent is not ablated: without any fire layer there is nothing to forecast.
    for removed in [None, *[a for a in PERCEPTION_AGENTS if a != "fire"]]:
        cfg = copy.deepcopy(config)
        cfg["agents"]["enabled"] = [a for a in cfg["agents"]["enabled"] if a != removed]
        cube = Coordinator(cfg).run(raw).cube
        label = "full system" if removed is None else f"without {removed}"
        results[label] = evaluate(cube.get(args.layer), truth) if cube.has(args.layer) else {k: float("nan") for k in ("precision", "recall", "f1", "iou", "auc_pr")}

    print(f"Ablation on '{raw.name}', layer '{args.layer}'\n")
    print(results_table(results, first_column="configuration"))


if __name__ == "__main__":
    main()
