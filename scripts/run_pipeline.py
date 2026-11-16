"""Run the full multi-agent pipeline on one scene and save datacube, report and quick-look.

    python scripts/run_pipeline.py --scene data/scenes/synthetic
    python scripts/run_pipeline.py --scene data/scenes/synthetic --disable smoke vegetation
    python scripts/run_pipeline.py --scene data/scenes/synthetic --config configs/my_run.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.config import load_config, resolve_path  # noqa: E402
from firespread.common.scene import TRUTH  # noqa: E402
from firespread.coordinator import Coordinator  # noqa: E402
from firespread.evaluation import evaluate_forecasts  # noqa: E402
from firespread.io import load_raw_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="data/scenes/synthetic", help="scene directory (contains scene.json)")
    parser.add_argument("--config", default=None, help="YAML merged on top of configs/default.yaml")
    parser.add_argument("--out", default=None, help="output directory (default: outputs/<scene name>)")
    parser.add_argument("--disable", nargs="*", default=[], help="agents to switch off (ablation)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.disable:
        config["agents"]["enabled"] = [a for a in config["agents"]["enabled"] if a not in args.disable]

    raw = load_raw_scene(resolve_path(args.scene))
    result = Coordinator(config).run(raw)

    truth = raw.get(TRUTH)
    if truth is not None and "fire_next" in truth:
        result.report.evaluation = evaluate_forecasts(result.cube, raw.grid.resample(truth["fire_next"], "max"))

    out = result.save(resolve_path(args.out) if args.out else resolve_path(config["paths"]["outputs"]) / raw.name)
    print(result.report.to_markdown())
    print(f"Saved datacube, report and quicklook.png to {out}")


if __name__ == "__main__":
    main()
