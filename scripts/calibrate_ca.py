"""Calibrate the cellular automaton on the first days of a sequence, test it on the later days.

    python scripts/calibrate_ca.py --sequence data/sequences/evros_2023 --runs outputs/evros_2023 --train-end 2023-08-25

Needs the datacubes saved by run_sequence.py (fire, burned, fuel, terrain and wind of each day)
and the FIRMS t+1 truth of the sequence. Objective: micro-F1 of the next-day active fire
(sim_prob >= 0.5) on the calibration days. Writes the best coefficients to a config file that
run_pipeline.py / run_sequence.py accept with --config.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.agents.simulation.calibration import calibrate, case_from_cube, score  # noqa: E402
from firespread.agents.simulation.cellular_automaton import CAParams  # noqa: E402
from firespread.common.config import load_config, resolve_path  # noqa: E402
from firespread.common.scene import TRUTH, SceneDatacube  # noqa: E402
from firespread.io.scene_io import load_scene_raster  # noqa: E402

SEARCH = {
    "p_h": [0.1, 0.2, 0.3, 0.4, 0.58],  # base ignition probability: overall spread rate
    "c1": [0.0, 0.045, 0.1],  # wind speed effect
    "c2": [0.131, 0.3, 0.5],  # wind direction effect (anisotropy)
}
STEPS_PER_DAY = [1, 2, 3, 4, 6]


def fmt(s: dict) -> str:
    return f"P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--runs", required=True, help="outputs of run_sequence.py (per-day datacubes)")
    parser.add_argument("--train-end", required=True, help="last calibration day, YYYY-MM-DD")
    parser.add_argument("--config", default=None, help="config used for the runs (default coefficients)")
    parser.add_argument("--out", default=None, help="default: configs/<sequence>_calibrated.yaml")
    parser.add_argument("--n-runs", type=int, default=10, help="Monte Carlo runs per evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    sequence, runs = resolve_path(args.sequence), resolve_path(args.runs)
    threshold = config["coordinator"]["fire_threshold"]
    train, test = [], []
    for day_dir in sorted((sequence / "days").iterdir()):
        cube_dir = runs / day_dir.name / "datacube"
        if not cube_dir.exists():
            continue
        cube = SceneDatacube.load(cube_dir)
        if not cube.has("fire"):
            continue
        truth = cube.grid.resample(load_scene_raster(day_dir, TRUTH)["fire_next"], "max")
        (train if day_dir.name <= args.train_end else test).append(case_from_cube(cube, truth, threshold))
    print(f"{len(train)} calibration days (<= {args.train_end}), {len(test)} test days")

    sim = config["agents"]["simulation"]
    default_params, default_steps = CAParams(**sim["params"]), int(sim["steps_per_day"])
    print(f"Default ({default_steps} steps/day, {default_params}):")
    print(f"  calibration {fmt(score(train, default_params, default_steps, args.n_runs))}")
    print(f"  test        {fmt(score(test, default_params, default_steps, args.n_runs))}")

    n = len(STEPS_PER_DAY) * len(SEARCH["p_h"]) * len(SEARCH["c1"]) * len(SEARCH["c2"])
    print(f"Grid search over {n} combinations ...", flush=True)
    start = time.perf_counter()
    params, steps, best, tried = calibrate(train, SEARCH, STEPS_PER_DAY, n_runs=args.n_runs)
    print(f"done ({time.perf_counter() - start:.0f} s)")
    test_score = score(test, params, steps, args.n_runs)
    print(f"Calibrated ({steps} steps/day, {params}):")
    print(f"  calibration {fmt(best)}")
    print(f"  test        {fmt(test_score)}")
    print("Top 5 on calibration days:")
    for t in sorted(tried, key=lambda t: -(t["f1"] if t["f1"] == t["f1"] else -1))[:5]:
        print(f"  steps={t['steps_per_day']} p_h={t['p_h']} c1={t['c1']} c2={t['c2']}  {fmt(t)}")

    out = resolve_path(args.out or f"configs/{sequence.name}_calibrated.yaml")
    header = (f"# Cellular automaton calibrated by scripts/calibrate_ca.py on {date.today()}\n"
              f"# sequence {sequence.name}, calibration days <= {args.train_end}: {fmt(best)}\n"
              f"# test days: {fmt(test_score)}\n"
              f"# Use: --config {out.relative_to(resolve_path('.')).as_posix()}\n")
    body = {"agents": {"simulation": {"steps_per_day": steps, "params": {"p_h": params.p_h, "a": params.a, "c1": params.c1, "c2": params.c2}}}}
    out.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
