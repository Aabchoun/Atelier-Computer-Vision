"""Run the system day after day over a scene sequence and evaluate every day against FIRMS t+1.

One Coordinator is used for the whole sequence, so the last known layers (days without a
Sentinel-2 pass) and the fire history carry over from day to day, as in the real FIRMS-triggered flow.

    python scripts/run_sequence.py --sequence data/sequences/evros_2023 --split-date 2023-08-25
    python scripts/run_sequence.py --sequence data/sequences/evros_2023 --config configs/evros_calibrated.yaml --out outputs/evros_2023_calibrated --split-date 2023-08-25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.common.config import load_config, resolve_path  # noqa: E402
from firespread.common.message import json_default  # noqa: E402
from firespread.common.scene import S2_POST, TRUTH  # noqa: E402
from firespread.coordinator import Coordinator  # noqa: E402
from firespread.evaluation import FORECAST_LAYERS, evaluate_forecasts, micro_average  # noqa: E402
from firespread.io import load_raw_scene  # noqa: E402
from firespread.viz import SERIES, save_sequence_plot  # noqa: E402


def summarise(rows: list[dict], split_date: str | None) -> dict[str, dict[str, dict]]:
    groups = {
        "all days": rows,
        "days with Sentinel-2": [r for r in rows if r["s2"]],
        "days without Sentinel-2": [r for r in rows if not r["s2"]],
    }
    if split_date:
        groups[f"calibration days (<= {split_date})"] = [r for r in rows if r["day"] <= split_date]
        groups[f"test days (> {split_date})"] = [r for r in rows if r["day"] > split_date]
    return {name: {layer: micro_average(group, layer) for layer in FORECAST_LAYERS} for name, group in groups.items() if group}


def to_markdown(rows: list[dict], summary: dict, title: str) -> str:
    names = {layer: name for layer, (_, name) in SERIES.items()}
    lines = [f"# {title}", "", "## Pooled over days (micro average)", "",
             "| days | forecast | n | precision | recall | F1 | F1 (1-cell tolerance) |", "|---|---|---|---|---|---|---|"]
    for group, per_layer in summary.items():
        for layer, m in per_layer.items():
            lines.append(f"| {group} | {names[layer]} | {m['days']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['f1@1']:.3f} |")
    lines += ["", "## Day by day", "",
              "| day | Sentinel-2 | burning cells | truth cells | F1 persistence | F1 CA | F1 risk | F1@1 persistence | F1@1 CA | F1@1 risk |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f1 = [f"{r[layer]['f1']:.3f}" if layer in r else "-" for layer in FORECAST_LAYERS]
        f1_tol = [f"{r[layer]['f1@1']:.3f}" if layer in r else "-" for layer in FORECAST_LAYERS]
        lines.append(f"| {r['day']} | {'yes' if r['s2'] else ''} | {r['burning']} | {r['truth']} | " + " | ".join(f1 + f1_tol) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", required=True, help="data/sequences/<name>")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="default: outputs/<sequence name>")
    parser.add_argument("--split-date", default=None, help="last calibration day (YYYY-MM-DD) for the train/test summary")
    parser.add_argument("--no-quicklook", action="store_true", help="skip the per-day figures (faster)")
    parser.add_argument("--without-optical", action="store_true",
                        help="ablation: drop the during-fire Sentinel-2 scenes (fire from FIRMS only, no smoke, burned area from fire history)")
    args = parser.parse_args()

    config = load_config(args.config)
    sequence = resolve_path(args.sequence)
    out = resolve_path(args.out or f"outputs/{sequence.name}")
    day_dirs = sorted(p for p in (sequence / "days").iterdir() if (p / "scene.json").exists())
    threshold = config["coordinator"]["fire_threshold"]
    coordinator = Coordinator(config)

    rows = []
    for day_dir in day_dirs:
        start = time.perf_counter()
        raw = load_raw_scene(day_dir)
        had_s2 = S2_POST in raw.rasters  # days are grouped by what the real system would have had
        if args.without_optical:
            raw.rasters.pop(S2_POST, None)
        result = coordinator.run(raw)
        truth = raw.grid.resample(raw.get(TRUTH)["fire_next"], "max")
        result.report.evaluation = evaluate_forecasts(result.cube, truth)
        result.save(out / day_dir.name, quicklook=not args.no_quicklook)
        fire = result.cube.get("fire")
        row = {"day": day_dir.name, "s2": had_s2, "burning": int((fire >= threshold).sum()) if fire is not None else 0,
               "truth": int(truth.sum()), **evaluate_forecasts(result.cube, truth, label_with_model=False)}
        rows.append(row)
        scores = "  ".join(f"{layer}={row[layer]['f1']:.2f}/{row[layer]['f1@1']:.2f}" for layer in FORECAST_LAYERS if layer in row)
        print(f"{row['day']}  S2 {'yes' if row['s2'] else ' - '}  burning {row['burning']:4d}  truth {row['truth']:4d}  F1/F1@1: {scores}  ({time.perf_counter() - start:.0f} s)")
        del raw, result

    variant = [Path(args.config).stem] if args.config else []
    variant += ["without during-fire Sentinel-2"] if args.without_optical else []
    title = f"{sequence.name}: next-day active fire, day by day" + (f" ({', '.join(variant)})" if variant else "")
    summary = summarise(rows, args.split_date)
    markdown = to_markdown(rows, summary, title)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2, default=json_default), encoding="utf-8")
    save_sequence_plot(rows, out / "summary.png", title)
    print("\n" + markdown.split("## Day by day")[0])
    print(f"Saved per-day runs, summary.md and summary.png to {out}")


if __name__ == "__main__":
    main()
