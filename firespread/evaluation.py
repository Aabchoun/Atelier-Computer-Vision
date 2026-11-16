"""Evaluation of the forecasts against a ground-truth fire mask (benchmark table, ablation, sequences)."""

from __future__ import annotations

import numpy as np

from firespread.common.metrics import evaluate, tolerance_metrics
from firespread.common.scene import SceneDatacube

FORECAST_LAYERS = ("spread_prob", "sim_prob", "risk")
TABLE_COLUMNS = ("precision", "recall", "f1", "iou", "auc_pr", "f1@1")


def evaluate_forecasts(cube: SceneDatacube, truth: np.ndarray, threshold: float = 0.5, layers=FORECAST_LAYERS,
                       tolerance: int = 1, label_with_model: bool = True) -> dict[str, dict[str, float]]:
    """Exact metrics + metrics with a `tolerance`-cell margin for each forecast layer.

    With label_with_model the model name is added to the key (e.g. "spread_prob (persistence)").
    """
    results = {}
    for layer in layers:
        message = cube.message(layer)
        if message is None:
            continue
        label = f"{layer} ({message.extras['model']})" if label_with_model and "model" in message.extras else layer
        results[label] = {**evaluate(message.data, truth, threshold), **tolerance_metrics(message.data, truth, tolerance, threshold)}
    return results


def results_table(results: dict[str, dict[str, float]], columns=TABLE_COLUMNS, first_column: str = "forecast") -> str:
    if results:
        first = next(iter(results.values()))
        columns = [c for c in columns if c in first]
    lines = [f"| {first_column} | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for name, metrics in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{metrics[c]:.3f}" for c in columns) + " |")
    return "\n".join(lines)


def micro_average(per_day: list[dict[str, dict[str, float]]], layer: str) -> dict[str, float]:
    """Pool the counts of several days (micro average) for one forecast layer."""
    rows = [day[layer] for day in per_day if layer in day]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    pred, pred_near, true, true_near = (sum(r[k] for r in rows) for k in ("n_pred", "n_pred_near", "n_true", "n_true_near"))
    nan = float("nan")
    p_tol = pred_near / pred if pred else nan
    r_tol = true_near / true if true else nan
    return {
        "days": len(rows),
        "precision": tp / (tp + fp) if tp + fp else nan,
        "recall": tp / (tp + fn) if tp + fn else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan,
        "f1@1": nan if np.isnan(p_tol) or np.isnan(r_tol) else (2 * p_tol * r_tol / (p_tol + r_tol) if p_tol + r_tol > 0 else 0.0),
    }
