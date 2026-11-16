"""Segmentation metrics for imbalanced fire masks: never report plain accuracy.

Target pixels equal to `ignore_value` (-1 = cloud / missing in Next Day Wildfire Spread) and NaN
pixels are excluded.
"""

from __future__ import annotations

import numpy as np

from firespread.common.raster_ops import dilate


def _valid(pred, target, ignore_value: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    keep = (target != ignore_value) & np.isfinite(target) & np.isfinite(pred)
    return pred[keep], target[keep] > 0.5


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def segmentation_metrics(pred, target, threshold: float = 0.5, ignore_value: float = -1) -> dict[str, float]:
    p, t = _valid(pred, target, ignore_value)
    p = p >= threshold
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    return {
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def average_precision(scores, target, ignore_value: float = -1) -> float:
    """Area under the precision-recall curve (step-wise, same definition as scikit-learn)."""
    s, t = _valid(scores, target, ignore_value)
    if t.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, t = s[order], t[order]
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]  # last index of each distinct score
    tps = np.cumsum(t)[last]
    fps = last + 1 - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate(prob, target, threshold: float = 0.5, ignore_value: float = -1) -> dict[str, float]:
    """Precision, recall, F1, IoU at `threshold` + AUC-PR."""
    return {
        **segmentation_metrics(prob, target, threshold, ignore_value),
        "auc_pr": average_precision(prob, target, ignore_value),
    }


def iou(pred_mask, target_mask) -> float:
    return segmentation_metrics(np.asarray(pred_mask, float), np.asarray(target_mask, float))["iou"]


def tolerance_metrics(pred, target, radius: int = 1, threshold: float = 0.5, ignore_value: float = -1) -> dict[str, float]:
    """Precision / recall / F1 that accept an error of `radius` cells (2-D masks).

    A predicted cell counts as correct if a true cell is within `radius` cells, and a true cell
    as found if a predicted cell is within `radius`. Fair for thin fire fronts at 375 m.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = (target != ignore_value) & np.isfinite(target)
    p = (np.nan_to_num(pred) >= threshold) & valid
    t = (target > 0.5) & valid
    pred_near = int((p & dilate(t, radius)).sum())
    true_near = int((t & dilate(p, radius)).sum())
    precision, recall = _ratio(pred_near, int(p.sum())), _ratio(true_near, int(t.sum()))
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        f"precision@{radius}": precision,
        f"recall@{radius}": recall,
        f"f1@{radius}": f1,
        "n_pred": int(p.sum()),
        "n_pred_near": pred_near,
        "n_true": int(t.sum()),
        "n_true_near": true_near,
    }
