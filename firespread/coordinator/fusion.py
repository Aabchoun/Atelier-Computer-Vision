"""Fusion rules used by the coordinator."""

from __future__ import annotations

from datetime import datetime

import numpy as np


def staleness_weight(confidence: float, acquisition_time: datetime, reference_time: datetime, tau_days: float) -> float:
    """confidence * exp(-age / tau): an older observation counts less."""
    age_days = max((reference_time - acquisition_time).total_seconds() / 86_400.0, 0.0)
    return float(confidence * np.exp(-age_days / tau_days))


def weighted_nanmean(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Per-pixel weighted mean ignoring NaN (= not observed). NaN where no source observed the pixel."""
    stack = np.stack([np.asarray(a, dtype=np.float32) for a in arrays])
    observed = ~np.isnan(stack)
    w = np.asarray(weights, dtype=np.float32)[:, None, None] * observed
    total = np.sum(np.where(observed, stack, 0.0) * w, axis=0)
    norm = w.sum(axis=0)
    return np.where(norm > 0, total / np.maximum(norm, 1e-12), np.nan).astype(np.float32)
