"""Deep-learning helpers shared by the trainable agents (U-Net, loss, checkpoints, inference).

PyTorch and segmentation_models_pytorch are optional (`pip install -e ".[ml]"`) and imported
lazily, so the rest of the pipeline runs without them.

Checkpoint convention (one file per agent, e.g. models/spread_unet.pt), written by save_checkpoint:
    state_dict  - model weights
    encoder     - smp encoder name (e.g. "resnet34")
    inputs      - ordered list of input channels: band names (B12, B11, ...) for perception agents,
                  datacube layer names (fire, wind_u, ...) for the spread agent
    mean, std   - per-channel normalisation computed on the TRAINING split
    metrics     - validation metrics (report them; use F1 as the agent's confidence)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError('This feature needs PyTorch: pip install -e ".[ml]"') from exc
    return torch


def build_unet(in_channels: int, classes: int = 1, encoder: str = "resnet34", encoder_weights: str | None = None):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)


def masked_focal_dice_loss(logits, target, ignore_value: float = -1.0, alpha: float = 0.25, gamma: float = 2.0, dice_weight: float = 1.0, eps: float = 1.0):
    """Focal + Dice loss for rare fire pixels; target pixels equal to `ignore_value` are skipped.

    logits, target: tensors of shape (N, 1, H, W); target in {0, 1, ignore_value}.
    """
    torch = _torch()
    valid = target != ignore_value
    if not bool(valid.any()):
        return logits.sum() * 0.0
    x = logits[valid]
    t = target[valid].float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(x, t, reduction="none")
    p = torch.sigmoid(x)
    p_t = p * t + (1 - p) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    focal = (alpha_t * (1 - p_t) ** gamma * bce).mean()
    dice = 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return focal + dice_weight * dice


def save_checkpoint(model, path: str | Path, *, inputs: list[str], encoder: str, mean=None, std=None, metrics: dict | None = None) -> None:
    torch = _torch()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoder": encoder,
            "inputs": list(inputs),
            "mean": None if mean is None else [float(m) for m in mean],
            "std": None if std is None else [float(s) for s in std],
            "metrics": {k: float(v) for k, v in (metrics or {}).items()},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_unet(len(ckpt["inputs"]), encoder=ckpt["encoder"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}


def predict_proba(model, x: np.ndarray, mean=None, std=None, device: str | None = None, tile: int = 512, overlap: int = 64) -> np.ndarray:
    """Probability map (H, W) for one (C, H, W) input. NaN inputs are set to the channel mean.

    Inputs larger than `tile` are processed in overlapping windows (averaged where they overlap),
    so a full Sentinel-2 scene fits in memory. Uses the GPU when available unless `device` is set.
    """
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = np.array(x, dtype=np.float32)
    if mean is not None and std is not None:
        x = (x - np.asarray(mean, np.float32)[:, None, None]) / np.asarray(std, np.float32)[:, None, None]
    x = np.nan_to_num(x, nan=0.0)
    _, h, w = x.shape
    model = model.to(device).eval()

    def run(window: np.ndarray) -> np.ndarray:
        wh, ww = window.shape[1:]
        pad_h, pad_w = (-wh) % 32, (-ww) % 32  # U-Net needs sizes divisible by 32
        tensor = torch.from_numpy(np.ascontiguousarray(window))[None].to(device)
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad():
            return torch.sigmoid(model(tensor).float())[0, 0, :wh, :ww].cpu().numpy()

    if h <= tile and w <= tile:
        return run(x).astype(np.float32)

    total = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    step = tile - overlap
    for top in sorted({*range(0, max(h - tile, 0) + 1, step), max(h - tile, 0)}):
        for left in sorted({*range(0, max(w - tile, 0) + 1, step), max(w - tile, 0)}):
            window = x[:, top : top + tile, left : left + tile]
            total[top : top + window.shape[1], left : left + window.shape[2]] += run(window)
            count[top : top + window.shape[1], left : left + window.shape[2]] += 1
    return total / np.maximum(count, 1)
