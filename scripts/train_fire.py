"""Train the fire-agent U-Net on Sen2Fire and save a checkpoint the fire agent can load.

    pip install -e ".[ml]"
    python scripts/train_fire.py --inspect                       # check the .npz layout first
    python scripts/train_fire.py --epochs 1 --max-train-samples 64 # quick check
    python scripts/train_fire.py --epochs 40 --pretrained          # full run

Inputs are the 7 Sentinel-2 bands the pipeline provides (B2 B3 B4 B8 B8A B11 B12, reflectance), so
the trained model runs inside the fire agent: set `agents.fire.checkpoint: models/fire_unet.pt`.

The test table compares the U-Net with the agent's current baseline (SWIR threshold test) on the
official Sen2Fire test scene. Note: the labels come from MODIS (1 km) resampled to 10 m, so they
are coarse; the scores measure agreement with MODIS, not with exact fire pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firespread.agents.fire.agent import swir_fire_test  # noqa: E402
from firespread.common.config import load_config, resolve_path  # noqa: E402
from firespread.common.nn import build_unet, masked_focal_dice_loss, save_checkpoint  # noqa: E402
from firespread.io.sen2fire import PIPELINE_BANDS, describe, load_patch, split_files  # noqa: E402

BINS = 1000


def build_cache(root: Path, split: str, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """All patches of a split as memory-mapped arrays: x (N, 7, H, W) float16, y (N, H, W) uint8."""
    x_path, y_path = cache_dir / f"{split}_x.npy", cache_dir / f"{split}_y.npy"
    if x_path.exists() and y_path.exists():
        return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")
    files = split_files(root, split)
    first_x, first_y = load_patch(files[0])
    cache_dir.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(x_path.with_suffix(".tmp.npy"), "w+", np.float16, (len(files), *first_x.shape))
    y = np.lib.format.open_memmap(y_path.with_suffix(".tmp.npy"), "w+", np.uint8, (len(files), *first_y.shape))
    for i, path in enumerate(files):
        xi, yi = load_patch(path)
        x[i], y[i] = xi, yi
        if (i + 1) % 200 == 0:
            print(f"  {split}: cached {i + 1}/{len(files)}")
    x.flush(), y.flush()
    del x, y
    x_path.with_suffix(".tmp.npy").replace(x_path)
    y_path.with_suffix(".tmp.npy").replace(y_path)
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")


class Histogram:
    """Streaming precision / recall / F1 / IoU at any threshold + AUC-PR, over hundreds of millions of pixels."""

    def __init__(self) -> None:
        self.pos = np.zeros(BINS, np.int64)
        self.neg = np.zeros(BINS, np.int64)

    def add(self, prob: np.ndarray, target: np.ndarray) -> None:
        idx = np.clip((np.asarray(prob, np.float32) * BINS).astype(np.int64), 0, BINS - 1)
        t = np.asarray(target) > 0.5
        self.pos += np.bincount(idx[t], minlength=BINS)
        self.neg += np.bincount(idx[~t], minlength=BINS)

    def metrics(self, threshold: float = 0.5) -> dict[str, float]:
        k = int(round(threshold * BINS))
        tp, fp = self.pos[k:].sum(), self.neg[k:].sum()
        fn = self.pos[:k].sum()
        ratio = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
        tp_c, fp_c = np.cumsum(self.pos[::-1]), np.cumsum(self.neg[::-1])  # thresholds from high to low
        precision = tp_c / np.maximum(tp_c + fp_c, 1)
        recall = tp_c / max(self.pos.sum(), 1)
        auc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        return {
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "iou": ratio(tp, tp + fp + fn),
            "auc_pr": auc,
        }

    def best_threshold(self) -> float:
        candidates = np.round(np.arange(0.05, 0.96, 0.05), 2)
        return float(max(candidates, key=lambda t: np.nan_to_num(self.metrics(t)["f1"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/raw/sen2fire")
    parser.add_argument("--inspect", action="store_true", help="print the layout of a few patches and stop")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--pretrained", action="store_true", help="ImageNet encoder weights (needs internet once)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256, help="random training crop (multiple of 32)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fire-oversampling", type=float, default=4.0, help="sampling weight of patches that contain fire")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/fire_unet.pt")
    args = parser.parse_args()
    root = resolve_path(args.data)

    if args.inspect:
        for split in ("train", "eval", "test"):
            files = split_files(root, split)
            print(f"== {split}: {len(files)} patches")
            print(describe(files[0]))
        return

    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}{' (' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ''}")
    print(f"inputs: {list(PIPELINE_BANDS)}")

    # --- data --------------------------------------------------------------------------------
    t0 = time.time()
    cache_dir = root / "cache"
    data = {split: build_cache(root, split, cache_dir) for split in ("train", "eval", "test")}
    for split, (x, y) in data.items():
        has_fire = np.array([y[i].any() for i in range(len(y))])
        print(f"{split:5s}: {len(x):5d} patches {tuple(x.shape[2:])}, with fire {has_fire.mean():.1%}, fire pixels {np.mean([y[i].mean() for i in range(len(y))]):.3%}")
    print(f"data ready in {time.time() - t0:.0f} s")

    x_train, y_train = data["train"]
    train_idx = np.arange(len(x_train))
    if args.max_train_samples and args.max_train_samples < len(train_idx):
        train_idx = rng.choice(train_idx, args.max_train_samples, replace=False)

    sample = rng.choice(train_idx, min(200, len(train_idx)), replace=False)
    stats = np.stack([np.asarray(x_train[i], np.float32) for i in sorted(sample)])
    mean = np.nanmean(stats, axis=(0, 2, 3))
    std = np.nanstd(stats, axis=(0, 2, 3))
    std = np.where(std > 1e-6, std, 1.0)
    del stats
    print("band statistics (reflectance):", " ".join(f"{b}={m:.3f}±{s:.3f}" for b, m, s in zip(PIPELINE_BANDS, mean, std)))
    m_t, s_t = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]

    class Patches(Dataset):
        def __init__(self, x, y, idx, crop: int | None):
            self.x, self.y, self.idx, self.crop = x, y, idx, crop

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            x = torch.from_numpy(np.asarray(self.x[j], np.float32))
            y = torch.from_numpy(np.asarray(self.y[j], np.float32))[None]
            if self.crop:
                h, w = x.shape[1:]
                top, left = np.random.randint(0, h - self.crop + 1), np.random.randint(0, w - self.crop + 1)
                x, y = x[:, top : top + self.crop, left : left + self.crop], y[:, top : top + self.crop, left : left + self.crop]
                k = np.random.randint(4)  # no wind here: all rotations / flips are valid
                x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
                if np.random.rand() < 0.5:
                    x, y = x.flip(-1), y.flip(-1)
            x = torch.nan_to_num((x - m_t) / s_t, nan=0.0)
            return x.contiguous(), y.contiguous()

    has_fire = np.array([y_train[j].any() for j in train_idx])
    weights = np.where(has_fire, args.fire_oversampling, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(train_idx), replacement=True)
    train_loader = DataLoader(
        Patches(x_train, y_train, train_idx, args.crop),
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        persistent_workers=args.workers > 0,
    )

    def loader(split: str) -> DataLoader:
        x, y = data[split]
        return DataLoader(Patches(x, y, np.arange(len(x)), None), batch_size=8, num_workers=args.workers)

    # --- model -------------------------------------------------------------------------------
    model = build_unet(len(PIPELINE_BANDS), encoder=args.encoder, encoder_weights="imagenet" if args.pretrained else None).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    use_amp = device == "cuda"

    def score(split: str) -> Histogram:
        model.eval()
        hist = Histogram()
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
            for xb, yb in loader(split):
                prob = torch.sigmoid(model(xb.to(device)).float()).cpu().numpy()
                hist.add(prob, yb.numpy())
        return hist

    # --- training ----------------------------------------------------------------------------
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(xb)
            loss = masked_focal_dice_loss(logits.float(), yb)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.item())
        val = score("eval").metrics()
        flag = ""
        if val["auc_pr"] > best_auc:
            best_auc, best_epoch = val["auc_pr"], epoch
            torch.save(model.state_dict(), out_path.with_suffix(".best_state.pt"))
            flag = "  <- best"
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {np.mean(losses):.4f}  "
            f"val P {val['precision']:.3f} R {val['recall']:.3f} F1 {val['f1']:.3f} AUC-PR {val['auc_pr']:.3f}  "
            f"({time.time() - t0:.0f} s){flag}"
        )

    # --- test --------------------------------------------------------------------------------
    model.load_state_dict(torch.load(out_path.with_suffix(".best_state.pt"), map_location=device, weights_only=True))
    threshold = score("eval").best_threshold()
    unet = score("test")

    fire_cfg = load_config()["agents"]["fire"]
    baseline = Histogram()
    x_test, y_test = data["test"]
    b = {name: PIPELINE_BANDS.index(name) for name in ("B8A", "B11", "B12")}
    for i in range(len(x_test)):
        xi = np.asarray(x_test[i], np.float32)
        baseline.add(swir_fire_test(xi[b["B8A"]], xi[b["B11"]], xi[b["B12"]], fire_cfg["swir2_min"], fire_cfg["ratio_min"]), y_test[i])

    results = {
        "swir-threshold (baseline)": baseline.metrics(),
        "unet@0.5": unet.metrics(),
        f"unet@{threshold}": unet.metrics(threshold),
    }
    keys = ("precision", "recall", "f1", "iou", "auc_pr")
    print(f"\nTest scene ({len(x_test)} patches), best epoch {best_epoch} (val AUC-PR {best_auc:.3f})")
    print("| model | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for name, metrics in results.items():
        print(f"| {name} | " + " | ".join(f"{metrics[k]:.3f}" for k in keys) + " |")

    save_checkpoint(model, out_path, inputs=list(PIPELINE_BANDS), encoder=args.encoder, mean=mean, std=std, metrics=results["unet@0.5"])
    report = {
        "dataset": "sen2fire",
        "inputs": list(PIPELINE_BANDS),
        "encoder": args.encoder,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_patches": int(len(train_idx)),
        "best_val_threshold": threshold,
        "test": results,
    }
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_path.with_suffix(".best_state.pt").unlink(missing_ok=True)
    print(f"\nSaved {out_path} and {out_path.with_suffix('.json')}")
    print(f"Use it: agents.fire.checkpoint: {args.out}, agents.fire.s2_confidence = test F1")


if __name__ == "__main__":
    main()
